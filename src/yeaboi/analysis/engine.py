"""Team-analysis engine — the headless pipeline behind the TUI Analysis mode.

# See docs: "Architecture" — engines are UI-free pipelines; the TUI, CLI and
# MCP server are thin adapters over them (AGENTS.md "REQUIRED: Surface Parity").

Design choice — standalone pipeline, not a LangGraph node (same rationale as
``standup/engine.py``): the analysis is a deterministic gather step
(``_fetch_*_history``) followed by the 4-worker parallel analysis in
``tools/team_learning.py`` (which already handles its own LLM calls with
regex fallbacks), so a compiled graph would add checkpointing overhead for
nothing.

Error contract:
- Missing tracker / no closed sprints / fetch failures **raise** — with no
  board there is nothing to analyse, and every caller (TUI worker, CLI, MCP
  ``run_engine``) has its own error surface for that.
- LLM failures never raise: the parsers inside ``_run_parallel_analysis`` fall
  back to regex extraction, and the optional insights/samples steps degrade to
  a ``warnings`` entry.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from yeaboi.analysis import repo_inventory
from yeaboi.analysis.cancellation import AnalysisCancelledError
from yeaboi.analysis.setup import available_doc_sources, available_trackers, scannable_code_sources

# Re-exported: callers (TUI worker, tests) import the error from the engine,
# but it lives in analysis/cancellation.py so the fetch layers can raise it
# without importing this module (engine imports them — reverse would cycle).
from yeaboi.context.labels import label_run
from yeaboi.context.scope import ContextScope, coerce_scope

logger = logging.getLogger(__name__)


# Friendly tracker labels for 'both'-mode output (mirrors reporting/engine.py's
# _source_names). Note analysis uses "azdevops" (not reporting's "azuredevops").
_SOURCE_NAMES = {"jira": "Jira", "azdevops": "Azure DevOps"}

# The three analysis components are decoupled — each runs over its OWN sub-sources,
# not the tracker. Delivery (the sprint/ticket pipeline → TeamProfile) runs PER
# tracker (velocity isn't comparable across trackers). Code (remote AI-usage scan)
# and Docs (doc-quality read) are each ONE global scan over their selected hosts.
# Note: the code Azure-Repos tag is "azdo", distinct from the delivery tracker key
# "azdevops" — they are different systems.
_COMPONENTS = ("delivery", "code", "docs", "ops")
_DELIVERY_SOURCES = ("jira", "azdevops")
_CODE_SOURCES = ("github", "azdo")
_DOC_SOURCES = ("confluence", "notion")
_COMPONENT_SOURCES: dict[str, tuple[str, ...]] = {
    "delivery": _DELIVERY_SOURCES,
    "code": _CODE_SOURCES,
    "docs": _DOC_SOURCES,
}
_ANALYSIS_FEATURES = ("delivery", "ai_footprint", "code_health", "documentation", "operational")
_CODE_FEATURES = ("ai_footprint", "code_health")
#: Which component each feature reads. Ops is the one component whose known
#: sub-sources are a registry rather than a literal — a connector added later
#: must not need an edit here to be selectable.
_FEATURE_COMPONENT = {
    "delivery": "delivery",
    "ai_footprint": "code",
    "code_health": "code",
    "documentation": "docs",
    "operational": "ops",
}


def _ops_sources() -> tuple[str, ...]:
    """The connector keys an Operations run may read, from the live registry."""
    from yeaboi.analysis.setup import available_ops_sources

    return tuple(available_ops_sources())


def _resolve_analysis_features(
    analysis_features: list[str] | tuple[str, ...] | set[str] | None,
    comps: dict[str, list[str]],
) -> list[str]:
    """Validate feature selection and intersect it with selected integrations."""
    requested = list(_ANALYSIS_FEATURES if analysis_features is None else analysis_features)
    unknown = [feature for feature in requested if feature not in _ANALYSIS_FEATURES]
    if unknown:
        raise ValueError(f"analysis_features must be a subset of {_ANALYSIS_FEATURES!r} — got {unknown!r}")
    selected = []
    for feature in _ANALYSIS_FEATURES:
        if feature not in requested:
            continue
        component = _FEATURE_COMPONENT[feature]
        if comps[component]:
            selected.append(feature)
    if not selected:
        raise ValueError("Nothing to analyse — select at least one analysis feature with a configured integration.")
    return selected


def _resolve_components(
    source: str,
    components: dict[str, list[str]] | None,
    include_ai_usage: bool,
    include_doc_quality: bool,
) -> dict[str, list[str]]:
    """Resolve the component → sub-source map that will actually run.

    An explicit ``components`` (keyed ``delivery``/``code``/``docs``) wins, filtered
    to each component's known sub-sources. Otherwise derive from ``source`` + the
    legacy booleans: delivery over the resolved tracker(s) (``source``/'both'/auto),
    code/docs over all their sub-sources when the booleans are set — reproducing
    today's behaviour, except code/docs now run **once** rather than per tracker.
    """
    if components is not None:

        def _pick(comp: str) -> list[str]:
            allowed = _COMPONENT_SOURCES[comp] if comp in _COMPONENT_SOURCES else _ops_sources()
            return [v for v in (components.get(comp) or []) if v in allowed]

        return {comp: _pick(comp) for comp in _COMPONENTS}

    if source == "both":
        delivery = available_trackers()
    elif source in _DELIVERY_SOURCES:
        delivery = [source]
    else:
        from yeaboi.tools.team_learning import _detect_source

        detected = _detect_source()
        delivery = [detected] if detected in _DELIVERY_SOURCES else []
    result = {
        "delivery": delivery,
        "code": scannable_code_sources() if include_ai_usage else [],
        "docs": available_doc_sources() if include_doc_quality else [],
        # No legacy boolean turns Operations on: it is opt-in through the
        # feature list only, so no existing caller starts reading a vendor it
        # never asked about.
        "ops": [],
    }
    return result


def _resolve_source(source: str) -> str:
    from yeaboi.tools.team_learning import _detect_source

    resolved = source or _detect_source()
    if resolved not in ("jira", "azdevops"):
        raise ValueError(
            "No tracker configured for analysis — set JIRA_BASE_URL/JIRA_EMAIL/JIRA_API_TOKEN "
            "or AZURE_DEVOPS_ORG_URL/AZURE_DEVOPS_TOKEN (source: 'jira' or 'azdevops')."
        )
    return resolved


def _resolve_project(source: str, project_key: str, team_name: str) -> tuple[str, str]:
    if project_key:
        return project_key, team_name
    try:
        if source == "jira":
            from yeaboi.config import get_jira_project_key

            return get_jira_project_key() or "", team_name
        from yeaboi.config import get_azure_devops_project, get_azure_devops_team

        return get_azure_devops_project() or "", team_name or (get_azure_devops_team() or "")
    except Exception:
        return project_key, team_name


def _generate_samples(profile, examples: dict, warnings: list[str]) -> dict | None:
    """Auto-accepted sample tickets in the team's style (the TUI preview flow,
    minus the interactive accept/edit loop)."""
    try:
        from yeaboi.agent.nodes import _format_team_calibration
        from yeaboi.tools.team_learning import (
            generate_sample_epic,
            generate_sample_sprint,
            generate_sample_stories,
            generate_sample_tasks,
        )

        calibration = _format_team_calibration(profile, examples=examples)
        epic = generate_sample_epic(calibration, examples)
        stories = generate_sample_stories(calibration, epic, examples)
        tasks = generate_sample_tasks(calibration, stories, examples)
        sprint = generate_sample_sprint(calibration, stories, tasks, examples)
        return {"epic": epic, "stories": stories, "tasks": tasks, "sprint": sprint}
    except Exception as exc:  # LLM/parse trouble → warning, never a crash
        logger.warning("Sample-ticket generation failed: %s", exc)
        warnings.append(f"Sample-ticket generation failed: {exc}")
        return None


# Headline rows shown in the 'both' side-by-side comparison. Each entry is
# (label, formatter) where formatter renders one profile's value; values are
# never blended across trackers — they sit in separate columns.
_COMPARISON_ROWS: tuple[tuple[str, callable], ...] = (
    ("Sprints analysed", lambda p: str(p.sample_sprints)),
    ("Stories analysed", lambda p: str(p.sample_stories)),
    ("Avg velocity", lambda p: f"{p.velocity_avg:.0f} ± {p.velocity_stddev:.0f}"),
    ("Completion rate", lambda p: f"{p.sprint_completion_rate:.0f}%"),
    ("Estimation accuracy", lambda p: f"{p.estimation_accuracy_pct:.0f}%"),
)


def _build_comparison(delivery: dict) -> list[tuple[str, str, str]]:
    """Side-by-side delivery headline rows: (label, jira_value, azdevops_value). Kept
    deliberately separate (not aggregated) so each number names its tracker."""
    jira = delivery.get("jira", {}).get("profile")
    azdo = delivery.get("azdevops", {}).get("profile")
    rows: list[tuple[str, str, str]] = []
    for label, fmt in _COMPARISON_ROWS:
        rows.append((label, fmt(jira) if jira else "—", fmt(azdo) if azdo else "—"))
    return rows


def get_team_roster_result(
    source: str = "",
    project_key: str = "",
    sprint_count: int = 8,
    db_path=None,
    *,
    days: int = 30,
    force_refresh: bool = False,
):
    """Return status-aware recent/WIP assignee discovery for one tracker.

    ``sprint_count`` is retained for API compatibility but no longer controls
    roster discovery. Unlike a full analysis, this path never reads sprint
    history, comments, documentation, repositories, or an LLM.
    """
    del sprint_count
    from yeaboi.team_roster import fetch_roster_result

    resolved_source = _resolve_source(source)
    resolved_project, _ = _resolve_project(resolved_source, project_key, "")
    result = fetch_roster_result(
        jira_project=resolved_project if resolved_source == "jira" else "",
        azdo_project=resolved_project if resolved_source == "azdevops" else "",
        days=days,
        db_path=db_path,
        force_refresh=force_refresh,
    )
    logger.info(
        "Roster for %s/%s: %d member(s), status=%s",
        resolved_source,
        resolved_project,
        len(result.members),
        result.status,
    )
    return result


def get_team_roster(
    source: str = "",
    project_key: str = "",
    sprint_count: int = 8,
    db_path=None,
    *,
    days: int = 30,
    force_refresh: bool = False,
) -> list[str]:
    """Return sorted assignee names while preserving the legacy list API."""
    result = get_team_roster_result(
        source,
        project_key,
        sprint_count,
        db_path,
        days=days,
        force_refresh=force_refresh,
    )
    return sorted(
        {member.name.strip() for member in result.members if member.name.strip()},
        key=str.casefold,
    )


def run_team_analysis(
    source: str = "",
    project_key: str = "",
    team_name: str = "",
    sprint_count: int = 8,
    generate_samples: bool = False,
    include_insights: bool = True,
    include_ai_usage: bool = True,
    include_doc_quality: bool = True,
    components: dict[str, list[str]] | None = None,
    members: dict[str, list[str]] | None = None,
    *,
    analysis_depth: str = "deep",
    analysis_window_days: int = 120,
    analysis_scope: dict[str, list[str]] | None = None,
    analysis_model: str | None = None,
    analysis_features: list[str] | None = None,
    progress: list | None = None,
    db_path=None,
    cancel_event: threading.Event | None = None,
    context: ContextScope | dict | str | None = None,
    project_label: str = "",
    tags: Sequence[str] = (),
) -> dict:
    """Analyse the team into decoupled Delivery / Code / Docs components.

    The three components run independently over their **own** sub-sources:
    **Delivery** (velocity/calibration/contributors → a ``TeamProfile``) runs once
    per selected tracker (jira/azdevops; never blended). **Code** (remote AI-usage
    scan over github/azdo) and **Docs** (doc-quality over confluence/notion) are each
    a single **global** scan. Returns:
    ``{"delivery": {tracker: {profile, examples, ...}}, "code": {signal|None, examples}|None,
    "docs": {signal, examples}|None, "comparison": [...], "components": {...},
    "warnings": [...]}``. The global code/docs signals are also attached to every
    saved delivery profile (so the stored-profile browser keeps showing them).

    Args:
        source: 'jira', 'azdevops', or 'both'; blank auto-detects a single
            tracker from configured creds.
        project_key: tracker project; blank falls back to the configured one.
        team_name: AzDO team name attached to the profile (blank = configured).
        sprint_count: closed sprints to analyse (TUI uses 8).
        generate_samples: also generate auto-accepted sample tickets
            (epic/stories/tasks/sprint) in the team's style — extra LLM calls.
        include_insights: also generate the start/stop/keep/try coaching
            insights (one extra LLM call).
        include_ai_usage: legacy toggle folded into ``components`` when the latter is
            None — scan commits/PRs for AI-tool markers (Code component).
        include_doc_quality: legacy toggle folded into ``components`` when None — read
            recent Notion/Confluence pages (Docs component).
        analysis_depth: ``quick`` makes no LLM calls and uses deterministic
            explanations; ``deep`` adds cached ticket classification and AI-written
            enrichments. Defaults to ``deep``.
        analysis_window_days: changed-content window shared by Code and Docs.
        analysis_scope: provider → configured containers, such as GitHub owners,
            Azure projects, Confluence spaces, and Notion roots.
        analysis_model: optional per-run model for lightweight structured Analysis
            tasks. The primary model still writes the final synthesis.
        analysis_features: independently selectable result areas: ``delivery``,
            ``ai_footprint``, ``code_health``, and ``documentation``. ``None``
            enables every feature supported by the selected component integrations.
        components: component → sub-source map, e.g.
            ``{"delivery": ["jira"], "code": ["github", "azdo"], "docs": ["confluence"]}``.
            Each component runs over ONLY its listed sub-sources; an absent/empty
            component is skipped. ``None`` derives the default from ``source`` + the
            two booleans (delivery over source/both/auto; code/docs over all their
            sub-sources).
        members: per delivery-tracker subset of assignee names, e.g.
            ``{"jira": ["Alice", "Bob"]}`` — re-scopes that tracker's velocity/
            contributors. The global code scan strictly filters activity and changed
            files by the union of selected members. A blank or unmatched code scope
            stays empty and never broadens to whole-team code.
        progress: optional shared list the analysis workers append activity strings
            and explicit component lifecycle events to.
        db_path: sessions DB override (tests). Defaults to paths.get_db_path().
        cancel_event: injected in-process cancel seam (TUI worker thread), like
            ``progress``/``db_path``. When set, queued jobs abort at pickup and
            the whole run raises ``AnalysisCancelledError`` before anything persists.

    Raises ValueError when nothing at all can be analysed (no tracker/component
    configured); per-tracker board errors degrade to a ``warnings`` entry.
    """
    if analysis_depth not in ("quick", "deep"):
        raise ValueError(f"analysis_depth must be 'quick' or 'deep' — got {analysis_depth!r}")
    if not 1 <= int(analysis_window_days) <= 3650:
        raise ValueError("analysis_window_days must be between 1 and 3650")
    if analysis_depth == "quick" and generate_samples:
        raise ValueError("generate_samples requires analysis_depth='deep' because sample generation uses the LLM.")
    from yeaboi.paths import get_db_path

    effective_db_path = db_path or get_db_path()
    from yeaboi.analysis.llm_runtime import reset_analysis_llm_execution

    reset_analysis_llm_execution(model=analysis_model)
    comps = _resolve_components(source, components, include_ai_usage, include_doc_quality)
    if not any(comps.values()):
        _resolve_source("")  # preserve the canonical no-integration error
    features = _resolve_analysis_features(analysis_features, comps)
    feature_set = set(features)
    comps = {
        "delivery": comps["delivery"] if "delivery" in feature_set else [],
        "code": comps["code"] if feature_set & set(_CODE_FEATURES) else [],
        "docs": comps["docs"] if "documentation" in feature_set else [],
        "ops": comps["ops"] if "operational" in feature_set else [],
    }
    members = members or {}
    warnings: list[str] = []
    progress_list = progress if progress is not None else []
    logger.info(
        "Team analysis starting: delivery=%s code=%s docs=%s ops=%s members=%s",
        comps["delivery"],
        comps["code"],
        comps["docs"],
        comps["ops"],
        members or "all",
    )

    # Build one independent top-level job per delivery tracker plus one global Code
    # and Docs job.  Delivery's own four-worker analysis remains unchanged; this
    # outer pool removes the serial wait between otherwise unrelated components.
    delivery_results: dict[str, dict] = {}
    single = len(comps["delivery"]) == 1
    union_members = sorted({m for names in members.values() for m in (names or [])}) or None
    jobs: list[tuple[str, str, tuple, dict]] = []
    for tracker in comps["delivery"] if "delivery" in feature_set else []:
        jobs.append(
            (
                "delivery",
                tracker,
                (
                    tracker,
                    project_key if single else "",
                    team_name if single else "",
                    members.get(tracker),
                    sprint_count,
                    generate_samples,
                    include_insights,
                    analysis_depth,
                    progress_list,
                    effective_db_path,
                    cancel_event,
                ),
                {},
            )
        )

    selected_code_features = [feature for feature in _CODE_FEATURES if feature in feature_set]
    if comps["code"] and selected_code_features:
        from yeaboi.tools.team_learning import _run_ai_usage_component

        jobs.append(
            (
                "code",
                "code",
                ("", "", [], [], union_members, progress_list),
                {
                    "sub_sources": comps["code"],
                    "analysis_depth": analysis_depth,
                    "window_days": analysis_window_days,
                    "analysis_scope": analysis_scope,
                    "db_path": effective_db_path,
                    "code_features": selected_code_features,
                    "cancel_event": cancel_event,
                },
            )
        )
    if comps["docs"] and "documentation" in feature_set:
        from yeaboi.tools.team_learning import _run_doc_quality_component

        jobs.append(
            (
                "docs",
                "docs",
                ("", "", progress_list),
                {
                    "sub_sources": comps["docs"],
                    "analysis_depth": analysis_depth,
                    "window_days": analysis_window_days,
                    "analysis_scope": analysis_scope,
                    "db_path": effective_db_path,
                },
            )
        )

    if comps["ops"] and "operational" in feature_set:
        from yeaboi.analysis.operational import run_operational

        jobs.append(
            (
                "ops",
                "ops",
                (analysis_window_days,),
                {"sub_sources": comps["ops"], "progress": progress_list},
            )
        )

    code = None
    docs = None
    ops = None
    if jobs:
        from yeaboi.analysis.progress import append_component_progress

        def _job_progress(kind: str, key: str) -> list[tuple[str, str]]:
            if kind == "delivery":
                label = f"Fetching sprint history · {_SOURCE_NAMES.get(key, key)}"
                return [(f"{kind}:{key}", label)]
            elif kind == "code":
                labels = {
                    "ai_footprint": "Scanning selected-user AI footprint",
                    "code_health": "Analysing selected-user code-change health",
                }
                return [(f"code:{feature}", labels[feature]) for feature in selected_code_features]
            elif kind == "ops":
                return [("ops:operational", "Reading production")]
            else:
                label = "Assessing documentation quality"
                return [("docs:documentation", label)]

        def _guarded(fn, /, *args, **kwargs):
            # Cooperative cancel: a queued job aborts the moment a worker picks it
            # up. A job already running finishes (its threads can't be killed) but
            # its result is discarded by the pre-persist gate below.
            def _run():
                if cancel_event is not None and cancel_event.is_set():
                    raise AnalysisCancelledError("Analysis cancelled")
                return fn(*args, **kwargs)

            return _run

        max_workers = min(4, len(jobs))
        logger.info("Running %d top-level analysis job(s) with %d worker(s)", len(jobs), max_workers)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="team-analysis") as executor:
            futures = {}
            for kind, key, args, kwargs in jobs:
                for component_id, label in _job_progress(kind, key):
                    append_component_progress(
                        progress_list,
                        component_id=component_id,
                        label=label,
                        status="running",
                        phase="Discovering read-only repository scope" if kind == "code" else "",
                        read_only=kind == "code",
                    )
                if kind == "delivery":
                    future = executor.submit(_guarded(_run_delivery, *args, **kwargs))
                elif kind == "code":
                    future = executor.submit(_guarded(_run_ai_usage_component, *args, **kwargs))
                elif kind == "ops":
                    future = executor.submit(_guarded(run_operational, *args, **kwargs))
                else:
                    future = executor.submit(_guarded(_run_doc_quality_component, *args, **kwargs))
                futures[future] = (kind, key)

            for future in as_completed(futures):
                kind, key = futures[future]
                try:
                    result = future.result()
                except AnalysisCancelledError:
                    # Not a real failure — mark the component cancelled without
                    # polluting the warnings list.
                    for component_id, label in _job_progress(kind, key):
                        append_component_progress(
                            progress_list,
                            component_id=component_id,
                            label=label,
                            status="failed",
                            detail="cancelled",
                        )
                    continue
                except Exception as exc:
                    for component_id, label in _job_progress(kind, key):
                        append_component_progress(
                            progress_list,
                            component_id=component_id,
                            label=label,
                            status="failed",
                            detail=str(exc),
                        )
                    if kind == "delivery":
                        logger.warning("Delivery analysis failed for %s: %s", key, exc)
                        warnings.append(f"{_SOURCE_NAMES.get(key, key)} delivery analysis failed: {exc}")
                    else:
                        label = {"code": "Code", "ops": "Operations"}.get(kind, "Docs")
                        logger.warning("%s analysis failed: %s", label, exc)
                        warnings.append(f"{label} analysis failed: {exc}")
                    continue

                if kind == "delivery":
                    delivery_results[key] = result
                elif kind == "code":
                    signal, blob = result
                    if blob is not None:
                        code = {"signal": signal, "examples": blob}
                elif kind == "ops":
                    signal, blob = result
                    if blob is not None:
                        ops = {"signal": signal, "examples": blob}
                else:
                    signal, blob = result
                    if signal is not None:
                        docs = {"signal": signal, "examples": blob}

                for component_id, label in _job_progress(kind, key):
                    lifecycle_status = "completed"
                    detail = ""
                    if kind in {"code", "docs", "ops"}:
                        blob = result[1]
                        if blob is None:
                            lifecycle_status = "failed"
                            detail = "analysis failed"
                        else:
                            coverage_key = (
                                "activity_coverage" if component_id == "code:ai_footprint" else "coverage_report"
                            )
                            coverage_report = blob.get(coverage_key, {})
                            coverage_status = coverage_report.get("status", "complete")
                            lifecycle_status = {
                                "complete": "completed",
                                "partial": "partial",
                                "failed": "failed",
                                "no_data": "no_data",
                            }.get(coverage_status, "failed")
                            if kind == "code" and lifecycle_status == "completed":
                                if component_id == "code:ai_footprint":
                                    summary = blob.get("summary", {})
                                    detail = (
                                        f"{int(summary.get('scanned_commits', 0)):,} commits · "
                                        f"{int(summary.get('scanned_prs', 0)):,} authored PRs"
                                    )
                                else:
                                    health = blob.get("repository_health", {})
                                    detail = (
                                        f"{int(health.get('files_analysed', 0)):,} file records analysed · "
                                        f"{int(health.get('repositories_touched', 0)):,} repositories"
                                    )
                                    cached = int(health.get("cached_change_lookups", 0))
                                    if cached:
                                        detail += f" · {cached:,} cached changes reused"
                            if lifecycle_status != "completed":
                                detail = (
                                    f"{coverage_report.get('completed', 0):,}/"
                                    f"{coverage_report.get('eligible', 0):,} completed"
                                )
                                grouped_errors = coverage_report.get("grouped_errors") or []
                                if grouped_errors:
                                    error_detail = str(grouped_errors[0].get("detail", "")).strip()
                                    if error_detail:
                                        detail = f"{detail} · {error_detail}"
                    append_component_progress(
                        progress_list,
                        component_id=component_id,
                        label=label,
                        status=lifecycle_status,
                        detail=detail,
                    )

    # Pre-persist gate: even when a running job outlived the caller's bounded
    # wait, a set cancel_event guarantees nothing is saved.
    if cancel_event is not None and cancel_event.is_set():
        logger.info("Team analysis cancelled — discarding results; no profile or analysis run saved")
        raise AnalysisCancelledError("Analysis cancelled — nothing was saved.")

    # Futures complete in arbitrary order. Rebuild delivery in configured order so
    # the comparison table and TUI's initial tracker stay deterministic.
    delivery = {tracker: delivery_results[tracker] for tracker in comps["delivery"] if tracker in delivery_results}

    # Attach the global code/docs signals to every delivery profile, then persist.
    if delivery:
        _persist_delivery(delivery, code, docs, effective_db_path)
        # Analysis is a producer, so ``context`` narrows nothing here; it is
        # recorded on the profile's label row so the export that reads ceremony
        # history later runs under the scope this analysis was asked for.
        scope = coerce_scope(context)
        for tracker, sub in delivery.items():
            team_id = getattr(sub.get("profile"), "team_id", "") if isinstance(sub, dict) else ""
            if team_id:
                label_run(
                    "analysis",
                    team_id,
                    project_label=project_label,
                    tags=tags,
                    scope=scope,
                    defaults={"tracker_key": getattr(sub.get("profile"), "project_key", "") or ""},
                    db_path=effective_db_path,
                )
    for sub in delivery.values():
        warnings.extend(sub.get("warnings", []))

    if not delivery and code is None and docs is None and ops is None:
        # Nothing produced a result. If literally nothing is selected/available,
        # raise the canonical "no tracker configured" error; else a softer message.
        if not any(comps.values()):
            _resolve_source("")  # raises
        raise ValueError("Nothing to analyse — no component produced a result (see warnings).")

    ran = [t for t in delivery if delivery[t].get("profile") is not None]
    component_coverages: dict[str, dict] = {}
    if code:
        code_examples = code.get("examples", {})
        enabled_code = set(code_examples.get("enabled_features") or _CODE_FEATURES)
        if "ai_footprint" in enabled_code:
            component_coverages["ai_footprint"] = code_examples.get("activity_coverage", {})
        if "code_health" in enabled_code:
            component_coverages["code_health"] = code_examples.get("coverage_report", {})
    if docs:
        component_coverages["documentation"] = docs.get("examples", {}).get("coverage_report", {})
    if ops:
        component_coverages["operational"] = ops.get("examples", {}).get("coverage_report", {})
    incomplete = any(report.get("status") in {"partial", "failed"} for report in component_coverages.values())
    for name, report in component_coverages.items():
        if report.get("status") in {"partial", "failed"}:
            warnings.append(
                f"{name.replace('_', ' ').title()} coverage is {report.get('status')}: "
                f"{report.get('failed', 0)} failed, "
                f"{report.get('inaccessible', 0)} inaccessible, {report.get('truncated', 0)} truncated."
            )
    all_actions: list[dict] = []
    for payload in (code, docs, ops):
        if payload:
            all_actions.extend(payload.get("examples", {}).get("action_plan", []))
    result = {
        "delivery": delivery,
        "code": code,
        "docs": docs,
        "ops": ops,
        "comparison": _build_comparison(delivery) if len(ran) >= 2 else [],
        "components": comps,
        "analysis_features": features,
        "analysis_depth": analysis_depth,
        "analysis_window_days": analysis_window_days,
        "analysis_scope": analysis_scope or {},
        "coverage": {
            "status": (
                "failed"
                if incomplete
                and not delivery
                and not any(report.get("has_data") for report in component_coverages.values())
                else "partial"
                if incomplete
                else "complete"
            ),
            "components": component_coverages,
        },
        "action_plan": sorted(
            all_actions,
            key=lambda action: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(action.get("priority"), 9),
                -int(action.get("breadth", 1)),
                str(action.get("title", "")),
            ),
        ),
        "warnings": warnings,
    }
    from yeaboi.analysis.llm_runtime import get_analysis_llm_execution

    result["llm_execution"] = get_analysis_llm_execution()
    try:
        from yeaboi.team_profile import TeamProfileStore

        with TeamProfileStore(effective_db_path) as store:
            result["analysis_run_id"] = store.save_analysis_run(result)
    except Exception as exc:
        logger.warning("Could not persist normalized analysis run: %s", exc)
        warnings.append(f"Could not persist normalized analysis run: {exc}")
    return result


def _run_delivery(
    tracker: str,
    project_key: str,
    team_name: str,
    members: list[str] | None,
    sprint_count: int,
    generate_samples: bool,
    include_insights: bool,
    analysis_depth: str,
    progress: list,
    db_path,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Run the Delivery component for one tracker → a per-tracker result sub-dict.

    Fetches the board and runs the 4-worker parallel analysis (code/docs are NOT run
    here — they are separate global scans). Does NOT save; ``_persist_delivery``
    attaches the global code/docs signals and persists afterwards."""
    from yeaboi.tools.team_learning import (
        _fetch_azdevops_history,
        _fetch_jira_history,
        _run_parallel_analysis,
        compute_headline_stats,
    )

    started = time.monotonic()
    warnings: list[str] = []
    resolved_source = _resolve_source(tracker)
    resolved_project, resolved_team = _resolve_project(resolved_source, project_key, team_name)
    logger.info(
        "Delivery analysis: source=%s project=%s sprints=%d members=%s",
        resolved_source,
        resolved_project,
        sprint_count,
        members or "all",
    )
    fetch_started = time.monotonic()
    fetch = _fetch_jira_history if resolved_source == "jira" else _fetch_azdevops_history
    sprint_data = fetch(resolved_project, sprint_count, progress=progress, cancel_event=cancel_event)
    fetch_secs = time.monotonic() - fetch_started
    if not sprint_data:
        raise ValueError("No closed sprints found on the board — nothing to analyse.")
    sprint_names = [sd.get("sprint_name", "") for sd in sprint_data]

    analysis_started = time.monotonic()
    cache_updates: dict[str, tuple[str, dict]] = {}
    profile, examples = _run_parallel_analysis(
        resolved_source,
        resolved_project or "unknown",
        sprint_data,
        progress,
        include_ai_usage=False,
        include_doc_quality=False,
        members=members,
        warnings=warnings,
        analysis_depth=analysis_depth,
        include_insights=include_insights,
        db_path=db_path,
        cache_updates=cache_updates,
    )
    analysis_secs = time.monotonic() - analysis_started
    if resolved_team and not profile.team_name:
        from dataclasses import replace

        profile = replace(profile, team_name=resolved_team)

    duration = time.monotonic() - started
    examples["analysis_depth"] = analysis_depth
    insights = examples.get("insights") if include_insights else None
    samples = _generate_samples(profile, examples or {}, warnings) if generate_samples else None
    return {
        "source": resolved_source,
        "project_key": resolved_project,
        "sprint_names": sprint_names,
        "duration_secs": round(duration, 1),
        "profile": profile,
        "examples": examples,
        "headline_stats": compute_headline_stats(profile, examples),
        "insights": insights,
        "samples": samples,
        "analysis_depth": analysis_depth,
        "stage_timings": {
            "fetch_secs": round(fetch_secs, 1),
            "analysis_secs": round(analysis_secs, 1),
            "total_secs": round(duration, 1),
        },
        "_ticket_cache_updates": cache_updates,
        "log_path": "",
        "warnings": warnings,
    }


def _persist_delivery(delivery: dict, code: dict | None, docs: dict | None, db_path) -> None:
    """Attach the global code/docs signals to each delivery profile, save it, and
    write the analysis log. Scanning happens once; the same signal is written onto
    every tracker's profile so the stored-profile browser keeps rendering them."""
    from dataclasses import replace

    from yeaboi.paths import get_db_path
    from yeaboi.team_profile import TeamProfileStore
    from yeaboi.team_profile_exporter import write_analysis_log

    code_sig = code["signal"] if code else None
    docs_sig = docs["signal"] if docs else None
    with TeamProfileStore(db_path or get_db_path()) as store:
        for sub in delivery.values():
            profile = sub["profile"]
            examples = sub["examples"]
            cache_updates = sub.pop("_ticket_cache_updates", {})
            if cache_updates:
                store.save_ticket_parse_cache(profile.source, profile.project_key, cache_updates)
            if code_sig is not None:
                profile = replace(profile, ai_adoption=code_sig)
            if code is not None:
                # Move (not copy) the repo inventory to the top of the examples
                # blob. It is produced by the code scan but consumed by
                # planning, which should not have to know that — and a reader
                # that guesses at a nested path is a reader that silently finds
                # nothing. Popping it first matters: up to 300 rows carrying
                # descriptions would otherwise be serialised twice into the
                # same examples_json.
                ai_adoption = dict(code["examples"])
                inventory = ai_adoption.pop(repo_inventory.INVENTORY_KEY, None) or []
                examples["ai_adoption"] = ai_adoption
                if inventory:
                    examples[repo_inventory.INVENTORY_KEY] = inventory
            if docs_sig is not None:
                profile = replace(profile, doc_quality=docs_sig)
                examples["doc_quality"] = docs["examples"]
            sub["profile"] = profile
            store.save(profile, examples=examples)
            try:
                sub["log_path"] = str(
                    write_analysis_log(
                        profile,
                        examples=examples,
                        sprint_names=sub["sprint_names"],
                        duration_secs=sub["duration_secs"],
                    )
                )
            except Exception as exc:  # best-effort artifact
                logger.warning("Analysis log write failed: %s", exc)
                sub["warnings"].append(f"Analysis log not written: {exc}")
