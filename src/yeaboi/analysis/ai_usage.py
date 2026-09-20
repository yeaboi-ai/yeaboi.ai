"""AI-adoption footprint — detect how much of a team's tracked work shows an AI-tool trace.

# See docs: "Architecture" — engines are UI-free pipelines; this is a sub-analysis
# of team-analysis mode (AGENTS.md "REQUIRED: Surface Parity" — the TUI/CLI/MCP are
# thin adapters over ``analysis/engine.py:run_team_analysis``, which calls into here).

What this does
--------------
Fans out over the team's **remote** code sources (GitHub, Azure DevOps),
pulls recent commits + PRs *with their message bodies / descriptions*, and scans
that text for markers left by AI coding tools — ``Co-Authored-By: Claude``,
"Generated with Claude Code", Copilot's co-author line, Cursor / aider / Devin /
Codeium, and a catch-all AI trailer. It then aggregates a per-tool / per-author /
per-activity-type breakdown into an :class:`AiAdoptionSignal` and coaches the lead
on improving adoption (start / stop / keep / try).

Honesty contract — LOWER BOUND, never ground truth
--------------------------------------------------
Only tools that leave a *textual* trace in commit/PR metadata are counted. Inline
IDE assist (Copilot ghost-text, Cursor Tab) leaves no trace, so real usage is
always *at least* the reported footprint. Every surface must frame it that way;
``AiAdoptionSignal.is_lower_bound`` stays ``True`` to force it.

Error contract
--------------
Everything here is best-effort and NEVER raises: a missing SDK/credential or a
failing source contributes zero and is recorded as a coverage gap. ``run_ai_adoption``
wraps the whole thing so the analysis pipeline can call it unguarded.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from yeaboi.analysis import repo_inventory
from yeaboi.analysis.cancellation import AnalysisCancelledError
from yeaboi.team_profile import AiAdoptionSignal

logger = logging.getLogger(__name__)

# Default changed-content window. Callers can override this per run.
_SCAN_DAYS = 120
_CODE_COMPONENT_LABELS = {
    "ai_footprint": "Scanning selected-user AI footprint",
    "code_health": "Analysing selected-user code-change health",
}


def _code_workers(item_count: int) -> int:
    from yeaboi.config import get_team_analysis_code_max_concurrency

    return max(1, min(get_team_analysis_code_max_concurrency(), max(1, item_count)))


def _report_code_progress(
    progress: list | None,
    features: list[str] | tuple[str, ...] | set[str],
    *,
    phase: str,
    current: int | None = None,
    total: int | None = None,
    unit: str = "",
    secondary_count: int | None = None,
    secondary_unit: str = "",
) -> None:
    if progress is None:
        return
    from yeaboi.analysis.progress import append_component_progress

    for feature in features:
        label = _CODE_COMPONENT_LABELS.get(feature)
        if label:
            append_component_progress(
                progress,
                component_id=f"code:{feature}",
                label=label,
                status="running",
                phase=phase,
                current=current,
                total=total,
                unit=unit,
                secondary_count=secondary_count,
                secondary_unit=secondary_unit,
                read_only=True,
            )


# ---------------------------------------------------------------------------
# Marker table — extensible. Each entry: (tool_id, compiled regex over commit/PR text).
# Order matters: ``other_ai`` is the last-resort catch-all and is suppressed when a
# specific tool already matched (see _classify_ai_markers).
# ---------------------------------------------------------------------------
_AI_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "claude",
        re.compile(
            r"co-authored-by:\s*claude|generated with \[?claude code|noreply@anthropic\.com|claude\.com/claude-code",
            re.IGNORECASE,
        ),
    ),
    (
        "copilot",
        re.compile(
            r"github-copilot\[bot\]|co-authored-by:.*copilot|copilot@github\.com|gpt-4-copilot",
            re.IGNORECASE,
        ),
    ),
    # Tool markers below require an ATTRIBUTION shape (co-author trailer, bot
    # account, "generated with", agent domain), never a bare product name — a
    # commit that merely *mentions* "windsurf" or "aider" in prose is not AI
    # authorship, and precision matters more than recall for a lower bound.
    (
        "codex",
        # Codex cloud PR bodies carry a chatgpt.com/codex/tasks/... footer link.
        re.compile(
            r"co-authored-by:.*\bcodex\b|generated (?:with|by) codex|chatgpt\.com/codex|openai\.com/codex",
            re.IGNORECASE,
        ),
    ),
    (
        "cursor",
        re.compile(
            r"co-authored-by:.*\bcursor\b|generated (?:with|by) cursor|agent@cursor\.com|cursor\.com/agents",
            re.IGNORECASE,
        ),
    ),
    (
        "aider",
        # aider's real attribution forms: co-author trailer, "(aider)" author
        # suffix, "aider: " subject prefix, aider.chat links.
        re.compile(
            r"co-authored-by:.*\baider\b|\baider\.chat\b|\(aider\)|^\s*aider:\s",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "devin",
        re.compile(
            r"co-authored-by:.*\bdevin\b|devin-ai-integration\[bot\]|\bdevin\.ai\b",
            re.IGNORECASE,
        ),
    ),
    (
        "codeium",
        re.compile(
            r"co-authored-by:.*\b(codeium|windsurf)\b|\bcodeium\.com\b|\bwindsurf\.com\b"
            r"|generated (?:with|by) (?:codeium|windsurf)",
            re.IGNORECASE,
        ),
    ),
)

# Agent/bot AUTHOR identities — checked against the git author name and email,
# which the text markers above never see. Anchored full-string / suffix matches
# only, so a human named "Devin Smith" or a branch of prose never matches:
# these patterns describe ACCOUNT shapes, not words.
_AI_AUTHOR_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("claude", re.compile(r"^claude(\[bot\])?$|noreply@anthropic\.com$", re.IGNORECASE)),
    (
        "codex",
        re.compile(r"^(openai-)?codex(\[bot\])?$|^chatgpt-codex-connector(\[bot\])?$", re.IGNORECASE),
    ),
    (
        "copilot",
        re.compile(
            r"^(github-)?copilot(-swe-agent)?(\[bot\])?$|^copilot@github\.com$|copilot@users\.noreply\.github\.com$",
            re.IGNORECASE,
        ),
    ),
    ("cursor", re.compile(r"^cursor\s?agent$|@cursor\.com$", re.IGNORECASE)),
    ("aider", re.compile(r"\(aider\)$", re.IGNORECASE)),  # aider suffixes the human author name
    ("devin", re.compile(r"^devin-ai-integration(\[bot\])?$|@devin\.ai$", re.IGNORECASE)),
    ("gemini", re.compile(r"^google-labs-jules(\[bot\])?$", re.IGNORECASE)),
)

# Agent-created PR source branches — the strongest default trace some cloud
# agents leave (Codex cloud and the Copilot coding agent name their branches
# this way). Prefix-anchored so a human branch like "feature/codex-docs" or
# "fix-cursor-styles" never matches.
_AI_BRANCH_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("codex", re.compile(r"^codex/", re.IGNORECASE)),
    ("copilot", re.compile(r"^copilot/", re.IGNORECASE)),
    ("cursor", re.compile(r"^cursor/", re.IGNORECASE)),
    ("devin", re.compile(r"^devin/", re.IGNORECASE)),
)

# Catch-all handled in _classify_ai_markers (not the table): a co-author LINE
# whose name looks like an AI but matches no specific tool. Parsed per line so
# dependency-automation co-authors (dependabot[bot], renovate[bot], ...) can be
# excluded — they are automation, not AI codegen, and used to inflate the count.
_COAUTHOR_LINE = re.compile(r"^\s*co-authored-by:\s*(?P<name>[^<\n]+)", re.IGNORECASE | re.MULTILINE)
_OTHER_AI_NAME = re.compile(r"\b(ai|assistant|llm|gpt|chatgpt|openai|gemini|agent)\b", re.IGNORECASE)
_AUTOMATION_BOT_NAME = re.compile(
    r"\b(dependabot|renovate|greenkeeper|snyk|github-actions|imgbot|allcontributors|whitesource|mend"
    r"|pre-commit-ci|codecov|semantic-release|release-please|pyup)\b",
    re.IGNORECASE,
)

# A commit whose subject looks documentation-shaped is bucketed as "docs", not "code".
_DOCS_TITLE = re.compile(r"\breadme\b|\bdocs?/|\.md\b|\bdocumentation\b|\bchangelog\b", re.IGNORECASE)

# Human-readable source labels — the raw tags ("github"/"azdo") name the remote
# each scan hit. Single source of truth for renderers. Only remote sources are
# scanned (local-clone scanning was removed — it was environment-dependent and
# meaningless for a hosted team).
_SOURCE_LABELS: dict[str, str] = {
    "github": "GitHub (remote)",
    "azdo": "Azure DevOps (remote)",
}


def _source_label(tag: str) -> str:
    """Friendly label for a source tag ('github' → 'GitHub (remote)'); passthrough otherwise."""
    return _SOURCE_LABELS.get(tag, tag)


def _classify_ai_markers(text: str) -> set[str]:
    """Return the set of AI-tool ids whose markers appear in ``text``.

    Pure, no I/O — the core unit-test seam. ``other_ai`` fires only when NO
    specific tool matched (so a Claude commit with a generic ``Co-Authored-By``
    line is credited to "claude", not double-counted) and only for a co-author
    line whose name looks like an AI — dependency-automation bots (dependabot,
    renovate, ...) are excluded. Returns ``set()`` for empty text.
    """
    if not text:
        return set()
    hits: set[str] = set()
    for tool_id, pattern in _AI_MARKERS:
        if pattern.search(text):
            hits.add(tool_id)
    if not hits:
        for match in _COAUTHOR_LINE.finditer(text):
            name = match.group("name")
            if _OTHER_AI_NAME.search(name) and not _AUTOMATION_BOT_NAME.search(name):
                hits.add("other_ai")
                break
    return hits


def _classify_ai_authors(author: str, author_email: str = "") -> set[str]:
    """Tool ids whose agent/bot account shape matches the git author identity.

    Complements the text markers: an agent that commits under its own account
    (devin-ai-integration[bot], copilot-swe-agent, ...) often leaves a plain
    message with no trailer. Generic catch-all: an unknown ``[bot]`` name that
    looks like an AI (and is not dependency automation) counts as ``other_ai``.
    """
    name = (author or "").strip()
    email = (author_email or "").strip()
    if not name and not email:
        return set()
    hits: set[str] = set()
    for tool_id, pattern in _AI_AUTHOR_MARKERS:
        if pattern.search(name) or pattern.search(email):
            hits.add(tool_id)
    if not hits and name.lower().endswith("[bot]"):
        if _OTHER_AI_NAME.search(name) and not _AUTOMATION_BOT_NAME.search(name):
            hits.add("other_ai")
    return hits


def _classify_ai_item(item: dict) -> set[str]:
    """All AI-tool ids detectable on a normalized activity item.

    Union of text markers (title+body), author-account markers, and PR
    source-branch markers, with the same precedence rule as
    :func:`_classify_ai_markers`: a specific tool hit suppresses ``other_ai``.
    """
    hits = _classify_ai_markers(f"{item.get('title', '')}\n{item.get('body', '')}")
    hits |= _classify_ai_authors(str(item.get("author", "")), str(item.get("author_email", "")))
    branch = str(item.get("branch", "") or "")
    if branch:
        for tool_id, pattern in _AI_BRANCH_MARKERS:
            if pattern.search(branch):
                hits.add(tool_id)
    if len(hits) > 1:
        hits.discard("other_ai")
    return hits


# Below this many scanned items a percentage swings wildly on single commits —
# a member-scoped scan of a quiet fortnight can read "50% AI" off 2 of 4 items.
_MIN_FOOTPRINT_SAMPLE = 20


def footprint_small_sample(signal: AiAdoptionSignal) -> bool:
    """True when too little work was scanned for ``footprint_pct`` to be stable.

    Shared by every surface (TUI, CLI, exporters — Surface Parity) so they agree
    on when to show "N of M items" instead of a definitive-looking percentage.
    """
    return (signal.scanned_commits + signal.scanned_prs) < _MIN_FOOTPRINT_SAMPLE


def _activity_bucket(item: dict) -> str:
    """Map a normalized activity item to an adoption activity type: pr / docs / code."""
    if item.get("kind") == "pr":
        return "pr"
    if _DOCS_TITLE.search(str(item.get("title", ""))):
        return "docs"
    return "code"


def aggregate_ai_markers(items: list[dict]) -> AiAdoptionSignal:
    """Aggregate scanned commit/PR items into an :class:`AiAdoptionSignal`.

    Pure over its input (no network). Each item is a normalized activity dict with
    ``kind`` ('commit'/'pr'), ``author``, ``title``, optional ``body``/``branch``,
    and ``source``. An item is "AI-marked" when :func:`_classify_ai_item` finds a
    text, author-account, or source-branch marker. Returns an all-zero signal for
    an empty list.
    """
    scanned_commits = scanned_prs = ai_commits = ai_prs = 0
    per_tool: dict[str, int] = {}
    per_author: dict[str, int] = {}
    per_activity: dict[str, int] = {}
    per_source: dict[str, int] = {}
    sources: list[str] = []

    for item in items:
        kind = item.get("kind")
        is_pr = kind == "pr"
        if is_pr:
            scanned_prs += 1
        elif kind == "commit":
            scanned_commits += 1
        else:
            continue  # only commits/PRs carry an AI footprint

        src = str(item.get("source", "")).strip()
        if src and src not in sources:
            sources.append(src)

        tools = _classify_ai_item(item)
        if not tools:
            continue

        if is_pr:
            ai_prs += 1
        else:
            ai_commits += 1
        for t in tools:
            per_tool[t] = per_tool.get(t, 0) + 1
        author = (item.get("author") or "").strip() or "unknown"
        per_author[author] = per_author.get(author, 0) + 1
        bucket = _activity_bucket(item)
        per_activity[bucket] = per_activity.get(bucket, 0) + 1
        if src:
            per_source[src] = per_source.get(src, 0) + 1

    scanned = scanned_commits + scanned_prs
    footprint = round((ai_commits + ai_prs) / scanned * 100, 1) if scanned else 0.0

    def _sorted_pairs(d: dict[str, int]) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))

    return AiAdoptionSignal(
        scanned_commits=scanned_commits,
        scanned_prs=scanned_prs,
        ai_commits=ai_commits,
        ai_prs=ai_prs,
        footprint_pct=footprint,
        per_tool=_sorted_pairs(per_tool),
        per_author=_sorted_pairs(per_author),
        per_activity=_sorted_pairs(per_activity),
        per_source=_sorted_pairs(per_source),
        sources_scanned=tuple(sources),
        is_lower_bound=True,
    )


# ---------------------------------------------------------------------------
# Data gathering — graceful, best-effort fan-out (mirrors standup/collector.py)
# ---------------------------------------------------------------------------


def collect_ai_activity(
    source: str,
    project_key: str,
    sub_sources: list[str] | None = None,
    *,
    window_days: int = _SCAN_DAYS,
    analysis_scope: dict[str, list[str]] | None = None,
    progress: list[str] | None = None,
    code_features: list[str] | tuple[str, ...] | set[str] = ("ai_footprint", "code_health"),
    cancel_event: threading.Event | None = None,
    _return_inventory: bool = False,
) -> (
    tuple[list[dict], list[str], list[str], list[str]]
    | tuple[list[dict], list[str], list[str], list[str], list[dict], dict]
):
    """Fan out over GitHub + Azure DevOps (remote only) for recent commits/PRs with bodies.

    Returns ``(items, sources_scanned, coverage_notes, repos_scanned)``. Every source
    is best-effort and lazily imported (optional SDKs); a missing credential/SDK or a
    failing source contributes zero and is added to ``coverage_notes`` so absent
    coverage is visible rather than silent. ``repos_scanned`` holds friendly
    "what was actually scanned" labels (remote slug / project). Only remote sources
    are scanned — local-clone scanning was removed. ``sub_sources`` restricts which
    hosts to scan (subset of ``{"github", "azdo"}``; None = both). Never raises,
    except ``AnalysisCancelledError`` when ``cancel_event`` is set mid-scan.
    """
    from yeaboi.analysis.coverage import CoverageTracker, coverage_notes
    from yeaboi.config import (
        get_azure_devops_token,
        get_github_token,
        get_standup_github_repo,
        get_team_analysis_azdo_projects,
        get_team_analysis_github_owners,
    )

    def _want(tag: str) -> bool:
        return sub_sources is None or tag in sub_sources

    items: list[dict] = []
    sources_scanned: list[str] = []
    coverage: list[str] = []
    repos_scanned: list[str] = []
    inventory: list[dict] = []
    coverage_tracker = CoverageTracker("code", window_days)

    scope = analysis_scope or {}

    github_owners = scope.get("github") or list(get_team_analysis_github_owners())
    if _want("github") and github_owners and get_github_token():
        from yeaboi.tools.github import github_analysis_inventory, github_recent_commits, github_recent_prs

        legacy_repo = get_standup_github_repo()
        if not scope.get("github") and not os.getenv("TEAM_ANALYSIS_GITHUB_OWNERS") and legacy_repo:
            gh_inventory = [
                {
                    "provider": "github",
                    "container": legacy_repo.split("/", 1)[0],
                    "name": legacy_repo,
                    "active": True,
                    "paths": [],
                    "url": f"https://github.com/{legacy_repo}",
                    "default_branch": "",
                    "error": "",
                }
            ]
        else:
            try:
                gh_inventory = github_analysis_inventory(tuple(github_owners), days=window_days, include_trees=False)
            except TypeError:
                gh_inventory = github_analysis_inventory(tuple(github_owners), days=window_days)
        inventory.extend(gh_inventory)
        github_succeeded = False
        active_repos = [
            (index, repo)
            for index, repo in enumerate(gh_inventory)
            if not repo.get("discovery_error") and repo.get("active")
        ]
        _report_code_progress(
            progress,
            code_features,
            phase="Reading GitHub activity",
            current=0,
            total=len(active_repos),
            unit="repositories",
        )
        repo_results: dict[int, tuple[list[dict], Exception | None]] = {}

        def _read_github_repo(repo: dict) -> list[dict]:
            name = str(repo.get("name", ""))
            try:
                commit_items = github_recent_commits(name, days=window_days, include_changed_files=False)
                pr_items = github_recent_prs(name, days=window_days, include_changed_files=False, exhaustive=True)
            except TypeError:
                commit_items = github_recent_commits(name, days=window_days)
                pr_items = github_recent_prs(name, days=window_days)
            # A full-cap result means older in-window work went unread — surface it
            # through the coverage loop below ("truncated" status reads repo["error"]).
            from yeaboi.tools.github import _MAX_REPO_COMMITS, _MAX_REPO_PRS

            if len(commit_items) >= _MAX_REPO_COMMITS and not repo.get("error"):
                repo["error"] = f"commit scan capped at {_MAX_REPO_COMMITS} newest in window"
            if sum(1 for item in pr_items if item.get("kind") == "pr") >= _MAX_REPO_PRS and not repo.get("error"):
                repo["error"] = f"pull-request scan capped at {_MAX_REPO_PRS} most recently updated"
            return commit_items + pr_items

        if active_repos:
            with ThreadPoolExecutor(
                max_workers=_code_workers(len(active_repos)),
                thread_name_prefix="code-github",
            ) as executor:
                futures = {executor.submit(_read_github_repo, repo): (index, repo) for index, repo in active_repos}
                completed = 0
                for future in as_completed(futures):
                    if cancel_event is not None and cancel_event.is_set():
                        for pending in futures:
                            pending.cancel()
                        raise AnalysisCancelledError("Analysis cancelled")
                    index, _repo = futures[future]
                    try:
                        repo_results[index] = (future.result(), None)
                    except Exception as exc:
                        repo_results[index] = ([], exc)
                    completed += 1
                    _report_code_progress(
                        progress,
                        code_features,
                        phase="Reading GitHub activity",
                        current=completed,
                        total=len(active_repos),
                        unit="repositories",
                    )

        for index, repo in enumerate(gh_inventory):
            name = str(repo.get("name", ""))
            container = str(repo.get("container", ""))
            if repo.get("discovery_error"):
                coverage_tracker.add("github", container, name, "inaccessible", str(repo.get("error", "")))
                continue
            if not repo.get("active"):
                detail = str(repo.get("skip_reason") or "no changes in window")
                coverage_tracker.add("github", container, name, "unchanged", detail, eligible=False)
                continue
            raw, read_error = repo_results.get(index, ([], RuntimeError("repository activity was not attempted")))
            try:
                if read_error is not None:
                    raise read_error
                for item in raw:
                    item["source"] = "github"
                    item["repository"] = name
                    item["container"] = container
                    items.append(item)
                status = "truncated" if repo.get("error") else "succeeded"
                coverage_tracker.add("github", container, name, status, str(repo.get("error", "")))
                repos_scanned.append(f"GitHub (remote): {name}")
                github_succeeded = True
            except Exception as exc:
                coverage_tracker.add("github", container, name, "failed", str(exc))
        if github_succeeded:
            sources_scanned.append("github")
    elif _want("github"):
        coverage_tracker.add(
            "github",
            ",".join(github_owners) or "unconfigured",
            "repository estate",
            "inaccessible",
            "no GitHub token or owners — set Settings → GitHub, pass --github-owner, or pick owners in Analysis setup",
        )

    azdo_projects = scope.get("azdo") or list(get_team_analysis_azdo_projects())
    if source == "azdevops" and project_key and not scope.get("azdo"):
        azdo_projects = [project_key]
    if _want("azdo") and azdo_projects and get_azure_devops_token():
        from yeaboi.tools.azure_devops import (
            azdevops_analysis_inventory,
            azdevops_recent_commits,
            azdevops_recent_prs,
        )

        try:
            az_inventory = azdevops_analysis_inventory(tuple(azdo_projects), include_trees=False)
        except TypeError:
            az_inventory = azdevops_analysis_inventory(tuple(azdo_projects))
        inventory.extend(az_inventory)
        activity_by_repo: dict[tuple[str, str], int] = {}
        azure_inventory_by_project = {
            project: [repo for repo in az_inventory if str(repo.get("container", "")) == project]
            for project in azdo_projects
        }
        azure_repo_total = sum(not repo.get("discovery_error") for repo in az_inventory)
        azure_operations_total = azure_repo_total * 2
        azure_operations_completed = 0
        azure_progress_lock = threading.Lock()
        _report_code_progress(
            progress,
            code_features,
            phase="Reading Azure DevOps activity",
            current=0,
            total=azure_operations_total,
            unit="repository checks",
        )

        def _read_azure_project(project: str) -> tuple[str, list[dict]]:
            def _repo_completed(_current: int, _total: int) -> None:
                nonlocal azure_operations_completed
                with azure_progress_lock:
                    azure_operations_completed += 1
                    current = azure_operations_completed
                _report_code_progress(
                    progress,
                    code_features,
                    phase="Reading Azure DevOps activity",
                    current=current,
                    total=azure_operations_total,
                    unit="repository checks",
                )

            repositories = azure_inventory_by_project.get(project, [])
            try:
                raw = azdevops_recent_commits(
                    project,
                    days=window_days,
                    include_repository=True,
                    repositories=repositories,
                    progress_callback=_repo_completed,
                ) + azdevops_recent_prs(
                    project,
                    days=window_days,
                    include_repository=True,
                    repositories=repositories,
                    progress_callback=_repo_completed,
                )
            except TypeError:
                raw = azdevops_recent_commits(project, days=window_days, include_repository=True) + azdevops_recent_prs(
                    project, days=window_days, include_repository=True
                )
            return project, raw

        project_results: dict[str, list[dict]] = {}
        if azdo_projects:
            with ThreadPoolExecutor(
                max_workers=_code_workers(len(azdo_projects)),
                thread_name_prefix="code-azure-project",
            ) as executor:
                futures = {executor.submit(_read_azure_project, project): project for project in azdo_projects}
                for future in as_completed(futures):
                    if cancel_event is not None and cancel_event.is_set():
                        for pending in futures:
                            pending.cancel()
                        raise AnalysisCancelledError("Analysis cancelled")
                    project = futures[future]
                    try:
                        _project, raw = future.result()
                    except Exception as exc:
                        logger.warning("Azure activity collection failed for %s: %s", project, exc)
                        raw = []
                    project_results[project] = raw

        for project in azdo_projects:
            raw = project_results.get(project, [])
            for item in raw:
                item["source"] = "azdo"
                item["container"] = project
                items.append(item)
                rname = str(item.get("repository", "")).lower()
                activity_by_repo[(project, rname)] = activity_by_repo.get((project, rname), 0) + 1
        for repo in az_inventory:
            name = str(repo.get("name", ""))
            project = str(repo.get("container", ""))
            if repo.get("discovery_error"):
                coverage_tracker.add("azdo", project, name, "inaccessible", str(repo.get("error", "")))
                continue
            if activity_by_repo.get((project, name.lower()), 0) == 0:
                repo["active"] = False
                coverage_tracker.add("azdo", project, name, "unchanged", "no changes in window", eligible=False)
                continue
            status = "truncated" if repo.get("error") else "succeeded"
            coverage_tracker.add("azdo", project, name, status, str(repo.get("error", "")))
            repos_scanned.append(f"Azure DevOps (remote): {project}/{name}")
        if az_inventory:
            sources_scanned.append("azdo")
    elif _want("azdo"):
        coverage_tracker.add(
            "azdo",
            ",".join(azdo_projects) or "unconfigured",
            "repository estate",
            "inaccessible",
            "TEAM_ANALYSIS_AZDO_PROJECTS / AZURE_DEVOPS_TOKEN not set",
        )

    coverage_blob = coverage_tracker.as_dict()
    coverage.extend(coverage_notes(coverage_blob))
    if _return_inventory:
        return items, sources_scanned, coverage, repos_scanned, inventory, coverage_blob
    return items, sources_scanned, coverage, repos_scanned


def _identity_key(value: str) -> str:
    """Normalize display names, usernames, and email local-parts for strict matching."""
    return re.sub(r"[^a-z0-9]", "", (value or "").strip().lower())


def _filter_items_by_members(items: list[dict], members: list[str]) -> tuple[list[dict], dict, list[str], int]:
    """Keep commit/PR items authored by one of ``members`` or by an AI agent account.

    Matches a member name (case-insensitive) against the item's ``author`` OR the
    local-part of its ``author_email`` — the tracker's assignee display name and the
    git commit-author name are different identity spaces and often disagree, so we
    check both. Items authored by an AI agent/bot account (:func:`_classify_ai_authors`)
    are also retained: agents act on behalf of the selected members, and dropping
    them silently zeroes the very footprint this scan measures. The retention is
    disclosed via the returned count. Kept items are annotated with
    ``matched_members`` (human matches) or ``agent_authored`` (bot matches) so
    downstream per-member breakdowns don't have to re-derive the attribution.
    Returns
    ``(filtered_items, distinct_authors_matched, unmatched_members, agent_items_retained)``.
    """
    selected = [m.strip() for m in members if m and m.strip()]
    norm = {_identity_key(m): m for m in selected}
    if not norm:
        return [], {}, selected, 0
    kept: list[dict] = []
    matched: dict[str, set[str]] = {m: set() for m in selected}
    agents_retained = 0
    for it in items:
        author = (it.get("author", "") or "").strip()
        email = (it.get("author_email", "") or "").strip().lower()
        local = email.split("@", 1)[0] if email else ""
        candidates = {_identity_key(author), _identity_key(local)}
        member_keys = candidates & set(norm)
        if member_keys:
            it["matched_members"] = sorted(norm[key] for key in member_keys)
            kept.append(it)
            for key in member_keys:
                matched[norm[key]].add(author or local)
        elif _classify_ai_authors(author, email):
            it["agent_authored"] = True
            kept.append(it)
            agents_retained += 1
    resolved = {member: sorted(identities) for member, identities in matched.items() if identities}
    unmatched = [member for member in selected if member not in resolved]
    return kept, resolved, unmatched, agents_retained


def _dedupe_fanout_items(items: list[dict]) -> tuple[list[dict], int]:
    """Collapse fan-out automation: one piece of work pushed to many repos.

    Platform teams script the same maintenance commit/PR into hundreds of
    repositories under their own identities (observed live: one Dockerfile fix
    ×93 repos, one scripted PR per YL.Domain.* repo). Counting every copy
    multiplies the footprint denominator by the estate size and drowns the AI
    signal, so items with the same kind, author, message, and day collapse to
    the first occurrence regardless of repository. The survivor is annotated
    with ``fanout_copies`` and the collapsed total is returned for disclosure.
    """
    kept: list[dict] = []
    first_by_key: dict[tuple, dict] = {}
    collapsed = 0
    for it in items:
        title = str(it.get("title", "") or "")
        repo_short = str(it.get("repository", "") or "").rsplit("/", 1)[-1]
        if repo_short:
            # Azure DevOps titles carry a " (repo)" suffix; strip it so copies
            # of one message differ only by repository, not by title.
            title = title.removesuffix(f" ({repo_short})")
        author = (it.get("author", "") or "").strip()
        key = (
            str(it.get("kind", "")),
            _identity_key(author) or (it.get("author_email", "") or "").strip().lower(),
            title,
            str(it.get("body", "") or ""),
            str(it.get("timestamp", "") or "")[:10],
        )
        survivor = first_by_key.get(key)
        if survivor is None:
            first_by_key[key] = it
            kept.append(it)
        else:
            survivor["fanout_copies"] = survivor.get("fanout_copies", 1) + 1
            collapsed += 1
    return kept, collapsed


def run_ai_adoption(
    source: str,
    project_key: str,
    delivery_stories: list[dict],
    all_stories: list[dict],
    members: list[str] | None = None,
    sub_sources: list[str] | None = None,
    window_days: int = _SCAN_DAYS,
    analysis_scope: dict[str, list[str]] | None = None,
    progress: list[str] | None = None,
    code_features: list[str] | tuple[str, ...] | set[str] | None = None,
    db_path=None,
    generate_insights: bool = False,
    cancel_event: threading.Event | None = None,
) -> tuple[AiAdoptionSignal, dict]:
    """Orchestrate the AI-adoption scan: discover sources → collect → aggregate.

    Returns ``(signal, examples_blob)``. ``examples_blob`` carries the aggregated
    summary, up to ~20 illustrative samples for the report, and coverage notes.
    Wholly best-effort — any failure yields an empty signal and a coverage note,
    never an exception (the pipeline calls this unguarded). ``delivery_stories`` /
    ``all_stories`` are accepted for future ticket-derived repo discovery and to
    keep the signature stable; scanning currently uses configured code sources.

    ``members`` is authoritative: only commits and PRs authored by a selected,
    resolved identity are retained — plus items authored by AI agent/bot accounts
    (disclosed in coverage), since agents act on behalf of the selected members.
    An absent selection or zero identity matches produces an explicit empty
    result; it never broadens to whole-team *human* activity.
    """
    enabled_features = [
        feature for feature in ("ai_footprint", "code_health") if code_features is None or feature in code_features
    ]
    footprint_enabled = "ai_footprint" in enabled_features
    health_enabled = "code_health" in enabled_features
    logger.info(
        "run_ai_adoption: source=%s project=%s members=%s features=%s",
        source,
        project_key,
        members or "all",
        enabled_features,
    )
    try:
        # Dispatch on the collector's signature instead of catching TypeError from
        # the call: a genuine TypeError raised inside collection must propagate,
        # not silently rerun the whole scan with legacy (unscoped) semantics.
        # Integrations/tests may replace the collector with a legacy 3-arg stub.
        try:
            parameters = inspect.signature(collect_ai_activity).parameters
            legacy_collector = "cancel_event" not in parameters and not any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
            )
        except (TypeError, ValueError):
            legacy_collector = False
        if legacy_collector:
            collected = collect_ai_activity(source, project_key, sub_sources)
        else:
            collected = collect_ai_activity(
                source,
                project_key,
                sub_sources,
                window_days=window_days,
                analysis_scope=analysis_scope,
                progress=progress,
                code_features=enabled_features,
                cancel_event=cancel_event,
                _return_inventory=True,
            )
        if len(collected) == 4:
            items, sources_scanned, coverage, repos_scanned = collected
            inventory = []
            coverage_blob = {
                "component": "code",
                "status": "complete",
                "window_days": window_days,
                "discovered": len(repos_scanned),
                "eligible": len(repos_scanned),
                "attempted": len(repos_scanned),
                "succeeded": len(repos_scanned),
                "failed": 0,
                "unchanged": 0,
                "inaccessible": 0,
                "truncated": 0,
                "per_container": {},
                "assets": [],
            }
        else:
            items, sources_scanned, coverage, repos_scanned, inventory, coverage_blob = collected
        selected_users = [m.strip() for m in (members or []) if m and m.strip()]
        if selected_users:
            items, matched_identities, unmatched_users, agents_retained = _filter_items_by_members(
                items, selected_users
            )
            _report_code_progress(
                progress,
                enabled_features,
                phase="Matching selected-user activity",
                current=len(matched_identities),
                total=len(selected_users),
                unit="users",
                secondary_count=len(items),
                secondary_unit="activity records",
            )
            logger.info(
                "AI-usage member filter retained %d item(s) for %d/%d selected user(s)",
                len(items),
                len(matched_identities),
                len(selected_users),
            )
            if agents_retained:
                logger.info("AI-usage member filter retained %d agent-authored item(s)", agents_retained)
                coverage.append(
                    f"{agents_retained} agent-authored item(s) retained "
                    "(AI bot accounts act on behalf of the selected members)"
                )
            if unmatched_users:
                coverage.append("unmatched selected users: " + ", ".join(unmatched_users))
            if not items:
                coverage.append("no commits or authored PRs matched the selected users")
        else:
            items = []
            matched_identities = {}
            unmatched_users = []
            coverage.append("no users selected — code analysis requires an explicit member scope")
        # Repo provenance must reflect what the scan actually covered, so keep
        # the pre-dedup member-scoped list for it; everything else (footprint
        # numerator AND denominator, samples, activity summary, code-health
        # change lookups) counts distinct work only.
        member_scoped_items = items
        items, fanout_collapsed = _dedupe_fanout_items(items)
        if fanout_collapsed:
            logger.info("AI-usage fan-out dedup collapsed %d duplicate item(s)", fanout_collapsed)
            coverage.append(
                f"{fanout_collapsed} duplicate fan-out item(s) collapsed "
                "(same author, message, and day across multiple repositories)"
            )
        # Marker classification (analysis/aggregate.py) happens HERE, before
        # the change-metadata fetch, because the insights thread below consumes
        # signal+samples concurrently with that fetch.
        # Evidence is no longer a first-N sample: every AI-marked item is kept so
        # recommendations and JSON/MCP consumers can inspect the complete basis.
        if footprint_enabled:
            from yeaboi.analysis.aggregate import (
                build_classify_inputs,
                classify_markers,
                signal_from_wire,
            )

            classify_inputs = build_classify_inputs(items=items)
            classified = classify_markers(classify_inputs)
            signal = signal_from_wire(classified["signal"])
            samples = classified["samples"]
            _report_code_progress(
                progress,
                ("ai_footprint",),
                phase="Classifying detectable AI markers",
                current=signal.scanned_commits + signal.scanned_prs,
                total=signal.scanned_commits + signal.scanned_prs,
                unit="commits and PRs",
            )
        else:
            signal = AiAdoptionSignal()
            samples = []

        # Repo/source provenance onto the signal for honest, source-aware rendering.
        from dataclasses import replace

        touched_repositories = sorted(
            {
                (str(item.get("source", "")), str(item.get("container", "")), str(item.get("repository", "")))
                for item in member_scoped_items
                if item.get("repository")
            }
        )
        repos_scanned = [
            (
                f"GitHub (remote): {repository}"
                if provider == "github"
                else f"Azure DevOps (remote): {container}/{repository}"
            )
            for provider, container, repository in touched_repositories
        ]
        selected_sources = sorted({provider for provider, _container, _repository in touched_repositories})
        signal = replace(signal, sources_scanned=tuple(selected_sources), repos_scanned=tuple(repos_scanned))

        insights_executor = None
        insights_future = None
        if footprint_enabled and generate_insights and signal.scanned_commits + signal.scanned_prs > 0:
            _report_code_progress(
                progress,
                ("ai_footprint",),
                phase="Generating AI-footprint recommendations",
            )
            insights_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai-footprint-guidance")
            insights_future = insights_executor.submit(
                generate_ai_adoption_insights,
                signal,
                {"samples": samples},
                db_path=db_path,
            )
        change_requests: list[tuple[str, str, str, dict]] = []
        seen_changes: set[tuple[str, str, str, str]] = set()
        for provider, container, repository in touched_repositories if health_enabled else []:
            repo_activity = [
                item
                for item in items
                if item.get("source") == provider
                and str(item.get("container", "")) == container
                and str(item.get("repository", "")) == repository
            ]
            attributable_changes = [
                item
                for item in repo_activity
                if (item.get("kind") == "commit" and item.get("commit_id"))
                or (item.get("kind") == "pr" and item.get("pr_id"))
            ]
            for item in attributable_changes:
                change_id = str(item.get("commit_id") or f"pr:{item.get('pr_id', '')}")
                dedupe_key = (provider, container, repository, change_id)
                if change_id and dedupe_key not in seen_changes:
                    seen_changes.add(dedupe_key)
                    change_requests.append((provider, container, repository, item))

        def _change_cache_key(provider: str, container: str, repository: str, item: dict) -> str:
            identity = str(item.get("commit_id") or f"pr:{item.get('pr_id', '')}")
            version = "" if item.get("commit_id") else str(item.get("timestamp", ""))
            raw = "\0".join((provider, container, repository, identity, version))
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()

        def _cacheable_change(item: dict) -> bool:
            return bool(item.get("commit_id")) or str(item.get("status", "")).lower() in {
                "merged",
                "closed",
                "completed",
            }

        def _read_changed_files(request: tuple[str, str, str, dict]) -> list[dict]:
            provider, container, repository, item = request
            if provider == "github":
                from yeaboi.tools.github import github_changed_files

                return github_changed_files(repository, [item])
            from yeaboi.tools.azure_devops import azdevops_changed_files

            return azdevops_changed_files(container, repository, [item])

        changed_files: list[dict] = []
        change_results: dict[int, list[dict]] = {}
        pending: list[tuple[int, tuple[str, str, str, dict]]] = []
        changed_file_cache_hits = 0
        cache_store = None
        if health_enabled and db_path:
            try:
                from pathlib import Path

                from yeaboi.team_profile import TeamProfileStore

                cache_store = TeamProfileStore(Path(db_path))
            except Exception:
                logger.debug("Code-change cache unavailable", exc_info=True)
        try:
            for index, request in enumerate(change_requests):
                provider, container, repository, item = request
                cached = None
                if cache_store is not None and _cacheable_change(item):
                    try:
                        cached = cache_store.load_analysis_enrichment(
                            "code_changed_files",
                            _change_cache_key(provider, container, repository, item),
                            "provider-metadata-v1",
                        )
                    except Exception:
                        logger.debug("Code-change cache lookup failed", exc_info=True)
                if isinstance(cached, dict) and isinstance(cached.get("files"), list):
                    changed_file_cache_hits += 1
                    change_results[index] = [{**file, "cache_status": "hit"} for file in cached["files"]]
                else:
                    pending.append((index, request))

            # Global cold-run cap: each pending entry is one live API call. Cache
            # hits above are never capped, so warm re-runs keep full coverage;
            # newest-first ordering degrades quality toward older evidence.
            from yeaboi.config import get_team_analysis_max_change_lookups

            lookup_cap = get_team_analysis_max_change_lookups()
            skipped_lookups = 0
            if len(pending) > lookup_cap:
                pending.sort(key=lambda entry: str(entry[1][3].get("timestamp", "")), reverse=True)
                skipped_lookups = len(pending) - lookup_cap
                pending = pending[:lookup_cap]
                coverage.append(
                    f"code-change inspection capped at {lookup_cap} of {lookup_cap + skipped_lookups} "
                    "uncached changes (newest first; older changes were not inspected)"
                )
            lookup_total = len(change_requests) - skipped_lookups

            completed_changes = len(change_results)
            files_found = sum(len(files) for files in change_results.values())
            _report_code_progress(
                progress,
                ("code_health",) if health_enabled else (),
                phase="Reading code-change metadata",
                current=completed_changes,
                total=lookup_total,
                unit="changes inspected",
                secondary_count=files_found,
                secondary_unit="file records found",
            )
            if pending:
                with ThreadPoolExecutor(
                    max_workers=_code_workers(len(pending)),
                    thread_name_prefix="code-change-files",
                ) as executor:
                    futures = {
                        executor.submit(_read_changed_files, request): (index, request) for index, request in pending
                    }
                    for future in as_completed(futures):
                        if cancel_event is not None and cancel_event.is_set():
                            for pending_future in futures:
                                pending_future.cancel()
                            raise AnalysisCancelledError("Analysis cancelled")
                        index, request = futures[future]
                        provider, container, repository, item = request
                        try:
                            files = future.result()
                        except Exception as exc:
                            logger.warning(
                                "Code-change metadata failed for %s/%s: %s",
                                container,
                                repository,
                                exc,
                            )
                            files = [
                                {
                                    "provider": provider,
                                    "container": container,
                                    "repository": repository,
                                    "path": str(item.get("key", "unknown change")),
                                    "status": "failed",
                                    "error": str(exc),
                                }
                            ]
                        change_results[index] = files
                        if (
                            cache_store is not None
                            and _cacheable_change(item)
                            and not any(str(file.get("status", "")).lower() == "failed" for file in files)
                        ):
                            try:
                                cache_store.save_analysis_enrichment(
                                    "code_changed_files",
                                    _change_cache_key(provider, container, repository, item),
                                    "provider-metadata-v1",
                                    {"files": files},
                                )
                            except Exception:
                                logger.debug("Code-change cache checkpoint failed", exc_info=True)
                        completed_changes += 1
                        files_found += len(files)
                        _report_code_progress(
                            progress,
                            ("code_health",),
                            phase="Reading code-change metadata",
                            current=completed_changes,
                            total=lookup_total,
                            unit="changes inspected",
                            secondary_count=files_found,
                            secondary_unit="file records found",
                        )
            changed_files = [file for index in range(len(change_requests)) for file in change_results.get(index, [])]
        finally:
            if cache_store is not None:
                cache_store.close()

        # Annotate each item with the file paths its change touched so the
        # practices scorer can measure tests/docs hygiene. Items whose lookup
        # was cap-skipped or failed get NO annotation (not an empty list) —
        # they must stay out of the file-based denominators.
        if health_enabled:
            for index, (_prov, _cont, _repo, item) in enumerate(change_requests):
                files = change_results.get(index)
                if files is None:
                    continue
                ok_files = [file for file in files if str(file.get("status", "")).lower() != "failed"]
                if files and not ok_files:
                    continue
                item["changed_file_paths"] = [str(file.get("path", "")) for file in ok_files if file.get("path")]

        # The deterministic tail — code health, the per-member activity tally,
        # and practice hygiene (tests / docs / tickets / descriptions, the lead
        # signal of the card) — lives in analysis/aggregate.py.
        from yeaboi.analysis.aggregate import build_score_inputs, score_code

        score_inputs = build_score_inputs(
            items=items,
            changed_files=changed_files,
            selected_users=selected_users,
            window_days=window_days,
            health_enabled=health_enabled,
            changed_file_cache_hits=changed_file_cache_hits,
        )
        scored = score_code(score_inputs)
        health = scored["health"]
        file_reports = health["file_reports"]
        health_findings = health["findings"]
        action_plan = health["action_plan"]
        file_coverage = health["file_coverage"]
        repository_health = health["repository_health"]
        if health_enabled:
            _report_code_progress(
                progress,
                ("code_health",),
                phase="Evaluating code-change health",
                current=len(file_reports),
                total=len(file_reports),
                unit="file records",
            )
            coverage.extend(health["coverage_notes"])
        commit_count = scored["activity_counts"]["commits"]
        pr_count = scored["activity_counts"]["prs"]
        member_activity = scored["member_activity"]
        practices = scored["practices"]
        file_data = practices["file_data"]
        if health_enabled and file_data["total"] and file_data["with_file_data"] < file_data["total"]:
            coverage.append(
                f"file-level practice signals (tests, docs files) based on "
                f"{file_data['with_file_data']} of {file_data['total']} items with change metadata"
            )
        elif not health_enabled and file_data["total"]:
            coverage.append("file-level practice signals unavailable — code health disabled")
        blob: dict = {
            "enabled_features": enabled_features,
            "summary": {
                "scanned_commits": signal.scanned_commits if footprint_enabled else commit_count,
                "scanned_prs": signal.scanned_prs if footprint_enabled else pr_count,
                "ai_commits": signal.ai_commits,
                "ai_prs": signal.ai_prs,
                "footprint_pct": signal.footprint_pct,
                "per_tool": [list(p) for p in signal.per_tool],
                "per_author": [list(p) for p in signal.per_author],
                "per_activity": [list(p) for p in signal.per_activity],
                "per_source": [list(p) for p in signal.per_source],
                "repos_scanned": list(repos_scanned),
                "is_lower_bound": True,
                "small_sample": footprint_small_sample(signal) if footprint_enabled else False,
                "selected_users": selected_users,
                "matched_users": len(matched_identities),
                "unmatched_users": unmatched_users,
                "fanout_collapsed": fanout_collapsed,
            },
            "samples": samples,
            "coverage": coverage,
            "coverage_report": file_coverage if health_enabled else coverage_blob,
            "activity_coverage": coverage_blob,
            "selected_users": selected_users,
            "matched_identities": matched_identities,
            "unmatched_users": unmatched_users,
            "member_activity": member_activity,
            "member_practices": practices,
            "activity_summary": {
                "commits": commit_count,
                "authored_prs": pr_count,
                "reviews": scored["activity_counts"]["reviews"],
                "comments": scored["activity_counts"]["comments"],
                "repositories_touched": len(touched_repositories),
            },
            "changed_files": file_reports,
            "attribution": {
                "authored_commit": "high confidence",
                "authored_pr": "medium confidence; the PR may include collaborators' commits",
            },
            "repository_health": repository_health,
            # The estate itself, kept rather than discarded: planning reads this
            # back offline to shortlist the team's own repos as prior art for a
            # greenfield project. Trees are excluded here on purpose — see
            # analysis.repo_inventory.
            repo_inventory.INVENTORY_KEY: repo_inventory.normalise(inventory),
            "findings": health_findings,
            "action_plan": action_plan,
            "window_days": window_days,
        }
        if insights_future is not None:
            try:
                blob["insights"] = insights_future.result()
            finally:
                insights_executor.shutdown(wait=True)
        logger.info(
            "run_ai_adoption: scanned=%d ai=%d footprint=%.1f%% sources=%s",
            signal.scanned_commits + signal.scanned_prs,
            signal.ai_commits + signal.ai_prs,
            signal.footprint_pct,
            ",".join(sources_scanned) or "none",
        )
        return signal, blob
    except AnalysisCancelledError:
        # Cancellation is not a failure — propagate so the engine discards the run.
        raise
    except Exception:  # pragma: no cover - collect/aggregate already guard
        logger.exception("run_ai_adoption failed; returning empty signal")
        return AiAdoptionSignal(), {
            "enabled_features": enabled_features,
            "summary": {},
            "samples": [],
            "coverage": ["code analysis failed"],
            "coverage_report": {},
            "repository_health": {},
            "changed_files": [],
            "findings": [],
            "action_plan": [],
        }


def _collect_samples(items: list[dict], limit: int | None = 20) -> list[dict]:
    """AI-marked evidence items for the report (never bodies)."""
    out: list[dict] = []
    for item in items:
        tools = _classify_ai_item(item)
        if not tools:
            continue
        out.append(
            {
                "author": (item.get("author") or "").strip() or "unknown",
                "tool": sorted(tools)[0],
                "activity": _activity_bucket(item),
                "title": str(item.get("title", ""))[:80],
                "source": item.get("source", ""),
                "key": str(item.get("key", "")),
                "url": item.get("url", ""),
            }
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def _pick_sample(samples: list[dict], **filters) -> dict | None:
    """First sample matching all ``filters`` (e.g. activity="code"), or None."""
    for s in samples:
        if all(str(s.get(k, "")) == str(v) for k, v in filters.items()):
            return s
    return samples[0] if samples and not filters else None


def _sample_ref(sample: dict) -> str:
    """Short human reference to a sampled item, e.g. "commit a1b2c3d4 'Fix login' by Dinho"."""
    kind = "PR" if sample.get("activity") == "pr" else "commit"
    key = sample.get("key", "") or ""
    title = (sample.get("title", "") or "").strip()
    author = (sample.get("author", "") or "").strip()
    ref = f"{kind} {key}".strip()
    if title:
        ref += f" '{title}'"
    if author and author != "unknown":
        ref += f" by {author}"
    return ref


def _with_link(item: dict, sample: dict | None) -> dict:
    """Attach a best-effort ``link`` (the sample's url) to an insight item when present."""
    if sample and sample.get("url"):
        item["link"] = sample["url"]
    return item


# ---------------------------------------------------------------------------
# Coaching insights — start / stop / keep / try (mirrors team_learning insights)
# ---------------------------------------------------------------------------

_LOWER_BOUND_NOTE = (
    "This footprint is a lower bound — it only counts AI tools that leave a marker in "
    "commit messages or PR descriptions. Inline IDE assist (Copilot ghost-text, Cursor "
    "Tab) leaves no trace, so real usage is at least this."
)


def _fallback_ai_adoption_insights(signal: AiAdoptionSignal, samples: list[dict] | None = None) -> dict:
    """Deterministic AI-adoption coaching when the LLM is unavailable.

    Pure — no LLM, no I/O, never raises. Every category is guaranteed non-empty so
    the screen always has content. Framed as a lower bound throughout. When
    ``samples`` are given, relevant items cite a concrete example (with a link).
    """
    from yeaboi.tools.team_learning import _INSIGHT_MAX_ITEMS, _insight_item

    samples = samples or []
    footprint = signal.footprint_pct
    scanned = signal.scanned_commits + signal.scanned_prs
    top_tool = signal.per_tool[0][0] if signal.per_tool else ""
    n_authors = len(signal.per_author)
    activity = dict(signal.per_activity)

    start: list[dict] = []
    stop: list[dict] = []
    keep: list[dict] = []
    try_items: list[dict] = []

    # START — grow adoption where it's thin.
    if footprint < 25:
        start.append(
            _insight_item(
                "Adopt an AI pairing tool team-wide",
                "Only a small share of tracked work shows an AI marker. Pick one tool and "
                "roll it out so the whole team benefits, not just early adopters.",
                f"{footprint:.0f}% detectable AI footprint across {scanned} commits/PRs",
            )
        )
    if activity.get("pr", 0) == 0 and signal.ai_commits > 0:
        # Cite an actual AI-marked code commit so the advice points at real work.
        raw_commit = _pick_sample(samples, activity="code") or _pick_sample(samples)
        evidence = "AI shows up in commits but no PRs were scanned"
        if raw_commit:
            evidence = f"e.g. {_sample_ref(raw_commit)} — an AI-assisted commit with no PR"
        start.append(
            _with_link(
                _insight_item(
                    "Use AI to draft PR descriptions",
                    "AI shows up in commits but not PR descriptions. Move that work through a PR "
                    "and have authors generate a first-draft summary — it improves review context.",
                    evidence,
                ),
                raw_commit,
            )
        )
    if not start:
        start.append(
            _insight_item(
                "Standardise AI co-author trailers",
                "Agree on a Co-Authored-By convention so AI-assisted work is visible and "
                "this footprint reflects reality more closely.",
                _LOWER_BOUND_NOTE,
            )
        )

    # STOP — avoid mismeasurement / over-reliance blind spots.
    if any(t == "other_ai" for t, _ in signal.per_tool):
        stop.append(
            _insight_item(
                "Stop relying on unlabelled AI trailers",
                "Some commits carry a generic AI co-author with no tool name. Standardising "
                "the tool makes adoption measurable and reviewable.",
                "Generic 'AI' co-author trailers detected",
            )
        )
    if not stop:
        stop.append(
            _insight_item(
                "Don't treat this number as the whole picture",
                "Inline AI assist is invisible here. Avoid concluding low usage from the "
                "footprint alone — pair it with a quick team check-in.",
                _LOWER_BOUND_NOTE,
            )
        )

    # KEEP — reinforce what's working.
    if top_tool:
        tool_sample = _pick_sample(samples, tool=top_tool)
        evidence = f"{top_tool} is the most-seen tool across scanned work"
        if tool_sample:
            evidence = f"e.g. {_sample_ref(tool_sample)} ({top_tool})"
        keep.append(
            _with_link(
                _insight_item(
                    f"Your investment in {top_tool}",
                    "There is a consistent AI footprint on the team's work — keep sharing "
                    "prompts and workflows so the habit spreads.",
                    evidence,
                ),
                tool_sample,
            )
        )
    if footprint >= 40:
        keep.append(
            _insight_item(
                "A healthy AI-assisted cadence",
                "A large share of tracked work already shows an AI trace — keep the momentum "
                "and capture what's working in a short playbook.",
                f"{footprint:.0f}% detectable footprint",
            )
        )
    if not keep:
        keep.append(
            _insight_item(
                "Making AI-assisted work visible",
                "Even a partial footprint means the team is leaving a trail — keep tagging "
                "AI-assisted commits so adoption stays measurable.",
                f"{scanned} commits/PRs scanned",
            )
        )

    # TRY — experiments to broaden or deepen adoption.
    if n_authors and n_authors <= 3 and scanned > 0:
        try_items.append(
            _insight_item(
                "Run an AI-tooling brown-bag",
                "AI markers cluster on a few people. A 30-minute demo from an adopter often "
                "unblocks the rest of the team.",
                f"AI markers seen from {n_authors} author(s)",
            )
        )
    if activity.get("docs", 0) == 0:
        try_items.append(
            _insight_item(
                "Try AI for documentation, not just code",
                "No AI footprint on docs/README changes. Drafting docs with AI is a low-risk way to widen adoption.",
                "No AI markers on documentation-shaped commits",
            )
        )
    if not try_items:
        try_items.append(
            _insight_item(
                "A shared prompt library",
                "Collect the prompts your adopters use into a team doc — it turns individual "
                "wins into a repeatable practice.",
                _LOWER_BOUND_NOTE,
            )
        )

    return {
        "start": start[:_INSIGHT_MAX_ITEMS],
        "stop": stop[:_INSIGHT_MAX_ITEMS],
        "keep": keep[:_INSIGHT_MAX_ITEMS],
        "try": try_items[:_INSIGHT_MAX_ITEMS],
    }


def generate_ai_adoption_insights(signal: AiAdoptionSignal, examples: dict, *, db_path=None) -> dict:
    """Use the LLM to coach on AI adoption: start / stop / keep / try.

    Returns ``{"start": [...], "stop": [...], "keep": [...], "try": [...]}`` where
    each item is ``{"title", "detail", "evidence"}``. Falls back to deterministic
    insights on any failure — must never raise (runs inside the analysis pipeline).
    The prompt explicitly frames the footprint as a lower bound.
    """
    import json

    from yeaboi.tools.team_learning import _INSIGHT_KEYS, _INSIGHT_MAX_ITEMS, _insight_item, _llm_invoke

    samples = examples.get("samples", []) if isinstance(examples, dict) else []
    fallback = _fallback_ai_adoption_insights(signal, samples)

    # Valid link set — LLM-returned links are accepted only if they cite a real sample.
    valid_links = {str(s.get("url", "")) for s in samples if s.get("url")}

    per_tool = ", ".join(f"{t}={n}" for t, n in signal.per_tool) or "none detected"
    per_activity = ", ".join(f"{a}={n}" for a, n in signal.per_activity) or "none"
    per_source = ", ".join(f"{_source_label(s)}={n}" for s, n in signal.per_source) or "none"
    digest = (
        f"Scanned {signal.scanned_commits} commits and {signal.scanned_prs} PRs from sources: "
        f"{', '.join(_source_label(s) for s in signal.sources_scanned) or 'none'}.\n"
        f"AI-marked: {signal.ai_commits} commits, {signal.ai_prs} PRs "
        f"(detectable footprint {signal.footprint_pct:.0f}%).\n"
        f"By tool: {per_tool}.\n"
        f"By activity type: {per_activity}.\n"
        f"By source: {per_source}.\n"
        f"AI markers seen from {len(signal.per_author)} distinct author(s)."
    )

    # Concrete items the LLM can cite (with links) so coaching points at real work.
    example_lines = []
    for s in samples[:12]:
        ref = _sample_ref(s)
        url = s.get("url", "")
        example_lines.append(f"- [{_source_label(s.get('source', ''))}] {ref}" + (f" — {url}" if url else ""))
    examples_block = "\n".join(example_lines) or "(no illustrative samples available)"

    # See docs: "Prompt Construction" — ARC: Ask (coach adoption), Requirements
    # (categories, item shape, lower-bound honesty), Context (footprint digest).
    prompt = (
        "You are an engineering enablement coach helping a team lead grow effective, "
        "healthy use of AI coding tools. A scan of the team's commits and pull requests "
        "for AI-tool markers produced the digest below.\n\n"
        "CRITICAL framing: this footprint is a LOWER BOUND. It only counts AI tools that "
        "leave a textual marker in commit messages or PR descriptions; inline IDE assist "
        "(Copilot autocomplete, Cursor Tab) leaves no trace. Never claim the team does not "
        "use AI from a low number — coach on making usage more visible, broader, and more "
        "effective.\n\n"
        "Requirements:\n"
        '- Four categories: "start" (things to start), "stop" (things to stop/avoid), '
        '"keep" (things working well), "try" (experiments worth trying).\n'
        '- 2-4 items per category. Each item: "title" (imperative, max 10 words), '
        '"detail" (1-2 plain-English sentences of practical advice), "evidence" (one short '
        'phrase; where possible cite a specific example from the list below, e.g. "e.g. commit '
        'a1b2c3d4 \'Fix login\'"), and optionally "link" (the exact URL of that example, copied '
        "verbatim from the list — omit if none applies).\n"
        "- Prefer coaching that references a real example: 'here is where you did Y (link), do X "
        "instead'. Do NOT invent links; only use URLs from the list.\n"
        "- Ground every item in the digest. At least one item must remind the lead the "
        "footprint is a lower bound.\n\n"
        "## Footprint digest\n" + digest + "\n\n"
        "## Examples you can cite (use these exact URLs)\n" + examples_block + "\n\n"
        "Return ONLY a JSON object: "
        '{"start": [{"title": "...", "detail": "...", "evidence": "...", "link": "..."}], '
        '"stop": [...], "keep": [...], "try": [...]}'
    )

    try:
        from yeaboi.agent.llm import get_analysis_fast_model
        from yeaboi.config import get_llm_model, get_llm_provider

        fast_model = get_analysis_fast_model()
        cache_model = fast_model or get_llm_model() or f"{get_llm_provider()}:default"
        cache_key = hashlib.sha256(
            json.dumps(
                {"version": "ai-footprint-coaching-v2", "digest": digest, "examples": example_lines},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if db_path:
            from yeaboi.analysis.llm_runtime import record_analysis_cache_hit
            from yeaboi.team_profile import TeamProfileStore

            try:
                with TeamProfileStore(db_path) as store:
                    cached = store.load_analysis_enrichment("ai_footprint_coaching", cache_key, cache_model)
                if cached:
                    record_analysis_cache_hit(records=signal.scanned_commits + signal.scanned_prs)
                    return cached
            except Exception:
                logger.debug("AI-footprint coaching cache read failed", exc_info=True)

        from yeaboi.analysis.llm_runtime import record_analysis_input

        record_analysis_input(records=signal.scanned_commits + signal.scanned_prs)

        response = _llm_invoke(
            prompt,
            temperature=0.0,
            max_reasks=0,
            model=fast_model,
            task="ai_footprint_coaching",
            records=signal.scanned_commits + signal.scanned_prs,
            request_timeout=60,
        )
        text = response.content if hasattr(response, "content") else str(response)
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        result = json.loads(text)
        if isinstance(result, dict):
            insights: dict = {}
            for key in _INSIGHT_KEYS:
                raw = result.get(key)
                items = []
                if isinstance(raw, list):
                    for it in raw:
                        if isinstance(it, dict) and isinstance(it.get("title"), str) and it["title"].strip():
                            item = _insight_item(
                                it["title"].strip(),
                                it["detail"].strip() if isinstance(it.get("detail"), str) else "",
                                it["evidence"].strip() if isinstance(it.get("evidence"), str) else "",
                            )
                            # Accept a link only if it cites a real sample URL (no hallucinations).
                            link = it.get("link")
                            if isinstance(link, str) and link.strip() in valid_links:
                                item["link"] = link.strip()
                            items.append(item)
                insights[key] = items[:_INSIGHT_MAX_ITEMS] if items else fallback[key]
            logger.info(
                "LLM AI-adoption insights generated (%s)",
                ", ".join(f"{k}={len(v)}" for k, v in insights.items()),
            )
            if db_path:
                try:
                    with TeamProfileStore(db_path) as store:
                        store.save_analysis_enrichment("ai_footprint_coaching", cache_key, cache_model, insights)
                except Exception:
                    logger.debug("AI-footprint coaching cache write failed", exc_info=True)
            return insights
        logger.warning("LLM AI-adoption insights had unexpected shape; using fallback")
    except Exception as exc:
        logger.warning("LLM AI-adoption insights generation failed: %s", exc)

    return fallback
