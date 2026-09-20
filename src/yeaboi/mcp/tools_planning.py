"""MCP tools: the planning pipeline (intake contract, plan read/export/generate)."""

from __future__ import annotations

import dataclasses
import json
import logging

import anyio

# Context must be importable from module globals — FastMCP evaluates the
# stringified type hints (PEP 563) of tool functions against this namespace.
from mcp.server.fastmcp import Context

from yeaboi.mcp.runtime import context_kwargs, run_engine, run_readonly

logger = logging.getLogger(__name__)

# The auto-driven pipeline takes ~7 graph steps (confirm, analyzer, epics,
# stories, tasks, sprints, occasionally a capacity re-plan) — used as the
# progress denominator reported to the client.
_EXPECTED_PIPELINE_STEPS = 8


def _intake_questions() -> dict:
    from yeaboi.prompts.intake import (
        ESSENTIAL_QUESTIONS,
        INTAKE_QUESTIONS,
        PHASE_LABELS,
        QUESTION_DEFAULTS,
        QUESTION_METADATA,
        SMART_ESSENTIALS,
    )

    return {
        "questions": {str(number): text for number, text in INTAKE_QUESTIONS.items()},
        "defaults": {str(number): value for number, value in QUESTION_DEFAULTS.items()},
        "choice_metadata": {str(number): dataclasses.asdict(meta) for number, meta in QUESTION_METADATA.items()},
        "phases": PHASE_LABELS,
        "essential_questions": sorted(ESSENTIAL_QUESTIONS),
        "smart_essentials": sorted(SMART_ESSENTIALS),
        "usage": (
            "Ask the user the smart_essentials questions conversationally (plus Q1, the project "
            "description), then call plan_generate with the collected answers keyed by question "
            "number. Unanswered questions fall back to `defaults`."
        ),
    }


def _load_state(session_id: str) -> tuple[str, dict]:
    from yeaboi.mcp.tools_sessions import resolve_session_id
    from yeaboi.paths import get_db_path
    from yeaboi.sessions import SessionStore

    resolved = resolve_session_id(session_id)
    with SessionStore(get_db_path()) as store:
        state = store.load_state(resolved)
    if state is None:
        raise ValueError(f"Session not found or has no saved state: {resolved}")
    return resolved, state


def _plan_get(session_id: str) -> dict:
    from yeaboi.json_exporter import export_plan_json

    resolved, state = _load_state(session_id)
    plan = json.loads(export_plan_json(state))
    plan["session_id"] = resolved
    return plan


def _build_questionnaire(
    description: str,
    answers: dict | None,
    team_size: int,
    sprint_length_weeks: int,
    project_context: str,
):
    """Merge the tool's inputs into a confirmed-ready QuestionnaireState.

    Mirrors cli._run_headless: explicit answers win, then the convenience
    params, then keywords deterministically extracted from project_context
    (no LLM), then QUESTION_DEFAULTS via build_questionnaire_from_answers.
    """
    from yeaboi.questionnaire_io import build_questionnaire_from_answers

    if not description.strip():
        raise ValueError("description is required — a few sentences about the project.")

    merged: dict[int, str] = {1: description}
    if team_size:
        merged[6] = str(team_size)
    if sprint_length_weeks:
        merged[8] = str(sprint_length_weeks)

    # Deterministic keyword extraction fills gaps (tech stack, integrations,
    # infra) from free-form context — same mechanism the SCRUM.md file uses.
    if project_context.strip():
        try:
            from yeaboi.agent.nodes import _keyword_extract_fallback

            extracted: dict[int, str] = {}
            _keyword_extract_fallback(f"{description}\n{project_context}", extracted)
            for number, answer in extracted.items():
                merged.setdefault(number, answer)
        except Exception:
            logger.warning("project_context keyword extraction failed (continuing)", exc_info=True)

    # Explicit answers always win — the host agent gathered these from the user.
    for key, value in (answers or {}).items():
        try:
            number = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"answers keys must be question numbers 1-30, got {key!r}") from None
        if not 1 <= number <= 30:
            raise ValueError(f"answers keys must be question numbers 1-30, got {number}")
        merged[number] = str(value)

    return build_questionnaire_from_answers(merged)


def _plan_generate(
    description: str,
    answers: dict | None,
    team_size: int,
    sprint_length_weeks: int,
    project_context: str,
    prior_art: list[str] | None,
    ac_format: str,
    architecture_spike: str,
    solo: bool,
    context,
    project_label: str,
    tags: list | None,
    integrations: list | None,
    refs: list | None,
    on_progress,
) -> dict:
    from yeaboi.agent.headless import run_planning_pipeline
    from yeaboi.json_exporter import export_plan_json

    questionnaire = _build_questionnaire(description, answers, team_size, sprint_length_weeks, project_context)
    state = run_planning_pipeline(
        questionnaire,
        on_progress=on_progress,
        prior_art=prior_art,
        ac_format=ac_format,
        architecture_spike=architecture_spike or "auto",
        solo=solo,
        integrations=list(integrations) if integrations is not None else None,
        refs=list(refs or ()),
        **context_kwargs(context, project_label, tags),
    )
    plan = json.loads(export_plan_json(state))
    plan["session_id"] = state.get("_session_id", "")
    return plan


def _plan_export(session_id: str, format: str) -> dict:
    resolved, state = _load_state(session_id)
    if format == "html":
        from yeaboi.html_exporter import export_plan_html

        path = export_plan_html(state)
    elif format == "markdown":
        from yeaboi.repl._io import _export_plan_markdown

        path = _export_plan_markdown(state)
    elif format == "prd":
        from yeaboi.prd_exporter import build_prd_markdown, export_prd_markdown

        result = build_prd_markdown(state)
        path = export_prd_markdown(state, result=result)
        logger.info("PRD exported via MCP: session=%s path=%s llm_mode=%s", resolved, path, result.llm_mode)
        return {
            "session_id": resolved,
            "format": format,
            "path": str(path),
            "llm_mode": result.llm_mode,
            "warnings": list(result.warnings),
        }
    else:
        raise ValueError(f"Unsupported format {format!r} — use 'markdown', 'html' or 'prd'.")
    logger.info("Plan exported via MCP: session=%s format=%s path=%s", resolved, format, path)
    return {"session_id": resolved, "format": format, "path": str(path)}


def _plan_publish(session_id: str, destination: str, content: str = "plan") -> dict:
    if destination not in ("notion", "confluence"):
        raise ValueError(f"Unsupported destination {destination!r} — use 'notion' or 'confluence'.")
    if content not in ("plan", "prd"):
        raise ValueError(f"Unsupported content {content!r} — use 'plan' or 'prd'.")
    resolved, state = _load_state(session_id)
    from yeaboi.export_targets import publish_markdown

    name = getattr(state.get("project_analysis"), "project_name", "")
    warnings: list[str] = []
    if content == "prd":
        from yeaboi.prd_exporter import build_prd_markdown

        built = build_prd_markdown(state)
        title = f"PRD — {name}" if name else "PRD"
        markdown = built.markdown
        warnings = list(built.warnings)
    else:
        from yeaboi.repl._io import build_plan_markdown

        title = f"Sprint Plan — {name}" if name else "Sprint Plan"
        markdown = build_plan_markdown(state)
    result = publish_markdown(destination, title=title, markdown=markdown)
    if not result.ok:
        # publish_markdown never raises — surface its failure message as the
        # tool error so the agent gets the setup hint instead of a silent no-op.
        raise ValueError(result.message)
    logger.info(
        "Plan published via MCP: session=%s dest=%s content=%s url=%s", resolved, destination, content, result.url
    )
    return {
        "session_id": resolved,
        "destination": destination,
        "content": content,
        "url": result.url,
        "message": result.message,
        "warnings": warnings,
    }


def _plan_sync(session_id: str, destination: str, target_sprint: str = "", on_progress=None) -> dict:
    from yeaboi import trackers

    spec = trackers.by_key(destination)
    if spec is None:
        known = "', '".join(trackers.TRACKERS)
        raise ValueError(f"Unsupported destination {destination!r} — use '{known}'.")
    resolved, state = _load_state(session_id)
    if target_sprint.strip():
        # Route the whole plan into an existing sprint/iteration instead of
        # creating any: digits = a Jira sprint id, anything else = the sprint /
        # iteration name (resolved among active/future targets at sync time).
        # The keyword "backlog" creates the stories and assigns them nowhere.
        value = target_sprint.strip()
        state = dict(state)
        if value.lower() == "backlog":
            state["sprint_target_mode"] = "backlog"
            state["target_sprint_name"] = ""
            state["target_sprint_external_id"] = ""
        elif value.isdigit():
            state["sprint_target_mode"] = "existing"
            state["target_sprint_external_id"] = value
            state["target_sprint_name"] = ""
        else:
            state["sprint_target_mode"] = "existing"
            state["target_sprint_name"] = value
            state["target_sprint_external_id"] = ""
    sync_all = spec.sync_all()
    result, updated_state = sync_all(state, on_progress)
    # Persist the created-key mappings so a re-run skips what already exists
    # (the sync modules are idempotent through these state fields).
    from yeaboi.paths import get_db_path
    from yeaboi.sessions import SessionStore

    with SessionStore(get_db_path()) as store:
        store.save_state(resolved, updated_state)
    logger.info("Plan synced via MCP: session=%s dest=%s errors=%d", resolved, destination, len(result.errors))
    summary = spec.result_summary(result)
    return {
        "session_id": resolved,
        "destination": destination,
        "epic": summary["epic"],
        "stories_created": dict(result.stories_created),
        "tasks_created": dict(result.tasks_created),
        "sprints_created": summary["sprints_created"],
        "sprints_updated": summary["sprints_updated"],
        "skipped_existing": result.skipped,
        "warnings": list(result.errors),
    }


def _plan_prior_art(description: str, answers: dict | None, profile_id: str) -> dict:
    """Shortlist the team's own repositories as prior art for a new project."""
    from yeaboi.agent import prior_art as engine

    if not description.strip():
        raise ValueError("description is required — a few sentences about the project.")
    merged: dict[int, str] = {1: description, 2: "Greenfield"}
    for key, value in (answers or {}).items():
        try:
            number = int(key)
        except (TypeError, ValueError):
            raise ValueError(f"answers keys must be question numbers 1-30, got {key!r}") from None
        merged[number] = str(value)

    result = engine.shortlist(merged, profile_id=profile_id)
    return {
        "candidates": [
            {
                "key": c.key,
                "name": c.name,
                "platform": c.platform,
                "url": c.url,
                "pitch": list(c.pitch),
                "stack": list(c.stack),
                "last_activity": c.last_activity,
            }
            for c in result.candidates
        ],
        "empty_reason": result.empty_reason,
        "message": result.message,
    }


def _plan_prior_art_feedback(repo_key: str, verdict: str, reason: str, repo_name: str) -> dict:
    """Record that a repository is (or is not) useful prior art."""
    from yeaboi.agent import prior_art_feedback

    if verdict not in prior_art_feedback.VERDICTS:
        raise ValueError(f"verdict must be one of {prior_art_feedback.VERDICTS}, got {verdict!r}")
    recorded = prior_art_feedback.apply_verdict(
        repo_key=repo_key,
        verdict=verdict,
        reason=reason,
        repo_name=repo_name,
    )
    if not recorded:
        raise ValueError(f"Could not record a verdict for {repo_key!r} — check the key is non-empty.")
    return {"repo_key": repo_key.strip().lower(), "verdict": verdict, "recorded": True}


def register(app) -> None:
    """Attach the planning tools to the FastMCP app."""

    @app.tool()
    async def plan_generate(
        description: str,
        ctx: Context,
        answers: dict | None = None,
        team_size: int = 0,
        sprint_length_weeks: int = 0,
        project_context: str = "",
        prior_art: list[str] | None = None,
        ac_format: str = "",
        architecture_spike: str = "auto",
        solo: bool = False,
        context: str | dict | None = None,
        project_label: str = "",
        tags: list[str] | None = None,
        integrations: list[str] | None = None,
        refs: list[dict] | None = None,
    ) -> dict:
        """Generate a full sprint plan (analysis, epics, stories, tasks, sprints) from a project
        description. Gather the intake_questions smart_essentials from the user first and pass
        them as `answers` {question_number: answer}; `project_context` takes free-form notes
        (tech stack, constraints, goals). Takes a few minutes — several LLM calls. The plan is
        saved as a session (see data.session_id) for plan_get/plan_export and the other modes.
        `prior_art` takes repository keys from plan_prior_art that the user confirmed are
        relevant — pass only what they approved; the plan builds on them.
        `ac_format`: acceptance-criteria style — 'gwt' (Given/When/Then) or 'bullets'
        (clear testable statements); empty follows the learned team profile.
        `architecture_spike`: when the analyzer's architecture decision is open (2+ options),
        whether to add a validation spike — 'include' / 'skip', or 'auto' (default: add it
        unless the analyzer's confidence is high).
        `solo`: true when the user is running their own delivery with no team — the team
        questions default to one developer and no member picker is offered.
        `context`: what this run may read from other sessions — 'all' (default), 'none', or a
        spec like 'standup,retro:1@2sprints project=apollo tags=q3' (sources, an optional
        window, project labels and tags); a JSON object of the same shape is accepted. Call
        context_preview first when the user names a timeframe. `project_label` is the free-text
        project label recorded on the run; `tags` are recorded beside its default tags.
        `integrations`: the connection keys (jira, github, notion, …) this plan may consult —
        omit for every connected one, [] for none. `refs`: what the plan should read first,
        as {kind: plan|run|integration|link, label, id?, mode?, source?, subject?, url?} rows —
        another plan's session id, a mode's run (a run with no id is that mode's latest), an
        item from an integration, or a URL."""

        def report(node_name: str, step: int) -> None:
            # Called from the engine's worker thread — bridge the async
            # progress notification back to the server's event loop.
            try:
                anyio.from_thread.run(ctx.report_progress, step, _EXPECTED_PIPELINE_STEPS, node_name)
            except Exception:
                logger.debug("progress report failed (continuing)", exc_info=True)

        return await run_engine(
            ctx,
            _plan_generate,
            description,
            answers,
            team_size,
            sprint_length_weeks,
            project_context,
            prior_art,
            ac_format,
            architecture_spike,
            solo,
            context,
            project_label,
            tags,
            integrations,
            refs,
            report,
        )

    @app.tool()
    async def plan_prior_art(ctx: Context, description: str, answers: dict | None = None, profile_id: str = "") -> dict:
        """Shortlist the team's OWN existing repositories that could help a new greenfield
        project, each with why it might be relevant. Candidates come from the saved
        team-analysis profile; the shortlist is then enriched from GitHub (needs a token to go
        beyond the stored row) and pitched by the LLM. Show the user the list and ask which are
        actually relevant before passing the approved keys to plan_generate's `prior_art`. When
        `empty_reason` is set, relay `message` — it tells the user what to do about it."""
        # Not `run_readonly`: that path is for deterministic work — it reports
        # `llm_mode: "n/a"`, skips the engine lock and never injects the
        # sampling model. The pitch step calls the LLM, so a sampling-only host
        # would silently fall back to deterministic bullets while the envelope
        # still claimed no model was involved.
        return await run_engine(ctx, _plan_prior_art, description, answers, profile_id)

    @app.tool()
    async def plan_prior_art_feedback(repo_key: str, verdict: str, reason: str = "", repo_name: str = "") -> dict:
        """Record the user's verdict on a prior-art suggestion (verdict: 'up' or 'down').
        A 'down' permanently stops that repository being suggested for any future project, so
        only call it when the user actually said it is not relevant — pass their reason."""
        return await run_readonly(_plan_prior_art_feedback, repo_key, verdict, reason, repo_name)

    @app.tool()
    async def intake_questions() -> dict:
        """Get yeaboi's intake contract: the 30 planning questions, which are essential, their
        defaults and choice options. Call this before gathering answers for plan_generate."""
        return await run_readonly(_intake_questions)

    @app.tool()
    async def plan_get(session_id: str = "") -> dict:
        """Get a saved sprint plan as JSON (analysis, epics, stories, tasks, sprints).
        Blank session_id = most recent session."""
        return await run_readonly(_plan_get, session_id)

    @app.tool()
    async def plan_export(ctx: Context, session_id: str = "", format: str = "markdown") -> dict:
        """Export a saved plan to a file and return its path. format: 'markdown', 'html' or
        'prd' — 'prd' generates a full Product Requirements Document (one LLM call for the
        prose sections; deterministic skeleton when no LLM is available, see llm_mode).
        Blank session_id = most recent session."""
        if format == "prd":
            # run_readonly reports llm_mode "n/a" and never injects the sampling
            # model — the PRD's prose call must go through run_engine.
            return await run_engine(ctx, _plan_export, session_id, format)
        return await run_readonly(_plan_export, session_id, format)

    @app.tool()
    async def plan_publish(
        ctx: Context, session_id: str = "", destination: str = "notion", content: str = "plan"
    ) -> dict:
        """Publish a saved plan as a page in the user's configured Notion or Confluence
        (destination: 'notion' or 'confluence') and return the page URL. content: 'plan'
        (the sprint plan) or 'prd' (a full Product Requirements Document — one LLM call).
        This creates a page in an external workspace — confirm with the user before calling.
        Blank session_id = most recent session."""
        if content == "prd":
            return await run_engine(ctx, _plan_publish, session_id, destination, content)
        return await run_readonly(_plan_publish, session_id, destination, content)

    @app.tool()
    async def plan_sync(ctx: Context, session_id: str = "", destination: str = "jira", target_sprint: str = "") -> dict:
        """Push a saved plan into the user's issue tracker (destination: 'jira',
        'azdevops', 'linear' or 'trello'):
        creates the epic, stories, tasks and sprints/iterations as REAL tickets on the
        configured board — always confirm with the user before calling. Idempotent: items
        created by an earlier sync are skipped, so a partial run can be safely retried.
        target_sprint: "backlog" creates the stories without assigning them to any
            sprint; otherwise add the plan's stories to this EXISTING active/future sprint instead
        of creating sprints — a Jira sprint id (digits) or the sprint/iteration name.
        Blank session_id = most recent session."""

        def report(current: int, total: int, label: str) -> None:
            # Called from the sync's worker thread — bridge the async progress
            # notification back to the server's event loop.
            try:
                anyio.from_thread.run(ctx.report_progress, current, total or None, label)
            except Exception:
                logger.debug("sync progress report failed (continuing)", exc_info=True)

        return await run_engine(ctx, _plan_sync, session_id, destination, target_sprint, report, needs_llm=False)
