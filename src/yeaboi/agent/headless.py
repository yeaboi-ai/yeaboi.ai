"""Headless planning-pipeline driver — run the full plan generation without a UI.

# See docs: "Agentic Blueprint Reference" — Core Graph Setup
# See docs: "MCP Server" — how external coding agents invoke this pipeline

The interactive REPL drives the LangGraph graph by prompting the user at every
review checkpoint. ``--export-only`` mode already auto-drives those checkpoints
by injecting synthetic inputs ("confirm", "accept", "continue"), but that
logic lives inside the ~1300-line ``run_repl`` loop, entangled with
prompt-toolkit and Rich rendering.

This module extracts the auto-drive into a plain function so callers that have
no terminal at all — the MCP server first among them — can run the pipeline
and get back the final graph state. The loop mirrors ``run_repl``'s
export-only branch exactly:

  - questionnaire awaiting confirmation  → inject "confirm"
  - a generation node set pending_review → clear review state, inject "accept"
  - capacity warning (sprint overflow)   → accept the recommended sprint count
  - anything else mid-pipeline           → inject "continue"
  - next node would be "agent"           → pipeline complete, stop

# See docs: "Memory & State" — stateless invocation requires manual history
# The graph is compiled without a checkpointer, so we thread the full state
# dict (messages + questionnaire + artifacts) between invoke() calls manually,
# just like the REPL does.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from langchain_core.messages import HumanMessage

from yeaboi.agent.state import QuestionnaireState, prior_art_refs
from yeaboi.context.labels import label_run
from yeaboi.context.resolve import scope_for
from yeaboi.context.scope import ContextScope

logger = logging.getLogger(__name__)


class HeadlessPipelineError(RuntimeError):
    """The pipeline could not be auto-driven to completion.

    Raised when the questionnaire is in a state that needs a human (an
    unanswered essential question mid-intake) or when the auto-drive loop
    exceeds ``max_steps`` without finishing — both indicate a bug or bad
    input rather than a transient failure, so callers should surface them.
    """


def _predict_next_node(state: dict) -> str:
    """Predict which graph node will run next, mirroring route_entry() logic.

    Used by the REPL to pick spinner messages and by the headless driver to
    detect pipeline completion. We duplicate the routing checks here (rather
    than importing route_entry) because route_entry expects a ScrumState
    TypedDict while callers work with a plain dict. The logic is
    intentionally kept in sync.
    """
    qs = state.get("questionnaire")
    if qs is None or not qs.completed:
        return "project_intake"
    if state.get("project_analysis") is None:
        return "project_analyzer"
    if not state.get("features"):
        analysis = state.get("project_analysis")
        if analysis and getattr(analysis, "skip_features", False):
            return "feature_skip"
        return "feature_generator"
    if not state.get("stories"):
        return "story_writer"
    if not state.get("tasks"):
        return "task_decomposer"
    if not state.get("sprints"):
        return "sprint_planner"
    return "agent"


def _next_auto_input(graph_state: dict) -> str | None:
    """Decide the synthetic input for the current state, or None when complete.

    An auto-responder over the same stage machine the chat drives
    (agent/chat_session.py): every gate a human would answer gets the answer
    nobody is there to give, plus the review-intercept bookkeeping that
    normally happens between prompts.

    Raises:
        HeadlessPipelineError: when the state needs a human to progress
            (questionnaire mid-intake, neither complete nor awaiting
            confirmation).
    """
    # Lazily, because chat_session reaches this module through streaming.
    from yeaboi.agent.chat_session import clear_review_state, stage_of

    stage = stage_of(graph_state)

    # Capacity warning — sprint_planner found the stories exceed the sprint
    # target and parked a negative "recommended count" in state. The chat asks
    # extend / grow the team / overload; headless always accepts the
    # recommendation, same as --export-only.
    if stage == "capacity":
        recommended = abs(graph_state.get("capacity_override_target", 0))
        logger.info("Capacity warning auto-accepted: %d sprints", recommended)
        graph_state["capacity_override_target"] = recommended
        return "accept recommended sprints"

    # Architecture-spike question — the node parked _spike_prompt because the
    # architecture decision is open and spike_choice is unset. Headless has
    # nobody to ask, so it applies the confidence auto-rule: validate unless
    # the analyzer's confidence is high. Callers override via
    # run_planning_pipeline(architecture_spike=...).
    if stage == "spike":
        from yeaboi.agent.nodes import spike_recommended

        spike_prompt = graph_state.get("_spike_prompt") or {}
        choice = "include" if spike_recommended(spike_prompt.get("confidence", "")) else "skip"
        logger.info("Spike question auto-answered: %s (confidence=%s)", choice, spike_prompt.get("confidence", ""))
        graph_state["spike_choice"] = choice
        graph_state["_spike_prompt"] = {}
        return f"{choice} the architecture spike"

    # Review checkpoint — a generation node produced artifacts and set
    # pending_review. The chat's intercept clears the review fields on accept
    # and re-invokes; we do the same. project_intake's pending_review is the
    # intake confirmation gate — the intake node itself consumes the "accept"
    # (via _is_confirm_intent), so only the pipeline checkpoints clear the
    # review-feedback fields here.
    pending = graph_state.get("pending_review")
    if pending:
        if pending == "project_intake":
            graph_state.pop("pending_review", None)
        else:
            clear_review_state(graph_state)
            if _predict_next_node(graph_state) == "agent":
                return None  # accepted the final artifact — plan complete
        return "accept"

    # The epic step is a chat affordance (it reformats before the feature
    # stage); headless never sets _epic_reviewed, so it reads as one more
    # pipeline step and gets the same "continue".
    if stage in ("pipeline", "epic"):
        return "continue"
    if stage == "chat":
        return None  # pipeline complete

    qs = graph_state.get("questionnaire")
    if isinstance(qs, QuestionnaireState) and qs.awaiting_confirmation:
        # Prior-art sub-loop — the intake is asking which existing repositories
        # are relevant. Headless has nobody to ask, so it answers "none":
        # accepting on the user's behalf would put repositories into a plan
        # nobody vetted, and "none" writes nothing to the ledger — an unpicked
        # batch is passed over for this run only, never suppressed. Callers
        # that DO know pass them in via run_planning_pipeline(prior_art=...).
        # A legacy "reason" stage (resumed old session) gets "none" too: the
        # node re-asks the batch on that input and the next pass answers it.
        # "empty" is not a question, so it falls through to the confirmation.
        if qs._prior_art_stage in ("ask", "reason"):
            logger.info("Headless: skipping the prior-art step (no user to ask)")
            return "none"

        # Intake summary shown — confirm it. Skipped/defaulted questions were
        # already resolved by build_questionnaire_from_answers().
        return "confirm"

    raise HeadlessPipelineError(
        "Questionnaire is mid-intake (not completed, not awaiting confirmation) — "
        "the headless pipeline needs a questionnaire built with "
        "build_questionnaire_from_answers() or an already-completed session."
    )


def run_planning_pipeline(
    questionnaire: QuestionnaireState,
    *,
    session_id: str | None = None,
    db_path: Path | None = None,
    save_session: bool = True,
    on_progress: Callable[[str, int], None] | None = None,
    max_steps: int = 40,
    prior_art: list[str] | None = None,
    ac_format: str = "",
    architecture_spike: str = "auto",
    solo: bool = False,
    context: ContextScope | dict | str | None = None,
    project_label: str = "",
    tags: Sequence[str] = (),
    integrations: Sequence[str] | None = None,
    refs: Sequence[dict] = (),
) -> dict:
    """Run the full planning pipeline headlessly and return the final graph state.

    Auto-accepts every review checkpoint (like ``--export-only``) and persists
    the session after each step so the result is resumable/inspectable from
    the TUI afterwards.

    Args:
        questionnaire: Intake answers, typically from
            questionnaire_io.build_questionnaire_from_answers(). Must be
            awaiting confirmation or already completed.
        session_id: Session row to write to. A fresh ID is minted when None.
        db_path: Sessions DB override (tests). Defaults to paths.get_db_path().
        save_session: When False, skip all SessionStore writes (dry runs).
        on_progress: Optional callback ``(node_name, step_index)`` invoked
            before each graph step — the MCP server forwards this to the
            client as progress notifications.
        max_steps: Safety cap on graph invocations; the happy path needs ~8.
        prior_art: Repository keys (``"github:acme/auth"``) to treat as
            accepted prior art — existing repositories the plan should build
            on. The interactive intake asks about these one at a time; a
            headless caller states them up front or gets none, because the
            step will not guess on a user's behalf. Unknown keys are used as
            given: this is an assertion by the caller, not a lookup.
        ac_format: Acceptance-criteria style override ("gwt" | "bullets").
            "" (default) resolves from YEABOI_AC_FORMAT / the learned team
            profile — see resolve_ac_style in agent/state.py.
        architecture_spike: Whether to add the architecture-validation spike
            when the analyzer's decision is open: "include" / "skip" force it,
            "auto" (default) applies the confidence rule (validate unless the
            analyzer's confidence is high). Irrelevant when the architecture
            is pinned or has a single option — nothing is added then.
        solo: A one-developer run (the Solo world). The intake defaults the
            team questions to one person and never offers a member picker,
            so the plan is sized for you alone.
        context: What this plan may read from other sessions — a
            ``ContextScope``, its dict twin or its spec string (``"all"``,
            ``"none"``, ``"standup,retro:1@2sprints"``). ``None`` reads as
            today: every source, no window.
        project_label: The free-text project label recorded on the session.
        tags: Tags recorded beside the defaults every plan gets.
        integrations: The connection keys this plan may consult (``None`` =
            every one). Seeds ``session_integrations``; see tools/risk.py.
        refs: References the plan reads during intake — ``{kind, label, id?,
            mode?, source?, subject?, url?}`` rows as agent/chat_refs.py
            validates them. Rendered once into ``pasted_context``.

    Returns:
        The final graph state dict (analysis, features, stories, tasks,
        sprints, questionnaire, messages) — feed it to
        json_exporter.export_plan_json() or the HTML/Markdown exporters.

    Raises:
        HeadlessPipelineError: if the state cannot be auto-driven or the loop
            exceeds max_steps.
        Exception: LLM/provider errors propagate to the caller (the MCP
            layer converts them into structured error payloads).
    """
    from yeaboi.agent.graph import create_graph
    from yeaboi.logging_setup import attach_session_log, detach_session_log
    from yeaboi.paths import get_db_path
    from yeaboi.sessions import SessionStore, make_session_id

    session_id = session_id or make_session_id()
    logger.info("Headless pipeline started: session=%s", session_id)
    attach_session_log(session_id)

    try:
        # Compile once — create_graph() validates topology and auto-loads the
        # tool belt; recompiling per step would waste ~seconds.
        graph = create_graph()

        # Close the prior-art step before the graph starts. `_next_auto_input`
        # answers "3" when the sub-loop opens, but by then the node has already
        # paid for it — five repository reads, five recursive tree walks and an
        # LLM call, all discarded, and any auth error from that call raised at
        # a caller who never asked for the step. Nobody to ask means nobody
        # pays for the asking; a caller who knows passes `prior_art=`.
        if isinstance(questionnaire, QuestionnaireState) and not questionnaire._prior_art_stage:
            questionnaire._prior_art_stage = "done"

        graph_state: dict = {
            "messages": [],
            "questionnaire": questionnaire,
            # Matches _run_headless: quick mode skips smart-intake follow-ups.
            "_intake_mode": "quick",
            "prior_art": prior_art_refs(prior_art),
        }
        if ac_format:
            from yeaboi.agent.state import AC_STYLES

            if ac_format not in AC_STYLES:
                raise HeadlessPipelineError(f"Unknown ac_format {ac_format!r} — use one of {AC_STYLES}.")
            # Seeding state["ac_format"] wins the resolve_ac_style precedence.
            graph_state["ac_format"] = ac_format
        if architecture_spike not in ("auto", "include", "skip"):
            raise HeadlessPipelineError(
                f"Unknown architecture_spike {architecture_spike!r} — use 'auto', 'include' or 'skip'."
            )
        if architecture_spike != "auto":
            # A pre-made choice means the spike question is never asked;
            # "auto" leaves it unset so _next_auto_input's confidence rule answers.
            graph_state["spike_choice"] = architecture_spike
        if solo:
            graph_state["solo"] = True
            logger.info("Headless: solo run — team questions default to one developer")
        scope = scope_for("planning", context)
        if scope is not None:
            # The JSON twin on state, so the nodes' _wants_dep and the saved
            # session both carry what this run may read.
            graph_state["context_scope"] = json.dumps(scope.to_dict(), sort_keys=True)
            logger.info("Headless: context scope %s", scope.to_spec())
        if integrations is not None:
            graph_state["session_integrations"] = list(integrations)
            logger.info("Headless: integrations restricted to %s", ", ".join(integrations) or "none")
        if refs:
            from yeaboi.agent.chat_refs import render_context_block, validate_refs

            checked = validate_refs(list(refs))
            graph_state["pasted_context"] = render_context_block(checked, db_path=db_path)
            logger.info("Headless: %d reference(s) read into the intake", len(checked))
        if project_label:
            graph_state["project_label"] = project_label

        store = SessionStore(db_path or get_db_path()) if save_session else None
        session_created = False
        project_name_recorded = False

        step = 0
        while True:
            injected = _next_auto_input(graph_state)
            if injected is None:
                logger.info("Headless pipeline complete: session=%s steps=%d", session_id, step)
                break
            if step >= max_steps:
                raise HeadlessPipelineError(
                    f"Pipeline did not complete within {max_steps} steps — "
                    f"stuck at node {_predict_next_node(graph_state)!r}."
                )

            invoke_state = {
                **graph_state,
                "messages": [*graph_state.get("messages", []), HumanMessage(content=injected)],
            }
            node_name = _predict_next_node(invoke_state)
            if on_progress is not None:
                on_progress(node_name, step)

            logger.info("Headless invoke: step=%d node=%s input=%r", step, node_name, injected)
            start = time.time()
            graph_state = graph.invoke(invoke_state)
            logger.info("Headless invoke done: node=%s (%.1fs)", node_name, time.time() - start)
            step += 1

            # Persist after every successful invoke — same best-effort pattern
            # as run_repl, so a crash mid-pipeline still leaves a resumable row.
            if store is not None:
                try:
                    if not session_created:
                        store.create_session(session_id)
                        session_created = True
                        label_run(
                            "planning",
                            session_id,
                            project_label=project_label,
                            tags=tags,
                            scope=scope,
                            defaults={"world": "solo" if solo else "team", "plan_size": "small_project"},
                            db_path=db_path,
                        )
                    store.save_state(session_id, graph_state)
                    if not project_name_recorded:
                        analysis = graph_state.get("project_analysis")
                        name = getattr(analysis, "project_name", "") if analysis else ""
                        if not name:
                            qs = graph_state.get("questionnaire")
                            if isinstance(qs, QuestionnaireState):
                                name = qs.answers.get(1, "")[:50]
                        if name:
                            store.update_project_name(session_id, name)
                            project_name_recorded = True
                    store.update_last_node(session_id, node_name)
                except Exception:
                    logger.warning("Session persistence failed (continuing)", exc_info=True)

        graph_state["_session_id"] = session_id
        return graph_state
    finally:
        detach_session_log()
