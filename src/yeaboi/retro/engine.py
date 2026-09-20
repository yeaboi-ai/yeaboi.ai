"""Retro engine — the one LLM call that turns retro cards into action items.

Like the standup engine, this is a standalone helper (NOT a LangGraph node): it
calls ``get_llm()`` directly and follows the same **parse → fallback** convention
the graph nodes use (agent/nodes.py). The team fills the "What didn't go well"
grid; this reads those cards (plus "What went well" for context) and appends
AI-suggested action items to the board's "Action items" grid.

An LLM auth/billing error is NOT re-raised — it is turned into a user-facing
status message and the deterministic fallback is used, so the retro never
crashes over a missing key (same policy as standup/engine.py).

# See docs: "The ReAct Loop" — using the LLM outside the main graph
# See docs: "Prompt Construction" — the retro action-items prompt
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from yeaboi.agent.state import RetroCard, RetroReport
from yeaboi.context.labels import label_run
from yeaboi.retro.board import CARRIED_OPEN_STATUSES, RetroBoard

if TYPE_CHECKING:
    from collections.abc import Sequence

    from yeaboi.context.resolve import Selection
    from yeaboi.context.scope import ContextScope

logger = logging.getLogger(__name__)


def carried_action_items_for_session(
    session_id: str,
    *,
    project_name: str = "",
    db_path: Path | None = None,
    selection: Selection | None = None,
) -> tuple[RetroCard, ...]:
    """Return the previous retro's action items for review, reset to ``pending``.

    The headless carry-forward entrypoint (the TUI + browser are adapters over it):
    finds "the retro before this one" and returns its ``action_items`` cards with
    ``status="pending"`` and ``origin="carryover"`` so the new board can seed its
    "Last sprint's actions" review column.

    "Previous retro" is resolved **across sessions**, not just this ``session_id``:
    retros run under auto-created quick sessions, so each one typically lands on a
    different session and a same-session lookup would almost always come up empty.
    We reuse ``RetroStore.get_recent_reports(limit, project_name)`` (the same
    cross-session, project-first primitive ``ceremony_history`` uses) and take the most
    recent recorded report. This intentionally carries forward across a reopen of the
    *same* session (close a retro, open it again → last run's actions appear) — at
    board-open the current run isn't recorded yet, so the newest report is always a
    genuinely prior retro. ``project_name`` biases toward the same project's retros;
    ``session_id`` is used only for logging.

    A ``selection`` with retros switched off carries nothing; its run ids
    replace the name bias with a hard filter. Graceful — returns an empty tuple
    when there's no prior retro or on any read error (never raises).

    # See AGENTS.md — Retro action-item carry-forward loop (mirrors Performance 1:1s)
    """
    if selection is not None and not selection.wants("retro"):
        logger.info("retro: carry-forward switched off for session=%s", session_id)
        return ()
    try:
        from yeaboi.paths import get_db_path
        from yeaboi.retro.store import RetroStore

        path = db_path or get_db_path()
        run_ids = selection.run_ids("retro") if selection is not None else None
        with RetroStore(path) as store:
            # Project-first, newest-first across ALL sessions (see docstring).
            reports = store.get_recent_reports(
                limit=5, project_name=project_name if run_ids is None else "", run_ids=run_ids
            )
    except Exception as exc:  # pragma: no cover - defensive; carry-forward is best-effort
        logger.warning("retro: could not load carried action items (session=%s): %s", session_id, exc)
        return ()

    # "The retro before this one" = the most recent recorded report (project-first via
    # get_recent_reports). At board-open the current run is NOT recorded yet, so the
    # newest report is always a genuinely prior retro — INCLUDING a reopen of the same
    # session, which is the common case: close a retro, open it again, and last run's
    # actions should carry forward. We deliberately do NOT skip same-session reports;
    # that guard used to eat the only prior report whenever retros reused the latest
    # quick session (each retro auto-creates a session that stays "latest"), so nothing
    # ever carried. ``session_id`` is kept for logging/telemetry only.
    prior_report = reports[0] if reports else None
    if prior_report is None:
        return ()
    return _carried_from_report(prior_report)


def standup_blocker_cards(
    selection: Selection | None, *, db_path: Path | None = None, existing: tuple[RetroCard, ...] = ()
) -> tuple[RetroCard, ...]:
    """The standup→retro edge: the selected standups' recent blockers as review cards.

    Returns dismissible ``pending`` cards (text badged ``[Standup]``), deduped
    against the ``existing`` carried cards they are seeded beside. An unscoped
    board (``selection`` None or unscoped) gets none — the team-wide board
    keeps its carry-forward-only seeding. Never raises.
    """
    from yeaboi.context.reads import recent_standup_blockers

    blockers = recent_standup_blockers(selection, db_path=db_path)
    if not blockers:
        return ()
    seen = {c.text.strip().lower() for c in existing}
    cards: list[RetroCard] = []
    for i, blocker in enumerate(blockers):
        text = f"[Standup] {blocker}"
        if text.lower() in seen:
            continue
        cards.append(
            RetroCard(
                id=f"standup-{i}",
                grid="action_items",
                text=text,
                author="Standup",
                origin="carryover",
                status="pending",
            )
        )
    if cards:
        logger.info("retro: %d standup blocker card(s) seeded", len(cards))
    return tuple(cards)


def record_retro_run(
    report: RetroReport,
    *,
    db_path: Path | None = None,
    project_label: str = "",
    tags: Sequence[str] = (),
    scope: ContextScope | dict | None = None,
) -> int:
    """Persist a finished retro and label it. The one record site every host calls."""
    from yeaboi.paths import get_db_path
    from yeaboi.retro.store import RetroStore

    path = db_path or get_db_path()
    with RetroStore(path) as store:
        run_id = store.record_run(report)
    label_run(
        "retro",
        report.session_id,
        run_id,
        project_label=project_label or report.project_name,
        tags=tags,
        scope=scope,
        db_path=path,
    )
    logger.info("retro: run %d recorded for session=%s", run_id, report.session_id)
    return run_id


def _carried_from_report(prior_report: RetroReport) -> tuple[RetroCard, ...]:
    from dataclasses import replace

    # Source = last retro's action_items grid PLUS any items it explicitly kept open in
    # its own review column (its carried_action_items with a still-open status). The
    # latter matters when the team marked something "Carried Over" but never clicked
    # Generate to re-add it to the grid — without this it would silently vanish. Dedup
    # by normalised text, grid items first.
    kept_open = [c for c in prior_report.carried_action_items if c.status in CARRIED_OPEN_STATUSES]
    seen: set[str] = set()
    combined = []
    for c in (*prior_report.by_grid().get("action_items", []), *kept_open):
        text = c.text.strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        combined.append(c)
    carried = tuple(replace(c, origin="carryover", status="pending") for c in combined)
    logger.info(
        "retro: %d carried-over action item(s) available from session %s",
        len(carried),
        prior_report.session_id,
    )
    return carried


def history_providers(
    *, project_name: str = "", db_path: Path | None = None, selection: Selection | None = None
) -> tuple[Callable[[], list[dict]], Callable[[int], dict | None]]:
    """Readers the browser board uses to step back through previous retros.

    Two callables rather than a store handle: the server never learns what a
    retro is persisted in, and a board with nothing behind it (a dev fixture, a
    retro run outside a session) simply reports no history.

    ``selection`` narrows both readers: with retros switched off they report
    nothing, and a run-narrowed selection only lists (and can only fetch by id)
    its own retros. Both are best-effort — a store that cannot be read is a
    board with no past, never a board that fails to load.
    """
    want = selection is None or selection.wants("retro")
    run_ids = selection.run_ids("retro") if selection is not None else None

    def _open() -> Any:
        from yeaboi.paths import get_db_path
        from yeaboi.retro.store import RetroStore

        return RetroStore(db_path or get_db_path())

    def listing() -> list[dict]:
        if not want:
            return []
        # Across sessions, like the carry-forward reader above and for the same
        # reason: a retro runs under whatever quick session was open that day,
        # so a same-session history is almost always empty. Project-first, so a
        # board for project X shows X's retros before anyone else's.
        try:
            with _open() as store:
                runs = store.get_all_history(limit=48, run_ids=run_ids)
        except Exception as exc:  # pragma: no cover - defensive; history is best-effort
            logger.warning("retro: could not list previous retros: %s", exc)
            return []
        # Already newest-first; a *stable* sort on the project match keeps it
        # that way inside each group.
        if project_name:
            runs.sort(key=lambda r: r.get("project_name") != project_name)
        return runs[:24]

    def one(run_id: int) -> dict | None:
        if not want or (run_ids is not None and run_id not in run_ids):
            return None
        try:
            with _open() as store:
                report = store.get_run_by_id(run_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("retro: could not read retro id=%s: %s", run_id, exc)
            return None
        return report_payload(report) if report else None

    return listing, one


def report_payload(report: RetroReport) -> dict:
    """A finished retro in the shape the browser board already renders.

    The same card fields the live poll sends, so a past retro is drawn by the
    same components — there is no second card renderer to keep in step.
    """
    return {
        "date": report.date,
        "sprint_name": report.sprint_name,
        "project_name": report.project_name,
        "participants": list(report.participants),
        "cards": [asdict(c) | {"mine": False} for c in report.cards],
        "carried": [asdict(c) | {"mine": False} for c in report.carried_action_items],
    }


def _parse_action_items(raw: str) -> list[str]:
    """Extract the action-item list from an LLM response, tolerating markdown fences."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[: raw.rfind("```")]
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("retro: could not parse LLM JSON response")
        return []
    items = parsed.get("action_items", []) if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        return []
    return [str(x).strip() for x in items if str(x).strip()]


def _build_fallback_action_items(didnt_go_well: list[str]) -> list[str]:
    """Deterministic action items when the LLM is unavailable.

    Turns each problem card into a plain "Address: <problem>" follow-up so the
    grid is never left empty just because AI is offline.
    """
    return [f"Address: {p}" for p in didnt_go_well[:6] if p.strip()]


def generate_action_items(board: RetroBoard) -> str:
    """Generate action items from the board's feedback and append them (origin="ai").

    Returns a short human-facing status message for the TUI (never raises).
    """
    grids = board.cards_by_grid()

    def _annotate(card) -> str:
        # Tag a card with its total reactions so the AI can weight team sentiment.
        total = sum(board.reaction_counts(card.id).values())
        return f"{card.text}  [{total} reactions]" if total else card.text

    # Raw text drives the deterministic fallback; reaction-annotated text drives the LLM.
    didnt_raw = [c.text for c in grids.get("didnt_go_well", [])]
    didnt = [_annotate(c) for c in grids.get("didnt_go_well", [])]
    went = [_annotate(c) for c in grids.get("went_well", [])]

    # Carry the loop forward: last sprint's actions the team marked "Carried Over" are
    # re-added to this sprint's grid (origin="carryover"); items still open (pending /
    # in-progress / carried-over) are handed to the LLM as context so it doesn't
    # duplicate them. Done / Not Relevant items are dropped.
    carried = board.carried_snapshot()
    carried_over_texts = [c.text for c in carried if c.status == "carried_over"]
    still_open = [c.text for c in carried if c.status in CARRIED_OPEN_STATUSES]

    if not didnt and not went:
        # A no-op click on an empty board must not mutate the grid.
        logger.info("retro: no feedback cards yet — nothing to generate")
        return "Add some cards first — no feedback to work from yet."

    if carried_over_texts:
        readded = board.add_carryover_cards(carried_over_texts)
        logger.info("retro: re-added %d carried-over action item(s) to the grid", readded)

    logger.info("retro: generating action items from %d problem / %d positive card(s)", len(didnt), len(went))

    from yeaboi.config import is_llm_configured

    configured, why = is_llm_configured()
    if not configured:
        logger.warning("retro: LLM not configured (%s) — using deterministic fallback", why)
        added = board.add_ai_cards(_build_fallback_action_items(didnt_raw))
        return f"AI unavailable ({why}) — added {added} basic action item(s)."

    # invoke_json tracks usage + turns on JSON mode + re-asks once on bad JSON.
    # See docs: "Local Mode (Ollama)" — reliability layer.
    from yeaboi.agent.llm import invoke_json
    from yeaboi.agent.nodes import _is_llm_auth_or_billing_error, _local_llm_hint
    from yeaboi.prompts.retro import get_retro_action_items_prompt

    prompt = get_retro_action_items_prompt(went_well=went, didnt_go_well=didnt, still_open=still_open)
    try:
        response = invoke_json(prompt, temperature=0.2)
        items = _parse_action_items(response.content)
    except Exception as exc:
        if _is_llm_auth_or_billing_error(exc):
            logger.warning("retro: LLM auth/billing error — surfacing as warning: %s", exc)
            added = board.add_ai_cards(_build_fallback_action_items(didnt_raw))
            return f"AI unavailable (API key/billing) — added {added} basic action item(s)."
        local_hint = _local_llm_hint(exc)
        if local_hint:
            logger.warning("retro: local Ollama failure: %s", exc)
            added = board.add_ai_cards(_build_fallback_action_items(didnt_raw))
            return f"{local_hint} Added {added} basic action item(s)."
        logger.warning("retro: LLM request failed, using fallback: %s", exc)
        added = board.add_ai_cards(_build_fallback_action_items(didnt_raw))
        return f"AI request failed — added {added} basic action item(s) (see logs)."

    if not items:
        added = board.add_ai_cards(_build_fallback_action_items(didnt_raw))
        return f"AI returned nothing usable — added {added} basic action item(s)."

    added = board.add_ai_cards(items)
    logger.info("retro: added %d AI action item(s)", added)
    return f"Generated {added} action item(s) from the team's feedback."
