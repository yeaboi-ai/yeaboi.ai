"""Session store for yeaboi: persists session metadata and state to SQLite.

Each terminal session gets a unique internal ID (new-<8hex>-<YYYY-MM-DD>).
Once the project name is known (after the analyzer node runs), the display
name is derived as <project-slug>-<YYYY-MM-DD> for human readability.

Phase 8A stores metadata (project name, timestamps, last node).
Phase 8B adds full state serialisation for --resume: questionnaire answers,
project analysis, features, stories, tasks, sprints, and all scalar fields are
persisted as JSON so interrupted sessions can be resumed from where they left off.

# See docs: "Memory & State" — MemorySaver, thread_id, session persistence
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import asdict
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4

from yeaboi.agent.state import (
    AcceptanceCriterion,
    Discipline,
    Feature,
    OutputFormat,
    Priority,
    ProjectAnalysis,
    QuestionnaireState,
    ReviewDecision,
    Sprint,
    StoryPointValue,
    Task,
    TaskLabel,
    UserStory,
    prior_art_from_dicts,
    prior_art_to_dicts,
)
from yeaboi.context.labels import SESSION_LABELS_SCHEMA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session ID helpers
# ---------------------------------------------------------------------------


def make_session_id() -> str:
    """Generate a stable, collision-resistant internal session ID.

    Format: new-<8 hex chars>-<YYYY-MM-DD>
    The UUID prefix ensures uniqueness even when the same project is run
    multiple times on the same day (see Phase 8B collision handling).
    """
    return f"new-{uuid4().hex[:8]}-{date.today().isoformat()}"


def make_display_name(meta: dict) -> str:
    """Derive a human-readable session name from a metadata row.

    When a project name is known: <project-slug>-<YYYY-MM-DD>
    Otherwise: the raw session_id (e.g. "new-a3f91b2c-2024-03-06").

    Args:
        meta: Dict with keys session_id, project_name, created_at.

    Returns:
        A short human-readable label for the session.
    """
    project_name = meta.get("project_name", "")
    created_at = meta.get("created_at", "")
    if project_name and created_at:
        slug = re.sub(r"[^a-z0-9]+", "-", project_name.lower()).strip("-")[:40] or "project"
        date_part = created_at[:10]  # ISO date: YYYY-MM-DD
        return f"{slug}-{date_part}"
    return meta.get("session_id", "unknown")


#: Where a provisional plan title is cut when the description runs long.
PROVISIONAL_TITLE_CHARS = 60


def provisional_title(description: str, *, limit: int = PROVISIONAL_TITLE_CHARS) -> str:
    """A plan's name before the analysis gives it one: the description's first sentence.

    Cut at a word boundary past ``limit`` characters with an ellipsis; empty
    when the description is blank.
    """
    text = " ".join(description.split())
    if not text:
        return ""
    first = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    if len(first) <= limit:
        return first
    cut = first[:limit].rsplit(" ", 1)[0].rstrip(" ,;:") or first[:limit]
    return cut + "…"


def plan_title(title: str, project_name: str, description: str) -> str:
    """The name a plan shows: the user's title, else the analysed name, else a provisional one."""
    return title or project_name or provisional_title(description)


def make_unique_display_names(sessions: list[dict]) -> dict[str, str]:
    """Compute collision-free display names for a list of sessions.

    Phase 8C: when the same project is run twice on the same day, both would
    get the same ``make_display_name()`` result (e.g. ``lendflow-2026-03-06``).
    This function appends ``-2``, ``-3``, etc. to duplicates. The first
    occurrence keeps the bare name.

    Args:
        sessions: List of session metadata dicts (from ``list_sessions()``).

    Returns:
        ``{session_id: unique_display_name}`` mapping.
    """
    # First pass: compute base names and track how many times each appears.
    base_names: list[tuple[str, str]] = []  # (session_id, base_name)
    for meta in sessions:
        sid = meta.get("session_id", "unknown")
        base_names.append((sid, make_display_name(meta)))

    # Second pass: append suffix for duplicates.
    seen: dict[str, int] = {}  # base_name → count so far
    result: dict[str, str] = {}
    for sid, base in base_names:
        count = seen.get(base, 0) + 1
        seen[base] = count
        result[sid] = base if count == 1 else f"{base}-{count}"
    return result


# ---------------------------------------------------------------------------
# Database schema
# ---------------------------------------------------------------------------

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS sessions_meta (
    session_id          TEXT PRIMARY KEY,
    project_name        TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL,
    last_modified       TEXT NOT NULL,
    last_node_completed TEXT NOT NULL DEFAULT '',
    session_state       TEXT NOT NULL DEFAULT '',
    session_mode        TEXT NOT NULL DEFAULT 'planning',
    title               TEXT NOT NULL DEFAULT ''
);"""

# One row per accepted plan section: the history a plan's blueprint shows.
_PLAN_VERSIONS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS plan_versions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    section    TEXT NOT NULL,
    version    INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload    TEXT NOT NULL,
    UNIQUE (session_id, section, version)
);"""
_PLAN_VERSIONS_INDEX = "CREATE INDEX IF NOT EXISTS idx_plan_versions_session ON plan_versions(session_id)"

# Phase 8C: schema version tracking — a single-row table that records which
# schema version this database was created/migrated to. On open, the code
# compares stored vs current version:
#   stored > current → schema_mismatch=True (newer DB, older code)
#   stored < current → run migrations, UPDATE to current
#   stored == current → schema_mismatch=False
# See docs: "Memory & State" — session persistence
CURRENT_SCHEMA_VERSION = 35  # v1=8A, v2=8B, v3=team_profiles, v4=session_mode, v5=token_usage, v6=standup, v7=retro, v8=performance, v9=reporting, v10=roadmap, v11=roadmap list, v12=token usage perf, v13=analysis ticket cache, v14=standup roster, v15=standup code scope, v16=standup documentation scope, v17=standup Azure project scope, v18=poker, v19=analysis enrichment cache, v20=analysis feature selection, v21=artifact edits, v22=standup transcript review, v23=standup practices, v24=standup practice AI matching, v25=standup practice feedback, v26=edit-provenance collision repair, v27=agentwatch, v28=standup GitHub owner scope, v29=standup GitHub repo exclusions, v30=planning prior-art feedback, v31=projects, v32=weekly review, v33=project status, v34=projects removed, v35=session title + plan versions + session labels  # noqa: E501

_SCHEMA_INFO = """\
CREATE TABLE IF NOT EXISTS schema_info (
    schema_version INT NOT NULL
);"""


# ---------------------------------------------------------------------------
# State serialisation helpers
# ---------------------------------------------------------------------------
# Phase 8B: persist graph state as JSON so --resume can reconstruct it.
# Messages are serialised through LangChain's message codec so a resumed
# conversation carries its transcript; pipeline nodes still read artifacts.
#
# Custom handling needed for:
# - Frozen dataclasses (Feature, UserStory, Task, Sprint, ProjectAnalysis, AcceptanceCriterion)
# - Enums (Priority, StoryPointValue, Discipline, ReviewDecision, OutputFormat)
# - Sets (skipped_questions, probed_questions, etc.) → lists in JSON
# - Tuples (story_ids, goals, etc.) → lists in JSON, reconstructed as tuples
# - QuestionnaireState (mutable dataclass with sets and dicts)
#
# See docs: "Memory & State" — session persistence, state serialisation


# Keys never written. Empty today; the seam stays for transient state.
_SKIP_KEYS: set[str] = set()

# ScrumState fields and the types they map to, used by the deserialiser to
# reconstruct the correct Python objects from JSON primitives.
_SCALAR_KEYS = {
    "project_name",
    "project_description",
    "team_size",
    "sprint_length_weeks",
    "velocity_per_sprint",
    "target_sprints",
    "repo_context",
    "confluence_context",
    "notion_context",
    "user_context",
    "pending_review",
    "last_review_feedback",
    "_intake_mode",
    "output_format",
    "context_sources",
    "solo",
    "_chat_greeting_done",
    "_chat_preamble",
    "_chat_fast_forward",
    "_epic_reviewed",
    "analysis_profile_id",
    "context_scope",
    "project_label",
    "session_integrations",
    "pasted_context",
    "chat_context",
}


class _StateEncoder(json.JSONEncoder):
    """JSON encoder that handles dataclasses, enums, and sets.

    # See docs: "Memory & State" — session persistence
    # Custom encoder so we don't need to manually convert every nested
    # structure before calling json.dumps(). The decoder side uses explicit
    # reconstruction helpers since JSON→Python needs type awareness.
    """

    def default(self, o: object) -> object:
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, set):
            return list(o)
        if isinstance(o, tuple):
            return list(o)
        return super().default(o)


def _messages_to_json(messages: list) -> list:
    """LangChain messages as their dict form; anything else passes through."""
    from langchain_core.messages import BaseMessage, messages_to_dict

    if messages and isinstance(messages[0], BaseMessage):
        return messages_to_dict(messages)
    return list(messages)


def _messages_from_json(value: list) -> list:
    from langchain_core.messages import messages_from_dict

    if value and isinstance(value[0], dict) and "type" in value[0]:
        return messages_from_dict(value)
    return list(value)


def _serialize_state(graph_state: dict) -> str:
    """Serialize graph_state to JSON, handling dataclasses, enums, and sets.

    Returns:
        JSON string of the serialisable subset of graph_state.
    """
    out: dict = {}
    for key, value in graph_state.items():
        if key in _SKIP_KEYS or value is None:
            continue
        if key == "messages":
            out[key] = _messages_to_json(value)
        elif key == "questionnaire" and isinstance(value, QuestionnaireState):
            out[key] = _questionnaire_to_dict(value)
        elif key == "project_analysis" and isinstance(value, ProjectAnalysis):
            out[key] = asdict(value)
        elif key == "features":
            out[key] = [asdict(e) for e in value]
        elif key == "stories":
            out[key] = [asdict(s) for s in value]
        elif key == "tasks":
            out[key] = [asdict(t) for t in value]
        elif key == "sprints":
            out[key] = [asdict(sp) for sp in value]
        elif key == "last_review_decision" and isinstance(value, ReviewDecision):
            out[key] = value.value
        elif key == "output_format" and isinstance(value, OutputFormat):
            out[key] = value.value
        elif key == "prior_art":
            # A tuple of frozen dataclasses: _StateEncoder would flatten it to
            # a list of un-reconstructable dicts, and _SCALAR_KEYS does not
            # cover dataclasses, so it needs its own pair like project_analysis.
            out[key] = prior_art_to_dicts(value)
        else:
            out[key] = value
    return json.dumps(out, cls=_StateEncoder, ensure_ascii=False)


def _questionnaire_to_dict(qs: QuestionnaireState) -> dict:
    """Convert QuestionnaireState to a JSON-friendly dict.

    Sets → lists, int dict keys → string keys (JSON requires string keys),
    tuple values in _follow_up_choices → lists.
    """
    return {
        "current_question": qs.current_question,
        # JSON keys must be strings — convert int keys
        "answers": {str(k): v for k, v in qs.answers.items()},
        "skipped_questions": list(qs.skipped_questions),
        "suggested_answers": {str(k): v for k, v in qs.suggested_answers.items()},
        "probed_questions": list(qs.probed_questions),
        "defaulted_questions": list(qs.defaulted_questions),
        "completed": qs.completed,
        "awaiting_confirmation": qs.awaiting_confirmation,
        "editing_question": qs.editing_question,
        "intake_mode": qs.intake_mode,
        "extracted_questions": list(qs.extracted_questions),
        "_pending_merged_questions": list(qs._pending_merged_questions),
        "_follow_up_choices": {str(k): list(v) for k, v in qs._follow_up_choices.items()},
        "_preferred_tracker": qs._preferred_tracker,
    }


def _dict_to_questionnaire(d: dict) -> QuestionnaireState:
    """Reconstruct a QuestionnaireState from a JSON-parsed dict.

    Reverses _questionnaire_to_dict: string keys → int, lists → sets/tuples.
    """
    return QuestionnaireState(
        current_question=d.get("current_question", 1),
        answers={int(k): v for k, v in d.get("answers", {}).items()},
        skipped_questions=set(d.get("skipped_questions", [])),
        suggested_answers={int(k): v for k, v in d.get("suggested_answers", {}).items()},
        probed_questions=set(d.get("probed_questions", [])),
        defaulted_questions=set(d.get("defaulted_questions", [])),
        completed=d.get("completed", False),
        awaiting_confirmation=d.get("awaiting_confirmation", False),
        editing_question=d.get("editing_question"),
        # Legacy sessions may still store "standard"; project_intake coerces it to
        # "smart" at its first invocation, so the stored/default value is harmless.
        intake_mode=d.get("intake_mode", "standard"),
        extracted_questions=set(d.get("extracted_questions", [])),
        _pending_merged_questions=d.get("_pending_merged_questions", []),
        _follow_up_choices={int(k): tuple(v) for k, v in d.get("_follow_up_choices", {}).items()},
        _preferred_tracker=d.get("_preferred_tracker", ""),
    )


def _dict_to_analysis(d: dict) -> ProjectAnalysis:
    """Reconstruct a ProjectAnalysis from a JSON-parsed dict.

    Lists → tuples for frozen dataclass tuple[str, ...] fields.
    """
    from yeaboi.agent.state import architecture_from_dict

    return ProjectAnalysis(
        architecture=architecture_from_dict(d.get("architecture")),
        project_name=d["project_name"],
        project_description=d["project_description"],
        project_type=d["project_type"],
        goals=tuple(d.get("goals", ())),
        end_users=tuple(d.get("end_users", ())),
        target_state=d["target_state"],
        tech_stack=tuple(d.get("tech_stack", ())),
        integrations=tuple(d.get("integrations", ())),
        constraints=tuple(d.get("constraints", ())),
        sprint_length_weeks=d["sprint_length_weeks"],
        target_sprints=d["target_sprints"],
        risks=tuple(d.get("risks", ())),
        out_of_scope=tuple(d.get("out_of_scope", ())),
        assumptions=tuple(d.get("assumptions", ())),
        skip_features=d.get("skip_features", False),
        is_low_code=d.get("is_low_code", False),
        low_code_reason=d.get("low_code_reason", ""),
        scrum_md_contributions=tuple(d.get("scrum_md_contributions", ())),
    )


def _dict_to_feature(d: dict) -> Feature:
    """Reconstruct a Feature from a JSON-parsed dict."""
    return Feature(
        id=d["id"],
        title=d["title"],
        description=d["description"],
        priority=Priority(d["priority"]),
    )


def _dict_to_story(d: dict) -> UserStory:
    """Reconstruct a UserStory from a JSON-parsed dict.

    Handles nested AcceptanceCriterion, enum fields, and tuple conversions.
    """
    acs = tuple(AcceptanceCriterion(**ac) for ac in d.get("acceptance_criteria", ()))
    return UserStory(
        id=d["id"],
        feature_id=d["feature_id"],
        persona=d["persona"],
        goal=d["goal"],
        benefit=d["benefit"],
        acceptance_criteria=acs,
        story_points=StoryPointValue(d["story_points"]),
        priority=Priority(d["priority"]),
        title=d.get("title", ""),
        discipline=Discipline(d.get("discipline", "fullstack")),
        dod_applicable=tuple(d.get("dod_applicable", (True,) * 7)),
        points_rationale=d.get("points_rationale", ""),
        points_confidence=d.get("points_confidence", ""),
    )


def _dict_to_task(d: dict) -> Task:
    """Reconstruct a Task from a JSON-parsed dict.

    label/test_plan/ai_prompt used to be silently dropped on --resume — a
    resumed session lost every task's label and test plan. Defaults keep old
    rows (without those keys) loading unchanged.
    """
    try:
        label = TaskLabel(d.get("label", TaskLabel.CODE.value))
    except ValueError:
        label = TaskLabel.CODE
    return Task(
        id=d["id"],
        story_id=d["story_id"],
        title=d["title"],
        description=d["description"],
        label=label,
        test_plan=d.get("test_plan", ""),
        ai_prompt=d.get("ai_prompt", ""),
    )


def _dict_to_sprint(d: dict) -> Sprint:
    """Reconstruct a Sprint from a JSON-parsed dict."""
    return Sprint(
        id=d["id"],
        name=d["name"],
        goal=d["goal"],
        capacity_points=d["capacity_points"],
        story_ids=tuple(d.get("story_ids", ())),
    )


def _deserialize_state(json_str: str) -> dict:
    """Reconstruct graph_state from a JSON string.

    Rebuilds all dataclasses, enums, sets, and tuples from their JSON
    representations. Injects an empty ``messages`` list so the state is
    ready for graph.invoke().

    Raises:
        json.JSONDecodeError: If json_str is not valid JSON.
        KeyError/TypeError: If required dataclass fields are missing or wrong type.
    """
    return state_from_raw(json.loads(json_str))


def state_from_raw(raw: dict) -> dict:
    """The deserialiser over an already-parsed blob (the list reads one straight from its row)."""
    state: dict = {"messages": []}

    for key, value in raw.items():
        if key == "messages":
            state[key] = _messages_from_json(value)
        elif key == "questionnaire":
            state[key] = _dict_to_questionnaire(value)
        elif key == "project_analysis":
            state[key] = _dict_to_analysis(value)
        elif key == "features":
            state[key] = [_dict_to_feature(e) for e in value]
        elif key == "stories":
            state[key] = [_dict_to_story(s) for s in value]
        elif key == "tasks":
            state[key] = [_dict_to_task(t) for t in value]
        elif key == "sprints":
            state[key] = [_dict_to_sprint(sp) for sp in value]
        elif key == "last_review_decision":
            state[key] = ReviewDecision(value)
        elif key == "output_format":
            state[key] = OutputFormat(value)
        elif key == "prior_art":
            state[key] = prior_art_from_dicts(value)
        else:
            # Scalar fields, context_sources (list[dict]), jira mappings (dict),
            # _intake_mode (str), etc. — pass through as-is.
            state[key] = value

    return state


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class SessionStore:
    """SQLite-backed metadata and state store for yeaboi sessions.

    Manages a ``sessions_meta`` table with both metadata columns (project name,
    timestamps, last node) and a ``session_state`` TEXT column containing the
    full serialised graph state as JSON. This avoids a separate table while
    keeping the schema simple.

    Usage (context manager — preferred):
        with SessionStore(db_path) as store:
            store.create_session(session_id)
            ...

    Usage (explicit close — for code paths where context manager is awkward):
        store = SessionStore(db_path)
        try:
            ...
        finally:
            store.close()

    # See docs: "Memory & State" — MemorySaver, thread_id, session persistence
    """

    def __init__(self, db_path: Path) -> None:
        # check_same_thread=False: the store is created on the main thread and
        # only ever accessed from the same thread (the REPL loop). The flag is
        # set to False to avoid spurious errors if the thread identity changes
        # (e.g. pytest reuses threads across fixtures).
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        # Belt-and-braces for direct-path callers that bypassed get_db_path():
        # the session DB holds planning content, keep it owner-only like .env.
        from yeaboi.config import restrict_permissions

        restrict_permissions(Path(db_path), mode=0o600)
        # isolation_level=None → autocommit: each execute() commits immediately.
        # Avoids manual transaction management for simple single-row writes.
        self._conn.isolation_level = None
        self._conn.execute(_SCHEMA)
        self._conn.execute(_PLAN_VERSIONS_SCHEMA)
        self._conn.execute(_PLAN_VERSIONS_INDEX)
        self._conn.executescript(SESSION_LABELS_SCHEMA)
        # Phase 8B: migrate existing Phase 8A databases that lack session_state.
        # ALTER TABLE ADD COLUMN is idempotent-safe with the try/except pattern:
        # if the column already exists (new schema or already migrated), SQLite
        # raises OperationalError which we silently ignore.
        try:
            self._conn.execute("ALTER TABLE sessions_meta ADD COLUMN session_state TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # column already exists

        # Phase 8C: schema version tracking.
        # Create the schema_info table, read the stored version, and compare
        # against CURRENT_SCHEMA_VERSION. Pre-8C databases will have no row —
        # we stamp the current version. If stored > current, set schema_mismatch
        # so callers can warn the user (newer DB opened by older code).
        # See docs: "Memory & State" — session persistence
        self._conn.execute(_SCHEMA_INFO)
        # Concurrent first-opens race the stamp INSERT below (nothing enforces
        # the single-row assumption), leaving duplicate rows and making the
        # fetchone() below read an arbitrary one. Keep exactly one row — the
        # highest version, so a newer build's stamp (and schema_mismatch
        # detection) survives the dedupe. Count first so the steady-state open
        # stays read-only (a DELETE takes a write lock even when it matches
        # nothing, and this DB is shared by the TUI, the MCP server and the
        # scheduler); the repair itself is one atomic DELETE — the connection
        # is autocommit, so a delete-all-then-reinsert would open a zero-row
        # window for another process.
        (info_rows,) = self._conn.execute("SELECT COUNT(*) FROM schema_info").fetchone()
        if info_rows > 1:
            cursor = self._conn.execute(
                "DELETE FROM schema_info WHERE rowid NOT IN "
                "(SELECT rowid FROM schema_info ORDER BY schema_version DESC, rowid DESC LIMIT 1)"
            )
            logger.warning("schema_info held %d duplicate rows; deduped to the highest version", cursor.rowcount)
        row = self._conn.execute("SELECT schema_version FROM schema_info").fetchone()
        if row is None:
            # Pre-8C DB or brand-new DB — stamp with current version. Guarded
            # so two processes racing this branch leave one row, not two.
            self._conn.execute(
                "INSERT INTO schema_info (schema_version) SELECT ? WHERE NOT EXISTS (SELECT 1 FROM schema_info)",
                (CURRENT_SCHEMA_VERSION,),
            )
            self._run_migrations(0)
            self.schema_mismatch = False
        elif row[0] > CURRENT_SCHEMA_VERSION:
            # DB was written by a newer version of the code — warn but don't crash.
            # Still self-heal the v21 collision: a future lineage stamping past
            # 26 would otherwise skip the repair forever, which is exactly the
            # failure v26 exists to fix. The body is idempotent and purely
            # additive, so it is safe on a newer schema.
            self._apply_edit_provenance()
            self.schema_mismatch = True
        else:
            # row[0] <= CURRENT_SCHEMA_VERSION — up to date (or migrated above)
            if row[0] < CURRENT_SCHEMA_VERSION:
                self._run_migrations(row[0])
                self._conn.execute("UPDATE schema_info SET schema_version = ?", (CURRENT_SCHEMA_VERSION,))
            self.schema_mismatch = False

    def _run_migrations(self, from_version: int) -> None:
        """Run schema migrations from from_version to CURRENT_SCHEMA_VERSION.

        v3: Create team_profiles table for team learning calibration data.
        """
        if from_version < 3:
            from yeaboi.team_profile import _TEAM_PROFILES_SCHEMA

            self._conn.execute(_TEAM_PROFILES_SCHEMA)
            logger.info("Migration v3: created team_profiles table")
        if from_version < 4:
            try:
                self._conn.execute("ALTER TABLE sessions_meta ADD COLUMN session_mode TEXT NOT NULL DEFAULT 'planning'")
                logger.info("Migration v4: added session_mode column")
            except sqlite3.OperationalError:
                pass  # column already exists

        if from_version < 5:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    input_tokens INT NOT NULL DEFAULT 0,
                    output_tokens INT NOT NULL DEFAULT 0,
                    model TEXT NOT NULL DEFAULT '',
                    provider TEXT NOT NULL DEFAULT ''
                )"""
            )
            logger.info("Migration v5: created token_usage table")

        if from_version < 6:
            # v6: Daily Standup mode — config, run history, self-reported updates.
            # Schema lives in standup/store.py (executescript handles the 3 CREATEs).
            from yeaboi.standup.store import _STANDUP_SCHEMA

            self._conn.executescript(_STANDUP_SCHEMA)
            logger.info("Migration v6: created standup tables")

        if from_version < 7:
            # v7: Retro mode — one retro_history table. Schema lives in retro/store.py.
            from yeaboi.retro.store import _RETRO_SCHEMA

            self._conn.executescript(_RETRO_SCHEMA)
            logger.info("Migration v7: created retro tables")

        if from_version < 8:
            # v8: Performance mode — per-engineer 1:1s, reviews, and lead notes.
            # Schema lives in performance/store.py (executescript handles the 3 CREATEs).
            from yeaboi.performance.store import _PERFORMANCE_SCHEMA

            self._conn.executescript(_PERFORMANCE_SCHEMA)
            logger.info("Migration v8: created performance tables")

        if from_version < 9:
            # v9: Reporting mode — business-friendly delivery reports per run.
            # Schema lives in reporting/store.py (executescript handles the CREATE).
            from yeaboi.reporting.store import _REPORTING_SCHEMA

            self._conn.executescript(_REPORTING_SCHEMA)
            logger.info("Migration v9: created reporting tables")

        if from_version < 10:
            # v10: Roadmap intake — saved roadmap source (singleton) + analysis
            # history. Schema lives in roadmap/store.py (executescript handles
            # both CREATEs).
            from yeaboi.roadmap.store import _ROADMAP_SCHEMA

            self._conn.executescript(_ROADMAP_SCHEMA)
            logger.info("Migration v10: created roadmap tables")

        if from_version < 11:
            # v11: Roadmaps become a LIST (open/create/delete like planning
            # projects) — new multi-row `roadmaps` table, seeded with one row
            # from the v10 singleton (roadmap_config + newest roadmap_history
            # analysis) so an already-analyzed roadmap survives the upgrade.
            from yeaboi.roadmap.store import _ROADMAP_SCHEMA

            self._conn.executescript(_ROADMAP_SCHEMA)  # idempotent; adds `roadmaps`
            # Seed only when empty — RoadmapStore may have pre-created the table.
            if self._conn.execute("SELECT COUNT(*) FROM roadmaps").fetchone()[0] == 0:
                cfg = self._conn.execute(
                    "SELECT source_type, source_locator, source_label, updated_at FROM roadmap_config WHERE id = 1"
                ).fetchone()
                hist = self._conn.execute(
                    "SELECT source_type, source_locator, project_count, analysis_json, run_at "
                    "FROM roadmap_history ORDER BY run_at DESC LIMIT 1"
                ).fetchone()
                # Source fields come from the config; fall back to the newest
                # history row when the config was never saved. The analysis
                # payload always comes from history (config never held one).
                src = (cfg[0], cfg[1], cfg[2], cfg[3]) if cfg and cfg[0] else None
                if src is None and hist and hist[0]:
                    src = (hist[0], hist[1], hist[1], hist[4])  # locator doubles as label
                if src is not None:
                    analysis_json = hist[3] if hist else ""
                    project_count = hist[2] if hist else 0
                    self._conn.execute(
                        """INSERT INTO roadmaps
                           (label, source_type, source_locator, source_label, analysis_json, project_count,
                            created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (src[2] or src[1], src[0], src[1], src[2], analysis_json, project_count, src[3], src[3]),
                    )
                    logger.info("Migration v11: seeded roadmaps row from v10 singleton (type=%s)", src[0])
            logger.info("Migration v11: created roadmaps table")

        if from_version < 12:
            # v12: local-model performance metrics on token_usage. Nullable, so
            # cloud rows (which have no timing data) simply stay NULL. Additive
            # ALTERs are idempotent via the OperationalError guard — a fresh DB
            # runs the v5 CREATE then these ALTERs in the same migration pass.
            for col in ("duration_ms", "eval_duration_ms", "load_duration_ms", "tokens_per_sec"):
                try:
                    self._conn.execute(f"ALTER TABLE token_usage ADD COLUMN {col} REAL")
                except sqlite3.OperationalError:
                    pass  # column already exists
            logger.info("Migration v12: added token_usage performance columns")

        if from_version < 13:
            from yeaboi.team_profile import _ANALYSIS_TICKET_CACHE_SCHEMA

            self._conn.execute(_ANALYSIS_TICKET_CACHE_SCHEMA)
            logger.info("Migration v13: created analysis ticket parse cache")
        if from_version < 14:
            for statement in (
                """ALTER TABLE standup_config
                   ADD COLUMN tracker_sources TEXT NOT NULL DEFAULT '["jira"]'""",
                """ALTER TABLE standup_config
                   ADD COLUMN team_members TEXT NOT NULL DEFAULT '[]'""",
                """ALTER TABLE standup_config
                   ADD COLUMN roster_configured INTEGER NOT NULL DEFAULT 0""",
            ):
                try:
                    self._conn.execute(statement)
                except sqlite3.OperationalError:
                    pass  # column already exists
            logger.info("Migration v14: added standup tracker and roster scope")

        if from_version < 15:
            for statement in (
                """ALTER TABLE standup_config
                   ADD COLUMN code_sources TEXT NOT NULL DEFAULT '[]'""",
                """ALTER TABLE standup_config
                   ADD COLUMN github_repositories TEXT NOT NULL DEFAULT '[]'""",
                """ALTER TABLE standup_config
                   ADD COLUMN azdo_repositories TEXT NOT NULL DEFAULT '[]'""",
                """ALTER TABLE standup_config
                   ADD COLUMN code_scope_configured INTEGER NOT NULL DEFAULT 0""",
            ):
                try:
                    self._conn.execute(statement)
                except sqlite3.OperationalError:
                    pass
            logger.info("Migration v15: added standup code repository scope")

        if from_version < 16:
            for statement in (
                """ALTER TABLE standup_config
                   ADD COLUMN documentation_sources TEXT NOT NULL DEFAULT '[]'""",
                """ALTER TABLE standup_config
                   ADD COLUMN documentation_scope_configured INTEGER NOT NULL DEFAULT 0""",
            ):
                try:
                    self._conn.execute(statement)
                except sqlite3.OperationalError:
                    pass
            logger.info("Migration v16: added standup documentation scope")

        if from_version < 17:
            try:
                self._conn.execute(
                    """ALTER TABLE standup_config
                       ADD COLUMN azdo_projects TEXT NOT NULL DEFAULT '[]'"""
                )
            except sqlite3.OperationalError:
                pass
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(standup_config)").fetchall()}
            rows = (
                self._conn.execute(
                    "SELECT session_id, azdo_repositories FROM standup_config WHERE azdo_projects = '[]'"
                ).fetchall()
                if {"session_id", "azdo_projects", "azdo_repositories"} <= columns
                else []
            )
            for session_id, repositories_json in rows:
                try:
                    repositories = json.loads(repositories_json or "[]")
                except (json.JSONDecodeError, TypeError):
                    repositories = []
                projects = list(
                    dict.fromkeys(
                        project
                        for repository in repositories
                        for project, separator, _name in [str(repository).partition("/")]
                        if separator and project
                    )
                )
                if projects:
                    self._conn.execute(
                        "UPDATE standup_config SET azdo_projects = ? WHERE session_id = ?",
                        (json.dumps(projects), session_id),
                    )
            logger.info("Migration v17: added standup Azure project scope")

        if from_version < 18:
            # v18: Scrum Poker mode — one poker_history table. Schema lives in poker/store.py.
            from yeaboi.poker.store import _POKER_SCHEMA

            self._conn.executescript(_POKER_SCHEMA)
            logger.info("Migration v18: created poker tables")

        # v19/v20 (renumbered from 18/19 when poker landed on main as v18): a DB
        # that already ran poker's v18 must still receive the analysis migrations.
        if from_version < 19:
            from yeaboi.team_profile import _ANALYSIS_ENRICHMENT_CACHE_SCHEMA

            self._conn.execute(_ANALYSIS_ENRICHMENT_CACHE_SCHEMA)
            logger.info("Migration v19: created analysis enrichment cache")
        if from_version < 20:
            try:
                self._conn.execute("ALTER TABLE analysis_runs ADD COLUMN features_json TEXT NOT NULL DEFAULT '[]'")
                logger.info("Migration v20: added Analysis run feature selection")
            except sqlite3.OperationalError:
                pass  # column already exists (pre-rebase lineage) — nothing to do
        if from_version < 21:
            # v21: browser-editable shared artifacts. The append-only edit log
            # gets its own table; each history table learns where a row came
            # from, so a corrected report can be told from a generated one.
            self._apply_edit_provenance()
            logger.info("Migration v21: created artifact_edits and edit-provenance columns")

        if from_version < 22:
            # v22: standup transcript review — three new tables plus two
            # standup_config columns. The whole schema script is idempotent
            # (CREATE TABLE IF NOT EXISTS), so replaying it only adds what's missing.
            from yeaboi.standup.store import _STANDUP_SCHEMA

            self._conn.executescript(_STANDUP_SCHEMA)
            for statement in (
                """ALTER TABLE standup_config
                   ADD COLUMN transcript_dir TEXT NOT NULL DEFAULT ''""",
                """ALTER TABLE standup_config
                   ADD COLUMN transcript_review_enabled INTEGER NOT NULL DEFAULT 1""",
            ):
                try:
                    self._conn.execute(statement)
                except sqlite3.OperationalError:
                    pass  # column already exists
            logger.info("Migration v22: created standup transcript-review tables")

        if from_version < 23:
            # v23: standup practice detection (standup/habits.py) — on by
            # default, with an optional rule subset. StandupStore.__init__ runs
            # the same ALTERs, so a DB that only ever opens through the store is
            # already correct; this is for databases migrated ahead of it.
            for statement in (
                "ALTER TABLE standup_config ADD COLUMN habit_detection TEXT NOT NULL DEFAULT 'on'",
                "ALTER TABLE standup_config ADD COLUMN habit_rules TEXT NOT NULL DEFAULT ''",
            ):
                try:
                    self._conn.execute(statement)
                except sqlite3.OperationalError:
                    pass  # column already exists
            logger.info("Migration v23: added standup practice detection config")

        if from_version < 24:
            # v24: the language-model pass that excuses a change belonging to a
            # ticket it never names (standup/adjudicate.py). Its own switch
            # because it is the only part of practice detection that costs money.
            try:
                self._conn.execute("ALTER TABLE standup_config ADD COLUMN habit_ai_match TEXT NOT NULL DEFAULT 'on'")
            except sqlite3.OperationalError:
                pass  # column already exists
            logger.info("Migration v24: added standup practice AI matching config")

        if from_version < 25:
            # v25: per-change thumbs up/down on a practice signal
            # (standup/practice_feedback.py). A new table rather than a config
            # column — the unit is one change, not one setting, and the ledger
            # has to outlive the report it was voted on.
            from yeaboi.standup.store import _STANDUP_PRACTICE_FEEDBACK_SCHEMA

            self._conn.execute(_STANDUP_PRACTICE_FEEDBACK_SCHEMA)
            logger.info("Migration v25: created standup practice feedback ledger")

        if 21 <= from_version < 26:
            # v26: repair the v21 number collision. A pre-rebase lineage stamped
            # shared databases at 21 for standup transcript review, so main's
            # v21 (edit-provenance columns + artifact_edits) was skipped while
            # the DB went on to v25 — every origin-reading query then fails
            # with "no such column: origin". The body is idempotent, so
            # re-running it on a healthy DB is a no-op; the lower bound only
            # skips the double-run when the v21 branch above just did the same
            # work. Same idiom as the v19/v20 renumbering above.
            self._apply_edit_provenance()
            logger.info("Migration v26: re-applied edit-provenance columns (v21 number collision)")

        if from_version < 27:
            # v27: the agentwatch (Agents family) tables — monitored-agent
            # session rollups, ingest cursors, security findings, and the three
            # report-history tables. Schema lives in agentwatch/store.py.
            from yeaboi.agentwatch.store import _AGENTWATCH_SCHEMA

            self._conn.executescript(_AGENTWATCH_SCHEMA)
            logger.info("Migration v27: created agentwatch tables")

        if from_version < 28:
            # v28: standup GitHub owner scope — GitHub is picked by owner/org, the
            # way Azure DevOps is picked by project, and the owner is expanded to
            # its active repositories per run.
            #
            # Column only, deliberately no data backfill: deriving "acme" from a
            # saved "acme/api" would widen an existing standup from one repo to
            # every repo in that org without anyone asking. Saved repositories keep
            # working untouched; the picker offers the owner upgrade explicitly.
            #
            # The bare except covers both "column already there" and "no
            # standup_config yet"; neither is a problem, because StandupStore._migrate
            # adds the same column independently — `--standup-run` and the MCP server
            # open that store without ever constructing a SessionStore.
            added = True
            try:
                self._conn.execute(
                    """ALTER TABLE standup_config
                       ADD COLUMN github_owners TEXT NOT NULL DEFAULT '[]'"""
                )
            except sqlite3.OperationalError:
                added = False
            if added:
                logger.info("Migration v28: added standup GitHub owner scope")

        if from_version < 29:
            # v29: standup GitHub repo exclusions — an owner still means "every
            # active repo inside it"; this column is the narrow opt-out, letting
            # someone drop one noisy repo without losing the "stays fresh
            # automatically" property an inclusion list would give up.
            #
            # Same bare-except reasoning as v28: StandupStore._migrate adds this
            # column independently for callers that never construct a SessionStore.
            added = True
            try:
                self._conn.execute(
                    """ALTER TABLE standup_config
                       ADD COLUMN github_excluded_repositories TEXT NOT NULL DEFAULT '[]'"""
                )
            except sqlite3.OperationalError:
                added = False
            if added:
                logger.info("Migration v29: added standup GitHub repo exclusions")

        if from_version < 30:
            # v30: planning prior-art feedback — the user's verdict on a repo
            # offered as prior art for a greenfield project. Rejections are
            # global and permanent, so the ledger has to outlive the session
            # that recorded it.
            from yeaboi.agent.prior_art_feedback import PRIOR_ART_FEEDBACK_SCHEMA

            self._conn.execute(PRIOR_ART_FEEDBACK_SCHEMA)
            logger.info("Migration v30: created planning_prior_art_feedback table")

        # v31 (projects) and v33 (project status) are no-ops: v34 drops what
        # they created. A database that never saw them simply never has it.

        if from_version < 32:
            # v32: the Solo world's weekly reviews. Schema lives in solo/store.py
            # (also created on that store's open, for the CLI and MCP paths).
            from yeaboi.solo.store import _WEEKLY_REVIEW_SCHEMA

            self._conn.executescript(_WEEKLY_REVIEW_SCHEMA)
            logger.info("Migration v32: created weekly_review_history table")

        if from_version < 34:
            # v34: projects removed — one kind of thing, and it is a session.
            # The index goes first: SQLite refuses DROP COLUMN on an indexed
            # column, so leaving it would make the ALTER below always raise.
            self._conn.execute("DROP INDEX IF EXISTS idx_sessions_meta_project")
            self._conn.execute("DROP TABLE IF EXISTS projects")
            try:
                self._conn.execute("ALTER TABLE sessions_meta DROP COLUMN project_id")
                logger.info("Migration v34: dropped the projects table and sessions_meta.project_id")
            except sqlite3.OperationalError:
                # DROP COLUMN needs SQLite >= 3.35. An older one keeps the
                # column, which is inert: _SCHEMA never declares it and every
                # INSERT names its columns.
                logger.info("Migration v34: dropped the projects table; project_id left in place (SQLite < 3.35)")

        if from_version < 35:
            # v35: a user-given title beside the derived project_name, and the
            # plan_versions table (also created on open, like every table a
            # CLI or MCP path may reach first).
            try:
                self._conn.execute("ALTER TABLE sessions_meta ADD COLUMN title TEXT NOT NULL DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # column already exists
            self._conn.execute(_PLAN_VERSIONS_SCHEMA)
            self._conn.execute(_PLAN_VERSIONS_INDEX)
            # session_labels: the project label, tags and scope every mode's run
            # carries (context/labels.py owns the DDL; the label store also
            # creates it for the CLI/MCP paths that never open a SessionStore).
            self._conn.executescript(SESSION_LABELS_SCHEMA)
            logger.info("Migration v35: added sessions_meta.title, plan_versions and session_labels")

    def _apply_edit_provenance(self) -> None:
        """The v21 migration body — idempotent, so v26 re-runs it verbatim.

        A pre-rebase lineage stamped shared databases at 21 with a different
        meaning (standup transcript review), so a DB could reach v25 with the
        provenance columns missing. See migration v26.
        """
        from yeaboi.artifacts.store import _ARTIFACT_EDITS_SCHEMA

        self._conn.executescript(_ARTIFACT_EDITS_SCHEMA)
        # `origin` and not a new `status` value: get_previous_report filters
        # status IN ('success','partial'), so a third status would silently
        # drop every corrected standup out of the next day's comparison.
        for table in (
            "standup_history",
            "retro_history",
            "reporting_history",
            "roadmap_history",
            "performance_one_on_ones",
            "performance_reviews",
        ):
            for column, kind, default in (("origin", "TEXT", "'generated'"), ("edited_from_id", "INTEGER", "0")):
                try:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind} NOT NULL DEFAULT {default}")
                except sqlite3.OperationalError:
                    pass  # column already exists — the block stays idempotent

    # ── Token usage persistence ──────────────────────────────────────────

    def record_token_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "",
        provider: str = "",
        *,
        duration_ms: float | None = None,
        eval_duration_ms: float | None = None,
        load_duration_ms: float | None = None,
        tokens_per_sec: float | None = None,
    ) -> None:
        """Record a single LLM call's token usage (+ optional local timing)."""
        self._conn.execute(
            "INSERT INTO token_usage "
            "(timestamp, input_tokens, output_tokens, model, provider, "
            "duration_ms, eval_duration_ms, load_duration_ms, tokens_per_sec) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._now(),
                input_tokens,
                output_tokens,
                model,
                provider,
                duration_ms,
                eval_duration_ms,
                load_duration_ms,
                tokens_per_sec,
            ),
        )

    def get_lifetime_usage(self) -> dict:
        """Return cumulative token usage across all sessions."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0), COUNT(*) FROM token_usage"
        ).fetchone()
        inp, out, calls = row if row else (0, 0, 0)
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out, "call_count": calls}

    def get_lifetime_usage_by_provider(self) -> dict[str, dict]:
        """Return cumulative token usage grouped by provider.

        Lets the Usage page price each provider's tokens at its own rate — a
        history mixing Anthropic and (free) Ollama sessions must neither hide
        real past cloud spend behind a $0 local rate nor price local tokens at
        cloud rates. Rows recorded before providers were stamped group under "".
        """
        usage: dict[str, dict] = {}
        for provider, inp, out, calls in self._conn.execute(
            "SELECT provider, COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0), COUNT(*) "
            "FROM token_usage GROUP BY provider"
        ).fetchall():
            usage[provider] = {
                "input_tokens": inp,
                "output_tokens": out,
                "total_tokens": inp + out,
                "call_count": calls,
            }
        return usage

    def get_local_perf_summary(self) -> dict:
        """Aggregate local-model (Ollama) throughput/latency across all calls.

        Powers the Usage page's "Local Model Performance" section. Returns {}
        when no local call has recorded timing yet (cloud-only history), so the
        UI can simply hide the section.
        """
        row = self._conn.execute(
            "SELECT COUNT(*), AVG(tokens_per_sec), MAX(tokens_per_sec), AVG(duration_ms), AVG(load_duration_ms) "
            "FROM token_usage WHERE provider = 'ollama' AND tokens_per_sec IS NOT NULL"
        ).fetchone()
        if not row or not row[0]:
            return {}
        calls, avg_tps, max_tps, avg_dur, avg_load = row
        last = self._conn.execute(
            "SELECT model, tokens_per_sec, duration_ms FROM token_usage "
            "WHERE provider = 'ollama' AND tokens_per_sec IS NOT NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        summary = {
            "calls": calls,
            "avg_tps": avg_tps or 0.0,
            "max_tps": max_tps or 0.0,
            "avg_duration_ms": avg_dur or 0.0,
            "avg_load_ms": avg_load or 0.0,
        }
        if last:
            summary["last"] = {"model": last[0], "tps": last[1] or 0.0, "duration_ms": last[2] or 0.0}
        return summary

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Fallback: close on GC for early-return paths that bypass __exit__.
        self.close()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Write operations ──────────────────────────────────────────────────

    def create_session(
        self,
        session_id: str,
        project_name: str = "",
        *,
        mode: str = "planning",
        title: str = "",
    ) -> None:
        """Insert a new session row. Silently ignores duplicate session IDs."""
        logger.info("Creating session: %s (mode=%s)", session_id, mode)
        now = self._now()
        self._conn.execute(
            """INSERT OR IGNORE INTO sessions_meta
               (session_id, project_name, created_at, last_modified,
                last_node_completed, session_state, session_mode, title)
               VALUES (?, ?, ?, ?, '', '', ?, ?)""",
            (session_id, project_name, now, now, mode, title),
        )

    def update_session_meta(self, session_id: str, *, title: str | None = None) -> None:
        """Change what the user may rename. ``None`` leaves a column alone."""
        if title is None:
            return
        logger.info("Session %s renamed (len=%d)", session_id, len(title))
        self._conn.execute(
            "UPDATE sessions_meta SET title = ?, last_modified = ? WHERE session_id = ?",
            (title, self._now(), session_id),
        )

    # ── Plan versions ─────────────────────────────────────────────────────

    def record_plan_version(self, session_id: str, section: str, payload: dict) -> int:
        """Keep an accepted plan section. Returns its version number (1-based)."""
        row = self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM plan_versions WHERE session_id = ? AND section = ?",
            (session_id, section),
        ).fetchone()
        version = int(row[0]) + 1
        self._conn.execute(
            "INSERT INTO plan_versions (session_id, section, version, created_at, payload) VALUES (?, ?, ?, ?, ?)",
            (session_id, section, version, self._now(), json.dumps(payload, cls=_StateEncoder, default=str)),
        )
        logger.info("Plan version recorded: session=%s section=%s version=%d", session_id, section, version)
        return version

    def list_plan_versions(self, session_id: str, section: str = "") -> list[dict]:
        """The accepted versions, oldest first, without their payloads."""
        params: list[object] = [session_id]
        where = ""
        if section:
            where = " AND section = ?"
            params.append(section)
        rows = self._conn.execute(
            "SELECT section, version, created_at FROM plan_versions "  # noqa: S608 — placeholders, not values
            f"WHERE session_id = ?{where} ORDER BY id",
            params,
        ).fetchall()
        return [{"section": r[0], "version": r[1], "created_at": r[2]} for r in rows]

    def get_plan_version(self, session_id: str, section: str, version: int) -> dict | None:
        """One accepted snapshot with its payload, or None."""
        row = self._conn.execute(
            "SELECT created_at, payload FROM plan_versions WHERE session_id = ? AND section = ? AND version = ?",
            (session_id, section, version),
        ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row[1])
        except json.JSONDecodeError:
            logger.error("Plan version %s/%s/%d holds invalid JSON", session_id, section, version)
            return None
        return {"section": section, "version": version, "created_at": row[0], "payload": payload}

    def update_project_name(self, session_id: str, project_name: str) -> None:
        """Set the display name once the project name becomes known.

        Called once after the project_analyzer node returns a ProjectAnalysis
        with a non-empty project_name.
        """
        self._conn.execute(
            "UPDATE sessions_meta SET project_name = ?, last_modified = ? WHERE session_id = ?",
            (project_name, self._now(), session_id),
        )

    def update_last_node(self, session_id: str, node_name: str) -> None:
        """Record the most recently completed pipeline node.

        Called after each successful graph.invoke() so the session picker
        can show 'Last step: epic_generator' etc.
        """
        self._conn.execute(
            "UPDATE sessions_meta SET last_node_completed = ?, last_modified = ? WHERE session_id = ?",
            (node_name, self._now(), session_id),
        )

    def save_state(self, session_id: str, graph_state: dict) -> None:
        """Persist the full graph state as JSON.

        Called after each successful graph.invoke(). Replaces the previous
        snapshot entirely — the latest state is always the full picture.

        # See docs: "Memory & State" — session persistence
        """
        json_str = _serialize_state(graph_state)
        self._conn.execute(
            "UPDATE sessions_meta SET session_state = ?, last_modified = ? WHERE session_id = ?",
            (json_str, self._now(), session_id),
        )

    # ── Read operations ───────────────────────────────────────────────────

    def get_session(self, session_id: str) -> dict | None:
        """Return the metadata row as a dict, or None if not found.

        Includes ``session_state_raw`` — the raw JSON string for the state.
        Use ``load_state()`` to deserialise it into a graph-ready dict.
        """
        row = self._conn.execute(
            "SELECT session_id, project_name, created_at, last_modified, "
            "last_node_completed, session_state, session_mode, title "
            "FROM sessions_meta WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        keys = (
            "session_id",
            "project_name",
            "created_at",
            "last_modified",
            "last_node_completed",
            "session_state_raw",
            "session_mode",
            "title",
        )
        return dict(zip(keys, row))

    def list_sessions(self, *, mode: str = "", limit: int = 0) -> list[dict]:
        """Return sessions ordered by last_modified descending.

        Used by the interactive session picker (--resume), --list-sessions and
        the cross-mode recent list. ``mode`` narrows the rows (blank = all);
        ``limit`` caps them (0 = every row). Each row carries ``session_mode``
        beside the legacy keys.
        """
        logger.debug("Listing sessions (mode=%s limit=%d)", mode or "-", limit)
        params: list[object] = []
        where = ""
        if mode:
            where = " WHERE session_mode = ?"
            params.append(mode)
        tail = " LIMIT ?" if limit > 0 else ""
        if limit > 0:
            params.append(limit)
        rows = self._conn.execute(
            "SELECT session_id, project_name, created_at, last_modified, "  # noqa: S608 — placeholders, not values
            "last_node_completed, session_state, session_mode, title "
            f"FROM sessions_meta{where} ORDER BY last_modified DESC{tail}",
            params,
        ).fetchall()
        keys = (
            "session_id",
            "project_name",
            "created_at",
            "last_modified",
            "last_node_completed",
            "session_state_raw",
            "session_mode",
            "title",
        )
        result = [dict(zip(keys, row)) for row in rows]
        logger.debug("Found %d session(s)", len(result))
        return result

    def list_analysis_sessions(self) -> list[dict]:
        """Return analysis-mode sessions ordered by last_modified descending."""
        rows = self._conn.execute(
            "SELECT session_id, project_name, created_at, last_modified, "
            "last_node_completed, session_state "
            "FROM sessions_meta WHERE session_mode = 'analysis' "
            "ORDER BY last_modified DESC"
        ).fetchall()
        keys = (
            "session_id",
            "project_name",
            "created_at",
            "last_modified",
            "last_node_completed",
            "session_state_raw",
        )
        return [dict(zip(keys, row)) for row in rows]

    def load_state(self, session_id: str) -> dict | None:
        """Load and reconstruct graph state from JSON.

        Returns the deserialised graph state dict ready for graph.invoke(),
        or None if the session doesn't exist, has no saved state, or the
        state is corrupt (malformed JSON, missing fields, etc.).

        # See docs: "Memory & State" — session persistence, --resume
        """
        meta = self.get_session(session_id)
        if not meta or not meta.get("session_state_raw"):
            logger.debug("No saved state for session %s", session_id)
            return None
        try:
            state = _deserialize_state(meta["session_state_raw"])
            logger.debug("Loaded state for session %s (%d keys)", session_id, len(state))
            return state
        except Exception:
            logger.error("Failed to deserialize state for session %s", session_id)
            return None

    def get_latest_session_id(self) -> str | None:
        """Return the session_id of the most recently modified session, or None."""
        row = self._conn.execute("SELECT session_id FROM sessions_meta ORDER BY last_modified DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def recent_session_ids(self, limit: int = 25) -> list[str]:
        """The most recently modified session ids, newest first.

        Lightweight — reads ids only, never the (potentially large) state blob —
        so a caller can scan a bounded window for the first session that carries
        what it needs without loading every session's state.
        """
        rows = self._conn.execute(
            "SELECT session_id FROM sessions_meta ORDER BY last_modified DESC LIMIT ?", (max(1, limit),)
        ).fetchall()
        return [r[0] for r in rows]

    def delete_session(self, session_id: str) -> bool:
        """Delete a single session by ID.

        Returns True if a row was deleted, False if the session_id didn't exist.
        """
        cursor = self._conn.execute("DELETE FROM sessions_meta WHERE session_id = ?", (session_id,))
        self._conn.execute("DELETE FROM plan_versions WHERE session_id = ?", (session_id,))
        self._conn.execute(
            "DELETE FROM session_labels WHERE session_id = ? AND mode IN ('planning', 'analysis')", (session_id,)
        )
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted session %s", session_id)
        else:
            logger.debug("Session not found for deletion: %s", session_id)
        return deleted

    def delete_all_sessions(self) -> int:
        """Delete all sessions. Returns the number of rows deleted."""
        cursor = self._conn.execute("DELETE FROM sessions_meta")
        self._conn.execute("DELETE FROM plan_versions")
        # Analysis labels are keyed by team id, not by a session row.
        self._conn.execute("DELETE FROM session_labels WHERE mode = 'planning'")
        logger.info("Deleted all sessions (count=%d)", cursor.rowcount)
        return cursor.rowcount

    def prune_old_sessions(self, max_age_days: int) -> int:
        """Delete sessions whose last_modified is older than *max_age_days*.

        Phase 8C: prevents unbounded DB growth. Called at REPL startup.
        Configurable via ``SESSION_PRUNE_DAYS`` env var (default 30, 0=disabled).

        Args:
            max_age_days: Sessions older than this are deleted. 0 means disabled.

        Returns:
            Number of sessions deleted.

        # See docs: "Memory & State" — session persistence
        """
        if max_age_days <= 0:
            return 0
        # SQLite datetime('now', '-N days') computes a UTC cutoff timestamp.
        # last_modified is stored as ISO-8601 UTC so string comparison works.
        cursor = self._conn.execute(
            "DELETE FROM sessions_meta WHERE last_modified < datetime('now', ?)",
            (f"-{max_age_days} days",),
        )
        if cursor.rowcount > 0:
            logger.info("Pruned %d session(s) older than %d days", cursor.rowcount, max_age_days)
        return cursor.rowcount
