"""SQLite store for the Daily Standup mode.

Persists seven things in the shared ~/.scrum-agent/sessions.db:
- ``standup_config``      — per-session schedule + delivery preferences
- ``standup_history``     — every run's serialized StandupReport + delivery status
- ``standup_updates``     — user-typed "my update" text, consumed verbatim by the engine
- ``standup_reviews``     — serialized TranscriptReview per audited standup
- ``standup_transcripts`` — which transcripts have been reviewed, keyed by CONTENT
  hash rather than path, so a renamed file isn't re-reviewed and an edited one is.
  The DB is the bookkeeping precisely so we never move or rewrite the user's files.
- ``standup_gap_issues``  — the gap→GitHub-issue dedup ledger. Deliberately NOT
  session-scoped: the review loop improves yeaboi itself, so the same gap raised
  in two different projects belongs on the same issue.
- ``standup_practice_feedback`` — one thumbs up/down per (rule, change), the team
  telling the practice rules where they were wrong (see practice_feedback.py)

Follows the exact patterns used by TeamProfileStore (team_profile.py): a separate
store class opening its own connection to the same DB, autocommit mode, context
manager support, idempotent CREATE-IF-NOT-EXISTS schema. The schema constant is
also referenced by sessions.py's v6 migration so an existing DB gets the tables.

# See docs: "Session Management" — SQLite persistence, schema versioning
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from yeaboi.agent.state import (
    ActivityEvidence,
    ConflictCard,
    MemberUpdate,
    OpsSignal,
    PracticeSignal,
    StandupGap,
    StandupReport,
    TranscriptClaim,
    TranscriptReview,
    TranscriptSource,
    annotations_from,
)
from yeaboi.context._sql import id_filter, limit_clause
from yeaboi.context.labels import drop_run_labels

logger = logging.getLogger(__name__)


def _scope_dict(raw: str | None) -> dict | None:
    """A stored ``context_scope`` JSON column as a dict, ``None`` when blank or unreadable."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# Schema — referenced by sessions.py migration v6 AND created on store open
# ---------------------------------------------------------------------------

_STANDUP_SCHEMA = """\
CREATE TABLE IF NOT EXISTS standup_config (
    session_id        TEXT PRIMARY KEY,
    enabled           INTEGER NOT NULL DEFAULT 0,
    time              TEXT NOT NULL DEFAULT '10:00',
    lead_minutes      INTEGER NOT NULL DEFAULT 10,
    timezone          TEXT NOT NULL DEFAULT '',
    weekdays          TEXT NOT NULL DEFAULT '1-5',
    delivery_channels TEXT NOT NULL DEFAULT '["terminal"]',
    repo_path         TEXT NOT NULL DEFAULT '',
    my_aliases        TEXT NOT NULL DEFAULT '',
    tracker_sources   TEXT NOT NULL DEFAULT '["jira"]',
    team_members      TEXT NOT NULL DEFAULT '[]',
    roster_configured INTEGER NOT NULL DEFAULT 0,
    code_sources      TEXT NOT NULL DEFAULT '[]',
    github_owners     TEXT NOT NULL DEFAULT '[]',
    github_repositories TEXT NOT NULL DEFAULT '[]',
    github_excluded_repositories TEXT NOT NULL DEFAULT '[]',
    azdo_projects     TEXT NOT NULL DEFAULT '[]',
    azdo_repositories TEXT NOT NULL DEFAULT '[]',
    code_scope_configured INTEGER NOT NULL DEFAULT 0,
    documentation_sources TEXT NOT NULL DEFAULT '[]',
    documentation_scope_configured INTEGER NOT NULL DEFAULT 0,
    automation_markers TEXT NOT NULL DEFAULT '',
    automation_handling TEXT NOT NULL DEFAULT 'exclude',
    transcript_dir    TEXT NOT NULL DEFAULT '',
    transcript_review_enabled INTEGER NOT NULL DEFAULT 1,
    habit_detection   TEXT NOT NULL DEFAULT 'on',
    habit_rules       TEXT NOT NULL DEFAULT '',
    habit_ai_match    TEXT NOT NULL DEFAULT 'on',
    context_deps      TEXT NOT NULL DEFAULT '',
    context_scope     TEXT NOT NULL DEFAULT '',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS standup_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    run_at          TEXT NOT NULL,
    standup_date    TEXT NOT NULL DEFAULT '',
    sprint_day      INTEGER NOT NULL DEFAULT 0,
    confidence_pct  INTEGER NOT NULL DEFAULT 0,
    report_json     TEXT NOT NULL DEFAULT '',
    -- Where this row came from: 'generated' or 'edited'. Provenance, not
    -- status: get_previous_report filters on status, so a third status value
    -- would silently drop corrected rows out of the next day's comparison.
    origin          TEXT NOT NULL DEFAULT 'generated',
    edited_from_id  INTEGER NOT NULL DEFAULT 0,
    delivery_status TEXT NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'success',
    error           TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS standup_updates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    standup_date TEXT NOT NULL,
    member       TEXT NOT NULL,
    update_text  TEXT NOT NULL DEFAULT '',
    images_json  TEXT NOT NULL DEFAULT '[]',
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS standup_reviews (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    run_id       INTEGER NOT NULL DEFAULT 0,
    standup_date TEXT NOT NULL DEFAULT '',
    reviewed_at  TEXT NOT NULL,
    review_json  TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'drafted',
    warnings_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS standup_transcripts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    path         TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    covered_date TEXT NOT NULL DEFAULT '',
    reviewed_at  TEXT NOT NULL,
    review_id    INTEGER NOT NULL DEFAULT 0,
    UNIQUE(session_id, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_standup_transcripts_date
    ON standup_transcripts (session_id, covered_date);
CREATE TABLE IF NOT EXISTS standup_gap_issues (
    fingerprint  TEXT PRIMARY KEY,
    category     TEXT NOT NULL DEFAULT '',
    title        TEXT NOT NULL DEFAULT '',
    issue_number INTEGER NOT NULL DEFAULT 0,
    issue_url    TEXT NOT NULL DEFAULT '',
    state        TEXT NOT NULL DEFAULT 'drafted',
    via          TEXT NOT NULL DEFAULT '',
    first_seen_at TEXT NOT NULL DEFAULT '',
    filed_at     TEXT NOT NULL DEFAULT '',
    last_seen_at TEXT NOT NULL DEFAULT '',
    last_commented_at TEXT NOT NULL DEFAULT '',
    occurrences  INTEGER NOT NULL DEFAULT 0,
    last_review_id INTEGER NOT NULL DEFAULT 0
);"""

# Its own constant so sessions.py's v25 migration can create exactly this table
# on a database that was migrated ahead of ever opening a StandupStore.
#
# UNIQUE(session_id, rule, handle) is the whole conflict policy: one verdict per
# change per rule, and a re-vote flips it instead of stacking a second row.
# Keyed by rule, not by member — excusing a PR for ``untracked-work`` says
# nothing about whether it is also an oversized change.
_STANDUP_PRACTICE_FEEDBACK_SCHEMA = """\
CREATE TABLE IF NOT EXISTS standup_practice_feedback (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    rule         TEXT NOT NULL,
    handle       TEXT NOT NULL,
    verdict      TEXT NOT NULL,
    note         TEXT NOT NULL DEFAULT '',
    member       TEXT NOT NULL DEFAULT '',
    subject      TEXT NOT NULL DEFAULT '',
    standup_date TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    UNIQUE(session_id, rule, handle)
);"""

_STANDUP_SCHEMA += "\n" + _STANDUP_PRACTICE_FEEDBACK_SCHEMA


# ---------------------------------------------------------------------------
# Serialisation helpers — StandupReport <-> JSON (same pattern as sessions.py)
# ---------------------------------------------------------------------------


def _standup_report_to_json(report: StandupReport) -> str:
    """Serialize a StandupReport to a JSON string.

    ``asdict`` recursively turns the frozen MemberUpdate tuples into dicts and
    the activity_counts tuple-of-tuples into nested lists; both are rebuilt with
    the correct types by ``_dict_to_standup_report``.
    """
    return json.dumps(asdict(report), ensure_ascii=False)


def _dict_to_evidence(items: object) -> tuple[ActivityEvidence, ...]:
    """Rebuild an evidence tuple from JSON-parsed dicts (missing → empty)."""
    if not isinstance(items, list):
        return ()
    return tuple(
        ActivityEvidence(
            kind=str(e.get("kind", "")),
            key=str(e.get("key", "")),
            title=str(e.get("title", "")),
            url=str(e.get("url", "")),
            repository=str(e.get("repository", "")),
            status=str(e.get("status", "")),
            timestamp=str(e.get("timestamp", "")),
            # One level deep in practice (commits under a PR); recursion keeps
            # the rebuild honest either way.
            children=_dict_to_evidence(e.get("children")),
            issue_type=str(e.get("issue_type", "")),
            parent_key=str(e.get("parent_key", "")),
            subtask=bool(e.get("subtask", False)),
            ticket_keys=tuple(str(k) for k in (e.get("ticket_keys") or []) if str(k)),
        )
        for e in items
        if isinstance(e, dict)
    )


def _dict_to_practices(items: object) -> tuple[PracticeSignal, ...]:
    """Rebuild practice signals from JSON-parsed dicts (missing → empty)."""
    if not isinstance(items, list):
        return ()
    return tuple(
        PracticeSignal(
            rule=str(p.get("rule", "")),
            title=str(p.get("title", "")),
            detail=str(p.get("detail", "")),
            # JSON turned each (label, url) tuple into a list — rebuild tuples.
            evidence=tuple((str(e[0]), str(e[1])) for e in p.get("evidence") or () if len(e) == 2),
            repeat=bool(p.get("repeat", False)),
            # Absent on reports written before feedback existed, which simply
            # means none of their signals can be voted on.
            handles=tuple(str(h) for h in (p.get("handles") or ()) if str(h)),
        )
        for p in items
        if isinstance(p, dict)
    )


def _dict_to_conflicts(items: object) -> tuple[ConflictCard, ...]:
    """Rebuild conflict cards from JSON-parsed dicts (missing → empty)."""
    if not isinstance(items, list):
        return ()
    return tuple(
        ConflictCard(
            fingerprint=str(c.get("fingerprint", "")),
            title=str(c.get("title", "")),
            detail=str(c.get("detail", "")),
            severity=str(c.get("severity", "medium")),
            entity_id=str(c.get("entity_id", "")),
            property_name=str(c.get("property_name", "")),
            # JSON turned each (source, value, label, url) tuple into a list —
            # rebuild tuples.
            claims=tuple(
                (str(cl[0]), str(cl[1]), str(cl[2]), str(cl[3])) for cl in c.get("claims") or () if len(cl) == 4
            ),
            recommended_action=str(c.get("recommended_action", "")),
            members=tuple(str(m) for m in (c.get("members") or ()) if str(m)),
        )
        for c in items
        if isinstance(c, dict)
    )


def _dict_to_ops_signals(items: object) -> tuple[OpsSignal, ...]:
    """Rebuild ops signals from JSON-parsed dicts (missing → empty)."""
    if not isinstance(items, list):
        return ()
    return tuple(
        OpsSignal(
            kind=str(s.get("kind", "")),
            family=str(s.get("family", "")),
            source=str(s.get("source", "")),
            count=int(s.get("count", 0)),
            resolved=int(s.get("resolved", 0)),
            severity=str(s.get("severity", "")),
            services=tuple(str(x) for x in (s.get("services") or ()) if str(x)),
            window_start=str(s.get("window_start", "")),
            window_end=str(s.get("window_end", "")),
            samples=tuple(str(x) for x in (s.get("samples") or ()) if str(x)),
        )
        for s in items
        if isinstance(s, dict)
    )


def _dict_to_standup_report(d: dict) -> StandupReport:
    """Reconstruct a StandupReport from a JSON-parsed dict.

    Uses ``.get()`` with defaults for every field so reports serialized by an
    older version (missing keys) still deserialize — see AGENTS.md
    "Frozen dataclass backward compatibility".
    """
    members = tuple(
        MemberUpdate(
            name=m.get("name", ""),
            summary=m.get("summary", ""),
            blockers=m.get("blockers", ""),
            progress_note=m.get("progress_note", ""),
            outlook=m.get("outlook", ""),
            source=m.get("source", "inferred"),
            self_report=m.get("self_report", ""),
            # JSON turned each (label, url) tuple into a list — rebuild tuples.
            links=tuple((str(li[0]), str(li[1])) for li in m.get("links", ()) if len(li) == 2),
            activity_count=int(m.get("activity_count", 0)),
            code_summary=m.get("code_summary", ""),
            code_links=tuple((str(li[0]), str(li[1])) for li in m.get("code_links", ()) if len(li) == 2),
            code_activity_count=int(m.get("code_activity_count", 0)),
            documentation_summary=m.get("documentation_summary", ""),
            documentation_links=tuple(
                (str(li[0]), str(li[1])) for li in m.get("documentation_links", ()) if len(li) == 2
            ),
            documentation_activity_count=int(m.get("documentation_activity_count", 0)),
            ticketing_summary=m.get("ticketing_summary", ""),
            ticketing_links=tuple((str(li[0]), str(li[1])) for li in m.get("ticketing_links", ()) if len(li) == 2),
            ticketing_activity_count=int(m.get("ticketing_activity_count", 0)),
            ticketing_evidence=_dict_to_evidence(m.get("ticketing_evidence")),
            code_evidence=_dict_to_evidence(m.get("code_evidence")),
            documentation_evidence=_dict_to_evidence(m.get("documentation_evidence")),
            practices=_dict_to_practices(m.get("practices")),
        )
        for m in d.get("member_updates", ())
    )
    # JSON turned each (source, count) tuple into a [source, count] list — rebuild tuples.
    counts = tuple((str(c[0]), int(c[1])) for c in d.get("activity_counts", ()) if len(c) == 2)
    skipped = tuple((str(s[0]), str(s[1])) for s in d.get("skipped_sources", ()) if len(s) == 2)
    category_coverage = tuple((str(item[0]), str(item[1])) for item in d.get("category_coverage", ()) if len(item) == 2)
    return StandupReport(
        date=d.get("date", ""),
        session_id=d.get("session_id", ""),
        sprint_name=d.get("sprint_name", ""),
        sprint_day=d.get("sprint_day", 0),
        sprint_total_days=d.get("sprint_total_days", 0),
        confidence_pct=d.get("confidence_pct", 0),
        confidence_label=d.get("confidence_label", ""),
        confidence_rationale=d.get("confidence_rationale", ""),
        confidence_delta=int(d.get("confidence_delta", 0)),
        confidence_trend=d.get("confidence_trend", ""),
        team_summary=d.get("team_summary", ""),
        member_updates=members,
        activity_counts=counts,
        activity_window=d.get("activity_window", ""),
        activity_window_start=d.get("activity_window_start", ""),
        activity_window_end=d.get("activity_window_end", ""),
        skipped_sources=skipped,
        unmet_sources=tuple(str(s) for s in d.get("unmet_sources", ())),
        category_coverage=category_coverage,
        my_name=d.get("my_name", ""),
        solo=bool(d.get("solo", False)),
        warnings=tuple(d.get("warnings", ())),
        images=tuple(d.get("images", ())),
        annotations=annotations_from(d.get("annotations")),
        practice_rollup=tuple((str(p[0]), int(p[1])) for p in d.get("practice_rollup", ()) if len(p) == 2),
        conflicts=_dict_to_conflicts(d.get("conflicts")),
        ops_signals=_dict_to_ops_signals(d.get("ops_signals")),
    )


# ---------------------------------------------------------------------------
# Serialisation helpers — TranscriptReview <-> JSON
# ---------------------------------------------------------------------------


def _review_to_json(review: TranscriptReview) -> str:
    """Serialize a TranscriptReview to a JSON string."""
    return json.dumps(asdict(review), ensure_ascii=False)


def _dict_to_claims(items: object) -> tuple[TranscriptClaim, ...]:
    """Rebuild a claim tuple from JSON-parsed dicts (missing → empty)."""
    if not isinstance(items, list):
        return ()
    return tuple(
        TranscriptClaim(
            member=str(c.get("member", "")),
            claim=str(c.get("claim", "")),
            quote=str(c.get("quote", "")),
            status=str(c.get("status", "")),
            matched_key=str(c.get("matched_key", "")),
            system_hint=str(c.get("system_hint", "")),
            artifact_hint=str(c.get("artifact_hint", "")),
            source_path=str(c.get("source_path", "")),
        )
        for c in items
        if isinstance(c, dict)
    )


def _dict_to_gaps(items: object) -> tuple[StandupGap, ...]:
    """Rebuild a gap tuple from JSON-parsed dicts (missing → empty)."""
    if not isinstance(items, list):
        return ()
    return tuple(
        StandupGap(
            fingerprint=str(g.get("fingerprint", "")),
            category=str(g.get("category", "")),
            scope=str(g.get("scope", "")),
            title=str(g.get("title", "")),
            detail=str(g.get("detail", "")),
            root_cause=str(g.get("root_cause", "")),
            priority=str(g.get("priority", "medium")),
            confidence=str(g.get("confidence", "medium")),
            feedback_kind=str(g.get("feedback_kind", "Improvement")),
            members=tuple(str(m) for m in g.get("members", ())),
            claims=_dict_to_claims(g.get("claims")),
            evidence=tuple(str(e) for e in g.get("evidence", ())),
            next_steps=tuple(str(s) for s in g.get("next_steps", ())),
            affected_systems=tuple(str(s) for s in g.get("affected_systems", ())),
            remedy=str(g.get("remedy", "")),
        )
        for g in items
        if isinstance(g, dict)
    )


def _dict_to_sources(items: object) -> tuple[TranscriptSource, ...]:
    """Rebuild a transcript-source tuple from JSON-parsed dicts (missing → empty)."""
    if not isinstance(items, list):
        return ()
    return tuple(
        TranscriptSource(
            path=str(s.get("path", "")),
            filename=str(s.get("filename", "")),
            fmt=str(s.get("fmt", "")),
            covered_date=str(s.get("covered_date", "")),
            char_count=int(s.get("char_count", 0) or 0),
            truncated=bool(s.get("truncated", False)),
            speakers=tuple(str(sp) for sp in s.get("speakers", ())),
            attribution=str(s.get("attribution", "labelled")),
            external=bool(s.get("external", False)),
        )
        for s in items
        if isinstance(s, dict)
    )


def _dict_to_review(d: dict) -> TranscriptReview:
    """Reconstruct a TranscriptReview from a JSON-parsed dict.

    ``.get()`` with a default for every field, so a review serialized by an
    older version still deserializes — see AGENTS.md "Frozen dataclass
    backward compatibility".
    """
    return TranscriptReview(
        review_id=int(d.get("review_id", 0) or 0),
        session_id=str(d.get("session_id", "")),
        standup_date=str(d.get("standup_date", "")),
        run_id=int(d.get("run_id", 0) or 0),
        reviewed_at=str(d.get("reviewed_at", "")),
        sources=_dict_to_sources(d.get("sources")),
        claims=_dict_to_claims(d.get("claims")),
        gaps=_dict_to_gaps(d.get("gaps")),
        config_suggestions=_dict_to_gaps(d.get("config_suggestions")),
        accuracy_note=str(d.get("accuracy_note", "")),
        claims_matched=int(d.get("claims_matched", 0) or 0),
        claims_missing=int(d.get("claims_missing", 0) or 0),
        claims_contradicted=int(d.get("claims_contradicted", 0) or 0),
        untracked_count=int(d.get("untracked_count", 0) or 0),
        llm_mode=str(d.get("llm_mode", "")),
        warnings=tuple(str(w) for w in d.get("warnings", ())),
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class StandupStore:
    """SQLite-backed store for standup config, run history, and self-updates.

    Uses the same database as SessionStore (sessions.db) with dedicated standup
    tables. Follows the same patterns: autocommit mode, context-manager support,
    explicit close.

    # See docs: "Session Management" — SQLite persistence
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.isolation_level = None  # autocommit
        self._conn.executescript(_STANDUP_SCHEMA)
        # Idempotent migration: add lead_minutes to standup_config tables created
        # before it existed (same try/except pattern SessionStore uses).
        try:
            self._conn.execute("ALTER TABLE standup_config ADD COLUMN lead_minutes INTEGER NOT NULL DEFAULT 10")
        except sqlite3.OperationalError:
            pass  # column already exists
        # Idempotent migration: screenshots pasted (Ctrl+V) into "My Update" — a
        # JSON list of file paths under ~/.yeaboi/attachments/, attached to the
        # summary LLM call as multimodal image blocks at run time.
        try:
            self._conn.execute("ALTER TABLE standup_updates ADD COLUMN images_json TEXT NOT NULL DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass  # column already exists
        # Idempotent migration: the user's comma-separated identity aliases
        # (GitHub handle, Jira display name, …) for alias-aware activity attribution.
        try:
            self._conn.execute("ALTER TABLE standup_config ADD COLUMN my_aliases TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # column already exists
        # Idempotent migration: Standup owns an explicit tracker/member scope.
        # The configured bit distinguishes "not chosen yet" from a deliberate
        # self-only roster (an empty team_members list).
        for statement in (
            """ALTER TABLE standup_config
               ADD COLUMN tracker_sources TEXT NOT NULL DEFAULT '["jira"]'""",
            """ALTER TABLE standup_config
               ADD COLUMN team_members TEXT NOT NULL DEFAULT '[]'""",
            """ALTER TABLE standup_config
               ADD COLUMN roster_configured INTEGER NOT NULL DEFAULT 0""",
            """ALTER TABLE standup_config
               ADD COLUMN code_sources TEXT NOT NULL DEFAULT '[]'""",
            """ALTER TABLE standup_config
               ADD COLUMN github_owners TEXT NOT NULL DEFAULT '[]'""",
            """ALTER TABLE standup_config
               ADD COLUMN github_repositories TEXT NOT NULL DEFAULT '[]'""",
            # Repos the owner picker would otherwise cover but the user deliberately
            # excluded — see code_scope.expand_github_owners. An exclusion list (not
            # an inclusion list) so a repo created after this was set is still scanned.
            """ALTER TABLE standup_config
               ADD COLUMN github_excluded_repositories TEXT NOT NULL DEFAULT '[]'""",
            """ALTER TABLE standup_config
               ADD COLUMN azdo_projects TEXT NOT NULL DEFAULT '[]'""",
            """ALTER TABLE standup_config
               ADD COLUMN azdo_repositories TEXT NOT NULL DEFAULT '[]'""",
            """ALTER TABLE standup_config
               ADD COLUMN code_scope_configured INTEGER NOT NULL DEFAULT 0""",
            """ALTER TABLE standup_config
               ADD COLUMN documentation_sources TEXT NOT NULL DEFAULT '[]'""",
            """ALTER TABLE standup_config
               ADD COLUMN documentation_scope_configured INTEGER NOT NULL DEFAULT 0""",
            # Service-hook/bot detection (see standup/automation.py): the user's
            # custom comma-separated content markers, and whether detected
            # automation is excluded from member credit ('exclude') or left
            # alone ('off').
            """ALTER TABLE standup_config
               ADD COLUMN automation_markers TEXT NOT NULL DEFAULT ''""",
            """ALTER TABLE standup_config
               ADD COLUMN automation_handling TEXT NOT NULL DEFAULT 'exclude'""",
            # Standup transcript review (standup/transcripts.py): an optional
            # external drop folder alongside the managed ~/.yeaboi/transcripts,
            # and the kill switch for the automatic sweep inside run_standup.
            """ALTER TABLE standup_config
               ADD COLUMN transcript_dir TEXT NOT NULL DEFAULT ''""",
            """ALTER TABLE standup_config
               ADD COLUMN transcript_review_enabled INTEGER NOT NULL DEFAULT 1""",
            # Engineering-practice detection (see standup/habits.py): whether
            # the deterministic habit signals run at all ('on'/'off'), and an
            # optional comma-separated subset of rule ids (empty = all of them).
            """ALTER TABLE standup_config
               ADD COLUMN habit_detection TEXT NOT NULL DEFAULT 'on'""",
            """ALTER TABLE standup_config
               ADD COLUMN habit_rules TEXT NOT NULL DEFAULT ''""",
            """ALTER TABLE standup_config
               ADD COLUMN habit_ai_match TEXT NOT NULL DEFAULT 'on'""",
            # Context-source toggles (projects/scope.py): a JSON list of dep
            # tokens; '' inherits the project default, '[]' is incognito.
            """ALTER TABLE standup_config
               ADD COLUMN context_deps TEXT NOT NULL DEFAULT ''""",
            # The scope a scheduled standup reads under (context/scope.py JSON);
            # '' means the caller's default. Kept off save_config's full upsert
            # so a config-form save never resets it.
            """ALTER TABLE standup_config
               ADD COLUMN context_scope TEXT NOT NULL DEFAULT ''""",
            # Edit-provenance columns (sessions.py v21/v26): a v21 version-number
            # collision could leave a DB stamped past 21 without them, and
            # several entry points (--standup-run, the MCP tools) open this
            # store without ever constructing a SessionStore. record_run,
            # get_previous_run and get_base_run all read `origin`, so heal here
            # too.
            """ALTER TABLE standup_history
               ADD COLUMN origin TEXT NOT NULL DEFAULT 'generated'""",
            """ALTER TABLE standup_history
               ADD COLUMN edited_from_id INTEGER NOT NULL DEFAULT 0""",
        ):
            try:
                self._conn.execute(statement)
            except sqlite3.OperationalError:
                pass  # column already exists

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> StandupStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Config ────────────────────────────────────────────────────────────

    def save_config(
        self,
        session_id: str,
        *,
        enabled: bool,
        time: str,
        weekdays: str,
        delivery_channels: list[str],
        lead_minutes: int = 10,
        timezone: str = "",
        repo_path: str = "",
        my_aliases: str = "",
        tracker_sources: list[str] | None = None,
        team_members: list[str] | None = None,
        roster_configured: bool = False,
        code_sources: list[str] | None = None,
        github_owners: list[str] | None = None,
        github_repositories: list[str] | None = None,
        github_excluded_repositories: list[str] | None = None,
        azdo_projects: list[str] | None = None,
        azdo_repositories: list[str] | None = None,
        code_scope_configured: bool = False,
        documentation_sources: list[str] | None = None,
        documentation_scope_configured: bool = False,
        automation_markers: str = "",
        automation_handling: str = "exclude",
        transcript_dir: str = "",
        transcript_review_enabled: bool = True,
        habit_detection: str = "on",
        habit_rules: str = "",
        habit_ai_match: str = "on",
    ) -> None:
        """Insert or update the standup schedule/delivery config for a session.

        ``time`` is the STANDUP time (e.g. "10:00"); the scheduler fires
        ``lead_minutes`` earlier. ``my_aliases`` is the user's comma-separated
        identity list across tools (GitHub handle, Jira display name, …) used
        for alias-aware activity attribution. ``automation_markers`` /
        ``automation_handling`` tune service-hook detection (standup/automation.py).
        ``transcript_dir`` is an optional EXTERNAL transcript folder (the managed
        ~/.yeaboi/transcripts is always swept); ``transcript_review_enabled``
        turns the automatic sweep inside run_standup off.

        NOTE: this writes EVERY column, so a caller that omits a keyword resets
        it to the default. Every full-pass call site must therefore pass every
        field — enforced by tests/unit/test_standup_config_call_sites.py.
        ``automation_handling`` tune service-hook detection (standup/automation.py);
        ``habit_detection`` / ``habit_rules`` tune practice detection
        (standup/habits.py), and ``habit_ai_match`` switches off the
        language-model pass that excuses a change belonging to a ticket it never
        names (standup/adjudicate.py) — a separate switch because it is the only
        part of practice detection that spends money.

        **This is a full upsert with defaulted keywords**, so a caller that omits
        a field resets it. Every call site must pass through the values it read.
        """
        now = self._now()
        channels_json = json.dumps(delivery_channels)
        tracker_sources_json = json.dumps(tracker_sources or ["jira"])
        team_members_json = json.dumps(team_members or [])
        code_sources_json = json.dumps(code_sources or [])
        github_owners_json = json.dumps(github_owners or [])
        github_repositories_json = json.dumps(github_repositories or [])
        github_excluded_repositories_json = json.dumps(github_excluded_repositories or [])
        azdo_projects_json = json.dumps(azdo_projects or [])
        azdo_repositories_json = json.dumps(azdo_repositories or [])
        documentation_sources_json = json.dumps(documentation_sources or [])
        logger.info(
            "Saving standup config: session=%s enabled=%s standup_time=%s lead=%d channels=%s",
            session_id,
            enabled,
            time,
            lead_minutes,
            delivery_channels,
        )
        self._conn.execute(
            """INSERT INTO standup_config
                   (session_id, enabled, time, lead_minutes, timezone, weekdays, delivery_channels,
                    repo_path, my_aliases, tracker_sources, team_members, roster_configured,
                    code_sources, github_owners, github_repositories, github_excluded_repositories,
                    azdo_projects, azdo_repositories, code_scope_configured,
                    documentation_sources, documentation_scope_configured,
                    automation_markers, automation_handling,
                    transcript_dir, transcript_review_enabled,
                    habit_detection, habit_rules, habit_ai_match, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                   enabled = excluded.enabled,
                   time = excluded.time,
                   lead_minutes = excluded.lead_minutes,
                   timezone = excluded.timezone,
                   weekdays = excluded.weekdays,
                   delivery_channels = excluded.delivery_channels,
                   repo_path = excluded.repo_path,
                   my_aliases = excluded.my_aliases,
                   tracker_sources = excluded.tracker_sources,
                   team_members = excluded.team_members,
                   roster_configured = excluded.roster_configured,
                   code_sources = excluded.code_sources,
                   github_owners = excluded.github_owners,
                   github_repositories = excluded.github_repositories,
                   github_excluded_repositories = excluded.github_excluded_repositories,
                   azdo_projects = excluded.azdo_projects,
                   azdo_repositories = excluded.azdo_repositories,
                   code_scope_configured = excluded.code_scope_configured,
                   documentation_sources = excluded.documentation_sources,
                   documentation_scope_configured = excluded.documentation_scope_configured,
                   automation_markers = excluded.automation_markers,
                   automation_handling = excluded.automation_handling,
                   transcript_dir = excluded.transcript_dir,
                   transcript_review_enabled = excluded.transcript_review_enabled,
                   habit_detection = excluded.habit_detection,
                   habit_rules = excluded.habit_rules,
                   habit_ai_match = excluded.habit_ai_match,
                   updated_at = excluded.updated_at""",
            (
                session_id,
                int(enabled),
                time,
                int(lead_minutes),
                timezone,
                weekdays,
                channels_json,
                repo_path,
                my_aliases,
                tracker_sources_json,
                team_members_json,
                int(roster_configured),
                code_sources_json,
                github_owners_json,
                github_repositories_json,
                github_excluded_repositories_json,
                azdo_projects_json,
                azdo_repositories_json,
                int(code_scope_configured),
                documentation_sources_json,
                int(documentation_scope_configured),
                automation_markers,
                automation_handling or "exclude",
                transcript_dir,
                int(transcript_review_enabled),
                habit_detection or "on",
                habit_rules,
                habit_ai_match or "on",
                now,
                now,
            ),
        )

    def get_enabled_schedule_sessions(self) -> list[str]:
        """Session ids with an enabled schedule, most recently updated first.

        The hub's schedule card prefers one of these over the bare latest session
        so an already-installed schedule stays visible/editable (instead of the
        wizard silently creating a second schedule for a newer session).
        """
        rows = self._conn.execute(
            "SELECT session_id FROM standup_config WHERE enabled = 1 ORDER BY updated_at DESC"
        ).fetchall()
        return [r[0] for r in rows]

    def get_latest_configured_session(self) -> str | None:
        """Newest session whose roster was actually confirmed, or None.

        The standup page targets "the latest session" across every mode, so any
        other activity that opens a session leaves the saved standup setup on an
        older one. The schedule card already works around this for schedules
        (see :meth:`get_enabled_schedule_sessions`); this is the same escape
        hatch for the setup itself. ``roster_configured`` is the filter because
        a row written by a half-finished walk is not a setup worth offering —
        the caller still validates the remaining, conditionally-required flags.
        """
        row = self._conn.execute(
            "SELECT session_id FROM standup_config WHERE roster_configured = 1 ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def load_config(self, session_id: str) -> dict | None:
        """Return the standup config for a session as a dict, or None if unset."""
        row = self._conn.execute(
            "SELECT session_id, enabled, time, timezone, weekdays, delivery_channels, repo_path, lead_minutes, "
            "my_aliases, tracker_sources, team_members, roster_configured, "
            "code_sources, github_repositories, azdo_projects, azdo_repositories, code_scope_configured, "
            "documentation_sources, documentation_scope_configured, automation_markers, automation_handling, "
            "transcript_dir, transcript_review_enabled, "
            "habit_detection, habit_rules, habit_ai_match, github_owners, github_excluded_repositories, "
            "context_scope "
            "FROM standup_config WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            channels = json.loads(row[5]) if row[5] else ["terminal"]
        except (json.JSONDecodeError, TypeError):
            channels = ["terminal"]
        try:
            tracker_sources = json.loads(row[9]) if row[9] else ["jira"]
        except (json.JSONDecodeError, TypeError):
            tracker_sources = ["jira"]
        try:
            team_members = json.loads(row[10]) if row[10] else []
        except (json.JSONDecodeError, TypeError):
            team_members = []

        def _json_list(value) -> list:
            try:
                parsed = json.loads(value) if value else []
                return parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, TypeError):
                return []

        azdo_projects = _json_list(row[14])
        legacy_azdo_repositories = _json_list(row[15])
        if not azdo_projects and legacy_azdo_repositories:
            azdo_projects = list(
                dict.fromkeys(
                    project
                    for repository in legacy_azdo_repositories
                    for project, separator, _name in [str(repository).partition("/")]
                    if separator and project
                )
            )

        return {
            "session_id": row[0],
            "enabled": bool(row[1]),
            "time": row[2],
            "timezone": row[3],
            "weekdays": row[4],
            "delivery_channels": channels,
            "repo_path": row[6],
            "lead_minutes": row[7] if row[7] is not None else 10,
            "my_aliases": row[8] or "",
            "tracker_sources": tracker_sources,
            "team_members": team_members,
            "roster_configured": bool(row[11]),
            "code_sources": _json_list(row[12]),
            # NOT derived from github_repositories when empty (unlike azdo_projects
            # above): a saved repo list is a deliberately narrow scope, and turning
            # "acme/api" into the owner "acme" would silently widen a standup to
            # every repo in the org. The picker offers that upgrade explicitly.
            "github_owners": _json_list(row[26]),
            "github_repositories": _json_list(row[13]),
            # Same never-widen rule as github_repositories: an excluded repo stays
            # excluded, but nothing here ever ADDS a repo — only the picker does.
            "github_excluded_repositories": _json_list(row[27]),
            "azdo_projects": azdo_projects,
            "azdo_repositories": legacy_azdo_repositories,
            "code_scope_configured": bool(row[16]),
            "documentation_sources": _json_list(row[17]),
            "documentation_scope_configured": bool(row[18]),
            "automation_markers": row[19] or "",
            "automation_handling": row[20] or "exclude",
            "transcript_dir": row[21] or "",
            # Default ON: a row written before this column existed still gets the
            # sweep, which is the behaviour a user who drops a transcript expects.
            "transcript_review_enabled": bool(row[22]) if row[22] is not None else True,
            "habit_detection": row[23] or "on",
            "habit_rules": row[24] or "",
            "habit_ai_match": row[25] or "on",
            "context_scope": _scope_dict(row[28]),
        }

    def set_context_scope(self, session_id: str, scope) -> None:
        """Persist the scope a session's standups read under (``None`` clears it).

        Its own writer rather than a ``save_config`` keyword: that method is a
        full upsert every config form must pass in full, and the scope is
        chosen on the run page, not the config form.
        """
        from yeaboi.context.scope import ContextScope

        payload = ""
        if scope is not None:
            payload = json.dumps(scope.to_dict() if isinstance(scope, ContextScope) else scope, sort_keys=True)
        now = self._now()
        cursor = self._conn.execute(
            "UPDATE standup_config SET context_scope = ?, updated_at = ? WHERE session_id = ?",
            (payload, now, session_id),
        )
        if (cursor.rowcount or 0) == 0:
            self._conn.execute(
                "INSERT INTO standup_config (session_id, context_scope, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, payload, now, now),
            )
        logger.info("Standup context scope %s for session=%s", "set" if payload else "cleared", session_id)

    def get_context_scope(self, session_id: str) -> dict | None:
        """The persisted scope dict for a session, or ``None``."""
        row = self._conn.execute(
            "SELECT context_scope FROM standup_config WHERE session_id = ?", (session_id,)
        ).fetchone()
        return _scope_dict(row[0]) if row else None

    # ── Self-reported updates ─────────────────────────────────────────────

    def save_my_update(
        self, session_id: str, standup_date: str, member: str, update_text: str, images: list[str] | None = None
    ) -> None:
        """Store a user-typed update for a member on a given date.

        A member submitting again for the same date overwrites the prior entry
        (delete-then-insert) so the latest text always wins.

        images: file paths of screenshots pasted into the update (Ctrl+V) —
            attached to the summary LLM call when the standup runs.
        """
        logger.info(
            "Saving self-reported update: session=%s date=%s member=%s images=%d",
            session_id,
            standup_date,
            member,
            len(images or []),
        )
        self._conn.execute(
            "DELETE FROM standup_updates WHERE session_id = ? AND standup_date = ? AND member = ?",
            (session_id, standup_date, member),
        )
        self._conn.execute(
            """INSERT INTO standup_updates (session_id, standup_date, member, update_text, images_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, standup_date, member, update_text, json.dumps(images or []), self._now()),
        )

    def get_my_updates(self, session_id: str, standup_date: str) -> dict[str, str]:
        """Return ``{member: update_text}`` for all self-reported updates on a date."""
        rows = self._conn.execute(
            "SELECT member, update_text FROM standup_updates WHERE session_id = ? AND standup_date = ?",
            (session_id, standup_date),
        ).fetchall()
        return {member: text for member, text in rows}

    def get_my_update_images(self, session_id: str, standup_date: str) -> dict[str, list[str]]:
        """Return ``{member: [image paths]}`` for self-reported updates on a date.

        Paths whose file no longer exists are pruned here so the engine only ever
        sees attachable screenshots (deleted files degrade silently).
        """
        rows = self._conn.execute(
            "SELECT member, images_json FROM standup_updates WHERE session_id = ? AND standup_date = ?",
            (session_id, standup_date),
        ).fetchall()
        out: dict[str, list[str]] = {}
        for member, images_json in rows:
            try:
                paths = json.loads(images_json) if images_json else []
            except (json.JSONDecodeError, TypeError):
                paths = []
            live = [p for p in paths if isinstance(p, str) and Path(p).exists()]
            if len(live) < len(paths):
                logger.warning("standup: %d pasted image(s) missing on disk for %s", len(paths) - len(live), member)
            if live:
                out[member] = live
        return out

    # ── Run history ───────────────────────────────────────────────────────

    def record_run(
        self,
        report: StandupReport,
        *,
        delivery_status: dict[str, bool] | None = None,
        status: str = "success",
        error: str = "",
        origin: str = "generated",
        edited_from_id: int = 0,
    ) -> int:
        """Persist a completed standup run and return its history row id."""
        report_json = _standup_report_to_json(report)
        cursor = self._conn.execute(
            """INSERT INTO standup_history
                   (session_id, run_at, standup_date, sprint_day, confidence_pct,
                    report_json, delivery_status, status, error, origin, edited_from_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                report.session_id,
                self._now(),
                report.date,
                report.sprint_day,
                report.confidence_pct,
                report_json,
                json.dumps(delivery_status or {}),
                status,
                error,
                origin,
                edited_from_id,
            ),
        )
        logger.info("Recorded standup run: session=%s date=%s status=%s", report.session_id, report.date, status)
        return int(cursor.lastrowid or 0)

    def get_latest_report(self, session_id: str) -> StandupReport | None:
        """Return the most recent StandupReport for a session, or None."""
        row = self._conn.execute(
            "SELECT report_json FROM standup_history WHERE session_id = ? ORDER BY run_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is None or not row[0]:
            return None
        try:
            return _dict_to_standup_report(json.loads(row[0]))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Failed to deserialize standup report for %s: %s", session_id, exc)
            return None

    def get_previous_run(self, session_id: str, before_date: str) -> tuple[int, str, int, StandupReport] | None:
        """Return ``(row id, origin, edited_from_id, report)`` for the newest report before ``before_date``.

        Date-scoped (not run-scoped) so a same-day rerun never becomes
        "yesterday" — the engine uses this as the previous standup when
        comparing each member's update day-over-day.

        The row id and origin ride along because a *corrected* previous standup
        is worth more to the next run than a generated one: it says the team
        looked at this and told us it was wrong, and the engine can go and read
        exactly what they changed.

        ``edited_from_id`` rides along because that is the row the corrections
        are filed under. A log is anchored to the artifact it was written
        against, so looking it up by *this* row's id — the corrected one — finds
        nothing, and the whole feed-forward hint quietly never fires.
        """
        row = self._conn.execute(
            "SELECT id, origin, edited_from_id, report_json FROM standup_history "
            "WHERE session_id = ? AND standup_date != '' AND standup_date < ? "
            "AND status IN ('success', 'partial') "
            "ORDER BY standup_date DESC, run_at DESC LIMIT 1",
            (session_id, before_date),
        ).fetchone()
        if row is None or not row[3]:
            return None
        try:
            return (
                int(row[0]),
                str(row[1] or "generated"),
                int(row[2] or 0),
                _dict_to_standup_report(json.loads(row[3])),
            )
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Failed to deserialize previous standup report for %s: %s", session_id, exc)
            return None

    def get_previous_report(self, session_id: str, before_date: str) -> StandupReport | None:
        """The previous standup's report alone. See :meth:`get_previous_run`."""
        found = self.get_previous_run(session_id, before_date)
        return found[3] if found else None

    def get_history(self, session_id: str, limit: int = 30) -> list[dict]:
        """Return recent run metadata (newest first) for a session.

        Each row carries its ``id`` so callers (the saved-runs hub) can reopen or
        delete a specific run via ``get_run_by_id`` / ``delete_run``.
        """
        rows = self._conn.execute(
            "SELECT id, run_at, standup_date, sprint_day, confidence_pct, status "
            "FROM standup_history WHERE session_id = ? ORDER BY run_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [
            {
                "id": r[0],
                "run_at": r[1],
                "standup_date": r[2],
                "sprint_day": r[3],
                "confidence_pct": r[4],
                "status": r[5],
            }
            for r in rows
        ]

    def get_run_by_id(self, run_id: int) -> StandupReport | None:
        """Return the StandupReport for a single history row, or None if missing/corrupt."""
        row = self._conn.execute(
            "SELECT report_json FROM standup_history WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None or not row[0]:
            return None
        try:
            return _dict_to_standup_report(json.loads(row[0]))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Failed to deserialize standup run id=%s: %s", run_id, exc)
            return None

    def get_base_run(self, *, session_id: str = "", run_id: int = 0) -> tuple[int, StandupReport] | None:
        """Return ``(id, report)`` for the *generated* run a correction log is anchored to.

        Not `get_latest_report`, which deliberately returns the **corrected**
        row — that is what makes "edits become the artifact" true for every
        reader. A correction log is the opposite question: it was recorded
        against the original, and it is replayed onto the original, so a caller
        that replays it onto the latest row applies every earlier correction a
        second time. Appends and notes have no compare-and-swap to save them, so
        they duplicate silently, once per call.

        Given a ``run_id`` that names a corrected row, this follows
        ``edited_from_id`` back to its parent, so naming either row in a chain
        anchors to the same log.
        """
        if run_id:
            row = self._conn.execute(
                "SELECT id, origin, edited_from_id, report_json FROM standup_history WHERE id = ?",
                (run_id,),
            ).fetchone()
            if row is not None and row[1] == "edited" and row[2]:
                row = self._conn.execute(
                    "SELECT id, origin, edited_from_id, report_json FROM standup_history WHERE id = ?",
                    (row[2],),
                ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT id, origin, edited_from_id, report_json FROM standup_history "
                "WHERE session_id = ? AND origin != 'edited' ORDER BY run_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row is None or not row[3]:
            return None
        try:
            return int(row[0]), _dict_to_standup_report(json.loads(row[3]))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Failed to deserialize standup base run id=%s: %s", row[0], exc)
            return None

    def delete_run(self, run_id: int) -> bool:
        """Delete a single standup history row. Returns True if a row was removed."""
        cursor = self._conn.execute("DELETE FROM standup_history WHERE id = ?", (run_id,))
        deleted = (cursor.rowcount or 0) > 0
        if deleted:
            drop_run_labels(self._db_path, "standup", run_id)
            logger.info("Deleted standup run id=%s", run_id)
        return deleted

    def get_latest_run_id(self, session_id: str) -> int | None:
        """The history row id of the most recent run, or None.

        The standup page loads its report through ``get_latest_report``, which
        answers with no id — and a thumbs-down has to write one row back. This
        is the missing half, ordered identically so the two can never disagree.
        """
        row = self._conn.execute(
            "SELECT id FROM standup_history WHERE session_id = ? ORDER BY run_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return int(row[0]) if row else None

    def update_run_report(self, run_id: int, report: StandupReport) -> bool:
        """Rewrite a stored run's report. Returns True if a row was updated.

        The one mutating path over ``standup_history``, and deliberately narrow:
        only ``report_json`` and the confidence it carries, never the run's
        identity or timestamps. Used when a thumbs-down removes a signal, so
        every later read — the TUI, an export, a re-share — sees the corrected
        report rather than each filtering the same signal out again.
        """
        cursor = self._conn.execute(
            "UPDATE standup_history SET report_json = ?, confidence_pct = ? WHERE id = ?",
            (_standup_report_to_json(report), report.confidence_pct, run_id),
        )
        updated = (cursor.rowcount or 0) > 0
        if updated:
            logger.info("Updated standup run id=%s after practice feedback", run_id)
        else:
            logger.warning("Standup run id=%s not found — practice feedback report rewrite skipped", run_id)
        return updated

    # ── Practice feedback ledger (see standup/practice_feedback.py) ───────

    def record_practice_feedback(
        self,
        session_id: str,
        *,
        rule: str,
        handle: str,
        verdict: str,
        note: str = "",
        member: str = "",
        subject: str = "",
        standup_date: str = "",
    ) -> None:
        """Upsert one verdict about one change.

        ``ON CONFLICT`` rather than an insert: the same change can be voted on
        again tomorrow (it is still open), and the team's latest word is the only
        one that should count. ``created_at`` moves with it so the prompt's
        "most recent examples" window means what it says.
        """
        self._conn.execute(
            """INSERT INTO standup_practice_feedback
                   (session_id, rule, handle, verdict, note, member, subject, standup_date, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, rule, handle) DO UPDATE SET
                   verdict = excluded.verdict,
                   note = excluded.note,
                   member = excluded.member,
                   subject = excluded.subject,
                   standup_date = excluded.standup_date,
                   created_at = excluded.created_at""",
            (session_id, rule, handle, verdict, note, member, subject, standup_date, self._now()),
        )

    def load_practice_feedback(self, session_id: str, limit: int = 0) -> list[dict]:
        """Every verdict for a session, newest first. Unbounded by default.

        No cap, deliberately. A thumbs-down promises a change is excused
        *forever* (see practice_feedback.py), and a LIMIT here would quietly
        break that promise at the worst moment: past the cap the oldest excuses
        fall out of the window and a signal someone already answered fires again
        at the same person, months later, with no way to tell why.

        The prompt is what actually needs bounding, and it is bounded where the
        examples are chosen (``_MAX_CORRECTIONS`` / ``_MAX_CONFIRMATIONS``). This
        row is eight short columns keyed by ``(session_id, rule, handle)`` with
        one row per change ever voted on, so reading them all is cheap.
        ``limit`` stays available for callers that want a page of recent
        verdicts; 0 means all of them.
        """
        sql = (
            "SELECT rule, handle, verdict, note, member, subject, standup_date, created_at "
            "FROM standup_practice_feedback WHERE session_id = ? ORDER BY created_at DESC, id DESC"
        )
        params: tuple = (session_id,)
        if limit > 0:
            sql += " LIMIT ?"
            params = (session_id, limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "rule": r[0],
                "handle": r[1],
                "verdict": r[2],
                "note": r[3],
                "member": r[4],
                "subject": r[5],
                "standup_date": r[6],
                "created_at": r[7],
            }
            for r in rows
        ]

    # ── Team-wide (cross-session) reads — used by ceremony_history to feed
    #    Planning / Analysis with the team's recent standups. standup_history has
    #    no project_name column, so these are recency-based (team-wide).

    def get_recent_reports(self, limit: int = 10, run_ids: tuple[int, ...] | None = None) -> list[StandupReport]:
        """Return recent StandupReports across ALL sessions, newest first.

        ``run_ids`` is the hard filter a resolved context scope hands over:
        ``None`` reads every row, ``()`` none. ``limit`` 0 means no limit.
        """
        where, params = id_filter(run_ids)
        limit_sql, limit_params = limit_clause(limit)
        rows = self._conn.execute(
            f"SELECT report_json FROM standup_history WHERE status = 'success' AND {where} "  # noqa: S608 — placeholders only
            f"ORDER BY run_at DESC{limit_sql}",
            (*params, *limit_params),
        ).fetchall()
        reports: list[StandupReport] = []
        for row in rows:
            if not row[0]:
                continue
            try:
                reports.append(_dict_to_standup_report(json.loads(row[0])))
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                logger.warning("Failed to deserialize a standup report: %s", exc)
        return reports

    def get_run_row_by_date(self, session_id: str, standup_date: str) -> int:
        """Return the history row id of the newest usable run ON a date, or 0.

        Scoped like ``get_previous_report`` (success/partial only) so a
        transcript review audits the run the team actually saw, not a failed
        attempt from the same morning.
        """
        row = self._conn.execute(
            "SELECT id FROM standup_history "
            "WHERE session_id = ? AND standup_date = ? AND status IN ('success', 'partial') "
            "ORDER BY run_at DESC LIMIT 1",
            (session_id, standup_date),
        ).fetchone()
        return int(row[0]) if row else 0

    # ── Transcript reviews ────────────────────────────────────────────────

    def record_review(self, review: TranscriptReview, *, status: str = "drafted") -> int:
        """Persist a transcript review and return its row id."""
        cursor = self._conn.execute(
            """INSERT INTO standup_reviews
                   (session_id, run_id, standup_date, reviewed_at, review_json, status, warnings_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                review.session_id,
                review.run_id,
                review.standup_date,
                review.reviewed_at or self._now(),
                _review_to_json(review),
                status,
                json.dumps(list(review.warnings)),
            ),
        )
        review_id = int(cursor.lastrowid or 0)
        logger.info(
            "Recorded transcript review: session=%s date=%s id=%d gaps=%d suggestions=%d",
            review.session_id,
            review.standup_date,
            review_id,
            len(review.gaps),
            len(review.config_suggestions),
        )
        return review_id

    def get_review(self, review_id: int) -> TranscriptReview | None:
        """Return one review by row id, or None if missing/corrupt."""
        row = self._conn.execute("SELECT id, review_json FROM standup_reviews WHERE id = ?", (review_id,)).fetchone()
        if row is None or not row[1]:
            return None
        try:
            review = _dict_to_review(json.loads(row[1]))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Failed to deserialize transcript review id=%s: %s", review_id, exc)
            return None
        # The id is assigned by SQLite on insert, so the serialized copy predates
        # it; hand callers back a review that knows its own row.
        return replace(review, review_id=int(row[0]))

    def get_reviews(self, session_id: str, limit: int = 30) -> list[dict]:
        """Return recent review metadata (newest first) for a session."""
        rows = self._conn.execute(
            "SELECT id, run_id, standup_date, reviewed_at, status FROM standup_reviews "
            "WHERE session_id = ? ORDER BY reviewed_at DESC, id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [{"id": r[0], "run_id": r[1], "standup_date": r[2], "reviewed_at": r[3], "status": r[4]} for r in rows]

    def get_latest_review(self, session_id: str) -> TranscriptReview | None:
        """Return the most recent transcript review for a session, or None."""
        row = self._conn.execute(
            "SELECT id FROM standup_reviews WHERE session_id = ? ORDER BY reviewed_at DESC, id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return self.get_review(int(row[0])) if row else None

    def set_review_status(self, review_id: int, status: str) -> None:
        """Update a review's filing status ('drafted' | 'filed' | 'partial')."""
        self._conn.execute("UPDATE standup_reviews SET status = ? WHERE id = ?", (status, review_id))

    # ── Transcript bookkeeping ────────────────────────────────────────────

    def mark_transcript_reviewed(
        self, session_id: str, *, path: str, content_hash: str, covered_date: str, review_id: int
    ) -> None:
        """Record that a transcript has been reviewed, keyed by content hash.

        Content-keyed on purpose: renaming or re-dropping the same file must not
        re-spend an LLM call, while editing it genuinely is new material.
        """
        self._conn.execute(
            """INSERT INTO standup_transcripts
                   (session_id, path, content_hash, covered_date, reviewed_at, review_id)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, content_hash) DO UPDATE SET
                   path = excluded.path,
                   covered_date = excluded.covered_date,
                   reviewed_at = excluded.reviewed_at,
                   review_id = excluded.review_id""",
            (session_id, path, content_hash, covered_date, self._now(), review_id),
        )

    def reviewed_transcript_hashes(self, session_id: str) -> set[str]:
        """Return the content hashes already reviewed for a session."""
        rows = self._conn.execute(
            "SELECT content_hash FROM standup_transcripts WHERE session_id = ?", (session_id,)
        ).fetchall()
        return {r[0] for r in rows}

    def reviewed_dates(self, session_id: str, *, since: str = "") -> set[str]:
        """Standup dates that a transcript was reviewed FOR.

        ``mark_transcript_reviewed`` fires for every source in a group regardless
        of what the review concluded, so this table means exactly "a transcript
        covering that date was read" — a stronger predicate than
        ``standup_reviews``, which can hold a review with no report behind it.
        Backed by ``idx_standup_transcripts_date``.
        """
        sql = "SELECT DISTINCT covered_date FROM standup_transcripts WHERE session_id = ? AND covered_date != ''"
        params: list = [session_id]
        if since:
            sql += " AND covered_date >= ?"
            params.append(since)
        return {r[0] for r in self._conn.execute(sql, params).fetchall()}

    def run_dates(self, session_id: str, *, since: str = "", before: str = "") -> set[str]:
        """Distinct dates a standup actually ran, over a half-open ``[since, before)``.

        Counts only ``success``/``partial`` runs — the same scoping
        ``get_run_row_by_date`` uses to decide a report exists at all, so a
        failed run is never something to be reproached for not transcribing.

        Deliberately not built on ``get_history``: that limit is in ROWS while
        this question is in DAYS, so a team that reruns standup twice a day
        would silently shorten the window.
        """
        sql = (
            "SELECT DISTINCT standup_date FROM standup_history "
            "WHERE session_id = ? AND standup_date != '' AND status IN ('success', 'partial')"
        )
        params: list = [session_id]
        if since:
            sql += " AND standup_date >= ?"
            params.append(since)
        if before:
            sql += " AND standup_date < ?"
            params.append(before)
        return {r[0] for r in self._conn.execute(sql, params).fetchall()}

    # ── Gap → GitHub issue ledger (cross-session by design) ───────────────

    def get_gap_issue(self, fingerprint: str) -> dict | None:
        """Return the issue ledger row for a gap fingerprint, or None."""
        row = self._conn.execute(
            "SELECT fingerprint, category, title, issue_number, issue_url, state, via, "
            "first_seen_at, filed_at, last_seen_at, last_commented_at, occurrences, last_review_id "
            "FROM standup_gap_issues WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        if row is None:
            return None
        return {
            "fingerprint": row[0],
            "category": row[1],
            "title": row[2],
            "issue_number": int(row[3] or 0),
            "issue_url": row[4] or "",
            "state": row[5] or "drafted",
            "via": row[6] or "",
            "first_seen_at": row[7] or "",
            "filed_at": row[8] or "",
            "last_seen_at": row[9] or "",
            "last_commented_at": row[10] or "",
            "occurrences": int(row[11] or 0),
            "last_review_id": int(row[12] or 0),
        }

    def upsert_gap_issue(
        self,
        fingerprint: str,
        *,
        category: str = "",
        title: str = "",
        issue_number: int | None = None,
        issue_url: str | None = None,
        state: str | None = None,
        via: str | None = None,
        filed_at: str | None = None,
        last_commented_at: str | None = None,
        review_id: int = 0,
        bump_occurrence: bool = True,
    ) -> dict:
        """Insert or update a gap's ledger row and return it.

        Every optional field uses COALESCE-on-NULL semantics: passing None keeps
        whatever is stored. That matters most for ``filed_at`` and
        ``issue_number`` — a later "seen again" must never erase the fact that
        the gap already has an issue, which is exactly how dedup would silently
        start filing duplicates.
        """
        now = self._now()
        self._conn.execute(
            """INSERT INTO standup_gap_issues
                   (fingerprint, category, title, issue_number, issue_url, state, via,
                    first_seen_at, filed_at, last_seen_at, last_commented_at, occurrences, last_review_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(fingerprint) DO UPDATE SET
                   category = excluded.category,
                   title = excluded.title,
                   issue_number = COALESCE(?, standup_gap_issues.issue_number),
                   issue_url = COALESCE(?, standup_gap_issues.issue_url),
                   state = COALESCE(?, standup_gap_issues.state),
                   via = COALESCE(?, standup_gap_issues.via),
                   filed_at = COALESCE(?, standup_gap_issues.filed_at),
                   last_commented_at = COALESCE(?, standup_gap_issues.last_commented_at),
                   last_seen_at = excluded.last_seen_at,
                   occurrences = standup_gap_issues.occurrences + ?,
                   last_review_id = excluded.last_review_id""",
            (
                fingerprint,
                category,
                title,
                issue_number or 0,
                issue_url or "",
                state or "drafted",
                via or "",
                now,
                filed_at or "",
                now,
                last_commented_at or "",
                1 if bump_occurrence else 0,
                review_id,
                issue_number,
                issue_url,
                state,
                via,
                filed_at,
                last_commented_at,
                1 if bump_occurrence else 0,
            ),
        )
        row = self.get_gap_issue(fingerprint)
        return row if row is not None else {}

    def get_gap_issues(self, limit: int = 50) -> list[dict]:
        """Return the gap ledger, most recently seen first (cross-session)."""
        rows = self._conn.execute(
            "SELECT fingerprint FROM standup_gap_issues ORDER BY last_seen_at DESC LIMIT ?", (limit,)
        ).fetchall()
        out: list[dict] = []
        for (fingerprint,) in rows:
            entry = self.get_gap_issue(fingerprint)
            if entry is not None:
                out.append(entry)
        return out

    def get_all_history(self, limit: int = 100, run_ids: tuple[int, ...] | None = None) -> list[dict]:
        """Return recent standup run metadata across ALL sessions (for cadence + the hub).

        ``run_ids`` narrows to those rows (``()`` = none); ``limit`` 0 = every row.
        """
        where, params = id_filter(run_ids)
        limit_sql, limit_params = limit_clause(limit)
        rows = self._conn.execute(
            "SELECT id, session_id, run_at, standup_date, sprint_day, confidence_pct, status "  # noqa: S608 — placeholders only
            f"FROM standup_history WHERE {where} ORDER BY run_at DESC{limit_sql}",
            (*params, *limit_params),
        ).fetchall()
        return [
            {
                "id": r[0],
                "session_id": r[1],
                "run_at": r[2],
                "standup_date": r[3],
                "sprint_day": r[4],
                "confidence_pct": r[5],
                "status": r[6],
            }
            for r in rows
        ]
