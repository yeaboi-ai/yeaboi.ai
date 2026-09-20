"""SQLite store for the Retro mode.

Persists each completed retrospective in the shared ~/.scrum-agent/sessions.db:
- ``retro_history`` — every run's serialized RetroReport (all cards + participants)

Follows the exact patterns used by StandupStore (standup/store.py): a separate
store class opening its own connection to the same DB, autocommit mode, context
manager support, idempotent CREATE-IF-NOT-EXISTS schema. The schema constant is
also referenced by sessions.py's v7 migration so an existing DB gets the table.

# See docs: "Session Management" — SQLite persistence, schema versioning
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from yeaboi.agent.state import RetroCard, RetroReport, annotations_from
from yeaboi.context._sql import id_filter, limit_clause
from yeaboi.context.labels import drop_run_labels

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema — referenced by sessions.py migration v7 AND created on store open
# ---------------------------------------------------------------------------

_RETRO_SCHEMA = """\
CREATE TABLE IF NOT EXISTS retro_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    run_at       TEXT NOT NULL,
    retro_date   TEXT NOT NULL DEFAULT '',
    project_name TEXT NOT NULL DEFAULT '',
    card_count   INTEGER NOT NULL DEFAULT 0,
    report_json  TEXT NOT NULL DEFAULT '',
    -- Where this row came from: 'generated' or 'edited'. Provenance, not
    -- status: get_previous_report filters on status, so a third status value
    -- would silently drop corrected rows out of the next day's comparison.
    origin          TEXT NOT NULL DEFAULT 'generated',
    edited_from_id  INTEGER NOT NULL DEFAULT 0
);"""


# ---------------------------------------------------------------------------
# Serialisation helpers — RetroReport <-> JSON (same pattern as standup/store.py)
# ---------------------------------------------------------------------------


def _retro_report_to_json(report: RetroReport) -> str:
    """Serialize a RetroReport to a JSON string (asdict recurses into RetroCard)."""
    return json.dumps(asdict(report), ensure_ascii=False)


def _dict_to_retro_report(d: dict) -> RetroReport:
    """Reconstruct a RetroReport from a JSON-parsed dict.

    Uses ``.get()`` with defaults for every field so reports serialized by an
    older version (missing keys) still deserialize — see AGENTS.md
    "Frozen dataclass backward compatibility".
    """

    def _card(c: dict) -> RetroCard:
        return RetroCard(
            id=c.get("id", ""),
            grid=c.get("grid", ""),
            text=c.get("text", ""),
            author=c.get("author", ""),
            created_at=c.get("created_at", ""),
            origin=c.get("origin", "web"),
            # JSON turned each (emoji, count) tuple into an [emoji, count] list — rebuild tuples.
            reactions=tuple((str(r[0]), int(r[1])) for r in c.get("reactions", ()) if len(r) == 2),
            status=c.get("status", ""),
        )

    cards = tuple(_card(c) for c in d.get("cards", ()))
    # Carried action items are RetroCards too (last sprint's items + their statuses).
    carried = tuple(_card(c) for c in d.get("carried_action_items", ()))
    return RetroReport(
        date=d.get("date", ""),
        session_id=d.get("session_id", ""),
        project_name=d.get("project_name", ""),
        sprint_name=d.get("sprint_name", ""),
        cards=cards,
        participants=tuple(d.get("participants", ())),
        generated_at=d.get("generated_at", ""),
        carried_action_items=carried,
        annotations=annotations_from(d.get("annotations")),
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class RetroStore:
    """SQLite-backed store for completed retrospectives.

    Uses the same database as SessionStore (sessions.db) with a dedicated
    ``retro_history`` table. Follows the same patterns: autocommit mode,
    context-manager support, explicit close.

    # See docs: "Session Management" — SQLite persistence
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.isolation_level = None  # autocommit
        self._conn.executescript(_RETRO_SCHEMA)
        # Idempotent migration: edit-provenance columns (sessions.py v21/v26).
        # A v21 version-number collision could leave a shared DB stamped past
        # 21 without them, and the CLI and MCP tools open this store without
        # ever constructing a SessionStore; record_run and get_base_run read
        # `origin`, so heal here too.
        for statement in (
            "ALTER TABLE retro_history ADD COLUMN origin TEXT NOT NULL DEFAULT 'generated'",
            "ALTER TABLE retro_history ADD COLUMN edited_from_id INTEGER NOT NULL DEFAULT 0",
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

    def __enter__(self) -> RetroStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Run history ───────────────────────────────────────────────────────

    def record_run(self, report: RetroReport, *, origin: str = "generated", edited_from_id: int = 0) -> int:
        """Persist a completed retro and return its history row id."""
        report_json = _retro_report_to_json(report)
        cursor = self._conn.execute(
            """INSERT INTO retro_history
                   (session_id, run_at, retro_date, project_name, card_count, report_json,
                    origin, edited_from_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                report.session_id,
                self._now(),
                report.date,
                report.project_name,
                len(report.cards),
                report_json,
                origin,
                edited_from_id,
            ),
        )
        logger.info(
            "Recorded retro run: session=%s date=%s cards=%d",
            report.session_id,
            report.date,
            len(report.cards),
        )
        return int(cursor.lastrowid or 0)

    def get_latest_report(self, session_id: str) -> RetroReport | None:
        """Return the most recent RetroReport for a session, or None."""
        row = self._conn.execute(
            "SELECT report_json FROM retro_history WHERE session_id = ? ORDER BY run_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is None or not row[0]:
            return None
        try:
            return _dict_to_retro_report(json.loads(row[0]))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Failed to deserialize retro report for %s: %s", session_id, exc)
            return None

    def get_history(self, session_id: str, limit: int = 30) -> list[dict]:
        """Return recent retro run metadata (newest first) for a session.

        Each row carries its ``id`` so the saved-runs hub can reopen or delete a
        specific run via ``get_run_by_id`` / ``delete_run``.
        """
        rows = self._conn.execute(
            "SELECT id, run_at, retro_date, project_name, card_count FROM retro_history "
            "WHERE session_id = ? ORDER BY run_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [
            {"id": r[0], "run_at": r[1], "retro_date": r[2], "project_name": r[3], "card_count": r[4]} for r in rows
        ]

    def get_run_by_id(self, run_id: int) -> RetroReport | None:
        """Return the RetroReport for a single history row, or None if missing/corrupt."""
        row = self._conn.execute(
            "SELECT report_json FROM retro_history WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None or not row[0]:
            return None
        try:
            return _dict_to_retro_report(json.loads(row[0]))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Failed to deserialize retro run id=%s: %s", run_id, exc)
            return None

    def get_base_run(self, *, session_id: str = "", run_id: int = 0) -> tuple[int, RetroReport] | None:
        """Return ``(id, report)`` for the *generated* run a correction log is anchored to.

        See :meth:`StandupStore.get_base_run` — same reason, same shape. Not
        `get_latest_report`, which returns the corrected row on purpose; a log
        replayed onto that applies every earlier correction again, and appends
        and notes have no compare-and-swap to stop them duplicating.
        """
        if run_id:
            row = self._conn.execute(
                "SELECT id, origin, edited_from_id, report_json FROM retro_history WHERE id = ?",
                (run_id,),
            ).fetchone()
            if row is not None and row[1] == "edited" and row[2]:
                row = self._conn.execute(
                    "SELECT id, origin, edited_from_id, report_json FROM retro_history WHERE id = ?",
                    (row[2],),
                ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT id, origin, edited_from_id, report_json FROM retro_history "
                "WHERE session_id = ? AND origin != 'edited' ORDER BY run_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row is None or not row[3]:
            return None
        try:
            return int(row[0]), _dict_to_retro_report(json.loads(row[3]))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Failed to deserialize retro base run id=%s: %s", row[0], exc)
            return None

    def delete_run(self, run_id: int) -> bool:
        """Delete a single retro history row. Returns True if a row was removed."""
        cursor = self._conn.execute("DELETE FROM retro_history WHERE id = ?", (run_id,))
        deleted = (cursor.rowcount or 0) > 0
        if deleted:
            drop_run_labels(self._db_path, "retro", run_id)
            logger.info("Deleted retro run id=%s", run_id)
        return deleted

    # ── Team-wide (cross-session) reads — used by ceremony_history to feed
    #    Planning / Analysis with the team's recent retros regardless of which
    #    session they ran under. See docs: "Session Management".

    def get_recent_reports(
        self, limit: int = 5, project_name: str = "", run_ids: tuple[int, ...] | None = None
    ) -> list[RetroReport]:
        """Return recent RetroReports across ALL sessions, newest first.

        When ``project_name`` is given, rows matching it sort first (project-first),
        then by recency — so a plan for project X sees X's retros ahead of others'.
        ``run_ids`` is the hard filter a resolved context scope hands over
        (``None`` = every row, ``()`` = none) and replaces the name bias when given.
        ``limit`` 0 means no limit.
        """
        where, params = id_filter(run_ids)
        limit_sql, limit_params = limit_clause(limit)
        if project_name and run_ids is None:
            # (project_name = ?) is 1 for matches, 0 otherwise → matches sort first.
            rows = self._conn.execute(
                f"SELECT report_json FROM retro_history ORDER BY (project_name = ?) DESC, run_at DESC{limit_sql}",  # noqa: S608
                (project_name, *limit_params),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT report_json FROM retro_history WHERE {where} ORDER BY run_at DESC{limit_sql}",  # noqa: S608 — placeholders only
                (*params, *limit_params),
            ).fetchall()
        reports: list[RetroReport] = []
        for row in rows:
            if not row[0]:
                continue
            try:
                reports.append(_dict_to_retro_report(json.loads(row[0])))
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                logger.warning("Failed to deserialize a retro report: %s", exc)
        return reports

    def get_all_history(self, limit: int = 100, run_ids: tuple[int, ...] | None = None) -> list[dict]:
        """Return recent retro run metadata across ALL sessions (for cadence + the hub).

        ``run_ids`` narrows to those rows (``()`` = none); ``limit`` 0 = every row.
        """
        where, params = id_filter(run_ids)
        limit_sql, limit_params = limit_clause(limit)
        rows = self._conn.execute(
            "SELECT id, session_id, run_at, retro_date, project_name, card_count "  # noqa: S608 — placeholders only
            f"FROM retro_history WHERE {where} ORDER BY run_at DESC{limit_sql}",
            (*params, *limit_params),
        ).fetchall()
        return [
            {
                "id": r[0],
                "session_id": r[1],
                "run_at": r[2],
                "retro_date": r[3],
                "project_name": r[4],
                "card_count": r[5],
            }
            for r in rows
        ]
