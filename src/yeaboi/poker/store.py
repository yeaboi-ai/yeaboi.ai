"""SQLite store for the Scrum Poker mode.

Persists each completed poker session in the shared ~/.yeaboi/sessions.db:
- ``poker_history`` — every run's serialized PokerReport (all tickets + votes)

Follows the exact patterns used by RetroStore (retro/store.py): a separate
store class opening its own connection to the same DB, autocommit mode, context
manager support, idempotent CREATE-IF-NOT-EXISTS schema. The schema constant is
also referenced by sessions.py's v18 migration so an existing DB gets the table.

# See docs: "Session Management" — SQLite persistence, schema versioning
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from yeaboi.agent.state import PokerReport, PokerTicketResult, PokerVote
from yeaboi.context._sql import id_filter, limit_clause
from yeaboi.context.labels import drop_run_labels

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema — referenced by sessions.py migration v18 AND created on store open
# ---------------------------------------------------------------------------

_POKER_SCHEMA = """\
CREATE TABLE IF NOT EXISTS poker_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    run_at          TEXT NOT NULL,
    poker_date      TEXT NOT NULL DEFAULT '',
    project_name    TEXT NOT NULL DEFAULT '',
    source          TEXT NOT NULL DEFAULT '',
    scope_label     TEXT NOT NULL DEFAULT '',
    ticket_count    INTEGER NOT NULL DEFAULT 0,
    estimated_count INTEGER NOT NULL DEFAULT 0,
    report_json     TEXT NOT NULL DEFAULT ''
);"""


# ---------------------------------------------------------------------------
# Serialisation helpers — PokerReport <-> JSON (same pattern as retro/store.py)
# ---------------------------------------------------------------------------


def _poker_report_to_json(report: PokerReport) -> str:
    """Serialize a PokerReport to a JSON string (asdict recurses into tickets/votes)."""
    return json.dumps(asdict(report), ensure_ascii=False)


def _dict_to_poker_report(d: dict) -> PokerReport:
    """Reconstruct a PokerReport from a JSON-parsed dict.

    Uses ``.get()`` with defaults for every field so reports serialized by an
    older version (missing keys) still deserialize — see AGENTS.md
    "Frozen dataclass backward compatibility".
    """

    def _float_or_none(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _vote(v: dict) -> PokerVote:
        return PokerVote(voter=v.get("voter", ""), avatar=v.get("avatar", ""), value=v.get("value", ""))

    def _ticket(t: dict) -> PokerTicketResult:
        return PokerTicketResult(
            key=t.get("key", ""),
            url=t.get("url", ""),
            summary=t.get("summary", ""),
            description=t.get("description", ""),
            state=t.get("state", ""),
            assignee=t.get("assignee", ""),
            initial_points=_float_or_none(t.get("initial_points")),
            final_points=_float_or_none(t.get("final_points")),
            estimated=bool(t.get("estimated")),
            votes=tuple(_vote(v) for v in t.get("votes", ())),
            ai_note=t.get("ai_note", ""),
            duel_transcript=t.get("duel_transcript", ""),
            duel_low=t.get("duel_low", ""),
            duel_high=t.get("duel_high", ""),
        )

    return PokerReport(
        date=d.get("date", ""),
        session_id=d.get("session_id", ""),
        project_name=d.get("project_name", ""),
        source=d.get("source", ""),
        scope_label=d.get("scope_label", ""),
        tickets=tuple(_ticket(t) for t in d.get("tickets", ())),
        participants=tuple(d.get("participants", ())),
        generated_at=d.get("generated_at", ""),
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class PokerStore:
    """SQLite-backed store for completed poker sessions.

    Uses the same database as SessionStore (sessions.db) with a dedicated
    ``poker_history`` table. Follows the same patterns: autocommit mode,
    context-manager support, explicit close.

    # See docs: "Session Management" — SQLite persistence
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.isolation_level = None  # autocommit
        self._conn.executescript(_POKER_SCHEMA)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> PokerStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Run history ───────────────────────────────────────────────────────

    def record_run(self, report: PokerReport) -> int:
        """Persist a completed poker session and return its history row id."""
        estimated = sum(1 for t in report.tickets if t.estimated)
        cursor = self._conn.execute(
            """INSERT INTO poker_history
                   (session_id, run_at, poker_date, project_name, source, scope_label,
                    ticket_count, estimated_count, report_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                report.session_id,
                self._now(),
                report.date,
                report.project_name,
                report.source,
                report.scope_label,
                len(report.tickets),
                estimated,
                _poker_report_to_json(report),
            ),
        )
        logger.info(
            "Recorded poker run: session=%s date=%s tickets=%d estimated=%d",
            report.session_id,
            report.date,
            len(report.tickets),
            estimated,
        )
        return int(cursor.lastrowid or 0)

    def get_latest_report(self, session_id: str) -> PokerReport | None:
        """Return the most recent PokerReport for a session, or None."""
        row = self._conn.execute(
            "SELECT report_json FROM poker_history WHERE session_id = ? ORDER BY run_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is None or not row[0]:
            return None
        try:
            return _dict_to_poker_report(json.loads(row[0]))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Failed to deserialize poker report for %s: %s", session_id, exc)
            return None

    def get_history(self, session_id: str, limit: int = 30) -> list[dict]:
        """Return recent poker run metadata (newest first) for a session.

        Each row carries its ``id`` so the saved-runs hub can reopen or delete
        a specific run via ``get_run_by_id`` / ``delete_run``.
        """
        rows = self._conn.execute(
            "SELECT id, run_at, poker_date, project_name, source, scope_label, ticket_count, estimated_count "
            "FROM poker_history WHERE session_id = ? ORDER BY run_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [self._history_row(r) for r in rows]

    @staticmethod
    def _history_row(r: tuple) -> dict:
        return {
            "id": r[0],
            "run_at": r[1],
            "poker_date": r[2],
            "project_name": r[3],
            "source": r[4],
            "scope_label": r[5],
            "ticket_count": r[6],
            "estimated_count": r[7],
        }

    def get_run_by_id(self, run_id: int) -> PokerReport | None:
        """Return the PokerReport for a single history row, or None if missing/corrupt."""
        row = self._conn.execute(
            "SELECT report_json FROM poker_history WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None or not row[0]:
            return None
        try:
            return _dict_to_poker_report(json.loads(row[0]))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Failed to deserialize poker run id=%s: %s", run_id, exc)
            return None

    def delete_run(self, run_id: int) -> bool:
        """Delete a single poker history row. Returns True if a row was removed."""
        cursor = self._conn.execute("DELETE FROM poker_history WHERE id = ?", (run_id,))
        deleted = (cursor.rowcount or 0) > 0
        if deleted:
            drop_run_labels(self._db_path, "poker", run_id)
            logger.info("Deleted poker run id=%s", run_id)
        return deleted

    def get_all_history(self, limit: int = 100, run_ids: tuple[int, ...] | None = None) -> list[dict]:
        """Return recent poker run metadata across ALL sessions (for the hub).

        ``run_ids`` narrows to those rows (``()`` = none); ``limit`` 0 = every row.
        """
        where, params = id_filter(run_ids)
        limit_sql, limit_params = limit_clause(limit)
        rows = self._conn.execute(
            "SELECT id, run_at, poker_date, project_name, source, scope_label, ticket_count, estimated_count, "  # noqa: S608 — placeholders only
            f"session_id FROM poker_history WHERE {where} ORDER BY run_at DESC{limit_sql}",
            (*params, *limit_params),
        ).fetchall()
        return [{**self._history_row(r), "session_id": r[8]} for r in rows]

    def get_recent_reports(self, limit: int = 5, run_ids: tuple[int, ...] | None = None) -> list[PokerReport]:
        """Recent PokerReports across ALL sessions, newest first — the cross-mode read.

        ``run_ids`` is the hard filter a resolved context scope hands over; ``limit`` 0 = no limit.
        """
        where, params = id_filter(run_ids)
        limit_sql, limit_params = limit_clause(limit)
        rows = self._conn.execute(
            f"SELECT report_json FROM poker_history WHERE {where} ORDER BY run_at DESC{limit_sql}",  # noqa: S608
            (*params, *limit_params),
        ).fetchall()
        reports: list[PokerReport] = []
        for row in rows:
            if not row[0]:
                continue
            try:
                reports.append(_dict_to_poker_report(json.loads(row[0])))
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                logger.warning("Failed to deserialize a poker report: %s", exc)
        return reports
