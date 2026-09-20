"""SQLite store for the Roadmap intake card.

Persists in the shared ~/.scrum-agent/sessions.db:
- ``roadmaps``        — the list of saved roadmaps (source + latest analysis
  inline), managed like planning projects: open / re-analyze / delete. Not
  keyed by session_id: a roadmap exists *before* any planning session.
- ``roadmap_history`` — every analysis run's serialized RoadmapAnalysis
  (append-only run log).
- ``roadmap_config``  — LEGACY v10 singleton source row (id=1). Superseded by
  the ``roadmaps`` table in schema v11; retained so the migration can seed the
  first ``roadmaps`` row from it.

Follows the exact patterns used by ReportingStore (reporting/store.py): a
separate store class opening its own connection to the same DB, autocommit
mode, context-manager support, idempotent CREATE-IF-NOT-EXISTS schema. The
``_ROADMAP_SCHEMA`` constant is also referenced by sessions.py's v10/v11
migrations so an existing DB gets the tables.

# See docs: "Session Management" — SQLite persistence, schema versioning
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path, PurePath

from yeaboi.agent.state import RoadmapAnalysis, RoadmapProject, annotations_from
from yeaboi.roadmap.ingest import RoadmapSource

logger = logging.getLogger(__name__)

_ROADMAP_FILE_SUFFIXES = {"md", "txt", "rst", "pdf", "docx", "pptx"}


def friendly_label(raw: str) -> str:
    """Humanize a file-ish roadmap label: 'q3-2026-roadmap.md' → 'Q3 2026 Roadmap'.

    Labels that already look like human titles (contain a space, e.g. a
    Confluence/Notion page title) pass through unchanged. Applied both when
    saving and when reading, so rows saved with a raw filename display nicely.
    """
    label = raw.strip()
    if not label:
        return label
    if "/" in label or "\\" in label:  # a path — keep just the file name
        label = PurePath(label.replace("\\", "/")).name
    stem, dot, ext = label.rpartition(".")
    if dot and ext.lower() in _ROADMAP_FILE_SUFFIXES:
        label = stem
    if " " in label:  # already a human title
        return label
    words = [w for w in re.split(r"[-_]+", label) if w]
    if not words:
        return raw.strip()
    pretty = []
    for w in words:
        if re.fullmatch(r"[qQ][1-4]", w):
            pretty.append(w.upper())  # quarter token: q3 → Q3
        elif w.isdigit():
            pretty.append(w)  # year / number — leave as-is
        else:
            pretty.append(w[:1].upper() + w[1:])
    return " ".join(pretty)


# ---------------------------------------------------------------------------
# Schema — referenced by sessions.py migration v10 AND created on store open
# ---------------------------------------------------------------------------

_ROADMAP_SCHEMA = """\
CREATE TABLE IF NOT EXISTS roadmap_config (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    source_type    TEXT NOT NULL DEFAULT '',
    source_locator TEXT NOT NULL DEFAULT '',
    source_label   TEXT NOT NULL DEFAULT '',
    updated_at     TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS roadmap_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at         TEXT NOT NULL,
    source_type    TEXT NOT NULL DEFAULT '',
    source_locator TEXT NOT NULL DEFAULT '',
    project_count  INTEGER NOT NULL DEFAULT 0,
    analysis_json  TEXT NOT NULL DEFAULT '',
    -- Where this row came from: 'generated' or 'edited'. Provenance, not
    -- status: get_previous_report filters on status, so a third status value
    -- would silently drop corrected rows out of the next day's comparison.
    origin          TEXT NOT NULL DEFAULT 'generated',
    edited_from_id  INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS roadmaps (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    label          TEXT NOT NULL DEFAULT '',
    source_type    TEXT NOT NULL DEFAULT '',
    source_locator TEXT NOT NULL DEFAULT '',
    source_label   TEXT NOT NULL DEFAULT '',
    analysis_json  TEXT NOT NULL DEFAULT '',
    project_count  INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL DEFAULT '',
    updated_at     TEXT NOT NULL DEFAULT ''
);"""


# ---------------------------------------------------------------------------
# Serialisation helpers — RoadmapAnalysis <-> JSON (same pattern as reporting)
# ---------------------------------------------------------------------------


def _analysis_to_json(analysis: RoadmapAnalysis) -> str:
    """Serialize a RoadmapAnalysis to a JSON string (asdict recurses into projects)."""
    return json.dumps(asdict(analysis), ensure_ascii=False)


def _dict_to_analysis(d: dict) -> RoadmapAnalysis:
    """Reconstruct a RoadmapAnalysis from a JSON-parsed dict.

    Uses ``.get()`` with defaults for every field so analyses serialized by an
    older version (missing keys) still deserialize — see AGENTS.md "Frozen
    dataclass backward compatibility". JSON turns tuples into lists, so the
    projects/themes/warnings collections are rebuilt back into tuples.
    """
    projects = tuple(
        RoadmapProject(
            name=p.get("name", ""),
            description=p.get("description", ""),
            size=p.get("size", "small"),
            rationale=p.get("rationale", ""),
            priority=int(p.get("priority", 0) or 0),
            themes=tuple(str(t) for t in p.get("themes", ())),
            quarter=p.get("quarter", ""),
        )
        for p in d.get("projects", ())
        if isinstance(p, dict)
    )
    return RoadmapAnalysis(
        source_type=d.get("source_type", ""),
        source_locator=d.get("source_locator", ""),
        source_label=d.get("source_label", ""),
        summary=d.get("summary", ""),
        projects=projects,
        warnings=tuple(d.get("warnings", ())),
        generated_at=d.get("generated_at", ""),
        annotations=annotations_from(d.get("annotations")),
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class RoadmapStore:
    """SQLite-backed store for the saved roadmap source + analysis history.

    Uses the same database as SessionStore (sessions.db) with dedicated
    ``roadmap_config`` / ``roadmap_history`` tables. Follows the same patterns
    as ReportingStore: autocommit mode, context-manager support, explicit close.

    # See docs: "Session Management" — SQLite persistence
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.isolation_level = None  # autocommit
        self._conn.executescript(_ROADMAP_SCHEMA)
        # Idempotent migration: edit-provenance columns (sessions.py v21/v26).
        # A v21 version-number collision could leave a shared DB stamped past
        # 21 without them, and the CLI and MCP tools open this store without
        # ever constructing a SessionStore, so heal here too.
        for statement in (
            "ALTER TABLE roadmap_history ADD COLUMN origin TEXT NOT NULL DEFAULT 'generated'",
            "ALTER TABLE roadmap_history ADD COLUMN edited_from_id INTEGER NOT NULL DEFAULT 0",
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

    def __enter__(self) -> RoadmapStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Saved roadmaps (the list the TUI manages) ─────────────────────────

    def save_roadmap(
        self, source: RoadmapSource, analysis: RoadmapAnalysis | None, *, roadmap_id: int | None = None
    ) -> int:
        """Insert a new roadmap (roadmap_id=None) or update an existing one; returns its id.

        Each row stores the source plus the LATEST analysis inline — Re-analyze
        updates the row in place rather than appending (the append-only run log
        stays in ``roadmap_history`` via record_run).
        """
        now = self._now()
        analysis_json = _analysis_to_json(analysis) if analysis is not None else ""
        project_count = len(analysis.projects) if analysis is not None else 0
        label = friendly_label(source.label or source.locator)
        if roadmap_id is None:
            cursor = self._conn.execute(
                """INSERT INTO roadmaps
                   (label, source_type, source_locator, source_label, analysis_json, project_count,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (label, source.source_type, source.locator, source.label, analysis_json, project_count, now, now),
            )
            new_id = int(cursor.lastrowid or 0)
            logger.info("Saved new roadmap id=%d: type=%s projects=%d", new_id, source.source_type, project_count)
            return new_id
        self._conn.execute(
            """UPDATE roadmaps
               SET label = ?, source_type = ?, source_locator = ?, source_label = ?,
                   analysis_json = ?, project_count = ?, updated_at = ?
               WHERE id = ?""",
            (label, source.source_type, source.locator, source.label, analysis_json, project_count, now, roadmap_id),
        )
        logger.info("Updated roadmap id=%d: type=%s projects=%d", roadmap_id, source.source_type, project_count)
        return roadmap_id

    def list_roadmaps(self) -> list[dict]:
        """Return saved-roadmap metadata (newest first). Excludes analysis_json — cheap for the list view."""
        rows = self._conn.execute(
            "SELECT id, label, source_type, source_locator, source_label, project_count, created_at, updated_at, "
            "analysis_json != '' FROM roadmaps ORDER BY updated_at DESC, id DESC"
        ).fetchall()
        return [
            {
                "id": r[0],
                "label": friendly_label(r[1]),  # rows saved pre-humanizer display nicely too
                "source_type": r[2],
                "source_locator": r[3],
                "source_label": r[4],
                "project_count": r[5],
                "created_at": r[6],
                "updated_at": r[7],
                "analyzed": bool(r[8]),
            }
            for r in rows
        ]

    def get_roadmap(self, roadmap_id: int) -> dict | None:
        """Return one roadmap row with its deserialized analysis, or None if missing.

        The returned dict adds ``"analysis"`` (RoadmapAnalysis | None — None when
        the roadmap was saved before ever being analyzed, or its JSON is corrupt)
        and ``"source"`` (a RoadmapSource rebuilt from the stored locator).
        """
        r = self._conn.execute(
            "SELECT id, label, source_type, source_locator, source_label, analysis_json, project_count, "
            "created_at, updated_at FROM roadmaps WHERE id = ?",
            (roadmap_id,),
        ).fetchone()
        if r is None:
            return None
        analysis: RoadmapAnalysis | None = None
        if r[5]:
            try:
                analysis = _dict_to_analysis(json.loads(r[5]))
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                logger.warning("Failed to deserialize roadmap %d analysis: %s", roadmap_id, exc)
        return {
            "id": r[0],
            "label": friendly_label(r[1]),
            "source_type": r[2],
            "source_locator": r[3],
            "source_label": r[4],
            "project_count": r[6],
            "created_at": r[7],
            "updated_at": r[8],
            "analysis": analysis,
            "source": RoadmapSource(source_type=r[2], locator=r[3], label=r[4]),
        }

    def delete_roadmap(self, roadmap_id: int) -> None:
        """Delete a saved roadmap; no-op if the id doesn't exist."""
        self._conn.execute("DELETE FROM roadmaps WHERE id = ?", (roadmap_id,))
        logger.info("Deleted roadmap id=%d", roadmap_id)

    # ── Saved source (LEGACY v10 singleton — superseded by `roadmaps`) ────

    def save_config(self, source: RoadmapSource) -> None:
        """Save (or overwrite) the single configured roadmap source.

        Legacy v10 API: superseded by save_roadmap() in v11. Retained so the
        v11 migration can seed from existing data; no production callers.
        """
        self._conn.execute(
            """INSERT OR REPLACE INTO roadmap_config (id, source_type, source_locator, source_label, updated_at)
               VALUES (1, ?, ?, ?, ?)""",
            (source.source_type, source.locator, source.label, self._now()),
        )
        logger.info("Saved roadmap source: type=%s locator=%s", source.source_type, source.locator)

    def load_config(self) -> RoadmapSource | None:
        """Return the saved roadmap source, or None if never configured.

        Legacy v10 API — see save_config.
        """
        row = self._conn.execute(
            "SELECT source_type, source_locator, source_label FROM roadmap_config WHERE id = 1"
        ).fetchone()
        if row is None or not row[0]:
            return None
        return RoadmapSource(source_type=row[0], locator=row[1], label=row[2])

    # ── Run history ───────────────────────────────────────────────────────

    def record_run(self, analysis: RoadmapAnalysis) -> int:
        """Persist an analysis run and return its history row id."""
        cursor = self._conn.execute(
            """INSERT INTO roadmap_history (run_at, source_type, source_locator, project_count, analysis_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                self._now(),
                analysis.source_type,
                analysis.source_locator,
                len(analysis.projects),
                _analysis_to_json(analysis),
            ),
        )
        logger.info(
            "Recorded roadmap analysis: source=%s projects=%d",
            analysis.source_type,
            len(analysis.projects),
        )
        return int(cursor.lastrowid or 0)

    def get_latest_analysis(self) -> RoadmapAnalysis | None:
        """Return the most recent RoadmapAnalysis, or None."""
        row = self._conn.execute("SELECT analysis_json FROM roadmap_history ORDER BY run_at DESC LIMIT 1").fetchone()
        if row is None or not row[0]:
            return None
        try:
            return _dict_to_analysis(json.loads(row[0]))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Failed to deserialize roadmap analysis: %s", exc)
            return None

    def get_history(self, limit: int = 30) -> list[dict]:
        """Return recent analysis-run metadata (newest first)."""
        rows = self._conn.execute(
            "SELECT run_at, source_type, source_locator, project_count FROM roadmap_history "
            "ORDER BY run_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"run_at": r[0], "source_type": r[1], "source_locator": r[2], "project_count": r[3]} for r in rows]
