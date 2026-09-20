"""Unit tests for the Daily Standup SQLite store."""

from dataclasses import replace

import pytest

from yeaboi.agent.state import (
    ConflictCard,
    MemberUpdate,
    OpsSignal,
    StandupGap,
    StandupReport,
    TranscriptClaim,
    TranscriptReview,
    TranscriptSource,
)
from yeaboi.standup.store import StandupStore


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "sessions.db"


def _make_report(**overrides) -> StandupReport:
    base = dict(
        date="2026-07-10",
        session_id="s1",
        sprint_name="Sprint 5",
        sprint_day=3,
        sprint_total_days=10,
        confidence_pct=82,
        confidence_label="At risk",
        member_updates=(MemberUpdate(name="Alice", summary="login"),),
        activity_counts=(("jira", 4),),
    )
    base.update(overrides)
    return StandupReport(**base)


class TestConfig:
    def test_save_and_load(self, db_path):
        with StandupStore(db_path) as store:
            store.save_config(
                "s1",
                enabled=True,
                time="10:00",
                lead_minutes=15,
                weekdays="1-5",
                delivery_channels=["terminal", "slack"],
                repo_path="/tmp/repo",
            )
            cfg = store.load_config("s1")
        assert cfg is not None
        assert cfg["enabled"] is True
        assert cfg["time"] == "10:00"
        assert cfg["lead_minutes"] == 15
        assert cfg["delivery_channels"] == ["terminal", "slack"]
        assert cfg["repo_path"] == "/tmp/repo"

    def test_lead_minutes_defaults_to_10(self, db_path):
        with StandupStore(db_path) as store:
            store.save_config("s1", enabled=True, time="10:00", weekdays="1-5", delivery_channels=["terminal"])
            cfg = store.load_config("s1")
        assert cfg["lead_minutes"] == 10

    def test_load_missing_returns_none(self, db_path):
        with StandupStore(db_path) as store:
            assert store.load_config("nope") is None

    def test_upsert_updates_existing(self, db_path):
        with StandupStore(db_path) as store:
            store.save_config("s1", enabled=True, time="09:50", weekdays="1-5", delivery_channels=["terminal"])
            store.save_config("s1", enabled=False, time="10:00", weekdays="1-5", delivery_channels=["email"])
            cfg = store.load_config("s1")
        assert cfg["enabled"] is False
        assert cfg["time"] == "10:00"
        assert cfg["delivery_channels"] == ["email"]

    def test_corrupt_channels_falls_back(self, db_path):
        with StandupStore(db_path) as store:
            store.save_config("s1", enabled=True, time="09:50", weekdays="1-5", delivery_channels=["terminal"])
            store._conn.execute("UPDATE standup_config SET delivery_channels = 'not json' WHERE session_id='s1'")
            cfg = store.load_config("s1")
        assert cfg["delivery_channels"] == ["terminal"]

    def test_my_aliases_round_trip(self, db_path):
        with StandupStore(db_path) as store:
            store.save_config(
                "s1",
                enabled=True,
                time="10:00",
                weekdays="1-5",
                delivery_channels=["terminal"],
                my_aliases="omardin14, Omar N",
            )
            cfg = store.load_config("s1")
        assert cfg["my_aliases"] == "omardin14, Omar N"

    def test_my_aliases_defaults_empty(self, db_path):
        with StandupStore(db_path) as store:
            store.save_config("s1", enabled=True, time="10:00", weekdays="1-5", delivery_channels=["terminal"])
            cfg = store.load_config("s1")
        assert cfg["my_aliases"] == ""

    def test_team_scope_round_trip(self, db_path):
        with StandupStore(db_path) as store:
            store.save_config(
                "s1",
                enabled=True,
                time="10:00",
                weekdays="1-5",
                delivery_channels=["terminal"],
                tracker_sources=["jira", "azure_devops"],
                team_members=["Alice", "Bob"],
                roster_configured=True,
            )
            cfg = store.load_config("s1")
        assert cfg["tracker_sources"] == ["jira", "azure_devops"]
        assert cfg["team_members"] == ["Alice", "Bob"]
        assert cfg["roster_configured"] is True

    def test_team_scope_defaults_to_unconfigured_jira(self, db_path):
        with StandupStore(db_path) as store:
            store.save_config("s1", enabled=True, time="10:00", weekdays="1-5", delivery_channels=["terminal"])
            cfg = store.load_config("s1")
        assert cfg["tracker_sources"] == ["jira"]
        assert cfg["team_members"] == []
        assert cfg["roster_configured"] is False

    def test_documentation_scope_round_trip(self, db_path):
        with StandupStore(db_path) as store:
            store.save_config(
                "s1",
                enabled=True,
                time="10:00",
                weekdays="1-5",
                delivery_channels=["terminal"],
                documentation_sources=["confluence", "notion"],
                documentation_scope_configured=True,
            )
            cfg = store.load_config("s1")
        assert cfg["documentation_sources"] == ["confluence", "notion"]
        assert cfg["documentation_scope_configured"] is True

    def test_github_excluded_repositories_round_trip(self, db_path):
        with StandupStore(db_path) as store:
            store.save_config(
                "s1",
                enabled=True,
                time="10:00",
                weekdays="1-5",
                delivery_channels=["terminal"],
                code_sources=["github"],
                github_owners=["acme"],
                github_excluded_repositories=["acme/legacy", "acme/archive-mirror"],
                code_scope_configured=True,
            )
            cfg = store.load_config("s1")
        assert cfg["github_excluded_repositories"] == ["acme/legacy", "acme/archive-mirror"]

    def test_github_excluded_repositories_defaults_empty(self, db_path):
        with StandupStore(db_path) as store:
            store.save_config("s1", enabled=True, time="10:00", weekdays="1-5", delivery_channels=["terminal"])
            cfg = store.load_config("s1")
        assert cfg["github_excluded_repositories"] == []

    def test_automation_fields_round_trip(self, db_path):
        with StandupStore(db_path) as store:
            store.save_config(
                "s1",
                enabled=True,
                time="10:00",
                weekdays="1-5",
                delivery_channels=["terminal"],
                automation_markers="wiz, acme-scanner",
                automation_handling="off",
            )
            cfg = store.load_config("s1")
        assert cfg["automation_markers"] == "wiz, acme-scanner"
        assert cfg["automation_handling"] == "off"

    def test_automation_fields_default(self, db_path):
        with StandupStore(db_path) as store:
            store.save_config("s1", enabled=True, time="10:00", weekdays="1-5", delivery_channels=["terminal"])
            cfg = store.load_config("s1")
        assert cfg["automation_markers"] == ""
        assert cfg["automation_handling"] == "exclude"

    def test_my_aliases_column_migrates_old_db(self, db_path):
        """A standup_config table created before my_aliases existed gains the column on open."""
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """CREATE TABLE standup_config (
                   session_id TEXT PRIMARY KEY,
                   enabled INTEGER NOT NULL DEFAULT 0,
                   time TEXT NOT NULL DEFAULT '10:00',
                   timezone TEXT NOT NULL DEFAULT '',
                   weekdays TEXT NOT NULL DEFAULT '1-5',
                   delivery_channels TEXT NOT NULL DEFAULT '["terminal"]',
                   repo_path TEXT NOT NULL DEFAULT '',
                   created_at TEXT NOT NULL,
                   updated_at TEXT NOT NULL
               );
               INSERT INTO standup_config (session_id, enabled, created_at, updated_at)
               VALUES ('s1', 1, 'now', 'now');"""
        )
        conn.close()
        with StandupStore(db_path) as store:
            cfg = store.load_config("s1")
        assert cfg is not None
        assert cfg["my_aliases"] == ""
        assert cfg["tracker_sources"] == ["jira"]
        assert cfg["team_members"] == []
        assert cfg["roster_configured"] is False
        # Automation-filter columns also arrive via migration with safe defaults.
        assert cfg["automation_markers"] == ""
        assert cfg["automation_handling"] == "exclude"


class TestSelfUpdates:
    def test_save_and_get(self, db_path):
        with StandupStore(db_path) as store:
            store.save_my_update("s1", "2026-07-10", "Alice", "shipped the login page")
            updates = store.get_my_updates("s1", "2026-07-10")
        assert updates == {"Alice": "shipped the login page"}

    def test_resubmit_overwrites(self, db_path):
        with StandupStore(db_path) as store:
            store.save_my_update("s1", "2026-07-10", "Alice", "first")
            store.save_my_update("s1", "2026-07-10", "Alice", "second")
            updates = store.get_my_updates("s1", "2026-07-10")
        assert updates == {"Alice": "second"}

    def test_scoped_by_date(self, db_path):
        with StandupStore(db_path) as store:
            store.save_my_update("s1", "2026-07-10", "Alice", "today")
            assert store.get_my_updates("s1", "2026-07-11") == {}

    def test_images_round_trip(self, db_path, tmp_path):
        img = tmp_path / "burndown.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        with StandupStore(db_path) as store:
            store.save_my_update("s1", "2026-07-10", "Alice", "see chart [image #1]", images=[str(img)])
            assert store.get_my_update_images("s1", "2026-07-10") == {"Alice": [str(img)]}

    def test_missing_image_files_pruned(self, db_path, tmp_path):
        with StandupStore(db_path) as store:
            store.save_my_update("s1", "2026-07-10", "Alice", "x", images=[str(tmp_path / "gone.png")])
            assert store.get_my_update_images("s1", "2026-07-10") == {}

    def test_update_without_images_has_none(self, db_path):
        with StandupStore(db_path) as store:
            store.save_my_update("s1", "2026-07-10", "Alice", "no pics")
            assert store.get_my_update_images("s1", "2026-07-10") == {}


class TestRunHistory:
    def test_record_and_get_latest(self, db_path):
        report = _make_report()
        with StandupStore(db_path) as store:
            row_id = store.record_run(report, delivery_status={"terminal": True}, status="success")
            latest = store.get_latest_report("s1")
        assert row_id > 0
        assert latest == report

    def test_get_latest_missing_returns_none(self, db_path):
        with StandupStore(db_path) as store:
            assert store.get_latest_report("s1") is None

    def test_latest_is_most_recent(self, db_path):
        with StandupStore(db_path) as store:
            store.record_run(_make_report(date="2026-07-09", confidence_pct=50))
            store.record_run(_make_report(date="2026-07-10", confidence_pct=90))
            latest = store.get_latest_report("s1")
        assert latest.date == "2026-07-10"
        assert latest.confidence_pct == 90

    def test_report_images_round_trip(self, db_path):
        # New tuple field must survive JSON serialization (list → tuple rebuild).
        report = _make_report(images=("/tmp/a.png", "/tmp/b.png"))
        with StandupStore(db_path) as store:
            store.record_run(report)
            latest = store.get_latest_report("s1")
        assert latest.images == ("/tmp/a.png", "/tmp/b.png")

    def test_old_report_without_images_deserializes(self, db_path):
        # Reports recorded before the images field existed must still load.
        report = _make_report()
        with StandupStore(db_path) as store:
            store.record_run(report)
            latest = store.get_latest_report("s1")
        assert latest.images == ()

    def test_get_history(self, db_path):
        with StandupStore(db_path) as store:
            store.record_run(_make_report(date="2026-07-09"))
            store.record_run(_make_report(date="2026-07-10"))
            history = store.get_history("s1")
        assert len(history) == 2
        assert history[0]["standup_date"] == "2026-07-10"  # newest first
        assert history[0]["confidence_pct"] == 82
        assert "id" in history[0]  # saved-runs hub needs the row id

    def test_corrupt_report_json_returns_none(self, db_path):
        with StandupStore(db_path) as store:
            store.record_run(_make_report())
            store._conn.execute("UPDATE standup_history SET report_json = 'garbage'")
            assert store.get_latest_report("s1") is None


class TestProvenanceSelfHeal:
    """A standup_history table missing the v21 provenance columns heals on open.

    The v21 schema-version collision could leave a shared DB stamped past 21
    without `origin`/`edited_from_id`, and several entry points (--standup-run,
    the MCP tools) open this store without ever constructing a SessionStore —
    whose v26 migration is the other repair path.
    """

    def test_pre_v21_history_table_heals_on_open(self, db_path):
        import sqlite3

        from yeaboi.standup.store import _standup_report_to_json

        legacy = _make_report(date="2026-07-09")
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """CREATE TABLE standup_history (
                   id              INTEGER PRIMARY KEY AUTOINCREMENT,
                   session_id      TEXT NOT NULL,
                   run_at          TEXT NOT NULL,
                   standup_date    TEXT NOT NULL DEFAULT '',
                   sprint_day      INTEGER NOT NULL DEFAULT 0,
                   confidence_pct  INTEGER NOT NULL DEFAULT 0,
                   report_json     TEXT NOT NULL DEFAULT '',
                   delivery_status TEXT NOT NULL DEFAULT '{}',
                   status          TEXT NOT NULL DEFAULT 'success',
                   error           TEXT NOT NULL DEFAULT ''
               );"""
        )
        conn.execute(
            "INSERT INTO standup_history (session_id, run_at, standup_date, report_json) VALUES (?, ?, ?, ?)",
            ("s1", "2026-07-09T10:00:00", "2026-07-09", _standup_report_to_json(legacy)),
        )
        conn.commit()
        conn.close()

        with StandupStore(db_path) as store:
            # The exact origin-reading queries Generate runs, in run order.
            previous = store.get_previous_run("s1", "2026-07-10")
            assert previous is not None
            row_id, origin, edited_from_id, report = previous
            assert origin == "generated"
            assert edited_from_id == 0
            assert report.date == "2026-07-09"
            assert store.record_run(_make_report()) > row_id
            base = store.get_base_run(session_id="s1")
            assert base is not None


class TestSavedRunsHub:
    """get_all_history / get_run_by_id / delete_run — power the TUI saved-runs hub."""

    def test_get_all_history_carries_id_and_session(self, db_path):
        with StandupStore(db_path) as store:
            store.record_run(_make_report(date="2026-07-09"))
            rows = store.get_all_history()
        assert rows and "id" in rows[0] and rows[0]["session_id"] == "s1"

    def test_get_run_by_id_round_trips_and_missing(self, db_path):
        report = _make_report(date="2026-07-10")
        with StandupStore(db_path) as store:
            rid = store.record_run(report)
            assert store.get_run_by_id(rid) == report
            assert store.get_run_by_id(999) is None

    def test_get_run_by_id_corrupt_returns_none(self, db_path):
        with StandupStore(db_path) as store:
            rid = store.record_run(_make_report())
            store._conn.execute("UPDATE standup_history SET report_json='{bad' WHERE id=?", (rid,))
            assert store.get_run_by_id(rid) is None

    def test_delete_run_removes_only_that_row(self, db_path):
        with StandupStore(db_path) as store:
            keep = store.record_run(_make_report(date="2026-07-09"))
            drop = store.record_run(_make_report(date="2026-07-10"))
            assert store.delete_run(drop) is True
            assert store.delete_run(drop) is False
            assert {r["id"] for r in store.get_all_history()} == {keep}

    def test_self_report_round_trips(self, db_path):
        report = _make_report(
            member_updates=(
                MemberUpdate(name="Me", summary="Merged auth PR", source="combined", self_report="paired\nall day"),
            )
        )
        with StandupStore(db_path) as store:
            store.record_run(report)
            latest = store.get_latest_report("s1")
        assert latest.member_updates[0].self_report == "paired\nall day"
        assert latest.member_updates[0].source == "combined"

    def test_old_report_json_without_self_report_deserializes(self, db_path):
        """Reports persisted before the self_report field existed still load."""
        import json

        with StandupStore(db_path) as store:
            store.record_run(_make_report())
            # Strip self_report from the stored JSON to simulate an old row.
            (raw,) = store._conn.execute("SELECT report_json FROM standup_history").fetchone()
            d = json.loads(raw)
            for m in d["member_updates"]:
                m.pop("self_report", None)
            store._conn.execute("UPDATE standup_history SET report_json = ?", (json.dumps(d),))
            latest = store.get_latest_report("s1")
        assert latest is not None
        assert latest.member_updates[0].self_report == ""

    def test_evidence_round_trips_as_dataclasses(self, db_path):
        from yeaboi.agent.state import ActivityEvidence

        report = _make_report(
            member_updates=(
                MemberUpdate(
                    name="Me",
                    summary="x",
                    code_evidence=(
                        ActivityEvidence(
                            kind="commit",
                            key="78e4201d",
                            title="Fix login redirect",
                            url="https://g/c1",
                            repository="yeaboi/web",
                            timestamp="2026-07-30T09:15:00",
                        ),
                    ),
                ),
            )
        )
        with StandupStore(db_path) as store:
            store.record_run(report)
            latest = store.get_latest_report("s1")
        (row,) = latest.member_updates[0].code_evidence
        assert isinstance(row, ActivityEvidence)
        assert (row.key, row.title, row.repository) == ("78e4201d", "Fix login redirect", "yeaboi/web")
        assert latest.member_updates[0].ticketing_evidence == ()

    def test_pr_children_round_trip_nested(self, db_path):
        from yeaboi.agent.state import ActivityEvidence

        report = _make_report(
            member_updates=(
                MemberUpdate(
                    name="Me",
                    summary="x",
                    code_evidence=(
                        ActivityEvidence(
                            kind="pr",
                            key="!91",
                            title="Enable SSO",
                            url="https://a/pr/91",
                            status="merged",
                            children=(ActivityEvidence(kind="commit", key="aaa1", title="Fix", url="https://a/c1"),),
                        ),
                    ),
                ),
            )
        )
        with StandupStore(db_path) as store:
            store.record_run(report)
            latest = store.get_latest_report("s1")
        (pr,) = latest.member_updates[0].code_evidence
        (child,) = pr.children
        assert isinstance(child, ActivityEvidence)
        assert (child.kind, child.key, child.children) == ("commit", "aaa1", ())

    def test_hierarchy_fields_round_trip(self, db_path):
        from yeaboi.agent.state import ActivityEvidence

        report = _make_report(
            member_updates=(
                MemberUpdate(
                    name="Me",
                    summary="x",
                    ticketing_evidence=(
                        ActivityEvidence(
                            kind="issue",
                            key="PSOT-3",
                            title="SSO error states",
                            url="https://j/browse/PSOT-3",
                            issue_type="Sub-task",
                            parent_key="PSOT-1",
                            subtask=True,
                        ),
                    ),
                    code_evidence=(
                        ActivityEvidence(
                            kind="pr",
                            key="#91",
                            title="PSOT-1 enable SSO",
                            url="https://g/pr/91",
                            ticket_keys=("PSOT-1", "#77"),
                        ),
                    ),
                ),
            )
        )
        with StandupStore(db_path) as store:
            store.record_run(report)
            latest = store.get_latest_report("s1")
        (ticket,) = latest.member_updates[0].ticketing_evidence
        assert (ticket.issue_type, ticket.parent_key, ticket.subtask) == ("Sub-task", "PSOT-1", True)
        (pr,) = latest.member_updates[0].code_evidence
        assert pr.ticket_keys == ("PSOT-1", "#77")

    def test_pre_hierarchy_evidence_dict_defaults(self, db_path):
        """Evidence stored before the hierarchy fields existed loads with defaults."""
        from yeaboi.standup.store import _dict_to_evidence

        (row,) = _dict_to_evidence([{"kind": "issue", "key": "PSOT-1", "title": "Old row"}])
        assert (row.issue_type, row.parent_key, row.subtask, row.ticket_keys) == ("", "", False, ())

    def test_old_report_json_without_evidence_deserializes(self, db_path):
        """Reports persisted before the *_evidence fields existed still load."""
        import json

        with StandupStore(db_path) as store:
            store.record_run(_make_report())
            (raw,) = store._conn.execute("SELECT report_json FROM standup_history").fetchone()
            d = json.loads(raw)
            for m in d["member_updates"]:
                for field in ("ticketing_evidence", "code_evidence", "documentation_evidence"):
                    m.pop(field, None)
            store._conn.execute("UPDATE standup_history SET report_json = ?", (json.dumps(d),))
            latest = store.get_latest_report("s1")
        assert latest is not None
        assert latest.member_updates[0].code_evidence == ()

    def test_activity_window_round_trips(self, db_path):
        report = _make_report(activity_window="Fri 2026-07-17 00:00 → now")
        with StandupStore(db_path) as store:
            store.record_run(report)
            latest = store.get_latest_report("s1")
        assert latest.activity_window == "Fri 2026-07-17 00:00 → now"

    def test_activity_window_bounds_round_trip(self, db_path):
        """The machine-readable pair the web timeline draws its axis from.

        ``activity_window`` beside it is the human string; only these two are
        parseable, and this is the path that feeds "open instantly from a saved
        report", so a report read back without them draws an axis derived from
        event times alone and silently loses the window's quiet edges.
        """
        report = _make_report(
            activity_window_start="2026-07-17T00:00:00+01:00",
            activity_window_end="2026-07-17T18:30:00+01:00",
        )
        with StandupStore(db_path) as store:
            store.record_run(report)
            latest = store.get_latest_report("s1")
        assert latest.activity_window_start == "2026-07-17T00:00:00+01:00"
        assert latest.activity_window_end == "2026-07-17T18:30:00+01:00"

    def test_a_report_stored_before_the_bounds_existed_still_reads(self, db_path):
        """Rows written by an older build carry neither key — they must default."""
        import json

        with StandupStore(db_path) as store:
            store.record_run(_make_report())
            (raw,) = store._conn.execute("SELECT report_json FROM standup_history").fetchone()
            d = json.loads(raw)
            d.pop("activity_window_start", None)
            d.pop("activity_window_end", None)
            store._conn.execute("UPDATE standup_history SET report_json = ?", (json.dumps(d),))
            latest = store.get_latest_report("s1")
        assert latest is not None
        assert latest.activity_window_start == ""
        assert latest.activity_window_end == ""

    def test_my_name_round_trips(self, db_path):
        report = _make_report(my_name="Omar Din")
        with StandupStore(db_path) as store:
            store.record_run(report)
            latest = store.get_latest_report("s1")
        assert latest.my_name == "Omar Din"


class TestMigrationCreatesTables:
    def test_session_store_v6_creates_standup_tables(self, db_path):
        """Opening a SessionStore should run the v6 migration and create standup tables."""
        from yeaboi.sessions import CURRENT_SCHEMA_VERSION, SessionStore

        assert CURRENT_SCHEMA_VERSION >= 6
        with SessionStore(db_path):
            pass
        # A fresh StandupStore on the same DB should find existing tables and work.
        with StandupStore(db_path) as store:
            store.save_config("s1", enabled=True, time="09:50", weekdays="1-5", delivery_channels=["terminal"])
            assert store.load_config("s1") is not None


class TestSkippedSourcesRoundTrip:
    def test_round_trips(self, db_path):
        report = _make_report(skipped_sources=(("github", "STANDUP_GITHUB_REPO not set"),))
        with StandupStore(db_path) as store:
            store.record_run(report)
            latest = store.get_latest_report("s1")
        assert latest.skipped_sources == (("github", "STANDUP_GITHUB_REPO not set"),)

    def test_old_report_without_field_deserializes(self, db_path):
        import json

        with StandupStore(db_path) as store:
            store.record_run(_make_report())
            (raw,) = store._conn.execute("SELECT report_json FROM standup_history").fetchone()
            d = json.loads(raw)
            d.pop("skipped_sources", None)
            store._conn.execute("UPDATE standup_history SET report_json = ?", (json.dumps(d),))
            latest = store.get_latest_report("s1")
        assert latest is not None
        assert latest.skipped_sources == ()

    def test_unmet_sources_round_trip(self, db_path):
        # The broadcast renderers read this long after the run that classified it,
        # so "asked for and not delivered" has to survive the store.
        report = _make_report(
            skipped_sources=(("github", "GITHUB_TOKEN not set"), ("local_git", "no repo path configured")),
            unmet_sources=("github",),
        )
        with StandupStore(db_path) as store:
            store.record_run(report)
            latest = store.get_latest_report("s1")
        assert latest.unmet_sources == ("github",)

    def test_old_report_without_unmet_sources_deserializes(self, db_path):
        # A run stored before the split has skips but no verdict on them; the
        # broadcast line must stay off rather than replay an old nag.
        import json

        with StandupStore(db_path) as store:
            store.record_run(_make_report(skipped_sources=(("github", "GITHUB_TOKEN not set"),)))
            (raw,) = store._conn.execute("SELECT report_json FROM standup_history").fetchone()
            d = json.loads(raw)
            d.pop("unmet_sources", None)
            store._conn.execute("UPDATE standup_history SET report_json = ?", (json.dumps(d),))
            latest = store.get_latest_report("s1")
        assert latest is not None
        assert latest.unmet_sources == ()


class TestMemberLinksRoundTrip:
    def test_round_trips(self, db_path):
        member = MemberUpdate(name="Alice", summary="login", links=(("PSOT-1", "https://j/browse/PSOT-1"),))
        with StandupStore(db_path) as store:
            store.record_run(_make_report(member_updates=(member,)))
            latest = store.get_latest_report("s1")
        assert latest.member_updates[0].links == (("PSOT-1", "https://j/browse/PSOT-1"),)

    def test_old_member_without_links_deserializes(self, db_path):
        import json

        with StandupStore(db_path) as store:
            store.record_run(_make_report())
            (raw,) = store._conn.execute("SELECT report_json FROM standup_history").fetchone()
            d = json.loads(raw)
            for m in d["member_updates"]:
                m.pop("links", None)
            store._conn.execute("UPDATE standup_history SET report_json = ?", (json.dumps(d),))
            latest = store.get_latest_report("s1")
        assert latest.member_updates[0].links == ()

    def test_structured_summaries_and_coverage_round_trip(self, db_path):
        member = MemberUpdate(
            name="Alice",
            summary="Delivered authentication and its runbook.",
            ticketing_summary="Moved PSOT-1 to Done.",
            ticketing_links=(("PSOT-1", "https://j/browse/PSOT-1"),),
            code_summary="Merged authentication.",
            code_links=(("#12", "https://github/pull/12"),),
            documentation_summary="Updated the authentication runbook.",
            documentation_links=(("Runbook", "https://wiki/runbook"),),
        )
        report = _make_report(
            member_updates=(member,),
            category_coverage=(
                ("ticketing", "covered"),
                ("code", "covered"),
                ("documentation", "partial"),
            ),
        )
        with StandupStore(db_path) as store:
            store.record_run(report)
            latest = store.get_latest_report("s1")
        assert latest.member_updates[0] == member
        assert latest.category_coverage == report.category_coverage


class TestActivityCountRoundTrip:
    def test_round_trips(self, db_path):
        member = MemberUpdate(name="Alice", summary="login", activity_count=3)
        with StandupStore(db_path) as store:
            store.record_run(_make_report(member_updates=(member,)))
            latest = store.get_latest_report("s1")
        assert latest.member_updates[0].activity_count == 3

    def test_old_member_without_count_deserializes(self, db_path):
        import json

        with StandupStore(db_path) as store:
            store.record_run(_make_report())
            (raw,) = store._conn.execute("SELECT report_json FROM standup_history").fetchone()
            d = json.loads(raw)
            for m in d["member_updates"]:
                m.pop("activity_count", None)
            store._conn.execute("UPDATE standup_history SET report_json = ?", (json.dumps(d),))
            latest = store.get_latest_report("s1")
        assert latest.member_updates[0].activity_count == 0


class TestEnabledScheduleSessions:
    @staticmethod
    def _save(store, session_id, enabled, time):
        store.save_config(session_id, enabled=enabled, time=time, weekdays="1-5", delivery_channels=["terminal"])

    def test_lists_enabled_sessions_most_recent_first(self, db_path):
        with StandupStore(db_path) as store:
            self._save(store, "old", True, "09:00")
            self._save(store, "off", False, "10:00")
            self._save(store, "new", True, "11:00")
            # Touch "old" again so it becomes the most recently updated.
            self._save(store, "old", True, "09:15")
            assert store.get_enabled_schedule_sessions() == ["old", "new"]

    def test_empty_when_no_enabled_config(self, db_path):
        with StandupStore(db_path) as store:
            self._save(store, "s1", False, "10:00")
            assert store.get_enabled_schedule_sessions() == []


class TestDayOverDayRoundTrip:
    def test_new_fields_round_trip(self, db_path):
        report = _make_report(
            confidence_delta=-8,
            confidence_trend="declining",
            member_updates=(
                MemberUpdate(
                    name="Alice",
                    summary="login",
                    progress_note="Still on PSOT-9 from yesterday.",
                    outlook="Likely to finish PSOT-9.",
                ),
            ),
        )
        with StandupStore(db_path) as store:
            store.record_run(report)
            latest = store.get_latest_report("s1")
        assert latest == report
        assert latest.confidence_delta == -8
        assert latest.confidence_trend == "declining"
        assert latest.member_updates[0].progress_note == "Still on PSOT-9 from yesterday."
        assert latest.member_updates[0].outlook == "Likely to finish PSOT-9."

    def test_old_report_json_defaults(self, db_path):
        # Reports recorded before the day-over-day fields existed must still load.
        report = _make_report()
        with StandupStore(db_path) as store:
            store.record_run(report)
            store._conn.execute(
                "UPDATE standup_history SET report_json = "
                '\'{"date": "2026-07-10", "member_updates": [{"name": "Alice"}]}\''
            )
            latest = store.get_latest_report("s1")
        assert latest.confidence_delta == 0
        assert latest.confidence_trend == ""
        assert latest.member_updates[0].progress_note == ""
        assert latest.member_updates[0].outlook == ""


class TestGetPreviousReport:
    def test_newest_before_date_wins(self, db_path):
        with StandupStore(db_path) as store:
            store.record_run(_make_report(date="2026-07-08", confidence_pct=70))
            store.record_run(_make_report(date="2026-07-09", confidence_pct=80))
            store.record_run(_make_report(date="2026-07-10", confidence_pct=90))
            prev = store.get_previous_report("s1", "2026-07-10")
        assert prev is not None
        assert prev.date == "2026-07-09"

    def test_same_day_rerun_excluded(self, db_path):
        # A rerun earlier TODAY is not "yesterday".
        with StandupStore(db_path) as store:
            store.record_run(_make_report(date="2026-07-10", confidence_pct=50))
            assert store.get_previous_report("s1", "2026-07-10") is None

    def test_failed_runs_excluded(self, db_path):
        with StandupStore(db_path) as store:
            store.record_run(_make_report(date="2026-07-09"), status="failed")
            assert store.get_previous_report("s1", "2026-07-10") is None

    def test_partial_runs_included(self, db_path):
        with StandupStore(db_path) as store:
            store.record_run(_make_report(date="2026-07-09"), status="partial")
            prev = store.get_previous_report("s1", "2026-07-10")
        assert prev is not None

    def test_corrupt_json_returns_none(self, db_path):
        with StandupStore(db_path) as store:
            store.record_run(_make_report(date="2026-07-09"))
            store._conn.execute("UPDATE standup_history SET report_json = 'garbage'")
            assert store.get_previous_report("s1", "2026-07-10") is None

    def test_other_session_ignored(self, db_path):
        with StandupStore(db_path) as store:
            store.record_run(_make_report(date="2026-07-09", session_id="other"))
            assert store.get_previous_report("s1", "2026-07-10") is None


# ---------------------------------------------------------------------------
# Transcript review — reviews, transcript bookkeeping, and the gap→issue ledger
# ---------------------------------------------------------------------------


def _make_review(**overrides) -> TranscriptReview:
    base = dict(
        session_id="s1",
        standup_date="2026-07-10",
        run_id=0,
        reviewed_at="2026-07-10T11:00:00+00:00",
        sources=(
            TranscriptSource(
                path="/tmp/t.vtt",
                filename="t.vtt",
                fmt="vtt",
                covered_date="2026-07-10",
                char_count=120,
                speakers=("Alice",),
            ),
        ),
        claims=(
            TranscriptClaim(
                member="Alice",
                claim="also commented on the design doc",
                quote="I also commented on the design doc",
                status="missing",
                system_hint="confluence",
                artifact_hint="comment on a page",
            ),
        ),
        gaps=(
            StandupGap(
                fingerprint="abc123",
                category="capability_gap_in_supported_source",
                scope="product",
                title="Standup misses Confluence page comments",
                members=("Alice",),
                affected_systems=("confluence",),
                next_steps=("Fetch page comments in collector.py",),
            ),
        ),
        config_suggestions=(
            StandupGap(
                fingerprint="def456",
                category="scope_gap_repository",
                scope="config",
                title="acme/infra is not in your code scope",
                remedy="Add acme/infra via Standup -> Configure -> Code",
            ),
        ),
        claims_matched=3,
        claims_missing=1,
        llm_mode="llm",
        warnings=("one warning",),
    )
    base.update(overrides)
    return TranscriptReview(**base)


class TestTranscriptReviews:
    def test_round_trips(self, db_path):
        review = _make_review()
        with StandupStore(db_path) as store:
            review_id = store.record_review(review)
            loaded = store.get_review(review_id)
        assert loaded is not None
        # review_id is assigned on insert, so compare the rest field-for-field.
        assert replace(loaded, review_id=0) == review
        assert loaded.review_id == review_id

    def test_nested_gaps_and_claims_rebuild_as_dataclasses(self, db_path):
        with StandupStore(db_path) as store:
            review_id = store.record_review(_make_review())
            loaded = store.get_review(review_id)
        assert isinstance(loaded.gaps[0], StandupGap)
        assert isinstance(loaded.claims[0], TranscriptClaim)
        assert isinstance(loaded.sources[0], TranscriptSource)
        assert loaded.gaps[0].affected_systems == ("confluence",)
        assert loaded.config_suggestions[0].scope == "config"

    def test_old_review_json_without_new_keys_deserializes(self, db_path):
        import json

        with StandupStore(db_path) as store:
            review_id = store.record_review(_make_review())
            (raw,) = store._conn.execute("SELECT review_json FROM standup_reviews").fetchone()
            stripped = json.loads(raw)
            for key in ("gaps", "config_suggestions", "sources", "claims", "llm_mode", "untracked_count"):
                stripped.pop(key, None)
            store._conn.execute("UPDATE standup_reviews SET review_json = ?", (json.dumps(stripped),))
            loaded = store.get_review(review_id)
        assert loaded is not None
        assert loaded.gaps == ()
        assert loaded.config_suggestions == ()
        assert loaded.sources == ()
        assert loaded.llm_mode == ""

    def test_corrupt_json_returns_none(self, db_path):
        with StandupStore(db_path) as store:
            review_id = store.record_review(_make_review())
            store._conn.execute("UPDATE standup_reviews SET review_json = 'garbage'")
            assert store.get_review(review_id) is None

    def test_get_review_missing_returns_none(self, db_path):
        with StandupStore(db_path) as store:
            assert store.get_review(999) is None

    def test_latest_and_list(self, db_path):
        with StandupStore(db_path) as store:
            store.record_review(_make_review(standup_date="2026-07-09", reviewed_at="2026-07-09T11:00:00+00:00"))
            store.record_review(_make_review(standup_date="2026-07-10", reviewed_at="2026-07-10T11:00:00+00:00"))
            latest = store.get_latest_review("s1")
            rows = store.get_reviews("s1")
        assert latest.standup_date == "2026-07-10"
        assert [r["standup_date"] for r in rows] == ["2026-07-10", "2026-07-09"]
        assert rows[0]["status"] == "drafted"

    def test_status_update(self, db_path):
        with StandupStore(db_path) as store:
            review_id = store.record_review(_make_review())
            store.set_review_status(review_id, "filed")
            assert store.get_reviews("s1")[0]["status"] == "filed"

    def test_other_session_ignored(self, db_path):
        with StandupStore(db_path) as store:
            store.record_review(_make_review(session_id="other"))
            assert store.get_latest_review("s1") is None


class TestRunRowByDate:
    def test_returns_newest_run_on_the_date(self, db_path):
        with StandupStore(db_path) as store:
            store.record_run(_make_report(date="2026-07-10"))
            second = store.record_run(_make_report(date="2026-07-10"))
            assert store.get_run_row_by_date("s1", "2026-07-10") == second

    def test_failed_run_ignored(self, db_path):
        with StandupStore(db_path) as store:
            store.record_run(_make_report(date="2026-07-10"), status="failed")
            assert store.get_run_row_by_date("s1", "2026-07-10") == 0

    def test_no_run_returns_zero(self, db_path):
        with StandupStore(db_path) as store:
            assert store.get_run_row_by_date("s1", "2026-07-10") == 0


class TestTranscriptBookkeeping:
    def test_marks_and_lists_hashes(self, db_path):
        with StandupStore(db_path) as store:
            store.mark_transcript_reviewed(
                "s1", path="/tmp/a.vtt", content_hash="h1", covered_date="2026-07-10", review_id=1
            )
            assert store.reviewed_transcript_hashes("s1") == {"h1"}

    def test_same_content_at_a_new_path_is_not_re_reviewed(self, db_path):
        # Renaming a transcript must not re-spend an LLM call: the key is content.
        with StandupStore(db_path) as store:
            store.mark_transcript_reviewed(
                "s1", path="/tmp/a.vtt", content_hash="h1", covered_date="2026-07-10", review_id=1
            )
            store.mark_transcript_reviewed(
                "s1", path="/tmp/renamed.vtt", content_hash="h1", covered_date="2026-07-10", review_id=2
            )
            rows = store._conn.execute("SELECT path, review_id FROM standup_transcripts").fetchall()
        assert rows == [("/tmp/renamed.vtt", 2)]

    def test_hashes_are_session_scoped(self, db_path):
        with StandupStore(db_path) as store:
            store.mark_transcript_reviewed(
                "s1", path="/tmp/a.vtt", content_hash="h1", covered_date="2026-07-10", review_id=1
            )
            assert store.reviewed_transcript_hashes("other") == set()


class TestReviewedAndRunDates:
    """The two set-difference halves behind the "unchecked standup" signal."""

    def _mark(self, store, day, *, session="s1", i=1):
        store.mark_transcript_reviewed(
            session, path=f"/t/{day}.vtt", content_hash=f"h-{day}", covered_date=day, review_id=i
        )

    def test_reviewed_dates_are_distinct(self, db_path):
        with StandupStore(db_path) as store:
            self._mark(store, "2026-07-10")
            store.mark_transcript_reviewed(
                "s1", path="/t/b.vtt", content_hash="h2", covered_date="2026-07-10", review_id=2
            )
            assert store.reviewed_dates("s1") == {"2026-07-10"}

    def test_reviewed_dates_honour_since(self, db_path):
        with StandupStore(db_path) as store:
            self._mark(store, "2026-01-05")
            self._mark(store, "2026-07-10")
            assert store.reviewed_dates("s1", since="2026-07-01") == {"2026-07-10"}

    def test_reviewed_dates_are_session_scoped(self, db_path):
        with StandupStore(db_path) as store:
            self._mark(store, "2026-07-10")
            assert store.reviewed_dates("other") == set()

    def test_reviewed_dates_skip_blank_dates(self, db_path):
        with StandupStore(db_path) as store:
            store.mark_transcript_reviewed("s1", path="/t/x", content_hash="hx", covered_date="", review_id=1)
            assert store.reviewed_dates("s1") == set()

    def test_run_dates_are_distinct_per_day(self, db_path):
        with StandupStore(db_path) as store:
            store.record_run(_make_report(date="2026-07-10"))
            store.record_run(_make_report(date="2026-07-10"))
            store.record_run(_make_report(date="2026-07-11"))
            assert store.run_dates("s1") == {"2026-07-10", "2026-07-11"}

    def test_run_dates_count_partial_but_not_failed(self, db_path):
        """Same scoping as get_run_row_by_date: a failed run produced no report,
        so it is not something to be reproached for not transcribing."""
        with StandupStore(db_path) as store:
            store.record_run(_make_report(date="2026-07-10"), status="partial")
            store.record_run(_make_report(date="2026-07-11"), status="failed")
            store.record_run(_make_report(date="2026-07-12"), status="error")
            assert store.run_dates("s1") == {"2026-07-10"}

    def test_run_dates_window_is_half_open(self, db_path):
        with StandupStore(db_path) as store:
            for day in ("2026-07-09", "2026-07-10", "2026-07-11"):
                store.record_run(_make_report(date=day))
            got = store.run_dates("s1", since="2026-07-10", before="2026-07-11")
            assert got == {"2026-07-10"}

    def test_run_dates_are_session_scoped(self, db_path):
        with StandupStore(db_path) as store:
            store.record_run(_make_report(date="2026-07-10"))
            assert store.run_dates("other") == set()


class TestGapIssueLedger:
    def test_insert_then_read(self, db_path):
        with StandupStore(db_path) as store:
            store.upsert_gap_issue("fp1", category="integration_missing", title="Slack", review_id=3)
            entry = store.get_gap_issue("fp1")
        assert entry["category"] == "integration_missing"
        assert entry["state"] == "drafted"
        assert entry["occurrences"] == 1
        assert entry["last_review_id"] == 3

    def test_recurrence_bumps_occurrences(self, db_path):
        with StandupStore(db_path) as store:
            store.upsert_gap_issue("fp1", category="c", title="t")
            store.upsert_gap_issue("fp1", category="c", title="t")
            store.upsert_gap_issue("fp1", category="c", title="t")
            assert store.get_gap_issue("fp1")["occurrences"] == 3

    def test_recurrence_preserves_filed_state(self, db_path):
        """A later 'seen again' must never erase the issue number — that is how
        dedup would silently start filing duplicates onto a public repo."""
        with StandupStore(db_path) as store:
            store.upsert_gap_issue(
                "fp1",
                category="c",
                title="t",
                issue_number=42,
                issue_url="https://example/42",
                state="filed",
                via="api",
                filed_at="2026-07-10T00:00:00+00:00",
            )
            store.upsert_gap_issue("fp1", category="c", title="t")
            entry = store.get_gap_issue("fp1")
        assert entry["issue_number"] == 42
        assert entry["issue_url"] == "https://example/42"
        assert entry["state"] == "filed"
        assert entry["filed_at"] == "2026-07-10T00:00:00+00:00"
        assert entry["occurrences"] == 2

    def test_bump_occurrence_can_be_suppressed(self, db_path):
        with StandupStore(db_path) as store:
            store.upsert_gap_issue("fp1", category="c", title="t")
            store.upsert_gap_issue("fp1", category="c", title="t", state="filed", bump_occurrence=False)
            assert store.get_gap_issue("fp1")["occurrences"] == 1

    def test_missing_returns_none(self, db_path):
        with StandupStore(db_path) as store:
            assert store.get_gap_issue("nope") is None

    def test_ledger_is_not_session_scoped(self, db_path):
        """The loop improves yeaboi itself, so the same gap in two projects is one issue."""
        with StandupStore(db_path) as store:
            store.upsert_gap_issue("fp1", category="c", title="t")
            store.upsert_gap_issue("fp2", category="c2", title="t2")
            assert {e["fingerprint"] for e in store.get_gap_issues()} == {"fp1", "fp2"}


class TestTranscriptConfig:
    def test_round_trips(self, db_path):
        with StandupStore(db_path) as store:
            store.save_config(
                "s1",
                enabled=True,
                time="10:00",
                weekdays="1-5",
                delivery_channels=["terminal"],
                transcript_dir="/tmp/meetings",
                transcript_review_enabled=False,
            )
            cfg = store.load_config("s1")
        assert cfg["transcript_dir"] == "/tmp/meetings"
        assert cfg["transcript_review_enabled"] is False

    def test_defaults_to_enabled(self, db_path):
        with StandupStore(db_path) as store:
            store.save_config("s1", enabled=True, time="10:00", weekdays="1-5", delivery_channels=["terminal"])
            cfg = store.load_config("s1")
        assert cfg["transcript_dir"] == ""
        assert cfg["transcript_review_enabled"] is True


class TestPracticeConfig:
    def _save(self, store, **over):
        kwargs = dict(enabled=True, time="10:00", weekdays="1-5", delivery_channels=["terminal"])
        kwargs.update(over)
        store.save_config("s1", **kwargs)

    def test_defaults_are_on_and_all_rules(self, db_path):
        with StandupStore(db_path) as store:
            self._save(store)
            config = store.load_config("s1")
        assert config["habit_detection"] == "on"
        assert config["habit_rules"] == ""
        assert config["habit_ai_match"] == "on"

    def test_round_trips_both_columns(self, db_path):
        with StandupStore(db_path) as store:
            self._save(store, habit_detection="off", habit_rules="wip-sprawl,large-change", habit_ai_match="off")
            config = store.load_config("s1")
        assert config["habit_detection"] == "off"
        assert config["habit_rules"] == "wip-sprawl,large-change"
        assert config["habit_ai_match"] == "off"

    def test_a_row_written_before_the_columns_existed_reads_as_on(self, db_path):
        # The migration adds the columns with defaults; a legacy row must not
        # come back with practices mysteriously disabled.
        with StandupStore(db_path) as store:
            self._save(store)
            store._conn.execute("UPDATE standup_config SET habit_detection = '', habit_rules = '', habit_ai_match = ''")
            config = store.load_config("s1")
        assert config["habit_detection"] == "on"
        assert config["habit_ai_match"] == "on"

    def test_migration_is_idempotent(self, db_path):
        with StandupStore(db_path) as store:
            self._save(store, habit_detection="off")
        # Re-opening runs the same ALTERs again; they must not raise or reset.
        with StandupStore(db_path) as store:
            assert store.load_config("s1")["habit_detection"] == "off"


class TestPracticeReportRoundTrip:
    def test_practices_and_rollup_survive_a_round_trip(self, db_path):
        from yeaboi.agent.state import PracticeSignal

        report = _make_report(
            member_updates=(
                MemberUpdate(
                    name="Alice",
                    summary="login",
                    practices=(
                        PracticeSignal(
                            rule="untracked-work",
                            title="Untracked work",
                            detail="#91 carries no ticket reference.",
                            evidence=(("#91", "https://x/pull/91"),),
                            repeat=True,
                            handles=("url:https://x/pull/91", "commit:acme/web:a1b2"),
                        ),
                    ),
                ),
            ),
            practice_rollup=(("untracked-work", 1),),
        )
        with StandupStore(db_path) as store:
            store.record_run(report)
            loaded = store.get_latest_report("s1")
        signal = loaded.member_updates[0].practices[0]
        assert (signal.rule, signal.title, signal.repeat) == ("untracked-work", "Untracked work", True)
        # JSON turned the pair into a list — it must come back as a tuple.
        assert signal.evidence == (("#91", "https://x/pull/91"),)
        # Without these a stored signal cannot be voted on, so they matter as
        # much as the prose does.
        assert signal.handles == ("url:https://x/pull/91", "commit:acme/web:a1b2")
        assert loaded.practice_rollup == (("untracked-work", 1),)

    def test_a_signal_stored_before_handles_existed_still_loads(self, db_path):
        import json

        from yeaboi.agent.state import PracticeSignal

        report = _make_report(
            member_updates=(MemberUpdate(name="Alice", practices=(PracticeSignal(rule="untracked-work"),)),),
        )
        with StandupStore(db_path) as store:
            store.record_run(report)
            row = store._conn.execute("SELECT id, report_json FROM standup_history").fetchone()
            payload = json.loads(row[1])
            payload["member_updates"][0]["practices"][0].pop("handles", None)
            store._conn.execute(
                "UPDATE standup_history SET report_json = ? WHERE id = ?", (json.dumps(payload), row[0])
            )
            loaded = store.get_latest_report("s1")
        assert loaded.member_updates[0].practices[0].handles == ()

    def test_a_legacy_report_json_with_neither_key_still_loads(self, db_path):
        import json

        with StandupStore(db_path) as store:
            store.record_run(_make_report())
            row = store._conn.execute("SELECT id, report_json FROM standup_history").fetchone()
            payload = json.loads(row[1])
            payload["member_updates"][0].pop("practices", None)
            payload.pop("practice_rollup", None)
            store._conn.execute(
                "UPDATE standup_history SET report_json = ? WHERE id = ?", (json.dumps(payload), row[0])
            )
            loaded = store.get_latest_report("s1")
        assert loaded.member_updates[0].practices == ()
        assert loaded.practice_rollup == ()


class TestConflictsRoundTrip:
    def test_conflict_cards_survive_the_store(self, db_path):
        card = ConflictCard(
            fingerprint="YEA-12:status:status_conflict",
            title="YEA-12 — the board says Done, but a pull request is still open",
            detail="YEA-12 is Done on the board while a PR still names it.",
            severity="medium",
            entity_id="YEA-12",
            property_name="status",
            claims=(("jira", "Done", "YEA-12", "https://j/12"), ("github", "open", "fix", "https://g/41")),
            recommended_action="Reopen YEA-12, or merge the pull request.",
            members=("Bob",),
        )
        report = _make_report(conflicts=(card,))
        with StandupStore(db_path) as store:
            store.record_run(report)
            loaded = store.get_latest_report("s1")
        assert loaded.conflicts == (card,)

    def test_report_without_conflicts_key_still_deserializes(self, db_path):
        import json

        # A report stored before conflict cards existed has no "conflicts" key.
        report = _make_report()
        with StandupStore(db_path) as store:
            store.record_run(report)
            row = store._conn.execute("SELECT id, report_json FROM standup_history").fetchone()
            payload = json.loads(row[1])
            payload.pop("conflicts", None)
            store._conn.execute(
                "UPDATE standup_history SET report_json = ? WHERE id = ?", (json.dumps(payload), row[0])
            )
            loaded = store.get_latest_report("s1")
        assert loaded.conflicts == ()


class TestOpsSignalsRoundTrip:
    def test_a_signal_comes_back_as_a_dataclass_with_its_window(self, db_path):
        report = _make_report(
            ops_signals=(
                OpsSignal(
                    kind="incident",
                    family="incidents",
                    source="pagerduty",
                    count=2,
                    resolved=1,
                    severity="high",
                    services=("checkout", "payments"),
                    window_start="2026-06-26T00:00:00+00:00",
                    window_end="2026-07-10T00:00:00+00:00",
                    samples=("Checkout latency",),
                ),
            )
        )
        with StandupStore(db_path) as store:
            store.record_run(report)
            latest = store.get_latest_report("s1")
        (signal,) = latest.ops_signals
        assert isinstance(signal, OpsSignal)
        assert (signal.count, signal.resolved, signal.severity) == (2, 1, "high")
        assert signal.services == ("checkout", "payments")
        assert signal.window_start == "2026-06-26T00:00:00+00:00"

    def test_a_report_saved_before_ops_existed_still_deserializes(self, db_path):
        import json

        with StandupStore(db_path) as store:
            store.record_run(_make_report())
            row = store._conn.execute("SELECT report_json FROM standup_history").fetchone()
            d = json.loads(row[0])
            d.pop("ops_signals", None)
            store._conn.execute("UPDATE standup_history SET report_json = ?", (json.dumps(d),))
            latest = store.get_latest_report("s1")
        assert latest is not None and latest.ops_signals == ()


class TestScopeFilter:
    """``run_ids`` is the hard filter a resolved context scope hands over."""

    def test_recent_reports_and_history(self, db_path):
        with StandupStore(db_path) as store:
            first = store.record_run(_make_report(date="2026-07-10"))
            second = store.record_run(_make_report(date="2026-07-11"))
            assert [r.date for r in store.get_recent_reports(run_ids=(first,))] == ["2026-07-10"]
            assert store.get_recent_reports(run_ids=()) == []
            assert len(store.get_recent_reports(limit=0)) == 2
            assert [r["id"] for r in store.get_all_history(run_ids=(second,))] == [second]
            assert store.get_all_history(run_ids=()) == []
            assert len(store.get_all_history(limit=0)) == 2

    def test_delete_drops_the_label_row(self, db_path):
        from yeaboi.context.labels import LabelStore

        with StandupStore(db_path) as store:
            run_id = store.record_run(_make_report())
        with LabelStore(db_path) as labels:
            labels.set_labels("standup", "s1", str(run_id))
        with StandupStore(db_path) as store:
            assert store.delete_run(run_id)
        with LabelStore(db_path) as labels:
            assert labels.get_labels("standup", "s1", str(run_id)) is None


class TestContextScopeColumn:
    """The scope a session's standups read under, kept off save_config's full upsert."""

    def test_set_get_and_load_config(self, db_path):
        from yeaboi.context.scope import ContextScope

        with StandupStore(db_path) as store:
            store.save_config("s1", enabled=True, time="09:30", weekdays="1-5", delivery_channels=["terminal"])
            assert store.get_context_scope("s1") is None
            assert store.load_config("s1")["context_scope"] is None
            store.set_context_scope("s1", ContextScope(sources=frozenset({"retro"})))
            assert store.get_context_scope("s1") == {
                "sources": ["retro"],
                "window": {"kind": "all"},
                "projects": [],
                "tags": [],
                "limits": {},
                "sessions": [],
            }
            assert store.load_config("s1")["context_scope"]["sources"] == ["retro"]
            # a config-form save never resets it
            store.save_config("s1", enabled=False, time="09:30", weekdays="1-5", delivery_channels=["terminal"])
            assert store.get_context_scope("s1")["sources"] == ["retro"]
            store.set_context_scope("s1", None)
            assert store.get_context_scope("s1") is None

    def test_set_without_a_config_row_creates_one(self, db_path):
        with StandupStore(db_path) as store:
            store.set_context_scope("fresh", {"sources": None})
            assert store.get_context_scope("fresh") == {"sources": None}
            assert store.get_context_scope("nobody") is None

    def test_unreadable_column_reads_as_none(self, db_path):
        import sqlite3

        with StandupStore(db_path) as store:
            store.set_context_scope("s1", {"sources": None})
        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE standup_config SET context_scope = '{bad'")
        conn.commit()
        conn.close()
        with StandupStore(db_path) as store:
            assert store.get_context_scope("s1") is None
