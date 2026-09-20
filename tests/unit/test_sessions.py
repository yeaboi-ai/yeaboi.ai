"""Tests for sessions.py — SessionStore file hardening and schema bookkeeping.

(The store's behaviour is covered indirectly across the mode suites; this file
holds the direct SessionStore unit tests, starting with the security bits.)
"""

import os
import sqlite3
import stat

import pytest

from yeaboi.sessions import CURRENT_SCHEMA_VERSION, SessionStore, plan_title, provisional_title


class TestSessionStoreFilePermissions:
    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_db_file_restricted_on_connect(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        store = SessionStore(db_path)
        try:
            assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
        finally:
            store._conn.close()

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_existing_lax_db_repaired(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        db_path.touch(mode=0o644)
        db_path.chmod(0o644)
        store = SessionStore(db_path)
        try:
            assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
        finally:
            store._conn.close()


class TestSchemaInfoSingleRow:
    """schema_info is a single-row table only by convention — opens must enforce it.

    Concurrent first-opens (TUI + MCP server + scheduler on the shared DB) race
    the stamp INSERT, leaving duplicate rows and making the version read
    arbitrary — one observed DB held 37 rows. Every open now dedupes to the
    single highest-version row.
    """

    def _rows(self, db_path):
        conn = sqlite3.connect(str(db_path))
        try:
            return [r[0] for r in conn.execute("SELECT schema_version FROM schema_info")]
        finally:
            conn.close()

    def _insert_rows(self, db_path, versions):
        conn = sqlite3.connect(str(db_path))
        for v in versions:
            conn.execute("INSERT INTO schema_info (schema_version) VALUES (?)", (v,))
        conn.commit()
        conn.close()

    def test_duplicate_rows_are_deduped_on_open(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        SessionStore(db_path).close()
        self._insert_rows(db_path, [CURRENT_SCHEMA_VERSION] * 30 + [1, 20, 25])

        store = SessionStore(db_path)
        try:
            assert not store.schema_mismatch
        finally:
            store.close()
        assert self._rows(db_path) == [CURRENT_SCHEMA_VERSION]

    def test_dedupe_keeps_the_newest_stamp(self, tmp_path):
        # A row stamped by a newer build must survive the dedupe, so the
        # newer-DB-older-code warning still fires and nothing downgrades it.
        db_path = tmp_path / "sessions.db"
        SessionStore(db_path).close()
        self._insert_rows(db_path, [CURRENT_SCHEMA_VERSION + 1, 3])

        store = SessionStore(db_path)
        try:
            assert store.schema_mismatch
        finally:
            store.close()
        assert self._rows(db_path) == [CURRENT_SCHEMA_VERSION + 1]

    def test_single_row_open_is_untouched(self, tmp_path):
        db_path = tmp_path / "sessions.db"
        SessionStore(db_path).close()
        SessionStore(db_path).close()
        assert self._rows(db_path) == [CURRENT_SCHEMA_VERSION]


class TestArchitectureRoundTrip:
    """ArchitectureDecision and full Task fields survive save/load."""

    def _analysis_with_architecture(self):
        from yeaboi.agent.state import ArchitectureDecision, ArchitectureOption

        return ArchitectureDecision(
            options=(
                ArchitectureOption(name="Monolith", summary="s", pros=("a",), cons=("b",)),
                ArchitectureOption(name="Serverless", summary="s2"),
            ),
            chosen="Monolith",
            confidence="medium",
            rationale="why",
        )

    def test_architecture_round_trips(self, tmp_path):
        from tests._node_helpers import make_dummy_analysis
        from yeaboi.sessions import SessionStore

        analysis = make_dummy_analysis(architecture=self._analysis_with_architecture())
        with SessionStore(tmp_path / "sessions.db") as store:
            store.create_session("s1", "Test")
            store.save_state("s1", {"messages": [], "project_analysis": analysis})
            loaded = store.load_state("s1")
        arch = loaded["project_analysis"].architecture
        assert arch is not None
        assert arch.chosen == "Monolith"
        assert arch.options[0].pros == ("a",)

    def test_old_analysis_without_architecture_loads(self, tmp_path):
        from tests._node_helpers import make_dummy_analysis
        from yeaboi.sessions import SessionStore

        with SessionStore(tmp_path / "sessions.db") as store:
            store.create_session("s1", "Test")
            store.save_state("s1", {"messages": [], "project_analysis": make_dummy_analysis()})
            loaded = store.load_state("s1")
        assert loaded["project_analysis"].architecture is None

    def test_task_label_and_plans_survive_resume(self, tmp_path):
        # Regression: _dict_to_task used to drop label/test_plan/ai_prompt.
        from yeaboi.agent.state import Task, TaskLabel
        from yeaboi.sessions import SessionStore

        task = Task(
            id="T-1",
            story_id="US-1",
            title="[Spike] Validate architecture: X",
            description="d",
            label=TaskLabel.SPIKE,
            test_plan="",
            ai_prompt="research prompt",
        )
        with SessionStore(tmp_path / "sessions.db") as store:
            store.create_session("s1", "Test")
            store.save_state("s1", {"messages": [], "tasks": [task]})
            loaded = store.load_state("s1")
        restored = loaded["tasks"][0]
        assert restored.label is TaskLabel.SPIKE
        assert restored.ai_prompt == "research prompt"

    def test_old_task_dict_without_new_keys_loads(self):
        from yeaboi.agent.state import TaskLabel
        from yeaboi.sessions import _dict_to_task

        task = _dict_to_task({"id": "T-1", "story_id": "US-1", "title": "t", "description": "d"})
        assert task.label is TaskLabel.CODE
        assert task.test_plan == ""


class TestWeeklyReviewMigration:
    """Migration v32 — the weekly_review_history table."""

    def _v31_db(self, tmp_path):
        db = tmp_path / "sessions.db"
        conn = sqlite3.connect(str(db))
        conn.executescript(
            """CREATE TABLE sessions_meta (
                   session_id          TEXT PRIMARY KEY,
                   project_name        TEXT NOT NULL DEFAULT '',
                   created_at          TEXT NOT NULL,
                   last_modified       TEXT NOT NULL,
                   last_node_completed TEXT NOT NULL DEFAULT '',
                   session_state       TEXT NOT NULL DEFAULT '',
                   session_mode        TEXT NOT NULL DEFAULT 'planning',
                   project_id          TEXT NOT NULL DEFAULT ''
               );
               CREATE TABLE schema_info (schema_version INT NOT NULL);"""
        )
        conn.execute("INSERT INTO schema_info VALUES (31)")
        conn.commit()
        conn.close()
        return db

    def test_v31_db_gains_the_table(self, tmp_path):
        db = self._v31_db(tmp_path)
        with SessionStore(db) as store:
            assert store.schema_mismatch is False
        conn = sqlite3.connect(str(db))
        try:
            names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
            assert "weekly_review_history" in names
            (version,) = conn.execute("SELECT schema_version FROM schema_info").fetchone()
            assert version == CURRENT_SCHEMA_VERSION
        finally:
            conn.close()

    def test_reopen_is_idempotent(self, tmp_path):
        db = self._v31_db(tmp_path)
        SessionStore(db).close()
        with SessionStore(db) as store:
            assert store.schema_mismatch is False


class TestSoloStateRoundTrip:
    def test_solo_survives_save_and_load(self, tmp_path):
        with SessionStore(tmp_path / "sessions.db") as store:
            store.create_session("s1", "Test")
            store.save_state("s1", {"messages": [], "solo": True})
            loaded = store.load_state("s1")
        assert loaded["solo"] is True


class TestSessionIntegrationsRoundTrip:
    """The plan-level keys the room writes survive a save and a load unchanged."""

    def test_the_three_keys_round_trip(self, tmp_path):
        state = {
            "messages": [],
            "session_integrations": ["jira", "github"],
            "pasted_context": ["Reference: PROJ-1"],
            "chat_context": [],
        }
        with SessionStore(tmp_path / "sessions.db") as store:
            store.create_session("s1", "Test")
            store.save_state("s1", state)
            loaded = store.load_state("s1")
        assert loaded["session_integrations"] == ["jira", "github"]
        assert loaded["pasted_context"] == ["Reference: PROJ-1"]
        assert loaded["chat_context"] == []

    def test_an_absent_key_stays_absent(self, tmp_path):
        with SessionStore(tmp_path / "sessions.db") as store:
            store.create_session("s1", "Test")
            store.save_state("s1", {"messages": []})
            loaded = store.load_state("s1")
        assert "session_integrations" not in loaded


class TestListSessionsFilters:
    """The additive kwargs the cross-mode recent list reads through."""

    def _seed(self, path):
        with SessionStore(path) as store:
            store.create_session("p1", "Apollo")
            store.create_session("a1", "Apollo", mode="analysis")
            store.create_session("p2", "Borealis")
        return path

    def test_rows_carry_their_mode(self, tmp_path):
        with SessionStore(self._seed(tmp_path / "s.db")) as store:
            rows = {r["session_id"]: r for r in store.list_sessions()}
        assert rows["p1"]["session_mode"] == "planning"
        assert rows["a1"]["session_mode"] == "analysis"

    def test_mode_filter(self, tmp_path):
        with SessionStore(self._seed(tmp_path / "s.db")) as store:
            assert [r["session_id"] for r in store.list_sessions(mode="analysis")] == ["a1"]

    def test_limit_caps_and_zero_means_all(self, tmp_path):
        with SessionStore(self._seed(tmp_path / "s.db")) as store:
            assert len(store.list_sessions(limit=2)) == 2
            assert len(store.list_sessions(limit=0)) == 3


class TestProjectsRemovedMigration:
    """Migration v34 — the projects table, its index and the link column go."""

    def _v33_db(self, tmp_path):
        db = tmp_path / "sessions.db"
        conn = sqlite3.connect(str(db))
        conn.executescript(
            """CREATE TABLE sessions_meta (
                   session_id          TEXT PRIMARY KEY,
                   project_name        TEXT NOT NULL DEFAULT '',
                   created_at          TEXT NOT NULL,
                   last_modified       TEXT NOT NULL,
                   last_node_completed TEXT NOT NULL DEFAULT '',
                   session_state       TEXT NOT NULL DEFAULT '',
                   session_mode        TEXT NOT NULL DEFAULT 'planning',
                   project_id          TEXT NOT NULL DEFAULT ''
               );
               CREATE INDEX idx_sessions_meta_project ON sessions_meta(project_id);
               CREATE TABLE projects (
                   id     TEXT PRIMARY KEY,
                   name   TEXT NOT NULL DEFAULT '',
                   status TEXT NOT NULL DEFAULT 'active'
               );
               CREATE TABLE schema_info (schema_version INT NOT NULL);"""
        )
        conn.execute("INSERT INTO schema_info VALUES (33)")
        conn.execute("INSERT INTO projects VALUES ('proj-11112222', 'Apollo', 'active')")
        conn.execute(
            "INSERT INTO sessions_meta (session_id, created_at, last_modified, project_id) "
            "VALUES ('old-1', 't', 't', 'proj-11112222')"
        )
        conn.commit()
        conn.close()
        return db

    def _names(self, db):
        conn = sqlite3.connect(str(db))
        try:
            return {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
        finally:
            conn.close()

    def test_the_table_and_index_are_dropped_and_sessions_survive(self, tmp_path):
        db = self._v33_db(tmp_path)
        with SessionStore(db) as store:
            assert store.schema_mismatch is False
            assert [r["session_id"] for r in store.list_sessions()] == ["old-1"]
        names = self._names(db)
        assert "projects" not in names
        assert "idx_sessions_meta_project" not in names

    def test_reopen_is_idempotent(self, tmp_path):
        db = self._v33_db(tmp_path)
        SessionStore(db).close()
        with SessionStore(db) as store:
            assert store.schema_mismatch is False


class TestPlanVersionsMigration:
    """Migration v35 — a title column and the plan_versions table."""

    def _v34_db(self, tmp_path):
        db = tmp_path / "sessions.db"
        conn = sqlite3.connect(str(db))
        conn.executescript(
            """CREATE TABLE sessions_meta (
                   session_id          TEXT PRIMARY KEY,
                   project_name        TEXT NOT NULL DEFAULT '',
                   created_at          TEXT NOT NULL,
                   last_modified       TEXT NOT NULL,
                   last_node_completed TEXT NOT NULL DEFAULT '',
                   session_state       TEXT NOT NULL DEFAULT '',
                   session_mode        TEXT NOT NULL DEFAULT 'planning'
               );
               CREATE TABLE schema_info (schema_version INT NOT NULL);"""
        )
        conn.execute("INSERT INTO schema_info VALUES (34)")
        conn.execute("INSERT INTO sessions_meta (session_id, created_at, last_modified) VALUES ('old-1', 't', 't')")
        conn.commit()
        conn.close()
        return db

    def test_the_column_and_table_arrive_and_rows_survive(self, tmp_path):
        db = self._v34_db(tmp_path)
        with SessionStore(db) as store:
            assert store.schema_mismatch is False
            (row,) = store.list_sessions()
            assert row["session_id"] == "old-1" and row["title"] == ""
            store.update_session_meta("old-1", title="Apollo")
            assert store.get_session("old-1")["title"] == "Apollo"
            assert store.record_plan_version("old-1", "features", {"items": []}) == 1
        conn = sqlite3.connect(str(db))
        try:
            assert conn.execute("SELECT schema_version FROM schema_info").fetchone()[0] == CURRENT_SCHEMA_VERSION
        finally:
            conn.close()

    def test_reopen_is_idempotent(self, tmp_path):
        db = self._v34_db(tmp_path)
        SessionStore(db).close()
        with SessionStore(db) as store:
            assert store.schema_mismatch is False


class TestMessagesRoundTrip:
    def test_messages_survive_a_save_including_tool_calls(self, tmp_path):
        from langchain_core.messages import AIMessage, HumanMessage

        call = {"name": "jira_create_epic", "args": {"summary": "Login"}, "id": "call-1", "type": "tool_call"}
        with SessionStore(tmp_path / "s.db") as store:
            store.create_session("new-1")
            store.save_state(
                "new-1", {"messages": [HumanMessage(content="hi"), AIMessage(content="", tool_calls=[call])]}
            )
            back = store.load_state("new-1")
        assert [type(m).__name__ for m in back["messages"]] == ["HumanMessage", "AIMessage"]
        assert back["messages"][1].tool_calls == [call]

    def test_a_blob_without_messages_loads_an_empty_list(self, tmp_path):
        with SessionStore(tmp_path / "s.db") as store:
            store.create_session("new-1")
            store.save_state("new-1", {"solo": True})
            assert store.load_state("new-1")["messages"] == []


class TestPlanVersions:
    def test_versions_count_up_per_section(self, tmp_path):
        with SessionStore(tmp_path / "s.db") as store:
            store.create_session("new-1")
            assert store.record_plan_version("new-1", "features", {"items": [1]}) == 1
            assert store.record_plan_version("new-1", "features", {"items": [2]}) == 2
            assert store.record_plan_version("new-1", "stories", {"items": []}) == 1
            assert [(v["section"], v["version"]) for v in store.list_plan_versions("new-1")] == [
                ("features", 1),
                ("features", 2),
                ("stories", 1),
            ]
            assert [v["version"] for v in store.list_plan_versions("new-1", "stories")] == [1]
            assert store.get_plan_version("new-1", "features", 2)["payload"] == {"items": [2]}

    def test_an_unknown_version_is_none(self, tmp_path):
        with SessionStore(tmp_path / "s.db") as store:
            assert store.get_plan_version("new-1", "features", 1) is None

    def test_deleting_the_session_drops_its_versions(self, tmp_path):
        with SessionStore(tmp_path / "s.db") as store:
            store.create_session("new-1")
            store.record_plan_version("new-1", "features", {})
            assert store.delete_session("new-1") is True
            assert store.list_plan_versions("new-1") == []


class TestSessionTitle:
    def test_a_title_is_kept_beside_the_project_name(self, tmp_path):
        with SessionStore(tmp_path / "s.db") as store:
            store.create_session("new-1", "Derived", title="Given")
            row = store.get_session("new-1")
            assert (row["project_name"], row["title"], row["session_mode"]) == ("Derived", "Given", "planning")
            store.update_session_meta("new-1", title=None)
            assert store.get_session("new-1")["title"] == "Given"


class TestSessionLabelsMigration:
    """Migration v35 also creates ``session_labels``, and deletes clear a session's rows."""

    def test_the_table_arrives_with_its_key(self, tmp_path):
        db = TestPlanVersionsMigration()._v34_db(tmp_path)
        with SessionStore(db) as store:
            assert store.schema_mismatch is False
        conn = sqlite3.connect(str(db))
        try:
            names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master").fetchall()}
            assert "session_labels" in names and "idx_session_labels_project" in names
            columns = [r[1] for r in conn.execute("PRAGMA table_info(session_labels)").fetchall()]
            assert columns == [
                "mode",
                "session_id",
                "run_id",
                "project",
                "tags_json",
                "scope_json",
                "created_at",
                "updated_at",
            ]
        finally:
            conn.close()

    def test_reopen_is_idempotent(self, tmp_path):
        db = TestPlanVersionsMigration()._v34_db(tmp_path)
        SessionStore(db).close()
        with SessionStore(db) as store:
            assert store.schema_mismatch is False

    def test_deleting_a_session_drops_its_label_rows_only(self, tmp_path):
        from yeaboi.context.labels import LabelStore

        db = tmp_path / "sessions.db"
        with SessionStore(db) as store:
            store.create_session("p1", "Apollo")
            store.create_session("p2", "Ares")
        with LabelStore(db) as labels:
            labels.set_labels("planning", "p1", project="Apollo")
            labels.set_labels("planning", "p2", project="Ares")
            labels.set_labels("standup", "p1", "1", project="Apollo")
        with SessionStore(db) as store:
            assert store.delete_session("p1")
        with LabelStore(db) as labels:
            assert labels.get_labels("planning", "p1") is None
            assert labels.get_labels("planning", "p2") is not None
            assert labels.get_labels("standup", "p1", "1") is not None  # a run's row belongs to its own store
        with LabelStore(db) as labels:
            labels.set_labels("analysis", "jira-PROJ-20260401", project="Apollo")  # keyed by team id
        with SessionStore(db) as store:
            store.delete_all_sessions()
        with LabelStore(db) as labels:
            assert labels.get_labels("planning", "p2") is None
            assert labels.get_labels("analysis", "jira-PROJ-20260401") is not None


class TestProvisionalTitle:
    def test_blank_description_has_no_title(self):
        assert provisional_title("") == ""
        assert provisional_title("   \n ") == ""

    def test_a_short_description_is_its_own_title(self):
        assert provisional_title("A booking app for barbers") == "A booking app for barbers"

    def test_only_the_first_sentence_is_used(self):
        assert provisional_title("A booking app for barbers. It needs payments.") == "A booking app for barbers."

    def test_a_long_first_sentence_is_cut_at_a_word(self):
        text = "A booking platform for independent barbers with online payments and reminders for every customer"
        title = provisional_title(text)
        assert title.endswith("…") and len(title) <= 61
        assert title == "A booking platform for independent barbers with online…"

    def test_a_single_long_word_is_cut_hard(self):
        assert provisional_title("x" * 80) == "x" * 60 + "…"


class TestPlanTitle:
    def test_the_users_title_wins(self):
        assert plan_title("Mine", "Analysed", "described") == "Mine"

    def test_the_analysed_name_beats_the_description(self):
        assert plan_title("", "Analysed", "described") == "Analysed"

    def test_the_description_is_the_last_resort(self):
        assert plan_title("", "", "A booking app. More.") == "A booking app."
        assert plan_title("", "", "") == ""
