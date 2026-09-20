"""Tests for src/yeaboi/context/reads.py — the two common cross-mode reads under a selection."""

from __future__ import annotations

import pytest

from yeaboi.agent.state import MemberUpdate, Sprint, StandupReport
from yeaboi.context.reads import latest_planning_state, recent_standup_blockers
from yeaboi.context.resolve import Selection
from yeaboi.context.scope import ContextScope
from yeaboi.sessions import SessionStore
from yeaboi.standup.store import StandupStore


@pytest.fixture
def db(tmp_path):
    return tmp_path / "sessions.db"


def _selection(scope: ContextScope, **by_source) -> Selection:
    return Selection(scope=scope, by_source=by_source)


class TestLatestPlanningState:
    def _seed(self, db):
        with SessionStore(db) as store:
            store.create_session("old", "Old")
            store.save_state(
                "old",
                {
                    "sprints": [Sprint(id="S1", name="Sprint 1", goal="ship", capacity_points=10, story_ids=())],
                    "messages": [],
                },
            )
            store.create_session("intake", "Intake")
            store.save_state("intake", {"messages": []})

    def test_unscoped_selection_is_none(self, db):
        assert latest_planning_state(None, db_path=db) is None
        assert latest_planning_state(Selection(scope=None), db_path=db) is None

    def test_plans_switched_off_is_none(self, db):
        self._seed(db)
        assert latest_planning_state(_selection(ContextScope(sources=frozenset({"retro"}))), db_path=db) is None

    def test_unrestricted_plans_scan_newest_first_and_skip_sprintless(self, db):
        self._seed(db)
        found = latest_planning_state(_selection(ContextScope(sources=frozenset({"plan"}))), db_path=db)
        assert found is not None and found[0] == "old"

    def test_restricted_ids_are_honoured(self, db):
        self._seed(db)
        selection = _selection(ContextScope(sources=frozenset({"plan"})), plan=("intake",))
        assert latest_planning_state(selection, db_path=db) is None
        selection = _selection(ContextScope(sources=frozenset({"plan"})), plan=("old",))
        assert latest_planning_state(selection, db_path=db)[0] == "old"

    def test_missing_database_and_failures_never_raise(self, tmp_path, monkeypatch):
        selection = _selection(ContextScope(sources=frozenset({"plan"})))
        assert latest_planning_state(selection, db_path=tmp_path / "nope.db") is None

        def boom(*_a, **_k):
            raise RuntimeError("down")

        monkeypatch.setattr("yeaboi.sessions.SessionStore.__init__", boom)
        assert latest_planning_state(selection, db_path=tmp_path / "x.db") is None


class TestPinnedPlanComesFirst:
    def test_the_resolved_order_is_honoured(self, db):
        from yeaboi.context.resolve import resolve_scope
        from yeaboi.context.scope import SessionRef

        sprint = Sprint(id="S1", name="Sprint 1", goal="ship", capacity_points=10, story_ids=())
        with SessionStore(db) as store:
            store.create_session("older", "Older")
            store.save_state("older", {"sprints": [sprint], "messages": []})
            store.create_session("newer", "Newer")
            store.save_state("newer", {"sprints": [sprint], "messages": []})
            store._conn.execute(
                "UPDATE sessions_meta SET created_at = '2026-01-01T00:00:00' WHERE session_id = 'older'"
            )
        pinned = ContextScope(sessions=(SessionRef("planning", "older"),))
        selection = resolve_scope(pinned, db_path=db)
        # The pin alone puts the named plan first; the rest still follow.
        assert selection.ids("plan") == ("older", "newer")
        assert latest_planning_state(selection, db_path=db)[0] == "older"
        selection = resolve_scope(
            ContextScope(sessions=(SessionRef("planning", "older"),), limits=(("plan", 1),)), db_path=db
        )
        assert selection.ids("plan")[0] == "older"
        assert latest_planning_state(selection, db_path=db)[0] == "older"


class TestRecentStandupBlockers:
    def _seed(self, db):
        with StandupStore(db) as store:
            store.record_run(
                StandupReport(
                    session_id="s1",
                    date="2026-09-01",
                    member_updates=(
                        MemberUpdate(name="Ana", blockers="CI is red"),
                        MemberUpdate(name="Bo", blockers=""),
                    ),
                )
            )
            store.record_run(
                StandupReport(
                    session_id="s1",
                    date="2026-09-02",
                    member_updates=(
                        MemberUpdate(name="Ana", blockers="ci is RED"),
                        MemberUpdate(name="", blockers="No env"),
                    ),
                )
            )

    def test_unscoped_reads_nothing(self, db):
        self._seed(db)
        assert recent_standup_blockers(None, db_path=db) == []
        assert recent_standup_blockers(Selection(scope=None), db_path=db) == []

    def test_scoped_dedupes_case_insensitively(self, db):
        self._seed(db)
        blockers = recent_standup_blockers(_selection(ContextScope(sources=frozenset({"standup"}))), db_path=db)
        assert blockers == ["Ana: ci is RED", "No env"]

    def test_standups_off_and_restricted_ids(self, db):
        self._seed(db)
        assert recent_standup_blockers(_selection(ContextScope(sources=frozenset({"retro"}))), db_path=db) == []
        selection = _selection(ContextScope(sources=frozenset({"standup"})), standup=("1",))
        assert recent_standup_blockers(selection, db_path=db) == ["Ana: CI is red"]

    def test_missing_database_and_failures_never_raise(self, tmp_path, monkeypatch):
        selection = _selection(ContextScope(sources=frozenset({"standup"})))
        assert recent_standup_blockers(selection, db_path=tmp_path / "nope.db") == []

        def boom(*_a, **_k):
            raise RuntimeError("down")

        monkeypatch.setattr("yeaboi.standup.store.StandupStore.__init__", boom)
        assert recent_standup_blockers(selection, db_path=tmp_path / "x.db") == []
