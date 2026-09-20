"""Tests for src/yeaboi/context/resolve.py — a scope into the ids each store may read."""

from __future__ import annotations

from datetime import date

import pytest

from yeaboi.agent.state import DeliveryReport, OneOnOnePrep, PokerReport, RetroReport, StandupReport, WeeklyReview
from yeaboi.context import resolve as res
from yeaboi.context.labels import LabelStore
from yeaboi.context.resolve import SOURCE_MODES, Preview, Selection, SourceRow, preview_scope, resolve_scope
from yeaboi.context.scope import SOURCES, ContextScope, Window
from yeaboi.context.window import SprintCalendar

TODAY = date(2026, 9, 11)
GRID = SprintCalendar(anchor=date(2026, 8, 31), length_weeks=2, source="settings")


@pytest.fixture
def db(tmp_path):
    return tmp_path / "sessions.db"


@pytest.fixture(autouse=True)
def _fixed_calendar(monkeypatch):
    monkeypatch.setattr(res, "load_sprint_calendar", lambda **_kw: GRID)


@pytest.fixture
def seeded(db):
    """Two standups (one old), a retro, a poker run, a report, a review, a prep, two planning sessions."""
    from yeaboi.performance.store import PerformanceStore
    from yeaboi.poker.store import PokerStore
    from yeaboi.reporting.store import ReportingStore
    from yeaboi.retro.store import RetroStore
    from yeaboi.sessions import SessionStore
    from yeaboi.solo.store import WeeklyReviewStore
    from yeaboi.standup.store import StandupStore
    from yeaboi.team_profile import TeamProfile, TeamProfileStore

    with SessionStore(db) as store:
        store.create_session("p1", "Apollo")
        # Stamped inside the window rather than on the real clock, so the test
        # does not depend on the day it runs.
        store._conn.execute("UPDATE sessions_meta SET created_at = '2026-09-02T10:00:00+00:00'")
    # Analysis rows are team profiles — the id analysis labels its runs with.
    with TeamProfileStore(db) as profiles:
        profiles.save(TeamProfile(team_id="a1", source="jira", project_key="APO"))
        stamp = "2026-09-02T10:00:00+00:00"
        profiles._conn.execute("UPDATE team_profiles SET created_at = ?, updated_at = ?", (stamp, stamp))
    with StandupStore(db) as store:
        old = store.record_run(StandupReport(session_id="p1", date="2026-06-01"))
        new = store.record_run(StandupReport(session_id="p1", date="2026-09-03"))
    with RetroStore(db) as store:
        retro = store.record_run(RetroReport(session_id="p1", date="2026-09-05"))
    with PokerStore(db) as store:
        store.record_run(PokerReport(session_id="p1", date="2026-09-04"))
    with ReportingStore(db) as store:
        store.record_run(DeliveryReport(period_label="Last week", period_end="2026-09-06"), session_id="p1")
    with WeeklyReviewStore(db) as store:
        store.record_run(WeeklyReview(session_id="p1", week_label="2026-W36", week_end="2026-09-06"))
    with PerformanceStore(db) as store:
        store.record_prep(OneOnOnePrep(engineer="Ada", date="2026-09-02"), session_id="p1")
    with LabelStore(db) as labels:
        labels.set_labels("standup", "p1", str(new), project="Apollo", tags=["q3"])
        labels.set_labels("standup", "p1", str(old), project="Zeus", tags=["q2"])
        labels.set_labels("retro", "p1", str(retro), project="Apollo")
    return {"db": db, "old": str(old), "new": str(new), "retro": str(retro)}


class TestSelection:
    def test_unscoped_selection_answers_none_for_everything(self):
        selection = Selection(scope=None)
        assert selection.ids("standup") is None and selection.run_ids("standup") is None
        assert selection.wants("retro")

    def test_switched_off_source_is_empty(self):
        selection = Selection(scope=ContextScope(sources=frozenset({"plan"})), by_source={"plan": None})
        assert selection.ids("retro") == () and selection.run_ids("retro") == ()
        assert selection.ids("plan") is None

    def test_run_ids_parse_ints_and_kind_prefixes(self):
        selection = Selection(scope=ContextScope(), by_source={"standup": ("3", "7"), "performance": ("prep:4", "x")})
        assert selection.run_ids("standup") == (3, 7)
        assert selection.run_ids("performance") == (4,)

    def test_source_row_key(self):
        assert SourceRow("plan", "p1", "", "2026-01-01", "", "").key == "p1"
        assert SourceRow("standup", "p1", "5", "2026-01-01", "", "").key == "5"
        assert SOURCE_MODES["plan"] == "planning" and SOURCE_MODES["review"] == "review"


class TestResolveScope:
    def test_none_reads_no_store(self, db, monkeypatch):
        def boom(_path):
            raise AssertionError("a None scope must not read a store")

        monkeypatch.setattr(res, "_SOURCE_READERS", {name: boom for name in SOURCES})
        selection = resolve_scope(None, today=TODAY, db_path=db)
        assert selection.scope is None and all(selection.ids(s) is None for s in SOURCES)

    def test_a_scope_that_narrows_nothing_reads_no_store(self, db, monkeypatch):
        def boom(_path):
            raise AssertionError("an all-on scope must not read a store")

        monkeypatch.setattr(res, "_SOURCE_READERS", {name: boom for name in SOURCES})
        selection = resolve_scope(ContextScope(), today=TODAY, db_path=db)
        assert all(selection.ids(s) is None for s in SOURCES)

    def test_sources_only_restricts_without_reading(self, db, monkeypatch):
        def boom(_path):
            raise AssertionError("sources alone need no read")

        monkeypatch.setattr(res, "_SOURCE_READERS", {name: boom for name in SOURCES})
        selection = resolve_scope("standup,retro", today=TODAY, db_path=db)
        assert selection.ids("standup") is None and selection.ids("plan") == ()

    def test_window_selects_by_each_source_date_column(self, seeded):
        selection = resolve_scope("all@2sprints", today=TODAY, db_path=seeded["db"])
        assert (selection.start, selection.end) == ("2026-08-17", "2026-09-11")
        assert selection.calendar_source == "settings"
        assert selection.ids("standup") == (seeded["new"],)
        assert selection.ids("retro") == (seeded["retro"],)
        assert len(selection.ids("poker")) == 1
        assert len(selection.ids("reporting")) == 1
        assert len(selection.ids("review")) == 1
        assert selection.ids("performance") == ("prep:1",)
        assert selection.ids("plan") == ("p1",) and selection.ids("analysis") == ("a1",)

    def test_window_that_excludes_everything_is_empty_not_none(self, seeded):
        selection = resolve_scope("standup@2020-01-01..2020-02-01", today=TODAY, db_path=seeded["db"])
        assert selection.ids("standup") == ()

    def test_labels_intersect(self, seeded):
        assert resolve_scope("standup project=apollo", today=TODAY, db_path=seeded["db"]).ids("standup") == (
            seeded["new"],
        )
        assert resolve_scope("standup tags=q2", today=TODAY, db_path=seeded["db"]).ids("standup") == (seeded["old"],)
        assert resolve_scope("standup project=Apollo tags=q2", today=TODAY, db_path=seeded["db"]).ids("standup") == ()
        assert resolve_scope("retro project=zeus", today=TODAY, db_path=seeded["db"]).ids("retro") == ()

    def test_limits_cap_newest_first(self, seeded):
        selection = resolve_scope("standup:1", today=TODAY, db_path=seeded["db"])
        assert selection.ids("standup") == (seeded["new"],)
        assert selection.run_ids("standup") == (int(seeded["new"]),)

    def test_a_broken_store_degrades_to_unrestricted(self, seeded, monkeypatch):
        def boom(_path):
            raise RuntimeError("locked")

        monkeypatch.setitem(res._SOURCE_READERS, "retro", boom)
        selection = resolve_scope("all@month", today=TODAY, db_path=seeded["db"])
        assert selection.ids("retro") is None
        assert selection.ids("standup") == (seeded["new"],)
        assert any("Retros" in w for w in selection.warnings)

    def test_missing_database_reads_nothing(self, tmp_path):
        selection = resolve_scope("all@month", today=TODAY, db_path=tmp_path / "nope.db")
        assert selection.ids("standup") == ()

    def test_bad_custom_window_is_ignored_with_a_warning(self, seeded):
        scope = ContextScope(window=Window(kind="custom", start="2026-02-30"))
        selection = resolve_scope(scope, today=TODAY, db_path=seeded["db"])
        assert selection.start == "" and any("window ignored" in w for w in selection.warnings)

    def test_bad_spec_raises(self, db):
        with pytest.raises(ValueError):
            resolve_scope("stanup", today=TODAY, db_path=db)


class TestPreview:
    def test_counts_and_label(self, seeded):
        preview = preview_scope("standup,retro@2sprints", today=TODAY, db_path=seeded["db"])
        assert isinstance(preview, Preview)
        assert preview.counts["standup"] == 1 and preview.counts["retro"] == 1 and preview.counts["plan"] == 0
        assert preview.label == "1 standup · 1 retro · 17 Aug – 11 Sep"

    def test_unscoped_preview_counts_everything(self, seeded):
        preview = preview_scope(None, today=TODAY, db_path=seeded["db"])
        assert preview.counts["standup"] == 2 and preview.counts["performance"] == 1
        assert preview.selection.ids("standup") is None
        assert "2 standups" in preview.label and "1 1:1s and reviews" in preview.label

    def test_rows_on_request(self, seeded):
        preview = preview_scope("retro", today=TODAY, db_path=seeded["db"], rows=True)
        assert preview.rows["retro"][0].title == "Retro — 2026-09-05"
        # The rows a picker lists carry their labels (the seeded retro is Apollo's).
        assert preview.rows["retro"][0].project == "Apollo"
        assert preview_scope("retro", today=TODAY, db_path=seeded["db"]).rows == {}

    def test_incognito_preview(self, seeded):
        preview = preview_scope("none", today=TODAY, db_path=seeded["db"])
        assert preview.label == "nothing to read" and all(n == 0 for n in preview.counts.values())


class TestSelectionFor:
    def test_scope_for_is_the_rule_planning_shares(self, monkeypatch):
        monkeypatch.setattr("yeaboi.config.get_last_context_scope", lambda mode: {"sources": ["retro"]})
        assert res.scope_for("planning", "standup@month").sources == frozenset({"standup"})
        assert res.scope_for("planning", None).sources == frozenset({"retro"})
        monkeypatch.setattr("yeaboi.config.get_last_context_scope", lambda mode: None)
        assert res.scope_for("planning", None) is None
        assert res.scope_for("planning", None, fallback="stanup") is None  # a stored typo degrades
        with pytest.raises(ValueError, match="stanup"):
            res.scope_for("planning", "stanup")

    """The surfaces' precedence: caller → the mode's saved scope → last used → unscoped."""

    @pytest.fixture(autouse=True)
    def _no_last_used(self, monkeypatch):
        monkeypatch.setattr("yeaboi.config.get_last_context_scope", lambda mode: None)

    def test_none_reads_no_store(self, tmp_path):
        selection = res.selection_for("standup", None, db_path=tmp_path / "never.db")
        assert selection.scope is None and selection.ids("standup") is None

    def test_the_caller_wins(self, seeded, monkeypatch):
        monkeypatch.setattr("yeaboi.config.get_last_context_scope", lambda mode: {"sources": ["retro"]})
        selection = res.selection_for(
            "standup", "standup@month", fallback={"sources": ["plan"]}, today=TODAY, db_path=seeded["db"]
        )
        assert selection.scope.sources == frozenset({"standup"})
        assert selection.ids("standup") == (seeded["new"],)

    def test_the_saved_scope_beats_the_last_used_one(self, seeded, monkeypatch):
        monkeypatch.setattr("yeaboi.config.get_last_context_scope", lambda mode: {"sources": ["retro"]})
        selection = res.selection_for("standup", None, fallback={"sources": ["plan"]}, db_path=seeded["db"])
        assert selection.scope.sources == frozenset({"plan"})

    def test_the_last_used_scope_is_the_final_fallback(self, seeded, monkeypatch):
        monkeypatch.setattr("yeaboi.config.get_last_context_scope", lambda mode: {"sources": ["retro"]})
        selection = res.selection_for("standup", None, db_path=seeded["db"])
        assert selection.scope.sources == frozenset({"retro"})

    def test_a_bad_caller_value_raises_and_a_bad_stored_one_degrades(self, seeded, monkeypatch):
        with pytest.raises(ValueError, match="stanup"):
            res.selection_for("standup", "stanup", fallback={"sources": ["plan"]}, db_path=seeded["db"])
        selection = res.selection_for("standup", None, fallback="stanup", db_path=seeded["db"])
        assert selection.scope is None
        monkeypatch.setattr("yeaboi.config.get_last_context_scope", lambda mode: "stanup")
        assert res.selection_for("standup", None, db_path=seeded["db"]).scope is None


class TestPinnedSessions:
    """A pin is always read: past the window, under a switched-off source, ahead of the newest."""

    def test_a_pinned_session_bypasses_the_window(self, db, seeded):
        from yeaboi.context.scope import SessionRef

        old = seeded["old"]
        scope = ContextScope(window=Window(kind="sprints", count=1), sessions=(SessionRef("standup", "p1", old),))
        ids = resolve_scope(scope, today=TODAY, db_path=db).ids("standup")
        assert ids is not None and ids[0] == old and seeded["new"] in ids

    def test_a_pinned_run_reads_under_an_excluded_source(self, db, seeded):
        from yeaboi.context.scope import SessionRef

        scope = ContextScope(sources=frozenset({"plan"}), sessions=(SessionRef("retro", "p1", seeded["retro"]),))
        selection = resolve_scope(scope, today=TODAY, db_path=db)
        assert selection.ids("retro") == (seeded["retro"],)
        assert selection.ids("standup") == ()
        assert selection.ids("plan") is None

    def test_pins_come_first_and_count(self, db, seeded):
        from yeaboi.context.scope import SessionRef

        scope = ContextScope(
            window=Window(kind="sprints", count=1), sessions=(SessionRef("standup", "p1", seeded["old"]),)
        )
        preview = preview_scope(scope, today=TODAY, db_path=db, rows=True)
        assert preview.counts["standup"] == 2
        assert preview.rows["standup"][0].key == seeded["old"]
