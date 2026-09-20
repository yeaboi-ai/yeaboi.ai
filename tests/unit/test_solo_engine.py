"""Tests for the Weekly Review engine (solo/engine.py) — mocked LLM and tracker."""

from __future__ import annotations

import json
from datetime import date

import pytest

from yeaboi import paths
from yeaboi.agent.state import DeliveredItem, MemberUpdate, ReviewAction, Sprint, StandupReport, UserStory, WeeklyReview
from yeaboi.reporting import activity as activity_mod
from yeaboi.sessions import SessionStore
from yeaboi.solo import engine
from yeaboi.solo.store import WeeklyReviewStore
from yeaboi.standup.store import StandupStore

TODAY = date(2026, 9, 4)  # Friday; Monday is the 31st


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "SOLO_EXPORTS_DIR", tmp_path / "exports")
    monkeypatch.setattr("yeaboi.persistence.load_projects", lambda: [])
    monkeypatch.setattr("yeaboi.config.get_standup_user_name", lambda: "Dinho")


class _FakeResp:
    def __init__(self, content):
        self.content = content
        self.response_metadata = {}


def _patch_llm(monkeypatch, content):
    monkeypatch.setattr("yeaboi.config.is_llm_configured", lambda: (True, ""))
    monkeypatch.setattr("yeaboi.agent.llm.track_usage", lambda resp: None)
    monkeypatch.setattr(
        "yeaboi.agent.llm.get_llm",
        lambda **k: type("L", (), {"invoke": lambda self, m: _FakeResp(content)})(),
    )


def _patch_llm_raising(monkeypatch, exc):
    monkeypatch.setattr("yeaboi.config.is_llm_configured", lambda: (True, ""))
    monkeypatch.setattr("yeaboi.agent.llm.track_usage", lambda resp: None)

    def _boom(self, m):
        raise exc

    monkeypatch.setattr("yeaboi.agent.llm.get_llm", lambda **k: type("L", (), {"invoke": _boom})())


def _patch_activity(monkeypatch, items=(), warnings=(), calls=None):
    def fake(period, **kw):
        if calls is not None:
            calls.append((period, kw))
        return list(items), [], list(warnings)

    monkeypatch.setattr(activity_mod, "gather_delivered_work", fake)


_GOOD = json.dumps(
    {
        "summary": "A steady week.",
        "went_well": ["Shipped S-1", "Kept the standups honest"],
        "to_change": ["Ask for keys earlier"],
        "actions": ["Split S-2 before starting", "Book the keys on Monday"],
    }
)


def _story(sid, title):
    from yeaboi.agent.state import Priority, StoryPointValue

    return UserStory(
        id=sid,
        feature_id="F-1",
        persona="dev",
        goal="ship it",
        benefit="",
        acceptance_criteria=(),
        story_points=StoryPointValue.THREE,
        priority=Priority.HIGH,
        title=title,
    )


def _plan_state():
    return {
        "project_name": "Apollo",
        "sprint_start_date": "2026-08-31",
        "sprint_length_weeks": 1,
        "stories": [_story("S-1", "Login"), _story("S-2", "Audit log"), _story("S-3", "Search")],
        "sprints": [
            Sprint(id="SP-1", name="Sprint 1", goal="", capacity_points=8, story_ids=("S-1", "S-2")),
            Sprint(id="SP-2", name="Sprint 2", goal="", capacity_points=8, story_ids=("S-3",)),
        ],
    }


def _report(session_id, day, *, pct, label="On track", blockers="", summary="Closed S-1"):
    return StandupReport(
        date=day,
        session_id=session_id,
        sprint_name="Sprint 1",
        sprint_day=2,
        sprint_total_days=5,
        confidence_pct=pct,
        confidence_label=label,
        my_name="Dinho",
        member_updates=(
            MemberUpdate(name="Other", summary="Reviewed"),
            MemberUpdate(name="Dinho", summary=summary, blockers=blockers),
        ),
    )


def _seed(tmp_path, *, standups=True, plan=True):
    db = tmp_path / "sessions.db"
    with SessionStore(db) as sessions:
        sessions.create_session("plan-1", "Apollo")
        if plan:
            sessions.save_state("plan-1", _plan_state())
    if standups:
        with StandupStore(db) as store:
            store.record_run(_report("plan-1", "2026-08-31", pct=60, blockers="waiting on keys"))
            store.record_run(_report("plan-1", "2026-09-02", pct=72, summary="Started S-2"))
            store.record_run(_report("plan-1", "2026-08-25", pct=40, label="Behind"))  # last week
            store.record_run(_report("plan-1", "2026-09-03", pct=50), status="error")  # failed run
    return db


def _items():
    return [
        DeliveredItem(key="S-1", title="Login", status="Done", source="jira", assignee="Dinho"),
        DeliveredItem(key="X-9", title="Someone else's", status="Done", source="jira", assignee="Other"),
    ]


class TestHappyPath:
    def test_full_review(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        _patch_llm(monkeypatch, _GOOD)
        _patch_activity(monkeypatch, items=_items())
        phases = []
        review = engine.run_weekly_review(db_path=db, today=TODAY, on_progress=phases.append)

        assert review.week_label == "2026-W36"
        assert (review.week_start, review.week_end) == ("2026-08-31", "2026-09-04")
        assert review.project_name == "Apollo"
        assert review.my_name == "Dinho"
        # only this week's successful standups, oldest first
        assert review.standup_dates == ("2026-08-31", "2026-09-02")
        assert review.standup_lines[0].startswith("Mon 2026-08-31: Closed S-1 — blocked: waiting on keys")
        assert (review.confidence_start, review.confidence_end) == (60, 72)
        # delivered narrowed to the user
        assert [i.key for i in review.delivered_items] == ["S-1"]
        assert review.planned_story_count == 2 and review.sprint_name == "Sprint 1"
        assert review.plan_status == "on_track"
        assert (
            review.plan_line
            == "Day 2/5 of Sprint 1 · On track (72%, up 12 since Monday) · 1 ticket closed against 2 planned"
        )
        assert review.summary == "A steady week."
        assert review.went_well == ("Shipped S-1", "Kept the standups honest")
        assert [a.text for a in review.actions] == ["Split S-2 before starting", "Book the keys on Monday"]
        assert all(a.origin == "ai" and a.status == "pending" and len(a.id) == 12 for a in review.actions)
        assert review.carried_actions == ()
        assert review.warnings == ()
        # one lifecycle event per transition: every phase runs, then finishes, in order
        from yeaboi.analysis.progress import is_component_progress

        assert all(is_component_progress(e) for e in phases)
        running = [e["component_id"] for e in phases if e["status"] == "running"]
        assert [p for i, p in enumerate(running) if p not in running[:i]] == list(engine.PHASES)
        assert [e["component_id"] for e in phases if e["status"] == "completed"] == list(engine.PHASES)
        assert all(e["label"] == engine.PHASE_LABELS[e["component_id"]] for e in phases)

    def test_partial_standups_still_count(self, monkeypatch, tmp_path):
        # A run with one failed source still carries the user's own update — the
        # same rule the Today strip applies (REVIEWABLE_STANDUP_STATUSES).
        db = _seed(tmp_path, standups=False)
        with StandupStore(db) as store:
            store.record_run(_report("plan-1", "2026-09-01", pct=55, summary="Fixed S-3"), status="partial")
        _patch_llm(monkeypatch, _GOOD)
        _patch_activity(monkeypatch)
        review = engine.run_weekly_review(db_path=db, today=TODAY)
        assert review.standup_dates == ("2026-09-01",)
        assert "Fixed S-3" in review.standup_lines[0]

    def test_saved_and_exported(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        _patch_llm(monkeypatch, _GOOD)
        _patch_activity(monkeypatch)
        review = engine.run_weekly_review(db_path=db, today=TODAY)
        with WeeklyReviewStore(db) as store:
            assert store.get_latest_report() == review
        assert (tmp_path / "exports" / "apollo" / "weekly-review-2026-W36.md").exists()

    def test_fenced_json_is_tolerated(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        _patch_llm(monkeypatch, f"```json\n{_GOOD}\n```")
        _patch_activity(monkeypatch)
        review = engine.run_weekly_review(db_path=db, today=TODAY)
        assert review.summary == "A steady week."

    def test_lists_are_capped_and_cleaned(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        _patch_llm(monkeypatch, json.dumps({"summary": "s", "went_well": ["", " a "] + ["x"] * 10, "actions": "no"}))
        _patch_activity(monkeypatch)
        review = engine.run_weekly_review(db_path=db, today=TODAY)
        assert review.went_well == ("a", "x", "x", "x", "x", "x")
        assert review.actions == ()


class TestFallbacks:
    def test_llm_not_configured(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        monkeypatch.setattr("yeaboi.config.is_llm_configured", lambda: (False, "no API key"))
        _patch_activity(monkeypatch, items=_items())
        review = engine.run_weekly_review(db_path=db, today=TODAY)
        assert review.summary == review.plan_line
        assert review.went_well == ("S-1 Login",)
        assert review.to_change == ("waiting on keys",)
        assert review.actions == ()
        assert any("no API key" in w for w in review.warnings)

    def test_garbage_json(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        _patch_llm(monkeypatch, "not json at all")
        _patch_activity(monkeypatch)
        review = engine.run_weekly_review(db_path=db, today=TODAY)
        assert review.summary == review.plan_line

    def test_llm_exception_is_a_warning(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        _patch_llm_raising(monkeypatch, RuntimeError("boom"))
        _patch_activity(monkeypatch)
        review = engine.run_weekly_review(db_path=db, today=TODAY)
        assert any("LLM request failed" in w for w in review.warnings)
        assert review.summary == review.plan_line

    def test_tracker_failure_is_a_warning(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        _patch_llm(monkeypatch, _GOOD)

        def boom(period, **kw):
            raise ConnectionError("tracker down")

        monkeypatch.setattr(activity_mod, "gather_delivered_work", boom)
        review = engine.run_weekly_review(db_path=db, today=TODAY)
        assert review.delivered_items == ()
        assert "could not read delivered work from the tracker" in review.warnings

    def test_dry_run_skips_tracker_and_llm(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        calls = []
        _patch_activity(monkeypatch, items=_items(), calls=calls)
        monkeypatch.setattr("yeaboi.config.is_llm_configured", lambda: (_ for _ in ()).throw(AssertionError("no LLM")))
        review = engine.run_weekly_review(db_path=db, today=TODAY, dry_run=True)
        assert calls == []
        assert review.delivered_items == ()
        assert "dry run — the tracker was not read" in review.warnings
        assert review.summary == review.plan_line

    def test_missing_db_is_an_empty_review(self, monkeypatch, tmp_path):
        monkeypatch.setattr("yeaboi.config.is_llm_configured", lambda: (False, "no key"))
        _patch_activity(monkeypatch)
        review = engine.run_weekly_review(db_path=tmp_path / "sessions.db", today=TODAY)
        assert review.plan_status == "no_plan"
        assert review.standup_lines == ()
        assert review.my_name == "Dinho"


class TestWeekWindow:
    @pytest.mark.parametrize(
        ("week_end", "today", "expected"),
        [
            ("", date(2026, 9, 4), ("2026-08-31", "2026-09-04", "2026-W36")),
            ("2026-09-02", date(2026, 9, 20), ("2026-08-31", "2026-09-02", "2026-W36")),
            ("2026-08-31", date(2026, 9, 20), ("2026-08-31", "2026-08-31", "2026-W36")),
            ("2026-01-02", date(2026, 9, 20), ("2025-12-29", "2026-01-02", "2026-W01")),
        ],
    )
    def test_monday_to_end(self, week_end, today, expected):
        monday, end, label = engine._week_window(week_end, today)
        assert (monday.isoformat(), end.isoformat(), label) == expected

    def test_junk_is_refused(self):
        with pytest.raises(ValueError, match="week_end must be an ISO date"):
            engine._week_window("next tuesday", TODAY)

    def test_window_reaches_the_tracker(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        _patch_llm(monkeypatch, _GOOD)
        calls = []
        _patch_activity(monkeypatch, calls=calls)
        engine.run_weekly_review(db_path=db, today=TODAY, week_end="2026-09-02")
        ((period, kw),) = calls
        assert period == activity_mod.PERIOD_WINDOW
        assert (kw["window_start"], kw["window_end"], kw["days_override"]) == ("2026-08-31", "2026-09-02", 3)


class TestPlanVerdict:
    def _verdict(self, **kw):
        base = dict(
            has_plan=True,
            has_standups=True,
            confidence_label="On track",
            confidence_start=60,
            confidence_end=72,
            sprint_name="Sprint 1",
            sprint_day=2,
            sprint_total_days=5,
            delivered=1,
            planned=2,
        )
        base.update(kw)
        return engine._plan_verdict(**base)

    def test_no_plan(self):
        assert self._verdict(has_plan=False)[0] == "no_plan"

    def test_no_standups(self):
        status, line = self._verdict(has_standups=False)
        assert status == "no_data" and "No standups this week" in line

    @pytest.mark.parametrize(
        ("label", "status"),
        [("On track", "on_track"), ("At risk", "at_risk"), ("Behind", "behind"), ("Insufficient data", "no_data")],
    )
    def test_label_maps_to_status(self, label, status):
        assert self._verdict(confidence_label=label)[0] == status

    def test_trend_wording(self):
        assert "up 12" in self._verdict()[1]
        assert "down 8" in self._verdict(confidence_start=80)[1]
        assert "flat" in self._verdict(confidence_start=72)[1]

    def test_plural_and_sprint_fallback(self):
        line = self._verdict(delivered=3, sprint_name="")[1]
        assert "3 tickets closed against 2 planned" in line and "of the sprint" in line


class TestCarryForward:
    def _first(self, monkeypatch, db):
        _patch_llm(monkeypatch, _GOOD)
        _patch_activity(monkeypatch)
        return engine.run_weekly_review(db_path=db, today=TODAY)

    def test_second_review_carries_the_first_actions(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        first = self._first(monkeypatch, db)
        second = engine.run_weekly_review(db_path=db, today=date(2026, 9, 11))
        assert [a.text for a in second.carried_actions] == [a.text for a in first.actions]
        assert all(a.origin == "carryover" and a.status == "pending" for a in second.carried_actions)
        assert [a.id for a in second.carried_actions] == [a.id for a in first.actions]
        assert all(a.week_label == "2026-W36" for a in second.carried_actions)

    def test_statuses_are_marked_on_the_next_review(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        first = self._first(monkeypatch, db)
        a, b = first.actions
        second = engine.run_weekly_review(
            db_path=db, today=date(2026, 9, 11), carried_statuses={a.id: "done", b.id: "dropped"}
        )
        assert [c.status for c in second.carried_actions] == ["done", "dropped"]
        # the first review is an append-only record — untouched
        with WeeklyReviewStore(db) as store:
            assert store.get_recent_reports()[1].actions == first.actions

    def test_open_carried_actions_survive_a_third_week(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        first = self._first(monkeypatch, db)
        a, b = first.actions
        _patch_llm(monkeypatch, json.dumps({"summary": "w2", "actions": ["Book the keys on Monday", "New thing"]}))
        second = engine.run_weekly_review(
            db_path=db, today=date(2026, 9, 11), carried_statuses={a.id: "done", b.id: "carried"}
        )
        third = engine.run_weekly_review(db_path=db, today=date(2026, 9, 18))
        texts = [c.text for c in third.carried_actions]
        # second's new actions first, then what it kept open — deduplicated by text; the done one is gone
        assert texts == ["Book the keys on Monday", "New thing"]
        assert a.text not in texts
        assert second.carried_actions[1].status == "carried"

    def test_unknown_id_or_status_is_a_warning(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        first = self._first(monkeypatch, db)
        a = first.actions[0]
        second = engine.run_weekly_review(
            db_path=db, today=date(2026, 9, 11), carried_statuses={"nope": "done", a.id: "finished"}
        )
        assert second.carried_actions[0].status == "pending"
        assert "no carried action with id 'nope'" in second.warnings
        assert f"unknown status 'finished' for action {a.id!r}" in second.warnings

    def test_carried_actions_reader(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        assert engine.carried_actions(db_path=db) == ()
        first = self._first(monkeypatch, db)
        carried = engine.carried_actions(db_path=db)
        assert [c.text for c in carried] == [a.text for a in first.actions]

    def test_missing_db_reads_nothing(self, tmp_path):
        assert engine.carried_actions(db_path=tmp_path / "none.db") == ()

    def test_a_same_week_rerun_does_not_carry_its_own_draft(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        first = self._first(monkeypatch, db)
        again = engine.run_weekly_review(db_path=db, today=TODAY)
        assert again.week_label == first.week_label
        assert again.carried_actions == ()
        # the reader: skip this week's drafts; without a week the newest wins
        assert engine.carried_actions(db_path=db, week_label=first.week_label) == ()
        assert [a.text for a in engine.carried_actions(db_path=db)] == [a.text for a in again.actions]
        # the next week carries from the newest draft of the earlier week
        following = engine.run_weekly_review(db_path=db, today=date(2026, 9, 11))
        assert [a.text for a in following.carried_actions] == [a.text for a in again.actions]


class TestApplyStatuses:
    def test_no_statuses_is_identity(self):
        carried = (ReviewAction(id="a", text="x"),)
        assert engine._apply_statuses(carried, None, []) is carried

    def test_prompt_splits_open_and_done(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        _patch_activity(monkeypatch)
        _patch_llm(monkeypatch, _GOOD)
        first = engine.run_weekly_review(db_path=db, today=TODAY)
        a, b = first.actions
        captured = {}

        def fake_prompt(**kw):
            captured.update(kw)
            return "prompt"

        monkeypatch.setattr("yeaboi.prompts.weekly_review.get_weekly_review_prompt", fake_prompt)
        engine.run_weekly_review(db_path=db, today=date(2026, 9, 11), carried_statuses={a.id: "done"})
        assert captured["carried_done"] == [a.text]
        assert captured["carried_open"] == [b.text]
        assert captured["week_label"] == "2026-W37"


class TestArtifactContract:
    def test_two_public_entry_points(self):
        public = [
            n
            for n in dir(engine)
            if not n.startswith("_")
            and callable(getattr(engine, n))
            and getattr(engine, n).__module__ == engine.__name__
        ]
        assert sorted(public) == ["carried_actions", "run_weekly_review"]

    def test_review_is_frozen(self):
        review = WeeklyReview(week_label="x")
        with pytest.raises(Exception):
            review.week_label = "y"  # type: ignore[misc]


class TestContextScope:
    """What a review may read, and how it is labelled."""

    def test_incognito_reads_nothing_and_says_so(self, monkeypatch, tmp_path):
        from yeaboi.context.labels import LabelStore

        db = _seed(tmp_path)
        monkeypatch.setattr("yeaboi.config.get_last_context_scope", lambda mode: None)
        phases: list = []
        review = engine.run_weekly_review(
            db_path=db,
            today=TODAY,
            dry_run=True,
            context="none",
            project_label="Apollo",
            tags=["Q3"],
            on_progress=lambda e: phases.append(e["component_id"]),
        )
        assert phases[0] == "scope"
        assert sum("switched off" in w for w in review.warnings) == 2
        with LabelStore(db) as labels:
            rows = labels.list_labels(mode="review")
        assert rows and rows[0].project == "Apollo"
        assert {"q3", "mode:review", "world:solo"} <= set(rows[0].tags)
        assert any(tag.startswith("week:") for tag in rows[0].tags)
        assert rows[0].scope == {
            "sources": [],
            "window": {"kind": "all"},
            "projects": [],
            "tags": [],
            "limits": {},
            "sessions": [],
        }

    def test_an_unscoped_review_reads_as_before(self, monkeypatch, tmp_path):
        db = _seed(tmp_path)
        monkeypatch.setattr("yeaboi.config.get_last_context_scope", lambda mode: None)
        review = engine.run_weekly_review(db_path=db, today=TODAY, dry_run=True)
        assert not any("switched off" in w for w in review.warnings)

    def test_the_carried_reader_honours_the_selection(self, tmp_path):
        from yeaboi.context.resolve import Selection
        from yeaboi.context.scope import ContextScope

        db = _seed(tmp_path)
        off = Selection(scope=ContextScope(sources=frozenset({"plan"})), by_source={})
        assert engine.carried_actions(db_path=db, selection=off) == ()
