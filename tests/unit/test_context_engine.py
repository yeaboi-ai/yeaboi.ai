"""Tests for src/yeaboi/context/engine.py — the entry points every surface calls."""

from __future__ import annotations

from datetime import date

import pytest

from yeaboi.context import engine
from yeaboi.context.labels import LabelStore
from yeaboi.context.scope import ContextScope

TODAY = date(2026, 9, 11)


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "sessions.db"
    monkeypatch.setattr("yeaboi.paths.get_db_path", lambda: path)
    monkeypatch.delenv("YEABOI_CONTEXT_STANDUP", raising=False)
    monkeypatch.setattr("yeaboi.context.window._from_tracker", lambda: None)
    return path


@pytest.fixture
def labelled(db):
    from yeaboi.agent.state import StandupReport
    from yeaboi.standup.store import StandupStore

    with StandupStore(db) as store:
        run = store.record_run(StandupReport(session_id="p1", date="2026-09-03"))
    with LabelStore(db) as labels:
        labels.set_labels("standup", "p1", str(run), project="Apollo", tags=["q3", "mode:standup"])
    return str(run)


class TestReExports:
    def test_parse_context_spec_is_the_grammar(self):
        scope = engine.parse_context_spec("standup,retro:1@2sprints project=apollo tags=q3")
        assert scope.sources == frozenset({"standup", "retro"}) and scope.limit_for("retro") == 1
        assert engine.parse_context_spec("") is None

    def test_parse_context_spec_names_the_valid_sources_on_a_typo(self):
        with pytest.raises(ValueError, match="standup"):
            engine.parse_context_spec("stanup")

    def test_resolve_scope_reads_nothing_for_none(self, db):
        selection = engine.resolve_scope(None, today=TODAY, db_path=db)
        assert selection.scope is None and selection.ids("standup") is None

    def test_preview_scope_counts(self, db, labelled):
        preview = engine.preview_scope("standup@month", mode="planning", today=TODAY, db_path=db)
        assert preview.counts["standup"] == 1 and "1 standup" in preview.label

    def test_label_run_never_raises(self, monkeypatch):
        monkeypatch.setattr("yeaboi.paths.get_db_path", lambda: 1 / 0)
        assert engine.label_run("standup", "p1", 3, project_label="x") is None


class TestContextOptions:
    def test_shape_and_counts(self, db, labelled, monkeypatch):
        monkeypatch.setenv("YEABOI_CONTEXT_STANDUP", '{"sources": ["retro"]}')
        options = engine.context_options(mode="standup", today=TODAY, db_path=db)
        by_key = {row["key"]: row for row in options.sources}
        assert set(by_key) == {"plan", "standup", "retro", "poker", "performance", "analysis", "reporting", "review"}
        assert by_key["standup"] == {
            "key": "standup",
            "label": "Standups",
            "hint": by_key["standup"]["hint"],
            "count": 1,
        }
        assert [w["kind"] for w in options.windows] == ["all", "sprints", "month", "quarter", "year", "custom"]
        assert next(w for w in options.windows if w["kind"] == "sprints")["needs_count"] is True
        assert next(w for w in options.windows if w["kind"] == "custom")["needs_range"] is True
        assert options.projects == ("Apollo",)
        assert {t["tag"] for t in options.tags} == {"q3", "mode:standup"}
        assert options.calendar["source"] in {"settings", "default"} and options.calendar["current"]["start"]
        assert options.default == {"sources": ["retro"]}
        assert "mode:standup" in options.defaults["tags"] and "2026-09" in options.defaults["tags"]

    def test_no_mode_means_no_default_and_planning_tags(self, db):
        options = engine.context_options(today=TODAY, db_path=db)
        assert options.default is None and "mode:planning" in options.defaults["tags"]

    def test_unknown_mode_raises(self, db):
        with pytest.raises(ValueError, match="unknown mode"):
            engine.context_options(mode="stanup", today=TODAY, db_path=db)

    def test_missing_database_is_empty_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("yeaboi.context.window._from_tracker", lambda: None)
        options = engine.context_options(today=TODAY, db_path=tmp_path / "none.db")
        assert options.projects == () and options.tags == ()
        assert all(row["count"] == 0 for row in options.sources)


class TestLabels:
    def test_set_get_list_round_trip(self, db):
        row = engine.set_session_labels("retro", "p1", "7", project_label="Apollo", tags=["Q3 Push"], db_path=db)
        assert row.project == "Apollo" and row.tags == ("q3-push",)
        assert engine.get_session_labels("retro", "p1", "7", db_path=db) == row
        listed = engine.list_session_labels(mode="retro", project_label="Apollo", db_path=db)
        assert [r.run_id for r in listed] == ["7"]
        assert engine.list_session_labels(mode="retro", tags=["nope"], db_path=db) == []

    def test_merge_and_replace_tags(self, db):
        engine.set_session_labels("retro", "p1", "7", tags=["a"], db_path=db)
        assert engine.set_session_labels("retro", "p1", "7", tags=["b"], db_path=db).tags == ("a", "b")
        assert engine.set_session_labels("retro", "p1", "7", tags=["c"], merge_tags=False, db_path=db).tags == ("c",)

    def test_unknown_mode_raises_on_every_read_and_write(self, db):
        with pytest.raises(ValueError):
            engine.set_session_labels("nope", "p1", db_path=db)
        with pytest.raises(ValueError):
            engine.get_session_labels("nope", "p1", db_path=db)
        with pytest.raises(ValueError):
            engine.list_session_labels(mode="nope", db_path=db)

    def test_missing_rows_and_missing_database(self, db, tmp_path):
        assert engine.get_session_labels("retro", "p1", db_path=db) is None
        assert engine.get_session_labels("retro", "p1", db_path=tmp_path / "none.db") is None
        assert engine.list_session_labels(db_path=tmp_path / "none.db") == []

    def test_scope_survives_a_label_write(self, db):
        engine.label_run("standup", "p1", 3, scope=ContextScope(sources=frozenset({"retro"})), db_path=db, today=TODAY)
        row = engine.set_session_labels("standup", "p1", "3", project_label="Apollo", db_path=db)
        assert row.scope == {
            "sources": ["retro"],
            "window": {"kind": "all"},
            "projects": [],
            "tags": [],
            "limits": {},
            "sessions": [],
        }
