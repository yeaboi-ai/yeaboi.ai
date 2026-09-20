"""Tests for src/yeaboi/context/scope.py — the scope vocabulary, its JSON twin and the spec grammar."""

from __future__ import annotations

import pytest

from yeaboi.context.scope import (
    SOURCES,
    WINDOW_KINDS,
    ContextScope,
    Window,
    coerce_scope,
    incognito,
    parse_context_spec,
    wants,
)


class TestWindow:
    def test_default_is_unbounded(self):
        assert Window().kind == "all"
        assert not Window().bounded
        assert Window().label() == "everything"

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown window kind"):
            Window(kind="fortnight")

    def test_bad_date_raises(self):
        with pytest.raises(ValueError, match="ISO date"):
            Window(kind="custom", start="12/06/2026")

    def test_negative_count_raises(self):
        with pytest.raises(ValueError, match="negative"):
            Window(kind="sprints", count=-1)

    @pytest.mark.parametrize(
        ("window", "label"),
        [
            (Window(kind="sprints", count=1), "last sprint"),
            (Window(kind="sprints", count=2), "last 2 sprints"),
            (Window(kind="month"), "last month"),
            (Window(kind="custom", start="2026-06-01", end="2026-08-31"), "2026-06-01 to 2026-08-31"),
            (Window(kind="custom", start="2026-06-01"), "2026-06-01 to today"),
        ],
    )
    def test_labels(self, window, label):
        assert window.label() == label

    def test_dict_round_trip(self):
        for window in (Window(), Window(kind="sprints", count=3), Window(kind="custom", start="2026-01-01", end="")):
            assert Window.from_dict(window.to_dict()) == window

    def test_from_dict_is_tolerant(self):
        assert Window.from_dict(None) == Window()
        assert Window.from_dict({"kind": "fortnight"}) == Window()
        assert Window.from_dict({"kind": "custom", "start": "junk"}) == Window()
        assert Window.from_dict({"kind": "sprints", "count": "x"}) == Window(kind="sprints", count=0)


class TestContextScope:
    def test_default_narrows_nothing(self):
        scope = ContextScope()
        assert scope.sources is None and not scope.narrows and not scope.incognito
        assert all(scope.wants(s) for s in SOURCES)

    def test_incognito_is_the_empty_set(self):
        scope = ContextScope(sources=frozenset())
        assert scope.incognito and scope.narrows
        assert not scope.wants("standup")

    def test_module_helpers_treat_none_as_unscoped(self):
        assert wants(None, "retro") is True
        assert incognito(None) is False
        assert wants(ContextScope(sources=frozenset({"plan"})), "retro") is False
        assert incognito(ContextScope(sources=frozenset())) is True

    def test_limit_for(self):
        scope = ContextScope(limits=(("retro", 1),))
        assert scope.limit_for("retro") == 1 and scope.limit_for("standup") == 0

    def test_dict_round_trip(self):
        scope = ContextScope(
            sources=frozenset({"standup", "retro"}),
            window=Window(kind="sprints", count=2),
            projects=("apollo",),
            tags=("q3", "team-a"),
            limits=(("retro", 1),),
        )
        assert ContextScope.from_dict(scope.to_dict()) == scope
        assert scope.to_dict()["sources"] == ["standup", "retro"]

    def test_from_dict_drops_unknown_sources(self):
        scope = ContextScope.from_dict({"sources": ["standup", "stanup"]})
        assert scope.sources == frozenset({"standup"})

    def test_all_unknown_sources_read_as_all_on_not_incognito(self):
        assert ContextScope.from_dict({"sources": ["stanup"]}).sources is None
        assert ContextScope.from_dict({"sources": []}).incognito

    def test_from_dict_ignores_junk(self):
        scope = ContextScope.from_dict({"sources": "standup", "limits": {"retro": "x", "nope": 2}, "extra": 1})
        assert scope == ContextScope()
        assert ContextScope.from_dict(None) == ContextScope()


class TestSpecGrammar:
    def test_inherit_forms_are_none(self):
        assert parse_context_spec("") is None
        assert parse_context_spec("  inherit ") is None

    def test_none_is_incognito(self):
        assert parse_context_spec("none") == ContextScope(sources=frozenset())

    def test_all(self):
        assert parse_context_spec("all") == ContextScope()

    def test_the_owners_example(self):
        scope = parse_context_spec("standup,retro:1@2sprints")
        assert scope.sources == frozenset({"standup", "retro"})
        assert scope.window == Window(kind="sprints", count=2)
        assert scope.limits == (("retro", 1),)

    def test_all_with_window_project_and_tags(self):
        scope = parse_context_spec("all@quarter project=apollo tags=team-a,q3")
        assert scope.sources is None
        assert scope.window == Window(kind="quarter")
        assert scope.projects == ("apollo",) and scope.tags == ("team-a", "q3")

    def test_window_clause_and_custom_range(self):
        scope = parse_context_spec("plan,standup window=2026-06-01..2026-08-31")
        assert scope.window == Window(kind="custom", start="2026-06-01", end="2026-08-31")
        assert parse_context_spec("standup@2026-08-01").window == Window(kind="custom", start="2026-08-01")
        assert parse_context_spec("standup@2026-08-01..").window == Window(kind="custom", start="2026-08-01")

    def test_quoted_project_label(self):
        assert parse_context_spec('project="Apollo Two"').projects == ("Apollo Two",)

    def test_single_sprint_forms(self):
        assert parse_context_spec("all@sprint").window == Window(kind="sprints", count=1)
        assert parse_context_spec("all@1sprint").window == Window(kind="sprints", count=1)

    def test_typo_raises_with_the_valid_list(self):
        with pytest.raises(ValueError, match="unknown context source 'stanup'.*standup"):
            parse_context_spec("stanup")

    def test_bad_window_raises(self):
        with pytest.raises(ValueError, match="unknown window"):
            parse_context_spec("all@fortnight")
        with pytest.raises(ValueError, match="ISO"):
            parse_context_spec("all@2026-06-01..soon")

    def test_unbalanced_quote_raises(self):
        with pytest.raises(ValueError, match="could not read"):
            parse_context_spec('project="Apollo')

    @pytest.mark.parametrize(
        "spec",
        [
            "all",
            "none",
            "standup,retro:1@2sprints",
            "all@quarter project=apollo tags=team-a,q3",
            "plan,standup@2026-06-01..2026-08-31",
            'plan project="Apollo Two"',
        ],
    )
    def test_spec_round_trips(self, spec):
        scope = parse_context_spec(spec)
        assert parse_context_spec(scope.to_spec()) == scope


class TestPinnedSessions:
    """``sessions`` pins runs by name: always read, whatever the window or the labels say."""

    def test_sessions_round_trip_dict_and_spec(self):
        from yeaboi.context.scope import SessionRef

        scope = ContextScope(
            sources=frozenset({"standup"}),
            sessions=(SessionRef("planning", "new-1"), SessionRef("performance", "", "1on1:12")),
        )
        assert ContextScope.from_dict(scope.to_dict()) == scope
        assert scope.to_dict()["sessions"] == [
            {"mode": "planning", "session_id": "new-1", "run_id": ""},
            {"mode": "performance", "session_id": "", "run_id": "1on1:12"},
        ]
        assert scope.to_spec() == "standup session=planning:new-1,performance::1on1:12"
        assert parse_context_spec(scope.to_spec()) == scope

    def test_a_pin_makes_the_source_wanted(self):
        from yeaboi.context.scope import SessionRef

        scope = ContextScope(sources=frozenset(), sessions=(SessionRef("retro", "p1", "3"),))
        assert scope.wants("retro") and not scope.wants("standup")
        assert scope.pinned("retro") == ("3",) and scope.pinned("standup") == ()
        assert not scope.incognito and scope.narrows
        assert parse_context_spec(scope.to_spec()) == scope

    def test_pins_survive_from_dict_junk(self):
        scope = ContextScope.from_dict(
            {
                "sessions": [
                    {"mode": "nope", "session_id": "x"},
                    {"mode": "standup", "run_id": "4"},
                    "garbage",
                    {"mode": "standup", "run_id": "4"},
                    {"mode": "planning"},
                ]
            }
        )
        assert [ref.to_dict() for ref in scope.sessions] == [{"mode": "standup", "session_id": "", "run_id": "4"}]
        assert ContextScope.from_dict({"sessions": "x"}).sessions == ()

    def test_pin_sessions_merges_without_duplicates(self):
        from yeaboi.context.scope import SessionRef, pin_sessions

        pinned = pin_sessions(None, [SessionRef("planning", "a")])
        again = pin_sessions(pinned, [SessionRef("planning", "a"), SessionRef("standup", "", "2")])
        assert [r.key for r in again.sessions] == ["a", "2"]
        assert pinned.sources is None

    def test_a_bad_pin_spec_raises(self):
        from yeaboi.context.scope import SessionRef

        with pytest.raises(ValueError, match="mode:session_id"):
            SessionRef.from_spec("planning")
        with pytest.raises(ValueError, match="unknown session mode"):
            parse_context_spec("all session=ship:x")


class TestCoerce:
    def test_accepts_every_twin(self):
        scope = ContextScope(sources=frozenset({"plan"}))
        assert coerce_scope(None) is None
        assert coerce_scope(scope) is scope
        assert coerce_scope(scope.to_dict()) == scope
        assert coerce_scope("plan") == scope

    def test_rejects_other_types(self):
        with pytest.raises(TypeError):
            coerce_scope(42)

    def test_bad_spec_raises(self):
        with pytest.raises(ValueError):
            coerce_scope("stanup")


def test_vocabulary_is_pinned():
    assert SOURCES == ("plan", "standup", "retro", "poker", "performance", "analysis", "reporting", "review")
    assert WINDOW_KINDS == ("all", "sprints", "month", "quarter", "year", "custom")


class TestCoerceJsonString:
    def test_a_json_string_is_the_dict_twin(self):
        from yeaboi.context.scope import ContextScope, coerce_scope

        scope = coerce_scope('{"sources": ["standup"], "window": {"kind": "month"}}')
        assert scope == ContextScope(sources=frozenset({"standup"}), window=Window(kind="month"))

    def test_broken_json_is_refused_not_read_as_a_spec(self):
        from yeaboi.context.scope import coerce_scope

        with pytest.raises(ValueError, match="JSON"):
            coerce_scope("{not json")


class TestSpecKeepsCapsUnderAll:
    def test_all_with_caps_round_trips(self):
        scope = parse_context_spec("all standup:5")
        assert scope.sources is None and scope.limits == (("standup", 5),)
        assert scope.to_spec() == "all standup:5"
        assert parse_context_spec(scope.to_spec()) == scope

    def test_all_with_window_and_caps_round_trips(self):
        scope = ContextScope(window=Window(kind="sprints", count=2), limits=(("standup", 5), ("retro", 1)))
        spec = scope.to_spec()
        assert spec == "all@2sprints standup:5,retro:1"
        assert parse_context_spec(spec) == scope

    def test_dict_twin_with_all_and_caps_matches_the_spec_twin(self):
        scope = ContextScope.from_dict({"sources": None, "limits": {"retro": 1}})
        assert parse_context_spec(scope.to_spec()) == scope
