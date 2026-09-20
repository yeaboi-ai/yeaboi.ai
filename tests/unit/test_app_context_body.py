"""Tests for src/yeaboi/app/_context_body.py — the three context keys every run body may carry."""

from __future__ import annotations

import pytest

from yeaboi.app._context_body import MAX_INTEGRATIONS, MAX_PROJECT_LABEL, MAX_TAGS, read_context, read_integrations
from yeaboi.app.router import HTTPError
from yeaboi.context.scope import ContextScope


class TestReadContext:
    def test_an_empty_body_reads_as_today(self):
        assert read_context({}) == (None, "", ())
        assert read_context({"context": "", "project_label": "", "tags": []}) == (None, "", ())

    def test_a_spec_string_and_a_dict_both_parse(self):
        scope, _label, _tags = read_context({"context": "standup,retro:1@2sprints"})
        assert isinstance(scope, ContextScope) and scope.wants("standup") and not scope.wants("plan")
        scope, _label, _tags = read_context({"context": {"sources": ["retro"], "window": {"kind": "month"}}})
        assert scope.sources == frozenset({"retro"}) and scope.window.kind == "month"

    def test_a_typo_is_a_400_naming_the_valid_sources(self):
        with pytest.raises(HTTPError) as exc:
            read_context({"context": "stanup"})
        assert exc.value.code == 400 and "standup" in exc.value.message

    def test_labels_and_tags_are_normalised(self):
        _scope, label, tags = read_context({"project_label": "  Apollo   Two ", "tags": ["Team A", "q3", "q3", ""]})
        assert label == "Apollo Two" and tags == ("team-a", "q3")

    @pytest.mark.parametrize(
        "payload",
        [
            {"project_label": 3},
            {"project_label": "x" * (MAX_PROJECT_LABEL + 1)},
            {"tags": "q3"},
            {"tags": [1]},
            {"tags": ["t"] * (MAX_TAGS + 1)},
            {"tags": ["x" * 41]},
        ],
    )
    def test_malformed_labels_are_400s(self, payload):
        with pytest.raises(HTTPError) as exc:
            read_context(payload)
        assert exc.value.code == 400


class TestReadIntegrations:
    def test_absent_and_null_read_as_unrestricted(self):
        assert read_integrations({}) is None
        assert read_integrations({"integrations": None}) is None

    def test_a_list_is_kept_normalised_and_deduped(self):
        assert read_integrations({"integrations": [" Jira", "github", "jira"]}) == ["jira", "github"]
        assert read_integrations({"integrations": []}) == []

    def test_an_unknown_key_is_a_400_naming_the_known_ones(self):
        with pytest.raises(HTTPError) as exc:
            read_integrations({"integrations": ["fax"]})
        assert exc.value.code == 400 and "fax" in exc.value.message and "jira" in exc.value.message

    @pytest.mark.parametrize("raw", ["jira", [1], {"jira": True}])
    def test_a_malformed_list_is_a_400(self, raw):
        with pytest.raises(HTTPError) as exc:
            read_integrations({"integrations": raw})
        assert exc.value.code == 400

    def test_too_many_is_a_400(self):
        with pytest.raises(HTTPError) as exc:
            read_integrations({"integrations": [f"k{n}" for n in range(MAX_INTEGRATIONS + 1)]})
        assert exc.value.code == 400
