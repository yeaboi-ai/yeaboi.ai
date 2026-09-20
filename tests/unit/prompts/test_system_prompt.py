"""Tests for the Scrum Master system prompt."""

from yeaboi.prompts.system import get_system_prompt

# ── Content tests ─────────────────────────────────────────────────────


class TestGetSystemPrompt:
    """Verify the system prompt contains all required scrum concepts."""

    def test_returns_nonempty_string(self):
        result = get_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_persona(self):
        prompt = get_system_prompt()
        assert "Scrum Master" in prompt

    def test_contains_story_format(self):
        prompt = get_system_prompt()
        assert "As a" in prompt
        assert "I want to" in prompt
        assert "so that" in prompt

    def test_contains_acceptance_criteria_format(self):
        prompt = get_system_prompt()
        assert "Given" in prompt
        assert "When" in prompt
        assert "Then" in prompt

    def test_contains_fibonacci_scale(self):
        prompt = get_system_prompt()
        assert "1, 2, 3, 5, 8" in prompt

    def test_contains_eight_point_maximum(self):
        prompt = get_system_prompt()
        assert "8 points" in prompt
        assert "split" in prompt.lower()

    def test_contains_issue_hierarchy(self):
        prompt = get_system_prompt()
        assert "Feature" in prompt
        assert "User Story" in prompt
        assert "Sub-Task" in prompt
        assert "Spike" in prompt

    def test_contains_sprint_capacity_rule(self):
        prompt = get_system_prompt()
        assert "sprint" in prompt.lower()
        assert "capacity" in prompt.lower()

    def test_contains_default_velocity(self):
        prompt = get_system_prompt()
        assert "5" in prompt
        assert "per engineer" in prompt.lower()

    def test_contains_guardrail_behaviours(self):
        prompt = get_system_prompt()
        lower = prompt.lower()
        assert "push back" in lower
        assert "scope creep" in lower

    def test_contains_stay_on_topic(self):
        prompt = get_system_prompt()
        lower = prompt.lower()
        assert "stay on topic" in lower
        assert "off-topic" in lower


# ── Structure tests ───────────────────────────────────────────────────


class TestSystemPromptStructure:
    """Verify structural properties of the prompt factory."""

    def test_deterministic(self):
        """Two calls must return the exact same string."""
        assert get_system_prompt() == get_system_prompt()

    def test_importable_from_package(self):
        """get_system_prompt should be re-exported from yeaboi.prompts."""
        from yeaboi.prompts import get_system_prompt as imported_fn

        assert imported_fn() == get_system_prompt()


class TestIntegrationsLine:
    def test_default_is_unchanged(self):
        assert "Integrations enabled" not in get_system_prompt()
        assert get_system_prompt(None) == get_system_prompt()

    def test_a_list_names_the_enabled_set(self):
        prompt = get_system_prompt(["jira", "github"])
        assert prompt.startswith(get_system_prompt())
        assert "Integrations enabled for this plan: jira, github." in prompt

    def test_an_empty_list_says_none(self):
        assert "Integrations enabled for this plan: none." in get_system_prompt([])
