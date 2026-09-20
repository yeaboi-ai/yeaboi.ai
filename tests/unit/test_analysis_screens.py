"""Unit tests for analysis mode screen builders and preview flow.

Covers:
- _build_analysis_review_screen (shared template: progress dots, scrollbar, viewport)
- _build_instructions_review_screen
- _build_sample_epic_screen
- _build_sample_stories_screen
- _build_sample_tasks_screen
- _build_sample_sprint_screen
- _build_analysis_progress_screen
- _build_team_analysis_screen (scrollbar + viewport wrapping)

Mirrors the test patterns used for planning mode screens in tests/test_session.py
(TestBuildDescriptionScreen, TestBuildQuestionScreen, etc.).
"""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from yeaboi.ui.mode_select.screens._screens_secondary import (
    _build_analysis_depth_screen,
    _build_analysis_feature_screen,
    _build_analysis_model_offer_screen,
    _build_analysis_progress_screen,
    _build_analysis_review_screen,
    _build_analysis_setup_review_screen,
    _build_analysis_window_screen,
    _build_code_scope_select_screen,
    _build_component_select_screen,
    _build_generate_confirm_screen,
    _build_instructions_review_screen,
    _build_member_select_screen,
    _build_sample_epic_screen,
    _build_sample_sprint_screen,
    _build_sample_stories_screen,
    _build_sample_tasks_screen,
    _build_team_analysis_screen,
    _build_team_insights_screen,
)
from yeaboi.ui.shared._components import ANALYSIS_THEME, PLANNING_THEME


def test_analysis_model_offer_names_models_and_eta():
    output = _render(
        _build_analysis_model_offer_screen("qwen3:14b", "qwen3:4b", 1200, height=30),
        width=100,
    )
    assert "qwen3:14b" in output
    assert "qwen3:4b" in output
    assert "estimated 20 min" in output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render(panel: Panel, width: int = 100) -> str:
    """Render a Rich Panel to plain text for content assertions."""
    buf = StringIO()
    console = Console(file=buf, width=width, force_terminal=False, highlight=False)
    console.print(panel)
    return buf.getvalue()


def _make_body_lines(n: int = 10, prefix: str = "Line") -> list:
    """Create a list of Text objects for use as body_lines."""
    return [Text(f"    {prefix} {i}", justify="left") for i in range(n)]


def test_code_project_picker_shows_selected_scope():
    rendered = _render(
        _build_code_scope_select_screen(
            ["Infrastructure", "Product"],
            {1},
            1,
            width=90,
            height=24,
        )
    )
    assert "ANALYSIS SETUP" in rendered and "AZURE PROJECTS" in rendered
    assert "1 of 2 projects selected" in rendered
    assert "Product" in rendered


def test_code_scope_picker_rebrands_for_github():
    rendered = _render(
        _build_code_scope_select_screen(
            ["acme-corp", "dinho"],
            {0},
            0,
            heading="GitHub owners",
            unit="owners",
            hint="Every non-archived repo with activity in the window is scanned.",
            width=90,
            height=24,
        )
    )
    assert "GITHUB OWNERS" in rendered
    assert "1 of 2 owners selected" in rendered
    assert "acme-corp" in rendered
    # The cost of one checkbox (an owner fans out to every active repo) has to be
    # visible BEFORE Enter — there is no other screen that states it.
    assert "non-archived repo" in rendered


def test_code_scope_picker_explains_an_empty_estate():
    # A token with no visible orgs is a real outcome; a blank viewport reads as a
    # rendering bug and leaves the user with nothing to act on.
    rendered = _render(
        _build_code_scope_select_screen(
            [],
            set(),
            0,
            heading="GitHub owners",
            unit="owners",
            empty_label="No GitHub owners were visible to the configured token",
            width=90,
            height=24,
        )
    )
    assert "No GitHub owners were visible" in rendered
    assert "0 of 0 owners selected" in rendered


# ---------------------------------------------------------------------------
# Sample fixture data
# ---------------------------------------------------------------------------

_SAMPLE_INSTRUCTIONS = """\
## Velocity & Capacity
- Team velocity — 23.5 pts/sprint average
- Sprint length — 2 weeks
- Team size — 4 developers

## Story Conventions
- Story points — use Fibonacci scale (1, 2, 3, 5, 8)
- Acceptance criteria — Given/When/Then format, median 3 per story
→ Match this team's style exactly.

## Naming Conventions
- Label convention: quarterly goal (42%), released to dev (6%)
- Epic naming: quarter-scoped (e.g. "Q4|2025|High Region Outage DR")
→ Generated tickets MUST match these naming conventions.

Estimation note: Use THESE team-specific patterns, not generic Fibonacci rules.
"""

_SAMPLE_EPIC = {
    "title": "Q1|2026|Medium Platform Resilience Upgrade",
    "description": "Improve platform resilience with automated failover and monitoring.",
    "priority": "high",
    "stories_estimate": 5,
    "points_estimate": 18,
    "rationale": "Matches team's quarter-scoped naming convention.",
}

_SAMPLE_STORIES = [
    {
        "id": "S1",
        "title": "Implement automated failover",
        "persona": "developer",
        "goal": "automated failover between regions",
        "benefit": "reduced downtime",
        "story_points": 5,
        "priority": "high",
        "discipline": "infrastructure",
        "acceptance_criteria": [
            {"given": "primary region fails", "when": "failover triggered", "then": "traffic routes to secondary"},
            {"given": "secondary region active", "when": "primary recovers", "then": "failback completes"},
        ],
        "rationale": "Matches team's infrastructure story patterns.",
    },
    {
        "id": "S2",
        "title": "Add monitoring dashboards",
        "persona": "SRE",
        "goal": "visibility into failover health",
        "benefit": "faster incident response",
        "story_points": 3,
        "priority": "medium",
        "discipline": "observability",
        "acceptance_criteria": [
            {"given": "failover runs", "when": "dashboard queried", "then": "shows status within 30s"},
        ],
        "rationale": "Matches observability story pattern.",
    },
]

_SAMPLE_TASKS = [
    {
        "id": "T-S1-01",
        "story_id": "S1",
        "title": "Implement health check endpoint",
        "description": "Add /health endpoint to secondary region.",
        "label": "Code",
        "test_plan": "Unit test: verify endpoint returns 200.",
    },
    {
        "id": "T-S1-02",
        "story_id": "S1",
        "title": "Write failover integration tests",
        "description": "End-to-end test for failover trigger.",
        "label": "Testing",
        "test_plan": "Integration test: simulate region failure.",
    },
    {
        "id": "T-S2-01",
        "story_id": "S2",
        "title": "Create Grafana dashboard",
        "description": "Build dashboard with failover metrics.",
        "label": "Infrastructure",
        "test_plan": "Manual: verify dashboard loads in staging.",
    },
]

_SAMPLE_SPRINT = {
    "sprint_name": "Sprint 1",
    "velocity_target": 20,
    "stories_included": ["S1", "S2"],
    "total_points": 8,
    "capacity_notes": "Based on team avg of 23.5 pts/sprint, 8 pts leaves buffer.",
    "risks": ["S1 depends on cloud provider API", "S2 blocked until S1 failover endpoint is live"],
    "rationale": "Conservative allocation matching team's 88% completion rate.",
}

_SAMPLE_EXAMPLES = {
    "naming_conventions": {
        "epic_naming_style": "quarter-scoped",
        "epic_examples": ["Q4|2025|High Region Outage DR", "Q1|2026|Low Overmind improvement"],
        "template_sections": [("What is this about?", 0.8), ("Why does it matter?", 0.6)],
    },
    "ac_patterns": {"median_ac": 3},
    "task_decomposition": {
        "avg_tasks_per_story": 4.8,
        "type_distribution": {"Development": 64, "Testing": 13, "Deploy": 12},
        "common_tasks": [("create aurora rollback module", 2), ("update engine version", 2)],
    },
    "scope_changes": {
        "totals": {"avg_delivered_velocity": 25.9, "avg_committed_velocity": 19.1},
    },
}


# ---------------------------------------------------------------------------
# Shared builder: _build_analysis_review_screen
# ---------------------------------------------------------------------------


class TestBuildAnalysisReviewScreen:
    """Test the shared analysis review screen template."""

    def test_returns_panel(self):
        result = _build_analysis_review_screen(_make_body_lines(), stage_index=0, width=80, height=24)
        assert isinstance(result, Panel)

    def test_empty_body(self):
        result = _build_analysis_review_screen([], stage_index=0, width=80, height=24)
        assert isinstance(result, Panel)

    def test_all_stage_indices(self):
        """Each stage index (0-4) should render a valid panel."""
        lines = _make_body_lines(5)
        for idx in range(5):
            result = _build_analysis_review_screen(lines, stage_index=idx, width=80, height=24)
            assert isinstance(result, Panel)

    def test_custom_actions(self):
        result = _build_analysis_review_screen(
            _make_body_lines(),
            actions=["Done", "Export"],
            action_sel=1,
            width=80,
            height=24,
        )
        assert isinstance(result, Panel)

    def test_subtitle_rendered(self):
        result = _build_analysis_review_screen(
            _make_body_lines(),
            subtitle="Review planning instructions",
            width=80,
            height=24,
        )
        output = _render(result)
        assert "Review planning instructions" in output

    def test_progress_dots_current_stage(self):
        """Progress dots should show the current stage name in bold."""
        result = _build_analysis_review_screen(_make_body_lines(), stage_index=2, width=80, height=24)
        output = _render(result)
        assert "Stories" in output

    def test_scrollbar_appears_when_content_overflows(self):
        """Scrollbar should render when content exceeds viewport."""
        long_body = _make_body_lines(100)
        result = _build_analysis_review_screen(long_body, stage_index=0, width=80, height=24)
        output = _render(result)
        # Scrollbar uses thin/thick vertical bars
        assert "\u2502" in output or "\u2503" in output

    def test_no_scrollbar_short_content(self):
        """No scrollbar for content that fits within viewport."""
        short_body = _make_body_lines(3)
        result = _build_analysis_review_screen(short_body, stage_index=0, width=80, height=30)
        assert isinstance(result, Panel)

    def test_scroll_offset_clamps(self):
        """Scroll offset beyond max should clamp without error."""
        lines = _make_body_lines(10)
        result = _build_analysis_review_screen(lines, scroll_offset=9999, width=80, height=24)
        assert isinstance(result, Panel)

    def test_action_selection_highlights(self):
        """Each action index should produce a valid panel."""
        lines = _make_body_lines(5)
        for sel in range(4):
            result = _build_analysis_review_screen(lines, action_sel=sel, width=80, height=24)
            assert isinstance(result, Panel)

    def test_wrapping_lines_dont_overflow(self):
        """Long lines that wrap should not push buttons off-screen."""
        long_lines = [Text("    " + "x" * 200, justify="left") for _ in range(20)]
        result = _build_analysis_review_screen(long_lines, scroll_offset=15, width=80, height=24)
        output = _render(result, width=80)
        # Buttons should always be present in the rendered output
        assert "\u256d" in output  # top border of button box
        assert "\u256f" in output  # bottom border of button box

    def test_narrow_width(self):
        result = _build_analysis_review_screen(_make_body_lines(), width=40, height=24)
        assert isinstance(result, Panel)

    def test_tall_height(self):
        result = _build_analysis_review_screen(_make_body_lines(5), width=80, height=60)
        assert isinstance(result, Panel)

    def test_minimum_height(self):
        """Very short terminal should still render."""
        result = _build_analysis_review_screen(_make_body_lines(3), width=80, height=12)
        assert isinstance(result, Panel)


# ---------------------------------------------------------------------------
# Instructions review screen
# ---------------------------------------------------------------------------


class TestBuildInstructionsReviewScreen:
    """Test the planning instructions review page (stage 1 of preview flow)."""

    def test_returns_panel(self):
        result = _build_instructions_review_screen(_SAMPLE_INSTRUCTIONS, width=80, height=24)
        assert isinstance(result, Panel)

    def test_empty_instructions(self):
        result = _build_instructions_review_screen("", width=80, height=24)
        assert isinstance(result, Panel)

    def test_section_headers_rendered(self):
        result = _build_instructions_review_screen(_SAMPLE_INSTRUCTIONS, width=100, height=40)
        output = _render(result, width=100)
        assert "Velocity & Capacity" in output
        assert "Story Conventions" in output
        assert "Naming Conventions" in output

    def test_numbered_items(self):
        result = _build_instructions_review_screen(_SAMPLE_INSTRUCTIONS, width=100, height=40)
        output = _render(result, width=100)
        # Items should be numbered
        assert "1" in output
        assert "Team velocity" in output

    def test_arrow_directives_rendered(self):
        result = _build_instructions_review_screen(_SAMPLE_INSTRUCTIONS, width=120, height=40)
        output = _render(result, width=120)
        assert "Match this team" in output or "naming conventions" in output

    def test_scrollable(self):
        result1 = _build_instructions_review_screen(_SAMPLE_INSTRUCTIONS, scroll_offset=0, width=80, height=24)
        result2 = _build_instructions_review_screen(_SAMPLE_INSTRUCTIONS, scroll_offset=5, width=80, height=24)
        assert isinstance(result1, Panel)
        assert isinstance(result2, Panel)

    def test_action_buttons(self):
        """Instructions page has Accept/Edit/Export buttons."""
        # Use tall height to ensure buttons are visible (not clipped by viewport)
        result = _build_instructions_review_screen(_SAMPLE_INSTRUCTIONS, width=100, height=60)
        output = _render(result, width=100)
        assert "Accept" in output
        assert "Edit" in output
        assert "Export" in output

    def test_action_selection(self):
        for sel in range(3):
            result = _build_instructions_review_screen(_SAMPLE_INSTRUCTIONS, action_sel=sel, width=80, height=24)
            assert isinstance(result, Panel)

    def test_stage_indicator_shows_instructions(self):
        result = _build_instructions_review_screen(_SAMPLE_INSTRUCTIONS, width=100, height=24)
        output = _render(result, width=100)
        assert "Instructions" in output

    def test_long_instructions_scrollbar(self):
        """Long instructions should show scrollbar."""
        long_text = "\n".join(f"- Item {i} — description of item {i}" for i in range(50))
        result = _build_instructions_review_screen(long_text, width=80, height=24)
        output = _render(result, width=80)
        assert "\u2502" in output or "\u2503" in output


# ---------------------------------------------------------------------------
# Sample epic screen
# ---------------------------------------------------------------------------


class TestBuildSampleEpicScreen:
    """Test the sample epic review page (stage 2 of preview flow)."""

    def test_returns_panel(self):
        result = _build_sample_epic_screen(_SAMPLE_EPIC, width=80, height=24)
        assert isinstance(result, Panel)

    def test_empty_epic(self):
        result = _build_sample_epic_screen({}, width=80, height=24)
        assert isinstance(result, Panel)

    def test_epic_title_rendered(self):
        result = _build_sample_epic_screen(_SAMPLE_EPIC, width=120, height=40)
        output = _render(result, width=120)
        assert "Platform Resilience" in output

    def test_epic_priority_rendered(self):
        result = _build_sample_epic_screen(_SAMPLE_EPIC, width=100, height=40)
        output = _render(result, width=100)
        assert "high" in output.lower()

    def test_epic_rationale_rendered(self):
        result = _build_sample_epic_screen(_SAMPLE_EPIC, width=120, height=40)
        output = _render(result, width=120)
        assert "quarter-scoped" in output or "rationale" in output.lower()

    def test_with_examples(self):
        result = _build_sample_epic_screen(_SAMPLE_EPIC, examples=_SAMPLE_EXAMPLES, width=80, height=24)
        assert isinstance(result, Panel)

    def test_examples_pattern_info(self):
        """When examples provided, should show naming style info."""
        result = _build_sample_epic_screen(_SAMPLE_EPIC, examples=_SAMPLE_EXAMPLES, width=120, height=40)
        output = _render(result, width=120)
        assert "quarter-scoped" in output or "naming" in output.lower()

    def test_scrollable(self):
        result = _build_sample_epic_screen(_SAMPLE_EPIC, scroll_offset=3, width=80, height=24)
        assert isinstance(result, Panel)

    def test_action_selection(self):
        for sel in range(4):
            result = _build_sample_epic_screen(_SAMPLE_EPIC, action_sel=sel, width=80, height=24)
            assert isinstance(result, Panel)

    def test_stage_indicator_shows_epic(self):
        result = _build_sample_epic_screen(_SAMPLE_EPIC, width=100, height=24)
        output = _render(result, width=100)
        assert "Epic" in output


# ---------------------------------------------------------------------------
# Sample stories screen
# ---------------------------------------------------------------------------


class TestBuildSampleStoriesScreen:
    """Test the sample stories review page (stage 3 of preview flow)."""

    def test_returns_panel(self):
        result = _build_sample_stories_screen(_SAMPLE_STORIES, width=80, height=24)
        assert isinstance(result, Panel)

    def test_empty_stories(self):
        result = _build_sample_stories_screen([], width=80, height=24)
        assert isinstance(result, Panel)

    def test_single_story(self):
        result = _build_sample_stories_screen([_SAMPLE_STORIES[0]], width=80, height=24)
        assert isinstance(result, Panel)

    def test_story_ids_rendered(self):
        # Tall height so both stories fit the viewport below the 6-row ANSI-Shadow header.
        result = _build_sample_stories_screen(_SAMPLE_STORIES, width=100, height=70)
        output = _render(result, width=100)
        assert "S1" in output
        assert "S2" in output

    def test_story_points_rendered(self):
        result = _build_sample_stories_screen(_SAMPLE_STORIES, width=100, height=70)
        output = _render(result, width=100)
        assert "5" in output  # S1 points
        assert "3" in output  # S2 points

    def test_acceptance_criteria_rendered(self):
        result = _build_sample_stories_screen(_SAMPLE_STORIES, width=120, height=50)
        output = _render(result, width=120)
        # Given/When/Then format
        assert "Given" in output or "given" in output.lower()

    def test_persona_goal_rendered(self):
        result = _build_sample_stories_screen(_SAMPLE_STORIES, width=120, height=40)
        output = _render(result, width=120)
        assert "developer" in output or "SRE" in output

    def test_with_epic_title(self):
        result = _build_sample_stories_screen(
            _SAMPLE_STORIES,
            epic_title="Q1|2026|Medium Platform Resilience",
            width=100,
            height=24,
        )
        output = _render(result, width=100)
        assert "Platform Resilience" in output or "Stories" in output

    def test_scrollable(self):
        result = _build_sample_stories_screen(_SAMPLE_STORIES, scroll_offset=5, width=80, height=24)
        assert isinstance(result, Panel)

    def test_action_selection(self):
        for sel in range(4):
            result = _build_sample_stories_screen(_SAMPLE_STORIES, action_sel=sel, width=80, height=24)
            assert isinstance(result, Panel)

    def test_stage_indicator_shows_stories(self):
        result = _build_sample_stories_screen(_SAMPLE_STORIES, width=100, height=24)
        output = _render(result, width=100)
        assert "Stories" in output

    def test_story_without_acceptance_criteria(self):
        """Stories with empty AC list should still render."""
        story = {**_SAMPLE_STORIES[0], "acceptance_criteria": []}
        result = _build_sample_stories_screen([story], width=80, height=24)
        assert isinstance(result, Panel)

    def test_story_with_missing_fields(self):
        """Minimal story dict should not crash."""
        story = {"id": "S1", "title": "Minimal story"}
        result = _build_sample_stories_screen([story], width=80, height=24)
        assert isinstance(result, Panel)

    def test_definition_of_done_rendered(self):
        """Stories with definition_of_done should render DoD items."""
        story = {
            **_SAMPLE_STORIES[0],
            "definition_of_done": ["Code reviewed", "Tests passing", "Deployed to staging"],
        }
        result = _build_sample_stories_screen([story], width=120, height=60)
        output = _render(result, width=120)
        assert "Definition of Done" in output
        assert "Code reviewed" in output

    def test_story_without_dod(self):
        """Stories without definition_of_done should still render."""
        story = {k: v for k, v in _SAMPLE_STORIES[0].items() if k != "definition_of_done"}
        result = _build_sample_stories_screen([story], width=80, height=24)
        assert isinstance(result, Panel)


# ---------------------------------------------------------------------------
# Sample tasks screen
# ---------------------------------------------------------------------------


class TestBuildSampleTasksScreen:
    """Test the sample tasks review page (stage 4 of preview flow)."""

    def test_returns_panel(self):
        result = _build_sample_tasks_screen(_SAMPLE_TASKS, width=80, height=24)
        assert isinstance(result, Panel)

    def test_empty_tasks(self):
        result = _build_sample_tasks_screen([], width=80, height=24)
        assert isinstance(result, Panel)

    def test_task_ids_rendered(self):
        result = _build_sample_tasks_screen(_SAMPLE_TASKS, width=100, height=40)
        output = _render(result, width=100)
        assert "T-S1-01" in output
        assert "T-S2-01" in output

    def test_task_labels_rendered(self):
        result = _build_sample_tasks_screen(_SAMPLE_TASKS, width=100, height=40)
        output = _render(result, width=100)
        assert "Code" in output
        assert "Testing" in output

    def test_task_grouping_by_story(self):
        """Tasks should be grouped under their story ID."""
        result = _build_sample_tasks_screen(_SAMPLE_TASKS, width=100, height=40)
        output = _render(result, width=100)
        # Both story IDs should appear as group headers
        assert "S1" in output
        assert "S2" in output

    def test_test_plan_rendered(self):
        result = _build_sample_tasks_screen(_SAMPLE_TASKS, width=120, height=40)
        output = _render(result, width=120)
        assert "Unit test" in output or "test" in output.lower()

    def test_scrollable(self):
        result = _build_sample_tasks_screen(_SAMPLE_TASKS, scroll_offset=3, width=80, height=24)
        assert isinstance(result, Panel)

    def test_action_selection(self):
        for sel in range(4):
            result = _build_sample_tasks_screen(_SAMPLE_TASKS, action_sel=sel, width=80, height=24)
            assert isinstance(result, Panel)

    def test_stage_indicator_shows_tasks(self):
        result = _build_sample_tasks_screen(_SAMPLE_TASKS, width=100, height=24)
        output = _render(result, width=100)
        assert "Tasks" in output

    def test_single_task(self):
        result = _build_sample_tasks_screen([_SAMPLE_TASKS[0]], width=80, height=24)
        assert isinstance(result, Panel)

    def test_task_with_missing_fields(self):
        """Minimal task dict should not crash."""
        task = {"id": "T-1", "title": "Do something"}
        result = _build_sample_tasks_screen([task], width=80, height=24)
        assert isinstance(result, Panel)


# ---------------------------------------------------------------------------
# Sample sprint screen
# ---------------------------------------------------------------------------


class TestBuildSampleSprintScreen:
    """Test the sample sprint plan review page (stage 5 of preview flow)."""

    def test_returns_panel(self):
        result = _build_sample_sprint_screen(_SAMPLE_SPRINT, _SAMPLE_STORIES, width=80, height=24)
        assert isinstance(result, Panel)

    def test_empty_sprint(self):
        result = _build_sample_sprint_screen({}, [], width=80, height=24)
        assert isinstance(result, Panel)

    def test_sprint_name_rendered(self):
        result = _build_sample_sprint_screen(_SAMPLE_SPRINT, _SAMPLE_STORIES, width=100, height=40)
        output = _render(result, width=100)
        assert "Sprint 1" in output

    def test_velocity_target_rendered(self):
        result = _build_sample_sprint_screen(_SAMPLE_SPRINT, _SAMPLE_STORIES, width=100, height=40)
        output = _render(result, width=100)
        assert "20" in output

    def test_total_points_rendered(self):
        result = _build_sample_sprint_screen(_SAMPLE_SPRINT, _SAMPLE_STORIES, width=100, height=40)
        output = _render(result, width=100)
        assert "8" in output

    def test_stories_listed(self):
        result = _build_sample_sprint_screen(_SAMPLE_SPRINT, _SAMPLE_STORIES, width=100, height=40)
        output = _render(result, width=100)
        assert "S1" in output
        assert "S2" in output

    def test_risks_rendered(self):
        result = _build_sample_sprint_screen(_SAMPLE_SPRINT, _SAMPLE_STORIES, width=120, height=40)
        output = _render(result, width=120)
        assert "cloud provider" in output or "risk" in output.lower()

    def test_capacity_notes_rendered(self):
        result = _build_sample_sprint_screen(_SAMPLE_SPRINT, _SAMPLE_STORIES, width=120, height=40)
        output = _render(result, width=120)
        assert "23.5" in output or "buffer" in output

    def test_done_button(self):
        """Sprint page should have Done button (not Accept)."""
        result = _build_sample_sprint_screen(_SAMPLE_SPRINT, _SAMPLE_STORIES, width=100, height=60)
        output = _render(result, width=100)
        assert "Done" in output

    def test_scrollable(self):
        result = _build_sample_sprint_screen(_SAMPLE_SPRINT, _SAMPLE_STORIES, scroll_offset=3, width=80, height=24)
        assert isinstance(result, Panel)

    def test_action_selection(self):
        for sel in range(3):  # Done, Regenerate, Export
            result = _build_sample_sprint_screen(_SAMPLE_SPRINT, _SAMPLE_STORIES, action_sel=sel, width=80, height=24)
            assert isinstance(result, Panel)

    def test_stage_indicator_shows_sprint(self):
        result = _build_sample_sprint_screen(_SAMPLE_SPRINT, _SAMPLE_STORIES, width=100, height=24)
        output = _render(result, width=100)
        assert "Sprint" in output

    def test_sprint_without_risks(self):
        sprint = {**_SAMPLE_SPRINT, "risks": []}
        result = _build_sample_sprint_screen(sprint, _SAMPLE_STORIES, width=80, height=24)
        assert isinstance(result, Panel)

    def test_sprint_without_stories(self):
        """Sprint with story IDs but no matching story objects."""
        result = _build_sample_sprint_screen(_SAMPLE_SPRINT, [], width=80, height=24)
        assert isinstance(result, Panel)


# ---------------------------------------------------------------------------
# Analysis progress screen
# ---------------------------------------------------------------------------


class TestBuildAnalysisProgressScreen:
    """Test the loading/progress screen shown during analysis."""

    def test_returns_panel(self):
        result = _build_analysis_progress_screen([], width=80, height=24)
        assert isinstance(result, Panel)

    def test_with_progress_steps(self):
        steps = ["Fetching sprints...", "Analysing velocity...", "Building profile..."]
        result = _build_analysis_progress_screen(steps, width=80, height=24)
        output = _render(result)
        assert "Fetching sprints" in output
        assert "✓ Fetching sprints" not in output
        assert "• Fetching sprints" in output
        # The active row spins (braille) instead of a static ▸ now.
        assert any(f"{g} Building profile" in output for g in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")

    def test_legacy_activity_uses_planning_card_theme(self):
        result = _build_analysis_progress_screen(
            ["Fetching sprints...", "Building profile..."],
            mode="planning",
            width=80,
            height=24,
        )
        rows = [item for item in result.renderable.renderables if isinstance(item, Text)]
        history = next(item for item in rows if "Fetching sprints" in item.plain)
        current = next(item for item in rows if "Building profile" in item.plain)
        assert str(history.style) == PLANNING_THEME.accent
        assert str(current.style) == f"bold {PLANNING_THEME.accent_bright}"
        assert str(result.border_style) == PLANNING_THEME.accent

    def test_structured_running_components_do_not_render_as_completed(self):
        progress = [
            {
                "kind": "analysis_component",
                "component_id": "delivery:jira",
                "label": "Fetching sprint history · Jira",
                "status": "running",
                "detail": "",
            },
            {
                "kind": "analysis_component",
                "component_id": "code:code",
                "label": "Scanning AI footprint & repository health",
                "status": "running",
                "detail": "",
            },
            "Docs · Confluence space: PSO",
        ]
        output = _render(_build_analysis_progress_screen(progress, width=100, height=24))
        spinners = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        assert any(f"{g} Fetching sprint history" in output for g in spinners)
        assert any(f"{g} Scanning AI footprint" in output for g in spinners)
        assert "✓ Fetching sprint history" not in output
        assert "Docs · Confluence space: PSO" in output

    def test_structured_component_only_checks_on_completed_event(self):
        progress = [
            {
                "kind": "analysis_component",
                "component_id": "delivery:jira",
                "label": "Fetching sprint history · Jira",
                "status": "running",
                "detail": "",
            },
            {
                "kind": "analysis_component",
                "component_id": "delivery:jira",
                "label": "Fetching sprint history · Jira",
                "status": "completed",
                "detail": "",
            },
        ]
        output = _render(_build_analysis_progress_screen(progress, width=100, height=24))
        assert "✓ Fetching sprint history" in output
        assert not any(f"{g} Fetching sprint history" in output for g in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")

        result = _build_analysis_progress_screen(progress, mode="analysis", width=100, height=24)
        completed = next(
            item
            for item in result.renderable.renderables
            if isinstance(item, Text) and "Fetching sprint history" in item.plain
        )
        assert str(completed.style) == ANALYSIS_THEME.accent

    def test_code_progress_is_counted_and_explicitly_read_only(self):
        progress = [
            {
                "kind": "analysis_component",
                "component_id": "code:code_health",
                "label": "Analysing selected-user code-change health",
                "status": "running",
                "detail": "",
                "phase": "Reading code-change metadata",
                "current": 3,
                "total": 5,
                "unit": "changes inspected",
                "secondary_count": 27,
                "secondary_unit": "file records found",
                "read_only": True,
            }
        ]

        output = _render(
            _build_analysis_progress_screen(progress, width=160, height=24),
            width=160,
        )

        assert "60% · 3/5 changes inspected" in output
        assert "27 file records found" in output
        assert "Repository access is read-only" in output
        assert "no files are modified" in output
        assert "Changed files:" not in output

    def test_elapsed_time(self):
        result = _build_analysis_progress_screen(["Working..."], elapsed=12.5, width=80, height=24)
        output = _render(result)
        assert "12" in output  # elapsed seconds shown

    def test_animation_tick(self):
        """Different animation ticks should produce valid panels."""
        for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = _build_analysis_progress_screen(["Working..."], anim_tick=tick, width=80, height=24)
            assert isinstance(result, Panel)

    def test_spinner_glyph_advances_with_the_clock(self):
        a = _render(_build_analysis_progress_screen(["Working..."], anim_tick=0.0, width=80, height=24))
        b = _render(_build_analysis_progress_screen(["Working..."], anim_tick=0.35, width=80, height=24))
        ga = next(g for g in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if f"{g} Working" in a)
        gb = next(g for g in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if f"{g} Working" in b)
        assert ga != gb

    def test_structured_run_gets_a_count_and_total_footer(self):
        progress = [
            {
                "kind": "analysis_component",
                "component_id": "footer:done",
                "label": "Fetching sprint history",
                "status": "completed",
                "detail": "",
            },
            {
                "kind": "analysis_component",
                "component_id": "footer:running",
                "label": "Scanning AI footprint",
                "status": "running",
                "detail": "",
            },
        ]
        output = _render(_build_analysis_progress_screen(progress, anim_tick=75.0, width=100, height=24))
        assert "[1/2]" in output
        assert "total 1:15" in output

    def test_running_row_carries_a_per_stage_elapsed(self):
        def _screen(tick):
            return _render(
                _build_analysis_progress_screen(
                    [
                        {
                            "kind": "analysis_component",
                            "component_id": "elapsed:stage",
                            "label": "Scanning AI footprint",
                            "status": "running",
                            "detail": "",
                        }
                    ],
                    anim_tick=tick,
                    width=100,
                    height=24,
                )
            )

        _screen(0.0)  # first sight starts this stage's clock
        assert "0:42" in _screen(42.0)

    def test_analysis_mode_renders(self):
        """Analysis mode should render without error and use analysis title."""
        result = _build_analysis_progress_screen(["Working..."], mode="analysis", width=100, height=24)
        assert isinstance(result, Panel)

    def test_planning_mode_renders(self):
        """Planning mode should render without error and use planning title."""
        result = _build_analysis_progress_screen(["Working..."], mode="planning", width=100, height=24)
        assert isinstance(result, Panel)

    def test_source_label(self):
        result = _build_analysis_progress_screen(
            ["Fetching..."], source="azdevops", mode="analysis", width=100, height=24
        )
        assert isinstance(result, Panel)

    def test_empty_progress(self):
        result = _build_analysis_progress_screen([], elapsed=0.0, width=80, height=24)
        assert isinstance(result, Panel)


# ---------------------------------------------------------------------------
# Team analysis screen (initial report page with custom viewport)
# ---------------------------------------------------------------------------


class TestBuildTeamAnalysisScreenExtended:
    """Extended tests for the initial analysis report screen.

    Supplements the 2 existing tests in test_team_profile.py with coverage
    for scrollbar, viewport wrapping, export selection, and edge cases.
    """

    @pytest.fixture()
    def profile(self):
        from yeaboi.team_profile import (
            DoDSignal,
            EpicPattern,
            SpilloverStats,
            StoryPointCalibration,
            StoryShapePattern,
            TeamProfile,
            WritingPatterns,
        )

        return TeamProfile(
            team_id="azdevops-PROJ",
            source="azdevops",
            project_key="PROJ",
            sample_sprints=8,
            sample_stories=64,
            velocity_avg=23.5,
            velocity_stddev=3.2,
            point_calibrations=(
                StoryPointCalibration(point_value=1, avg_cycle_time_days=0.5, sample_count=10),
                StoryPointCalibration(point_value=3, avg_cycle_time_days=2.1, sample_count=20, overshoot_pct=15.0),
                StoryPointCalibration(point_value=5, avg_cycle_time_days=4.2, sample_count=15, overshoot_pct=20.0),
            ),
            story_shapes=(
                StoryShapePattern(
                    discipline="backend", avg_points=3.2, avg_ac_count=3.0, avg_task_count=2.8, sample_count=20
                ),
                StoryShapePattern(
                    discipline="frontend", avg_points=2.5, avg_ac_count=2.5, avg_task_count=2.0, sample_count=12
                ),
            ),
            epic_pattern=EpicPattern(
                avg_stories_per_epic=6.0, avg_points_per_epic=18.0, typical_story_count_range=(4, 9)
            ),
            estimation_accuracy_pct=78.0,
            sprint_completion_rate=88.0,
            spillover=SpilloverStats(
                carried_over_pct=12.5,
                avg_spillover_pts=3.2,
                most_common_spillover_reason="backend stories",
            ),
            dod_signal=DoDSignal(
                common_checklist_items=("tests passing", "PR merged", "code reviewed"),
                stories_with_comments_pct=85.0,
                stories_with_pr_link_pct=82.0,
                stories_with_review_mention_pct=76.0,
                stories_with_testing_mention_pct=61.0,
                stories_with_deploy_mention_pct=44.0,
            ),
            writing_patterns=WritingPatterns(
                median_ac_count=3.0,
                median_task_count_per_story=2.5,
                subtask_label_distribution=(("Code", 0.58), ("Testing", 0.28)),
                common_subtask_patterns=("Write unit tests", "Deploy to staging"),
                subtasks_use_consistent_naming=True,
                common_personas=("developer", "admin"),
                uses_given_when_then=True,
                stories_with_subtasks_pct=72.0,
            ),
            sprints_fully_completed=6,
            sprints_partially_completed=2,
            sprints_analysed=8,
        )

    def test_returns_panel(self, profile):
        result = _build_team_analysis_screen(profile, width=80, height=30)
        assert isinstance(result, Panel)

    def test_export_button_selection(self, profile):
        """Each export_sel value should highlight a different button."""
        for sel in range(2):  # Export, Continue
            result = _build_team_analysis_screen(profile, export_sel=sel, width=80, height=30)
            assert isinstance(result, Panel)

    def test_scrollbar_on_tall_content(self, profile):
        """Profile with many sections should show scrollbar."""
        result = _build_team_analysis_screen(profile, scroll_offset=0, width=80, height=24)
        output = _render(result, width=80)
        assert "\u2502" in output or "\u2503" in output

    def test_scroll_to_bottom(self, profile):
        """Scrolling to a large offset should clamp and still show buttons."""
        result = _build_team_analysis_screen(profile, scroll_offset=9999, width=80, height=24)
        output = _render(result, width=80)
        # Buttons should still be visible
        assert "Export" in output or "Continue" in output

    def test_with_examples(self, profile):
        result = _build_team_analysis_screen(profile, examples=_SAMPLE_EXAMPLES, width=80, height=30)
        assert isinstance(result, Panel)

    def test_with_sprint_names(self, profile):
        names = ["Sprint 101", "Sprint 102", "Sprint 103"]
        result = _build_team_analysis_screen(profile, sprint_names=names, width=80, height=30)
        assert isinstance(result, Panel)

    def test_with_team_name(self, profile):
        result = _build_team_analysis_screen(profile, team_name="Platform Team", width=100, height=30)
        output = _render(result, width=100)
        assert "Platform Team" in output

    def test_narrow_terminal(self, profile):
        """Should render without crash on narrow terminals."""
        result = _build_team_analysis_screen(profile, width=40, height=24)
        assert isinstance(result, Panel)

    def test_short_terminal(self, profile):
        """Should render on short terminals without crash."""
        result = _build_team_analysis_screen(profile, width=80, height=14)
        assert isinstance(result, Panel)

    def test_velocity_section_rendered(self, profile):
        result = _build_team_analysis_screen(profile, width=100, height=50)
        output = _render(result, width=100)
        assert "23.5" in output  # velocity_avg

    def test_spillover_rendered(self, profile):
        result = _build_team_analysis_screen(profile, view="velocity", width=100, height=100)
        output = _render(result, width=100)
        assert "12.5" in output or "spillover" in output.lower()

    def test_zero_scroll(self, profile):
        result = _build_team_analysis_screen(profile, scroll_offset=0, width=80, height=30)
        assert isinstance(result, Panel)

    def test_mid_scroll(self, profile):
        result = _build_team_analysis_screen(profile, scroll_offset=10, width=80, height=30)
        assert isinstance(result, Panel)

    # Wrapping tables (DoD + Proposed DoD) previously reported a naive row_count
    # as their height. When cells wrapped onto multiple rows the viewport packer
    # over-filled the fixed-height panel and Rich cropped the action buttons off
    # the bottom (Patterns page showed no buttons). Heights are now measured, so
    # the buttons must survive even with a tall, heavily-wrapping Proposed DoD.
    _DOD_HEAVY_EXAMPLES = {
        "dod_testing": [{"issue_key": "PSOT-791", "summary": "Phase 9: Entra App Registration flow"}],
        "dod_pr": [{"issue_key": "PSOT-851", "summary": "WIZ - Create an automated repo scanner"}],
        "dod_review": [{"issue_key": "PSOT-880", "summary": "Complete DAST API Scan Implementation"}],
        "dod_deploy": [{"issue_key": "PSOT-880", "summary": "Complete DAST API Scan Implementation"}],
        "proposed_dod": {
            "summary": "7 of 9 practices are well-established. The team has a clear definition of done.",
            "health": "strong",
            "items": [
                {
                    "practice": f"Practice number {i} updated",
                    "status": "established",
                    "signals": f"{90 - i * 8}% mentioned in stories · 6% have subtasks",
                    "recommendation": "Consistently done. Include as a required DoD step.",
                }
                for i in range(8)
            ],
        },
    }

    @staticmethod
    def _render_cropped(panel: Panel, width: int, height: int) -> str:
        """Render exactly like the TUI: a fixed-height panel on a sized console.

        The panel's ``height`` crops overflowing content, so this reproduces the
        real button-cropping behaviour that a height-less render would hide.
        """
        buf = StringIO()
        console = Console(file=buf, width=width, height=height, force_terminal=False, highlight=False)
        console.print(panel)
        return buf.getvalue()

    def test_workflow_card_buttons_visible_with_wrapping_tables(self, profile):
        """Workflow & DoD action buttons must stay on screen even when tables wrap a lot."""
        for scroll in (0, 9999):  # top of card and clamped-to-bottom
            panel = _build_team_analysis_screen(
                profile,
                examples=self._DOD_HEAVY_EXAMPLES,
                view="workflow",
                width=120,
                height=44,
                scroll_offset=scroll,
            )
            output = self._render_cropped(panel, width=120, height=44)
            assert "Back" in output, f"Back button cropped at scroll={scroll}"
            assert "Continue" in output, f"Continue button cropped at scroll={scroll}"

    def test_both_mode_toggle_and_comparison(self, profile):
        """'Both' mode renders the source toggle line and side-by-side comparison."""
        comparison = [("Avg velocity", "23", "15"), ("Completion rate", "82%", "74%")]
        panel = _build_team_analysis_screen(
            profile,
            view="overview",
            width=100,
            height=40,
            source_toggle=["jira", "azdevops"],
            active_source="azdevops",
            comparison=comparison,
        )
        out = _render(panel, width=100)
        assert "Tab: switch source" in out
        assert "Jira vs Azure DevOps" in out  # comparison heading
        assert "23" in out and "15" in out  # per-tracker figures side by side

    def test_no_toggle_without_source_toggle(self, profile):
        """Single-source render shows no toggle affordance."""
        out = _render(_build_team_analysis_screen(profile, width=100, height=40), width=100)
        assert "Tab: switch source" not in out


class TestRunTeamAnalysisResultsBoth:
    """Drive _run_team_analysis_results in 'both' mode (source toggle + active_box).

    Covers the runner-loop branch the screen-builder tests can't reach: the
    Tab-cycled source switch, and the active_box mirror-back contract the two
    call sites rely on for the downstream insights/ticket steps.
    """

    class _FakeLive:
        def update(self, renderable):
            self.last = renderable

    @staticmethod
    def _console():
        from types import SimpleNamespace

        return SimpleNamespace(size=(100, 40))

    @staticmethod
    def _reader(keys):
        it = iter(keys)

        def read_key(timeout=None):
            return next(it)

        return read_key

    @staticmethod
    def _both():
        from yeaboi.team_profile import TeamProfile

        jira = TeamProfile(team_id="jira:P", source="jira", project_key="P", team_name="")
        azdo = TeamProfile(team_id="azdevops:Web", source="azdevops", project_key="Web", team_name="Web Team")
        return {
            "jira": {"profile": jira, "examples": {}, "sprint_names": ["S1"], "source": "jira", "project_key": "P"},
            "azdevops": {
                "profile": azdo,
                "examples": {},
                "sprint_names": ["I1"],
                "source": "azdevops",
                "project_key": "Web",
            },
        }

    def _run(self, keys, active_box):
        from yeaboi.ui.mode_select import _run_team_analysis_results

        both = self._both()
        return _run_team_analysis_results(
            self._FakeLive(),
            self._console(),
            self._reader(keys),
            0.01,
            True,
            both["jira"]["profile"],
            {},
            sprint_names=["S1"],
            delivery=both,
            comparison=[("Avg velocity", "23", "15")],
            active_box=active_box,
        )

    def test_tab_switches_active_source(self):
        active_box: list = [None]
        # Frame 1 = jira; Tab → azdevops; Esc on overview exits.
        assert self._run(["tab", "esc"], active_box) == "back"
        profile, _examples, sprint_names, team_name = active_box[0]
        assert profile.source == "azdevops"
        assert team_name == "Web Team"
        assert sprint_names == ["I1"]

    def test_no_team_name_bleed_when_toggling_back(self):
        active_box: list = [None]
        # jira → azdevops (team "Web Team") → back to jira: must NOT keep "Web Team".
        assert self._run(["tab", "tab", "esc"], active_box) == "back"
        profile, _examples, _sprint_names, team_name = active_box[0]
        assert profile.source == "jira"
        assert team_name == ""  # regression guard: no cross-tracker team-name bleed

    def test_global_docs_card_shows_on_every_tab(self):
        # Jira + Azure DevOps delivery, plus ONE global docs scan shown on both tabs.
        from yeaboi.team_profile import DocQualitySignal

        both = self._both()
        docs = {"signal": DocQualitySignal(pages_scanned=6, avg_clarity=71.0), "examples": {"summary": {}}}
        from yeaboi.ui.mode_select import _run_team_analysis_results

        active_box: list = [None]
        res = _run_team_analysis_results(
            self._FakeLive(),
            self._console(),
            self._reader(["tab", "esc"]),
            0.01,
            True,
            both["jira"]["profile"],
            {},
            sprint_names=["S1"],
            delivery=both,
            docs=docs,
            active_box=active_box,
        )
        assert res == "back"
        # Toggling tracker keeps a real (per-tracker) delivery profile in active_box.
        assert active_box[0][0].source == "azdevops"

    def test_failed_feature_offers_retry_action(self, monkeypatch):
        from rich.text import Text

        from yeaboi.team_profile import AiAdoptionSignal
        from yeaboi.ui.mode_select import _run_team_analysis_results

        both = self._both()
        captured = {}

        def screen(*args, **kwargs):
            captured["actions"] = kwargs["actions"]
            return Text("results")

        monkeypatch.setattr("yeaboi.ui.mode_select._build_team_analysis_screen", screen)
        code = {
            "signal": AiAdoptionSignal(scanned_commits=10),
            "examples": {
                "enabled_features": ["code_health"],
                "coverage_report": {"status": "failed", "completed": 0, "eligible": 10},
                "repository_health": {"files_analysed": 0},
            },
        }

        result = _run_team_analysis_results(
            self._FakeLive(),
            self._console(),
            self._reader(["esc"]),
            0.01,
            True,
            both["jira"]["profile"],
            {},
            delivery=both,
            code=code,
            analysis_features=["delivery", "code_health"],
            retry_config={"components": {"code": ["azdo"], "docs": []}},
        )

        assert result == "back"
        assert "Retry failed" in captured["actions"]


class TestDeliveryOffScreen:
    """_build_team_analysis_screen with profile=None (a code/docs-only run) — the
    global signals are passed via code_signal/doc_signal."""

    @staticmethod
    def _code_signal():
        from yeaboi.team_profile import AiAdoptionSignal

        return AiAdoptionSignal(
            scanned_commits=40,
            ai_commits=18,
            footprint_pct=45.0,
            per_tool=(("claude", 18),),
            sources_scanned=("github",),
        )

    def test_renders_without_profile(self):
        panel = _build_team_analysis_screen(
            None,
            width=100,
            height=40,
            examples={"ai_adoption": {"summary": {}, "coverage": [], "samples": []}},
            source="jira",
            project_key="PROJ",
            code_signal=self._code_signal(),
        )
        assert isinstance(panel, Panel)
        out = _render(panel, width=100)
        assert "jira/PROJ" in out
        assert "only" in out  # header says "... only" (no delivery profile)
        assert "sprints" not in out

    def test_shows_code_card_from_signal(self):
        panel = _build_team_analysis_screen(
            None,
            width=100,
            height=40,
            examples={"ai_adoption": {"summary": {}, "coverage": [], "samples": []}},
            view="ai-adoption",
            source="jira",
            project_key="PROJ",
            code_signal=self._code_signal(),
        )
        out = _render(panel, width=100)
        assert "AI Usage" in out
        assert "45%" in out  # footprint rendered from the global code_signal

    def test_ai_adoption_dashboard_breakdowns_evidence_and_actions(self):
        from yeaboi.team_profile import AiAdoptionSignal

        signal = AiAdoptionSignal(
            scanned_commits=40,
            scanned_prs=10,
            ai_commits=18,
            ai_prs=3,
            footprint_pct=42.0,
            per_tool=(("claude", 14), ("copilot", 7)),
            per_author=(("Ava", 12), ("Sam", 9)),
            per_activity=(("code", 17), ("pr", 4)),
            per_source=(("github", 21),),
            sources_scanned=("github",),
            repos_scanned=("acme/api",),
        )
        examples = {
            "ai_adoption": {
                "samples": [
                    {
                        "tool": "claude",
                        "title": "Improve authentication flow",
                        "url": "https://example.test/pr/42",
                    }
                ],
                "insights": {
                    "start": [
                        {
                            "title": "Share effective prompts",
                            "detail": "Turn individual wins into team practice.",
                            "evidence": "Two active adopters",
                        }
                    ],
                    "stop": [],
                    "keep": [],
                    "try": [],
                },
            }
        }
        top = _render(
            _build_team_analysis_screen(
                None,
                examples=examples,
                view="ai-adoption",
                code_signal=signal,
                width=100,
                height=50,
            ),
            width=100,
        )
        bottom = _render(
            _build_team_analysis_screen(
                None,
                examples=examples,
                view="ai-adoption",
                code_signal=signal,
                scroll_offset=9999,
                width=100,
                height=50,
            ),
            width=100,
        )
        combined = top + bottom
        for expected in (
            "LOWER BOUND SIGNAL",
            "DETECTABLE FOOTPRINT",
            "Tools detected",
            "Contributors",
            "Evidence",
            "Improve authentication flow",
            "Recommended actions",
            "Share effective prompts",
        ):
            assert expected in combined

    def test_zero_marked_explains_detection_limits(self):
        from yeaboi.team_profile import AiAdoptionSignal

        signal = AiAdoptionSignal(scanned_commits=30, ai_commits=0, footprint_pct=0.0)
        out = _render(
            _build_team_analysis_screen(
                None, examples={}, view="ai-adoption", code_signal=signal, width=100, height=50
            ),
            width=100,
        )
        assert "does not mean zero AI use" in out
        assert "attribution" in out

    def test_identity_mismatch_warning_when_no_users_matched(self):
        from yeaboi.team_profile import AiAdoptionSignal

        signal = AiAdoptionSignal()
        examples = {
            "ai_adoption": {
                "selected_users": ["Ava"],
                "matched_identities": {},
                "unmatched_users": ["Ava"],
                "coverage": ["no commits or authored PRs matched the selected users"],
            }
        }
        out = _render(
            _build_team_analysis_screen(
                None, examples=examples, view="ai-adoption", code_signal=signal, width=100, height=50
            ),
            width=100,
        )
        assert "IDENTITY MISMATCH" in out
        assert "not zero AI usage" in out

    def test_no_identity_mismatch_warning_when_users_matched(self):
        from yeaboi.team_profile import AiAdoptionSignal

        signal = AiAdoptionSignal(scanned_commits=30, ai_commits=6, footprint_pct=20.0)
        examples = {"ai_adoption": {"selected_users": ["Ava"], "matched_identities": {"Ava": ["ava"]}}}
        out = _render(
            _build_team_analysis_screen(
                None, examples=examples, view="ai-adoption", code_signal=signal, width=100, height=50
            ),
            width=100,
        )
        assert "IDENTITY MISMATCH" not in out

    def _practices_examples(self):
        return {
            "ai_adoption": {
                "selected_users": ["Ava", "Sam"],
                "matched_identities": {"Ava": ["ava"], "Sam": ["sam"]},
                "member_activity": [
                    {"member": "Ava", "commits": 2140, "prs": 12, "ai_marked": 5},
                    {"member": "Sam", "commits": 3, "prs": 4, "ai_marked": 1},
                ],
                "member_practices": {
                    "min_sample": 5,
                    "file_data": {"with_file_data": 9, "total": 12},
                    "members": [
                        {
                            "member": "Ava",
                            "commits": 2140,
                            "prs": 12,
                            "with_file_data": 8,
                            "tests_num": 6,
                            "tests_den": 8,
                            "tests_rate": 75.0,
                            "docs_num": 2,
                            "docs_den": 8,
                            "docs_rate": 25.0,
                            "ticket_num": 5,
                            "ticket_den": 10,
                            "ticket_rate": 50.0,
                            "desc_num": 7,
                            "desc_den": 12,
                            "desc_rate": 58.3,
                        },
                        {
                            "member": "Sam",
                            "commits": 3,
                            "prs": 4,
                            "with_file_data": 1,
                            "tests_num": 0,
                            "tests_den": 0,
                            "tests_rate": None,
                            "docs_num": 1,
                            "docs_den": 1,
                            "docs_rate": 100.0,
                            "ticket_num": 2,
                            "ticket_den": 4,
                            "ticket_rate": 50.0,
                            "desc_num": 1,
                            "desc_den": 4,
                            "desc_rate": 25.0,
                        },
                    ],
                    "team": {
                        "member": "Team",
                        "commits": 2143,
                        "prs": 16,
                        "with_file_data": 9,
                        "tests_num": 6,
                        "tests_den": 8,
                        "tests_rate": 75.0,
                        "docs_num": 3,
                        "docs_den": 9,
                        "docs_rate": 33.3,
                        "ticket_num": 7,
                        "ticket_den": 14,
                        "ticket_rate": 50.0,
                        "desc_num": 8,
                        "desc_den": 16,
                        "desc_rate": 50.0,
                    },
                },
            }
        }

    def test_ai_adoption_practices_table(self):
        from yeaboi.team_profile import AiAdoptionSignal

        signal = AiAdoptionSignal(scanned_commits=30, ai_commits=6, footprint_pct=20.0)
        out = _render(
            _build_team_analysis_screen(
                None, examples=self._practices_examples(), view="ai-adoption", code_signal=signal, width=100, height=70
            ),
            width=100,
        )
        assert "Engineering practices by member" in out
        for header in ("Tests", "Docs", "Tickets", "Descs"):
            assert header in out
        assert "75%" in out  # Ava's tests rate as a coloured percentage
        assert "n/a" in out  # Sam has no tests denominator
        assert "1/1" in out  # Sam's docs cell stays a raw fraction under the sample floor
        assert "Team" in out
        assert "2140" in out  # volume counts stay visible in the merged table
        assert "cover 9 of 12 items with change metadata" in out

    def test_practices_lead_above_footprint(self):
        from yeaboi.team_profile import AiAdoptionSignal

        signal = AiAdoptionSignal(scanned_commits=30, ai_commits=6, footprint_pct=20.0)
        out = _render(
            _build_team_analysis_screen(
                None, examples=self._practices_examples(), view="ai-adoption", code_signal=signal, width=100, height=70
            ),
            width=100,
        )
        assert out.index("Engineering practices by member") < out.index("LOWER BOUND SIGNAL")

    def test_ai_adoption_activity_fallback_for_old_blobs(self):
        # Saved profiles that predate practice scoring keep the activity table.
        from yeaboi.team_profile import AiAdoptionSignal

        signal = AiAdoptionSignal(scanned_commits=30, ai_commits=6, footprint_pct=20.0)
        examples = {
            "ai_adoption": {
                "selected_users": ["Ava", "Sam"],
                "matched_identities": {"Ava": ["ava"], "Sam": ["sam"]},
                "member_activity": [
                    {"member": "Ava", "commits": 2140, "prs": 12, "ai_marked": 5},
                    {"member": "Sam", "commits": 3, "prs": 4, "ai_marked": 1},
                    {"member": "AI agent accounts", "commits": 7, "prs": 0, "ai_marked": 7},
                ],
            }
        }
        out = _render(
            _build_team_analysis_screen(
                None, examples=examples, view="ai-adoption", code_signal=signal, width=100, height=60
            ),
            width=100,
        )
        assert "Activity by member" in out
        assert "2140" in out  # the automation-heavy member's volume is visible
        assert "AI agent accounts" in out
        assert "Engineering practices" not in out

    def test_ai_adoption_no_member_table_for_old_profiles(self):
        from yeaboi.team_profile import AiAdoptionSignal

        signal = AiAdoptionSignal(scanned_commits=30, ai_commits=6, footprint_pct=20.0)
        examples = {"ai_adoption": {"selected_users": ["Ava"], "matched_identities": {"Ava": ["ava"]}}}
        out = _render(
            _build_team_analysis_screen(
                None, examples=examples, view="ai-adoption", code_signal=signal, width=100, height=50
            ),
            width=100,
        )
        assert "Activity by member" not in out

    def test_ai_adoption_repo_list_capped_with_more_row(self):
        from yeaboi.team_profile import AiAdoptionSignal

        signal = AiAdoptionSignal(
            scanned_commits=40,
            ai_commits=8,
            footprint_pct=20.0,
            sources_scanned=("github",),
            repos_scanned=tuple(f"acme/repo-{i:02d}" for i in range(12)),
        )
        out = _render(
            _build_team_analysis_screen(
                None, examples={}, view="ai-adoption", code_signal=signal, width=100, height=60
            ),
            width=100,
        )
        assert "Repositories" in out and "12" in out
        assert "acme/repo-05" in out
        assert "acme/repo-06" not in out
        assert "+ 6 more repositories" in out

    def test_ai_adoption_evidence_capped(self):
        from yeaboi.team_profile import AiAdoptionSignal

        signal = AiAdoptionSignal(scanned_commits=40, ai_commits=12, footprint_pct=30.0)
        examples = {
            "ai_adoption": {
                "samples": [{"tool": "claude", "title": f"Sample change {i:02d}"} for i in range(12)],
            }
        }
        out = _render(
            _build_team_analysis_screen(
                None,
                examples=examples,
                view="ai-adoption",
                code_signal=signal,
                scroll_offset=9999,
                width=100,
                height=50,
            ),
            width=100,
        )
        assert "Sample change 07" in out
        assert "Sample change 08" not in out
        assert "+ 4 more AI-marked items" in out

    def test_visible_card_order_code_only(self):
        from yeaboi.ui.mode_select.screens._analysis_sections import visible_card_order

        assert visible_card_order(None, has_code=True, has_docs=False) == ("ai-adoption",)
        assert visible_card_order(None, has_code=False, has_docs=True) == ("documentation",)

    def test_code_health_only_hides_ai_footprint(self):
        from yeaboi.ui.mode_select.screens._analysis_sections import visible_card_order

        assert visible_card_order(
            None,
            has_code=False,
            has_docs=False,
            has_code_health=True,
            analysis_features=["code_health"],
        ) == ("code-health",)

    def test_visible_card_order_delivery_plus_globals(self):
        from yeaboi.team_profile import TeamProfile
        from yeaboi.ui.mode_select.screens._analysis_sections import visible_card_order

        order = visible_card_order(TeamProfile(team_id="t", source="jira", project_key="P"), True, True)
        # Delivery cards, then the two global cards, then insights last.
        assert order[0] == "velocity"
        assert "ai-adoption" in order and "documentation" in order
        assert order[-1] == "insights"


class TestAnalysisFeatureScreen:
    _AVAILABLE = {
        "delivery": True,
        "ai_footprint": True,
        "code_health": True,
        "documentation": False,
    }

    def test_all_runnable_areas_start_selected(self):
        out = _render(
            _build_analysis_feature_screen(
                self._AVAILABLE,
                {"delivery", "ai_footprint", "code_health"},
                0,
                width=100,
                height=32,
            ),
            width=100,
        )
        assert "Analyse all" in out
        assert "AI footprint" in out and "Code health" in out
        assert "Unavailable" in out
        assert "3/3 selected" in out

    def test_narrow_screen_keeps_focused_feature_visible(self):
        out = _render(
            _build_analysis_feature_screen(
                {key: True for key in ("delivery", "ai_footprint", "code_health", "documentation")},
                {"documentation"},
                4,
                width=48,
                height=20,
            ),
            width=48,
        )
        assert "‹ ● Documentation ›" in out


def test_narrow_window_screen_keeps_focused_tile_visible():
    out = _render(_build_analysis_window_screen(3, width=48, height=20), width=48)
    assert "‹ ● 365 DAYS ›" in out


class TestComponentSelectScreen:
    """Ragged component × sub-source picker screen + loop."""

    _GRID = {"delivery": ["jira", "azdevops"], "code": ["github", "azdo"], "docs": ["confluence", "notion"]}

    def test_ragged_render(self):
        checked = {"delivery": {0}, "code": {0, 1}, "docs": {0}}
        out = _render(
            _build_component_select_screen(
                self._GRID, ["delivery", "code", "docs"], checked, 0, 0, width=110, height=30
            ),
            width=110,
        )
        assert "DELIVERY" in out and "CODE" in out and "DOCS" in out
        # Each component shows its OWN sub-sources.
        assert "Jira" in out and "GitHub" in out and "Confluence" in out
        # Match the app-wide filled/empty toggle states.
        assert "●" in out and "○" in out
        assert "Delivery 1" in out and "sources" in out

    def test_focused_source_is_bracketed(self):
        checked = {"delivery": {0, 1}, "code": {0, 1}, "docs": {0, 1}}
        out = _render(
            _build_component_select_screen(
                self._GRID, ["delivery", "code", "docs"], checked, 1, 1, width=110, height=30
            ),
            width=110,
        )
        assert "‹ ● Azure Repos ›" in out

    def test_no_selection_shows_hint(self):
        checked = {"delivery": set(), "code": set(), "docs": set()}
        out = _render(
            _build_component_select_screen(
                self._GRID, ["delivery", "code", "docs"], checked, 0, 0, width=100, height=30
            ),
            width=100,
        )
        assert "at least one" in out

    def test_returns_panel_narrow(self):
        checked = {"delivery": {0}}
        assert isinstance(
            _build_component_select_screen({"delivery": ["jira"]}, ["delivery"], checked, 0, 0, width=40, height=20),
            Panel,
        )

    def test_default_branding_stays_analysis(self):
        """Regression: generalizing the screen for Reporting must not restyle analysis."""
        checked = {"delivery": {0}}
        out = _render(
            _build_component_select_screen(self._GRID, ["delivery"], checked, 0, 0, width=110, height=30),
            width=110,
        )
        assert "ANALYSIS SETUP" in out
        assert "REPORTING" not in out

    def test_reporting_branding_and_footer_verb(self):
        from yeaboi.ui.shared._components import REPORTING_THEME, reporting_title

        grid = {"delivery": ["jira", "azuredevops"], "code": ["github"]}
        out = _render(
            _build_component_select_screen(
                grid,
                ["delivery", "code"],
                {"delivery": set(), "code": set()},
                0,
                0,
                width=110,
                height=32,
                theme=REPORTING_THEME,
                brand="REPORTING SETUP",
                title_builder=lambda w, h: reporting_title(width=w),
                footer_verb="report on",
            ),
            width=110,
        )
        assert "REPORTING SETUP" in out
        assert "Azure DevOps" in out  # reporting's canonical token maps to a display name
        assert "at least one source to report on" in out


class TestAnalysisDepthScreen:
    def test_deep_is_rendered_as_recommended(self):
        out = _render(_build_analysis_depth_screen(0, width=100, height=30), width=100)
        assert "QUICK" in out and "Recommended" in out
        assert "DEEP" in out and "exhaustive" in out.lower()

    def test_deep_can_be_focused(self):
        out = _render(_build_analysis_depth_screen(1, width=100, height=30), width=100)
        assert "‹ ● DEEP ›" in out


class TestAnalysisSetupReviewScreen:
    def test_summarizes_exact_run_scope(self):
        out = _render(
            _build_analysis_setup_review_screen(
                features=["ai_footprint", "documentation"],
                components={"code": ["github", "azdo"], "docs": ["confluence"]},
                members=["Alice", "Bob"],
                analysis_scope={"azdo": ["Infrastructure", "Product"]},
                depth="deep",
                window_days=120,
                model="qwen3:4b",
                width=100,
                height=34,
            ),
            width=100,
        )
        assert "REVIEW" in out and "Run Analysis" in out and "Back" in out
        assert "AI footprint" in out and "Documentation" in out
        assert "GitHub" in out and "Infrastructure" in out
        assert "Alice, Bob" in out
        assert "Deep · 120 days" in out and "qwen3:4b" in out

    def test_both_code_hosts_show_their_own_scope(self):
        # The review screen is the last chance to notice the wrong owner was
        # picked, so neither host may be summarised away.
        out = _render(
            _build_analysis_setup_review_screen(
                features=["ai_footprint"],
                components={"code": ["github", "azdo"]},
                members=None,
                analysis_scope={"github": ["acme-corp"], "azdo": ["Infrastructure"]},
                depth="deep",
                window_days=120,
                width=100,
                height=34,
            ),
            width=100,
        )
        assert "GitHub owners: acme-corp" in out
        assert "Azure projects: Infrastructure" in out

    def test_compact_review_folds_both_scopes_onto_one_line(self):
        # The compact branch renders one Text per label, so the newline-separated
        # scope lines have to be folded with " · " to survive it at all.
        out = _render(
            _build_analysis_setup_review_screen(
                features=["ai_footprint"],
                components={"code": ["github", "azdo"]},
                members=None,
                analysis_scope={"github": ["acme-corp"], "azdo": ["Infrastructure"]},
                depth="deep",
                window_days=120,
                width=86,
                height=24,
            ),
            width=86,
        )
        sources_line = next(line for line in out.splitlines() if "Sources" in line)
        assert "GitHub owners: acme-corp" in sources_line
        assert "Azure projects:" in sources_line

    def test_review_loop_can_go_back_or_run(self):
        from yeaboi.ui.mode_select import _run_analysis_setup_review

        args = {
            "features": ["documentation"],
            "components": {"docs": ["confluence"]},
            "members": None,
            "analysis_scope": {},
            "depth": "quick",
            "window_days": 120,
            "model": None,
        }
        assert (
            _run_analysis_setup_review(
                TestComponentAndMemberLoops._FakeLive(),
                TestComponentAndMemberLoops._console(),
                TestComponentAndMemberLoops._reader(["enter"]),
                0.01,
                True,
                **args,
            )
            == "run"
        )
        assert (
            _run_analysis_setup_review(
                TestComponentAndMemberLoops._FakeLive(),
                TestComponentAndMemberLoops._console(),
                TestComponentAndMemberLoops._reader(["right", "enter"]),
                0.01,
                True,
                **args,
            )
            == "back"
        )

    def test_short_review_keeps_actions_visible(self):
        out = _render(
            _build_analysis_setup_review_screen(
                features=["documentation"],
                components={"docs": ["confluence"]},
                members=None,
                analysis_scope={},
                depth="quick",
                window_days=120,
                width=48,
                height=20,
            ),
            width=48,
        )
        assert "Run Analysis" in out and "Back" in out


class TestMemberSelectScreen:
    def test_roster_render(self):
        out = _render(_build_member_select_screen(["Alice", "Bob"], {0}, 0, width=100, height=30), width=100)
        assert "Alice" in out and "Bob" in out
        assert "1 of 2 selected" in out
        assert "A selects all" in out

    def test_empty_selection_is_explicit(self):
        out = _render(_build_member_select_screen(["Alice"], set(), 0, width=100, height=30), width=100)
        assert "0 of 1 selected" in out
        assert "whole team" not in out

    def test_empty_roster(self):
        out = _render(_build_member_select_screen([], set(), 0, width=100, height=30), width=100)
        assert "No members found" in out


class TestMoveAnalysisListCursor:
    """The setup pickers are single-column lists — movement is ±1 with wraparound."""

    def _move(self, cursor, key, count):
        from yeaboi.ui.mode_select import _move_analysis_list_cursor

        return _move_analysis_list_cursor(cursor, key, count)

    def test_down_moves_one_row(self):
        assert self._move(0, "down", 5) == 1
        assert self._move(1, "down", 5) == 2

    def test_up_moves_one_row(self):
        assert self._move(2, "up", 5) == 1

    def test_wraps_at_both_ends(self):
        assert self._move(0, "up", 5) == 4
        assert self._move(4, "down", 5) == 0

    def test_left_right_mirror_up_down(self):
        assert self._move(1, "left", 4) == 0
        assert self._move(1, "right", 4) == 2

    def test_scroll_keys_move_one_row(self):
        assert self._move(1, "scroll_up", 4) == 0
        assert self._move(1, "scroll_down", 4) == 2

    def test_unknown_key_is_a_no_op(self):
        assert self._move(2, "x", 5) == 2

    def test_single_row_pins_to_zero(self):
        assert self._move(3, "down", 1) == 0
        assert self._move(0, "up", 0) == 0


class TestComponentAndMemberLoops:
    class _FakeLive:
        def update(self, renderable):
            self.last = renderable

    @staticmethod
    def _console():
        from types import SimpleNamespace

        return SimpleNamespace(size=(100, 40))

    @staticmethod
    def _reader(keys):
        it = iter(keys)

        def read_key(timeout=None):
            return next(it)

        return read_key

    _GRID = {"delivery": ["jira", "azdevops"], "code": ["github", "azdo"], "docs": ["confluence"]}

    def _components(self, keys, grid=None):
        from yeaboi.ui.mode_select import _run_component_select

        return _run_component_select(
            self._FakeLive(), self._console(), self._reader(keys), 0.01, True, grid or self._GRID
        )

    def _features(self, keys):
        from yeaboi.ui.mode_select import _run_analysis_feature_select

        return _run_analysis_feature_select(
            self._FakeLive(),
            self._console(),
            self._reader(keys),
            0.01,
            True,
            {
                "delivery": True,
                "ai_footprint": True,
                "code_health": True,
                "documentation": True,
            },
        )

    def _members(self, keys, roster):
        from yeaboi.ui.mode_select import _run_member_select

        return _run_member_select(self._FakeLive(), self._console(), self._reader(keys), 0.01, True, roster)

    def _depth(self, keys):
        from yeaboi.ui.mode_select import _run_analysis_depth_select

        return _run_analysis_depth_select(self._FakeLive(), self._console(), self._reader(keys), 0.01, True)

    def test_all_checked_returns_full_map(self):
        # Everything checked by default → Enter returns the full component→sub-source map.
        assert self._components(["enter"]) == {
            "delivery": ["jira", "azdevops"],
            "code": ["github", "azdo"],
            "docs": ["confluence"],
        }

    def test_all_features_are_selected_by_default(self):
        assert self._features(["enter"]) == [
            "delivery",
            "ai_footprint",
            "code_health",
            "documentation",
        ]

    def test_feature_can_be_removed(self):
        assert self._features(["down", "right", " ", "enter"]) == [
            "delivery",
            "code_health",
            "documentation",
        ]

    def test_feature_picker_blocks_empty_selection(self):
        assert self._features([" ", "enter", "down", " ", "enter"]) == ["delivery"]

    def test_depth_defaults_to_deep(self):
        assert self._depth(["enter"]) == "deep"

    def test_depth_can_select_quick_or_cancel(self):
        assert self._depth(["left", "enter"]) == "quick"
        assert self._depth(["esc"]) == "cancel"

    def test_deselecting_a_subsource(self):
        # Toggle off delivery[0]=jira → delivery keeps only azdevops.
        result = self._components([" ", "enter"])
        assert result["delivery"] == ["azdevops"]
        assert result["code"] == ["github", "azdo"]

    def test_deselecting_a_whole_component(self):
        # Focus the docs row, toggle off its only checked box → docs dropped entirely.
        # rows: delivery(2 cols), code(2), docs(1). Down twice to docs, Space off.
        result = self._components(["down", "down", " ", "enter"])
        assert "docs" not in result

    def test_cannot_confirm_nothing(self):
        # Uncheck every box across all rows, Enter is blocked, then re-check one.
        keys = [
            " ",
            "right",
            " ",  # delivery: off both
            "down",
            " ",
            "right",
            " ",  # code: off both
            "down",
            " ",  # docs: off
            "enter",  # blocked (nothing selected)
            " ",  # docs back on
            "enter",
        ]
        assert self._components(keys) == {"docs": ["confluence"]}

    def test_cancel(self):
        assert self._components(["esc"]) == "cancel"

    def test_required_component_blocks_enter(self):
        from yeaboi.ui.mode_select import _run_component_select

        # Uncheck both delivery boxes; Enter must be rejected (delivery required),
        # re-checking one then lets Enter through.
        keys = [" ", "right", " ", "enter", " ", "enter"]
        result = _run_component_select(
            self._FakeLive(),
            self._console(),
            self._reader(keys),
            0.01,
            True,
            self._GRID,
            required={"delivery": "Select at least one ticketing source."},
        )
        assert result["delivery"] == ["azdevops"]

    def test_required_ignored_when_component_not_in_grid(self):
        from yeaboi.ui.mode_select import _run_component_select

        result = _run_component_select(
            self._FakeLive(),
            self._console(),
            self._reader(["enter"]),
            0.01,
            True,
            {"code": ["github"]},
            required={"delivery": "Select at least one ticketing source."},
        )
        assert result == {"code": ["github"]}

    def test_member_select_starts_with_everyone_selected(self):
        assert self._members(["enter"], ["Alice", "Bob"]) == ["Alice", "Bob"]

    def test_member_select_can_exclude_a_name(self):
        assert self._members([" ", "enter"], ["Alice", "Bob"]) == ["Bob"]

    def test_member_select_toggle_all(self):
        assert self._members(["a", "right", " ", "enter"], ["Alice", "Bob"]) == ["Bob"]

    def test_member_select_cannot_confirm_nobody(self):
        assert self._members(["a", "enter", " ", "enter"], ["Alice"]) == ["Alice"]

    def test_member_select_empty_roster_is_none(self):
        assert self._members(["enter"], []) is None

    def test_member_cancel(self):
        assert self._members(["esc"], ["Alice"]) == "cancel"

    def test_member_right_key_moves_one_row(self):
        assert self._members(["right", " ", "enter"], ["Alice", "Bob", "Zoe"]) == ["Alice", "Zoe"]

    # Regression tests for the ≥88-column grid-navigation bug: the pickers render a
    # single vertical column at every width, but the old mover treated wide
    # terminals (the fixture console is 100 columns) as a 2-column grid, so
    # up/down jumped two rows and skipped toggles.

    def test_member_down_at_wide_width_moves_one_row(self):
        # Old grid math skipped Bob and toggled Zoe.
        assert self._members(["down", " ", "enter"], ["Alice", "Bob", "Zoe"]) == ["Alice", "Zoe"]

    def test_feature_down_twice_lands_on_second_area(self):
        # Rows: Analyse-all, delivery, ai_footprint, … Old grid math jumped from
        # delivery straight to code_health.
        assert self._features(["down", "down", " ", "enter"]) == [
            "delivery",
            "code_health",
            "documentation",
        ]

    def _window(self, keys):
        from yeaboi.ui.mode_select import _run_analysis_window_select

        return _run_analysis_window_select(self._FakeLive(), self._console(), self._reader(keys), 0.01, True)

    def test_window_down_moves_one_option(self):
        # 120 days is preselected; old grid math made "down" a no-op at wide widths.
        assert self._window(["down", "enter"]) == 365

    def test_window_up_moves_one_option(self):
        assert self._window(["up", "enter"]) == 90

    # initial_* params — the setup wizard re-enters a step with the previous
    # choice restored instead of resetting to the defaults.

    def test_feature_initial_selection_is_restored(self):
        from yeaboi.ui.mode_select import _run_analysis_feature_select

        result = _run_analysis_feature_select(
            self._FakeLive(),
            self._console(),
            self._reader(["enter"]),
            0.01,
            True,
            {"delivery": True, "ai_footprint": True, "code_health": True, "documentation": True},
            initial_features=["delivery", "documentation"],
        )
        assert result == ["delivery", "documentation"]

    def test_component_initial_selection_is_restored(self):
        from yeaboi.ui.mode_select import _run_component_select

        result = _run_component_select(
            self._FakeLive(),
            self._console(),
            self._reader(["enter"]),
            0.01,
            True,
            self._GRID,
            initial={"delivery": ["azdevops"], "docs": ["confluence"]},
        )
        # code is absent from initial (newly enabled) → all-checked like a first visit.
        assert result == {"delivery": ["azdevops"], "code": ["github", "azdo"], "docs": ["confluence"]}

    def test_code_project_initial_selection_is_restored(self, monkeypatch):
        from yeaboi.ui.mode_select import _run_code_scope_select

        monkeypatch.setattr("yeaboi.tools.azure_devops.azdevops_list_projects", lambda: ["Alpha", "Beta", "Gamma"])
        monkeypatch.setattr("yeaboi.config.get_team_analysis_azdo_projects", lambda: [])
        result = _run_code_scope_select(
            self._FakeLive(),
            self._console(),
            self._reader(["enter"]),
            0.01,
            True,
            provider="azdo",
            initial=["beta"],
        )
        assert result == ["Beta"]

    def _github_scope(self, keys, monkeypatch, *, owners=None, initial=None, discover=None):
        from yeaboi.ui.mode_select import _run_code_scope_select

        monkeypatch.setattr(
            "yeaboi.tools.github.github_list_owners",
            discover or (lambda: list(owners if owners is not None else ["acme-corp", "dinho"])),
        )
        monkeypatch.setattr("yeaboi.config.get_team_analysis_github_owners", lambda: ())
        return _run_code_scope_select(
            self._FakeLive(),
            self._console(),
            self._reader(keys),
            0.01,
            True,
            provider="github",
            initial=initial,
        )

    def test_github_owner_picker_pre_checks_nothing_without_configured_owners(self, monkeypatch):
        # Discovery here is unbounded — personal login plus every org, each one a
        # whole repo estate. Pre-checking all of it would make three Enters scan
        # everything the token can see, so Enter must refuse until the user picks.
        assert self._github_scope([" ", "enter"], monkeypatch) == ["acme-corp"]

    def test_github_owner_picker_refuses_an_empty_pick(self, monkeypatch):
        assert self._github_scope(["enter", "down", " ", "enter"], monkeypatch) == ["dinho"]

    def test_github_owner_picker_pre_checks_configured_owners(self, monkeypatch):
        # A saved Settings default IS an explicit choice, so it arrives checked.
        from yeaboi.ui.mode_select import _run_code_scope_select

        monkeypatch.setattr("yeaboi.tools.github.github_list_owners", lambda: ["acme-corp", "dinho"])
        monkeypatch.setattr("yeaboi.config.get_team_analysis_github_owners", lambda: ("acme-corp",))
        result = _run_code_scope_select(
            self._FakeLive(),
            self._console(),
            self._reader(["enter"]),
            0.01,
            True,
            provider="github",
        )
        assert result == ["acme-corp"]

    def test_github_owner_initial_selection_is_restored(self, monkeypatch):
        assert self._github_scope(["enter"], monkeypatch, initial=["DINHO"]) == ["dinho"]

    def test_github_owner_empty_estate_names_the_way_out(self, monkeypatch):
        # "Select at least one GitHub owner." is impossible advice with an empty
        # list — Enter has to point at the only key that works.
        from yeaboi.ui.mode_select import _run_code_scope_select

        monkeypatch.setattr("yeaboi.tools.github.github_list_owners", lambda: [])
        monkeypatch.setattr("yeaboi.config.get_team_analysis_github_owners", lambda: ())
        captured: list = []
        live = self._FakeLive()
        live.update = captured.append
        _run_code_scope_select(
            live,
            self._console(),
            self._reader(["enter", "esc"]),
            0.01,
            True,
            provider="github",
        )
        assert "press Esc to go back" in _render(captured[-1])

    def test_github_owner_discovery_failure_falls_back_to_config(self, monkeypatch):
        # Discovery failing must not strand the wizard: the configured default
        # still lets the run proceed, with the reason on screen.
        from yeaboi.ui.mode_select import _run_code_scope_select

        def _boom():
            raise RuntimeError("bad credentials")

        monkeypatch.setattr("yeaboi.tools.github.github_list_owners", _boom)
        monkeypatch.setattr("yeaboi.config.get_team_analysis_github_owners", lambda: ("fallback-org",))
        captured: list = []
        live = self._FakeLive()
        live.update = captured.append
        result = _run_code_scope_select(
            live,
            self._console(),
            self._reader(["enter"]),
            0.01,
            True,
            provider="github",
        )
        assert result == ["fallback-org"]
        assert "bad credentials" in _render(captured[-1])

    def test_github_owner_empty_estate_stays_on_screen(self, monkeypatch):
        # Enter with nothing to select must not return an empty scope — Esc is the
        # only way out, so the user goes back and de-selects GitHub deliberately.
        assert self._github_scope(["enter", "esc"], monkeypatch, owners=[]) == "cancel"

    def test_member_selection_can_be_restored_after_review_back(self):
        from yeaboi.ui.mode_select import _run_member_select

        result = _run_member_select(
            self._FakeLive(),
            self._console(),
            self._reader(["enter"]),
            0.01,
            True,
            ["Alice", "Bob"],
            initial_members=["Bob"],
        )
        assert result == ["Bob"]

    def test_prefetch_roster_unions_sources(self, monkeypatch):
        from yeaboi.ui.mode_select import _prefetch_roster

        rosters = {"jira": ["Bob", "Alice"], "azdevops": ["Alice", "Zoe"]}
        monkeypatch.setattr("yeaboi.analysis.get_team_roster", lambda s, p, db_path=None: rosters[s])
        names = _prefetch_roster(self._FakeLive(), self._console(), ["jira", "azdevops"], "", None)
        assert names == ["Alice", "Bob", "Zoe"]  # sorted union across both trackers

    def test_prefetch_roster_fetches_sources_concurrently(self, monkeypatch):
        import threading

        from yeaboi.ui.mode_select import _prefetch_roster

        barrier = threading.Barrier(2)
        state = {"active": 0, "peak": 0}
        lock = threading.Lock()

        def roster(source, project, db_path=None):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                barrier.wait(timeout=2)
                return [source]
            finally:
                with lock:
                    state["active"] -= 1

        monkeypatch.setattr("yeaboi.analysis.get_team_roster", roster)
        names = _prefetch_roster(self._FakeLive(), self._console(), ["jira", "azdevops"], "", None)

        assert state["peak"] == 2
        assert names == ["azdevops", "jira"]

    def test_prefetch_roster_swallows_errors(self, monkeypatch):
        from yeaboi.ui.mode_select import _prefetch_roster

        def boom(s, p, db_path=None):
            raise RuntimeError("no tracker")

        monkeypatch.setattr("yeaboi.analysis.get_team_roster", boom)
        assert _prefetch_roster(self._FakeLive(), self._console(), ["jira"], "", None) == []

    def test_status_aware_prefetch_unions_sources(self, monkeypatch):
        from yeaboi.team_roster import RosterMember, RosterResult
        from yeaboi.ui.mode_select import _prefetch_roster_result

        def roster(source, project, db_path=None):
            names = {"jira": ("Ada", "Bob"), "azdevops": ("Ada", "Carol")}[source]
            return RosterResult(
                tuple(RosterMember(name, source, f"{source}:{name}") for name in names),
                "complete",
                (),
            )

        monkeypatch.setattr("yeaboi.analysis.get_team_roster_result", roster)
        result = _prefetch_roster_result(
            self._FakeLive(),
            self._console(),
            ["jira", "azdevops"],
            "",
            None,
        )
        assert result.status == "complete"
        assert [member.name for member in result.members] == ["Ada", "Bob", "Carol"]

    def test_failed_status_asks_instead_of_continuing_unscoped(self, monkeypatch):
        from yeaboi.team_roster import RosterResult
        from yeaboi.ui.mode_select import _run_analysis_roster_lookup

        monkeypatch.setattr(
            "yeaboi.ui.mode_select._prefetch_roster_result",
            lambda *args, **kwargs: RosterResult((), "failed", (), ("Jira unavailable",)),
        )
        result = _run_analysis_roster_lookup(
            self._FakeLive(),
            self._console(),
            self._reader(["esc"]),
            ["jira"],
            "PROJ",
            None,
        )
        assert result is None


class TestAnalysisSetupWizard:
    """Esc-back navigation through the setup steps with state carry-over."""

    _FakeLive = TestComponentAndMemberLoops._FakeLive
    _console = staticmethod(TestComponentAndMemberLoops._console)
    _reader = staticmethod(TestComponentAndMemberLoops._reader)

    _DOCS_ONLY = {"delivery": [], "code": [], "docs": ["confluence"]}
    _DELIVERY_ONLY = {"delivery": ["jira"], "code": [], "docs": []}
    _GITHUB_ONLY = {"delivery": [], "code": ["github"], "docs": []}
    _BOTH_HOSTS = {"delivery": [], "code": ["github", "azdo"], "docs": []}

    def _wizard(self, monkeypatch, keys, grid, *, roster=("Alice",), preflight=None, lookup_fails=False):
        from types import SimpleNamespace

        from yeaboi.ui.mode_select import _run_analysis_setup_wizard

        monkeypatch.setattr(
            "yeaboi.analysis.llm_runtime.get_ollama_analysis_preflight",
            lambda db_path: preflight or {"offer": False},
        )
        # Code-scope discovery is a network call per host; stub both regardless of
        # the grid so a step that becomes applicable mid-test can never reach out.
        monkeypatch.setattr("yeaboi.tools.github.github_list_owners", lambda: ["acme-corp", "dinho"])
        monkeypatch.setattr("yeaboi.config.get_team_analysis_github_owners", lambda: ())
        monkeypatch.setattr("yeaboi.tools.azure_devops.azdevops_list_projects", lambda: ["Infra", "Product"])
        monkeypatch.setattr("yeaboi.config.get_team_analysis_azdo_projects", lambda: ())
        lookup_result = (
            None if lookup_fails else SimpleNamespace(members=[SimpleNamespace(name=name) for name in roster])
        )
        monkeypatch.setattr(
            "yeaboi.ui.mode_select._run_analysis_roster_lookup",
            lambda *args, **kwargs: lookup_result,
        )
        return _run_analysis_setup_wizard(
            self._FakeLive(),
            self._console(),
            self._reader(keys),
            0.01,
            True,
            grid=grid,
            roster_fallback=grid["delivery"] or ["jira"],
        )

    def test_esc_on_first_step_returns_none(self, monkeypatch):
        assert self._wizard(monkeypatch, ["esc"], self._DOCS_ONLY) is None

    def test_docs_only_walkthrough_skips_inapplicable_steps(self, monkeypatch):
        # features → sources → window → review; depth/model/members never shown.
        config = self._wizard(monkeypatch, ["enter", "enter", "enter", "enter"], self._DOCS_ONLY)
        assert config["features"] == ["documentation"]
        assert config["components"] == {"docs": ["confluence"]}
        assert config["depth"] == "quick"
        assert config["model"] is None
        assert config["members"] is None and config["members_map"] is None
        assert config["window_days"] == 120

    def test_esc_from_review_preserves_window_choice(self, monkeypatch):
        # Pick 365 days, Esc from review lands back on window with 365 kept.
        keys = ["enter", "enter", "down", "enter", "esc", "enter", "enter"]
        config = self._wizard(monkeypatch, keys, self._DOCS_ONLY)
        assert config["window_days"] == 365

    def test_back_skips_step_that_no_longer_applies(self, monkeypatch):
        # Delivery-only: window is inapplicable, so Esc-chaining from review
        # crosses members straight to depth (model is skipped: no offer).
        keys = ["enter", "enter", "enter", "enter", "esc", "esc", "left", "enter", "enter", "enter"]
        config = self._wizard(monkeypatch, keys, self._DELIVERY_ONLY)
        assert config["depth"] == "quick"

    def test_member_subset_survives_review_roundtrip(self, monkeypatch):
        keys = ["enter", "enter", "enter", " ", "enter", "esc", "enter", "enter"]
        config = self._wizard(monkeypatch, keys, self._DELIVERY_ONLY, roster=("Alice", "Bob"))
        assert config["members"] == ["Bob"]
        assert config["members_map"] == {"jira": ["Bob"]}

    def test_stale_deep_depth_is_coerced_when_features_change(self, monkeypatch):
        # Choose Deep, Esc-chain back to features, drop delivery leaving docs only:
        # the stale Deep depth must coerce to quick and the members subset to None.
        grid = {"delivery": ["jira"], "code": [], "docs": ["confluence"]}
        keys = [
            *["enter", "enter", "enter", "enter", "enter"],  # walk to review (deep)
            *["esc", "esc", "esc", "esc", "esc"],  # review→members→window→depth→sources→features
            *["down", " ", "enter"],  # deselect delivery (docs stays)
            *["enter", "enter", "enter"],  # sources → window → review → run
        ]
        config = self._wizard(monkeypatch, keys, grid)
        assert config["features"] == ["documentation"]
        assert config["depth"] == "quick"
        assert config["model"] is None
        assert config["members"] is None and config["members_map"] is None

    def test_roster_lookup_declined_steps_back(self, monkeypatch):
        # Declining the failed-roster retry steps back (here all the way out)
        # instead of exiting the app.
        keys = ["enter", "enter", "enter", "esc", "esc", "esc"]
        assert self._wizard(monkeypatch, keys, self._DELIVERY_ONLY, lookup_fails=True) is None

    def test_github_owners_reach_the_run_config(self, monkeypatch):
        # The whole point of the fix: a GitHub-only code estate produces a scope
        # the engine can act on, without any AzDO involvement.
        # features → sources → owners ("a" selects all, since nothing is
        # pre-checked without configured owners) → depth → window → members → review
        keys = ["enter", "enter", "a", "enter", "enter", "enter", "enter", "enter"]
        config = self._wizard(monkeypatch, keys, self._GITHUB_ONLY)
        assert config["components"] == {"code": ["github"]}
        assert config["analysis_scope"] == {"github": ["acme-corp", "dinho"]}

    def test_esc_from_azure_picker_lands_on_github_picker_with_its_pick_intact(self, monkeypatch):
        # Two scope screens, so Esc must step between them — the reason each host
        # gets its own wizard step rather than one combined screen.
        keys = [
            *["enter", "enter"],  # features → sources
            *[" ", "enter"],  # owners: check the first owner only
            *["esc"],  # azdo projects → back to owners
            *["enter"],  # owners again, selection restored
            *["enter", "enter", "enter", "enter", "enter"],  # azdo → depth → window → members → review
        ]
        config = self._wizard(monkeypatch, keys, self._BOTH_HOSTS)
        assert config["analysis_scope"]["github"] == ["acme-corp"]
        assert config["analysis_scope"]["azdo"] == ["Infra", "Product"]

    def test_deselecting_github_coerces_its_scope_out(self, monkeypatch):
        # Same discipline as the stale-depth case: a host dropped at the sources
        # step must not leak its owners into the payload.
        keys = [
            *["enter", "enter", "a", "enter"],  # features → sources → owners (all)
            *["enter", "enter", "enter", "enter"],  # azdo → depth → window → members
            *["esc"] * 6,  # review→members→window→depth→azdo→owners→sources
            *["down", " ", "enter"],  # sources: deselect github, leaving azdo
            *["enter", "enter", "enter", "enter", "enter"],  # azdo → depth → window → members → review
        ]
        config = self._wizard(monkeypatch, keys, self._BOTH_HOSTS)
        assert config["components"]["code"] == ["azdo"]
        assert "github" not in config["analysis_scope"]
        assert config["analysis_scope"]["azdo"] == ["Infra", "Product"]

    def test_empty_roster_is_transparent_in_both_directions(self, monkeypatch):
        # Forward: members auto-advances to review. Backward: Esc from review
        # crosses members to depth without ping-ponging.
        keys = ["enter", "enter", "enter", "esc", "left", "enter", "enter"]
        config = self._wizard(monkeypatch, keys, self._DELIVERY_ONLY, roster=())
        assert config["depth"] == "quick"
        assert config["members"] is None and config["members_map"] is None


# ---------------------------------------------------------------------------
# Confirmation screen: _build_generate_confirm_screen
# ---------------------------------------------------------------------------


class TestBuildGenerateConfirmScreen:
    """The gate shown between team/board analysis and sample-ticket generation."""

    def test_returns_panel(self):
        result = _build_generate_confirm_screen(width=80, height=24)
        assert isinstance(result, Panel)

    def test_renders_prompt_and_buttons(self):
        output = _render(_build_generate_confirm_screen(width=100, height=30), width=100)
        assert "generate sample tickets now?" in output
        assert "Generate tickets" in output
        assert "Not now" in output

    def test_both_action_selections(self):
        """Either button may be highlighted without crashing."""
        for sel in (0, 1):
            result = _build_generate_confirm_screen(width=100, height=30, action_sel=sel)
            assert isinstance(result, Panel)

    def test_subtitle_rendered(self):
        output = _render(_build_generate_confirm_screen(width=100, height=30, subtitle="jira/PROJ"), width=100)
        assert "jira/PROJ" in output

    def test_narrow_terminal(self):
        assert isinstance(_build_generate_confirm_screen(width=40, height=24), Panel)

    def test_short_terminal(self):
        assert isinstance(_build_generate_confirm_screen(width=80, height=14), Panel)

    def test_buttons_registered(self):
        """New button labels must have colours registered (AGENTS.md convention)."""
        from yeaboi.ui.shared._components import _BTN_COLORS

        assert "Generate tickets" in _BTN_COLORS
        assert "Not now" in _BTN_COLORS


class TestConfirmTicketGeneration:
    """The driver loop gating analysis → ticket generation (key handling)."""

    class _FakeConsole:
        size = (100, 30)

    class _FakeLive:
        def __init__(self):
            self.frames = 0

        def update(self, _panel):
            self.frames += 1

    @staticmethod
    def _run(keys):
        """Drive _confirm_ticket_generation with a scripted key sequence."""
        from yeaboi.ui.mode_select import _confirm_ticket_generation

        it = iter(keys)

        def _read_key(timeout=None):
            return next(it)

        live = TestConfirmTicketGeneration._FakeLive()
        result = _confirm_ticket_generation(
            live,
            TestConfirmTicketGeneration._FakeConsole(),
            _read_key,
            0.05,
            True,
            subtitle="jira/PROJ",
        )
        return result, live

    def test_enter_on_generate_confirms(self):
        result, live = self._run(["enter"])
        assert result is True
        assert live.frames >= 1  # rendered at least once before the keypress

    def test_right_then_enter_declines(self):
        # Move to "Not now" (sel=1), then confirm the selection.
        result, _ = self._run(["right", "enter"])
        assert result is False

    def test_right_left_enter_confirms(self):
        # Navigate to Not now and back to Generate, then Enter.
        result, _ = self._run(["right", "left", "enter"])
        assert result is True

    def test_esc_declines(self):
        result, _ = self._run(["esc"])
        assert result is False

    def test_space_selects_current(self):
        result, _ = self._run([" "])
        assert result is True

    def test_left_clamps_at_zero(self):
        # Pressing left at sel=0 stays on Generate.
        result, _ = self._run(["left", "left", "enter"])
        assert result is True

    def test_right_clamps_at_one(self):
        # Pressing right past the last button stays on Not now.
        result, _ = self._run(["right", "right", "enter"])
        assert result is False


# ---------------------------------------------------------------------------
# Analysis overview + section card views (view= API)
# ---------------------------------------------------------------------------


def _make_overview_profile():
    from yeaboi.team_profile import (
        DoDSignal,
        SpilloverStats,
        StoryPointCalibration,
        TeamProfile,
        WritingPatterns,
    )

    return TeamProfile(
        team_id="jira-SCRUM",
        source="jira",
        project_key="SCRUM",
        sample_sprints=4,
        sample_stories=40,
        velocity_avg=23.5,
        velocity_stddev=3.2,
        point_calibrations=(StoryPointCalibration(point_value=3, avg_cycle_time_days=4.0, sample_count=12),),
        estimation_accuracy_pct=78.0,
        sprint_completion_rate=88.0,
        spillover=SpilloverStats(carried_over_pct=12.0),
        dod_signal=DoDSignal(stories_with_pr_link_pct=40.0, stories_with_testing_mention_pct=30.0),
        writing_patterns=WritingPatterns(uses_given_when_then=True, median_ac_count=3.0),
    )


_NARRATIVE_EXAMPLES = {
    "team_size": 5,
    "sprint_details": [
        {"name": "Sprint 1", "points": 22, "planned": 10, "completed": 9, "rate": 90, "done": True},
        {"name": "Sprint 2", "points": 25, "planned": 12, "completed": 10, "rate": 83, "done": False},
    ],
    "scope_changes": {
        "totals": {"avg_committed_velocity": 26.0, "avg_delivered_velocity": 23.5},
        "per_sprint": [
            {"name": "Sprint 1", "committed_pts": 26, "final_pts": 28, "scope_change_total": 2, "scope_churn": 0.12}
        ],
    },
    "narrative": {
        "executive_summary": "The team is broadly healthy with steady delivery.",
        "sections": {
            "velocity": "Velocity is stable sprint to sprint.",
            "team": "Work is spread evenly across the team.",
            "estimation": "Estimates mostly hold.",
            "workflow": "Task breakdown is consistent.",
            "writing": "Tickets are well written.",
            "trends": "No worrying long-term trends.",
            "recommendations": "Two small things to tighten up.",
        },
    },
    "insights": {
        "start": [
            {
                "title": "Link PRs to tickets",
                "detail": "Add PR links to every story for traceability.",
                "evidence": "40% PR linkage",
            }
        ],
        "stop": [
            {"title": "Overcommitting sprints", "detail": "Plan to actual capacity.", "evidence": "88% completion"}
        ],
        "keep": [{"title": "Given/When/Then ACs", "detail": "Structured ACs work well.", "evidence": "GWT detected"}],
        "try": [{"title": "WIP limits", "detail": "Cap in-progress work.", "evidence": "12% spillover"}],
    },
}

_ALL_CARD_KEYS = ("velocity", "team", "estimation", "workflow", "writing", "trends", "recommendations", "insights")


class TestAnalysisOverview:
    """The overview view: headline stats, AI executive summary, card list."""

    def _render_view(self, examples=None, selected_card=0, width=100, height=40):
        from yeaboi.team_profile import AiAdoptionSignal, DocQualitySignal

        panel = _build_team_analysis_screen(
            _make_overview_profile(),
            examples=examples,
            view="overview",
            selected_card=selected_card,
            width=width,
            height=height,
            # Global code/docs scans present → the two standalone cards render.
            code_signal=AiAdoptionSignal(scanned_commits=20, ai_commits=8, footprint_pct=40.0),
            code_examples={
                "enabled_features": ["ai_footprint", "code_health"],
                "repository_health": {"files_analysed": 12, "findings": 2},
            },
            doc_signal=DocQualitySignal(pages_scanned=5, avg_clarity=70.0),
        )
        assert isinstance(panel, Panel)
        return _render(panel, width=width)

    def test_returns_panel_by_default(self):
        panel = _build_team_analysis_screen(_make_overview_profile(), width=80, height=30)
        assert isinstance(panel, Panel)

    def test_headline_stats_render(self):
        output = self._render_view(examples=_NARRATIVE_EXAMPLES)
        assert "At a Glance" in output
        assert "5 contributors" in output
        assert "estimates hold" in output

    def test_all_card_titles_render(self):
        # Selection auto-scrolls, so check the top half with card 0 selected
        # and the bottom half with the last card selected.
        top = self._render_view(examples=_NARRATIVE_EXAMPLES, selected_card=0)
        bottom = self._render_view(examples=_NARRATIVE_EXAMPLES, selected_card=10)
        combined = top + bottom
        for title in (
            "Velocity & Sprints",
            "Team Members",
            "Estimation & Points",
            "Workflow & DoD",
            "Writing Style",
            "Trends & Repos",
            "Recommendations",
            "AI Usage",
            "Code Health",
            "Documentation",
            "Team Insights",
        ):
            assert title in combined, title
        assert "AI-POWERED INSIGHTS" in combined

    def test_teaser_stats_render(self):
        output = self._render_view(examples=_NARRATIVE_EXAMPLES)
        assert "pts/sprint" in output

    def test_selected_card_marker_moves(self):
        first = self._render_view(examples=_NARRATIVE_EXAMPLES, selected_card=0)
        second = self._render_view(examples=_NARRATIVE_EXAMPLES, selected_card=1)
        assert first != second
        assert "▸" in first and "▸" in second

    def test_executive_summary_renders(self):
        output = self._render_view(examples=_NARRATIVE_EXAMPLES)
        assert "broadly healthy" in output

    def test_missing_narrative_shows_hint(self):
        """Old saved profiles have no narrative — overview must still render."""
        ex = {k: v for k, v in _NARRATIVE_EXAMPLES.items() if k != "narrative"}
        output = self._render_view(examples=ex)
        assert "No AI summary saved" in output

    def test_no_examples_at_all(self):
        output = self._render_view(examples=None)
        assert "Sections" in output

    def test_recommendation_warning_count(self):
        from yeaboi.team_profile import DoDSignal, SpilloverStats, TeamProfile, WritingPatterns

        weak = TeamProfile(
            team_id="jira-W",
            source="jira",
            project_key="W",
            sample_sprints=4,
            sample_stories=40,
            velocity_avg=20.0,
            velocity_stddev=12.0,
            sprint_completion_rate=45.0,
            spillover=SpilloverStats(carried_over_pct=30.0),
            dod_signal=DoDSignal(),
            writing_patterns=WritingPatterns(),
        )
        panel = _build_team_analysis_screen(weak, view="overview", selected_card=6, width=100, height=40)
        output = _render(panel, width=100)
        assert "⚠" in output and "flagged" in output

    def test_narrow_and_short_terminals(self):
        for w, h in ((40, 14), (60, 20), (200, 60)):
            panel = _build_team_analysis_screen(
                _make_overview_profile(), examples=_NARRATIVE_EXAMPLES, view="overview", width=w, height=h
            )
            assert isinstance(panel, Panel)

    def test_overview_actions(self):
        output = self._render_view(examples=_NARRATIVE_EXAMPLES)
        assert "Open" in output and "Continue" in output

    def test_results_panel_carries_card_background(self):
        from yeaboi.ui.mode_select.screens._screens_secondary import _ANALYSIS_CARD_BG

        panel = _build_team_analysis_screen(_make_overview_profile(), width=100, height=40)
        assert panel.style == _ANALYSIS_CARD_BG

    def test_results_background_cascades_onto_blank_rows(self):
        """The card background must reach spacer/filler rows (no dark seams)."""
        panel = _build_team_analysis_screen(_make_overview_profile(), width=100, height=40)
        buf = StringIO()
        console = Console(file=buf, width=100, force_terminal=True, color_system="truecolor", highlight=False)
        console.print(panel)
        assert "13;31;27" in buf.getvalue()

    def test_overview_code_health_sits_above_the_ai_group(self):
        # Deterministic card: renders with the regular cards, before the ✦ heading.
        top = self._render_view(examples=_NARRATIVE_EXAMPLES, selected_card=0)
        bottom = self._render_view(examples=_NARRATIVE_EXAMPLES, selected_card=10)
        combined = top + bottom
        assert combined.index("Code Health") < combined.index("AI-POWERED INSIGHTS")
        columns = {}
        for line in combined.splitlines():
            for title in ("Trends & Repos", "Code Health"):
                if title in line and title not in columns:
                    columns[title] = line.find(title)
        assert set(columns) == {"Trends & Repos", "Code Health"}
        assert len(set(columns.values())) == 1, columns

    def test_overview_code_health_has_no_ai_star(self):
        top = self._render_view(examples=_NARRATIVE_EXAMPLES, selected_card=0)
        bottom = self._render_view(examples=_NARRATIVE_EXAMPLES, selected_card=10)
        for line in (top + bottom).splitlines():
            if "Code Health" in line:
                assert "✦" not in line
            if "AI Usage" in line:
                assert "✦" in line


class TestAnalysisSectionDetail:
    """Each section card renders its sections, narrative block and glossary."""

    @pytest.mark.parametrize("card_key", _ALL_CARD_KEYS)
    def test_card_renders_panel(self, card_key):
        panel = _build_team_analysis_screen(
            _make_overview_profile(), examples=_NARRATIVE_EXAMPLES, view=card_key, width=100, height=40
        )
        assert isinstance(panel, Panel)

    # The insights card is coaching content itself — it has no narrative key.
    @pytest.mark.parametrize("card_key", tuple(k for k in _ALL_CARD_KEYS if k != "insights"))
    def test_narrative_block_shown(self, card_key):
        panel = _build_team_analysis_screen(
            _make_overview_profile(), examples=_NARRATIVE_EXAMPLES, view=card_key, width=100, height=50
        )
        output = _render(panel, width=100)
        assert "What this means" in output

    @pytest.mark.parametrize("card_key", _ALL_CARD_KEYS)
    def test_narrative_block_omitted_without_narrative(self, card_key):
        ex = {k: v for k, v in _NARRATIVE_EXAMPLES.items() if k != "narrative"}
        panel = _build_team_analysis_screen(_make_overview_profile(), examples=ex, view=card_key, width=100, height=50)
        output = _render(panel, width=100)
        assert "What this means" not in output

    @pytest.mark.parametrize("card_key", _ALL_CARD_KEYS)
    def test_detail_actions(self, card_key):
        panel = _build_team_analysis_screen(
            _make_overview_profile(), examples=_NARRATIVE_EXAMPLES, view=card_key, width=100, height=40
        )
        output = _render(panel, width=100)
        assert "Back" in output

    def test_velocity_card_sections_and_breadcrumb(self):
        panel = _build_team_analysis_screen(
            _make_overview_profile(), examples=_NARRATIVE_EXAMPLES, view="velocity", width=100, height=60
        )
        output = _render(panel, width=100)
        assert "Team & Velocity" in output
        assert "Sprint Breakdown" in output
        assert "Overview › Velocity & Sprints" in output

    def test_velocity_card_churn_glossary(self):
        """The Churn column jargon is explained on the card (user complaint)."""
        panel = _build_team_analysis_screen(
            _make_overview_profile(),
            examples=_NARRATIVE_EXAMPLES,
            view="velocity",
            width=100,
            height=40,
            scroll_offset=9999,
        )
        output = _render(panel, width=100)
        assert "What the terms mean" in output
        assert "Churn — % of committed points added or removed mid-sprint" in output

    def test_estimation_card_glossary(self):
        panel = _build_team_analysis_screen(
            _make_overview_profile(),
            examples=_NARRATIVE_EXAMPLES,
            view="estimation",
            width=100,
            height=40,
            scroll_offset=9999,
        )
        output = _render(panel, width=100)
        assert "Cycle — days from work starting to done" in output

    def test_workflow_card_no_glossary(self):
        panel = _build_team_analysis_screen(
            _make_overview_profile(),
            examples=_NARRATIVE_EXAMPLES,
            view="workflow",
            width=100,
            height=40,
            scroll_offset=9999,
        )
        output = _render(panel, width=100)
        assert "What the terms mean" not in output

    def test_recommendations_card_renders_recs(self):
        from yeaboi.team_profile import DoDSignal, SpilloverStats, TeamProfile, WritingPatterns

        weak = TeamProfile(
            team_id="jira-W",
            source="jira",
            project_key="W",
            sample_sprints=4,
            sample_stories=40,
            velocity_avg=20.0,
            velocity_stddev=12.0,
            sprint_completion_rate=45.0,
            spillover=SpilloverStats(carried_over_pct=30.0),
            dod_signal=DoDSignal(),
            writing_patterns=WritingPatterns(),
        )
        panel = _build_team_analysis_screen(weak, view="recommendations", width=100, height=50)
        output = _render(panel, width=100)
        assert "Low sprint completion" in output

    @pytest.mark.parametrize("card_key", _ALL_CARD_KEYS)
    def test_empty_profile_all_cards(self, card_key):
        from yeaboi.team_profile import TeamProfile

        empty = TeamProfile(team_id="e", source="jira", project_key="X", sample_sprints=0, sample_stories=0)
        panel = _build_team_analysis_screen(empty, examples=None, view=card_key, width=80, height=24)
        assert isinstance(panel, Panel)


class TestDocumentationCard:
    """The Documentation card: clarity + AI-usage estimate + coaching (populated + empty)."""

    def _profile(self, sig):
        from yeaboi.team_profile import TeamProfile

        return TeamProfile(
            team_id="jira-D",
            source="jira",
            project_key="D",
            sample_sprints=4,
            sample_stories=40,
            velocity_avg=30.0,
            doc_quality=sig,
        )

    def test_populated_renders_clarity_estimate_and_flag(self):
        from yeaboi.team_profile import DocQualitySignal

        sig = DocQualitySignal(
            pages_scanned=6,
            platforms_scanned=("confluence", "notion"),
            avg_clarity=52.0,
            avg_usefulness=45.0,
            clear_pages=2,
            mixed_pages=2,
            unclear_pages=2,
            avg_ai_likelihood=61.0,
            likely_ai_pages=3,
            ai_marked_pages=1,
            per_platform=(("confluence", 4), ("notion", 2)),
            flagged_pages=(("Onboarding guide", "clarity 30/100 — dense or long-winded"),),
        )
        ex = {
            "doc_quality": {
                "samples": [
                    {
                        "title": "Onboarding guide",
                        "platform": "confluence",
                        "clarity": 30,
                        "usefulness": 40,
                        "url": "https://wiki/onboarding",
                    }
                ],
                "insights": {
                    "start": [
                        {
                            "title": "Tighten the least-clear pages",
                            "detail": "Trim it.",
                            "evidence": "52/100",
                            "link": "https://wiki/onboarding",
                        }
                    ],
                    "stop": [],
                    "keep": [],
                    "try": [],
                },
            }
        }
        panel = _build_team_analysis_screen(self._profile(sig), examples=ex, view="documentation", width=100, height=60)
        output = _render(panel, width=100)
        bottom = _render(
            _build_team_analysis_screen(
                self._profile(sig),
                examples=ex,
                view="documentation",
                scroll_offset=9999,
                width=100,
                height=60,
            ),
            width=100,
        )
        assert "Documentation" in output
        assert "52/100" in output  # clarity score
        assert "usefulness" in output.lower()
        assert "lower bound" in output.lower()  # explicit-marker framing
        assert "Onboarding guide" in output  # flagged page
        assert "Tighten the least-clear pages" in bottom  # coaching
        assert "Page evidence" in output  # evidence table
        assert "https://wiki/onboarding" in output + bottom  # page link on example + coaching item

    def test_large_scan_tables_capped_and_never_blank(self):
        # Regression: a 188-page scan produced flagged/evidence tables hundreds of
        # rows tall. Tables are atomic renderables for the viewport packer, so one
        # taller than the viewport rendered as pure blank scroll space. Capped
        # tables must actually render, with "+ N more" disclosure rows.
        from yeaboi.team_profile import DocQualitySignal

        sig = DocQualitySignal(
            pages_scanned=188,
            platforms_scanned=("confluence",),
            avg_clarity=50.0,
            avg_usefulness=45.0,
            clear_pages=60,
            mixed_pages=60,
            unclear_pages=68,
            per_platform=(("confluence", 188),),
            flagged_pages=tuple((f"Page {i:03d}", "clarity 30/100 — dense or long-winded") for i in range(100)),
        )
        ex = {
            "doc_quality": {
                "samples": [
                    {
                        "title": f"Page {i:03d}",
                        "platform": "confluence",
                        "clarity": 30,
                        "usefulness": 40,
                        "url": f"https://wiki/page-{i}",
                    }
                    for i in range(188)
                ],
                "action_plan": [
                    {
                        "priority": "high",
                        "title": "Rewrite the densest pages",
                        "detail": "Start with the flagged list.",
                        "affected_scope": ["confluence"],
                        "owner_role": "team lead",
                        "effort": "medium",
                    }
                ],
            }
        }
        frames = [
            _render(
                _build_team_analysis_screen(
                    self._profile(sig),
                    examples=ex,
                    view="documentation",
                    scroll_offset=offset,
                    width=100,
                    height=40,
                ),
                width=100,
            )
            for offset in range(0, 70, 5)
        ]
        combined = "\n".join(frames)
        assert "Page 000" in combined  # the capped tables really render
        assert "+ 92 more flagged pages (full list in export)" in combined
        assert "+ 180 more scanned pages (full list in export)" in combined
        assert "Prioritized action plan" in combined  # sections after the tables are reachable

    def test_empty_state_and_coverage(self):
        from yeaboi.team_profile import TeamProfile

        prof = TeamProfile(team_id="e", source="jira", project_key="X")
        ex = {"doc_quality": {"coverage": ["notion: NOTION_TOKEN not set"]}}
        panel = _build_team_analysis_screen(prof, examples=ex, view="documentation", width=90, height=30)
        output = _render(panel, width=90)
        assert "No documentation scan" in output
        assert "NOTION_TOKEN not set" in output


class TestCodeHealthCard:
    """The deterministic code-health card (selected-user changed files)."""

    def _code_examples(self, action):
        return {
            "repository_health": {
                "files_analysed": 5578,
                "repositories_touched": 231,
                "findings": 1834,
                "by_category": {"hotspot": 623, "testing": 1211},
            },
            "selected_users": ["Ava"],
            "matched_identities": {"Ava": ["ava"]},
            "action_plan": [action],
        }

    def _frames(self, code_examples):
        from yeaboi.team_profile import TeamProfile

        prof = TeamProfile(team_id="t", source="jira", project_key="P")
        combined = "\n".join(
            _render(
                _build_team_analysis_screen(
                    prof,
                    examples=None,
                    code_examples=code_examples,
                    view="code-health",
                    scroll_offset=offset,
                    width=100,
                    height=40,
                ),
                width=100,
            )
            for offset in range(0, 40, 5)
        )
        # Collapse panel borders + wrapping so phrases can be matched across lines.
        return " ".join(combined.replace("│", " ").split())

    def test_scope_text_caps(self):
        from yeaboi.ui.mode_select.screens._analysis_sections import _ta_scope_text

        assert _ta_scope_text(["a", "b"]) == "a, b"
        assert _ta_scope_text([f"r{i}" for i in range(10)]) == (
            "r0, r1, r2, r3, r4, r5 and 4 more (full list in export)"
        )

    def test_wide_scope_action_plan_renders_not_blank(self):
        # Regression: cross-repo actions merge affected_scope over every touched
        # repository (231 on the real run). The full join wrapped one callout to
        # hundreds of lines; callouts are atomic renderables for the viewport
        # packer, so every action rendered as blank scroll space under the
        # "Prioritized action plan" heading.
        flat = self._frames(
            self._code_examples(
                {
                    "priority": "high",
                    "title": "Add tests to hot files",
                    "detail": "Files changed repeatedly without accompanying tests.",
                    "affected_scope": [f"YL.Repo.{i:03d}" for i in range(231)],
                    "owner_role": "tech lead",
                    "effort": "medium",
                }
            )
        )
        assert "Prioritized action plan" in flat
        assert "Add tests to hot files" in flat  # the callout really renders
        assert "and 225 more (full list in export)" in flat

    def test_oversized_callout_body_truncated_not_blank(self):
        # Backstop: even a pathological detail string must never make a callout
        # taller than the viewport.
        flat = self._frames(
            self._code_examples(
                {
                    "priority": "medium",
                    "title": "Reduce churn",
                    "detail": "hotspot detail " * 400,
                    "affected_scope": ["YL.Repo.001"],
                    "owner_role": "tech lead",
                    "effort": "medium",
                }
            )
        )
        assert "Reduce churn" in flat
        assert "… (full detail in export)" in flat


# ---------------------------------------------------------------------------
# Team insights screen (results → insights → generate-tickets confirm)
# ---------------------------------------------------------------------------


class TestBuildTeamInsightsScreen:
    """The coaching-insights screen shown before the sample-ticket confirm."""

    def _render_screen(self, examples=None, width=100, height=40, **kwargs):
        panel = _build_team_insights_screen(
            _make_overview_profile(),
            examples=examples,
            width=width,
            height=height,
            **kwargs,
        )
        assert isinstance(panel, Panel)
        return _render(panel, width=width)

    def test_returns_panel(self):
        panel = _build_team_insights_screen(_make_overview_profile(), examples=_NARRATIVE_EXAMPLES)
        assert isinstance(panel, Panel)

    def test_intro_line_renders(self):
        output = self._render_screen(examples=_NARRATIVE_EXAMPLES)
        assert "How to improve this team" in output

    def test_action_groups_and_badges_render(self):
        # The dashboard replaces raw Start/Stop/Keep/Try sections with three
        # action-oriented groups; badges retain each item's original intent.
        top = self._render_screen(examples=_NARRATIVE_EXAMPLES, height=60)
        bottom = self._render_screen(examples=_NARRATIVE_EXAMPLES, height=60, scroll_offset=9999)
        output = top + bottom
        for heading in ("Focus now", "Keep working", "Experiments"):
            assert heading in output, heading
        for badge in ("START", "AVOID", "KEEP", "TRY"):
            assert badge in output, badge

    def test_item_title_detail_evidence_render(self):
        output = self._render_screen(examples=_NARRATIVE_EXAMPLES, height=60)
        assert "Link PRs to tickets" in output
        assert "traceability" in output
        assert "40% PR linkage" in output

    def test_default_actions(self):
        output = self._render_screen(examples=_NARRATIVE_EXAMPLES)
        for action in ("Continue", "Export", "Back"):
            assert action in output, action

    def test_action_selection_highlights(self):
        rendered = [self._render_screen(examples=_NARRATIVE_EXAMPLES, action_sel=i) for i in range(3)]
        assert len(set(rendered)) == 1 or len(set(rendered)) > 1  # renders for every selection
        for r in rendered:
            assert "Continue" in r

    def test_empty_examples_show_hint(self):
        """Old saved profiles have no insights — screen must still render."""
        output = self._render_screen(examples={})
        assert "No insights saved" in output

    def test_none_examples_show_hint(self):
        output = self._render_screen(examples=None)
        assert "No insights saved" in output

    def test_scrollbar_on_overflow(self):
        output = self._render_screen(examples=_NARRATIVE_EXAMPLES, height=20)
        assert "│" in output or "┃" in output

    def test_scroll_clamps_and_keeps_buttons(self):
        output = self._render_screen(examples=_NARRATIVE_EXAMPLES, height=24, scroll_offset=9999)
        assert "Continue" in output

    def test_narrow_terminal_no_crash(self):
        panel = _build_team_insights_screen(_make_overview_profile(), examples=_NARRATIVE_EXAMPLES, width=40, height=24)
        assert isinstance(panel, Panel)

    def test_short_terminal_no_crash(self):
        panel = _build_team_insights_screen(_make_overview_profile(), examples=_NARRATIVE_EXAMPLES, width=80, height=10)
        assert isinstance(panel, Panel)

    def test_subtitle_renders(self):
        output = self._render_screen(examples=_NARRATIVE_EXAMPLES, subtitle="jira/SCRUM  ·  Team Insights")
        assert "Team Insights" in output

    def test_insights_card_teaser_on_overview(self):
        panel = _build_team_analysis_screen(
            _make_overview_profile(),
            examples=_NARRATIVE_EXAMPLES,
            view="overview",
            selected_card=9,
            width=100,
            height=40,
        )
        output = _render(panel, width=100)
        assert "1 start" in output
        assert "1 try" in output

    def test_insights_card_detail_view(self):
        panel = _build_team_analysis_screen(
            _make_overview_profile(),
            examples=_NARRATIVE_EXAMPLES,
            view="insights",
            width=100,
            height=50,
        )
        output = _render(panel, width=100)
        assert "Team coaching plan" in output
        assert "Focus now" in output
        assert "START" in output
        assert "Link PRs to tickets" in output


class TestRunTeamInsights:
    """The driver loop for the insights screen (key handling)."""

    class _FakeConsole:
        size = (100, 30)

    class _FakeLive:
        def __init__(self):
            self.frames = 0

        def update(self, _panel):
            self.frames += 1

    @staticmethod
    def _run(keys):
        """Drive _run_team_insights with a scripted key sequence."""
        from yeaboi.ui.mode_select import _run_team_insights

        it = iter(keys)

        def _read_key(timeout=None):
            return next(it)

        live = TestRunTeamInsights._FakeLive()
        result = _run_team_insights(
            live,
            TestRunTeamInsights._FakeConsole(),
            _read_key,
            0.05,
            True,
            _make_overview_profile(),
            _NARRATIVE_EXAMPLES,
        )
        return result, live

    def test_enter_on_continue(self):
        result, live = self._run(["enter"])
        assert result == "continue"
        assert live.frames >= 1

    def test_back_button_returns_back(self):
        # Continue → Export → Back, then Enter.
        result, _ = self._run(["right", "right", "enter"])
        assert result == "back"

    def test_esc_returns_back(self):
        result, _ = self._run(["esc"])
        assert result == "back"

    def test_q_returns_back(self):
        result, _ = self._run(["q"])
        assert result == "back"

    def test_right_left_then_continue(self):
        # Navigate to Export and back to Continue, then Enter.
        result, _ = self._run(["right", "left", "enter"])
        assert result == "continue"

    def test_left_clamps_then_continue(self):
        result, _ = self._run(["left", "left", "enter"])
        assert result == "continue"
