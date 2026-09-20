"""Tests for project analyzer node and its helper functions."""

from unittest.mock import MagicMock

import pytest

try:
    import pymupdf  # noqa: F401

    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False
from langchain_core.messages import AIMessage, HumanMessage

from tests._node_helpers import (
    VALID_ANALYSIS_JSON,
    make_completed_questionnaire,
    make_dummy_analysis,
)
from yeaboi.agent.nodes import (
    _build_answers_block,
    _build_fallback_analysis,
    _extract_confluence_page_ids,
    _fetch_confluence_context,
    _format_analysis,
    _load_user_context,
    _parse_analysis_response,
    _scan_repo_context,
    compute_prompt_quality,
    project_analyzer,
)
from yeaboi.agent.state import (
    TOTAL_QUESTIONS,
    ProjectAnalysis,
    QuestionnaireState,
)
from yeaboi.prompts.intake import QUESTION_DEFAULTS


class TestBuildAnswersBlock:
    """Tests for _build_answers_block() helper."""

    def test_includes_all_questions(self):
        """All 26 questions should appear in the output."""
        qs = make_completed_questionnaire()
        block = _build_answers_block(qs)
        for i in range(1, TOTAL_QUESTIONS + 1):
            assert f"Q{i}." in block

    def test_includes_answers(self):
        """Each answer should appear next to its question."""
        qs = make_completed_questionnaire()
        block = _build_answers_block(qs)
        assert "Answer for Q1" in block
        assert "Answer for Q26" in block

    def test_marks_defaulted_questions(self):
        """Defaulted questions should be marked with *(assumed default)*."""
        qs = make_completed_questionnaire()
        qs.defaulted_questions.add(5)
        block = _build_answers_block(qs)
        # Q5 should have the marker
        lines = block.split("\n")
        q5_answer_line = [line for line in lines if "Answer for Q5" in line]
        assert q5_answer_line
        assert "*(assumed default)*" in q5_answer_line[0]

    def test_marks_skipped_questions(self):
        """Questions with no answer should be marked with *(skipped)*."""
        qs = make_completed_questionnaire()
        del qs.answers[3]  # Remove Q3 answer
        block = _build_answers_block(qs)
        assert "(no answer)" in block
        assert "*(skipped)*" in block

    def test_returns_string(self):
        """Output should be a non-empty string."""
        qs = make_completed_questionnaire()
        block = _build_answers_block(qs)
        assert isinstance(block, str)
        assert len(block) > 0


class TestParseAnalysisResponse:
    """Tests for _parse_analysis_response() helper."""

    def _qs(self) -> QuestionnaireState:
        return make_completed_questionnaire()

    def test_valid_json_returns_project_analysis(self):
        """Valid JSON should produce a ProjectAnalysis dataclass."""
        result = _parse_analysis_response(VALID_ANALYSIS_JSON, self._qs(), 3, 15)
        assert isinstance(result, ProjectAnalysis)
        assert result.project_name == "Todo App"
        assert result.project_type == "greenfield"
        assert result.sprint_length_weeks == 2
        assert result.target_sprints == 4

    def test_lists_become_tuples(self):
        """JSON arrays should be converted to tuples (frozen dataclass requirement)."""
        result = _parse_analysis_response(VALID_ANALYSIS_JSON, self._qs(), 3, 15)
        assert isinstance(result.goals, tuple)
        assert isinstance(result.tech_stack, tuple)
        assert result.goals == ("Task management", "User authentication")

    def test_code_fence_stripping(self):
        """JSON wrapped in markdown code fences should be handled."""
        fenced = f"```json\n{VALID_ANALYSIS_JSON}\n```"
        result = _parse_analysis_response(fenced, self._qs(), 3, 15)
        assert result.project_name == "Todo App"

    def test_bad_json_returns_fallback(self):
        """Invalid JSON should fall back to deterministic extraction."""
        result = _parse_analysis_response("this is not json", self._qs(), 3, 15)
        assert isinstance(result, ProjectAnalysis)
        # Fallback uses Q1 answer as project name
        assert "Answer for Q1" in result.project_name

    def test_empty_response_returns_fallback(self):
        """Empty response should fall back."""
        result = _parse_analysis_response("", self._qs(), 3, 15)
        assert isinstance(result, ProjectAnalysis)

    def test_non_dict_json_returns_fallback(self):
        """JSON that's not a dict (e.g. array) should fall back."""
        result = _parse_analysis_response("[1, 2, 3]", self._qs(), 3, 15)
        assert isinstance(result, ProjectAnalysis)

    def test_missing_fields_use_defaults(self):
        """JSON with missing fields should use sensible defaults."""
        minimal = '{"project_name": "Mini"}'
        result = _parse_analysis_response(minimal, self._qs(), 3, 15)
        assert result.project_name == "Mini"
        assert result.sprint_length_weeks == 2  # default
        assert result.target_sprints == 0  # default

    def test_skip_features_true_when_small_project(self):
        """skip_features=true should be kept when project is small (≤2 sprints, ≤3 goals)."""
        # Build a small project JSON: 1 sprint, 2 goals
        small_json = """{
            "project_name": "Tiny API",
            "project_description": "A simple REST endpoint",
            "project_type": "greenfield",
            "goals": ["Serve data", "Auth"],
            "end_users": ["developers"],
            "target_state": "Deployed",
            "tech_stack": ["FastAPI"],
            "integrations": [],
            "constraints": [],
            "sprint_length_weeks": 2,
            "target_sprints": 1,
            "risks": [],
            "out_of_scope": [],
            "assumptions": [],
            "skip_features": true
        }"""
        result = _parse_analysis_response(small_json, self._qs(), 1, 10)
        assert result.skip_features is True

    def test_skip_features_overridden_when_many_sprints(self):
        """skip_features=true should be forced to False when target_sprints > 2."""
        json_with_skip = VALID_ANALYSIS_JSON.rstrip("}") + ', "skip_features": true}'
        result = _parse_analysis_response(json_with_skip, self._qs(), 3, 15)
        # VALID_ANALYSIS_JSON has target_sprints=4, so guardrail overrides
        assert result.skip_features is False

    def test_skip_features_overridden_when_many_goals(self):
        """skip_features=true should be forced to False when goals > 3."""
        many_goals_json = """{
            "project_name": "Big App",
            "project_description": "A complex app",
            "project_type": "greenfield",
            "goals": ["Goal 1", "Goal 2", "Goal 3", "Goal 4"],
            "end_users": ["users"],
            "target_state": "Done",
            "tech_stack": ["React"],
            "integrations": [],
            "constraints": [],
            "sprint_length_weeks": 2,
            "target_sprints": 2,
            "risks": [],
            "out_of_scope": [],
            "assumptions": [],
            "skip_features": true
        }"""
        result = _parse_analysis_response(many_goals_json, self._qs(), 3, 15)
        assert result.skip_features is False

    def test_skip_features_false_extracted(self):
        """skip_features=false in JSON should be extracted as False."""
        json_with_skip = VALID_ANALYSIS_JSON.rstrip("}") + ', "skip_features": false}'
        result = _parse_analysis_response(json_with_skip, self._qs(), 3, 15)
        assert result.skip_features is False

    def test_skip_features_missing_defaults_false(self):
        """Missing skip_features should default to False."""
        result = _parse_analysis_response(VALID_ANALYSIS_JSON, self._qs(), 3, 15)
        assert result.skip_features is False


class TestBuildFallbackAnalysis:
    """Tests for _build_fallback_analysis() helper."""

    def test_returns_project_analysis(self):
        """Should return a ProjectAnalysis dataclass."""
        qs = make_completed_questionnaire()
        result = _build_fallback_analysis(qs, 3, 15)
        assert isinstance(result, ProjectAnalysis)

    def test_uses_q1_as_name(self):
        """project_name should come from Q1."""
        qs = make_completed_questionnaire()
        result = _build_fallback_analysis(qs, 3, 15)
        assert "Answer for Q1" in result.project_name

    def test_parses_sprint_length_from_q8(self):
        """Sprint length should be parsed from Q8."""
        qs = make_completed_questionnaire()
        qs.answers[8] = "2 weeks"
        result = _build_fallback_analysis(qs, 3, 15)
        assert result.sprint_length_weeks == 2

    def test_default_sprint_length_when_unparseable(self):
        """Sprint length should default to 2 when Q8 is not parseable."""
        qs = make_completed_questionnaire()
        qs.answers[8] = "not a number"
        result = _build_fallback_analysis(qs, 3, 15)
        assert result.sprint_length_weeks == 2

    def test_tracks_defaulted_assumptions(self):
        """Defaulted questions should appear in assumptions."""
        qs = make_completed_questionnaire()
        qs.defaulted_questions.add(5)
        result = _build_fallback_analysis(qs, 3, 15)
        assert any("Q5" in a for a in result.assumptions)

    def test_tracks_skipped_assumptions(self):
        """Skipped questions with no answer should appear in assumptions."""
        qs = make_completed_questionnaire()
        del qs.answers[3]  # Remove answer
        qs.skipped_questions.add(3)
        result = _build_fallback_analysis(qs, 3, 15)
        assert any("Q3" in a for a in result.assumptions)


class TestFormatAnalysis:
    """Tests for _format_analysis() helper."""

    def test_returns_string(self):
        """Should return a non-empty markdown string."""
        analysis = make_dummy_analysis()
        result = _format_analysis(analysis)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_project_name(self):
        """The project name should appear in the output."""
        analysis = make_dummy_analysis(project_name="Widget Builder")
        result = _format_analysis(analysis)
        assert "Widget Builder" in result

    def test_includes_sections(self):
        """Key sections should be present."""
        analysis = make_dummy_analysis()
        result = _format_analysis(analysis)
        assert "## Goals" in result
        assert "## Tech Stack" in result
        assert "## Risks" in result
        assert "## Sprint Planning" in result

    def test_no_continue_footer(self):
        """The old 'type anything to continue' footer should be removed — review menu is used instead."""
        analysis = make_dummy_analysis()
        result = _format_analysis(analysis)
        assert "Type anything to continue" not in result

    def test_shows_assumptions_when_present(self):
        """Assumptions section should appear when there are assumptions."""
        analysis = make_dummy_analysis(assumptions=("Q5 was defaulted",))
        result = _format_analysis(analysis)
        assert "## Assumptions" in result
        assert "Q5 was defaulted" in result

    def test_no_assumptions_section_when_empty(self):
        """Assumptions section should not appear when empty."""
        analysis = make_dummy_analysis(assumptions=())
        result = _format_analysis(analysis)
        assert "## Assumptions" not in result


class TestProjectAnalyzer:
    """Tests for the project_analyzer() node function."""

    def _make_state(self, **extras: object) -> dict:
        """Build a minimal state with completed questionnaire for analyzer tests."""
        qs = make_completed_questionnaire()
        state = {
            "messages": [HumanMessage(content="continue")],
            "questionnaire": qs,
            "team_size": 3,
            "velocity_per_sprint": 15,
        }
        state.update(extras)
        return state

    def test_returns_project_analysis(self, monkeypatch):
        """project_analyzer should return a ProjectAnalysis in the state update."""
        fake_response = MagicMock()
        fake_response.content = VALID_ANALYSIS_JSON
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)

        result = project_analyzer(self._make_state())
        assert "project_analysis" in result
        assert isinstance(result["project_analysis"], ProjectAnalysis)

    def test_returns_ai_message(self, monkeypatch):
        """project_analyzer should return an AIMessage with the formatted analysis."""
        fake_response = MagicMock()
        fake_response.content = VALID_ANALYSIS_JSON
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)

        result = project_analyzer(self._make_state())
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    def test_populates_metadata_fields(self, monkeypatch):
        """project_analyzer should set project_name, project_description, sprint_length_weeks, target_sprints."""
        fake_response = MagicMock()
        fake_response.content = VALID_ANALYSIS_JSON
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)

        result = project_analyzer(self._make_state())
        assert result["project_name"] == "Todo App"
        assert result["project_description"] == "A full-stack todo application"
        assert result["sprint_length_weeks"] == 2
        assert result["target_sprints"] == 4

    def test_bad_json_uses_fallback(self, monkeypatch):
        """When LLM returns bad JSON, the fallback should produce a valid analysis."""
        fake_response = MagicMock()
        fake_response.content = "not valid json at all"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)

        result = project_analyzer(self._make_state())
        assert isinstance(result["project_analysis"], ProjectAnalysis)
        # Fallback pulls project_name from Q1 answer
        assert "Answer for Q1" in result["project_name"]

    def test_llm_exception_uses_fallback(self, monkeypatch):
        """When the LLM call raises an exception, the fallback should be used."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API down")
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)

        result = project_analyzer(self._make_state())
        assert isinstance(result["project_analysis"], ProjectAnalysis)
        assert "messages" in result

    def test_defaults_team_size_when_missing(self, monkeypatch):
        """When team_size is not in state, should default to 1."""
        fake_response = MagicMock()
        fake_response.content = VALID_ANALYSIS_JSON
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)

        state = self._make_state()
        del state["team_size"]
        del state["velocity_per_sprint"]
        result = project_analyzer(state)
        assert isinstance(result["project_analysis"], ProjectAnalysis)

    def test_pasted_images_sent_as_multimodal_blocks(self, monkeypatch, tmp_path):
        """With pasted_images in state, the LLM receives text + image content blocks."""
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        fake_response = MagicMock()
        fake_response.content = VALID_ANALYSIS_JSON
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)

        result = project_analyzer(self._make_state(pasted_images=[str(img)]))
        assert isinstance(result["project_analysis"], ProjectAnalysis)
        sent = mock_llm.invoke.call_args[0][0][0].content
        assert isinstance(sent, list)
        assert sent[0]["type"] == "text"
        assert sent[1]["type"] == "image"
        assert sent[1]["mime_type"] == "image/png"

    def test_no_images_sends_plain_string_prompt(self, monkeypatch):
        """Without pasted images the prompt stays a plain string (regression)."""
        fake_response = MagicMock()
        fake_response.content = VALID_ANALYSIS_JSON
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)

        project_analyzer(self._make_state())
        sent = mock_llm.invoke.call_args[0][0][0].content
        assert isinstance(sent, str)

    def test_missing_image_file_degrades_to_plain_prompt(self, monkeypatch, tmp_path):
        """A deleted attachment (e.g. after --resume) degrades to text-only, no crash."""
        fake_response = MagicMock()
        fake_response.content = VALID_ANALYSIS_JSON
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)

        result = project_analyzer(self._make_state(pasted_images=[str(tmp_path / "gone.png")]))
        assert isinstance(result["project_analysis"], ProjectAnalysis)
        sent = mock_llm.invoke.call_args[0][0][0].content
        assert isinstance(sent, str)


class TestScanRepoContext:
    """Tests for the _scan_repo_context() helper function."""

    def _make_qs(self, url: str = "https://github.com/owner/repo", platform: str = "GitHub") -> QuestionnaireState:
        qs = QuestionnaireState()
        qs.answers[17] = url
        qs.answers[16] = platform
        return qs

    def test_no_url_returns_none(self):
        """Returns None when Q17 is not set."""
        qs = QuestionnaireState()
        ctx, status = _scan_repo_context(qs)
        assert ctx is None
        assert status["status"] == "skipped"

    def test_empty_url_returns_none(self):
        """Returns None when Q17 is an empty string."""
        qs = QuestionnaireState()
        qs.answers[17] = ""
        ctx, status = _scan_repo_context(qs)
        assert ctx is None
        assert status["status"] == "skipped"

    def test_default_url_value_returns_none(self):
        """Returns None when Q17 holds the 'No repo URL provided' default."""
        qs = QuestionnaireState()
        qs.answers[17] = QUESTION_DEFAULTS[17]
        ctx, status = _scan_repo_context(qs)
        assert ctx is None
        assert status["status"] == "skipped"

    def test_gitlab_platform_returns_none(self):
        """Returns None for GitLab (no tools implemented yet)."""
        qs = self._make_qs("https://gitlab.com/owner/repo", "GitLab")
        ctx, status = _scan_repo_context(qs)
        assert ctx is None
        assert status["status"] == "skipped"

    def test_unsupported_platform_returns_none(self):
        """Returns None for Bitbucket (no tools implemented yet)."""
        qs = self._make_qs("https://bitbucket.org/owner/repo", "Bitbucket")
        ctx, status = _scan_repo_context(qs)
        assert ctx is None
        assert status["status"] == "skipped"

    def test_github_calls_read_repo_and_readme(self, monkeypatch):
        """GitHub platform calls github_read_repo and github_read_readme."""
        mock_repo = MagicMock()
        mock_repo.invoke.return_value = "## File Tree\n- src/"
        mock_readme = MagicMock()
        mock_readme.invoke.return_value = "# MyProject\nA great project."

        monkeypatch.setattr("yeaboi.tools.github.github_read_repo", mock_repo)
        monkeypatch.setattr("yeaboi.tools.github.github_read_readme", mock_readme)

        qs = self._make_qs("https://github.com/owner/repo", "GitHub")
        result, status = _scan_repo_context(qs)

        mock_repo.invoke.assert_called_once_with({"repo_url": "https://github.com/owner/repo"})
        mock_readme.invoke.assert_called_once_with({"repo_url": "https://github.com/owner/repo"})
        assert result is not None
        assert "File Tree" in result
        assert "MyProject" in result
        assert status["status"] == "success"

    def test_github_combines_results_with_separator(self, monkeypatch):
        """GitHub results are joined with '---' separator."""
        mock_repo = MagicMock()
        mock_repo.invoke.return_value = "tree data"
        mock_readme = MagicMock()
        mock_readme.invoke.return_value = "readme data"

        monkeypatch.setattr("yeaboi.tools.github.github_read_repo", mock_repo)
        monkeypatch.setattr("yeaboi.tools.github.github_read_readme", mock_readme)

        qs = self._make_qs("https://github.com/owner/repo", "GitHub")
        result, _status = _scan_repo_context(qs)

        assert "---" in result
        assert "tree data" in result
        assert "readme data" in result

    def test_github_all_tools_fail_returns_none(self, monkeypatch):
        """Returns None when all GitHub tool calls return error strings."""
        mock_repo = MagicMock()
        mock_repo.invoke.return_value = "Error: 404 not found"
        mock_readme = MagicMock()
        mock_readme.invoke.return_value = "Error: network error"

        monkeypatch.setattr("yeaboi.tools.github.github_read_repo", mock_repo)
        monkeypatch.setattr("yeaboi.tools.github.github_read_readme", mock_readme)

        qs = self._make_qs("https://github.com/owner/repo", "GitHub")
        ctx, status = _scan_repo_context(qs)
        assert ctx is None
        assert status["status"] == "error"

    def test_github_one_tool_fails_returns_other(self, monkeypatch):
        """Returns the successful result when only one GitHub tool fails."""
        mock_repo = MagicMock()
        mock_repo.invoke.return_value = "Error: 404 not found"
        mock_readme = MagicMock()
        mock_readme.invoke.return_value = "# README content"

        monkeypatch.setattr("yeaboi.tools.github.github_read_repo", mock_repo)
        monkeypatch.setattr("yeaboi.tools.github.github_read_readme", mock_readme)

        qs = self._make_qs("https://github.com/owner/repo", "GitHub")
        result, status = _scan_repo_context(qs)

        assert result is not None
        assert "README content" in result
        assert status["status"] == "success"

    def test_github_rate_limit_excluded(self, monkeypatch):
        """Rate-limit responses are excluded (treated as failure)."""
        mock_repo = MagicMock()
        mock_repo.invoke.return_value = "GitHub rate limit exceeded"
        mock_readme = MagicMock()
        mock_readme.invoke.return_value = "# README"

        monkeypatch.setattr("yeaboi.tools.github.github_read_repo", mock_repo)
        monkeypatch.setattr("yeaboi.tools.github.github_read_readme", mock_readme)

        qs = self._make_qs("https://github.com/owner/repo", "GitHub")
        result, status = _scan_repo_context(qs)

        assert result is not None
        assert "rate limit" not in result
        assert "README" in result
        assert status["status"] == "success"

    def test_azdo_platform_calls_azdevops_read_repo(self, monkeypatch):
        """Azure DevOps platform calls azdevops_read_repo."""
        mock_fn = MagicMock()
        mock_fn.invoke.return_value = "## AzDO File Tree\n- src/"

        monkeypatch.setattr("yeaboi.tools.azure_devops.azdevops_read_repo", mock_fn)

        qs = self._make_qs("https://dev.azure.com/org/proj/_git/repo", "Azure DevOps")
        result, status = _scan_repo_context(qs)

        mock_fn.invoke.assert_called_once_with({"repo_url": "https://dev.azure.com/org/proj/_git/repo"})
        assert result is not None
        assert "AzDO File Tree" in result
        assert status["status"] == "success"

    def test_azdo_error_returns_none(self, monkeypatch):
        """Returns None when AzDO tool returns an error string."""
        mock_fn = MagicMock()
        mock_fn.invoke.return_value = "Error: unauthorized"

        monkeypatch.setattr("yeaboi.tools.azure_devops.azdevops_read_repo", mock_fn)

        qs = self._make_qs("https://dev.azure.com/org/proj/_git/repo", "Azure DevOps")
        ctx, status = _scan_repo_context(qs)
        assert ctx is None
        assert status["status"] == "error"


class TestProjectAnalyzerRepoContext:
    """Tests that project_analyzer integrates _scan_repo_context correctly."""

    def _make_state(self, **extras: object) -> dict:
        qs = make_completed_questionnaire()
        state = {
            "messages": [HumanMessage(content="continue")],
            "questionnaire": qs,
            "team_size": 3,
            "velocity_per_sprint": 15,
        }
        state.update(extras)
        return state

    def test_includes_repo_context_when_scan_succeeds(self, monkeypatch):
        """project_analyzer includes 'repo_context' in return dict when scan succeeds."""
        fake_response = MagicMock()
        fake_response.content = VALID_ANALYSIS_JSON
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)
        monkeypatch.setattr(
            "yeaboi.agent.nodes._scan_repo_context",
            lambda _qs: ("## File Tree\n- src/", {"name": "Repository", "status": "success", "detail": "test"}),
        )

        result = project_analyzer(self._make_state())
        assert "repo_context" in result
        assert result["repo_context"] == "## File Tree\n- src/"

    def test_omits_repo_context_when_scan_returns_none(self, monkeypatch):
        """project_analyzer omits 'repo_context' key when scan returns None."""
        fake_response = MagicMock()
        fake_response.content = VALID_ANALYSIS_JSON
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)
        monkeypatch.setattr(
            "yeaboi.agent.nodes._scan_repo_context",
            lambda _qs: (None, {"name": "Repository", "status": "skipped", "detail": "test"}),
        )

        result = project_analyzer(self._make_state())
        assert "repo_context" not in result


class TestFetchConfluenceContext:
    """Tests for the _fetch_confluence_context() helper function."""

    def _make_qs(self, project_name: str = "My Project") -> QuestionnaireState:
        qs = QuestionnaireState()
        qs.answers[1] = project_name
        return qs

    def test_returns_none_when_no_project_name(self, monkeypatch):
        """Returns None when Q1 is not set."""
        qs = QuestionnaireState()
        ctx, status = _fetch_confluence_context(qs)
        assert ctx is None
        assert status["status"] == "skipped"

    def test_returns_none_when_credentials_missing(self, monkeypatch):
        """Returns None when Jira/Confluence env vars are not configured."""
        # get_jira_base_url/email/token just call os.getenv — clear the vars to simulate
        # missing credentials without needing to monkeypatch the imported functions.
        monkeypatch.delenv("JIRA_BASE_URL", raising=False)
        monkeypatch.delenv("JIRA_EMAIL", raising=False)
        monkeypatch.delenv("JIRA_API_TOKEN", raising=False)

        qs = self._make_qs("My Project")
        ctx, status = _fetch_confluence_context(qs)
        assert ctx is None
        assert status["status"] == "skipped"

    def test_returns_none_when_search_errors(self, monkeypatch):
        """Returns None when confluence_search_docs returns an Error string."""
        # StructuredTool is a Pydantic v2 model — can't set .invoke directly.
        # Replace the whole module-level object with a MagicMock instead.
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "Error: Confluence is not configured."
        monkeypatch.setattr("yeaboi.tools.confluence.confluence_search_docs", mock_tool)
        # Patch credentials to appear configured so we reach the search call
        monkeypatch.setenv("JIRA_BASE_URL", "https://example.atlassian.net")
        monkeypatch.setenv("JIRA_EMAIL", "user@example.com")
        monkeypatch.setenv("JIRA_API_TOKEN", "token")

        qs = self._make_qs("My Project")
        ctx, status = _fetch_confluence_context(qs)
        assert ctx is None
        assert status["status"] == "error"

    def test_returns_none_when_no_results(self, monkeypatch):
        """Returns None when confluence_search_docs returns 'No Confluence pages found'."""
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "No Confluence pages found for 'My Project'."
        monkeypatch.setattr("yeaboi.tools.confluence.confluence_search_docs", mock_tool)
        monkeypatch.setenv("JIRA_BASE_URL", "https://example.atlassian.net")
        monkeypatch.setenv("JIRA_EMAIL", "user@example.com")
        monkeypatch.setenv("JIRA_API_TOKEN", "token")

        qs = self._make_qs("My Project")
        ctx, status = _fetch_confluence_context(qs)
        assert ctx is None
        assert status["status"] == "error"

    def test_returns_result_string_on_success(self, monkeypatch):
        """Returns the search result string when Confluence returns pages."""
        fake_results = "Confluence search results for 'My Project':\n\n[Arch Docs] (ID: 1)\n  Auth system..."
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = fake_results
        monkeypatch.setattr("yeaboi.tools.confluence.confluence_search_docs", mock_tool)
        monkeypatch.setenv("JIRA_BASE_URL", "https://example.atlassian.net")
        monkeypatch.setenv("JIRA_EMAIL", "user@example.com")
        monkeypatch.setenv("JIRA_API_TOKEN", "token")

        qs = self._make_qs("My Project")
        result, status = _fetch_confluence_context(qs)
        assert result == fake_results
        assert status["status"] == "success"

    def test_returns_none_on_exception(self, monkeypatch):
        """Returns None (does not raise) when an unexpected error occurs."""
        mock_tool = MagicMock()
        mock_tool.invoke.side_effect = RuntimeError("Network error")
        monkeypatch.setattr("yeaboi.tools.confluence.confluence_search_docs", mock_tool)
        monkeypatch.setenv("JIRA_BASE_URL", "https://example.atlassian.net")
        monkeypatch.setenv("JIRA_EMAIL", "user@example.com")
        monkeypatch.setenv("JIRA_API_TOKEN", "token")

        qs = self._make_qs("My Project")
        ctx, status = _fetch_confluence_context(qs)
        assert ctx is None
        assert status["status"] == "error"

    def test_fetches_pages_from_scrum_md_urls(self, monkeypatch):
        """Fetches Confluence pages directly when SCRUM.md contains wiki URLs."""
        # Search returns nothing, but SCRUM.md has a direct page URL
        mock_search = MagicMock()
        mock_search.invoke.return_value = "No Confluence pages found for 'My Project'."
        monkeypatch.setattr("yeaboi.tools.confluence.confluence_search_docs", mock_search)

        mock_read = MagicMock()
        mock_read.invoke.return_value = "=== RunBook: Container Restart ===\nURL: ...\n\nStep 1: Check pods..."
        monkeypatch.setattr("yeaboi.tools.confluence.confluence_read_page", mock_read)

        monkeypatch.setenv("JIRA_BASE_URL", "https://example.atlassian.net")
        monkeypatch.setenv("JIRA_EMAIL", "user@example.com")
        monkeypatch.setenv("JIRA_API_TOKEN", "token")

        user_context = (
            "## Key Links\n"
            "- RunBook: https://example.atlassian.net/wiki/spaces/ISS/pages/1234567890/Container+Restart\n"
        )
        qs = self._make_qs("My Project")
        result, status = _fetch_confluence_context(qs, user_context=user_context)
        assert result is not None
        assert "Container Restart" in result
        assert status["status"] == "success"
        mock_read.invoke.assert_called_once_with({"page_id": "1234567890"})

    def test_combines_search_and_linked_pages(self, monkeypatch):
        """Combines keyword search results with directly fetched pages from SCRUM.md URLs."""
        mock_search = MagicMock()
        mock_search.invoke.return_value = "Confluence search results for 'My Project':\n\n[ADR-001] ..."
        monkeypatch.setattr("yeaboi.tools.confluence.confluence_search_docs", mock_search)

        mock_read = MagicMock()
        mock_read.invoke.return_value = "=== RunBook ===\nStep 1..."
        monkeypatch.setattr("yeaboi.tools.confluence.confluence_read_page", mock_read)

        monkeypatch.setenv("JIRA_BASE_URL", "https://example.atlassian.net")
        monkeypatch.setenv("JIRA_EMAIL", "user@example.com")
        monkeypatch.setenv("JIRA_API_TOKEN", "token")

        user_context = "- https://example.atlassian.net/wiki/spaces/ISS/pages/999/RunBook\n"
        qs = self._make_qs("My Project")
        result, status = _fetch_confluence_context(qs, user_context=user_context)
        assert "ADR-001" in result
        assert "RunBook" in result
        assert "---" in result  # separator between parts
        assert "linked page" in status["detail"]

    def test_deduplicates_page_ids(self, monkeypatch):
        """Does not fetch the same Confluence page ID twice."""
        mock_search = MagicMock()
        mock_search.invoke.return_value = "No Confluence pages found."
        monkeypatch.setattr("yeaboi.tools.confluence.confluence_search_docs", mock_search)

        mock_read = MagicMock()
        mock_read.invoke.return_value = "=== Page ===\nContent"
        monkeypatch.setattr("yeaboi.tools.confluence.confluence_read_page", mock_read)

        monkeypatch.setenv("JIRA_BASE_URL", "https://example.atlassian.net")
        monkeypatch.setenv("JIRA_EMAIL", "user@example.com")
        monkeypatch.setenv("JIRA_API_TOKEN", "token")

        # Same page ID appears twice
        user_context = (
            "- https://example.atlassian.net/wiki/spaces/ISS/pages/111/Page1\n"
            "- https://example.atlassian.net/wiki/spaces/ISS/pages/111/Page1?v=2\n"
        )
        qs = self._make_qs("My Project")
        _fetch_confluence_context(qs, user_context=user_context)
        assert mock_read.invoke.call_count == 1

    def test_skips_failed_page_fetches_gracefully(self, monkeypatch):
        """Continues when individual page fetches fail."""
        mock_search = MagicMock()
        mock_search.invoke.return_value = "No Confluence pages found."
        monkeypatch.setattr("yeaboi.tools.confluence.confluence_search_docs", mock_search)

        mock_read = MagicMock()
        mock_read.invoke.side_effect = [
            RuntimeError("Page deleted"),
            "=== Good Page ===\nContent",
        ]
        monkeypatch.setattr("yeaboi.tools.confluence.confluence_read_page", mock_read)

        monkeypatch.setenv("JIRA_BASE_URL", "https://example.atlassian.net")
        monkeypatch.setenv("JIRA_EMAIL", "user@example.com")
        monkeypatch.setenv("JIRA_API_TOKEN", "token")

        user_context = (
            "- https://example.atlassian.net/wiki/spaces/ISS/pages/111/Bad\n"
            "- https://example.atlassian.net/wiki/spaces/ISS/pages/222/Good\n"
        )
        qs = self._make_qs("My Project")
        result, status = _fetch_confluence_context(qs, user_context=user_context)
        assert result is not None
        assert "Good Page" in result
        assert status["status"] == "success"


class TestExtractConfluencePageIds:
    """Tests for _extract_confluence_page_ids() helper."""

    def test_extracts_page_id_from_standard_url(self):
        text = "https://example.atlassian.net/wiki/spaces/ISS/pages/1359905111/Domain+Container"
        assert _extract_confluence_page_ids(text) == ["1359905111"]

    def test_extracts_multiple_page_ids(self):
        text = (
            "- https://x.atlassian.net/wiki/spaces/A/pages/111/Page1\n"
            "- https://x.atlassian.net/wiki/spaces/B/pages/222/Page2\n"
        )
        assert _extract_confluence_page_ids(text) == ["111", "222"]

    def test_handles_urls_with_query_params(self):
        text = "https://x.atlassian.net/wiki/spaces/ISS/pages/999/Title?atlOrigin=abc123"
        assert _extract_confluence_page_ids(text) == ["999"]

    def test_returns_empty_for_no_urls(self):
        assert _extract_confluence_page_ids("No links here") == []

    def test_returns_empty_for_non_confluence_urls(self):
        assert _extract_confluence_page_ids("https://github.com/org/repo") == []

    def test_handles_real_scrum_md_content(self):
        """Parses the URL format from a real SCRUM.md file."""
        text = """## Key Links
- Run Book 1: https://youlend.atlassian.net/wiki/spaces/ISS/pages/1359905111/Domain+Container+Service+Restart+Critical?atlOrigin=eyJpIjoi
- Run Book 2: https://youlend.atlassian.net/wiki/spaces/ISS/pages/1359904787/Frequent+Container+Restarts
- BITS SRE: https://www.datadoghq.com/blog/bits-ai-sre/
"""
        ids = _extract_confluence_page_ids(text)
        assert ids == ["1359905111", "1359904787"]


class TestProjectAnalyzerConfluenceContext:
    """Tests that project_analyzer integrates _fetch_confluence_context correctly."""

    def _make_state(self, **extras: object) -> dict:
        qs = make_completed_questionnaire()
        state = {
            "messages": [HumanMessage(content="continue")],
            "questionnaire": qs,
            "team_size": 3,
            "velocity_per_sprint": 15,
        }
        state.update(extras)
        return state

    def test_includes_confluence_context_when_fetch_succeeds(self, monkeypatch):
        """project_analyzer includes 'confluence_context' in return dict when fetch succeeds."""
        fake_response = MagicMock()
        fake_response.content = VALID_ANALYSIS_JSON
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)
        monkeypatch.setattr(
            "yeaboi.agent.nodes._scan_repo_context",
            lambda _qs: (None, {"name": "Repository", "status": "skipped", "detail": "test"}),
        )
        monkeypatch.setattr(
            "yeaboi.agent.nodes._fetch_confluence_context",
            lambda _qs, **kw: (
                "Confluence search results for 'My Project':\n\n[ADR-001] ...",
                {"name": "Confluence", "status": "success", "detail": "test"},
            ),
        )

        result = project_analyzer(self._make_state())
        assert "confluence_context" in result
        assert "ADR-001" in result["confluence_context"]

    def test_omits_confluence_context_when_fetch_returns_none(self, monkeypatch):
        """project_analyzer omits 'confluence_context' key when fetch returns None."""
        fake_response = MagicMock()
        fake_response.content = VALID_ANALYSIS_JSON
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)
        monkeypatch.setattr(
            "yeaboi.agent.nodes._scan_repo_context",
            lambda _qs: (None, {"name": "Repository", "status": "skipped", "detail": "test"}),
        )
        monkeypatch.setattr(
            "yeaboi.agent.nodes._fetch_confluence_context",
            lambda _qs, **kw: (None, {"name": "Confluence", "status": "skipped", "detail": "test"}),
        )

        result = project_analyzer(self._make_state())
        assert "confluence_context" not in result


class TestIntegrationGates:
    """A plan restricted to some integrations skips the analyzer's own reads of the others."""

    def _run(self, monkeypatch, repo="https://github.com/acme/app", **extras):
        fake_response = MagicMock()
        fake_response.content = VALID_ANALYSIS_JSON
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)
        calls: list[str] = []
        monkeypatch.setattr(
            "yeaboi.agent.nodes._scan_repo_context",
            lambda _qs: calls.append("repo") or (None, {"name": "Repository", "status": "skipped", "detail": "t"}),
        )
        monkeypatch.setattr(
            "yeaboi.agent.nodes._fetch_confluence_context",
            lambda _qs, **kw: calls.append("confluence") or (None, {"name": "Confluence", "status": "skipped"}),
        )
        monkeypatch.setattr(
            "yeaboi.agent.nodes._fetch_notion_context",
            lambda _qs, **kw: calls.append("notion") or (None, {"name": "Notion", "status": "skipped"}),
        )
        qs = make_completed_questionnaire()
        qs.answers[17] = repo
        state = {"messages": [HumanMessage(content="continue")], "questionnaire": qs, **extras}
        return project_analyzer(state), calls

    def test_a_local_checkout_scans_whatever_the_plan_allows(self, monkeypatch):
        _result, calls = self._run(monkeypatch, repo="/Users/me/app", session_integrations=["confluence"])
        assert calls == ["repo", "confluence"]

    def test_confluence_and_notion_are_skipped_when_not_enabled(self, monkeypatch):
        result, calls = self._run(monkeypatch, session_integrations=["github"])
        assert calls == ["repo"]
        statuses = {row["name"]: row for row in result["context_sources"]}
        assert statuses["Confluence"]["status"] == "skipped"
        assert statuses["Confluence"]["detail"] == "not enabled for this plan"
        assert statuses["Notion"]["status"] == "skipped"

    def test_the_repository_scan_is_skipped_when_github_is_not_enabled(self, monkeypatch):
        result, calls = self._run(monkeypatch, session_integrations=["confluence", "notion"])
        assert calls == ["confluence", "notion"]
        statuses = {row["name"]: row for row in result["context_sources"]}
        assert statuses["Repository"]["detail"] == "not enabled for this plan"

    def test_absent_key_reads_everything(self, monkeypatch):
        _result, calls = self._run(monkeypatch)
        assert calls == ["repo", "confluence", "notion"]

    def test_pasted_context_joins_the_user_context(self, monkeypatch):
        captured: dict = {}

        def mock_prompt(answers_block, team_size, velocity_per_sprint, **kwargs):
            captured.update(kwargs)
            return "mock prompt"

        monkeypatch.setattr("yeaboi.agent.nodes.get_analyzer_prompt", mock_prompt)
        monkeypatch.setattr(
            "yeaboi.agent.nodes._load_user_context",
            lambda: ("# SCRUM notes", {"name": "SCRUM.md", "status": "success", "detail": "t"}),
        )
        self._run(monkeypatch, pasted_context=["Reference 1 (link): spec — https://s"])
        assert captured["user_context"] == "# SCRUM notes\n\nReference 1 (link): spec — https://s"


class TestLoadUserContext:
    """Tests for the _load_user_context() helper function."""

    def test_returns_none_when_file_missing(self, tmp_path, monkeypatch):
        """Returns None when SCRUM.md does not exist in the working directory."""
        monkeypatch.chdir(tmp_path)
        ctx, status = _load_user_context()
        assert ctx is None
        assert status["status"] == "skipped"

    def test_returns_none_when_file_empty(self, tmp_path, monkeypatch):
        """Returns None when SCRUM.md exists but is blank."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "SCRUM.md").write_text("   \n\n")
        ctx, status = _load_user_context()
        assert ctx is None
        assert status["status"] == "skipped"

    def test_returns_content_when_file_present(self, tmp_path, monkeypatch):
        """Returns the file content (stripped) when SCRUM.md exists."""
        monkeypatch.chdir(tmp_path)
        content = "# My Project\nWe use React + FastAPI."
        (tmp_path / "SCRUM.md").write_text(content)
        result, status = _load_user_context()
        assert result == content
        assert status["status"] == "success"

    def test_strips_leading_trailing_whitespace(self, tmp_path, monkeypatch):
        """Content is stripped so blank padding around the real notes is removed."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "SCRUM.md").write_text("\n\n# Notes\nSome content\n\n")
        result, _status = _load_user_context()
        assert result == "# Notes\nSome content"

    def test_accepts_explicit_path(self, tmp_path):
        """An explicit path argument bypasses CWD lookup."""
        p = tmp_path / "custom.md"
        p.write_text("custom context")
        result, status = _load_user_context(path=str(p))
        assert result == "custom context"
        assert status["status"] == "success"

    def test_returns_none_on_read_error(self, tmp_path, monkeypatch):
        """Returns None (does not raise) when the file cannot be read."""
        bad_path = str(tmp_path / "nonexistent" / "SCRUM.md")
        ctx, status = _load_user_context(path=bad_path)
        assert ctx is None
        assert status["status"] == "skipped"

    def test_loads_scrum_docs_directory(self, tmp_path):
        """Reads .md/.txt/.rst files from scrum-docs/ directory."""
        docs = tmp_path / "scrum-docs"
        docs.mkdir()
        (docs / "prd.md").write_text("# PRD\nThe product requirements.")
        (docs / "design.txt").write_text("Design notes here.")
        # Non-matching extension should be ignored
        (docs / "data.json").write_text('{"ignored": true}')
        ctx, status = _load_user_context(path=str(tmp_path / "MISSING"), docs_dir=str(docs))
        assert ctx is not None
        assert "PRD" in ctx
        assert "Design notes" in ctx
        assert "ignored" not in ctx
        assert status["status"] == "success"
        assert "prd.md" in status["detail"]
        assert "design.txt" in status["detail"]

    def test_combines_scrum_md_and_docs_dir(self, tmp_path):
        """Both SCRUM.md and scrum-docs/ are loaded and combined."""
        (tmp_path / "SCRUM.md").write_text("Main context file.")
        docs = tmp_path / "scrum-docs"
        docs.mkdir()
        (docs / "arch.md").write_text("Architecture decisions.")
        ctx, status = _load_user_context(path=str(tmp_path / "SCRUM.md"), docs_dir=str(docs))
        assert "Main context file." in ctx
        assert "Architecture decisions." in ctx
        assert "SCRUM.md" in status["detail"]
        assert "arch.md" in status["detail"]

    def test_docs_dir_respects_budget(self, tmp_path):
        """Files in scrum-docs/ are truncated when they exceed the character budget."""
        from yeaboi.tools.codebase import _MAX_DOCS_CHARS

        docs = tmp_path / "scrum-docs"
        docs.mkdir()
        (docs / "huge.md").write_text("A" * (_MAX_DOCS_CHARS + 5000))
        ctx, status = _load_user_context(path=str(tmp_path / "MISSING"), docs_dir=str(docs))
        assert ctx is not None
        assert "Truncated" in ctx
        assert len(ctx) < _MAX_DOCS_CHARS + 500  # allow for header + truncation note

    def test_docs_dir_skipped_when_missing(self, tmp_path):
        """No error when scrum-docs/ directory doesn't exist."""
        (tmp_path / "SCRUM.md").write_text("Just SCRUM.md")
        ctx, status = _load_user_context(path=str(tmp_path / "SCRUM.md"), docs_dir=str(tmp_path / "nope"))
        assert ctx == "Just SCRUM.md"
        assert status["status"] == "success"

    @pytest.mark.skipif(not _HAS_PYMUPDF, reason="pymupdf not installed")
    def test_reads_pdf_files(self, tmp_path):
        """PDF files in scrum-docs/ are read via pymupdf."""
        import pymupdf

        docs = tmp_path / "scrum-docs"
        docs.mkdir()
        # Create a minimal PDF with text content
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Product Requirements Document\nThis is the PRD content.")
        doc.save(str(docs / "prd.pdf"))
        doc.close()

        ctx, status = _load_user_context(path=str(tmp_path / "MISSING"), docs_dir=str(docs))
        assert ctx is not None
        assert "Product Requirements Document" in ctx
        assert "prd.pdf" in status["detail"]

    def test_skips_pdf_when_pymupdf_missing(self, tmp_path, monkeypatch):
        """PDF files are silently skipped when pymupdf is not installed."""
        docs = tmp_path / "scrum-docs"
        docs.mkdir()
        # Write a fake PDF (won't be parseable but that's fine — we block the import)
        (docs / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
        (docs / "notes.md").write_text("Markdown notes here.")

        # Block pymupdf import
        import builtins

        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "pymupdf":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)

        ctx, status = _load_user_context(path=str(tmp_path / "MISSING"), docs_dir=str(docs))
        assert ctx is not None
        assert "Markdown notes" in ctx
        # PDF was skipped, only markdown loaded
        assert "doc.pdf" not in status["detail"]
        assert "notes.md" in status["detail"]

    @pytest.mark.skipif(not _HAS_PYMUPDF, reason="pymupdf not installed")
    def test_pdf_mixed_with_text_files(self, tmp_path):
        """PDF and text files are combined in the same docs directory."""
        import pymupdf

        docs = tmp_path / "scrum-docs"
        docs.mkdir()
        (docs / "arch.md").write_text("Architecture: microservices.")
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Design spec for API gateway.")
        doc.save(str(docs / "design.pdf"))
        doc.close()

        ctx, status = _load_user_context(path=str(tmp_path / "MISSING"), docs_dir=str(docs))
        assert "Architecture: microservices" in ctx
        assert "Design spec for API gateway" in ctx
        assert "arch.md" in status["detail"]
        assert "design.pdf" in status["detail"]


# ---------------------------------------------------------------------------
# TestProjectAnalyzerUserContext
# ---------------------------------------------------------------------------


class TestProjectAnalyzerUserContext:
    """Tests that project_analyzer integrates _load_user_context correctly."""

    def _make_state(self, **extras):
        qs = make_completed_questionnaire()
        state = {
            "messages": [HumanMessage(content="start")],
            "questionnaire": qs,
            **extras,
        }
        return state

    def test_user_context_passed_to_prompt_when_present(self, monkeypatch):
        """project_analyzer passes user_context to get_analyzer_prompt when file exists."""
        fake_response = MagicMock()
        fake_response.content = VALID_ANALYSIS_JSON
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)
        monkeypatch.setattr(
            "yeaboi.agent.nodes._scan_repo_context",
            lambda _qs: (None, {"name": "Repository", "status": "skipped", "detail": "test"}),
        )
        monkeypatch.setattr(
            "yeaboi.agent.nodes._fetch_confluence_context",
            lambda _qs, **kw: (None, {"name": "Confluence", "status": "skipped", "detail": "test"}),
        )
        monkeypatch.setattr(
            "yeaboi.agent.nodes._load_user_context",
            lambda: ("# SCRUM notes\nUse Postgres.", {"name": "SCRUM.md", "status": "success", "detail": "test"}),
        )

        captured: dict = {}

        def mock_prompt(answers_block, team_size, velocity_per_sprint, **kwargs):
            captured.update(kwargs)
            return "mock prompt"

        monkeypatch.setattr("yeaboi.agent.nodes.get_analyzer_prompt", mock_prompt)

        project_analyzer(self._make_state())
        assert captured.get("user_context") == "# SCRUM notes\nUse Postgres."

    def test_user_context_absent_when_no_file(self, monkeypatch):
        """project_analyzer passes user_context=None when _load_user_context returns None."""
        fake_response = MagicMock()
        fake_response.content = VALID_ANALYSIS_JSON
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)
        monkeypatch.setattr(
            "yeaboi.agent.nodes._scan_repo_context",
            lambda _qs: (None, {"name": "Repository", "status": "skipped", "detail": "test"}),
        )
        monkeypatch.setattr(
            "yeaboi.agent.nodes._fetch_confluence_context",
            lambda _qs, **kw: (None, {"name": "Confluence", "status": "skipped", "detail": "test"}),
        )
        monkeypatch.setattr(
            "yeaboi.agent.nodes._load_user_context",
            lambda: (None, {"name": "SCRUM.md", "status": "skipped", "detail": "test"}),
        )

        captured: dict = {}

        def mock_prompt(answers_block, team_size, velocity_per_sprint, **kwargs):
            captured.update(kwargs)
            return "mock prompt"

        monkeypatch.setattr("yeaboi.agent.nodes.get_analyzer_prompt", mock_prompt)

        project_analyzer(self._make_state())
        assert captured.get("user_context") is None


class TestComputePromptQuality:
    """Tests for compute_prompt_quality() — deterministic scoring from QuestionnaireState."""

    def test_all_answered_grade_a(self):
        """All 26 questions answered directly should produce grade A."""
        qs = make_completed_questionnaire()
        rating = compute_prompt_quality(qs, has_user_context=True)
        assert rating.grade == "A"
        assert rating.score_pct >= 85
        assert rating.answered_count == TOTAL_QUESTIONS
        assert rating.extracted_count == 0
        assert rating.defaulted_count == 0
        assert rating.skipped_count == 0
        assert rating.suggestions == ()

    def test_all_skipped_grade_d(self):
        """Empty questionnaire (no answers) should produce grade D."""
        qs = QuestionnaireState(completed=True)
        rating = compute_prompt_quality(qs)
        assert rating.grade == "D"
        assert rating.score_pct == 0
        assert rating.skipped_count == TOTAL_QUESTIONS
        assert rating.answered_count == 0
        assert len(rating.suggestions) > 0  # should have suggestions for essential Qs

    def test_mixed_scoring(self):
        """Mixed answered/extracted/defaulted/skipped should produce an intermediate grade."""
        qs = QuestionnaireState(completed=True)
        # Answer essential questions (Q1-Q4, Q6)
        for q in [1, 2, 3, 4, 6]:
            qs.answers[q] = f"Answer for Q{q}"
        # Extract Q11, Q15
        for q in [11, 15]:
            qs.answers[q] = f"Extracted Q{q}"
            qs.extracted_questions.add(q)
        # Default a few non-essential
        for q in [5, 7, 8, 9, 10]:
            qs.answers[q] = f"Default for Q{q}"
            qs.defaulted_questions.add(q)
        # Skip the rest (12-14, 16-30 minus 15) — 18 questions skipped
        rating = compute_prompt_quality(qs)
        assert rating.answered_count == 5
        assert rating.extracted_count == 2
        assert rating.defaulted_count == 5
        assert rating.skipped_count == 18
        assert rating.grade in ("C", "D")  # more skipped → lower grade

    def test_probing_bonus(self):
        """Probed questions should add bonus points."""
        qs = make_completed_questionnaire()
        qs.probed_questions = {3, 4, 11}
        rating = compute_prompt_quality(qs)
        assert rating.probed_count == 3
        assert rating.grade == "A"  # still A with all answered + probing

    def test_suggestions_for_defaulted_essential(self):
        """Suggestions should reference defaulted essential questions."""
        qs = make_completed_questionnaire()
        # Default Q11 (essential — tech stack)
        qs.defaulted_questions.add(11)
        rating = compute_prompt_quality(qs)
        assert any("Q11" in s for s in rating.suggestions)

    def test_suggestions_for_skipped_essential(self):
        """Suggestions should reference skipped essential questions."""
        qs = QuestionnaireState(completed=True)
        # Answer only non-essential questions
        for q in range(5, TOTAL_QUESTIONS + 1):
            if q not in {6, 11, 15}:
                qs.answers[q] = f"Answer for Q{q}"
        # Q1-Q4, Q6, Q11, Q15 are skipped (no answers)
        rating = compute_prompt_quality(qs)
        # Should have suggestions for the skipped essential questions
        assert len(rating.suggestions) > 0
        assert len(rating.suggestions) <= 4  # max 4 suggestions

    def test_extracted_gets_full_points(self):
        """Extracted answers should get the same points as user-answered."""
        # All answered
        qs_answered = make_completed_questionnaire()
        rating_answered = compute_prompt_quality(qs_answered)

        # All extracted
        qs_extracted = QuestionnaireState(completed=True)
        for q in range(1, TOTAL_QUESTIONS + 1):
            qs_extracted.answers[q] = f"Extracted Q{q}"
            qs_extracted.extracted_questions.add(q)
        rating_extracted = compute_prompt_quality(qs_extracted)

        assert rating_answered.score_pct == rating_extracted.score_pct

    def test_scrum_md_suggestion_when_no_user_context(self):
        """SCRUM.md suggestion should appear when has_user_context is False."""
        qs = make_completed_questionnaire()
        rating = compute_prompt_quality(qs, has_user_context=False)
        assert any("SCRUM.md" in s for s in rating.suggestions)

    def test_no_scrum_md_suggestion_when_user_context_present(self):
        """SCRUM.md suggestion should NOT appear when has_user_context is True."""
        qs = make_completed_questionnaire()
        rating = compute_prompt_quality(qs, has_user_context=True)
        assert not any("SCRUM.md" in s for s in rating.suggestions)

    def test_high_value_question_suggestions(self):
        """Suggestions should include high-value non-essential Qs (Q14, Q17) when skipped."""
        qs = make_completed_questionnaire()
        # Remove answers for Q14 and Q17
        del qs.answers[14]
        del qs.answers[17]
        rating = compute_prompt_quality(qs)
        suggestion_text = " ".join(rating.suggestions)
        assert "Q14" in suggestion_text or "Q17" in suggestion_text


class TestContextScopeGating:
    """The state's ``context_scope`` decides what the planning nodes may read."""

    def test_no_scope_wants_everything(self):
        from yeaboi.agent.nodes import _effective_analysis_profile_id, _state_scope, _wants_dep

        state = {"analysis_profile_id": "jira-PROJ"}
        assert _wants_dep(state, "analysis") and _state_scope(state) is None
        assert _effective_analysis_profile_id(state) == "jira-PROJ"

    def test_analysis_off_drops_the_profile(self):
        import json

        from yeaboi.agent.nodes import _effective_analysis_profile_id, _state_scope, _wants_dep

        state = {"analysis_profile_id": "jira-PROJ", "context_scope": json.dumps({"sources": ["standup"]})}
        assert not _wants_dep(state, "analysis")
        assert _effective_analysis_profile_id(state) == ""
        selection = _state_scope(state)
        assert selection is not None and not selection.wants("analysis") and selection.wants("standup")

    def test_an_unreadable_scope_reads_everything(self):
        from yeaboi.agent.nodes import _state_scope, _wants_dep

        assert _wants_dep({"context_scope": "{"}, "analysis")
        assert _state_scope({"context_scope": "{"}) is None

    def test_every_call_resolves_afresh(self, monkeypatch):
        """A run must see what was recorded since the last one — no day-long cache."""
        import json

        from yeaboi.agent.nodes import _state_scope

        answers = iter(["first", "second"])
        monkeypatch.setattr("yeaboi.context.resolve.resolve_scope", lambda *_a, **_k: next(answers))
        state = {"context_scope": json.dumps({"sources": ["standup"]})}
        assert _state_scope(state) == "first"
        assert _state_scope(state) == "second"

    def test_the_analyzer_skips_calibration_and_hands_the_selection_on(self, monkeypatch):
        import json

        from yeaboi.agent.ceremony_history import CeremonyContext

        fake_response = MagicMock()
        fake_response.content = VALID_ANALYSIS_JSON
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)

        def never(*_a, **_k):
            raise AssertionError("the team profile must not be read with analysis switched off")

        monkeypatch.setattr("yeaboi.agent.nodes._load_team_profile", never)
        seen: dict = {}
        monkeypatch.setattr(
            "yeaboi.agent.nodes.gather_ceremony_context", lambda *a, **kw: seen.update(kw) or CeremonyContext()
        )
        state = {
            "messages": [HumanMessage(content="continue")],
            "questionnaire": make_completed_questionnaire(),
            "team_size": 3,
            "velocity_per_sprint": 15,
            "analysis_profile_id": "jira-PROJ",
            "context_scope": json.dumps({"sources": ["standup"]}),
        }
        result = project_analyzer(state)
        assert isinstance(result["project_analysis"], ProjectAnalysis)
        assert seen["selection"] is not None and not seen["selection"].wants("retro")
