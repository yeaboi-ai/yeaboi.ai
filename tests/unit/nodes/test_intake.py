"""Tests for project_intake node and questionnaire-related helpers."""

import re
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from yeaboi.agent.nodes import (
    _Q2_REPO_URL_PROMPT,
    _VELOCITY_PER_ENGINEER,
    _apply_solo_defaults,
    _auto_apply_extractions,
    _batch_defaults_for_phase,
    _build_answers_block,
    _build_extraction_summary,
    _build_gap_prompt,
    _build_intake_summary,
    _check_vague_answer,
    _derive_q15_from_q2,
    _derive_q27_from_locale,
    _extract_answers_from_description,
    _extract_capacity_deductions,
    _extract_team_and_velocity,
    _find_essential_gaps,
    _is_confirm_intent,
    _is_defaults_intent,
    _is_skip_intent,
    _needs_repo_url_prompt,
    _next_unskipped_question,
    _parse_edit_intent,
    _parse_first_int,
    _solo_velocity_from_profile,
    _sync_platform_from_url,
    project_intake,
    route_entry,
)
from yeaboi.agent.state import (
    TOTAL_QUESTIONS,
    QuestionnaireState,
)
from yeaboi.prompts.intake import (
    CONDITIONAL_ESSENTIALS,
    INTAKE_QUESTIONS,
    PHASE_INTROS,
    PHASE_LABELS,
    QUESTION_DEFAULTS,
    QUESTION_METADATA,
    QUICK_FALLBACK_DEFAULTS,
    SMART_ESSENTIALS,
    SOLO_ROLES_DEFAULT,
    AnswerSource,
    QuestionMeta,
    is_choice_question,
)


class TestProjectIntake:
    """Tests for the project_intake() node function."""

    @pytest.fixture(autouse=True)
    def _disable_vague_check(self, monkeypatch):
        """Disable vague-answer checking so existing tests don't hit the LLM.

        The subsequent-calls path now calls _check_vague_answer(), which would
        try to invoke the real LLM without an API key. This autouse fixture
        patches it to always return None (accept the answer as-is), keeping
        existing test logic unchanged.
        """
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)

    def test_first_call_initializes_questionnaire(self):
        """No questionnaire in state → returns a new QuestionnaireState (smart mode)."""
        state = {"messages": []}
        result = project_intake(state)
        assert "questionnaire" in result
        assert isinstance(result["questionnaire"], QuestionnaireState)
        # The default (smart) flow asks an essential gap, not Q1-first — that
        # Q1-first behavior belonged to the retired "standard" mode.
        assert result["questionnaire"].intake_mode == "smart"
        assert isinstance(result["messages"][0], AIMessage)

    def test_records_answer(self):
        """After answering, the answer should be stored in questionnaire.answers."""
        qs = QuestionnaireState(current_question=1)
        state = {
            "messages": [HumanMessage(content="A todo app for tracking tasks")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        assert result["questionnaire"].answers[1] == "A todo app for tracking tasks"

    def test_advances_question(self):
        """After recording an answer, current_question should increment."""
        qs = QuestionnaireState(current_question=1)
        state = {
            "messages": [HumanMessage(content="A todo app")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        assert result["questionnaire"].current_question == 2

    def test_asks_next_question(self):
        """After answering Q1, should ask Q2."""
        qs = QuestionnaireState(current_question=1)
        state = {
            "messages": [HumanMessage(content="A todo app")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        assert INTAKE_QUESTIONS[2] in result["messages"][0].content

    def test_includes_progress(self):
        """Response should include progress indicator like Q5/26."""
        qs = QuestionnaireState(current_question=4)
        state = {
            "messages": [HumanMessage(content="answer")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        assert f"Q5/{TOTAL_QUESTIONS}" in result["messages"][0].content

    def test_phase_transition(self):
        """When moving from Q5 to Q6, the phase label should change to Phase 2."""
        qs = QuestionnaireState(current_question=5)
        state = {
            "messages": [HumanMessage(content="No deadlines")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        assert "Phase 2: Team & Capacity" in result["messages"][0].content

    def test_completion_sets_awaiting_confirmation(self):
        """After answering Q26, awaiting_confirmation should be True (not completed)."""
        qs = QuestionnaireState(current_question=TOTAL_QUESTIONS)
        # Fill in prior answers so the summary has content
        qs.answers = {i: f"answer {i}" for i in range(1, TOTAL_QUESTIONS)}
        state = {
            "messages": [HumanMessage(content="Markdown export please")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        assert result["questionnaire"].awaiting_confirmation is True
        assert result["questionnaire"].completed is False

    def test_completion_returns_summary(self):
        """Completion should return a summary message containing all phase headers."""
        qs = QuestionnaireState(current_question=TOTAL_QUESTIONS)
        qs.answers = {i: f"answer {i}" for i in range(1, TOTAL_QUESTIONS)}
        qs._leave_input_stage = "done"  # Skip PTO sub-loop — not under test here
        state = {
            "messages": [HumanMessage(content="Both Jira and Markdown")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        summary = result["messages"][0].content
        assert "Project Intake Summary" in summary
        assert "Phase 1: Project Context" in summary
        assert "Phase 5: Preferences & Process" in summary

    def test_returns_single_ai_message(self):
        """Each invocation should return exactly one AIMessage."""
        state = {"messages": []}
        result = project_intake(state)
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)


# ── project_intake import tests ──────────────────────────────────────


class TestProjectIntakeImports:
    """Verify project_intake is importable from the expected locations."""

    def test_importable_from_agent_package(self):
        from yeaboi.agent import project_intake as imported_fn

        assert imported_fn is project_intake

    def test_importable_from_nodes_module(self):
        from yeaboi.agent.nodes import project_intake as imported_fn

        assert imported_fn is project_intake


# ── Adaptive skip helpers ────────────────────────────────────────────


class TestExtractAnswersFromDescription:
    """Tests for _extract_answers_from_description() helper."""

    def test_valid_json_extraction(self, monkeypatch):
        """LLM returns valid JSON → parsed into dict[int, str]."""
        fake_response = AIMessage(content='{"1": "A todo app", "6": "3 engineers"}')
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _extract_answers_from_description("Building a todo app with 3 engineers")
        assert result == {1: "A todo app", 6: "3 engineers"}

    def test_empty_description_skips_llm(self, monkeypatch):
        """Empty description should return {} without calling the LLM."""
        mock_llm = MagicMock()
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _extract_answers_from_description("")
        assert result == {}
        mock_llm.invoke.assert_not_called()

    def test_whitespace_only_description_skips_llm(self, monkeypatch):
        """Whitespace-only description should return {} without calling the LLM."""
        mock_llm = MagicMock()
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _extract_answers_from_description("   \n  ")
        assert result == {}
        mock_llm.invoke.assert_not_called()

    def test_bad_json_fallback(self, monkeypatch):
        """LLM returns invalid JSON → graceful fallback to {}."""
        fake_response = AIMessage(content="This is not JSON at all")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _extract_answers_from_description("Some project description")
        assert result == {}

    def test_llm_exception_fallback(self, monkeypatch):
        """LLM raises an exception → graceful fallback to {}."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API timeout")
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _extract_answers_from_description("Some project description")
        assert result == {}

    def test_markdown_code_fence_handling(self, monkeypatch):
        """LLM wraps JSON in markdown code fences → fences are stripped."""
        fake_response = AIMessage(content='```json\n{"1": "A todo app"}\n```')
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _extract_answers_from_description("Building a todo app")
        assert result == {1: "A todo app"}

    def test_invalid_question_numbers_filtered(self, monkeypatch):
        """Question numbers outside 1–30 should be filtered out."""
        fake_response = AIMessage(content='{"0": "bad", "1": "good", "31": "bad", "abc": "bad"}')
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _extract_answers_from_description("Some description")
        assert result == {1: "good"}

    def test_empty_values_filtered(self, monkeypatch):
        """Empty string values should be filtered out."""
        fake_response = AIMessage(content='{"1": "A todo app", "2": "", "3": "   "}')
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _extract_answers_from_description("Some description")
        assert result == {1: "A todo app"}

    def test_non_dict_response(self, monkeypatch):
        """LLM returns valid JSON but not a dict → returns {}."""
        fake_response = AIMessage(content='["not", "a", "dict"]')
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _extract_answers_from_description("Some description")
        assert result == {}


class TestNextUnskippedQuestion:
    """Tests for _next_unskipped_question() helper."""

    def test_no_skips(self):
        """With no skipped questions, returns the current number."""
        assert _next_unskipped_question(1, set()) == 1

    def test_contiguous_skips(self):
        """Skipping Q1–Q3 from position 1 → returns Q4."""
        assert _next_unskipped_question(1, {1, 2, 3}) == 4

    def test_all_remaining_skipped(self):
        """All questions from current onward are skipped → returns None."""
        all_remaining = set(range(20, TOTAL_QUESTIONS + 1))
        assert _next_unskipped_question(20, all_remaining) is None

    def test_past_total(self):
        """Starting past TOTAL_QUESTIONS → returns None."""
        assert _next_unskipped_question(TOTAL_QUESTIONS + 1, set()) is None

    def test_non_contiguous_skips(self):
        """Non-contiguous skips: Q2 and Q4 skipped, starting at Q1 → returns Q1."""
        assert _next_unskipped_question(1, {2, 4}) == 1
        # Starting at Q2 → skip Q2, return Q3
        assert _next_unskipped_question(2, {2, 4}) == 3
        # Starting at Q4 → skip Q4, return Q5
        assert _next_unskipped_question(4, {2, 4}) == 5


class TestBuildExtractionSummary:
    """Tests for _build_extraction_summary() helper."""

    def test_formats_extracted_answers(self):
        """Should format question + answer pairs in order."""
        extracted = {1: "A todo app", 6: "3 engineers"}
        result = _build_extraction_summary(extracted)
        assert "Q1." in result
        assert "A todo app" in result
        assert "Q6." in result
        assert "3 engineers" in result

    def test_sorted_by_question_number(self):
        """Extracted answers should appear in question-number order."""
        extracted = {11: "React and FastAPI", 1: "A todo app", 6: "3 engineers"}
        result = _build_extraction_summary(extracted)
        q1_pos = result.index("Q1.")
        q6_pos = result.index("Q6.")
        q11_pos = result.index("Q11.")
        assert q1_pos < q6_pos < q11_pos


# ── Adaptive skip integration tests ─────────────────────────────────


class TestAdaptiveSkipIntegration:
    """Integration tests for adaptive skip logic in project_intake()."""

    @pytest.fixture(autouse=True)
    def _disable_vague_check(self, monkeypatch):
        """Disable vague-answer checking so adaptive-skip tests don't hit the LLM."""
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)

    def _mock_extract(self, monkeypatch, extracted: dict[int, str]):
        """Monkeypatch _extract_answers_from_description to return the given dict."""
        monkeypatch.setattr(
            "yeaboi.agent.nodes._extract_answers_from_description",
            lambda desc: extracted,
        )

    # NOTE: the first-call "adaptive skip" tests that lived here asserted the
    # retired "standard" mode's behavior (extractions stored as suggested_answers,
    # always start at Q1, show a "picked up N details" count). That flow has been
    # removed — smart mode (the default) auto-applies non-essential extractions and
    # asks the first essential gap instead. Smart first-invocation is covered by
    # TestSmartIntakeMode and test_smart_mode_first_invocation_* below.

    def test_suggestion_shown_on_subsequent_question(self, monkeypatch):
        """When advancing to a question with a suggestion, it should show inline."""
        qs = QuestionnaireState(current_question=5)
        qs.answers = {i: f"answer {i}" for i in range(1, 5)}
        qs.suggested_answers = {6: "3 engineers", 11: "React"}
        state = {
            "messages": [HumanMessage(content="No deadlines")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        # Q5 answered, advances to Q6 which has a suggestion
        assert result["questionnaire"].current_question == 6
        content = result["messages"][0].content
        assert "Extracted" in content
        assert "3 engineers" in content

    def test_confirmed_suggestion_recorded_as_answer(self, monkeypatch):
        """When the user confirms a suggestion (sends the suggestion text), it's recorded."""
        qs = QuestionnaireState(current_question=1)
        qs.suggested_answers = {1: "A todo app", 6: "3 engineers"}
        state = {
            # User confirmed by sending the suggestion text (REPL resolved Enter/Y)
            "messages": [HumanMessage(content="A todo app")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert result["questionnaire"].answers[1] == "A todo app"
        assert result["questionnaire"].current_question == 2

    def test_overridden_suggestion_recorded(self, monkeypatch):
        """When the user types a different answer, it overrides the suggestion."""
        qs = QuestionnaireState(current_question=1)
        qs.suggested_answers = {1: "A todo app"}
        state = {
            "messages": [HumanMessage(content="An e-commerce platform")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert result["questionnaire"].answers[1] == "An e-commerce platform"

    def test_no_suggestion_on_question_without_one(self, monkeypatch):
        """Questions without suggestions should NOT show a 'Suggested:' line."""
        qs = QuestionnaireState(current_question=1)
        qs.suggested_answers = {6: "3 engineers"}  # suggestion on Q6, not Q1
        state = {
            "messages": [HumanMessage(content="A todo app")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        # Q2 should not have a suggestion line
        content = result["messages"][0].content
        assert "Suggested" not in content


# ── _check_vague_answer unit tests ───────────────────────────────────


class TestCheckVagueAnswer:
    """Tests for the _check_vague_answer() helper.

    This function uses the LLM to judge whether an intake answer is too vague.
    All tests monkeypatch the LLM to avoid real API calls.
    """

    def test_long_answer_short_circuits(self, monkeypatch):
        """Answers longer than 100 characters skip the LLM call entirely."""
        mock_llm = MagicMock()
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        long_answer = "x" * 101
        result = _check_vague_answer("What is your project?", long_answer)
        assert result is None
        mock_llm.invoke.assert_not_called()

    def test_numeric_answer_short_circuits(self, monkeypatch):
        """Pure numeric answers (e.g. '7' for team size) skip the LLM call — never vague."""
        mock_llm = MagicMock()
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        assert _check_vague_answer("How many engineers?", "7") is None
        assert _check_vague_answer("How many sprints?", "3") is None
        assert _check_vague_answer("What velocity?", "15.5") is None
        mock_llm.invoke.assert_not_called()

    def test_no_preference_short_circuits_q11(self, monkeypatch):
        """'any' / 'no preference' for Q11 (tech stack) should skip the LLM — not vague."""
        mock_llm = MagicMock()
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        for answer in ("any", "Any", "anything", "no preference", "tbd", "flexible"):
            assert _check_vague_answer(INTAKE_QUESTIONS[11], answer, q_num=11) is None
        mock_llm.invoke.assert_not_called()

    def test_no_preference_short_circuits_q12(self, monkeypatch):
        """'none' / 'no integrations' for Q12 should skip the LLM — not vague."""
        mock_llm = MagicMock()
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        for answer in ("none", "None", "no integrations", "n/a", "no"):
            assert _check_vague_answer(INTAKE_QUESTIONS[12], answer, q_num=12) is None
        mock_llm.invoke.assert_not_called()

    def test_no_preference_only_applies_to_q11_q12(self, monkeypatch):
        """'any' for a non-tech question (e.g. Q1) should still go through the LLM."""
        mock_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = '{"vague": true, "follow_up": "Tell me more", "choices": ["A", "B"]}'
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _check_vague_answer("What is your project?", "any", q_num=1)
        assert result is not None
        mock_llm.invoke.assert_called_once()

    def test_exactly_100_chars_calls_llm(self, monkeypatch):
        """Answers of exactly 100 characters should still call the LLM (boundary)."""
        fake_response = AIMessage(content='{"vague": false}')
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        answer = "x" * 100
        _check_vague_answer("What is your project?", answer)
        mock_llm.invoke.assert_called_once()

    def test_vague_answer_returns_follow_up(self, monkeypatch):
        """LLM judges answer as vague → returns (follow_up, choices) tuple."""
        fake_response = AIMessage(
            content='{"vague": true, "follow_up": "Can you describe the main features?",'
            ' "choices": ["Task management", "E-commerce", "Social platform"]}'
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _check_vague_answer("What is your project?", "A web app")
        assert result is not None
        follow_up, choices = result
        assert follow_up == "Can you describe the main features?"
        assert choices == ("Task management", "E-commerce", "Social platform")

    def test_specific_answer_returns_none(self, monkeypatch):
        """LLM judges answer as specific → returns None."""
        fake_response = AIMessage(content='{"vague": false}')
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _check_vague_answer("What tech stack?", "React 18, FastAPI, PostgreSQL")
        assert result is None

    def test_llm_exception_returns_none(self, monkeypatch):
        """LLM raises an exception → graceful fallback, returns None."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API timeout")
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _check_vague_answer("What is your project?", "A web app")
        assert result is None

    def test_bad_json_returns_none(self, monkeypatch):
        """LLM returns invalid JSON → graceful fallback, returns None."""
        fake_response = AIMessage(content="I think this is vague")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _check_vague_answer("What is your project?", "stuff")
        assert result is None

    def test_markdown_code_fence_handling(self, monkeypatch):
        """LLM wraps JSON in markdown code fences → fences are stripped."""
        fake_response = AIMessage(
            content='```json\n{"vague": true, "follow_up": "What features?", "choices": ["Mobile app", "Web app"]}\n```'
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _check_vague_answer("What is your project?", "An app")
        assert result is not None
        follow_up, choices = result
        assert follow_up == "What features?"
        assert choices == ("Mobile app", "Web app")

    def test_non_dict_response_returns_none(self, monkeypatch):
        """LLM returns valid JSON but not a dict → returns None."""
        fake_response = AIMessage(content='["not", "a", "dict"]')
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _check_vague_answer("What is your project?", "stuff")
        assert result is None

    def test_vague_true_but_empty_follow_up(self, monkeypatch):
        """LLM says vague but follow_up is empty string → returns None."""
        fake_response = AIMessage(content='{"vague": true, "follow_up": "", "choices": ["A", "B"]}')
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _check_vague_answer("What is your project?", "stuff")
        assert result is None

    def test_vague_true_but_missing_follow_up_key(self, monkeypatch):
        """LLM says vague but omits follow_up key → returns None."""
        fake_response = AIMessage(content='{"vague": true}')
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _check_vague_answer("What is your project?", "stuff")
        assert result is None

    def test_whitespace_only_answer_not_short_circuited(self, monkeypatch):
        """Whitespace-only answer (len > 100 after strip) should still call LLM if stripped length <= 100."""
        fake_response = AIMessage(
            content='{"vague": true, "follow_up": "Please elaborate.", "choices": ["Option A", "Option B"]}'
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        # 200 spaces + 5 chars = len > 100 but strip() gives 5 chars
        padded = " " * 200 + "stuff"
        result = _check_vague_answer("What is your project?", padded)
        assert result is not None
        follow_up, choices = result
        assert follow_up == "Please elaborate."
        assert choices == ("Option A", "Option B")
        mock_llm.invoke.assert_called_once()

    def test_vague_with_no_choices_returns_empty_tuple(self, monkeypatch):
        """LLM says vague but provides no choices → returns (follow_up, ())."""
        fake_response = AIMessage(content='{"vague": true, "follow_up": "Tell me more."}')
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _check_vague_answer("What is your project?", "stuff")
        assert result is not None
        follow_up, choices = result
        assert follow_up == "Tell me more."
        assert choices == ()

    def test_vague_with_one_choice_returns_empty_tuple(self, monkeypatch):
        """LLM provides only 1 choice → too few, returns empty tuple."""
        fake_response = AIMessage(content='{"vague": true, "follow_up": "Tell me more.", "choices": ["Only one"]}')
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _check_vague_answer("What is your project?", "stuff")
        assert result is not None
        _, choices = result
        assert choices == ()

    def test_vague_with_five_choices_clamped_to_four(self, monkeypatch):
        """LLM provides 5 choices → clamped to first 4."""
        fake_response = AIMessage(
            content='{"vague": true, "follow_up": "Which type?", "choices": ["A", "B", "C", "D", "E"]}'
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        result = _check_vague_answer("What is your project?", "stuff")
        assert result is not None
        _, choices = result
        assert len(choices) == 4
        assert choices == ("A", "B", "C", "D")


# ── Follow-up probing integration tests ──────────────────────────────


class TestFollowUpProbing:
    """Integration tests for follow-up probing in project_intake().

    These test the full probing flow: vague answer → follow-up asked →
    user responds → answers combined → question advances.
    """

    def test_vague_answer_triggers_follow_up(self, monkeypatch):
        """A vague answer should trigger a follow-up and NOT advance the question."""
        choices = ("Task management", "E-commerce", "Social platform")
        monkeypatch.setattr(
            "yeaboi.agent.nodes._check_vague_answer",
            lambda q, a, n=0: ("Can you describe the main features?", choices),
        )

        qs = QuestionnaireState(current_question=1)
        state = {
            "messages": [HumanMessage(content="A web app")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        qs_out = result["questionnaire"]

        # Question should NOT advance — still on Q1
        assert qs_out.current_question == 1
        # Q1 should be marked as probed
        assert 1 in qs_out.probed_questions
        # The answer is still recorded
        assert qs_out.answers[1] == "A web app"
        # Response should contain the follow-up
        ai_msg = result["messages"][0]
        assert "Follow-up on Q1" in ai_msg.content
        assert "Can you describe the main features?" in ai_msg.content
        # Dynamic choices should be stored on questionnaire state
        assert qs_out._follow_up_choices[1] == ("Task management", "E-commerce", "Social platform")

    def test_vague_answer_without_choices_stores_nothing(self, monkeypatch):
        """A vague answer with no valid choices should NOT store _follow_up_choices."""
        monkeypatch.setattr(
            "yeaboi.agent.nodes._check_vague_answer",
            lambda q, a, n=0: ("Can you be more specific?", ()),
        )

        qs = QuestionnaireState(current_question=1)
        state = {
            "messages": [HumanMessage(content="A web app")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        qs_out = result["questionnaire"]
        assert 1 not in qs_out._follow_up_choices

    def test_follow_up_response_combines_answers_and_advances(self, monkeypatch):
        """After a follow-up, the user's second answer is combined with the first and the question advances."""
        # Disable vague check for the second pass (already probed → combines)
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)

        # Simulate: Q1 was probed, user is now responding to the follow-up
        qs = QuestionnaireState(current_question=1)
        qs.answers[1] = "A web app"
        qs.probed_questions.add(1)
        qs._follow_up_choices[1] = ("Task management", "E-commerce")
        state = {
            "messages": [HumanMessage(content="It's a task management tool with Kanban boards")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        qs_out = result["questionnaire"]

        # Question should now advance past Q1
        assert qs_out.current_question == 2
        # Answer should be combined
        assert "A web app" in qs_out.answers[1]
        assert "Follow-up detail:" in qs_out.answers[1]
        assert "task management tool with Kanban boards" in qs_out.answers[1]
        # Dynamic choices should be cleared after follow-up is answered
        assert 1 not in qs_out._follow_up_choices

    def test_max_one_probe_per_question(self, monkeypatch):
        """A question that was already probed should advance even if the follow-up answer is also vague."""
        # _check_vague_answer would say "vague" — but it shouldn't be called
        # because the probed_questions check happens first.
        call_count = {"n": 0}
        original_check = lambda q, a, n=0: None  # noqa: E731

        def tracking_check(q, a, n=0):
            call_count["n"] += 1
            return original_check(q, a, n)

        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", tracking_check)

        qs = QuestionnaireState(current_question=1)
        qs.answers[1] = "A web app"
        qs.probed_questions.add(1)
        state = {
            "messages": [HumanMessage(content="Still vague")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        # Should advance — no second probe
        assert result["questionnaire"].current_question == 2
        # _check_vague_answer should NOT be called for already-probed questions
        assert call_count["n"] == 0

    def test_specific_answer_advances_without_probing(self, monkeypatch):
        """A specific answer should advance the question without triggering a follow-up."""
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)

        qs = QuestionnaireState(current_question=1)
        state = {
            "messages": [HumanMessage(content="A task management tool for small teams with Kanban boards")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        qs_out = result["questionnaire"]

        assert qs_out.current_question == 2
        assert 1 not in qs_out.probed_questions

    def test_probing_does_not_affect_progress(self, monkeypatch):
        """A probed-but-not-advanced question should not double-count progress."""
        monkeypatch.setattr(
            "yeaboi.agent.nodes._check_vague_answer",
            lambda q, a, n=0: ("Tell me more?", ()),
        )

        qs = QuestionnaireState(current_question=1)
        state = {
            "messages": [HumanMessage(content="A web app")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        qs_out = result["questionnaire"]

        # Still on Q1 — progress should reflect 0 completed questions
        # (answer is recorded but question hasn't advanced yet)
        assert qs_out.current_question == 1
        # Progress = (answered + skipped) / total. Q1 is answered but
        # current_question hasn't moved, so the NEXT question isn't asked yet.
        # The exact progress depends on state — just verify it's not > 1/26.
        assert qs_out.progress <= 1 / TOTAL_QUESTIONS

    def test_follow_up_message_has_no_phase_label(self, monkeypatch):
        """Follow-up messages should feel conversational — no phase header or progress indicator."""
        monkeypatch.setattr(
            "yeaboi.agent.nodes._check_vague_answer",
            lambda q, a, n=0: ("What kind of users?", ()),
        )

        qs = QuestionnaireState(current_question=1)
        state = {
            "messages": [HumanMessage(content="An app")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        content = result["messages"][0].content

        # Should NOT contain phase labels or progress indicators
        assert "Phase 1" not in content
        assert f"Q2/{TOTAL_QUESTIONS}" not in content
        # Should contain the follow-up
        assert "What kind of users?" in content

    def test_probing_on_last_free_text_question_then_completes(self, monkeypatch):
        """Probing on the last free-text question (Q23), then answering follow-up → advances.

        Note: Q24, Q25, Q26 are choice questions (skip vagueness check), so we test
        with Q23 which is the last free-text question before the choices run.
        """
        # First call: vague answer on Q23 → follow-up
        monkeypatch.setattr(
            "yeaboi.agent.nodes._check_vague_answer",
            lambda q, a, n=0: ("What specifically is out of scope?", ("Mobile app", "Analytics")),
        )

        qs = QuestionnaireState(current_question=23)
        qs.answers = {i: f"answer {i}" for i in range(1, 23)}
        state = {
            "messages": [HumanMessage(content="Nothing specific")],
            "questionnaire": qs,
        }

        result1 = project_intake(state)
        assert result1["questionnaire"].completed is False
        assert 23 in result1["questionnaire"].probed_questions

        # Second call: follow-up answer on Q23 → should advance to Q24
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)

        state2 = {
            "messages": [HumanMessage(content="Mobile app and advanced analytics are out of scope")],
            "questionnaire": result1["questionnaire"],
        }

        result2 = project_intake(state2)
        # Should advance to Q24 (not complete yet)
        assert result2["questionnaire"].current_question == 24
        # Combined answer should be stored
        assert "Follow-up detail:" in result2["questionnaire"].answers[23]


# ── Skip intent detection tests ──────────────────────────────────────


class TestIsSkipIntent:
    """Tests for the _is_skip_intent() helper.

    Deterministic keyword matching — no LLM call. Tests cover exact matches,
    substring matches, case insensitivity, and false-positive prevention.
    """

    # ── Exact matches ────────────────────────────────────────────────

    @pytest.mark.parametrize("word", ["skip", "pass", "next", "n/a", "na", "idk", "-", "none"])
    def test_exact_matches(self, word):
        """Each exact-match keyword should be detected as skip intent."""
        assert _is_skip_intent(word) is True

    @pytest.mark.parametrize("word", ["SKIP", "Pass", "NEXT", "N/A", "NA", "IDK", "None", "NONE"])
    def test_exact_matches_case_insensitive(self, word):
        """Exact matches should be case-insensitive."""
        assert _is_skip_intent(word) is True

    def test_exact_match_with_whitespace(self):
        """Leading/trailing whitespace should be stripped before matching."""
        assert _is_skip_intent("  skip  ") is True
        assert _is_skip_intent("\tpass\n") is True

    # ── Substring matches ────────────────────────────────────────────

    @pytest.mark.parametrize(
        "phrase",
        [
            "I don't know",
            "I dont know",
            "not sure",
            "unsure",
            "no idea",
            "don't know",
            "dont know",
            "skip this",
            "pass on this",
            "move on",
            "no answer",
        ],
    )
    def test_substring_matches(self, phrase):
        """Each substring phrase should be detected as skip intent."""
        assert _is_skip_intent(phrase) is True

    def test_substring_in_longer_message(self):
        """Substring match should work even within a longer message."""
        assert _is_skip_intent("I'm not sure about this one") is True
        assert _is_skip_intent("Honestly, I don't know the answer") is True
        assert _is_skip_intent("Let's just move on to the next question") is True

    def test_substring_case_insensitive(self):
        """Substring matches should be case-insensitive."""
        assert _is_skip_intent("I DON'T KNOW") is True
        assert _is_skip_intent("Not Sure") is True
        assert _is_skip_intent("NO IDEA") is True

    # ── False-positive prevention ────────────────────────────────────

    def test_password_not_false_positive(self):
        """'pass' only exact-matches — 'password' should NOT trigger."""
        assert _is_skip_intent("We use a password manager") is False

    def test_substantial_answer_not_false_positive(self):
        """A real answer should not trigger skip detection."""
        assert _is_skip_intent("React 18, FastAPI, PostgreSQL, Redis") is False

    def test_next_in_sentence_not_false_positive(self):
        """'next' only exact-matches — 'Next.js' should NOT trigger."""
        assert _is_skip_intent("We use Next.js for the frontend") is False

    def test_empty_string(self):
        """Empty string should not be detected as skip intent."""
        assert _is_skip_intent("") is False

    def test_whitespace_only(self):
        """Whitespace-only string should not be detected as skip intent."""
        assert _is_skip_intent("   ") is False


# ── Question defaults tests ──────────────────────────────────────────


class TestQuestionDefaults:
    """Tests for the QUESTION_DEFAULTS dict in intake.py."""

    def test_valid_question_numbers(self):
        """All keys in QUESTION_DEFAULTS should be valid question numbers (1–26)."""
        for q_num in QUESTION_DEFAULTS:
            assert 1 <= q_num <= TOTAL_QUESTIONS, f"Q{q_num} is not a valid question number"

    def test_essential_questions_excluded(self):
        """Essential questions (Q1–Q4, Q6, Q11, Q15) should NOT have defaults."""
        essential = {1, 2, 3, 4, 6, 11, 15}
        for q_num in essential:
            assert q_num not in QUESTION_DEFAULTS, f"Q{q_num} is essential and should not have a default"

    def test_all_defaults_non_empty(self):
        """All default values should be non-empty strings."""
        for q_num, default in QUESTION_DEFAULTS.items():
            assert isinstance(default, str), f"Q{q_num} default is not a string"
            assert default.strip(), f"Q{q_num} default is empty"


# ── Skip handling integration tests ──────────────────────────────────


class TestSkipHandling:
    """Integration tests for skip handling in project_intake().

    Tests the full skip flow: user says "skip" → default applied (or gap
    flagged) → question advances → acknowledgment shown.
    """

    @pytest.fixture(autouse=True)
    def _disable_vague_check(self, monkeypatch):
        """Disable vague-answer checking so skip tests don't hit the LLM."""
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)

    # ── Default storage & advancement ────────────────────────────────

    def test_skip_with_default_stores_default_value(self):
        """Skipping a question with a default should store the default in answers."""
        qs = QuestionnaireState(current_question=5)  # Q5 has a default
        state = {
            "messages": [HumanMessage(content="skip")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert result["questionnaire"].answers[5] == QUESTION_DEFAULTS[5]

    def test_skip_with_default_marks_defaulted(self):
        """Skipping a question with a default should add it to defaulted_questions."""
        qs = QuestionnaireState(current_question=8)  # Q8 has a default
        state = {
            "messages": [HumanMessage(content="skip")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert 8 in result["questionnaire"].defaulted_questions

    def test_skip_with_default_advances_question(self):
        """After skip, should advance to the next question."""
        qs = QuestionnaireState(current_question=5)  # Q5 has default
        state = {
            "messages": [HumanMessage(content="skip")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert result["questionnaire"].current_question == 6

    def test_skip_essential_question_no_default(self):
        """Skipping an essential question (no default) should add to skipped_questions."""
        qs = QuestionnaireState(current_question=1)  # Q1 is essential — no default
        state = {
            "messages": [HumanMessage(content="skip")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert 1 in result["questionnaire"].skipped_questions
        assert 1 not in result["questionnaire"].answers

    def test_skip_essential_advances_question(self):
        """Skipping an essential question should still advance."""
        qs = QuestionnaireState(current_question=1)
        state = {
            "messages": [HumanMessage(content="skip")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert result["questionnaire"].current_question == 2

    # ── Acknowledgment messages ──────────────────────────────────────

    def test_skip_default_acknowledgment_mentions_assumption(self):
        """When a default is applied, the acknowledgment should mention the assumption."""
        qs = QuestionnaireState(current_question=8)  # Q8: "2 weeks"
        state = {
            "messages": [HumanMessage(content="skip")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        content = result["messages"][0].content
        assert "2 weeks" in content
        assert "assume" in content.lower()

    def test_skip_no_default_acknowledgment_mentions_skipped(self):
        """When no default exists, acknowledgment should say 'skipped'."""
        qs = QuestionnaireState(current_question=1)
        state = {
            "messages": [HumanMessage(content="skip")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        content = result["messages"][0].content
        assert "Skipped Q1" in content

    def test_skip_still_shows_next_question(self):
        """After skip acknowledgment, the next question should be shown."""
        qs = QuestionnaireState(current_question=5)
        state = {
            "messages": [HumanMessage(content="skip")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        content = result["messages"][0].content
        assert INTAKE_QUESTIONS[6] in content

    # ── Skip during follow-up probe ──────────────────────────────────

    def test_skip_during_probe_keeps_original_answer(self):
        """Skipping during a follow-up probe should keep the original answer unchanged."""
        qs = QuestionnaireState(current_question=1)
        qs.answers[1] = "A web app"
        qs.probed_questions.add(1)
        state = {
            "messages": [HumanMessage(content="skip")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        # Original answer preserved — NOT combined with "skip"
        assert result["questionnaire"].answers[1] == "A web app"
        # Should advance past Q1
        assert result["questionnaire"].current_question == 2

    def test_skip_during_probe_acknowledgment(self):
        """Skip during probe should acknowledge keeping the earlier answer."""
        qs = QuestionnaireState(current_question=3)
        qs.answers[3] = "Developers"
        qs.probed_questions.add(3)
        state = {
            "messages": [HumanMessage(content="I don't know")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        content = result["messages"][0].content
        assert "earlier answer" in content.lower()

    # ── Bypass vague check ───────────────────────────────────────────

    def test_skip_bypasses_vague_check(self, monkeypatch):
        """Skip should bypass the vague-answer check entirely."""
        check_called = {"n": 0}

        def tracking_check(q, a, n=0):
            check_called["n"] += 1
            return None

        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", tracking_check)

        qs = QuestionnaireState(current_question=5)
        state = {
            "messages": [HumanMessage(content="skip")],
            "questionnaire": qs,
        }

        project_intake(state)
        assert check_called["n"] == 0

    # ── Summary flagging ─────────────────────────────────────────────

    def test_defaulted_answer_flagged_in_summary(self):
        """Defaulted answers should be marked with '*(assumed default)*' in the summary."""
        qs = QuestionnaireState(current_question=TOTAL_QUESTIONS)
        qs.answers = {i: f"answer {i}" for i in range(1, TOTAL_QUESTIONS)}
        qs.answers[8] = "2 weeks"
        qs.defaulted_questions.add(8)
        qs._leave_input_stage = "done"  # Skip PTO sub-loop — not under test here
        state = {
            "messages": [HumanMessage(content="Markdown export")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        summary = result["messages"][0].content
        assert "*(assumed default)*" in summary

    def test_skipped_no_default_flagged_in_summary(self):
        """Questions skipped with no default should show '_skipped (no default available)_'."""
        qs = QuestionnaireState()
        qs.answers = {i: f"answer {i}" for i in range(1, TOTAL_QUESTIONS + 1)}
        # Remove Q1's answer and mark it as skipped (no default)
        del qs.answers[1]
        qs.skipped_questions.add(1)
        summary = _build_intake_summary(qs)
        assert "_skipped (no default available)_" in summary

    def test_summary_footer_counts_defaults(self):
        """Summary should include a footer noting how many defaults were used."""
        qs = QuestionnaireState()
        qs.answers = {i: f"answer {i}" for i in range(1, TOTAL_QUESTIONS + 1)}
        qs.answers[5] = "No hard deadlines"
        qs.answers[8] = "2 weeks"
        qs.defaulted_questions = {5, 8}
        summary = _build_intake_summary(qs)
        assert "2 answer(s) above are assumed defaults" in summary

    # ── Progress counting ────────────────────────────────────────────

    def test_defaulted_question_counts_toward_progress(self):
        """Defaulted questions have answers stored, so they count toward progress."""
        qs = QuestionnaireState(current_question=6)
        qs.answers = {5: "No hard deadlines"}
        qs.defaulted_questions = {5}
        assert qs.progress == pytest.approx(1 / TOTAL_QUESTIONS)

    def test_skipped_essential_counts_toward_progress(self):
        """Skipped essential questions (in skipped_questions) count toward progress."""
        qs = QuestionnaireState(current_question=2)
        qs.skipped_questions = {1}
        assert qs.progress == pytest.approx(1 / TOTAL_QUESTIONS)

    # ── Various skip phrases ─────────────────────────────────────────

    @pytest.mark.parametrize("phrase", ["skip", "I don't know", "idk", "n/a", "pass", "not sure", "no idea"])
    def test_various_skip_phrases_trigger_skip(self, phrase):
        """Multiple skip phrases should all trigger skip handling in project_intake."""
        qs = QuestionnaireState(current_question=8)  # Q8 has a default
        state = {
            "messages": [HumanMessage(content=phrase)],
            "questionnaire": qs,
        }

        result = project_intake(state)
        # Should have stored the default and advanced
        assert result["questionnaire"].answers[8] == QUESTION_DEFAULTS[8]
        assert result["questionnaire"].current_question == 9

    # ── Skip on last question completes questionnaire ────────────────

    def test_skip_last_question_sets_awaiting_confirmation(self):
        """Skipping Q26 should set awaiting_confirmation (not completed)."""
        qs = QuestionnaireState(current_question=TOTAL_QUESTIONS)
        qs.answers = {i: f"answer {i}" for i in range(1, TOTAL_QUESTIONS)}
        qs._leave_input_stage = "done"  # Skip PTO sub-loop — not under test here
        state = {
            "messages": [HumanMessage(content="skip")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert result["questionnaire"].awaiting_confirmation is True
        assert result["questionnaire"].completed is False
        assert "Project Intake Summary" in result["messages"][0].content
        assert "accept" in result["messages"][0].content.lower()


# ── Confirm intent detection tests ────────────────────────────────────


class TestIsConfirmIntent:
    """Tests for the _is_confirm_intent() helper.

    Deterministic keyword matching — no LLM call. Tests cover exact matches,
    case insensitivity, and false-positive prevention.
    """

    @pytest.mark.parametrize(
        "word",
        ["confirm", "confirmed", "yes", "y", "looks good", "lgtm", "proceed", "go ahead", "ok", "okay"],
    )
    def test_confirm_keywords(self, word):
        """Each confirm keyword should be detected as confirmation intent."""
        assert _is_confirm_intent(word) is True

    @pytest.mark.parametrize("word", ["CONFIRM", "Yes", "LGTM", "Looks Good", "OK", "Okay"])
    def test_confirm_case_insensitive(self, word):
        """Confirm matches should be case-insensitive."""
        assert _is_confirm_intent(word) is True

    def test_confirm_with_whitespace(self):
        """Leading/trailing whitespace should be stripped before matching."""
        assert _is_confirm_intent("  confirm  ") is True
        assert _is_confirm_intent("\tyes\n") is True

    @pytest.mark.parametrize(
        "text",
        [
            "I want to change Q5",
            "actually can we revise the team size",
            "no",
            "wait",
            "",
            "   ",
            "confirming something else",
        ],
    )
    def test_non_confirm_returns_false(self, text):
        """Non-confirm messages should return False."""
        assert _is_confirm_intent(text) is False


# ── Confirmation flow integration tests ───────────────────────────────


class TestConfirmation:
    """Integration tests for the confirmation gate in project_intake().

    After the last question is answered, the questionnaire enters
    awaiting_confirmation mode. The user must confirm before the
    questionnaire is marked as completed.
    """

    @pytest.fixture(autouse=True)
    def _disable_vague_check(self, monkeypatch):
        """Disable vague-answer checking so confirmation tests don't hit the LLM."""
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)

    def _make_awaiting_state(self) -> QuestionnaireState:
        """Build a QuestionnaireState in awaiting_confirmation mode."""
        qs = QuestionnaireState(current_question=TOTAL_QUESTIONS + 1)
        qs.answers = {i: f"answer {i}" for i in range(1, TOTAL_QUESTIONS + 1)}
        qs.awaiting_confirmation = True
        return qs

    def test_confirm_sets_completed(self):
        """Typing 'confirm' should set completed=True and awaiting_confirmation=False."""
        qs = self._make_awaiting_state()
        state = {
            "messages": [HumanMessage(content="confirm")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert result["questionnaire"].completed is True
        assert result["questionnaire"].awaiting_confirmation is False

    def test_yes_sets_completed(self):
        """Typing 'yes' should also confirm and complete."""
        qs = self._make_awaiting_state()
        state = {
            "messages": [HumanMessage(content="yes")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert result["questionnaire"].completed is True
        assert result["questionnaire"].awaiting_confirmation is False

    def test_confirm_returns_proceed_message(self):
        """Confirmation should return a message indicating the agent will proceed."""
        qs = self._make_awaiting_state()
        state = {
            "messages": [HumanMessage(content="confirm")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        content = result["messages"][0].content
        assert "locked in" in content.lower() or "analyze" in content.lower()

    def test_non_confirm_re_prompts(self):
        """Non-confirm text should re-show the summary and stay in awaiting_confirmation."""
        qs = self._make_awaiting_state()
        state = {
            "messages": [HumanMessage(content="I want to change Q5")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert result["questionnaire"].awaiting_confirmation is True
        assert result["questionnaire"].completed is False
        content = result["messages"][0].content
        assert "Project Intake Summary" in content
        assert "accept" in content.lower()

    def test_last_question_sets_awaiting_confirmation(self):
        """Answering the last question should set awaiting_confirmation=True, completed=False."""
        qs = QuestionnaireState(current_question=TOTAL_QUESTIONS)
        qs.answers = {i: f"answer {i}" for i in range(1, TOTAL_QUESTIONS)}
        qs._leave_input_stage = "done"  # Skip PTO sub-loop — not under test here
        state = {
            "messages": [HumanMessage(content="Final answer")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert result["questionnaire"].awaiting_confirmation is True
        assert result["questionnaire"].completed is False
        assert "Project Intake Summary" in result["messages"][0].content

    def test_confirm_after_re_prompt_succeeds(self):
        """After a re-prompt, confirming should complete the questionnaire."""
        qs = self._make_awaiting_state()

        # First: non-confirm → re-prompt
        state1 = {
            "messages": [HumanMessage(content="change something")],
            "questionnaire": qs,
        }
        result1 = project_intake(state1)
        assert result1["questionnaire"].awaiting_confirmation is True

        # Second: confirm → completed
        state2 = {
            "messages": [HumanMessage(content="confirm")],
            "questionnaire": result1["questionnaire"],
        }
        result2 = project_intake(state2)
        assert result2["questionnaire"].completed is True
        assert result2["questionnaire"].awaiting_confirmation is False

    def test_route_entry_still_routes_to_intake_while_awaiting(self):
        """While awaiting_confirmation, route_entry should still route to project_intake."""
        qs = self._make_awaiting_state()
        state = {"messages": [], "questionnaire": qs}
        assert route_entry(state) == "project_intake"


# ── Edit intent detection tests ────────────────────────────────────


class TestParseEditIntent:
    """Tests for the _parse_edit_intent() helper.

    Deterministic regex matching — no LLM call. Tests cover re-ask patterns,
    inline edit patterns, edge cases, and false-positive prevention.
    """

    # ── Re-ask patterns (no inline answer) ────────────────────────

    @pytest.mark.parametrize(
        "text",
        [
            "Q6",
            "q6",
            "Q06",
            "edit Q6",
            "Edit Q6",
            "change Q6",
            "revise Q6",
            "update Q6",
            "question 6",
            # Bare numbers — Q prefix is now optional (entered after "Which Q?" prompt)
            "6",
            "edit 6",
        ],
    )
    def test_reask_patterns(self, text):
        """Various re-ask formats should return (question_num, None)."""
        result = _parse_edit_intent(text)
        assert result is not None
        assert result[0] == 6
        assert result[1] is None

    def test_bare_number_large(self):
        """Bare two-digit question number should also be recognised."""
        assert _parse_edit_intent("25") == (25, None)

    # ── Inline edit patterns ──────────────────────────────────────

    @pytest.mark.parametrize(
        ("text", "expected_answer"),
        [
            ("Q6: 5 engineers", "5 engineers"),
            ("q6: 5 engineers", "5 engineers"),
            ("Q6 = 5 engineers", "5 engineers"),
            ("edit Q6: new answer here", "new answer here"),
            ("Q1: A task management tool", "A task management tool"),
            # Bare number with inline answer
            ("6: 5 engineers", "5 engineers"),
            ("25: new answer", "new answer"),
        ],
    )
    def test_inline_patterns(self, text, expected_answer):
        """Inline edit formats should return (question_num, answer)."""
        result = _parse_edit_intent(text)
        assert result is not None
        assert result[1] == expected_answer

    # ── Edge cases ────────────────────────────────────────────────

    def test_question_number_boundaries(self):
        """0 and 31 should return None (out of range 1–30) regardless of Q prefix."""
        assert _parse_edit_intent("Q0") is None
        assert _parse_edit_intent("Q31") is None
        assert _parse_edit_intent("0") is None
        assert _parse_edit_intent("31") is None

    def test_valid_boundary_questions(self):
        """Q1 and Q30 should be valid — also without the Q prefix."""
        assert _parse_edit_intent("Q1") == (1, None)
        assert _parse_edit_intent("Q30") == (30, None)
        assert _parse_edit_intent("1") == (1, None)
        assert _parse_edit_intent("30") == (30, None)

    def test_whitespace_handling(self):
        """Leading/trailing whitespace should be stripped."""
        result = _parse_edit_intent("  Q6  ")
        assert result == (6, None)

    def test_inline_with_empty_answer_returns_none(self):
        """'Q6: ' (colon then whitespace only) should return None — no valid answer provided."""
        assert _parse_edit_intent("Q6:   ") is None

    # ── Non-edit returns None ─────────────────────────────────────

    @pytest.mark.parametrize(
        "text",
        [
            "confirm",
            "yes",
            "I want to change something",
            "change the team size",
            "looks good",
            "",
            "   ",
            "Q",
            "edit",
            "42",
        ],
    )
    def test_non_edit_returns_none(self, text):
        """Non-edit messages should return None."""
        assert _parse_edit_intent(text) is None


# ── Edit flow integration tests ────────────────────────────────────


class TestEditFlow:
    """Integration tests for the edit flow in project_intake().

    Tests inline edits, re-ask flow, skip during re-ask, and
    confirm after editing.
    """

    @pytest.fixture(autouse=True)
    def _disable_vague_check(self, monkeypatch):
        """Disable vague-answer checking so edit tests don't hit the LLM."""
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)

    def _make_awaiting_state(self) -> QuestionnaireState:
        """Build a QuestionnaireState in awaiting_confirmation mode."""
        qs = QuestionnaireState(current_question=TOTAL_QUESTIONS + 1)
        qs.answers = {i: f"answer {i}" for i in range(1, TOTAL_QUESTIONS + 1)}
        qs.awaiting_confirmation = True
        return qs

    # ── Inline edit ───────────────────────────────────────────────

    def test_inline_edit_updates_answer(self):
        """'Q6: 5 engineers' should update Q6's answer and re-show summary."""
        qs = self._make_awaiting_state()
        state = {
            "messages": [HumanMessage(content="Q6: 5 engineers")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert result["questionnaire"].answers[6] == "5 engineers"
        assert result["questionnaire"].awaiting_confirmation is True
        assert result["questionnaire"].completed is False
        content = result["messages"][0].content
        assert "Updated Q6" in content
        assert "Project Intake Summary" in content
        assert "accept" in content.lower()

    def test_inline_edit_clears_defaulted_flag(self):
        """Inline editing a defaulted question should remove its defaulted flag."""
        qs = self._make_awaiting_state()
        qs.defaulted_questions.add(8)
        state = {
            "messages": [HumanMessage(content="Q8: 3 weeks")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert result["questionnaire"].answers[8] == "3 weeks"
        assert 8 not in result["questionnaire"].defaulted_questions

    def test_inline_edit_clears_skipped_flag(self):
        """Inline editing a skipped question should remove its skipped flag."""
        qs = self._make_awaiting_state()
        qs.skipped_questions.add(1)
        del qs.answers[1]
        state = {
            "messages": [HumanMessage(content="Q1: A task management app")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert result["questionnaire"].answers[1] == "A task management app"
        assert 1 not in result["questionnaire"].skipped_questions

    # ── Re-ask flow ───────────────────────────────────────────────

    def test_reask_sets_editing_question(self):
        """'edit Q6' should set editing_question and show the question text."""
        qs = self._make_awaiting_state()
        state = {
            "messages": [HumanMessage(content="edit Q6")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert result["questionnaire"].editing_question == 6
        content = result["messages"][0].content
        assert "Q6." in content
        assert "Current answer:" in content
        assert "answer 6" in content
        assert "Enter your new answer" in content

    def test_reask_then_answer_updates_and_reshows_summary(self):
        """After re-ask, the user's answer should update the answer and re-show summary."""
        qs = self._make_awaiting_state()
        qs.editing_question = 6
        state = {
            "messages": [HumanMessage(content="10 engineers")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert result["questionnaire"].answers[6] == "10 engineers"
        assert result["questionnaire"].editing_question is None
        assert result["questionnaire"].awaiting_confirmation is True
        content = result["messages"][0].content
        assert "Project Intake Summary" in content
        assert "accept" in content.lower()

    def test_reask_clears_defaulted_flag(self):
        """Re-answering a defaulted question should remove the defaulted flag."""
        qs = self._make_awaiting_state()
        qs.editing_question = 8
        qs.defaulted_questions.add(8)
        state = {
            "messages": [HumanMessage(content="3 weeks")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert result["questionnaire"].answers[8] == "3 weeks"
        assert 8 not in result["questionnaire"].defaulted_questions

    # ── Skip during re-ask ────────────────────────────────────────

    def test_skip_during_reask_keeps_current_answer(self):
        """Typing 'skip' during re-ask should keep the current answer unchanged."""
        qs = self._make_awaiting_state()
        qs.editing_question = 6
        state = {
            "messages": [HumanMessage(content="skip")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        assert result["questionnaire"].answers[6] == "answer 6"
        assert result["questionnaire"].editing_question is None
        content = result["messages"][0].content
        assert "Project Intake Summary" in content

    # ── Non-edit non-confirm shows help ───────────────────────────

    def test_non_edit_non_confirm_shows_help(self):
        """A message that's neither confirm nor edit should show edit format help."""
        qs = self._make_awaiting_state()
        state = {
            "messages": [HumanMessage(content="I want to change the team size")],
            "questionnaire": qs,
        }

        result = project_intake(state)
        content = result["messages"][0].content
        assert "Q6: new answer" in content
        assert "edit Q6" in content
        assert "Project Intake Summary" in content
        assert "accept" in content.lower()

    # ── Confirm after edit ────────────────────────────────────────

    def test_confirm_after_inline_edit(self):
        """After an inline edit, confirming should complete the questionnaire."""
        qs = self._make_awaiting_state()

        # Inline edit Q6
        state1 = {
            "messages": [HumanMessage(content="Q6: 5 engineers")],
            "questionnaire": qs,
        }
        result1 = project_intake(state1)
        assert result1["questionnaire"].answers[6] == "5 engineers"

        # Confirm
        state2 = {
            "messages": [HumanMessage(content="confirm")],
            "questionnaire": result1["questionnaire"],
        }
        result2 = project_intake(state2)
        assert result2["questionnaire"].completed is True
        assert result2["questionnaire"].awaiting_confirmation is False


# ── Velocity extraction helpers ──────────────────────────────────────


class TestParseFirstInt:
    """Tests for the _parse_first_int() helper.

    Deterministic regex extraction — no LLM call. Parametrized over
    typical intake answer formats.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("3", 3),
            ("3 engineers", 3),
            ("About 5 people", 5),
            ("velocity is 20 points", 20),
            ("I have 10 developers on the team", 10),
            ("42", 42),
        ],
    )
    def test_extracts_first_integer(self, text, expected):
        """Should extract the first integer from natural-language text."""
        assert _parse_first_int(text) == expected

    @pytest.mark.parametrize(
        "text",
        ["", "no numbers here", "N/A", "skip", "none"],
    )
    def test_returns_none_when_no_digits(self, text):
        """Should return None when text contains no digits."""
        assert _parse_first_int(text) is None


class TestExtractTeamAndVelocity:
    """Tests for the _extract_team_and_velocity() helper.

    Tests the orchestration of Q6/Q9 parsing + default velocity calculation.
    """

    def test_explicit_velocity(self):
        """When both Q6 and Q9 have numbers, use the explicit velocity."""
        qs = QuestionnaireState()
        qs.answers = {6: "3 engineers", 9: "20 points per sprint"}
        result = _extract_team_and_velocity(qs)
        assert result["team_size"] == 3
        assert result["velocity_per_sprint"] == 20
        assert result["_velocity_was_calculated"] is False

    def test_default_velocity_when_q9_defaulted(self):
        """When Q9 was defaulted, calculate velocity as team_size * 5."""
        qs = QuestionnaireState()
        qs.answers = {6: "4 engineers", 9: "No historical velocity — will use default"}
        qs.defaulted_questions.add(9)
        result = _extract_team_and_velocity(qs)
        assert result["team_size"] == 4
        assert result["velocity_per_sprint"] == 4 * _VELOCITY_PER_ENGINEER
        assert result["_velocity_was_calculated"] is True

    def test_default_velocity_when_q9_missing(self):
        """When Q9 has no answer at all, calculate velocity as team_size * 5."""
        qs = QuestionnaireState()
        qs.answers = {6: "3 engineers"}
        result = _extract_team_and_velocity(qs)
        assert result["team_size"] == 3
        assert result["velocity_per_sprint"] == 3 * _VELOCITY_PER_ENGINEER
        assert result["_velocity_was_calculated"] is True

    def test_default_velocity_when_q9_has_no_number(self):
        """When Q9 answer has no parseable number, calculate default velocity."""
        qs = QuestionnaireState()
        qs.answers = {6: "5 engineers", 9: "Not sure yet"}
        result = _extract_team_and_velocity(qs)
        assert result["team_size"] == 5
        assert result["velocity_per_sprint"] == 5 * _VELOCITY_PER_ENGINEER
        assert result["_velocity_was_calculated"] is True

    def test_empty_dict_when_q6_missing(self):
        """When Q6 has no answer, return empty dict."""
        qs = QuestionnaireState()
        qs.answers = {9: "20 points per sprint"}
        result = _extract_team_and_velocity(qs)
        assert result == {}

    def test_empty_dict_when_q6_unparseable(self):
        """When Q6 has no parseable number, return empty dict."""
        qs = QuestionnaireState()
        qs.answers = {6: "a few people", 9: "20 points"}
        result = _extract_team_and_velocity(qs)
        assert result == {}

    def test_zero_team_size_returns_empty(self):
        """When Q6 parses to 0, return empty dict (0 engineers is invalid)."""
        qs = QuestionnaireState()
        qs.answers = {6: "0 engineers"}
        result = _extract_team_and_velocity(qs)
        assert result == {}

    def test_jira_per_dev_scales_by_team_size(self):
        """When Jira per-dev velocity is set, velocity = per_dev × Q6 team size.

        E.g. Jira team avg 25 pts with 5 devs → per_dev = 5.0. If feature
        team is 2 devs → velocity = 10 pts/sprint.
        """
        qs = QuestionnaireState()
        qs.answers = {6: "2 engineers"}
        qs._jira_per_dev_velocity = 5.0  # from Jira: 25/5 team members
        result = _extract_team_and_velocity(qs)
        assert result["team_size"] == 2
        assert result["velocity_per_sprint"] == 10  # 5.0 × 2
        assert result["_velocity_was_calculated"] is False

    def test_jira_per_dev_recalculates_on_q6_change(self):
        """Changing Q6 should recalculate velocity from per-dev rate."""
        qs = QuestionnaireState()
        qs._jira_per_dev_velocity = 5.0

        # First with 2 engineers
        qs.answers = {6: "2 engineers"}
        result = _extract_team_and_velocity(qs)
        assert result["velocity_per_sprint"] == 10

        # Change to 3 engineers
        qs.answers[6] = "3 engineers"
        result = _extract_team_and_velocity(qs)
        assert result["velocity_per_sprint"] == 15  # 5.0 × 3

    def test_jira_per_dev_fallback_on_zero(self):
        """If per_dev × team_size is 0, fall back to default calculation."""
        qs = QuestionnaireState()
        qs.answers = {6: "2 engineers"}
        qs._jira_per_dev_velocity = 0.0
        result = _extract_team_and_velocity(qs)
        # Falls back to team_size * _VELOCITY_PER_ENGINEER
        assert result["velocity_per_sprint"] == 2 * _VELOCITY_PER_ENGINEER


# ── Velocity extraction integration tests ────────────────────────────


class TestConfirmationVelocity:
    """Integration tests for velocity extraction during confirmation.

    Verifies that confirming the intake summary extracts team_size and
    velocity_per_sprint into the state dict, and shows velocity info
    in the confirmation message.
    """

    @pytest.fixture(autouse=True)
    def _disable_vague_check(self, monkeypatch):
        """Disable vague-answer checking so confirmation tests don't hit the LLM."""
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)

    def _make_awaiting_state(self, **answer_overrides) -> QuestionnaireState:
        """Build a QuestionnaireState in awaiting_confirmation mode."""
        qs = QuestionnaireState(current_question=TOTAL_QUESTIONS + 1)
        qs.answers = {i: f"answer {i}" for i in range(1, TOTAL_QUESTIONS + 1)}
        qs.answers[6] = "3 engineers"  # parseable team size
        qs.answers[9] = "15 points per sprint"  # parseable velocity
        qs.answers.update(answer_overrides)
        qs.awaiting_confirmation = True
        return qs

    def test_confirm_extracts_team_size(self):
        """Confirming should extract team_size from Q6 into the state dict."""
        qs = self._make_awaiting_state()
        state = {"messages": [HumanMessage(content="confirm")], "questionnaire": qs}
        result = project_intake(state)
        assert result["team_size"] == 3

    def test_confirm_extracts_explicit_velocity(self):
        """Confirming should extract explicit velocity from Q9."""
        qs = self._make_awaiting_state()
        state = {"messages": [HumanMessage(content="confirm")], "questionnaire": qs}
        result = project_intake(state)
        assert result["velocity_per_sprint"] == 15

    def test_confirm_calculates_default_velocity(self):
        """When Q9 is defaulted, velocity should be team_size * 5."""
        qs = self._make_awaiting_state()
        qs.defaulted_questions.add(9)
        state = {"messages": [HumanMessage(content="confirm")], "questionnaire": qs}
        result = project_intake(state)
        assert result["team_size"] == 3
        assert result["velocity_per_sprint"] == 3 * _VELOCITY_PER_ENGINEER

    def test_confirm_message_shows_explicit_velocity(self):
        """Confirmation message should show velocity without 'calculated' note."""
        qs = self._make_awaiting_state()
        state = {"messages": [HumanMessage(content="confirm")], "questionnaire": qs}
        result = project_intake(state)
        content = result["messages"][0].content
        assert "3 engineer(s)" in content
        assert "15 pts/sprint" in content
        assert "calculated" not in content

    def test_confirm_message_shows_calculated_velocity(self):
        """Confirmation message should note when velocity was calculated."""
        qs = self._make_awaiting_state()
        qs.defaulted_questions.add(9)
        state = {"messages": [HumanMessage(content="confirm")], "questionnaire": qs}
        result = project_intake(state)
        content = result["messages"][0].content
        assert "3 engineer(s)" in content
        assert f"3 × {_VELOCITY_PER_ENGINEER}" in content

    def test_no_velocity_when_q6_unparseable(self):
        """When Q6 has no number, confirm still works but no velocity fields."""
        qs = self._make_awaiting_state()
        qs.answers[6] = "a few people"
        state = {"messages": [HumanMessage(content="confirm")], "questionnaire": qs}
        result = project_intake(state)
        assert result["questionnaire"].completed is True
        assert "team_size" not in result
        assert "velocity_per_sprint" not in result


class TestExtractCapacityDeductions:
    """Tests for _extract_capacity_deductions() — Q29 (unplanned %) extraction.

    Q27/Q28/Q30 moved to capacity_check node; only Q29 stays in intake.
    """

    def test_unplanned_pct_from_choice(self):
        """Q29 choice '15%' should extract 15."""
        qs = QuestionnaireState(completed=True)
        qs.answers[29] = "15%"
        result = _extract_capacity_deductions(qs)
        assert result["capacity_unplanned_leave_pct"] == 15

    def test_unplanned_pct_default_when_missing(self):
        """Missing Q29 should default to 10%."""
        qs = QuestionnaireState(completed=True)
        result = _extract_capacity_deductions(qs)
        assert result["capacity_unplanned_leave_pct"] == 10

    def test_returns_all_capacity_keys(self):
        """Result dict should contain all capacity deduction fields."""
        qs = QuestionnaireState(completed=True)
        result = _extract_capacity_deductions(qs)
        assert set(result.keys()) == {
            "capacity_bank_holiday_days",
            "capacity_planned_leave_days",
            "capacity_unplanned_leave_pct",
            "capacity_onboarding_engineer_sprints",
            "capacity_ktlo_engineers",
            "capacity_discovery_pct",
        }


# ── Project Analyzer helpers ────────────────────────────────────────


class TestQuestionMetadata:
    """Tests for QUESTION_METADATA structure and is_choice_question helper."""

    def test_metadata_has_fifteen_entries(self):
        """QUESTION_METADATA should define exactly 15 choice questions."""
        assert len(QUESTION_METADATA) == 15

    def test_all_entries_are_question_meta(self):
        """Every value in QUESTION_METADATA should be a QuestionMeta instance."""
        for q_num, meta in QUESTION_METADATA.items():
            assert isinstance(meta, QuestionMeta), f"Q{q_num} is not a QuestionMeta"

    def test_all_entries_are_valid_choice_type(self):
        """Every entry should have question_type 'single_choice' or 'multi_choice'."""
        for q_num, meta in QUESTION_METADATA.items():
            assert meta.question_type in ("single_choice", "multi_choice"), (
                f"Q{q_num} has invalid type '{meta.question_type}'"
            )

    def test_all_entries_have_options(self):
        """Every entry should have at least 2 options (except Q27/Q28 which are dynamic)."""
        for q_num, meta in QUESTION_METADATA.items():
            if q_num in (27, 28):
                continue  # Q27/Q28 have dynamic options — populated at runtime
            assert len(meta.options) >= 2, f"Q{q_num} has fewer than 2 options"

    def test_expected_question_numbers(self):
        """The 15 choice questions."""
        assert set(QUESTION_METADATA.keys()) == {2, 7, 8, 10, 13, 16, 18, 19, 24, 25, 26, 27, 28, 29, 30}

    def test_default_index_valid_or_none(self):
        """default_index must be None or a valid index into options."""
        for q_num, meta in QUESTION_METADATA.items():
            if meta.default_index is not None:
                assert 0 <= meta.default_index < len(meta.options), f"Q{q_num} default_index out of range"

    def test_q2_has_no_default(self):
        """Q2 (project type) is essential — no default."""
        assert QUESTION_METADATA[2].default_index is None

    def test_is_choice_question_true_for_known(self):
        """is_choice_question returns True for known choice questions."""
        for q_num in QUESTION_METADATA:
            assert is_choice_question(q_num) is True

    def test_is_choice_question_false_for_free_text(self):
        """is_choice_question returns False for free-text questions."""
        assert is_choice_question(1) is False
        assert is_choice_question(3) is False
        assert is_choice_question(11) is False
        assert is_choice_question(20) is False

    def test_defaults_consistency(self):
        """For single_choice Qs with a default_index, QUESTION_DEFAULTS should match the option text."""
        for q_num, meta in QUESTION_METADATA.items():
            if meta.question_type == "single_choice" and meta.default_index is not None:
                expected = meta.options[meta.default_index]
                assert QUESTION_DEFAULTS[q_num] == expected, (
                    f"Q{q_num}: QUESTION_DEFAULTS[{q_num}]='{QUESTION_DEFAULTS[q_num]}' "
                    f"!= options[{meta.default_index}]='{expected}'"
                )

    def test_question_meta_is_frozen(self):
        """QuestionMeta should be immutable (frozen dataclass)."""
        meta = QUESTION_METADATA[2]
        with pytest.raises(AttributeError):
            meta.question_type = "free_text"


# ── Phase intros tests ───────────────────────────────────────────────


class TestPhaseIntros:
    """Tests for PHASE_INTROS dictionary."""

    def test_all_phases_covered(self):
        """PHASE_INTROS should have an entry for every phase in PHASE_LABELS."""
        assert set(PHASE_INTROS.keys()) == set(PHASE_LABELS.keys())

    def test_all_intros_non_empty(self):
        """Every intro should be a non-empty string."""
        for phase, intro in PHASE_INTROS.items():
            assert isinstance(intro, str), f"Phase '{phase}' intro is not a string"
            assert len(intro.strip()) > 0, f"Phase '{phase}' intro is empty"

    def test_intros_are_different_from_labels(self):
        """Intros should not be the same as the phase labels."""
        for phase in PHASE_INTROS:
            assert PHASE_INTROS[phase] != PHASE_LABELS[phase]

    # NOTE: the former test_phase_intro_appears_in_first_question asserted the
    # retired "standard" mode's Q1-first phase-intro banner. Smart mode (the
    # default) asks an essential gap first, so that banner no longer appears on
    # the first call. Phase intros still render in the shared subsequent-call flow.


# ── Defaults command tests ───────────────────────────────────────────


class TestDefaultsCommand:
    """Tests for _is_defaults_intent and _batch_defaults_for_phase."""

    @pytest.mark.parametrize("text", ["defaults", "default", "use defaults", "DEFAULTS", "Default"])
    def test_is_defaults_intent_positive(self, text):
        """Known defaults keywords should be detected."""
        assert _is_defaults_intent(text) is True

    @pytest.mark.parametrize("text", ["skip", "help", "use all defaults", "default value", ""])
    def test_is_defaults_intent_negative(self, text):
        """Non-defaults input should not be detected."""
        assert _is_defaults_intent(text) is False

    def test_batch_defaults_applies_remaining(self):
        """_batch_defaults_for_phase should fill remaining Qs in the current phase."""
        qs = QuestionnaireState(current_question=8)
        qs.answers = {6: "3 engineers", 7: "2 backend, 1 frontend"}
        summary_lines, count = _batch_defaults_for_phase(qs)
        # Q8 (sprint length), Q9, Q10 should be defaulted
        assert count >= 2  # at least Q8 and Q9 and Q10
        assert 8 in qs.defaulted_questions
        assert 9 in qs.defaulted_questions
        assert 10 in qs.defaulted_questions

    def test_batch_defaults_skips_already_answered(self):
        """Already-answered questions should not be overwritten."""
        qs = QuestionnaireState(current_question=8)
        qs.answers = {6: "3 engineers", 7: "2 backend", 8: "1 week", 9: "20 pts", 10: "3 sprints"}
        _batch_defaults_for_phase(qs)
        # Existing answers should remain unchanged
        assert qs.answers[8] == "1 week"
        assert qs.answers[9] == "20 pts"

    def test_batch_defaults_skips_essential_with_no_default(self):
        """Essential questions with no default should be skipped, not defaulted."""
        qs = QuestionnaireState(current_question=1)
        _batch_defaults_for_phase(qs)
        # Q1 is essential, Q2 is essential (no default_index), Q3-Q4 essential
        # Q5 has a default in QUESTION_DEFAULTS
        assert 1 not in qs.answers
        assert 5 in qs.answers

    def test_defaults_command_in_intake_node(self, monkeypatch):
        """Typing 'defaults' during intake should apply defaults and advance past the phase."""
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)

        qs = QuestionnaireState(current_question=8)
        qs.answers = {i: f"answer {i}" for i in range(1, 8)}
        state = {
            "messages": [HumanMessage(content="defaults")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        # Should advance past phase 2 (Q6-Q10) to Q11 (phase 3)
        assert result["questionnaire"].current_question == 11
        # Q8, Q9, Q10 should all be defaulted
        assert 8 in result["questionnaire"].defaulted_questions
        assert "default" in result["messages"][0].content.lower()

    def test_defaults_on_last_phase_shows_summary(self, monkeypatch):
        """Typing 'defaults' on the last phase should show the intake summary."""
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)

        qs = QuestionnaireState(current_question=27)
        qs.answers = {i: f"answer {i}" for i in range(1, 27)}
        qs._leave_input_stage = "done"  # Skip PTO sub-loop — not under test here
        state = {
            "messages": [HumanMessage(content="defaults")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        assert result["questionnaire"].awaiting_confirmation is True
        assert "Project Intake Summary" in result["messages"][0].content


# ── Vagueness skip for choice questions ──────────────────────────────


class TestVaguenessSkipForChoiceQuestions:
    """Tests that choice questions skip the vagueness check."""

    def test_choice_question_skips_vagueness(self, monkeypatch):
        """A choice question answer should NOT trigger _check_vague_answer."""
        vague_called = False

        def fake_vague(q, a, n=0):
            nonlocal vague_called
            vague_called = True
            return "This is vague!"

        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", fake_vague)

        qs = QuestionnaireState(current_question=2)
        qs.answers = {1: "A todo app"}
        state = {
            "messages": [HumanMessage(content="Greenfield")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        assert vague_called is False
        # Should advance normally
        assert result["questionnaire"].current_question == 3

    def test_free_text_still_probes(self, monkeypatch):
        """A free-text question should still trigger _check_vague_answer."""
        vague_called = False

        def fake_vague(q, a, n=0):
            nonlocal vague_called
            vague_called = True
            return None  # Accept the answer

        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", fake_vague)

        qs = QuestionnaireState(current_question=3)
        qs.answers = {1: "A todo app", 2: "Greenfield"}
        state = {
            "messages": [HumanMessage(content="It solves task management")],
            "questionnaire": qs,
        }
        project_intake(state)
        assert vague_called is True


# ── Smart / Quick intake mode tests ─────────────────────────────────


class TestAutoApplyExtractions:
    """Tests for _auto_apply_extractions helper."""

    def test_moves_extracted_to_answers(self):
        qs = QuestionnaireState()
        extracted = {1: "A todo app", 6: "3 engineers"}
        _auto_apply_extractions(qs, extracted)
        assert qs.answers[1] == "A todo app"
        assert qs.answers[6] == "3 engineers"
        assert qs.extracted_questions == {1, 6}

    def test_empty_extraction_no_change(self):
        qs = QuestionnaireState()
        _auto_apply_extractions(qs, {})
        assert qs.answers == {}
        assert qs.extracted_questions == set()


class TestDeriveQ15FromQ2:
    """Tests for _derive_q15_from_q2 helper."""

    @pytest.mark.parametrize(
        ("q2_answer", "expected_q15"),
        [
            ("Greenfield", "New build (greenfield)"),
            ("Existing codebase", "Yes, existing codebase"),
            ("Hybrid", "Partial — extending existing codebase with new components"),
        ],
    )
    def test_derives_from_q2(self, q2_answer, expected_q15):
        qs = QuestionnaireState()
        qs.answers[2] = q2_answer
        _derive_q15_from_q2(qs)
        assert qs.answers[15] == expected_q15
        assert 15 in qs.defaulted_questions

    def test_no_derivation_when_q15_already_answered(self):
        qs = QuestionnaireState()
        qs.answers[2] = "Greenfield"
        qs.answers[15] = "Custom answer"
        _derive_q15_from_q2(qs)
        assert qs.answers[15] == "Custom answer"

    def test_no_derivation_without_q2(self):
        qs = QuestionnaireState()
        _derive_q15_from_q2(qs)
        assert 15 not in qs.answers


class TestNeedsRepoUrlPrompt:
    """Tests for _needs_repo_url_prompt helper."""

    def test_existing_codebase_without_q17(self):
        qs = QuestionnaireState()
        qs.answers[2] = "Existing codebase"
        assert _needs_repo_url_prompt(qs) is True

    def test_hybrid_without_q17(self):
        qs = QuestionnaireState()
        qs.answers[2] = "Hybrid"
        assert _needs_repo_url_prompt(qs) is True

    def test_greenfield_returns_false(self):
        qs = QuestionnaireState()
        qs.answers[2] = "Greenfield"
        assert _needs_repo_url_prompt(qs) is False

    def test_existing_codebase_with_q17_already_answered(self):
        qs = QuestionnaireState()
        qs.answers[2] = "Existing codebase"
        qs.answers[17] = "https://github.com/org/repo"
        assert _needs_repo_url_prompt(qs) is False

    def test_no_q2_answer_returns_false(self):
        qs = QuestionnaireState()
        assert _needs_repo_url_prompt(qs) is False

    def test_existing_codebase_with_q17_defaulted_returns_true(self):
        # Regression: _auto_default_remaining pre-fills Q17 with "No repo URL provided"
        # and adds it to defaulted_questions. We must still ask the user for a real URL.
        qs = QuestionnaireState()
        qs.answers[2] = "Existing codebase"
        qs.answers[17] = "No repo URL provided"
        qs.defaulted_questions.add(17)
        assert _needs_repo_url_prompt(qs) is True

    def test_hybrid_with_q17_defaulted_returns_true(self):
        qs = QuestionnaireState()
        qs.answers[2] = "Hybrid"
        qs.answers[17] = "No repo URL provided"
        qs.defaulted_questions.add(17)
        assert _needs_repo_url_prompt(qs) is True


class TestQ2RepoUrlFollowUp:
    """Tests for the Q2 → repo URL follow-up in project_intake."""

    @pytest.fixture(autouse=True)
    def _disable_vague_check(self, monkeypatch):
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)

    def _make_state(self, current_q, answers):
        qs = QuestionnaireState(current_question=current_q)
        qs.answers.update(answers)
        return {
            "messages": [HumanMessage(content=answers.get(current_q, ""))],
            "questionnaire": qs,
        }

    def test_existing_codebase_triggers_repo_prompt_standard(self):
        """Standard mode: Q2 'Existing codebase' → ask for repo URL before advancing."""
        qs = QuestionnaireState(current_question=2)
        state = {
            "messages": [HumanMessage(content="Existing codebase")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        assert _Q2_REPO_URL_PROMPT in result["messages"][0].content
        # current_question stays at 2 (held via probed_questions)
        assert result["questionnaire"].current_question == 2
        assert 2 in result["questionnaire"].probed_questions

    def test_hybrid_triggers_repo_prompt_standard(self):
        """Standard mode: Q2 'Hybrid' → ask for repo URL before advancing."""
        qs = QuestionnaireState(current_question=2)
        state = {
            "messages": [HumanMessage(content="Hybrid")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        assert _Q2_REPO_URL_PROMPT in result["messages"][0].content

    def test_greenfield_does_not_trigger_repo_prompt(self):
        """Standard mode: Q2 'Greenfield' → advance normally to Q3."""
        qs = QuestionnaireState(current_question=2)
        state = {
            "messages": [HumanMessage(content="Greenfield")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        assert _Q2_REPO_URL_PROMPT not in result["messages"][0].content
        assert result["questionnaire"].current_question == 3

    def test_repo_url_stored_in_q17_standard(self):
        """Standard mode: follow-up answer is stored in Q17, not combined into Q2."""
        qs = QuestionnaireState(current_question=2)
        qs.answers[2] = "Existing codebase"
        qs.probed_questions.add(2)  # simulates Q2 already probed for repo URL
        state = {
            "messages": [HumanMessage(content="https://github.com/org/repo")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        assert result["questionnaire"].answers[17] == "https://github.com/org/repo"
        # Q2 answer unchanged
        assert result["questionnaire"].answers[2] == "Existing codebase"

    def test_repo_url_not_asked_if_q17_already_answered(self):
        """Standard mode: Q17 already in answers → skip repo URL prompt."""
        qs = QuestionnaireState(current_question=2)
        qs.answers[17] = "https://github.com/org/repo"
        state = {
            "messages": [HumanMessage(content="Existing codebase")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        assert _Q2_REPO_URL_PROMPT not in result["messages"][0].content
        assert result["questionnaire"].current_question == 3

    def test_smart_mode_q2_existing_triggers_repo_prompt(self, monkeypatch):
        """Smart mode: Q2 gap answered 'Existing codebase' → ask for repo URL."""
        monkeypatch.setattr("yeaboi.agent.nodes._extract_answers_from_description", lambda _: {})
        qs = QuestionnaireState(current_question=2, intake_mode="smart")
        qs.answers.update({3: "goals", 4: "done", 6: "4", 11: "React"})
        state = {
            "messages": [HumanMessage(content="Existing codebase")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        assert _Q2_REPO_URL_PROMPT in result["messages"][0].content
        assert 2 in result["questionnaire"].probed_questions

    def test_smart_mode_repo_url_stored_in_q17(self, monkeypatch):
        """Smart mode: repo URL follow-up stores answer in Q17."""
        monkeypatch.setattr("yeaboi.agent.nodes._extract_answers_from_description", lambda _: {})
        qs = QuestionnaireState(current_question=2, intake_mode="smart")
        qs.answers.update({3: "goals", 4: "done", 6: "4", 11: "React"})
        qs.answers[2] = "Existing codebase"
        qs.probed_questions.add(2)
        state = {
            "messages": [HumanMessage(content="https://github.com/org/repo")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        assert result["questionnaire"].answers[17] == "https://github.com/org/repo"
        assert result["questionnaire"].answers[2] == "Existing codebase"

    def test_smart_mode_first_invocation_asks_q2_with_suggestion_when_extracted(self, monkeypatch):
        """Smart mode first call: extracted Q2 'Existing codebase' goes to suggestions, not answers.

        Q2 is an essential question, so the extraction becomes a suggestion shown
        inline. Q17 (repo URL) is asked later when the user confirms Q2.
        """
        monkeypatch.setattr(
            "yeaboi.agent.nodes._extract_answers_from_description",
            lambda _: {2: "Existing codebase", 3: "goals", 4: "done", 6: "4", 11: "React"},
        )
        state = {
            "messages": [HumanMessage(content="We're working on an existing Django app.")],
            "_intake_mode": "smart",
        }
        result = project_intake(state)
        qs = result["questionnaire"]
        # Q2 is essential → goes to suggestions (user must confirm)
        assert 2 in qs.suggested_answers
        assert qs.suggested_answers[2] == "Existing codebase"
        # First gap question should be Q2 (the first essential gap)
        assert qs.current_question == 2


class TestTrackerChoiceHonoursIntegrations:
    """The intake offers only the trackers the plan's integrations allow."""

    def _run(self, monkeypatch, **extras):
        monkeypatch.setattr("yeaboi.trackers._jira_configured", lambda: True)
        monkeypatch.setattr("yeaboi.trackers._azdevops_configured", lambda: True)
        monkeypatch.setattr("yeaboi.agent.nodes._extract_answers_from_description", lambda _: {})
        monkeypatch.setattr("yeaboi.agent.nodes._fetch_tracker_velocity", lambda _pref: None)
        monkeypatch.setattr("yeaboi.agent.nodes._fetch_active_sprint_number", lambda _pref: (None, None, "no sprint"))
        state = {
            "messages": [HumanMessage(content="A pond with a pump and a bench.")],
            "_intake_mode": "smart",
            **extras,
        }
        return project_intake(state)

    def test_two_enabled_trackers_are_offered(self, monkeypatch):
        result = self._run(monkeypatch)
        assert result["questionnaire"]._awaiting_tracker_choice is True
        assert "Which tracker" in result["messages"][0].content

    def test_one_enabled_tracker_is_the_preference_without_asking(self, monkeypatch):
        result = self._run(monkeypatch, session_integrations=["azdevops", "github"])
        qs = result["questionnaire"]
        assert qs._awaiting_tracker_choice is False
        assert qs._preferred_tracker == "azdevops"
        assert "Which tracker" not in result["messages"][0].content

    def test_no_enabled_tracker_skips_the_offer(self, monkeypatch):
        result = self._run(monkeypatch, session_integrations=["github"])
        qs = result["questionnaire"]
        assert qs._awaiting_tracker_choice is False
        assert qs._preferred_tracker == ""
        assert "Which tracker" not in result["messages"][0].content


class TestFindEssentialGaps:
    """Tests for _find_essential_gaps helper."""

    def test_returns_unanswered_essentials(self):
        qs = QuestionnaireState()
        qs.answers = {2: "Greenfield", 6: "3"}
        gaps = _find_essential_gaps(qs, SMART_ESSENTIALS)
        # SMART_ESSENTIALS = {2, 3, 4, 6, 10, 11, 27}; answered: 2, 6
        # Conditional: Q6 answered → Q7 promoted; Q2 answered → Q13 promoted
        # Q29 no longer conditional — defaults to 10%, editable at confirmation
        # → gaps: 3, 4, 7, 10, 11, 13, 27
        assert gaps == [3, 4, 7, 10, 11, 13, 27]

    def test_no_gaps_when_all_answered(self):
        qs = QuestionnaireState()
        qs.answers = {
            2: "Greenfield",
            3: "Problem",
            4: "Done state",
            6: "3",
            7: "Backend, Frontend",
            8: "2 weeks",
            10: "5 sprints",
            11: "React",
            12: "No integrations",
            13: "AWS",
            27: "Fresh start (today)",
            29: "10%",
        }
        gaps = _find_essential_gaps(qs, SMART_ESSENTIALS)
        assert gaps == []


class TestBuildGapPrompt:
    """Tests for _build_gap_prompt helper."""

    def test_asks_first_gap_individually(self):
        """Each gap is asked one at a time — Q3 and Q4 are never merged."""
        qs = QuestionnaireState()
        prompt, q_nums = _build_gap_prompt([3, 4, 11], qs)
        assert prompt == INTAKE_QUESTIONS[3]
        assert q_nums == [3]

    def test_single_question(self):
        qs = QuestionnaireState()
        prompt, q_nums = _build_gap_prompt([6, 11], qs)
        assert prompt == INTAKE_QUESTIONS[6]
        assert q_nums == [6]

    def test_empty_gaps_returns_empty(self):
        qs = QuestionnaireState()
        prompt, q_nums = _build_gap_prompt([], qs)
        assert prompt == ""
        assert q_nums == []


class TestBuildGapPromptCapacity:
    """Tests for capacity questions in _build_gap_prompt (Q28/Q30 deferred to capacity_check)."""

    def test_q29_asked_individually_if_gap(self):
        """Q29 is the only capacity question that stays in intake — asked individually."""
        qs = QuestionnaireState()
        prompt, q_nums = _build_gap_prompt([29], qs)
        assert q_nums == [29]
        assert "unplanned" in prompt.lower()

    def test_non_capacity_gap_asked_individually(self):
        """Non-capacity questions are asked one at a time."""
        qs = QuestionnaireState()
        prompt, q_nums = _build_gap_prompt([3, 29], qs)
        assert q_nums == [3]  # Q3 asked first, individually


class TestDeriveQ27FromLocale:
    """Tests for _derive_q27_from_locale — auto-detecting bank holidays."""

    def test_populates_q27_from_locale(self, monkeypatch):
        """Q27 should be auto-populated with 'Fresh start (today)' when no Jira."""
        qs = QuestionnaireState(completed=True)
        qs.answers = {i: f"answer {i}" for i in range(1, TOTAL_QUESTIONS + 1)}
        qs.defaulted_questions.add(27)
        _derive_q27_from_locale(qs)
        assert 27 in qs.extracted_questions
        assert qs.answers[27] == "Fresh start (today)"
        assert 27 not in qs.defaulted_questions

    def test_skips_when_already_answered(self):
        """If Q27 is already explicitly answered, don't override."""
        qs = QuestionnaireState()
        qs.answers[27] = "Sprint 105"
        _derive_q27_from_locale(qs)
        assert qs.answers[27] == "Sprint 105"
        assert 27 not in qs.extracted_questions

    def test_no_locale_still_populates_with_fresh_start(self, monkeypatch):
        """Q27 is populated with 'Fresh start (today)' regardless of locale."""
        qs = QuestionnaireState()
        qs.answers[27] = "None"
        qs.defaulted_questions.add(27)
        _derive_q27_from_locale(qs)
        assert 27 in qs.extracted_questions
        assert qs.answers[27] == "Fresh start (today)"


class TestSmartIntakeMode:
    """Tests for smart intake mode in project_intake node."""

    @pytest.fixture(autouse=True)
    def _disable_vague_check(self, monkeypatch):
        """Disable vague-answer checking so smart mode tests don't hit the LLM."""
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)

    @pytest.fixture(autouse=True)
    def _no_scrum_md(self, monkeypatch):
        """Prevent SCRUM.md loading from CWD so tests are hermetic."""
        monkeypatch.setattr("yeaboi.agent.nodes._load_user_context", lambda *a, **kw: (None, {}))

    def test_auto_applies_extractions_and_asks_gaps(self, monkeypatch):
        """Smart mode: non-essential extractions go to answers, essential ones to suggestions."""
        monkeypatch.setattr(
            "yeaboi.agent.nodes._extract_answers_from_description",
            lambda desc: {1: "A todo app", 3: "Task management", 4: "Users can track tasks"},
        )
        state = {
            "messages": [HumanMessage(content="Building a todo app for task management")],
            "_intake_mode": "smart",
        }
        result = project_intake(state)
        qs = result["questionnaire"]
        assert qs.intake_mode == "smart"
        # Q1 auto-applied (not in SMART_ESSENTIALS — handled separately)
        assert 1 in qs.extracted_questions
        # Q3 and Q4 are essential → routed to suggested_answers, not answers
        assert 3 in qs.suggested_answers
        assert 4 in qs.suggested_answers
        # Should ask for remaining gaps (Q2, Q3, Q4, Q6, Q11 — Q3+Q4 merged)
        ai_msg = result["messages"][0].content
        assert "remaining" in ai_msg.lower() or "more question" in ai_msg.lower()

    def test_no_gaps_jumps_to_summary(self, monkeypatch):
        """Smart mode: essential extractions become suggestions, so gaps remain for user confirmation."""
        # Even when all SMART_ESSENTIALS are extracted, they go to suggestions
        # (not answers), so gaps remain and the user is asked to confirm them.
        monkeypatch.setattr(
            "yeaboi.agent.nodes._extract_answers_from_description",
            lambda desc: {
                1: "A todo app",
                2: "Greenfield",
                3: "Task management",
                4: "Users can track tasks",
                6: "3 engineers",
                11: "React, Node, PostgreSQL",
            },
        )
        state = {
            "messages": [HumanMessage(content="Building a todo app")],
            "_intake_mode": "smart",
        }
        result = project_intake(state)
        qs = result["questionnaire"]
        # Should ask gaps (essential extractions are suggestions, not answers)
        assert qs.awaiting_confirmation is False
        ai_msg = result["messages"][0].content
        assert "remaining" in ai_msg.lower() or "more question" in ai_msg.lower()
        # All essential extractions should be in suggested_answers
        assert 2 in qs.suggested_answers
        assert 6 in qs.suggested_answers
        assert 11 in qs.suggested_answers

    def test_auto_defaults_non_essential(self, monkeypatch):
        """Smart mode auto-defaults optional questions."""
        monkeypatch.setattr(
            "yeaboi.agent.nodes._extract_answers_from_description",
            lambda desc: {1: "A todo app"},
        )
        state = {
            "messages": [HumanMessage(content="Building a todo app")],
            "_intake_mode": "smart",
        }
        result = project_intake(state)
        qs = result["questionnaire"]
        # Q5 (deadlines) should be auto-defaulted
        assert 5 in qs.answers
        assert 5 in qs.defaulted_questions

    def test_q2_derives_q15(self, monkeypatch):
        """Smart mode: Q2 extraction goes to suggestions; Q15 derived when user confirms Q2."""
        monkeypatch.setattr(
            "yeaboi.agent.nodes._extract_answers_from_description",
            lambda desc: {1: "A todo app", 2: "Greenfield"},
        )
        state = {
            "messages": [HumanMessage(content="Building a greenfield todo app")],
            "_intake_mode": "smart",
        }
        result = project_intake(state)
        qs = result["questionnaire"]
        # Q2 is essential → goes to suggestions, Q15 not derived yet
        assert 2 in qs.suggested_answers
        assert qs.suggested_answers[2] == "Greenfield"
        # Q15 is NOT derived yet because Q2 hasn't been confirmed by the user.
        # It will be derived later when the user answers Q2.
        assert 15 not in qs.answers or 15 in qs.defaulted_questions

    def test_q3_q4_asked_separately(self, monkeypatch):
        """Smart mode: Q3 and Q4 are asked as separate questions, not merged."""
        qs = QuestionnaireState(intake_mode="smart", current_question=2)
        qs.answers = {1: "A todo app"}
        qs.extracted_questions = {1}
        # Fill defaults for non-essential questions
        for q_num in range(1, TOTAL_QUESTIONS + 1):
            if q_num not in qs.answers and q_num in QUESTION_DEFAULTS:
                qs.answers[q_num] = QUESTION_DEFAULTS[q_num]
                qs.defaulted_questions.add(q_num)
        # Answer Q2 — user confirms "Greenfield"
        state = {
            "messages": [HumanMessage(content="Greenfield")],
            "questionnaire": qs,
            "_intake_mode": "smart",
        }
        result = project_intake(state)
        qs2 = result["questionnaire"]
        # Q2 answered → next gap is Q3 (asked individually, not merged with Q4)
        ai_msg = result["messages"][0].content
        assert "problem" in ai_msg
        assert qs2._pending_merged_questions == [3]

    def test_gap_fill_records_answer(self, monkeypatch):
        """Smart mode: user's answer to a gap question is recorded correctly."""
        qs = QuestionnaireState(intake_mode="smart", current_question=6)
        qs.answers = {1: "A todo app", 2: "Greenfield", 3: "Problem", 4: "Done"}
        qs.extracted_questions = {1}
        # Fill defaults for non-essential questions
        for q_num in range(1, TOTAL_QUESTIONS + 1):
            if q_num not in qs.answers and q_num in QUESTION_DEFAULTS:
                qs.answers[q_num] = QUESTION_DEFAULTS[q_num]
                qs.defaulted_questions.add(q_num)
        qs.answers[15] = "New build (greenfield)"
        qs.defaulted_questions.add(15)

        state = {
            "messages": [HumanMessage(content="5 engineers")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        qs_out = result["questionnaire"]
        assert qs_out.answers[6] == "5 engineers"

    def test_gap_fill_q3_then_q4_separate(self, monkeypatch):
        """Smart mode: answering Q3 advances to Q4 as the next gap."""
        qs = QuestionnaireState(intake_mode="smart", current_question=3)
        qs.answers = {1: "A todo app", 2: "Greenfield"}
        qs.extracted_questions = {1, 2}
        qs._pending_merged_questions = [3]
        # Fill defaults
        for q_num in range(1, TOTAL_QUESTIONS + 1):
            if q_num not in qs.answers and q_num in QUESTION_DEFAULTS:
                qs.answers[q_num] = QUESTION_DEFAULTS[q_num]
                qs.defaulted_questions.add(q_num)
        qs.answers[15] = "New build (greenfield)"
        qs.defaulted_questions.add(15)

        state = {
            "messages": [HumanMessage(content="Solve task management for small teams")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        qs_out = result["questionnaire"]
        assert 3 in qs_out.answers
        # Q4 should be the next gap asked
        assert qs_out.current_question == 4

    def test_standard_mode_coerced_to_smart(self, monkeypatch):
        """The retired "standard" mode is coerced to smart at first invocation.

        The legacy 30-question one-at-a-time flow has been removed; any lingering
        "standard" value (old sessions, stale state) now follows the smart path,
        which asks only essential gaps rather than starting at Q1.
        """
        monkeypatch.setattr(
            "yeaboi.agent.nodes._extract_answers_from_description",
            lambda desc: {},
        )
        state = {
            "messages": [HumanMessage(content="Building a todo app")],
            "_intake_mode": "standard",
        }
        result = project_intake(state)
        qs = result["questionnaire"]
        assert qs.intake_mode == "smart"
        # Smart flow does not walk Q1-first; it jumps to essential gaps.
        assert qs.current_question != 1

    def test_vague_answer_triggers_follow_up(self, monkeypatch):
        """Smart mode: a vague answer should trigger a follow-up probe."""
        # Override the autouse fixture for this test — we need vague detection active.
        monkeypatch.setattr(
            "yeaboi.agent.nodes._check_vague_answer",
            lambda q, a, n=0: ("Can you be more specific?", ("Option A", "Option B")),
        )
        qs = QuestionnaireState(intake_mode="smart", current_question=6)
        qs.answers = {1: "A todo app", 2: "Greenfield", 3: "Problem", 4: "Done"}
        qs.extracted_questions = {1}
        for q_num in range(1, TOTAL_QUESTIONS + 1):
            if q_num not in qs.answers and q_num in QUESTION_DEFAULTS:
                qs.answers[q_num] = QUESTION_DEFAULTS[q_num]
                qs.defaulted_questions.add(q_num)
        qs.answers[15] = "New build (greenfield)"
        qs.defaulted_questions.add(15)

        state = {
            "messages": [HumanMessage(content="some")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        qs_out = result["questionnaire"]
        # Should probe Q6, not advance to next gap
        assert 6 in qs_out.probed_questions
        assert qs_out._follow_up_choices.get(6) == ("Option A", "Option B")
        assert "Follow-up" in result["messages"][0].content

    def test_follow_up_response_combines_answers(self, monkeypatch):
        """Smart mode: follow-up answer is combined with original, then advances."""
        qs = QuestionnaireState(intake_mode="smart", current_question=6)
        qs.answers = {1: "A todo app", 2: "Greenfield", 3: "Problem", 4: "Done", 6: "some"}
        qs.extracted_questions = {1}
        qs.probed_questions.add(6)
        qs._follow_up_choices[6] = ("Option A", "Option B")
        for q_num in range(1, TOTAL_QUESTIONS + 1):
            if q_num not in qs.answers and q_num in QUESTION_DEFAULTS:
                qs.answers[q_num] = QUESTION_DEFAULTS[q_num]
                qs.defaulted_questions.add(q_num)
        qs.answers[15] = "New build (greenfield)"
        qs.defaulted_questions.add(15)

        state = {
            "messages": [HumanMessage(content="5 engineers")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        qs_out = result["questionnaire"]
        # Original + follow-up combined
        assert "some" in qs_out.answers[6]
        assert "5 engineers" in qs_out.answers[6]
        # Dynamic choices cleared after consumption
        assert 6 not in qs_out._follow_up_choices

    def test_vague_check_skipped_for_choice_questions(self, monkeypatch):
        """Smart mode: choice question answers should NOT trigger vague checking."""
        tracking = {"called": False}

        def tracking_check(q, a, n=0):
            tracking["called"] = True
            return ("follow-up?", ("A", "B"))

        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", tracking_check)
        # Q2 is a choice question — vague check should be skipped
        qs = QuestionnaireState(intake_mode="smart", current_question=2)
        qs.answers = {1: "A todo app"}
        qs.extracted_questions = {1}
        for q_num in range(1, TOTAL_QUESTIONS + 1):
            if q_num not in qs.answers and q_num in QUESTION_DEFAULTS:
                qs.answers[q_num] = QUESTION_DEFAULTS[q_num]
                qs.defaulted_questions.add(q_num)
        qs.answers[15] = "New build (greenfield)"
        qs.defaulted_questions.add(15)

        state = {
            "messages": [HumanMessage(content="Greenfield")],
            "questionnaire": qs,
        }
        project_intake(state)
        assert not tracking["called"], "_check_vague_answer should not be called for choice questions"

    def test_vague_check_runs_for_q3(self, monkeypatch):
        """Smart mode: a vague Q3 answer should trigger follow-up probing."""
        tracking = {"called": False, "question_text": None}

        def tracking_check(q, a, n=0):
            tracking["called"] = True
            tracking["question_text"] = q
            return ("Can you be more specific?", ("Option A", "Option B"))

        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", tracking_check)
        qs = QuestionnaireState(intake_mode="smart", current_question=3)
        qs.answers = {1: "A todo app", 2: "Greenfield"}
        qs.extracted_questions = {1, 2}
        qs._pending_merged_questions = [3]
        for q_num in range(1, TOTAL_QUESTIONS + 1):
            if q_num not in qs.answers and q_num in QUESTION_DEFAULTS:
                qs.answers[q_num] = QUESTION_DEFAULTS[q_num]
                qs.defaulted_questions.add(q_num)
        qs.answers[15] = "New build (greenfield)"
        qs.defaulted_questions.add(15)

        state = {
            "messages": [HumanMessage(content="no problem")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        assert tracking["called"], "_check_vague_answer should be called"
        # Should check against Q3's text
        assert tracking["question_text"] == INTAKE_QUESTIONS[3]
        # Should return a follow-up probe
        ai_content = result["messages"][-1].content
        assert "Follow-up" in ai_content

    def test_q3_non_vague_advances_to_q4(self, monkeypatch):
        """Smart mode: a detailed Q3 answer advances to Q4 as the next gap."""
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)
        qs = QuestionnaireState(intake_mode="smart", current_question=3)
        qs.answers = {1: "A todo app", 2: "Greenfield", 6: "3 engineers", 11: "Python, React"}
        qs.extracted_questions = {1, 2}
        qs._pending_merged_questions = [3]
        for q_num in range(1, TOTAL_QUESTIONS + 1):
            if q_num not in qs.answers and q_num in QUESTION_DEFAULTS:
                qs.answers[q_num] = QUESTION_DEFAULTS[q_num]
                qs.defaulted_questions.add(q_num)
        qs.answers[15] = "New build (greenfield)"
        qs.defaulted_questions.add(15)

        state = {
            "messages": [HumanMessage(content="Solve task management for teams, users are devs")],
            "questionnaire": qs,
        }
        result = project_intake(state)
        # Q3 answered, Q4 is the next gap
        assert result["questionnaire"].current_question == 4


class TestQuickIntakeMode:
    """Tests for quick intake mode in project_intake node."""

    @pytest.fixture(autouse=True)
    def _no_scrum_md(self, monkeypatch):
        """Prevent SCRUM.md loading from CWD so tests are hermetic."""
        monkeypatch.setattr("yeaboi.agent.nodes._load_user_context", lambda *a, **kw: (None, {}))

    def test_asks_only_q6_and_q11(self, monkeypatch):
        """Quick mode: only Q6 and Q11 are asked when not extracted."""
        monkeypatch.setattr(
            "yeaboi.agent.nodes._extract_answers_from_description",
            lambda desc: {1: "A todo app"},
        )
        state = {
            "messages": [HumanMessage(content="Building a todo app")],
            "_intake_mode": "quick",
        }
        result = project_intake(state)
        qs = result["questionnaire"]
        assert qs.intake_mode == "quick"
        # Q2, Q3, Q4 should be fallback-defaulted
        assert 2 in qs.answers
        assert 3 in qs.answers
        assert 4 in qs.answers
        # Should be asking Q6 (first quick essential)
        assert qs.current_question == 6

    def test_all_extracted_asks_essentials_with_suggestions(self, monkeypatch):
        """Quick mode: extracted essentials (Q6, Q11) become suggestions, not auto-accepted."""
        monkeypatch.setattr(
            "yeaboi.agent.nodes._extract_answers_from_description",
            lambda desc: {1: "A todo app", 6: "3 engineers", 11: "React"},
        )
        state = {
            "messages": [HumanMessage(content="Building a todo app with 3 engineers using React")],
            "_intake_mode": "quick",
        }
        result = project_intake(state)
        qs = result["questionnaire"]
        # Q6 and Q11 are essential → go to suggestions, user must confirm
        assert 6 in qs.suggested_answers
        assert 11 in qs.suggested_answers
        assert qs.awaiting_confirmation is False
        # Should be asking first essential gap (Q6)
        assert qs.current_question == 6

    def test_fallback_defaults_applied(self, monkeypatch):
        """Quick mode: QUICK_FALLBACK_DEFAULTS fill Q2, Q3, Q4, Q15."""
        monkeypatch.setattr(
            "yeaboi.agent.nodes._extract_answers_from_description",
            lambda desc: {1: "A todo app"},
        )
        state = {
            "messages": [HumanMessage(content="Building a todo app")],
            "_intake_mode": "quick",
        }
        result = project_intake(state)
        qs = result["questionnaire"]
        for q_num in QUICK_FALLBACK_DEFAULTS:
            if q_num not in qs.extracted_questions:
                assert q_num in qs.answers, f"Q{q_num} should be filled by fallback default"


class TestIntakeSummaryProvenance:
    """Tests for provenance markers in _build_intake_summary."""

    def test_extracted_marker_shown(self):
        """Extracted answers should show *(from your description)* in summary."""
        qs = QuestionnaireState()
        qs.answers = {i: f"Answer {i}" for i in range(1, TOTAL_QUESTIONS + 1)}
        qs.extracted_questions = {1, 3}
        summary = _build_intake_summary(qs)
        # Q1 and Q3 should have the extracted marker
        assert "*(from your description)*" in summary

    def test_stats_header_shown(self):
        """Summary should include a stats header with answer provenance counts."""
        qs = QuestionnaireState()
        qs.answers = {i: f"Answer {i}" for i in range(1, TOTAL_QUESTIONS + 1)}
        qs.extracted_questions = {1, 3}
        qs.defaulted_questions = {5, 7, 8}
        summary = _build_intake_summary(qs)
        assert "2 extracted" in summary
        assert "3 defaulted" in summary

    def test_answers_block_extracted_marker(self):
        """_build_answers_block should show extracted marker for extracted questions."""
        qs = QuestionnaireState()
        qs.answers = {i: f"Answer {i}" for i in range(1, TOTAL_QUESTIONS + 1)}
        qs.extracted_questions = {1}
        block = _build_answers_block(qs)
        assert "*(extracted from description)*" in block


# ---------------------------------------------------------------------------
# SCRUM.md auto-population during intake
# ---------------------------------------------------------------------------


class TestScrumMdAutoPopulation:
    """Tests for SCRUM.md auto-population of intake questions."""

    @pytest.fixture(autouse=True)
    def _disable_vague_check(self, monkeypatch):
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)

    def test_scrum_md_extractions_fill_gaps(self, monkeypatch):
        """SCRUM.md extractions fill questions not covered by the user's description."""
        # Description provides Q1; SCRUM.md provides Q8 and Q11
        call_count = {"n": 0}
        description_extractions = {1: "An AI agent"}
        scrum_md_extractions = {8: "2 weeks", 11: "Python, Langchain", 24: "Fibonacci"}

        def fake_extract(desc):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return description_extractions  # first call: from description
            return scrum_md_extractions  # second call: from SCRUM.md

        monkeypatch.setattr("yeaboi.agent.nodes._extract_answers_from_description", fake_extract)
        monkeypatch.setattr(
            "yeaboi.agent.nodes._load_user_context",
            lambda *a, **kw: ("## Tech\nPython & Langchain", {"name": "User context", "status": "success"}),
        )
        state = {
            "messages": [HumanMessage(content="An AI agent")],
            "_intake_mode": "smart",
        }
        result = project_intake(state)
        qs = result["questionnaire"]
        # Q8 (sprint length) and Q24 (story points) are non-essential → auto-applied
        assert qs.answers.get(8) == "2 weeks"
        assert qs.answers.get(24) == "Fibonacci"
        # Q11 (tech stack) is essential → goes to suggested_answers
        assert qs.suggested_answers.get(11) == "Python, Langchain"
        # Track SCRUM.md provenance
        assert 8 in qs._scrum_md_questions
        assert 24 in qs._scrum_md_questions

    def test_description_wins_over_scrum_md(self, monkeypatch):
        """User's description takes priority over SCRUM.md for the same question."""
        call_count = {"n": 0}

        def fake_extract(desc):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {1: "An AI agent", 11: "JavaScript, React"}
            return {11: "Python, Langchain"}  # SCRUM.md has different tech stack

        monkeypatch.setattr("yeaboi.agent.nodes._extract_answers_from_description", fake_extract)
        monkeypatch.setattr(
            "yeaboi.agent.nodes._load_user_context",
            lambda *a, **kw: ("## Tech\nPython", {"name": "User context", "status": "success"}),
        )
        state = {
            "messages": [HumanMessage(content="An AI agent with JavaScript")],
            "_intake_mode": "smart",
        }
        result = project_intake(state)
        qs = result["questionnaire"]
        # Q11 from description (JavaScript) should win
        assert qs.suggested_answers.get(11) == "JavaScript, React"
        # Q11 should NOT be in scrum_md_questions since description won
        assert 11 not in qs._scrum_md_questions

    def test_no_scrum_md_no_extra_extraction(self, monkeypatch):
        """When SCRUM.md is absent, only one extraction call is made."""
        call_count = {"n": 0}

        def fake_extract(desc):
            call_count["n"] += 1
            return {1: "An AI agent"}

        monkeypatch.setattr("yeaboi.agent.nodes._extract_answers_from_description", fake_extract)
        monkeypatch.setattr("yeaboi.agent.nodes._load_user_context", lambda *a, **kw: (None, {}))
        state = {
            "messages": [HumanMessage(content="An AI agent")],
            "_intake_mode": "smart",
        }
        project_intake(state)
        assert call_count["n"] == 1  # only called once (no SCRUM.md)

    def test_preamble_mentions_scrum_md(self, monkeypatch):
        """Preamble should distinguish SCRUM.md contributions from description."""
        call_count = {"n": 0}

        def fake_extract(desc):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {1: "An AI agent"}
            return {8: "2 weeks", 24: "Fibonacci"}

        monkeypatch.setattr("yeaboi.agent.nodes._extract_answers_from_description", fake_extract)
        monkeypatch.setattr(
            "yeaboi.agent.nodes._load_user_context",
            lambda *a, **kw: ("sprint: 2 weeks", {"name": "User context", "status": "success"}),
        )
        state = {
            "messages": [HumanMessage(content="An AI agent")],
            "_intake_mode": "smart",
        }
        result = project_intake(state)
        msg = result["messages"][0].content
        assert "from SCRUM.md" in msg

    def test_preamble_reads_as_a_sentence(self, monkeypatch):
        monkeypatch.setattr(
            "yeaboi.agent.nodes._extract_answers_from_description", lambda desc: {1: "An AI agent", 8: "2 weeks"}
        )
        state = {"messages": [HumanMessage(content="An AI agent")], "_intake_mode": "smart"}
        msg = project_intake(state)["messages"][0].content
        assert re.search(r"^I extracted \*\*\d+\*\* from your description and filled \*\*\d+\*\* with defaults\.", msg)
        assert "** extracted" not in msg

    def test_summary_provenance_shows_scrum_md(self):
        """Intake summary should show *(from SCRUM.md)* for SCRUM.md-sourced questions."""
        qs = QuestionnaireState()
        qs.answers = {i: f"Answer {i}" for i in range(1, TOTAL_QUESTIONS + 1)}
        qs.extracted_questions = {1, 8, 11}
        qs._scrum_md_questions = {8, 11}
        summary = _build_intake_summary(qs)
        assert "*(from SCRUM.md)*" in summary
        assert "*(from your description)*" in summary


# ---------------------------------------------------------------------------
# _sync_platform_from_url
# ---------------------------------------------------------------------------


class TestSyncPlatformFromUrl:
    def _make_qs(self, url: str, platform: str = "GitHub") -> QuestionnaireState:
        qs = QuestionnaireState()
        qs.answers[17] = url
        qs.answers[16] = platform
        return qs

    def test_github_url_no_change_when_already_github(self):
        qs = self._make_qs("https://github.com/owner/repo", "GitHub")
        _sync_platform_from_url(qs)
        assert qs.answers[16] == "GitHub"

    def test_azdo_url_updates_platform_from_github(self):
        qs = self._make_qs("https://dev.azure.com/org/proj/_git/repo", "GitHub")
        _sync_platform_from_url(qs)
        assert qs.answers[16] == "Azure DevOps"

    def test_azdo_url_removes_from_defaulted_questions(self):
        qs = self._make_qs("https://dev.azure.com/org/proj/_git/repo", "GitHub")
        qs.defaulted_questions.add(16)
        _sync_platform_from_url(qs)
        assert 16 not in qs.defaulted_questions

    def test_empty_url_no_change(self):
        qs = QuestionnaireState()
        qs.answers[16] = "GitHub"
        qs.answers[17] = ""
        _sync_platform_from_url(qs)
        assert qs.answers[16] == "GitHub"

    def test_default_url_value_no_change(self):
        qs = QuestionnaireState()
        qs.answers[16] = "GitHub"
        qs.answers[17] = QUESTION_DEFAULTS[17]  # "No repo URL provided"
        _sync_platform_from_url(qs)
        assert qs.answers[16] == "GitHub"

    def test_unknown_url_no_change(self):
        qs = self._make_qs("https://example.com/owner/repo", "GitHub")
        _sync_platform_from_url(qs)
        assert qs.answers[16] == "GitHub"

    def test_gitlab_url_updates_platform(self):
        qs = self._make_qs("https://gitlab.com/owner/repo", "GitHub")
        _sync_platform_from_url(qs)
        assert qs.answers[16] == "GitLab"

    def test_no_q17_answer_no_change(self):
        qs = QuestionnaireState()
        qs.answers[16] = "GitHub"
        # Q17 not set at all
        _sync_platform_from_url(qs)
        assert qs.answers[16] == "GitHub"


# ── PTO sub-loop tests ──────────────────────────────────────────────────


class TestPTOSubLoop:
    """Tests for the PTO/planned leave sub-loop in the confirmation gate."""

    def _make_qs_at_confirmation(self, intake_mode="standard"):
        """Create a QuestionnaireState ready for confirmation (all questions answered)."""
        qs = QuestionnaireState(current_question=TOTAL_QUESTIONS + 1)
        qs.answers = {i: f"answer {i}" for i in range(1, TOTAL_QUESTIONS + 1)}
        qs.awaiting_confirmation = True
        qs.intake_mode = intake_mode
        return qs

    def test_pto_question_shown_before_summary(self):
        """In standard mode, PTO question should be shown before the summary."""
        # Create a state at Q30 (last question) — answering it triggers the summary flow
        qs = QuestionnaireState(current_question=TOTAL_QUESTIONS)
        qs.answers = {i: f"answer {i}" for i in range(1, TOTAL_QUESTIONS)}
        qs.intake_mode = "standard"
        state = {"messages": [HumanMessage(content="No engineers onboarding")], "questionnaire": qs}
        result = project_intake(state)
        # PTO question should be shown instead of summary
        assert "planned leave" in result["messages"][0].content.lower()
        assert result["questionnaire"]._awaiting_leave_input is True
        assert result["questionnaire"]._leave_input_stage == "ask"

    def test_pto_no_skips_to_summary(self):
        """Answering 'No' to PTO should proceed to summary."""
        qs = self._make_qs_at_confirmation("standard")
        qs._awaiting_leave_input = True
        qs._leave_input_stage = "ask"
        state = {"messages": [HumanMessage(content="2")], "questionnaire": qs}
        result = project_intake(state)
        assert "Project Intake Summary" in result["messages"][0].content
        assert result["questionnaire"]._awaiting_leave_input is False

    def test_pto_yes_asks_for_person(self):
        """Answering 'Yes' to PTO should ask for person name."""
        qs = self._make_qs_at_confirmation("standard")
        qs._awaiting_leave_input = True
        qs._leave_input_stage = "ask"
        state = {"messages": [HumanMessage(content="1")], "questionnaire": qs}
        result = project_intake(state)
        assert "name" in result["messages"][0].content.lower()
        assert result["questionnaire"]._leave_input_stage == "person"

    def test_pto_person_asks_for_start(self):
        """After entering person name, should ask for start date."""
        qs = self._make_qs_at_confirmation("standard")
        qs._awaiting_leave_input = True
        qs._leave_input_stage = "person"
        state = {"messages": [HumanMessage(content="Alice")], "questionnaire": qs}
        result = project_intake(state)
        assert "start date" in result["messages"][0].content.lower()
        assert result["questionnaire"]._leave_input_stage == "start"
        assert result["questionnaire"]._leave_input_buffer["person"] == "Alice"

    def test_pto_invalid_start_date_retries(self):
        """Invalid start date should show error and retry."""
        qs = self._make_qs_at_confirmation("standard")
        qs._awaiting_leave_input = True
        qs._leave_input_stage = "start"
        qs._leave_input_buffer = {"person": "Alice"}
        state = {"messages": [HumanMessage(content="not a date")], "questionnaire": qs}
        result = project_intake(state)
        assert "invalid" in result["messages"][0].content.lower()
        assert result["questionnaire"]._leave_input_stage == "start"  # Still on start

    def test_pto_valid_start_asks_for_end(self):
        """Valid start date should ask for end date."""
        from datetime import date, timedelta

        qs = self._make_qs_at_confirmation("standard")
        qs._awaiting_leave_input = True
        qs._leave_input_stage = "start"
        qs._leave_input_buffer = {"person": "Alice"}
        # Q8=2 weeks, Q10=2 sprints → window is today + 4 weeks; pick a date inside it.
        # Computed relative to today so the test doesn't rot once the date passes.
        qs.answers[8] = "2 weeks"
        qs.answers[10] = "2 sprints"
        start = (date.today() + timedelta(days=7)).strftime("%d/%m/%Y")
        state = {"messages": [HumanMessage(content=start)], "questionnaire": qs}
        result = project_intake(state)
        assert "end date" in result["messages"][0].content.lower()
        assert result["questionnaire"]._leave_input_stage == "end"

    def test_pto_end_before_start_retries(self):
        """End date before start date should show error and retry."""
        qs = self._make_qs_at_confirmation("standard")
        qs._awaiting_leave_input = True
        qs._leave_input_stage = "end"
        qs._leave_input_buffer = {"person": "Alice", "start_date": "2026-04-10"}
        state = {"messages": [HumanMessage(content="06/04/2026")], "questionnaire": qs}
        result = project_intake(state)
        assert "on or after" in result["messages"][0].content.lower()

    def test_pto_valid_end_shows_summary_and_more(self):
        """Valid end date should show entry summary with add/done options."""
        qs = self._make_qs_at_confirmation("standard")
        qs._awaiting_leave_input = True
        qs._leave_input_stage = "end"
        qs._leave_input_buffer = {"person": "Alice", "start_date": "2026-04-06"}
        state = {"messages": [HumanMessage(content="10/04/2026")], "questionnaire": qs}
        result = project_intake(state)
        content = result["messages"][0].content
        assert "Alice" in content
        assert "5 working day(s)" in content
        assert "Add another" in content
        assert len(result["questionnaire"]._planned_leave_entries) == 1

    def test_pto_add_another_loops_back(self):
        """Choosing 'Add another' should loop back to person prompt."""
        qs = self._make_qs_at_confirmation("standard")
        qs._awaiting_leave_input = True
        qs._leave_input_stage = "more?"
        qs._planned_leave_entries = [
            {"person": "Alice", "start_date": "2026-04-06", "end_date": "2026-04-10", "working_days": 5}
        ]
        state = {"messages": [HumanMessage(content="1")], "questionnaire": qs}
        result = project_intake(state)
        assert "name" in result["messages"][0].content.lower()
        assert result["questionnaire"]._leave_input_stage == "person"

    def test_pto_done_exits_to_summary(self):
        """Choosing 'Done' should exit PTO loop and show summary."""
        qs = self._make_qs_at_confirmation("standard")
        qs._awaiting_leave_input = True
        qs._leave_input_stage = "more?"
        qs._planned_leave_entries = [
            {"person": "Alice", "start_date": "2026-04-06", "end_date": "2026-04-10", "working_days": 5}
        ]
        state = {"messages": [HumanMessage(content="2")], "questionnaire": qs}
        result = project_intake(state)
        assert "Project Intake Summary" in result["messages"][0].content
        assert result["questionnaire"]._awaiting_leave_input is False

    def test_pto_ask_invalid_input_reprompts(self):
        """Invalid input on PTO ask stage should re-prompt, not skip."""
        qs = self._make_qs_at_confirmation("standard")
        qs._awaiting_leave_input = True
        qs._leave_input_stage = "ask"
        state = {"messages": [HumanMessage(content="4")], "questionnaire": qs}
        result = project_intake(state)
        assert result["questionnaire"]._awaiting_leave_input is True
        assert result["questionnaire"]._leave_input_stage == "ask"
        assert "[1]" in result["messages"][0].content

    def test_pto_more_invalid_input_reprompts(self):
        """Invalid input on more? stage should re-prompt, not skip."""
        qs = self._make_qs_at_confirmation("standard")
        qs._awaiting_leave_input = True
        qs._leave_input_stage = "more?"
        qs._planned_leave_entries = [
            {"person": "Alice", "start_date": "2026-04-06", "end_date": "2026-04-10", "working_days": 5}
        ]
        state = {"messages": [HumanMessage(content="3")], "questionnaire": qs}
        result = project_intake(state)
        assert result["questionnaire"]._awaiting_leave_input is True
        assert result["questionnaire"]._leave_input_stage == "more?"
        assert "[1]" in result["messages"][0].content

    def test_pto_start_date_before_window_rejected(self):
        """Start date before planning window should be rejected."""
        from datetime import date, timedelta

        qs = self._make_qs_at_confirmation("standard")
        qs._awaiting_leave_input = True
        qs._leave_input_stage = "start"
        qs._leave_input_buffer = {"person": "Alice"}
        # Q8=2 weeks, Q10=2 sprints → window is today + 4 weeks
        qs.answers[8] = "2 weeks"
        qs.answers[10] = "2 sprints"
        # Use a date within 6 months but before the planning window (yesterday)
        yesterday = (date.today() - timedelta(days=1)).strftime("%d/%m/%Y")
        state = {"messages": [HumanMessage(content=yesterday)], "questionnaire": qs}
        result = project_intake(state)
        assert "before" in result["messages"][0].content.lower()
        assert result["questionnaire"]._leave_input_stage == "start"

    def test_pto_start_date_after_window_rejected(self):
        """Start date after planning window should be rejected."""
        qs = self._make_qs_at_confirmation("standard")
        qs._awaiting_leave_input = True
        qs._leave_input_stage = "start"
        qs._leave_input_buffer = {"person": "Alice"}
        qs.answers[8] = "2 weeks"
        qs.answers[10] = "2 sprints"
        state = {"messages": [HumanMessage(content="01/01/2030")], "questionnaire": qs}
        result = project_intake(state)
        assert "after" in result["messages"][0].content.lower()
        assert result["questionnaire"]._leave_input_stage == "start"

    def test_pto_end_date_after_window_rejected(self):
        """End date after planning window should be rejected."""
        from datetime import date, timedelta

        qs = self._make_qs_at_confirmation("standard")
        qs._awaiting_leave_input = True
        qs._leave_input_stage = "end"
        # Start date is tomorrow (valid within window)
        tomorrow = date.today() + timedelta(days=1)
        qs._leave_input_buffer = {"person": "Alice", "start_date": tomorrow.isoformat()}
        qs.answers[8] = "1 week"
        qs.answers[10] = "1 sprint"
        # End date far in the future
        state = {"messages": [HumanMessage(content="01/01/2030")], "questionnaire": qs}
        result = project_intake(state)
        assert "after" in result["messages"][0].content.lower()
        assert result["questionnaire"]._leave_input_stage == "end"

    def test_quick_mode_skips_pto(self):
        """Quick mode should skip PTO entirely and go straight to summary."""
        qs = self._make_qs_at_confirmation("quick")
        state = {"messages": [HumanMessage(content="1")], "questionnaire": qs}
        result = project_intake(state)
        # Quick mode doesn't trigger PTO — "1" is treated as velocity accept
        assert result["questionnaire"]._awaiting_leave_input is False

    def test_all_four_date_formats_accepted(self):
        """All four date formats (DD/MM/YYYY, DD/MM/YY, DD-MM-YYYY, DD-MM-YY) should work."""
        from datetime import date

        from yeaboi.agent.nodes import _parse_date_dmy

        assert _parse_date_dmy("06/04/2026") == date(2026, 4, 6)
        assert _parse_date_dmy("06/04/26") == date(2026, 4, 6)
        assert _parse_date_dmy("06-04-2026") == date(2026, 4, 6)
        assert _parse_date_dmy("06-04-26") == date(2026, 4, 6)

    def test_rejects_dates_far_in_past(self):
        """Dates more than 6 months in the past should be rejected (catches 2-digit year typos)."""
        from yeaboi.agent.nodes import _parse_date_dmy

        assert _parse_date_dmy("12/12/12") is None  # 2012 — clearly in the past
        assert _parse_date_dmy("01/01/2020") is None


# ── Smart Intake Improvements tests ─────────────────────────────────


class TestAnswerSources:
    """Tests for answer_sources tracking (Step 1: Answer Confidence Signalling)."""

    def test_direct_answer_sets_source(self):
        """A direct user answer should set answer_sources to 'direct'."""
        qs = QuestionnaireState(current_question=1)
        state = {
            "messages": [HumanMessage(content="A todo app for tracking tasks")],
            "questionnaire": qs,
        }
        # Disable vague check and extraction
        from unittest.mock import patch

        with patch("yeaboi.agent.nodes._check_vague_answer", return_value=None):
            result = project_intake(state)
        assert result["questionnaire"].answer_sources.get(1) == "direct"

    def test_extracted_answer_sets_source(self):
        """Auto-applied extractions should set answer_sources to 'extracted'."""
        from yeaboi.agent.nodes import _auto_apply_extractions

        qs = QuestionnaireState()
        _auto_apply_extractions(qs, {6: "5 engineers", 11: "Python, FastAPI"})
        assert qs.answer_sources[6] == "extracted"
        assert qs.answer_sources[11] == "extracted"

    def test_defaulted_answer_sets_source(self):
        """Defaulted answers should set answer_sources to 'defaulted'."""
        from yeaboi.agent.nodes import _auto_default_remaining

        qs = QuestionnaireState()
        qs.answers[1] = "A project"
        _auto_default_remaining(qs, frozenset({1, 2, 3, 4, 6, 11}))
        # Q5 has a default ("No hard deadlines") and is not essential
        assert qs.answer_sources.get(5) == "defaulted"

    def test_batch_defaults_sets_source(self):
        """_batch_defaults_for_phase should set answer_sources to 'defaulted'."""
        qs = QuestionnaireState(current_question=6)
        summary_lines, count = _batch_defaults_for_phase(qs)
        # Q8 has a default (2 weeks)
        assert qs.answer_sources.get(8) == "defaulted"

    def test_summary_shows_confidence_breakdown(self):
        """_build_intake_summary should show 'direct | extracted | defaulted' breakdown."""
        qs = QuestionnaireState()
        qs.answers = {i: f"Answer {i}" for i in range(1, TOTAL_QUESTIONS + 1)}
        qs.answer_sources = {i: "direct" for i in range(1, TOTAL_QUESTIONS + 1)}
        qs.answer_sources[1] = "extracted"
        qs.answer_sources[2] = "extracted"
        qs.answer_sources[5] = "defaulted"
        qs.extracted_questions = {1, 2}
        qs.defaulted_questions = {5}
        summary = _build_intake_summary(qs)
        assert "2 extracted" in summary
        assert "1 defaulted" in summary
        assert "direct" in summary

    def test_low_confidence_areas_in_prompt_quality(self):
        """compute_prompt_quality should populate low_confidence_areas for defaulted essentials."""
        from yeaboi.agent.nodes import compute_prompt_quality

        qs = QuestionnaireState()
        qs.answers = {i: f"Answer {i}" for i in range(1, TOTAL_QUESTIONS + 1)}
        # Default Q2 (essential) and Q6 (essential)
        qs.defaulted_questions = {2, 6}
        qs.answer_sources = {i: "direct" for i in range(1, TOTAL_QUESTIONS + 1)}
        qs.answer_sources[2] = "defaulted"
        qs.answer_sources[6] = "defaulted"
        rating = compute_prompt_quality(qs)
        assert "Project type" in rating.low_confidence_areas
        assert "Team size" in rating.low_confidence_areas

    def test_no_low_confidence_when_all_direct(self):
        """No low_confidence_areas when all essentials are direct."""
        from yeaboi.agent.nodes import compute_prompt_quality

        qs = QuestionnaireState()
        qs.answers = {i: f"Answer {i}" for i in range(1, TOTAL_QUESTIONS + 1)}
        qs.answer_sources = {i: "direct" for i in range(1, TOTAL_QUESTIONS + 1)}
        rating = compute_prompt_quality(qs)
        assert rating.low_confidence_areas == ()


class TestKeywordExtraction:
    """Tests for _keyword_extract_fallback (Step 2: Smarter Extraction)."""

    def test_refactor_infers_existing_codebase(self):
        """'refactor' keyword should infer Q2 as 'Existing codebase'."""
        from yeaboi.agent.nodes import _keyword_extract_fallback

        extracted: dict[int, str] = {}
        _keyword_extract_fallback("We're refactoring our legacy API", extracted)
        assert extracted[2] == "Existing codebase"

    def test_greenfield_keyword_infers_greenfield(self):
        """'from scratch' keyword should infer Q2 as 'Greenfield'."""
        from yeaboi.agent.nodes import _keyword_extract_fallback

        extracted: dict[int, str] = {}
        _keyword_extract_fallback("Building a new project from scratch", extracted)
        assert extracted[2] == "Greenfield"

    def test_llm_extracted_q2_takes_priority(self):
        """LLM-extracted Q2 should not be overwritten by keyword fallback."""
        from yeaboi.agent.nodes import _keyword_extract_fallback

        extracted: dict[int, str] = {2: "Hybrid"}
        _keyword_extract_fallback("We're refactoring our legacy API", extracted)
        assert extracted[2] == "Hybrid"  # LLM value preserved

    def test_stripe_extracts_q12(self):
        """'stripe' keyword should extract Q12 integration."""
        from yeaboi.agent.nodes import _keyword_extract_fallback

        extracted: dict[int, str] = {}
        _keyword_extract_fallback("integrating Stripe for payments and Sentry for monitoring", extracted)
        assert "Sentry" in extracted[12]
        assert "Stripe" in extracted[12]

    def test_kubernetes_extracts_q13(self):
        """'kubernetes' keyword should extract Q13 constraint."""
        from yeaboi.agent.nodes import _keyword_extract_fallback

        extracted: dict[int, str] = {}
        _keyword_extract_fallback("deployed on kubernetes with docker containers", extracted)
        assert "docker" in extracted[13]
        assert "kubernetes" in extracted[13]

    def test_no_keywords_no_extraction(self):
        """Description without any keywords should not add Q2/Q12/Q13."""
        from yeaboi.agent.nodes import _keyword_extract_fallback

        extracted: dict[int, str] = {}
        _keyword_extract_fallback("A simple web application", extracted)
        assert 2 not in extracted
        assert 12 not in extracted
        assert 13 not in extracted

    def test_case_insensitive(self):
        """Keywords should match case-insensitively."""
        from yeaboi.agent.nodes import _keyword_extract_fallback

        extracted: dict[int, str] = {}
        _keyword_extract_fallback("Running on AWS with Docker", extracted)
        assert 13 in extracted


class TestAdaptiveQuestionText:
    """Tests for _resolve_adaptive_text (Step 3: Adaptive Question Text)."""

    def test_q7_personalized_with_team_size(self):
        """Q7 should reference Q6 team size when available."""
        from yeaboi.agent.nodes import _resolve_adaptive_text

        qs = QuestionnaireState()
        qs.answers[6] = "5"
        text = _resolve_adaptive_text(7, qs)
        assert "5 engineers" in text

    def test_q12_personalized_with_tech_stack(self):
        """Q12 should reference Q11 tech stack when available."""
        from yeaboi.agent.nodes import _resolve_adaptive_text

        qs = QuestionnaireState()
        qs.answers[11] = "Python, FastAPI, PostgreSQL"
        text = _resolve_adaptive_text(12, qs)
        assert "Python, FastAPI, PostgreSQL" in text

    def test_q13_personalized_with_project_type(self):
        """Q13 should reference Q2 project type and show appropriate hints."""
        from yeaboi.agent.nodes import _resolve_adaptive_text

        qs = QuestionnaireState()
        qs.answers[2] = "Existing codebase"
        text = _resolve_adaptive_text(13, qs)
        assert "Existing codebase" in text
        assert "backward compatibility" in text

    def test_fallback_when_dependency_missing(self):
        """Should return default question text when dependency is missing."""
        from yeaboi.agent.nodes import _resolve_adaptive_text

        qs = QuestionnaireState()
        # Q6 not answered — Q7 should fall back to default
        text = _resolve_adaptive_text(7, qs)
        assert text == INTAKE_QUESTIONS[7]

    def test_fallback_when_dependency_defaulted(self):
        """Should return default question text when dependency was defaulted."""
        from yeaboi.agent.nodes import _resolve_adaptive_text

        qs = QuestionnaireState()
        qs.answers[6] = "Roles not specified"
        qs.defaulted_questions.add(6)
        text = _resolve_adaptive_text(7, qs)
        assert text == INTAKE_QUESTIONS[7]

    def test_no_template_returns_default(self):
        """Questions without templates should return INTAKE_QUESTIONS text."""
        from yeaboi.agent.nodes import _resolve_adaptive_text

        qs = QuestionnaireState()
        text = _resolve_adaptive_text(1, qs)
        assert text == INTAKE_QUESTIONS[1]

    def test_gap_prompt_uses_adaptive_text(self):
        """_build_gap_prompt should use adaptive text when available."""
        qs = QuestionnaireState()
        qs.answers[6] = "3"
        prompt, q_nums = _build_gap_prompt([7], qs)
        assert "3 engineers" in prompt


class TestFollowUpTemplates:
    """Tests for custom follow-up templates (Step 4: Follow-up Quality)."""

    def test_custom_template_injected_for_known_question(self, monkeypatch):
        """When q_num has a FOLLOW_UP_TEMPLATES entry, it should appear in the LLM prompt."""
        fake_response = AIMessage(content='{"vague": true, "follow_up": "Who are the users?", "choices": ["A", "B"]}')
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response

        def capturing_llm(**kwargs):
            return mock_llm

        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", capturing_llm)

        result = _check_vague_answer("What problem?", "stuff", q_num=3)
        assert result is not None
        # The prompt sent to the LLM should contain the custom template hint
        call_args = mock_llm.invoke.call_args[0][0]
        prompt_text = call_args[0].content
        assert "Who experiences this problem?" in prompt_text

    def test_no_template_for_unknown_question(self, monkeypatch):
        """Questions without a FOLLOW_UP_TEMPLATES entry should NOT get a custom hint."""
        fake_response = AIMessage(content='{"vague": true, "follow_up": "Tell me more", "choices": ["A", "B"]}')
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        _check_vague_answer("What is the project?", "stuff", q_num=1)
        call_args = mock_llm.invoke.call_args[0][0]
        prompt_text = call_args[0].content
        assert "If vague, use this follow-up:" not in prompt_text

    def test_q_num_zero_no_template(self, monkeypatch):
        """q_num=0 (default) should not inject any custom template."""
        fake_response = AIMessage(content='{"vague": false}')
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kwargs: mock_llm)

        _check_vague_answer("Question?", "answer")
        call_args = mock_llm.invoke.call_args[0][0]
        prompt_text = call_args[0].content
        assert "If vague, use this follow-up:" not in prompt_text


class TestCrossQuestionValidation:
    """Tests for _validate_cross_questions (Step 5: Cross-Question Validation)."""

    def test_greenfield_with_repo_url_warns(self):
        """Greenfield + repo URL should produce a warning."""
        from yeaboi.agent.nodes import _validate_cross_questions

        answers = {2: "Greenfield", 17: "https://github.com/org/repo"}
        warnings = _validate_cross_questions(answers)
        assert len(warnings) == 1
        assert "Greenfield" in warnings[0].message
        assert "repo URL" in warnings[0].message

    def test_greenfield_case_insensitive(self):
        """Greenfield check should be case-insensitive."""
        from yeaboi.agent.nodes import _validate_cross_questions

        for variant in ("greenfield", "GREENFIELD", "Greenfield", " greenfield "):
            answers = {2: variant, 17: "https://github.com/org/repo"}
            warnings = _validate_cross_questions(answers)
            assert len(warnings) == 1, f"Failed for q2={variant!r}"

    def test_existing_codebase_with_repo_url_no_warning(self):
        """Existing codebase + repo URL should NOT produce a warning."""
        from yeaboi.agent.nodes import _validate_cross_questions

        answers = {2: "Existing codebase", 17: "https://github.com/org/repo"}
        warnings = _validate_cross_questions(answers)
        assert all("Greenfield" not in w.message for w in warnings)

    def test_long_timeline_warns(self):
        """Sprint weeks × sprint count > 26 weeks should produce an info warning."""
        from yeaboi.agent.nodes import _validate_cross_questions

        answers = {8: "2 weeks", 10: "15 sprints"}
        warnings = _validate_cross_questions(answers)
        assert any("months" in w.message for w in warnings)

    def test_short_timeline_no_warning(self):
        """Sprint weeks × sprint count ≤ 26 weeks should NOT warn."""
        from yeaboi.agent.nodes import _validate_cross_questions

        answers = {8: "2 weeks", 10: "5 sprints"}
        warnings = _validate_cross_questions(answers)
        assert not any("months" in w.message for w in warnings)

    def test_velocity_sanity_warns_too_high(self):
        """Velocity/team_size > 15 should produce a warning."""
        from yeaboi.agent.nodes import _validate_cross_questions

        answers = {6: "2 engineers", 9: "50 points"}
        warnings = _validate_cross_questions(answers)
        assert any("pts/engineer" in w.message for w in warnings)

    def test_velocity_sanity_warns_too_low(self):
        """Velocity/team_size < 2 should produce a warning."""
        from yeaboi.agent.nodes import _validate_cross_questions

        answers = {6: "10 engineers", 9: "5 points"}
        warnings = _validate_cross_questions(answers)
        assert any("pts/engineer" in w.message for w in warnings)

    def test_normal_velocity_no_warning(self):
        """Velocity/team_size in 2-15 range should NOT warn."""
        from yeaboi.agent.nodes import _validate_cross_questions

        answers = {6: "5 engineers", 9: "30 points"}
        warnings = _validate_cross_questions(answers)
        assert not any("pts/engineer" in w.message for w in warnings)

    def test_clean_answers_no_warnings(self):
        """A complete, consistent answer set should produce no warnings."""
        from yeaboi.agent.nodes import _validate_cross_questions

        answers = {
            2: "Existing codebase",
            6: "5 engineers",
            8: "2 weeks",
            9: "30 points",
            10: "5 sprints",
            17: "https://github.com/org/repo",
        }
        warnings = _validate_cross_questions(answers)
        assert warnings == []

    def test_warnings_appear_in_summary(self):
        """Validation warnings should appear in the intake summary."""
        qs = QuestionnaireState()
        qs.answers = {i: f"Answer {i}" for i in range(1, TOTAL_QUESTIONS + 1)}
        qs.answers[2] = "Greenfield"
        qs.answers[17] = "https://github.com/org/repo"
        summary = _build_intake_summary(qs)
        assert "Heads up" in summary
        assert "Greenfield" in summary


class TestConditionalEssentials:
    """Tests for CONDITIONAL_ESSENTIALS and their integration with _find_essential_gaps."""

    def test_q7_promoted_when_q6_answered(self):
        """Q7 (team roles) becomes a gap when Q6 (team size) has a real answer."""
        qs = QuestionnaireState()
        qs.answers = {6: "5"}
        gaps = _find_essential_gaps(qs, SMART_ESSENTIALS)
        assert 7 in gaps

    def test_q7_not_promoted_when_q6_defaulted(self):
        """Q7 stays defaulted when Q6 was itself defaulted — no point asking roles."""
        qs = QuestionnaireState()
        qs.answers = {6: "3"}
        qs.defaulted_questions.add(6)
        gaps = _find_essential_gaps(qs, SMART_ESSENTIALS)
        assert 7 not in gaps

    def test_q12_promoted_when_q11_answered(self):
        """Q12 (integrations) becomes a gap when Q11 (tech stack) is answered."""
        qs = QuestionnaireState()
        qs.answers = {11: "React, FastAPI"}
        gaps = _find_essential_gaps(qs, SMART_ESSENTIALS)
        assert 12 in gaps

    def test_q12_not_promoted_when_q11_defaulted(self):
        qs = QuestionnaireState()
        qs.answers = {11: "Not specified"}
        qs.defaulted_questions.add(11)
        gaps = _find_essential_gaps(qs, SMART_ESSENTIALS)
        assert 12 not in gaps

    def test_q13_promoted_when_q2_answered(self):
        """Q13 (constraints) becomes a gap when Q2 (project type) is answered."""
        qs = QuestionnaireState()
        qs.answers = {2: "Greenfield"}
        gaps = _find_essential_gaps(qs, SMART_ESSENTIALS)
        assert 13 in gaps

    def test_q13_not_promoted_when_q2_defaulted(self):
        qs = QuestionnaireState()
        qs.answers = {2: "Greenfield"}
        qs.defaulted_questions.add(2)
        gaps = _find_essential_gaps(qs, SMART_ESSENTIALS)
        assert 13 not in gaps

    def test_conditional_not_promoted_when_directly_answered(self):
        """If Q7 has a direct (non-defaulted) answer, it should not appear as a gap."""
        qs = QuestionnaireState()
        qs.answers = {6: "5", 7: "Backend, Frontend"}
        gaps = _find_essential_gaps(qs, SMART_ESSENTIALS)
        assert 7 not in gaps

    def test_conditional_promoted_when_defaulted_and_prereq_answered(self):
        """Q7 defaulted by _auto_default_remaining should become a gap when Q6 is answered."""
        qs = QuestionnaireState()
        qs.answers = {6: "5", 7: "Roles not specified — assuming generalist/fullstack team"}
        qs.defaulted_questions.add(7)
        gaps = _find_essential_gaps(qs, SMART_ESSENTIALS)
        assert 7 in gaps

    def test_q29_not_promoted_in_smart_mode(self):
        """Q29 (unplanned leave %) no longer promoted — defaults to 10%."""
        qs = QuestionnaireState()
        qs.answers = {6: "5"}
        gaps = _find_essential_gaps(qs, SMART_ESSENTIALS)
        assert 29 not in gaps

    def test_multiple_conditionals_promoted_together(self):
        """All four conditionals can fire at once when prerequisites are met."""
        qs = QuestionnaireState()
        qs.answers = {2: "Existing codebase", 6: "4", 11: "Python, Django"}
        gaps = _find_essential_gaps(qs, SMART_ESSENTIALS)
        assert 7 in gaps
        assert 12 in gaps
        assert 13 in gaps
        assert 29 not in gaps  # Q29 no longer conditional

    def test_conditional_essentials_mapping_is_complete(self):
        """Verify the mapping covers Q7→Q6, Q12→Q11, Q13→Q2."""
        assert CONDITIONAL_ESSENTIALS == {7: 6, 12: 11, 13: 2}


class TestMultiChoiceMetadata:
    """Tests for multi_choice question metadata and is_choice_question."""

    def test_q7_is_multi_choice(self):
        meta = QUESTION_METADATA[7]
        assert meta.question_type == "multi_choice"
        assert "Backend" in meta.options
        assert "Frontend" in meta.options
        assert "Fullstack" in meta.options

    def test_q13_is_multi_choice(self):
        meta = QUESTION_METADATA[13]
        assert meta.question_type == "multi_choice"
        assert "AWS" in meta.options
        assert "Microservices" in meta.options

    def test_q19_is_single_choice(self):
        meta = QUESTION_METADATA[19]
        assert meta.question_type == "single_choice"
        assert meta.options == ("Yes", "No", "Partial/in progress")
        assert meta.default_index == 1

    def test_q25_is_single_choice(self):
        meta = QUESTION_METADATA[25]
        assert meta.question_type == "single_choice"
        assert len(meta.options) == 2
        assert meta.default_index == 1

    def test_q30_is_single_choice(self):
        meta = QUESTION_METADATA[30]
        assert meta.question_type == "single_choice"
        assert "None" in meta.options
        assert meta.default_index == 0

    def test_is_choice_question_includes_multi_choice(self):
        """is_choice_question should return True for both single and multi choice."""
        assert is_choice_question(7) is True  # multi_choice
        assert is_choice_question(13) is True  # multi_choice
        assert is_choice_question(2) is True  # single_choice
        assert is_choice_question(1) is False  # free text

    def test_multi_choice_has_no_default_index(self):
        """Multi-choice questions don't use default_index — users toggle selections."""
        for q_num, meta in QUESTION_METADATA.items():
            if meta.question_type == "multi_choice":
                assert meta.default_index is None, f"Q{q_num} multi_choice should have no default_index"


# ── Defaults-all command tests (the /finish fast path) ───────────────


class TestDefaultsAllCommand:
    """Tests for _is_defaults_all_intent and the all-phases defaults handler."""

    @pytest.mark.parametrize("text", ["defaults all", "all defaults", "use defaults for everything", "DEFAULTS ALL"])
    def test_is_defaults_all_intent_positive(self, text):
        from yeaboi.agent.nodes import _is_defaults_all_intent

        assert _is_defaults_all_intent(text) is True

    @pytest.mark.parametrize("text", ["finish", "defaults", "default", "use defaults", "finish it", ""])
    def test_is_defaults_all_intent_negative(self, text):
        """Bare "finish" is a plausible free-text answer and must NOT match."""
        from yeaboi.agent.nodes import _is_defaults_all_intent

        assert _is_defaults_all_intent(text) is False

    def test_intents_are_disjoint(self):
        from yeaboi.agent.nodes import _DEFAULTS_ALL_EXACT, _DEFAULTS_EXACT, _is_defaults_all_intent

        assert not (_DEFAULTS_ALL_EXACT & _DEFAULTS_EXACT)
        for text in _DEFAULTS_EXACT:
            assert _is_defaults_all_intent(text) is False
        for text in _DEFAULTS_ALL_EXACT:
            assert _is_defaults_intent(text) is False

    def test_batch_defaults_with_last_q_spans_phases(self):
        """last_q=TOTAL_QUESTIONS defaults across every remaining phase in one pass."""
        qs = QuestionnaireState(current_question=8)
        qs.answers = {6: "3 engineers", 7: "2 backend"}
        _summary, count = _batch_defaults_for_phase(qs, last_q=TOTAL_QUESTIONS)
        # Far past the current phase: Q16-Q26 and Q28-Q30 all have defaults.
        assert 16 in qs.defaulted_questions
        assert 30 in qs.defaulted_questions
        assert count > 10
        # Essential questions with no default are flagged, not invented.
        assert 11 in qs.skipped_questions
        assert 11 not in qs.answers

    def test_defaults_all_in_intake_node_goes_to_summary(self, monkeypatch):
        """Smart mode: one turn lands at the summary with PTO auto-defaulted."""
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)

        qs = QuestionnaireState(current_question=8, intake_mode="smart")
        qs.answers = {i: f"answer {i}" for i in range(1, 8)}
        state = {"messages": [HumanMessage(content="defaults all")], "questionnaire": qs}
        result = project_intake(state)
        out_qs = result["questionnaire"]
        assert out_qs.awaiting_confirmation is True
        assert result.get("pending_review") == "project_intake"
        # PTO skipped, matching quick mode — no "planned leave" prompt.
        assert out_qs._leave_input_stage == "done"
        assert out_qs._awaiting_leave_input is False
        assert "planned leave" not in result["messages"][0].content
        assert "Fast-forward" in result["messages"][0].content
        assert "Project Intake Summary" in result["messages"][0].content

    def test_defaults_all_small_project_leaves_pto_state_alone(self, monkeypatch):
        """small_project never plans capacity — the PTO bypass must not touch it."""
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)

        qs = QuestionnaireState(current_question=8, intake_mode="small_project")
        qs.answers = {i: f"answer {i}" for i in range(1, 8)}
        state = {"messages": [HumanMessage(content="defaults all")], "questionnaire": qs}
        result = project_intake(state)
        assert result["questionnaire"]._leave_input_stage == ""
        assert result["questionnaire"].awaiting_confirmation is True

    def test_defaults_all_reports_gaps(self, monkeypatch):
        """Unanswered essentials with no default are counted in the ack."""
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)

        qs = QuestionnaireState(current_question=6, intake_mode="smart")
        qs.answers = {i: f"answer {i}" for i in range(1, 6)}  # Q6 and Q11 still open
        state = {"messages": [HumanMessage(content="defaults all")], "questionnaire": qs}
        result = project_intake(state)
        assert "essential question(s) had no default" in result["messages"][0].content


class TestGapModeSkipDefaults:
    """Regression: in smart/quick/small_project (gap) mode the skip/defaults
    literals used to be recorded verbatim as the gap answer — the standard-mode
    handlers sit below the gap-filling block and never ran."""

    def _qs(self, current_q=8, mode="smart"):
        qs = QuestionnaireState(current_question=current_q, intake_mode=mode)
        qs.answers = {i: f"answer {i}" for i in range(1, current_q)}
        return qs

    def test_skip_stores_default_not_the_literal(self, monkeypatch):
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)
        qs = self._qs(8)
        result = project_intake({"messages": [HumanMessage(content="skip")], "questionnaire": qs})
        out = result["questionnaire"]
        assert out.answers[8] != "skip"
        assert 8 in out.defaulted_questions
        assert "assume" in result["messages"][0].content.lower()

    def test_skip_flags_no_default_essential_and_moves_on(self, monkeypatch):
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)
        qs = QuestionnaireState(current_question=6, intake_mode="smart")
        qs.answers = {2: "a", 3: "b", 4: "c"}
        result = project_intake({"messages": [HumanMessage(content="skip")], "questionnaire": qs})
        out = result["questionnaire"]
        assert 6 in out.skipped_questions
        assert 6 not in out.answers
        # The next prompt is a different question — a skipped gap is never re-asked.
        assert out.current_question != 6

    def test_defaults_lands_at_pto_in_smart_mode(self, monkeypatch):
        """Per-phase "defaults" fills everything but still stops at PTO — only
        /finish's "defaults all" bypasses it."""
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)
        qs = self._qs(8)
        result = project_intake({"messages": [HumanMessage(content="defaults")], "questionnaire": qs})
        out = result["questionnaire"]
        assert out.answers[8] != "defaults"
        assert out._awaiting_leave_input is True
        assert "planned leave" in result["messages"][0].content

    def test_skipped_essential_is_not_a_gap(self):
        from yeaboi.agent.nodes import _find_essential_gaps
        from yeaboi.prompts.intake import SMART_ESSENTIALS

        qs = QuestionnaireState(current_question=6, intake_mode="smart")
        qs.answers = {2: "a", 3: "b", 4: "c"}
        qs.skipped_questions.add(6)
        assert 6 not in _find_essential_gaps(qs, SMART_ESSENTIALS)


class TestAnalysisProfileInIntake:
    """A seeded analysis profile id is read once and stashed for prior art."""

    @pytest.fixture(autouse=True)
    def _hermetic(self, monkeypatch):
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)
        monkeypatch.setattr("yeaboi.agent.nodes._load_user_context", lambda *a, **kw: (None, {}))
        monkeypatch.setattr("yeaboi.agent.nodes._extract_answers_from_description", lambda desc: {})

    def _run(self, monkeypatch, extra_state):
        from yeaboi.agent import nodes

        loads: list = []
        monkeypatch.setattr(nodes, "_load_profile_by_id", lambda pid: loads.append(pid) or (None, None))
        state = {
            "messages": [HumanMessage(content="Building a todo app for task management")],
            "_intake_mode": "smart",
            "analysis_profile_id": "jira-DEMO-1",
            **extra_state,
        }
        result = project_intake(state)
        return result["questionnaire"], loads

    def test_the_profile_seeds_intake(self, monkeypatch):
        qs, loads = self._run(monkeypatch, {})
        assert loads == ["jira-DEMO-1"]  # the Q6/Q8/Q9 auto-fill read
        assert qs._analysis_profile_id == "jira-DEMO-1"
        assert qs._analysis_enabled is True


class TestSoloIntake:
    """A Solo-world run plans for one developer: the team questions are defaulted, never asked."""

    @pytest.fixture(autouse=True)
    def _hermetic(self, monkeypatch):
        monkeypatch.setattr("yeaboi.agent.nodes._check_vague_answer", lambda q, a, n=0: None)
        monkeypatch.setattr("yeaboi.agent.nodes._load_user_context", lambda *a, **kw: (None, {}))

    @staticmethod
    def _run(monkeypatch, extracted, *, solo=True, **state_extra):
        monkeypatch.setattr("yeaboi.agent.nodes._extract_answers_from_description", lambda desc: dict(extracted))
        state = {
            "messages": [HumanMessage(content="Building a todo app")],
            "_intake_mode": "smart",
            "analysis_profile_id": "team-apollo",
            **({"solo": True} if solo else {}),
            **state_extra,
        }
        return project_intake(state)

    @staticmethod
    def _profile(monkeypatch, contributors):
        from types import SimpleNamespace

        profile = SimpleNamespace(velocity_avg=30.0, tech_stack=(), integrations=())
        examples = {
            "contributor_stats": [{"name": n, "per_sprint": 15.0, "top_discipline": "backend"} for n in contributors]
        }
        monkeypatch.setattr("yeaboi.agent.nodes._load_profile_by_id", lambda pid: (profile, examples))

    def test_team_questions_are_defaulted_not_asked(self, monkeypatch):
        result = self._run(monkeypatch, {1: "A todo app", 6: "3 engineers"})
        qs = result["questionnaire"]
        assert qs.answers[6] == "1"
        assert qs.answers[7] == SOLO_ROLES_DEFAULT
        assert qs.answers[30] == "None"
        for q in (6, 7, 30):
            assert q in qs.defaulted_questions
            assert qs.answer_sources[q] is AnswerSource.DEFAULTED
        # The described "3 engineers" never survives as a suggestion.
        assert 6 not in qs.suggested_answers
        assert INTAKE_QUESTIONS[6] not in result["messages"][0].content

    def test_none_of_the_team_questions_is_a_gap(self, monkeypatch):
        qs = self._run(monkeypatch, {1: "A todo app"})["questionnaire"]
        assert set(_find_essential_gaps(qs, SMART_ESSENTIALS)) & {6, 7, 30} == set()
        # The other essentials are still asked.
        assert 2 in _find_essential_gaps(qs, SMART_ESSENTIALS)

    def test_the_plan_is_sized_for_one_developer(self, monkeypatch):
        qs = self._run(monkeypatch, {1: "A todo app"})["questionnaire"]
        sizing = _extract_team_and_velocity(qs)
        assert sizing["team_size"] == 1
        assert sizing["velocity_per_sprint"] == _VELOCITY_PER_ENGINEER
        assert qs._jira_org_team_size == 1

    def test_a_two_contributor_profile_offers_no_member_picker(self, monkeypatch):
        self._profile(monkeypatch, ["Alice", "Bob"])
        qs = self._run(monkeypatch, {1: "A todo app"})["questionnaire"]
        assert not qs._q6_member_select
        assert 6 not in qs._follow_up_choices
        assert qs.answers[6] == "1"
        # The profile's velocity lands as a per-developer rate, not the team's.
        assert qs.answers[9].startswith("15 points per sprint")

    def test_the_team_world_still_offers_the_picker(self, monkeypatch):
        self._profile(monkeypatch, ["Alice", "Bob"])
        qs = self._run(monkeypatch, {1: "A todo app"}, solo=False)["questionnaire"]
        assert qs._q6_member_select

    def test_apply_solo_defaults_drops_q6_from_the_essentials(self):
        qs = QuestionnaireState()
        qs.suggested_answers[6] = "4 engineers"
        qs.extracted_questions.add(6)
        assert _apply_solo_defaults(qs, SMART_ESSENTIALS) == SMART_ESSENTIALS - {6}
        assert qs.answers[6] == "1" and 6 not in qs.suggested_answers and 6 not in qs.extracted_questions
        assert qs._jira_org_team_size == 1

    def test_solo_velocity_from_profile(self):
        from types import SimpleNamespace

        contributors = {"contributor_stats": [{"name": "A"}, {"name": "B"}]}
        assert _solo_velocity_from_profile(SimpleNamespace(velocity_avg=30.0), contributors).startswith("15 points")
        assert _solo_velocity_from_profile(SimpleNamespace(velocity_avg=30.0), {}).startswith("30 points")
        assert _solo_velocity_from_profile(SimpleNamespace(velocity_avg=0.0), contributors) == ""
