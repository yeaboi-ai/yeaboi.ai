"""Tests for core agent node functions (call_model, should_continue, human_review)."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END

from yeaboi.agent.nodes import (
    _HIGH_RISK_TOOLS,
    _is_llm_connection_error,
    _is_local_llm_down,
    _is_ollama_model_missing,
    _local_llm_hint,
    _should_reraise_llm_error,
    _trim_history_for_local,
    _user_confirmed,
    call_model,
    human_review,
    make_call_model,
    should_continue,
)
from yeaboi.prompts import get_system_prompt

# ── LLM error classification ─────────────────────────────────────────


class TestLlmConnectionError:
    """_is_llm_connection_error must catch a down local server, not auth errors."""

    def test_builtin_connection_refused(self):
        assert _is_llm_connection_error(ConnectionRefusedError("refused")) is True

    def test_httpx_connect_error(self):
        import httpx

        assert _is_llm_connection_error(httpx.ConnectError("connection failed")) is True

    def test_wrapped_cause_chain(self):
        """langchain/ollama wrap the transport error — the __cause__ chain must be walked."""
        inner = ConnectionRefusedError("refused")
        outer = RuntimeError("request failed")
        outer.__cause__ = inner
        assert _is_llm_connection_error(outer) is True

    def test_errno_message_patterns(self):
        assert _is_llm_connection_error(Exception("[Errno 61] Connection refused")) is True
        assert _is_llm_connection_error(Exception("[Errno 111] connection refused")) is True

    def test_plain_error_is_not_connection(self):
        assert _is_llm_connection_error(ValueError("bad json")) is False

    def test_auth_error_is_not_connection(self):
        assert _is_llm_connection_error(Exception("api key invalid")) is False


class TestLocalLlmDown:
    """_is_local_llm_down gates connection errors on the ollama provider."""

    def test_true_for_ollama_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        assert _is_local_llm_down(ConnectionRefusedError("refused")) is True

    def test_false_for_cloud_provider(self, monkeypatch):
        """Cloud transient network blips keep the graceful-fallback behaviour."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        assert _is_local_llm_down(ConnectionRefusedError("refused")) is False

    def test_false_for_non_connection_error(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        assert _is_local_llm_down(ValueError("bad json")) is False


class TestShouldReraiseLlmError:
    def test_ollama_connection_error_reraises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        assert _should_reraise_llm_error(ConnectionRefusedError("refused")) is True

    def test_cloud_connection_error_falls_back(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        assert _should_reraise_llm_error(ConnectionRefusedError("refused")) is False

    def test_generic_error_falls_back(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        assert _should_reraise_llm_error(ValueError("parse failed")) is False

    def test_ollama_model_missing_reraises(self, monkeypatch):
        """A not-pulled model must surface, never silently fall back."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        assert _should_reraise_llm_error(Exception("model 'qwen3:8b' not found, try pulling it first")) is True


class TestOllamaModelMissing:
    """_is_ollama_model_missing must catch the server's 404 for un-pulled models."""

    def test_message_pattern(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        assert _is_ollama_model_missing(Exception("model 'qwen3:32b' not found")) is True

    def test_real_ollama_response_error_404(self, monkeypatch):
        ollama = pytest.importorskip("ollama", reason="ollama client not installed")
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        err = ollama.ResponseError("model 'x' not found", 404)
        assert _is_ollama_model_missing(err) is True

    def test_wrapped_cause_chain(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        inner = Exception("model 'qwen3:8b' not found")
        outer = RuntimeError("request failed")
        outer.__cause__ = inner
        assert _is_ollama_model_missing(outer) is True

    def test_false_for_cloud_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        assert _is_ollama_model_missing(Exception("model 'x' not found")) is False

    def test_false_for_other_errors(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        assert _is_ollama_model_missing(ValueError("bad json")) is False


class TestLocalLlmHint:
    """_local_llm_hint is the shared decision point for REPL/TUI/engine messages."""

    def test_model_missing_gets_pull_hint(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_MODEL", "qwen3:32b")
        hint = _local_llm_hint(Exception("model 'qwen3:32b' not found"))
        assert hint is not None
        assert "ollama pull qwen3:32b" in hint

    def test_model_missing_uses_provider_default_when_unset(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        hint = _local_llm_hint(Exception("model 'qwen3:8b' not found"))
        assert "ollama pull qwen3:8b" in hint

    def test_connection_refused_gets_serve_hint(self, monkeypatch):
        import yeaboi.ollama_control as ollama_control

        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setattr(ollama_control, "is_ollama_installed", lambda: True)
        hint = _local_llm_hint(ConnectionRefusedError("refused"))
        assert hint is not None
        assert "ollama serve" in hint
        assert "https://ollama.com" not in hint  # installed → don't tell them to install

    def test_connection_refused_when_not_installed_says_install(self, monkeypatch):
        import yeaboi.ollama_control as ollama_control

        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setattr(ollama_control, "is_ollama_installed", lambda: False)
        hint = _local_llm_hint(ConnectionRefusedError("refused"))
        assert hint is not None
        assert "https://ollama.com" in hint
        assert "ollama serve" in hint

    def test_read_timeout_gets_slowness_hint(self, monkeypatch):
        import httpx

        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        hint = _local_llm_hint(httpx.ReadTimeout("timed out"))
        assert hint is not None
        assert "timed out" in hint.lower()
        assert "ollama serve" not in hint  # not misreported as a down server

    def test_none_for_cloud_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        assert _local_llm_hint(ConnectionRefusedError("refused")) is None

    def test_none_for_unrelated_errors(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        assert _local_llm_hint(ValueError("bad json")) is None

    def test_langchain_ollama_missing_gets_install_hint(self, monkeypatch):
        # The optional extra was never installed — the wizard warns, but a
        # hand-edited .env can still reach get_llm()'s ImportError at call time.
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        exc = ImportError("langchain-ollama is not installed. Run: uv sync --extra ollama")
        hint = _local_llm_hint(exc)
        assert hint is not None
        assert "uv sync --extra ollama" in hint

    def test_unrelated_import_error_no_hint(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        assert _local_llm_hint(ImportError("No module named 'pandas'")) is None


class TestTrimHistoryForLocal:
    """Chat history must fit Ollama's context window without splitting tool pairs."""

    def _tokens_of(self, n_chars: int) -> str:
        return "x" * n_chars

    def test_cloud_provider_untouched(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        msgs = [SystemMessage(content="s"), HumanMessage(content=self._tokens_of(500_000))]
        assert _trim_history_for_local(msgs) is msgs

    def test_ollama_fits_untouched(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
        msgs = [SystemMessage(content="s"), HumanMessage(content="hi"), AIMessage(content="hello")]
        assert _trim_history_for_local(msgs) is msgs

    def test_overflow_drops_oldest_at_human_boundary(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        # Budget = 12000 − 8192 − sys ≈ 3800 tokens. Each message below is
        # 1000 tokens (4000 chars): all four (4000) overflow, the last two
        # (2000) fit — so the cut lands on the second HumanMessage.
        monkeypatch.setenv("OLLAMA_NUM_CTX", "12000")
        h1 = HumanMessage(content=self._tokens_of(4000))
        a1 = AIMessage(content=self._tokens_of(4000))
        h2 = HumanMessage(content=self._tokens_of(4000))
        a2 = AIMessage(content=self._tokens_of(4000))
        result = _trim_history_for_local([SystemMessage(content="s"), h1, a1, h2, a2])
        assert result == [result[0], h2, a2]
        assert isinstance(result[0], SystemMessage)

    def test_tool_call_pair_never_split(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_NUM_CTX", "12000")
        h1 = HumanMessage(content=self._tokens_of(4000))
        a1 = AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}])
        t1 = ToolMessage(content=self._tokens_of(4000), tool_call_id="1")
        a1b = AIMessage(content=self._tokens_of(4000))
        h2 = HumanMessage(content=self._tokens_of(4000))
        a2 = AIMessage(content="ok")
        result = _trim_history_for_local([SystemMessage(content="s"), h1, a1, t1, a1b, h2, a2])
        # The whole first exchange (incl. the AI/Tool pair) is dropped together.
        assert result == [result[0], h2, a2]
        assert not any(isinstance(m, ToolMessage) for m in result)

    def test_latest_exchange_kept_even_over_budget(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_NUM_CTX", "9000")  # budget ≈ 800 tokens
        h1 = HumanMessage(content=self._tokens_of(8000))
        a1 = AIMessage(content=self._tokens_of(8000))
        h2 = HumanMessage(content=self._tokens_of(8000))  # alone exceeds budget
        result = _trim_history_for_local([SystemMessage(content="s"), h1, a1, h2])
        assert result == [result[0], h2]

    def test_no_human_boundary_returns_unchanged(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_NUM_CTX", "9000")
        msgs = [SystemMessage(content="s"), AIMessage(content=self._tokens_of(40_000))]
        assert _trim_history_for_local(msgs) is msgs


class TestToolsUnsupportedDegradation:
    """make_call_model must degrade to plain chat when a local model rejects tools."""

    class _PlainLLM:
        def __init__(self):
            self.plain_invokes = 0
            self.bind_calls = 0

        def invoke(self, messages):
            self.plain_invokes += 1
            return AIMessage(content="plain reply")

        def bind_tools(self, tools):
            self.bind_calls += 1

            class _Bound:
                def invoke(self, messages):
                    raise RuntimeError("registry.ollama.ai/library/foo does not support tools")

            return _Bound()

    def test_degrades_to_plain_chat_with_note(self, monkeypatch):
        llm = self._PlainLLM()
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: llm)
        node = make_call_model([])
        result = node({"messages": [HumanMessage(content="hi")]})
        reply = result["messages"][0]
        assert "plain reply" in reply.content
        assert "doesn't support tool calling" in reply.content
        assert llm.plain_invokes == 1

    def test_skips_bound_attempt_on_later_turns(self, monkeypatch):
        llm = self._PlainLLM()
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: llm)
        node = make_call_model([])
        node({"messages": [HumanMessage(content="hi")]})
        result2 = node({"messages": [HumanMessage(content="again")]})
        assert llm.bind_calls == 1  # doomed binding not retried
        assert result2["messages"][0].content == "plain reply"  # note appended only once

    def test_other_errors_still_propagate(self, monkeypatch):
        class _Llm:
            def bind_tools(self, tools):
                class _Bound:
                    def invoke(self, messages):
                        raise RuntimeError("boom")

                return _Bound()

        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: _Llm())
        node = make_call_model([])
        with pytest.raises(RuntimeError, match="boom"):
            node({"messages": [HumanMessage(content="hi")]})


class TestProseInvokesTrackUsage:
    """Prose chat invokes must feed track_usage so local timing/tokens are counted."""

    def test_call_model_tracks(self, monkeypatch):
        calls = []
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="hi")
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: mock_llm)
        monkeypatch.setattr("yeaboi.agent.nodes.track_usage", lambda resp: calls.append(resp))
        call_model({"messages": [HumanMessage(content="hello")]})
        assert len(calls) == 1

    def test_bound_path_tracks(self, monkeypatch):
        calls = []

        class _Llm:
            def bind_tools(self, tools):
                class _Bound:
                    def invoke(self, messages):
                        return AIMessage(content="bound reply")

                return _Bound()

        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: _Llm())
        monkeypatch.setattr("yeaboi.agent.nodes.track_usage", lambda resp: calls.append(resp))
        node = make_call_model([])
        node({"messages": [HumanMessage(content="hi")]})
        assert len(calls) == 1

    def test_degraded_path_tracks(self, monkeypatch):
        calls = []

        class _Llm:
            def invoke(self, messages):
                return AIMessage(content="plain reply")

            def bind_tools(self, tools):
                class _Bound:
                    def invoke(self, messages):
                        raise RuntimeError("model does not support tools")

                return _Bound()

        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda **kw: _Llm())
        monkeypatch.setattr("yeaboi.agent.nodes.track_usage", lambda resp: calls.append(resp))
        node = make_call_model([])
        node({"messages": [HumanMessage(content="hi")]})
        assert len(calls) == 1  # tracked once on the degraded plain invoke, not double


# ── Core behaviour ───────────────────────────────────────────────────


class TestCallModel:
    """Tests for the call_model() node function."""

    def test_returns_dict_with_messages_key(self, monkeypatch):
        """call_model must return a dict containing a 'messages' key."""
        fake_response = AIMessage(content="Hello!")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda: mock_llm)

        state = {"messages": [HumanMessage(content="Hi")]}
        result = call_model(state)
        assert "messages" in result

    def test_returns_single_item_list(self, monkeypatch):
        """The messages value should be a list with exactly one response."""
        fake_response = AIMessage(content="I can help with that.")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda: mock_llm)

        state = {"messages": [HumanMessage(content="Hello")]}
        result = call_model(state)
        assert len(result["messages"]) == 1

    def test_response_is_ai_message(self, monkeypatch):
        """The returned message should be the AIMessage from the LLM."""
        fake_response = AIMessage(content="Sure, let me help.")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda: mock_llm)

        state = {"messages": [HumanMessage(content="Hello")]}
        result = call_model(state)
        assert result["messages"][0] is fake_response
        assert isinstance(result["messages"][0], AIMessage)

    def test_system_prompt_prepended(self, monkeypatch):
        """The system prompt must be the first message sent to the LLM."""
        fake_response = AIMessage(content="Acknowledged.")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda: mock_llm)

        state = {"messages": [HumanMessage(content="Plan my project")]}
        call_model(state)

        # Extract the message list passed to invoke()
        call_args = mock_llm.invoke.call_args[0][0]
        assert isinstance(call_args[0], SystemMessage)
        assert call_args[0].content == get_system_prompt()

    def test_user_messages_forwarded(self, monkeypatch):
        """User messages from state must be passed through to the LLM."""
        fake_response = AIMessage(content="Got it.")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda: mock_llm)

        user_msg = HumanMessage(content="Build a todo app")
        state = {"messages": [user_msg]}
        call_model(state)

        call_args = mock_llm.invoke.call_args[0][0]
        # System prompt first, then the user message
        assert call_args[1] is user_msg

    def test_multiple_messages_forwarded(self, monkeypatch):
        """All messages in state should be forwarded after the system prompt."""
        fake_response = AIMessage(content="Understood.")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda: mock_llm)

        msg1 = HumanMessage(content="Build a todo app")
        msg2 = AIMessage(content="Tell me more about the project.")
        msg3 = HumanMessage(content="It should have CRUD operations")
        state = {"messages": [msg1, msg2, msg3]}
        call_model(state)

        call_args = mock_llm.invoke.call_args[0][0]
        # System prompt + 3 conversation messages = 4 total
        assert len(call_args) == 4
        assert isinstance(call_args[0], SystemMessage)
        assert call_args[1] is msg1
        assert call_args[2] is msg2
        assert call_args[3] is msg3

    def test_system_prompt_not_in_returned_messages(self, monkeypatch):
        """The system prompt is injected for the LLM call but NOT returned in state."""
        fake_response = AIMessage(content="Here's a plan.")
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = fake_response
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda: mock_llm)

        state = {"messages": [HumanMessage(content="Hello")]}
        result = call_model(state)

        # Only the AI response is returned — no SystemMessage
        for msg in result["messages"]:
            assert not isinstance(msg, SystemMessage)


# ── Import tests ─────────────────────────────────────────────────────


class TestCallModelImports:
    """Verify call_model is importable from the expected locations."""

    def test_importable_from_agent_package(self):
        """call_model should be re-exported from yeaboi.agent."""
        from yeaboi.agent import call_model as imported_fn

        assert imported_fn is call_model

    def test_importable_from_nodes_module(self):
        """call_model should be importable directly from yeaboi.agent.nodes."""
        from yeaboi.agent.nodes import call_model as imported_fn

        assert imported_fn is call_model


# ── should_continue routing function ────────────────────────────────


class TestShouldContinue:
    """Tests for the should_continue() conditional edge function."""

    def test_returns_end_when_no_tool_calls(self):
        """Plain AIMessage with no tool_calls should route to END."""
        state = {"messages": [AIMessage(content="Here's your plan.")]}
        assert should_continue(state) == END

    def test_returns_tools_when_tool_calls_present(self):
        """AIMessage with tool_calls should route to 'tools'."""
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "read_codebase", "args": {}, "id": "call_123"}],
        )
        state = {"messages": [msg]}
        assert should_continue(state) == "tools"

    def test_only_inspects_last_message(self):
        """Only the last message matters — earlier messages are ignored."""
        # First message has tool calls, but the last one doesn't
        msg_with_tools = AIMessage(
            content="",
            tool_calls=[{"name": "read_codebase", "args": {}, "id": "call_001"}],
        )
        msg_without_tools = AIMessage(content="Done!")
        state = {"messages": [msg_with_tools, msg_without_tools]}
        assert should_continue(state) == END

    def test_returns_end_when_tool_calls_empty_list(self):
        """An empty tool_calls list should route to END (same as no tool calls)."""
        msg = AIMessage(content="All done.", tool_calls=[])
        state = {"messages": [msg]}
        assert should_continue(state) == END

    def test_multiple_tool_calls_returns_tools(self):
        """Multiple tool calls in one message should still route to 'tools'."""
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "read_codebase", "args": {}, "id": "call_001"},
                {"name": "export_markdown", "args": {"path": "plan.md"}, "id": "call_002"},
            ],
        )
        state = {"messages": [msg]}
        assert should_continue(state) == "tools"

    def test_does_not_mutate_state(self):
        """should_continue is a pure function — it must not modify the state."""
        original_msg = AIMessage(content="Hello!")
        state = {"messages": [HumanMessage(content="Hi"), original_msg]}
        original_messages = list(state["messages"])

        should_continue(state)

        assert state["messages"] == original_messages


# ── should_continue import tests ────────────────────────────────────


class TestShouldContinueImports:
    """Verify should_continue is importable from the expected locations."""

    def test_importable_from_agent_package(self):
        """should_continue should be re-exported from yeaboi.agent."""
        from yeaboi.agent import should_continue as imported_fn

        assert imported_fn is should_continue

    def test_importable_from_nodes_module(self):
        """should_continue should be importable directly from yeaboi.agent.nodes."""
        from yeaboi.agent.nodes import should_continue as imported_fn

        assert imported_fn is should_continue


# ── Risk-level routing ────────────────────────────────────────────────


class TestShouldContinueRiskRouting:
    """Tests for the three-way risk-level routing in should_continue."""

    def test_low_risk_tool_routes_to_tools(self):
        """read_codebase is not high-risk — routes directly to 'tools'."""
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "read_codebase", "args": {}, "id": "call_1", "type": "tool_call"}],
        )
        state = {"messages": [msg]}
        assert should_continue(state) == "tools"

    def test_github_read_routes_to_tools(self):
        """github_read_repo is low-risk — auto-executes."""
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "github_read_repo", "args": {}, "id": "call_1", "type": "tool_call"}],
        )
        state = {"messages": [msg]}
        assert should_continue(state) == "tools"

    def test_jira_create_epic_routes_to_human_review(self):
        """jira_create_epic is high-risk — routes to human_review for confirmation."""
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "jira_create_epic", "args": {"title": "Auth Feature"}, "id": "call_1", "type": "tool_call"}
            ],
        )
        state = {"messages": [msg]}
        assert should_continue(state) == "human_review"

    def test_jira_create_story_routes_to_human_review(self):
        """jira_create_story is high-risk."""
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "jira_create_story", "args": {}, "id": "call_1", "type": "tool_call"}],
        )
        state = {"messages": [msg]}
        assert should_continue(state) == "human_review"

    def test_jira_create_sprint_routes_to_human_review(self):
        """jira_create_sprint is high-risk."""
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "jira_create_sprint", "args": {}, "id": "call_1", "type": "tool_call"}],
        )
        state = {"messages": [msg]}
        assert should_continue(state) == "human_review"

    def test_confluence_create_page_routes_to_human_review(self):
        """confluence_create_page is high-risk."""
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "confluence_create_page", "args": {}, "id": "call_1", "type": "tool_call"}],
        )
        state = {"messages": [msg]}
        assert should_continue(state) == "human_review"

    def test_confluence_update_page_routes_to_human_review(self):
        """confluence_update_page is high-risk."""
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "confluence_update_page", "args": {}, "id": "call_1", "type": "tool_call"}],
        )
        state = {"messages": [msg]}
        assert should_continue(state) == "human_review"

    def test_confirmed_high_risk_routes_to_tools(self):
        """After user confirms, high-risk tool routes to 'tools' (not human_review again)."""
        confirmation_msg = AIMessage(content="I'd like to create a feature. Please confirm — yes/no?")
        user_yes = HumanMessage(content="yes")
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "jira_create_epic", "args": {}, "id": "call_2", "type": "tool_call"}],
        )
        state = {"messages": [confirmation_msg, user_yes, tool_call_msg]}
        assert should_continue(state) == "tools"

    def test_confirmed_with_full_yes_phrase(self):
        """'yes please go ahead' is treated as affirmative."""
        confirmation_msg = AIMessage(content="Confirm?")
        user_yes = HumanMessage(content="yes please go ahead")
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "confluence_create_page", "args": {}, "id": "call_2", "type": "tool_call"}],
        )
        state = {"messages": [confirmation_msg, user_yes, tool_call_msg]}
        assert should_continue(state) == "tools"

    def test_no_confirmation_stays_human_review(self):
        """If user declined, next attempt still routes to human_review."""
        confirmation_msg = AIMessage(content="Confirm?")
        user_no = HumanMessage(content="no, cancel that")
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "jira_create_epic", "args": {}, "id": "call_2", "type": "tool_call"}],
        )
        state = {"messages": [confirmation_msg, user_no, tool_call_msg]}
        assert should_continue(state) == "human_review"

    def test_insufficient_history_stays_human_review(self):
        """Without prior confirmation messages, high-risk routes to human_review."""
        user_msg = HumanMessage(content="Create a feature for auth")
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "jira_create_epic", "args": {}, "id": "call_1", "type": "tool_call"}],
        )
        state = {"messages": [user_msg, tool_call_msg]}
        assert should_continue(state) == "human_review"

    def test_azdevops_create_epic_routes_to_human_review(self):
        """azdevops_create_epic is a write — must pause for confirmation."""
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "azdevops_create_epic", "args": {}, "id": "call_1", "type": "tool_call"}],
        )
        state = {"messages": [msg]}
        assert should_continue(state) == "human_review"

    def test_azdevops_create_story_and_iteration_route_to_human_review(self):
        for name in ("azdevops_create_story", "azdevops_create_iteration"):
            msg = AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": "c1", "type": "tool_call"}])
            assert should_continue({"messages": [msg]}) == "human_review"

    def test_confirmed_azdevops_write_routes_to_tools(self):
        """The confirmation round-trip works for the newly gated Azure DevOps writes too."""
        state = {
            "messages": [
                AIMessage(content="Create the epic in Azure DevOps — confirm?"),
                HumanMessage(content="yes"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "azdevops_create_epic", "args": {}, "id": "c2", "type": "tool_call"}],
                ),
            ]
        }
        assert should_continue(state) == "tools"

    def test_high_risk_tools_constant_contains_expected_names(self):
        """_HIGH_RISK_TOOLS should list all Jira/AzDO/Confluence/Notion write operations."""
        assert "jira_create_epic" in _HIGH_RISK_TOOLS
        assert "jira_create_story" in _HIGH_RISK_TOOLS
        assert "jira_create_sprint" in _HIGH_RISK_TOOLS
        assert "confluence_create_page" in _HIGH_RISK_TOOLS
        assert "confluence_update_page" in _HIGH_RISK_TOOLS
        assert "azdevops_create_epic" in _HIGH_RISK_TOOLS
        assert "azdevops_create_story" in _HIGH_RISK_TOOLS
        assert "azdevops_create_iteration" in _HIGH_RISK_TOOLS
        assert "notion_create_page" in _HIGH_RISK_TOOLS
        assert "notion_update_page" in _HIGH_RISK_TOOLS
        # Read tools must NOT be in the set
        assert "jira_read_board" not in _HIGH_RISK_TOOLS
        assert "confluence_read_page" not in _HIGH_RISK_TOOLS
        assert "read_codebase" not in _HIGH_RISK_TOOLS
        assert "azdevops_read_repo" not in _HIGH_RISK_TOOLS

    def test_high_risk_tools_derived_from_risk_registry(self):
        """The gate set is exactly the WRITE rows of tools/risk.py."""
        from yeaboi.tools.risk import high_risk_tool_names

        assert _HIGH_RISK_TOOLS == high_risk_tool_names()


class TestUserConfirmed:
    """Tests for the _user_confirmed() helper."""

    def test_yes(self):
        assert _user_confirmed("yes")

    def test_y(self):
        assert _user_confirmed("y")

    def test_ok(self):
        assert _user_confirmed("ok")

    def test_yes_with_suffix(self):
        assert _user_confirmed("yes please go ahead")

    def test_no(self):
        assert not _user_confirmed("no")

    def test_cancel(self):
        assert not _user_confirmed("cancel")

    def test_empty(self):
        assert not _user_confirmed("")

    def test_case_insensitive(self):
        assert _user_confirmed("YES")
        assert _user_confirmed("Yes please")

    # Negation veto — a qualified or negated "yes" is NOT consent. The old
    # prefix matcher accepted all of these; each was a real false positive
    # that would have written to an external tracker.
    def test_surely_not_is_not_consent(self):
        assert not _user_confirmed("surely not")

    def test_yes_but_cancel_is_not_consent(self):
        assert not _user_confirmed("yes but actually cancel it")

    def test_y_not_is_not_consent(self):
        assert not _user_confirmed("y not do something else instead")

    def test_hold_on_is_not_consent(self):
        assert not _user_confirmed("ok wait, hold on")

    def test_bare_go_is_not_consent(self):
        # "go" alone is discourse ("go back", "go to the board"), not a write approval.
        assert not _user_confirmed("go back to the plan first")
        assert not _user_confirmed("go to the board")

    def test_dont_is_not_consent(self):
        assert not _user_confirmed("don't")
        assert not _user_confirmed("please don't proceed")

    def test_affirmative_with_elaboration_still_confirms(self):
        assert _user_confirmed("yes please")
        assert _user_confirmed("ok go for it")
        assert _user_confirmed("go ahead")
        assert _user_confirmed("please proceed")
        assert _user_confirmed("yes, create the epic")

    def test_okay_and_punctuation(self):
        assert _user_confirmed("okay!")
        assert _user_confirmed("yes.")

    def test_unrelated_text_is_not_consent(self):
        assert not _user_confirmed("what does this tool do?")
        assert not _user_confirmed("show me the plan first")


# ── make_call_model factory ───────────────────────────────────────────


class TestMakeCallModel:
    """Tests for the make_call_model() factory function."""

    def test_returns_callable(self, monkeypatch):
        """make_call_model must return a callable node function."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda: mock_llm)

        fn = make_call_model([])
        assert callable(fn)

    def test_bind_tools_called_with_tool_list(self, monkeypatch):
        """bind_tools() must be called with the provided tools on first invocation (lazy init)."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda: mock_llm)

        from langchain_core.tools import tool

        @tool
        def fake_tool(x: str) -> str:
            """A fake tool."""
            return x

        fn = make_call_model([fake_tool])
        # bind_tools is lazy — not called at factory time, only on first invocation.
        mock_llm.bind_tools.assert_not_called()
        fn({"messages": [HumanMessage(content="hi")]})
        mock_llm.bind_tools.assert_called_once_with([fake_tool])

    def test_call_model_binds_only_the_allowed_integrations(self, monkeypatch):
        from langchain_core.tools import tool

        @tool
        def jira_read_board(x: str) -> str:
            """Jira."""
            return x

        @tool
        def notion_read_page(x: str) -> str:
            """Notion."""
            return x

        @tool
        def estimate_complexity(x: str) -> str:
            """Local."""
            return x

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="ok")
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda: mock_llm)
        fn = make_call_model([jira_read_board, notion_read_page, estimate_complexity])
        fn({"messages": [HumanMessage(content="hi")], "session_integrations": ["jira"]})
        mock_llm.bind_tools.assert_called_once_with([jira_read_board, estimate_complexity])
        system = mock_llm.invoke.call_args[0][0][0]
        assert "Integrations enabled for this plan: jira" in system.content
        # A second set binds again; the same set is cached.
        fn({"messages": [HumanMessage(content="hi")], "session_integrations": ["jira"]})
        fn({"messages": [HumanMessage(content="hi")]})
        assert mock_llm.bind_tools.call_count == 2
        assert mock_llm.bind_tools.call_args[0][0] == [jira_read_board, notion_read_page, estimate_complexity]

    def test_chat_context_is_attached_then_cleared(self, monkeypatch):
        mock_bound_llm = MagicMock()
        mock_bound_llm.invoke.return_value = AIMessage(content="Ok")
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound_llm
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda: mock_llm)

        fn = make_call_model([])
        state = {"messages": [HumanMessage(content="Plan this")], "chat_context": ["Reference 1 (link): spec"]}
        out = fn(state)
        sent = mock_bound_llm.invoke.call_args[0][0]
        assert sent[-1].content == "Plan this\n\n---\nReferences and files:\nReference 1 (link): spec"
        assert state["messages"][0].content == "Plan this"  # stored history untouched
        assert out["chat_context"] == []

    def test_returned_function_invokes_bound_llm(self, monkeypatch):
        """The returned node function must call the bound LLM (not the raw LLM)."""
        mock_bound_llm = MagicMock()
        fake_response = AIMessage(content="Response with tools")
        mock_bound_llm.invoke.return_value = fake_response

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound_llm
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda: mock_llm)

        fn = make_call_model([])
        state = {"messages": [HumanMessage(content="Hello")]}
        result = fn(state)

        assert result["messages"] == [fake_response]
        mock_bound_llm.invoke.assert_called_once()
        # Raw LLM must NOT be invoked directly
        mock_llm.invoke.assert_not_called()

    def test_returned_function_returns_messages_dict(self, monkeypatch):
        """The node function must return a dict with a 'messages' key."""
        mock_bound_llm = MagicMock()
        mock_bound_llm.invoke.return_value = AIMessage(content="Ok")
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound_llm
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda: mock_llm)

        fn = make_call_model([])
        result = fn({"messages": [HumanMessage(content="Hi")]})
        assert "messages" in result
        assert len(result["messages"]) == 1

    def test_system_prompt_prepended_by_returned_fn(self, monkeypatch):
        """The node function must prepend the system prompt to messages."""
        mock_bound_llm = MagicMock()
        mock_bound_llm.invoke.return_value = AIMessage(content="Ok")
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound_llm
        monkeypatch.setattr("yeaboi.agent.nodes.get_llm", lambda: mock_llm)

        fn = make_call_model([])
        fn({"messages": [HumanMessage(content="Plan this")]})

        call_args = mock_bound_llm.invoke.call_args[0][0]
        assert isinstance(call_args[0], SystemMessage)


# ── human_review node ─────────────────────────────────────────────────


class TestHumanReview:
    """Tests for the human_review() node function."""

    def _make_state(self, tool_name: str, args: dict | None = None, msg_id: str | None = None) -> dict:
        msg = AIMessage(
            id=msg_id,
            content="",
            tool_calls=[{"name": tool_name, "args": args or {}, "id": "call_1", "type": "tool_call"}],
        )
        return {"messages": [msg]}

    def test_returns_messages_dict(self):
        """human_review must return a dict with a 'messages' key."""
        state = self._make_state("jira_create_epic", {"title": "Auth"}, msg_id="id-1")
        result = human_review(state)
        assert "messages" in result

    def test_replacement_has_same_id(self):
        """The replacement message must carry the same ID so add_messages replaces it."""
        state = self._make_state("jira_create_epic", msg_id="my-msg-id")
        result = human_review(state)
        assert result["messages"][0].id == "my-msg-id"

    def test_replacement_has_no_tool_calls(self):
        """The replacement message must NOT have tool_calls (routes to END, not tools)."""
        state = self._make_state("jira_create_epic", msg_id="id-1")
        result = human_review(state)
        replacement = result["messages"][0]
        assert not getattr(replacement, "tool_calls", None)

    def test_content_mentions_tool_name(self):
        """Confirmation text must name the tool the agent wants to call."""
        state = self._make_state("jira_create_sprint", msg_id="id-1")
        result = human_review(state)
        assert "jira_create_sprint" in result["messages"][0].content

    def test_content_includes_arg_values(self):
        """Confirmation text should include the tool's argument values."""
        state = self._make_state("confluence_create_page", {"title": "Sprint 1 Plan"}, msg_id="id-1")
        result = human_review(state)
        assert "Sprint 1 Plan" in result["messages"][0].content

    def test_content_asks_for_confirmation(self):
        """Confirmation text must ask the user to confirm."""
        state = self._make_state("jira_create_epic", msg_id="id-1")
        result = human_review(state)
        content = result["messages"][0].content.lower()
        assert "yes" in content or "confirm" in content

    def test_handles_none_id(self):
        """When the original message has no ID, human_review still returns a message."""
        state = self._make_state("jira_create_epic")  # no msg_id
        result = human_review(state)
        assert len(result["messages"]) == 1
        replacement = result["messages"][0]
        assert isinstance(replacement, AIMessage)


class TestHumanReviewImports:
    """Verify human_review is importable from expected locations."""

    def test_importable_from_agent_package(self):
        from yeaboi.agent import human_review as fn

        assert fn is human_review

    def test_importable_from_nodes_module(self):
        from yeaboi.agent.nodes import human_review as fn

        assert fn is human_review


class TestIsLlmRateLimited:
    """A 429 is told apart from an auth failure because the remedy differs:
    a rate limit clears on its own, a bad key does not."""

    def test_an_anthropic_rate_limit(self):
        import anthropic
        import httpx

        from yeaboi.agent.nodes import _is_llm_rate_limited

        exc = anthropic.RateLimitError(
            "rate limited",
            response=httpx.Response(429, request=httpx.Request("POST", "https://api.anthropic.com")),
            body=None,
        )
        assert _is_llm_rate_limited(exc) is True

    def test_a_bare_429_from_any_provider(self):
        from yeaboi.agent.nodes import _is_llm_rate_limited

        exc = RuntimeError("too many requests")
        exc.status_code = 429  # type: ignore[attr-defined]
        assert _is_llm_rate_limited(exc) is True

    def test_anything_else_is_not(self):
        from yeaboi.agent.nodes import _is_llm_rate_limited

        assert _is_llm_rate_limited(RuntimeError("boom")) is False


class TestMakeToolNode:
    """The tools node refuses a call whose integration the plan did not enable."""

    def _tools(self):
        from langchain_core.tools import tool

        @tool
        def jira_read_board(x: str) -> str:
            """Jira."""
            return f"board {x}"

        @tool
        def notion_read_page(x: str) -> str:
            """Notion."""
            return f"page {x}"

        return [jira_read_board, notion_read_page]

    def _run(self, calls, integrations=None):
        """Run the node inside a one-node graph: ToolNode needs the graph's runtime config."""
        from langgraph.graph import START, StateGraph

        from yeaboi.agent.nodes import make_tool_node
        from yeaboi.agent.state import ScrumState

        graph = StateGraph(ScrumState)
        graph.add_node("tools", make_tool_node(self._tools()))
        graph.add_edge(START, "tools")
        graph.add_edge("tools", END)
        state = {"messages": [HumanMessage(content="go"), AIMessage(content="", tool_calls=calls)]}
        if integrations is not None:
            state["session_integrations"] = integrations
        out = graph.compile().invoke(state)
        return [m for m in out["messages"] if isinstance(m, ToolMessage)]

    def test_absent_key_runs_everything(self):
        calls = [{"name": "notion_read_page", "args": {"x": "1"}, "id": "c1"}]
        assert [m.content for m in self._run(calls)] == ["page 1"]

    def test_an_excluded_tool_call_is_refused_and_the_rest_run(self):
        calls = [
            {"name": "notion_read_page", "args": {"x": "1"}, "id": "c1"},
            {"name": "jira_read_board", "args": {"x": "2"}, "id": "c2"},
        ]
        by_id = {m.tool_call_id: m for m in self._run(calls, integrations=["jira"])}
        assert by_id["c1"].content == "notion is not enabled for this plan"
        assert by_id["c2"].content == "board 2"

    def test_an_empty_list_refuses_every_integration(self):
        calls = [{"name": "jira_read_board", "args": {"x": "2"}, "id": "c2"}]
        assert self._run(calls, integrations=[])[0].content == "jira is not enabled for this plan"
