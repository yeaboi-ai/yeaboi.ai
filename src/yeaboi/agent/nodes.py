"""LangGraph node functions for the Scrum Agent.

# See docs: "Agentic Blueprint Reference" — two core nodes (call_model + tool_node)
# See docs: "The ReAct Loop" — Thought → Action → Observation pattern

Node functions are the building blocks of a LangGraph graph. Each node is a
plain Python function that takes the current state and returns a partial state
update. LangGraph merges the returned dict into the existing state using the
reducers defined on the state schema (e.g. add_messages for the messages list).

This file is kept separate from graph wiring (Step 6) so that node functions
remain unit-testable — they are pure functions of state, with no dependency
on the graph object itself.
"""

import dataclasses
import json
import logging
import math
import os
import re
from collections.abc import Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END

from yeaboi.agent.ceremony_history import gather_ceremony_context
from yeaboi.agent.llm import get_llm, invoke_json, strip_think_tags, track_usage
from yeaboi.agent.repo_signals import analyze_context, scan_repo_signals
from yeaboi.agent.state import (
    DOD_ITEMS,
    PHASE_QUESTION_RANGES,
    TOTAL_QUESTIONS,
    AcceptanceCriterion,
    ArchitectureDecision,
    ArchitectureOption,
    Discipline,
    Feature,
    Priority,
    ProjectAnalysis,
    PromptQualityRating,
    QuestionnairePhase,
    QuestionnaireState,
    ReviewDecision,
    ScrumState,
    Sprint,
    StoryPointValue,
    Task,
    TaskLabel,
    UserStory,
)
from yeaboi.prompts.analyzer import get_analyzer_prompt
from yeaboi.prompts.feature_generator import get_feature_generator_prompt
from yeaboi.prompts.intake import (
    ADAPTIVE_QUESTION_TEMPLATES,
    CONDITIONAL_ESSENTIALS,
    ESSENTIAL_QUESTIONS,
    FOLLOW_UP_TEMPLATES,
    INTAKE_QUESTIONS,
    PHASE_INTROS,
    PHASE_LABELS,
    Q2_CONSTRAINT_HINTS,
    Q2_INFERENCE_KEYWORDS,
    Q2_TO_Q15_MAP,
    Q12_SERVICE_KEYWORDS,
    Q13_INFRA_KEYWORDS,
    QUESTION_DEFAULTS,
    QUESTION_IMPROVEMENT_HINTS,
    QUESTION_METADATA,
    QUESTION_SHORT_LABELS,
    QUICK_ESSENTIALS,
    QUICK_FALLBACK_DEFAULTS,
    SCRUM_MD_HINT,
    SMALL_PROJECT_ESSENTIALS,
    SMART_ESSENTIALS,
    SOLO_ROLES_DEFAULT,
    AnswerSource,
    ValidationWarning,
    is_choice_question,
)
from yeaboi.prompts.sprint_planner import get_sprint_planner_prompt
from yeaboi.prompts.story_writer import (
    MAX_STORIES_PER_FEATURE,
    MIN_STORIES_PER_FEATURE,
    SMALL_PROJECT_MAX_STORIES,
    get_story_writer_prompt,
)
from yeaboi.prompts.system import get_system_prompt  # noqa: E402 — direct submodule imports avoid circular import
from yeaboi.prompts.task_decomposer import get_task_decomposer_prompt
from yeaboi.timeparse import parse_date, parse_datetime
from yeaboi.tools import detect_platform
from yeaboi.tools.risk import allowed_tools, high_risk_tool_names, integration_of, tool_allowed

logger = logging.getLogger(__name__)

# The first line of the write-tool confirmation human_review asks for; the
# chat surfaces route on it, so it lives in one place.
TOOL_CONFIRM_PREFIX = "I'd like to perform the following write operation(s):"


def _is_llm_rate_limited(exc: Exception) -> bool:
    """Check whether an exception is the provider saying "too many requests".

    Separate from the auth check because the remedy is: a rate limit clears on
    its own, so the user is told to wait rather than to go and look at their
    credentials.
    """
    for module, attr in (("anthropic", "RateLimitError"), ("openai", "RateLimitError")):
        try:
            provider = __import__(module)
        except ImportError:
            continue
        error_class = getattr(provider, attr, None)
        if error_class is not None and isinstance(exc, error_class):
            return True
    # Anything else that carries a 429, including providers with no class of
    # their own for it.
    return getattr(exc, "status_code", None) == 429


def _is_llm_auth_or_billing_error(exc: Exception) -> bool:
    """Check whether an exception is an LLM authentication or billing error.

    These errors should NOT be silently swallowed — the user needs to know
    their API key is invalid or their balance is too low, otherwise every
    LLM feature silently degrades and the app appears broken.

    Covers Anthropic, OpenAI, and Google provider error classes.

    Every mode's engine routes its LLM failures through here, which makes this the
    one place that learns a subscription token has stopped working — so a positive
    answer also records that (in memory only; see :mod:`yeaboi.auth_state`). The
    alternative was the same three lines in eleven engines.
    """
    if not _auth_or_billing_error(exc):
        return False
    from yeaboi.auth_state import note_auth_failure

    note_auth_failure(exc)
    return True


def _auth_or_billing_error(exc: Exception) -> bool:
    """The pure classification behind :func:`_is_llm_auth_or_billing_error`."""
    # Anthropic: AuthenticationError, PermissionDeniedError, BadRequestError (billing)
    try:
        import anthropic

        if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            return True
        if isinstance(exc, anthropic.BadRequestError) and "credit balance" in str(exc).lower():
            return True
    except ImportError:
        pass

    # OpenAI: AuthenticationError (401), PermissionDeniedError (403)
    try:
        import openai

        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return True
    except ImportError:
        pass

    # Google: catch by message pattern (no typed billing error class)
    msg = str(exc).lower()
    if "api key" in msg and ("invalid" in msg or "not valid" in msg):
        return True
    if "quota" in msg and "exceeded" in msg:
        return True

    return False


def _is_llm_connection_error(exc: Exception) -> bool:
    """Check whether an exception means the LLM server could not be reached.

    Matters for the local Ollama provider: a down server would otherwise be
    swallowed by the deterministic-fallback net and silently degrade every
    artifact. Guard sites re-raise these (only when provider == "ollama") so
    the user gets an actionable "is Ollama running?" message instead.

    Walks the __cause__/__context__ chain because langchain and the ollama
    client wrap the underlying httpx transport error.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        # Builtin connection errors (ConnectionRefusedError, ConnectionResetError, ...)
        if isinstance(current, ConnectionError):
            return True
        # httpx transport errors — the ollama client and langchain use httpx
        try:
            import httpx

            if isinstance(current, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)):
                return True
        except ImportError:
            pass
        msg = str(current).lower()
        # "[Errno 61]" = macOS connection refused, "[Errno 111]" = Linux
        if "connection refused" in msg or "[errno 61]" in msg or "[errno 111]" in msg:
            return True
        if "failed to connect" in msg and "ollama" in msg:
            return True
        current = current.__cause__ or current.__context__
    return False


def _ollama_connection_hint() -> str:
    """Actionable message for a down/unreachable local Ollama server.

    Mirrors the setup-screen copy (_ollama_unreachable_message): distinguish
    "not installed" from "installed but not running" so the hint matches reality.
    """
    from yeaboi.config import get_ollama_base_url
    from yeaboi.ollama_control import is_ollama_installed

    base = get_ollama_base_url()
    if is_ollama_installed():
        return f"Can't reach Ollama at {base}. Start it with: ollama serve — or check OLLAMA_BASE_URL in your .env."
    return (
        f"Can't reach Ollama at {base}. Ollama isn't installed — get it at https://ollama.com "
        "(or: brew install ollama), then start it with: ollama serve."
    )


def _is_local_llm_down(exc: Exception) -> bool:
    """True when the active provider is the local Ollama server and it's unreachable.

    Cloud transient network blips deliberately return False — they keep today's
    graceful-fallback behaviour. Used by the mode engines to swap their generic
    "request failed" warning for the actionable Ollama hint.
    """
    if not _is_llm_connection_error(exc):
        return False
    from yeaboi.config import get_llm_provider

    return get_llm_provider() == "ollama"


def _exception_chain(exc: Exception):
    """Yield exc and every exception in its __cause__/__context__ chain (cycle-safe)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_ollama_model_missing(exc: Exception) -> bool:
    """True when the active provider is Ollama and the configured model isn't pulled.

    The Ollama server answers 404 ``model '…' not found`` (ollama.ResponseError)
    for a model that was never pulled or has been deleted. That's neither a
    connection error nor an auth error, so without this check it would silently
    degrade every artifact to its deterministic fallback — exactly what the
    reliability layer exists to prevent.
    """
    from yeaboi.config import get_llm_provider

    if get_llm_provider() != "ollama":
        return False
    for current in _exception_chain(exc):
        try:
            import ollama

            if isinstance(current, ollama.ResponseError) and getattr(current, "status_code", None) == 404:
                return True
        except ImportError:
            pass
        msg = str(current).lower()
        if "model" in msg and "not found" in msg:
            return True
    return False


def _is_langchain_ollama_missing(exc: Exception) -> bool:
    """True when the langchain-ollama package itself isn't installed.

    It's an optional extra — a user can point LLM_PROVIDER=ollama at a working
    server without ever installing the python package, and the first real call
    would raise ImportError from get_llm(). Detected here so it surfaces as an
    actionable install hint instead of a raw traceback or silent fallback.
    """
    for current in _exception_chain(exc):
        if isinstance(current, ImportError) and "langchain" in str(current).lower():
            return True
    return False


def _is_llm_read_timeout(exc: Exception) -> bool:
    """True when the failure is a read timeout (server reachable but too slow)."""
    try:
        import httpx
    except ImportError:
        return False
    return any(isinstance(current, httpx.ReadTimeout) for current in _exception_chain(exc))


def _local_llm_hint(exc: Exception) -> str | None:
    """Actionable message for a local-Ollama failure, or None for anything else.

    The single decision point the REPL, TUI, and mode engines all share. Four
    cases (package missing, model missing, timeout, server down), checked
    most-specific first; cloud-provider errors always return None so their
    graceful-fallback behaviour is untouched.
    """
    from yeaboi.config import get_llm_provider

    if get_llm_provider() != "ollama":
        return None
    if _is_langchain_ollama_missing(exc):
        return (
            "Ollama support isn't installed — run: uv sync --extra ollama "
            "(or: pip install langchain-ollama), then retry."
        )
    if _is_ollama_model_missing(exc):
        from yeaboi.agent.llm import _PROVIDER_DEFAULTS
        from yeaboi.config import get_llm_model

        model = get_llm_model() or _PROVIDER_DEFAULTS["ollama"]
        return f"Model '{model}' is not installed on the Ollama server. Pull it with: ollama pull {model}"
    if _is_llm_read_timeout(exc):
        return (
            "Ollama timed out — the model may be too large or slow for this machine. "
            "Try a smaller model (e.g. qwen3:8b) or a shorter project description."
        )
    if _is_llm_connection_error(exc):
        return _ollama_connection_hint()
    return None


def _should_reraise_llm_error(exc: Exception) -> bool:
    """Decide whether an LLM failure must surface to the user instead of falling back.

    Cases that re-raise:
    - Auth/billing errors on any provider (the user must fix credentials).
    - Local-Ollama failures with an actionable fix — server down, model not
      pulled, or timeout — which would otherwise silently degrade every
      artifact to its deterministic fallback. Cloud transient network blips
      keep today's graceful-fallback behaviour (the REPL already retries
      rate limits).
    """
    return _is_llm_auth_or_billing_error(exc) or _local_llm_hint(exc) is not None


def _invoke_json(prompt: str, *, image_paths=None):
    """JSON-mode LLM invoke for this module's generation nodes.

    Wraps agent/llm.py::invoke_json (constrained JSON decoding on Ollama + a
    one-shot "fix your JSON" re-ask + track_usage) while routing model creation
    through THIS module's ``get_llm`` reference — the lambda resolves it at call
    time, so tests that patch ``yeaboi.agent.nodes.get_llm`` keep working.
    # See docs: "Local Mode (Ollama)" — reliability layer.
    """
    return invoke_json(prompt, image_paths=image_paths, get_llm_fn=lambda **kw: get_llm(**kw))


# ---------------------------------------------------------------------------
# High-risk tool constants
# ---------------------------------------------------------------------------

# Write operations that modify external systems (Jira, Azure DevOps, Confluence,
# Notion). They require explicit user confirmation before execution — the agent
# must ask and receive a "yes" before the graph routes to the ToolNode. The
# source of truth is the per-tool registry in tools/risk.py, where every
# registered tool is classified READ or WRITE (enforced two-way by
# tests/unit/tools/test_risk.py, so a new tool cannot ship ungated silently).
# See docs: "Guardrails" — human-in-the-loop pattern (Tool layer)
_HIGH_RISK_TOOLS: frozenset[str] = high_risk_tool_names()

# Words that signal the user is declining, hesitating, or qualifying —
# checked before any affirmative so "surely not" / "yes but cancel it" never
# count as consent. Erring toward not-confirmed is safe: the agent just asks
# again; a false positive would write to an external tracker.
_NEGATION_RE = re.compile(r"\b(no|not|don'?t|never|stop|cancel|wait|hold)\b")

# Bare "go" is deliberately NOT here — "go back to the plan" / "go to the board"
# are not consent to a write. It only confirms as the phrase "go ahead" below.
_AFFIRM_TOKENS = {"yes", "y", "ok", "okay", "sure", "confirm", "proceed", "yep", "yup"}
_AFFIRM_PHRASES = ("go ahead", "please proceed", "please go ahead")


def _user_confirmed(text: str) -> bool:
    """Return True if the text is an unqualified affirmative confirmation.

    Used by should_continue to detect whether the user approved a high-risk
    tool call in the preceding HumanMessage. Negations veto first (so
    "surely not" or "yes but actually cancel" never confirm); then the first
    token (punctuation-stripped) must be an affirmative, or the message must
    start with an exact affirmative phrase.
    """
    lowered = text.strip().lower()
    if not lowered or _NEGATION_RE.search(lowered):
        return False
    if lowered.startswith(_AFFIRM_PHRASES):
        return True
    first_token = lowered.split()[0].strip(".,!?:;")
    return first_token in _AFFIRM_TOKENS


def _attach_chat_images(state: ScrumState, all_messages: list[BaseMessage]) -> list[BaseMessage]:
    """Rebuild the latest human message with pasted screenshots at invoke time.

    state["messages"] must stay text-only — many nodes string-op on .content —
    so Ctrl+V screenshots ride in state["chat_images"] (file paths) and are
    converted to multimodal content blocks only in the message list handed to
    the LLM, never in the stored history. See agent/llm.py:build_multimodal_content.
    """
    images = list(state.get("chat_images") or [])
    if not images:
        return all_messages
    from yeaboi.agent.llm import build_multimodal_content

    for i in range(len(all_messages) - 1, -1, -1):
        msg = all_messages[i]
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            content = build_multimodal_content(msg.content, images)
            if isinstance(content, list):
                logger.info("chat: attaching %d pasted image(s) to the latest user message", len(images))
                return [*all_messages[:i], HumanMessage(content=content), *all_messages[i + 1 :]]
            break  # images unreadable — degrade to text-only
    return all_messages


def _attach_chat_context(state: ScrumState, all_messages: list[BaseMessage]) -> list[BaseMessage]:
    """Append the turn's rendered refs and files to the latest human message, at invoke time only.

    Same rule as the images: state["messages"] keeps the person's own words,
    state["chat_context"] carries the text agent/chat_refs.py rendered, and the
    two meet only in the list handed to the LLM. Runs before the images so the
    content is still a string here.
    """
    block = list(state.get("chat_context") or [])
    if not block:
        return all_messages
    for i in range(len(all_messages) - 1, -1, -1):
        msg = all_messages[i]
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            content = msg.content + "\n\n---\nReferences and files:\n" + "\n\n".join(block)
            logger.info("chat: attaching %d reference/file paragraph(s) to the latest user message", len(block))
            return [
                *all_messages[:i],
                HumanMessage(content=content, additional_kwargs=dict(msg.additional_kwargs)),
                *all_messages[i + 1 :],
            ]
    return all_messages


def _trim_history_for_local(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Trim chat history to fit a local model's context window. No-op for cloud.

    # See docs: "Local Mode (Ollama)" — context window reliability
    Cloud windows are 200k+ tokens, but Ollama silently LEFT-truncates anything
    beyond num_ctx − num_predict — which eats the *system prompt* first, the
    worst possible loss. Instead, drop the OLDEST turns explicitly: keep the
    head message (the system prompt) pinned, walk the rest accumulating a rough
    token estimate, and cut only at HumanMessage boundaries so an
    AIMessage(tool_calls) is never separated from its ToolMessage results
    (a split pair is an API error on every provider). The latest exchange is
    always kept, even if it alone exceeds the budget.
    """
    from yeaboi.config import get_llm_provider

    if get_llm_provider() != "ollama" or len(messages) < 2:
        return messages

    from langchain_core.messages import HumanMessage

    from yeaboi.agent.llm import _OLLAMA_NUM_PREDICT, estimate_tokens
    from yeaboi.config import get_ollama_num_ctx

    def _msg_tokens(msg: BaseMessage) -> int:
        content = getattr(msg, "content", "")
        return estimate_tokens(content if isinstance(content, str) else str(content))

    system, history = messages[0], messages[1:]
    budget = get_ollama_num_ctx() - _OLLAMA_NUM_PREDICT - _msg_tokens(system)
    sizes = [_msg_tokens(m) for m in history]
    remaining = sum(sizes)  # at index i this holds sum(sizes[i:])
    if remaining <= budget:
        return messages

    cut = None  # first HumanMessage index whose tail fits the budget
    last_human = None
    for i, msg in enumerate(history):
        if isinstance(msg, HumanMessage):
            last_human = i
            if remaining <= budget:
                cut = i
                break
        remaining -= sizes[i]
    if cut is None:
        cut = last_human  # over budget even at the latest exchange — keep it anyway
    if not cut:  # None (no human message to cut at) or 0 (nothing to drop)
        return messages

    logger.info(
        "local context trim: dropping %d oldest chat messages to fit OLLAMA_NUM_CTX (budget ~%d tokens)",
        cut,
        budget,
    )
    return [system, *history[cut:]]


def make_call_model(tools: list[BaseTool]) -> Callable[[ScrumState], dict[str, list[BaseMessage]]]:
    """Return a call_model node function with the given tools bound to the LLM.

    # See docs: "Agentic Blueprint Reference" — bind_tools wires tools into the LLM
    # See docs: "Tools" — tool types, @tool decorator
    #
    # bind_tools() is a LangChain method on ChatModels. It takes a list of
    # tool definitions (functions decorated with @tool) and returns a new Runnable
    # that includes those tool schemas in every API call to Claude.
    #
    # Why does this matter?
    # Without bind_tools(), the LLM has NO knowledge of available tools. It can
    # never produce tool_calls in its response — so should_continue always routes
    # to END and the ReAct loop never invokes any tools.
    #
    # The binding is LAZY — it happens on the first invocation of the returned
    # closure, not at factory / graph-creation time. This is important for
    # testability: create_graph() can compile the graph structure in CI without
    # an ANTHROPIC_API_KEY; the key is only required when the node is actually
    # called (i.e. during a real agent run or in tests that mock get_llm).
    #
    # Why a factory function and not a class?
    # A simple closure is the idiomatic LangGraph pattern for parameterised nodes.
    # The graph wires it as: graph.add_node("agent", make_call_model(tools))
    """
    # One binding per allowed-integration set: a plan restricted to some
    # integrations never sends the other tools' schemas to the model.
    _bound: dict = {}
    _tools_unsupported = False  # set once a local model rejects tool schemas

    def _is_tools_unsupported_error(exc: Exception) -> bool:
        """True when the (local) model rejected the request because of tool schemas."""
        return any("does not support tools" in str(current).lower() for current in _exception_chain(exc))

    def call_model_with_tools(state: ScrumState) -> dict[str, list[BaseMessage]]:
        """LangGraph node: invoke the LLM with bound tools."""
        nonlocal _tools_unsupported
        allowed = state.get("session_integrations")
        cache_key = None if allowed is None else frozenset(allowed)
        if cache_key not in _bound and not _tools_unsupported:
            # bind_tools() returns a new Runnable (RunnableBinding) that wraps
            # the LLM and injects the tool schemas into every API request.
            # Claude reads these schemas to know what tools are available and
            # generates tool_calls when it wants to use one.
            _bound[cache_key] = get_llm().bind_tools(allowed_tools(tools, allowed))
        system_message = SystemMessage(content=get_system_prompt(integrations=allowed))
        # Trim BEFORE attaching images: the estimate is text-based, and the
        # latest human message (where images attach) is always kept.
        all_messages = _attach_chat_images(
            state, _attach_chat_context(state, _trim_history_for_local([system_message, *state["messages"]]))
        )
        if _tools_unsupported:
            response = get_llm().invoke(all_messages)
        else:
            try:
                response = _bound[cache_key].invoke(all_messages)
            except Exception as exc:
                # Some local models can't call tools — Ollama rejects the whole
                # request ("… does not support tools"). Degrade to plain chat
                # (once discovered, skip the doomed bound attempt on later turns)
                # instead of surfacing "Unexpected error". Cloud models always
                # support tools, so any other exception propagates unchanged.
                if not _is_tools_unsupported_error(exc):
                    raise
                logger.warning("model does not support tool calling — degrading to plain chat: %s", exc)
                _tools_unsupported = True
                _bound.clear()
                response = get_llm().invoke(all_messages)
                track_usage(response)  # prose chat path — count tokens + local timing
                _strip_response_think_tags(response)
                if isinstance(response.content, str):
                    response.content += (
                        "\n\n_Note: this local model doesn't support tool calling, so actions like "
                        "Jira/Confluence sync are unavailable in chat. Pick a tool-capable model "
                        "(e.g. qwen3:8b) in Setup to enable them._"
                    )
                out_deg: dict = {"messages": [response]}
                if state.get("chat_images"):
                    out_deg["chat_images"] = []
                if state.get("chat_context"):
                    out_deg["chat_context"] = []
                return out_deg
        track_usage(response)  # covers both the tools-unsupported and bound-LLM invokes above
        _strip_response_think_tags(response)
        out: dict = {"messages": [response]}
        if state.get("chat_images"):
            out["chat_images"] = []  # consumed — clear so later turns don't re-send
        if state.get("chat_context"):
            out["chat_context"] = []
        return out

    return call_model_with_tools


def make_tool_node(tools: list[BaseTool]):
    """The "tools" node: LangGraph's ToolNode behind a per-plan integration guard.

    A plan restricted to some integrations never sees the other tools' schemas
    (make_call_model), but a model can still name one; such a call is answered
    with a refusing ToolMessage and the allowed calls run as usual.
    # See docs: "Tools" — tool types and ToolNode
    """
    from langgraph.prebuilt import ToolNode

    node = ToolNode(list(tools), handle_tool_errors=True)

    def guarded_tools(state: ScrumState, config: RunnableConfig) -> dict:
        allowed = state.get("session_integrations")
        messages = state["messages"]
        last = messages[-1] if messages else None
        calls = list(getattr(last, "tool_calls", None) or [])
        refused = [call for call in calls if not tool_allowed(call["name"], allowed)]
        if not refused:
            return node.invoke(state, config)
        out: list[BaseMessage] = []
        for call in refused:
            logger.warning(
                "tools: refused %s — %s is not enabled for this plan", call["name"], integration_of(call["name"])
            )
            out.append(
                ToolMessage(
                    content=f"{integration_of(call['name'])} is not enabled for this plan",
                    tool_call_id=call.get("id", ""),
                    name=call["name"],
                )
            )
        kept = [call for call in calls if tool_allowed(call["name"], allowed)]
        if kept:
            trimmed = AIMessage(content=last.content, tool_calls=kept, id=getattr(last, "id", None))
            result = node.invoke({**state, "messages": [*messages[:-1], trimmed]}, config)
            out.extend(result.get("messages", []))
        return {"messages": out}

    return guarded_tools


def _strip_response_think_tags(response) -> None:
    """Strip <think> blocks from a chat response's prose content, in place.

    Local think-by-default models (qwen3) embed reasoning in .content — see
    agent/llm.py:strip_think_tags. Mutates the message (tool_calls untouched)
    so callers keep returning the original AIMessage object.
    """
    from yeaboi.agent.llm import strip_think_tags

    content = getattr(response, "content", None)
    if isinstance(content, str) and "<think>" in content:
        response.content = strip_think_tags(content)


def call_model(state: ScrumState) -> dict[str, list[BaseMessage]]:
    """LangGraph node: invoke the LLM with the current conversation.

    # See docs: "Agentic Blueprint Reference" — this is the "agent" node
    # See docs: "The ReAct Loop" — this node is the "Thought" step
    #
    # How this works:
    # 1. Build a SystemMessage from the Scrum Master prompt — this sets the
    #    LLM's persona and constraints for every call.
    # 2. Prepend it to the existing conversation messages from state.
    # 3. Call the LLM with the full message list.
    # 4. Return {"messages": [response]} — LangGraph's add_messages reducer
    #    will APPEND this to the existing state["messages"], not replace it.
    #
    # Why the SystemMessage is injected here (not stored in state):
    # - System messages are infrastructure, not conversation history —
    #   they shouldn't pollute the user/assistant message list.
    # - Different nodes can inject different system context later
    #   (e.g. a story_writer node could add DoD rules to its system prompt).
    # - Keeps state clean: messages only contains user/assistant turns.
    #
    # Why invoke() and not stream():
    # invoke() returns the complete response in one call. Streaming is handled
    # at the REPL layer (repl.py) by iterating over graph.stream(), not here.
    # The node's job is to produce the response; the UI decides how to display it.

    Args:
        state: The current LangGraph state containing the conversation messages.

    Returns:
        A dict with a single "messages" key containing the LLM's response
        in a list. The add_messages reducer on ScrumState will append this
        to the existing conversation history.
    """
    system_message = SystemMessage(content=get_system_prompt(integrations=state.get("session_integrations")))

    # Prepend system prompt to conversation history for each call.
    # The system message is NOT stored in state — it's injected fresh
    # each invocation so different nodes can use different system context.
    # Pasted screenshots (chat_images) are attached to the latest human
    # message here, at invoke time only — stored history stays text-only.
    # Local models get their history trimmed to the context window first.
    all_messages = _attach_chat_images(
        state, _attach_chat_context(state, _trim_history_for_local([system_message, *state["messages"]]))
    )

    response = get_llm().invoke(all_messages)
    track_usage(response)  # prose chat path — count tokens + local timing
    _strip_response_think_tags(response)

    # Return single-item list — the add_messages reducer on ScrumState["messages"]
    # will append this response to the existing conversation history.
    # See docs: "Agentic Blueprint Reference" — node return format
    out: dict = {"messages": [response]}
    if state.get("chat_images"):
        out["chat_images"] = []  # consumed — clear so later turns don't re-send
    if state.get("chat_context"):
        out["chat_context"] = []
    return out


def should_continue(state: ScrumState) -> str:
    """Route after call_model: continue to tools, request human review, or end.

    # See docs: "Agentic Blueprint Reference" — conditional edges
    # See docs: "The ReAct Loop" — this is the decision point
    # See docs: "Guardrails" — human-in-the-loop pattern
    #
    # Three-way routing:
    #   no tool_calls        → END             (LLM is done; return response)
    #   low/medium-risk      → "tools"         (auto-execute via ToolNode)
    #   high-risk            → "human_review"  (pause for user confirmation)
    #
    # High-risk tools are external write operations (Jira, Azure DevOps,
    # Confluence, Notion) that create or modify records and cannot be easily
    # undone. _HIGH_RISK_TOOLS lists them (the WRITE rows of tools/risk.py).
    #
    # Confirmation detection: if the immediately preceding messages show that
    # the user already confirmed (human_review asked → user said "yes"), route
    # directly to "tools" on the second trip through this function.
    # Pattern: [..., AIMessage(confirmation request), HumanMessage("yes"), AIMessage(tool_calls)]
    """
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return END

    # Check if any requested tool is high-risk.
    tool_names = {tc["name"] for tc in last_message.tool_calls}
    if not (tool_names & _HIGH_RISK_TOOLS):
        # All tools are low/medium risk — auto-execute.
        return "tools"

    # High-risk tool detected. Check for a prior confirmation in message history.
    # After human_review runs, the conversation has:
    #   messages[-3] = AIMessage(confirmation request, no tool_calls)
    #   messages[-2] = HumanMessage("yes")
    #   messages[-1] = AIMessage(new tool_calls)  ← current
    messages = state["messages"]
    if len(messages) >= 3:
        prev_ai = messages[-3]
        prev_human = messages[-2]
        if (
            isinstance(prev_ai, AIMessage)
            and not getattr(prev_ai, "tool_calls", None)  # Was a confirmation request
            and isinstance(prev_human, HumanMessage)
            and _user_confirmed(prev_human.content)
        ):
            logger.info("high-risk tool confirmed by user: %s", sorted(tool_names & _HIGH_RISK_TOOLS))
            return "tools"

    logger.info("high-risk tool gated for confirmation: %s", sorted(tool_names & _HIGH_RISK_TOOLS))
    return "human_review"


def human_review(state: ScrumState) -> dict[str, list[BaseMessage]]:
    """LangGraph node: pause before high-risk write operations for user confirmation.

    # See docs: "Guardrails" — human-in-the-loop pattern
    #
    # This node is reached when should_continue detects a high-risk tool call
    # (Jira/Confluence write). It replaces the tool_calls AIMessage with a
    # plain-text confirmation request so the graph routes to END and the REPL
    # shows the user what the agent wants to do.
    #
    # The ID trick: add_messages (the LangGraph reducer) replaces messages that
    # share the same ID rather than appending a new one. By returning a new
    # AIMessage with the same ID as the tool_calls message, we replace it in
    # state — this prevents two consecutive AIMessages, which would violate
    # Anthropic's alternating-message rule. In the rare case where the original
    # message has no ID (some test scenarios), langchain-anthropic merges
    # consecutive same-role messages before the API call, so it's still safe.
    #
    # Flow after this node:
    #   → END → REPL shows confirmation text
    #   → user types "yes" → call_model re-generates the tool call
    #   → should_continue detects prior confirmation → "tools" (auto-executes)
    """
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    lines = [TOOL_CONFIRM_PREFIX + "\n"]
    for tc in tool_calls:
        lines.append(f"  \u2022 **{tc['name']}**")
        for k, v in tc["args"].items():
            lines.append(f"    - {k}: {v!r}")
    lines.append("\nPlease confirm \u2014 type **yes** to proceed or **no** to cancel.")

    # Replace the tool_calls AIMessage with a plain confirmation request.
    # Same ID → add_messages replaces; None ID → add_messages appends (safe fallback).
    # See docs: "Guardrails" — human-in-the-loop pattern
    replacement = AIMessage(id=last_message.id, content="\n".join(lines))
    return {"messages": [replacement]}


# ── Intake questionnaire ─────────────────────────────────────────────

# ── Adaptive skip helpers ────────────────────────────────────────────
# See docs: "Project Intake Questionnaire" — adaptive skip logic
#
# When the user's initial description (typed at scrum> before Q1) already
# answers some intake questions, these helpers extract those answers and
# skip the corresponding questions. This avoids asking the user to repeat
# information they already provided.
#
# Why LLM-powered analysis (not keyword matching)?
# Natural language like "3 engineers working on React" answers Q6 (team size)
# and Q11 (tech stack), but no keyword pattern would reliably catch Q6. The
# LLM understands context and maps descriptions to specific questions. The
# LLM call only happens once (on the first project_intake call) so the
# latency cost is minimal.


def _extract_answers_from_description(description: str) -> dict[int, str]:
    """Use the LLM to extract answers to intake questions from a project description.

    Sends the description + all 30 questions to the LLM with a structured prompt
    asking for a JSON object mapping question numbers to extracted answers.

    Args:
        description: The user's initial project description.

    Returns:
        A dict mapping question numbers (1–30) to extracted answer strings.
        Returns {} on empty description, bad JSON, or any exception.
    """
    if not description or not description.strip():
        return {}

    # Build the question list for the prompt
    question_list = "\n".join(f"Q{num}: {text}" for num, text in INTAKE_QUESTIONS.items())

    prompt = (
        "You are analyzing a project description to extract answers to intake questions.\n\n"
        f'Project description:\n"{description}"\n\n'
        f"Questions:\n{question_list}\n\n"
        "For each question that the description EXPLICITLY and CLEARLY answers, return a JSON object "
        "mapping the question number (as a string key) to the extracted answer.\n\n"
        "STRICT RULES:\n"
        "- Only include questions where the user DIRECTLY stated the answer.\n"
        "- Q2 (project type): INFER from strong signals like 'refactor', 'migrate', 'legacy' → Existing codebase; "
        "'from scratch', 'new project', 'greenfield' → Greenfield.\n"
        "- Q12 (integrations): EXTRACT named services (e.g. 'Stripe', 'Auth0', 'Firebase').\n"
        "- Q13 (constraints): EXTRACT named infra (e.g. 'Kubernetes', 'AWS', 'microservices').\n"
        "- Do NOT infer Q11 (tech stack) from product/service names (e.g. 'Teleport', 'Kubernetes').\n"
        "- Do NOT infer Q4 (end-state) from vague goals like 'improve security'.\n"
        "- Do NOT extract Q3 (problem/users) unless the description names specific users or problems.\n"
        "- When in doubt, leave the question OUT — the user will be asked directly.\n\n"
        "Return ONLY the JSON object, no other text.\n\n"
        'Example: {"1": "A todo app for tracking tasks", "6": "3 engineers"}'
    )

    try:
        # Single LLM call with low temperature for deterministic extraction.
        # See docs: "Agentic Blueprint Reference" — using the LLM outside the main graph
        response = _invoke_json(prompt)
        raw = response.content

        # Strip markdown code fences that LLMs sometimes wrap JSON in
        # (e.g. ```json\n{...}\n```)
        raw = raw.strip()
        if raw.startswith("```"):
            # Remove opening fence (```json or ```)
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()

        parsed = json.loads(raw)

        # Validate: must be a dict, keys must be valid question numbers, values non-empty strings
        if not isinstance(parsed, dict):
            return {}

        result: dict[int, str] = {}
        for key, value in parsed.items():
            try:
                q_num = int(key)
            except (ValueError, TypeError):
                continue
            if 1 <= q_num <= TOTAL_QUESTIONS and isinstance(value, str) and value.strip():
                result[q_num] = value.strip()

        return result

    except Exception as exc:
        if _should_reraise_llm_error(exc):
            raise
        # Graceful fallback: if anything goes wrong (bad JSON, LLM timeout,
        # network error), return empty dict. The questionnaire will ask all
        # 26 questions as usual — the feature never breaks the existing flow.
        logger.debug("Failed to extract answers from description, falling back to full questionnaire", exc_info=True)
        return {}


def _keyword_extract_fallback(description: str, extracted: dict[int, str]) -> dict[int, str]:
    """Deterministic keyword-based extraction fallback after LLM extraction.

    # See docs: "Project Intake Questionnaire" — smart intake
    #
    # Scans the description for strong keyword signals for Q2 (project type),
    # Q12 (integrations), and Q13 (architectural constraints). Only fills
    # questions not already extracted by the LLM — LLM-extracted values
    # always take priority.

    Args:
        description: The user's project description text.
        extracted: The LLM-extracted answers dict (mutated in place with additions).

    Returns:
        The same extracted dict with any new keyword-based additions.
    """
    desc_lower = description.lower()

    # Q2: project type from keywords
    if 2 not in extracted:
        for keyword, value in Q2_INFERENCE_KEYWORDS.items():
            if keyword in desc_lower:
                extracted[2] = value
                break

    # Q12: service/integration names
    if 12 not in extracted:
        found_services = [kw for kw in Q12_SERVICE_KEYWORDS if kw in desc_lower]
        if found_services:
            # Capitalize service names for readability
            extracted[12] = ", ".join(s.capitalize() for s in sorted(found_services))

    # Q13: infrastructure/architecture keywords
    if 13 not in extracted:
        found_infra = [kw for kw in Q13_INFRA_KEYWORDS if kw in desc_lower]
        if found_infra:
            extracted[13] = ", ".join(sorted(found_infra))

    return extracted


def _next_unskipped_question(current: int, skipped: set[int]) -> int | None:
    """Find the next question number that hasn't been skipped.

    Scans forward from `current` through TOTAL_QUESTIONS.

    Args:
        current: The question number to start scanning from (inclusive).
        skipped: Set of question numbers to skip over.

    Returns:
        The first question number >= current not in skipped, or None if all
        remaining questions are skipped.
    """
    for q_num in range(current, TOTAL_QUESTIONS + 1):
        if q_num not in skipped:
            return q_num
    return None


def _build_extraction_summary(extracted: dict[int, str]) -> str:
    """Format what was extracted from the initial description for user review.

    Shows each extracted question + answer so the user can see what was
    auto-detected and verify correctness before continuing.

    Args:
        extracted: Dict mapping question numbers to extracted answers.

    Returns:
        A formatted string summarizing the extracted answers.
    """
    lines = ["I extracted the following from your description:\n"]
    for q_num in sorted(extracted):
        question = INTAKE_QUESTIONS[q_num]
        answer = extracted[q_num]
        lines.append(f"  **Q{q_num}.** {question}")
        lines.append(f"  > {answer}\n")
    return "\n".join(lines)


# ── Skip intent detection ────────────────────────────────────────────
# See docs: "Project Intake Questionnaire" — adaptive behavior
#
# Deterministic keyword matching for "skip" / "I don't know" responses.
# No LLM call needed — these are unambiguous user signals. Exact matches
# use a frozenset for O(1) lookup; substring matches use a tuple for
# ordered iteration via `in`.

_SKIP_EXACT: frozenset[str] = frozenset({"skip", "pass", "next", "n/a", "na", "idk", "-", "none"})

_SKIP_SUBSTRINGS: tuple[str, ...] = (
    "i don't know",
    "i dont know",
    "not sure",
    "unsure",
    "no idea",
    "don't know",
    "dont know",
    "skip this",
    "pass on this",
    "move on",
    "no answer",
)


def _is_skip_intent(message: str) -> bool:
    """Detect whether a user message is a skip/don't-know signal.

    Uses deterministic keyword matching — no LLM call. Normalized to
    lowercase + stripped before checking.

    Args:
        message: The raw user message text.

    Returns:
        True if the message indicates the user wants to skip the question.
    """
    normalized = message.strip().lower()
    if normalized in _SKIP_EXACT:
        return True
    return any(phrase in normalized for phrase in _SKIP_SUBSTRINGS)


# ── Defaults intent detection ────────────────────────────────────────
# See docs: "Project Intake Questionnaire" — batch defaults
#
# When the user types "defaults" during the questionnaire, all remaining
# questions in the current phase are answered with their defaults (from
# QUESTION_METADATA for choice Qs, QUESTION_DEFAULTS for free-text).
# Essential questions with no default are skipped over. This lets the
# user fast-forward through a phase without answering each question.

_DEFAULTS_EXACT: frozenset[str] = frozenset({"defaults", "default", "use defaults"})

# The all-phases variant (/finish sends this literal). Deliberately NOT bare
# "finish" — that is a plausible free-text answer ("finish the migration") and
# intent matching must never swallow a real answer.
_DEFAULTS_ALL_EXACT: frozenset[str] = frozenset({"defaults all", "all defaults", "use defaults for everything"})


def _is_defaults_all_intent(message: str) -> bool:
    """Detect an 'apply defaults to every remaining question' signal.

    Uses deterministic keyword matching — no LLM call. Disjoint from
    _DEFAULTS_EXACT, so the per-phase and all-phases intents can't collide.
    """
    return message.strip().lower() in _DEFAULTS_ALL_EXACT


def _is_defaults_intent(message: str) -> bool:
    """Detect whether a user message is a 'use defaults' signal.

    Uses deterministic keyword matching — no LLM call.

    Args:
        message: The raw user message text.

    Returns:
        True if the message indicates the user wants to apply defaults.
    """
    return message.strip().lower() in _DEFAULTS_EXACT


def _batch_defaults_for_phase(questionnaire: QuestionnaireState, last_q: int | None = None) -> tuple[list[str], int]:
    """Apply defaults to all remaining questions in the current phase.

    # See docs: "Project Intake Questionnaire" — batch defaults
    #
    # Iterates from current_question through the end of the current phase
    # (or through last_q when given — the "defaults all" intent passes
    # TOTAL_QUESTIONS to span every remaining phase in one pass).
    # For each question:
    #   - Choice Q with default_index → use option at default_index
    #   - Free-text Q with QUESTION_DEFAULTS entry → use that default
    #   - Essential Q with no default → skip (flagged in summary)
    #
    # Returns a list of summary lines and the count of questions defaulted.

    Args:
        questionnaire: The mutable QuestionnaireState to update.
        last_q: Inclusive end question; defaults to the current phase's end.

    Returns:
        A tuple of (summary_lines, count_defaulted).
    """
    from yeaboi.agent.state import PHASE_QUESTION_RANGES

    phase = questionnaire.current_phase
    _start, end = PHASE_QUESTION_RANGES[phase]
    if last_q is not None:
        end = last_q
    summary_lines: list[str] = []
    count = 0

    for q_num in range(questionnaire.current_question, end + 1):
        if q_num in questionnaire.answers or q_num in questionnaire.skipped_questions:
            continue  # already answered or skipped

        meta = QUESTION_METADATA.get(q_num)
        if meta and meta.default_index is not None:
            # Choice question with a default — use the option
            default_val = meta.options[meta.default_index]
            questionnaire.answers[q_num] = default_val
            questionnaire.defaulted_questions.add(q_num)
            questionnaire.answer_sources[q_num] = AnswerSource.DEFAULTED
            summary_lines.append(f"  Q{q_num}: {default_val}")
            count += 1
        elif q_num in QUESTION_DEFAULTS:
            # Free-text question with a default
            default_val = QUESTION_DEFAULTS[q_num]
            questionnaire.answers[q_num] = default_val
            questionnaire.defaulted_questions.add(q_num)
            questionnaire.answer_sources[q_num] = AnswerSource.DEFAULTED
            summary_lines.append(f"  Q{q_num}: {default_val}")
            count += 1
        else:
            # Essential question — no default, flag as skipped
            questionnaire.skipped_questions.add(q_num)

    return summary_lines, count


# ── Confirm intent detection ─────────────────────────────────────────
# See docs: "Project Intake Questionnaire" — confirmation gate
#
# After the last question is answered, the intake node shows a summary and
# asks the user to confirm before proceeding to the main agent. This helper
# detects confirmation signals — deterministic keyword matching, same
# pattern as _is_skip_intent().

_CONFIRM_KEYWORDS: frozenset[str] = frozenset(
    {
        "confirm",
        "confirmed",
        "accept",
        "accepted",
        "yes",
        "y",
        "looks good",
        "lgtm",
        "proceed",
        "go ahead",
        "ok",
        "okay",
    }
)


def _is_confirm_intent(text: str) -> bool:
    """Detect whether a user message is a confirmation signal.

    Uses deterministic keyword matching — no LLM call. Normalized to
    lowercase + stripped before checking.

    Args:
        text: The raw user message text.

    Returns:
        True if the message indicates the user wants to confirm and proceed.
    """
    return text.strip().lower() in _CONFIRM_KEYWORDS


# ── Edit intent detection ─────────────────────────────────────────
# See docs: "Project Intake Questionnaire" — edit flow
#
# During the confirmation gate the user can reference a question by
# number to revise their answer. Two modes:
#   - Inline: "Q6: 5 engineers" → update immediately, re-show summary
#   - Re-ask: "Q6" or "edit Q6" → show question, collect new answer
#
# Deterministic regex matching (same pattern as _is_skip_intent and
# _is_confirm_intent) — no LLM call needed.

_EDIT_PATTERN = re.compile(
    r"^(?:edit|change|revise|update)?\s*"
    r"(?:q(?:uestion)?\s*)?"  # Q prefix is optional — bare numbers like "25" also match
    r"(\d{1,2})"
    r"(?:\s*[:=]\s*(.+))?$",
    re.IGNORECASE,
)


def _parse_edit_intent(text: str) -> tuple[int, str | None] | None:
    """Detect whether a user message is an edit request for a specific question.

    Supports formats like:
    - "Q6" or "q6" → re-ask Q6
    - "6" or "25" → bare number, re-ask that question (Q prefix optional)
    - "edit Q6" / "change Q6" / "revise Q6" / "update Q6" → re-ask Q6
    - "edit 6" / "change 25" → re-ask without Q prefix
    - "Q6: 5 engineers" or "Q6 = new answer" → inline edit Q6
    - "question 6: new answer" → inline edit Q6

    Args:
        text: The raw user message text.

    Returns:
        A tuple of (question_number, inline_answer_or_None) if an edit intent
        is detected, or None if the message is not an edit request.
    """
    match = _EDIT_PATTERN.match(text.strip())
    if not match:
        return None

    q_num = int(match.group(1))
    if not (1 <= q_num <= TOTAL_QUESTIONS):
        return None

    inline_answer = match.group(2)
    if inline_answer is not None:
        inline_answer = inline_answer.strip()
        if not inline_answer:
            inline_answer = None

    return (q_num, inline_answer)


_EDIT_HELP = (
    "To edit an answer, use one of these formats:\n"
    "- **Q6: new answer** — update Q6 immediately\n"
    "- **edit Q6** or just **Q6** — re-answer Q6 interactively\n\n"
)


# ── Velocity extraction helpers ──────────────────────────────────────
# See docs: "Scrum Standards" — velocity and capacity planning
#
# After the user confirms the intake summary, we parse team size (Q6)
# and velocity (Q9) into typed ScrumState fields. Downstream nodes
# (sprint planner, Phase 5) need these numeric values to allocate
# stories to sprints without exceeding capacity.
#
# Deterministic parsing (not LLM) — same philosophy as _is_skip_intent().
# A simple regex extracts the first integer from natural-language answers
# like "3 engineers" or "20 points per sprint". No LLM call needed for
# unambiguous numeric extraction.

_VELOCITY_PER_ENGINEER = 5


def _parse_first_int(text: str) -> int | None:
    """Extract the first integer from a natural-language string.

    Uses a simple regex to find the first sequence of digits. This is
    sufficient for answers like "3 engineers", "About 5 people", or
    "velocity is 20 points" where the number is unambiguous.

    Args:
        text: The raw string to extract an integer from.

    Returns:
        The first integer found, or None if no digits are present.
    """
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def _parse_jira_team_size(questionnaire: QuestionnaireState) -> int | None:
    """Extract the Jira org team size if available.

    Primary source: ``_jira_org_team_size`` transient field — set during
    smart intake even when Jira velocity is zero (team size is derived from
    sub-task assignees independently of story point completion).

    Fallback: parse Q9 answer text for "from Jira: … N team member(s)".

    The Jira team size is the total headcount on the board — used to cap
    the "increase team" recommendation so we never suggest more engineers
    than actually exist.
    """
    # Primary: transient field (always set when Jira is reachable)
    if getattr(questionnaire, "_jira_org_team_size", None):
        return questionnaire._jira_org_team_size
    # Fallback: parse Q9 text (for sessions started before this field existed)
    q9 = questionnaire.answers.get(9, "")
    if "from Jira" not in q9 and "from jira" not in q9:
        return None
    m = re.search(r"(\d+)\s+team member", q9)
    return int(m.group(1)) if m else None


def _extract_team_and_velocity(questionnaire: QuestionnaireState) -> dict:
    """Parse team size and velocity from intake answers, calculating defaults.

    # See docs: "Scrum Standards" — velocity and capacity planning
    #
    # The system prompt documents the rule: "When team velocity is unknown,
    # use 5 story points per engineer per sprint as the baseline estimate."
    # This function implements that calculation deterministically.
    #
    # Why at confirmation time (not inside the sprint planner)?
    # The confirmation gate is the natural boundary — answers are finalized,
    # and the state update can include the extracted numeric fields alongside
    # the questionnaire. This keeps the sprint planner focused on allocation
    # rather than parsing.

    Args:
        questionnaire: The completed questionnaire with all answers.

    Returns:
        A dict with team_size, velocity_per_sprint, and _velocity_was_calculated
        (transient flag for the confirmation message). Returns {} if Q6 has
        no parseable team size — confirmation still works, just without
        velocity info.
    """
    # Parse Q6 → team_size
    q6_answer = questionnaire.answers.get(6)
    if not q6_answer:
        return {}
    team_size = _parse_first_int(q6_answer)
    if not team_size:  # None or 0
        return {}

    # When Jira per-dev velocity is available, recompute the feature velocity
    # from the stored per-dev rate × current Q6 team size. This handles the
    # case where the user edits Q6 at the confirmation gate — the velocity
    # automatically adjusts to the new team size.
    # See docs: "Scrum Standards" — capacity planning
    if questionnaire._jira_per_dev_velocity is not None:
        velocity = round(questionnaire._jira_per_dev_velocity * team_size)
        if velocity <= 0:
            velocity = team_size * _VELOCITY_PER_ENGINEER
        return {
            "team_size": team_size,
            "velocity_per_sprint": velocity,
            "_velocity_was_calculated": False,
        }

    # Parse Q9 → velocity. Skip if Q9 was defaulted (the default text says
    # "No historical velocity — will use default of 5 points per engineer
    # per sprint"), missing, or has no parseable number.
    velocity = None
    q9_answer = questionnaire.answers.get(9)
    if q9_answer and 9 not in questionnaire.defaulted_questions:
        velocity = _parse_first_int(q9_answer)
        if velocity is not None and velocity <= 0:
            velocity = None

    velocity_was_calculated = velocity is None
    if velocity is None:
        velocity = team_size * _VELOCITY_PER_ENGINEER

    return {
        "team_size": team_size,
        "velocity_per_sprint": velocity,
        "_velocity_was_calculated": velocity_was_calculated,
    }


def _is_jira_configured() -> bool:
    """Whether Jira is fully configured — a thin alias over the tracker registry.

    # See docs: "Tools" — tool types, Jira integration
    """
    from yeaboi import trackers

    return trackers.TRACKERS["jira"].is_configured()


def _is_azdevops_configured() -> bool:
    """Whether Azure DevOps Boards is configured — alias over the tracker registry."""
    from yeaboi import trackers

    return trackers.TRACKERS["azdevops"].is_configured()


def _is_tracker_configured() -> bool:
    """Whether any issue tracker is configured.

    Used by the intake node to decide whether to ask Q27 (sprint selection)
    interactively and whether to fetch velocity data.
    """
    from yeaboi import trackers

    return bool(trackers.configured())


def _fetch_jira_velocity() -> dict | None:
    """Alias over the tracker registry — see :mod:`yeaboi.trackers`."""
    from yeaboi import trackers

    return trackers.TRACKERS["jira"].fetch_velocity()


def _fetch_azdevops_velocity() -> dict | None:
    """Alias over the tracker registry — see :mod:`yeaboi.trackers`."""
    from yeaboi import trackers

    return trackers.TRACKERS["azdevops"].fetch_velocity()


def _fetch_tracker_velocity(preferred: str = "") -> dict | None:
    """Fetch velocity from the preferred tracker, else the first configured one.

    # See docs: "Scrum Standards" — capacity planning
    """
    from yeaboi import trackers

    return trackers.fetch_velocity(preferred)


def _fetch_active_sprint_number(preferred: str = "") -> tuple[int | None, str | None, str]:
    """The configured tracker's current active sprint/iteration.

    # See docs: "Scrum Standards" — sprint planning

    Returns:
        Tuple of (sprint_number, start_date, status_message).
        sprint_number is None on failure. start_date is ISO string or None.
        status_message explains what happened — shown to the user so they know
        why sprint selection fell back to "Fresh start".
    """
    from yeaboi import trackers

    return trackers.fetch_active_sprint(preferred)


def _fetch_sprint_targets(preferred: str = "") -> tuple[list[dict], str]:
    """The board's open (active + future) sprints/iterations as targets.

    # See docs: "Scrum Standards" — sprint planning
    #
    # Small-project mode offers "add this work to an existing sprint" and needs
    # the real board sprints to offer as targets — the active one first (small
    # work often lands mid-sprint), then upcoming future sprints. The registry
    # entries use the same paginated, scrum-filtered lookups as the sync
    # modules so intake and sync can never disagree about what exists.

    Returns:
        ([{name, external_id, state, start_date, number|None}], status_message).
        Empty list on failure, with the reason in status_message — same
        contract as _fetch_active_sprint_number.
    """
    from yeaboi import trackers

    return trackers.fetch_sprint_targets(preferred)


# Cap on how many existing sprints the small-mode Q27 menu offers — beyond
# the active sprint and the next few, "add to a sprint two months out" is
# backlog management, not small-project planning.
_MAX_SPRINT_TARGET_CHOICES = 4

_SPRINT_TARGET_CREATE_CHOICE = "Create a new sprint"

_SPRINT_TARGET_BACKLOG_CHOICE = "Backlog (no sprint)"


def _setup_small_sprint_target_question(questionnaire: QuestionnaireState) -> str | None:
    """Park the small-mode Q27 choices: existing sprint, create new, or backlog.

    # See docs: "Scrum Standards" — sprint planning
    #
    # Fetches the board's active + future sprints and offers them by their real
    # names ("Add to PSOT Sprint 104 (active)"), plus "Create a new sprint" and
    # "Backlog (no sprint)". Mirrors the smart-mode Q27 dynamic-choice pattern:
    # sentinel answer + _follow_up_choices, resolved on the next pass.

    Returns:
        The prompt text to show, or None when no targets could be fetched —
        the caller then falls back to _setup_small_sprint_fallback_question,
        which keeps the create/backlog halves of the decision askable.
    """
    targets, status = _fetch_sprint_targets(questionnaire._preferred_tracker)
    if not targets:
        logger.warning("Small-mode Q27: sprint target fetch failed: %s", status)
        return None

    active = next((t for t in targets if t["state"] == "active"), None)
    if active is not None and active["number"] is not None:
        # Feed the start-date offset machinery the same transients smart mode sets.
        questionnaire._active_sprint_number = active["number"]
        questionnaire._active_sprint_start_date = active["start_date"] or None

    offered = targets[:_MAX_SPRINT_TARGET_CHOICES]
    questionnaire._sprint_target_options = {t["name"]: t["external_id"] for t in offered}
    choices = [f"Add to {t['name']}{' (active)' if t['state'] == 'active' else ''}" for t in offered]
    choices.append(_SPRINT_TARGET_CREATE_CHOICE)
    choices.append(_SPRINT_TARGET_BACKLOG_CHOICE)
    questionnaire._follow_up_choices[27] = tuple(choices)
    if active is not None and active["number"] is not None:
        questionnaire.answers[27] = f"_active:{active['number']}"
    else:
        questionnaire.answers[27] = "_targets"

    from yeaboi import trackers

    pref = questionnaire._preferred_tracker
    label = trackers.label_for(pref) or "Jira"
    logger.info("Small-mode Q27: offering %d sprint target(s) from %s", len(offered), label)
    active_line = f"Detected active sprint in {label}: **{active['name']}**.\n\n" if active else ""
    return f"{active_line}Add this work to an existing sprint, create a new one, or leave it in the backlog?"


def _setup_small_sprint_fallback_question(
    questionnaire: QuestionnaireState, last_num: int | None, source_line: str
) -> str:
    """Small-mode Q27 when the board's sprint list cannot be fetched.

    The small-project question is a landing decision — existing sprint, new
    sprint, or backlog. Without a target list the first option is gone, but
    the other two remain askable; the generic numbered menu used to swallow
    them (a bare "Sprint 25" choice with no create/backlog language).
    last_num is the best-known latest sprint number (live active sprint or
    team analysis), used to name the create choice; None means no number is
    known anywhere and the sync will number from the board at sync time.
    """
    if last_num is not None:
        questionnaire._active_sprint_number = last_num
        questionnaire.answers[27] = f"_active:{last_num}"
        choices = (f"Create Sprint {last_num + 1} (next)", _SPRINT_TARGET_BACKLOG_CHOICE)
    else:
        questionnaire.answers[27] = "_targets"
        choices = (_SPRINT_TARGET_CREATE_CHOICE, _SPRINT_TARGET_BACKLOG_CHOICE)
    questionnaire._follow_up_choices[27] = choices
    logger.info("Small-mode Q27 fallback: create/backlog choices (last_num=%s)", last_num)
    return f"{source_line}Create a new sprint for this work, or leave it in the backlog?"


def _resolve_small_sprint_target_answer(questionnaire: QuestionnaireState) -> None:
    """Normalize the small-mode Q27 answer after the user picked a choice.

    "Add to <name> (active)" → "Add to <name>" (the real board name, kept for
    the confirmation write). "Create a new sprint" → "Sprint <max+1>" so the
    plan's Sprint artifact carries the number the sync will actually use.
    """
    answer = questionnaire.answers.get(27, "")
    if answer.startswith("Add to "):
        target_name = answer.removeprefix("Add to ").strip()
        target_name = target_name.removesuffix("(active)").strip()
        questionnaire.answers[27] = f"Add to {target_name}"
        questionnaire._follow_up_choices.pop(27, None)
        return
    if "backlog" in answer.lower():
        questionnaire.answers[27] = _SPRINT_TARGET_BACKLOG_CHOICE
        questionnaire._follow_up_choices.pop(27, None)
        return
    if _SPRINT_TARGET_CREATE_CHOICE.lower() in answer.lower():
        numbers = [
            int(m.group(1))
            for name in questionnaire._sprint_target_options
            if (m := re.search(r"(\d+)\s*$", name)) is not None
        ]
        if questionnaire._active_sprint_number is not None:
            numbers.append(questionnaire._active_sprint_number)
        if numbers:
            questionnaire.answers[27] = f"Sprint {max(numbers) + 1}"
        else:
            questionnaire.answers[27] = "Fresh start (today)"
        questionnaire._follow_up_choices.pop(27, None)


# ---------------------------------------------------------------------------
# Intake-mode helpers — Small project / Large / Offline.
# See docs: "Project Intake Questionnaire" — intake modes.
#
# "small_project" is a lightweight mode for 1-2 tickets in one quick sprint.
# It reuses the smart-intake extraction engine but asks fewer questions
# (SMALL_PROJECT_ESSENTIALS) and switches OFF the capacity machinery: no bank
# holidays, no PTO, no per-sprint velocity, no capacity-overflow advisory.
# "smart" is the Epic-wide engine (full capacity planning), unchanged.
# ---------------------------------------------------------------------------


def _essentials_for_mode(intake_mode: str) -> frozenset[int]:
    """Return the essential-question set for the given intake mode.

    Smart (Large) is the default; quick and small_project are leaner subsets.
    """
    if intake_mode == "quick":
        return QUICK_ESSENTIALS
    if intake_mode == "small_project":
        return SMALL_PROJECT_ESSENTIALS
    return SMART_ESSENTIALS


def _is_small_project_mode(intake_mode: str | None) -> bool:
    """True when the intake is running in the lightweight Small-project mode."""
    return intake_mode == "small_project"


def _extract_capacity_deductions(questionnaire: QuestionnaireState) -> dict:
    """Parse all capacity questions (Q27-Q30) from intake answers.

    # See docs: "Scrum Standards" — capacity planning
    #
    # All capacity questions are now collected during intake (Phase 6).
    # Q27 (sprint selection) determines which sprint to plan for.
    # Q28 (bank holidays) is auto-detected from locale and confirmed by user.
    # Q29 (unplanned %), Q30 (onboarding) are defaulted in smart mode.

    Args:
        questionnaire: The completed questionnaire with all answers.

    Returns:
        A dict with all capacity fields for ScrumState.
    """
    # Small-project mode does no capacity planning — return zero deductions so
    # net velocity equals gross velocity (no bank holidays / PTO / discovery tax).
    if _is_small_project_mode(questionnaire.intake_mode):
        return {
            "capacity_bank_holiday_days": 0,
            "capacity_planned_leave_days": 0,
            "capacity_unplanned_leave_pct": 0,
            "capacity_onboarding_engineer_sprints": 0,
            "capacity_ktlo_engineers": 0,
            "capacity_discovery_pct": 0,
        }

    # Q28 — bank holidays (auto-detected, confirmed by user as a choice)
    # The _detected_bank_holiday_days transient field is set during Q28 processing.
    # Fall back to parsing the Q28 answer text for a count.
    bank_holidays = questionnaire._detected_bank_holiday_days
    if bank_holidays == 0:
        q28 = questionnaire.answers.get(28, "No bank holidays detected")
        parsed = _parse_first_int(q28)
        if parsed is not None:
            bank_holidays = parsed

    # Q29 — unplanned leave % (choice question)
    q29 = questionnaire.answers.get(29, "10%")
    unplanned_pct = _parse_first_int(q29)
    if unplanned_pct is None:
        unplanned_pct = 10  # default 10%

    # Q30 — onboarding engineer-sprints
    q30 = questionnaire.answers.get(30, "No engineers onboarding")
    onboarding = _parse_first_int(q30) or 0

    # Planned leave — sum of working days from per-person leave entries
    # collected in the PTO sub-loop after Q28. Defaults to 0 if no entries.
    # See docs: "Scrum Standards" — capacity planning
    planned_leave = sum(e.get("working_days", 0) for e in questionnaire._planned_leave_entries)

    return {
        "capacity_bank_holiday_days": bank_holidays,
        "capacity_planned_leave_days": planned_leave,
        "capacity_unplanned_leave_pct": unplanned_pct,
        "capacity_onboarding_engineer_sprints": onboarding,
        "capacity_ktlo_engineers": 0,
        "capacity_discovery_pct": 5,
    }


def _compute_net_velocity(
    team_size: int,
    velocity_per_sprint: int,
    sprint_length_weeks: int,
    target_sprints: int,
    bank_holiday_days: int,
    planned_leave_days: int,
    unplanned_leave_pct: int,
    onboarding_engineer_sprints: int,
    ktlo_engineers: int = 0,
    discovery_pct: int = 5,
) -> int:
    """Compute net velocity per sprint after capacity deductions.

    # See docs: "Scrum Standards" — capacity planning
    #
    # Formula:
    #   gross_days = team_size × sprint_length_days × num_sprints
    #   ktlo_days = ktlo_engineers × sprint_length_days × num_sprints
    #   available_days = gross_days - ktlo_days
    #   deductions = bank_holidays + planned_leave + unplanned + onboarding_days
    #   discovery_days = (available_days - deductions) × discovery_pct / 100
    #   net_days = max(available_days - deductions - discovery_days, 0)
    #   net_velocity = round(net_days / gross_days × velocity_per_sprint)

    Args:
        ktlo_engineers: Engineers dedicated to KTLO/BAU work (default 0).
        discovery_pct: Discovery/design tax as a percentage (default 5).

    Returns:
        The adjusted velocity (always >= 1).
    """
    sprint_length_days = sprint_length_weeks * 5  # working days per sprint
    gross_days = team_size * sprint_length_days * target_sprints
    ktlo_days = ktlo_engineers * sprint_length_days * target_sprints
    available_days = gross_days - ktlo_days
    unplanned_days = gross_days * unplanned_leave_pct / 100
    onboarding_days = onboarding_engineer_sprints * sprint_length_days
    total_deductions = bank_holiday_days + planned_leave_days + unplanned_days + onboarding_days
    after_deductions = max(available_days - total_deductions, 0)
    discovery_days = after_deductions * discovery_pct / 100
    net_days = max(after_deductions - discovery_days, 0)

    if gross_days > 0:
        net_ratio = net_days / gross_days
        return max(1, round(net_ratio * velocity_per_sprint))
    return velocity_per_sprint


def _assign_holidays_to_sprints(
    holidays: list[dict],
    sprint_start_date: str,
    sprint_length_weeks: int,
    target_sprints: int,
) -> dict[int, list[dict]]:
    """Map each bank holiday to its 0-based sprint index.

    # See docs: "Scrum Standards" — capacity planning
    #
    # Each holiday dict has {"date": date, "name": str, "weekday": str}
    # from get_bank_holidays_structured(). We compute which sprint window
    # the holiday falls in so per-sprint velocity can be calculated.

    Returns:
        Dict mapping sprint_index → list of holiday dicts in that sprint.
    """
    from datetime import date

    if not holidays or target_sprints <= 0:
        return {}

    try:
        start = parse_date(sprint_start_date) if sprint_start_date else date.today()
    except (ValueError, TypeError):
        start = date.today()

    sprint_days = sprint_length_weeks * 7
    result: dict[int, list[dict]] = {}

    for holiday in holidays:
        h_date = holiday.get("date")
        if h_date is None:
            continue
        # Convert string dates to date objects
        if isinstance(h_date, str):
            try:
                h_date = parse_date(h_date)
            except ValueError:
                continue

        delta = (h_date - start).days
        if delta < 0:
            continue
        sprint_idx = delta // sprint_days
        if sprint_idx < target_sprints:
            result.setdefault(sprint_idx, []).append(holiday)

    return result


def _parse_date_dmy(text: str):
    """Parse a user-entered date in DD/MM/YYYY, DD/MM/YY, DD-MM-YYYY, or DD-MM-YY format.

    # See docs: "Scrum Standards" — capacity planning
    #
    # Users enter leave dates in day-first format (common in UK/EU).
    # Two-digit years assume 2000s (e.g. 26 → 2026).

    Returns:
        A datetime.date or None if the input is unrecognised.
    """
    import re as _re
    from datetime import date

    m = _re.match(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})$", text.strip())
    if not m:
        return None
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if year < 100:
        year += 2000
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None
    # Reject dates obviously in the past (> 6 months ago) — catches 2-digit year
    # typos like "12/12/12" → 2012 which is clearly not a future leave date.
    if (date.today() - parsed).days > 180:
        return None
    return parsed


def _count_working_days(start_date, end_date) -> int:
    """Count weekdays (Mon–Fri) between two dates, inclusive.

    # See docs: "Scrum Standards" — capacity planning
    #
    # Used to compute working days lost per leave entry. Only counts
    # Mon(0)–Fri(4); Sat(5) and Sun(6) are excluded.

    Returns:
        Number of working days (>= 0).
    """
    from datetime import timedelta

    if end_date < start_date:
        return 0
    count = 0
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Mon=0 .. Fri=4
            count += 1
        current += timedelta(days=1)
    return count


def _assign_leave_to_sprints(
    leave_entries: list[dict],
    sprint_start_date: str,
    sprint_length_weeks: int,
    target_sprints: int,
) -> dict[int, list[dict]]:
    """Map each PTO leave entry to sprint windows, handling multi-sprint spans.

    # See docs: "Scrum Standards" — capacity planning
    #
    # Similar to _assign_holidays_to_sprints but:
    # - Each entry can span multiple sprints (clipped to sprint boundaries)
    # - No × team_size multiplier (PTO is per-person, not team-wide)
    # - Returns working days per sprint, not holiday count

    Returns:
        Dict mapping sprint_index → list of {"person": str, "days": int}.
    """
    from datetime import date, timedelta

    if not leave_entries or target_sprints <= 0:
        return {}

    try:
        start = parse_date(sprint_start_date) if sprint_start_date else date.today()
    except (ValueError, TypeError):
        start = date.today()

    sprint_days = sprint_length_weeks * 7  # calendar days per sprint
    result: dict[int, list[dict]] = {}

    for entry in leave_entries:
        person = entry.get("person", "")
        try:
            leave_start = parse_date(entry["start_date"])
            leave_end = parse_date(entry["end_date"])
        except (ValueError, TypeError, KeyError):
            continue

        # Check each sprint window for overlap
        for i in range(target_sprints):
            sprint_begin = start + timedelta(days=i * sprint_days)
            sprint_end = start + timedelta(days=(i + 1) * sprint_days - 1)

            # Clip leave to sprint boundaries
            overlap_start = max(leave_start, sprint_begin)
            overlap_end = min(leave_end, sprint_end)

            if overlap_start <= overlap_end:
                working = _count_working_days(overlap_start, overlap_end)
                if working > 0:
                    result.setdefault(i, []).append({"person": person, "days": working})

    return result


def _compute_per_sprint_velocities(
    team_size: int,
    velocity_per_sprint: int,
    sprint_length_weeks: int,
    target_sprints: int,
    holidays_by_sprint: dict[int, list[dict]],
    planned_leave_days: int,
    unplanned_leave_pct: int,
    onboarding_engineer_sprints: int,
    ktlo_engineers: int = 0,
    discovery_pct: int = 5,
    leave_by_sprint: dict[int, list[dict]] | None = None,
) -> list[dict]:
    """Compute per-sprint velocities with sprint-specific bank holiday and PTO deductions.

    # See docs: "Scrum Standards" — capacity planning
    #
    # Unlike _compute_net_velocity (which spreads bank holidays evenly),
    # this function only reduces velocity for sprints that actually contain
    # bank holidays or PTO. Other deductions (unplanned leave, discovery, KTLO)
    # are still applied uniformly since they're probabilistic averages.
    #
    # leave_by_sprint maps sprint_index → list of {"person": str, "days": int}.
    # PTO is per-person (1 × days), not team-wide like bank holidays (team_size × days).
    # The None default preserves backward compatibility.

    Returns:
        List of dicts: [{"sprint_index": 0, "bank_holiday_days": 2,
                         "bank_holiday_names": ["Good Friday", ...],
                         "pto_days": 5, "pto_entries": [...],
                         "net_velocity": 3}, ...]
    """
    sprint_length_days = sprint_length_weeks * 5  # working days per sprint
    gross_days_per_sprint = team_size * sprint_length_days

    if gross_days_per_sprint == 0 or target_sprints == 0:
        return [
            {
                "sprint_index": i,
                "bank_holiday_days": 0,
                "bank_holiday_names": [],
                "pto_days": 0,
                "pto_entries": [],
                "net_velocity": velocity_per_sprint,
            }
            for i in range(target_sprints)
        ]

    # Flat deductions per sprint (uniform across all sprints)
    ktlo_days_per_sprint = ktlo_engineers * sprint_length_days
    unplanned_days_per_sprint = gross_days_per_sprint * unplanned_leave_pct / 100
    # When leave_by_sprint is provided, planned leave is already tracked per-sprint —
    # don't also spread the aggregate total, or PTO gets double-counted.
    planned_leave_per_sprint = 0 if leave_by_sprint else (planned_leave_days / target_sprints if target_sprints else 0)
    onboarding_days_total = onboarding_engineer_sprints * sprint_length_days
    onboarding_per_sprint = onboarding_days_total / target_sprints if target_sprints else 0

    result = []
    for i in range(target_sprints):
        # Sprint-specific bank holidays — each holiday costs team_size person-days
        sprint_holidays = holidays_by_sprint.get(i, [])
        bank_days = len(sprint_holidays) * team_size

        # Sprint-specific PTO — per-person, no team_size multiplier
        pto_entries = leave_by_sprint.get(i, []) if leave_by_sprint else []
        pto_days = sum(e["days"] for e in pto_entries)

        available = gross_days_per_sprint - ktlo_days_per_sprint
        total_deductions = (
            bank_days + pto_days + planned_leave_per_sprint + unplanned_days_per_sprint + onboarding_per_sprint
        )
        after_deductions = max(available - total_deductions, 0)
        discovery_days = after_deductions * discovery_pct / 100
        net_days = max(after_deductions - discovery_days, 0)

        net_ratio = net_days / gross_days_per_sprint
        net_vel = max(1, round(net_ratio * velocity_per_sprint))

        result.append(
            {
                "sprint_index": i,
                "bank_holiday_days": len(sprint_holidays),
                "bank_holiday_names": [h.get("name", "") for h in sprint_holidays],
                "pto_days": pto_days,
                "pto_entries": pto_entries,
                "net_velocity": net_vel,
            }
        )

    return result


def _build_velocity_breakdown(
    velocity_per_sprint: int,
    velocity_source: str,
    team_size: int,
    sprint_length_weeks: int,
    target_sprints: int,
    bank_holiday_days: int,
    planned_leave_days: int,
    unplanned_leave_pct: int,
    onboarding_engineer_sprints: int,
    ktlo_engineers: int = 0,
    discovery_pct: int = 5,
    planned_leave_entries: list[dict] | None = None,
) -> tuple[int, str]:
    """Compute net velocity and build a transparent markdown breakdown.

    # See docs: "Scrum Standards" — capacity planning
    #
    # Shows the user exactly how the recommended velocity was calculated,
    # converting each deduction into points-per-sprint so the impact is
    # immediately visible. Displayed at the confirmation gate before the
    # user accepts or overrides the velocity.

    Returns:
        Tuple of (net_velocity, breakdown_text).
    """
    net_vel = _compute_net_velocity(
        team_size=team_size,
        velocity_per_sprint=velocity_per_sprint,
        sprint_length_weeks=sprint_length_weeks,
        target_sprints=target_sprints,
        bank_holiday_days=bank_holiday_days,
        planned_leave_days=planned_leave_days,
        unplanned_leave_pct=unplanned_leave_pct,
        onboarding_engineer_sprints=onboarding_engineer_sprints,
        ktlo_engineers=ktlo_engineers,
        discovery_pct=discovery_pct,
    )

    # Convert each deduction to points-per-sprint for the breakdown.
    # The ratio is: deduction_days / gross_days × velocity_per_sprint.
    sprint_length_days = sprint_length_weeks * 5
    gross_days = team_size * sprint_length_days * target_sprints
    if gross_days == 0:
        return net_vel, f"**Net velocity: {net_vel} pts/sprint**"

    def _to_pts(days: float) -> float:
        return days / gross_days * velocity_per_sprint

    ktlo_days = ktlo_engineers * sprint_length_days * target_sprints
    unplanned_days = gross_days * unplanned_leave_pct / 100
    onboarding_days = onboarding_engineer_sprints * sprint_length_days
    available_days = gross_days - ktlo_days
    total_leave = bank_holiday_days + planned_leave_days + unplanned_days + onboarding_days
    after_deductions = max(available_days - total_leave, 0)
    discovery_days = after_deductions * discovery_pct / 100

    source_map = {
        "jira": "from Jira: avg of last 3 sprints",
        "manual": "user-provided",
        "estimated": "estimated",
    }
    source_label = source_map.get(velocity_source, velocity_source)

    lines = ["**Recommended Velocity**\n"]
    lines.append(f"Gross velocity:       {velocity_per_sprint} pts/sprint ({source_label})")

    deduction_lines: list[str] = []
    if bank_holiday_days > 0:
        pts = _to_pts(bank_holiday_days)
        detail = f"{bank_holiday_days} day(s) across {target_sprints} sprints"
        deduction_lines.append(f"  Bank holidays:      −{pts:.1f} pts  ({detail})")
    if planned_leave_days > 0:
        pts = _to_pts(planned_leave_days)
        # Show person names when leave entries are available
        if planned_leave_entries:
            names = ", ".join(f"{e['person']} {e['working_days']}d" for e in planned_leave_entries)
            deduction_lines.append(f"  Planned leave:      −{pts:.1f} pts  ({names})")
        else:
            deduction_lines.append(f"  Planned leave:      −{pts:.1f} pts  ({planned_leave_days} day(s))")
    if unplanned_leave_pct > 0:
        pts = _to_pts(unplanned_days)
        deduction_lines.append(f"  Unplanned absence:  −{pts:.1f} pts  ({unplanned_leave_pct}%)")
    if onboarding_engineer_sprints > 0:
        pts = _to_pts(onboarding_days)
        deduction_lines.append(
            f"  Onboarding:         −{pts:.1f} pts  ({onboarding_engineer_sprints} engineer × sprint(s))"
        )
    if ktlo_engineers > 0:
        pts = _to_pts(ktlo_days)
        deduction_lines.append(f"  KTLO/BAU:           −{pts:.1f} pts  ({ktlo_engineers} dedicated engineer(s))")
    if discovery_pct > 0:
        pts = _to_pts(discovery_days)
        deduction_lines.append(f"  Discovery/design:   −{pts:.1f} pts  ({discovery_pct}%)")

    if deduction_lines:
        lines.extend(deduction_lines)
        lines.append("                      ─────────")

    lines.append(f"**Net velocity:       {net_vel} pts/sprint**")

    return net_vel, "\n".join(lines)


def _parse_velocity_override(text: str) -> int | None:
    """Parse a velocity override from user input.

    Accepts bare numbers ("14"), numbers with units ("12 pts", "15 points"),
    or sentences like "use 18 points per sprint".

    Returns:
        The parsed velocity as int, or None if the input can't be parsed.
    """
    normalized = text.strip().lower()
    # Match patterns like "12", "12 pts", "12 points", "12 pts/sprint"
    match = re.search(r"(\d+)\s*(?:pts|points|pts/sprint)?", normalized)
    if match:
        val = int(match.group(1))
        if val > 0:
            return val
    return None


def _build_suggestion_line(questionnaire: QuestionnaireState, q_num: int) -> str:
    """Return a markdown suggestion line if the question has a suggested answer.

    Args:
        questionnaire: The current questionnaire state.
        q_num: The question number to check for suggestions.

    Returns:
        A formatted suggestion line, or empty string if no suggestion.
    """
    suggestion = questionnaire.suggested_answers.get(q_num)
    if suggestion:
        return f"\n\n> Extracted: **{suggestion}** *(Enter to accept, or type your own)*"
    return ""


# ── Smart / quick intake helpers ──────────────────────────────────────
# See docs: "Project Intake Questionnaire" — smart intake
#
# These helpers implement the three-mode intake system. Smart mode auto-
# applies LLM-extracted answers and defaults, only asking unfilled essential
# gaps. Quick mode is even more aggressive — only team size and tech stack.
# Standard mode is the original 26-question flow (unchanged).


def _load_user_context(path: str | None = None, docs_dir: str | None = None) -> tuple[str | None, dict]:
    """Read SCRUM.md and scrum-docs/ from the working directory and return combined content + status.

    # See docs: "Tools" — read-only tool pattern
    #
    # Thin wrapper around the load_project_context @tool in tools/codebase.py.
    # The tool handles all filesystem I/O and returns a JSON string; this wrapper
    # parses it back to the (context, status) tuple expected by the analyzer node.
    # Same pattern as _prepare_bank_holiday_choices calling detect_bank_holidays.invoke().

    Args:
        path: Override the SCRUM.md file path (used in tests). Defaults to SCRUM.md in CWD.
        docs_dir: Override the docs directory path (used in tests). Defaults to scrum-docs/ in CWD.

    Returns:
        Tuple of (context string or None, status dict with name/status/detail).
    """
    try:
        from yeaboi.tools.codebase import load_project_context

        result = load_project_context.invoke({"path": path or "", "docs_dir": docs_dir or ""})
        data = json.loads(result)
        return data["context"], data["status"]
    except Exception as e:
        logger.debug("User context load failed (non-fatal)", exc_info=True)
        return None, {"name": "User context", "status": "error", "detail": str(e)[:80]}


def _load_team_profile(profile_id: str = "") -> object | None:
    """Load the team profile from SQLite.

    If profile_id is given (from user's profile picker selection), loads that
    specific profile. Otherwise falls back to auto-detecting from configured
    Jira/AzDO project keys.

    Returns:
        A TeamProfile instance or None. Non-fatal — never raises.
    """
    try:
        from yeaboi.paths import get_db_path
        from yeaboi.team_profile import TeamProfileStore

        db_path = get_db_path()
        if not db_path.exists():
            return None

        with TeamProfileStore(db_path) as store:
            # If a specific profile was selected, use it
            if profile_id:
                profile = store.load(profile_id)
                if profile:
                    logger.debug("Loaded selected team profile: %s", profile_id)
                    return profile
                # Prefix match for old-format IDs (without timestamp)
                row = store._conn.execute(
                    "SELECT team_id FROM team_profiles WHERE team_id LIKE ? ORDER BY updated_at DESC LIMIT 1",
                    (f"{profile_id}%",),
                ).fetchone()
                if row:
                    profile = store.load(row[0])
                    if profile:
                        logger.debug("Loaded team profile via prefix: %s → %s", profile_id, row[0])
                        return profile

            # Auto-detect: try Jira first
            try:
                from yeaboi.config import get_jira_project_key

                jira_key = get_jira_project_key()
                if jira_key:
                    profile = store.load_by_project(jira_key, "jira")
                    if profile:
                        logger.debug("Loaded team profile for jira/%s", jira_key)
                        return profile
            except Exception:
                pass

            # Auto-detect: try AzDO
            try:
                from yeaboi.config import get_azdevops_project

                azdevops_project = get_azdevops_project()
                if azdevops_project:
                    profile = store.load_by_project(azdevops_project, "azdevops")
                    if profile:
                        logger.debug("Loaded team profile for azdevops/%s", azdevops_project)
                        return profile
            except Exception:
                pass

        return None
    except Exception:
        logger.debug("Team profile load failed (non-fatal)", exc_info=True)
        return None


def _load_team_examples(profile_id: str = "", db_path=None) -> dict | None:
    """Load the examples dict for the team profile from SQLite.

    If profile_id is given, loads examples for that specific profile.
    Otherwise auto-detects from configured Jira/AzDO project keys.

    `db_path` is the injection seam callers test against — without it a test
    that passes a temporary database would still read the developer's real
    `~/.yeaboi/sessions.db` and quietly depend on whatever is in it.
    """
    try:
        from yeaboi.paths import get_db_path
        from yeaboi.team_profile import TeamProfileStore

        db_path = db_path or get_db_path()
        if not db_path.exists():
            return None

        with TeamProfileStore(db_path) as store:
            if profile_id:
                _, ex = store.load_with_examples(profile_id)
                if ex:
                    return ex
                # Prefix match for old-format IDs
                row = store._conn.execute(
                    "SELECT team_id FROM team_profiles WHERE team_id LIKE ? ORDER BY updated_at DESC LIMIT 1",
                    (f"{profile_id}%",),
                ).fetchone()
                if row:
                    _, ex = store.load_with_examples(row[0])
                    if ex:
                        return ex

            try:
                from yeaboi.config import get_jira_project_key

                jira_key = get_jira_project_key()
                if jira_key:
                    row = store._conn.execute(
                        "SELECT team_id FROM team_profiles WHERE team_id LIKE ? ORDER BY updated_at DESC LIMIT 1",
                        (f"jira-{jira_key}%",),
                    ).fetchone()
                    if row:
                        _, ex = store.load_with_examples(row[0])
                        if ex:
                            return ex
            except Exception:
                pass
            try:
                from yeaboi.config import get_azdevops_project

                azdevops_project = get_azdevops_project()
                if azdevops_project:
                    row = store._conn.execute(
                        "SELECT team_id FROM team_profiles WHERE team_id LIKE ? ORDER BY updated_at DESC LIMIT 1",
                        (f"azdevops-{azdevops_project}%",),
                    ).fetchone()
                    if row:
                        _, ex = store.load_with_examples(row[0])
                        if ex:
                            return ex
            except Exception:
                pass

        return None
    except Exception:
        return None


def _load_profile_by_id(profile_id: str):
    """Load a specific team profile and examples by team_id.

    If exact match fails, tries prefix match to handle old-format IDs
    (e.g. "azdevops-Proj" matching "azdevops-Proj-202604010430").
    """
    try:
        from yeaboi.paths import get_db_path
        from yeaboi.team_profile import TeamProfileStore

        db_path = get_db_path()
        if not db_path.exists():
            return None, None

        with TeamProfileStore(db_path) as store:
            # Exact match first
            result = store.load_with_examples(profile_id)
            if result[0] is not None:
                return result

            # Prefix match — handles old IDs without timestamp
            row = store._conn.execute(
                "SELECT team_id FROM team_profiles WHERE team_id LIKE ? ORDER BY updated_at DESC LIMIT 1",
                (f"{profile_id}%",),
            ).fetchone()
            if row:
                logger.info("Profile prefix match: %s → %s", profile_id, row[0])
                return store.load_with_examples(row[0])

            return None, None
    except Exception:
        logger.debug("Failed to load profile %s", profile_id, exc_info=True)
        return None, None


def _extract_answers_from_profile(profile, examples: dict | None = None) -> dict[int, str]:
    """Extract intake questionnaire answers from a team analysis profile.

    Returns a dict mapping question numbers to answer strings.
    Used to auto-fill Q6 (team size), Q8 (sprint length), Q9 (velocity),
    Q11 (tech stack), Q12 (integrations) when the user selects an analysis
    profile in planning mode.
    """
    answers: dict[int, str] = {}
    _ex = examples or {}

    # Q6: Team size — from contributor stats
    contrib = _ex.get("contributor_stats", [])
    if isinstance(contrib, list) and contrib:
        answers[6] = str(len(contrib))
        logger.info("Analysis auto-fill: Q6 (team size) = %s from %d contributors", answers[6], len(contrib))

    # Q8: Sprint length — derive from sprint date ranges
    sprint_details = _ex.get("sprint_details", [])
    if sprint_details:
        try:
            durations = []
            for sd in sprint_details:
                start = sd.get("start", "")
                end = sd.get("end", "")
                if start and end:
                    s_dt = parse_datetime(start)
                    e_dt = parse_datetime(end)
                    days = (e_dt - s_dt).days
                    durations.append(days)
            if durations:
                avg_days = sum(durations) / len(durations)
                weeks = round(avg_days / 7)
                weeks = max(1, min(4, weeks))
                answers[8] = f"{weeks} week{'s' if weeks > 1 else ''}"
                logger.info("Analysis auto-fill: Q8 (sprint length) = %s from %d sprints", answers[8], len(durations))
        except Exception:
            pass

    # Q9: Velocity — use total team velocity as default suggestion
    # (will be recalculated when specific team members are selected in Q6)
    vel = getattr(profile, "velocity_avg", 0.0)
    if vel > 0:
        answers[9] = f"{vel:.0f} points per sprint (from team analysis)"
        logger.info("Analysis auto-fill: Q9 (velocity) = %.0f pts/sprint", vel)

    # Q11: Tech stack — from analysis profile's tech_stack field
    tech = getattr(profile, "tech_stack", ())
    if tech:
        answers[11] = ", ".join(tech)
        logger.info("Analysis auto-fill: Q11 (tech stack) = %s", answers[11])

    # Q12: Integrations — from analysis profile's integrations field
    integrations = getattr(profile, "integrations", ())
    if integrations:
        answers[12] = ", ".join(integrations)
        logger.info("Analysis auto-fill: Q12 (integrations) = %s", answers[12])

    return answers


def _get_contributor_names(examples: dict | None) -> list[str]:
    """Extract contributor names from analysis examples for Q6 multi-select."""
    if not examples:
        return []
    contrib = examples.get("contributor_stats", [])
    if isinstance(contrib, list):
        return [c.get("name", "") for c in contrib if c.get("name")]
    return []


def _calculate_velocity_for_members(member_names: list[str], examples: dict | None) -> float:
    """Calculate combined velocity from selected team members' per_sprint values.

    Delegates to the shared ``team_learning.selected_member_velocity`` (analysis mode
    uses the same helper when a run is re-scoped to a member subset) so there is one
    source of truth for the calculation.
    """
    if not examples or not member_names:
        return 0.0
    contrib = examples.get("contributor_stats", [])
    if not isinstance(contrib, list):
        return 0.0
    from yeaboi.tools.team_learning import selected_member_velocity

    return selected_member_velocity(contrib, member_names)


def _extract_confluence_page_ids(text: str) -> list[str]:
    """Extract Confluence page IDs from URLs found in free-form text (e.g. SCRUM.md).

    # See docs: "Tools" — read-only tool pattern
    #
    # Confluence Cloud URLs embed the numeric page ID in the path, e.g.:
    #   https://example.atlassian.net/wiki/spaces/SPACE/pages/1234567890/Page+Title
    # This regex extracts those IDs so we can call confluence_read_page directly
    # instead of relying on keyword search which often misses specific pages.
    """
    # Match /wiki/spaces/<key>/pages/<page_id> — the page_id is always numeric.
    return re.findall(r"/wiki/spaces/[^/]+/pages/(\d+)", text)


def _fetch_confluence_context(
    questionnaire: QuestionnaireState,
    user_context: str | None = None,
) -> tuple[str | None, dict]:
    """Search Confluence for docs related to the project and return combined context + status.

    # See docs: "Tools" — read-only tool pattern
    #
    # Two strategies are used to find relevant Confluence pages:
    # 1. Keyword search using the project name (Q1) — broad discovery.
    # 2. Direct page fetch for any Confluence URLs found in SCRUM.md — precise
    #    targeting of pages the user has explicitly linked (e.g. RunBook URLs).
    #
    # Both are combined into a single context string. Strategy 2 is the fix for
    # the issue where RunBook URLs in SCRUM.md's "Key Links" section were ignored
    # because the keyword search used the full project description as a CQL query,
    # which rarely matched specific page titles.

    Uses the project name (Q1) as the search query, and also extracts Confluence
    page IDs from user_context (SCRUM.md) URLs to fetch those pages directly.
    Falls back to (None, status) when Confluence is not configured.
    The caller proceeds gracefully with reduced context when None is returned.

    Args:
        questionnaire: The completed QuestionnaireState with Q1 (project name).
        user_context: Raw SCRUM.md content — scanned for Confluence URLs to fetch directly.

    Returns:
        Tuple of (context string or None, status dict with name/status/detail).
    """
    try:
        from yeaboi.config import get_jira_base_url, get_jira_email, get_jira_token

        # Only proceed if Confluence/Jira credentials are configured.
        if not all([get_jira_base_url(), get_jira_email(), get_jira_token()]):
            return None, {"name": "Confluence", "status": "skipped", "detail": "not configured"}

        from yeaboi.tools.confluence import confluence_read_page, confluence_search_docs

        parts: list[str] = []

        # Strategy 1: Keyword search using project name (Q1).
        project_name = questionnaire.answers.get(1, "").strip()
        if project_name and project_name != QUESTION_DEFAULTS.get(1):
            result = confluence_search_docs.invoke({"query": project_name, "limit": 5})
            if result and not result.startswith("Error") and not result.startswith("No Confluence"):
                parts.append(result)
                logger.debug("CONFLUENCE: keyword search returned results for %r", project_name)
            else:
                logger.debug("CONFLUENCE: keyword search returned no results for %r", project_name)

        # Strategy 2: Fetch pages directly from Confluence URLs in SCRUM.md.
        # This covers explicit links the user added (e.g. RunBook URLs) that
        # keyword search would miss.
        if user_context:
            page_ids = _extract_confluence_page_ids(user_context)
            logger.debug("CONFLUENCE: extracted %d page ID(s) from user_context: %s", len(page_ids), page_ids[:10])
            seen = set()
            for pid in page_ids:
                if pid in seen:
                    continue
                seen.add(pid)
                try:
                    logger.debug("CONFLUENCE: fetching page ID %s", pid)
                    page_content = confluence_read_page.invoke({"page_id": pid})
                    if page_content and not page_content.startswith("Error"):
                        parts.append(page_content)
                        logger.debug("CONFLUENCE: fetched page ID %s (%d chars)", pid, len(page_content))
                    else:
                        logger.debug(
                            "CONFLUENCE: page ID %s returned error: %s",
                            pid,
                            page_content[:100] if page_content else "empty",
                        )
                except Exception:
                    logger.debug("CONFLUENCE: failed to fetch page %s (non-fatal)", pid, exc_info=True)
        else:
            logger.debug("CONFLUENCE: no user_context provided, skipping URL extraction")

        if not parts:
            return None, {"name": "Confluence", "status": "error", "detail": "no docs found"}

        combined = "\n\n---\n\n".join(parts)
        detail = f"search + {len(parts) - 1} linked page(s)" if len(parts) > 1 else f"docs for '{project_name}'"
        return combined, {"name": "Confluence", "status": "success", "detail": detail}
    except Exception as e:
        logger.debug("Confluence context fetch failed (non-fatal)", exc_info=True)
        return None, {"name": "Confluence", "status": "error", "detail": str(e)[:80]}


def _extract_notion_page_ids(text: str) -> list[str]:
    """Extract Notion page IDs from URLs found in free-form text (e.g. SCRUM.md).

    # See docs: "Tools" — read-only tool pattern
    #
    # Notion URLs end with a 32-hex-character page ID (optionally dash-formatted),
    # usually appended to a slugified title, e.g.:
    #   https://www.notion.so/My-Runbook-1234567890abcdef1234567890abcdef
    #   https://www.notion.so/workspace/Title-12345678-90ab-cdef-1234-567890abcdef
    # We extract the raw 32-hex id (stripping dashes) so we can call notion_read_page
    # directly instead of relying on keyword search which often misses specific pages.
    """
    ids: list[str] = []
    # Match a 32-hex run, allowing the dash-separated 8-4-4-4-12 UUID form too.
    hex32 = r"[0-9a-fA-F]{32}"
    uuid = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    for m in re.findall(rf"({hex32}|{uuid})", text):
        ids.append(m.replace("-", ""))
    return ids


def _fetch_notion_context(
    questionnaire: QuestionnaireState,
    user_context: str | None = None,
) -> tuple[str | None, dict]:
    """Search Notion for docs related to the project and return combined context + status.

    # See docs: "Tools" — read-only tool pattern
    #
    # Mirrors _fetch_confluence_context exactly, with two strategies:
    # 1. Keyword search using the project name (Q1) — broad discovery.
    # 2. Direct page fetch for any Notion URLs found in SCRUM.md — precise
    #    targeting of pages the user has explicitly linked.
    #
    # Falls back to (None, status) when Notion is not configured; the caller
    # proceeds gracefully with reduced context when None is returned.

    Args:
        questionnaire: The completed QuestionnaireState with Q1 (project name).
        user_context: Raw SCRUM.md content — scanned for Notion URLs to fetch directly.

    Returns:
        Tuple of (context string or None, status dict with name/status/detail).
    """
    try:
        from yeaboi.config import get_notion_token

        # Only proceed if the Notion integration token is configured.
        if not get_notion_token():
            return None, {"name": "Notion", "status": "skipped", "detail": "not configured"}

        from yeaboi.tools.notion import notion_read_page, notion_search_pages

        parts: list[str] = []

        # Strategy 1: Keyword search using project name (Q1).
        project_name = questionnaire.answers.get(1, "").strip()
        if project_name and project_name != QUESTION_DEFAULTS.get(1):
            result = notion_search_pages.invoke({"query": project_name, "limit": 5})
            if result and not result.startswith("Error") and not result.startswith("No Notion"):
                parts.append(result)
                logger.debug("NOTION: keyword search returned results for %r", project_name)
            else:
                logger.debug("NOTION: keyword search returned no results for %r", project_name)

        # Strategy 2: Fetch pages directly from Notion URLs in SCRUM.md.
        if user_context:
            page_ids = _extract_notion_page_ids(user_context)
            logger.debug("NOTION: extracted %d page ID(s) from user_context: %s", len(page_ids), page_ids[:10])
            seen = set()
            for pid in page_ids:
                if pid in seen:
                    continue
                seen.add(pid)
                try:
                    logger.debug("NOTION: fetching page ID %s", pid)
                    page_content = notion_read_page.invoke({"page_id": pid})
                    if page_content and not page_content.startswith("Error"):
                        parts.append(page_content)
                        logger.debug("NOTION: fetched page ID %s (%d chars)", pid, len(page_content))
                    else:
                        logger.debug(
                            "NOTION: page ID %s returned error: %s",
                            pid,
                            page_content[:100] if page_content else "empty",
                        )
                except Exception:
                    logger.debug("NOTION: failed to fetch page %s (non-fatal)", pid, exc_info=True)
        else:
            logger.debug("NOTION: no user_context provided, skipping URL extraction")

        if not parts:
            return None, {"name": "Notion", "status": "error", "detail": "no docs found"}

        combined = "\n\n---\n\n".join(parts)
        detail = f"search + {len(parts) - 1} linked page(s)" if len(parts) > 1 else f"docs for '{project_name}'"
        return combined, {"name": "Notion", "status": "success", "detail": detail}
    except Exception as e:
        logger.debug("Notion context fetch failed (non-fatal)", exc_info=True)
        return None, {"name": "Notion", "status": "error", "detail": str(e)[:80]}


def _repo_integration(questionnaire: QuestionnaireState) -> str:
    """The connection key the Q17 repository scan would read through; ``""`` for a local path or none."""
    url = questionnaire.answers.get(17, "") or ""
    if not url or url == QUESTION_DEFAULTS.get(17) or "://" not in url:
        return ""
    platform = detect_platform(url) or questionnaire.answers.get(16, "")
    return {"GitHub": "github", "Azure DevOps": "azdevops"}.get(platform, "")


def _scan_repo_context(questionnaire: QuestionnaireState) -> tuple[str | None, dict]:
    """Scan the repo referenced in Q17 and return a combined context string + status.

    Calls GitHub or AzDO read tools directly as Python functions — no LLM
    ReAct loop needed, they are plain functions that return strings.
    Returns (None, status) if no URL was provided, the remote cannot be read
    (GitLab and Bitbucket have ops connectors but no repo-read tools — a local
    path scans regardless of platform), or all tool calls fail. The caller
    proceeds gracefully with reduced context when None is returned.

    Returns:
        Tuple of (context string or None, status dict with name/status/detail).

    # See docs: "Tools" — read-only tool pattern
    """
    url = questionnaire.answers.get(17, "")
    if not url or url == QUESTION_DEFAULTS.get(17):
        return None, {"name": "Repository", "status": "skipped", "detail": "no URL provided"}

    platform = questionnaire.answers.get(16, "GitHub")
    sections: list[str] = []

    try:
        if platform == "GitHub":
            from yeaboi.tools.github import github_read_readme, github_read_repo

            for fn, kwargs in [
                (github_read_repo, {"repo_url": url}),
                (github_read_readme, {"repo_url": url}),
            ]:
                result = fn.invoke(kwargs)
                if result and not result.startswith("Error:") and not result.startswith("GitHub rate limit"):
                    sections.append(result)

        elif platform == "Azure DevOps":
            from yeaboi.tools.azure_devops import azdevops_read_repo

            result = azdevops_read_repo.invoke({"repo_url": url})
            if result and not result.startswith("Error:"):
                sections.append(result)

        elif not url.startswith(("http://", "https://")):
            # Treat as a local filesystem path (user selected "local only" in Q16
            # or typed an absolute/relative path instead of a remote URL).
            from yeaboi.tools.codebase import read_codebase

            result = read_codebase.invoke({"path": url})
            if result and not result.startswith("Error:"):
                sections.append(result)

        else:
            # GitLab and Bitbucket remotes: the connectors read their pipelines,
            # but there is no repo-read tool — a local checkout path scans fine.
            return None, {
                "name": "Repository",
                "status": "skipped",
                "detail": f"{platform} remotes are not read — give a local checkout path to scan",
            }

    except Exception as e:
        return None, {"name": "Repository", "status": "error", "detail": str(e)[:80]}

    if sections:
        context = "\n\n---\n\n".join(sections)
        return context, {"name": "Repository", "status": "success", "detail": f"{platform} — scanned"}

    return None, {"name": "Repository", "status": "error", "detail": f"{platform} scan returned no data"}


def _apply_repo_signals(qs: QuestionnaireState) -> None:
    """Scan the repo once at intake and apply repo-informed suggestions.

    Runs `repo_signals.scan_repo_signals` (graceful — never raises, no-ops when
    no repo is configured/available), stashes the raw scan + low-code verdict on
    the questionnaire for `project_analyzer` to reuse, and pre-fills:

    - **Q11 (tech stack)** — the detected stack as a *suggestion* (shown when Q11
      is asked; the user confirms or overrides). Only when the user hasn't
      already given / been suggested a stack.
    - **Q12 (integrations)** — manifest-detected SDKs, auto-applied so they
      surface (editable) in the confirmation summary. Only when Q12 is unanswered.

    # See docs: "Project Intake Questionnaire" — smart intake
    """
    try:
        raw_context, signals, _status = scan_repo_signals(qs)
    except Exception:  # noqa: BLE001 — scan is best-effort; never block intake
        logger.debug("repo_signals scan failed (non-fatal)", exc_info=True)
        return

    qs._repo_context = raw_context or ""
    qs._repo_low_code = signals.low_code
    qs._repo_low_code_reason = signals.low_code_reasons[0] if signals.low_code_reasons else ""

    if signals.detected_stack and 11 not in qs.answers and 11 not in qs.suggested_answers:
        qs.suggested_answers[11] = ", ".join(signals.detected_stack)
        logger.info("repo_signals: suggested tech stack for Q11: %s", signals.detected_stack)

    if signals.integrations and 12 not in qs.answers:
        _auto_apply_extractions(qs, {12: ", ".join(signals.integrations)})
        logger.info("repo_signals: applied integrations for Q12: %s", signals.integrations)


def _sync_platform_from_url(questionnaire: QuestionnaireState) -> None:
    """Auto-update Q16 (platform) when Q17 (repo URL) is stored.

    If the URL implies a known platform and Q16 differs, Q16 is silently
    corrected so the LLM sees the right platform in the intake summary.
    """
    url = questionnaire.answers.get(17, "")
    if not url or url == QUESTION_DEFAULTS.get(17):
        return  # "No repo URL provided" or empty — skip
    platform = detect_platform(url)
    if platform and questionnaire.answers.get(16) != platform:
        questionnaire.answers[16] = platform
        questionnaire.defaulted_questions.discard(16)


def _auto_apply_extractions(questionnaire: QuestionnaireState, extracted: dict[int, str]) -> None:
    """Move extracted answers directly into questionnaire.answers (not suggestions).

    # In smart/quick mode, extracted answers are auto-accepted (not suggested
    # for confirmation). This is the key UX difference from standard mode.

    Args:
        questionnaire: The mutable QuestionnaireState to update.
        extracted: Dict mapping question numbers to extracted answer strings.
    """
    for q_num, answer in extracted.items():
        questionnaire.answers[q_num] = answer
        questionnaire.extracted_questions.add(q_num)
        questionnaire.answer_sources[q_num] = AnswerSource.EXTRACTED


def _apply_solo_defaults(questionnaire: QuestionnaireState, essential_set: frozenset[int]) -> frozenset[int]:
    """Answer the team questions for a one-developer run; Q6 leaves the essentials.

    Q6/Q7/Q30 land as DEFAULTED, never extracted, so CONDITIONAL_ESSENTIALS
    cannot promote Q7 and the gap finder asks none of them.
    """
    prior = questionnaire.suggested_answers.get(6) or questionnaire.answers.get(6)
    if prior and _parse_first_int(prior) not in (None, 1):
        logger.info("Intake: solo run overrides a described team size of %r with 1", prior)
    q30_meta = QUESTION_METADATA[30]
    solo_answers = ((6, "1"), (7, SOLO_ROLES_DEFAULT), (30, q30_meta.options[q30_meta.default_index or 0]))
    for q_num, answer in solo_answers:
        questionnaire.answers[q_num] = answer
        questionnaire.defaulted_questions.add(q_num)
        questionnaire.answer_sources[q_num] = AnswerSource.DEFAULTED
        questionnaire.extracted_questions.discard(q_num)
        questionnaire.suggested_answers.pop(q_num, None)
    # Caps the "add engineers" recommendation at the one person there is.
    questionnaire._jira_org_team_size = 1
    return essential_set - {6}


def _solo_velocity_from_profile(profile, examples: dict | None) -> str:
    """Q9 for a solo run: the profile's per-developer rate, not the whole team's."""
    vel = float(getattr(profile, "velocity_avg", 0.0) or 0.0)
    if vel <= 0:
        return ""
    contrib = (examples or {}).get("contributor_stats", [])
    heads = len(contrib) if isinstance(contrib, list) and contrib else 1
    return f"{vel / max(1, heads):.0f} points per sprint (your per-developer rate from team analysis)"


def _auto_default_remaining(
    questionnaire: QuestionnaireState,
    essential_set: frozenset[int],
    fallbacks: dict[int, str] | None = None,
) -> None:
    """Apply defaults to all non-essential, non-answered questions.

    # See docs: "Project Intake Questionnaire" — smart intake
    #
    # In smart/quick mode, optional questions are auto-defaulted so only
    # essential gaps remain. This is the mechanism that reduces 26 questions
    # to 2-4.
    #
    # For each unanswered question:
    #   - Choice Q with default_index → use that option
    #   - Free-text Q with QUESTION_DEFAULTS entry → use that default
    #   - Fallback provided → use the fallback (quick mode uses these)
    #   - Otherwise → skip it (essential Qs stay as gaps)

    Args:
        questionnaire: The mutable QuestionnaireState to update.
        essential_set: The set of essential question numbers (not defaulted).
        fallbacks: Optional fallback defaults for questions that have none
            in QUESTION_DEFAULTS (used by quick mode).
    """
    for q_num in range(1, TOTAL_QUESTIONS + 1):
        if q_num in questionnaire.answers or q_num in questionnaire.skipped_questions:
            continue  # already answered or skipped

        if q_num in essential_set:
            continue  # essential gap — will be asked interactively

        # Try standard defaults first
        meta = QUESTION_METADATA.get(q_num)
        if meta and meta.default_index is not None:
            questionnaire.answers[q_num] = meta.options[meta.default_index]
            questionnaire.defaulted_questions.add(q_num)
            questionnaire.answer_sources[q_num] = AnswerSource.DEFAULTED
        elif q_num in QUESTION_DEFAULTS:
            questionnaire.answers[q_num] = QUESTION_DEFAULTS[q_num]
            questionnaire.defaulted_questions.add(q_num)
            questionnaire.answer_sources[q_num] = AnswerSource.DEFAULTED
        elif fallbacks and q_num in fallbacks:
            questionnaire.answers[q_num] = fallbacks[q_num]
            questionnaire.defaulted_questions.add(q_num)
            questionnaire.answer_sources[q_num] = AnswerSource.DEFAULTED


# Prompt shown immediately after Q2 is answered as "Existing codebase" or "Hybrid",
# before advancing to the next question. Stored in Q17 when answered.
_Q2_REPO_URL_PROMPT = (
    "Since you're building on an existing codebase, could you share the repository URL? "
    "The agent can scan it for tech stack context.\n\n"
    "Paste a URL, or hit enter to skip"
)

# Q2 answers that trigger the repo URL follow-up.
_EXISTING_CODEBASE_ANSWERS = frozenset({"Existing codebase", "Hybrid"})


def _needs_repo_url_prompt(questionnaire: QuestionnaireState) -> bool:
    """Return True if Q2 was answered 'Existing codebase'/'Hybrid' and Q17 needs asking.

    Q17 is considered unanswered if it is absent OR was only auto-defaulted by
    _auto_default_remaining (i.e. in defaulted_questions). A defaulted Q17 means
    the LLM never saw a real URL — we still need to ask the user explicitly.
    """
    if questionnaire.answers.get(2) not in _EXISTING_CODEBASE_ANSWERS:
        return False
    # Ask if Q17 is missing entirely, or was only filled by auto-default (not by user/extraction)
    return 17 not in questionnaire.answers or 17 in questionnaire.defaulted_questions


def _derive_q15_from_q2(questionnaire: QuestionnaireState) -> None:
    """Deterministically derive Q15 (existing codebase?) from Q2 (project type).

    # Q15 asks "Does the project have an existing codebase, or is this a new build?"
    # Q2 asks "Is this a greenfield project or are you building on an existing codebase?"
    # These are redundant — Q15 can be derived from Q2 without an LLM call.
    #
    # Only applies if Q2 is answered and Q15 is not already answered.

    Args:
        questionnaire: The mutable QuestionnaireState to update.
    """
    if 15 in questionnaire.answers:
        return  # already answered

    q2_answer = questionnaire.answers.get(2, "")
    derived = Q2_TO_Q15_MAP.get(q2_answer)
    if derived:
        questionnaire.answers[15] = derived
        questionnaire.defaulted_questions.add(15)
        questionnaire.answer_sources[15] = AnswerSource.DEFAULTED


def _detect_bank_holidays_for_window(
    sprint_start_date: str | None,
    sprint_length_weeks: int,
    target_sprints: int,
) -> tuple[int, str]:
    """Auto-detect bank holidays from system locale for a sprint planning window.

    # See docs: "Scrum Standards" — capacity planning
    #
    # Uses get_bank_holidays_structured with locale auto-detection to count
    # weekday bank holidays in the planning window.

    Args:
        sprint_start_date: ISO date string for sprint start, or None for today.
        sprint_length_weeks: Length of each sprint in weeks.
        target_sprints: Number of sprints to plan for.

    Returns:
        Tuple of (count, summary_text) where summary_text is human-readable.
    """
    from yeaboi.tools.calendar_tools import _detect_country_from_locale, get_bank_holidays_structured

    country = _detect_country_from_locale()
    if not country:
        return 0, "No bank holidays detected (locale not detected)"

    from datetime import date

    start = sprint_start_date or date.today().isoformat()
    holidays = get_bank_holidays_structured(
        country_code=country,
        sprint_length_weeks=sprint_length_weeks,
        num_sprints=target_sprints,
        start_date=start,
    )

    count = len(holidays)
    if count == 0:
        return 0, "No bank holidays detected in planning window"

    # Build summary with holiday names and sprint mapping
    try:
        start_dt = parse_date(start)
    except ValueError:
        start_dt = date.today()

    sprint_days = sprint_length_weeks * 7
    names = []
    for h in holidays:
        h_date = h["date"]
        offset_days = (h_date - start_dt).days
        sprint_num = offset_days // sprint_days + 1
        day_name = h["weekday"][:3]
        names.append(f"{h['name']} (Sprint {sprint_num}, {day_name} {h_date.strftime('%-d %b')})")

    summary = f"{count} bank holiday(s): " + ", ".join(names)
    return count, summary


def _derive_q27_from_locale(questionnaire: QuestionnaireState) -> None:
    """Auto-default Q27 (sprint selection) when Jira is unavailable.

    # See docs: "Scrum Standards" — capacity planning
    #
    # In smart/quick mode without Jira (or when Jira fetch fails), Q27 is
    # auto-defaulted to "Fresh start (today)". Bank holiday detection is
    # handled separately via Q28.

    Args:
        questionnaire: The mutable QuestionnaireState to update.
    """
    if 27 in questionnaire.answers and 27 not in questionnaire.defaulted_questions:
        return  # already answered explicitly

    questionnaire.answers[27] = "Fresh start (today)"
    questionnaire.extracted_questions.add(27)
    questionnaire.defaulted_questions.discard(27)
    questionnaire.answer_sources[27] = AnswerSource.EXTRACTED


def _resolve_sprint_start_date(questionnaire: QuestionnaireState) -> str:
    """Derive the sprint start date from Q27's answer.

    # See docs: "Scrum Standards" — capacity planning
    #
    # When Jira is configured and the user selects a future sprint (e.g. Sprint 107
    # when active is Sprint 104), compute the start date by adding the sprint offset:
    #   start = today + (selected - active) × sprint_weeks
    # This ensures bank holiday detection uses the correct planning window.
    # When Q27 is "Fresh start (today)" or no Jira, we use today's date.

    Returns:
        ISO date string (YYYY-MM-DD) for the planning window start.
    """
    from datetime import date, timedelta

    q27 = questionnaire.answers.get(27, "")
    active = questionnaire._active_sprint_number
    if active is not None:
        # User selected a specific sprint via Jira — compute start date
        # using the active sprint's actual Jira start date + offset.
        selected_match = re.search(r"Sprint\s+(\d+)", q27)
        if selected_match:
            selected = int(selected_match.group(1))
            offset_sprints = selected - active
            sprint_weeks = _parse_first_int(questionnaire.answers.get(8, "2 weeks")) or 2
            # Use Jira's actual start date as anchor when available
            anchor = questionnaire._active_sprint_start_date
            if anchor:
                anchor_date = parse_date(anchor)
            else:
                anchor_date = date.today()
            return (anchor_date + timedelta(weeks=offset_sprints * sprint_weeks)).isoformat()

    # Default to today — covers "Fresh start (today)" and fallbacks
    return date.today().isoformat()


def _get_planning_window(questionnaire: QuestionnaireState):
    """Compute the planning window (start, end) from Q8/Q10/Q27.

    Returns (start_date, end_date) as datetime.date objects, or (None, None)
    if the window can't be determined (missing answers).
    """
    from datetime import timedelta

    q8 = questionnaire.answers.get(8, "")
    sprint_weeks = _parse_first_int(q8)
    if not sprint_weeks:
        return None, None

    q10 = questionnaire.answers.get(10, "")
    q10_nums = re.findall(r"\d+", q10)
    num_sprints = int(q10_nums[-1]) if q10_nums else None
    if not num_sprints:
        return None, None

    start_str = _resolve_sprint_start_date(questionnaire)
    start = parse_date(start_str)
    end = start + timedelta(weeks=sprint_weeks * num_sprints)
    return start, end


def _prepare_bank_holiday_choices(questionnaire: QuestionnaireState) -> None:
    """Detect bank holidays using the detect_bank_holidays @tool.

    # See docs: "Scrum Standards" — capacity planning
    # See docs: "Tools" — tool types, @tool decorator
    #
    # Called after Q27 (sprint selection) is resolved — regardless of whether
    # Jira is configured. Uses the actual sprint window dates:
    #   - Start date: from Jira active sprint or today for "Fresh start"
    #   - Duration: sprint length (Q8) × target sprints (Q10)
    # The tool auto-detects the user's region from system locale.
    #
    # In smart/quick mode, Q28 is NOT asked interactively — the result is
    # auto-filled and shown in the velocity breakdown at confirmation.
    # In standard mode, choice menu options are populated for Q28.

    Args:
        questionnaire: The mutable QuestionnaireState to update.
    """
    # Small-project mode skips bank-holiday planning entirely. Zero the
    # transient fields so downstream capacity code sees "no holidays" and the
    # confirmation summary shows no bank-holiday line. This single guard
    # neutralizes every _prepare_bank_holiday_choices call site for Small mode.
    if _is_small_project_mode(questionnaire.intake_mode):
        questionnaire._detected_bank_holiday_days = 0
        questionnaire._detected_bank_holidays = []
        return

    from yeaboi.tools.calendar_tools import detect_bank_holidays

    q8 = questionnaire.answers.get(8, "2 weeks")
    sprint_weeks = _parse_first_int(q8) or 2
    # Q10 uses ranges like "1–2 sprints" — use the upper bound (last number)
    # to match the project_analyzer logic. _parse_first_int would grab "1"
    # from "1–2 sprints", missing holidays that fall in the second sprint.
    q10 = questionnaire.answers.get(10, "")
    q10_nums = re.findall(r"\d+", q10)
    num_sprints = int(q10_nums[-1]) if q10_nums else 6
    start_date = _resolve_sprint_start_date(questionnaire)

    logger.debug(
        "BANK_HOLIDAY: _prepare_bank_holiday_choices called — weeks=%d sprints=%d start=%s",
        sprint_weeks,
        num_sprints,
        start_date,
    )

    # Call the @tool's underlying function directly (bypassing LangChain
    # .invoke() machinery) for reliability in all runtime contexts.
    try:
        tool_output = detect_bank_holidays.func(
            country_code="",
            sprint_length_weeks=sprint_weeks,
            num_sprints=num_sprints,
            start_date=start_date,
        )
        logger.debug("BANK_HOLIDAY: tool returned %d chars: %.200s", len(tool_output), tool_output)
    except Exception:
        logger.warning("BANK_HOLIDAY: detection failed", exc_info=True)
        tool_output = ""

    # Also fetch structured holiday data so per-sprint velocity can be computed.
    # The structured data includes individual dates we can map to sprint windows.
    from yeaboi.tools.calendar_tools import get_bank_holidays_structured

    try:
        structured = get_bank_holidays_structured(
            country_code="",
            sprint_length_weeks=sprint_weeks,
            num_sprints=num_sprints,
            start_date=start_date,
        )
        # Convert date objects to ISO strings for serialization safety
        questionnaire._detected_bank_holidays = [
            {
                "date": h["date"].isoformat() if hasattr(h["date"], "isoformat") else str(h["date"]),
                "name": h["name"],
                "weekday": h["weekday"],
            }
            for h in structured
        ]
        logger.debug("BANK_HOLIDAY: structured data — %d holidays stored", len(structured))
    except Exception:
        logger.warning("BANK_HOLIDAY: structured detection failed", exc_info=True)
        questionnaire._detected_bank_holidays = []

    # Parse the tool output to extract the count and summary.
    # The tool outputs: "Total working days lost to bank holidays: **N**"
    count = 0
    count_match = re.search(r"Total working days lost.*?\*\*(\d+)\*\*", tool_output)
    if count_match:
        count = int(count_match.group(1))

    # Build a concise summary from the tool output
    summary = ""
    if count > 0:
        holiday_lines = re.findall(r"- \d{4}-\d{2}-\d{2} \(\w+\): (.+)", tool_output)
        if holiday_lines:
            summary = f"{count} bank holiday(s): " + ", ".join(holiday_lines)
        else:
            summary = f"{count} bank holiday(s) in planning window"

    questionnaire._detected_bank_holiday_days = count
    logger.debug("BANK_HOLIDAY: count=%d summary=%r", count, summary)

    # Auto-fill Q28 answer so _extract_capacity_deductions reads it correctly.
    # In smart/quick mode this is the final answer (shown in velocity breakdown).
    # In standard mode the user can still override via the choice menu.
    if count > 0:
        questionnaire.answers[28] = summary
        questionnaire.extracted_questions.add(28)
        questionnaire.defaulted_questions.discard(28)
        questionnaire.answer_sources[28] = AnswerSource.EXTRACTED
        logger.debug("BANK_HOLIDAY: Q28 set to %r (extracted)", summary)
    else:
        questionnaire.answers[28] = "No bank holidays detected"
        questionnaire.extracted_questions.add(28)
        questionnaire.defaulted_questions.discard(28)
        questionnaire.answer_sources[28] = AnswerSource.EXTRACTED
        logger.debug("BANK_HOLIDAY: Q28 set to 'No bank holidays detected'")

    # Populate choice menu for standard mode (Q28 is still interactive there)
    if count > 0:
        questionnaire._follow_up_choices[28] = (
            f"Accept: {summary}",
            "No bank holidays",
            "Enter manually",
        )
    else:
        questionnaire._follow_up_choices[28] = (
            "No bank holidays detected — accept",
            "Enter manually",
        )


def _find_essential_gaps(questionnaire: QuestionnaireState, essential_set: frozenset[int]) -> list[int]:
    """Return sorted list of essential question numbers that are still unanswered.

    # See docs: "Project Intake Questionnaire" — conditional essentials
    #
    # In addition to the static essential set, CONDITIONAL_ESSENTIALS maps
    # questions to their prerequisites. A conditional question becomes a gap
    # when its prerequisite has a real (non-defaulted) answer — e.g., Q7
    # (team roles) is only asked when Q6 (team size) was actually answered.

    Args:
        questionnaire: The current questionnaire state.
        essential_set: The set of essential question numbers to check.

    Returns:
        Sorted list of question numbers that have no answer recorded.
    """
    # A question the user explicitly skipped is a flagged hole (shown in the
    # summary), not a gap to re-ask — without this, "skip" on a no-default
    # essential would loop forever on the same question.
    gaps = set(q for q in essential_set if q not in questionnaire.answers and q not in questionnaire.skipped_questions)

    # Conditionals: promote to essential when prerequisite is answered (not defaulted).
    # A conditional question becomes a gap when:
    #   - it has no answer at all, OR it was auto-defaulted (worth re-asking)
    #   - its prerequisite has a real (non-defaulted) answer
    for q, prereq in CONDITIONAL_ESSENTIALS.items():
        if (
            (q not in questionnaire.answers or q in questionnaire.defaulted_questions)
            and q not in questionnaire.skipped_questions
            and prereq in questionnaire.answers
            and prereq not in questionnaire.defaulted_questions
        ):
            gaps.add(q)

    return sorted(gaps)


def _resolve_adaptive_text(q_num: int, questionnaire: QuestionnaireState) -> str:
    """Return personalized question text if a template exists and dependencies are met.

    # See docs: "Project Intake Questionnaire" — adaptive question text
    #
    # Checks ADAPTIVE_QUESTION_TEMPLATES for q_num. If all referenced prior
    # answers are available (and not defaulted), formats the template with
    # those answers. Otherwise falls back to INTAKE_QUESTIONS[q_num].

    Args:
        q_num: The question number to resolve text for.
        questionnaire: The current questionnaire state with answers.

    Returns:
        The personalized question text, or the original INTAKE_QUESTIONS text.
    """
    template = ADAPTIVE_QUESTION_TEMPLATES.get(q_num)
    if not template:
        return INTAKE_QUESTIONS[q_num]

    try:
        # Build format kwargs from prior answers
        kwargs: dict[str, str] = {}
        if "{q6}" in template:
            q6 = questionnaire.answers.get(6)
            if not q6 or 6 in questionnaire.defaulted_questions:
                return INTAKE_QUESTIONS[q_num]
            kwargs["q6"] = q6
        if "{q11}" in template:
            q11 = questionnaire.answers.get(11)
            if not q11 or 11 in questionnaire.defaulted_questions:
                return INTAKE_QUESTIONS[q_num]
            kwargs["q11"] = q11
        if "{q2}" in template:
            q2 = questionnaire.answers.get(2)
            if not q2 or 2 in questionnaire.defaulted_questions:
                return INTAKE_QUESTIONS[q_num]
            kwargs["q2"] = q2
        if "{hint}" in template:
            q2 = questionnaire.answers.get(2, "")
            hint = Q2_CONSTRAINT_HINTS.get(q2, "microservices vs monolith, cloud provider")
            kwargs["hint"] = hint
        return template.format(**kwargs)
    except (KeyError, ValueError):
        return INTAKE_QUESTIONS[q_num]


def _build_gap_prompt(gaps: list[int], questionnaire: QuestionnaireState) -> tuple[str, list[int]]:
    """Build the prompt text for the next essential gap to ask.

    Asks one question at a time so each essential question gets a focused answer.
    Appends a suggestion line when the question has an extracted value in
    suggested_answers — lets the user confirm with Enter or override.

    Args:
        gaps: Sorted list of remaining essential gap question numbers.
        questionnaire: The current questionnaire state.

    Returns:
        A tuple of (prompt_text, list_of_q_nums_being_asked).
    """
    if not gaps:
        return ("", [])

    q_num = gaps[0]

    prompt = _resolve_adaptive_text(q_num, questionnaire)
    suggest = _build_suggestion_line(questionnaire, q_num)
    return (f"{prompt}{suggest}", [q_num])


def _build_skip_acknowledgment(question_num: int, *, during_probe: bool, default: str | None) -> str:
    """Build a user-facing acknowledgment message for a skipped question.

    Three cases:
    - Skip during follow-up probe → keep the original answer.
    - Default available → tell the user what assumption is being made.
    - No default → flag that we'll work without this information.

    Args:
        question_num: The question number being skipped.
        during_probe: True if the skip happened on a follow-up probe.
        default: The default value, or None if no default exists.

    Returns:
        A short acknowledgment string (no phase label or progress info).
    """
    if during_probe:
        return f"Got it, I'll keep your earlier answer for Q{question_num}."
    if default is not None:
        return f'Noted — I\'ll assume: **"{default}"** (you can change this in the summary).'
    return f"Skipped Q{question_num} — I'll work without this information."


def _check_vague_answer(question: str, answer: str, q_num: int = 0) -> tuple[str, tuple[str, ...]] | None:
    """Use the LLM to judge whether an intake answer is too vague.

    # See docs: "Project Intake Questionnaire" — follow-up probing
    #
    # Why LLM detection (not heuristics)?
    # "React" is specific for Q11 (tech stack) but vague for Q1 (project
    # description). Vagueness depends on question context — only the LLM
    # can judge this reliably across all 26 questions.
    #
    # Performance short-circuit: answers longer than 100 characters are
    # assumed to be detailed enough — skip the LLM call entirely.
    #
    # Dynamic choices: when the answer is vague, the LLM also generates 2-4
    # contextual options for the follow-up so the user can pick a number
    # instead of composing a free-text answer from scratch.
    #
    # Custom follow-up templates (Step 4): when q_num has a FOLLOW_UP_TEMPLATES
    # entry, the template is injected into the prompt as a hint. This produces
    # more targeted follow-ups for key questions.

    Args:
        question: The intake question that was asked.
        answer: The user's response to the question.
        q_num: The question number (used for custom follow-up templates).

    Returns:
        A (follow_up, choices) tuple if the answer is vague, where choices
        is a tuple of 2-4 option strings (may be empty if the LLM didn't
        provide valid choices — degrades to open-ended follow-up). Returns
        None if the answer is specific enough or on any error (graceful
        fallback — the feature never breaks the existing flow).
    """
    # Short-circuit: long answers are assumed to be detailed enough.
    if len(answer.strip()) > 100:
        return None

    # Short-circuit: numeric answers (e.g. "7" for team size, "3" for
    # sprint count) are precise by definition — never vague.
    try:
        float(answer.strip())
        return None
    except ValueError:
        pass

    # Short-circuit: "no preference" / "none" answers for tech stack (Q11) and
    # integrations (Q12). These are valid answers — the user genuinely has no
    # preference or no integrations. Without this, the LLM vagueness check
    # generates biased follow-up options (e.g. always suggesting JavaScript).
    _no_pref_patterns = frozenset(
        {
            "any",
            "anything",
            "no preference",
            "none",
            "no integrations",
            "n/a",
            "not sure yet",
            "tbd",
            "no",
            "whatever",
            "flexible",
        }
    )
    if q_num in (11, 12) and answer.strip().lower() in _no_pref_patterns:
        logger.debug("_check_vague_answer: Q%d no-preference answer accepted: %s", q_num, answer.strip())
        return None

    # Build custom follow-up hint if available for this question
    custom_hint = ""
    if q_num and q_num in FOLLOW_UP_TEMPLATES:
        custom_hint = f"\nIf vague, use this follow-up: '{FOLLOW_UP_TEMPLATES[q_num]}'\n"

    prompt = (
        "You are evaluating whether a user's answer to a project intake question "
        "is specific enough to be useful for Scrum planning.\n\n"
        f'Question: "{question}"\n'
        f'Answer: "{answer}"\n\n'
        "IMPORTANT rules for judging vagueness:\n"
        "- Named technologies, tools, or frameworks (e.g. 'Angular', 'Python', 'PostgreSQL') "
        "are SPECIFIC answers to tech stack questions — NOT vague.\n"
        "- Short answers can still be specific. 'React' is specific, 'something modern' is vague.\n"
        "- Only flag an answer as vague if it truly lacks actionable detail "
        "(e.g. 'stuff', 'not sure', 'the usual', 'some things').\n\n"
        f"{custom_hint}"
        "If the answer is too vague or generic to be actionable, respond with JSON:\n"
        '{"vague": true, "follow_up": "A specific follow-up question to get more detail", '
        '"choices": ["Option A", "Option B", "Option C"]}\n\n'
        "The follow_up MUST ask for more detail about the SAME topic as the original question. "
        "Do NOT change the subject or ask about a different topic.\n"
        "The choices array should contain 2-4 concrete, contextually relevant options "
        "that the user can pick from to clarify their answer. Each choice should be a "
        "short phrase (not a full sentence).\n\n"
        "If the answer is specific enough, respond with JSON:\n"
        '{"vague": false}\n\n'
        "Return ONLY the JSON object, no other text."
    )

    try:
        response = _invoke_json(prompt)
        raw = response.content.strip()

        # Strip markdown code fences that LLMs sometimes wrap JSON in
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()

        parsed = json.loads(raw)

        if not isinstance(parsed, dict):
            return None

        if parsed.get("vague") is True:
            follow_up = parsed.get("follow_up", "")
            if isinstance(follow_up, str) and follow_up.strip():
                # Parse and validate choices — graceful fallback to empty tuple
                choices = _parse_follow_up_choices(parsed.get("choices"))
                return (follow_up.strip(), choices)

        return None

    except Exception as exc:
        if _should_reraise_llm_error(exc):
            raise
        # Graceful fallback: if anything goes wrong (bad JSON, LLM timeout,
        # network error), accept the answer as-is. The feature never breaks
        # the existing flow.
        logger.debug("Vague-answer check failed, accepting answer as-is", exc_info=True)
        return None


def _parse_follow_up_choices(raw_choices: object) -> tuple[str, ...]:
    """Validate and clamp LLM-generated follow-up choices to 2-4 non-empty strings.

    Graceful degradation: if the input is missing, not a list, or contains
    non-string / empty items, returns an empty tuple (the follow-up degrades
    to the current open-ended behavior — no menu shown).

    Args:
        raw_choices: The raw "choices" value from the LLM JSON response.

    Returns:
        A tuple of 2-4 non-empty strings, or an empty tuple if invalid.
    """
    if not isinstance(raw_choices, list):
        return ()

    # Filter to non-empty strings only
    cleaned = [c.strip() for c in raw_choices if isinstance(c, str) and c.strip()]

    # Enforce 2-4 range — too few is useless, clamp excess
    if len(cleaned) < 2:
        return ()
    return tuple(cleaned[:4])


def _validate_cross_questions(answers: dict[int, str]) -> list[ValidationWarning]:
    """Run deterministic cross-question validation rules on completed answers.

    # See docs: "Project Intake Questionnaire" — cross-question validation
    #
    # Three rules (no LLM call):
    # 1. Greenfield + repo URL → contradiction
    # 2. Timeline > 6 months → advisory
    # 3. Velocity / team_size outside 2-15 range → sanity check

    Args:
        answers: The questionnaire answers dict (q_num → answer string).

    Returns:
        A list of ValidationWarning instances (may be empty).
    """
    warnings: list[ValidationWarning] = []

    # Rule 1: Greenfield + repo URL
    q2 = answers.get(2, "")
    q17 = answers.get(17, "")
    if q2.lower().strip() == "greenfield" and q17 and "http" in q17.lower():
        warnings.append(
            ValidationWarning(
                question_nums=(2, 17),
                message="Q2 says Greenfield but Q17 has a repo URL — did you mean Existing codebase?",
            )
        )

    # Rule 2: Timeline > 6 months
    q8 = answers.get(8, "")
    q10 = answers.get(10, "")
    sprint_weeks = _parse_first_int(q8) or 2
    q10_nums = re.findall(r"\d+", q10)
    if q10_nums:
        target_sprints = int(q10_nums[-1])
        total_weeks = sprint_weeks * target_sprints
        if total_weeks > 26:
            months = round(total_weeks / 4.3, 1)
            warnings.append(
                ValidationWarning(
                    question_nums=(8, 10),
                    message=f"Plan spans ~{months} months. Consider breaking into phases.",
                    severity="info",
                )
            )

    # Rule 3: Velocity sanity check
    q6 = answers.get(6, "")
    q9 = answers.get(9, "")
    team_size = _parse_first_int(q6)
    velocity = _parse_first_int(q9)
    if team_size and team_size > 0 and velocity and velocity > 0:
        ratio = velocity / team_size
        if ratio < 2 or ratio > 15:
            warnings.append(
                ValidationWarning(
                    question_nums=(6, 9),
                    message=(
                        f"Velocity of {velocity} with {team_size} engineers = "
                        f"{ratio:.1f} pts/engineer. Typical range: 3-10."
                    ),
                )
            )

    return warnings


def _build_intake_summary(questionnaire: QuestionnaireState) -> str:
    """Build a formatted markdown summary of all intake answers grouped by phase.

    Called once when the questionnaire is complete. The summary is sent as an
    AIMessage so it appears in the conversation history for the main agent to
    reference when generating Scrum artifacts.

    Four-tier answer display:
    - Extracted: `> {answer}  *(from your description)*`
    - Normal answer: `> {answer}`
    - Defaulted: `> {answer}  *(assumed default)*`
    - Skipped with no default: `> _skipped (no default available)_`
    """
    sections: list[str] = []
    for phase, (start, end) in PHASE_QUESTION_RANGES.items():
        label = PHASE_LABELS[phase]
        lines = [f"## {label}\n"]
        for q_num in range(start, end + 1):
            question = INTAKE_QUESTIONS[q_num]
            answer = questionnaire.answers.get(q_num)
            lines.append(f"**Q{q_num}.** {question}")
            if answer is None:
                # No answer recorded and not defaulted — truly skipped
                lines.append("> _skipped (no default available)_\n")
            elif q_num in questionnaire.extracted_questions:
                source = "from SCRUM.md" if q_num in questionnaire._scrum_md_questions else "from your description"
                lines.append(f"> {answer}  *({source})*\n")
            elif q_num in questionnaire.defaulted_questions:
                lines.append(f"> {answer}  *(assumed default)*\n")
            else:
                lines.append(f"> {answer}\n")
        sections.append("\n".join(lines))

    # Stats header — shows answer provenance at a glance
    # Use answer_sources for the confidence breakdown when available,
    # fall back to the legacy tracking sets for backward compatibility.
    sources = questionnaire.answer_sources
    if sources:
        num_direct = sum(1 for v in sources.values() if v == AnswerSource.DIRECT)
        num_extracted = sum(1 for v in sources.values() if v == AnswerSource.EXTRACTED)
        num_defaulted = sum(1 for v in sources.values() if v == AnswerSource.DEFAULTED)
        num_probed = sum(1 for v in sources.values() if v == AnswerSource.PROBED)
    else:
        num_direct = len(
            questionnaire.answers.keys() - questionnaire.extracted_questions - questionnaire.defaulted_questions
        )
        num_extracted = len(questionnaire.extracted_questions)
        num_defaulted = len(questionnaire.defaulted_questions)
        num_probed = len(questionnaire.probed_questions)
    stats_parts: list[str] = []
    if num_direct > 0:
        stats_parts.append(f"{num_direct} direct")
    if num_extracted > 0:
        stats_parts.append(f"{num_extracted} extracted")
    if num_defaulted > 0:
        stats_parts.append(f"{num_defaulted} defaulted")
    if num_probed > 0:
        stats_parts.append(f"{num_probed} probed")
    stats_line = f"**{' | '.join(stats_parts)}**\n\n" if stats_parts else ""

    # Cross-question validation — advisory warnings for contradictions/unrealistic combos
    validation_warnings = _validate_cross_questions(questionnaire.answers)
    warnings_block = ""
    if validation_warnings:
        warning_lines = ["**Heads up** — a few things worth double-checking:\n"]
        for w in validation_warnings:
            q_refs = ", ".join(f"Q{q}" for q in w.question_nums)
            warning_lines.append(f"- {w.message} (edit {q_refs})")
        warnings_block = "\n".join(warning_lines) + "\n\n"

    header = (
        f"# Project Intake Summary\n\n{stats_line}{warnings_block}"
        "Here's a summary of your answers. I'll use this to generate your Scrum plan.\n"
    )
    body = "\n---\n\n".join(sections)

    # Prior art — the existing repositories the user accepted as reference.
    # Rendered above the answers because it is the one part of the summary the
    # user chose rather than typed, and the part they will scan for.
    accepted = questionnaire._prior_art_accepted
    if accepted:
        prior_lines = ["## Prior art\n", "Existing repositories you said are relevant:\n"]
        for candidate in accepted:
            name = candidate.get("name", "")
            pitch = candidate.get("pitch") or ()
            prior_lines.append(f"- **{name}**" + (f" — {pitch[0]}" if pitch else ""))
        body = "\n".join(prior_lines) + "\n\n---\n\n" + body

    # Add a footer when defaults were used so the user knows to review assumptions
    if num_defaulted > 0:
        footer = (
            f"\n\n**Note:** {num_defaulted} answer(s) above are assumed defaults (marked with *assumed default*)."
            " You can revise these before I generate the Scrum plan."
        )
        result = header + body + footer
    else:
        result = header + body

    # Append velocity breakdown section if team size is available.
    # See docs: "Scrum Standards" — capacity planning
    extracted = _extract_team_and_velocity(questionnaire)
    if extracted:
        ts = extracted["team_size"]
        vel = extracted["velocity_per_sprint"]
        velocity_was_calculated = extracted.get("_velocity_was_calculated", False)
        velocity_source = "estimated" if velocity_was_calculated else "manual"
        # Check if velocity came from Jira (Q9 has "from Jira" marker)
        q9 = questionnaire.answers.get(9, "")
        if "from Jira" in q9 or "from jira" in q9:
            velocity_source = "jira"

        sprint_weeks = _parse_first_int(questionnaire.answers.get(8, "2 weeks")) or 2
        q10_nums = re.findall(r"\d+", questionnaire.answers.get(10, ""))
        target = int(q10_nums[-1]) if q10_nums else 6
        capacity = _extract_capacity_deductions(questionnaire)

        # Use override if set
        override = questionnaire._velocity_override
        if _is_small_project_mode(questionnaire.intake_mode):
            # Small-project mode does no capacity planning — net equals gross
            # velocity and there is no bank-holiday / PTO / discovery breakdown.
            net_vel = vel
            breakdown = f"**Small project** — no bank-holiday or capacity deductions (using {vel} pts/sprint)."
        else:
            net_vel, breakdown = _build_velocity_breakdown(
                velocity_per_sprint=vel,
                velocity_source=velocity_source,
                team_size=ts,
                sprint_length_weeks=sprint_weeks,
                target_sprints=target,
                bank_holiday_days=capacity["capacity_bank_holiday_days"],
                planned_leave_days=capacity["capacity_planned_leave_days"],
                unplanned_leave_pct=capacity["capacity_unplanned_leave_pct"],
                onboarding_engineer_sprints=capacity["capacity_onboarding_engineer_sprints"],
                ktlo_engineers=capacity["capacity_ktlo_engineers"],
                discovery_pct=capacity["capacity_discovery_pct"],
                planned_leave_entries=list(questionnaire._planned_leave_entries),
            )

        if override is not None:
            display_vel = override
            breakdown += f"\n\n**User override: {override} pts/sprint**"
        else:
            display_vel = net_vel

        result += f"\n\n---\n\n{breakdown}"

        # Per-sprint breakdown — when bank holidays or PTO hit specific sprints,
        # show which sprints are impacted so the user sees the uneven capacity.
        from datetime import date

        holidays_by_sprint = _assign_holidays_to_sprints(
            questionnaire._detected_bank_holidays,
            date.today().isoformat(),
            sprint_weeks,
            target,
        )
        leave_by_sprint = _assign_leave_to_sprints(
            questionnaire._planned_leave_entries,
            date.today().isoformat(),
            sprint_weeks,
            target,
        )
        if holidays_by_sprint or leave_by_sprint:
            sprint_caps = _compute_per_sprint_velocities(
                team_size=ts,
                velocity_per_sprint=vel,
                sprint_length_weeks=sprint_weeks,
                target_sprints=target,
                holidays_by_sprint=holidays_by_sprint,
                planned_leave_days=capacity["capacity_planned_leave_days"],
                unplanned_leave_pct=capacity["capacity_unplanned_leave_pct"],
                onboarding_engineer_sprints=capacity["capacity_onboarding_engineer_sprints"],
                ktlo_engineers=capacity["capacity_ktlo_engineers"],
                discovery_pct=capacity["capacity_discovery_pct"],
                leave_by_sprint=leave_by_sprint,
            )
            has_uneven = any(sc["bank_holiday_days"] > 0 or sc.get("pto_days", 0) > 0 for sc in sprint_caps)
            if has_uneven:
                per_sprint_lines = ["\n**Per-sprint capacity:**"]
                total_pts = 0
                for sc in sprint_caps:
                    nv = sc["net_velocity"]
                    total_pts += nv
                    label = f"Sprint {sc['sprint_index'] + 1}"
                    annotations = []
                    if sc["bank_holiday_names"]:
                        names = ", ".join(sc["bank_holiday_names"])
                        annotations.append(f"−{sc['bank_holiday_days']}d: {names}")
                    if sc.get("pto_days", 0) > 0:
                        pto_names = ", ".join(f"{e['person']} {e['days']}d" for e in sc.get("pto_entries", []))
                        annotations.append(f"PTO: {pto_names}")
                    if annotations:
                        per_sprint_lines.append(f"  {label}: **{nv} pts** ({'; '.join(annotations)})")
                    else:
                        per_sprint_lines.append(f"  {label}: **{nv} pts**")
                per_sprint_lines.append(f"  Total: **{total_pts} pts** across {len(sprint_caps)} sprints")
                result += "\n".join(per_sprint_lines)

        result += f"\n\n[1] Accept {display_vel} pts/sprint"
        result += "\n[2] Override — enter a custom velocity"

    return result


_CONFIRM_PROMPT = (
    "\n\nPlease review the summary above. Select a velocity option above, or type **edit Q6** to change an answer."
)


# ── Review intent detection ─────────────────────────────────────────
# See docs: "Guardrails" — human-in-the-loop pattern
#
# After each generation node (feature_generator, story_writer, task_decomposer,
# sprint_planner), the user reviews the output and chooses Accept / Edit / Reject.
# This is deterministic keyword matching — same pattern as _is_skip_intent() and
# _is_confirm_intent(). No LLM call needed.
#
# Design: unrecognized text defaults to REJECT with the full text as feedback.
# This is the most natural UX — typing "add a security feature" without a keyword
# prefix is clearly rejection feedback.

_ACCEPT_KEYWORDS: frozenset[str] = frozenset(
    {"accept", "approve", "ok", "yes", "y", "looks good", "lgtm", "proceed", "continue", "good", "fine"}
)

_REJECT_KEYWORDS: frozenset[str] = frozenset({"reject", "redo", "regenerate", "again", "try again", "no", "n"})

_EDIT_BARE_KEYWORDS: frozenset[str] = frozenset({"edit", "change", "modify", "update", "adjust", "tweak", "revise"})

_EDIT_PREFIXES: tuple[str, ...] = (
    "edit:",
    "change:",
    "modify:",
    "update:",
    "adjust:",
    "tweak:",
    "revise:",
    "edit ",
    "change ",
    "modify ",
    "update ",
    "adjust ",
    "tweak ",
    "revise ",
)


def _parse_review_intent(text: str) -> tuple[ReviewDecision, str]:
    """Detect the user's review intent from their input text.

    # See docs: "Guardrails" — human-in-the-loop pattern
    #
    # Three possible outcomes:
    # - ACCEPT: user approves the output, pipeline continues
    # - EDIT: user wants specific modifications (feedback extracted)
    # - REJECT: user wants a full regeneration (feedback extracted)
    #
    # Fallback: unrecognized text → REJECT with full text as feedback.
    # This is the most natural UX — "add a security feature" without a prefix
    # is clearly rejection feedback.

    Args:
        text: The raw user input text.

    Returns:
        A tuple of (ReviewDecision, feedback_string). Feedback is "" for
        accept, and the user's feedback text for edit/reject.
    """
    normalized = text.strip().lower()

    # Check accept keywords first — exact match only
    if normalized in _ACCEPT_KEYWORDS:
        return (ReviewDecision.ACCEPT, "")

    # Check edit prefixes — extract feedback after the prefix
    for prefix in _EDIT_PREFIXES:
        if normalized.startswith(prefix):
            feedback = text.strip()[len(prefix) :].strip()
            return (ReviewDecision.EDIT, feedback)

    # Check bare edit keywords — exact match returns EDIT with no feedback
    # (the REPL will prompt for feedback). This handles "2" → "edit" from
    # the numbered menu, as well as typing "edit" directly.
    if normalized in _EDIT_BARE_KEYWORDS:
        return (ReviewDecision.EDIT, "")

    # Check reject keywords — may have inline feedback after ":"
    for keyword in _REJECT_KEYWORDS:
        if normalized == keyword:
            return (ReviewDecision.REJECT, "")
        if normalized.startswith(keyword + ":"):
            feedback = text.strip()[len(keyword) + 1 :].strip()
            return (ReviewDecision.REJECT, feedback)

    # Fallback: unrecognized text → REJECT with the full text as feedback
    return (ReviewDecision.REJECT, text.strip())


def route_entry(state: ScrumState) -> str:
    """Route from START: seven-way branch based on pipeline progress.

    # See docs: "Agentic Blueprint Reference" — conditional edges
    #
    # This is a conditional edge function that runs at the START of every
    # graph invocation. It checks questionnaire, analysis, feature, story,
    # task, and sprint state for seven-way routing:
    #   - Questionnaire not completed → "project_intake" node
    #   - Questionnaire completed, no project_analysis → "project_analyzer"
    #   - Analysis done, no features → "feature_generator"
    #   - Features done, no stories → "story_writer"
    #   - Stories done, no tasks → "task_decomposer"
    #   - Tasks done, no sprints → "sprint_planner"
    #   - Sprints populated → "agent" node (main ReAct loop)
    #
    # Why check `not state.get("sprints")`?
    # Same pattern as tasks — empty list is falsy, None is falsy. Once the
    # sprint_planner populates the list, it becomes truthy and we route to the agent.
    """
    questionnaire = state.get("questionnaire")
    if questionnaire is None or not questionnaire.completed:
        return "project_intake"
    if state.get("project_analysis") is None:
        return "project_analyzer"
    if not state.get("features"):
        # When the analyzer determined the project is too small for features,
        # skip feature generation and use a sentinel feature instead.
        # See docs: "Scrum Standards" — feature generation
        analysis = state.get("project_analysis")
        if analysis and analysis.skip_features:
            return "feature_skip"
        return "feature_generator"
    if not state.get("stories"):
        return "story_writer"
    if not state.get("tasks"):
        return "task_decomposer"
    if not state.get("sprints"):
        return "sprint_planner"
    return "agent"


def _show_summary_or_pto(questionnaire: QuestionnaireState, prefix: str = "") -> dict:
    """Show the confirmation summary OR trigger the PTO sub-loop first.

    # See docs: "Scrum Standards" — capacity planning
    #
    # In smart and standard modes, PTO is always asked interactively before the
    # confirmation summary because it's per-person knowledge the system can't
    # auto-detect. In quick mode, auto-default to "no planned leave".
    #
    # After PTO is resolved (or skipped), the confirmation summary is shown
    # with the velocity choice menu as usual.
    """
    # Check if PTO should be asked before the summary. Quick and small_project
    # modes skip PTO — small_project does no capacity planning at all, quick
    # auto-defaults to "no planned leave".
    if (
        questionnaire.intake_mode not in ("quick", "small_project")
        and not questionnaire._leave_input_stage
        and not questionnaire._awaiting_leave_input
        and not questionnaire._planned_leave_entries
    ):
        questionnaire._awaiting_leave_input = True
        questionnaire._leave_input_stage = "ask"
        # Set awaiting_confirmation so the PTO handler runs in the confirmation gate
        questionnaire.awaiting_confirmation = True
        # Show PTO under Q28 (Holidays & leave) in the accordion — semantically
        # PTO belongs with the leave/holidays section, not onboarding (Q30).
        questionnaire.current_question = 28
        return {
            "questionnaire": questionnaire,
            "messages": [
                AIMessage(
                    content=f"{prefix}Does anyone have planned leave (PTO/vacation) during the planning window?\n\n"
                    "[1] Yes\n[2] No"
                )
            ],
        }

    # ── Prior-art sub-loop ────────────────────────────────────────────
    # See docs: "Project Intake Questionnaire" — prior art
    #
    # A greenfield project is planned from a blank page even when the team
    # already owns most of the pieces. Before the summary is signed off, offer
    # the repositories analysis found and let the user rule each in or out.
    # Same shape as the PTO sub-loop above: a stage flag on the questionnaire,
    # awaiting_confirmation set so the handler runs in the confirmation gate.
    if not questionnaire._prior_art_stage and not questionnaire._analysis_enabled:
        # Analysis is toggled off for this run: the shortlist would auto-detect
        # a profile even with a blank id, so the stage settles without a scan.
        logger.info("analysis dep off — skipping the prior-art offer")
        questionnaire._prior_art_stage = "done"
    if not questionnaire._prior_art_stage and _prior_art_applies(questionnaire):
        prompt = _start_prior_art(questionnaire)
        if prompt is not None:
            questionnaire.current_question = TOTAL_QUESTIONS + 1
            questionnaire.awaiting_confirmation = True
            return {
                "questionnaire": questionnaire,
                "messages": [AIMessage(content=f"{prefix}{prompt}")],
            }

    # PTO already handled (or quick mode) — show the confirmation summary
    questionnaire.current_question = TOTAL_QUESTIONS + 1
    questionnaire.awaiting_confirmation = True
    summary = _build_intake_summary(questionnaire)
    return {
        "questionnaire": questionnaire,
        "messages": [AIMessage(content=f"{prefix}{summary}{_CONFIRM_PROMPT}")],
        "pending_review": "project_intake",
    }


# ---------------------------------------------------------------------------
# Prior art — the team's own repositories, offered as reference for a new build
# ---------------------------------------------------------------------------
# See docs: "Project Intake Questionnaire" — prior art

# The one grammar every surface answers in — the chat widget submits it, and
# REPL / form / headless / typed-chat users type it against the numbered list
# the prompt shows. Documented in _prior_art_prompt and _PRIOR_ART_GRAMMAR_HINT.
_PRIOR_ART_GRAMMAR_HINT = (
    'Reply with the numbers that are relevant (e.g. "1 3"), "all", or "none". '
    'Prefix ! to never suggest a repo again (e.g. "1 !2").'
)


def _prior_art_applies(questionnaire: QuestionnaireState) -> bool:
    """Whether this project gets the prior-art step at all (greenfield only)."""
    from yeaboi.agent import prior_art

    return prior_art.applies(questionnaire.answers)


def _start_prior_art(questionnaire: QuestionnaireState) -> str | None:
    """Run the scan and open the sub-loop. None means "nothing to ask".

    Returns the one batch prompt when there is a shortlist. When there
    is not, the stage is closed out and the reason recorded so the card can say
    which of "no profile" / "profile too old" / "nothing matched" happened —
    going quiet would leave the user unable to tell a gap from a verdict.
    """
    from yeaboi.agent import prior_art

    try:
        result = prior_art.shortlist(
            questionnaire.answers,
            profile_id=questionnaire._analysis_profile_id,
        )
    except Exception as exc:
        if _should_reraise_llm_error(exc):
            raise
        logger.warning("prior_art: scan failed, skipping the step", exc_info=True)
        questionnaire._prior_art_stage = "done"
        return None
    if not result.candidates:
        questionnaire._prior_art_empty_reason = result.empty_reason
        logger.info("prior_art: no candidates (%s)", result.empty_reason)
        # Say so rather than going quiet: a user who has never run Team
        # Analysis, one whose profile predates this feature, and one whose
        # repositories genuinely do not match are three different situations
        # with three different things to do about them, and silence reads as
        # "your repos were considered and rejected" in all three.
        if result.message:
            questionnaire._prior_art_stage = "empty"
            return result.message
        questionnaire._prior_art_stage = "done"
        return None
    # `stack` is a property, so asdict() drops it — add it explicitly or the
    # prompt and the promoted reference silently fall back to languages alone.
    questionnaire._prior_art_candidates = [{**dataclasses.asdict(c), "stack": list(c.stack)} for c in result.candidates]
    questionnaire._prior_art_index = 0
    questionnaire._prior_art_stage = "ask"
    logger.info("prior_art: asking about %d repositories", len(result.candidates))
    return _prior_art_prompt(questionnaire)


def _accepted_prior_art(questionnaire: QuestionnaireState) -> tuple:
    """The accepted candidates as durable PriorArtRefs."""
    from yeaboi.agent.state import PriorArtRef

    refs = []
    for candidate in questionnaire._prior_art_accepted:
        refs.append(
            PriorArtRef(
                key=str(candidate.get("key", "") or ""),
                name=str(candidate.get("name", "") or ""),
                url=str(candidate.get("url", "") or ""),
                platform=str(candidate.get("platform", "") or ""),
                pitch=tuple(candidate.get("pitch") or ()),
                stack=tuple(candidate.get("stack") or candidate.get("languages") or ()),
            )
        )
    return tuple(refs)


def _prior_art_prompt(questionnaire: QuestionnaireState) -> str:
    """The one batch question listing every shortlisted repository.

    All candidates appear numbered with condensed pitches, because this same
    markdown is the whole interface on the text surfaces (REPL, form intake,
    typed chat) — the answer grammar it documents is what
    _parse_prior_art_answer accepts, so nothing depends on the chat widget.
    """
    candidates = questionnaire._prior_art_candidates
    if not candidates:
        return ""
    plural = "repository" if len(candidates) == 1 else "repositories"
    lines = [
        f"**Prior art** — you already have {len(candidates)} {plural} that might be relevant.",
        "",
    ]
    for position, candidate in enumerate(candidates, start=1):
        stack = candidate.get("stack") or candidate.get("languages") or ()
        title = f"**{position}. {candidate.get('name', '')}**"
        if stack:
            title += f" _({', '.join(stack)})_"
        lines.append(title)
        for bullet in (candidate.get("pitch") or ())[:2]:
            lines.append(f"   - {bullet}")
    lines += ["", _PRIOR_ART_GRAMMAR_HINT]
    return "\n".join(lines)


def _parse_prior_art_answer(text: str, count: int) -> tuple[set[int], set[int]] | None:
    """Parse a batch verdict into 0-based (relevant, banned) index sets.

    Grammar (case-insensitive, tokens split on commas/whitespace):
      - ``N``            → relevant (1-based against the prompt's numbering)
      - ``!N``           → banned — never suggest again (permanent ledger write)
      - ``all``          → every candidate relevant
      - ``none``/``skip``/empty → nothing relevant, nothing banned
    A ban wins over a select of the same index. Any out-of-range digit or
    unrecognised token returns None so the caller can re-prompt — silently
    dropping half an answer would record a verdict the user never gave.
    """
    words = [w for w in re.split(r"[,\s]+", text.strip().lower()) if w]
    relevant: set[int] = set()
    banned: set[int] = set()
    for word in words:
        if word in ("none", "skip"):
            # An explicit "nothing" — inert beside other tokens rather than an
            # error, so "none" typed after second thoughts still parses.
            continue
        if word == "all":
            relevant.update(range(count))
            continue
        is_ban = word.startswith("!")
        digits = word[1:] if is_ban else word
        # isdecimal, not isdigit: isdigit accepts superscripts ("²") that
        # int() then refuses, turning a typo into a raised turn error.
        if not digits.isdecimal():
            return None
        index = int(digits) - 1
        if not 0 <= index < count:
            return None
        (banned if is_ban else relevant).add(index)
    return relevant - banned, banned


def _finish_prior_art(questionnaire: QuestionnaireState) -> dict:
    """Close the sub-loop: persist verdicts, then show the summary.

    Only explicit verdicts reach the permanent ledger: accepted → up, banned
    (``!N``) → down. A candidate the user simply did not tick is passed over
    for this project only and appears in neither list — writing it down would
    make one quick Enter silently blacklist the whole shortlist forever.
    """
    from yeaboi.agent import prior_art_feedback

    questionnaire._prior_art_stage = "done"
    for rejection in questionnaire._prior_art_rejected:
        prior_art_feedback.apply_verdict(
            repo_key=rejection.get("key", ""),
            verdict=prior_art_feedback.VERDICT_DOWN,
            reason=rejection.get("reason", ""),
            repo_name=rejection.get("name", ""),
            project=str(questionnaire.answers.get(1, ""))[:120],
        )
    for accepted in questionnaire._prior_art_accepted:
        prior_art_feedback.apply_verdict(
            repo_key=accepted.get("key", ""),
            verdict=prior_art_feedback.VERDICT_UP,
            repo_name=accepted.get("name", ""),
            project=str(questionnaire.answers.get(1, ""))[:120],
        )
    logger.info(
        "prior_art: %d accepted, %d banned",
        len(questionnaire._prior_art_accepted),
        len(questionnaire._prior_art_rejected),
    )
    return _show_summary_or_pto(questionnaire)


def apply_size_switch(graph_state: dict, target_mode: str) -> None:
    """Reset session state to switch intake mode mid-session, preserving answers.

    See README: "Guardrails" — human-in-the-loop (advisory)

    Generalizes the original Small → Large switch to both directions
    ("smart" | "small_project"), driven by the /large and /small chat
    commands as well as the oversize advisory. Preserves the questionnaire
    ANSWERS and project description (so the user re-types nothing), flips the
    mode, and clears the analysis + downstream artifacts so the pipeline
    regenerates them. The _reopen_for_epic flag (name kept for persistence
    compatibility — it now means "reopen after a size switch" in either
    direction) tells project_intake to ask only the still-missing essentials
    for the new mode on the next pass.

    When switching DOWN to small_project, the smart-only capacity transients
    (PTO sub-loop, velocity override, detected bank holidays) and the derived
    capacity_* state are also cleared: small mode skips the capacity
    questions, and _extract_capacity_deductions returns all-zeros for it at
    confirmation, so stale smart-mode values must not leak into the leaner
    plan. Extra answered questions from the wider mode are inert — small-mode
    artifacts depend only on the small essential set.
    """
    if target_mode not in ("smart", "small_project"):
        raise ValueError(f"Unsupported intake mode for size switch: {target_mode!r}")
    qs = graph_state.get("questionnaire")
    if qs is not None:
        qs.intake_mode = target_mode
        qs.completed = False
        qs.awaiting_confirmation = False
        qs.editing_question = None
        qs._reopen_for_epic = True
        # Prior art resets in both directions, unlike the PTO block below: the
        # guard skips a stage that is already open, but the confirmation-gate
        # handler still claims the turn — so a switch made mid-loop would eat
        # the user's first reply at the new summary and answer a card about a
        # repository they can no longer see. Accepted refs are already on
        # graph_state["prior_art"], which the confirm path falls back to.
        qs._prior_art_stage = ""
        qs._prior_art_candidates = []
        qs._prior_art_index = 0
        qs._prior_art_accepted = []
        qs._prior_art_rejected = []
        if target_mode == "small_project":
            qs._planned_leave_entries = []
            qs._awaiting_leave_input = False
            qs._leave_input_stage = ""
            qs._leave_input_buffer = {}
            qs._velocity_override = None
            qs._awaiting_velocity_input = False
            qs._detected_bank_holiday_days = 0
            qs._detected_bank_holidays = []
    graph_state["_intake_mode"] = target_mode
    # Drop the analysis + all generated artifacts + review bookkeeping so the
    # pipeline re-runs cleanly under the new mode's rules. Answers are untouched.
    keys_to_clear = [
        "project_analysis",
        "features",
        "stories",
        "tasks",
        "sprints",
        "pending_review",
        "_small_project_oversized",
        "_epic_reviewed",
        "last_review_decision",
        "last_review_feedback",
    ]
    if target_mode == "small_project":
        keys_to_clear += [
            "capacity_bank_holiday_days",
            "capacity_planned_leave_days",
            "capacity_unplanned_leave_pct",
            "capacity_onboarding_engineer_sprints",
            "capacity_ktlo_engineers",
            "capacity_discovery_pct",
            "net_velocity_per_sprint",
            "velocity_source",
            "sprint_start_date",
            "sprint_capacities",
            "planned_leave_entries",
        ]
    for key in keys_to_clear:
        graph_state.pop(key, None)


def apply_epic_switch(graph_state: dict) -> None:
    """Switch Small project → Large intake (kept as the historical entry point)."""
    apply_size_switch(graph_state, "smart")


def _reopen_intake_for_epic(state: ScrumState, questionnaire: QuestionnaireState) -> dict:
    """Ask the next Large-mode essential after a Small → Large switch (no answer to record).

    Reuses the same gap-finding helpers as the smart-mode intake loop. Dynamic
    Q27 tracker menus are populated on the subsequent normal gap pass, so here we
    just surface the first missing essential (or the summary if none remain).
    """
    questionnaire._reopen_for_epic = False
    questionnaire.awaiting_confirmation = False
    # The switch works in both directions now (/large and /small): the label
    # and the gap set both derive from whatever mode we just switched INTO.
    mode_label = "Small" if _is_small_project_mode(questionnaire.intake_mode) else "Large"
    essential_set = _essentials_for_mode(questionnaire.intake_mode)
    gaps = _find_essential_gaps(questionnaire, essential_set)
    if not gaps:
        # Every essential for the new mode is already answered — go straight to
        # the summary (with bank-holiday detection + PTO where the mode uses them).
        if not _is_small_project_mode(questionnaire.intake_mode):
            _prepare_bank_holiday_choices(questionnaire)
        return _show_summary_or_pto(
            questionnaire,
            prefix=f"Switched to **{mode_label}** — using your existing answers.\n\n",
        )
    prompt_text, q_nums = _build_gap_prompt(gaps, questionnaire)
    questionnaire._pending_merged_questions = q_nums
    questionnaire.current_question = q_nums[0]
    logger.info("Size switch to %s: asking remaining essentials %s", questionnaire.intake_mode, sorted(gaps))
    tail = "Just a few more questions:" if mode_label == "Small" else "Just a few more questions for the fuller plan:"
    return {
        "questionnaire": questionnaire,
        "messages": [
            AIMessage(content=f"Switched to **{mode_label}** — I kept all your answers. {tail}\n\n{prompt_text}")
        ],
    }


def project_intake(state: ScrumState) -> dict:
    """LangGraph node: ask one intake question per graph invocation.

    # See docs: "Scrum Standards" — questionnaire phases
    # See docs: "Agentic Blueprint Reference" — node return format
    #
    # How this works:
    # 1. First call (no questionnaire in state) → initialize QuestionnaireState,
    #    ask Q1 with a phase header.
    # 2. Subsequent calls → record the user's answer from the last HumanMessage,
    #    advance to the next question, and ask it.
    # 3. After Q26 is answered → mark completed, return a formatted summary
    #    of all answers grouped by phase.
    #
    # Why one question per invocation (not all at once)?
    # Each graph.invoke() call processes one user turn. The REPL collects the
    # user's answer, then calls graph.invoke() again with the updated state.
    # This gives the user a conversational, one-question-at-a-time experience
    # rather than a wall of 26 questions.
    #
    # Why deterministic questions (not LLM-generated)?
    # Using a predefined question list ensures all 26 questions are asked
    # consistently, in order, and enables progress tracking. Follow-up probing
    # (a separate TODO) will add LLM-powered refinement later.
    """
    questionnaire = state.get("questionnaire")
    if questionnaire is not None:
        # Refresh the stash on every invoke, so a resumed session — whose
        # transients were dropped — reaches the prior-art step with a profile.
        questionnaire._analysis_profile_id = _effective_analysis_profile_id(state)
        questionnaire._analysis_enabled = _wants_dep(state, "analysis")

    # ── Tracker choice resolution ────────────────────────────────────
    # When both Jira and Azure DevOps are configured, the user picks one
    # for velocity/sprint data before intake begins. This resolves that choice.
    if questionnaire is not None and questionnaire._awaiting_tracker_choice:
        from yeaboi import trackers

        last_msg = state["messages"][-1]
        choice_text = last_msg.content.strip().lower() if isinstance(last_msg, HumanMessage) else ""
        # Resolve against the same configured list the prompt offered, so a
        # typed "2" means the second option the user actually read.
        options = trackers.configured() or ["jira"]
        questionnaire._preferred_tracker = trackers.resolve_choice(choice_text, options)
        questionnaire._awaiting_tracker_choice = False
        questionnaire._follow_up_choices.pop(1, None)  # clear the tracker choice from Q1
        _chosen_tracker = questionnaire._preferred_tracker
        logger.info("User chose tracker for velocity/sprint: %s", _chosen_tracker)
        # Reset questionnaire so the first-call init block runs fresh.
        # This prevents the tracker choice (e.g. "Jira") from being treated
        # as the Q1 project description. We pass the preference via a transient
        # key so the new QuestionnaireState picks it up.
        questionnaire = None
        # Store in a local var — the init block below will read it.
        _pending_tracker_pref = _chosen_tracker
    else:
        _pending_tracker_pref = ""

    # ── Small project → Large switch re-entry ────────────────────────
    # See docs: "Guardrails" — human-in-the-loop (advisory)
    #
    # When the user switched modes at the analysis review, the questionnaire is
    # re-opened in Epic (smart) mode with all Small-project answers preserved.
    # There's no answer to record on this pass — jump straight to asking the
    # remaining Large-mode essentials (Q7/Q10/Q27), or the summary if none are missing.
    if questionnaire is not None and getattr(questionnaire, "_reopen_for_epic", False):
        return _reopen_intake_for_epic(state, questionnaire)

    if questionnaire is None:
        # First call — initialize questionnaire and attempt adaptive skip.
        # See docs: "Project Intake Questionnaire" — adaptive skip logic
        #
        # If the user typed a project description at the scrum> prompt before
        # Q1, we send it to the LLM to extract any answers it can find. This
        # pre-populates the questionnaire so the user only answers remaining
        # questions they haven't already covered.
        qs = QuestionnaireState()
        # The prior-art step runs from _show_summary_or_pto, which sees only the
        # questionnaire — stash the profile id rather than thread graph state
        # through eleven call sites. Same pattern as _repo_context.
        qs._analysis_profile_id = _effective_analysis_profile_id(state)
        qs._analysis_enabled = _wants_dep(state, "analysis")

        # Apply tracker preference from the choice resolution above (if any).
        if _pending_tracker_pref:
            qs._preferred_tracker = _pending_tracker_pref

        # Read intake_mode from the REPL-injected state key.
        # The legacy 30-question "standard" flow has been retired — smart intake
        # is the single interactive path. Default to (and coerce any legacy
        # "standard" / unknown value from old sessions to) "smart" so the
        # pipeline always follows one code path below.
        intake_mode = state.get("_intake_mode") or "smart"
        if intake_mode not in ("smart", "quick", "small_project"):
            intake_mode = "smart"
        qs.intake_mode = intake_mode
        # The Solo world's key, seeded by the caller (see ScrumState.solo).
        solo = bool(state.get("solo"))
        if solo:
            logger.info("Intake: solo run — Q6/Q7/Q30 default to one developer, member selection skipped")

        # Extract initial description from the first message (if present).
        # The REPL sends the user's first input as a HumanMessage before
        # the first graph invocation reaches this node.
        description = ""
        if state.get("messages"):
            first_msg = state["messages"][0]
            if isinstance(first_msg, HumanMessage):
                description = first_msg.content

        extracted = _extract_answers_from_description(description)
        # Deterministic keyword fallback — catches strong signals the LLM
        # may have been too conservative to infer (Q2, Q12, Q13).
        _keyword_extract_fallback(description, extracted)

        # ── SCRUM.md auto-population ─────────────────────────────────
        # Load SCRUM.md early so its content can pre-fill intake answers,
        # avoiding duplicate data entry when the user has already documented
        # project context in the file.
        # See docs: "Tools" — read-only tool pattern
        scrum_md_context, _scrum_status = _load_user_context()
        scrum_extracted: dict[int, str] = {}
        _scrum_md_contributed: set[int] = set()
        if scrum_md_context:
            scrum_extracted = _extract_answers_from_description(scrum_md_context)
            # Merge: user's typed description wins over SCRUM.md.
            # SCRUM.md fills gaps the description didn't cover.
            for q_num, answer in scrum_extracted.items():
                if q_num not in extracted:
                    extracted[q_num] = answer
                    _scrum_md_contributed.add(q_num)
            if scrum_extracted:
                logger.debug(
                    "SCRUM.md extracted %d answers (questions: %s), contributed %d new",
                    len(scrum_extracted),
                    sorted(scrum_extracted.keys()),
                    len(_scrum_md_contributed),
                )

        # ── Analysis profile auto-fill ────────────────────────────────
        # If the user selected an analysis profile in the profile picker,
        # extract Q6/Q8/Q9 from it. Priority: description > SCRUM.md > analysis.
        _analysis_profile_id = _effective_analysis_profile_id(state)
        if _analysis_profile_id:
            _ap, _ap_ex = _load_profile_by_id(_analysis_profile_id)
            if _ap:
                _analysis_answers = _extract_answers_from_profile(_ap, _ap_ex)
                if solo:
                    _analysis_answers.pop(6, None)
                    _solo_vel = _solo_velocity_from_profile(_ap, _ap_ex)
                    if _solo_vel:
                        _analysis_answers[9] = _solo_vel
                    else:
                        _analysis_answers.pop(9, None)
                for q_num, answer in _analysis_answers.items():
                    if q_num not in extracted:
                        extracted[q_num] = answer
                        _scrum_md_contributed.add(q_num)  # track as auto-filled
                logger.info(
                    "Analysis profile %s auto-filled %d answers: %s",
                    _analysis_profile_id,
                    len(_analysis_answers),
                    sorted(_analysis_answers.keys()),
                )

                # Set up Q6 as team member multi-select when contributor data exists
                _member_names = _get_contributor_names(_ap_ex)
                if not solo and _member_names and len(_member_names) >= 2:
                    # Build member labels with velocity info
                    _member_labels = []
                    _contrib_list = _ap_ex.get("contributor_stats", [])
                    for c in _contrib_list:
                        name = c.get("name", "")
                        ps = c.get("per_sprint", 0)
                        disc = c.get("top_discipline", "")
                        if name:
                            label = f"{name} ({ps:.1f} pts/sprint, {disc})" if ps > 0 else name
                            _member_labels.append(label)
                    if _member_labels:
                        qs._follow_up_choices[6] = tuple(_member_labels)
                        qs._q6_member_select = True  # flag for velocity recalc
                        # Don't auto-fill Q6 — let user select members
                        extracted.pop(6, None)
                        logger.info("Q6 set up as team member multi-select: %d members", len(_member_labels))

        # ── Smart / quick mode first-invocation ──────────────────────
        # See docs: "Project Intake Questionnaire" — smart intake
        #
        # In smart/quick mode, extracted answers are auto-accepted (not
        # shown for confirmation). Defaults are auto-applied to all
        # non-essential questions. Only essential gaps are asked.
        if intake_mode in ("smart", "quick", "small_project"):
            logger.debug("BANK_HOLIDAY: entering %s intake mode", intake_mode)
            # Store SCRUM.md provenance so the preamble can report it
            qs._scrum_md_questions = _scrum_md_contributed
            # Q1 is always the user's description (first message)
            if description.strip():
                qs.answers[1] = description.strip()
                qs.extracted_questions.add(1)
                qs.answer_sources[1] = AnswerSource.EXTRACTED

            # Pick essential set based on mode (needed before extraction split)
            essential_set = _essentials_for_mode(intake_mode)
            # If analysis provides velocity, skip Q9 from essentials (auto-accept).
            # Q11 (tech stack) stays essential so it appears as a suggestion
            # the user can confirm or override.
            if _analysis_profile_id and 9 in extracted:
                essential_set = essential_set - {9}
                logger.info("Analysis auto-fill: removed Q9 from essentials")

            # Split extractions: essential questions go to suggested_answers
            # (user confirms or overrides), non-essential go to answers (auto-accepted).
            # This ensures essential questions are always asked interactively,
            # with the extracted value shown as a pre-filled suggestion.
            if extracted:
                essential_extractions = {q: a for q, a in extracted.items() if q in essential_set}
                non_essential_extractions = {q: a for q, a in extracted.items() if q not in essential_set}
                if non_essential_extractions:
                    _auto_apply_extractions(qs, non_essential_extractions)
                if essential_extractions:
                    qs.suggested_answers.update(essential_extractions)
                if 17 in extracted:
                    _sync_platform_from_url(qs)
            fallbacks = QUICK_FALLBACK_DEFAULTS if intake_mode == "quick" else None
            if solo:
                essential_set = _apply_solo_defaults(qs, essential_set)

            # Auto-default all non-essential, non-answered questions
            _auto_default_remaining(qs, essential_set, fallbacks)

            # ── Repository-informed suggestions ──────────────────────────
            # Scan the project's repo (the Q17 URL, or a configured GitHub repo)
            # and derive a suggested tech stack + integrations, plus a low-code
            # verdict. Graceful: no configured/available repo → a no-op that only
            # applies description-based low-code markers. Runs once here at first
            # invocation; the raw scan + verdict are stashed on the questionnaire
            # for project_analyzer to reuse (avoids a second live scan).
            # See docs: "Project Intake Questionnaire" — smart intake
            _apply_repo_signals(qs)

            # Derive Q15 from Q2 if available
            _derive_q15_from_q2(qs)

            # Q27 sprint selection: no Jira → auto-default to "Fresh start (today)".
            # Q27 is in SMART_ESSENTIALS, but auto-deriving fills the answer so it
            # won't appear as a gap when Jira is absent.
            # Q28 (bank holidays) choices are prepared so the user sees a confirmation.
            # See docs: "Scrum Standards" — capacity planning
            # ── Derive tracker preference from analysis profile if selected ──
            _ap_id = _effective_analysis_profile_id(state)
            if _ap_id and not qs._preferred_tracker:
                from yeaboi import trackers as _trackers

                _ap_source = _ap_id.split("-", 1)[0]
                if _ap_source in _trackers.TRACKERS:
                    qs._preferred_tracker = _ap_source
                    logger.info("Tracker preference derived from analysis profile: %s", _ap_source)

            # ── Tracker choice prompt when two or more are configured ──
            from yeaboi import trackers as _trackers

            # Only the trackers this plan may consult are offered; when the plan's
            # integrations leave exactly one, it is the preference without asking.
            _configured_trackers = [k for k in _trackers.configured() if _wants_integration(state, k)]
            if len(_configured_trackers) == 1 and not qs._preferred_tracker and state.get("session_integrations"):
                qs._preferred_tracker = _configured_trackers[0]
                logger.info("Tracker preference set by the plan's integrations: %s", qs._preferred_tracker)
            if len(_configured_trackers) >= 2 and not qs._preferred_tracker:
                qs._awaiting_tracker_choice = True
                # Use Q1 slot with follow-up choices so the TUI accordion renders
                # a proper choice menu (Q0 doesn't exist in the accordion height map).
                qs.current_question = 1
                _labels = tuple(_trackers.TRACKERS[k].label for k in _configured_trackers)
                qs._follow_up_choices[1] = _labels
                _named = " and ".join(f"**{label}**" for label in _labels)
                return {
                    "questionnaire": qs,
                    "messages": [
                        AIMessage(
                            content=(
                                f"{_named} are configured.\n\nWhich tracker should I use for velocity and sprint data?"
                            )
                        )
                    ],
                }

            if _configured_trackers and _is_tracker_configured():
                # Fire both tracker calls concurrently — they are independent HTTP
                # requests and running them in parallel halves the wait time.
                # See docs: "Scrum Standards" — capacity planning
                from concurrent.futures import ThreadPoolExecutor

                need_velocity = 9 not in qs.answers or 9 in qs.defaulted_questions
                # Determine which tracker to use — explicit choice or auto-detect
                _tracker_label = _trackers.label_for(qs._preferred_tracker) or "Jira"
                _pref = qs._preferred_tracker
                with ThreadPoolExecutor(max_workers=2) as pool:
                    vel_future = pool.submit(_fetch_tracker_velocity, _pref) if need_velocity else None
                    sprint_future = pool.submit(_fetch_active_sprint_number, _pref)
                    jira_data = vel_future.result() if vel_future else None
                    active_result = sprint_future.result()

                # Apply velocity data from tracker if Q9 hasn't been answered yet.
                # Uses per-dev velocity × feature team size (Q6) so the velocity
                # reflects the subset of the team working on this feature.
                if need_velocity and jira_data is not None:
                    jira_team_size = jira_data["jira_team_size"]
                    # Always store the org team size — used to cap the
                    # "increase team" recommendation even when velocity is zero.
                    # A solo run is capped at the one person there is.
                    qs._jira_org_team_size = 1 if solo else jira_team_size

                    if "velocity_error" not in jira_data:
                        per_dev = jira_data["per_dev_velocity"]
                        team_vel = jira_data["team_velocity"]
                        qs._jira_per_dev_velocity = per_dev

                        # Scale by Q6 (feature team size) if available
                        q6 = _parse_first_int(qs.answers.get(6, ""))
                        if q6 and q6 > 0:
                            feature_vel = round(per_dev * q6)
                            qs.answers[9] = (
                                f"{feature_vel} pts/sprint "
                                f"({per_dev:.0f} pts/dev × {q6} dev(s) — "
                                f"from {_tracker_label}: {team_vel} pts team avg, "
                                f"{jira_team_size} team member(s))"
                            )
                        else:
                            # Q6 not yet answered — store per-dev rate,
                            # will be recomputed at confirmation
                            qs.answers[9] = (
                                f"{per_dev:.0f} pts/dev/sprint "
                                f"(from {_tracker_label}: {team_vel} pts team avg, "
                                f"{jira_team_size} team member(s))"
                            )
                        qs.extracted_questions.add(9)
                        qs.defaulted_questions.discard(9)
                        qs.answer_sources[9] = AnswerSource.EXTRACTED
                    else:
                        logger.debug(
                            "Jira velocity zero but org team size=%d stored",
                            jira_team_size,
                        )

                # Unpack active sprint result (fetched concurrently above)
                active_num, active_start, jira_status = active_result
            else:
                _derive_q27_from_locale(qs)
                active_num, active_start, jira_status = None, None, ""
            # Find remaining essential gaps (bank holidays are detected later,
            # once Q10 has its final answer — not here where Q10 may still be
            # the default "6 sprints").
            gaps = _find_essential_gaps(qs, essential_set)
            logger.info(
                "Essential gaps: %s (essentials=%s, Q27 in answers=%s, Q27 in defaults=%s)",
                gaps,
                sorted(essential_set),
                27 in qs.answers,
                27 in qs.defaulted_questions,
            )

            # Build extraction summary for the preamble
            num_from_desc = len(qs.extracted_questions - qs._scrum_md_questions)
            num_from_scrum = len(qs.extracted_questions & qs._scrum_md_questions)
            num_defaulted = len(qs.defaulted_questions)
            preamble_parts: list[str] = []
            if num_from_desc > 0:
                preamble_parts.append(f"extracted **{num_from_desc}** from your description")
            if num_from_scrum > 0:
                preamble_parts.append(f"took **{num_from_scrum}** from SCRUM.md")
            if num_defaulted > 0:
                preamble_parts.append(f"filled **{num_defaulted}** with defaults")
            preamble = ""
            if preamble_parts:
                preamble = "I " + " and ".join(preamble_parts) + ".\n\n"

            if not gaps:
                # All essentials filled — detect bank holidays now that Q10 is
                # finalized, then ask PTO or jump straight to summary.
                _prepare_bank_holiday_choices(qs)
                return _show_summary_or_pto(qs, prefix=preamble)

            # Ask the first gap (or merged Q3+Q4)
            prompt_text, q_nums = _build_gap_prompt(gaps, qs)
            qs._pending_merged_questions = q_nums
            # Set current_question to the first gap being asked
            qs.current_question = q_nums[0]

            # Q27 with tracker: use the active sprint/iteration number (fetched concurrently above)
            # to populate dynamic choices for the sprint selection menu.
            # Small mode asks a different Q27 — "add to an existing sprint, or
            # create a new one?" — with the board's real sprint names as targets.
            small_q27 = q_nums[0] == 27 and _is_small_project_mode(qs.intake_mode) and _is_tracker_configured()
            if small_q27 and (small_prompt := _setup_small_sprint_target_question(qs)) is not None:
                prompt_text = small_prompt
            elif small_q27:
                # Target list unavailable — the landing decision (new sprint
                # vs backlog) is still real, so ask it instead of silently
                # defaulting like the generic path would.
                if active_num is not None:
                    qs._active_sprint_start_date = active_start
                source = (
                    f"Detected active sprint in {_tracker_label}: **Sprint {active_num}**.\n\n"
                    if active_num is not None
                    else ""
                )
                prompt_text = _setup_small_sprint_fallback_question(qs, active_num, source)
            elif not small_q27 and q_nums[0] == 27 and active_num is not None:
                qs._active_sprint_number = active_num
                qs._active_sprint_start_date = active_start
                qs.answers[27] = f"_active:{active_num}"
                qs._follow_up_choices[27] = (
                    f"Sprint {active_num + 1} (next)",
                    f"Sprint {active_num + 2}",
                    f"Sprint {active_num + 3}",
                )
                prompt_text = (
                    f"Detected active sprint in {_tracker_label}: **Sprint {active_num}**.\n\n"
                    f"Which sprint are you planning for?"
                )
            elif q_nums[0] == 27 and _is_tracker_configured():
                # Couldn't fetch sprint/targets — tell the user why, then fall back
                logger.warning("Tracker sprint fetch failed: %s", jira_status)
                _derive_q27_from_locale(qs)
                gaps = _find_essential_gaps(qs, essential_set)
                if not gaps:
                    # All essentials filled — detect bank holidays with
                    # finalized Q10, then ask PTO or show summary.
                    _prepare_bank_holiday_choices(qs)
                    return _show_summary_or_pto(qs, prefix=preamble)
                prompt_text, q_nums = _build_gap_prompt(gaps, qs)
                qs._pending_merged_questions = q_nums
                qs.current_question = q_nums[0]

            remaining_text = f"A few more questions ({len(gaps)} remaining):" if len(gaps) > 1 else "One more question:"
            return {
                "questionnaire": qs,
                "messages": [AIMessage(content=f"{preamble}{remaining_text}\n\n{prompt_text}")],
            }

        # NOTE: the legacy "standard" (30-question, one-at-a-time) first-invocation
        # flow used to live here. It has been retired — intake_mode is coerced to
        # smart/quick/small_project above, so the branch just above always returns.

    # Record the user's answer to the current question.
    # The last message in state is always the HumanMessage with the user's reply.
    last_msg = state["messages"][-1]
    current_q = questionnaire.current_question
    logger.info(
        "Intake subsequent: current_q=%s, _q6_member=%s, msg=%.60s",
        current_q,
        getattr(questionnaire, "_q6_member_select", "N/A"),
        last_msg.content if hasattr(last_msg, "content") else "?",
    )

    # ── Edit re-ask handler ──────────────────────────────────────────
    # See docs: "Project Intake Questionnaire" — edit flow
    #
    # When editing_question is set, the user is answering a re-asked
    # question from the confirmation summary. Record their new answer
    # (or keep the current one on skip), clear editing state, and
    # re-show the updated summary.
    if questionnaire.editing_question is not None:
        eq = questionnaire.editing_question
        if _is_skip_intent(last_msg.content):
            # Skip during re-ask → keep the current answer unchanged
            pass
        else:
            # Record new answer, clear any defaulted/skipped flags for this question
            questionnaire.answers[eq] = last_msg.content
            questionnaire.defaulted_questions.discard(eq)
            questionnaire.skipped_questions.discard(eq)

        questionnaire.editing_question = None
        summary = _build_intake_summary(questionnaire)
        return {
            "questionnaire": questionnaire,
            "messages": [AIMessage(content=f"{summary}{_CONFIRM_PROMPT}")],
            "pending_review": "project_intake",
        }

    # ── Smart / quick mode gap-filling ──────────────────────────────
    # See docs: "Project Intake Questionnaire" — smart intake
    #
    # In smart/quick mode, after the first invocation, the user is
    # answering essential gap questions one at a time. Record the
    # answer, derive Q15 if Q2 was just answered, then find the next
    # gap or show the summary.
    if (
        questionnaire.intake_mode in ("smart", "quick", "small_project")
        and not questionnaire.awaiting_confirmation
        and questionnaire.editing_question is None
    ):
        # Confirmation prefix for repo URL detection — set after _sync_platform_from_url
        # and prepended to the next AIMessage so the user sees immediate feedback.
        repo_confirm = ""

        # ── Skip / defaults intents (gap mode) ────────────────────
        # These literals must be caught BEFORE answer recording, or the word
        # itself is stored as the gap answer — the standard-mode handlers
        # further down are unreachable in smart/quick/small_project modes.
        if _is_defaults_all_intent(last_msg.content) or _is_defaults_intent(last_msg.content):
            # In gap mode the non-essentials are already auto-defaulted, so
            # "defaults" and "defaults all" both mean "stop asking": fill
            # everything defaultable, flag the rest, land at the summary.
            # Only "defaults all" (the /finish fast path) also bypasses the
            # PTO sub-loop — /defaults still stops there in smart mode.
            summary_lines, count = _batch_defaults_for_phase(questionnaire, last_q=TOTAL_QUESTIONS)
            if _is_defaults_all_intent(last_msg.content) and questionnaire.intake_mode not in (
                "quick",
                "small_project",
            ):
                # PTO defaults to "no planned leave", matching quick mode. A
                # truthy _leave_input_stage bypasses the PTO gate in
                # _show_summary_or_pto; the PTO stage machine only runs while
                # _awaiting_leave_input is True, so "done" is inert there.
                questionnaire._leave_input_stage = "done"
            questionnaire._pending_merged_questions = []
            gaps_flagged = len(questionnaire.skipped_questions - set(questionnaire.answers))
            logger.info("Intake defaults-all (gap mode): defaulted=%d gaps=%d", count, gaps_flagged)
            ack = f"Fast-forward: applied **{count}** default(s) across all remaining questions."
            if gaps_flagged:
                ack += f" {gaps_flagged} essential question(s) had no default — check them in the summary."
            return _show_summary_or_pto(questionnaire, prefix=f"{ack}\n\n")

        if _is_skip_intent(last_msg.content):
            if current_q in questionnaire.probed_questions:
                # Skip during a follow-up probe — keep the original answer.
                questionnaire._follow_up_choices.pop(current_q, None)
                repo_confirm = f"{_build_skip_acknowledgment(current_q, during_probe=True, default=None)}\n\n"
            else:
                # Default-or-flag every question in the current gap prompt
                # (merged gaps like Q3+Q4 are asked together).
                targets = questionnaire._pending_merged_questions or [current_q]
                for q_num in targets:
                    if q_num in questionnaire.answers:
                        continue
                    if q_num in QUESTION_DEFAULTS:
                        questionnaire.answers[q_num] = QUESTION_DEFAULTS[q_num]
                        questionnaire.defaulted_questions.add(q_num)
                        questionnaire.answer_sources[q_num] = AnswerSource.DEFAULTED
                    else:
                        questionnaire.skipped_questions.add(q_num)
                default = QUESTION_DEFAULTS.get(current_q)
                repo_confirm = f"{_build_skip_acknowledgment(current_q, during_probe=False, default=default)}\n\n"
            questionnaire._pending_merged_questions = []
            questionnaire._follow_up_choices.pop(current_q, None)
            logger.info("Intake skip (gap mode): Q%d", current_q)

        # ── Follow-up probe response ──────────────────────────────
        # If this question was already probed for vagueness, the user
        # is responding to the follow-up. Combine original + follow-up
        # answers (same logic as standard mode) then advance.
        elif current_q in questionnaire.probed_questions:
            if current_q == 2 and questionnaire.answers.get(2) in _EXISTING_CODEBASE_ANSWERS:
                # This is the repo URL follow-up — store in Q17, not combined into Q2.
                # Q17 is "Can you share the repo URL(s)?" — storing here lets the
                # repo scan and platform detection use it downstream.
                questionnaire.answers[17] = last_msg.content
                questionnaire.defaulted_questions.discard(17)
                _sync_platform_from_url(questionnaire)
                platform = questionnaire.answers.get(16, "")
                if platform:
                    repo_confirm = f"*✓ {platform} repo detected — will be scanned during analysis.*\n\n"
            else:
                original = questionnaire.answers.get(current_q, "")
                combined = f"{original}\n\n(Follow-up detail: {last_msg.content})"
                questionnaire.answers[current_q] = combined
            questionnaire._follow_up_choices.pop(current_q, None)
        else:
            # Record answer for the primary question first. For merged Q3+Q4,
            # only store the primary question's answer here — secondary questions
            # are stored AFTER the vagueness check passes, so they don't show
            # as completed in the accordion while a follow-up probe is active.
            pending = questionnaire._pending_merged_questions
            if pending:
                for q_num in pending:
                    questionnaire.answers[q_num] = last_msg.content
                    questionnaire.defaulted_questions.discard(q_num)
                    questionnaire.skipped_questions.discard(q_num)
                    questionnaire.answer_sources[q_num] = AnswerSource.DIRECT
                questionnaire._pending_merged_questions = []
            else:
                # Single gap question
                questionnaire.answers[current_q] = last_msg.content
                questionnaire.defaulted_questions.discard(current_q)
                questionnaire.skipped_questions.discard(current_q)
                questionnaire.answer_sources[current_q] = AnswerSource.DIRECT

            # Q6 team member multi-select → recalculate velocity + auto-fill Q7
            # (runs after both pending and single-gap answer recording)
            if current_q == 6 and getattr(questionnaire, "_q6_member_select", False):
                _sel_text = last_msg.content
                # Parse member names — handle both "; " and ", " separators
                # Labels look like "Name (X pts/sprint, discipline)"
                # Split on "; " first (multi-select default), then try ", " between labels
                _sel_names = []
                if "; " in _sel_text:
                    for _p in _sel_text.split("; "):
                        name = _p.strip().split(" (")[0].strip()
                        if name:
                            _sel_names.append(name)
                else:
                    # Separator is ", " — use regex to find "Name (info)" patterns
                    import re as _q6re

                    _labels = _q6re.findall(r"([A-Za-z][^(]*?)\s*\([^)]+\)", _sel_text)
                    _sel_names = [lbl.strip() for lbl in _labels if lbl.strip()]
                    if not _sel_names:
                        # No parentheses — plain comma-separated names
                        _sel_names = [p.strip() for p in _sel_text.split(",") if p.strip()]
                # Deduplicate preserving order
                _seen: set[str] = set()
                _unique: list[str] = []
                for _n in _sel_names:
                    if _n.lower() not in _seen:
                        _seen.add(_n.lower())
                        _unique.append(_n)
                _sel_names = _unique

                if _sel_names:
                    _names_str = ", ".join(_sel_names)
                    questionnaire.answers[6] = f"{len(_sel_names)} ({_names_str})"
                    logger.info("Q6 member select: %d members: %s", len(_sel_names), _names_str)

                    _ap_id = _effective_analysis_profile_id(state)
                    if _ap_id:
                        try:
                            _, _ap_ex2 = _load_profile_by_id(_ap_id)
                            if _ap_ex2:
                                _vel = _calculate_velocity_for_members(_sel_names, _ap_ex2)
                                if _vel > 0:
                                    questionnaire.answers[9] = (
                                        f"{_vel:.0f} points per sprint (from {len(_sel_names)} selected members)"
                                    )
                                    questionnaire.answer_sources[9] = AnswerSource.EXTRACTED
                                    questionnaire.extracted_questions.add(9)
                                    questionnaire.defaulted_questions.discard(9)

                                # Auto-fill Q7 from selected members' disciplines
                                _contrib_list = _ap_ex2.get("contributor_stats", [])
                                _roles: dict[str, int] = {}
                                _lower_names = [n.lower() for n in _sel_names]
                                for c in _contrib_list:
                                    if isinstance(c, dict) and c.get("name", "").lower() in _lower_names:
                                        disc = c.get("top_discipline", "fullstack")
                                        _dm = {
                                            "backend": "Backend",
                                            "frontend": "Frontend",
                                            "fullstack": "Fullstack",
                                            "infrastructure": "DevOps/Infra",
                                            "devops": "DevOps/Infra",
                                            "ci-cd": "DevOps/Infra",
                                            "testing": "QA/Testing",
                                            "security": "DevOps/Infra",
                                            "data": "Data/ML",
                                            "design": "Design",
                                            "observability": "DevOps/Infra",
                                            "platform": "DevOps/Infra",
                                            "networking": "DevOps/Infra",
                                            "database": "Backend",
                                        }
                                        role = _dm.get(disc, "Fullstack")
                                        _roles[role] = _roles.get(role, 0) + 1
                                if _roles:
                                    _rp = [
                                        f"{c} {r}" if c > 1 else r
                                        for r, c in sorted(_roles.items(), key=lambda x: -x[1])
                                    ]
                                    questionnaire.answers[7] = ", ".join(_rp)
                                    questionnaire.answer_sources[7] = AnswerSource.EXTRACTED
                                    questionnaire.extracted_questions.add(7)
                                    questionnaire.defaulted_questions.discard(7)
                                    logger.info("Q7 auto-filled: %s", questionnaire.answers[7])
                        except Exception:
                            pass
                questionnaire._q6_member_select = False
                questionnaire._follow_up_choices.pop(6, None)

            if current_q == 17:
                _sync_platform_from_url(questionnaire)
                platform = questionnaire.answers.get(16, "")
                if platform:
                    repo_confirm = f"*✓ {platform} repo detected — will be scanned during analysis.*\n\n"

            # Q2 repo URL follow-up: when Q2 = "Existing codebase"/"Hybrid",
            # prompt for the repo URL inline as part of Q2. The answer is stored
            # in Q17 so downstream repo scan and platform detection can use it.
            # Q2 stays active in the accordion during this follow-up.
            # Check defaulted_questions because _auto_default_remaining may have
            # already filled Q17 with a placeholder default.
            if questionnaire.answers.get(2) in _EXISTING_CODEBASE_ANSWERS and (
                17 not in questionnaire.answers or 17 in questionnaire.defaulted_questions
            ):
                questionnaire.probed_questions.add(2)
                return {
                    "questionnaire": questionnaire,
                    "messages": [AIMessage(content=_Q2_REPO_URL_PROMPT)],
                }

            # ── Vague answer probing ──────────────────────────────
            # See docs: "Project Intake Questionnaire" — follow-up probing
            #
            # Same vagueness check as standard mode — even in smart mode,
            # a vague answer to an essential question defeats the purpose.
            # Skip for choice questions (selections are never vague).
            if not is_choice_question(current_q):
                check_text = INTAKE_QUESTIONS[current_q]
                vague_result = _check_vague_answer(check_text, last_msg.content, current_q)
                if vague_result:
                    follow_up, choices = vague_result
                    questionnaire.probed_questions.add(current_q)
                    questionnaire.answer_sources[current_q] = AnswerSource.PROBED
                    if choices:
                        questionnaire._follow_up_choices[current_q] = choices
                    return {
                        "questionnaire": questionnaire,
                        "messages": [AIMessage(content=f"**Follow-up on Q{current_q}:**\n\n{follow_up}")],
                    }

            # Clear pending merged list after answer is stored.
            if pending:
                questionnaire._pending_merged_questions = []

        # Derive Q15 from Q2 if Q2 was just answered
        _derive_q15_from_q2(questionnaire)

        # Q27 sprint selection: resolve the selected sprint.
        # The resolved choice text is "Sprint 105 (next)" — extract the sprint number.
        # Small mode resolves its "Add to <name>" / "Create a new sprint" choices
        # FIRST: "Add to PSOT Sprint 104" must keep its add-to intent, so the
        # generic Sprint-N rewrite below must never see it.
        # Bank holidays are now a separate question (Q28).
        if current_q == 27 and _is_tracker_configured():
            if _is_small_project_mode(questionnaire.intake_mode):
                _resolve_small_sprint_target_answer(questionnaire)
            q27_answer = questionnaire.answers.get(27, "")
            if not q27_answer.startswith("Add to "):
                sprint_num_match = re.search(r"Sprint\s+(\d+)", q27_answer)
                if sprint_num_match:
                    questionnaire.answers[27] = f"Sprint {sprint_num_match.group(1)}"
            questionnaire._follow_up_choices.pop(27, None)
            # Prepare bank holiday detection choices for Q28 (no-op in small mode)
            _prepare_bank_holiday_choices(questionnaire)

        # Q28 bank holiday: parse the user's answer (same logic as standard mode)
        if current_q == 28:
            q28_answer = questionnaire.answers.get(28, "")
            if "accept" in q28_answer.lower():
                count = questionnaire._detected_bank_holiday_days
                questionnaire.answers[28] = f"{count} bank holiday(s)" if count > 0 else "No bank holidays"
            elif "no bank holidays" in q28_answer.lower():
                questionnaire._detected_bank_holiday_days = 0
                questionnaire.answers[28] = "No bank holidays"
            elif "enter manually" in q28_answer.lower():
                questionnaire._follow_up_choices.pop(28, None)
                questionnaire.answers.pop(28, None)
                questionnaire.probed_questions.add(28)
                return {
                    "questionnaire": questionnaire,
                    "messages": [AIMessage(content="How many bank/public holiday days fall in your planning window?")],
                }
            else:
                parsed = _parse_first_int(q28_answer)
                if parsed is not None:
                    questionnaire._detected_bank_holiday_days = parsed
                    questionnaire.answers[28] = f"{parsed} bank holiday(s)"
            questionnaire._follow_up_choices.pop(28, None)

        # Find remaining essential gaps
        essential_set = _essentials_for_mode(questionnaire.intake_mode)
        gaps = _find_essential_gaps(questionnaire, essential_set)

        if not gaps:
            # All essentials filled — detect bank holidays with finalized
            # Q10 answer, then ask PTO or show summary.
            _prepare_bank_holiday_choices(questionnaire)
            return _show_summary_or_pto(questionnaire, prefix=repo_confirm)

        # Ask the next gap
        prompt_text, q_nums = _build_gap_prompt(gaps, questionnaire)
        questionnaire._pending_merged_questions = q_nums
        questionnaire.current_question = q_nums[0]

        # Q27 with tracker: populate dynamic choices for the sprint selection menu.
        # Small mode asks its own Q27 (existing sprint vs new) with board names.
        if (
            q_nums[0] == 27
            and _is_small_project_mode(questionnaire.intake_mode)
            and _is_tracker_configured()
            and (small_prompt := _setup_small_sprint_target_question(questionnaire)) is not None
        ):
            prompt_text = small_prompt
        elif q_nums[0] == 27 and _is_tracker_configured():
            from yeaboi import trackers as _trk_registry

            _pref_trk = questionnaire._preferred_tracker
            _trk_label = _trk_registry.label_for(_pref_trk) or "Jira"
            logger.info("Q27: fetching active sprint from %s (preferred=%s)", _trk_label, _pref_trk)
            active_num, active_start, jira_status = _fetch_active_sprint_number(_pref_trk)
            logger.info("Q27: active_num=%s, active_start=%s, status=%s", active_num, active_start, jira_status)
            if active_num is not None and not _is_small_project_mode(questionnaire.intake_mode):
                questionnaire._active_sprint_number = active_num
                questionnaire._active_sprint_start_date = active_start
                questionnaire.answers[27] = f"_active:{active_num}"
                questionnaire._follow_up_choices[27] = (
                    f"Sprint {active_num + 1} (next)",
                    f"Sprint {active_num + 2}",
                    f"Sprint {active_num + 3}",
                )
                prompt_text = (
                    f"Detected active sprint in {_trk_label}: **Sprint {active_num}**.\n\n"
                    f"Which sprint are you planning for?"
                )
            elif active_num is not None:
                # Small mode with a live active number but no target list —
                # the landing decision (new sprint vs backlog) is still real.
                questionnaire._active_sprint_start_date = active_start
                prompt_text = _setup_small_sprint_fallback_question(
                    questionnaire,
                    active_num,
                    f"Detected active sprint in {_trk_label}: **Sprint {active_num}**.\n\n",
                )
            else:
                # Couldn't fetch active sprint from live tracker
                logger.warning("Tracker sprint fetch failed: %s", jira_status)
                # Try analysis sprint data as fallback
                _ap_id = _effective_analysis_profile_id(state)
                _used_analysis = False
                if _ap_id:
                    try:
                        _, _ap_ex3 = _load_profile_by_id(_ap_id)
                        if _ap_ex3:
                            _sd = _ap_ex3.get("sprint_details", [])
                            if _sd:
                                # Find the latest sprint number from analysis
                                _last_sprint = _sd[-1]
                                _last_name = _last_sprint.get("name", "")
                                import re as _s27re

                                _num_m = _s27re.search(r"(\d+)", _last_name)
                                if _num_m:
                                    _last_num = int(_num_m.group(1))
                                    _src_line = (
                                        f"Last analysed sprint: **Sprint {_last_num}** "
                                        f"(from team analysis — live tracker unavailable).\n\n"
                                    )
                                    if _is_small_project_mode(questionnaire.intake_mode):
                                        # Small mode's landing decision: create
                                        # a new sprint, or leave it in the backlog.
                                        prompt_text = _setup_small_sprint_fallback_question(
                                            questionnaire, _last_num, _src_line
                                        )
                                    else:
                                        questionnaire._active_sprint_number = _last_num
                                        questionnaire.answers[27] = f"_active:{_last_num}"
                                        questionnaire._follow_up_choices[27] = (
                                            f"Sprint {_last_num + 1} (next)",
                                            f"Sprint {_last_num + 2}",
                                            f"Sprint {_last_num + 3}",
                                        )
                                        prompt_text = f"{_src_line}Which sprint are you planning for?"
                                    _used_analysis = True
                                    logger.info("Q27 fallback: using analysis sprint %d", _last_num)
                    except Exception:
                        pass
                if not _used_analysis and _is_small_project_mode(questionnaire.intake_mode):
                    # No live board and no analysis number — the small-mode
                    # landing decision (new sprint vs backlog) is still real.
                    prompt_text = _setup_small_sprint_fallback_question(questionnaire, None, "")
                elif not _used_analysis:
                    _derive_q27_from_locale(questionnaire)
                    gaps = _find_essential_gaps(questionnaire, essential_set)
                    if not gaps:
                        _prepare_bank_holiday_choices(questionnaire)
                        return _show_summary_or_pto(questionnaire, prefix=repo_confirm)
                    prompt_text, q_nums = _build_gap_prompt(gaps, questionnaire)
                    questionnaire._pending_merged_questions = q_nums
                    questionnaire.current_question = q_nums[0]

        return {
            "questionnaire": questionnaire,
            "messages": [AIMessage(content=f"{repo_confirm}{prompt_text}")],
        }

    # ── Confirmation gate ────────────────────────────────────────────
    # See docs: "Project Intake Questionnaire" — confirmation gate
    #
    # After the last question is answered the summary is shown and
    # awaiting_confirmation is set. On the NEXT invocation, check here:
    #   - User confirms → set completed=True, route to main agent
    #   - User edits a question → inline update or re-ask
    #   - User says anything else → show edit format help + re-show summary
    if questionnaire.awaiting_confirmation:
        # ── Velocity override input handler ──────────────────────────
        # When user picked "Override" from the velocity choice menu,
        # we're waiting for them to type a number.
        if questionnaire._awaiting_velocity_input:
            override = _parse_velocity_override(last_msg.content)
            if override is not None:
                questionnaire._velocity_override = override
                questionnaire._awaiting_velocity_input = False
                # Re-show summary with overridden velocity
                summary = _build_intake_summary(questionnaire)
                return {
                    "questionnaire": questionnaire,
                    "messages": [AIMessage(content=f"{summary}{_CONFIRM_PROMPT}")],
                    "pending_review": "project_intake",
                }
            else:
                return {
                    "questionnaire": questionnaire,
                    "messages": [AIMessage(content="Please enter a number (e.g. **14** or **14 pts/sprint**):")],
                }

        # ── PTO leave sub-loop handler ────────────────────────────────
        # See docs: "Scrum Standards" — capacity planning
        #
        # After bank holidays (Q28) are resolved, the PTO sub-loop collects
        # per-person leave entries. This mirrors the _awaiting_velocity_input
        # pattern: a transient state machine driven by flags on QuestionnaireState.
        # In quick mode, PTO is auto-defaulted to "no planned leave".
        if questionnaire._awaiting_leave_input:
            stage = questionnaire._leave_input_stage
            user_text = last_msg.content.strip()

            if stage == "ask":
                # User answering "Does anyone have planned leave?"
                choice = user_text.lower()
                if choice in ("1", "yes", "y"):
                    questionnaire._leave_input_stage = "person"
                    return {
                        "questionnaire": questionnaire,
                        "messages": [AIMessage(content="Who is taking leave? (name or initials):")],
                    }
                elif choice in ("2", "no", "n"):
                    # No planned leave — exit sub-loop back through the funnel.
                    # "done" rather than "" because the PTO guard tests the
                    # stage for falsiness; clearing it would re-ask forever.
                    questionnaire._awaiting_leave_input = False
                    questionnaire._leave_input_stage = "done"
                    return _show_summary_or_pto(questionnaire)
                else:
                    # Invalid input — re-prompt
                    return {
                        "questionnaire": questionnaire,
                        "messages": [AIMessage(content="Please choose [1] Yes or [2] No:")],
                    }

            elif stage == "person":
                questionnaire._leave_input_buffer = {"person": user_text}
                questionnaire._leave_input_stage = "start"
                return {
                    "questionnaire": questionnaire,
                    "messages": [AIMessage(content="Start date (DD/MM/YYYY):")],
                }

            elif stage == "start":
                parsed = _parse_date_dmy(user_text)
                if parsed is None:
                    return {
                        "questionnaire": questionnaire,
                        "messages": [
                            AIMessage(
                                content="Invalid date format. Please enter "
                                "the start date as **DD/MM/YYYY** (e.g. 06/04/2026):"
                            )
                        ],
                    }
                # Validate date falls within a reasonable window around the planning period
                window_start, window_end = _get_planning_window(questionnaire)
                if window_start and parsed < window_start:
                    return {
                        "questionnaire": questionnaire,
                        "messages": [
                            AIMessage(
                                content=f"Date is before the planning window starts "
                                f"({window_start.strftime('%d/%m/%Y')}). Try again:"
                            )
                        ],
                    }
                if window_end and parsed > window_end:
                    return {
                        "questionnaire": questionnaire,
                        "messages": [
                            AIMessage(
                                content=f"Date is after the planning window ends "
                                f"({window_end.strftime('%d/%m/%Y')}). Try again:"
                            )
                        ],
                    }
                questionnaire._leave_input_buffer["start_date"] = parsed.isoformat()
                questionnaire._leave_input_stage = "end"
                return {
                    "questionnaire": questionnaire,
                    "messages": [AIMessage(content="End date (DD/MM/YYYY):")],
                }

            elif stage == "end":
                parsed = _parse_date_dmy(user_text)
                if parsed is None:
                    return {
                        "questionnaire": questionnaire,
                        "messages": [
                            AIMessage(
                                content="Invalid date format. Please enter "
                                "the end date as **DD/MM/YYYY** (e.g. 10/04/2026):"
                            )
                        ],
                    }

                start = parse_date(questionnaire._leave_input_buffer["start_date"])
                # Validate end date within planning window
                _, window_end = _get_planning_window(questionnaire)
                if window_end and parsed > window_end:
                    return {
                        "questionnaire": questionnaire,
                        "messages": [
                            AIMessage(
                                content=f"Date is after the planning window ends "
                                f"({window_end.strftime('%d/%m/%Y')}). Try again:"
                            )
                        ],
                    }
                if parsed < start:
                    return {
                        "questionnaire": questionnaire,
                        "messages": [
                            AIMessage(
                                content=f"End date must be on or after start date "
                                f"({start.strftime('%d/%m/%Y')}). Try again:"
                            )
                        ],
                    }
                working = _count_working_days(start, parsed)
                entry = {
                    "person": questionnaire._leave_input_buffer["person"],
                    "start_date": start.isoformat(),
                    "end_date": parsed.isoformat(),
                    "working_days": working,
                }
                questionnaire._planned_leave_entries.append(entry)
                questionnaire._leave_input_buffer = {}
                questionnaire._leave_input_stage = "more?"

                person = entry["person"]
                start_fmt = start.strftime("%d/%m")
                end_fmt = parsed.strftime("%d/%m")
                summary_line = f"**{person}:** {start_fmt} – {end_fmt} ({working} working day(s))"
                return {
                    "questionnaire": questionnaire,
                    "messages": [AIMessage(content=f"{summary_line}\n\n[1] Add another\n[2] Done")],
                }

            elif stage == "more?":
                if user_text.lower() in ("1", "add", "another", "add another"):
                    questionnaire._leave_input_stage = "person"
                    return {
                        "questionnaire": questionnaire,
                        "messages": [AIMessage(content="Who is taking leave? (name or initials):")],
                    }
                elif user_text.lower() in ("2", "done", "d"):
                    # Done — exit the sub-loop back through the funnel so any
                    # step that sits between PTO and the summary still runs.
                    questionnaire._awaiting_leave_input = False
                    questionnaire._leave_input_stage = "done"
                    return _show_summary_or_pto(questionnaire)
                else:
                    # Invalid input — re-prompt
                    return {
                        "questionnaire": questionnaire,
                        "messages": [AIMessage(content="Please choose [1] Add another or [2] Done:")],
                    }

        # ── Prior-art sub-loop handler ────────────────────────────────
        # See docs: "Project Intake Questionnaire" — prior art
        #
        # Sits after PTO and before the velocity menu, mirroring the PTO
        # handler's shape: a transient stage flag drives a small state machine
        # and every exit routes back through _show_summary_or_pto.
        if questionnaire._prior_art_stage == "empty":
            # Nothing to decide — the card only had to be read. Any input
            # acknowledges it and moves on to the summary.
            questionnaire._prior_art_stage = "done"
            return _show_summary_or_pto(questionnaire)

        if questionnaire._prior_art_stage in ("ask", "reason"):
            candidates = questionnaire._prior_art_candidates
            if not candidates:
                # Defensive: the list emptied underneath us (a resumed session
                # drops the transients). Close out rather than loop.
                return _finish_prior_art(questionnaire)

            if questionnaire._prior_art_stage == "reason":
                # Legacy tolerance: "reason" was the per-repo "why isn't it
                # relevant?" free-text stage, which only a session serialized
                # by an older build can still be in. The user's text answers a
                # question this build no longer asks, so re-ask the whole
                # batch rather than guess what the words meant.
                questionnaire._prior_art_stage = "ask"
                return {
                    "questionnaire": questionnaire,
                    "messages": [AIMessage(content=_prior_art_prompt(questionnaire))],
                }

            parsed = _parse_prior_art_answer(last_msg.content, len(candidates))
            if parsed is None:
                return {
                    "questionnaire": questionnaire,
                    "messages": [AIMessage(content=_PRIOR_ART_GRAMMAR_HINT)],
                }
            relevant, banned = parsed
            # The submission is the whole verdict: both lists are re-derived
            # from it, so a half-answered legacy loop carried in by a resumed
            # session cannot double-write its earlier per-repo answers.
            questionnaire._prior_art_accepted = [candidates[i] for i in sorted(relevant)]
            questionnaire._prior_art_rejected = [
                {"key": candidates[i].get("key", ""), "name": candidates[i].get("name", ""), "reason": ""}
                for i in sorted(banned)
            ]
            logger.info(
                "prior_art: batch verdict — %d relevant, %d banned, %d passed over",
                len(relevant),
                len(banned),
                len(candidates) - len(relevant) - len(banned),
            )
            return _finish_prior_art(questionnaire)

        # ── Velocity choice menu handler ─────────────────────────────
        # "1" or "accept" → accept computed/overridden velocity
        # "2" or "override" → prompt for custom number
        choice_text = last_msg.content.strip().lower()
        if choice_text == "2" or choice_text == "override":
            questionnaire._awaiting_velocity_input = True
            return {
                "questionnaire": questionnaire,
                "messages": [AIMessage(content="Enter your velocity (pts/sprint):")],
            }

        # Choice "1" maps to confirm intent (accept velocity + proceed)
        if choice_text == "1":
            # Treat as confirm — fall through to the confirm block below
            pass
        elif not _is_confirm_intent(last_msg.content):
            # Check for edit intent — inline edit or re-ask
            edit = _parse_edit_intent(last_msg.content)
            if edit is not None:
                q_num, inline_answer = edit
                if inline_answer is not None:
                    # Inline edit: update answer immediately, re-show summary
                    questionnaire.answers[q_num] = inline_answer
                    questionnaire.defaulted_questions.discard(q_num)
                    questionnaire.skipped_questions.discard(q_num)
                    # Clear velocity override when answers change — will be recomputed
                    questionnaire._velocity_override = None
                    summary = _build_intake_summary(questionnaire)
                    return {
                        "questionnaire": questionnaire,
                        "messages": [AIMessage(content=f"Updated Q{q_num}.\n\n{summary}{_CONFIRM_PROMPT}")],
                        "pending_review": "project_intake",
                    }
                else:
                    # Re-ask: show the question text + current answer, collect new answer
                    questionnaire.editing_question = q_num
                    question_text = INTAKE_QUESTIONS[q_num]
                    current_answer = questionnaire.answers.get(q_num, "_no answer_")
                    return {
                        "questionnaire": questionnaire,
                        "messages": [
                            AIMessage(
                                content=(
                                    f"**Q{q_num}.** {question_text}\n\n"
                                    f"Current answer: {current_answer}\n\n"
                                    "Enter your new answer:"
                                )
                            )
                        ],
                    }

            # Not a confirmation, choice, or edit — show edit format help + re-show summary
            summary = _build_intake_summary(questionnaire)
            return {
                "questionnaire": questionnaire,
                "messages": [AIMessage(content=f"{_EDIT_HELP}{summary}{_CONFIRM_PROMPT}")],
                "pending_review": "project_intake",
            }

        # ── Confirm: lock in answers + velocity ──────────────────────
        questionnaire.awaiting_confirmation = False
        questionnaire.completed = True

        # Extract team size and velocity from answers (Q6/Q9).
        extracted = _extract_team_and_velocity(questionnaire)
        velocity_was_calculated = extracted.pop("_velocity_was_calculated", False)

        # Extract all capacity deductions (Q27-Q30).
        # See docs: "Scrum Standards" — capacity planning
        capacity = _extract_capacity_deductions(questionnaire)

        # Determine velocity source
        q9 = questionnaire.answers.get(9, "")
        if "from Jira" in q9 or "from jira" in q9:
            velocity_source = "jira"
        elif velocity_was_calculated:
            velocity_source = "estimated"
        else:
            velocity_source = "manual"

        # Compute per-sprint velocities — only sprints with bank holidays
        # get reduced capacity. Other deductions are applied uniformly.
        ts = extracted.get("team_size", 1)
        vel = extracted.get("velocity_per_sprint", ts * _VELOCITY_PER_ENGINEER)
        sprint_weeks = _parse_first_int(questionnaire.answers.get(8, "2 weeks")) or 2

        # Determine sprint start date — use _resolve_sprint_start_date which
        # computes the offset for future sprints (e.g. Sprint 107 when active is 104).
        sprint_start = _resolve_sprint_start_date(questionnaire)
        starting_sprint = -1  # default: no Jira
        q27_answer = questionnaire.answers.get(27, "")
        sprint_num_match = re.search(r"Sprint\s+(\d+)", q27_answer)
        if sprint_num_match:
            starting_sprint = int(sprint_num_match.group(1))

        # Small-mode "Add to <existing sprint>" — carry the target so the syncs
        # assign stories to that sprint instead of creating one. The trailing
        # digits (already captured above) also name the Sprint artifact after
        # the real board sprint.
        sprint_target_mode = ""
        target_sprint_name = ""
        target_sprint_external_id = ""
        if q27_answer.startswith("Add to "):
            sprint_target_mode = "existing"
            target_sprint_name = q27_answer.removeprefix("Add to ").strip()
            target_sprint_external_id = questionnaire._sprint_target_options.get(target_sprint_name, "")
            logger.info(
                "Sprint targeting: existing sprint %r (external_id=%r)",
                target_sprint_name,
                target_sprint_external_id or "resolve-by-name",
            )
        elif q27_answer == _SPRINT_TARGET_BACKLOG_CHOICE:
            # Small-mode "Backlog (no sprint)": stories are created but never
            # assigned — the syncs skip sprint creation/assignment entirely.
            sprint_target_mode = "backlog"
            logger.info("Sprint targeting: backlog — no sprint will be created or assigned")

        if _is_small_project_mode(questionnaire.intake_mode):
            # Small-project mode: no capacity planning. Net velocity equals gross,
            # and there is no per-sprint breakdown (project_analyzer caps the plan
            # at 1-2 sprints). Leave sprint_caps empty so the sprint_planner uses a
            # simple velocity × target_sprints allocation.
            sprint_caps = []
            net_vel = vel
        else:
            q10_nums = re.findall(r"\d+", questionnaire.answers.get(10, ""))
            target = int(q10_nums[-1]) if q10_nums else 6

            # Map detected holidays and PTO to sprint windows for per-sprint velocity
            holidays_by_sprint = _assign_holidays_to_sprints(
                questionnaire._detected_bank_holidays,
                sprint_start,
                sprint_weeks,
                target,
            )
            leave_by_sprint = _assign_leave_to_sprints(
                questionnaire._planned_leave_entries,
                sprint_start,
                sprint_weeks,
                target,
            )
            sprint_caps = _compute_per_sprint_velocities(
                team_size=ts,
                velocity_per_sprint=vel,
                sprint_length_weeks=sprint_weeks,
                target_sprints=target,
                holidays_by_sprint=holidays_by_sprint,
                planned_leave_days=capacity["capacity_planned_leave_days"],
                unplanned_leave_pct=capacity["capacity_unplanned_leave_pct"],
                onboarding_engineer_sprints=capacity["capacity_onboarding_engineer_sprints"],
                ktlo_engineers=capacity["capacity_ktlo_engineers"],
                discovery_pct=capacity["capacity_discovery_pct"],
                leave_by_sprint=leave_by_sprint,
            )

            # Use minimum per-sprint velocity as the conservative net velocity
            # (for backward compat and capacity overflow checks)
            if sprint_caps:
                net_vel = min(sc["net_velocity"] for sc in sprint_caps)
            else:
                net_vel = _compute_net_velocity(
                    team_size=ts,
                    velocity_per_sprint=vel,
                    sprint_length_weeks=sprint_weeks,
                    target_sprints=target,
                    bank_holiday_days=capacity["capacity_bank_holiday_days"],
                    planned_leave_days=capacity["capacity_planned_leave_days"],
                    unplanned_leave_pct=capacity["capacity_unplanned_leave_pct"],
                    onboarding_engineer_sprints=capacity["capacity_onboarding_engineer_sprints"],
                    ktlo_engineers=capacity["capacity_ktlo_engineers"],
                    discovery_pct=capacity["capacity_discovery_pct"],
                )

        # Apply user override if set
        if questionnaire._velocity_override is not None:
            net_vel = questionnaire._velocity_override
            # Override replaces per-sprint velocities too
            sprint_caps = [{**sc, "net_velocity": questionnaire._velocity_override} for sc in sprint_caps]

        # Build confirmation message with velocity info
        msg = "Great, your answers are locked in!"
        if extracted:
            velocity_note = f" (calculated as {ts} × {_VELOCITY_PER_ENGINEER})" if velocity_was_calculated else ""
            msg += f"\n\nPlanning with: **{ts} engineer(s), {vel} pts/sprint**{velocity_note}"
        if sprint_caps and any(sc["bank_holiday_days"] > 0 or sc.get("pto_days", 0) > 0 for sc in sprint_caps):
            # Show per-sprint breakdown when bank holidays or PTO affect specific sprints
            sprint_label_start = starting_sprint if starting_sprint > 0 else 1
            sprint_lines = []
            for sc in sprint_caps:
                label = f"Sprint {sprint_label_start + sc['sprint_index']}"
                annotations = []
                if sc["bank_holiday_names"]:
                    annotations.append(", ".join(sc["bank_holiday_names"]))
                if sc.get("pto_days", 0) > 0:
                    pto_names = ", ".join(f"{e['person']} {e['days']}d" for e in sc.get("pto_entries", []))
                    annotations.append(f"PTO: {pto_names}")
                if annotations:
                    sprint_lines.append(f"  {label}: **{sc['net_velocity']} pts** ({'; '.join(annotations)})")
                else:
                    sprint_lines.append(f"  {label}: **{sc['net_velocity']} pts**")
            msg += "\n**Per-sprint velocity** (after capacity deductions):\n" + "\n".join(sprint_lines)
        elif net_vel != vel:
            msg += f"\n**Net velocity: {net_vel} pts/sprint** (after capacity deductions)"
        if questionnaire._velocity_override is not None:
            msg += " *(user override)*"
        msg += "\n\n---\nI'll analyze your project next."

        return {
            "questionnaire": questionnaire,
            **extracted,
            **capacity,
            "net_velocity_per_sprint": net_vel,
            "sprint_capacities": sprint_caps,
            "velocity_source": velocity_source,
            "sprint_start_date": sprint_start,
            "starting_sprint_number": starting_sprint,
            "sprint_target_mode": sprint_target_mode,
            "target_sprint_name": target_sprint_name,
            "target_sprint_external_id": target_sprint_external_id,
            "planned_leave_entries": list(questionnaire._planned_leave_entries),
            # Promote the accepted references off the transient questionnaire
            # onto durable state — the analyzer, the summary and the exports
            # all read them from here, and a resumed session must keep them.
            #
            # `prior_art` is a plain LastValue channel, so whatever this node
            # returns *replaces* the channel. A headless run seeds the state
            # directly (`run_planning_pipeline(prior_art=…)`, and the MCP and
            # CLI surfaces behind it) and never walks the sub-loop, so
            # returning the empty questionnaire tuple here would silently
            # discard the caller's references before the analyzer ever sees
            # them. Fall back to what is already on the state instead.
            "prior_art": _accepted_prior_art(questionnaire) or tuple(state.get("prior_art") or ()),
            "messages": [AIMessage(content=msg)],
        }

    # ── Defaults command handling ─────────────────────────────────────
    # See docs: "Project Intake Questionnaire" — batch defaults
    #
    # When the user types "defaults", apply defaults to all remaining
    # questions in the current phase and advance past it. This lets users
    # fast-forward through phases they don't have strong opinions on.
    if _is_defaults_intent(last_msg.content):
        prev_phase = questionnaire.current_phase
        summary_lines, count = _batch_defaults_for_phase(questionnaire)

        # Advance to the next unskipped question after the current phase
        from yeaboi.agent.state import PHASE_QUESTION_RANGES

        _start, end = PHASE_QUESTION_RANGES[prev_phase]
        next_q = _next_unskipped_question(end + 1, questionnaire.skipped_questions)

        if count > 0:
            ack = f"Applied **{count}** default(s) for {PHASE_LABELS[prev_phase]}:\n" + "\n".join(summary_lines)
        else:
            ack = f"No remaining questions in {PHASE_LABELS[prev_phase]} needed defaults."

        if next_q is None:
            # All done — ask PTO or show summary
            return _show_summary_or_pto(questionnaire, prefix=f"{ack}\n\n")

        questionnaire.current_question = next_q
        question = _resolve_adaptive_text(next_q, questionnaire)
        new_phase = questionnaire.current_phase
        phase_label = PHASE_LABELS[new_phase]
        phase_intro = PHASE_INTROS.get(new_phase, "")
        intro_line = f"*{phase_intro}*\n\n" if phase_intro else ""
        suggest_line = _build_suggestion_line(questionnaire, next_q)
        return {
            "questionnaire": questionnaire,
            "messages": [
                AIMessage(
                    content=(
                        f"{ack}\n\n**{phase_label}** (Q{next_q}/{TOTAL_QUESTIONS})"
                        f"\n\n{intro_line}{question}{suggest_line}"
                    )
                )
            ],
        }

    # ── Skip / "I don't know" handling ───────────────────────────────
    # See docs: "Project Intake Questionnaire" — adaptive behavior
    #
    # Check for skip intent BEFORE the probed_questions check. This ensures
    # "skip" during a follow-up probe is handled correctly (keeps original
    # answer, advances) rather than combining "skip" as follow-up detail.
    if _is_skip_intent(last_msg.content):
        if current_q in questionnaire.probed_questions:
            # Skip during a follow-up probe — keep the original answer as-is
            ack = _build_skip_acknowledgment(current_q, during_probe=True, default=None)
        elif current_q in QUESTION_DEFAULTS:
            # Default available — store it and mark as defaulted
            default = QUESTION_DEFAULTS[current_q]
            questionnaire.answers[current_q] = default
            questionnaire.defaulted_questions.add(current_q)
            questionnaire.answer_sources[current_q] = AnswerSource.DEFAULTED
            ack = _build_skip_acknowledgment(current_q, during_probe=False, default=default)
        else:
            # Essential question with no default — flag the gap
            questionnaire.skipped_questions.add(current_q)
            ack = _build_skip_acknowledgment(current_q, during_probe=False, default=None)

        # Advance to next question (same logic as normal flow)
        next_q = _next_unskipped_question(current_q + 1, questionnaire.skipped_questions)
        if next_q is None:
            return _show_summary_or_pto(questionnaire, prefix=f"{ack}\n\n")

        prev_skip_phase = questionnaire.current_phase
        questionnaire.current_question = next_q
        question = _resolve_adaptive_text(next_q, questionnaire)
        new_skip_phase = questionnaire.current_phase
        phase_label = PHASE_LABELS[new_skip_phase]
        # Show phase intro when entering a new phase after a skip
        phase_intro = PHASE_INTROS.get(new_skip_phase, "") if new_skip_phase != prev_skip_phase else ""
        intro_line = f"*{phase_intro}*\n\n" if phase_intro else ""
        suggest_line = _build_suggestion_line(questionnaire, next_q)
        return {
            "questionnaire": questionnaire,
            "messages": [
                AIMessage(
                    content=(
                        f"{ack}\n\n**{phase_label}** (Q{next_q}/{TOTAL_QUESTIONS})"
                        f"\n\n{intro_line}{question}{suggest_line}"
                    )
                )
            ],
        }

    # ── Follow-up probing logic ──────────────────────────────────────
    # See docs: "Project Intake Questionnaire" — follow-up probing
    #
    # After each answer, the LLM judges whether it's specific enough.
    # If vague, one targeted follow-up is asked before moving on.
    # Max 1 follow-up per question — if still vague, accept and advance.
    #
    # Why inline probing (not batch at end)? Probing right after the
    # answer keeps context fresh. A batch pass would require the user
    # to context-switch back to earlier questions.
    #
    # Flow:
    #   IF current_question already probed → combine original + follow-up, advance
    #   ELSE → record answer, check vagueness:
    #     vague → mark probed, DON'T advance, return follow-up
    #     not vague → advance normally

    # Confirmation prefix for repo URL detection — set after _sync_platform_from_url
    # and prepended to the next AIMessage so the user sees immediate feedback.
    repo_confirm = ""

    if current_q in questionnaire.probed_questions:
        # This is a follow-up response.
        if current_q == 2 and questionnaire.answers.get(2) in _EXISTING_CODEBASE_ANSWERS:
            # The follow-up was our repo URL prompt — store answer in Q17, not Q2.
            # Q17 is "Can you share the repo URL(s)?", storing here lets the repo
            # scan and platform detection use it downstream.
            questionnaire.answers[17] = last_msg.content
            questionnaire.defaulted_questions.discard(17)
            _sync_platform_from_url(questionnaire)
            platform = questionnaire.answers.get(16, "")
            if platform:
                repo_confirm = f"*✓ {platform} repo detected — will be scanned during analysis.*\n\n"
        else:
            # Normal vagueness follow-up — combine original + follow-up detail.
            # Format: "{original}\n\n(Follow-up detail: {follow_up_answer})"
            # so downstream nodes get full context even if the follow-up is
            # incremental (e.g. "React and Node" expanding "A web app").
            original = questionnaire.answers.get(current_q, "")
            combined = f"{original}\n\n(Follow-up detail: {last_msg.content})"
            questionnaire.answers[current_q] = combined
        # Clear dynamic choices — they were consumed when the user answered.
        questionnaire._follow_up_choices.pop(current_q, None)
    else:
        # First answer to this question — record it and check vagueness.
        questionnaire.answers[current_q] = last_msg.content
        questionnaire.answer_sources[current_q] = AnswerSource.DIRECT

        # Q6 team member multi-select → recalculate velocity from selected members
        logger.info(
            "Q6 check: current_q=%s, _q6_member_select=%s, answer=%s",
            current_q,
            getattr(questionnaire, "_q6_member_select", "MISSING"),
            last_msg.content[:80],
        )
        if current_q == 6 and getattr(questionnaire, "_q6_member_select", False):
            _selected_text = last_msg.content
            logger.info("Q6 member select raw answer: %s", _selected_text[:200])
            # Parse member names from labels like "Name (X.X pts/sprint, discipline)"
            # Split on "), " which separates complete labels, then extract name before " ("
            import re as _q6_re

            _selected_names = []
            # Find all "Name (..." patterns
            _label_parts = _q6_re.findall(r"([^,;]+?\s*\([^)]+\))", _selected_text)
            if _label_parts:
                for lbl in _label_parts:
                    name = lbl.strip().split(" (")[0].strip()
                    if name:
                        _selected_names.append(name)
            else:
                # Fallback: try simple comma split (no parentheses in labels)
                for part in _selected_text.split(","):
                    name = part.strip().split(" (")[0].strip()
                    if name:
                        _selected_names.append(name)
            logger.info("Q6 parsed members: %s", _selected_names)
            if _selected_names:
                # Store clean display: "2 engineers (Alice, Bob)"
                _names_str = ", ".join(_selected_names)
                questionnaire.answers[6] = f"{len(_selected_names)} ({_names_str})"
                # Calculate velocity from selected members
                _ap_id = _effective_analysis_profile_id(state)
                if _ap_id:
                    try:
                        _, _ap_ex2 = _load_profile_by_id(_ap_id)
                        if _ap_ex2:
                            _vel = _calculate_velocity_for_members(_selected_names, _ap_ex2)
                            if _vel > 0:
                                questionnaire.answers[9] = (
                                    f"{_vel:.0f} points per sprint (from {len(_selected_names)} selected members)"
                                )
                                questionnaire.answer_sources[9] = AnswerSource.EXTRACTED
                                questionnaire.extracted_questions.add(9)
                                logger.info(
                                    "Q6 member select: %d members, velocity=%.0f",
                                    len(_selected_names),
                                    _vel,
                                )
                            # Auto-fill Q7 (team roles) from selected members' disciplines
                            _contrib_list = _ap_ex2.get("contributor_stats", [])
                            _roles: dict[str, int] = {}
                            for c in _contrib_list:
                                if isinstance(c, dict) and c.get("name", "").lower() in [
                                    n.lower() for n in _selected_names
                                ]:
                                    disc = c.get("top_discipline", "fullstack")
                                    # Map analysis disciplines to Q7 options
                                    _disc_map = {
                                        "backend": "Backend",
                                        "frontend": "Frontend",
                                        "fullstack": "Fullstack",
                                        "infrastructure": "DevOps/Infra",
                                        "devops": "DevOps/Infra",
                                        "ci-cd": "DevOps/Infra",
                                        "testing": "QA/Testing",
                                        "qa": "QA/Testing",
                                        "security": "DevOps/Infra",
                                        "data": "Data/ML",
                                        "design": "Design",
                                        "observability": "DevOps/Infra",
                                        "platform": "DevOps/Infra",
                                        "networking": "DevOps/Infra",
                                        "database": "Backend",
                                    }
                                    role = _disc_map.get(disc, "Fullstack")
                                    _roles[role] = _roles.get(role, 0) + 1
                            if _roles:
                                _role_parts = []
                                for role, count in sorted(_roles.items(), key=lambda x: -x[1]):
                                    _role_parts.append(f"{count} {role}" if count > 1 else role)
                                questionnaire.answers[7] = ", ".join(_role_parts)
                                questionnaire.answer_sources[7] = AnswerSource.EXTRACTED
                                questionnaire.extracted_questions.add(7)
                                questionnaire.defaulted_questions.discard(7)
                                logger.info("Q7 auto-filled from members: %s", questionnaire.answers[7])
                    except Exception:
                        pass
            questionnaire._q6_member_select = False
            questionnaire._follow_up_choices.pop(6, None)

        if current_q == 17:
            _sync_platform_from_url(questionnaire)
            platform = questionnaire.answers.get(16, "")
            if platform:
                repo_confirm = f"*✓ {platform} repo detected — will be scanned during analysis.*\n\n"

        # Q2 repo URL follow-up: when Q2 is answered "Existing codebase" or
        # "Hybrid" and Q17 is not yet set, ask for the repo URL immediately
        # before advancing to Q3. The answer is stored in Q17 (not combined
        # into Q2) so downstream repo scan and platform detection can use it.
        # Reuses probed_questions so the existing skip path handles it gracefully.
        if _needs_repo_url_prompt(questionnaire):
            questionnaire.probed_questions.add(2)
            return {
                "questionnaire": questionnaire,
                "messages": [AIMessage(content=_Q2_REPO_URL_PROMPT)],
            }

        # Skip vagueness check for choice questions — a selection from a
        # predefined list is never vague. Only probe free-text answers.
        if is_choice_question(current_q):
            vague_result = None
        else:
            vague_result = _check_vague_answer(INTAKE_QUESTIONS[current_q], last_msg.content, current_q)
        if vague_result:
            # Unpack follow-up question and dynamic choices from the LLM.
            # choices may be empty — the REPL gracefully degrades to open-ended.
            follow_up, choices = vague_result
            # Answer is vague — ask a follow-up. Don't advance current_question.
            # The follow-up message has NO phase label or progress indicator
            # — it feels like a conversational clarification, not a new question.
            questionnaire.probed_questions.add(current_q)
            questionnaire.answer_sources[current_q] = AnswerSource.PROBED
            # Store dynamic choices so the REPL can render a numbered menu.
            # Same lifecycle as probed_questions — cleared when follow-up is answered.
            if choices:
                questionnaire._follow_up_choices[current_q] = choices
            return {
                "questionnaire": questionnaire,
                "messages": [AIMessage(content=f"**Follow-up on Q{current_q}:**\n\n{follow_up}")],
            }

    # ── Q27 sprint selection handling (standard mode) ────────────────
    # See docs: "Scrum Standards" — capacity planning
    #
    # When Q27 is the current question and the user just answered it:
    # If Jira configured, the answer is a sprint selection (1/2/3/custom).
    # Parse it using resolve_sprint_selection. If not Jira, Q27 was auto-filled
    # so this path isn't reached.
    if current_q == 27 and current_q not in questionnaire.probed_questions:
        # Check if Q27 answer is a sprint selection response
        q27_answer = questionnaire.answers.get(27, "")
        if _is_tracker_configured() and not q27_answer.startswith("Fresh start"):
            # Try to parse as sprint selection — the active sprint number was
            # stored temporarily in the answer as "_active:N" by the Q27 prompt.
            active_match = re.search(r"_active:(\d+)", q27_answer)
            if active_match:
                active_num = int(active_match.group(1))
                user_answer = last_msg.content

                # The TUI/REPL resolves dynamic choices to the full option text
                # (e.g. "Sprint 105 (next)"), so try extracting the sprint number
                # from the resolved text first, then fall back to resolve_sprint_selection
                # for raw numeric input (e.g. "1", "105").
                sprint_num_match = re.search(r"Sprint\s+(\d+)", user_answer)
                if sprint_num_match:
                    resolved = int(sprint_num_match.group(1))
                else:
                    resolved = resolve_sprint_selection(user_answer, active_num)

                if resolved is not None and resolved > 0:
                    questionnaire.answers[27] = f"Sprint {resolved}"
                    # Clear dynamic choices — they were consumed
                    questionnaire._follow_up_choices.pop(27, None)
                    # Prepare Q28 bank holiday choices for the next question
                    _prepare_bank_holiday_choices(questionnaire)
                else:
                    # Invalid selection — re-ask
                    questionnaire.answers.pop(27, None)
                    return {
                        "questionnaire": questionnaire,
                        "messages": [AIMessage(content="Please pick 1–3, or type a sprint number.")],
                    }

    # ── Q28 bank holiday answer processing ────────────────────────────
    # When Q28 is answered, parse the user's choice to set the bank holiday count.
    if current_q == 28 and current_q not in questionnaire.probed_questions:
        q28_answer = questionnaire.answers.get(28, "")
        if "accept" in q28_answer.lower():
            # User accepted the detected count — keep _detected_bank_holiday_days as-is
            count = questionnaire._detected_bank_holiday_days
            questionnaire.answers[28] = f"{count} bank holiday(s)" if count > 0 else "No bank holidays"
        elif "no bank holidays" in q28_answer.lower():
            questionnaire._detected_bank_holiday_days = 0
            questionnaire.answers[28] = "No bank holidays"
        elif "enter manually" in q28_answer.lower():
            # User wants to enter manually — clear choices and re-ask as free text
            questionnaire._follow_up_choices.pop(28, None)
            questionnaire.answers.pop(28, None)
            questionnaire.probed_questions.add(28)
            return {
                "questionnaire": questionnaire,
                "messages": [AIMessage(content="How many bank/public holiday days fall in your planning window?")],
            }
        else:
            # Free-text answer (manual entry or follow-up) — parse the number
            parsed = _parse_first_int(q28_answer)
            if parsed is not None:
                questionnaire._detected_bank_holiday_days = parsed
                questionnaire.answers[28] = f"{parsed} bank holiday(s)"
        questionnaire._follow_up_choices.pop(28, None)

    # ── Advance to next question ─────────────────────────────────────
    # _next_unskipped_question scans forward, skipping any questions that
    # were already answered from the initial description.
    next_q = _next_unskipped_question(current_q + 1, questionnaire.skipped_questions)

    if next_q is None:
        # All remaining questions have been answered or skipped —
        # ask PTO or show summary and wait for confirmation.
        return _show_summary_or_pto(questionnaire, prefix=repo_confirm)

    prev_advance_phase = questionnaire.current_phase
    questionnaire.current_question = next_q

    # ── Q27 sprint selection auto-handling when advancing ─────────────
    # See docs: "Scrum Standards" — capacity planning
    #
    # When advancing to Q27 in standard mode:
    # - No Jira: auto-fill with bank holiday detection and skip to Q28
    # - Jira: fetch active sprint and present options
    if next_q == 27:
        if not _is_tracker_configured():
            # No tracker — auto-fill Q27 as "Fresh start (today)" and advance to Q28
            _derive_q27_from_locale(questionnaire)
            # Prepare bank holiday choices for Q28
            _prepare_bank_holiday_choices(questionnaire)
            # Advance past Q27 to Q28 (bank holidays)
            next_q = _next_unskipped_question(28, questionnaire.skipped_questions)
            if next_q is None:
                return _show_summary_or_pto(questionnaire, prefix=repo_confirm)
            questionnaire.current_question = next_q
            prev_advance_phase = QuestionnairePhase.CAPACITY_PLANNING
        else:
            # Tracker configured — fetch active sprint/iteration and show selection as a choice menu
            from yeaboi import trackers as _std_registry

            _pref_std = questionnaire._preferred_tracker
            _std_tracker_label = _std_registry.label_for(_pref_std) or "Jira"
            active_num, active_start, jira_status = _fetch_active_sprint_number(_pref_std)
            if active_num is not None:
                # Store active sprint number and start date in transient fields
                questionnaire._active_sprint_number = active_num
                questionnaire._active_sprint_start_date = active_start
                questionnaire.answers[27] = f"_active:{active_num}"
                # Populate dynamic choices so the TUI accordion / REPL renders a proper
                # numbered menu instead of a free-text input box.
                questionnaire._follow_up_choices[27] = (
                    f"Sprint {active_num + 1} (next)",
                    f"Sprint {active_num + 2}",
                    f"Sprint {active_num + 3}",
                )
                phase_label = PHASE_LABELS[questionnaire.current_phase]
                phase_intro = PHASE_INTROS.get(questionnaire.current_phase, "")
                intro_line = f"*{phase_intro}*\n\n" if phase_intro else ""
                prompt = (
                    f"Detected active sprint in {_std_tracker_label}: **Sprint {active_num}**.\n\n"
                    f"Which sprint are you planning for?"
                )
                return {
                    "questionnaire": questionnaire,
                    "messages": [
                        AIMessage(
                            content=(
                                f"{repo_confirm}**{phase_label}** (Q{next_q}/{TOTAL_QUESTIONS})\n\n{intro_line}{prompt}"
                            )
                        )
                    ],
                }
            else:
                # Jira configured but couldn't fetch sprint — fall back with feedback
                logger.warning("Jira sprint fetch failed: %s", jira_status)
                _derive_q27_from_locale(questionnaire)
                _prepare_bank_holiday_choices(questionnaire)
                next_q = _next_unskipped_question(28, questionnaire.skipped_questions)
                if next_q is None:
                    return _show_summary_or_pto(questionnaire, prefix=repo_confirm)
                questionnaire.current_question = next_q
                prev_advance_phase = QuestionnairePhase.CAPACITY_PLANNING

    # Ask the next question, with phase label and progress indicator.
    # The phase label changes when transitioning between phases (e.g. Q5→Q6).
    question = INTAKE_QUESTIONS[questionnaire.current_question]
    new_advance_phase = questionnaire.current_phase
    phase_label = PHASE_LABELS[new_advance_phase]
    # Show phase intro when entering a new phase
    phase_intro = PHASE_INTROS.get(new_advance_phase, "") if new_advance_phase != prev_advance_phase else ""
    intro_line = f"*{phase_intro}*\n\n" if phase_intro else ""
    suggest_line = _build_suggestion_line(questionnaire, questionnaire.current_question)
    return {
        "questionnaire": questionnaire,
        "messages": [
            AIMessage(
                content=(
                    f"{repo_confirm}"
                    f"**{phase_label}** (Q{questionnaire.current_question}/{TOTAL_QUESTIONS})"
                    f"\n\n{intro_line}{question}{suggest_line}"
                )
            )
        ],
    }


# ── Project analyzer ────────────────────────────────────────────────
# See docs: "Architecture" — project_analyzer sits between intake and agent
# See docs: "Scrum Standards" — project analysis
#
# After the user confirms the 26-question intake questionnaire, this node
# synthesizes all answers into a structured ProjectAnalysis dataclass.
# Downstream nodes (feature_generator, story_writer, sprint_planner) read
# ProjectAnalysis instead of re-parsing raw conversation history.
#
# Why LLM-powered extraction (not deterministic)?
# The 26 answers are natural language — extracting structured fields like
# "goals" from free-text like "We want to improve onboarding and reduce churn"
# requires semantic understanding. A single LLM call with a JSON-schema prompt
# handles this, with a deterministic fallback on parse failure.


def _build_answers_block(questionnaire: QuestionnaireState) -> str:
    """Format all 26 Q&A pairs for the analyzer prompt.

    Marks defaulted answers with *(assumed default)* and skipped questions
    with *(skipped)* so the LLM can flag them as assumptions.

    Args:
        questionnaire: The completed questionnaire with all answers.

    Returns:
        A formatted string with one Q/A pair per block.
    """
    lines: list[str] = []
    for q_num in range(1, TOTAL_QUESTIONS + 1):
        question = INTAKE_QUESTIONS[q_num]
        answer = questionnaire.answers.get(q_num)
        marker = ""
        if q_num in questionnaire.extracted_questions:
            source = "SCRUM.md" if q_num in questionnaire._scrum_md_questions else "description"
            marker = f" *(extracted from {source})*"
        elif q_num in questionnaire.defaulted_questions:
            marker = " *(assumed default)*"
        elif answer is None:
            marker = " *(skipped)*"
        display_answer = answer if answer is not None else "(no answer)"
        lines.append(f"Q{q_num}. {question}\nA: {display_answer}{marker}\n")
    return "\n".join(lines)


def _parse_architecture(raw: object) -> ArchitectureDecision | None:
    """Parse the analyzer's architecture block with deterministic guardrails.

    # See docs: "Scrum Standards" — DoD Spike
    #
    # Clamps to at most 3 options, drops nameless ones, normalizes confidence
    # to high/medium/low, and coerces `chosen` to a real option name. Returns
    # None (no architecture section, no spike downstream) for anything that
    # isn't a dict with at least one named option — including the deterministic
    # fallback path, which never fabricates options.
    """
    if not isinstance(raw, dict):
        return None

    def to_str_tuple(val: object) -> tuple[str, ...]:
        if isinstance(val, list):
            return tuple(str(item) for item in val if item)
        if isinstance(val, str) and val.strip():
            return (val,)
        return ()

    options: list[ArchitectureOption] = []
    raw_options = raw.get("options")
    if isinstance(raw_options, list):
        for item in raw_options[:3]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            options.append(
                ArchitectureOption(
                    name=name,
                    summary=str(item.get("summary", "")).strip(),
                    pros=to_str_tuple(item.get("pros")),
                    cons=to_str_tuple(item.get("cons")),
                )
            )
    if not options:
        return None

    confidence = str(raw.get("confidence", "")).strip().lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"
    chosen = str(raw.get("chosen", "")).strip()
    if chosen not in {o.name for o in options}:
        chosen = options[0].name

    return ArchitectureDecision(
        options=tuple(options),
        chosen=chosen,
        confidence=confidence,
        rationale=str(raw.get("rationale", "")).strip(),
        pinned_by_constraint=bool(raw.get("pinned_by_constraint", False)) or len(options) == 1,
    )


def _parse_analysis_response(
    raw: str,
    questionnaire: QuestionnaireState,
    team_size: int,
    velocity: int,
) -> ProjectAnalysis:
    """Parse the LLM's JSON response into a ProjectAnalysis dataclass.

    Strips markdown code fences, parses JSON, and converts list fields to
    tuples (frozen dataclass requires immutable sequences). Falls back to
    _build_fallback_analysis on any parse error.

    Args:
        raw: The raw LLM response string (expected to be JSON).
        questionnaire: The completed questionnaire (for fallback).
        team_size: Team size (for fallback).
        velocity: Velocity per sprint (for fallback).

    Returns:
        A ProjectAnalysis instance.
    """
    try:
        # Strip a leaked <think> block, then the markdown fences LLMs wrap JSON
        # in. Reasoning left in place defeats the fence strip and the parse falls
        # through to the deterministic fallback with no error.
        text = strip_think_tags(raw).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return _build_fallback_analysis(questionnaire, team_size, velocity)

        # Helper to safely convert list-like values to tuple[str, ...]
        def to_str_tuple(val: object) -> tuple[str, ...]:
            if isinstance(val, list):
                return tuple(str(item) for item in val if item)
            if isinstance(val, str) and val.strip():
                return (val,)
            return ()

        # Parse sprint_length_weeks with default of 2
        sprint_weeks_raw = parsed.get("sprint_length_weeks", 2)
        try:
            sprint_weeks = int(sprint_weeks_raw)
        except (ValueError, TypeError):
            sprint_weeks = 2

        # Parse target_sprints with default of 0
        target_raw = parsed.get("target_sprints", 0)
        try:
            target_sprints = int(target_raw)
        except (ValueError, TypeError):
            target_sprints = 0

        return ProjectAnalysis(
            project_name=str(parsed.get("project_name", "Untitled Project")),
            project_description=str(parsed.get("project_description", "")),
            project_type=str(parsed.get("project_type", "unknown")),
            goals=to_str_tuple(parsed.get("goals")),
            end_users=to_str_tuple(parsed.get("end_users")),
            target_state=str(parsed.get("target_state", "")),
            tech_stack=to_str_tuple(parsed.get("tech_stack")),
            integrations=to_str_tuple(parsed.get("integrations")),
            constraints=to_str_tuple(parsed.get("constraints")),
            sprint_length_weeks=sprint_weeks,
            target_sprints=target_sprints,
            risks=to_str_tuple(parsed.get("risks")),
            out_of_scope=to_str_tuple(parsed.get("out_of_scope")),
            assumptions=to_str_tuple(parsed.get("assumptions")),
            # Deterministic guardrail: only allow skip_features when the project is
            # genuinely small (target_sprints ≤ 2 AND goals ≤ 3). The LLM may
            # over-eagerly set skip_features=true for larger projects.
            skip_features=bool(parsed.get("skip_features", False))
            and target_sprints <= 2
            and len(to_str_tuple(parsed.get("goals"))) <= 3,
            is_low_code=bool(parsed.get("is_low_code", False)),
            low_code_reason=str(parsed.get("low_code_reason", "")),
            scrum_md_contributions=to_str_tuple(parsed.get("scrum_md_contributions")),
            architecture=_parse_architecture(parsed.get("architecture")),
        )

    except Exception:
        logger.debug("Failed to parse analysis JSON, falling back to deterministic extraction", exc_info=True)
        return _build_fallback_analysis(questionnaire, team_size, velocity)


def _build_fallback_analysis(
    questionnaire: QuestionnaireState,
    team_size: int,
    velocity: int,
) -> ProjectAnalysis:
    """Build a best-effort ProjectAnalysis from raw answers when LLM fails.

    Deterministic extraction — pulls answers directly by question number.
    No LLM call, so this always succeeds. The result may be less polished
    than the LLM version but captures all the raw data.

    Args:
        questionnaire: The completed questionnaire with all answers.
        team_size: Team size from state.
        velocity: Velocity per sprint from state.

    Returns:
        A ProjectAnalysis with fields populated from raw answers.
    """
    answers = questionnaire.answers

    # Parse sprint length from Q8 answer
    sprint_weeks = 2
    q8 = answers.get(8, "")
    sprint_int = _parse_first_int(q8) if q8 else None
    if sprint_int and 1 <= sprint_int <= 4:
        sprint_weeks = sprint_int

    # Parse target sprints from Q10 answer.
    # Q10 uses ranges like "3–5 sprints" — extract the upper bound so the
    # planner has room to spread work. "No preference" has no digits → 0.
    target_sprints = 0
    q10 = answers.get(10, "")
    if q10:
        q10_nums = re.findall(r"\d+", q10)
        if q10_nums:
            target_sprints = int(q10_nums[-1])  # upper bound of range

    # Collect assumptions from defaulted/skipped questions
    assumptions: list[str] = []
    for q_num in sorted(questionnaire.defaulted_questions):
        assumptions.append(f"Q{q_num}: used default — {answers.get(q_num, 'N/A')}")
    for q_num in sorted(questionnaire.skipped_questions - set(answers.keys())):
        assumptions.append(f"Q{q_num}: skipped with no answer")

    return ProjectAnalysis(
        project_name=answers.get(1, "Untitled Project")[:50],
        project_description=answers.get(1, ""),
        project_type=answers.get(2, "unknown").lower().strip(),
        goals=(answers.get(3, ""),) if answers.get(3) else (),
        end_users=(answers.get(3, ""),) if answers.get(3) else (),
        target_state=answers.get(4, ""),
        tech_stack=(answers.get(11, ""),) if answers.get(11) else (),
        integrations=(answers.get(12, ""),) if answers.get(12) else (),
        constraints=(answers.get(13, ""),) if answers.get(13) else (),
        sprint_length_weeks=sprint_weeks,
        target_sprints=target_sprints,
        risks=(answers.get(21, ""),) if answers.get(21) else (),
        out_of_scope=(answers.get(23, ""),) if answers.get(23) else (),
        assumptions=tuple(assumptions),
        scrum_md_contributions=(),
    )


def _format_architecture_section(architecture: ArchitectureDecision | None) -> str:
    """Render the analysis review's Architecture block ("" when absent).

    Shows WHAT was considered — every option with pros/cons — beside the
    recommendation and its confidence, so the pick is a reviewable decision
    rather than a silent default.
    """
    if architecture is None or not architecture.options:
        return ""
    lines = [
        "## Architecture",
        f"  **Recommended:** {architecture.chosen} · confidence: **{architecture.confidence}**",
    ]
    if architecture.rationale:
        lines.append(f"  {architecture.rationale}")
    for opt in architecture.options:
        marker = "✓ " if opt.name == architecture.chosen else ""
        summary = f" — {opt.summary}" if opt.summary else ""
        lines.append(f"  - {marker}**{opt.name}**{summary}")
        if opt.pros:
            lines.append(f"      pros: {'; '.join(opt.pros)}")
        if opt.cons:
            lines.append(f"      cons: {'; '.join(opt.cons)}")
    if architecture.pinned_by_constraint:
        lines.append("  _(decision already pinned — no validation spike needed)_")
    else:
        lines.append(
            "  _(the choice is open — you'll be asked whether to add a validation spike; "
            "to pick a different option, choose Edit and say so)_"
        )
    return "\n".join(lines) + "\n"


def _format_analysis(
    analysis: ProjectAnalysis,
    *,
    sprint_capacities: list[dict] | None = None,
    net_velocity: int | None = None,
    velocity_per_sprint: int | None = None,
    team_size: int | None = None,
    velocity_source: str | None = None,
) -> str:
    """Format a ProjectAnalysis as a markdown display for the user.

    Matches the project's intake summary style — sections with bullet points.
    The REPL renders this as a Rich panel and waits for user
    input before routing to the main agent.

    Args:
        analysis: The completed ProjectAnalysis to display.
        sprint_capacities: Per-sprint velocity breakdown (from capacity analysis).
        net_velocity: Net velocity after deductions.
        velocity_per_sprint: Gross velocity before deductions.
        team_size: Number of engineers.
        velocity_source: How velocity was determined ("jira", "estimated", "manual").

    Returns:
        A formatted markdown string.
    """

    def _bullet_list(items: tuple[str, ...]) -> str:
        if not items:
            return "  - _(none)_"
        return "\n".join(f"  - {item}" for item in items)

    low_code_line = ""
    if analysis.is_low_code:
        reason = f" — {analysis.low_code_reason}" if analysis.low_code_reason else ""
        low_code_line = f"**⚙ Low-code project{reason}** (estimates and tasks scaled lighter)\n\n"

    sections = [
        f"# Project Analysis: {analysis.project_name}\n",
        f"**Description:** {analysis.project_description}\n",
        f"**Type:** {analysis.project_type}\n",
        low_code_line,
        f"## Goals\n{_bullet_list(analysis.goals)}\n",
        f"## End Users\n{_bullet_list(analysis.end_users)}\n",
        f"## Target State\n{analysis.target_state or '_(not specified)_'}\n",
        f"## Tech Stack\n{_bullet_list(analysis.tech_stack)}\n",
        _format_architecture_section(analysis.architecture),
        f"## Integrations\n{_bullet_list(analysis.integrations)}\n",
        f"## Constraints\n{_bullet_list(analysis.constraints)}\n",
        f"## Sprint Planning\n"
        f"  - Sprint length: **{analysis.sprint_length_weeks} week(s)**\n"
        f"  - Target sprints: **{analysis.target_sprints or 'scope-based'}**\n",
        f"## Risks\n{_bullet_list(analysis.risks)}\n",
        f"## Out of Scope\n{_bullet_list(analysis.out_of_scope)}\n",
    ]

    # ── Capacity Analysis section ─────────────────────────────────────
    # Shows the velocity calculation and per-sprint breakdown when bank
    # holidays affect specific sprints. This gives the user visibility into
    # exactly how planning capacity was derived before the agent proceeds.
    if net_velocity is not None and velocity_per_sprint is not None:
        ts = team_size or 1
        gross = velocity_per_sprint
        source_label = {"jira": "from Jira", "estimated": "estimated", "manual": "manual"}.get(
            velocity_source or "", ""
        )
        cap_lines = [
            "## Capacity",
            f"  - Team: **{ts} engineer(s)** · Gross velocity: **{gross} pts/sprint**"
            + (f" ({source_label})" if source_label else ""),
        ]

        has_per_sprint = sprint_capacities and any(sc.get("bank_holiday_days", 0) > 0 for sc in sprint_capacities)
        if has_per_sprint:
            cap_lines.append("  - Per-sprint breakdown:")
            total_pts = 0
            for sc in sprint_capacities:
                idx = sc["sprint_index"] + 1
                nv = sc["net_velocity"]
                total_pts += nv
                if sc["bank_holiday_names"]:
                    names = ", ".join(sc["bank_holiday_names"])
                    cap_lines.append(f"    - Sprint {idx}: **{nv} pts** (−{sc['bank_holiday_days']}d: {names})")
                else:
                    cap_lines.append(f"    - Sprint {idx}: **{nv} pts**")
            cap_lines.append(f"  - Total capacity: **{total_pts} pts** across {len(sprint_capacities)} sprints")
        cap_lines.append(f"  - Net velocity: **{net_velocity} pts/sprint**")
        sections.append("\n".join(cap_lines) + "\n")

    if analysis.assumptions:
        sections.append(f"## Assumptions\n{_bullet_list(analysis.assumptions)}\n")

    if analysis.scrum_md_contributions:
        fields = ", ".join(analysis.scrum_md_contributions)
        sections.append(f"## User Docs Enriched\n  - {fields}\n")

    sections.append("")  # Trailing newline for clean formatting

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Prompt quality scoring — deterministic rating from QuestionnaireState
# ---------------------------------------------------------------------------
# See docs: "Scrum Standards" — prompt quality rating
#
# Pure function: computes a quality score from the questionnaire tracking sets
# (answered, extracted, defaulted, skipped, probed). No LLM call. Used by
# project_analyzer to attach a PromptQualityRating to the ProjectAnalysis.

# Essential questions worth 5 pts each; all others worth 2 pts each.
_ESSENTIAL_QUESTIONS: frozenset[int] = frozenset({1, 2, 3, 4, 6, 11, 15})
_ESSENTIAL_WEIGHT = 5
_OTHER_WEIGHT = 2
_PROBING_BONUS = 1
_DEFAULTED_FACTOR = 0.4  # defaulted answers get 40% of points
_MAX_SUGGESTIONS = 4
# High-value non-essential questions — worth suggesting when defaulted/skipped.
# These provide context that significantly improves analysis quality (repo URLs,
# existing docs) but aren't strictly required to generate a plan.
_HIGH_VALUE_QUESTIONS: frozenset[int] = frozenset({14, 17})


def compute_prompt_quality(qs: QuestionnaireState, *, has_user_context: bool = False) -> PromptQualityRating:
    """Compute a deterministic prompt quality rating from questionnaire tracking sets.

    Scoring formula:
    - 7 essential questions (Q1-Q4, Q6, Q11, Q15): 5 pts each = 35 pts max
    - 19 other questions: 2 pts each = 38 pts max
    - Probing bonus: 1 pt per probed question
    - Total ~78 pts max, normalized to percentage

    Deductions:
    - User-answered or extracted from description → full points
    - Defaulted → 40% of points
    - Skipped (no answer at all) → 0 points

    Grade: A (≥85%), B (≥70%), C (≥50%), D (<50%)

    Args:
        qs: The completed QuestionnaireState with tracking sets populated.
        has_user_context: Whether a SCRUM.md file was loaded. When False,
            a suggestion to add one is included.

    Returns:
        A PromptQualityRating with score, grade, counts, and suggestions.
    """
    total_possible = 0.0
    total_earned = 0.0
    answered_count = 0
    extracted_count = 0
    defaulted_count = 0
    skipped_count = 0

    for q_num in range(1, TOTAL_QUESTIONS + 1):
        weight = _ESSENTIAL_WEIGHT if q_num in _ESSENTIAL_QUESTIONS else _OTHER_WEIGHT
        total_possible += weight

        if q_num in qs.extracted_questions:
            # Extracted from description — full points
            total_earned += weight
            extracted_count += 1
        elif q_num in qs.defaulted_questions:
            # Defaulted — partial credit
            total_earned += weight * _DEFAULTED_FACTOR
            defaulted_count += 1
        elif q_num in qs.answers:
            # User answered directly — full points
            total_earned += weight
            answered_count += 1
        else:
            # Skipped with no answer — 0 points
            skipped_count += 1

    # Probing bonus — 1 pt per probed question (shows engagement)
    probed_count = len(qs.probed_questions)
    total_earned += probed_count * _PROBING_BONUS
    total_possible += probed_count * _PROBING_BONUS  # keep ratio fair

    # Normalize to percentage
    score_pct = round((total_earned / total_possible) * 100) if total_possible > 0 else 0

    # Grade thresholds
    if score_pct >= 85:
        grade = "A"
    elif score_pct >= 70:
        grade = "B"
    elif score_pct >= 50:
        grade = "C"
    else:
        grade = "D"

    # Generate suggestions — essential questions first, then high-value, then SCRUM.md
    suggestions: list[str] = []
    # Essential questions that were defaulted or skipped
    for q_num in sorted(_ESSENTIAL_QUESTIONS):
        if len(suggestions) >= _MAX_SUGGESTIONS:
            break
        if q_num in qs.defaulted_questions or (q_num not in qs.answers and q_num not in qs.extracted_questions):
            hint = QUESTION_IMPROVEMENT_HINTS.get(q_num)
            if hint:
                suggestions.append(f"{hint} (Q{q_num})")
    # High-value non-essential questions (repo URL, docs) that were defaulted or skipped
    for q_num in sorted(_HIGH_VALUE_QUESTIONS):
        if len(suggestions) >= _MAX_SUGGESTIONS:
            break
        if q_num in qs.defaulted_questions or (q_num not in qs.answers and q_num not in qs.extracted_questions):
            hint = QUESTION_IMPROVEMENT_HINTS.get(q_num)
            if hint:
                suggestions.append(f"{hint} (Q{q_num})")
    # SCRUM.md suggestion — shown when no user context file was loaded
    if not has_user_context and len(suggestions) < _MAX_SUGGESTIONS:
        suggestions.append(SCRUM_MD_HINT)

    # Low-confidence areas — essential questions that were defaulted.
    # These are flagged as assumptions in ProjectAnalysis for downstream
    # spike recommendations. Uses answer_sources when available.
    low_confidence: list[str] = []
    for q_num in sorted(ESSENTIAL_QUESTIONS):
        is_defaulted = (
            qs.answer_sources.get(q_num) == AnswerSource.DEFAULTED
            if qs.answer_sources
            else q_num in qs.defaulted_questions
        )
        if is_defaulted:
            label = QUESTION_SHORT_LABELS.get(q_num, f"Q{q_num}")
            low_confidence.append(label)

    return PromptQualityRating(
        score_pct=score_pct,
        grade=grade,
        answered_count=answered_count,
        extracted_count=extracted_count,
        defaulted_count=defaulted_count,
        skipped_count=skipped_count,
        probed_count=probed_count,
        suggestions=tuple(suggestions),
        low_confidence_areas=tuple(low_confidence),
    )


def _gather_performance_summary(selection=None) -> str:
    """Return the team's Performance signal as a markdown block (empty if unused).

    Thin, graceful wrapper around performance.gather_performance_context so the
    analyzer / planner stay decoupled from the Performance package internals and a
    missing DB or unused mode never affects a plan. A ``selection`` with
    performance switched off yields "". See README: "Performance Mode".
    """
    try:
        from yeaboi.performance.context import gather_performance_context

        return gather_performance_context(selection=selection).summary_md
    except Exception:  # noqa: BLE001 — performance context is best-effort
        logger.debug("_gather_performance_summary failed (non-fatal)", exc_info=True)
        return ""


def _wants_integration(state, key: str) -> bool:
    """Whether this plan may consult the ``key`` connection (absent ``session_integrations`` = all on)."""
    allowed = state.get("session_integrations")
    return allowed is None or key in set(allowed)


def _skipped_source(name: str) -> dict:
    return {"name": name, "status": "skipped", "detail": "not enabled for this plan"}


def _wants_dep(state, source: str) -> bool:
    """Whether this run's context scope allows ``source`` (absent key = all on).

    Reads the ``context_scope`` state key seeded by run_planning_pipeline and
    the chat/TUI launch sites — the JSON twin of ``context.ContextScope``.
    """
    raw = state.get("context_scope", "")
    if not raw:
        return True
    from yeaboi.context.scope import coerce_scope, wants

    try:
        return wants(coerce_scope(raw), source)
    except (TypeError, ValueError):
        logger.warning("context_scope on state is unreadable — reading every source")
        return True


def _state_scope(state):
    """The run's resolved ``Selection``, or ``None`` when the state carries no scope.

    Resolved on each call: a run must see what was recorded since the last
    one, and only two nodes ask.
    """
    raw = state.get("context_scope", "")
    if not raw:
        return None
    from yeaboi.context.resolve import resolve_scope

    try:
        return resolve_scope(str(raw))
    except (TypeError, ValueError):
        logger.warning("context_scope on state is unreadable — reading every source")
        return None


def _effective_analysis_profile_id(state) -> str:
    """The state's analysis profile id, unless the analysis source is switched off.

    The scope beats an explicit pick: with ``analysis`` off, a seeded or
    resumed profile id is dropped (logged) so intake never auto-fills from it.
    """
    pid = state.get("analysis_profile_id", "") or ""
    if pid and not _wants_dep(state, "analysis"):
        logger.info("analysis source off — dropping analysis profile %s for this run", pid)
        return ""
    return pid


#: How far back the planner's production note looks. Wider than a sprint on
#: purpose: an incident rate is the thing that bears on capacity, and one
#: sprint's worth of incidents is a mood, not a rate.
_OPS_PLANNING_WINDOW_DAYS = 30


def _gather_ops_summary() -> str:
    """Production over the last month as a markdown block ('' when unused).

    Prose only, and prose the planner is told to treat as a constraint it may
    not compute with. Nothing here touches velocity, capacity or confidence:
    silently haircutting a sprint because Datadog was chatty is exactly the
    behaviour the conflicts vocabulary was written to replace.
    """
    try:
        from yeaboi.connectors import registry

        if not registry.any_fetchable():
            return ""
        from yeaboi.connectors.fetching import gather
        from yeaboi.ops.signals import describe

        result = gather(since=f"{_OPS_PLANNING_WINDOW_DAYS}d")
        if not result.signals:
            return ""
        lines = [f"Over the last {_OPS_PLANNING_WINDOW_DAYS} days, the connected monitoring tools saw:"]
        lines += [f"- {describe(sig)}" for sig in result.signals]
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 — production context is best-effort
        logger.debug("_gather_ops_summary failed (non-fatal)", exc_info=True)
        return ""


def project_analyzer(state: ScrumState) -> dict:
    """LangGraph node: synthesize intake answers into a structured ProjectAnalysis.

    # See docs: "Agentic Blueprint Reference" — node return format
    # See docs: "Architecture" — project_analyzer node
    #
    # How this works:
    # 1. Read the confirmed questionnaire answers + team_size + velocity from state.
    # 2. Build a formatted answers block for the LLM prompt.
    # 3. Call the LLM with the analyzer prompt (temperature=0.0 for deterministic JSON).
    # 4. Parse the JSON response into a ProjectAnalysis dataclass.
    # 5. Return the analysis + populate ScrumState metadata fields.
    #
    # Why this returns to END (not to agent)?
    # Same pattern as feature_generator — the node produces output, the REPL
    # displays it, and the user reviews with [Accept / Edit / Reject]. On
    # accept, route_entry sees project_analysis populated and routes to
    # the next pipeline node.

    Args:
        state: The current LangGraph state with completed questionnaire.

    Returns:
        A dict updating project_analysis, project_name, project_description,
        sprint_length_weeks, target_sprints, and messages.
    """
    questionnaire = state["questionnaire"]
    team_size = state.get("team_size", 1)
    velocity = state.get("velocity_per_sprint", team_size * _VELOCITY_PER_ENGINEER)

    # See docs: "Guardrails" — human-in-the-loop pattern
    # Read review state from previous edit decision. When present, the
    # REPL has cleared the old analysis and set last_review_decision/feedback
    # so this node regenerates with user feedback injected into the prompt.
    review_decision = state.get("last_review_decision")
    review_feedback = state.get("last_review_feedback", "")
    review_mode = review_decision.value if review_decision else None

    # For edit mode, extract previous output from the feedback string.
    # The REPL packs it as: "{feedback}\n\n---PREVIOUS OUTPUT---\n{serialized}"
    previous_output = None
    if review_mode == "edit" and "---PREVIOUS OUTPUT---" in review_feedback:
        parts = review_feedback.split("---PREVIOUS OUTPUT---", 1)
        review_feedback = parts[0].strip()
        previous_output = parts[1].strip()

    # Repository signals — grounds analysis in real codebase data and drives the
    # deterministic low-code verdict + the detected-stack hint. To avoid scanning
    # the repo twice, reuse the raw scan stashed by project_intake if present;
    # either way analyze_context re-derives the structured signals (cheap, pure)
    # from the current questionnaire answers.
    # See docs: "Project Intake Questionnaire" — smart intake
    _stashed_repo_context = getattr(questionnaire, "_repo_context", "")
    repo_key = _repo_integration(questionnaire)
    if _stashed_repo_context:
        repo_context = _stashed_repo_context
        repo_status = {"name": "Repository", "status": "success", "detail": "reused intake scan"}
    elif repo_key and not _wants_integration(state, repo_key):
        repo_context, repo_status = None, _skipped_source("Repository")
    else:
        # _scan_repo_context is kept as the analysis-time scan seam (many
        # integration/golden tests monkeypatch it to skip live scans).
        repo_context, repo_status = _scan_repo_context(questionnaire)
    # Re-derive structured signals from the raw scan (pure, no I/O) so the
    # detected-stack hint and low-code verdict reflect the current answers.
    repo_signals = analyze_context(
        repo_context or "",
        description=questionnaire.answers.get(1, "") or "",
        tech_stack=questionnaire.answers.get(11, "") or "",
    )

    # Load SCRUM.md first — the user's own project context file. Loaded before
    # Confluence so that any Confluence URLs in SCRUM.md can be fetched directly.
    # Similar to CLAUDE.md for Claude Code: a free-form markdown file containing URLs,
    # design notes, tech decisions, and anything else the user wants the agent to know.
    user_context, user_status = _load_user_context()
    # What the person pointed at during intake (@-references, attached files)
    # reads as project context alongside SCRUM.md.
    pasted_context = list(state.get("pasted_context") or [])
    if pasted_context:
        user_context = "\n\n".join(part for part in (user_context or "", *pasted_context) if part)
        logger.info(
            "project_analyzer: %d pasted reference/file paragraph(s) join the user context", len(pasted_context)
        )

    # Search Confluence for docs related to the project name AND fetch any pages
    # linked directly in SCRUM.md (e.g. RunBook URLs). Passing user_context allows
    # _fetch_confluence_context to extract page IDs from Confluence URLs.
    # Returns (None, status) gracefully when Confluence is not configured or no docs found.
    # See docs: "Tools" — read-only tool pattern
    logger.debug(
        "CONFLUENCE: passing user_context=%s to _fetch_confluence_context", "present" if user_context else "None"
    )
    if _wants_integration(state, "confluence"):
        confluence_context, confluence_status = _fetch_confluence_context(questionnaire, user_context=user_context)
    else:
        confluence_context, confluence_status = None, _skipped_source("Confluence")
    logger.debug(
        "CONFLUENCE: result status=%s detail=%s", confluence_status.get("status"), confluence_status.get("detail")
    )

    # Same two-strategy discovery for Notion (keyword search + direct page fetch
    # from URLs in SCRUM.md). Notion is an independent doc source with its own
    # token — graceful (None, status) when it's not configured or no docs found.
    # See docs: "Tools" — read-only tool pattern
    if _wants_integration(state, "notion"):
        notion_context, notion_status = _fetch_notion_context(questionnaire, user_context=user_context)
    else:
        notion_context, notion_status = None, _skipped_source("Notion")
    logger.debug("NOTION: result status=%s detail=%s", notion_status.get("status"), notion_status.get("detail"))

    # Load team profile for calibration-aware analysis.
    # Non-fatal — if no profile exists, the prompt runs without calibration context.
    # See docs: "Scrum Standards" — team learning, self-calibrating estimates
    if _wants_dep(state, "analysis"):
        team_profile = _load_team_profile(state.get("analysis_profile_id", ""))
        team_calibration_text = _format_team_calibration(
            team_profile, examples=_load_team_examples(state.get("analysis_profile_id", ""))
        )
    else:
        team_profile, team_calibration_text = None, ""
    team_profile_summary = team_calibration_text.strip()

    # Gather the team's recent Standup + Retro history (graceful — an empty
    # context when nothing has run). Recency-dominant: pre-analysis we have no
    # reliable project name to match on, so we pass "" and let the most recent
    # ceremonies inform the plan. A run with a context scope reads only the
    # sessions it selected. Action items are stashed for story_writer.
    # See docs: "Session Management" — SQLite persistence
    selection = _state_scope(state)
    ceremony = gather_ceremony_context(selection=selection)

    # Gather per-engineer Performance signal (open 1:1 action items + review focus
    # areas) so the analysis is *person-aware* — e.g. flags an engineer's growth
    # area as a staffing consideration. Graceful: empty when Performance mode is
    # unused. See README: "Performance Mode".
    performance = _gather_performance_summary(selection)

    # Build the formatted answers block for the prompt
    answers_block = _build_answers_block(questionnaire)
    prompt = get_analyzer_prompt(
        answers_block,
        team_size,
        velocity,
        repo_context=repo_context,
        detected_stack=repo_signals.detected_stack or None,
        confluence_context=confluence_context,
        notion_context=notion_context,
        user_context=user_context,
        team_profile_summary=team_profile_summary,
        ceremony_history=ceremony.summary_md,
        performance_context=performance,
        review_feedback=review_feedback if review_mode else None,
        review_mode=review_mode,
        previous_output=previous_output,
        prior_art=tuple(state.get("prior_art") or ()),
    )

    # Pasted screenshots (Ctrl+V in the description/questionnaire inputs, plus any
    # attached to review-edit feedback) — file paths converted to multimodal image
    # blocks at invoke time only. See agent/llm.py:invoke_json/invoke_with_images.
    images = list(state.get("pasted_images") or []) + list(state.get("review_feedback_images") or [])
    if images:
        logger.info("project_analyzer: attaching %d pasted image(s)", len(images))

    try:
        # Single LLM call with low temperature for deterministic JSON extraction.
        # See docs: "Agentic Blueprint Reference" — using the LLM outside the main graph
        response = _invoke_json(prompt, image_paths=images)
        analysis = _parse_analysis_response(response.content, questionnaire, team_size, velocity)
        if analysis.architecture is not None:
            logger.info(
                "project_analyzer: architecture options=%d chosen=%s confidence=%s pinned=%s",
                len(analysis.architecture.options),
                analysis.architecture.chosen,
                analysis.architecture.confidence,
                analysis.architecture.pinned_by_constraint,
            )
    except Exception as exc:
        if _should_reraise_llm_error(exc):
            raise
        # LLM call failed entirely — use deterministic fallback.
        logger.warning("LLM call failed in project_analyzer, using fallback", exc_info=True)
        analysis = _build_fallback_analysis(questionnaire, team_size, velocity)

    # Compute prompt quality rating from questionnaire tracking sets (no LLM call).
    # Attach to the analysis via dataclasses.replace since ProjectAnalysis is frozen.
    quality = compute_prompt_quality(questionnaire, has_user_context=user_context is not None)
    analysis = dataclasses.replace(analysis, prompt_quality=quality)

    # ── Low-code reconciliation ──────────────────────────────────────────
    # Three inputs OR together (a strong signal from any one wins): the analyzer
    # LLM's is_low_code, the deterministic verdict stashed by project_intake's
    # broadened scan, and a fresh analyze_context pass over the reused raw scan.
    stashed_low_code = getattr(questionnaire, "_repo_low_code", False)
    stashed_reason = getattr(questionnaire, "_repo_low_code_reason", "")
    det_reason = stashed_reason or (repo_signals.low_code_reasons[0] if repo_signals.low_code_reasons else "")
    final_low_code = analysis.is_low_code or repo_signals.low_code or stashed_low_code
    if final_low_code != analysis.is_low_code:
        analysis = dataclasses.replace(analysis, is_low_code=final_low_code)
    if final_low_code and not analysis.low_code_reason:
        analysis = dataclasses.replace(analysis, low_code_reason=det_reason)
    if final_low_code:
        logger.info("project_analyzer: low-code project detected (reason: %s)", analysis.low_code_reason or det_reason)

    # ── Small-project scope detection & coercion ─────────────────────────
    # See docs: "Guardrails" — human-in-the-loop (advisory)
    #
    # In Small-project mode the plan is always kept flat: a single implicit epic
    # (skip_features) and 1-2 sprints max. We coerce the analysis so the
    # downstream nodes stay lean — but FIRST read the analyzer's honest size
    # estimate. If it judged the project bigger (needs feature grouping, >2
    # sprints, or many goals), we set _small_project_oversized so the analysis
    # review can advise switching to Large (answers are preserved on switch).
    small_mode = _is_small_project_mode(state.get("_intake_mode"))
    oversized = False
    honest_target = analysis.target_sprints
    if small_mode:
        oversized = not analysis.skip_features or analysis.target_sprints > 2 or len(analysis.goals) > 3
        # Targeting one existing sprint (or the backlog) means one Sprint
        # artifact — the stories are being added to a sprint that already
        # exists / left unscheduled, not spread over new ones.
        targeting_existing = state.get("sprint_target_mode") in ("existing", "backlog")
        analysis = dataclasses.replace(
            analysis,
            skip_features=True,
            target_sprints=1 if targeting_existing else min(max(analysis.target_sprints or 1, 1), 2),
        )

    # Format the analysis for display — include capacity data so the user
    # sees velocity breakdown and per-sprint bank holiday impact on this screen.
    display = _format_analysis(
        analysis,
        sprint_capacities=state.get("sprint_capacities"),
        net_velocity=state.get("net_velocity_per_sprint"),
        velocity_per_sprint=state.get("velocity_per_sprint"),
        team_size=state.get("team_size"),
        velocity_source=state.get("velocity_source"),
    )

    # When the Small-project scope looks bigger than 1-2 tickets, append a
    # non-blocking advisory. The analysis review adds a "Switch to Large"
    # action (see _phases_review / _review) — switching preserves the user's
    # answers so they never re-type anything.
    if oversized:
        display += (
            "\n\n---\n\n"
            "> ⚠ **This looks bigger than a small project.**\n"
            f"> Your answers point to multiple feature areas (~{honest_target} sprint(s) of work).\n"
            "> Small-project mode will keep this as a flat 1-2 ticket plan. You can\n"
            "> **continue** as a small project, or **switch to Large** for full\n"
            "> epic / story / sprint planning — your answers are kept, nothing is re-typed."
        )

    # Set pending_review so the REPL intercepts the next user input for
    # the [Accept / Edit / Reject] review flow — same pattern as features/stories.
    return_dict: dict = {
        "project_analysis": analysis,
        "project_name": analysis.project_name,
        "project_description": analysis.project_description,
        "sprint_length_weeks": analysis.sprint_length_weeks,
        "target_sprints": analysis.target_sprints,
        "pending_review": "project_analyzer",
        "_small_project_oversized": oversized,
        # Unresolved retro action items — story_writer seeds the backlog from these
        # (badged [Retro]); sprint_planner reuses the ceremony summary for load caution.
        "_ceremony_action_items": ceremony.action_items,
        "_ceremony_history": ceremony.summary_md,
        # Per-engineer Performance signal reused by sprint_planner (capacity/assignment).
        "_performance_context": performance,
        "messages": [AIMessage(content=display)],
        # Context source diagnostics — tells the REPL which external sources
        # were used, skipped, or failed so the user gets transparency.
        "context_sources": [repo_status, confluence_status, notion_status, user_status],
    }
    # Only write repo_context / confluence_context / notion_context / user_context
    # when present — avoids overwriting a prior value on re-runs (e.g. edit review)
    # where the scan might silently fail (network down, token expired, file deleted).
    if repo_context is not None:
        return_dict["repo_context"] = repo_context
    if confluence_context is not None:
        return_dict["confluence_context"] = confluence_context
    if notion_context is not None:
        return_dict["notion_context"] = notion_context
    if user_context is not None:
        return_dict["user_context"] = user_context
    return return_dict


# ---------------------------------------------------------------------------
# Feature generator node — decomposes ProjectAnalysis into 3-6 Feature dataclasses
# ---------------------------------------------------------------------------
# See docs: "Architecture" — feature_generator sits between analyzer and agent
# See docs: "Scrum Standards" — feature decomposition
#
# Same pattern as project_analyzer: extract from state → build prompt →
# LLM call → parse JSON → fallback → format display → return state update.
# The node returns to END so the REPL can display the features and wait for user
# input. On the next invocation, route_entry sees features populated and routes
# to the agent node.


def _format_epic_list(items: tuple[str, ...]) -> str:
    """Format a tuple of strings as a markdown bullet list for the prompt.

    Args:
        items: Tuple of string items from ProjectAnalysis.

    Returns:
        A bullet list string, or "_(none)_" if empty.
    """
    if not items:
        return "_(none)_"
    return "\n".join(f"- {item}" for item in items)


def _parse_features_response(raw: str, analysis: ProjectAnalysis) -> list[Feature]:
    """Parse the LLM's JSON response into a list of Feature dataclasses.

    # See docs: "Scrum Standards" — feature format
    #
    # Strips markdown code fences, parses JSON array, validates priority
    # against the Priority enum, and falls back to _build_fallback_features
    # on any parse error. Same defensive pattern as _parse_analysis_response.

    Args:
        raw: The raw LLM response string (expected to be a JSON array).
        analysis: The ProjectAnalysis (for fallback).

    Returns:
        A list of Feature instances.
    """
    try:
        # Strip a leaked <think> block, then the markdown fences LLMs wrap JSON
        # in. Reasoning left in place defeats the fence strip and the parse falls
        # through to the deterministic fallback with no error.
        text = strip_think_tags(raw).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return _build_fallback_features(analysis)

        # Validate and convert each feature dict into a Feature dataclass
        # Priority validation: default to MEDIUM if the LLM returns an invalid value
        valid_priorities = {p.value for p in Priority}
        features: list[Feature] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            raw_priority = str(item.get("priority", "medium")).lower().strip()
            priority = Priority(raw_priority) if raw_priority in valid_priorities else Priority.MEDIUM
            features.append(
                Feature(
                    id=str(item.get("id", f"F{len(features) + 1}")),
                    title=str(item.get("title", "Untitled Feature")),
                    description=str(item.get("description", "")),
                    priority=priority,
                )
            )

        if not features:
            return _build_fallback_features(analysis)

        return features

    except Exception:
        logger.debug("Failed to parse features JSON, falling back to deterministic extraction", exc_info=True)
        return _build_fallback_features(analysis)


def _build_fallback_features(analysis: ProjectAnalysis) -> list[Feature]:
    """Build 3 deterministic fallback features from ProjectAnalysis fields.

    # See docs: "Scrum Standards" — feature decomposition
    #
    # When the LLM fails to produce valid JSON, this function creates 3
    # generic features that give downstream nodes something to work with:
    # 1. Core functionality — derived from the first goal
    # 2. Infrastructure & setup — always needed
    # 3. Integrations & extensions — derived from integrations/tech stack

    Args:
        analysis: The ProjectAnalysis to derive features from.

    Returns:
        A list of 3 Feature instances.
    """
    # Derive core feature title from the first goal, or use a generic label
    core_title = "Core Functionality"
    core_desc = "Implement the primary features of the project."
    if analysis.goals:
        core_title = analysis.goals[0][:60]
        core_desc = f"Implement: {analysis.goals[0]}"

    return [
        Feature(
            id="F1",
            title=core_title,
            description=core_desc,
            priority=Priority.HIGH,
        ),
        Feature(
            id="F2",
            title="Infrastructure & Setup",
            description=f"Project scaffolding, CI/CD, and deployment for {analysis.project_name}.",
            priority=Priority.HIGH,
        ),
        Feature(
            id="F3",
            title="Integrations & Extensions",
            description="Third-party integrations, APIs, and extensibility features.",
            priority=Priority.MEDIUM,
        ),
    ]


def _format_features(features: list[Feature], project_name: str) -> str:
    """Format a list of Features as a markdown display for the user.

    Matches the project's analysis summary style — sections with structured
    content. The REPL renders this as a Rich table and waits for user
    review before routing to the next pipeline step.

    Args:
        features: List of Feature dataclasses to display.
        project_name: Project name for the header.

    Returns:
        A formatted markdown string.
    """
    sections = [
        f"# Feature Decomposition: {project_name}\n",
        f"**{len(features)} feature(s) identified**\n",
    ]

    for feature in features:
        sections.append(
            f"## {feature.id}: {feature.title}\n**Priority:** {feature.priority.value}\n{feature.description}\n"
        )

    sections.append("\n---\n**[Accept / Edit / Reject]** — Review the features above.")

    return "\n".join(sections)


def feature_skip(state: ScrumState) -> dict:
    """LangGraph node: create a single feature for small projects.

    # See docs: "Scrum Standards" — feature generation
    #
    # When the analyzer sets skip_features=True, this node runs instead of
    # feature_generator. It creates a single feature (id=F1) named after the project
    # so the user sees 1 feature instead of the usual 3-6. This keeps downstream
    # code (story_writer, task_decomposer, sprint_planner, renderers) working
    # unchanged — they still see a feature_id on every story.
    #
    # Sets pending_review to "feature_generator" so the review checkpoint fires
    # and the user sees the feature before moving to stories. This keeps
    # the flow consistent: analyze → features (review) → stories → tasks → sprints.

    Args:
        state: The current LangGraph state with project_analysis.

    Returns:
        A dict updating features, messages, and pending_review.
    """
    analysis: ProjectAnalysis = state["project_analysis"]
    sentinel = Feature(
        id="F1",
        title=analysis.project_name,
        description=analysis.project_description,
        priority=Priority.HIGH,
    )

    display = (
        f"# {analysis.project_name}\n\n"
        f"Project scope is small — 1 feature covers all planned work.\n\n"
        f"**{analysis.project_name}:** {analysis.project_description}\n"
    )

    return {
        "features": [sentinel],
        "messages": [AIMessage(content=display)],
        "pending_review": "feature_generator",
    }


def feature_generator(state: ScrumState) -> dict:
    """LangGraph node: decompose ProjectAnalysis into 3-6 features.

    # See docs: "Agentic Blueprint Reference" — node return format
    # See docs: "Architecture" — feature_generator node
    # See docs: "Scrum Standards" — feature decomposition
    #
    # How this works:
    # 1. Read the ProjectAnalysis from state.
    # 2. Format analysis fields into strings for the prompt.
    # 3. Call the LLM with the feature generator prompt (temperature=0.0).
    # 4. Parse the JSON array response into Feature dataclasses.
    # 5. Return {"features": [...], "messages": [AIMessage]}.
    #
    # Why this returns to END (not to agent)?
    # Same pattern as project_analyzer — the node produces output, the REPL
    # displays it, and the user types anything to continue. On the next
    # invocation, route_entry sees features populated and routes to the agent.

    Args:
        state: The current LangGraph state with project_analysis.

    Returns:
        A dict updating features and messages.
    """
    analysis: ProjectAnalysis = state["project_analysis"]
    # Read repo context scanned during project_analyzer — grounds feature scope
    # in real directory structure and README goals vs. user descriptions only.
    repo_context: str | None = state.get("repo_context")

    # See docs: "Guardrails" — human-in-the-loop pattern
    # Read review state from previous reject/edit decision. When present, the
    # REPL has cleared the old artifacts and set last_review_decision/feedback
    # so this node regenerates with user feedback injected into the prompt.
    review_decision = state.get("last_review_decision")
    review_feedback = state.get("last_review_feedback", "")
    review_mode = review_decision.value if review_decision else None

    # For edit mode, extract previous output from the feedback string.
    # The REPL packs it as: "{feedback}\n\n---PREVIOUS OUTPUT---\n{serialized}"
    previous_output = None
    if review_mode == "edit" and "---PREVIOUS OUTPUT---" in review_feedback:
        parts = review_feedback.split("---PREVIOUS OUTPUT---", 1)
        review_feedback = parts[0].strip()
        previous_output = parts[1].strip()

    # Format ProjectAnalysis fields into strings for the prompt.
    # This avoids passing the dataclass directly, keeping the prompt module
    # free of imports from agent.state (avoiding circular imports).
    target_sprints_str = str(analysis.target_sprints) if analysis.target_sprints else "scope-based"

    prompt = get_feature_generator_prompt(
        project_name=analysis.project_name,
        project_description=analysis.project_description,
        project_type=analysis.project_type,
        goals=_format_epic_list(analysis.goals),
        end_users=_format_epic_list(analysis.end_users),
        target_state=analysis.target_state or "_(not specified)_",
        tech_stack=_format_epic_list(analysis.tech_stack),
        constraints=_format_epic_list(analysis.constraints),
        risks=_format_epic_list(analysis.risks),
        target_sprints=target_sprints_str,
        out_of_scope=_format_epic_list(analysis.out_of_scope),
        repo_context=repo_context,
        review_feedback=review_feedback if review_mode else None,
        review_mode=review_mode,
        previous_output=previous_output,
        prior_art=tuple(state.get("prior_art") or ()),
    )

    # Screenshots attached to review-edit feedback (Ctrl+V) — sent only when the
    # user is asking for changes; the normal pass works from the analysis alone.
    review_images = list(state.get("review_feedback_images") or []) if review_mode else []

    try:
        # Single LLM call with low temperature for deterministic JSON output.
        # See docs: "Agentic Blueprint Reference" — using the LLM outside the main graph
        response = _invoke_json(prompt, image_paths=review_images)
        features = _parse_features_response(response.content, analysis)
    except Exception as exc:
        if _should_reraise_llm_error(exc):
            raise
        # LLM call failed entirely — use deterministic fallback.
        logger.warning("LLM call failed in feature_generator, using fallback", exc_info=True)
        features = _build_fallback_features(analysis)

    # Format the features for display
    display = _format_features(features, analysis.project_name)

    # Set pending_review so the REPL intercepts the next user input for
    # the [Accept / Edit / Reject] flow instead of invoking the graph.
    return {
        "features": features,
        "messages": [AIMessage(content=display)],
        "pending_review": "feature_generator",
    }


# ---------------------------------------------------------------------------
# Story writer node — decomposes Features into UserStory dataclasses
# ---------------------------------------------------------------------------
# See docs: "Architecture" — story_writer sits between feature_generator and agent
# See docs: "Scrum Standards" — story format, acceptance criteria, story points
#
# Same pattern as feature_generator: extract from state → build prompt →
# LLM call → parse JSON → fallback → format display → return state update.
# The node returns to END so the REPL can display the stories and wait for
# user input. On the next invocation, route_entry sees stories populated
# and routes to the agent node.


def _format_features_for_prompt(features: list[Feature]) -> str:
    """Format a list of features as a readable text block for the story writer prompt.

    Each feature is shown with its ID, title, description, and priority so the LLM
    can reference them when generating stories.

    Args:
        features: List of Feature dataclasses from the feature_generator node.

    Returns:
        A formatted multi-line string describing all features.
    """
    lines: list[str] = []
    for feature in features:
        lines.append(f"**{feature.id}: {feature.title}** (Priority: {feature.priority.value})")
        lines.append(f"  {feature.description}")
        lines.append("")
    return "\n".join(lines)


def _format_team_calibration(profile: object, *, examples: dict | None = None) -> str:
    """Render a TeamProfile into prompt-ready calibration text.

    # See docs: "Scrum Standards" — team learning, self-calibrating estimates
    #
    # Injected into story_writer and sprint_planner prompts so the LLM uses
    # team-specific patterns instead of generic Fibonacci rules. The profile
    # is passed as `object` to avoid importing TeamProfile (circular import risk).
    # The optional ``examples`` dict provides analysis data (spillover
    # correlation, discipline calibration, task patterns, scope changes) that
    # lives outside the TeamProfile dataclass.

    Args:
        profile: A TeamProfile instance (from team_profile.py).
        examples: Optional analysis examples dict from TeamProfileStore.

    Returns:
        A formatted markdown section, or "" if the profile has no useful data.
    """
    if profile is None:
        return ""

    sample_sprints = getattr(profile, "sample_sprints", 0)
    sample_stories = getattr(profile, "sample_stories", 0)
    if sample_sprints == 0:
        return ""

    _ex = examples or {}
    lines = [
        f"\n## Team Calibration Data (from {sample_sprints} sprints, {sample_stories} stories)\n",
    ]

    # Point calibrations — prefer LLM-generated descriptions, fall back to raw metrics
    calibrations = getattr(profile, "point_calibrations", ())
    point_descs = _ex.get("point_descriptions", {}) if isinstance(_ex.get("point_descriptions"), dict) else {}
    if calibrations:
        lines.append("### What story points mean for THIS team:\n")
        for cal in calibrations:
            if cal.sample_count > 0:
                desc = point_descs.get(str(cal.point_value), "")
                if desc:
                    lines.append(f"- **{cal.point_value} pt**: {desc}")
                else:
                    # Fallback to raw metrics
                    patterns = ", ".join(cal.common_patterns) if cal.common_patterns else ""
                    pattern_note = f' — typically "{patterns}"' if patterns else ""
                    overshoot_note = f", overshoots {cal.overshoot_pct:.0f}% of time" if cal.overshoot_pct > 10 else ""
                    if cal.sample_count >= 15:
                        conf = " **(HIGH confidence)**"
                    elif cal.sample_count >= 5:
                        conf = " *(medium confidence)*"
                    else:
                        conf = " *(low confidence — few samples, use cautiously)*"
                    lines.append(
                        f"- **{cal.point_value} pt**: avg {cal.avg_cycle_time_days:.1f} day cycle time, "
                        f"~{cal.typical_task_count:.0f} tasks ({cal.sample_count} samples)"
                        f"{pattern_note}{overshoot_note}{conf}"
                    )
        lines.append("")

    # Example stories per point value — for LLM to reference in points_rationale
    _has_examples = False
    for pts in (1, 2, 3, 5, 8):
        ex_key = f"calibration_{pts}pt"
        ex_items = _ex.get(ex_key, [])
        if ex_items:
            if not _has_examples:
                lines.append("### Similar completed stories (reference these in points_rationale):\n")
                _has_examples = True
            lines.append(f"**{pts} pt examples:**")
            for ex in ex_items[:3]:
                ek = ex.get("issue_key", "")
                sm = ex.get("summary", "")
                detail = ex.get("detail", "")
                lines.append(f"  - {ek}: {sm}" + (f" ({detail})" if detail else ""))
    if _has_examples:
        lines.append("")

    # Story shapes with discipline-specific guidance
    shapes = getattr(profile, "story_shapes", ())
    if shapes:
        lines.append("### Story shape patterns by discipline:\n")
        for shape in shapes:
            if shape.sample_count > 0:
                if shape.sample_count >= 15:
                    conf = " **(reliable)**"
                elif shape.sample_count >= 5:
                    conf = ""
                else:
                    conf = " *(few samples)*"
                lines.append(
                    f"- **{shape.discipline}** stories: avg {shape.avg_points:.1f} pts, "
                    f"{shape.avg_ac_count:.0f} ACs, {shape.avg_task_count:.1f} tasks "
                    f"({shape.sample_count} samples){conf}"
                )
        lines.append("")

    # Velocity with trend context
    vel_avg = getattr(profile, "velocity_avg", 0)
    vel_std = getattr(profile, "velocity_stddev", 0)
    if vel_avg > 0:
        var_pct = vel_std / vel_avg * 100 if vel_avg else 0
        stability = "stable" if var_pct < 20 else ("moderate variance" if var_pct < 35 else "HIGH variance")
        lines.append(f"### Velocity: {vel_avg:.0f} ± {vel_std:.0f} pts/sprint ({stability})\n")

    # Sprint completion
    completion = getattr(profile, "sprint_completion_rate", 0)
    if completion > 0:
        lines.append(f"### Sprint completion rate: {completion:.0f}%")
        if completion < 70:
            lines.append("**WARNING:** Low completion rate — plan conservatively, target 80% of historical velocity.\n")
        else:
            lines.append("")

    # Spillover with root-cause hints
    spillover = getattr(profile, "spillover", None)
    if spillover and getattr(spillover, "carried_over_pct", 0) > 0:
        pct = spillover.carried_over_pct
        lines.append(
            f"### Spillover: {pct:.0f}% of stories "
            f"carried to next sprint (avg {spillover.avg_spillover_pts:.1f} pts/sprint)"
        )
        if spillover.most_common_spillover_reason:
            lines.append(f"- Most common cause: {spillover.most_common_spillover_reason}")
        # Actionable guidance based on spillover severity
        if pct > 20:
            lines.append(
                "- **High spillover — split stories aggressively.** "
                "Prefer 2-3pt stories over 5-8pt. Add buffer for unknowns.\n"
            )
        elif pct > 10:
            lines.append(
                "- **Moderate spillover — be cautious with large stories.** "
                "Consider splitting 8pt stories into smaller pieces.\n"
            )
        else:
            lines.append("")

    # Estimation accuracy
    accuracy = getattr(profile, "estimation_accuracy_pct", 0)
    if accuracy > 0:
        lines.append(f"### Estimation accuracy: {accuracy:.0f}% of stories completed at original estimate")
        if accuracy < 60:
            lines.append("- Estimates are frequently wrong — add explicit unknowns/risks to ACs.\n")
        else:
            lines.append("")

    # Epic patterns
    epic = getattr(profile, "epic_pattern", None)
    if epic and getattr(epic, "sample_count", 0) > 0:
        lines.append(
            f"### Epic sizing: avg {epic.avg_stories_per_epic:.0f} stories/epic, "
            f"{epic.avg_points_per_epic:.0f} pts/epic "
            f"(range {epic.typical_story_count_range[0]}–{epic.typical_story_count_range[1]} stories)\n"
        )

    # Definition of Done signals
    dod = getattr(profile, "dod_signal", None)
    if dod:
        pr_pct = getattr(dod, "stories_with_pr_link_pct", 0)
        review_pct = getattr(dod, "stories_with_review_mention_pct", 0)
        test_pct = getattr(dod, "stories_with_testing_mention_pct", 0)
        deploy_pct = getattr(dod, "stories_with_deploy_mention_pct", 0)
        if pr_pct > 0 or review_pct > 0:
            lines.append("### Definition of Done (observed behaviour):\n")
            if pr_pct > 0:
                lines.append(f"- PR linked before close: {pr_pct:.0f}% of stories")
            if review_pct > 0:
                lines.append(f"- Code review mentioned: {review_pct:.0f}% of stories")
            if test_pct > 0:
                lines.append(f"- Testing mentioned: {test_pct:.0f}% of stories")
            if deploy_pct > 0:
                lines.append(f"- Deploy/release mentioned: {deploy_pct:.0f}% of stories")
            checklist = getattr(dod, "common_checklist_items", ())
            if checklist:
                items = ", ".join(f'"{c}"' for c in checklist[:4])
                lines.append(f"- Common checklist items: {items}")
            lines.append(
                "\nWhen generating tasks, include steps that match this team's "
                "DoD pattern (e.g. code review, testing, PR).\n"
            )

    # Writing patterns
    wp = getattr(profile, "writing_patterns", None)
    if wp:
        gwt = getattr(wp, "uses_given_when_then", False)
        median_ac = getattr(wp, "median_ac_count", 0)
        median_tasks = getattr(wp, "median_task_count_per_story", 0)
        personas = getattr(wp, "common_personas", ())
        if gwt or median_ac > 0 or personas:
            lines.append("### Writing patterns (match this team's style):\n")
            if gwt:
                lines.append("- Acceptance criteria use Given/When/Then format")
            if median_ac > 0:
                lines.append(f"- Median acceptance criteria per story: {median_ac:.0f}")
            if median_tasks > 0:
                lines.append(f"- Median tasks per story: {median_tasks:.0f}")
            if personas:
                lines.append(f"- Common personas: {', '.join(personas[:5])}")
            lines.append("")

    # ── Sections from examples dict (analysis data not in TeamProfile) ──

    # 1A: Spillover correlation — which sizes/disciplines spill most
    spill_corr = _ex.get("spillover_correlation", {})
    if isinstance(spill_corr, dict):
        by_size = spill_corr.get("by_size", {})
        by_disc = spill_corr.get("by_discipline", {})
        if by_size or by_disc:
            lines.append("### Spillover risk factors:\n")
            for sz in sorted(by_size.keys(), key=lambda x: int(x)):
                pct = by_size[sz]
                if pct > 0:
                    risk = " ⚠ HIGH RISK" if pct >= 30 else ""
                    lines.append(f"- {sz}pt stories spill {pct:.0f}% of the time{risk}")
            for disc, pct in sorted(by_disc.items(), key=lambda x: -x[1]):
                if pct > 0:
                    lines.append(f"- {disc} stories spill {pct:.0f}%")
            # Actionable guidance
            high_spill_sizes = [sz for sz, p in by_size.items() if p >= 25]
            if high_spill_sizes:
                sizes = ", ".join(f"{s}pt" for s in sorted(high_spill_sizes, key=int))
                lines.append(f"\n→ {sizes} stories are spill-prone. Prefer smaller slices.\n")
            else:
                lines.append("")

    # 1B: Velocity trend — improving/degrading/stable
    vel_trend = _ex.get("velocity_trend", {})
    if isinstance(vel_trend, dict) and vel_trend.get("trend"):
        trend = vel_trend["trend"]
        slope = vel_trend.get("slope", 0)
        first_v = vel_trend.get("first_velocity", 0)
        last_v = vel_trend.get("last_velocity", 0)
        if trend not in ("insufficient_data", ""):
            lines.append(f"### Velocity trend: {trend.upper()} ({first_v}→{last_v}, {slope:+.1f}/sprint)\n")
            if trend == "degrading":
                lines.append(
                    f"→ Velocity is declining. Plan using recent velocity ({last_v} pts), "
                    "not the average. Investigate root causes.\n"
                )
            elif trend == "improving":
                lines.append(
                    f"→ Velocity is improving. Recent sprints deliver {last_v} pts. "
                    "Can plan slightly above the historical average.\n"
                )

    # 1C: Discipline-specific calibration
    disc_cal = _ex.get("discipline_calibration", {})
    if isinstance(disc_cal, dict) and len(disc_cal) > 1:
        lines.append("### Discipline calibration:\n")
        for disc, entries in sorted(disc_cal.items()):
            if not isinstance(entries, list):
                continue
            for e in entries:
                if not isinstance(e, dict):
                    continue
                pts = e.get("points", 0)
                cycle = e.get("avg_cycle_days", 0)
                spill = e.get("spill_pct", 0)
                samples = e.get("samples", 0)
                conf = "reliable" if samples >= 10 else ("moderate" if samples >= 5 else "low data")
                risk = " — ⚠ high spill" if spill >= 25 else ""
                lines.append(
                    f"- **{disc}** {pts}pt: {cycle:.1f}d cycle, {spill:.0f}% spill ({samples} samples) — {conf}{risk}"
                )
        lines.append("")

    # 1D: Task decomposition patterns
    task_decomp = _ex.get("task_decomposition", {})
    if isinstance(task_decomp, dict) and task_decomp.get("total_stories", 0) > 0:
        avg_tasks = task_decomp.get("avg_tasks_per_story", 0)
        sw_tasks = task_decomp.get("stories_with_tasks", 0)
        tot_s = task_decomp.get("total_stories", 0)
        task_pct = round(sw_tasks / tot_s * 100) if tot_s else 0
        completion = task_decomp.get("task_completion_rate", 100)
        type_dist = task_decomp.get("type_distribution", {})
        bottlenecks = task_decomp.get("bottlenecks", [])
        common = task_decomp.get("common_tasks", [])

        lines.append("### Task decomposition patterns (historical):\n")
        if avg_tasks > 0:
            lines.append(f"- Avg tasks per story: {avg_tasks:.1f} ({task_pct}% of stories have subtasks)")
        if type_dist:
            dist_parts = ", ".join(f"{t} {p:.0f}%" for t, p in type_dist.items() if p > 0)
            if dist_parts:
                lines.append(f"- Type distribution: {dist_parts}")
        if completion < 70:
            lines.append(f"- ⚠ Task completion rate: {completion:.0f}% — make tasks smaller and more actionable")
        for cat, rate, count in bottlenecks[:2]:
            lines.append(f"- ⚠ {cat} bottleneck: only {rate}% completion ({count} tasks)")
        if common:
            task_names = ", ".join(f'"{t}"' for t in common[:5])
            lines.append(f"- Common task patterns: {task_names}")
        lines.append("\n→ When generating subtasks, match this team's task style and naming conventions.\n")

    # 1E: Committed vs delivered + scope churn
    scope_changes = _ex.get("scope_changes", {})
    if isinstance(scope_changes, dict) and scope_changes.get("totals"):
        sc_totals = scope_changes["totals"]
        committed = sc_totals.get("avg_committed_velocity", 0.0)
        delivered = sc_totals.get("avg_delivered_velocity", 0.0)
        if committed > 0:
            accuracy = round(delivered / committed * 100)
            lines.append("### Committed vs Delivered:\n")
            lines.append(f"- Avg committed: {committed:g} pts/sprint")
            lines.append(f"- Avg delivered: {delivered:g} pts/sprint ({accuracy}% delivery rate)")
            # Compute avg churn from per-sprint data
            per_sprint = scope_changes.get("per_sprint", [])
            churns = [s.get("scope_churn", 0) for s in per_sprint if s.get("scope_churn") is not None]
            if churns:
                avg_churn = sum(churns) / len(churns)
                lines.append(f"- Avg scope churn: {avg_churn:.0%}")
            if accuracy < 80:
                lines.append(
                    f"\n→ Plan sprints at {delivered:g} pts (delivered velocity), "
                    f"not {committed:g}. Leave buffer for mid-sprint scope changes.\n"
                )
            else:
                lines.append("")

    # Ticket naming conventions
    naming = _ex.get("naming_conventions", {})
    if isinstance(naming, dict):
        naming_parts: list[str] = []
        prefixes = naming.get("title_prefixes", [])
        if prefixes:
            p_str = ", ".join(f"{p} ({pct}%)" for p, pct in prefixes[:5])
            naming_parts.append(f"- Title prefixes in use: {p_str} — use matching prefixes on generated stories")
        labels = naming.get("label_distribution", [])
        if labels:
            l_str = ", ".join(f"{lbl} ({pct}%)" for lbl, pct in labels[:5])
            naming_parts.append(f"- Label convention: {l_str}")
        epic_style = naming.get("epic_naming_style", "")
        epic_ex = naming.get("epic_examples", [])
        if epic_style and epic_ex:
            ex_str = ", ".join(f'"{e}"' for e in epic_ex[:3])
            naming_parts.append(f"- Epic naming: {epic_style} (e.g. {ex_str})")
        sections = naming.get("template_sections", [])
        if sections:
            s_str = ", ".join(f'"{s}"' for s, _ in sections[:5])
            naming_parts.append(f"- Description template: use sections {s_str}")
        if naming_parts:
            lines.append("### Ticket naming conventions (match this team's style):\n")
            lines.extend(naming_parts)
            lines.append("\n→ Generated tickets MUST match these naming conventions.\n")

    # Board workflow style
    wf = _ex.get("workflow_style", {})
    if isinstance(wf, dict) and wf.get("style") == "columns-as-dod":
        workflow = wf.get("workflow", [])
        dod_cols = wf.get("dod_columns", {})
        if workflow and dod_cols:
            col_str = " → ".join(workflow)
            dod_str = ", ".join(f"{c} ({r}%)" for c, r in dod_cols.items())
            lines.append("### Team workflow style: columns-as-DoD\n")
            lines.append(f"- Board columns: {col_str}")
            lines.append(f"- DoD columns: {dod_str}")
            lines.append(
                "\n→ This team uses board columns as workflow stages. "
                "DO NOT create separate subtasks for steps that are tracked as column transitions "
                "(e.g. Documentation, PR review). Instead, generate implementation tasks only.\n"
            )

    # Estimation bias and seasonal patterns
    addl = _ex.get("additional_patterns", {})
    if isinstance(addl, dict):
        est = addl.get("estimation_bias", {})
        if isinstance(est, dict) and est.get("underestimated_pct", 0) >= 20:
            worst = est.get("worst_sizes", [])
            w_str = ", ".join(f"{p}pt" for p in worst) if worst else "larger stories"
            lines.append(
                f"### Estimation warning: {est['underestimated_pct']}% of stories take >2x expected.\n"
                f"- Most affected: {w_str}\n"
                "- Add estimation buffer for these sizes or break into smaller pieces.\n"
            )
        seas = addl.get("seasonal", {})
        if isinstance(seas, dict) and seas.get("low_months"):
            low = seas["low_months"]
            low_str = ", ".join(f"{m} ({v:g} pts)" for m, v in low.items())
            lines.append(
                f"### Seasonal velocity dips: {low_str}\n"
                f"- Average velocity is {seas.get('overall_avg', 0):g} pts/sprint.\n"
                "- Plan lighter sprints in low-velocity months.\n"
            )

    lines.append(
        "### Estimation note: Use THESE team-specific patterns, not generic Fibonacci rules. "
        "Weight HIGH confidence calibrations heavily; treat low confidence data as rough guidance only.\n"
    )

    return "\n".join(lines)


def _snap_to_fibonacci(value: int) -> StoryPointValue:
    """Clamp an integer to [1, 8] and snap to the nearest Fibonacci story point.

    # See docs: "Scrum Standards" — story points on Fibonacci scale
    #
    # The LLM may return non-Fibonacci values (e.g. 4, 6, 10). This helper
    # ensures all story points are valid StoryPointValue members:
    # 1. Clamp to [1, 8] range
    # 2. Find the closest Fibonacci value

    Args:
        value: The raw story point value from the LLM.

    Returns:
        The nearest valid StoryPointValue (1, 2, 3, 5, or 8).
    """
    clamped = max(1, min(8, value))
    # Find the Fibonacci value with the smallest distance
    fibonacci_values = list(StoryPointValue)
    closest = min(fibonacci_values, key=lambda fib: abs(fib.value - clamped))
    return closest


# ── Discipline inference ──────────────────────────────────────────────
# See docs: "Scrum Standards" — discipline tagging
#
# Keyword-based discipline inference. The LLM prompt asks for a discipline
# field in the JSON, but if the LLM omits it or gives a bad value, this
# function guesses based on keywords in the story text. Best-effort —
# the TODO says "where possible". If both frontend and backend keywords
# match, defaults to FULLSTACK.

_FRONTEND_KEYWORDS = frozenset(
    {
        "ui",
        "component",
        "css",
        "frontend",
        "page",
        "form",
        "button",
        "layout",
        "responsive",
        "style",
    }
)
_BACKEND_KEYWORDS = frozenset(
    {
        "api",
        "endpoint",
        "database",
        "server",
        "backend",
        "query",
        "migration",
        "schema",
    }
)
_INFRASTRUCTURE_KEYWORDS = frozenset(
    {
        "ci",
        "cd",
        "deploy",
        "pipeline",
        "docker",
        "infrastructure",
        "monitoring",
        "logging",
    }
)
_DESIGN_KEYWORDS = frozenset({"design", "wireframe", "mockup", "prototype", "ux"})
_TESTING_KEYWORDS = frozenset({"test", "qa", "automation", "coverage"})


def _infer_discipline(story: UserStory) -> Discipline:
    """Infer a discipline tag from story text using keyword matching.

    Combines the story's persona, goal, benefit, and acceptance criteria text
    into a single searchable string and checks for discipline keywords.
    If both frontend and backend keywords match, returns FULLSTACK.

    Args:
        story: A UserStory to classify.

    Returns:
        The inferred Discipline value.
    """
    # Combine all textual fields into a single lowercase string for matching
    parts = [story.persona, story.goal, story.benefit]
    for ac in story.acceptance_criteria:
        parts.extend([ac.given, ac.when, ac.then, ac.text])
    text = " ".join(parts).lower()

    words = set(text.split())

    has_frontend = bool(words & _FRONTEND_KEYWORDS)
    has_backend = bool(words & _BACKEND_KEYWORDS)

    # If both frontend and backend match, it's fullstack
    if has_frontend and has_backend:
        return Discipline.FULLSTACK
    if has_frontend:
        return Discipline.FRONTEND
    if has_backend:
        return Discipline.BACKEND
    if words & _INFRASTRUCTURE_KEYWORDS:
        return Discipline.INFRASTRUCTURE
    if words & _DESIGN_KEYWORDS:
        return Discipline.DESIGN
    if words & _TESTING_KEYWORDS:
        return Discipline.TESTING
    return Discipline.FULLSTACK


# ── Story validation ──────────────────────────────────────────────────
# See docs: "Scrum Standards" — Story Checklist
#
# Deterministic post-processing after LLM parsing. Checks acceptance criteria
# count (>= 3), non-empty required fields, and per-feature story counts. Auto-fixes
# what it can and collects warnings for the user. No LLM call needed — all rules
# are deterministic.


def _generic_acs(ac_style: str = "gwt") -> tuple[AcceptanceCriterion, ...]:
    """Generic coverage-padding criteria, in the plan's resolved AC style.

    Padding must match how the team's real criteria are written — a bullets
    plan with Gherkin padding would read as two formats in one ticket.
    """
    if ac_style == "bullets":
        return (
            AcceptanceCriterion(text="Happy path: the main workflow completes with the expected outcome."),
            AcceptanceCriterion(text="Error path: invalid input produces a clear, appropriate error message."),
            AcceptanceCriterion(text="Edge cases: unusual conditions are handled gracefully."),
        )
    return (
        AcceptanceCriterion(
            given="the feature is available",
            when="the user performs the happy-path workflow",
            then="the expected outcome is achieved",
        ),
        AcceptanceCriterion(
            given="invalid input is provided",
            when="the user attempts the action",
            then="an appropriate error message is shown",
        ),
        AcceptanceCriterion(
            given="an edge case scenario occurs",
            when="the user encounters unusual conditions",
            then="the system handles it gracefully",
        ),
    )


def _validate_stories(
    stories: list[UserStory],
    features: list[Feature],
    ac_style: str = "gwt",
    min_acs: int = 3,
) -> tuple[list[UserStory], list[str]]:
    """Validate stories against the Story Checklist and auto-fix where possible.

    # See docs: "Scrum Standards" — Story Checklist
    #
    # Checks:
    # 1. AC count >= min_acs — pads with style-matched generic ACs if fewer.
    #    min_acs honours a learned team median below 3 (a 1-AC team must not
    #    be force-padded to 3), and never exceeds 3.
    # 2. Non-empty persona, goal, benefit — sets defaults if empty.
    # 3. Per-feature story count in [MIN_STORIES_PER_FEATURE, MAX_STORIES_PER_FEATURE] — warns if out of range.
    #
    # Returns new UserStory instances (frozen, so must rebuild) with fixes applied.

    Args:
        stories: The parsed stories to validate.
        features: The feature list for per-feature count validation.
        ac_style: The resolved AC style — padding matches it.
        min_acs: Minimum AC count per story (clamped to 1..3).

    Returns:
        A tuple of (validated_stories, warnings).
    """
    min_acs = max(1, min(3, min_acs))
    warnings: list[str] = []
    validated: list[UserStory] = []

    for story in stories:
        needs_rebuild = False
        new_persona = story.persona
        new_goal = story.goal
        new_benefit = story.benefit
        new_title = story.title
        new_acs = list(story.acceptance_criteria)

        # Check title — generate from goal if missing
        if not story.title.strip():
            # Capitalise the goal and truncate to ~7 words for a concise heading
            goal_words = story.goal.strip().split()
            new_title = " ".join(goal_words[:7]).rstrip(",;:.")
            # Capitalise first letter of each word for a title-case heading
            new_title = new_title.title()
            needs_rebuild = True

        # Check non-empty fields
        if not story.persona.strip():
            new_persona = "user"
            needs_rebuild = True
            warnings.append(f"Story {story.id} had empty persona — defaulted to 'user'.")
        if not story.goal.strip():
            new_goal = "perform the required action"
            needs_rebuild = True
            warnings.append(f"Story {story.id} had empty goal — set default.")
        if not story.benefit.strip():
            new_benefit = "the expected value is delivered"
            needs_rebuild = True
            warnings.append(f"Story {story.id} had empty benefit — set default.")

        # Check AC count >= min_acs
        if len(new_acs) < min_acs:
            added = 0
            for generic_ac in _generic_acs(ac_style):
                if len(new_acs) >= min_acs:
                    break
                # Only add generic ACs that aren't already present
                if generic_ac not in new_acs:
                    new_acs.append(generic_ac)
                    added += 1
            if added > 0:
                needs_rebuild = True
                warnings.append(
                    f"Story {story.id} had only {len(story.acceptance_criteria)} AC(s) "
                    f"— added {added} generic AC(s) to reach {min_acs}."
                )

        if needs_rebuild:
            validated.append(
                dataclasses.replace(
                    story,
                    persona=new_persona,
                    goal=new_goal,
                    benefit=new_benefit,
                    acceptance_criteria=tuple(new_acs),
                    title=new_title,
                )
            )
        else:
            validated.append(story)

    # Per-feature story count warnings.

    feature_counts: dict[str, int] = {}
    for story in validated:
        feature_counts[story.feature_id] = feature_counts.get(story.feature_id, 0) + 1

    for feature in features:
        count = feature_counts.get(feature.id, 0)
        if count < MIN_STORIES_PER_FEATURE:
            warnings.append(
                f"Feature {feature.id} ({feature.title}) has only {count} story(ies)"
                f" — minimum recommended is {MIN_STORIES_PER_FEATURE}."
            )
        elif count > MAX_STORIES_PER_FEATURE:
            warnings.append(
                f"Feature {feature.id} ({feature.title}) has {count} stories "
                f"— maximum recommended is {MAX_STORIES_PER_FEATURE}."
            )

    return validated, warnings


def _parse_stories_response(
    raw: str, features: list[Feature], analysis: ProjectAnalysis, ac_style: str = "gwt", dod_count: int = 0
) -> list[UserStory]:
    """Parse the LLM's JSON response into a list of UserStory dataclasses.

    # See docs: "Scrum Standards" — story format, acceptance criteria
    #
    # Strips markdown code fences, parses JSON array, validates priorities
    # and story points, parses nested acceptance criteria, generates IDs
    # when missing, and falls back to _build_fallback_stories on error.
    # Same defensive pattern as _parse_features_response.

    Args:
        raw: The raw LLM response string (expected to be a JSON array).
        features: The list of features (for fallback and ID validation).
        analysis: The ProjectAnalysis (for fallback).

    Returns:
        A list of UserStory instances.
    """
    try:
        # Strip a leaked <think> block, then the markdown fences LLMs wrap JSON
        # in. Reasoning left in place defeats the fence strip and the parse falls
        # through to the deterministic fallback with no error.
        text = strip_think_tags(raw).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return _build_fallback_stories(features, analysis, ac_style)

        # Build a set of valid feature IDs for validation
        valid_feature_ids = {e.id for e in features}

        # Track per-feature story counts for auto-ID generation
        feature_story_counts: dict[str, int] = {}

        valid_priorities = {p.value for p in Priority}
        stories: list[UserStory] = []

        for item in parsed:
            if not isinstance(item, dict):
                continue

            # Validate feature_id — skip stories with unknown feature IDs
            feature_id = str(item.get("feature_id", ""))
            if feature_id not in valid_feature_ids:
                continue

            # Auto-generate ID if missing
            feature_story_counts.setdefault(feature_id, 0)
            feature_story_counts[feature_id] += 1
            story_id = str(item.get("id", ""))
            if not story_id:
                story_id = f"US-{feature_id}-{feature_story_counts[feature_id]:03d}"

            # Validate and snap story points to Fibonacci
            raw_points = item.get("story_points", 3)
            try:
                points = _snap_to_fibonacci(int(raw_points))
            except (ValueError, TypeError):
                points = StoryPointValue.THREE

            # Validate priority — default to MEDIUM if invalid
            raw_priority = str(item.get("priority", "medium")).lower().strip()
            priority = Priority(raw_priority) if raw_priority in valid_priorities else Priority.MEDIUM

            # Parse discipline — fall back to inference if missing or invalid.
            # See docs: "Scrum Standards" — discipline tagging
            valid_disciplines = {d.value for d in Discipline}
            raw_discipline = str(item.get("discipline", "")).lower().strip()
            discipline = Discipline(raw_discipline) if raw_discipline in valid_disciplines else None

            # Parse nested acceptance criteria — accept every shape regardless
            # of the requested style (the LLM may disobey): a GWT triple dict,
            # a free-text dict ({"text": …}), or a bare string.
            raw_acs = item.get("acceptance_criteria", [])
            acs: list[AcceptanceCriterion] = []
            if isinstance(raw_acs, list):
                for ac_item in raw_acs:
                    if isinstance(ac_item, dict):
                        given = str(ac_item.get("given", "")).strip()
                        when = str(ac_item.get("when", "")).strip()
                        then = str(ac_item.get("then", "")).strip()
                        text_ac = str(ac_item.get("text", "")).strip()
                        if given and when and then:
                            acs.append(AcceptanceCriterion(given=given, when=when, then=then))
                        elif text_ac:
                            acs.append(AcceptanceCriterion(text=text_ac))
                    elif isinstance(ac_item, str) and ac_item.strip():
                        acs.append(AcceptanceCriterion(text=ac_item.strip()))

            # Fallback: if no valid ACs parsed, add a generic happy-path AC
            # in the plan's resolved style.
            if not acs:
                persona = str(item.get("persona", "user"))
                goal = str(item.get("goal", "perform the action"))
                if ac_style == "bullets":
                    acs.append(AcceptanceCriterion(text=f"The {persona} can {goal} successfully."))
                else:
                    acs.append(
                        AcceptanceCriterion(
                            given=f"the {persona} is authenticated",
                            when=f"they {goal}",
                            then="the operation completes successfully",
                        )
                    )

            # Parse Definition of Done applicability flags. The LLM is asked
            # for a boolean array matching the plan's RESOLVED DoD list (a
            # custom list may not be 7 items), so flags are normalised to that
            # length — truncated if long, padded with True (fully applicable)
            # if short or missing. The renderers zip flags against the same
            # resolved list, so length agreement here is what keeps the DoD
            # section on every synced ticket.
            n_dod = dod_count or len(DOD_ITEMS)
            raw_dod = item.get("dod_applicable")
            if isinstance(raw_dod, list) and raw_dod:
                flags = [bool(f) for f in raw_dod[:n_dod]]
                flags += [True] * (n_dod - len(flags))
                dod_applicable: tuple[bool, ...] = tuple(flags)
            else:
                dod_applicable = (True,) * n_dod

            points_rationale = str(item.get("points_rationale", ""))
            points_confidence = str(item.get("points_confidence", ""))
            if points_confidence not in ("high", "medium", "low"):
                points_confidence = ""

            story = UserStory(
                id=story_id,
                feature_id=feature_id,
                persona=str(item.get("persona", "user")),
                goal=str(item.get("goal", "")),
                benefit=str(item.get("benefit", "")),
                acceptance_criteria=tuple(acs),
                story_points=points,
                priority=priority,
                title=str(item.get("title", "")),
                discipline=discipline if discipline is not None else Discipline.FULLSTACK,
                dod_applicable=dod_applicable,
                points_rationale=points_rationale,
                points_confidence=points_confidence,
            )
            # Infer discipline from text when the LLM didn't provide a valid one
            if discipline is None:
                story = dataclasses.replace(story, discipline=_infer_discipline(story))
            stories.append(story)

        if not stories:
            return _build_fallback_stories(features, analysis, ac_style)

        return stories

    except Exception:
        logger.debug("Failed to parse stories JSON, falling back to deterministic extraction", exc_info=True)
        return _build_fallback_stories(features, analysis, ac_style)


def _build_fallback_stories(
    features: list[Feature], analysis: ProjectAnalysis, ac_style: str = "gwt"
) -> list[UserStory]:
    """Build deterministic fallback stories per feature.

    # See docs: "Scrum Standards" — story format
    #
    # When the LLM fails to produce valid JSON, this function creates generic
    # stories that give downstream nodes something to work with.
    # Generates 2 stories per feature (core functionality + setup/testing).

    Args:
        features: The list of features to generate stories for.
        analysis: The ProjectAnalysis for context.

    Returns:
        A list of UserStory instances.
    """
    stories: list[UserStory] = []
    end_user = analysis.end_users[0] if analysis.end_users else "user"

    for feature in features:
        # Story 1: core functionality
        stories.append(
            UserStory(
                id=f"US-{feature.id}-001",
                feature_id=feature.id,
                persona=end_user,
                goal=f"use the core features of {feature.title}",
                benefit=f"I can accomplish the primary objectives of {feature.title}",
                acceptance_criteria=(
                    AcceptanceCriterion(text=f"The {end_user} can complete the main {feature.title} workflow.")
                    if ac_style == "bullets"
                    else AcceptanceCriterion(
                        given=f"the {end_user} has access to {feature.title}",
                        when="they perform the main workflow",
                        then="the expected outcome is achieved",
                    ),
                ),
                story_points=StoryPointValue.FIVE,
                priority=feature.priority,
                title=f"Core {feature.title}",
                discipline=Discipline.FULLSTACK,
            )
        )

        # Story 2: setup and testing
        stories.append(
            UserStory(
                id=f"US-{feature.id}-002",
                feature_id=feature.id,
                persona="developer",
                goal=f"set up and validate {feature.title}",
                benefit="the feature is reliable and properly tested",
                acceptance_criteria=(
                    AcceptanceCriterion(text=f"All tests pass and {feature.title} works as deployed.")
                    if ac_style == "bullets"
                    else AcceptanceCriterion(
                        given="the development environment is configured",
                        when=f"the {feature.title} feature is deployed",
                        then="all tests pass and the feature works as expected",
                    ),
                ),
                story_points=StoryPointValue.THREE,
                priority=feature.priority,
                title=f"Validate {feature.title}",
                discipline=Discipline.TESTING,
            )
        )

    return stories


def _format_stories(
    stories: list[UserStory],
    features: list[Feature],
    project_name: str,
    warnings: list[str] | None = None,
) -> str:
    """Format a list of UserStories as a markdown display for the user.

    Groups stories by their parent feature for readability. Shows the full
    story text, acceptance criteria, story points, priority, and discipline.
    Includes optional validation warnings. The REPL renders this as Rich
    tables and waits for user review.

    Args:
        stories: List of UserStory dataclasses to display.
        features: List of Feature dataclasses (for grouping headers).
        project_name: Project name for the header.
        warnings: Optional list of validation warning strings to display.

    Returns:
        A formatted markdown string.
    """
    sections = [
        f"# User Stories: {project_name}\n",
        f"**{len(stories)} user story(ies) across {len(features)} feature(s)**\n",
    ]

    # Build a feature lookup for grouping
    feature_map = {e.id: e for e in features}

    # Group stories by feature_id, preserving order
    stories_by_feature: dict[str, list[UserStory]] = {}
    for story in stories:
        stories_by_feature.setdefault(story.feature_id, []).append(story)

    for feature_id, feature_stories in stories_by_feature.items():
        feature = feature_map.get(feature_id)
        feature_title = feature.title if feature else feature_id
        sections.append(f"## {feature_id}: {feature_title}\n")

        for story in feature_stories:
            sections.append(f"### {story.id}")
            sections.append(f"**As a** {story.persona}, **I want to** {story.goal}, **so that** {story.benefit}.")
            sections.append(
                f"**Priority:** {story.priority.value} | **Points:** {story.story_points.value} "
                f"| **Discipline:** {story.discipline.value}\n"
            )

            sections.append("**Acceptance Criteria:**")
            for i, ac in enumerate(story.acceptance_criteria, 1):
                if ac.text:
                    sections.append(f"  {i}. {ac.text}")
                else:
                    sections.append(f"  {i}. **Given** {ac.given}")
                    sections.append(f"     **When** {ac.when}")
                    sections.append(f"     **Then** {ac.then}")
            sections.append("")

    # Show validation warnings if any were collected
    if warnings:
        sections.append("## Validation Notes\n")
        for warning in warnings:
            sections.append(f"- {warning}")
        sections.append("")

    sections.append("\n---\n**[Accept / Edit / Reject]** — Review the stories above.")

    return "\n".join(sections)


# ── Architecture-validation spike ─────────────────────────────────────
# See docs: "Scrum Standards" — DoD Spike
#
# When the analyzer produced a genuinely OPEN architecture decision (2+
# options, not pinned by a constraint), the user is asked whether to add a
# time-boxed validation spike to the plan. The ask follows the capacity-popup
# pattern: the node returns early with a sentinel (_spike_prompt) + message,
# the driver renders a choice, and the answer lands in state["spike_choice"].
# Injection itself is deterministic Python (never an LLM instruction), so every
# entry path — TUI, headless, MCP, roadmap — gets an identical artifact.

_SPIKE_TITLE_PREFIX = "[Spike] "


def _spike_eligible(analysis: ProjectAnalysis | None) -> bool:
    """True when the architecture decision is open enough to be worth a spike."""
    arch = getattr(analysis, "architecture", None)
    return arch is not None and len(arch.options) >= 2 and not arch.pinned_by_constraint


def _spike_choice_override() -> str:
    """Pre-made spike choice from YEABOI_ARCHITECTURE_SPIKE (CLI --architecture-spike).

    "include"/"skip" suppress the interactive question entirely; anything else
    (including "auto" and unset) means ask/auto-rule as usual.
    """
    val = (os.getenv("YEABOI_ARCHITECTURE_SPIKE") or "").strip().lower()
    return val if val in ("include", "skip") else ""


def spike_recommended(confidence: str) -> bool:
    """The confidence auto-rule: validate unless the AI is confident.

    Used both as the recommended default in the interactive popup and as the
    unattended answer in headless/MCP runs.
    """
    return confidence != "high"


def _parse_spike_reply(state: ScrumState) -> str:
    """Resolve a pending spike question from the user's textual reply.

    The TUI popups write state["spike_choice"] directly and never reach this;
    the REPL (and any plain-text driver) answers in words. An ambiguous reply
    (including the REPL export-only "continue") falls back to the recommended
    default parked in the sentinel — which is the confidence auto-rule.
    """
    messages = state.get("messages") or []
    text = str(getattr(messages[-1], "content", "")).strip().lower() if messages else ""
    if "skip" in text or text in ("2", "no", "n"):
        return "skip"
    # A negated include ("no spike", "don't add one") must not fall through to
    # the substring include-match below and inject the story the user refused.
    negated = re.search(r"\b(no|not|don'?t|dont|never|without)\b", text)
    if negated and ("add" in text or "include" in text or "spike" in text):
        return "skip"
    if "add" in text or "include" in text or "spike" in text or text in ("1", "yes", "y"):
        return "include"
    return (state.get("_spike_prompt") or {}).get("recommended", "include")


def _maybe_prompt_spike_choice(state: ScrumState, analysis: ProjectAnalysis | None) -> dict | None:
    """Return the ask-the-user sentinel when the spike question is still open.

    None when nothing needs asking: ineligible architecture, or the choice was
    already made (interactively, via the architecture_spike param, the
    YEABOI_ARCHITECTURE_SPIKE env override, or the headless auto-rule).
    """
    if not _spike_eligible(analysis) or state.get("spike_choice") or _spike_choice_override():
        return None
    arch = analysis.architecture
    rec = "add" if spike_recommended(arch.confidence) else "skip"
    question = (
        f"The architecture choice is open — recommended: **{arch.chosen}** "
        f"(confidence: **{arch.confidence}**), against {len(arch.options) - 1} alternative(s).\n\n"
        f"Add a time-boxed **validation spike** (1-3 days) to the plan? "
        f"Recommended: **{rec}** — "
        + (
            "confidence is not high, so validating before committing de-risks the build."
            if rec == "add"
            else "confidence is high, so you can likely commit without a spike."
        )
    )
    logger.info("spike prompt: chosen=%s confidence=%s recommendation=%s", arch.chosen, arch.confidence, rec)
    return {
        "_spike_prompt": {
            "chosen": arch.chosen,
            "confidence": arch.confidence,
            "options": [o.name for o in arch.options],
            "recommended": "include" if rec == "add" else "skip",
        },
        "messages": [AIMessage(content=question)],
    }


def _spike_story_acs(analysis: ProjectAnalysis, ac_style: str) -> tuple[AcceptanceCriterion, ...]:
    """The spike story's ACs, mapped from the documented Spike DoD, style-matched."""
    arch = analysis.architecture
    option_names = ", ".join(o.name for o in arch.options)
    if ac_style == "bullets":
        return (
            AcceptanceCriterion(text=f"The spike objective and 1-3 day time-box are stated up front ({option_names})."),
            AcceptanceCriterion(
                text="Findings are documented with a recommendation including trade-offs and rejected alternatives."
            ),
            AcceptanceCriterion(text="Next steps (adopt or adjust the plan) are agreed with the team and linked."),
        )
    return (
        AcceptanceCriterion(
            given=f"the candidate architectures ({option_names})",
            when="the spike starts",
            then="the objective and 1-3 day time-box are stated",
        ),
        AcceptanceCriterion(
            given="the spike work",
            when="it concludes",
            then="findings are documented with a recommendation including trade-offs and rejected alternatives",
        ),
        AcceptanceCriterion(
            given="the recommendation",
            when="it is shared with the team",
            then="next steps (adopt or adjust the plan) are agreed and resources are linked",
        ),
    )


def _inject_architecture_spike_story(
    stories: list[UserStory],
    features: list[Feature],
    analysis: ProjectAnalysis,
    ac_style: str = "gwt",
    dod_count: int = 0,
) -> tuple[list[UserStory], str | None]:
    """Prepend the '[Spike] Validate architecture' story (large mode).

    Deterministic post-parse injection — attaches to the first (highest-value)
    feature, CRITICAL priority so the sprint planner schedules it first, 2
    points (the spike DoD's 1-3 day time-box). Idempotent across Edit re-runs
    via the title prefix. Docs/Knowledge-Sharing DoD items stay applicable
    (the findings write-up IS the deliverable); Testing/Merge/SDLC do not.
    """
    if not features or any(s.title.startswith(_SPIKE_TITLE_PREFIX) for s in stories):
        return stories, None
    arch = analysis.architecture
    runner_up = next((o.name for o in arch.options if o.name != arch.chosen), "the alternatives")
    # DoD flags are positional against the plan's RESOLVED list. The curated
    # pattern (Testing/Merge/SDLC not applicable) only means anything on the
    # default 7-item list; a custom DoD gets all-applicable of its own length.
    n_dod = dod_count or len(DOD_ITEMS)
    curated = (True, True, False, False, False, True, True)
    spike = UserStory(
        id=f"US-{features[0].id}-SPIKE",
        feature_id=features[0].id,
        persona="development team",
        goal=f"run a time-boxed spike (1-3 days) validating the recommended {arch.chosen} architecture "
        f"against {runner_up}",
        benefit="we commit to an architecture based on evidence rather than assumption",
        acceptance_criteria=_spike_story_acs(analysis, ac_style),
        story_points=StoryPointValue.TWO,
        priority=Priority.CRITICAL,
        title=f"{_SPIKE_TITLE_PREFIX}Validate architecture: {arch.chosen}",
        discipline=Discipline.FULLSTACK,
        dod_applicable=curated if n_dod == len(curated) else (True,) * n_dod,
        points_rationale=f"Time-boxed architecture spike; confidence in the recommendation is {arch.confidence}.",
    )
    logger.info("story_writer: injected architecture spike story %s", spike.id)
    note = (
        f"Added {spike.title} — scheduled first to de-risk the open architecture decision "
        f"(confidence: {arch.confidence})."
    )
    return [spike, *stories], note


def _inject_architecture_spike_task(
    tasks: list[Task],
    stories: list[UserStory],
    analysis: ProjectAnalysis,
) -> tuple[list[Task], str | None]:
    """Splice the spike TASK first among the first story's tasks (small mode).

    Small mode caps stories at 2, so a story-level spike would crowd out real
    work — the spike rides as the first sub-task instead. TaskLabel.SPIKE is
    set here directly (the LLM is never asked to emit it), and test_plan stays
    empty: the deliverable is a written recommendation, not code.
    """
    if not stories or any(t.title.startswith(_SPIKE_TITLE_PREFIX) for t in tasks):
        return tasks, None
    arch = analysis.architecture
    first_story = stories[0]
    trade_offs = "\n".join(
        f"- {o.name}: {o.summary or 'candidate architecture'}" + (f" (cons: {'; '.join(o.cons)})" if o.cons else "")
        for o in arch.options
    )
    spike = Task(
        id=f"T-{first_story.id}-SPIKE",
        story_id=first_story.id,
        title=f"{_SPIKE_TITLE_PREFIX}Validate architecture: {arch.chosen}",
        description=(
            f"Time-boxed spike (1-3 days): validate the recommended {arch.chosen} architecture "
            f"before building on it. Candidates considered:\n{trade_offs}\n"
            "Deliverables: documented findings, a recommendation with trade-offs and rejected "
            "alternatives, and agreed next steps (adopt or adjust the plan)."
        ),
        label=TaskLabel.SPIKE,
        test_plan="",
        ai_prompt=(
            f"You are a tech lead evaluating architectures for {analysis.project_name} "
            f"({', '.join(analysis.tech_stack) or 'stack TBD'}). Compare: "
            f"{', '.join(o.name for o in arch.options)}. Produce a written recommendation "
            "with concrete trade-offs, what evidence would change the pick, and next steps."
        ),
    )
    insert_at = next((i for i, t in enumerate(tasks) if t.story_id == first_story.id), len(tasks))
    logger.info("task_decomposer: injected architecture spike task %s at index %d", spike.id, insert_at)
    note = f"Added {spike.title} as the first task — a 1-3 day spike to validate the open architecture decision."
    return [*tasks[:insert_at], spike, *tasks[insert_at:]], note


def _pin_spike_to_first_sprint(sprints: list[Sprint], stories: list[UserStory]) -> list[Sprint]:
    """Guarantee the spike story lands in the first sprint.

    The sprint-planner prompt already asks for spikes first (rule 4); this
    enforces it deterministically after parse, moving a misplaced spike story
    into sprints[0] via dataclasses.replace (Sprint is frozen).
    """
    spike_ids = {s.id for s in stories if s.title.startswith(_SPIKE_TITLE_PREFIX)}
    if not spike_ids or not sprints:
        return sprints
    first = sprints[0]
    misplaced = [sid for sid in spike_ids if sid not in first.story_ids]
    if not misplaced:
        return sprints
    # capacity_points is defined as the sum of the sprint's story points
    # (_validate_sprints), so every sprint whose story_ids change here must
    # have it recomputed — the pin runs after validation.
    points = {s.id: s.story_points.value for s in stories}
    result: list[Sprint] = []
    moved: list[str] = []
    for i, sprint in enumerate(sprints):
        if i == 0:
            continue  # rebuilt last, once we know what moves
        remaining = tuple(sid for sid in sprint.story_ids if sid not in spike_ids)
        if len(remaining) != len(sprint.story_ids):
            moved.extend(sid for sid in sprint.story_ids if sid in spike_ids)
            sprint = dataclasses.replace(
                sprint, story_ids=remaining, capacity_points=sum(points.get(sid, 0) for sid in remaining)
            )
        result.append(sprint)
    if moved:
        logger.info("sprint_planner: pinned spike story(ies) %s into %s", moved, sprints[0].name)
        pinned_ids = (*moved, *first.story_ids)
        first = dataclasses.replace(
            first, story_ids=pinned_ids, capacity_points=sum(points.get(sid, 0) for sid in pinned_ids)
        )
    return [first, *result]


def story_writer(state: ScrumState) -> dict:
    """LangGraph node: decompose features into user stories with ACs and points.

    # See docs: "Agentic Blueprint Reference" — node return format
    # See docs: "Architecture" — story_writer sits between feature_generator and agent
    # See docs: "Scrum Standards" — story format, acceptance criteria, story points
    #
    # How this works:
    # 1. Read the ProjectAnalysis and features from state.
    # 2. Format analysis fields and features into strings for the prompt.
    # 3. Call the LLM with the story writer prompt (temperature=0.0).
    # 4. Parse the JSON array response into UserStory dataclasses.
    # 5. Return {"stories": [...], "messages": [AIMessage]}.
    #
    # Why a single LLM call for all features (not per-feature)?
    # Projects have 3-6 features producing 6-30 stories. A single call keeps
    # the pattern consistent with feature_generator, avoids per-feature prompt
    # complexity, and lets the LLM see all features to avoid cross-feature
    # duplication. ~6000 output tokens is well within Claude's limits.
    #
    # Why this returns to END (not to agent)?
    # Same pattern as feature_generator — the node produces output, the REPL
    # displays it, and the user types anything to continue. On the next
    # invocation, route_entry sees stories populated and routes to the agent.

    Args:
        state: The current LangGraph state with project_analysis and features.

    Returns:
        A dict updating stories and messages.
    """
    analysis: ProjectAnalysis = state["project_analysis"]
    features: list[Feature] = state["features"]

    # Architecture-validation spike: when the decision is open and the user
    # hasn't chosen yet, ask BEFORE generating stories (large mode asks here;
    # small mode asks in task_decomposer, where its spike lives). The TUI
    # popups answer by writing spike_choice; a plain-text driver's reply is
    # parsed here on the re-invoke.
    spike_updates: dict = {}
    if not _is_small_project_mode(state.get("_intake_mode")):
        if state.get("_spike_prompt") and not state.get("spike_choice"):
            choice = _parse_spike_reply(state)
            logger.info("story_writer: spike reply resolved to %s", choice)
            state = {**state, "spike_choice": choice}
            spike_updates = {"spike_choice": choice, "_spike_prompt": {}}
        else:
            spike_ask = _maybe_prompt_spike_choice(state, analysis)
            if spike_ask is not None:
                return spike_ask

    # Read review state (same pattern as feature_generator)
    review_decision = state.get("last_review_decision")
    review_feedback = state.get("last_review_feedback", "")
    review_mode = review_decision.value if review_decision else None

    previous_output = None
    if review_mode == "edit" and "---PREVIOUS OUTPUT---" in review_feedback:
        parts = review_feedback.split("---PREVIOUS OUTPUT---", 1)
        review_feedback = parts[0].strip()
        previous_output = parts[1].strip()

    # Format ProjectAnalysis fields and features into strings for the prompt.
    # This avoids passing dataclasses directly, keeping the prompt module
    # free of imports from agent.state (avoiding circular imports).
    # See docs: "Prompt Construction" — pre-formatted strings pattern
    features_block = _format_features_for_prompt(features)

    # Load team calibration for team-specific estimation rules.
    # See docs: "Scrum Standards" — team learning, self-calibrating estimates
    if _wants_dep(state, "analysis"):
        team_profile = _load_team_profile(state.get("analysis_profile_id", ""))
        team_calibration_text = _format_team_calibration(
            team_profile, examples=_load_team_examples(state.get("analysis_profile_id", ""))
        )
    else:
        team_profile, team_calibration_text = None, ""

    # Resolve DoD items — custom from analysis or default 7
    from yeaboi.agent.state import resolve_ac_style, resolve_dod_items

    _dod = resolve_dod_items(state)

    # Resolve the acceptance-criteria style once (session choice > env override
    # > learned team profile > GWT) and read the team's median AC count off the
    # profile directly — the prompt rules take these as structured params, not
    # regex-greps of the calibration markdown.
    # See docs: "Scrum Standards" — Acceptance Criteria
    ac_style = resolve_ac_style(state, team_profile)
    ac_median = round(getattr(getattr(team_profile, "writing_patterns", None), "median_ac_count", 0) or 0)
    logger.info("story_writer: ac_style=%s median_ac=%d", ac_style, ac_median)

    # The team's learned ticket-template section headings (naming conventions)
    # — persisted on state so exporters can adopt them on every surface.
    _tpl_sections: list[str] = []
    _examples = _load_team_examples(state.get("analysis_profile_id", "")) if _wants_dep(state, "analysis") else None
    if _examples:
        _naming = _examples.get("naming_conventions") or {}
        for entry in (_naming.get("template_sections") or [])[:8]:
            heading = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
            if isinstance(heading, str) and heading.strip():
                _tpl_sections.append(heading.strip())

    # Same predicate as the analyzer coercion — the two must not disagree on
    # what "small" means, or the sprint clamp and the story cap would drift.
    small_mode = _is_small_project_mode(state.get("_intake_mode"))

    prompt = get_story_writer_prompt(
        project_name=analysis.project_name,
        project_description=analysis.project_description,
        project_type=analysis.project_type,
        goals=_format_epic_list(analysis.goals),
        end_users=_format_epic_list(analysis.end_users),
        tech_stack=_format_epic_list(analysis.tech_stack),
        constraints=_format_epic_list(analysis.constraints),
        features_block=features_block,
        out_of_scope=_format_epic_list(analysis.out_of_scope),
        team_calibration=team_calibration_text,
        dod_items=_dod,
        ac_style=ac_style,
        ac_median_count=ac_median,
        is_low_code=analysis.is_low_code,
        carry_over_items=tuple(state.get("_ceremony_action_items", ()) or ()),
        review_feedback=review_feedback if review_mode else None,
        review_mode=review_mode,
        previous_output=previous_output,
        max_total_stories=SMALL_PROJECT_MAX_STORIES if small_mode else None,
    )

    # Screenshots attached to review-edit feedback (Ctrl+V) — review passes only.
    review_images = list(state.get("review_feedback_images") or []) if review_mode else []

    try:
        # Single LLM call with low temperature for deterministic JSON output.
        # See docs: "Agentic Blueprint Reference" — using the LLM outside the main graph
        response = _invoke_json(prompt, image_paths=review_images)
        stories = _parse_stories_response(response.content, features, analysis, ac_style, dod_count=len(_dod))
    except Exception as exc:
        if _should_reraise_llm_error(exc):
            raise
        # LLM call failed entirely — use deterministic fallback.
        logger.warning("LLM call failed in story_writer, using fallback", exc_info=True)
        stories = _build_fallback_stories(features, analysis, ac_style)

    # Validate stories against the Story Checklist — auto-fix where possible
    # and collect warnings for the user. Deterministic post-processing, no LLM.
    # A learned team median below 3 lowers the padding floor — a 1-AC team
    # must not have every story force-padded to 3.
    # See docs: "Scrum Standards" — Story Checklist
    stories, warnings = _validate_stories(stories, features, ac_style, min_acs=ac_median if ac_median > 0 else 3)

    # Inject the architecture-validation spike story when the user opted in
    # (large mode only — small mode's spike is a task; and never when the
    # small-mode cap below could immediately evict real work for it).
    effective_spike = state.get("spike_choice") or _spike_choice_override()
    if not small_mode and effective_spike == "include" and _spike_eligible(analysis):
        stories, spike_note = _inject_architecture_spike_story(stories, features, analysis, ac_style, len(_dod))
        if spike_note:
            warnings.append(spike_note)

    # Hard cap for small projects — the prompt asks, this enforces. Keep
    # document order: the LLM lists stories by importance, so the first
    # SMALL_PROJECT_MAX_STORIES are the ones to keep.
    if small_mode and len(stories) > SMALL_PROJECT_MAX_STORIES:
        dropped = stories[SMALL_PROJECT_MAX_STORIES:]
        stories = stories[:SMALL_PROJECT_MAX_STORIES]
        warnings.append(
            f"Small project cap: kept the first {SMALL_PROJECT_MAX_STORIES} stories and dropped "
            f"{len(dropped)}: " + "; ".join(f"{s.id} ({s.title or s.goal})" for s in dropped)
        )
        logger.info(
            "story_writer: small-project cap trimmed %d -> %d stories",
            SMALL_PROJECT_MAX_STORIES + len(dropped),
            SMALL_PROJECT_MAX_STORIES,
        )

    # Format the stories for display (with warnings if any)
    display = _format_stories(stories, features, analysis.project_name, warnings=warnings)

    return {
        "stories": stories,
        # Persist the resolved style + learned template headings so every later
        # consumer (exports, editor, re-runs, MCP) matches how the stories were
        # actually written — on all surfaces, not just the TUI.
        "ac_format": ac_style,
        "ticket_template_sections": _tpl_sections,
        **spike_updates,
        "messages": [AIMessage(content=display)],
        "pending_review": "story_writer",
    }


# ---------------------------------------------------------------------------
# Task decomposer node — breaks UserStories into Task dataclasses
# ---------------------------------------------------------------------------
# See docs: "Architecture" — task_decomposer sits between story_writer and agent
# See docs: "Scrum Standards" — task decomposition
#
# Same pattern as story_writer: extract from state → build prompt →
# LLM call → parse JSON → fallback → format display → return state update.
# The node returns to END so the REPL can display the tasks and wait for user
# input. On the next invocation, route_entry sees tasks populated and routes
# to the agent node.


def _build_doc_context(state: ScrumState) -> str | None:
    """Gather documentation references from intake answers and external sources.

    Collects documentation context from three places:
    1. Q14 answer — "Is there any existing documentation, PRDs, or design docs?"
    2. confluence_context — text scraped from Confluence during intake.
    3. user_context — free-form markdown from SCRUM.md (may contain doc URLs).

    Returns a formatted string for injection into the task decomposer prompt,
    or None if no documentation context is available.

    # See docs: "Tools" — read-only tool pattern
    # See docs: "Prompt Construction" — context injection
    """
    parts: list[str] = []

    # Q14: existing documentation, PRDs, design docs
    questionnaire = state.get("questionnaire")
    if questionnaire and questionnaire.answers.get(14):
        q14 = questionnaire.answers[14]
        # Skip the default "No existing documentation to reference"
        if "no existing documentation" not in q14.lower():
            parts.append(f"- **Existing docs (from intake):** {q14}")

    # Confluence context — scraped page content from confluence_search_docs / confluence_read_page
    confluence = state.get("confluence_context", "")
    if confluence and confluence.strip():
        parts.append(f"- **Confluence:** {confluence.strip()[:500]}")

    # Notion context — scraped page content from notion_search_pages / notion_read_page
    notion = state.get("notion_context", "")
    if notion and notion.strip():
        parts.append(f"- **Notion:** {notion.strip()[:500]}")

    # User context from SCRUM.md — may contain wiki URLs, design doc links, etc.
    user_ctx = state.get("user_context", "")
    if user_ctx and user_ctx.strip():
        parts.append(f"- **Project docs (SCRUM.md):** {user_ctx.strip()[:500]}")

    if not parts:
        return None

    return "\n".join(parts)


def _format_stories_for_prompt(stories: list[UserStory], features: list[Feature]) -> str:
    """Format stories grouped by feature for the task decomposer prompt.

    For each story shows: ID, story text, story points, discipline,
    a summary of acceptance criteria, and a `[Documentation in DoD]`
    annotation when the story's Definition of Done includes documentation
    (dod_applicable[1] == True). The task decomposer prompt uses this
    annotation to decide whether to generate a dedicated documentation sub-task.

    Args:
        stories: List of UserStory dataclasses from the story_writer node.
        features: List of Feature dataclasses for grouping headers.

    Returns:
        A formatted multi-line string describing all stories grouped by feature.
    """
    feature_map = {e.id: e for e in features}

    # Group stories by feature_id, preserving order
    stories_by_feature: dict[str, list[UserStory]] = {}
    for story in stories:
        stories_by_feature.setdefault(story.feature_id, []).append(story)

    lines: list[str] = []
    for feature_id, feature_stories in stories_by_feature.items():
        feature = feature_map.get(feature_id)
        feature_title = feature.title if feature else feature_id
        lines.append(f"### {feature_id}: {feature_title}\n")

        for story in feature_stories:
            # Check if Documentation (index 1 in DOD_ITEMS) is applicable for this story.
            # When True, the prompt instructs the LLM to generate a dedicated documentation
            # sub-task that consolidates all doc work for the story.
            # See docs: "Scrum Standards" — Definition of Done
            doc_in_dod = len(story.dod_applicable) > 1 and story.dod_applicable[1]
            dod_tag = " [Documentation in DoD]" if doc_in_dod else ""

            lines.append(f"**{story.id}** ({story.story_points.value} pts, {story.discipline.value}){dod_tag}")
            lines.append(f"  As a {story.persona}, I want to {story.goal}, so that {story.benefit}.")
            if story.acceptance_criteria:
                lines.append("  ACs:")
                for ac in story.acceptance_criteria:
                    lines.append(f"    - {ac.flat_text}")
            lines.append("")

    return "\n".join(lines)


def _parse_tasks_response(raw: str, stories: list[UserStory]) -> list[Task]:
    """Parse the LLM's JSON response into a list of Task dataclasses.

    # See docs: "Scrum Standards" — task decomposition
    #
    # Strips markdown code fences, parses JSON array, validates story_id
    # against known story IDs, auto-generates IDs when missing, and falls
    # back to _build_fallback_tasks on error. Same defensive pattern as
    # _parse_stories_response.

    Args:
        raw: The raw LLM response string (expected to be a JSON array).
        stories: The list of stories (for story_id validation and fallback).

    Returns:
        A list of Task instances.
    """
    try:
        # Strip a leaked <think> block, then the markdown fences LLMs wrap JSON
        # in. Reasoning left in place defeats the fence strip and the parse falls
        # through to the deterministic fallback with no error.
        text = strip_think_tags(raw).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return _build_fallback_tasks(stories)

        # Build a set of valid story IDs for validation
        valid_story_ids = {s.id for s in stories}

        # Track per-story task counts for auto-ID generation
        story_task_counts: dict[str, int] = {}

        tasks: list[Task] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue

            # Validate story_id — skip tasks with unknown story IDs
            story_id = str(item.get("story_id", ""))
            if story_id not in valid_story_ids:
                continue

            # Auto-generate ID if missing
            story_task_counts.setdefault(story_id, 0)
            story_task_counts[story_id] += 1
            task_id = str(item.get("id", ""))
            if not task_id:
                task_id = f"T-{story_id}-{story_task_counts[story_id]:02d}"

            # Validate non-empty title and description
            title = str(item.get("title", "")).strip()
            if not title:
                title = f"Implement task for {story_id}"

            description = str(item.get("description", "")).strip()
            if not description:
                description = f"Implementation task for story {story_id}"

            # Parse label — default to CODE if missing or invalid.
            # The LLM is prompted to return one of: Code, Documentation, Infrastructure, Testing.
            # See docs: "Scrum Standards" — task decomposition, task labels
            raw_label = str(item.get("label", "")).strip()
            try:
                label = TaskLabel(raw_label)
            except ValueError:
                label = TaskLabel.CODE

            # Parse test_plan — only expected for Code/Infrastructure tasks.
            # Empty string for Documentation/Testing tasks or if missing.
            test_plan = str(item.get("test_plan", "")).strip()

            # Parse ai_prompt — self-contained ARC-structured instruction for AI coding assistants.
            ai_prompt = str(item.get("ai_prompt", "")).strip()

            tasks.append(
                Task(
                    id=task_id,
                    story_id=story_id,
                    title=title,
                    description=description,
                    label=label,
                    test_plan=test_plan,
                    ai_prompt=ai_prompt,
                )
            )

        if not tasks:
            return _build_fallback_tasks(stories)

        return tasks

    except Exception:
        logger.debug("Failed to parse tasks JSON, falling back to deterministic extraction", exc_info=True)
        return _build_fallback_tasks(stories)


def _build_fallback_tasks(stories: list[UserStory]) -> list[Task]:
    """Build 2 deterministic fallback tasks per story.

    # See docs: "Scrum Standards" — task decomposition
    #
    # When the LLM fails to produce valid JSON, this function creates 2
    # generic tasks per story that give downstream nodes something to work with:
    # 1. Core implementation — implement the story's goal
    # 2. Testing — write tests for the story's goal

    Args:
        stories: The list of stories to generate tasks for.

    Returns:
        A list of Task instances (2 per story).
    """
    tasks: list[Task] = []

    for story in stories:
        # Task 1: core implementation
        tasks.append(
            Task(
                id=f"T-{story.id}-01",
                story_id=story.id,
                title=f"Implement {story.goal}",
                description=f"Core implementation for story {story.id}: {story.goal}",
            )
        )

        # Task 2: testing
        tasks.append(
            Task(
                id=f"T-{story.id}-02",
                story_id=story.id,
                title=f"Write tests for {story.goal}",
                description=f"Write unit and integration tests for story {story.id}: {story.goal}",
            )
        )

    return tasks


def _format_tasks(
    tasks: list[Task],
    stories: list[UserStory],
    features: list[Feature],
    project_name: str,
) -> str:
    """Format a list of Tasks as a markdown display for the user.

    Groups tasks by feature → story for readability. Shows the task ID, title,
    and description. The REPL renders this as Rich tables and waits for
    user review before routing to the next pipeline step.

    Args:
        tasks: List of Task dataclasses to display.
        stories: List of UserStory dataclasses (for grouping and context).
        features: List of Feature dataclasses (for grouping headers).
        project_name: Project name for the header.

    Returns:
        A formatted markdown string.
    """
    sections = [
        f"# Task Decomposition: {project_name}\n",
        f"**{len(tasks)} task(s) across {len(stories)} story(ies)**\n",
    ]

    # Build lookups for grouping
    feature_map = {e.id: e for e in features}

    # Group tasks by story_id
    tasks_by_story: dict[str, list[Task]] = {}
    for task in tasks:
        tasks_by_story.setdefault(task.story_id, []).append(task)

    # Group stories by feature_id for display
    stories_by_feature: dict[str, list[UserStory]] = {}
    for story in stories:
        stories_by_feature.setdefault(story.feature_id, []).append(story)

    for feature_id, feature_stories in stories_by_feature.items():
        feature = feature_map.get(feature_id)
        feature_title = feature.title if feature else feature_id
        sections.append(f"## {feature_id}: {feature_title}\n")

        for story in feature_stories:
            story_tasks = tasks_by_story.get(story.id, [])
            if not story_tasks:
                continue

            sections.append(f"### {story.id}: {story.goal}")
            sections.append(f"**Points:** {story.story_points.value} | **Priority:** {story.priority.value}\n")

            for task in story_tasks:
                _lbl = task.label.value if hasattr(task.label, "value") else str(task.label)
                sections.append(f"- **{task.id}** [{_lbl}]: {task.title}")
                sections.append(f"  {task.description}")
                if task.test_plan:
                    sections.append(f"  **Test plan:** {task.test_plan}")
                if task.ai_prompt:
                    sections.append(f"  **AI prompt:** {task.ai_prompt}")
            sections.append("")

    sections.append("\n---\n**[Accept / Edit / Reject]** — Review the tasks above.")

    return "\n".join(sections)


def _parallel_task_decompose(
    stories: list[UserStory],
    features: list[Feature],
    project_name: str,
    project_type: str,
    tech_stack: str,
    doc_context: str | None,
    team_calibration: str = "",
    is_low_code: bool = False,
) -> list[Task]:
    """Decompose stories into tasks with parallel LLM calls per feature.

    # See docs: "Architecture" — task_decomposer parallelisation
    #
    # Groups stories by feature, sends one LLM call per feature concurrently
    # (max 4 workers to stay within Bedrock concurrency limits), then merges
    # results. Falls back to deterministic tasks for any feature whose LLM
    # call fails.
    #
    # Why per-feature batches (not per-story)?
    # - Stories within a feature share context — the LLM can avoid duplicate tasks.
    # - Typical projects have 4-6 features → 4-6 concurrent calls → good parallelism.
    # - Per-story would mean 10-30 calls, risking Bedrock throttling.

    Args:
        stories: All stories from the story_writer node.
        features: All features from the feature_generator node.
        project_name: From ProjectAnalysis.
        project_type: From ProjectAnalysis.
        tech_stack: Pre-formatted tech stack string.
        doc_context: Optional documentation context.

    Returns:
        Merged list of Task dataclasses from all features.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Group stories by feature_id
    stories_by_feature: dict[str, list[UserStory]] = {}
    for story in stories:
        stories_by_feature.setdefault(story.feature_id, []).append(story)

    feature_map = {f.id: f for f in features}
    all_tasks: list[Task] = []

    def _decompose_feature(feature_id: str, feature_stories: list[UserStory]) -> list[Task]:
        """LLM call for a single feature's stories."""
        feature = feature_map.get(feature_id)
        feature_list = [feature] if feature else []
        stories_block = _format_stories_for_prompt(feature_stories, feature_list)

        prompt = get_task_decomposer_prompt(
            project_name=project_name,
            project_type=project_type,
            tech_stack=tech_stack,
            stories_block=stories_block,
            doc_context=doc_context,
            team_calibration=team_calibration,
            is_low_code=is_low_code,
        )

        try:
            response = _invoke_json(prompt)
            return _parse_tasks_response(response.content, feature_stories)
        except Exception as exc:
            if _should_reraise_llm_error(exc):
                raise
            logger.warning("LLM call failed for feature %s, using fallback", feature_id, exc_info=True)
            return _build_fallback_tasks(feature_stories)

    # Cap at 4 workers — Bedrock's per-account concurrency is typically 4-8.
    max_workers = min(4, len(stories_by_feature))

    if max_workers <= 1:
        # Only 1 feature — no benefit from threading
        for fid, fstories in stories_by_feature.items():
            all_tasks.extend(_decompose_feature(fid, fstories))
    else:
        logger.info("Parallel task decomposition: %d features, %d workers", len(stories_by_feature), max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_decompose_feature, fid, fstories): fid for fid, fstories in stories_by_feature.items()
            }
            for future in as_completed(futures):
                fid = futures[future]
                try:
                    all_tasks.extend(future.result())
                except Exception:
                    logger.warning("Feature %s task decomposition failed, using fallback", fid, exc_info=True)
                    all_tasks.extend(_build_fallback_tasks(stories_by_feature[fid]))

    return all_tasks


def task_decomposer(state: ScrumState) -> dict:
    """LangGraph node: decompose user stories into concrete implementation tasks.

    # See docs: "Agentic Blueprint Reference" — node return format
    # See docs: "Architecture" — task_decomposer sits between story_writer and agent
    # See docs: "Scrum Standards" — task decomposition
    #
    # How this works:
    # 1. Read the ProjectAnalysis, features, and stories from state.
    # 2. Format stories into a text block for the prompt.
    # 3. Call the LLM with the task decomposer prompt (temperature=0.0).
    # 4. Parse the JSON array response into Task dataclasses.
    # 5. Return {"tasks": [...], "messages": [AIMessage]}.
    #
    # Why a single LLM call for all stories (not per-story)?
    # Projects have 6-30 stories producing 12-150 tasks. A single call keeps
    # the pattern consistent with story_writer, avoids per-story prompt
    # complexity, and lets the LLM see all stories to avoid cross-story
    # task duplication.
    #
    # Why this returns to END (not to agent)?
    # Same pattern as story_writer — the node produces output, the REPL
    # displays it, and the user types anything to continue. On the next
    # invocation, route_entry sees tasks populated and routes to the agent.

    Args:
        state: The current LangGraph state with project_analysis, features, and stories.

    Returns:
        A dict updating tasks and messages.
    """
    analysis: ProjectAnalysis = state["project_analysis"]
    features: list[Feature] = state["features"]
    stories: list[UserStory] = state["stories"]

    # Architecture-validation spike (small mode): the spike rides as a task
    # here, so this is where small mode asks the opt-in question — same
    # sentinel + driver-popup pattern as story_writer's large-mode ask.
    spike_updates: dict = {}
    if _is_small_project_mode(state.get("_intake_mode")):
        if state.get("_spike_prompt") and not state.get("spike_choice"):
            choice = _parse_spike_reply(state)
            logger.info("task_decomposer: spike reply resolved to %s", choice)
            state = {**state, "spike_choice": choice}
            spike_updates = {"spike_choice": choice, "_spike_prompt": {}}
        else:
            spike_ask = _maybe_prompt_spike_choice(state, analysis)
            if spike_ask is not None:
                return spike_ask

    # Read review state (same pattern as feature_generator)
    review_decision = state.get("last_review_decision")
    review_feedback = state.get("last_review_feedback", "")
    review_mode = review_decision.value if review_decision else None

    previous_output = None
    if review_mode == "edit" and "---PREVIOUS OUTPUT---" in review_feedback:
        parts = review_feedback.split("---PREVIOUS OUTPUT---", 1)
        review_feedback = parts[0].strip()
        previous_output = parts[1].strip()

    # Build documentation context from intake answers and external sources.
    doc_context = _build_doc_context(state)
    tech_stack_str = _format_epic_list(analysis.tech_stack)

    # Load team calibration for team-specific task patterns.
    # See docs: "Scrum Standards" — team learning, self-calibrating estimates
    if _wants_dep(state, "analysis"):
        team_profile = _load_team_profile(state.get("analysis_profile_id", ""))
        team_calibration_text = _format_team_calibration(
            team_profile, examples=_load_team_examples(state.get("analysis_profile_id", ""))
        )
    else:
        team_profile, team_calibration_text = None, ""

    # ── Parallel task decomposition by feature ────────────────────────────
    # Split stories into per-feature groups and decompose concurrently.
    # Each feature is independent — tasks in Feature 1 don't depend on
    # Feature 2. This cuts wall-clock time from ~200s to ~60-80s for
    # typical projects with 4-6 features.
    #
    # When in review mode (edit/reject), fall back to the original single
    # call since the user's feedback may reference cross-feature concerns.
    # See docs: "Architecture" — task_decomposer parallelisation

    if review_mode:
        # Review mode: single call with all stories (user feedback may span features)
        stories_block = _format_stories_for_prompt(stories, features)
        prompt = get_task_decomposer_prompt(
            project_name=analysis.project_name,
            project_type=analysis.project_type,
            tech_stack=tech_stack_str,
            stories_block=stories_block,
            doc_context=doc_context,
            team_calibration=team_calibration_text,
            is_low_code=analysis.is_low_code,
            review_feedback=review_feedback,
            review_mode=review_mode,
            previous_output=previous_output,
        )
        # Screenshots attached to review-edit feedback (Ctrl+V) — review passes only.
        review_images = list(state.get("review_feedback_images") or [])
        try:
            response = _invoke_json(prompt, image_paths=review_images)
            tasks = _parse_tasks_response(response.content, stories)
        except Exception as exc:
            if _should_reraise_llm_error(exc):
                raise
            logger.warning("LLM call failed in task_decomposer (review), using fallback", exc_info=True)
            tasks = _build_fallback_tasks(stories)
    else:
        # Normal mode: parallel calls per feature
        tasks = _parallel_task_decompose(
            stories=stories,
            features=features,
            project_name=analysis.project_name,
            project_type=analysis.project_type,
            tech_stack=tech_stack_str,
            doc_context=doc_context,
            team_calibration=team_calibration_text,
            is_low_code=analysis.is_low_code,
        )

    # Inject the architecture-validation spike task when the user opted in
    # (small mode only — large mode injected a spike story instead).
    spike_note = None
    if (
        _is_small_project_mode(state.get("_intake_mode"))
        and (state.get("spike_choice") or _spike_choice_override()) == "include"
        and _spike_eligible(analysis)
    ):
        tasks, spike_note = _inject_architecture_spike_task(tasks, stories, analysis)

    # Format the tasks for display
    display = _format_tasks(tasks, stories, features, analysis.project_name)
    if spike_note:
        display = f"{display}\n\n_{spike_note}_"

    return {
        "tasks": tasks,
        **spike_updates,
        "messages": [AIMessage(content=display)],
        "pending_review": "task_decomposer",
    }


# ---------------------------------------------------------------------------
# Sprint planner node — allocates stories to sprints based on velocity
# ---------------------------------------------------------------------------
# See docs: "Architecture" — sprint_planner sits between task_decomposer and agent
# See docs: "Scrum Standards" — sprint planning, capacity allocation
#
# Hybrid approach: LLM allocates stories and writes natural-language sprint goals,
# then a deterministic validator ensures no sprint exceeds velocity, no stories are
# orphaned/duplicated, and capacity_points are correct. Fallback is a greedy
# bin-packing algorithm.
#
# Why not fully deterministic? Sprint goals like "Establish authentication foundation"
# require LLM reasoning. A greedy algorithm would produce generic goals.
#
# Why a single LLM call? Same pattern as task_decomposer — total stories fit in
# one call easily, and the LLM needs to see all stories to write coherent sprint goals.

# Priority sort order for the greedy fallback algorithm.
# Maps Priority enum values to sort keys (lower = higher priority = earlier sprint).
_PRIORITY_SORT_ORDER = {
    Priority.CRITICAL: 0,
    Priority.HIGH: 1,
    Priority.MEDIUM: 2,
    Priority.LOW: 3,
}


def _format_stories_for_sprint_planner(stories: list[UserStory], features: list[Feature]) -> str:
    """Format stories in a compact layout for the sprint planner prompt.

    # See docs: "Prompt Construction" — pre-formatted strings pattern
    #
    # Unlike _format_stories_for_prompt (used by task_decomposer), this format
    # omits acceptance criteria — the sprint planner only needs ID, points,
    # priority, discipline, and goal for capacity allocation decisions.

    Args:
        stories: List of UserStory dataclasses to format.
        features: List of Feature dataclasses for grouping headers.

    Returns:
        A formatted multi-line string with stories grouped by feature.
    """
    feature_map = {e.id: e for e in features}

    # Group stories by feature_id, preserving order
    stories_by_feature: dict[str, list[UserStory]] = {}
    for story in stories:
        stories_by_feature.setdefault(story.feature_id, []).append(story)

    lines: list[str] = []
    for feature_id, feature_stories in stories_by_feature.items():
        feature = feature_map.get(feature_id)
        feature_title = feature.title if feature else feature_id
        feature_priority = feature.priority.value if feature else "medium"
        lines.append(f"### {feature_id}: {feature_title} ({feature_priority})")

        for story in feature_stories:
            lines.append(
                f"- **{story.id}** | {story.story_points.value} pts | "
                f"{story.priority.value} | {story.discipline.value} — {story.goal}"
            )
        lines.append("")

    return "\n".join(lines)


def _validate_sprint_capacity(
    sprints: list[Sprint],
    stories: list[UserStory],
    velocity: int,
) -> list[Sprint]:
    """Post-parse validation: fix capacity math, redistribute over-packed sprints, handle orphans.

    # See docs: "Scrum Standards" — sprint capacity validation
    #
    # The LLM is good at allocating stories but sometimes gets the math wrong.
    # This function corrects three classes of errors:
    # 1. Wrong capacity_points — recalculate from actual story points
    # 2. Over-packed sprints — move excess stories to the next sprint
    # 3. Orphaned stories — append any missing stories to the last sprint
    #
    # Sprint dataclasses are frozen, so we rebuild them from scratch.

    Args:
        sprints: The parsed Sprint list (may have incorrect capacity/duplicates).
        stories: The full story list (for point lookup and orphan detection).
        velocity: Team velocity cap per sprint.

    Returns:
        A corrected list of Sprint dataclasses.
    """
    story_points_map = {s.id: s.story_points.value for s in stories}
    all_story_ids = {s.id for s in stories}

    # 1. Deduplicate: keep first occurrence of each story across all sprints
    seen: set[str] = set()
    deduped_sprints: list[list[str]] = []
    sprint_meta: list[tuple[str, str, str]] = []  # (id, name, goal) per sprint

    for sp in sprints:
        unique_ids: list[str] = []
        for sid in sp.story_ids:
            if sid not in seen and sid in all_story_ids:
                unique_ids.append(sid)
                seen.add(sid)
        deduped_sprints.append(unique_ids)
        sprint_meta.append((sp.id, sp.name, sp.goal))

    # 2. Redistribute: if any sprint exceeds velocity, move excess to next sprint
    redistributed: list[list[str]] = []
    overflow: list[str] = []
    for i, story_ids in enumerate(deduped_sprints):
        current = list(overflow) + story_ids
        overflow = []

        # Special case: single story exceeding velocity gets its own sprint
        in_sprint: list[str] = []
        total = 0
        for sid in current:
            pts = story_points_map.get(sid, 0)
            if total + pts > velocity and in_sprint:
                # This story would exceed capacity — overflow it
                overflow.append(sid)
            else:
                in_sprint.append(sid)
                total += pts

        redistributed.append(in_sprint)

    # Handle remaining overflow — create new sprints as needed
    while overflow:
        in_sprint: list[str] = []
        total = 0
        remaining: list[str] = []
        for sid in overflow:
            pts = story_points_map.get(sid, 0)
            if total + pts > velocity and in_sprint:
                remaining.append(sid)
            else:
                in_sprint.append(sid)
                total += pts
        redistributed.append(in_sprint)
        overflow = remaining

    # 3. Orphan check: find stories not in any sprint
    assigned = set()
    for story_ids in redistributed:
        assigned.update(story_ids)
    orphans = [sid for sid in all_story_ids if sid not in assigned]

    if orphans:
        # Sort orphans by priority for consistent ordering
        orphan_priority = {s.id: _PRIORITY_SORT_ORDER.get(s.priority, 3) for s in stories}
        orphans.sort(key=lambda sid: orphan_priority.get(sid, 3))

        # Try to fit orphans into the last sprint first
        if redistributed:
            last = redistributed[-1]
            last_total = sum(story_points_map.get(sid, 0) for sid in last)
            still_orphaned: list[str] = []
            for sid in orphans:
                pts = story_points_map.get(sid, 0)
                if last_total + pts <= velocity:
                    last.append(sid)
                    last_total += pts
                else:
                    still_orphaned.append(sid)
            orphans = still_orphaned

        # Create new sprints for remaining orphans
        while orphans:
            in_sprint: list[str] = []
            total = 0
            remaining: list[str] = []
            for sid in orphans:
                pts = story_points_map.get(sid, 0)
                if total + pts > velocity and in_sprint:
                    remaining.append(sid)
                else:
                    in_sprint.append(sid)
                    total += pts
            redistributed.append(in_sprint)
            orphans = remaining

    # 4. Build final Sprint objects with correct capacity_points
    result: list[Sprint] = []
    for i, story_ids in enumerate(redistributed):
        if not story_ids:
            continue  # skip empty sprints

        sprint_num = i + 1
        capacity = sum(story_points_map.get(sid, 0) for sid in story_ids)

        # Reuse original metadata if available, otherwise generate
        if i < len(sprint_meta):
            sp_id, sp_name, sp_goal = sprint_meta[i]
        else:
            sp_id = f"SP-{sprint_num}"
            sp_name = f"Sprint {sprint_num}"
            sp_goal = f"Sprint {sprint_num} stories"

        result.append(
            Sprint(
                id=sp_id,
                name=sp_name,
                goal=sp_goal,
                capacity_points=capacity,
                story_ids=tuple(story_ids),
            )
        )

    return result


def _parse_sprints_response(
    raw: str,
    stories: list[UserStory],
    velocity: int,
    starting_sprint_number: int = 0,
) -> list[Sprint]:
    """Parse the LLM's JSON response into a list of Sprint dataclasses.

    # See docs: "Scrum Standards" — sprint format
    #
    # Strips markdown code fences, parses JSON array, validates story_ids
    # against known story IDs, auto-generates IDs when missing, and calls
    # _validate_sprint_capacity for post-parse correction. Falls back to
    # _build_fallback_sprints on any parse error.
    # Same defensive pattern as _parse_tasks_response.

    Args:
        raw: The raw LLM response string (expected to be a JSON array).
        stories: The list of stories (for story_id validation and fallback).
        velocity: Team velocity for capacity validation.

    Returns:
        A list of Sprint instances.
    """
    try:
        # Strip a leaked <think> block, then the markdown fences LLMs wrap JSON
        # in. Reasoning left in place defeats the fence strip and the parse falls
        # through to the deterministic fallback with no error.
        text = strip_think_tags(raw).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return _build_fallback_sprints(stories, velocity, starting_sprint_number)

        valid_story_ids = {s.id for s in stories}
        # When starting_sprint_number > 0, use real sprint numbers (e.g. 105, 106).
        # Otherwise fall back to generic 1-based numbering.
        base = starting_sprint_number if starting_sprint_number > 0 else 1

        sprints: list[Sprint] = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue

            sprint_num = base + i
            # Always use computed numbering for id/name to ensure consistency.
            # The LLM may return "Sprint 1" even when starting_sprint_number=3.
            sprint_id = f"SP-{sprint_num}"
            sprint_name = f"Sprint {sprint_num}"
            sprint_goal = str(item.get("goal", "")).strip() or f"Sprint {sprint_num} stories"

            # Parse capacity_points (will be recalculated by validator)
            try:
                capacity = int(item.get("capacity_points", 0))
            except (ValueError, TypeError):
                capacity = 0

            # Parse and validate story_ids
            raw_ids = item.get("story_ids", [])
            if not isinstance(raw_ids, list):
                raw_ids = []
            story_ids = tuple(str(sid) for sid in raw_ids if str(sid) in valid_story_ids)

            sprints.append(
                Sprint(
                    id=sprint_id,
                    name=sprint_name,
                    goal=sprint_goal,
                    capacity_points=capacity,
                    story_ids=story_ids,
                )
            )

        if not sprints:
            return _build_fallback_sprints(stories, velocity, starting_sprint_number)

        # Post-parse validation: fix capacity math, redistribute, handle orphans
        return _validate_sprint_capacity(sprints, stories, velocity)

    except Exception:
        logger.debug("Failed to parse sprints JSON, falling back to greedy bin-packing", exc_info=True)
        return _build_fallback_sprints(stories, velocity, starting_sprint_number)


def _build_fallback_sprints(stories: list[UserStory], velocity: int, starting_sprint_number: int = 0) -> list[Sprint]:
    """Greedy bin-packing fallback when the LLM fails to produce valid sprint JSON.

    # See docs: "Scrum Standards" — sprint planning fallback
    #
    # Sorts stories by priority (Critical first), then packs them into sprints
    # greedily: fill the current sprint until adding the next story would exceed
    # velocity, then start a new sprint.
    #
    # Edge case: a story whose points exceed velocity gets its own sprint.
    # The alternative — orphaning it — would lose work.

    Args:
        stories: The full list of stories to allocate.
        velocity: Team velocity cap per sprint.

    Returns:
        A list of Sprint instances (one per sprint, all stories allocated).
    """
    if not stories:
        return []

    # Sort by priority (stable sort preserves relative order within same priority)
    sorted_stories = sorted(stories, key=lambda s: _PRIORITY_SORT_ORDER.get(s.priority, 3))

    sprints: list[list[UserStory]] = []
    current: list[UserStory] = []
    current_points = 0

    for story in sorted_stories:
        pts = story.story_points.value

        if current and current_points + pts > velocity:
            # Current sprint is full — start a new one
            sprints.append(current)
            current = [story]
            current_points = pts
        else:
            current.append(story)
            current_points += pts

    # Don't forget the last sprint
    if current:
        sprints.append(current)

    # Build Sprint dataclasses
    # When starting_sprint_number > 0, use real sprint numbers (e.g. 105, 106).
    base = starting_sprint_number if starting_sprint_number > 0 else 1
    result: list[Sprint] = []
    for i, sprint_stories in enumerate(sprints):
        sprint_num = base + i
        capacity = sum(s.story_points.value for s in sprint_stories)
        story_ids = tuple(s.id for s in sprint_stories)

        # Determine dominant priority for the fallback goal
        priority_counts: dict[str, int] = {}
        for s in sprint_stories:
            priority_counts[s.priority.value] = priority_counts.get(s.priority.value, 0) + 1
        dominant = max(priority_counts, key=priority_counts.get) if priority_counts else "medium"

        result.append(
            Sprint(
                id=f"SP-{sprint_num}",
                name=f"Sprint {sprint_num}",
                goal=f"Sprint {sprint_num}: {dominant} priority stories",
                capacity_points=capacity,
                story_ids=story_ids,
            )
        )

    return result


def _merge_sprints_to_target(
    sprints: list[Sprint],
    target: int,
    stories: list[UserStory],
    starting_sprint_number: int = 0,
) -> list[Sprint]:
    """Merge sprints down to a target count when the user enforced a deadline.

    # See docs: "Scrum Standards" — sprint planning, capacity allocation
    #
    # When the user rejects the capacity recommendation and keeps their original
    # target, the LLM may still produce more sprints than requested. This function
    # merges them deterministically:
    # 1. Collect all story IDs in priority order from the existing sprints
    # 2. Distribute them round-robin across `target` buckets, keeping priority
    #    ordering so critical stories still land in earlier sprints
    # 3. Rebuild Sprint objects with correct IDs, names, and capacity

    Args:
        sprints: The sprints produced by the LLM or fallback (may exceed target).
        target: The enforced sprint count (user's deadline).
        stories: Full story list for point lookups.
        starting_sprint_number: Starting sprint number (e.g. 105) for naming.

    Returns:
        Exactly `target` Sprint objects with all stories distributed.
    """
    if len(sprints) <= target or target <= 0:
        return sprints

    # Collect all story IDs in their current order (preserves LLM's priority ordering)
    all_story_ids: list[str] = []
    for sp in sprints:
        all_story_ids.extend(sp.story_ids)

    # Build point lookup
    points_map = {s.id: s.story_points.value for s in stories}

    # Distribute stories across target buckets using greedy bin-packing.
    # Each bucket fills until adding the next story would make it the largest;
    # this produces a roughly even distribution.
    buckets: list[list[str]] = [[] for _ in range(target)]
    bucket_points: list[int] = [0] * target

    for sid in all_story_ids:
        pts = points_map.get(sid, 0)
        # Put in the lightest bucket (greedy even distribution)
        lightest = min(range(target), key=lambda i: bucket_points[i])
        buckets[lightest].append(sid)
        bucket_points[lightest] += pts

    # Preserve original sprint goals where possible
    original_goals = [sp.goal for sp in sprints]

    base = starting_sprint_number if starting_sprint_number > 0 else 1
    merged: list[Sprint] = []
    for i in range(target):
        sprint_num = base + i
        goal = original_goals[i] if i < len(original_goals) else f"Sprint {sprint_num} stories"
        merged.append(
            Sprint(
                id=f"SP-{sprint_num}",
                name=f"Sprint {sprint_num}",
                goal=goal,
                capacity_points=bucket_points[i],
                story_ids=tuple(buckets[i]),
            )
        )

    return merged


def _format_sprints(
    sprints: list[Sprint],
    stories: list[UserStory],
    features: list[Feature],
    project_name: str,
    velocity: int,
) -> str:
    """Format a list of Sprints as a rich markdown display for the user.

    Shows a header with sprint count and velocity, then per-sprint details
    including goal, capacity bar, and story list with feature context.
    The REPL renders this as Rich panels and waits for user review.

    Args:
        sprints: List of Sprint dataclasses to display.
        stories: List of UserStory dataclasses (for story details).
        features: List of Feature dataclasses (for feature context).
        project_name: Project name for the header.
        velocity: Team velocity for capacity display.

    Returns:
        A formatted markdown string.
    """
    story_map = {s.id: s for s in stories}
    feature_map = {e.id: e for e in features}

    total_points = sum(sp.capacity_points for sp in sprints)

    sections = [
        f"# Sprint Plan: {project_name}\n",
        f"**{len(sprints)} sprint(s)** | Velocity: **{velocity} pts/sprint** | Total: **{total_points} pts**\n",
    ]

    for sp in sprints:
        sections.append(f"## {sp.name} ({sp.capacity_points}/{velocity} pts)")
        sections.append(f"**Goal:** {sp.goal}\n")

        for sid in sp.story_ids:
            story = story_map.get(sid)
            if story:
                feature = feature_map.get(story.feature_id)
                feature_label = f"[{feature.title}]" if feature else f"[{story.feature_id}]"
                sections.append(
                    f"- **{story.id}** ({story.story_points.value} pts, {story.priority.value}) "
                    f"{feature_label} — {story.goal}"
                )
            else:
                sections.append(f"- **{sid}** (unknown story)")
        sections.append("")

    sections.append("\n---\n**[Accept / Edit / Reject]** — Review the sprint plan above.")

    return "\n".join(sections)


# sprint_selector and capacity_check were removed — sprint selection and
# capacity planning are now handled during intake (Phase 6: Q27-Q30).
# See _is_jira_configured(), _fetch_active_sprint_number(), _derive_q27_from_locale(),
# _extract_capacity_deductions(), and _compute_net_velocity() above.


def resolve_sprint_selection(user_input: str, current_sprint_number: int) -> int | None:
    """Resolve the user's sprint selection input into a starting sprint number.

    # See docs: "Scrum Standards" — sprint planning
    #
    # Used by the intake node when processing Q27 (sprint selection) with Jira,
    # and by the REPL for backwards compatibility.

    Args:
        user_input: The user's response (e.g. "1", "2", "3", or a number like "110").
        current_sprint_number: The active Jira sprint number (e.g. 104).

    Returns:
        The starting sprint number (e.g. 105), or None if input is invalid.
    """
    text = user_input.strip()

    # Quick option selection: 1, 2, 3 map to next, +2, +3
    if text == "1":
        return current_sprint_number + 1
    if text == "2":
        return current_sprint_number + 2
    if text == "3":
        return current_sprint_number + 3

    # Try to parse as a direct sprint number
    try:
        num = int(text)
        if num > 0:
            return num
        return None
    except ValueError:
        return None


def sprint_planner(state: ScrumState) -> dict:
    """LangGraph node: allocate stories to sprints based on team velocity.

    # See docs: "Agentic Blueprint Reference" — node return format
    # See docs: "Architecture" — sprint_planner sits between task_decomposer and agent
    # See docs: "Scrum Standards" — sprint planning, capacity allocation
    #
    # How this works:
    # 1. Read the ProjectAnalysis, features, stories, velocity, and target_sprints from state.
    # 2. Format stories in a compact layout for the prompt (no ACs — just points/priority).
    # 3. Call the LLM with the sprint planner prompt (temperature=0.0).
    # 4. Parse the JSON array response into Sprint dataclasses.
    # 5. Validate sprint capacity and fix any LLM math errors.
    # 6. Return {"sprints": [...], "messages": [AIMessage]}.
    #
    # Why a hybrid LLM + deterministic approach?
    # The LLM writes natural-language sprint goals (e.g. "Establish authentication
    # foundation") that a greedy algorithm can't produce. But the LLM sometimes
    # gets the math wrong, so _validate_sprint_capacity corrects capacity_points,
    # redistributes over-packed sprints, and ensures no stories are orphaned.
    #
    # Default velocity: team_size × 5 (same formula as _extract_team_and_velocity).
    # If velocity_per_sprint isn't in state, we calculate it here as a fallback.

    Args:
        state: The current LangGraph state with project_analysis, features, stories, and velocity.

    Returns:
        A dict updating sprints and messages.
    """
    analysis: ProjectAnalysis = state["project_analysis"]
    features: list[Feature] = state["features"]
    stories: list[UserStory] = state["stories"]

    # Read review state (same pattern as feature_generator)
    review_decision = state.get("last_review_decision")
    review_feedback = state.get("last_review_feedback", "")
    review_mode = review_decision.value if review_decision else None

    previous_output = None
    if review_mode == "edit" and "---PREVIOUS OUTPUT---" in review_feedback:
        parts = review_feedback.split("---PREVIOUS OUTPUT---", 1)
        review_feedback = parts[0].strip()
        previous_output = parts[1].strip()

    # Extract velocity — prefer net velocity (after capacity deductions) when available.
    # Falls back to gross velocity, then to default calculation.
    # See docs: "Scrum Standards" — capacity planning
    team_size = state.get("team_size", 1)
    original_team_size = team_size  # Save before potential team override
    gross_velocity = state.get("velocity_per_sprint", team_size * _VELOCITY_PER_ENGINEER)
    velocity = state.get("net_velocity_per_sprint") or gross_velocity
    target_sprints = state.get("target_sprints", analysis.target_sprints)

    # Read the starting sprint number set by sprint_selector.
    # Positive values mean the user chose a real sprint number (e.g. 105).
    # Negative or zero means no Jira / no selection — use generic numbering.
    raw_start = state.get("starting_sprint_number", 0)
    starting_sprint_number = raw_start if raw_start > 0 else 0

    # Read the raw Q10 answer (e.g. "3–5 sprints (~1 quarter)") so the prompt
    # can show the user's chosen range rather than just the parsed upper bound.
    # This gives the LLM context like "aim for 3–5 sprints" instead of just "5".
    target_sprints_raw = ""
    qs = state.get("questionnaire")
    if isinstance(qs, QuestionnaireState):
        target_sprints_raw = qs.answers.get(10, "")

    # ── Capacity check ────────────────────────────────────────────────
    # See docs: "Guardrails" — human-in-the-loop pattern
    #
    # Before calling the LLM, verify that the user's sprint target can
    # actually hold all the story points. If total points exceed
    # velocity × target, the scope doesn't fit and the user needs to
    # know. We treat the sprint target as a deadline — the source of
    # truth — and warn rather than silently exceeding it.
    #
    # Encoding (same pattern as sprint_selector):
    #   0       → not yet checked
    #   < -1    → warning pending; abs(value) = recommended sprint count
    #   -1      → user rejected recommendation; proceed with original
    #   > 0     → user accepted; override target with this value
    capacity_override = state.get("capacity_override_target", 0)
    sprint_caps = state.get("sprint_capacities", [])

    # enforce_target is set when the user explicitly rejected the capacity
    # recommendation — tells the prompt to treat the target as a hard deadline.
    enforce_target = False

    if capacity_override > 0:
        # User accepted the recommendation — use it as the new target.
        target_sprints = capacity_override
        # Update the raw text so the prompt reflects the accepted target.
        target_sprints_raw = f"{capacity_override} sprints (accepted recommendation)"
    elif capacity_override == -1 and state.get("_capacity_team_override", 0) > 0:
        # User chose to increase team size — recalculate velocity to fit scope
        # in the original sprint count without using enforce_target.
        # See docs: "Guardrails" — human-in-the-loop pattern
        new_team = state["_capacity_team_override"]
        velocity_per_engineer = gross_velocity // team_size if team_size > 0 else gross_velocity
        gross_velocity = velocity_per_engineer * new_team
        team_size = new_team
        # Extract holiday/PTO data from existing sprint_caps before recomputing
        old_caps = sprint_caps
        holidays_by_sprint: dict[int, list[dict]] = {}
        leave_by_sprint: dict[int, list[dict]] | None = None
        if old_caps:
            holidays_by_sprint = {
                sc["sprint_index"]: [{"name": n} for n in sc.get("bank_holiday_names", [])] for sc in old_caps
            }
            leave_by_sprint = {sc["sprint_index"]: sc.get("pto_entries", []) for sc in old_caps}
        # Recompute sprint capacities from scratch with the new team size —
        # bank holidays affect ALL engineers (team_size × days), and other
        # deductions also scale with team size, so a simple multiplier won't work.
        sprint_caps = _compute_per_sprint_velocities(
            team_size=new_team,
            velocity_per_sprint=gross_velocity,
            sprint_length_weeks=state.get("sprint_length_weeks", analysis.sprint_length_weeks),
            target_sprints=target_sprints,
            holidays_by_sprint=holidays_by_sprint,
            planned_leave_days=state.get("capacity_planned_leave_days", 0),
            unplanned_leave_pct=state.get("capacity_unplanned_leave_pct", 0),
            onboarding_engineer_sprints=state.get("capacity_onboarding_engineer_sprints", 0),
            ktlo_engineers=state.get("capacity_ktlo_engineers", 0),
            discovery_pct=state.get("capacity_discovery_pct", 5),
            leave_by_sprint=leave_by_sprint,
        )
        # Use the minimum per-sprint velocity as the net velocity
        velocity = min(sc["net_velocity"] for sc in sprint_caps) if sprint_caps else gross_velocity
    elif capacity_override == -1:
        # User rejected — proceed with the original target_sprints but enforce it.
        enforce_target = True
    elif (
        capacity_override == 0
        and target_sprints > 0
        and velocity > 0
        and not review_mode
        and not _is_small_project_mode(state.get("_intake_mode"))
    ):
        # First time through — check if scope fits in the target.
        # Small-project mode skips this overflow advisory — it targets 1-2 sprints
        # of tiny scope, so the "you need more sprints/team" warning is just noise.
        # Use per-sprint velocities for a more accurate capacity check.
        total_points = sum(s.story_points for s in stories)
        if sprint_caps:
            total_capacity = sum(sc["net_velocity"] for sc in sprint_caps)
            min_sprints = math.ceil(total_points / velocity) if velocity > 0 else target_sprints
        else:
            total_capacity = velocity * target_sprints
            min_sprints = math.ceil(total_points / velocity)
        if total_points > total_capacity:
            # Scope overflow — warn the user and return early (no LLM call yet).
            # Compute alternative: minimum team size to fit scope in original sprints.
            # Velocity scales linearly with team size (velocity_per_engineer × team_size).
            # See docs: "Guardrails" — human-in-the-loop pattern
            velocity_per_engineer = velocity // team_size if team_size > 0 else velocity
            min_team_size = (
                math.ceil(total_points / (velocity_per_engineer * target_sprints))
                if velocity_per_engineer > 0
                else team_size + 1
            )
            # Cap min_team_size to the Jira org team size if available — can't
            # recommend more engineers than actually exist on the Jira board.
            jira_team_size = _parse_jira_team_size(qs) if isinstance(qs, QuestionnaireState) else None
            if jira_team_size and min_team_size > jira_team_size:
                min_team_size = jira_team_size
            target_label = target_sprints_raw or f"{target_sprints} sprints"
            warning = (
                f"Your stories total **{total_points} story points**. "
                f"At **{velocity} points/sprint** velocity, you need at least "
                f"**{min_sprints} sprints** — but your target is **{target_label}**.\n\n"
                f"You can extend the timeline, increase team capacity, or keep as-is."
            )
            return {
                "capacity_override_target": -(min_sprints),
                "_original_target_sprints": target_sprints,
                "_recommended_team_size": min_team_size,
                "messages": [AIMessage(content=warning)],
            }

    # Format stories into a compact text block for the prompt.
    # Unlike the task_decomposer prompt, we omit ACs — the sprint planner
    # only needs points, priority, and discipline for capacity allocation.
    stories_block = _format_stories_for_sprint_planner(stories, features)

    # Load team calibration for velocity-aware sprint planning.
    # See docs: "Scrum Standards" — team learning, self-calibrating estimates
    if _wants_dep(state, "analysis"):
        team_profile = _load_team_profile(state.get("analysis_profile_id", ""))
        team_calibration_text = _format_team_calibration(
            team_profile, examples=_load_team_examples(state.get("analysis_profile_id", ""))
        )
    else:
        team_profile, team_calibration_text = None, ""

    prompt = get_sprint_planner_prompt(
        project_name=analysis.project_name,
        project_description=analysis.project_description,
        velocity=velocity,
        target_sprints=target_sprints,
        stories_block=stories_block,
        target_sprints_raw=target_sprints_raw,
        starting_sprint_number=starting_sprint_number,
        enforce_target=enforce_target,
        sprint_capacities=sprint_caps or None,
        team_override_from=original_team_size if state.get("_capacity_team_override", 0) > 0 else None,
        team_calibration=team_calibration_text,
        ceremony_history=state.get("_ceremony_history", "") or "",
        performance_context=state.get("_performance_context", "") or _gather_performance_summary(_state_scope(state)),
        # Planner only. The analyzer's output feeds feature generation, and a
        # feature invented from an incident is a story nobody asked for.
        production_context=_gather_ops_summary(),
        review_feedback=review_feedback if review_mode else None,
        review_mode=review_mode,
        previous_output=previous_output,
    )

    # Screenshots attached to review-edit feedback (Ctrl+V) — review passes only.
    review_images = list(state.get("review_feedback_images") or []) if review_mode else []

    try:
        # Single LLM call with low temperature for deterministic JSON output.
        # See docs: "Agentic Blueprint Reference" — using the LLM outside the main graph
        response = _invoke_json(prompt, image_paths=review_images)
        sprints = _parse_sprints_response(response.content, stories, velocity, starting_sprint_number)
    except Exception as exc:
        if _should_reraise_llm_error(exc):
            raise
        # LLM call failed entirely — use greedy bin-packing fallback.
        logger.warning("LLM call failed in sprint_planner, using fallback", exc_info=True)
        sprints = _build_fallback_sprints(stories, velocity, starting_sprint_number)

    # When the user enforced a hard target (rejected the capacity recommendation)
    # OR chose to increase team size (which keeps the original sprint count),
    # merge sprints down to the target count. The LLM often ignores the constraint
    # and produces more sprints than requested — this is the programmatic safety net.
    team_override_active = state.get("_capacity_team_override", 0) > 0
    should_merge = (enforce_target or team_override_active) and target_sprints > 0
    if should_merge and len(sprints) > target_sprints:
        sprints = _merge_sprints_to_target(sprints, target_sprints, stories, starting_sprint_number)

    # The architecture spike must open the plan — prompt rule 4 asks for it,
    # this guarantees it (deterministic post-parse, like the capacity fixes).
    sprints = _pin_spike_to_first_sprint(sprints, stories)

    if state.get("sprint_target_mode") == "backlog" and sprints:
        # The plan's single bucket IS the backlog, not a numbered sprint —
        # naming it "Sprint 1" would promise a sprint the sync never creates.
        sprints = [dataclasses.replace(sprints[0], name="Backlog"), *sprints[1:]]

    # Format the sprints for display
    display = _format_sprints(sprints, stories, features, analysis.project_name, velocity)

    result: dict = {
        "sprints": sprints,
        "messages": [AIMessage(content=display)],
        "pending_review": "sprint_planner",
    }

    # When team size was overridden, persist the updated velocity and team size
    # back to state so TUI renderers and exports see the correct values.
    if state.get("_capacity_team_override", 0) > 0:
        result["velocity_per_sprint"] = velocity
        result["net_velocity_per_sprint"] = velocity
        result["team_size"] = team_size
        if sprint_caps:
            result["sprint_capacities"] = sprint_caps

    return result
