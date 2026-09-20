"""LangGraph graph factory — wires nodes and edges into a compiled graph.

# See docs: "Agentic Blueprint Reference" — Core Graph Setup, Wiring
# See docs: "The ReAct Loop" — Thought → Action → Observation pattern

This module contains the factory function that assembles the agent graph.
It is kept separate from node functions (nodes.py) and state (state.py) so that
the graph topology is defined in one place, while nodes remain independently testable.

Why a factory function instead of a module-level constant?
- Matches the project pattern: get_llm(), get_system_prompt() are also factories.
- Accepts parameters (tools, checkpointer) so the graph can be configured
  differently in tests vs. production without changing this file.
- A module-level constant would be created at import time, making it hard to
  inject different tools or checkpointers.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from yeaboi.agent.nodes import (
    feature_generator,
    feature_skip,
    human_review,
    make_call_model,
    make_tool_node,
    project_analyzer,
    project_intake,
    route_entry,
    should_continue,
    sprint_planner,
    story_writer,
    task_decomposer,
)
from yeaboi.agent.state import ScrumState

logger = logging.getLogger(__name__)


def create_graph(
    tools: Sequence[BaseTool] = (),
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build and compile the Scrum Agent LangGraph graph.

    # See docs: "Agentic Blueprint Reference" — this is the core graph setup
    #
    # The graph implements the intake questionnaire + ReAct loop:
    #
    #   START ──route_entry?──→ project_intake → END
    #              │
    #              ├──→ project_analyzer → END
    #              │
    #              ├──→ feature_generator → END
    #              │
    #              ├──→ story_writer → END
    #              │
    #              ├──→ task_decomposer → END
    #              │
    #              └──→ agent ──should_continue?──→ END
    #                     ▲          │
    #                     │       "tools"
    #                     │          ▼
    #                     └─────── tools
    #
    # How the ReAct loop works:
    # 1. The "agent" node (call_model) asks the LLM to think about the user's input.
    # 2. should_continue checks if the LLM wants to use a tool (tool_calls present).
    # 3. If yes → route to "tools" node, which executes the tool and loops back to "agent".
    # 4. If no  → route to END, returning the LLM's response to the user.
    # The loop repeats until the LLM decides it's done (no more tool_calls).
    #
    # Why auto-load tools when none are passed?
    # Callers (e.g. the REPL) can call create_graph() with no arguments and get
    # a fully-wired graph with all available tools. Tests that want an isolated
    # graph can pass tools=[] explicitly to opt out of auto-loading.
    # See docs: "Tools" — tool registration pattern

    Args:
        tools: Sequence of LangChain tools to bind to the agent. When empty (default),
            auto-loads all available tools via get_tools(). Pass tools=[] explicitly
            to skip tool loading (useful in tests).
        checkpointer: Optional LangGraph checkpointer for session persistence.
            Pass MemorySaver() (Phase 7) to enable conversation memory across invocations.

    Returns:
        A compiled LangGraph StateGraph ready to be invoked or streamed.
    """
    # Auto-load tools when none are provided. The empty-tuple default lets tests
    # pass tools=[] to get a no-tools graph without triggering auto-loading.
    logger.debug("create_graph called (tools=%d, checkpointer=%s)", len(tools), type(checkpointer).__name__)
    if not tools:
        from yeaboi.tools import get_tools

        tools = get_tools()
        logger.debug("Auto-loaded %d tool(s)", len(tools))
    # StateGraph is the core LangGraph class. It takes a state schema (TypedDict)
    # and manages how state flows between nodes. Each node receives the full state
    # and returns a partial update dict that gets merged back using the reducers
    # defined on the schema (e.g. add_messages for the messages list).
    # See docs: "Agentic Blueprint Reference" — Core Graph Setup
    graph = StateGraph(ScrumState)

    # ── Register nodes ──────────────────────────────────────────────
    # add_node(name, function) registers a node in the graph.
    # The name is a string label used in edges; the function is called with
    # the current state and must return a partial state update dict.

    # "project_intake" node — deterministic questionnaire (no LLM call).
    # Asks one intake question per invocation. When all 26 questions are
    # answered, sets questionnaire.completed = True so route_entry sends
    # future invocations to the "project_analyzer" node.
    # See docs: "Scrum Standards" — questionnaire phases
    graph.add_node("project_intake", project_intake)

    # "project_analyzer" node — synthesizes confirmed intake answers into
    # a structured ProjectAnalysis. Single LLM call with JSON-schema prompt.
    # After this runs, route_entry sees project_analysis populated and
    # routes future invocations to the "feature_generator" node.
    # See docs: "Architecture" — project_analyzer sits between intake and agent
    graph.add_node("project_analyzer", project_analyzer)

    # "feature_skip" node — creates a single feature for small projects.
    # When the analyzer determines skip_features=True, this node creates one feature
    # named after the project instead of running the full feature generator.
    # See docs: "Scrum Standards" — feature generation
    graph.add_node("feature_skip", feature_skip)

    # "feature_generator" node — decomposes ProjectAnalysis into 3-6 features.
    # Single LLM call with JSON-schema prompt. After this runs, route_entry
    # sees features populated and routes future invocations to the "story_writer" node.
    # See docs: "Architecture" — feature_generator sits between analyzer and story_writer
    graph.add_node("feature_generator", feature_generator)

    # "story_writer" node — decomposes features into 2-5 user stories each.
    # Single LLM call with JSON-schema prompt including nested acceptance criteria.
    # After this runs, route_entry sees stories populated and routes future
    # invocations to the "task_decomposer" node.
    # See docs: "Architecture" — story_writer sits between feature_generator and task_decomposer
    graph.add_node("story_writer", story_writer)

    # "task_decomposer" node — breaks user stories into 2-5 implementation tasks.
    # Single LLM call with JSON-schema prompt. After this runs, route_entry
    # sees tasks populated and routes future invocations to the "sprint_planner" node.
    # See docs: "Architecture" — task_decomposer sits between story_writer and sprint_planner
    graph.add_node("task_decomposer", task_decomposer)

    # "sprint_planner" node — allocates stories to sprints based on velocity.
    # Hybrid LLM + deterministic approach: LLM writes sprint goals and allocates
    # stories, then a validator corrects capacity math and handles orphans.
    # Uses starting_sprint_number (from sprint_selector) for real sprint names.
    # After this runs, route_entry sees sprints populated and routes to the "agent" node.
    # See docs: "Architecture" — sprint_planner sits between task_decomposer and agent
    graph.add_node("sprint_planner", sprint_planner)

    # "agent" node — the LLM reasoning step (Thought in ReAct).
    # make_call_model(tools) returns a node function with the tools bound to the LLM
    # via bind_tools(). Without bind_tools(), the LLM has no awareness of available
    # tools and can never generate tool_calls — the ReAct loop would be inert.
    # See docs: "Agentic Blueprint Reference" — bind_tools
    graph.add_node("agent", make_call_model(list(tools)))

    # "tools" node — the tool execution step (Action in ReAct).
    # ToolNode is a LangGraph prebuilt that automatically executes whatever
    # tool_calls the LLM requested. It reads the tool names from the last
    # AIMessage's tool_calls, runs the matching tool functions, and returns
    # ToolMessages with the results. These get appended to state["messages"]
    # so the LLM can see the tool output on the next loop iteration.
    #
    # handle_tool_errors=True: when a tool raises an exception, ToolNode
    # catches it and returns an error ToolMessage instead of crashing the
    # graph. The LLM receives the error text on the next turn and can
    # respond gracefully (e.g. apologise, suggest an alternative).
    # Without this, any tool exception propagates up and terminates the
    # REPL session unexpectedly.
    # See docs: "Tools" — tool types and ToolNode
    # See docs: "Guardrails" — graceful degradation on tool failure
    # make_tool_node wraps it in the per-plan integration guard: a tool whose
    # integration the plan did not enable answers with a refusal instead.
    graph.add_node("tools", make_tool_node(list(tools)))

    # "human_review" node — the human-in-the-loop step for high-risk writes.
    # Reached when should_continue detects an external write tool call
    # (Jira, Azure DevOps, Confluence, Notion) — the WRITE rows of the
    # per-tool registry in tools/risk.py.
    # Replaces the tool_calls AIMessage with a plain-text confirmation request,
    # then routes to END so the REPL displays the confirmation to the user.
    # If the user confirms, the next invocation's should_continue detects the
    # confirmation pattern and routes directly to "tools".
    # See docs: "Guardrails" — human-in-the-loop pattern
    graph.add_node("human_review", human_review)

    # ── Wire edges ──────────────────────────────────────────────────
    # START → route_entry → [intake | analyzer | feature_gen | story_writer | task_decomposer | sprint_planner | agent]
    #
    # route_entry is a conditional edge from START. On each invocation it
    # checks questionnaire, analysis, feature, story, task, and sprint state
    # for seven-way routing:
    #   - Questionnaire not completed → "project_intake"
    #   - Questionnaire completed, no analysis → "project_analyzer"
    #   - Analysis done, no features → "feature_generator"
    #   - Features done, no stories → "story_writer"
    #   - Stories done, no tasks → "task_decomposer"
    #   - Tasks done, no sprints → "sprint_planner"
    #   - Sprints populated → "agent" (ReAct loop takes over)
    #
    # See docs: "Agentic Blueprint Reference" — conditional edges
    graph.add_conditional_edges(
        START,
        route_entry,
        [
            "project_intake",
            "project_analyzer",
            "feature_skip",
            "feature_generator",
            "story_writer",
            "task_decomposer",
            "sprint_planner",
            "agent",
        ],
    )

    # project_intake always ends after asking one question. The REPL
    # collects the user's answer and calls graph.invoke() again, which
    # re-enters via route_entry.
    graph.add_edge("project_intake", END)

    # project_analyzer ends after synthesizing the analysis. The REPL
    # displays the formatted analysis and the user types anything to
    # continue. On the next invocation, route_entry routes to feature_generator.
    graph.add_edge("project_analyzer", END)

    # feature_skip ends after creating the single feature. The REPL loop re-invokes
    # the graph; route_entry sees features populated and routes to story_writer.
    graph.add_edge("feature_skip", END)

    # feature_generator ends after decomposing the project into features. The REPL
    # displays the feature list and the user types anything to continue. On the
    # next invocation, route_entry routes to the story_writer node.
    graph.add_edge("feature_generator", END)

    # story_writer ends after decomposing features into user stories. The REPL
    # displays the story list and the user types anything to continue. On the
    # next invocation, route_entry routes to the task_decomposer node.
    graph.add_edge("story_writer", END)

    # task_decomposer ends after breaking stories into implementation tasks.
    # The REPL displays the task list and the user types anything to continue.
    # On the next invocation, route_entry routes to the sprint_planner node.
    graph.add_edge("task_decomposer", END)

    # sprint_planner ends after allocating stories to sprints. The REPL
    # displays the sprint plan and the user types anything to continue.
    # On the next invocation, route_entry routes to the agent node.
    graph.add_edge("sprint_planner", END)

    # add_conditional_edges creates a branching point after a node.
    # After "agent" runs, LangGraph calls should_continue(state) to decide
    # where to go next:
    #   - no tool_calls   → END            (LLM is done)
    #   - low-risk tools  → "tools"        (auto-execute)
    #   - high-risk tools → "human_review" (pause for user confirmation)
    # The third argument lists all possible destinations for topology validation.
    # See docs: "Agentic Blueprint Reference" — conditional edges
    # See docs: "Guardrails" — human-in-the-loop pattern
    graph.add_conditional_edges("agent", should_continue, ["tools", "human_review", END])

    # After the tools node runs, always loop back to the agent node.
    # This closes the ReAct loop: agent → tools → agent → tools → ... → END.
    # The agent gets to see the tool results (ToolMessages) and can either
    # make another tool call or produce a final response.
    graph.add_edge("tools", "agent")

    # human_review routes to END after replacing the tool_calls message with a
    # confirmation request. The user responds in the REPL; if they confirm, the
    # next call_model invocation re-generates the tool call, and should_continue
    # detects the confirmation pattern and routes to "tools".
    graph.add_edge("human_review", END)

    # ── Compile ─────────────────────────────────────────────────────
    # compile() validates the graph topology (all edges point to existing nodes,
    # no orphan nodes, etc.) and returns a CompiledStateGraph. The compiled graph
    # is what you actually invoke — it has .invoke() and .stream() methods.
    # The optional checkpointer parameter enables state persistence across
    # invocations (Phase 7: Memory & Session Persistence).
    # See docs: "Memory & State" — MemorySaver, thread_id
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info(
        "Graph compiled: %d node(s), %d tool(s)",
        len(graph.nodes),
        len(tools),
    )
    return compiled
