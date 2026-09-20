"""One planning conversation, as a decision layer and an event stream.

# See docs: "Architecture" — the four layers; "The ReAct Loop"
# See docs: "Guardrails" — human-in-the-loop review gates

Everything here answers "what does this conversation need next?" without
knowing whether the answer becomes a Rich panel, an NDJSON line or a React
component — the TUI driver (``ui/session/chat/_driver.py``) renders these
events today and the desktop's chat route will stream the same ones.

:class:`ChatSession` owns the graph state and runs one turn at a time,
emitting typed events as they happen; everything above it is a pure function
of that state. Nothing in this module renders, reads keys or touches the
duck: those stay with the caller that owns a screen.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, is_dataclass

from langchain_core.messages import AIMessage, HumanMessage

from yeaboi.agent.state import TOTAL_QUESTIONS, QuestionnaireState, ReviewDecision
from yeaboi.agent.streaming import predict_next_node, stream_chat_turn

logger = logging.getLogger(__name__)

# The generation nodes that produce an artifact and park on a review gate.
PIPELINE_NODES = (
    "project_analyzer",
    "feature_skip",
    "feature_generator",
    "story_writer",
    "task_decomposer",
    "sprint_planner",
)

# What counts as "yes" at any review gate.
ACCEPT_WORDS = frozenset({"accept", "a", "ok", "yes", "looks good", "lgtm", "continue"})

# The line that stands in for a node's markdown summary once the card has
# rendered it. Module constants because the live turn and the resume replay
# both write them, and the two must not drift.
CONFIRM_VERDICT_PROMPT = (
    "Here's everything I've got. Pick an option below — or type **accept**, **edit N**, or just tell me what's off."
)

PRIOR_ART_VERDICT_PROMPT = (
    "You already own these. **Space** picks the relevant ones, **←/→** browses the details, "
    "**X** hides a repo forever, **Enter** confirms — or type e.g. **1 3**, **all**, or **none**."
)

# The epic gate parks under this pending_review value. It is not a graph node:
# the reformat runs between invokes, and the verdict is read like any other
# review so every surface answers it the same way.
EPIC_REVIEW_NODE = "epic_review"

EPIC_VERDICT_PROMPT = (
    "Here's the project epic. Reply **accept** to break it into epics and stories, or **edit** + your changes."
)

# AwaitConfirm.kind for the ReAct node's write-tool gate (agent/nodes.py human_review).
TOOL_WRITE_KIND = "tool_write"

# The build's checklist, in order. repl/_ui._PIPELINE_STEPS must stay equal.
PIPELINE_STEPS = (
    "project_analyzer",
    EPIC_REVIEW_NODE,
    "feature_generator",
    "story_writer",
    "task_decomposer",
    "sprint_planner",
)

# The plan's sections as every surface names them, in build order.
SECTION_KINDS = ("intake", "analysis", "epic", "features", "stories", "tasks", "sprints")

# Which artifact card a parked review gate shows.
REVIEW_ARTIFACT_KINDS = {
    "project_analyzer": "analysis",
    EPIC_REVIEW_NODE: "epic",
    "feature_generator": "features",
    "feature_skip": "features",
    "story_writer": "stories",
    "task_decomposer": "tasks",
    "sprint_planner": "sprints",
}

# Which state key proves a pipeline step has produced its artifact.
PROGRESS_DONE_KEYS = {
    "project_analyzer": "project_analysis",
    "epic_review": "_epic_reviewed",
    "feature_generator": "features",
    "story_writer": "stories",
    "task_decomposer": "tasks",
    "sprint_planner": "sprints",
}

# The review-state fields an accepted gate clears before the next invoke.
_REVIEW_STATE_KEYS = (
    "pending_review",
    "last_review_decision",
    "last_review_feedback",
    "review_feedback_images",
    "_small_project_oversized",
)

# Dry-run has no graph to predict against — the artifact keys carry the order.
_DRY_NODE_ORDER = (
    ("project_analysis", "project_analyzer"),
    ("features", "feature_generator"),
    ("stories", "story_writer"),
    ("tasks", "task_decomposer"),
    ("sprints", "sprint_planner"),
)


# --------------------------------------------------------------------- events


@dataclass(frozen=True)
class Token:
    """One streamed chunk of the reply being written."""

    text: str


@dataclass(frozen=True)
class Done:
    """The turn finished — state has moved on."""


@dataclass(frozen=True)
class Assistant:
    """A plain assistant bubble — the node's own words."""

    text: str


@dataclass(frozen=True)
class UserSaid:
    """Something the user said — only ever produced by the replay."""

    text: str


@dataclass(frozen=True)
class AskQuestion:
    """An intake question, decorated for chat."""

    text: str
    number: int


@dataclass(frozen=True)
class ShowArtifact:
    """An artifact card, rendered from state rather than from the reply."""

    kind: str


@dataclass(frozen=True)
class AwaitConfirm:
    """An intake-side gate: a card plus the one line that asks for a verdict."""

    kind: str
    prompt: str


@dataclass(frozen=True)
class AwaitReview:
    """A pipeline review gate: the node, its card, and the verdict grammar."""

    node: str
    kind: str
    prompt: str


@dataclass(frozen=True)
class AwaitChoice:
    """A capacity or spike question: the prompt and the option keys that answer it."""

    kind: str
    prompt: str
    options: tuple[tuple[str, str], ...]  # (key, label); the recommended option first


@dataclass(frozen=True)
class Progress:
    """A build step starting or finishing."""

    node: str
    step: int
    total: int
    status: str  # "running" | "done"


@dataclass(frozen=True)
class SectionChanged:
    """A plan section changed — whoever shows it refetches it."""

    kind: str
    status: str  # "empty" | "generating" | "awaiting_review" | "accepted"
    version: int


@dataclass(frozen=True)
class Notice:
    """A dim system line: something that happened, not something said."""

    text: str


@dataclass(frozen=True)
class Action:
    """Something the caller's own surface must do (a tracker sync)."""

    name: str
    detail: str = ""


ReplyEvent = Assistant | AskQuestion | AwaitConfirm
ChatEvent = (
    Token | ReplyEvent | ShowArtifact | AwaitReview | AwaitChoice | Progress | SectionChanged | Notice | Action | Done
)
EventSink = Callable[["ChatEvent"], None]


# ----------------------------------------------------------------- predicates


def questionnaire(state: dict) -> QuestionnaireState | None:
    qs = state.get("questionnaire")
    return qs if isinstance(qs, QuestionnaireState) else None


def next_node(state: dict, *, dry_run: bool = False) -> str:
    """The graph node that will run next (or its dry-run stand-in)."""
    if not dry_run:
        return predict_next_node(state)
    for key, node in _DRY_NODE_ORDER:
        if not state.get(key):
            return node
    return "agent"


def stage_of(state: dict, *, dry_run: bool = False) -> str:
    """What the conversation needs next: the one predicate every caller routes on.

    Resume, the PTO sub-loop and mid-chat size switches fall out of state
    inspection rather than control flow — which is why this is a function of
    state alone.
    """
    if state.get("capacity_override_target", 0) < -1 and not dry_run:
        return "capacity"
    if state.get("_spike_prompt") and not state.get("spike_choice") and not dry_run:
        return "spike"
    pending = state.get("pending_review")
    if pending == "project_intake":
        return "intake"  # confirmation gate — the node consumes the reply
    if pending in PIPELINE_NODES or pending == EPIC_REVIEW_NODE:
        return "review"
    node = next_node(state, dry_run=dry_run)
    if node == "project_intake":
        return "intake"
    if node in PIPELINE_NODES:
        if (
            node in ("feature_generator", "feature_skip")
            and state.get("project_analysis")
            and not state.get("_epic_reviewed")
        ):
            return "epic"
        return "pipeline"
    return "chat"


def at_intake_summary(state: dict) -> bool:
    """True when the newest reply is the intake summary awaiting a verdict.

    One predicate for both paths that render it — the live turn and the resume
    replay — or reopening a session parked on the gate resurrects the markdown
    wall the card replaced. The sub-states are excluded because each one
    re-asks something instead of re-showing the summary: a PTO prompt, a
    velocity prompt, the prior-art verdict, or the re-ask of the answer being
    edited.
    """
    qs = questionnaire(state)
    return (
        qs is not None
        and qs.awaiting_confirmation
        and not qs._awaiting_leave_input
        and not qs._awaiting_velocity_input
        and not at_prior_art(state)
        and qs.editing_question is None
        and qs.current_question > TOTAL_QUESTIONS
    )


def at_prior_art(state: dict) -> bool:
    """True while the prior-art sub-loop owns the turn.

    The summary card's condition is defined as "not this", so the two can never
    drift into both claiming the same turn.
    """
    qs = questionnaire(state)
    return qs is not None and getattr(qs, "_prior_art_stage", "") in ("ask", "reason", "empty")


def newest_reply(state: dict) -> str:
    messages = state.get("messages", [])
    return messages[-1].content if messages and isinstance(messages[-1], AIMessage) else ""


# ------------------------------------------------------------- reply routing


def reply_event(state: dict) -> ReplyEvent | None:
    """Route the newest assistant reply to what should be shown for it.

    None means there is nothing to show (no reply on this turn).
    """
    reply = newest_reply(state)
    if not reply:
        return None

    qs = questionnaire(state)
    # Intake confirmation summary → card + short prompt instead of the node's
    # markdown wall (the card is the same data, rendered properly).
    if at_intake_summary(state):
        return AwaitConfirm(kind="intake_summary", prompt=CONFIRM_VERDICT_PROMPT)

    prior_art_stage = getattr(qs, "_prior_art_stage", "") if qs is not None else ""
    if prior_art_stage == "ask":
        from yeaboi.agent.nodes import _PRIOR_ART_GRAMMAR_HINT

        if reply.strip() == _PRIOR_ART_GRAMMAR_HINT:
            # The node rejected a typed answer. Swallowing this and re-posting
            # the same card would read as a no-op — the one turn where the
            # node's own words must go out as prose.
            return Assistant(reply)
        return AwaitConfirm(kind="prior_art", prompt=PRIOR_ART_VERDICT_PROMPT)

    # Nothing found — the node's message is already the whole statement. An
    # explicit branch rather than falling through: the tail decorates replies
    # as intake questions, and this one is not a question.
    if prior_art_stage == "empty":
        return Assistant(reply)

    if qs is not None and not qs.completed:
        from yeaboi.prompts.intake import decorate_question_for_chat

        mode = qs.intake_mode or state.get("_intake_mode") or None
        return AskQuestion(
            text=decorate_question_for_chat(qs.current_question, reply, intake_mode=mode),
            number=qs.current_question,
        )
    from yeaboi.agent.nodes import TOOL_CONFIRM_PREFIX

    if reply.startswith(TOOL_CONFIRM_PREFIX):
        # The ReAct node's write gate: "yes" re-runs the tool call, "no" drops it.
        return AwaitConfirm(kind=TOOL_WRITE_KIND, prompt=reply)
    return Assistant(reply)


def review_gate(state: dict, node: str) -> AwaitReview:
    """The card and verdict grammar for a parked pipeline review."""
    if node == EPIC_REVIEW_NODE:
        return AwaitReview(node=node, kind="epic", prompt=EPIC_VERDICT_PROMPT)
    prompts = [
        "Reply **accept** to continue",
        "**edit** + your changes to refine",
        "/export to save",
        "/finish auto-accepts the rest",
    ]
    if node == "project_analyzer" and state.get("_small_project_oversized"):
        prompts.insert(1, "**switch to large** for a fuller plan (this looks bigger than a small project)")
    return AwaitReview(
        node=node,
        kind=REVIEW_ARTIFACT_KINDS.get(node, "analysis"),
        prompt=" · ".join(prompts) + ".",
    )


# ------------------------------------------------------------- resume replay


@dataclass(frozen=True)
class ReplayPlan:
    """Which replayed message indices become cards instead of prose.

    -1 means "no message qualifies". The two are mutually exclusive —
    :func:`at_intake_summary` excludes prior art.
    """

    summary_at: int
    prior_art_at: int


def replay_plan(state: dict) -> ReplayPlan:
    """Decide which stored replies a resumed session renders as cards.

    A live turn renders the intake summary and the prior-art batch as cards
    (:func:`reply_event`), so replaying their markdown would hand a resumed
    session the wall of text the cards exist to replace.
    """
    messages = state.get("messages", [])

    def newest(predicate) -> int:
        return max(
            (
                i
                for i, m in enumerate(messages)
                if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content and predicate(m.content)
            ),
            default=-1,
        )

    summary_at = newest(lambda _content: True) if at_intake_summary(state) else -1

    prior_art_at = -1
    qs = questionnaire(state)
    if qs is not None and getattr(qs, "_prior_art_stage", "") == "ask" and qs._prior_art_candidates:
        from yeaboi.agent.nodes import _PRIOR_ART_GRAMMAR_HINT

        # The card belongs to the newest reply that is NOT the grammar hint: a
        # rejected typed answer leaves [..., AI(batch prompt), Human, AI(hint)],
        # and pinning to the newest reply outright would card the one-liner
        # while the batch-prompt wall above it replayed raw.
        prior_art_at = newest(lambda content: content.strip() != _PRIOR_ART_GRAMMAR_HINT)
    return ReplayPlan(summary_at=summary_at, prior_art_at=prior_art_at)


# Which artifact card each stored artifact resumes as, in pipeline order.
ARTIFACT_STATE_KEYS = (
    ("analysis", "project_analysis"),
    ("epic", "_epic_reviewed"),
    ("features", "features"),
    ("stories", "stories"),
    ("tasks", "tasks"),
    ("sprints", "sprints"),
)

ReplayItem = UserSaid | Assistant | AwaitConfirm | ShowArtifact | AwaitReview | AwaitChoice


def parked_gate(state: dict) -> AwaitReview | AwaitChoice | None:
    """The gate a conversation is waiting on, or None when it is not parked."""
    stage = stage_of(state)
    if stage == "capacity":
        return capacity_choices(state)
    if stage == "spike":
        return spike_choices(state)
    if stage == "review":
        return review_gate(state, state.get("pending_review", ""))
    return None


def is_synthetic(message) -> bool:
    """True for a turn the driver injected (an advance's "continue")."""
    return bool(getattr(message, "additional_kwargs", {}).get("synthetic"))


def replay(state: dict) -> list[ReplayItem]:
    """Rebuild the whole conversation from stored state, in order.

    The greeting exchange lives in ``_chat_preamble`` rather than in messages
    (project_intake reads ``messages[0]`` as the description), so it leads;
    then the messages, with the gate replies routed through
    :func:`replay_plan`; then the artifacts the session already holds. A
    finished plan ends on its recap card — silently, because the celebration
    fired when the build completed and must not replay on every resume.
    """
    items: list[ReplayItem] = []
    for entry in state.get("_chat_preamble") or []:
        text = entry.get("text", "")
        items.append(UserSaid(text) if entry.get("role") == "user" else Assistant(text))

    plan = replay_plan(state)
    for i, message in enumerate(state.get("messages", [])):
        if not isinstance(message.content, str):
            continue
        if isinstance(message, HumanMessage):
            if not is_synthetic(message):
                items.append(UserSaid(message.content))
        elif isinstance(message, AIMessage) and message.content:
            if i == plan.summary_at:
                items.append(AwaitConfirm(kind="intake_summary", prompt=CONFIRM_VERDICT_PROMPT))
            elif i == plan.prior_art_at:
                items.append(AwaitConfirm(kind="prior_art", prompt=PRIOR_ART_VERDICT_PROMPT))
            else:
                items.append(Assistant(message.content))

    for kind, key in ARTIFACT_STATE_KEYS:
        if state.get(key) and (kind != "epic" or state.get("project_analysis")):
            items.append(ShowArtifact(kind))
    if state.get("sprints"):
        items.append(ShowArtifact("recap"))
    # A reopened window redraws the verdict buttons from this, not from prose.
    gate = parked_gate(state)
    if gate is not None:
        items.append(gate)
    return items


# ------------------------------------------------------------ review verdicts


@dataclass(frozen=True)
class Accept:
    """The gate was accepted — the pipeline moves on."""


@dataclass(frozen=True)
class SwitchSize:
    """ "switch to large" at the analysis gate."""

    target: str


@dataclass(frozen=True)
class TrackerSync:
    """A sync request. An empty tracker means "whichever is configured"."""

    tracker: str


@dataclass(frozen=True)
class EditFeedback:
    """Anything else — refine by chatting."""

    text: str


ReviewVerdict = Accept | SwitchSize | TrackerSync | EditFeedback


def review_verdict(text: str, pending: str) -> ReviewVerdict:
    """Classify a reply typed at a pipeline review gate."""
    lowered = text.lower().strip()
    if lowered in ACCEPT_WORDS:
        return Accept()
    if lowered in ("switch to large", "switch") and pending == "project_analyzer":
        return SwitchSize(target="smart")
    if lowered in ("sync jira", "sync azure", "sync azure devops", "sync"):
        tracker = "azdevops" if "azure" in lowered else ("jira" if "jira" in lowered else "")
        return TrackerSync(tracker=tracker)
    return EditFeedback(text=text.removeprefix("edit").removeprefix("regenerate").strip() or text)


def clear_review_state(state: dict) -> None:
    """Drop the review bookkeeping so the next invoke runs the next stage."""
    for key in _REVIEW_STATE_KEYS:
        state.pop(key, None)


def accept_review(state: dict) -> str:
    """Accept the parked gate. Returns the section kind that was accepted."""
    kind = REVIEW_ARTIFACT_KINDS.get(state.get("pending_review", ""), "")
    clear_review_state(state)
    return kind


def apply_edit_feedback(state: dict, pending: str, feedback: str, images: list[str] | None = None) -> str:
    """Record edit feedback on a parked gate so the next invoke regenerates from it.

    The previous output rides along in the feedback (the node reads both), and
    everything downstream is cleared because it was derived from what is being
    redone. The epic gate edits the analysis it was formatted from. Returns
    the section kind the feedback is about.
    """
    from yeaboi.repl._review import _clear_downstream_artifacts, _serialize_artifacts_for_review

    node = "project_analyzer" if pending == EPIC_REVIEW_NODE else pending
    serialized = _serialize_artifacts_for_review(state, node)
    _clear_downstream_artifacts(state, node)
    if node == "project_analyzer":
        # A redone analysis is formatted and reviewed as an epic again.
        state.pop("_epic_reviewed", None)
    state["last_review_decision"] = ReviewDecision.EDIT
    if images:
        state["review_feedback_images"] = list(images)
    state["last_review_feedback"] = f"{feedback}\n\n---PREVIOUS OUTPUT---\n{serialized}" if serialized else feedback
    state.pop("pending_review", None)
    logger.info("Review decision: edit %s (len=%d images=%d)", pending, len(feedback), len(images or []))
    return REVIEW_ARTIFACT_KINDS.get(pending, "")


def capacity_choices(state: dict) -> AwaitChoice:
    """The capacity-overflow question: extend, grow the team, or overload."""
    recommended = abs(state.get("capacity_override_target", 0))
    original_target = state.get("_original_target_sprints", recommended)
    recommended_team = state.get("_recommended_team_size", 0)
    current_team = state.get("team_size", 1)
    prompt = newest_reply(state).replace("**", "")
    options = [("extend", f"Extend to {recommended} sprints")]
    if recommended_team > current_team:
        options.append(("team", f"Keep {original_target} sprints — increase team to {recommended_team} engineers"))
    elif recommended_team > 0:
        prompt += (
            f"\n\nIncrease team is unavailable — your Jira board has "
            f"{current_team} team member(s), which is already the maximum."
        )
    options.append(
        ("overload", f"Keep {original_target} sprints, {current_team} engineer(s) — overload (not recommended)")
    )
    return AwaitChoice(kind="capacity", prompt=prompt, options=tuple(options))


def apply_capacity_choice(state: dict, key: str) -> str:
    """Answer the capacity question. Returns the chosen option's label."""
    choice = capacity_choices(state)
    labels = dict(choice.options)
    if key not in labels:
        raise ValueError(f"unknown capacity choice {key!r} — one of {', '.join(labels)}")
    recommended = abs(state.get("capacity_override_target", 0))
    if key == "team":
        state["capacity_override_target"] = -1
        state["_capacity_team_override"] = state.get("_recommended_team_size", 0)
    elif key == "overload":
        state["capacity_override_target"] = -1
    else:
        state["capacity_override_target"] = recommended
    state["_capacity_warning"] = {"text": choice.prompt, "recommended": recommended}
    logger.info("Capacity overflow: %s", key)
    return labels[key]


def spike_choices(state: dict) -> AwaitChoice:
    """The architecture-spike question, the recommended option first."""
    prompt = state.get("_spike_prompt") or {}
    recommended = prompt.get("recommended", "include")
    chosen = prompt.get("chosen", "the recommended architecture")
    confidence = prompt.get("confidence", "medium")
    add = "Add a validation spike (1-3 days)"
    skip = f"Skip — commit to {chosen}"
    if recommended == "include":
        options = (("include", f"{add} (recommended — confidence {confidence})"), ("skip", skip))
    else:
        options = (("skip", f"{skip} (recommended — confidence high)"), ("include", add))
    return AwaitChoice(kind="spike", prompt=newest_reply(state).replace("**", ""), options=options)


def apply_spike_choice(state: dict, key: str) -> str:
    """Answer the spike question. Returns the chosen option's label."""
    labels = dict(spike_choices(state).options)
    if key not in labels:
        raise ValueError(f"unknown spike choice {key!r} — one of {', '.join(labels)}")
    state["spike_choice"] = key
    state["_spike_prompt"] = {}
    logger.info("Spike question: %s", key)
    return labels[key]


def choice_key(choice: AwaitChoice, text: str) -> str | None:
    """Resolve a typed answer to an option key: the key, its 1-based number, or an accept word for the first."""
    lowered = text.strip().lower()
    keys = [key for key, _label in choice.options]
    if lowered in keys:
        return lowered
    if lowered.isdigit() and 1 <= int(lowered) <= len(keys):
        return keys[int(lowered) - 1]
    if not lowered or lowered in ACCEPT_WORDS:
        return keys[0]
    return None


def pipeline_step(node: str) -> tuple[int, int]:
    """A node's 1-based position in the build checklist (0 when it is not a step)."""
    step_node = "feature_generator" if node == "feature_skip" else node
    total = len(PIPELINE_STEPS)
    return (PIPELINE_STEPS.index(step_node) + 1 if step_node in PIPELINE_STEPS else 0), total


def last_completed_node(state: dict) -> str:
    """The newest build step whose artifact exists, "" before the build starts."""
    done = ""
    for node in PIPELINE_STEPS:
        if state.get(PROGRESS_DONE_KEYS.get(node, "")):
            done = node
    return done


def section_payload(state: dict, kind: str) -> dict:
    """A plan section as plain data, the shape a version snapshot keeps."""

    def plain(value):
        return asdict(value) if is_dataclass(value) else value

    if kind == "intake":
        qs = questionnaire(state)
        if qs is None:
            return {}
        return {
            "answers": {str(n): a for n, a in sorted(qs.answers.items())},
            "sources": {str(n): s for n, s in sorted(qs.answer_sources.items())},
        }
    if kind in ("analysis", "epic"):
        analysis = state.get("project_analysis")
        return asdict(analysis) if is_dataclass(analysis) else {}
    if kind in SECTION_KINDS:
        return {"items": [plain(item) for item in state.get(kind) or []]}
    raise ValueError(f"unknown plan section {kind!r}")


def start_state(
    description: str,
    *,
    intake_mode: str = "",
    solo: bool = False,
    analysis_profile_id: str = "",
    context_scope: dict | str | None = None,
    project_label: str = "",
    integrations: Sequence[str] | None = None,
) -> dict:
    """The state a fresh conversation starts from.

    The greeting and the size pick belong to ``_chat_preamble``, never to
    ``messages`` — project_intake reads ``messages[0]`` as the description, so
    anything else in front of it would be planned instead of the project. An
    unstated size is classified from the description, exactly as the chat's
    greeting does. ``solo`` seeds the Solo-world key so the intake plans for
    one developer; ``analysis_profile_id`` seeds the team calibration;
    ``context_scope`` (a dict or its JSON) and ``project_label`` are carried
    for the run to read later; ``integrations`` (None = unrestricted) names
    the connections the plan may consult.
    """
    from yeaboi.agent.chat_intake import GREETING_TEXT, resolve_intake_mode, seed_analysis_profile

    mode = intake_mode
    if not mode:
        mode = resolve_intake_mode(description)[0] or "smart"
    label = "Small" if mode == "small_project" else "Large"
    logger.info("Chat session opened: mode=%s solo=%s description_len=%d", mode, solo, len(description))
    state = {
        "messages": [],
        "questionnaire": None,
        "_intake_mode": mode,
        "_chat_greeting_done": True,
        # The opening line, held until it is sent as messages[0]. A caller that
        # never sends it gets an intake with nothing to plan, so the session
        # view carries it and the client's first turn is this text.
        "_chat_opening": description,
        # The description is deliberately absent: it becomes messages[0] on the
        # first turn, and replaying it here too would show it twice.
        "_chat_preamble": [
            {"role": "ai", "text": GREETING_TEXT},
            {"role": "ai", "text": f"Sounds like a {label} plan — switch any time with /small · /large."},
        ],
    }
    if solo:
        state["solo"] = True
    if context_scope:
        state["context_scope"] = context_scope if isinstance(context_scope, str) else json.dumps(context_scope)
    seed_analysis_profile(state, analysis_profile_id)
    if project_label:
        state["project_label"] = project_label
    if integrations is not None:
        state["session_integrations"] = list(integrations)
    return state


# --------------------------------------------------------------- the session


class ChatSession:
    """One planning conversation over the graph, driven a turn at a time.

    Owns the graph state; the caller owns the screen (or the socket) and
    decides what each event becomes. Every method blocks, so the TUI runs
    them on a worker thread and paints meanwhile while an HTTP route streams
    the events out as they arrive.

    :meth:`reply` answers whatever the conversation is parked on (a question,
    a review gate, a capacity or spike choice); :meth:`advance` runs the one
    step that needs no answer (a build stage, the epic reformat). ``on_version``
    is told every accepted section — ``(kind, payload) -> version`` — so a
    store can keep the plan's history.
    """

    def __init__(
        self,
        graph,
        state: dict,
        *,
        dry_run: bool = False,
        typewriter: bool = True,
        on_version: Callable[[str, dict], int] | None = None,
    ) -> None:
        self.graph = graph
        self.state = state
        self.dry_run = dry_run
        self.typewriter = typewriter
        self.on_version = on_version
        self.versions: dict[str, int] = {}

    @property
    def awaiting(self) -> str:
        """The stage this conversation is parked on — see :func:`stage_of`."""
        return stage_of(self.state, dry_run=self.dry_run)

    # ------------------------------------------------------------------ turns

    def send(
        self,
        text: str,
        on_event: EventSink,
        *,
        images: list[str] | None = None,
        files: list[str] | None = None,
        refs: list[dict] | None = None,
        cancel: threading.Event | None = None,
    ) -> bool:
        """Run one graph turn, emitting events as they happen.

        The first turn is an ordinary send: the description is ``messages[0]``
        and the graph builds the questionnaire from it. Returns True once the
        state has moved; provider and cancellation errors propagate, so the
        caller classifies them for its own surface. ``files`` and ``refs`` are
        rendered for the model by :mod:`yeaboi.agent.chat_refs`.
        """
        summary_open = at_intake_summary(self.state)
        if not self._turn(text, on_event, images=images, files=files, refs=refs, cancel=cancel):
            return False
        qs = questionnaire(self.state)
        if summary_open and qs is not None and qs.completed and not qs.awaiting_confirmation:
            on_event(SectionChanged("intake", "accepted", self._record_version("intake")))
        on_event(Done())
        return True

    def reply(
        self,
        text: str,
        on_event: EventSink,
        *,
        images: list[str] | None = None,
        files: list[str] | None = None,
        refs: list[dict] | None = None,
        cancel: threading.Event | None = None,
    ) -> bool:
        """Answer whatever the conversation is parked on.

        A review gate takes a verdict, a capacity or spike question takes an
        option, and anything else is a chat turn behind the input guardrails.
        Returns False when nothing moved (a blocked input, an unknown option,
        a stage that wants :meth:`advance` instead) — a Notice says why.
        """
        stage = self.awaiting
        logger.info("Chat reply: stage=%s len=%d files=%d refs=%d", stage, len(text), len(files or ()), len(refs or ()))
        if stage == "review":
            if files or refs:
                logger.info("Chat reply: refs and files are not read at a review gate")
            return self._review_reply(text, on_event, images)
        if stage in ("capacity", "spike"):
            if files or refs:
                logger.info("Chat reply: refs and files are not read at a %s gate", stage)
            return self._choice_reply(stage, text, on_event)
        if stage in ("pipeline", "epic"):
            on_event(Notice("The plan is being built — there is nothing to answer yet."))
            on_event(Done())
            return False
        if self._blocked(text, on_event):
            return False
        return self.send(text, on_event, images=images, files=files, refs=refs, cancel=cancel)

    @staticmethod
    def _blocked(text: str, on_event: EventSink) -> bool:
        """Run the input guardrails on text the person wrote; True (and a Notice) when it must not reach the model."""
        from yeaboi.input_guardrails import validate_chat_input

        block = validate_chat_input(text)
        if block is None:
            return False
        logger.info("Chat input blocked: layer=%s len=%d", block.layer, len(text))
        on_event(Notice(block.message))
        on_event(Done())
        return True

    def advance(self, on_event: EventSink, *, cancel: threading.Event | None = None) -> bool:
        """Run the one step that needs no answer: a build stage, or the epic reformat.

        Returns False when the conversation is parked on something that wants
        :meth:`reply`, or the stage failed (state unchanged) — a Notice says which.
        """
        stage = self.awaiting
        logger.info("Chat advance: stage=%s", stage)
        if stage == "epic":
            return self._epic_step(on_event)
        if stage != "pipeline":
            on_event(Notice("Nothing to run — the conversation is waiting for you."))
            on_event(Done())
            return False
        if self.dry_run:
            on_event(Notice("The build does not run in dry-run."))
            on_event(Done())
            return False
        node = next_node(self.state)
        step, total = pipeline_step(node)
        kind = REVIEW_ARTIFACT_KINDS.get(node, "")
        on_event(Progress(node, step, total, "running"))
        if kind:
            on_event(SectionChanged(kind, "generating", self.versions.get(kind, 0)))
        if not self._turn("continue", on_event, images=None, cancel=cancel, show_reply=False, synthetic=True):
            on_event(Done())
            return False
        on_event(Progress(node, step, total, "done"))
        self._emit_parked_gate(on_event)
        on_event(Done())
        return True

    # -------------------------------------------------------------- internals

    def _turn(
        self,
        text: str,
        on_event: EventSink,
        *,
        images: list[str] | None,
        cancel: threading.Event | None,
        show_reply: bool = True,
        synthetic: bool = False,
        files: list[str] | None = None,
        refs: list[dict] | None = None,
    ) -> bool:
        messages = list(self.state.get("messages", []))
        if text:
            # A synthetic turn is the driver's own "continue", not something
            # the person typed — replay leaves it out of the transcript.
            messages.append(HumanMessage(content=text, additional_kwargs={"synthetic": True} if synthetic else {}))
        intake_turn = next_node(self.state) == "project_intake"
        if intake_turn:
            # The intake confirmation is the one turn sent while a review gate
            # is still open — project_intake consumes the reply itself rather
            # than a review card doing it. Its confirm branch returns no
            # "pending_review", and pending_review is a plain LastValue channel
            # (agent/state.py), so whatever is in the input state survives the
            # invoke: leave it set and the gate never closes and the stage
            # stays "intake" forever. Safe to drop unconditionally — the node
            # re-sets it when the summary needs showing again ("edit").
            self.state.pop("pending_review", None)
        invoke_state = {**self.state, "messages": messages}
        if images:
            if intake_turn:
                invoke_state["pasted_images"] = list(self.state.get("pasted_images") or []) + images
            else:
                invoke_state["chat_images"] = images
        if files or refs:
            from yeaboi.agent.chat_refs import render_context_block

            block = render_context_block(refs or (), files or ())
            if intake_turn:
                invoke_state["pasted_context"] = list(self.state.get("pasted_context") or []) + block
            else:
                invoke_state["chat_context"] = block

        result = stream_chat_turn(
            self.graph,
            invoke_state,
            lambda chunk: on_event(Token(chunk)),
            cancel=cancel,
            typewriter=self.typewriter,
        )
        if result is None:
            # stream_chat_turn either returns a state or raises; a None here
            # would silently blank the session.
            logger.error("Chat turn produced no state")
            return False
        self.state = result
        if text:
            self.state.pop("_chat_opening", None)
        if show_reply:
            event = reply_event(self.state)
            if event is not None:
                on_event(event)
        return True

    def _emit_parked_gate(self, on_event: EventSink) -> None:
        gate = parked_gate(self.state)
        if isinstance(gate, AwaitReview):
            on_event(ShowArtifact(gate.kind))
            on_event(SectionChanged(gate.kind, "awaiting_review", self.versions.get(gate.kind, 0)))
        if gate is not None:
            on_event(gate)

    def _review_reply(self, text: str, on_event: EventSink, images: list[str] | None) -> bool:
        pending = self.state.get("pending_review", "")
        verdict = review_verdict(text, pending)
        if isinstance(verdict, Accept):
            kind = accept_review(self.state)
            logger.info("Review decision: accept %s", pending)
            on_event(SectionChanged(kind, "accepted", self._record_version(kind)))
        elif isinstance(verdict, SwitchSize):
            from yeaboi.agent.nodes import apply_size_switch

            apply_size_switch(self.state, verdict.target)
            self.state.pop("_prior_art_preview", None)
            logger.info("Review decision: switch size to %s", verdict.target)
            on_event(Notice("Switched to Large — your answers are kept and the plan regenerates from here."))
        elif isinstance(verdict, TrackerSync):
            on_event(Action("sync", verdict.tracker))
        elif self.dry_run:
            on_event(Notice("Edits are not available in dry-run — reply accept to continue."))
        else:
            # Edit feedback is the person's own words and lands in a prompt:
            # the same guardrails as a chat turn, whichever gate it came in on.
            if self._blocked(verdict.text, on_event):
                return False
            kind = apply_edit_feedback(self.state, pending, verdict.text, images)
            on_event(SectionChanged(kind, "empty", self.versions.get(kind, 0)))
        on_event(Done())
        return True

    def _choice_reply(self, stage: str, text: str, on_event: EventSink) -> bool:
        choice = capacity_choices(self.state) if stage == "capacity" else spike_choices(self.state)
        key = choice_key(choice, text)
        if key is None:
            on_event(Notice("Pick one of: " + ", ".join(k for k, _label in choice.options) + "."))
            on_event(choice)
            on_event(Done())
            return False
        if stage == "capacity":
            label = apply_capacity_choice(self.state, key)
            on_event(Notice(f"Capacity: {label}"))
        else:
            label = apply_spike_choice(self.state, key)
            on_event(Notice(f"Architecture spike: {label}."))
        on_event(Done())
        return True

    def _epic_step(self, on_event: EventSink) -> bool:
        from yeaboi.ui.session.chat._epic import reformat_epic_to_team_style

        step, total = pipeline_step(EPIC_REVIEW_NODE)
        on_event(Progress(EPIC_REVIEW_NODE, step, total, "running"))
        on_event(SectionChanged("epic", "generating", self.versions.get("epic", 0)))
        try:
            reformat_epic_to_team_style(self.state, dry_run=self.dry_run)
        except Exception:  # noqa: BLE001 — the original epic stands
            logger.error("Epic reformat step failed unexpectedly", exc_info=True)
        self.state["_epic_reviewed"] = True
        self.state["pending_review"] = EPIC_REVIEW_NODE
        on_event(Progress(EPIC_REVIEW_NODE, step, total, "done"))
        self._emit_parked_gate(on_event)
        on_event(Done())
        return True

    def _record_version(self, kind: str) -> int:
        version = self.versions.get(kind, 0) + 1
        if self.on_version is not None:
            try:
                version = int(self.on_version(kind, section_payload(self.state, kind)) or version)
            except Exception:  # noqa: BLE001 — history is best-effort, the plan is not
                logger.warning("Plan version for %s was not recorded", kind, exc_info=True)
        self.versions[kind] = version
        return version
