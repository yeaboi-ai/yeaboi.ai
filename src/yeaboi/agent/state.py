"""State schema for the LangGraph scrum agent.

Defines enums, artifact dataclasses, questionnaire state, and the main
ScrumState TypedDict that all graph nodes read from and write to.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from yeaboi._compat import IntEnum, StrEnum

# Imported rather than redeclared: OpsSignal is the connectors' own vocabulary
# (yeaboi.ops), and a second copy here would be two shapes to keep in step.
# ops/ is pure types with no I/O, so nothing cycles back into this module.
from yeaboi.ops.signals import OpsSignal

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Priority(StrEnum):
    """Priority levels for features and stories."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class StoryPointValue(IntEnum):
    """Allowed Fibonacci story-point values."""

    ONE = 1
    TWO = 2
    THREE = 3
    FIVE = 5
    EIGHT = 8


class Discipline(StrEnum):
    """Discipline tag for stories — indicates which team skillset owns the story.

    # See docs: "Scrum Standards" — discipline tagging
    #
    # Used to classify each story by the primary skillset needed to implement it.
    # The LLM prompt asks for a discipline field; if missing or invalid,
    # _infer_discipline() in nodes.py guesses from keywords. Default is FULLSTACK
    # (the safe catch-all when discipline is unclear).
    """

    FRONTEND = "frontend"
    BACKEND = "backend"
    FULLSTACK = "fullstack"
    INFRASTRUCTURE = "infrastructure"
    DESIGN = "design"
    TESTING = "testing"


class QuestionnairePhase(StrEnum):
    """High-level phases that map to question ranges.

    Seven phases matching the intake questionnaire design in the README.
    Each phase groups related questions to create a natural conversation flow.
    # See docs: "Scrum Standards" → questionnaire phases
    """

    PROJECT_CONTEXT = "project_context"  # Q1–Q5: project name, description, goals, users, scope
    TEAM_AND_CAPACITY = "team_and_capacity"  # Q6–Q10: team size, roles, velocity, sprint length
    TECHNICAL_CONTEXT = "technical_context"  # Q11–Q14: tech stack, architecture, integrations, constraints
    CODEBASE_CONTEXT = "codebase_context"  # Q15–Q20: repo URL, existing code, testing, CI/CD, docs
    RISKS_AND_UNKNOWNS = "risks_and_unknowns"  # Q21–Q23: risks, dependencies, unknowns
    PREFERENCES = "preferences"  # Q24–Q26: output format, naming conventions, process preferences
    CAPACITY_PLANNING = "capacity_planning"  # Q27–Q30: bank holidays, leave, unplanned %, onboarding


class TaskLabel(StrEnum):
    """Label classifying sub-tasks by the type of work involved.

    Auto-assigned by the task decomposer prompt based on task content.
    Used in REPL tables and TUI renderers to visually distinguish task types.
    # See docs: "Scrum Standards" — task decomposition
    """

    CODE = "Code"
    DOCUMENTATION = "Documentation"
    INFRASTRUCTURE = "Infrastructure"
    # Research/validation work (e.g. the architecture-validation spike). Only
    # ever set by the deterministic injectors in nodes.py — the task-decomposer
    # LLM is never asked to emit it. See docs: "Scrum Standards" — DoD Spike
    SPIKE = "Spike"
    TESTING = "Testing"


class ReviewDecision(StrEnum):
    """Possible outcomes when the user reviews generated artifacts."""

    ACCEPT = "accept"
    EDIT = "edit"
    REJECT = "reject"


class OutputFormat(StrEnum):
    """Supported export formats."""

    JIRA = "jira"
    MARKDOWN = "markdown"
    BOTH = "both"


# ---------------------------------------------------------------------------
# Phase-to-question mapping
# ---------------------------------------------------------------------------

PHASE_QUESTION_RANGES: dict[QuestionnairePhase, tuple[int, int]] = {
    QuestionnairePhase.PROJECT_CONTEXT: (1, 5),
    QuestionnairePhase.TEAM_AND_CAPACITY: (6, 10),
    QuestionnairePhase.TECHNICAL_CONTEXT: (11, 14),
    QuestionnairePhase.CODEBASE_CONTEXT: (15, 20),
    QuestionnairePhase.RISKS_AND_UNKNOWNS: (21, 23),
    QuestionnairePhase.PREFERENCES: (24, 26),
    QuestionnairePhase.CAPACITY_PLANNING: (27, 30),
}

TOTAL_QUESTIONS = 30

# ---------------------------------------------------------------------------
# Artifact dataclasses (frozen / immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptanceCriterion:
    """A single acceptance criterion — Given/When/Then, or the team's own style.

    Two shapes share this class: a Gherkin triple (given/when/then set, text
    empty) and a free-text criterion (text set, triple empty) for teams whose
    tickets don't use Given/When/Then. All fields are defaulted so old saved
    sessions (no ``text`` key) deserialize unchanged.
    See docs: "Scrum Standards" — Acceptance Criteria
    """

    given: str = ""
    when: str = ""
    then: str = ""
    # Free-text criterion ("Search results return within 200ms"). When set,
    # renderers show it verbatim; when empty, the GWT triple renders.
    text: str = ""

    @property
    def flat_text(self) -> str:
        """The criterion as one sentence, whichever shape it carries."""
        return self.text or f"Given {self.given}, when {self.when}, then {self.then}"


# Acceptance-criteria styles a plan can be generated in. "gwt" is the
# Given/When/Then default; "bullets" writes each criterion as a single clear,
# testable statement — for teams whose real tickets don't use Gherkin.
AC_STYLE_GWT = "gwt"
AC_STYLE_BULLETS = "bullets"
AC_STYLES: tuple[str, ...] = (AC_STYLE_GWT, AC_STYLE_BULLETS)


def resolve_ac_style(graph_state: dict | None = None, team_profile: object | None = None) -> str:
    """Resolve which acceptance-criteria style this plan should use.

    Precedence: the session's persisted choice (state["ac_format"], written by
    story_writer so exports of a saved plan match how it was generated) >
    the YEABOI_AC_FORMAT env override > the learned team profile
    (WritingPatterns.uses_given_when_then / evidence of writing data) >
    Given/When/Then.
    """
    if graph_state:
        persisted = graph_state.get("ac_format")
        if persisted in AC_STYLES:
            return persisted

    from yeaboi.config import get_ac_format  # lazy: config must not import state

    override = get_ac_format()
    if override in AC_STYLES:
        return override

    patterns = getattr(team_profile, "writing_patterns", None)
    if patterns is not None:
        if getattr(patterns, "uses_given_when_then", False):
            return AC_STYLE_GWT
        # The team has analysed writing data and it did NOT show GWT — follow
        # their real style instead of forcing the template on them.
        if getattr(patterns, "median_ac_count", 0) > 0 or getattr(patterns, "common_personas", ()):
            return AC_STYLE_BULLETS

    return AC_STYLE_GWT


def map_template_headings(sections: tuple[str, ...] | list[str] | None) -> dict[str, str]:
    """Map a team's learned description-section headings onto our canonical blocks.

    Team analysis records the section headings the team's real tickets use
    (e.g. "What is this about?", "Done looks like"). Exports keep their fixed
    structure but adopt the team's own heading names where one clearly maps:
    "summary" (the story sentence), "acceptance_criteria", and "dod". Unmatched
    headings are ignored — we have no content to put under them.
    """
    mapping: dict[str, str] = {}
    for heading in sections or ():
        low = str(heading).lower()
        if "acceptance" in low or low.rstrip(":?") in ("ac", "acs"):
            mapping.setdefault("acceptance_criteria", str(heading))
        elif "done" in low or "dod" in low:
            mapping.setdefault("dod", str(heading))
        elif any(word in low for word in ("summary", "about", "background", "description", "context", "what")):
            mapping.setdefault("summary", str(heading))
    return mapping


@dataclass(frozen=True)
class Feature:
    """A high-level feature grouping related user stories."""

    id: str
    title: str
    description: str
    priority: Priority


# Definition of Done — standard checklist applied to every user story.
# The LLM evaluates which items apply to each story and marks the rest as N/A.
# Rendered in the story table with strikethrough for non-applicable items.
# See docs: "Scrum Standards" — Definition of Done
DOD_ITEMS: tuple[str, ...] = (
    "Acceptance Criteria Met",
    "Documentation",
    "Proper Testing",
    "Code Merged to Main",
    "Released via SDLC",
    "Stakeholder Sign-off",
    "Knowledge Sharing",
)


def resolve_dod_items(graph_state: dict | None = None) -> tuple[str, ...]:
    """Return custom DoD items from state if set, else the default DOD_ITEMS.

    When an analysis profile provides team-specific DoD practices,
    they override the generic defaults for the entire planning session.
    """
    if graph_state:
        custom = graph_state.get("custom_dod_items")
        if custom and isinstance(custom, (tuple, list)) and len(custom) > 0:
            return tuple(custom)
    return DOD_ITEMS


def shorten_dod_items(items: tuple[str, ...]) -> tuple[str, ...]:
    """Generate short display labels from full DoD item names."""
    _known = {
        "Acceptance Criteria Met": "AC Met",
        "Documentation": "Docs",
        "Proper Testing": "Testing",
        "Code Merged to Main": "Code Merged",
        "Released via SDLC": "SDLC",
        "Stakeholder Sign-off": "Sign-off",
        "Knowledge Sharing": "Know. Sharing",
    }
    return tuple(_known.get(item, item[:14].strip()) for item in items)


@dataclass(frozen=True)
class UserStory:
    """A user story following the persona/goal/benefit template."""

    id: str
    feature_id: str
    persona: str
    goal: str
    benefit: str
    acceptance_criteria: tuple[AcceptanceCriterion, ...]
    story_points: StoryPointValue
    priority: Priority
    # Short summary title for the story, e.g. "Create Bookmark Endpoint".
    # Displayed in sprint views and used as headings in exports.
    # Default "" ensures backward compatibility with existing saved sessions.
    title: str = ""
    # Discipline tag — which team skillset owns this story.
    # Default is FULLSTACK so existing code (and fallback stories) work without changes.
    # See docs: "Scrum Standards" — discipline tagging
    discipline: Discipline = Discipline.FULLSTACK
    # Definition of Done flags — one bool per DOD_ITEMS entry.
    # True = applies to this story, False = not applicable (shown with strikethrough).
    # Default all-True so existing tests and fallback stories work without changes.
    dod_applicable: tuple[bool, ...] = (True, True, True, True, True, True, True)
    # LLM's reasoning for the story point estimate — explains what complexity,
    # uncertainty, or effort factors led to the assigned value. Used to calibrate
    # the AI's estimation against engineer expectations over time.
    points_rationale: str = ""
    # Confidence that the point estimate matches the team's historical data.
    # "high" (≥15 samples), "medium" (≥5), "low" (<5), "" (no data).
    points_confidence: str = ""

    @property
    def text(self) -> str:
        """Standard user-story sentence."""
        return f"As a {self.persona}, I want to {self.goal}, so that {self.benefit}."


@dataclass(frozen=True)
class Task:
    """A concrete implementation task tied to a user story."""

    id: str
    story_id: str
    title: str
    description: str
    # Auto-assigned by the task decomposer based on task content.
    # Default is CODE — the most common task type. The LLM picks the label
    # from the TaskLabel enum; the parser falls back to CODE if invalid.
    # See docs: "Scrum Standards" — task decomposition
    label: TaskLabel = TaskLabel.CODE
    # Auto-generated test plan for tasks labelled Code or Infrastructure.
    # Lists what to test (unit, integration, edge cases) so developers know
    # what verification is expected. Empty string for non-code tasks.
    # See docs: "Scrum Standards" — task decomposition, testing
    test_plan: str = ""
    # Self-contained instruction for AI coding assistants (Cursor, Claude Code,
    # GitHub Copilot). Includes project context, tech stack, and specific guidance
    # so a developer can paste it directly into an AI tool and start working.
    # See docs: "Scrum Standards" — task decomposition
    ai_prompt: str = ""


@dataclass(frozen=True)
class Sprint:
    """A sprint containing a subset of stories."""

    id: str
    name: str
    goal: str
    capacity_points: int
    story_ids: tuple[str, ...]


# Shared by every artifact a reader can correct in the browser. It lives here
# rather than in yeaboi.artifacts because it is *part of the artifact* — it
# serializes with it, exports with it, and has to deserialize out of a
# report_json written before the field existed, which is the whole reason every
# field is defaulted.
@dataclass(frozen=True)
class Annotation:
    """Something a reader added that the generated schema had no room for.

    Two shapes behind one dataclass, told apart by ``kind``:

    * ``note`` — free text hung off a section or a row; ``label`` is empty.
    * ``field`` — a named value the schema never had, e.g. "Risk owner: Ada".

    ``kind`` is an explicit discriminator rather than "an empty label means a
    note", because a sentinel is a rule a reader has to be told, and the first
    person to add a field with no name would silently create a note.

    ``anchor`` is the path of the thing this hangs off — a member, a project, a
    section — or ``""`` for the document as a whole. It is deliberately the same
    grammar the edits use, so a renderer has one way to ask "what belongs here".
    """

    kind: str = "note"
    anchor: str = ""
    label: str = ""
    text: str = ""
    author: str = ""
    avatar: str = ""
    at: str = ""


def annotations_from(value: object) -> tuple[Annotation, ...]:
    """Rebuild an annotation tuple from JSON-parsed dicts (missing → empty).

    One helper rather than the same six ``.get()`` blocks copied into six mode
    stores. Tolerant in the house style: anything that is not a sequence of
    dicts deserializes to ``()`` rather than raising, so a report written by a
    version that had never heard of annotations still loads.

    Accepts a tuple as well as a list, which is not pedantry: the JSON path
    hands over a list, but ``asdict`` rebuilds each container with
    ``type(obj)(...)`` and so hands over a *tuple*. Anonymize reconstructs from
    exactly that tree, so a list-only check would have dropped every annotation
    from a masked artifact — silently, and only on the path meant to make
    something safe to publish.
    """
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        Annotation(
            kind=str(a.get("kind", "note")),
            anchor=str(a.get("anchor", "")),
            label=str(a.get("label", "")),
            text=str(a.get("text", "")),
            author=str(a.get("author", "")),
            avatar=str(a.get("avatar", "")),
            at=str(a.get("at", "")),
        )
        for a in value
        if isinstance(a, dict)
    )


# See docs: "Session Management" — Daily Standup mode artifacts
#
# The Daily Standup mode produces a StandupReport for a given day: one
# MemberUpdate per team member (either self-reported by the person or inferred
# by the LLM from their recent ticket/code activity), a team-level narrative,
# and a deterministic sprint-progress confidence score. Like every other
# artifact in this module it is a FROZEN dataclass — immutable once built and
# serializable via asdict() — so it round-trips cleanly through the session
# store. Every field has a default so old serialized reports still deserialize
# (see AGENTS.md "Frozen dataclass backward compatibility").
# How many evidence rows one member's category carries in a report.
#
# It lives here, on the neutral module both readers already import, because two
# places have to agree on it and they cannot import each other: the engine's
# `_member_evidence` applies it, and `gap_taxonomy`'s truncation rule uses it to
# decide whether a category is *at* its cap — i.e. whether activity was provably
# cut. When those two drifted (the cap moved to 30, the rule kept assuming 8) the
# rule fired on any member whose commits merely nested under a PR, and
# `gap_issues` files that as a public GitHub issue.
MEMBER_EVIDENCE_CAP = 30


@dataclass(frozen=True)
class ActivityEvidence:
    """One attributable activity item kept as structured evidence.

    The collectors already fetch a title, kind, and repository for every commit,
    PR, and ticket; this keeps them past the point where ``MemberUpdate.*_links``
    narrows each item to a bare ``(label, url)`` pair — so an export can say
    "[commit] 78e4201 Fix login redirect · yeaboi/web" instead of a naked SHA.
    Evidence is for rendering only; the LLM never sees it (engine._for_llm).
    """

    # commit | pr | review | comment | issue | update | work_item | wip | page | page-created
    # — produced by the collectors, not validated
    kind: str = ""
    key: str = ""  # short handle: "#91", "YB-12", sha[:8]
    title: str = ""  # commit subject / PR title / ticket summary
    url: str = ""
    repository: str = ""  # repo slug or source container; "" for tickets
    status: str = ""  # "merged", "In Progress", a review state, or ""
    timestamp: str = ""  # ISO-8601 or ""
    # Commits folded under their PR row (one level deep); () everywhere else.
    # Story→subtask hierarchy deliberately does NOT reuse this — it rides the
    # flat fields below so gap/transcript iteration never misses a row.
    children: tuple[ActivityEvidence, ...] = ()
    issue_type: str = ""  # tracker's own word: "Story", "Sub-task", "Task", "Bug"; "" for code/docs
    parent_key: str = ""  # parent issue in sibling-key spelling: "PROJ-12" (Jira), "#123" (AzDO)
    # The tracker's authoritative child-of-a-story flag (Jira issuetype.subtask,
    # AzDO WorkItemType == Task). The ONLY licence to nest under parent_key —
    # a team-managed Jira Story also carries parent_key (its epic) with subtask False.
    subtask: bool = False
    # Code/doc rows only: exact tracker keys this change's own text or first-party
    # links name ("PROJ-12", "#123"). Never fuzzy-matched (references.display_ticket_keys).
    ticket_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class PracticeSignal:
    """One deterministic engineering-practice observation about a member's day.

    Produced by ``standup/habits.py`` from the collected activity — never by the
    LLM, and never shown to it. Each signal is an *observation with evidence*
    ("this PR has no ticket reference", with the PR's link), not a verdict:
    the standup names people, so a rule only fires on positive evidence and a
    missed signal is always preferred to a wrong one.

    ``rule`` is an engine-produced vocabulary, not a validated one — an
    unrecognised id renders with the muted fallback tone rather than failing a
    build (same treatment as confidence labels and coverage statuses).
    """

    rule: str = ""  # "untracked-work" | "board-not-updated" | … (habits.ALL_RULES)
    title: str = ""  # short label for the chip, e.g. "Untracked work"
    detail: str = ""  # the observation plus its nudge, one or two sentences
    evidence: tuple[tuple[str, str], ...] = ()  # (label, url) — the items observed
    repeat: bool = False  # the same signal also fired in the previous standup
    # Stable ids of EVERY change behind this signal — ``habits.change_handle``.
    # Internal identity, not display: this is what a thumbs-down remembers, so a
    # change excused once never fires again. Deliberately wider than ``evidence``,
    # which is capped at four links, and deliberately absent from the export
    # payload — a static HTML file can do nothing with it.
    handles: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConflictCard:
    """Two sources disagree about one property of one work item.

    Produced by standup/conflicts.py from the aggregate's grouped activity.
    Like a PracticeSignal, a conflict is an observation with evidence, not a
    verdict: the card names both claims and says what would settle them,
    instead of anyone's confidence being silently lowered because the sources
    disagree. Severity is a word (provenance.conflicts.Severity), never a
    colour — payload rules apply all the way down.
    """

    fingerprint: str = ""  # stable id, e.g. "YEA-12:status:status_conflict"
    title: str = ""  # "YEA-12 — the board says Done, a pull request is still open"
    detail: str = ""  # the observation, spelled out
    severity: str = "medium"  # low | medium | high | critical
    entity_id: str = ""  # the work item both sources are talking about
    property_name: str = ""  # what they disagree on, e.g. "status"
    # One claim per source: (source, value, label, url) — label/url are the
    # click-through evidence, same shape discipline as MemberUpdate.links.
    claims: tuple[tuple[str, str, str, str], ...] = ()
    recommended_action: str = ""  # what would settle it
    members: tuple[str, ...] = ()  # whose activity surfaced the disagreement


@dataclass(frozen=True)
class MemberUpdate:
    """One team member's standup update for a given day."""

    name: str = ""
    summary: str = ""  # general overview synthesizing all category evidence + self-report
    blockers: str = ""  # anything blocking them (empty if none)
    progress_note: str = ""  # one sentence linking yesterday's standup to today (continued/completed/stalled)
    outlook: str = ""  # one sentence predicting the member's likely focus for the day ahead
    source: str = "inferred"  # "inferred" (activity only) | "self-reported" (typed, no activity) | "combined" (both)
    self_report: str = ""  # the member's own typed update, kept verbatim as supporting context
    links: tuple[tuple[str, str], ...] = ()  # (label, url) refs from their activity — tuple-of-pairs stays frozen
    activity_count: int = 0  # attributed activity items today — drives the ●/○ active/quiet glyphs in the TUI
    code_summary: str = ""  # separate outcome-oriented summary of commits, PRs, and reviews
    code_links: tuple[tuple[str, str], ...] = ()  # repository evidence links, kept separate from tracker/docs links
    code_activity_count: int = 0  # attributed code events; never presented as a productivity score
    documentation_summary: str = ""  # Confluence/Notion and repository documentation outcomes
    documentation_links: tuple[tuple[str, str], ...] = ()
    documentation_activity_count: int = 0
    ticketing_summary: str = ""  # Jira/Azure Boards progress, including assigned in-progress work
    ticketing_links: tuple[tuple[str, str], ...] = ()
    ticketing_activity_count: int = 0
    # Structured evidence per category — additive alongside the *_links pairs,
    # which still feed the TUI, the ticket-key linkifier, and legacy exports.
    ticketing_evidence: tuple[ActivityEvidence, ...] = ()
    code_evidence: tuple[ActivityEvidence, ...] = ()
    documentation_evidence: tuple[ActivityEvidence, ...] = ()
    # Deterministic practice observations (standup/habits.py) — capped, and
    # empty whenever detection is off or the tracker coverage can't support it.
    practices: tuple[PracticeSignal, ...] = ()


@dataclass(frozen=True)
class StandupReport:
    """A full daily standup for one project session on one day.

    Produced by standup/engine.py:run_standup(). Rendered to the TUI and
    delivered to configured channels (terminal/desktop/Slack/email).
    """

    date: str = ""  # ISO date the standup covers, e.g. "2026-07-10"
    session_id: str = ""
    sprint_name: str = ""
    sprint_day: int = 0  # which working day of the sprint we're on (1-indexed)
    sprint_total_days: int = 0  # total working days in the sprint
    confidence_pct: int = 0  # 0-100 confidence we'll hit the sprint goal
    confidence_label: str = ""  # "On track" | "At risk" | "Behind" | "Insufficient data"
    confidence_rationale: str = ""  # short human-readable explanation
    confidence_delta: int = 0  # today's pct minus the previous standup's pct (0 when no usable history)
    confidence_trend: str = ""  # "improving" | "steady" | "declining" | "" (no usable history)
    team_summary: str = ""  # LLM-synthesized team-level narrative
    member_updates: tuple[MemberUpdate, ...] = ()
    activity_counts: tuple[tuple[str, int], ...] = ()  # (source, count) — tuple so it stays frozen/serializable
    activity_window: str = ""  # human-readable look-back window, e.g. "Fri 2026-07-17 00:00 → now"
    # Machine-readable window bounds (tz-aware ISO-8601) — the web timeline's
    # axis. Defaulted so a report stored before the timeline existed still
    # deserializes; the page derives the axis from event times when empty.
    activity_window_start: str = ""
    activity_window_end: str = ""
    skipped_sources: tuple[tuple[str, str], ...] = ()  # (source, reason) for sources NOT scanned — visible, not silent
    # The subset of skipped_sources the user actually ASKED for and did not get.
    # Diagnostic surfaces (the TUI panel, the HTML details) list every skip; the
    # broadcast ones (Slack/email plaintext, Markdown) list only these, or a
    # Jira-only team reads the same five-source apology in every standup forever.
    unmet_sources: tuple[str, ...] = ()
    category_coverage: tuple[tuple[str, str], ...] = ()  # category -> covered/partial/failed/not_configured
    my_name: str = ""  # the standup user's resolved display name (drives the "My Update" row)
    solo: bool = False  # a one-person run: one card, first-person summary; renderers voice from this
    warnings: tuple[str, ...] = ()  # surfaced problems (missing API key, source 401/403) — shown, never silent
    images: tuple[str, ...] = ()  # screenshot paths pasted into "My Update" — embedded in exports
    # Reader-authored additions; see Annotation. Defaulted so a report stored
    # before browser editing existed still deserializes.
    annotations: tuple[Annotation, ...] = ()
    # (rule, member count) for the overview rollup — same shape as activity_counts
    practice_rollup: tuple[tuple[str, int], ...] = ()
    # Cross-source disagreements (standup/conflicts.py) — defaulted so a
    # report stored before conflict cards existed still deserializes.
    conflicts: tuple[ConflictCard, ...] = ()
    # What production did over its own (wider) window — bounded counts per
    # source and kind, never per person. Empty whenever no ops vendor is
    # connected, which is what keeps the "Production" panel unearned.
    ops_signals: tuple[OpsSignal, ...] = ()


# See docs: "Session Management" — Daily Standup transcript-review artifacts
#
# After the standup MEETING, a transcript of it is the only place the report
# gets fact-checked out loud ("yes, but I also did 4 and 5"). The transcript
# review absorbs that correction and works out WHY the work was invisible, so
# each root cause can become a GitHub issue against yeaboi itself. The review is
# a SIBLING artifact, never a field on StandupReport: standup_history is an
# append-only record of what was said at the time, and reports must not grow a
# field that every older serialized report lacks.
#
# Same rules as every artifact here: FROZEN, every field defaulted, collections
# as tuples so asdict() round-trips (see AGENTS.md "Frozen dataclass backward
# compatibility").
@dataclass(frozen=True)
class TranscriptSource:
    """One transcript file that fed a review."""

    path: str = ""
    filename: str = ""
    fmt: str = ""  # txt | md | vtt | srt | json
    covered_date: str = ""  # ISO date of the standup this transcript is FOR
    char_count: int = 0
    truncated: bool = False  # hit the read cap — the tail was not reviewed
    speakers: tuple[str, ...] = ()  # raw speaker labels found, pre-attribution
    attribution: str = "labelled"  # "labelled" | "unlabelled" (too few speaker labels to trust)
    external: bool = False  # came from the configured external dir, not the managed one


@dataclass(frozen=True)
class TranscriptNudge:
    """Standups that were never checked against their meeting.

    DERIVED, never stored: the whole thing is a set difference between the dates
    a standup ran and the dates a transcript was reviewed for, both already in
    the database. That is deliberate — a "last nudged at" column would be state
    to migrate, to reset, and to get wrong, and the honest answer is recomputable
    at any moment.

    ``level`` is the anti-nag policy in one word, and it escalates toward the
    EXIT rather than toward volume: after enough unchecked standups the honest
    reading is that this team does not record theirs, so the wording turns into
    an offer to switch the feature off rather than a louder version of the same
    request.
    """

    session_id: str = ""
    missed_dates: tuple[str, ...] = ()  # newest first
    streak: int = 0  # consecutive unchecked standup dates, counting back from the newest
    standup_count: int = 0  # standups in the window — the denominator
    ever_reviewed: bool = False  # has this session EVER reviewed a transcript
    level: str = ""  # "" | invite | reminder | escalated
    message: str = ""

    def __bool__(self) -> bool:
        """True when there is something to say, so callers read ``if nudge:``."""
        return bool(self.level)


@dataclass(frozen=True)
class TranscriptClaim:
    """One thing a person said they did, checked against what the report knew.

    ``quote`` is verbatim transcript text and is what makes the claim
    falsifiable: the review drops any claim whose quote is not literally present
    in the transcript, which is the single strongest guard against an invented
    gap reaching a public issue tracker.
    """

    member: str = ""  # resolved roster name; "" when attribution failed
    claim: str = ""  # what they said they did, in the model's words
    quote: str = ""  # verbatim supporting transcript text (clipped)
    status: str = ""  # matched | missing | contradicted
    matched_key: str = ""  # the evidence key it corresponds to, when one was offered
    # jira | azure_devops | github | azure_repos | local_git | confluence | notion | none | unknown
    system_hint: str = ""
    artifact_hint: str = ""  # free text from the model — never fingerprinted, see gap_taxonomy
    source_path: str = ""  # which transcript it came from


@dataclass(frozen=True)
class StandupGap:
    """One diagnosed reason standup missed (or misstated) real work.

    ``scope`` decides what happens next and is the whole point of the taxonomy:
    "config" means the user can fix it by configuring standup, so it stays local
    as a suggestion; "product" means yeaboi is at fault and no amount of
    configuring helps, so it becomes a drafted GitHub issue. "none" is counted
    only (work with no digital footprint is expected, not a defect).
    """

    fingerprint: str = ""  # stable dedup key — see gap_taxonomy.fingerprint
    category: str = ""  # a gap_taxonomy GapCategory id
    scope: str = ""  # config | product | none
    title: str = ""
    detail: str = ""
    root_cause: str = ""
    priority: str = "medium"  # critical | high | medium | low
    confidence: str = "medium"  # high | medium | low
    feedback_kind: str = "Improvement"  # maps to feedback.FEEDBACK_TYPES for the issue title prefix
    members: tuple[str, ...] = ()
    claims: tuple[TranscriptClaim, ...] = ()
    evidence: tuple[str, ...] = ()  # human-readable "why we believe this" lines
    next_steps: tuple[str, ...] = ()
    affected_systems: tuple[str, ...] = ()
    remedy: str = ""  # for config gaps: the exact thing the user should change


@dataclass(frozen=True)
class TranscriptReview:
    """The audit of one standup report against the meeting that discussed it."""

    review_id: int = 0
    session_id: str = ""
    standup_date: str = ""  # the date the reviewed transcripts cover
    run_id: int = 0  # standup_history.id of the audited run; 0 when no run matched
    reviewed_at: str = ""
    sources: tuple[TranscriptSource, ...] = ()
    claims: tuple[TranscriptClaim, ...] = ()
    gaps: tuple[StandupGap, ...] = ()  # scope="product" — these draft GitHub issues
    config_suggestions: tuple[StandupGap, ...] = ()  # scope="config" — never filed
    accuracy_note: str = ""
    claims_matched: int = 0
    claims_missing: int = 0
    claims_contradicted: int = 0
    untracked_count: int = 0  # claims with no digital footprint — expected, not a defect
    llm_mode: str = ""  # "llm" | "deterministic"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class GapIssueLink:
    """What happened to one gap on GitHub — the dedup ledger's public half."""

    fingerprint: str = ""
    issue_number: int = 0  # 0 for the browser path, which cannot learn the number
    issue_url: str = ""
    state: str = ""  # drafted | filed | commented | browser | skipped | failed | blocked
    filed_at: str = ""
    last_commented_at: str = ""
    occurrences: int = 0
    via: str = ""  # "api" | "browser" | ""
    message: str = ""  # human status line, including why a filing was skipped/blocked


@dataclass(frozen=True)
class IssueFilingResult:
    """Outcome of one explicit "file these" act. Never raised, always reported."""

    review_id: int = 0
    links: tuple[GapIssueLink, ...] = ()
    filed: int = 0
    commented: int = 0
    skipped: int = 0
    warnings: tuple[str, ...] = ()


# See docs: "Session Management" — Retro mode artifacts
#
# The Retro (retrospective) mode produces a RetroReport: every sticky card the
# team added to the four-grid board (What went well / What didn't go well /
# Action items / Demos), plus the distinct participants who contributed. Like
# every other artifact here it is a FROZEN dataclass — immutable once built and
# serializable via asdict() — so it round-trips through the retro store. Cards
# are gathered live during the session by the mutable RetroBoard (retro/board.py,
# which owns the threading lock); RetroReport is the finalized snapshot the store
# and exporter consume. Every field is defaulted so old serialized reports still
# deserialize (see AGENTS.md "Frozen dataclass backward compatibility").
@dataclass(frozen=True)
class RetroCard:
    """One sticky card on the retro board.

    ``id`` is assigned server-side (never trusted from the browser client) so a
    LAN peer cannot forge or overwrite an existing card. ``origin`` distinguishes
    human-authored web cards from AI-generated action items so both the TUI and
    the browser can badge them.
    """

    id: str = ""  # server-assigned uuid4().hex[:12]
    grid: str = ""  # one of RETRO_GRIDS (went_well/didnt_go_well/action_items/demos)
    text: str = ""  # raw card text — escaped only at render time, never pre-escaped
    author: str = ""  # display name from the browser name prompt (or "AI")
    created_at: str = ""  # ISO-8601 UTC timestamp
    origin: str = "web"  # "web" (a teammate) | "ai" (LLM action item) | "carryover" (re-added from last retro)
    # (emoji, count) pairs — a tuple (not a dict) so the card stays frozen/hashable and
    # serializes cleanly, exactly like StandupReport.activity_counts. Populated only at
    # report time by board_to_report(); the live board keeps reactions in its own map.
    reactions: tuple[tuple[str, int], ...] = ()
    # Progress on a *carried-over* action item from the previous retro. Empty for
    # normal authoring-grid cards; one of retro.board.CARRIED_STATUSES for the items
    # surfaced in the "Last sprint's actions" review column so the team can close the
    # loop (pending/done/in_progress/carried_over/not_relevant). See AGENTS.md retro.
    status: str = ""


@dataclass(frozen=True)
class RetroReport:
    """A finished retrospective for one project session.

    Produced by retro/board.py:board_to_report(). Persisted by RetroStore and
    rendered to Markdown + HTML by retro/export.py.
    """

    date: str = ""  # ISO date the retro was held, e.g. "2026-07-10"
    session_id: str = ""
    project_name: str = ""
    sprint_name: str = ""
    cards: tuple[RetroCard, ...] = ()  # every card across all four grids
    participants: tuple[str, ...] = ()  # distinct human authors who contributed
    generated_at: str = ""  # ISO-8601 UTC timestamp the report was assembled
    # Last sprint's action items reviewed *during* this retro, each carrying the
    # ``status`` the team set (see RetroCard.status). Seeded at board open from the
    # prior run's action_items grid; the Prep↔Completion carry-forward loop for retro
    # (mirrors Performance mode). Empty when there was no prior retro for the session.
    carried_action_items: tuple[RetroCard, ...] = ()
    # Reader-authored additions; see Annotation. Defaulted so a report stored
    # before browser editing existed still deserializes.
    annotations: tuple[Annotation, ...] = ()

    def by_grid(self) -> dict[str, list[RetroCard]]:
        """Group this report's cards by grid key, preserving insertion order."""
        from yeaboi.retro.board import RETRO_GRIDS

        out: dict[str, list[RetroCard]] = {g: [] for g in RETRO_GRIDS}
        for c in self.cards:
            out.setdefault(c.grid, []).append(c)
        return out


# See docs: "Session Management" — Poker mode artifacts
#
# Scrum Poker (planning poker) mode: the team votes on tickets pulled from Jira
# or Azure DevOps, the admin reveals the votes and writes the agreed story
# points back to the board. Like Retro, the live session is run by a mutable,
# lock-guarded board (poker/board.py); these FROZEN dataclasses are the
# finalized snapshot the store and exporter consume. Every field is defaulted
# so old serialized reports still deserialize (see AGENTS.md "Frozen dataclass
# backward compatibility").
@dataclass(frozen=True)
class PokerVote:
    """One participant's revealed vote on a ticket (the round that was accepted)."""

    voter: str = ""  # display name from the browser join prompt
    avatar: str = ""  # emoji avatar picked at join (may be empty)
    value: str = ""  # a POKER_DECK string ("0".."21", "?" or "☕")


@dataclass(frozen=True)
class PokerTicketResult:
    """One ticket's outcome after a poker session.

    ``initial_points`` is what the tracker held when the session started;
    ``final_points`` is what the admin saved after the reveal (also pushed to
    Jira / Azure DevOps). ``estimated`` distinguishes "finalized this session"
    from "skipped / never voted".
    """

    key: str = ""  # Jira issue key ("PROJ-1") or AzDO work-item id ("101")
    url: str = ""  # browse URL on the tracker ("" for demo tickets)
    summary: str = ""
    description: str = ""  # plain display text (HTML already stripped for AzDO)
    state: str = ""  # tracker status name at fetch time
    assignee: str = ""
    initial_points: float | None = None
    final_points: float | None = None
    estimated: bool = False  # True once the admin finalized points this session
    votes: tuple[PokerVote, ...] = ()  # the revealed round the final points came from
    ai_note: str = ""  # the AI-perspective comment shown during the debate ("" if unused)
    # Duel (open the floor): the recorded low-vs-high debate, "" when no duel
    # ran. Defaulted so pre-duel report JSON keeps deserializing (frozen-
    # dataclass backward-compat rule — no schema bump needed).
    duel_transcript: str = ""  # capped, speaker-attributed where browser mics ran
    duel_low: str = ""  # "Alex (2)" — the low-extreme duelist and their vote
    duel_high: str = ""  # "Sam (13)"


@dataclass(frozen=True)
class PokerReport:
    """A finished poker session over one batch of tickets.

    Produced by poker/board.py:board_to_report(). Persisted by PokerStore and
    rendered to Markdown + HTML by poker/export.py.
    """

    date: str = ""  # ISO date the session was held
    session_id: str = ""
    project_name: str = ""
    source: str = ""  # "jira" | "azdevops" | "demo"
    scope_label: str = ""  # "Sprint 42" | "Backlog" — where the tickets came from
    tickets: tuple[PokerTicketResult, ...] = ()
    participants: tuple[str, ...] = ()  # distinct joiner names seen during the session
    generated_at: str = ""  # ISO-8601 UTC timestamp the report was assembled


# See docs: "Session Management" — Performance mode artifacts
#
# The Performance mode helps a team lead manage each engineer's growth. It has
# three connected workflows — 1:1 Prep, 1:1 Completion, and a 6-Month Review —
# each producing a FROZEN dataclass artifact (immutable + asdict()-serializable),
# exactly like the Standup / Retro reports above. Every field is defaulted so an
# artifact serialized by an older version still deserializes (see AGENTS.md
# "Frozen dataclass backward compatibility"). The roster (EngineerRef) is derived
# from the real people who did work in Jira / Azure DevOps, and EngineerActivity
# is the per-engineer slice of their recent-sprint tickets that seeds a 1:1 prep.
@dataclass(frozen=True)
class EngineerRef:
    """A team member the lead can run performance workflows for.

    Sourced from Jira / Azure DevOps assignees (the people who actually did work),
    not from the plan's team-size number — so the roster reflects reality.
    """

    name: str = ""  # display name, e.g. "Ada Lovelace"
    source: str = ""  # "jira" | "azuredevops" | "manual"
    external_id: str = ""  # accountId / descriptor — best-effort, may be empty
    email: str = ""  # best-effort; often hidden by tracker privacy settings.
    # Both of the above are alias seeds: the same person authors commits, pages
    # and retro cards under handles that are not their tracker display name, so
    # evidence gathered by display name alone silently misses their work.


@dataclass(frozen=True)
class PerfMetric:
    """One measured fact about an engineer, kept as a number.

    ``performance/evidence.py`` computes these — stories completed, points
    delivered, spill rate, cycle time, the share of changes carrying tests — and
    used to render them straight into English for the prompt, which is where the
    numbers stopped. Keeping the record means the prompt's sentence and the
    page's meter are formatted from the same value and cannot disagree.

    **A metric with no sample is omitted, never emitted with a value of 0.** An
    engineer whose spill rate was never measured has not spilled 0%; the
    coverage rows are what explain the absence.

    ``denominator`` of 0 means there is no ratio, so a renderer draws a bare
    count rather than a meter scaled against nothing.
    """

    key: str = ""  # "stories_completed" | "spill_rate" | "tests_rate" | …
    label: str = ""  # "Stories completed"
    value: float = 0.0
    denominator: float = 0.0
    unit: str = ""  # "" | "%" | "pts" | "d" — what the number IS, never how to draw it
    group: str = ""  # "delivery" | "practice" | "ceremony" | "volume"
    source: str = ""  # an evidence.SOURCE_* id, so a number traces to its coverage row
    detail: str = ""  # one sentence of context


@dataclass(frozen=True)
class EvidenceGroup:
    """A titled run of structured evidence rows, tied to the source that produced it.

    Grouped by source rather than by claim, deliberately: nothing in the pipeline
    maps an LLM-written bullet to a specific pull request, and a per-bullet
    citation invented on the way out would be indistinguishable from a real one
    in a document that decides someone's promotion.
    """

    source: str = ""  # SOURCE_* — pairs with the artifact's evidence_coverage row
    label: str = ""  # "Code activity"
    items: tuple[ActivityEvidence, ...] = ()
    note: str = ""  # "capped at 12 of 47"


@dataclass(frozen=True)
class PerformanceNote:
    """A lead's private note about an engineer, as an artifact the surfaces can render.

    The store has always returned these as a bare string; carrying who and when
    alongside is what lets a note be titled, dated, and masked like every other
    performance artifact.
    """

    engineer: str = ""
    date: str = ""
    text: str = ""


def metrics_from(value: object) -> tuple[PerfMetric, ...]:
    """Rebuild a metric tuple from JSON-parsed dicts (missing → empty).

    Same tolerance as :func:`annotations_from`, and for the same two reasons: an
    artifact stored before metrics existed must still load, and the anonymize
    path reconstructs from an ``asdict`` tree whose containers are tuples.
    """
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[PerfMetric] = []
    for m in value:
        if not isinstance(m, dict):
            continue
        try:
            out.append(
                PerfMetric(
                    key=str(m.get("key", "")),
                    label=str(m.get("label", "")),
                    value=float(m.get("value", 0.0) or 0.0),
                    denominator=float(m.get("denominator", 0.0) or 0.0),
                    unit=str(m.get("unit", "")),
                    group=str(m.get("group", "")),
                    source=str(m.get("source", "")),
                    detail=str(m.get("detail", "")),
                )
            )
        except (TypeError, ValueError):
            continue  # a row that cannot be read is dropped, never raised
    return tuple(out)


def evidence_from(value: object) -> tuple[ActivityEvidence, ...]:
    """Rebuild an ActivityEvidence tuple from JSON-parsed dicts, children included."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        ActivityEvidence(
            kind=str(e.get("kind", "")),
            key=str(e.get("key", "")),
            title=str(e.get("title", "")),
            url=str(e.get("url", "")),
            repository=str(e.get("repository", "")),
            status=str(e.get("status", "")),
            timestamp=str(e.get("timestamp", "")),
            children=evidence_from(e.get("children")),
            issue_type=str(e.get("issue_type", "")),
            parent_key=str(e.get("parent_key", "")),
            subtask=bool(e.get("subtask", False)),
            ticket_keys=tuple(str(k) for k in e.get("ticket_keys", ()) or ()),
        )
        for e in value
        if isinstance(e, dict)
    )


def evidence_groups_from(value: object) -> tuple[EvidenceGroup, ...]:
    """Rebuild an evidence-group tuple from JSON-parsed dicts (missing → empty)."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        EvidenceGroup(
            source=str(g.get("source", "")),
            label=str(g.get("label", "")),
            items=evidence_from(g.get("items")),
            note=str(g.get("note", "")),
        )
        for g in value
        if isinstance(g, dict)
    )


def coverage_from(value: object) -> tuple[tuple[str, str, str], ...]:
    """Rebuild (source, state, detail) coverage rows from JSON or a masked artifact.

    Both round-trips hand these back as lists of lists, so rows are rebuilt as
    tuples and anything not three-wide is dropped rather than crashing a reader.
    """
    rows: list[tuple[str, str, str]] = []
    for row in value or ():
        try:
            source, state, detail = row
        except (TypeError, ValueError):
            continue
        rows.append((str(source), str(state), str(detail)))
    return tuple(rows)


@dataclass(frozen=True)
class EngineerStory:
    """One ticket an engineer worked on in a recent sprint window."""

    key: str = ""  # e.g. "PROJ-123" or "#456"
    title: str = ""
    status: str = ""  # e.g. "In Progress", "Done"
    kind: str = ""  # "issue" | "work_item"
    sprint: str = "current"  # "current" | "previous" — which look-back window it came from
    source: str = ""  # "jira" | "azuredevops"


@dataclass(frozen=True)
class EngineerActivity:
    """An engineer's worked tickets across the current + prior sprint windows.

    Deterministic (no LLM) — assembled by performance/activity.py from the same
    recent-activity helpers the standup uses, grouped by author. Feeds the 1:1
    prep prompt as concrete evidence of what the person actually did.
    """

    engineer: str = ""
    current_sprint: str = ""  # active sprint name (best-effort, may be empty)
    previous_sprint: str = ""
    stories: tuple[EngineerStory, ...] = ()
    total_items: int = 0
    sources: tuple[tuple[str, int], ...] = ()  # (source, count) — frozen/serializable like activity_counts


@dataclass(frozen=True)
class OneOnOnePrep:
    """Talking points for an upcoming 1:1, derived from recent work + last 1:1.

    Produced by performance/engine.py:run_one_on_one_prep() (one LLM call, with a
    deterministic fallback). ``carried_action_items`` are the open actions from the
    engineer's most recent 1:1 completion — this is what closes the Prep↔Completion
    loop so nothing agreed in a 1:1 is silently dropped.
    """

    engineer: str = ""
    date: str = ""  # ISO date the prep was generated
    talking_points: tuple[str, ...] = ()
    feedback: tuple[str, ...] = ()  # positive + constructive feedback to give
    goals: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()  # skill / delivery gaps observed
    improvements: tuple[str, ...] = ()  # concrete things to improve
    carried_action_items: tuple[str, ...] = ()  # open actions from the previous 1:1
    activity_summary: str = ""  # short prose summary of the sprint work reviewed
    warnings: tuple[str, ...] = ()
    # What fed this artifact, and what did not. ``evidence_coverage`` rows are
    # (source, state, detail) — state uses the standup coverage vocabulary. Kept
    # so a reader can tell "did nothing" from "nobody looked"; see
    # performance/evidence.py. Defaulted for backward-compat.
    evidence_sources: tuple[str, ...] = ()
    evidence_coverage: tuple[tuple[str, str, str], ...] = ()
    # The numbers behind the prose, and the items behind the numbers. See
    # PerfMetric and EvidenceGroup; both empty on an artifact stored before they
    # existed, which every renderer treats as "nothing to show", not as zero.
    metrics: tuple[PerfMetric, ...] = ()
    evidence_items: tuple[EvidenceGroup, ...] = ()
    # (section_key, state, reason) using the coverage vocabulary, so an empty
    # section can say whether nothing was found or nothing was looked at.
    section_states: tuple[tuple[str, str, str], ...] = ()
    # The tickets this prep was built from, kept rather than only summarised —
    # a reader who disagrees with the summary needs the list it came from.
    activity: EngineerActivity = EngineerActivity()
    # Reader-authored additions; see Annotation. Defaulted so a report stored
    # before browser editing existed still deserializes.
    annotations: tuple[Annotation, ...] = ()


@dataclass(frozen=True)
class OneOnOneRecord:
    """A completed 1:1: the transcript plus the AI-written email summary + actions.

    Produced by performance/engine.py:complete_one_on_one(). The email summary is
    delivered via SMTP (reusing standup EmailDelivery); ``action_items`` are
    persisted so the *next* run_one_on_one_prep() picks them up as carried actions.
    """

    engineer: str = ""
    date: str = ""
    transcript: str = ""  # raw meeting notes/transcript the lead provided
    email_subject: str = ""
    email_summary: str = ""  # the summary email body (plain text)
    action_items: tuple[str, ...] = ()  # agreed next steps → carried into next prep
    highlights: tuple[str, ...] = ()  # key discussion points recorded
    warnings: tuple[str, ...] = ()
    # Whether the summary email actually went out, so the page states a fact
    # rather than inferring one from the absence of a warning.
    delivery_state: str = ""  # "sent" | "failed" | "not_configured"
    # Carried forward from the prep this 1:1 was run off. This is the artifact
    # that gets emailed to the engineer, so it is the one that most needs to be
    # able to say what it was based on. Empty when there was no recent prep;
    # ``evidence_date`` is that prep's date, so a reader is never shown a scan
    # without being told when it was taken.
    evidence_date: str = ""
    evidence_sources: tuple[str, ...] = ()
    evidence_coverage: tuple[tuple[str, str, str], ...] = ()
    metrics: tuple[PerfMetric, ...] = ()
    evidence_items: tuple[EvidenceGroup, ...] = ()
    # Reader-authored additions; see Annotation. Defaulted so a report stored
    # before browser editing existed still deserializes.
    annotations: tuple[Annotation, ...] = ()


@dataclass(frozen=True)
class SixMonthReview:
    """A performance review synthesised from ~6 months of signals.

    Produced by performance/engine.py:run_six_month_review() from past 1:1s,
    Jira/AzDO delivery history, ceremony history, the lead's notes, and a
    competency framework (bundled default or lead-imported template).
    """

    engineer: str = ""
    period_start: str = ""  # ISO date
    period_end: str = ""
    strengths: tuple[str, ...] = ()
    areas_for_improvement: tuple[str, ...] = ()
    achievements: tuple[str, ...] = ()
    goals: tuple[str, ...] = ()  # goals for the next period
    overall: str = ""  # overall summary paragraph
    framework_used: str = ""  # "default" | imported template name
    warnings: tuple[str, ...] = ()
    # What fed this review, and what did not — see OneOnOnePrep for the shape.
    # A review that cannot say which sources were scanned cannot be argued with.
    evidence_sources: tuple[str, ...] = ()
    evidence_coverage: tuple[tuple[str, str, str], ...] = ()
    metrics: tuple[PerfMetric, ...] = ()
    evidence_items: tuple[EvidenceGroup, ...] = ()
    section_states: tuple[tuple[str, str, str], ...] = ()
    activity: EngineerActivity = EngineerActivity()
    # Reader-authored additions; see Annotation. Defaulted so a report stored
    # before browser editing existed still deserializes.
    annotations: tuple[Annotation, ...] = ()


@dataclass(frozen=True)
class DeliveredItem:
    """One completed ticket that shipped in the reporting period.

    Deterministic evidence (no LLM) — assembled by reporting/activity.py from the
    same recent-activity helpers the standup / performance modes use, filtered to
    the tickets whose status means *done*. The business-friendly narrative in a
    DeliveryReport is grounded in these items.
    """

    key: str = ""  # e.g. "PROJ-123" or "#456"
    title: str = ""
    status: str = ""  # completed status as reported: Done / Closed / Released / ...
    source: str = ""  # "jira" | "azuredevops"
    assignee: str = ""  # who delivered it (best-effort; may be empty)


@dataclass(frozen=True)
class SupportingSignal:
    """One corroborating activity stream from the same reporting period.

    Deterministic reference context (no LLM) — assembled by reporting/context.py
    from the standup code collector and the analysis doc reader. These signals
    corroborate the delivered-ticket story ("backed by 24 merged PRs and 5 doc
    updates"); they are never the report's subject, so only bounded counts and
    sample titles are kept — never bodies.
    """

    kind: str = ""  # "pull_requests" | "commits" | "doc_updates"
    source: str = ""  # "github" | "azuredevops" | "confluence" | "notion"
    count: int = 0
    samples: tuple[str, ...] = ()  # bounded "Title (ref)" strings


@dataclass(frozen=True)
class DeliveryReport:
    """A business-friendly summary of delivered work over a reporting period.

    Produced by reporting/engine.py:run_delivery_report() — one deterministic gather
    of completed tickets + a single LLM "design" call that writes the executive
    narrative, groups outcomes into themes, and picks section emojis. Follows the
    parse → fallback convention: an LLM failure yields a deterministic report (counts
    + item list + generic emojis), never a crash. Rendered to Markdown, HTML, and a
    self-contained HTML slide deck (reporting/presentation.py).

    ``themes`` is a tuple of (theme title, bullet outcomes) pairs; ``metrics`` and
    ``emoji_theme`` are tuple-of-pairs so the whole artifact stays frozen/serializable.
    """

    period_label: str = ""  # "Last sprint" | "Last month (~2 sprints)"
    period_start: str = ""  # ISO date
    period_end: str = ""
    project_name: str = ""
    sprint_names: tuple[str, ...] = ()
    headline: str = ""  # one-line business headline (LLM)
    executive_summary: str = ""  # 1-2 paragraph business narrative (LLM)
    themes: tuple[tuple[str, tuple[str, ...]], ...] = ()  # (theme title, outcome bullets) — LLM grouping
    highlights: tuple[str, ...] = ()  # top business-impact wins (LLM)
    metrics: tuple[tuple[str, str], ...] = ()  # (label, value), e.g. ("Items delivered", "23")
    delivered_items: tuple[DeliveredItem, ...] = ()  # raw completed-ticket evidence (deterministic)
    emoji_theme: tuple[tuple[str, str], ...] = ()  # (slot, emoji) chosen by the LLM, e.g. ("highlights", "🚀")
    supporting_signals: tuple[SupportingSignal, ...] = ()  # code/docs corroboration (deterministic)
    # What production did over the SAME period — its own field and its own
    # heading, never folded into supporting_signals, whose sentence claims
    # corroboration. Empty whenever no ops vendor is connected. Deterministic,
    # and deliberately absent from the design prompt: two incidents means
    # nothing without a baseline this tool does not have.
    ops_signals: tuple[OpsSignal, ...] = ()
    warnings: tuple[str, ...] = ()
    generated_at: str = ""
    # Reader-authored additions; see Annotation. Defaulted so a report stored
    # before browser editing existed still deserializes.
    annotations: tuple[Annotation, ...] = ()


@dataclass(frozen=True)
class ReviewAction:
    """One commitment a Weekly Review proposes, and what became of it.

    ``status`` is marked on the *next* review's ``carried_actions`` (retro's
    carry-forward rule) — the review that created it is an append-only record.
    """

    id: str = ""  # engine-assigned, stable across carry-forward so a surface can mark it
    text: str = ""
    status: str = "pending"  # "pending" | "done" | "dropped" | "carried"
    origin: str = "ai"  # "ai" | "carryover" | "fallback"
    week_label: str = ""  # the review that created it, e.g. "2026-W35"


@dataclass(frozen=True)
class WeeklyReview:
    """A solo developer's review of their own week.

    Produced by solo/engine.py:run_weekly_review() — the Solo world's
    counterpart of a retro plus a performance check-in, over the user's own
    standups, delivered work and sprint plan. One deterministic gather, one
    LLM call for the prose (parse → fallback), and an honest "on track vs your
    plan" line computed without the model. All fields defaulted so a stored
    review keeps deserializing as the shape grows.
    """

    week_label: str = ""  # ISO week, e.g. "2026-W35"
    week_start: str = ""  # ISO date (Monday)
    week_end: str = ""
    project_name: str = ""
    session_id: str = ""
    my_name: str = ""
    standup_dates: tuple[str, ...] = ()
    standup_lines: tuple[str, ...] = ()  # "Mon 2026-08-31: <summary> — blocked: <b>" per standup
    confidence_start: int = 0
    confidence_end: int = 0
    confidence_label: str = ""
    sprint_name: str = ""
    sprint_day: int = 0
    sprint_total_days: int = 0
    delivered_items: tuple[DeliveredItem, ...] = ()
    planned_story_count: int = 0
    plan_status: str = ""  # "on_track" | "at_risk" | "behind" | "no_plan" | "no_data"
    plan_line: str = ""  # the deterministic sentence behind plan_status
    summary: str = ""
    went_well: tuple[str, ...] = ()
    to_change: tuple[str, ...] = ()
    actions: tuple[ReviewAction, ...] = ()  # new this week
    carried_actions: tuple[ReviewAction, ...] = ()  # last review's, with this week's statuses
    warnings: tuple[str, ...] = ()
    generated_at: str = ""
    annotations: tuple[Annotation, ...] = ()


@dataclass(frozen=True)
class AnonymizedOutput:
    """A privacy-masked copy of a mode's generated output, ready for public sharing.

    Produced by anonymize/engine.py:run_anonymize() — a post-processing step that
    takes the already-rendered Markdown any mode's Export button emits and masks the
    sensitive data (personal/team/project names, internal tool names, the company
    identity, URLs/emails/IDs) so a real plan/standup/report can be pasted into a
    README, website, or post. Follows the parse → fallback convention: a deterministic
    seed pass (known company terms from config) runs first and always, then one LLM
    call generalizes the masking; an LLM failure yields the seed-masked text plus a
    warning, never a crash.

    ``replacements`` pairs each original with its neutral placeholder — shown in the
    TUI review screen so the user can spot false positives/negatives, but never
    written to the exported/copied document (that would re-expose the originals).
    """

    anonymized_text: str = ""
    replacements: tuple[tuple[str, str], ...] = ()  # (original, placeholder) — TUI review only
    source_mode: str = ""  # which mode produced the input (for titling/logging)
    warnings: tuple[str, ...] = ()
    generated_at: str = ""


@dataclass(frozen=True)
class RoadmapProject:
    """One candidate project extracted from the team's quarterly roadmap.

    Produced by roadmap/engine.py:run_roadmap_analysis() — the LLM reads the
    ingested roadmap document and proposes concrete projects worth planning.
    ``description`` must be rich enough to pre-seed a planning session's
    Phase A description input on its own (it is pasted there verbatim when the
    user picks "Plan This"). ``size`` maps onto the intake cards:
    "small" → small_project intake, "large" → smart (Large) intake.

    All fields defaulted for backward-compat with serialized history rows.
    """

    name: str = ""
    description: str = ""  # self-contained — pre-seeds planning Phase A
    size: str = "small"  # "small" | "large" → intake_mode small_project | smart
    rationale: str = ""  # why this size + why start it now
    priority: int = 0  # 1-based recommended start order (0 = unranked)
    themes: tuple[str, ...] = ()  # roadmap themes/initiatives it belongs to
    quarter: str = ""  # e.g. "Q3 2026" when detectable


@dataclass(frozen=True)
class RoadmapAnalysis:
    """A full roadmap ingestion + analysis run (source descriptor + projects).

    Produced by roadmap/engine.py following the parse → fallback convention:
    an LLM failure yields a deterministic zero-project analysis carrying the
    warnings, never a crash. Persisted to roadmap_history (roadmap/store.py)
    so return visits to the Roadmap intake card show the last analysis
    immediately with a Re-analyze option.
    """

    source_type: str = ""  # "confluence" | "notion" | "local"
    source_locator: str = ""  # page id / file path
    source_label: str = ""  # page title / file name (display)
    summary: str = ""  # 1-2 sentence roadmap overview (LLM)
    projects: tuple[RoadmapProject, ...] = ()
    warnings: tuple[str, ...] = ()
    generated_at: str = ""  # ISO timestamp
    # Reader-authored additions; see Annotation. Defaulted so a report stored
    # before browser editing existed still deserializes.
    annotations: tuple[Annotation, ...] = ()


# See docs: "Scrum Standards" — prompt quality rating
@dataclass(frozen=True)
class PromptQualityRating:
    """Deterministic quality score for the user's intake questionnaire input.

    Computed purely from QuestionnaireState tracking sets (no LLM call).
    Displayed on the analysis review screen alongside assumptions.

    Scoring: 7 essential questions (Q1-Q4, Q6, Q11, Q15) worth 5 pts each,
    19 other questions worth 2 pts each, plus 1 pt per probed question.
    Answered/extracted = full points, defaulted = 40%, skipped = 0.
    """

    score_pct: int  # 0-100 percentage
    grade: str  # A, B, C, or D
    answered_count: int
    extracted_count: int
    defaulted_count: int
    skipped_count: int
    probed_count: int
    suggestions: tuple[str, ...]
    low_confidence_areas: tuple[str, ...] = ()  # QUESTION_SHORT_LABELS for defaulted essentials


# See docs: "Scrum Standards" — project analysis
@dataclass(frozen=True)
class ArchitectureOption:
    """One candidate architecture for a new project, with its trade-offs.

    Produced by the project_analyzer alongside the rest of the analysis so the
    user sees WHAT was considered, not just what was picked. All fields
    defaulted — old saved sessions have no architecture data at all.
    See docs: "Scrum Standards" — DoD Spike (trade-offs & alternatives)
    """

    name: str = ""  # short label, e.g. "Modular monolith"
    summary: str = ""
    pros: tuple[str, ...] = ()
    cons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArchitectureDecision:
    """The analyzer's architecture recommendation: options, pick, confidence.

    ``pinned_by_constraint`` is True when the decision was already made before
    planning (Q13 constraint, existing repo, team docs, prior art) — a pinned
    decision gets exactly one option and never triggers a validation spike.
    """

    options: tuple[ArchitectureOption, ...] = ()  # 1-3 candidates
    chosen: str = ""  # name of the recommended option
    confidence: str = ""  # "high" | "medium" | "low"
    rationale: str = ""  # why this pick; what evidence would change it
    pinned_by_constraint: bool = False


def architecture_from_dict(raw: object) -> ArchitectureDecision | None:
    """Rebuild an ArchitectureDecision from its asdict() form (or return None).

    Shared by sessions.py and persistence.py so a resumed session's analysis
    carries a real dataclass, not the raw nested dict.
    """
    if not isinstance(raw, dict):
        return raw if isinstance(raw, ArchitectureDecision) else None
    return ArchitectureDecision(
        options=tuple(
            ArchitectureOption(
                name=o.get("name", ""),
                summary=o.get("summary", ""),
                pros=tuple(o.get("pros", ())),
                cons=tuple(o.get("cons", ())),
            )
            for o in raw.get("options", ())
            if isinstance(o, dict)
        ),
        chosen=raw.get("chosen", ""),
        confidence=raw.get("confidence", ""),
        rationale=raw.get("rationale", ""),
        pinned_by_constraint=bool(raw.get("pinned_by_constraint", False)),
    )


@dataclass(frozen=True)
class ProjectAnalysis:
    """Structured synthesis of all 30 intake answers.

    Produced once by the project_analyzer node after the user confirms the
    questionnaire. Downstream nodes (feature_generator, story_writer, sprint_planner)
    read this instead of re-parsing raw conversation history.

    Frozen (immutable) — same pattern as Feature, UserStory, Task, Sprint.
    Uses tuple[str, ...] for list fields (same pattern as Sprint.story_ids).
    """

    project_name: str
    project_description: str
    project_type: str  # "greenfield", "existing codebase", etc.
    goals: tuple[str, ...]
    end_users: tuple[str, ...]
    target_state: str  # What "done" looks like
    tech_stack: tuple[str, ...]
    integrations: tuple[str, ...]
    constraints: tuple[str, ...]
    sprint_length_weeks: int
    target_sprints: int
    risks: tuple[str, ...]
    out_of_scope: tuple[str, ...]
    assumptions: tuple[str, ...]  # Defaults/skipped answers flagged
    # When True, the project is small enough for a single feature instead of 3-6.
    # The analyzer LLM sets this based on project scope (guideline: target_sprints ≤ 2
    # AND goals ≤ 3). Default False so existing projects are unaffected.
    # See docs: "Scrum Standards" — feature generation
    skip_features: bool = False
    # When True, the project is mostly configuration / content / no-code-platform
    # work rather than engineering. Set by reconciling the deterministic
    # repo_signals scan with the analyzer LLM's own read; drives lighter estimation
    # and config-oriented task decomposition downstream. Default False so ordinary
    # engineering projects are unaffected (and old saved sessions still deserialize).
    # See docs: "Scrum Standards" — estimation
    is_low_code: bool = False
    low_code_reason: str = ""  # human-readable why, shown in the analysis panel
    scrum_md_contributions: tuple[str, ...] = ()  # JSON field names enriched by SCRUM.md
    # Deterministic quality rating for the user's intake input. Computed by
    # compute_prompt_quality() in nodes.py from QuestionnaireState tracking sets.
    # None until the project_analyzer node runs. Displayed on the analysis review screen.
    prompt_quality: PromptQualityRating | None = None
    # Architecture options + recommendation (greenfield projects with an open
    # choice get 2-3 candidates; a pinned decision gets exactly one). None when
    # the analyzer produced no architecture data (fallback path, old sessions).
    # See docs: "Scrum Standards" — DoD Spike
    architecture: ArchitectureDecision | None = None


# ---------------------------------------------------------------------------
# agentwatch (Agents family) artifacts — see agentwatch/engine.py
#
# Same conventions as every mode artifact above: @dataclass(frozen=True), every
# field defaulted (older serialized rows keep deserializing), tuples not lists,
# tuple-of-pairs instead of dicts, trailing `annotations` for reader edits.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelUsageRow:
    """One model's token totals and cost within an AgentUsageReport."""

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0  # 5m + 1h writes combined (display total)
    cache_read_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0
    known_pricing: bool = True  # False = priced at the fallback tier


@dataclass(frozen=True)
class AgentUsageBreakdownRow:
    """Cost/volume rollup along one dimension (per project, per source)."""

    key: str = ""
    sessions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class DailyUsagePoint:
    """One day of the usage trend line."""

    date: str = ""  # YYYY-MM-DD
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    sessions: int = 0


@dataclass(frozen=True)
class AgentUsageReport:
    """Cost/token dashboard over locally monitored agent sessions."""

    period_start: str = ""
    period_end: str = ""
    session_count: int = 0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_cache_read_tokens: int = 0
    # Share of total cost priced at the fallback tier (unknown models) — an
    # honesty flag the renderers surface next to the total.
    unknown_model_cost_share: float = 0.0
    pricing_as_of: str = ""
    # "subscription" | "api" | "" (unknown). A subscription's total is an
    # API-equivalent figure, never a bill, and every renderer says so.
    billing_kind: str = ""
    cache_cost_share: float = 0.0  # share of the total that was cache reads + writes
    window_days: int = 0
    by_model: tuple[ModelUsageRow, ...] = ()
    by_project: tuple[AgentUsageBreakdownRow, ...] = ()
    by_source: tuple[AgentUsageBreakdownRow, ...] = ()
    daily_trend: tuple[DailyUsagePoint, ...] = ()
    insights: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    generated_at: str = ""
    annotations: tuple[Annotation, ...] = ()


@dataclass(frozen=True)
class McpServerRecord:
    """One MCP server found in an agent's configuration."""

    name: str = ""
    scope: str = ""  # global | project:<path>
    transport: str = ""  # stdio | http | sse
    target: str = ""  # command or URL (never credentials)
    flags: tuple[str, ...] = ()  # e.g. plain-http, unpinned-package


@dataclass(frozen=True)
class SecurityFix:
    """One concrete action a finding can be answered with — a button, never a sentence."""

    id: str = ""  # guard-hook | guard-hook-pr | settings-edit | mcp-edit-pr | rotate | mark-test-data | dismiss
    kind: str = ""  # write | pr | link | dismiss | manual
    label: str = ""  # the verb on the button, e.g. "Block this in Claude Code"
    target: str = ""  # the file the fix writes, the URL it opens, or the repo it PRs against
    detail: str = ""  # one sentence saying what happens
    scope: str = ""  # user | repo | ""


@dataclass(frozen=True)
class SecurityFinding:
    """One agent-security finding, deterministic and location-referenced."""

    severity: str = "info"  # critical | high | medium | info
    category: str = ""  # settings | mcp | secret | risky_tool
    title: str = ""
    location: str = ""  # file path (never file content)
    line_no: int = 0
    pattern: str = ""  # detector label, e.g. curl-pipe-shell
    detail: str = ""
    remediation: str = ""
    occurrences: int = 1  # matching lines rolled into this row
    key: str = ""  # the dismissal key: category:pattern:location[:context]
    scopes: tuple[str, ...] = ()  # MCP: every scope the same server spec appears in
    verdict: str = ""  # needs-decision | unsure | test-data | handled | info
    verdict_reason: str = ""
    context: str = ""  # command | heredoc | inline-script | write-input | tool-result | prose | user-prompt
    target: str = ""  # the file a write/read context pointed at (a path, never content)
    snippet: str = ""  # ≤120 redacted characters around the match, the span masked
    at: str = ""  # timestamp of the first matching message
    session_id: str = ""
    project_label: str = ""
    sessions: int = 1  # distinct sessions rolled into this row
    fixes: tuple[SecurityFix, ...] = ()


@dataclass(frozen=True)
class SecurityIssue:
    """One pattern across every place it fired — what the security page lists."""

    id: str = ""  # category:pattern
    category: str = ""
    pattern: str = ""
    title: str = ""  # plain words: "An agent ran curl piped into a shell"
    why: str = ""  # one paragraph on why the pattern matters
    verdict: str = ""  # the worst verdict among its findings
    severity: str = ""  # the worst severity among its findings
    signals: int = 0  # matched lines
    sessions: int = 0
    files: int = 0
    last_seen: str = ""  # YYYY-MM-DD
    finding_keys: tuple[str, ...] = ()
    fixes: tuple[SecurityFix, ...] = ()


@dataclass(frozen=True)
class AgentSecurityReport:
    """Security posture of local agent usage (settings, MCP, secrets, tools)."""

    scan_date: str = ""
    posture: str = ""  # good | needs-attention | at-risk
    sessions_scanned: int = 0
    files_scanned: int = 0
    secrets_found: int = 0
    findings: tuple[SecurityFinding, ...] = ()
    mcp_servers: tuple[McpServerRecord, ...] = ()
    settings_flags: tuple[str, ...] = ()
    summary: str = ""
    recommendations: tuple[str, ...] = ()
    finding_keys: tuple[str, ...] = ()  # every undismissed key, info included — what the next run diffs against
    new_findings: tuple[str, ...] = ()  # keys not in the previous saved report
    resolved_findings: tuple[str, ...] = ()  # keys the previous report had and this one lacks
    dismissed_count: int = 0
    hidden_info_count: int = 0
    posture_reason: str = ""
    pattern_totals: tuple[tuple[str, str], ...] = ()  # (pattern, "N matches across M files")
    issues: tuple[SecurityIssue, ...] = ()  # the page's rows: one per pattern, worst verdict first
    verdict_counts: tuple[tuple[str, int], ...] = ()  # (verdict, findings) in display order
    verdict_line: str = ""  # the deterministic one-sentence answer to "do I need to do anything?"
    warnings: tuple[str, ...] = ()
    generated_at: str = ""
    annotations: tuple[Annotation, ...] = ()


@dataclass(frozen=True)
class WasteLineItem:
    """One recoverable-spend mechanism sized by the advisor's transcript audit."""

    mechanism: str = ""  # identical-repeat | subset-containment | write-readback | stale-reread | line-number-overhead
    label: str = ""
    calls: int = 0
    content_bytes: int = 0  # UTF-8 bytes of Read tool_result content
    est_tokens: int = 0  # ≈ content_bytes / 4
    est_usd: float = 0.0  # est_tokens priced at the window's blended input rate
    share_of_read_bytes: float = 0.0
    # True = summed into the recoverable headline; False = sized but reported
    # as context only (e.g. stale re-reads need staleness-aware handling).
    recoverable: bool = True
    note: str = ""


@dataclass(frozen=True)
class VolatileFileSignal:
    """Volatile-shaped content counts for one prompt-prefix file (CLAUDE.md…)."""

    location: str = ""  # file path (never file content)
    counts: tuple[tuple[str, str], ...] = ()  # (label, count as str)
    total: int = 0


@dataclass(frozen=True)
class AgentAdvisorReport:
    """Recoverable agent spend + prompt-cache health, audited from local sessions."""

    period_start: str = ""
    period_end: str = ""
    session_count: int = 0
    files_audited: int = 0
    total_cost_usd: float = 0.0  # the window's estimated spend (context for the headline)
    read_calls: int = 0
    read_bytes: int = 0
    tool_bytes_total: int = 0
    recoverable_usd: float = 0.0  # sum of the recoverable line items
    recoverable_share: float = 0.0  # of total_cost_usd
    effective_input_rate_per_mtok: float = 0.0  # blended $/Mtok used to price waste
    unknown_rate_share: float = 0.0  # share of input tokens priced at the fallback tier
    pricing_as_of: str = ""
    line_items: tuple[WasteLineItem, ...] = ()
    residency_median: int = 0  # assistant turns a Read stays in context
    residency_p90: int = 0
    gaps_over_5m: int = 0  # cache-death windows (inter-message gaps past the TTL)
    gaps_over_1h: int = 0
    sessions_with_gap: int = 0
    volatile_signals: tuple[VolatileFileSignal, ...] = ()
    alignment_score: int = 100  # 0-100; lower = more volatile content in prefix files
    insights: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    generated_at: str = ""
    annotations: tuple[Annotation, ...] = ()


# Provenance audit artifacts (provenance/engine.py). The chain itself lives in
# yeaboi.provenance; these are the render-ready views of it — deterministic,
# no LLM anywhere in the pipeline, because a trust report that needs a model
# to read the tamper log would undermine the thing it reports on.
@dataclass(frozen=True)
class ProvenanceDecisionRow:
    """One decision record, summarised for display. Counts and keys only —
    the row never carries more than the chain already stores."""

    entity_id: str = ""
    entity_type: str = ""  # practice-signal | blocker-signal | confidence | conflict | …
    record_kind: str = "decision"  # decision | invalidation
    agent_id: str = ""  # the rule, model, or person behind it
    role: str = ""  # generator | suppressor | invalidator | …
    timestamp: str = ""
    detail: str = ""
    inputs: tuple[str, ...] = ()  # the evidence keys the decision rests on
    sequence_id: int = 0


@dataclass(frozen=True)
class ProvenanceAuditReport:
    """The chain's health plus what it recorded in the window.

    ``chain_valid`` covers the WHOLE chain, not the window: a tamper verdict
    scoped to recent rows would miss exactly the edits it exists to catch.
    """

    generated_at: str = ""
    window_days: int = 30
    chain_valid: bool = True
    total_records: int = 0
    window_records: int = 0
    records_by_type: tuple[tuple[str, int], ...] = ()  # (entity_type, count), whole chain
    recent: tuple[ProvenanceDecisionRow, ...] = ()  # newest first, capped
    # (sequence_id, entity_id, reason) per verification failure — reason is
    # "checksum_mismatch" (edited row), "chain_break" (deleted/renumbered),
    # or "truncated_tail" (newest rows removed; the walk fell short of the
    # head anchor).
    breaks: tuple[tuple[int, str, str], ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProvenanceTrace:
    """The "why" trail behind one entity: its records plus the latest record
    behind each of its inputs, breadth-first."""

    entity_id: str = ""
    found: bool = False
    records: tuple[ProvenanceDecisionRow, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PriorArtRef:
    """An existing team repository accepted as reference material for a plan.

    Produced by the prior-art step in the intake (see ``agent/prior_art.py``),
    carried into the analyzer and feature prompts, and rendered in the intake
    summary and the exports. Deliberately thin: the candidate that produced it
    knows a score and a last-push date, and neither belongs in a plan.

    # See docs: "Project Intake Questionnaire" — prior art
    """

    key: str = ""
    name: str = ""
    url: str = ""
    platform: str = ""
    pitch: tuple[str, ...] = ()
    stack: tuple[str, ...] = ()


def prior_art_to_dicts(refs) -> list[dict]:
    """Serialise accepted prior art for either persistence layer."""
    out: list[dict] = []
    for ref in refs or ():
        if not isinstance(ref, PriorArtRef):
            continue
        out.append(
            {
                "key": ref.key,
                "name": ref.name,
                "url": ref.url,
                "platform": ref.platform,
                "pitch": list(ref.pitch),
                "stack": list(ref.stack),
            }
        )
    return out


def prior_art_refs(keys: list[str] | None) -> tuple:
    """Turn caller-supplied repository keys into PriorArtRefs.

    The name falls back to the key's slug half so an export still reads as a
    repository rather than a bare identifier. No lookup: a headless caller
    naming a repository is asserting it is relevant, and failing the run
    because the estate has not been scanned would be worse than taking them
    at their word.
    """
    refs = []
    for raw in keys or ():
        key = str(raw or "").strip().lower()
        if not key:
            continue
        name = key.split(":", 1)[1] if ":" in key else key
        platform = key.split(":", 1)[0] if ":" in key else ""
        refs.append(PriorArtRef(key=key, name=name, platform=platform))
    return tuple(refs)


def prior_art_from_dicts(rows) -> tuple[PriorArtRef, ...]:
    """Rebuild accepted prior art from JSON.

    Lists become tuples (the dataclass is frozen and declares tuples), and a
    malformed row is skipped rather than failing the resume — losing one
    reference is recoverable, losing the session is not.
    """
    out: list[PriorArtRef] = []
    for row in rows or ():
        if not isinstance(row, dict):
            continue
        out.append(
            PriorArtRef(
                key=str(row.get("key", "") or ""),
                name=str(row.get("name", "") or ""),
                url=str(row.get("url", "") or ""),
                platform=str(row.get("platform", "") or ""),
                pitch=tuple(row.get("pitch") or ()),
                stack=tuple(row.get("stack") or ()),
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Ship artifacts (the supervised plan-item → PR pipeline)
# ---------------------------------------------------------------------------


# Run lifecycle. `awaiting_approval` is the pause the human gate owns; the
# resolve and resume transitions are independent CAS updates in ship/store.py.
SHIP_STATUSES = (
    "planned",
    "running",
    "awaiting_approval",
    "approved",
    "rejected",
    "failed",
    "cancelled",
)


@dataclass(frozen=True)
class ShipPhase:
    """One pipeline phase's outcome, for the progress screen and the record."""

    name: str = ""  # setup | implement | validate | gate | finalize
    status: str = ""  # completed | failed | skipped
    detail: str = ""
    duration_s: float = 0.0


@dataclass(frozen=True)
class ShipValidation:
    """What the deterministic validation step ran and what it said.

    ``configured`` False means no command was available — a visible state the
    approval screen must show, never a silent pass.
    """

    configured: bool = False
    command: str = ""
    passed: bool = False
    exit_code: int = -1
    output_tail: str = ""


@dataclass(frozen=True)
class ShipRun:
    """One supervised plan-item → PR run, end to end.

    Frozen like every artifact: the store persists status transitions, and the
    engine returns a fresh copy per phase. The gate fields mirror archon's
    protocol — ``gate_resolution`` is independent of ``status`` so resolve and
    resume cannot race each other.
    """

    run_id: str = ""
    # The plan item this run implements, at whichever level it lives. Stored
    # runs written before ship could target an epic or a task carry it under
    # the old key ``story_id``; the store reads both and keeps emitting the old
    # one, so the MCP payload and the plugin skill's contract are unchanged.
    item_id: str = ""
    level: str = "story"  # "epic" | "story" | "task"
    session_id: str = ""  # the *planning* session the item came from
    agent_session_id: str = ""  # the coding agent's own session (transcript key)
    repo: str = ""
    branch: str = ""
    worktree: str = ""
    base_sha: str = ""
    # The branch this run's PR targets, "" for the repo default. Persisted
    # because only the artifact knows it: resuming a stranded batch member has
    # nothing else to rebuild its stack parent from.
    pr_base: str = ""
    status: str = "planned"  # one of SHIP_STATUSES
    phases: tuple[ShipPhase, ...] = ()
    validation: ShipValidation = ShipValidation()
    diff_stat: str = ""  # `git diff --stat` summary shown at the gate
    diff_text: str = ""  # the capped patch itself — the gate approves a diff, not a file count
    cost_usd: float = 0.0
    transcript_findings: tuple[tuple[str, str, str], ...] = ()  # (kind, severity, label)
    transcript_path: str = ""
    pr_url: str = ""
    gate_resolution: str = ""  # "" (open) | approved | rejected
    gate_comment: str = ""  # the approver's words; lands in the PR body
    rejection_count: int = 0
    # Batch membership: N runs from one epic launched as "one PR per story",
    # each stacked on the branch before it. "" and 0 for a single run.
    batch_id: str = ""
    # The epic the batch was launched from. A member's own item_id is its
    # story, so without this nothing can find a batch by what the user picked.
    batch_item_id: str = ""
    batch_index: int = 0  # 1-based position within the batch
    batch_total: int = 0
    # The pid driving this run, stamped at record time and re-stamped on resume.
    # A run left at the gate is resumable only once that process is gone, so a
    # reused pid reads as "still owned" and refuses — the harmless direction.
    owner_pid: int = 0
    created_at: str = ""
    updated_at: str = ""
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Ceremony artifacts (the team's recurring runs — see ceremonies/)
# ---------------------------------------------------------------------------


# What one fired run did. Two of the four are *not* failures: a run the guards
# declined is a decision, and recording it as such is the difference between
# "nothing happened" and "something stopped it".
CEREMONY_OUTCOMES = (
    "ok",
    "failed",
    "skipped_stale",  # fired far enough after its slot that the output would mislead
    "skipped_over_cap",  # this month's spend on this ceremony is already at the cap
    "skipped_paused",  # a job fired for a ceremony the store says is paused
    "skipped_once",  # somebody asked for this one occurrence off
)


@dataclass(frozen=True)
class Ceremony:
    """One recurring run of one mode, declared once and fired by the OS.

    ``name`` is the key every surface addresses, and it also becomes part of a
    launchd label, a plist filename and a crontab marker — so it is whitelisted
    (``ceremonies.store.valid_name``) rather than trusted.

    ``args`` is ordered key/value *strings*; the catalog coerces them to the
    engine's real types. A dict would be the obvious choice and is the wrong
    one: this artifact is frozen, persisted as JSON and compared field-by-field
    in tests, and a tuple of pairs round-trips through all three unchanged.
    """

    session_id: str = ""
    name: str = ""
    mode: str = ""  # a ceremonies.catalog key
    args: tuple[tuple[str, str], ...] = ()
    weekdays: str = "1-5"  # the scheduler's spec form: "1-5", "1,3,5"
    at: str = "09:00"  # local time the ceremony is FOR (and when the job fires)
    channels: tuple[str, ...] = ("terminal",)
    enabled: bool = True
    stale_after_min: int = 120  # 0 disables the staleness guard
    monthly_cap_usd: float = 0.0  # 0 = uncapped
    # One occurrence off, as the ISO date of the slot being skipped — never a
    # bool. launchd coalesces missed calendar intervals, so a flag can be
    # consumed by a fire arriving the following morning *for yesterday's slot*,
    # burning the skip on the occurrence the user already saw. A date says
    # which one they meant, and the guard clears it once that slot has passed.
    skip_next: str = ""
    last_fired_at: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class CeremonyRun:
    """One fired run, recorded whatever happened to it.

    A scheduled run that fails at 06:00 with nobody watching is how this whole
    feature dies quietly, so ``error`` is a column rather than a log line: the
    fact of a failure is useless without the reason for it.
    """

    ceremony: str = ""
    session_id: str = ""
    fired_at: str = ""
    outcome: str = ""  # one of CEREMONY_OUTCOMES
    scheduled: bool = False  # fired by the OS job, not by a human
    duration_s: float = 0.0
    cost_usd: float = 0.0
    delivery: tuple[tuple[str, bool], ...] = ()  # (channel, delivered)
    detail: str = ""  # the headline the run produced, for the history screen
    error: str = ""


@dataclass(frozen=True)
class Dispatch:
    """What a finished ceremony has to say, in a form every channel can send.

    Delivery used to be typed on ``StandupReport``, which is why nothing else
    could be delivered — the agent standup ended up re-implementing the Slack
    POST rather than duck-typing another mode's report. This is the mode-neutral
    payload that replaced it: ``summary`` is the one-liner a desktop
    notification can hold, ``body`` is the plaintext Slack/email/terminal
    version.

    ``subject`` exists because an email subject line is not a notification
    title, even when they carry the same two facts. People build inbox filters
    and threading on subject lines, so a mode that has been mailing one shape
    for releases does not get to change it because a desktop banner reads better
    shorter. Empty means "use the title", which is right for every mode with no
    opinion.
    """

    title: str = ""
    summary: str = ""
    body: str = ""
    subject: str = ""


@dataclass(frozen=True)
class MessageRef:
    """Where a delivered message landed, so a later read can find it again.

    An incoming webhook answers a POST with the literal body ``ok`` and no
    message id, which is the whole reason the two-way lane needs a bot token:
    ``chat.postMessage`` replies with ``(channel, ts)``, and that pair is the
    only handle Slack ever gives you on a message you posted. Without it, a
    reaction can never be attributed back to the run that caused it.

    ``kind`` rather than a Slack-only shape because email could grow a
    Message-ID and be answered the same way; nothing else has a durable address
    today, and those channels simply return no ref.
    """

    kind: str = "slack"
    channel: str = ""  # Slack channel id
    ts: str = ""  # Slack message ts — the identity
    permalink: str = ""


# ---------------------------------------------------------------------------
# Niko — the global assistant's records
# ---------------------------------------------------------------------------

#: What Niko is allowed to do. Read and point; never change. The tool registry
#: (yeaboi/niko/tools.py) is built to match, and the parity test asserts it.
NIKO_READ_ONLY = True


@dataclass(frozen=True)
class NikoToolCall:
    """One tool Niko reached for, and how it went.

    ``result`` is the JSON-able payload the tool returned, kept so a replayed
    conversation shows the same cards it showed live — the platform this was
    ported from dropped tool blocks on replay and its resumed conversations
    forgot what they had looked up.
    """

    name: str = ""
    arguments: dict = field(default_factory=dict)
    ok: bool = True
    result: object = None
    error: str = ""


@dataclass(frozen=True)
class NikoMessage:
    """One turn in a Niko conversation, as stored.

    ``route`` is the desktop route (or TUI mode) the question was asked from.
    It is a snapshot, not a live value: "what was I looking at when I asked
    this?" is the whole reason the answer reads the way it does.
    """

    id: str = ""
    conversation_id: str = ""
    role: str = "user"  # "user" | "assistant"
    content: str = ""
    tool_calls: tuple[NikoToolCall, ...] = ()
    route: str = ""
    created_at: str = ""


@dataclass(frozen=True)
class NikoConversation:
    """A Niko thread. ``title`` is written once, from the opening question."""

    id: str = ""
    title: str = ""
    created_at: str = ""
    updated_at: str = ""
    archived: bool = False
    message_count: int = 0


@dataclass(frozen=True)
class NikoAnswer:
    """One finished turn — what every surface renders.

    ``route`` is Niko's navigation suggestion (the ``navigate`` tool), empty
    when it did not offer one. It is a suggestion: the desktop pushes it, the
    terminal prints it, and neither is obliged.
    """

    conversation_id: str = ""
    text: str = ""
    tool_calls: tuple[NikoToolCall, ...] = ()
    route: str = ""
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Questionnaire state (mutable — updated incrementally by intake node)
# ---------------------------------------------------------------------------


@dataclass
class QuestionnaireState:
    """Tracks progress through the 30-question intake flow.

    The questionnaire has 7 phases (see QuestionnairePhase). As the intake node
    runs, it updates current_question, records answers, and optionally marks
    questions as skipped (e.g. when the initial project description already
    covers them, or the user explicitly skips).

    Both answered and skipped questions count toward progress so the progress
    bar reflects true forward movement through the questionnaire.
    """

    current_question: int = 1
    answers: dict[int, str] = field(default_factory=dict)
    # Tracks questions the agent auto-skipped (already answered in the initial
    # description) or the user explicitly skipped. Needed for adaptive skip
    # logic — see TODO Phase 4: "Implement adaptive skip logic".
    skipped_questions: set[int] = field(default_factory=set)
    # Stores LLM-extracted answers from the initial project description as
    # confirmable suggestions. Instead of silently skipping extracted questions,
    # each is presented with its suggestion so the user can press Enter/Y to
    # confirm or type a different answer. Cleared per-question once confirmed.
    # See docs: "Project Intake Questionnaire" — adaptive skip logic
    suggested_answers: dict[int, str] = field(default_factory=dict)
    # Tracks which questions have already been probed with a follow-up.
    # Max 1 follow-up per question — if the answer is still vague after
    # probing, accept it and move on.
    # See docs: "Project Intake Questionnaire" — follow-up probing
    probed_questions: set[int] = field(default_factory=set)
    # Tracks which questions used a sensible default (user said "skip" / "I don't
    # know"). Needed to flag assumptions in the intake summary. Defaulted questions
    # have an entry in `answers` (the default value) so they don't affect progress
    # calculation — progress counts answer keys and skipped questions as usual.
    # See docs: "Project Intake Questionnaire" — adaptive behavior
    defaulted_questions: set[int] = field(default_factory=set)
    completed: bool = False
    # True after the last question is answered but before the user confirms
    # the summary. The intake node re-shows the summary until the user types
    # "confirm" (or similar). Only then does completed flip to True.
    # See docs: "Project Intake Questionnaire" — confirmation gate
    awaiting_confirmation: bool = False
    # Tracks which question the user is currently editing (via "Q6" or "edit Q6"
    # from the confirmation summary). Separate from current_question to avoid
    # corrupting the forward-progress model. None when not editing.
    # See docs: "Project Intake Questionnaire" — edit flow
    editing_question: int | None = None
    # Intake mode — controls how many questions are shown interactively.
    # The legacy 30-question "standard" flow has been retired as a user-facing
    # path: project_intake coerces any "standard" value to "smart" at its first
    # invocation. The dataclass default is left as "standard" only so directly
    # constructed states (in tests / shared subsequent-call helpers) keep their
    # historical value; production always sets this from _intake_mode.
    # See docs: "Project Intake Questionnaire" — smart intake
    intake_mode: str = "standard"  # coerced to "smart" | "quick" | "small_project"
    # Transient: set when the user switches Small project → Large at the
    # analysis review. On the next project_intake pass we skip answer-recording
    # and ask the remaining Large-mode essentials instead (answers are preserved).
    _reopen_for_epic: bool = False
    # Transient repository-scan carry-over. project_intake runs repo_signals once
    # at first invocation (broadened to configured repos) and stashes the raw
    # scan + deterministic low-code verdict here; project_analyzer reuses them so
    # the repo isn't scanned twice. Not serialized — a resumed session re-scans.
    # See docs: "Project Intake Questionnaire" — smart intake
    _repo_context: str = ""
    _repo_low_code: bool = False
    _repo_low_code_reason: str = ""
    # Tracks which question numbers had answers auto-applied from the
    # initial description (via LLM extraction). Used for provenance
    # markers in the intake summary ("from your description").
    extracted_questions: set[int] = field(default_factory=set)
    # Transient: when asking a merged question (e.g. Q3+Q4 combined),
    # this tracks which question numbers the current prompt covers.
    # Cleared after the answer is recorded.
    _pending_merged_questions: list[int] = field(default_factory=list)
    # Transient: LLM-generated choices for follow-up probes on vague answers.
    # Maps question number → tuple of 2-4 option strings. The REPL renders
    # these as a numbered menu so the user can pick instead of typing.
    # Cleared after the follow-up answer is recorded (same lifecycle as probed_questions).
    # See docs: "Project Intake Questionnaire" — follow-up probing
    _follow_up_choices: dict[int, tuple[str, ...]] = field(default_factory=dict)
    # Transient: bank holiday count auto-detected during Q27 processing.
    # Stored here so _extract_capacity_deductions can read it at confirmation time
    # and populate capacity_bank_holiday_days in ScrumState.
    # See docs: "Scrum Standards" — capacity planning
    _detected_bank_holiday_days: int = 0
    # Transient: structured holiday data from get_bank_holidays_structured().
    # Each dict has {"date": date, "name": str, "weekday": str}.
    # Used by _compute_per_sprint_velocities to map holidays to sprint windows
    # so only the sprints that contain bank holidays get reduced velocity.
    _detected_bank_holidays: list[dict] = field(default_factory=list)
    # Transient: user's velocity override from the confirmation gate velocity
    # accept/override choice menu. None means the computed velocity was accepted.
    # See docs: "Scrum Standards" — capacity planning
    _velocity_override: int | None = None
    # Transient: True when the user picked "Override" from the velocity choice
    # menu and we're waiting for them to enter a custom number.
    _awaiting_velocity_input: bool = False
    # Transient: per-developer velocity from Jira (team avg / team size).
    # Stored so that Q6 changes at the confirmation gate trigger recomputation
    # of the feature velocity (per_dev × feature_team_size).
    # See docs: "Scrum Standards" — capacity planning
    _jira_per_dev_velocity: float | None = None
    # Transient: PTO/planned leave entries collected via the leave sub-loop.
    # Each entry: {"person": str, "start_date": str (ISO), "end_date": str (ISO), "working_days": int}
    # PTO is per-person (unlike bank holidays which affect the whole team).
    # See docs: "Scrum Standards" — capacity planning
    _planned_leave_entries: list[dict] = field(default_factory=list)
    # Transient: True when in the PTO collection sub-loop after Q28.
    _awaiting_leave_input: bool = False
    # Transient: current stage of the leave sub-loop state machine.
    # Stages: "ask", "person", "start", "end", "more?"
    _leave_input_stage: str = ""
    # Transient: partial entry being built during the leave sub-loop.
    _leave_input_buffer: dict = field(default_factory=dict)
    # Transient prior-art sub-loop, modelled on the PTO sub-loop above. After
    # the questionnaire and before the confirmation summary, a greenfield
    # project is offered the team's own repositories as reference material and
    # the user picks the relevant ones in one batched answer.
    # Stages: "" (not started), "ask", "empty", "done". ("reason" was the old
    # per-repo rejection-reason stage — read-tolerated from sessions serialized
    # by older builds; the handler converts it back to "ask".)
    # See docs: "Project Intake Questionnaire" — prior art
    _prior_art_stage: str = ""
    # Transient: the shortlist on offer, each a RepoCandidate as a dict.
    _prior_art_candidates: list[dict] = field(default_factory=list)
    # Transient: kept for serialization compatibility with the old one-at-a-time
    # loop; no longer advanced (the whole shortlist shows at once).
    _prior_art_index: int = 0
    # Transient: accepted candidates, promoted to ScrumState.prior_art on confirm.
    _prior_art_accepted: list[dict] = field(default_factory=list)
    # Transient: banned candidates ("never suggest again") as {"key", "name",
    # "reason"}, written to the global feedback ledger when the sub-loop ends.
    # Reason is always "" now — the free-text "why?" stage is gone; a candidate
    # merely left unticked lands in neither list and is never written down.
    _prior_art_rejected: list[dict] = field(default_factory=list)
    # Transient: why the shortlist was empty, so the card can say which of
    # "no profile" / "profile too old" / "nothing matched" happened. Going
    # quiet would leave the user unable to tell a gap from a verdict.
    _prior_art_empty_reason: str = ""
    # Transient: the analysis profile the prior-art scan reads its repository
    # estate from. Stashed by project_intake at first invocation, the same way
    # _repo_context is, so the summary path can reach it without threading
    # graph state through eleven call sites.
    _analysis_profile_id: str = ""
    # Transient: whether this run's context toggles allow analysis reads at all.
    # A blank _analysis_profile_id still auto-detects from configured trackers,
    # so the prior-art step needs an explicit off switch, not just an empty id.
    _analysis_enabled: bool = True
    # Transient: active sprint number from Jira (e.g. 104). Used to compute
    # the start date offset when the user selects a future sprint (e.g. Sprint 107).
    # Set during Q27 processing; None when Jira is not configured.
    _active_sprint_number: int | None = None
    # Transient: the board's open sprint targets offered in small-project Q27
    # ("add to an existing sprint or create a new one?"). Maps the real board
    # sprint/iteration NAME to its external id (Jira sprint id / AzDO iteration
    # path). Consumed at confirmation to fill target_sprint_* on ScrumState.
    _sprint_target_options: dict[str, str] = field(default_factory=dict)
    # Transient: active sprint start date from Jira (ISO string, e.g. "2026-03-02").
    # Used with _active_sprint_number to compute exact start dates for future sprints.
    _active_sprint_start_date: str | None = None
    # Transient: total Jira org team size (unique assignees from closed sprints).
    # Used to cap the "increase team" recommendation so we never suggest more
    # engineers than exist on the board. Set even when velocity is zero.
    _jira_org_team_size: int | None = None
    # Transient: True when Q6 is set up as a team member multi-select
    # (from analysis contributor_stats). When set, Q6 answer is parsed
    # as comma-separated member names and velocity is recalculated.
    _q6_member_select: bool = False
    # Transient: tracks which question numbers were auto-populated from SCRUM.md
    # content (as opposed to the user's typed description). Used for provenance
    # markers in the intake preamble ("N from SCRUM.md").
    _scrum_md_questions: set[int] = field(default_factory=set)
    # Unified answer provenance — maps question number to AnswerSource value.
    # Populated alongside the existing tracking sets (extracted_questions,
    # defaulted_questions, probed_questions) for backward compatibility.
    # See docs: "Project Intake Questionnaire" — answer confidence signalling
    answer_sources: dict[int, str] = field(default_factory=dict)
    # Transient: preferred tracker for velocity/sprint data when both Jira and
    # Azure DevOps are configured. Set by the user at the start of intake via
    # a choice prompt. Values: "jira", "azdevops", or "" (not yet chosen).
    # When only one tracker is configured, this is ignored.
    _preferred_tracker: str = ""
    # Transient: True when waiting for the user to pick a tracker (before Q1).
    _awaiting_tracker_choice: bool = False

    @property
    def current_phase(self) -> QuestionnairePhase:
        """Return the phase that the current question belongs to."""
        for phase, (start, end) in PHASE_QUESTION_RANGES.items():
            if start <= self.current_question <= end:
                return phase
        return QuestionnairePhase.PREFERENCES  # clamp to last phase

    @property
    def progress(self) -> float:
        """Return completion ratio from 0.0 to 1.0.

        Both answered and skipped questions count toward progress so the
        progress bar reflects true forward movement through the questionnaire.
        Uses a union of answer keys and skipped questions to avoid double-
        counting questions that were auto-extracted (present in both sets).
        """
        completed_questions = set(self.answers.keys()) | self.skipped_questions
        return len(completed_questions) / TOTAL_QUESTIONS


# ---------------------------------------------------------------------------
# ScrumState TypedDict (LangGraph graph state)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Custom state reducers
# ---------------------------------------------------------------------------


def _merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts, with b's values overwriting a's on key collisions.

    Used as the reducer for Jira key-mapping dicts in ScrumState so that each
    node can return only the new mappings it created (a partial dict) and
    LangGraph merges them into the running total — the same append-semantics
    pattern that operator.add provides for list fields like `features` and `stories`.
    # See docs: "Memory & State" — reducers, Annotated fields
    """
    return {**a, **b}


class _RequiredState(TypedDict):
    """Keys that must always be present in the state."""

    messages: Annotated[list[BaseMessage], add_messages]


class ScrumState(_RequiredState, total=False):
    """Full yeaboi graph state.

    `messages` is required (inherited); everything else is optional and
    populated progressively as the agent runs through its nodes.
    """

    # Project metadata
    project_name: str
    project_description: str

    # Questionnaire
    questionnaire: QuestionnaireState
    # Intake mode — passed from REPL to the intake node on first invocation.
    # Stored as a ScrumState field so LangGraph doesn't strip it.
    # See docs: "Project Intake Questionnaire" — smart intake
    _intake_mode: str

    # Chat-planning presentation state (TUI live chat only; never read by
    # graph nodes). The greeting + size exchange happens BEFORE the first
    # graph invocation and must stay out of `messages` — project_intake reads
    # messages[0] as the project description. _chat_preamble records that
    # exchange for transcript rebuild/export: [{"role": "user"|"ai", "text": str}].
    _chat_greeting_done: bool
    _chat_preamble: list[dict]
    # /finish fast mode. The chat owns intake only, so in production this now
    # spans a single deterministic turn ("defaults all" → the summary) and the
    # driver pops it at the hand-off to the card pipeline — whose review gates
    # never read it, and which stops at every one of them. It stays a state
    # field, not a driver attribute, because the end-to-end chat path
    # (stop_after_intake=False) still auto-accepts reviews while it is set.
    _chat_fast_forward: bool
    # Which prior-art candidate the chat's carousel is previewing (0-based).
    # Presentation-only and driver-owned: the card renderer reads it, no graph
    # node ever does, and the driver pops it when the batch submits or a size
    # switch resets the sub-loop. Serializes harmlessly mid-browse.
    _prior_art_preview: int

    # Declared so they survive an invoke: LangGraph returns only the keys the
    # state schema names, and drops the rest of the input dict.
    # See docs: "Memory & State" — state schema
    #
    # The epic reformat ran and its card was shown (both planning surfaces
    # gate the feature stage on it).
    _epic_reviewed: bool
    # The capacity warning the user answered — {"text", "recommended"} — kept
    # for the recap.
    _capacity_warning: dict
    # What other sessions this run may read, as a ContextScope's JSON, and the
    # free-text project label it is filed under. Carried, not yet consumed.
    context_scope: str
    project_label: str
    # The connection keys this plan may consult (jira, github, …). Absent =
    # unrestricted; [] = none. Not ProjectAnalysis.integrations, which is what
    # the analysed product integrates with. Read by the agent node's tool
    # binding, the tool node's guard and the analyzer's direct reads.
    # See docs: "Memory & State" — StateGraph keeps only declared keys
    session_integrations: list[str]

    # Project analysis — structured synthesis of intake answers.
    # Set once by project_analyzer node; no reducer needed (single value).
    project_analysis: ProjectAnalysis

    # Artifacts (append-semantics via operator.add)
    features: Annotated[list[Feature], operator.add]
    stories: Annotated[list[UserStory], operator.add]
    tasks: Annotated[list[Task], operator.add]
    sprints: Annotated[list[Sprint], operator.add]

    # Custom DoD items from team analysis — overrides DOD_ITEMS when set.
    # Empty tuple means use the default 7 items.
    custom_dod_items: tuple[str, ...]

    # Acceptance-criteria style the plan was generated in ("gwt" | "bullets").
    # Written by story_writer after resolve_ac_style() so every later consumer
    # (exports, editor, re-runs) matches how the stories were actually written.
    # "" = not yet resolved. See docs: "Scrum Standards" — Acceptance Criteria
    ac_format: str
    # The team's learned ticket description section headings (from the analysis
    # profile's naming conventions). Exports map them onto our canonical blocks
    # via map_template_headings(). Empty = use the default headings.
    ticket_template_sections: list[str]

    # Selected team members from analysis profile (names from contributor_stats).
    # When set, velocity is calculated from these specific members' per_sprint values.
    # Empty tuple = no specific members selected (use total team velocity).
    selected_team_members: tuple[str, ...]

    # Team / planning knobs
    team_size: int
    sprint_length_weeks: int
    velocity_per_sprint: int
    target_sprints: int
    # Analysis profile selected by user in planning mode profile picker.
    # When set, intake auto-fills Q6/Q8/Q9 from the profile and nodes
    # use this profile for team calibration. Empty string = no profile selected.
    analysis_profile_id: str
    # True for a Solo-world run: the intake plans for one developer (team
    # questions default, no member picker). Seeded by the caller and declared as
    # a ScrumState field so LangGraph doesn't strip it — undeclared keys never
    # reach a node.
    # See docs: "Memory & State" — StateGraph keeps only declared keys
    solo: bool
    # Existing team repositories the user accepted as prior art for this plan.
    # Only greenfield projects are offered them. Feeds the analyzer and feature
    # prompts, and renders as a Prior Art section in the summary and exports.
    # See docs: "Project Intake Questionnaire" — prior art
    prior_art: tuple[PriorArtRef, ...]
    # Starting sprint number — set by the sprint_selector node after fetching
    # the active Jira sprint and asking the user which sprint to plan for.
    # e.g. if active sprint is "Sprint 104" and user picks next → 105.
    # When 0 (default), sprint_planner uses generic "Sprint 1, Sprint 2, ...".
    # See docs: "Scrum Standards" — sprint planning
    starting_sprint_number: int

    # Small-project sprint targeting — "add to an existing sprint" support.
    # sprint_target_mode is "" (create sprints, the default), "existing"
    # (assign the plan's stories to an existing tracker sprint; the syncs then
    # never create a sprint), or "backlog" (create the stories and assign them
    # to nothing — the syncs skip sprints entirely and unassigned items sit in
    # the tracker's backlog). target_sprint_name is the real board sprint /
    # iteration name ("PSOT Sprint 104"); target_sprint_external_id is the Jira
    # sprint id or AzDO iteration path, "" when only the name is known (the
    # sync resolves it by name among active/future sprints at execution time).
    # See docs: "Scrum Standards" — sprint planning
    sprint_target_mode: str
    target_sprint_name: str
    target_sprint_external_id: str

    # Capacity override — set by sprint_planner when total story points exceed
    # what fits in the user's target sprint range (Q10).
    # See docs: "Guardrails" — human-in-the-loop pattern
    #   0       → not yet checked (default)
    #   < -1    → capacity warning pending; abs(value) = recommended sprint count
    #   -1      → user rejected recommendation; proceed with original target
    #   > 0     → user accepted; use this value as the new target sprint count
    capacity_override_target: int

    # Original target sprint count — set alongside capacity_override_target
    # when a capacity overflow is detected. Lets the TUI show "Keep N sprints"
    # in the choice popup so the user knows what the original target was.
    _original_target_sprints: int

    # Recommended team size to fit scope in original sprint count — computed
    # during capacity overflow detection: ceil(total_points / (vel_per_eng × target)).
    # Displayed as option 2 in the capacity overflow choice screen.
    # See docs: "Guardrails" — human-in-the-loop pattern
    _recommended_team_size: int

    # Team size override chosen by the user via the capacity overflow screen.
    # When > 0, sprint_planner recalculates velocity = vel_per_eng × this value
    # instead of using enforce_target. 0 = not set (default).
    # See docs: "Guardrails" — human-in-the-loop pattern
    _capacity_team_override: int

    # Architecture-validation spike opt-in/out. "" = undecided, "include" /
    # "skip" = the user's (or the confidence auto-rule's) answer. Asked only
    # when the architecture decision is genuinely open (2+ options, not
    # pinned); see _maybe_prompt_spike_choice in nodes.py.
    # See docs: "Guardrails" — human-in-the-loop pattern
    spike_choice: str

    # Transient sentinel: set (with {chosen, confidence, options}) when a node
    # needs the driver to ask the spike question — same ask-the-user pattern as
    # capacity_override_target, but a dict instead of a sign-encoded int.
    # Cleared (empty dict) once answered.
    _spike_prompt: dict

    # Small-project scope advisory. Set True by project_analyzer when the intake
    # ran in "small_project" mode but the analyzer judged the project bigger than
    # 1-2 tickets (needs feature grouping, > 2 sprints, or many goals). The
    # analysis review surfaces a "Switch to Large" action when this is True.
    # See docs: "Guardrails" — human-in-the-loop (advisory)
    _small_project_oversized: bool

    # Team ceremony (Standup + Retro) history gathered by project_analyzer and
    # reused downstream: _ceremony_action_items seeds story_writer's backlog
    # ([Retro] stories); _ceremony_history is the markdown block reused by
    # sprint_planner. Transient; serialize harmlessly across --resume.
    # See docs: "Session Management" — SQLite persistence
    _ceremony_action_items: tuple[str, ...]
    _ceremony_history: str
    # Per-engineer Performance signal (open 1:1 actions + review focus areas)
    # gathered by project_analyzer and reused by sprint_planner. Transient markdown.
    # See docs: "Performance Mode"
    _performance_context: str

    # Capacity deductions — all collected during intake (Phase 6: Capacity Planning).
    # Q27 (sprint selection / bank holidays auto-detected), Q28 (planned leave),
    # Q29 (unplanned %), Q30 (onboarding). Net velocity computed at intake confirmation.
    # Used by sprint_planner to compute net feature capacity (gross - deductions).
    # See docs: "Scrum Standards" — capacity planning
    capacity_bank_holiday_days: int  # Total bank/public holiday days in planning window
    capacity_planned_leave_days: int  # Total planned leave days (vacation, training)
    capacity_unplanned_leave_pct: int  # Percentage lost to unplanned absences (0–100)
    capacity_onboarding_engineer_sprints: int  # Engineer-sprints lost to ramp-up
    capacity_ktlo_engineers: int  # Engineers dedicated to KTLO/BAU work (default 0)
    capacity_discovery_pct: int  # Discovery/design tax percentage (default 5)
    net_velocity_per_sprint: int  # Adjusted velocity after capacity deductions (min of per-sprint)
    velocity_source: str  # Provenance: "jira", "manual", or "estimated"
    sprint_start_date: str  # ISO date string for first sprint start (e.g. "2026-03-16")

    # Per-sprint velocity breakdown — only sprints with bank holidays or PTO get
    # reduced capacity. Each entry is a dict with keys: sprint_index (0-based),
    # bank_holiday_days, bank_holiday_names (list[str]), pto_days, pto_entries,
    # net_velocity. When empty, the flat net_velocity_per_sprint is used everywhere.
    # See docs: "Scrum Standards" — capacity planning
    sprint_capacities: list[dict]

    # Structured per-person leave entries — persisted for rendering in exports
    # and TUI. Each entry: {"person": str, "start_date": str, "end_date": str,
    # "working_days": int}. PTO is per-person (1 × days), unlike bank holidays
    # (team_size × days). See README: "Scrum Standards" — capacity planning
    planned_leave_entries: list[dict]

    # Repo context — raw string from tool scan, populated by project_analyzer
    # and read by epic_generator. None if no URL was provided or scan failed.
    repo_context: str

    # Confluence context — concatenated plain-text content from confluence_search_docs
    # and confluence_read_page tool calls during the intake phase. Populated by the
    # agent as it reads relevant docs; surfaced in the project_analyzer prompt alongside
    # repo context. Empty string if no Confluence tools were called.
    # See docs: "Tools" — tool types, read-only tool pattern
    confluence_context: str

    # Notion context — concatenated plain-text content from notion_search_pages and
    # notion_read_page tool calls during the intake phase. Notion is an independent
    # doc source (its own integration token); surfaced in the project_analyzer prompt
    # alongside repo and Confluence context. Empty string if no Notion tools were called.
    # See docs: "Tools" — tool types, read-only tool pattern
    notion_context: str

    # User-provided context from SCRUM.md — free-form markdown the user places in
    # their project root (URLs, design notes, screenshots as links, tech decisions,
    # team conventions). Read once by project_analyzer; injected into the prompt so
    # the LLM can ground analysis in the user's own documentation.
    user_context: str

    # Pasted screenshot attachments (Ctrl+V in TUI textboxes) — PNG/JPEG file
    # paths under ~/.yeaboi/attachments/. Paths, not bytes, so state stays small
    # and sessions survive --resume; a deleted file degrades to text-only at
    # invoke time (see agent/llm.py:load_image_b64). state["messages"] must stay
    # text-only (nodes string-op on .content), so images ride here instead and
    # become multimodal content blocks only inside get_llm().invoke() call sites.
    #
    # pasted_images: collected from the project-description input and questionnaire
    # answers; consumed by project_analyzer.
    pasted_images: list[str]
    # review_feedback_images: screenshots attached to the current review-edit
    # feedback; consumed once by the node being regenerated, then cleared
    # alongside last_review_feedback.
    review_feedback_images: list[str]
    # chat_images: screenshots attached to the current post-pipeline chat message;
    # consumed by the agent node (call_model) on the next invoke, then cleared.
    chat_images: list[str]
    # The rendered text of a turn's @-references and attached files (see
    # agent/chat_refs.py). Same two channels as the images: pasted_context
    # accumulates through intake for the analyzer, chat_context is consumed by
    # the agent node on the next invoke, then cleared. Stored history stays
    # the person's own words.
    pasted_context: list[str]
    chat_context: list[str]

    # Review loop
    # See docs: "Guardrails" — human-in-the-loop pattern
    # pending_review holds the name of the generation node awaiting user review
    # (e.g. "feature_generator"). When set, the REPL intercepts user input and
    # routes it through the [Accept / Edit / Reject] flow instead of invoking
    # the graph. Cleared after the user makes a decision.
    pending_review: str
    last_review_decision: ReviewDecision
    last_review_feedback: str

    # Output
    output_format: OutputFormat

    # Context source diagnostics — populated by project_analyzer to show the user
    # which external sources (repo scan, Confluence, SCRUM.md) were used, skipped,
    # or failed. Each entry is a dict with keys: name, status, detail.
    # Rendered by the REPL after the analysis panel for transparency.
    context_sources: list[dict]

    # Jira key mappings — populated after jira_create_epic / jira_create_story calls.
    # jira_feature_keys: maps internal feature IDs → Jira Epic keys (e.g. "PROJ-5").
    # jira_story_keys: maps internal story IDs → Jira story keys.
    # jira_task_keys: maps internal task IDs → Jira sub-task keys.
    # jira_sprint_keys: maps internal sprint IDs → Jira sprint IDs.
    # jira_epic_key: single project-level Epic key (e.g. "PROJ-42").
    # The _merge_dicts reducer appends new entries without overwriting existing ones,
    # so each node/tool call can return only the mappings it just created.
    # See docs: "Tools" — tool types, write tools, human-in-the-loop pattern
    jira_feature_keys: Annotated[dict[str, str], _merge_dicts]
    jira_story_keys: Annotated[dict[str, str], _merge_dicts]
    jira_task_keys: Annotated[dict[str, str], _merge_dicts]
    jira_sprint_keys: Annotated[dict[str, str], _merge_dicts]
    jira_epic_key: str

    # Azure DevOps key mappings — populated after azdevops_create_epic / azdevops_create_story calls.
    # azdevops_epic_id: project-level Epic work item ID.
    # azdevops_story_keys: maps internal story IDs → AzDO work item IDs.
    # azdevops_task_keys: maps internal task IDs → AzDO work item IDs.
    # azdevops_iteration_keys: maps internal sprint IDs → AzDO iteration paths.
    # The _merge_dicts reducer appends new entries without overwriting existing ones,
    # so each node/tool call can return only the mappings it just created.
    # See docs: "Tools" — tool types, write tools, human-in-the-loop pattern
    azdevops_epic_id: str
    azdevops_story_keys: Annotated[dict[str, str], _merge_dicts]
    azdevops_task_keys: Annotated[dict[str, str], _merge_dicts]
    azdevops_iteration_keys: Annotated[dict[str, str], _merge_dicts]

    # Linear / Trello key mappings — populated by linear_sync / trello_sync.
    # Declared as graph channels with the same _merge_dicts reducer as the Jira
    # mappings above: an undeclared key is dropped when a resumed session's
    # state passes back through the graph, which would lose the idempotency
    # mapping and duplicate cards/issues on the next sync.
    linear_story_keys: Annotated[dict[str, str], _merge_dicts]
    linear_story_ids: Annotated[dict[str, str], _merge_dicts]
    linear_task_keys: Annotated[dict[str, str], _merge_dicts]
    linear_cycle_keys: Annotated[dict[str, str], _merge_dicts]
    trello_story_keys: Annotated[dict[str, str], _merge_dicts]
    trello_task_keys: Annotated[dict[str, str], _merge_dicts]
    trello_list_keys: Annotated[dict[str, str], _merge_dicts]
