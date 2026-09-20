#!/usr/bin/env python3
"""Decide whether a pull request has review feedback that nobody answered.

Five things comment on a PR in this repo — ``claude-review.yml`` after CI goes
green, the ``pr-opened-dod-audit`` routine, ``auto-version.yml``,
``dependabot-auto.yml``, and ``claude.yml`` when someone types ``@claude`` — and
until this script existed, nothing ever read a word of it back. ``/ship`` exits at
``gh pr create``, minutes before the review posts; ``/babysit-prs`` reads
``gh pr checks`` and nothing else. So findings landed on the timeline and PRs
merged straight past them.

This turns that into a **commit status**, which is the only artefact GitHub will
actually refuse a merge over. Everything else in the loop stays advisory: the
reviewers keep reviewing and never block, and this counts.

**The gate is arithmetic, not judgment.** It does not read a finding and guess
whether it was addressed — the producers stamp a machine-readable verdict into
their own comment (``<!-- pr-feedback: claude-review open=2 -->``), exactly like
the ``<!-- cowork-dod -->`` marker already used to find-and-edit the DoD audit.
A checker that had to interpret prose would be wrong quietly and often; this one
is a subtraction.

Three rules carry the whole design:

* **Only the latest verdict per producer counts.** Every push re-runs CI, which
  re-runs the review, which posts a fresh verdict. Fix the finding and the next
  verdict is ``open=0`` and the gate clears on its own — no reply, no ceremony.
  That is what keeps this from becoming a box nobody can tick.
* **An unfixed finding needs an answer newer than the finding, from someone with
  write access.** A reply carrying ``<!-- addressed: claude-review -->`` clears
  the pass it follows. Deliberate won't-fixes are the only case that ever has to
  be re-stated, and only when the branch moved underneath them. "Newer" is
  measured against when the verdict was last *written*, not created — a producer
  that edits one comment in place would otherwise be answerable in advance — and
  the write-access requirement is because this repo is public.
* **"Not yet" and "never" are different answers.** CI still running is `pending`.
  CI green twenty minutes ago with no review in sight is a `failure` that names
  the missing review — ``claude-review.yml`` has silently stopped firing twice in
  this repo's history (see its own header), and both times nothing noticed.

Usage::

    uv run python scripts/pr_feedback.py --pr 123            # human-readable, exit 1 if open
    uv run python scripts/pr_feedback.py --pr 123 --json     # for /pr-feedback to iterate
    uv run python scripts/pr_feedback.py --pr 123 --status   # post the status (CI only)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

# scripts/ is not a package, so the sibling transport is imported by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _gh_transport as transport  # noqa: E402 - after the sys.path line that makes it importable

ROOT = Path(__file__).resolve().parent.parent

# The status context a branch ruleset makes required. Changing this string
# silently un-blocks every PR, because the ruleset keeps waiting on a context
# nothing posts any more.
STATUS_CONTEXT = "pr-feedback"

# The one comment this script owns, found and edited in place rather than added
# to. Same trick as `<!-- cowork-dod -->`: one marker, one comment, no pile-up.
STICKY_MARKER = "<!-- pr-feedback-status -->"

# The escape hatch. A gate with no override is a gate that eventually bricks the
# repo — a review that errors out, a producer that changes format, a finding that
# is simply wrong. The label is honoured, and the sticky comment says so out loud
# so an overridden PR never looks like a clean one.
OVERRIDE_LABEL = "feedback-override"

# Mirrors the author filter in `claude-review.yml` and `pr-opened-dod-audit.md`:
# bot-authored PRs are not reviewed unless they came from the cowork implement
# job. Kept identical on purpose — if this gate expected a review the reviewer
# was never going to write, the PR would sit red forever with nothing to fix.
BOT_AUTHORS = frozenset({"dependabot[bot]", "github-actions[bot]"})
COWORK_LABEL = "cowork"

# Branch namespaces an unattended run pushes to: `cowork/…` (cowork-builder),
# `feature/issue-N-…` (claude.yml's implement job), plus the two workflows that
# open their own fix PRs. A PR from one of these was written by a machine, which
# has two consequences below — it is never waved through as "bot-authored, review
# not applicable", and its author cannot answer its own review.
#
# Kept in step with the author filter in `claude-review.yml`: this gate must
# never expect a review that the reviewer was never going to write, or the PR
# sits red with nothing to fix. Widen both or neither.
UNATTENDED_BRANCH_PREFIXES = (
    "cowork/",
    "feature/issue-",
    "security/codeql-triage",
    "ci-sentinel/",
)

# How long after CI goes green a review may take before its absence is treated as
# a fault rather than as latency.
REVIEW_GRACE = timedelta(minutes=20)

# The workflow whose success gates the reviewers (`.github/workflows/ci.yml`).
CI_WORKFLOW_NAME = "CI"

# GitHub truncates a status description past 140 characters.
DESCRIPTION_LIMIT = 140

# How many findings-bearing verdicts Claude Review may post on one PR before it
# stops reviewing altogether. `BLOCKING_ROUNDS` below is the earlier, separate
# point at which its findings stop *holding the merge*.
#
# Two, because the loop has no natural terminator otherwise. An adversarial
# reviewer reading a large diff will always find *something* — four consecutive
# rounds on PR #222 each produced real should-fix findings, and there was no
# reason to expect a fifth to be empty. "Merge when the reviewer reports zero" is
# therefore not a condition that reliably arrives, and a gate whose exit
# condition may never occur is a gate that gets deleted.
#
# At the cap the findings do not vanish: they are still listed in the sticky
# comment, the PR is labelled `review-capped`, and the daily standup names it.
# What changes is that they stop holding the merge. That is a real loosening, and
# what makes it defensible is that a merge no longer reaches users — it publishes
# a pre-release, and the weekly promotion is a human checkpoint. If that ever
# stops being true, revisit this number with it.
MAX_REVIEW_ROUNDS = 2

# How many findings-bearing verdicts may block the merge on ordinary findings.
# Past this count only a `critical` one still does.
#
# Counted the way `review_rounds` counts: the *first* findings-bearing verdict is
# round 1, so the comparison below is `rounds > BLOCKING_ROUNDS` and not `>=`.
# With `>=` the very first review would stop blocking the moment it posted, which
# is the whole gate, deleted by an off-by-one.
#
# One, because the second round is where the loop stops paying for itself. Round
# one reads a diff nobody else has read and is worth waiting for. By round two
# the author has already fixed what round one found, and what comes back is
# overwhelmingly the adversarial tail — real should-fix findings, but ones a
# person weighing them would file rather than hold a merge for. Four consecutive
# rounds on PR #222 each produced findings, and none of them was a reason that PR
# should not have merged when it did.
#
# So after the first findings-bearing verdict the review keeps speaking and stops
# blocking — *unless* it says the finding is critical. That is what `critical=M`
# in the marker is for, and why it is worth having a second number at all: a
# blocker found on round two still holds the merge, and everything else is
# recorded. `MAX_REVIEW_ROUNDS` above is still the point the reviewer stops
# writing at all.
BLOCKING_ROUNDS = 1

# How many times the reviewer speaks on a *local* branch.
#
# Nothing in this file reads it — the local lane never blocks, so there is no
# arithmetic here for it to feed. It lives here because `claude-review.yml` needs
# the number and cannot hold one: a bash literal in a YAML file is invisible to
# every test in this repo, and the two caps that workflow applies are exactly the
# kind of pair that drifts apart quietly. `TestTheReviewCaps` asserts the YAML
# against both constants, the same way it always has for `MAX_REVIEW_ROUNDS`.
#
# One, because a second adversarial pass over a diff whose author is sitting
# right there produces findings they will read and weigh anyway — at the cost of
# a full model run, on every push, for the rest of the branch's life.
LOCAL_REVIEW_ROUNDS = 1

# Applied when the cap is reached with findings still open, so the state is
# visible on the PR list and queryable afterwards rather than buried in a comment.
CAPPED_LABEL = "review-capped"

# One page of review threads. Cheap to raise, and never silently exceeded — a
# truncated page becomes an open item of its own rather than a quiet undercount.
THREAD_PAGE = 100


@dataclass(frozen=True)
class Producer:
    """Something that reviews a PR and stamps a countable verdict on its comment.

    ``required`` is the difference between "this reviewer had nothing to say" and
    "this reviewer never spoke". Claude Review runs on every PR, so its silence is
    a fault. The DoD audit is a hand-registered cowork routine that may not be
    deployed at all, so it is honoured when present and never waited on.
    """

    key: str
    label: str
    required: bool
    pattern: re.Pattern[str]
    # The exact login this producer speaks under, when there is one. Without it a
    # verdict has no author check at all — weaker than the acknowledgement rule
    # below, despite being the stronger statement: an ack answers findings, a
    # verdict replaces the count of them. On a public repo that meant any account
    # could post `open=0` and turn the gate green, and with the round cap it would
    # also mean two comments could exhaust the cap on somebody else's PR.
    authors: frozenset[str] | None = None


# The optional severity half of a verdict marker, shared by both producers so
# `verdict_counts` can read either one the same way. Absent reads as zero, which
# is the whole reason this is optional rather than required: every marker written
# before the field existed keeps parsing, and keeps meaning the least blocking
# thing it could mean.
_CRITICAL_RE = r"(?:\s+critical=(?P<critical>\d+))?"


# Two of the five commenters named above are registered here, and the omissions
# are deliberate rather than pending: `auto-version` and `dependabot-auto` report
# facts about a bump and never produce a finding, and `claude.yml` answers a
# human who is already in the conversation and will speak again if it is wrong.
# A producer earns a row by being expected to raise something nobody asked for.
PRODUCERS: tuple[Producer, ...] = (
    Producer(
        key="claude-review",
        label="Claude Review",
        required=True,
        pattern=re.compile(r"<!--\s*pr-feedback:\s*claude-review\s+open=(?P<open>\d+)" + _CRITICAL_RE + r"\s*-->"),
        # `claude-review.yml` posts through the Claude GitHub App. Verified
        # against the live comments on PR #222, not assumed. A `[bot]` suffix
        # cannot be forged — GitHub logins may not contain brackets.
        authors=frozenset({"claude[bot]"}),
    ),
    Producer(
        key="cowork-dod",
        label="DoD audit",
        required=False,
        # No pinned `authors`, deliberately: a cowork routine writes this one, and
        # it may run as the maintainer or as a bot depending on how it is invoked.
        # So it falls back to write access, which still stops a stranger on a
        # public repo. It is advisory and never required — forging its `open=0`
        # suppresses DoD-audit findings and leaves `claude-review`, the producer
        # that gates the merge, untouched. Named here rather than left as an
        # asymmetry to notice.
        # `open=` is what makes it countable. A bare `<!-- cowork-dod -->` is the
        # older format and deliberately reads as no verdict rather than as zero:
        # a routine that has not been updated yet must not be able to clear a gate
        # by saying nothing.
        pattern=re.compile(r"<!--\s*cowork-dod\s+open=(?P<open>\d+)" + _CRITICAL_RE + r"\s*-->"),
    ),
)

# A reply that says what happened to a producer's findings. Two shapes, and the
# difference between them is what this gate had never asked for:
#
#     <!-- addressed: claude-review -->                     every finding, answered
#     <!-- addressed: claude-review fixed=2 answered=1 -->   two changed, one argued
#
# The counted shape exists because a *fix* used to be invisible here. "Push, and
# the next verdict reads open=0" cleared the gate with nothing on the PR saying
# what changed or why. That is fine when a person did it — they are the one
# merging, and they know. When a machine did it, the entire record of what
# happened to three findings is that a number went down, and nobody can review a
# subtraction.
#
# The two fields are separate numbers rather than one total because they are
# checked by different things:
#
# * `answered=` is a **dismissal** — the finding stands and is not being acted
#   on. Nothing but a reader can check that, so it keeps the author restriction
#   in `is_acknowledged`: on an unattended PR the account that wrote the change
#   may not be the one declaring the review of it answered.
# * `fixed=` is a **claim of work**, and the reviewer's next pass checks it. A
#   fix that is not there gets reported again and the gate never clears. So this
#   half may be written by the PR's own author — it has to be, because that is
#   who did the work.
#
# A bare marker is the older shape and reads as "all of them, answered": it
# dismisses, and it accounts for any count. Every reply already sitting on an
# open PR therefore keeps meaning exactly what it meant when it was written.
ACK_RE = re.compile(
    r"<!--\s*addressed:\s*(?P<key>[a-z0-9][a-z0-9-]*)(?P<fields>(?:\s+[a-z]+=\d+)*)\s*-->",
    re.IGNORECASE,
)
_RESPONSE_FIELD_RE = re.compile(r"([a-z]+)=(\d+)", re.IGNORECASE)

# Who may answer a finding. This repo is public, so without this any account on
# the internet could clear the gate on somebody else's PR with a single drive-by
# `<!-- addressed: claude-review -->`. An ack from outside the set is read as
# ordinary prose; the sticky comment still says how to clear the check, so the
# way out of a wrongly-red gate is never closed, only moved to someone with
# write access.
#
# Write access is necessary and, on an unattended PR, not sufficient: see
# `is_acknowledged`, which additionally refuses an ack from the PR's own author.
TRUSTED_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})


@dataclass(frozen=True)
class Comment:
    """One comment on the PR's issue timeline, or one submitted review's body.

    Two timestamps, because a producer is told to **edit its comment in place**
    rather than pile up a new one on every push (`pr-opened-dod-audit.md` step 5)
    and an edit does not move ``created_at``. ``written_at`` is when the text
    that is there now was last written, and it is what a *verdict* is dated by:
    without it, a reply from the day the PR opened would go on clearing a set of
    findings that were rewritten into that comment ten minutes ago.

    An *acknowledgement* is deliberately dated by ``created_at`` instead, so that
    editing an old reply cannot forward-date it over a finding it never saw. The
    asymmetry is the fail-closed direction of the same fact.

    ``association`` is GitHub's ``author_association``; ``kind`` separates an
    issue comment from a review body, whose ids come from different spaces.
    """

    id: int
    author: str
    body: str
    created_at: datetime
    updated_at: datetime | None = None
    association: str = "NONE"
    kind: str = "comment"  # "comment" | "review"

    @property
    def written_at(self) -> datetime:
        return max(self.created_at, self.updated_at) if self.updated_at else self.created_at


@dataclass(frozen=True)
class Response:
    """What one ``<!-- addressed: … -->`` marker claims about a producer's findings.

    ``bare`` is the marker written with no counts — the shape that predates the
    fields. It reads as "all of them", so it both dismisses and accounts for
    whatever number the verdict reported, which is exactly what it has always
    meant.
    """

    key: str
    fixed: int = 0
    answered: int = 0
    bare: bool = True

    @property
    def dismisses(self) -> bool:
        """Whether this claims a finding stands and is deliberately not being acted on."""
        return self.bare or self.answered > 0

    def covers(self, count: int) -> bool:
        """Whether this accounts for every one of ``count`` findings, fixed or argued."""
        return self.bare or (self.fixed + self.answered) >= count


@dataclass(frozen=True)
class Thread:
    """One inline review thread. ``is_resolved`` is GitHub's own resolve button.

    ``resolved_by`` is who pressed it, and it is the difference between a
    reviewer closing their own point and the PR's author closing it for them.
    Empty when the thread is open, or when the field could not be read.
    """

    id: str
    is_resolved: bool
    is_outdated: bool
    path: str | None
    line: int | None
    authors: tuple[str, ...]
    excerpt: str
    resolved_by: str = ""


@dataclass(frozen=True)
class CIState:
    """The CI run for this exact head SHA. ``conclusion`` is None when none ran."""

    conclusion: str | None
    completed_at: datetime | None


@dataclass(frozen=True)
class Snapshot:
    """Everything the verdict is computed from. Fetched once, then never re-read."""

    number: int
    head_sha: str
    author: str
    is_draft: bool
    labels: tuple[str, ...]
    review_decision: str | None
    comments: tuple[Comment, ...]
    threads: tuple[Thread, ...]
    ci: CIState
    threads_truncated: bool = False
    head_ref: str = ""
    # Who applied `feedback-override`, when it is present. Empty when nobody did,
    # or when the timeline could not be read — those are different, and
    # ``classify`` treats them differently.
    override_actor: str = ""


@dataclass(frozen=True)
class OpenItem:
    """One thing standing between this PR and a merge."""

    kind: str  # "producer" | "thread" | "changes-requested" | "truncated" | "override"
    key: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "key": self.key, "detail": self.detail}


@dataclass(frozen=True)
class Verdict:
    """The commit status to post, and why."""

    state: str  # "success" | "failure" | "pending"
    description: str
    items: tuple[OpenItem, ...] = ()
    # Why a *green* status is carrying open items, which is the one case where
    # the state alone is not the whole story. "" is an ordinary clean pass;
    # "capped" means the review ran out of blocking rounds; "advisory" means this
    # is a local branch the gate does not enforce on. The sticky comment and the
    # `review-capped` label both read this rather than re-deriving it, because
    # "green with findings" used to mean exactly one thing and now means two.
    reason: str = ""

    advisory_items: tuple[OpenItem, ...] = ()

    def as_dict(self) -> dict:
        return {
            "advisory_items": [item.as_dict() for item in self.advisory_items],
            "state": self.state,
            "description": self.description,
            "reason": self.reason,
            "items": [item.as_dict() for item in self.items],
        }


# --- classification (pure; every test in tests/unit/test_pr_feedback.py lands here)


def parse_timestamp(value: str | None) -> datetime | None:
    """GitHub's ISO-8601, normalised to aware UTC so comparisons cannot explode."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def responses(body: str) -> dict[str, Response]:
    """Every producer this comment responds to, and what it claims about each.

    Repeated markers for one producer are merged by taking the largest claim on
    each field rather than summing — two markers in one comment is somebody
    restating, not somebody doing the work twice.

    An unrecognised field (``<!-- addressed: claude-review fixt=2 -->``) leaves
    both counts at zero rather than falling back to the bare reading. That is the
    fail-closed direction: a typo under-claims and gets caught, where the
    alternative would silently dismiss everything.
    """
    found: dict[str, Response] = {}
    for match in ACK_RE.finditer(body):
        key = match.group("key").lower()
        raw = match.group("fields") or ""
        fields = {name.lower(): int(value) for name, value in _RESPONSE_FIELD_RE.findall(raw)}
        parsed = Response(
            key=key,
            fixed=fields.get("fixed", 0),
            answered=fields.get("answered", 0),
            bare=not raw,
        )
        previous = found.get(key)
        found[key] = (
            parsed
            if previous is None
            else Response(
                key=key,
                fixed=max(previous.fixed, parsed.fixed),
                answered=max(previous.answered, parsed.answered),
                bare=previous.bare or parsed.bare,
            )
        )
    return found


def acknowledged_producers(body: str) -> set[str]:
    """Producer keys this comment claims to have *answered* — dismissals only.

    A comment that only accounts for fixes (``fixed=2`` and nothing else) is
    deliberately not in here. It says work was done, and the reviewer's next pass
    is what confirms that; nothing about it dismisses a finding, so nothing about
    it should clear one.
    """
    return {key for key, response in responses(body).items() if response.dismisses}


def is_authentic_verdict(comment: Comment, producer: Producer) -> bool:
    """Whether this comment is allowed to *be* a verdict from ``producer``.

    Where a producer speaks under a known login, that login is the whole test.
    Claude Review always arrives from ``claude[bot]``, and a ``[bot]`` suffix
    cannot be forged because GitHub logins may not contain brackets.

    Association is deliberately *not* also required: a bot commenting through
    ``GITHUB_TOKEN`` carries ``author_association: NONE``, so demanding write
    access would reject every genuine review and wedge the PR on one that could
    never qualify — the deadlock this module exists to avoid.

    A producer with no pinned login (the DoD audit, which a cowork routine writes
    as the maintainer or as a bot depending on how it runs) falls back to write
    access, which still stops a stranger on a public repo.
    """
    if producer.authors is not None:
        return comment.author in producer.authors
    if comment.author.endswith("[bot]"):
        return True
    return comment.association.upper() in TRUSTED_ASSOCIATIONS


def verdict_counts(match: re.Match[str]) -> tuple[int, int]:
    """``(open, critical)`` out of a verdict marker.

    ``critical=`` is optional and absent reads as **zero**, which is the only
    direction this could safely default. The field's sole power is to *hold* a
    merge past the first round; a marker that predates it, or a reviewer that
    forgets it, therefore blocks less rather than more. Reading a missing field
    as "assume the worst" would have made every verdict already sitting on an
    open PR into a blocker the moment this shipped.
    """
    groups = match.groupdict()
    critical = groups.get("critical")
    return int(groups["open"]), int(critical) if critical else 0


def verdict_history(comments: Iterable[Comment], producer: Producer) -> list[tuple[Comment, int, int]]:
    """Every authentic verdict this producer posted, oldest first.

    ``(comment, open, critical)`` per entry. Three things read this — the round
    counter, the newest-wins lookup, and the ledger — and they used to walk the
    comment list separately with the ordering rule restated each time.

    Ordered by ``(written_at, id)``. The id is not decoration: two comments can
    share a timestamp at second granularity, and a coin-flip order there would
    make the round numbers and the ledger flap between runs.

    ``written_at``, not ``created_at``, because the DoD audit is instructed to
    edit one comment in place on every push — its ``created_at`` is the hour the
    PR opened while the verdict inside it may be a minute old.

    Comments that are not authentic verdicts (see ``is_authentic_verdict``) never
    enter, so a forged one can neither inflate the round count nor win the
    newest-wins race.
    """
    history: list[tuple[Comment, int, int]] = []
    for comment in comments:
        match = producer.pattern.search(comment.body)
        if match is None or not is_authentic_verdict(comment, producer):
            continue
        history.append((comment, *verdict_counts(match)))
    history.sort(key=lambda entry: (entry[0].written_at, entry[0].id))
    return history


def review_rounds(comments: Iterable[Comment], producer: Producer) -> int:
    """How many times this producer has posted a verdict **that found something**.

    *Non-empty only*, which is the difference between bounding the loop and
    breaking the gate. Counting every verdict would cap a PR whose first review
    was clean the moment a second one ran — so a regression introduced after a
    clean pass could never reopen the gate, and "review found a new problem" and
    "review ran out of patience" would be the same state. What runs forever is
    the *findings* loop: find, fix, find again. That is what gets counted.
    """
    return sum(1 for _comment, count, _critical in verdict_history(comments, producer) if count > 0)


def latest_verdict(comments: Iterable[Comment], producer: Producer) -> tuple[Comment, int, int] | None:
    """The newest countable verdict from one producer, or None if it never posted.

    Returns ``(comment, open, critical)``. ``critical`` is the blocker-severity
    subset of ``open`` and is what decides whether a *second* round still holds
    the merge — see ``BLOCKING_ROUNDS``.

    Newest wins outright — an older pass is stale the moment a newer one exists,
    and re-reviewing after a push is precisely how a fixed finding disappears
    without anyone replying to it. ``unaccounted_rounds`` is what now asks for a
    written account of that disappearance; this function keeps ignoring it, which
    is what stops a fixed finding from blocking forever.

    Ordering, authorship and the ``written_at`` dating all come from
    ``verdict_history``.
    """
    history = verdict_history(comments, producer)
    return history[-1] if history else None


def is_unattended(snapshot: Snapshot) -> bool:
    """Whether this PR was written by a machine rather than by a person.

    Two signals, either of which is enough: the ``cowork`` label, and a head
    branch in one of the namespaces an unattended run pushes to. The label alone
    was not enough once the auto lane widened — `cowork/house-rules.md` requires
    it on every routine PR, but a run truncated between `git push` and
    `gh pr create --label` leaves an unlabelled machine PR behind, and that is
    exactly the PR that must not be trusted more than a labelled one.
    """
    return COWORK_LABEL in snapshot.labels or snapshot.head_ref.startswith(UNATTENDED_BRANCH_PREFIXES)


def is_acknowledged(
    comments: Iterable[Comment],
    producer_key: str,
    after: datetime,
    after_id: int = 0,
    deny_author: str | None = None,
) -> bool:
    """Whether a trusted reply newer than ``after`` answers this producer's pass.

    ``after_id`` breaks the same second-granularity tie ``latest_verdict`` breaks:
    a reply posted in the same second as the verdict it answers is newer, and a
    strict ``>`` on the timestamp alone would drop it.

    ``deny_author`` is the PR's own author, passed only for an unattended PR, and
    it is what makes this gate mean anything once machines merge their own work.
    A cowork routine posts under an account with write access, so
    ``TRUSTED_ASSOCIATIONS`` alone would let the thing that wrote the change also
    declare the review of it answered — a gate whose key is held by the applicant.
    It may still *fix* a finding: a push triggers a re-review, and
    ``latest_verdict`` then reads ``open=0`` from the reviewer itself. What it may
    no longer do is disagree in a comment and merge anyway.

    A human answering the review on their own PR is untouched: a person read it,
    which is the whole thing being checked for.
    """
    return acknowledgement(comments, producer_key, after=after, after_id=after_id, deny_author=deny_author) is not None


def acknowledgement(
    comments: Iterable[Comment],
    producer_key: str,
    after: datetime,
    after_id: int = 0,
    deny_author: str | None = None,
) -> Comment | None:
    """The reply ``is_acknowledged`` found, so the ledger can name who wrote it.

    Earliest wins, for the same reason as ``account_for``: the reply that
    answered the round is the one worth naming, not a later comment that happened
    to repeat the marker.
    """
    best: Comment | None = None
    for comment in comments:
        if (comment.created_at, comment.id) <= (after, after_id):
            continue
        if comment.association.upper() not in TRUSTED_ASSOCIATIONS:
            continue
        if deny_author is not None and comment.author == deny_author:
            continue
        if producer_key not in acknowledged_producers(comment.body):
            continue
        if best is None or (comment.created_at, comment.id) < (best.created_at, best.id):
            best = comment
    return best


def account_for(
    comments: Iterable[Comment],
    producer_key: str,
    after: datetime,
    after_id: int,
    count: int,
    deny_dismissal_from: str = "",
) -> Comment | None:
    """The reply that says what was done about ``count`` findings, or None.

    Deliberately has no ``deny_author``, and that asymmetry with
    ``is_acknowledged`` is the point rather than an oversight. An account is a
    claim of *work*, and the reviewer's next pass checks it — claim a fix that is
    not there and the finding comes straight back. A dismissal has no such check
    behind it, which is why only that one is refused from the applicant.

    ``deny_dismissal_from`` is the one place that asymmetry has to be spelled
    out. An account that claims **no fix at all** — a bare marker, which means
    "all of them, answered", or an explicit ``answered=N fixed=0`` — is pure
    dismissal: it leaves nothing for the next review pass to check. Accepting one
    from the PR's own author would let an unattended PR close out every
    superseded round without ever claiming a fix, which is precisely the silence
    this check exists to end. Claim one fix and the whole reply is admissible,
    because the reviewer's re-read of the diff is then a real check on it.

    Testing ``fixed == 0`` rather than ``bare`` is deliberate and was a real gap:
    the bare shape was refused while ``answered=3`` from that same author sailed
    through, which is the identical dismissal with a number typed after it.

    Earliest match wins, so the ledger names the reply that answered the round
    rather than whichever later comment happened to repeat the marker.
    """
    best: Comment | None = None
    for comment in comments:
        if (comment.created_at, comment.id) <= (after, after_id):
            continue
        if comment.association.upper() not in TRUSTED_ASSOCIATIONS:
            continue
        response = responses(comment.body).get(producer_key)
        if response is None or not response.covers(count):
            continue
        if response.fixed == 0 and deny_dismissal_from and comment.author == deny_dismissal_from:
            continue
        if best is None or (comment.created_at, comment.id) < (best.created_at, best.id):
            best = comment
    return best


def settled_count(history: Sequence[tuple[Comment, int, int]], index: int) -> int:
    """How many of round ``index``'s findings must be accounted for.

    ``max(1, count - next_count)`` — the findings that stopped being reported
    between this verdict and the one after it. Zero for the newest verdict, which
    has no successor and is still being held by ``open_producers`` if it found
    anything.

    Shared by the gate and the ledger so a round cannot read as answered in one
    and unanswered in the other. The reasoning behind both halves of the formula
    is in ``unaccounted_rounds``.
    """
    count = history[index][1]
    if count == 0 or index + 1 >= len(history):
        return 0
    return max(1, count - history[index + 1][1])


def unaccounted_rounds(snapshot: Snapshot) -> list[OpenItem]:
    """Findings that stopped being reported with nothing saying what changed.

    The gate's original rule was that a fix needs no reply: push, the reviewer
    re-reads, the next verdict says ``open=0``, done — "no reply, no ceremony",
    as the module docstring puts it. That is right when a person did the fixing.
    They are the one merging, and they know what they changed.

    It is wrong for a machine. The whole record of what an agent did about three
    findings becomes a number going down, and a subtraction cannot be reviewed.
    So on the unattended lane a findings-bearing verdict that has been superseded
    needs an account newer than it — ``<!-- addressed: claude-review fixed=N -->``
    and a sentence per finding.

    Note what this does not ask for. It never re-opens a finding the reviewer
    stopped reporting, and it never asks anyone to agree with the fix. It asks
    for the fix to be written down, once, by whoever made it.

    **Superseded only**, because the newest verdict is already handled: if it
    still reports findings then ``open_producers`` is holding the gate on exactly
    those, and two items about one verdict would only make the check harder to
    read.

    **How much must be accounted for** is ``max(1, count - next_count)``: the
    number of findings that stopped being reported between this round and the
    one after it. Asking for all of ``count`` would be asking somebody to write a
    disposition for a finding still sitting open in the next verdict, which they
    have not resolved and should not claim to have. The floor of one is what
    stops a round from vanishing for free when the next pass happens to report
    the same number — a count that did not move is not evidence that nothing was
    done, and the marker is one line.

    The arithmetic cannot tell a carried-over finding from a brand-new one, so a
    round that fixes three things while the next pass finds three different ones
    is asked to account for one rather than three. That is the known slack, and
    it is in the direction of asking for less: the gate's job here is to make the
    work visible, not to audit the count.

    **Never capped**, unlike the findings themselves. ``BLOCKING_ROUNDS`` exists
    because an adversarial review of a large diff always finds something, so the
    findings loop has no natural end. This has one: it is a single comment, it is
    always satisfiable, and it does not get harder the more rounds run.
    """
    producer = PRODUCERS[0]
    history = verdict_history(snapshot.comments, producer)
    if not history:
        return []
    latest_id = history[-1][0].id
    items: list[OpenItem] = []
    round_number = 0
    for index, (comment, count, _critical) in enumerate(history):
        if count == 0:
            continue
        round_number += 1
        if comment.id == latest_id:
            continue
        required = settled_count(history, index)
        if (
            account_for(
                snapshot.comments,
                producer.key,
                comment.written_at,
                comment.id,
                required,
                deny_dismissal_from=snapshot.author,
            )
            is not None
        ):
            continue
        settled = "finding" if required == 1 else "findings"
        items.append(
            OpenItem(
                "account",
                f"{producer.key}:{comment.id}",
                f"{producer.label} round {round_number} reported {count}, and {required} {settled} "
                f"stopped being reported without anything saying what changed — reply with what you "
                f"did, ending `<!-- addressed: {producer.key} fixed=N answered=M -->`",
            )
        )
    return items


def _thread_location(thread: Thread) -> str:
    return f"{thread.path}:{thread.line}" if thread.path and thread.line else (thread.path or "the PR")


def open_threads(snapshot: Snapshot) -> list[OpenItem]:
    """Unresolved inline threads that somebody other than the author started.

    ``isOutdated`` is deliberately not a free pass: a thread going outdated means
    the line moved, not that the point was taken. Only the resolve button closes
    one, which is the same signal a human reviewer already uses.
    """
    items: list[OpenItem] = []
    for thread in snapshot.threads:
        if thread.is_resolved:
            continue
        if not any(author != snapshot.author for author in thread.authors):
            continue
        detail = f"unresolved thread on {_thread_location(thread)}"
        if thread.excerpt:
            detail = f"{detail} — {thread.excerpt}"
        items.append(OpenItem("thread", thread.id, detail))
    return items


def silently_resolved(snapshot: Snapshot) -> list[OpenItem]:
    """Threads this PR's own author resolved without ever replying in them.

    **Resolve conversation** is a claim that the reviewer was heard. Every agent
    prompt in this repo says never to press it on a thread you did not answer,
    and until now that was a convention held up by nothing — which is the exact
    shape of the problem the rest of this file exists for. A machine closing a
    human's thread in silence also scales in a way a person doing it does not.

    Narrow on purpose: it fires only when all three are true — the PR's own
    author resolved it, somebody else had spoken in it, and the author never did.
    A maintainer resolving somebody's thread is a person's call and is untouched.
    A thread nobody but the author ever wrote in has nothing in it to answer.

    Clearing it is a reply in the thread. Resolved threads still take comments,
    so nothing has to be re-opened and no state has to be undone.
    """
    items: list[OpenItem] = []
    for thread in snapshot.threads:
        if not thread.is_resolved or thread.resolved_by != snapshot.author or not snapshot.author:
            continue
        if not any(author != snapshot.author for author in thread.authors):
            continue
        if snapshot.author in thread.authors:
            continue
        items.append(
            OpenItem(
                "resolved",
                thread.id,
                f"thread on {_thread_location(thread)} was resolved by {thread.resolved_by}, who wrote "
                f"this PR, with no reply in it — answer it in the thread",
            )
        )
    return items


def waiting_reason(snapshot: Snapshot, producer: Producer, now: datetime) -> str | None:
    """Why a missing verdict is latency rather than a fault — None once it is a fault."""
    ci = snapshot.ci
    if ci.conclusion is None:
        return f"waiting for CI before {producer.label}"
    if ci.conclusion != "success":
        return f"CI is {ci.conclusion} — {producer.label} does not run until it is green"
    if ci.completed_at is not None and now - ci.completed_at < REVIEW_GRACE:
        return f"waiting for {producer.label}"
    return None


def open_producers(snapshot: Snapshot, now: datetime) -> tuple[list[OpenItem], str | None]:
    """Producers with unanswered findings, plus the first reason to still be waiting."""
    items: list[OpenItem] = []
    waiting: str | None = None
    for producer in PRODUCERS:
        latest = latest_verdict(snapshot.comments, producer)
        if latest is None:
            if not producer.required:
                continue
            reason = waiting_reason(snapshot, producer, now)
            if reason is not None:
                waiting = waiting or reason
            else:
                items.append(
                    OpenItem(
                        "producer",
                        producer.key,
                        f"{producer.label} never posted a verdict on {snapshot.head_sha[:7]} — "
                        f"check the workflow actually ran",
                    )
                )
            continue
        comment, count, _critical = latest
        if count == 0:
            continue
        if is_acknowledged(
            snapshot.comments,
            producer.key,
            after=comment.written_at,
            after_id=comment.id,
            deny_author=snapshot.author if is_unattended(snapshot) else None,
        ):
            continue
        plural = "finding" if count == 1 else "findings"
        items.append(
            OpenItem("producer", producer.key, f"{producer.label}: {count} unanswered {plural} (comment {comment.id})")
        )
    return items, waiting


def describe(items: Sequence[OpenItem]) -> str:
    """The one line GitHub shows next to the check. Counted, then named."""
    noun = "item" if len(items) == 1 else "items"
    head = f"{len(items)} unanswered review {noun}"
    if items:
        head = f"{head}: {items[0].detail}"
        if len(items) > 1:
            head = f"{head} (+{len(items) - 1} more)"
    return head[:DESCRIPTION_LIMIT]


CODEX_REVIEWER = "chatgpt-codex-connector[bot]"


def codex_only_thread(thread: Thread, author: str) -> bool:
    participants = set(thread.authors) - {author}
    return participants == {CODEX_REVIEWER}


def classify(snapshot: Snapshot, now: datetime) -> Verdict:
    """Keep native Codex findings visible without changing existing merge gates."""
    advisory_threads = tuple(t for t in snapshot.threads if codex_only_thread(t, snapshot.author))
    enforced = replace(snapshot, threads=tuple(t for t in snapshot.threads if t not in advisory_threads))
    verdict = _classify_enforced(enforced, now)
    advisory = tuple(open_threads(replace(snapshot, threads=advisory_threads)))
    return replace(verdict, advisory_items=advisory)


def _classify_enforced(snapshot: Snapshot, now: datetime) -> Verdict:
    """The whole decision. Everything above feeds this; nothing below re-decides it."""
    if OVERRIDE_LABEL in snapshot.labels:
        # The override is a stronger dismissal than any marker — it clears every
        # finding, every unresolved thread and a CHANGES_REQUESTED review at once —
        # so the rule that governs the marker has to govern it too, or the lane
        # simply uses the bigger lever. On an unattended PR the applicant holds
        # write access, and `gh pr edit --add-label` sits inside the sweeps' bare
        # `Bash` grant, so "a human's call" was a convention rather than a fact.
        #
        # Unknown actor still honours the override. This label exists to unbrick a
        # gate that has genuinely gone wrong, and refusing it on a timeline we
        # could not read would turn one API failure into a PR nobody can merge —
        # the exact outcome it is the escape hatch for.
        if is_unattended(snapshot) and snapshot.override_actor and snapshot.override_actor == snapshot.author:
            return Verdict(
                "failure",
                f"`{OVERRIDE_LABEL}` was applied by this PR's own author — it does not count here",
                (
                    OpenItem(
                        "override",
                        OVERRIDE_LABEL,
                        f"`{OVERRIDE_LABEL}` applied by {snapshot.override_actor}, who authored this "
                        f"machine PR — a human other than the author must apply it, or fix the findings",
                    ),
                ),
            )
        who = f" by {snapshot.override_actor}" if snapshot.override_actor else ""
        return Verdict("success", f"overridden by the `{OVERRIDE_LABEL}` label{who}")
    if snapshot.is_draft:
        return Verdict("success", "draft — review not applicable")
    if snapshot.author in BOT_AUTHORS and not is_unattended(snapshot):
        return Verdict("success", f"{snapshot.author} — review not applicable")

    items = open_threads(snapshot)
    if snapshot.review_decision == "CHANGES_REQUESTED":
        items.append(OpenItem("changes-requested", "review", "a reviewer requested changes and has not re-approved"))
    if snapshot.threads_truncated:
        items.append(
            OpenItem("truncated", "threads", f"more than {THREAD_PAGE} review threads — this gate cannot see them all")
        )

    producer_items, waiting = open_producers(snapshot, now)
    items.extend(producer_items)

    # --- the local lane ------------------------------------------------------
    # A branch a person is sitting at the keyboard for. The review still runs and
    # its findings are still written and listed here; what it never does is hold
    # the merge.
    #
    # `success`, and never `pending`, is the load-bearing half of this. A
    # *required* context sitting pending blocks a merge exactly as hard as a red
    # one, so a local PR whose review never fired — the reviewer has silently
    # stopped twice in this repo's history, see `claude-review.yml`'s header —
    # would hang with nothing to act on. Returning here, ahead of `waiting`,
    # means there is no path on which a human's own branch is held by this gate.
    # That is the precondition that made adding `pr-feedback` to the branch
    # ruleset safe: the gate can only ever refuse an unattended PR, and an
    # unattended PR has a machine on the other end that can fix what it is told.
    #
    # What goes advisory here is the *machine reviewer*, and only it. A second
    # person's unresolved thread and a `CHANGES_REQUESTED` review still block,
    # because the argument above does not reach them: "nobody on the other end to
    # weigh a finding" is false by construction when the finding is a human's,
    # and `definition-of-done.md` promises in as many words that a person waiting
    # for an answer is never capped. Letting those ride as advisory would have
    # printed "nothing here has to be answered before you merge" directly above a
    # reviewer's open thread — on the one lane where the reviewer is real.
    if not is_unattended(snapshot):
        human = [item for item in items if item.kind != "producer"]
        if human:
            return Verdict("failure", describe(human), tuple(human))
        if not items:
            return Verdict("success", "no open review feedback")
        noun = "finding" if len(items) == 1 else "findings"
        return Verdict(
            "success",
            f"advisory — {len(items)} review {noun} on a local branch, not enforced",
            tuple(items),
            reason="advisory",
        )

    # --- the unattended lane: was the process followed -----------------------
    # Two things a machine can do that a person doing the same thing would not
    # need checking on: fix three findings and say nothing, and close a human's
    # thread without answering it. Both were conventions in an agent prompt until
    # now, and a convention is not a gate.
    #
    # Unattended only, and not because a human is trusted more — because neither
    # question arises there. The local lane reviews once, so no verdict is ever
    # superseded and ``unaccounted_rounds`` is inert by construction; and a person
    # resolving a thread on their own PR is the ordinary use of the button.
    #
    # Appended *after* the findings so ``describe`` still leads with what the
    # reviewer said. These are about the answer, and the finding comes first.
    items.extend(unaccounted_rounds(snapshot))
    items.extend(silently_resolved(snapshot))

    # --- the unattended lane: the blocking cap -------------------------------
    # Reached only on the required producer's own findings — an unresolved human
    # thread or a requested-changes review is a person waiting for an answer, and
    # running out of *review rounds* says nothing about those. Capping them too
    # would let a PR merge past somebody who is still talking.
    #
    # The two process items above are outside it for the same reason and by the
    # same mechanism — they are not `kind == "producer"`, so they land in `human`
    # below and the cap never applies. The cap exists because the findings loop
    # has no natural end; "say what you did" has one, and it is one comment.
    #
    # Past `BLOCKING_ROUNDS` the review keeps speaking and stops blocking, with
    # one exception: a verdict that reports `critical=M` above zero still holds
    # the merge, however many rounds have run. Round two exists to catch the
    # blocker round one's fix introduced; everything else it finds is recorded.
    # The critical count is read from the *latest* verdict only, for the same
    # reason the open count is — an older pass is stale the moment a newer one
    # exists, so a blocker that has since been fixed does not keep blocking.
    review = PRODUCERS[0]
    rounds = review_rounds(snapshot.comments, review)
    latest = latest_verdict(snapshot.comments, review)
    critical = latest[2] if latest is not None else 0
    if rounds > BLOCKING_ROUNDS and critical == 0:
        human = [item for item in items if item.kind != "producer" or item.key != review.key]
        if not human:
            capped = [item for item in items if item.kind == "producer" and item.key == review.key]
            if capped:
                noun = "finding" if len(capped) == 1 else "findings"
                if rounds >= MAX_REVIEW_ROUNDS:
                    detail = f"review capped at {MAX_REVIEW_ROUNDS} rounds — {len(capped)} {noun} recorded, not fixed"
                else:
                    detail = f"{len(capped)} non-critical {noun} after round {rounds} — recorded, not blocking"
                return Verdict("success", detail, tuple(capped), reason="capped")

    # Open items win over waiting: something is already unanswered, and a later
    # review pass can only add to that.
    if items:
        return Verdict("failure", describe(items), tuple(items))
    if waiting is not None:
        return Verdict("pending", waiting[:DESCRIPTION_LIMIT])
    return Verdict("success", "no open review feedback")


# --- gh ----------------------------------------------------------------------


def _gh(*args: str) -> subprocess.CompletedProcess[str]:
    return transport.gh(*args)


def _gh_json(*args: str) -> object | None:
    result = _gh(*args)
    if result.returncode != 0:
        print(f"[pr-feedback] gh {' '.join(args[:2])} failed: {result.stderr.strip()}", file=sys.stderr)
        return None
    try:
        return json.loads(result.stdout or "null")
    except json.JSONDecodeError:
        print(f"[pr-feedback] gh {' '.join(args[:2])} returned non-JSON", file=sys.stderr)
        return None


# --- reads and writes, through whichever transport this machine has -----------
#
# `gh` when it is installed, REST with a token when it is not. Both branches ask
# for the same thing; only the spelling differs. This gate is why the distinction
# matters more here than anywhere: it runs in a cloud routine session that has a
# token and no CLI, and a gate that cannot read a PR does not fail loudly — it
# finds nothing, which reads exactly like a clean PR.


def _read_paged(path: str) -> list | None:
    """Every page of a list endpoint. None when the question could not be asked.

    None rather than ``[]`` deliberately, and the callers depend on it: an empty
    list of review threads means "nothing to answer" while a failed fetch means
    "unknown", and collapsing them opens the gate on a network blip.
    """
    if transport.gh_available():
        payload = _gh_json("api", path, "--paginate")
        return payload if isinstance(payload, list) else None
    result = transport.api_paged(f"/{path.lstrip('/')}")
    if not result.ok:
        print(f"[pr-feedback] GET {path} failed: {result.error}", file=sys.stderr)
        return None
    return result.data if isinstance(result.data, list) else None


def _read(path: str) -> object | None:
    if transport.gh_available():
        return _gh_json("api", path)
    result = transport.api("GET", f"/{path.lstrip('/')}")
    if not result.ok:
        print(f"[pr-feedback] GET {path} failed: {result.error}", file=sys.stderr)
        return None
    return result.data


# Why the last read failed, for `main` to explain with. Module state rather than
# a return value because `fetch_snapshot` has one caller and four ways to come
# back empty, and threading a reason through all of them would be more moving
# parts than the problem has.
#
# It exists because of what a routine session does to this script: GraphQL is
# refused there outright (403, recorded in tests/fixtures/cowork_github_access_live.json),
# so the gate reads *nothing* — and a gate that reads nothing must never be
# mistaken for a gate that found nothing.
LAST_FAILURE = ""

# The refusal a routine session gets. Matched on the two stable halves of
# GitHub's own message rather than the status code, because a 403 on this path
# from a *token scope* problem is a different fault with a different remedy.
_PROXY_REFUSAL = "not enabled for this session"


def _graphql(variables: dict) -> dict | None:
    """`PR_QUERY`, through whichever transport this machine has.

    The transport routes this one itself, because `gh api graphql` takes a shape
    (`-f query=`, `-F var=`) no caller should have to know, and because a
    GraphQL error arrives inside a 200 — something has to notice that, and it is
    not the caller.
    """
    global LAST_FAILURE
    result = transport.graphql(PR_QUERY, variables)
    if not result.ok:
        LAST_FAILURE = f"graphql: {result.error}"
        print(f"[pr-feedback] graphql failed: {result.error}", file=sys.stderr)
        return None
    return result.data if isinstance(result.data, dict) else None


def unreadable_reason() -> str:
    """What to print when the PR could not be read, in terms someone can act on.

    The distinction this draws is the whole point: a session whose egress refuses
    GraphQL cannot answer the question *at all*, and saying so is the difference
    between "go look at the CI run" and "this PR is fine". `reviewDecision` and
    whether a thread is resolved exist in v4 and nowhere in v3, so there is no
    partial answer to fall back to — only a smaller answer that would read as a
    whole one.
    """
    if _PROXY_REFUSAL in LAST_FAILURE:
        return (
            "this session's GitHub egress refuses GraphQL, and review threads plus "
            "reviewDecision exist only there — so NOTHING was determined about this PR. "
            "Do not read this as a clean PR. The full gate runs in "
            ".github/workflows/pr-feedback.yml, where the query is served."
        )
    return LAST_FAILURE or "see the run log"


def _write(method: str, path: str, fields: dict[str, object]) -> bool:
    """One write. `gh api -f k=v` and a JSON body say the same thing.

    `-f` sends strings, which is why the REST branch need not reproduce `-F`'s
    typing: nothing written from here is a number.
    """
    if transport.gh_available():
        args = ["api", "-X", method, path]
        for key, value in fields.items():
            # A list is `gh api`'s repeated-field spelling: `-f labels[]=x`. The
            # REST branch below sends the real list, so callers write one shape.
            if isinstance(value, list):
                args += [arg for item in value for arg in ("-f", f"{key}[]={item}")]
            else:
                args += ["-f", f"{key}={value}"]
        result = _gh(*args)
        if result.returncode != 0:
            print(f"[pr-feedback] {method} {path} failed: {result.stderr.strip()}", file=sys.stderr)
            return False
        return True
    result = transport.api(method, f"/{path.lstrip('/')}", dict(fields))
    if not result.ok:
        print(f"[pr-feedback] {method} {path} failed: {result.error}", file=sys.stderr)
        return False
    return True


# One query for the metadata *and* the threads, rather than `gh pr view` for the
# first and GraphQL for the second.
#
# Not a tidy-up: `reviewDecision` is a v4 field with no REST equivalent at all,
# and `classify` reads it to hold the gate on a CHANGES_REQUESTED review. Asking
# for it here is what lets both transports answer the same question — the
# alternative was a REST branch that silently never saw a requested change.
# It also costs one call instead of two, on every run.
PR_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $page: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      number
      headRefOid
      headRefName
      isDraft
      reviewDecision
      author { login }
      labels(first: 50) { nodes { name } }
      reviewThreads(first: $page) {
        pageInfo { hasNextPage }
        nodes {
          id
          isResolved
          isOutdated
          resolvedBy { login }
          path
          line
          comments(first: 50) { nodes { author { login } body } }
        }
      }
    }
  }
}
"""


def fetch_snapshot(number: int, slug: str) -> Snapshot | None:
    """Read the PR once, from four read-only calls, into the shape ``classify`` wants."""
    owner, _, name = slug.partition("/")

    graph = _graphql({"owner": owner, "name": name, "number": number, "page": THREAD_PAGE})
    pull = (((graph or {}).get("data") or {}).get("repository") or {}).get("pullRequest")
    if not isinstance(pull, dict):
        return None
    head_sha = pull.get("headRefOid") or ""
    labels = tuple(node.get("name", "") for node in ((pull.get("labels") or {}).get("nodes") or []))

    raw_comments = _read_paged(f"repos/{slug}/issues/{number}/comments")
    if raw_comments is None:
        return None
    comments = tuple(
        Comment(
            id=int(item.get("id", 0)),
            author=(item.get("user") or {}).get("login", ""),
            body=item.get("body") or "",
            created_at=parse_timestamp(item.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc),
            updated_at=parse_timestamp(item.get("updated_at")),
            association=item.get("author_association") or "NONE",
        )
        for item in raw_comments
    ) + fetch_reviews(slug, number)

    block = pull.get("reviewThreads") or {}
    truncated = bool((block.get("pageInfo") or {}).get("hasNextPage"))
    threads = tuple(_thread(node) for node in block.get("nodes") or [])

    return Snapshot(
        number=number,
        head_sha=head_sha,
        author=(pull.get("author") or {}).get("login", ""),
        is_draft=bool(pull.get("isDraft")),
        labels=labels,
        review_decision=pull.get("reviewDecision") or None,
        comments=comments,
        threads=threads,
        ci=fetch_ci(slug, head_sha),
        threads_truncated=truncated,
        head_ref=pull.get("headRefName") or "",
        # Only asked when the label is actually present — one paginated call for a
        # question that is almost always moot.
        override_actor=(fetch_override_actor(slug, number) if OVERRIDE_LABEL in labels else ""),
    )


def _thread(node: dict) -> Thread:
    nodes = ((node.get("comments") or {}).get("nodes")) or []
    authors = tuple((entry.get("author") or {}).get("login", "") for entry in nodes)
    first_body = (nodes[0].get("body") if nodes else "") or ""
    excerpt = " ".join(first_body.split())[:80]
    return Thread(
        id=node.get("id", ""),
        is_resolved=bool(node.get("isResolved")),
        is_outdated=bool(node.get("isOutdated")),
        path=node.get("path"),
        line=node.get("line"),
        authors=authors,
        excerpt=excerpt,
        # Absent on an open thread, and null on a resolved one only if the
        # resolver's account is gone. Either way empty, which `silently_resolved`
        # reads as "cannot tell" and lets through — this check accuses somebody
        # by name, so it never runs on a field it did not read.
        resolved_by=((node.get("resolvedBy") or {}).get("login") or ""),
    )


def fetch_reviews(slug: str, number: int) -> tuple[Comment, ...]:
    """Submitted review bodies, folded in beside the issue comments.

    A review's top-level body is not an issue comment and never appears in
    ``/issues/{n}/comments``, so a marker written through the review box rather
    than the comment box would be invisible in *both* directions — a producer's
    verdict, and a maintainer's ``<!-- addressed: … -->``.

    What is deliberately not here: a human review body that carries findings in
    prose and no marker is not counted as an open item. GitHub already has two
    unambiguous ways to say "act on this", an inline thread and a Request-changes
    review, and both are gated. Treating every ``COMMENTED`` review as blocking
    would turn "LGTM, nice work" into a merge block, which is how a check gets
    deleted rather than answered.
    """
    raw = _read_paged(f"repos/{slug}/pulls/{number}/reviews")
    if raw is None:
        return ()
    return tuple(
        Comment(
            id=int(item.get("id", 0)),
            author=(item.get("user") or {}).get("login", ""),
            body=item.get("body") or "",
            # Reviews expose no edited-at, so a verdict written into one is dated
            # by submission. Producers post comments, not reviews; this path is
            # for the maintainer who typed the ack into the review box.
            created_at=parse_timestamp(item.get("submitted_at")) or datetime.min.replace(tzinfo=timezone.utc),
            association=item.get("author_association") or "NONE",
            kind="review",
        )
        for item in raw
        if (item.get("body") or "").strip()
    )


def fetch_override_actor(slug: str, number: int) -> str:
    """Who most recently applied ``feedback-override``, or "" if unknown.

    Read from the issue events timeline, which is the only place the *actor* of a
    label is recorded — the labels on a PR carry no provenance at all. Empty on
    any failure, and ``classify`` reads that as "unknown" rather than as "the
    author", so a timeline this cannot fetch never turns into a blocked PR.
    """
    raw = _read_paged(f"repos/{slug}/issues/{number}/events")
    if raw is None:
        return ""
    actor = ""
    for item in raw:
        if not isinstance(item, dict) or item.get("event") != "labeled":
            continue
        if ((item.get("label") or {}).get("name")) != OVERRIDE_LABEL:
            continue
        # Events arrive oldest-first; the last one wins, matching "most recently
        # applied" after any remove/re-add.
        actor = ((item.get("actor") or {}).get("login")) or ""
    return actor


def fetch_ci(slug: str, head_sha: str) -> CIState:
    """The CI run for this exact SHA — not the branch's latest, which may be older."""
    if not head_sha:
        return CIState(None, None)
    payload = _read(f"repos/{slug}/actions/runs?head_sha={head_sha}&per_page=100")
    if not isinstance(payload, dict):
        return CIState(None, None)
    runs = [run for run in payload.get("workflow_runs") or [] if run.get("name") == CI_WORKFLOW_NAME]
    if not runs:
        return CIState(None, None)
    latest = max(runs, key=lambda run: run.get("created_at") or "")
    return CIState(latest.get("conclusion"), parse_timestamp(latest.get("updated_at")))


def repo_slug() -> str | None:
    return transport.resolve_slug(ROOT)


def current_pr() -> int | None:
    """The open PR for the branch this checkout is on.

    `gh pr view` works this out from the current branch. Without `gh` there is no
    "current" anything, so the REST branch asks the same question explicitly:
    which open PR has this branch as its head. A detached HEAD or a branch with
    no PR answers None, which the caller already reports as "say which PR".
    """
    if transport.gh_available():
        payload = _gh_json("pr", "view", "--json", "number")
        return int(payload["number"]) if isinstance(payload, dict) and payload.get("number") else None
    slug = repo_slug()
    branch = _current_branch()
    if not slug or not branch:
        return None
    owner, _, _name = slug.partition("/")
    payload = _read(f"repos/{slug}/pulls?head={owner}:{branch}&state=open&per_page=1")
    if not isinstance(payload, list) or not payload:
        return None
    number = payload[0].get("number")
    return int(number) if number else None


def _current_branch() -> str | None:
    result = subprocess.run(  # noqa: S603 - literal argv
        ["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    branch = result.stdout.strip()
    return branch if result.returncode == 0 and branch and branch != "HEAD" else None


# --- emit --------------------------------------------------------------------


def post_status(slug: str, head_sha: str, verdict: Verdict, target_url: str | None = None) -> bool:
    fields: dict[str, object] = {
        "state": verdict.state,
        "context": STATUS_CONTEXT,
        "description": verdict.description[:DESCRIPTION_LIMIT],
    }
    if target_url:
        fields["target_url"] = target_url
    return _write("POST", f"repos/{slug}/statuses/{head_sha}", fields)


def review_ledger(snapshot: Snapshot) -> list[str]:
    """One line per review round: what it found, and what was written back.

    The process, made readable in one place. All of it is already on the PR — the
    findings in the reviewer's comments, the answers in the replies below them —
    but on a PR with two rounds and a fix in between, working out which reply
    answered which round means reading the timeline in order and holding the
    dates in your head. The gate has already done that to decide the status, so
    it may as well show its working.
    """
    producer = PRODUCERS[0]
    history = verdict_history(snapshot.comments, producer)
    lines: list[str] = []
    round_number = 0
    for index, (comment, count, critical) in enumerate(history):
        if count == 0:
            lines.append("- A later pass reported no findings")
            continue
        round_number += 1
        severity = f", {critical} critical" if critical else ""
        noun = "finding" if count == 1 else "findings"
        required = settled_count(history, index)
        if required:
            # A round the reviewer has already moved past. What is being looked
            # for is the account of what changed.
            account = account_for(
                snapshot.comments,
                producer.key,
                comment.written_at,
                comment.id,
                required,
                deny_dismissal_from=snapshot.author if is_unattended(snapshot) else "",
            )
            missing = "**nothing written back yet**"
        else:
            # The newest round. Nothing has settled, so there is nothing to
            # account for yet — the only thing that closes it short of a fix is a
            # dismissal, which is a stricter rule and a different lookup.
            account = acknowledgement(
                snapshot.comments,
                producer.key,
                after=comment.written_at,
                after_id=comment.id,
                deny_author=snapshot.author if is_unattended(snapshot) else None,
            )
            missing = "still open"
        if account is None:
            answer = missing
        else:
            response = responses(account.body)[producer.key]
            claims = [
                f"{response.fixed} fixed" if response.fixed else "",
                f"{response.answered} answered" if response.answered else "",
            ]
            said = ", ".join(claim for claim in claims if claim) or "all answered"
            answer = f"{said} by @{account.author}"
        lines.append(f"- Round {round_number} — {count} {noun}{severity} → {answer}")
    return lines


def _ledger_block(snapshot: Snapshot) -> list[str]:
    """The ledger as sticky-comment lines, or nothing when no review has spoken."""
    ledger = review_ledger(snapshot)
    return ["", "**Review ledger**", "", *ledger] if ledger else []


def sticky_body(snapshot: Snapshot, verdict: Verdict) -> str:
    body = _sticky_body(snapshot, verdict)
    if verdict.advisory_items:
        body += "\n\n**Codex review — advisory**\n\n" + "\n".join(f"- {item.detail}" for item in verdict.advisory_items)
    return body


def _sticky_body(snapshot: Snapshot, verdict: Verdict) -> str:
    """The comment a human reads when the check goes red — what, and how to clear it."""
    lines = [STICKY_MARKER]
    if verdict.state == "pending":
        # Pending holds the merge just as firmly as a failure. Calling it clear is
        # the one thing this comment must never say.
        lines += [f"**Review feedback: waiting.** {verdict.description}"]
        return "\n".join(lines)
    if verdict.state == "success" and verdict.reason == "advisory":
        # Green because of *who* wrote the branch, not because there is nothing
        # to say. Saying "clear" here would be a lie about the findings below it,
        # and saying "blocked" would be a lie about the check.
        noun = "finding" if len(verdict.items) == 1 else "findings"
        lines += [
            f"**{len(verdict.items)} review {noun} — advisory.** This is a local branch, so this "
            "check does not hold the merge. Nothing here has to be answered before you merge; it "
            "is here because somebody read the diff.",
            "",
            *[f"- {item.detail}" for item in verdict.items],
            "",
            "The enforced lane is machine-authored PRs — `cowork/…`, `feature/issue-…`, "
            f"`security/codeql-triage…`, `ci-sentinel/…`, and anything labelled `{COWORK_LABEL}` — "
            "where there is nobody on the other end to weigh a finding. You are the person this "
            "would otherwise be arguing with, so it does not argue.",
            "",
            "A *person's* unresolved thread, or a `Request changes` review, still holds this check "
            "even here — that one has somebody on the other end by construction.",
        ]
        return "\n".join(lines)
    if verdict.state == "success" and verdict.items:
        # Green, but not clean, and the comment must not blur the two. These
        # findings were read by nobody and fixed by nobody; the merge stopped
        # waiting for them because the review ran out of blocking rounds, which
        # is a decision about the loop and not a judgement about the code.
        noun = "finding" if len(verdict.items) == 1 else "findings"
        lines += [
            f"**{len(verdict.items)} {noun} recorded, not fixed.** {verdict.description}. This PR can merge.",
            "",
            *[f"- {item.detail}" for item in verdict.items],
            "",
            "An adversarial review of a large diff finds something every time, so the loop is "
            f"bounded rather than run to zero — past round {BLOCKING_ROUNDS} only a finding the "
            "reviewer marks `critical` holds the merge. What is above was left undone deliberately "
            f"— it is worth reading before merging, and worth filing if it matters. The "
            f"`{CAPPED_LABEL}` label marks this PR so it can be found again.",
            *_ledger_block(snapshot),
        ]
        return "\n".join(lines)
    if verdict.state != "failure":
        lines += [f"**Review feedback: clear.** {verdict.description}"]
        return "\n".join(lines)
    single = len(verdict.items) == 1
    noun = "item" if single else "items"
    tail = "it is answered" if single else "they are answered"
    lines += [
        f"**{len(verdict.items)} unanswered review {noun}** — this PR is blocked until {tail}.",
        "",
    ]
    lines += [f"- {item.detail}" for item in verdict.items]
    # The advice has to match the PR. On an unattended PR an `<!-- addressed: -->`
    # reply from the PR's own author is discarded, and on the cowork lane that
    # author *is* the maintainer — so a human who reads this comment, disagrees
    # with a finding and replies exactly as instructed would get silence and a
    # re-rendered red check, with nothing anywhere saying why. Telling somebody to
    # do the one thing that cannot work is worse than telling them nothing.
    if is_unattended(snapshot):
        lines += [
            "",
            "Run `/pr-feedback " + str(snapshot.number) + "` to work through them, or by hand: fix the "
            "finding and push, **then reply saying what you changed**, ending the reply with "
            "`<!-- addressed: claude-review fixed=N answered=M -->`. Reply in a review thread before "
            "you hit **Resolve conversation** on it.",
            "",
            "**Pushing the fix is not enough on its own here.** The next review pass reports `open=0` "
            "and the finding stops being listed — which means the entire record of what happened to "
            "it is a number going down, and nobody can review a subtraction. `fixed=N` is that "
            "record, and the reviewer's own re-read is what checks it: claim a fix that is not there "
            "and the finding comes straight back.",
            "",
            "**A reply cannot *dismiss* a finding here.** An `answered=` claim from this PR's own "
            "author is ignored — the account that wrote the change would otherwise be answering the "
            "review of it. Fix it, or hand it to a human: the "
            f"`{OVERRIDE_LABEL}` label is their call, and is recorded here.",
            *_ledger_block(snapshot),
        ]
        return "\n".join(lines)
    lines += [
        "",
        "Run `/pr-feedback " + str(snapshot.number) + "` to work through them, or by hand: fix and push "
        "(the next review pass reports `open=0` and this clears itself), or reply with "
        "`<!-- addressed: <producer> -->` and a reason, or hit **Resolve conversation** on a thread you "
        "have answered.",
        "",
        f"_Stuck? The `{OVERRIDE_LABEL}` label clears this check and says so here._",
    ]
    return "\n".join(lines)


def upsert_sticky(slug: str, number: int, snapshot: Snapshot, verdict: Verdict) -> None:
    """One comment, edited in place. Never created just to say everything is fine."""
    existing = None
    for comment in snapshot.comments:
        # `kind` matters: a review body and an issue comment have separate id
        # spaces, so PATCHing the issue-comments endpoint with a review id would
        # edit an unrelated comment.
        if comment.kind == "comment" and STICKY_MARKER in comment.body:
            existing = comment.id
    body = sticky_body(snapshot, verdict)
    if existing is not None:
        _write("PATCH", f"repos/{slug}/issues/comments/{existing}", {"body": body})
    elif verdict.state == "failure":
        # Created only for a red check, so an advisory local PR never grows a
        # second comment restating what the review comment above it already
        # says. An advisory verdict still *edits* an existing sticky, which is
        # how a PR that went red before this lane existed corrects itself.
        _write("POST", f"repos/{slug}/issues/{number}/comments", {"body": body})


def apply_capped_label(slug: str, number: int, snapshot: Snapshot, verdict: Verdict) -> None:
    """Mark a PR that merged past unfixed findings, so it can be found again.

    Only added, never removed: the label records that a cap was hit on this PR,
    and a later push that happens to produce a clean review does not undo the
    fact. Idempotent — GitHub ignores adding a label that is already present, and
    the guard below keeps it to one call.

    Applying a label that does not exist silently does nothing, so
    ``review-capped`` is in ``expected_labels()`` and the repo's label setup creates
    it. A missing label costs the record, not the merge.
    """
    # `reason`, not `bool(items)`: a green status now carries items in two
    # different situations, and only one of them is a PR that merged past
    # findings. Labelling every advisory local PR `review-capped` would make the
    # label mean nothing within a week.
    if verdict.reason != "capped" or CAPPED_LABEL in snapshot.labels:
        return
    # `labels[]=` is `gh api`'s spelling of a repeated field; the JSON body wants
    # a real list. `_write` sends what each transport reads, so the value is a
    # list here and the gh branch flattens it. Adds rather than replaces either
    # way — POST to the collection, never PUT (PUT replaces the whole label set).
    _write("POST", f"repos/{slug}/issues/{number}/labels", {"labels": [CAPPED_LABEL]})


def unreadable_verdict(meta: object) -> Verdict:
    """What to post when the PR could not be read — decided by lane.

    Pending is the honest answer and, on the enforced lane, the right one: a
    machine PR that cannot be read must not merge on the strength of a read that
    failed.

    On a local branch it is the wrong one, and dangerously so. `pr-feedback` is a
    *required* context, and a required context sitting pending blocks the merge
    as hard as a red one — so one `gh` hiccup would wedge a person's own PR
    behind a gate that is not supposed to enforce on them at all. The lane is
    recoverable from the one-shot `pulls/{n}` read even when the full snapshot
    is not: it needs only the head ref and the labels, which is exactly what
    ``is_unattended`` reads.

    An unreadable *lane* falls to pending, because "we could not tell" is not "it
    is yours" — so this needs a head ref it positively read, not merely the
    absence of one. A payload thin enough to carry a SHA and no branch name is
    the unknown case, not the local one.
    """
    if isinstance(meta, dict):
        ref = str((meta.get("head") or {}).get("ref") or "")
        labels = {label.get("name", "") for label in meta.get("labels") or [] if isinstance(label, dict)}
        if ref and not (COWORK_LABEL in labels or ref.startswith(UNATTENDED_BRANCH_PREFIXES)):
            return Verdict(
                "success",
                "could not read the PR — advisory on a local branch, not enforced (see the run log)",
                reason="advisory",
            )
    return Verdict("pending", "could not read the PR — see the run log")


def render_report(snapshot: Snapshot, verdict: Verdict) -> str:
    icon = {"success": "OK", "pending": "..", "failure": "XX"}[verdict.state]
    lines = [f"[{icon}] PR #{snapshot.number} @ {snapshot.head_sha[:7]} — {verdict.description}"]
    for item in verdict.items:
        lines.append(f"       · {item.detail}")
    for item in verdict.advisory_items:
        lines.append(f"       · Codex (advisory): {item.detail}")
    # Printed on a clean PR too, and deliberately: "the gate is green" and "the
    # findings were answered" are different sentences, and this is the second.
    for line in review_ledger(snapshot):
        lines.append(f"       {line}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report unanswered review feedback on a pull request.")
    parser.add_argument("--pr", type=int, help="PR number (defaults to the current branch's PR)")
    parser.add_argument("--json", action="store_true", help="machine-readable verdict for /pr-feedback")
    parser.add_argument("--status", action="store_true", help="post the commit status + sticky comment (CI)")
    parser.add_argument("--target-url", help="with --status, the run URL the check links to")
    args = parser.parse_args(argv)

    if not transport.gh_available() and not transport.github_token():
        print(
            "[pr-feedback] no `gh` on PATH and no GH_TOKEN — install gh (brew install gh), "
            "or export GH_TOKEN for the REST fallback",
            file=sys.stderr,
        )
        return 2
    slug = repo_slug()
    if slug is None:
        print("[pr-feedback] could not resolve the repo — is this a GitHub checkout?", file=sys.stderr)
        return 2
    number = args.pr or current_pr()
    if number is None:
        print("[pr-feedback] no PR given and none open for this branch — pass --pr", file=sys.stderr)
        return 2

    snapshot = fetch_snapshot(number, slug)
    if snapshot is None:
        print(f"[pr-feedback] could not read PR #{number} — {unreadable_reason()}", file=sys.stderr)
        if args.status:
            # A *required* status that was never posted shows as "Expected —
            # waiting for status", which looks exactly like this workflow not
            # existing. Say what actually happened, on whatever SHA is reachable.
            # REST spells it `head.sha`; the GraphQL read above spells the same
            # value `headRefOid`. This path exists for when that read failed.
            meta = _read(f"repos/{slug}/pulls/{number}")
            head = (meta.get("head") or {}).get("sha") if isinstance(meta, dict) else None
            if head:
                post_status(slug, head, unreadable_verdict(meta), args.target_url)
        return 2
    verdict = classify(snapshot, datetime.now(timezone.utc))

    if args.status:
        # The step succeeds whatever the verdict — the *status* is what blocks a
        # merge. A red step here would look like a broken workflow instead of an
        # unanswered review, and would be the first thing anyone disabled.
        ok = post_status(slug, snapshot.head_sha, verdict, args.target_url)
        upsert_sticky(slug, number, snapshot, verdict)
        apply_capped_label(slug, number, snapshot, verdict)
        print(render_report(snapshot, verdict))
        return 0 if ok else 2

    if args.json:
        payload = {
            "number": snapshot.number,
            "head_sha": snapshot.head_sha,
            **verdict.as_dict(),
            # For `/pr-feedback` and the `pr-responder` agent: which rounds still
            # need an account, without re-deriving the timeline from `gh pr view`.
            "ledger": review_ledger(snapshot),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_report(snapshot, verdict))
    return 1 if verdict.state == "failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
