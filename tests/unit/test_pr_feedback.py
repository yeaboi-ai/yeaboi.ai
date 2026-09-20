"""Tests for scripts/pr_feedback.py — the gate that blocks a merge on unread review feedback.

The script's whole value is that it is *fail-closed and cannot deadlock*, and both
halves of that are easy to break in a way nothing notices. A gate that is too
lenient re-creates the original problem silently — PRs merge past findings again,
and the check sits green while they do. A gate that is too strict is worse in
practice, because the first PR it wedges is the one that gets it deleted.

So every state in the table below is pinned here, on hand-built snapshots. Nothing
in this file calls ``gh`` or touches the network: ``classify`` and everything it
calls are pure, which is exactly why the fetching was kept out of them.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# scripts/ is not a package, so load the module straight from its file path.
_MODULE_PATH = ROOT / "scripts" / "pr_feedback.py"
_spec = importlib.util.spec_from_file_location("pr_feedback", _MODULE_PATH)
prf = importlib.util.module_from_spec(_spec)
# Registered before exec: @dataclass resolves annotations through
# sys.modules[cls.__module__], which is None for a module loaded off a path.
sys.modules["pr_feedback"] = prf
_spec.loader.exec_module(prf)

NOW = datetime(2026, 8, 6, 12, 0, 0, tzinfo=timezone.utc)
AUTHOR = "dinho"
# A second human with write access — not the PR's author. On the unattended lane
# an acknowledgement from the author is discarded, so this is who answers.
MAINTAINER = "a-maintainer"
HEAD = "abc1234def5678"

REVIEW_MARK = "<!-- pr-feedback: claude-review open={n} -->"
CRITICAL_MARK = "<!-- pr-feedback: claude-review open={n} critical={c} -->"
DOD_MARK = "<!-- cowork-dod open={n} -->"


def comment(
    body: str,
    *,
    minutes_ago: int = 60,
    author: str = "github-actions[bot]",
    ident: int = 1,
    association: str = "NONE",
    edited_minutes_ago: int | None = None,
    kind: str = "comment",
):
    return prf.Comment(
        id=ident,
        author=author,
        body=body,
        created_at=NOW - timedelta(minutes=minutes_ago),
        updated_at=None if edited_minutes_ago is None else NOW - timedelta(minutes=edited_minutes_ago),
        association=association,
        kind=kind,
    )


# The login `claude-review.yml` actually posts under — the Claude GitHub App.
# `is_authentic_verdict` pins the producer to it, so a fixture using any other
# identity is not testing the real thing.
REVIEWER = "claude[bot]"


def review(
    n: int,
    *,
    critical: int | None = None,
    minutes_ago: int = 60,
    ident: int = 1,
    edited_minutes_ago: int | None = None,
    author: str = REVIEWER,
):
    """A review verdict. ``critical=None`` writes the pre-severity marker shape,
    which is what every verdict already sitting on an open PR looks like."""
    mark = REVIEW_MARK.format(n=n) if critical is None else CRITICAL_MARK.format(n=n, c=critical)
    return comment(
        f"Findings...\n\n{mark}",
        minutes_ago=minutes_ago,
        ident=ident,
        author=author,
        edited_minutes_ago=edited_minutes_ago,
    )


def forged(
    n: int = 0,
    *,
    producer: str = "claude-review",
    minutes_ago: int = 5,
    ident: int = 9,
    author: str = "a-stranger",
    association: str = "NONE",
):
    """A verdict somebody other than the producer wrote, claiming ``n`` findings."""
    mark = REVIEW_MARK if producer == "claude-review" else DOD_MARK
    return comment(mark.format(n=n), minutes_ago=minutes_ago, ident=ident, author=author, association=association)


def ack(
    producer: str = "claude-review",
    *,
    minutes_ago: int = 30,
    ident: int = 2,
    association: str = "OWNER",
    author: str = MAINTAINER,
    **kw,
):
    """A won't-fix reply. Trusted by default — the untrusted case is its own test.

    Written by somebody other than the PR's author, because the default snapshot
    below is an *unattended* PR and there the author's own acknowledgement is
    discarded by design. Tests about that rule pass ``author=AUTHOR`` explicitly.
    """
    return comment(
        f"Won't fix, because the caller already guards it.\n\n<!-- addressed: {producer} -->",
        minutes_ago=minutes_ago,
        author=author,
        ident=ident,
        association=association,
        **kw,
    )


def account(
    producer: str = "claude-review",
    *,
    fixed: int = 3,
    answered: int = 0,
    minutes_ago: int = 20,
    ident: int = 5,
    association: str = "OWNER",
    author: str = AUTHOR,
    **kw,
):
    """The reply that says what was done about a round's findings.

    Written by the PR's **own author** by default, which is the whole difference
    from ``ack()``. A fix is a claim of work that the reviewer's next pass
    checks, so the account that did the work is exactly who should be writing it
    down — where a *dismissal* from that same account is refused on the
    unattended lane, because nothing checks it.
    """
    claims = " ".join(part for part in (f"fixed={fixed}" if fixed else "", f"answered={answered}" if answered else ""))
    assert claims.strip(), "an account with no counts is a bare ack — use ack()"
    return comment(
        f"Round answered:\n- hoisted the guard (abc1234)\n\n<!-- addressed: {producer} {claims.strip()} -->",
        minutes_ago=minutes_ago,
        author=author,
        ident=ident,
        association=association,
        **kw,
    )


def thread(
    *,
    resolved: bool = False,
    authors: tuple[str, ...] = ("reviewer",),
    ident: str = "T1",
    outdated: bool = False,
    resolved_by: str = "",
):
    return prf.Thread(
        id=ident,
        is_resolved=resolved,
        is_outdated=outdated,
        path="src/yeaboi/standup/collector.py",
        line=88,
        authors=authors,
        excerpt="this drops the last page",
        resolved_by=resolved_by,
    )


def snapshot(**overrides):
    """A PR whose CI went green an hour ago — old enough that the grace has lapsed.

    **On the enforced lane by default.** Almost everything below is a test about
    when the gate blocks, and after the lane split the gate only ever blocks an
    unattended PR — so a fixture on a human branch would make every one of those
    tests assert `success` for the wrong reason. `local()` is the other lane, and
    `TestTheLocalLane` is where it is exercised.
    """
    base = dict(
        number=123,
        head_sha=HEAD,
        author=AUTHOR,
        head_ref="cowork/gate-fixture",
        is_draft=False,
        labels=(),
        review_decision=None,
        comments=(),
        threads=(),
        ci=prf.CIState("success", NOW - timedelta(minutes=60)),
        threads_truncated=False,
    )
    base.update(overrides)
    return prf.Snapshot(**base)


def local(**overrides):
    """The same PR on a branch a person is sitting at the keyboard for."""
    return snapshot(head_ref="feature/nice-thing", **overrides)


class TestProducerVerdicts:
    def test_a_clean_review_passes(self):
        verdict = prf.classify(snapshot(comments=(review(0),)), NOW)
        assert verdict.state == "success"
        assert verdict.items == ()

    def test_unanswered_findings_fail(self):
        verdict = prf.classify(snapshot(comments=(review(2),)), NOW)
        assert verdict.state == "failure"
        assert len(verdict.items) == 1
        assert "2 unanswered findings" in verdict.items[0].detail

    def test_one_finding_reads_singular(self):
        verdict = prf.classify(snapshot(comments=(review(1),)), NOW)
        assert "1 unanswered finding (" in verdict.items[0].detail

    def test_a_newer_acknowledgement_clears_them(self):
        snap = snapshot(comments=(review(2, minutes_ago=60), ack(minutes_ago=30)))
        assert prf.classify(snap, NOW).state == "success"

    def test_an_older_acknowledgement_does_not(self):
        # The reply predates the review pass, so it answered something else — the
        # case a naive "is there an ack anywhere" check would wave straight through.
        snap = snapshot(comments=(review(2, minutes_ago=30), ack(minutes_ago=60)))
        assert prf.classify(snap, NOW).state == "failure"

    def test_an_acknowledgement_is_scoped_to_its_producer(self):
        snap = snapshot(comments=(review(2, minutes_ago=60), ack("cowork-dod", minutes_ago=30)))
        assert prf.classify(snap, NOW).state == "failure"

    def test_only_the_latest_pass_counts(self):
        # The fix landed and the re-review said so. The *findings* stop blocking
        # on the strength of the newer verdict alone — nobody has to argue them
        # down, which is what stops the gate becoming a box nobody can tick. What
        # still has to exist is a line saying what changed: TestFixesAreAccounted.
        snap = snapshot(
            comments=(review(3, minutes_ago=90, ident=1), review(0, minutes_ago=10, ident=2), account(fixed=3))
        )
        assert prf.classify(snap, NOW).state == "success"

    def test_a_regression_reopens_the_gate(self):
        snap = snapshot(comments=(review(0, minutes_ago=90, ident=1), review(1, minutes_ago=10, ident=2)))
        assert prf.classify(snap, NOW).state == "failure"

    def test_equal_timestamps_break_ties_on_id(self):
        snap = snapshot(
            comments=(
                review(4, minutes_ago=10, ident=1),
                review(0, minutes_ago=10, ident=2),
                account(fixed=4, minutes_ago=5),
            )
        )
        assert prf.classify(snap, NOW).state == "success"

    def test_the_dod_audit_blocks_when_it_reports_unmet_items(self):
        snap = snapshot(comments=(review(0), comment(DOD_MARK.format(n=2), ident=3)))
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "failure"
        assert verdict.items[0].key == "cowork-dod"

    def test_the_dod_audit_is_never_waited_on(self):
        # It is a hand-registered cowork routine that may not be deployed at all.
        assert prf.classify(snapshot(comments=(review(0),)), NOW).state == "success"

    def test_a_markerless_dod_comment_is_not_a_verdict(self):
        # The pre-`open=` format must read as "said nothing", not as "said zero" —
        # otherwise a routine nobody updated could clear the gate by staying quiet.
        snap = snapshot(comments=(review(0), comment("<!-- cowork-dod -->", ident=3)))
        assert prf.latest_verdict(snap.comments, prf.PRODUCERS[1]) is None


class TestTheReviewCap:
    """The loop needs a terminator, and the terminator must not break the gate.

    Four consecutive rounds on PR #222 each produced real should-fix findings.
    An adversarial review of a large diff always finds something, so "merge when
    the reviewer reports zero" is not a condition that reliably arrives — and a
    gate whose exit may never occur is a gate that gets deleted.
    """

    def test_under_the_cap_findings_still_block(self):
        snap = snapshot(comments=(review(2, minutes_ago=60, ident=1),))
        assert prf.classify(snap, NOW).state == "failure"

    def test_at_the_cap_the_gate_opens(self):
        snap = snapshot(
            comments=(review(2, minutes_ago=90, ident=1), review(1, minutes_ago=10, ident=2), account(fixed=1))
        )
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "success"
        assert "capped" in verdict.description

    def test_the_findings_are_kept_not_discarded(self):
        """Green is not clean. The items ride along so the comment can list them."""
        snap = snapshot(
            comments=(review(2, minutes_ago=90, ident=1), review(3, minutes_ago=10, ident=2), account(fixed=1))
        )
        verdict = prf.classify(snap, NOW)
        assert verdict.items and verdict.items[0].key == "claude-review"
        assert "not fixed" in verdict.description

    def test_a_clean_pass_does_not_spend_a_round(self):
        """Otherwise a regression after a clean review could never reopen the gate.

        Counting every verdict would make "review found a new problem" and
        "review ran out of patience" the same state.
        """
        snap = snapshot(comments=(review(0, minutes_ago=90, ident=1), review(1, minutes_ago=10, ident=2)))
        assert prf.classify(snap, NOW).state == "failure"

    def test_a_forged_verdict_cannot_burn_a_round(self):
        """Otherwise two comments from anybody would cap somebody else's PR."""
        forged = comment(REVIEW_MARK.format(n=1), minutes_ago=50, ident=9, author="a-stranger")
        rounds = prf.review_rounds((review(1, minutes_ago=60, ident=1), forged), prf.PRODUCERS[0])
        assert rounds == 1

    def test_an_unresolved_human_thread_is_never_capped(self):
        """A person waiting for an answer is not a loop that has run too long."""
        snap = snapshot(
            comments=(review(1, minutes_ago=90, ident=1), review(1, minutes_ago=10, ident=2), account(fixed=1)),
            threads=(thread(authors=("a-reviewer",)),),
        )
        assert prf.classify(snap, NOW).state == "failure"

    def test_a_changes_requested_review_is_never_capped(self):
        snap = snapshot(
            comments=(review(1, minutes_ago=90, ident=1), review(1, minutes_ago=10, ident=2), account(fixed=1)),
            review_decision="CHANGES_REQUESTED",
        )
        assert prf.classify(snap, NOW).state == "failure"

    def test_the_capped_comment_does_not_read_as_clean(self):
        snap = snapshot(
            comments=(review(2, minutes_ago=90, ident=1), review(2, minutes_ago=10, ident=2), account(fixed=1))
        )
        body = prf.sticky_body(snap, prf.classify(snap, NOW))
        assert "recorded, not fixed" in body
        assert "clear" not in body.lower().split("recorded")[0]
        assert prf.CAPPED_LABEL in body

    def test_the_workflow_and_the_script_cap_at_the_same_numbers(self):
        """Reviewing past the cap writes findings nothing will ever act on.

        The workflow holds both caps as bash literals, which no other check in
        this repo can see. They are mirrored from the two constants here, so this
        is the only thing standing between them and a silent drift.
        """
        text = (ROOT / ".github" / "workflows" / "claude-review.yml").read_text(encoding="utf-8")
        assert f"cap={prf.MAX_REVIEW_ROUNDS}" in text, "claude-review.yml stopped agreeing with MAX_REVIEW_ROUNDS"
        assert f"cap={prf.LOCAL_REVIEW_ROUNDS}" in text, "claude-review.yml stopped agreeing with LOCAL_REVIEW_ROUNDS"
        assert '-ge "$cap" ]' in text, "the cap stopped being applied"

    def test_the_workflow_counts_a_marker_that_carries_a_severity(self):
        """The round counter greps the marker itself. Anchoring it on the closing
        `-->` — which the pre-severity shape ended with — would read zero rounds
        forever the moment the reviewer started emitting `critical=`, and the loop
        would never terminate."""
        text = (ROOT / ".github" / "workflows" / "claude-review.yml").read_text(encoding="utf-8")
        capture = re.search(r'capture\("(<!-- pr-feedback[^"]*)"\)', text)
        assert capture is not None, "the round counter's capture expression moved"
        pattern = capture.group(1).replace("(?<n>", "(?P<n>")
        for marker in (REVIEW_MARK.format(n=2), CRITICAL_MARK.format(n=2, c=1)):
            assert re.search(pattern, marker), f"the workflow would not count {marker!r}"


class TestTheLocalLane:
    """A branch a person is sitting at the keyboard for is never held by this gate.

    Requirement, and the reason: the enforced lane exists because an unattended
    PR has nobody on the other end to weigh a finding against the cost of not
    merging. A human's own branch has exactly that person, already reading the
    review, and holding their merge to make them type a marker at themselves is
    ceremony rather than review.
    """

    def test_findings_do_not_block(self):
        verdict = prf.classify(local(comments=(review(3),)), NOW)
        assert verdict.state == "success"
        assert verdict.reason == "advisory"
        assert "advisory" in verdict.description

    def test_the_findings_still_ride_along(self):
        """Advisory is not silent — the items are what the comment lists."""
        verdict = prf.classify(local(comments=(review(3),)), NOW)
        assert verdict.items and verdict.items[0].key == "claude-review"

    def test_a_clean_pr_reads_as_clean_not_as_advisory(self):
        verdict = prf.classify(local(comments=(review(0),)), NOW)
        assert verdict.state == "success"
        assert verdict.reason == ""
        assert verdict.items == ()

    @pytest.mark.parametrize(
        "ci,label",
        [
            (prf.CIState(None, None), "no CI run yet"),
            (prf.CIState("failure", NOW - timedelta(hours=2)), "red CI"),
            (prf.CIState("success", NOW - timedelta(minutes=5)), "inside the grace window"),
            (prf.CIState("success", NOW - timedelta(hours=3)), "past the grace window, review missing"),
        ],
    )
    def test_it_is_never_pending(self, ci, label):
        """The half that makes arming the required check safe.

        A *required* context sitting pending blocks a merge exactly as hard as a
        red one. Every state that produces `pending` on the enforced lane must
        produce `success` here, or one hiccup in a workflow that is explicitly
        allowed to not run wedges a person's own PR with nothing to act on.
        """
        assert prf.classify(local(ci=ci), NOW).state == "success", label

    def test_an_unresolved_human_thread_still_blocks(self):
        """The lane withdraws the machine reviewer, not the human one.

        This asserted `success` when the lane first landed, on the reasoning that
        GitHub's own review UI is where a human PR's author already looks. The
        argument for going advisory is "nobody on the other end to weigh a
        finding", and that is false by construction here — so this is the one
        part of item 10 the local lane can still meaningfully enforce, and
        `definition-of-done.md` promises it in as many words.
        """
        verdict = prf.classify(local(comments=(review(0),), threads=(thread(authors=("a-reviewer",)),)), NOW)
        assert verdict.state == "failure"
        assert any(item.kind == "thread" for item in verdict.items)

    def test_changes_requested_still_blocks(self):
        verdict = prf.classify(local(comments=(review(0),), review_decision="CHANGES_REQUESTED"), NOW)
        assert verdict.state == "failure"

    def test_the_comment_says_advisory_and_not_clear(self):
        body = prf.sticky_body(local(comments=(review(2),)), prf.classify(local(comments=(review(2),)), NOW))
        assert "advisory" in body.lower()
        assert "recorded, not fixed" not in body

    def test_it_is_not_labelled_review_capped(self):
        """`review-capped` records a PR that merged past findings on the enforced
        lane. Applying it to every advisory local PR would empty it of meaning."""
        assert prf.classify(local(comments=(review(2),)), NOW).reason != "capped"

    def test_the_enforced_lane_is_unchanged_by_all_of_this(self):
        assert prf.classify(snapshot(comments=(review(3),)), NOW).state == "failure"


class TestCriticalFindings:
    """`critical=M` is the only thing that survives the first blocking round."""

    def test_a_marker_without_the_field_reads_as_zero(self):
        """Every verdict already on an open PR looks like this. It must keep
        parsing, and must mean the *least* blocking thing it could mean — the
        field's only power is to hold a merge, so absent has to be zero."""
        match = prf.PRODUCERS[0].pattern.search(REVIEW_MARK.format(n=4))
        assert match is not None and prf.verdict_counts(match) == (4, 0)

    def test_the_field_is_read_when_present(self):
        match = prf.PRODUCERS[0].pattern.search(CRITICAL_MARK.format(n=4, c=2))
        assert match is not None and prf.verdict_counts(match) == (4, 2)

    def test_the_first_round_blocks_on_ordinary_findings(self):
        assert prf.classify(snapshot(comments=(review(2),)), NOW).state == "failure"

    def test_the_second_round_does_not(self):
        snap = snapshot(
            comments=(review(2, minutes_ago=90, ident=1), review(2, minutes_ago=10, ident=2), account(fixed=1))
        )
        assert prf.classify(snap, NOW).state == "success"

    def test_the_second_round_does_when_it_is_critical(self):
        """The whole reason the field exists: a blocker the first round's fix
        introduced still holds the merge."""
        snap = snapshot(
            comments=(
                review(2, minutes_ago=90, ident=1),
                review(2, critical=1, minutes_ago=10, ident=2),
                account(fixed=1),
            )
        )
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "failure"
        # And for the stated reason, not because round one went unaccounted for.
        assert [item.kind for item in verdict.items] == ["producer"]

    def test_a_stale_critical_does_not_keep_blocking(self):
        """Read from the latest verdict only, for the same reason `open` is: the
        fix landed and the reviewer said so."""
        snap = snapshot(
            comments=(
                review(2, critical=2, minutes_ago=90, ident=1),
                review(1, critical=0, minutes_ago=10, ident=2),
                account(fixed=1),
            )
        )
        assert prf.classify(snap, NOW).state == "success"

    def test_critical_is_irrelevant_on_the_local_lane(self):
        assert prf.classify(local(comments=(review(2, critical=5),)), NOW).state == "success"

    def test_a_critical_verdict_still_clears_when_it_is_fixed(self):
        snap = snapshot(
            comments=(
                review(2, critical=1, minutes_ago=90, ident=1),
                review(0, minutes_ago=10, ident=2),
                account(fixed=2),
            )
        )
        assert prf.classify(snap, NOW).state == "success"


class TestFixesAreAccounted:
    """A machine that fixes a finding has to say so. The old rule was the opposite.

    "Push, and the next verdict reads open=0" cleared the gate with nothing on
    the PR saying what changed — which is fine when a person did the fixing and
    is about to click merge, and is a silence when an agent did it and merges
    itself. The whole record of what happened to three findings was a number
    going down, and a subtraction cannot be reviewed.

    So a findings-bearing verdict the reviewer has moved past now needs an
    account newer than it. Note what is *not* asked for: the finding is never
    re-opened, and nobody has to agree with the fix.
    """

    def test_a_silent_fix_no_longer_clears_the_gate(self):
        snap = snapshot(comments=(review(3, minutes_ago=90, ident=1), review(0, minutes_ago=10, ident=2)))
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "failure"
        assert [item.kind for item in verdict.items] == ["account"]
        assert "stopped being reported" in verdict.items[0].detail

    def test_an_account_clears_it(self):
        snap = snapshot(
            comments=(review(3, minutes_ago=90, ident=1), review(0, minutes_ago=10, ident=2), account(fixed=3))
        )
        assert prf.classify(snap, NOW).state == "success"

    def test_the_pr_author_may_write_it(self):
        """The asymmetry with `is_acknowledged`, stated on its own.

        A claimed fix is checked by the reviewer's next read — claim one that is
        not there and the finding comes straight back. A claimed disagreement is
        checked by nobody, which is why only that one is refused from the account
        that wrote the change.
        """
        snap = snapshot(
            labels=("cowork",),
            comments=(
                review(2, minutes_ago=90, ident=1),
                review(0, minutes_ago=10, ident=2),
                account(fixed=2, author=AUTHOR, association="OWNER"),
            ),
        )
        assert prf.classify(snap, NOW).state == "success"

    def test_a_fixed_claim_still_cannot_dismiss_an_open_finding(self):
        """`fixed=` accounts for work; it never argues a finding down.

        Without this the author's own marker would be a dismissal in disguise —
        the exact hole `is_acknowledged`'s deny_author was added to close.
        """
        snap = snapshot(comments=(review(2, minutes_ago=60, ident=1), account(fixed=2, minutes_ago=10)))
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "failure"
        assert "2 unanswered findings" in verdict.items[0].detail

    def test_an_account_that_claims_too_little_does_not_cover_the_round(self):
        snap = snapshot(
            comments=(review(3, minutes_ago=90, ident=1), review(0, minutes_ago=10, ident=2), account(fixed=2))
        )
        assert prf.classify(snap, NOW).state == "failure"

    def test_only_the_settled_findings_have_to_be_accounted_for(self):
        """Round one found three, round two still reports two of them.

        Asking for a disposition on all three would be asking somebody to write
        one for a finding they have not resolved and should not claim to have.
        """
        snap = snapshot(
            comments=(review(3, minutes_ago=90, ident=1), review(2, minutes_ago=10, ident=2), account(fixed=1))
        )
        assert prf.classify(snap, NOW).state == "success"

    def test_a_round_that_did_not_shrink_still_owes_one_line(self):
        """The floor of one. A count that did not move is not evidence that
        nothing was done — and the marker is a single line either way."""
        snap = snapshot(comments=(review(2, minutes_ago=90, ident=1), review(3, minutes_ago=10, ident=2)))
        assert any(item.kind == "account" for item in prf.classify(snap, NOW).items)

    def test_an_account_older_than_the_round_does_not_count(self):
        snap = snapshot(
            comments=(
                account(fixed=3, minutes_ago=120),
                review(3, minutes_ago=90, ident=1),
                review(0, minutes_ago=10, ident=2),
            )
        )
        assert prf.classify(snap, NOW).state == "failure"

    def test_a_stranger_cannot_write_one(self):
        """Same reason as the ack: this repo is public."""
        snap = snapshot(
            comments=(
                review(3, minutes_ago=90, ident=1),
                review(0, minutes_ago=10, ident=2),
                account(fixed=3, author="a-stranger", association="NONE"),
            )
        )
        assert prf.classify(snap, NOW).state == "failure"

    def test_a_bare_marker_still_accounts_for_everything(self):
        """Every reply already sitting on an open PR is this shape, and it has
        always meant "all of them". It must not become an under-claim overnight."""
        snap = snapshot(
            comments=(review(3, minutes_ago=90, ident=1), review(0, minutes_ago=10, ident=2), ack(minutes_ago=20))
        )
        assert prf.classify(snap, NOW).state == "success"

    def test_a_bare_marker_from_the_pr_author_does_not_account(self):
        """The one hole the split marker left open.

        A bare marker means "all of them, **answered**" — pure dismissal, with no
        claim of work for the next review pass to check. Accepting one from the
        applicant would have let an unattended PR close its whole account with a
        contentless comment, which is exactly the silence this check exists to
        stop. From anyone else it still means what it always meant.
        """
        comments = (review(3, minutes_ago=90, ident=1), review(0, minutes_ago=10, ident=2))
        bare_from_author = comment(
            "Addressed.\n\n<!-- addressed: claude-review -->",
            minutes_ago=20,
            author=AUTHOR,
            ident=7,
            association="OWNER",
        )
        assert prf.classify(snapshot(comments=(*comments, bare_from_author)), NOW).state == "failure"

    def test_a_counted_marker_from_the_pr_author_still_accounts(self):
        """The asymmetry has to stay narrow: `fixed=` is the half the reviewer's
        next read of the diff verifies, so the account that did the work is
        exactly who should be writing it down."""
        comments = (review(3, minutes_ago=90, ident=1), review(0, minutes_ago=10, ident=2))
        assert prf.classify(snapshot(comments=(*comments, account(fixed=3))), NOW).state == "success"

    def test_it_is_not_capped(self):
        """`BLOCKING_ROUNDS` bounds a loop with no natural end. This has one — it
        is a single comment, and it does not get harder the more rounds run."""
        snap = snapshot(
            comments=(
                review(2, minutes_ago=120, ident=1),
                review(2, minutes_ago=60, ident=2),
                review(0, minutes_ago=10, ident=3),
            )
        )
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "failure"
        assert {item.kind for item in verdict.items} == {"account"}

    def test_the_newest_round_is_not_reported_twice(self):
        """It is already held by `open_producers`; two items about one verdict
        would only make the check harder to read."""
        verdict = prf.classify(snapshot(comments=(review(3, minutes_ago=60),)), NOW)
        assert [item.kind for item in verdict.items] == ["producer"]

    def test_the_local_lane_never_reaches_it(self):
        """Inert by construction there — `LOCAL_REVIEW_ROUNDS` is one, so no
        verdict is ever superseded. Pinned anyway, because the thing that would
        change that is a number in another file."""
        snap = local(comments=(review(3, minutes_ago=90, ident=1), review(0, minutes_ago=10, ident=2)))
        assert prf.classify(snap, NOW).state == "success"


class TestSilentlyResolvedThreads:
    """**Resolve conversation** is a claim that the reviewer was heard.

    Every agent prompt in this repo says never to press it on a thread you did
    not answer, and until now that was a convention held up by nothing — the same
    shape as the problem the rest of this gate exists for, and one that scales
    when a machine does it.
    """

    def _snap(self, **kw):
        return snapshot(comments=(review(0),), threads=(thread(**kw),))

    def test_the_author_resolving_without_replying_blocks(self):
        snap = self._snap(resolved=True, resolved_by=AUTHOR, authors=("a-reviewer",))
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "failure"
        assert [item.kind for item in verdict.items] == ["resolved"]

    def test_replying_first_is_the_way_through(self):
        """A resolved thread still takes comments, so nothing has to be re-opened."""
        snap = self._snap(resolved=True, resolved_by=AUTHOR, authors=("a-reviewer", AUTHOR))
        assert prf.classify(snap, NOW).state == "success"

    def test_somebody_else_resolving_it_is_their_call(self):
        snap = self._snap(resolved=True, resolved_by="a-maintainer", authors=("a-reviewer",))
        assert prf.classify(snap, NOW).state == "success"

    def test_a_thread_only_the_author_wrote_in_has_nothing_to_answer(self):
        snap = self._snap(resolved=True, resolved_by=AUTHOR, authors=(AUTHOR,))
        assert prf.classify(snap, NOW).state == "success"

    def test_an_unreadable_resolver_is_never_accused(self):
        """`resolvedBy` is null when the account is gone. This check names
        somebody, so it does not run on a field it did not read."""
        snap = self._snap(resolved=True, resolved_by="", authors=("a-reviewer",))
        assert prf.classify(snap, NOW).state == "success"

    def test_a_person_on_their_own_pr_is_untouched(self):
        snap = local(comments=(review(0),), threads=(thread(resolved=True, resolved_by=AUTHOR),))
        assert prf.classify(snap, NOW).state == "success"


class TestTheLedger:
    """What the process looked like, in one place, without reading the timeline."""

    def test_it_names_the_round_the_count_and_the_answer(self):
        snap = snapshot(
            comments=(
                review(3, critical=1, minutes_ago=90, ident=1),
                review(0, minutes_ago=10, ident=2),
                account(fixed=2, answered=1),
            )
        )
        lines = prf.review_ledger(snap)
        assert lines[0] == f"- Round 1 — 3 findings, 1 critical → 2 fixed, 1 answered by @{AUTHOR}"
        assert lines[1] == "- A later pass reported no findings"

    def test_an_unanswered_round_says_so(self):
        snap = snapshot(comments=(review(3, minutes_ago=90, ident=1), review(0, minutes_ago=10, ident=2)))
        assert "nothing written back yet" in prf.review_ledger(snap)[0]

    def test_the_round_in_progress_is_not_called_unanswered(self):
        """Nothing has settled on the newest verdict, so there is nothing to
        account for — it is open, which is a different state from closed quietly."""
        assert prf.review_ledger(snapshot(comments=(review(2),)))[0].endswith("still open")

    def test_a_dismissal_closes_the_newest_round_in_the_ledger(self):
        snap = snapshot(comments=(review(2, minutes_ago=60), ack(minutes_ago=30)))
        assert prf.review_ledger(snap)[0].endswith(f"all answered by @{MAINTAINER}")

    def test_it_reaches_the_sticky_comment(self):
        snap = snapshot(comments=(review(2, minutes_ago=60),))
        body = prf.sticky_body(snap, prf.classify(snap, NOW))
        assert "**Review ledger**" in body
        assert "- Round 1 — 2 findings" in body

    def test_the_ledger_cannot_clear_the_gate_it_describes(self):
        """The sticky is a comment on the PR, so the next run reads it back."""
        snap = snapshot(
            comments=(review(3, minutes_ago=90, ident=1), review(0, minutes_ago=10, ident=2), account(fixed=3))
        )
        body = prf.sticky_body(snap, prf.classify(snap, NOW))
        assert prf.acknowledged_producers(body) == set()
        assert prf.responses(body) == {}
        for producer in prf.PRODUCERS:
            assert producer.pattern.search(body) is None

    def test_no_review_no_ledger(self):
        assert prf.review_ledger(snapshot()) == []


class TestWaitingVersusMissing:
    def test_no_ci_run_yet_is_pending(self):
        snap = snapshot(ci=prf.CIState(None, None))
        assert prf.classify(snap, NOW).state == "pending"

    def test_red_ci_is_pending_and_says_so(self):
        snap = snapshot(ci=prf.CIState("failure", NOW - timedelta(hours=2)))
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "pending"
        assert "CI is failure" in verdict.description

    def test_inside_the_grace_window_is_pending(self):
        snap = snapshot(ci=prf.CIState("success", NOW - timedelta(minutes=5)))
        assert prf.classify(snap, NOW).state == "pending"

    def test_past_the_grace_window_a_missing_review_is_a_failure(self):
        # claude-review.yml has silently stopped firing twice in this repo. That
        # is the failure this state exists to make loud.
        snap = snapshot(ci=prf.CIState("success", NOW - timedelta(minutes=45)))
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "failure"
        assert "never posted a verdict" in verdict.items[0].detail

    def test_an_open_thread_beats_waiting_for_the_review(self):
        snap = snapshot(ci=prf.CIState("success", NOW - timedelta(minutes=1)), threads=(thread(),))
        assert prf.classify(snap, NOW).state == "failure"


class TestHumanThreads:
    def test_an_unresolved_thread_blocks(self):
        verdict = prf.classify(snapshot(comments=(review(0),), threads=(thread(),)), NOW)
        assert verdict.state == "failure"
        assert "collector.py:88" in verdict.items[0].detail

    def test_a_resolved_thread_does_not(self):
        snap = snapshot(comments=(review(0),), threads=(thread(resolved=True),))
        assert prf.classify(snap, NOW).state == "success"

    def test_the_author_talking_to_themselves_does_not(self):
        snap = snapshot(comments=(review(0),), threads=(thread(authors=(AUTHOR, AUTHOR)),))
        assert prf.classify(snap, NOW).state == "success"

    def test_an_outdated_thread_still_blocks(self):
        # The line moved; the point was not taken. Only Resolve closes one.
        snap = snapshot(comments=(review(0),), threads=(thread(outdated=True),))
        assert prf.classify(snap, NOW).state == "failure"

    def test_changes_requested_blocks(self):
        snap = snapshot(comments=(review(0),), review_decision="CHANGES_REQUESTED")
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "failure"
        assert verdict.items[0].kind == "changes-requested"

    def test_truncated_thread_pages_are_reported_not_swallowed(self):
        snap = snapshot(comments=(review(0),), threads_truncated=True)
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "failure"
        assert verdict.items[0].kind == "truncated"


class TestNotApplicable:
    def test_a_draft_passes(self):
        # Mirrors claude-review.yml's own filter: it never reviews a draft, so a
        # gate that waited for one would wedge the PR with nothing to fix.
        snap = snapshot(is_draft=True, ci=prf.CIState("success", NOW - timedelta(hours=3)))
        assert prf.classify(snap, NOW).state == "success"

    def test_a_dependabot_pr_passes(self):
        snap = snapshot(
            author="dependabot[bot]",
            head_ref="dependabot/pip/rich-14",
            ci=prf.CIState("success", NOW - timedelta(hours=3)),
        )
        assert prf.classify(snap, NOW).state == "success"

    def test_a_bot_pr_labelled_cowork_is_still_gated(self):
        # The claude.yml implement job opens PRs unattended as a bot. Those are
        # exactly the PRs nobody watched being written.
        snap = snapshot(
            author="github-actions[bot]",
            labels=("cowork",),
            ci=prf.CIState("success", NOW - timedelta(hours=3)),
        )
        assert prf.classify(snap, NOW).state == "failure"

    def test_the_override_label_clears_everything(self):
        snap = snapshot(comments=(review(5),), threads=(thread(),), labels=("feedback-override",))
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "success"
        assert "feedback-override" in verdict.description

    def test_the_override_is_reported_in_the_sticky_comment(self):
        snap = snapshot(comments=(review(5),), labels=("feedback-override",))
        body = prf.sticky_body(snap, prf.classify(snap, NOW))
        assert "feedback-override" in body


class TestDescription:
    def test_it_fits_githubs_limit(self):
        items = tuple(prf.OpenItem("thread", f"T{i}", "x" * 200) for i in range(5))
        assert len(prf.describe(items)) <= prf.DESCRIPTION_LIMIT

    def test_it_counts_then_names(self):
        snap = snapshot(comments=(review(0),), threads=(thread(ident="T1"), thread(ident="T2")))
        verdict = prf.classify(snap, NOW)
        assert verdict.description.startswith("2 unanswered review items")
        assert "+1 more" in verdict.description


class TestMarkers:
    @pytest.mark.parametrize(
        "body,expected",
        [
            ("<!-- addressed: claude-review -->", {"claude-review"}),
            ("<!--addressed:claude-review-->", {"claude-review"}),
            ("<!-- Addressed: Claude-Review -->", {"claude-review"}),
            ("text\n<!-- addressed: cowork-dod -->\nmore", {"cowork-dod"}),
            ("nothing here", set()),
            # Counted shapes. `answered=` dismisses; `fixed=` alone does not —
            # it records work, and work is checked by the reviewer's next read.
            ("<!-- addressed: claude-review fixed=2 answered=1 -->", {"claude-review"}),
            ("<!-- addressed: claude-review answered=1 -->", {"claude-review"}),
            ("<!-- addressed: claude-review fixed=2 -->", set()),
            ("<!-- addressed: claude-review fixed=0 answered=0 -->", set()),
            # A typo in a field is not a bare marker. Under-claiming is caught;
            # silently dismissing everything would not be.
            ("<!-- addressed: claude-review fixt=2 -->", set()),
        ],
    )
    def test_acknowledgement_parsing(self, body, expected):
        assert prf.acknowledged_producers(body) == expected

    @pytest.mark.parametrize(
        "body,fixed,answered,bare",
        [
            ("<!-- addressed: claude-review -->", 0, 0, True),
            ("<!-- addressed: claude-review fixed=2 answered=1 -->", 2, 1, False),
            ("<!-- addressed: claude-review answered=1 fixed=2 -->", 2, 1, False),
            ("<!-- addressed: claude-review fixed=9 -->", 9, 0, False),
        ],
    )
    def test_response_parsing(self, body, fixed, answered, bare):
        response = prf.responses(body)["claude-review"]
        assert (response.fixed, response.answered, response.bare) == (fixed, answered, bare)

    def test_a_bare_marker_covers_any_count(self):
        """It has always meant "all of them", and every reply already sitting on
        an open PR is this shape."""
        assert prf.responses("<!-- addressed: claude-review -->")["claude-review"].covers(99) is True

    def test_a_counted_marker_covers_only_what_it_claims(self):
        response = prf.responses("<!-- addressed: claude-review fixed=2 answered=1 -->")["claude-review"]
        assert response.covers(3) is True
        assert response.covers(4) is False

    def test_repeated_markers_take_the_larger_claim(self):
        """Two markers in one comment is somebody restating, not somebody doing
        the work twice — so the fields are maxed rather than summed."""
        body = "<!-- addressed: claude-review fixed=2 -->\n<!-- addressed: claude-review fixed=1 answered=1 -->"
        response = prf.responses(body)["claude-review"]
        assert (response.fixed, response.answered) == (2, 1)

    def test_timestamps_are_normalised_to_utc(self):
        assert prf.parse_timestamp("2026-08-06T12:00:00Z") == NOW
        assert prf.parse_timestamp("2026-08-06T12:00:00+00:00") == NOW
        assert prf.parse_timestamp(None) is None
        assert prf.parse_timestamp("not a date") is None


class TestStickyComment:
    def test_a_failing_verdict_lists_every_item_and_the_way_out(self):
        snap = snapshot(comments=(review(2),), threads=(thread(),))
        body = prf.sticky_body(snap, prf.classify(snap, NOW))
        assert prf.STICKY_MARKER in body
        assert "/pr-feedback 123" in body
        assert "collector.py:88" in body
        assert "<!-- addressed:" in body

    @pytest.mark.parametrize(
        "threads,expected",
        [((), "until it is answered"), ((thread(ident="T2"),), "until they are answered")],
    )
    def test_it_agrees_with_itself_on_number(self, threads, expected):
        snap = snapshot(comments=(review(2),), threads=threads)
        assert expected in prf.sticky_body(snap, prf.classify(snap, NOW))

    def test_a_clear_verdict_says_so_in_one_line(self):
        snap = snapshot(comments=(review(0),))
        body = prf.sticky_body(snap, prf.classify(snap, NOW))
        assert "clear" in body.lower()


class TestEditedComments:
    """A verdict is dated by when its text was last written, not when it appeared.

    ``pr-opened-dod-audit.md`` step 5 tells the DoD audit to **edit one comment in
    place** on every push rather than pile up a new one, and an edit does not move
    ``created_at``. Dating a verdict by creation therefore let a reply written the
    hour the PR opened go on clearing findings that were rewritten into that same
    comment minutes ago — a silent false green in the one producer designed to
    behave this way.
    """

    def test_an_edited_in_verdict_outranks_an_older_acknowledgement(self):
        snap = snapshot(
            comments=(
                comment(DOD_MARK.format(n=3), minutes_ago=600, edited_minutes_ago=5, ident=1),
                ack("cowork-dod", minutes_ago=300, ident=2),
                review(0, minutes_ago=60, ident=3),
            )
        )
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "failure"
        assert verdict.items[0].key == "cowork-dod"

    def test_an_acknowledgement_after_the_edit_clears_it(self):
        snap = snapshot(
            comments=(
                comment(DOD_MARK.format(n=3), minutes_ago=600, edited_minutes_ago=5, ident=1),
                ack("cowork-dod", minutes_ago=1, ident=2),
                review(0, minutes_ago=60, ident=3),
            )
        )
        assert prf.classify(snap, NOW).state == "success"

    def test_the_edited_comment_is_the_latest_verdict_even_when_created_first(self):
        old_but_edited = comment(
            REVIEW_MARK.format(n=4), minutes_ago=600, edited_minutes_ago=1, ident=1, author=REVIEWER
        )
        newer = review(0, minutes_ago=60, ident=2)
        found = prf.latest_verdict((old_but_edited, newer), prf.PRODUCERS[0])
        assert found[1] == 4

    def test_editing_an_old_reply_cannot_forward_date_it_over_a_finding(self):
        """The mirror rule, and the fail-closed half of the same fact.

        An acknowledgement is dated by ``created_at`` on purpose: if an edit moved
        it, fixing a typo in last week's reply would silently answer this
        morning's findings.
        """
        snap = snapshot(
            comments=(
                ack(minutes_ago=300, ident=1, edited_minutes_ago=1),
                review(2, minutes_ago=60, ident=2),
            )
        )
        assert prf.classify(snap, NOW).state == "failure"


class TestWhoMayAnswer:
    """This repo is public, so anybody at all can comment on a pull request."""

    @pytest.mark.parametrize("association", ["OWNER", "MEMBER", "COLLABORATOR", "owner"])
    def test_write_access_clears_a_finding(self, association):
        snap = snapshot(comments=(review(2, minutes_ago=60), ack(minutes_ago=30, association=association)))
        assert prf.classify(snap, NOW).state == "success"

    @pytest.mark.parametrize("association", ["NONE", "CONTRIBUTOR", "FIRST_TIME_CONTRIBUTOR", "MANNEQUIN"])
    def test_a_drive_by_acknowledgement_does_not(self, association):
        """One stranger's comment must not be able to clear somebody else's gate."""
        snap = snapshot(comments=(review(2, minutes_ago=60), ack(minutes_ago=30, association=association)))
        assert prf.classify(snap, NOW).state == "failure"

    def test_an_acknowledgement_in_the_same_second_still_counts(self):
        """`latest_verdict` breaks second-granularity ties on id; so does this."""
        snap = snapshot(comments=(review(2, minutes_ago=30, ident=1), ack(minutes_ago=30, ident=2)))
        assert prf.classify(snap, NOW).state == "success"


class TestForgedVerdicts:
    """A verdict is stronger than an ack, and until this existed it had no checks.

    Blocking the `<!-- addressed: -->` marker closed the weaker hole. Anyone could
    still post `<!-- pr-feedback: claude-review open=0 -->`, which does not
    *answer* the findings — it replaces the count of them — and the gate went
    green. On a public repo that was any account at all; on a cowork PR it was the
    routine that wrote the code. Survivable while a human clicked merge, and the
    entire gate now that `sweep-procedure.md` arms `gh pr merge --auto`.
    """

    def test_a_stranger_cannot_post_a_verdict(self):
        snap = snapshot(comments=(review(3, minutes_ago=60), forged(0, minutes_ago=5)))
        assert prf.classify(snap, NOW).state == "failure"

    def test_write_access_alone_cannot_post_one_either(self):
        """This is the applicant's route: a cowork routine posts as the maintainer."""
        snap = snapshot(
            labels=("cowork",),
            author="cowork-bot",
            comments=(review(3, minutes_ago=60), forged(0, minutes_ago=5, author="cowork-bot", association="OWNER")),
        )
        assert prf.classify(snap, NOW).state == "failure"

    def test_the_real_reviewer_still_clears_it(self):
        """The honest path must survive: a re-review reports zero and the gate opens."""
        snap = snapshot(
            comments=(
                review(3, minutes_ago=60, ident=1),
                review(0, minutes_ago=5, ident=2),
                account(fixed=3, minutes_ago=2),
            )
        )
        assert prf.classify(snap, NOW).state == "success"

    def test_a_bot_association_is_not_required_to_be_trusted(self):
        """A bot commenting through GITHUB_TOKEN carries `author_association: NONE`.

        Requiring write access *as well* would reject every genuine Claude Review
        and wedge every PR on a review that could never qualify — the deadlock
        this module is built to avoid. Pinned so the rule is not "hardened" into
        one later.
        """
        assert prf.is_authentic_verdict(
            comment(REVIEW_MARK.format(n=0), author=REVIEWER, association="NONE"),
            prf.PRODUCERS[0],
        )

    def test_a_forged_verdict_cannot_win_the_newest_wins_race(self):
        """It is skipped outright, not merely outranked."""
        latest = prf.latest_verdict(
            (review(3, minutes_ago=60, ident=1), forged(0, minutes_ago=1, ident=2)),
            prf.PRODUCERS[0],
        )
        assert latest is not None and latest[1] == 3

    def test_a_rejected_verdict_reads_as_absent_not_as_zero(self):
        """The fail-closed direction: red saying the review never posted."""
        snap = snapshot(comments=(forged(0, minutes_ago=5),))
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "failure"
        assert "never posted a verdict" in verdict.description

    def test_only_the_reviewer_login_may_post_a_review_verdict(self):
        """ "Any bot" is not specific enough where the applicant is also a bot.

        `claude.yml`, `codeql-triage.yml` and `ci-sentinel.yml` all open their PRs
        from an Actions job, so on those lanes the author and the reviewer are
        indistinguishable by bot-ness, and a job on the applicant side could post
        the reviewer's `open=0`. Only comment ordering stopped it — a coincidence,
        not a check.
        """
        producer = prf.PRODUCERS[0]
        assert prf.is_authentic_verdict(comment(REVIEW_MARK.format(n=0), author=REVIEWER), producer)
        for impostor in ("github-actions[bot]", "dependabot[bot]", "some-app[bot]"):
            assert not prf.is_authentic_verdict(comment(REVIEW_MARK.format(n=0), author=impostor), producer)

    def test_another_bot_cannot_clear_the_gate_on_a_machine_pr(self):
        snap = snapshot(
            author="github-actions[bot]",
            head_ref="feature/issue-9-thing",
            comments=(
                review(2, minutes_ago=60, ident=1),
                comment(REVIEW_MARK.format(n=0), minutes_ago=5, ident=2, author="github-actions[bot]"),
            ),
        )
        assert prf.classify(snap, NOW).state == "failure"

    def test_the_dod_audit_accepts_a_maintainer_because_a_routine_writes_it(self):
        """Advisory and never required — and a stranger still cannot write it."""
        producer = next(p for p in prf.PRODUCERS if p.key == "cowork-dod")
        maintainer = comment(DOD_MARK.format(n=1), author=AUTHOR, association="OWNER")
        stranger = comment(DOD_MARK.format(n=1), author="a-stranger", association="NONE")
        assert prf.is_authentic_verdict(maintainer, producer)
        assert not prf.is_authentic_verdict(stranger, producer)


class TestUnattendedPRs:
    """A machine PR may fix a finding. It may not declare one answered.

    Once security, bug and chore fixes merge without a human clicking anything,
    `TRUSTED_ASSOCIATIONS` stops being a gate: a cowork routine posts under an
    account with write access, so the thing that wrote the change could also
    write `<!-- addressed: claude-review -->` under the review of it and merge.
    The way out is a push — a re-review then reports `open=0` on its own — which
    is why these tests check that the honest path still clears.
    """

    @pytest.mark.parametrize(
        "branch",
        ["cowork/security-pin-shas", "feature/issue-231-fix", "security/codeql-triage-2026-08", "ci-sentinel/red-main"],
    )
    def test_every_machine_branch_counts_as_unattended(self, branch):
        assert prf.is_unattended(snapshot(head_ref=branch)) is True

    @pytest.mark.parametrize("branch", ["", "feature/nice-thing", "main", "coworker/typo"])
    def test_an_ordinary_branch_does_not(self, branch):
        assert prf.is_unattended(snapshot(head_ref=branch)) is False

    def test_the_label_alone_is_enough(self):
        assert prf.is_unattended(snapshot(labels=("cowork",))) is True

    def test_the_author_cannot_answer_its_own_review(self):
        snap = snapshot(
            labels=("cowork",),
            comments=(review(2, minutes_ago=60), ack(minutes_ago=30, association="OWNER", author=AUTHOR)),
        )
        assert prf.classify(snap, NOW).state == "failure"

    def test_an_unlabelled_machine_branch_is_gated_the_same_way(self):
        """A run truncated between `git push` and `gh pr create --label` lands here."""
        snap = snapshot(
            head_ref="cowork/standup-confidence",
            comments=(review(2, minutes_ago=60), ack(minutes_ago=30, association="OWNER", author=AUTHOR)),
        )
        assert prf.classify(snap, NOW).state == "failure"

    def test_somebody_else_with_write_access_still_can(self):
        snap = snapshot(
            labels=("cowork",),
            comments=(
                review(2, minutes_ago=60),
                ack(minutes_ago=30, association="OWNER", author="a-different-human"),
            ),
        )
        assert prf.classify(snap, NOW).state == "success"

    def test_a_fix_still_clears_it(self):
        """The honest path: push, get re-reviewed, and say what you changed.

        The account is written by the PR's own author — the same account whose
        *dismissal* is refused two tests up. That asymmetry is the design: a
        claimed fix is checked by the reviewer's next read, and a claimed
        disagreement is checked by nothing.
        """
        snap = snapshot(
            labels=("cowork",),
            comments=(
                review(2, minutes_ago=60, ident=1),
                review(0, minutes_ago=10, ident=3),
                account(fixed=2, minutes_ago=5),
            ),
        )
        assert prf.classify(snap, NOW).state == "success"

    def test_a_bot_pr_on_a_machine_branch_is_not_waved_through(self):
        """Without this, the widest lane in the fleet has no gate at all."""
        snap = snapshot(
            author="github-actions[bot]",
            head_ref="cowork/platform-pin-actions",
            comments=(review(2),),
            ci=prf.CIState("success", NOW - timedelta(hours=3)),
        )
        assert prf.classify(snap, NOW).state == "failure"

    def test_a_human_answering_their_own_pr_is_untouched(self):
        """Nothing above applies to a person: one already read the review."""
        snap = snapshot(comments=(review(2, minutes_ago=60), ack(minutes_ago=30, association="OWNER")))
        assert prf.classify(snap, NOW).state == "success"


class TestReviewBodies:
    """A review's top-level body is not an issue comment and needs folding in."""

    def test_an_acknowledgement_typed_into_the_review_box_counts(self):
        snap = snapshot(comments=(review(2, minutes_ago=60), ack(minutes_ago=30, kind="review", ident=9)))
        assert prf.classify(snap, NOW).state == "success"

    def test_an_untrusted_review_body_still_does_not(self):
        snap = snapshot(
            comments=(review(2, minutes_ago=60), ack(minutes_ago=30, kind="review", ident=9, association="NONE"))
        )
        assert prf.classify(snap, NOW).state == "failure"


class TestTheOverrideIsNotAFreeLever:
    """`feedback-override` clears more than a marker ever could, so it needs the rule.

    It returns success before the producers, the threads and a
    CHANGES_REQUESTED review are even looked at. The sweeps hold bare `Bash` and
    already run `gh pr merge --auto` on their own PR, so `gh pr edit --add-label
    feedback-override` sits inside the same grant — meaning "a human's call" was a
    convention, and the lane that just lost the `<!-- addressed: -->` route had a
    bigger lever sitting beside it.
    """

    def _snap(self, **overrides):
        base = dict(labels=("cowork", prf.OVERRIDE_LABEL), comments=(review(3),))
        base.update(overrides)
        return snapshot(**base)

    def test_the_author_of_a_machine_pr_cannot_override_their_own(self):
        snap = self._snap(author="cowork-bot", override_actor="cowork-bot")
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "failure"
        assert "own author" in verdict.description

    def test_another_human_still_can(self):
        snap = self._snap(author="cowork-bot", override_actor="a-different-human")
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "success"
        assert "a-different-human" in verdict.description

    def test_an_unknown_actor_still_honours_it(self):
        """This label exists to unbrick a wedged gate.

        Refusing it on a timeline we could not read would turn one API failure
        into a PR nobody can merge — precisely what it is the escape hatch for.
        """
        snap = self._snap(author="cowork-bot", override_actor="")
        assert prf.classify(snap, NOW).state == "success"

    def test_an_ordinary_pr_is_untouched(self):
        """A person overriding their own PR is a person making a call."""
        snap = self._snap(
            labels=(prf.OVERRIDE_LABEL,), head_ref="feature/nice-thing", author=AUTHOR, override_actor=AUTHOR
        )
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "success"
        # And it is the *override* that cleared it, not the local lane below it —
        # otherwise this test would keep passing with the override rule deleted.
        assert prf.OVERRIDE_LABEL in verdict.description

    def test_the_actor_is_read_from_the_timeline_newest_wins(self):
        events = [
            {"event": "labeled", "label": {"name": prf.OVERRIDE_LABEL}, "actor": {"login": "first"}},
            {"event": "labeled", "label": {"name": "cowork"}, "actor": {"login": "noise"}},
            {"event": "labeled", "label": {"name": prf.OVERRIDE_LABEL}, "actor": {"login": "second"}},
        ]
        with_json = lambda *a: events  # noqa: E731 - a one-line stub
        original = prf._gh_json
        try:
            prf._gh_json = with_json
            assert prf.fetch_override_actor("o/r", 1) == "second"
        finally:
            prf._gh_json = original

    def test_an_unreadable_timeline_reads_as_unknown_not_as_the_author(self):
        original = prf._gh_json
        try:
            prf._gh_json = lambda *a: None
            assert prf.fetch_override_actor("o/r", 1) == ""
        finally:
            prf._gh_json = original


class TestStickyAdviceMatchesThePR:
    """The check must not instruct a human to do the one thing that cannot work.

    On a cowork PR the author is the maintainer's own account, so a person who
    reads the red check, disagrees with a finding and replies exactly as told gets
    silence and the same red check re-rendered, with no diagnostic anywhere.
    """

    def _body(self, **overrides):
        snap = snapshot(comments=(review(2),), **overrides)
        return prf.sticky_body(snap, prf.classify(snap, NOW))

    def test_the_reply_route_is_still_spelled_out_where_it_applies(self):
        """`classify` no longer produces a red verdict on a local branch, so this
        exercises `sticky_body` directly. The copy stays because the rule it
        describes is about *who may answer*, not about which lane blocks — flip
        the lane policy back and this is the comment that has to be right."""
        body = prf.sticky_body(
            local(),
            prf.Verdict("failure", "1 unanswered review item", (prf.OpenItem("producer", "claude-review", "2 x"),)),
        )
        assert "<!-- addressed: <producer> -->" in body

    def test_a_machine_pr_says_the_reply_route_does_not_apply(self):
        body = self._body(labels=("cowork",))
        assert "cannot *dismiss* a finding here" in body
        assert "<!-- addressed: <producer> -->" not in body

    def test_a_machine_pr_still_names_a_way_out(self):
        """Never a dead end: fixing, or the override, and the override is a human's."""
        body = self._body(head_ref="cowork/platform-x")
        assert "fix the finding and push" in body
        assert prf.OVERRIDE_LABEL in body

    def test_a_machine_pr_is_told_that_pushing_is_not_the_whole_job(self):
        """The advice used to end at "fix and push — this clears itself", which is
        the exact behaviour `unaccounted_rounds` stopped being true."""
        body = self._body(head_ref="cowork/platform-x")
        assert "then reply saying what you changed" in body
        assert "fixed=N" in body


class TestStickyIsInert:
    def test_a_pending_verdict_is_not_called_clear(self):
        """Pending holds the merge exactly as firmly as a failure does."""
        body = prf.sticky_body(snapshot(), prf.Verdict("pending", "waiting for Claude Review"))
        assert "waiting" in body.lower()
        assert "clear" not in body.lower()

    @pytest.mark.parametrize(
        "verdict",
        [
            prf.Verdict("success", "no open review feedback"),
            prf.Verdict("pending", "waiting for Claude Review"),
            prf.Verdict(
                "failure", "1 unanswered review item", (prf.OpenItem("producer", "claude-review", "2 findings"),)
            ),
        ],
    )
    def test_the_gate_cannot_clear_itself_through_its_own_comment(self, verdict):
        """The sticky is a comment on the PR, so the next run reads it back.

        Nothing in it may parse as a verdict or as an acknowledgement — a wording
        change that made it do so would make every red gate go green one run later.
        """
        body = prf.sticky_body(snapshot(), verdict)
        assert prf.acknowledged_producers(body) == set()
        for producer in prf.PRODUCERS:
            assert producer.pattern.search(body) is None


class FakeGh:
    """Record every ``gh`` invocation; reply from a per-test script.

    ``_gh`` is the single seam every GitHub call goes through, so the whole
    fetching and posting half is reachable with one monkeypatch and none of it
    touches the network. Replies are matched on a substring of the joined args,
    because these calls are URLs rather than the two-word subcommands
    ``test_cowork_setup.py`` keys on.
    """

    def __init__(self):
        self.calls: list[tuple[str, ...]] = []
        self.replies: list[tuple[str, int, str]] = []

    def reply(self, needle, payload, code=0):
        self.replies.append((needle, code, payload if isinstance(payload, str) else json.dumps(payload)))

    def sent(self, needle):
        return [args for args in self.calls if needle in " ".join(args)]

    def __call__(self, *args):
        self.calls.append(args)
        joined = " ".join(args)
        for needle, code, out in self.replies:
            if needle in joined:
                return subprocess.CompletedProcess(args, code, out, "boom" if code else "")
        return subprocess.CompletedProcess(args, 0, "", "")


@pytest.fixture
def gh(monkeypatch):
    fake = FakeGh()
    monkeypatch.setattr(prf, "_gh", fake)
    # `transport.graphql` routes itself and calls the transport's own `gh`, so
    # patching only `prf._gh` leaves the GraphQL read going out over the network.
    # It did, briefly, and the test failure named a real repository.
    monkeypatch.setattr(prf.transport, "gh", fake)
    monkeypatch.setattr(prf.transport, "gh_available", lambda: True)
    return fake


@pytest.fixture(autouse=True)
def _no_live_github(monkeypatch):
    """No ambient token, no memoised repository, and no unstubbed REST call.

    ``_gh`` is no longer the only seam: the transport authenticates from the
    *environment*, so a developer with GH_TOKEN exported would have any gap in a
    stub go live against their own repo rather than fail.
    """
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    prf.transport.reset_slug_cache()

    def refuse(method, path, body=None):
        raise AssertionError(f"a test reached the REST transport unstubbed: {method} {path}")

    monkeypatch.setattr(prf.transport, "api", refuse)


def _seed_pr(gh, **over):
    gh.reply("issues/123/comments", [])
    gh.reply("pulls/123/reviews", [])
    gh.reply("graphql", {"data": {"repository": {"pullRequest": _pull(**over)}}})
    gh.reply("actions/runs", {"workflow_runs": []})


def _pull(threads=None, has_next=False, **over):
    """The `pullRequest` node `PR_QUERY` returns — metadata and threads together.

    They used to come from two calls, `gh pr view` and a GraphQL query. One call
    now, because `reviewDecision` has no REST equivalent and asking for it here
    is what lets both transports answer the same question.
    """
    return {
        "number": 123,
        "headRefOid": HEAD,
        "headRefName": "feature/x",
        "isDraft": False,
        "reviewDecision": None,
        "author": {"login": AUTHOR},
        "labels": {"nodes": [{"name": "cowork"}]},
        "reviewThreads": {"pageInfo": {"hasNextPage": has_next}, "nodes": threads or []},
        **over,
    }


class TestFetch:
    def test_a_pr_is_read_into_the_shape_classify_wants(self, gh):
        _seed_pr(gh, reviewDecision="CHANGES_REQUESTED")
        gh.replies.insert(
            0,
            (
                "issues/123/comments",
                0,
                json.dumps(
                    [
                        {
                            "id": 7,
                            "user": {"login": "claude[bot]"},
                            "body": REVIEW_MARK.format(n=1),
                            "created_at": "2026-08-06T10:00:00Z",
                            "updated_at": "2026-08-06T11:00:00Z",
                            "author_association": "NONE",
                        }
                    ]
                ),
            ),
        )
        snap = prf.fetch_snapshot(123, "o/r")
        assert snap.head_sha == HEAD
        assert snap.author == AUTHOR
        assert snap.labels == ("cowork",)
        assert snap.review_decision == "CHANGES_REQUESTED"
        assert snap.comments[0].written_at > snap.comments[0].created_at

    def test_a_failed_read_is_none_rather_than_a_half_snapshot(self, gh):
        """The metadata read is the GraphQL one now. A half snapshot would be
        read as a PR with no threads and no labels — a clean one."""
        gh.reply("graphql", "", code=1)
        assert prf.fetch_snapshot(123, "o/r") is None

    def test_threads_are_normalised_and_truncation_is_carried(self, gh):
        _seed_pr(gh)
        gh.replies.insert(
            0,
            (
                "graphql",
                0,
                json.dumps(
                    {
                        "data": {
                            "repository": {
                                "pullRequest": {
                                    "reviewThreads": {
                                        "pageInfo": {"hasNextPage": True},
                                        "nodes": [
                                            {
                                                "id": "T1",
                                                "isResolved": False,
                                                "isOutdated": True,
                                                "path": "a.py",
                                                "line": 3,
                                                "comments": {
                                                    "nodes": [{"author": {"login": "r"}, "body": "  spaced\n out "}]
                                                },
                                            }
                                        ],
                                    }
                                }
                            }
                        }
                    }
                ),
            ),
        )
        snap = prf.fetch_snapshot(123, "o/r")
        assert snap.threads_truncated is True
        assert snap.threads[0].authors == ("r",)
        assert snap.threads[0].excerpt == "spaced out"

    def test_review_bodies_are_folded_in_and_empty_ones_dropped(self, gh):
        _seed_pr(gh)
        gh.replies.insert(
            0,
            (
                "pulls/123/reviews",
                0,
                json.dumps(
                    [
                        {"id": 1, "user": {"login": "r"}, "body": "", "submitted_at": "2026-08-06T10:00:00Z"},
                        {
                            "id": 2,
                            "user": {"login": AUTHOR},
                            "body": "<!-- addressed: claude-review -->",
                            "submitted_at": "2026-08-06T11:00:00Z",
                            "author_association": "OWNER",
                        },
                    ]
                ),
            ),
        )
        bodies = [c for c in prf.fetch_snapshot(123, "o/r").comments if c.kind == "review"]
        assert len(bodies) == 1
        assert bodies[0].association == "OWNER"

    def test_ci_takes_the_latest_run_for_this_sha_and_ignores_other_workflows(self, gh):
        gh.reply(
            "actions/runs",
            {
                "workflow_runs": [
                    {
                        "name": "CI",
                        "conclusion": "failure",
                        "created_at": "2026-08-06T09:00:00Z",
                        "updated_at": "2026-08-06T09:10:00Z",
                    },
                    {
                        "name": "CI",
                        "conclusion": "success",
                        "created_at": "2026-08-06T10:00:00Z",
                        "updated_at": "2026-08-06T10:10:00Z",
                    },
                    {
                        "name": "CodeQL",
                        "conclusion": "failure",
                        "created_at": "2026-08-06T11:00:00Z",
                        "updated_at": "2026-08-06T11:10:00Z",
                    },
                ]
            },
        )
        state = prf.fetch_ci("o/r", HEAD)
        assert state.conclusion == "success"
        assert state.completed_at == datetime(2026, 8, 6, 10, 10, tzinfo=timezone.utc)

    def test_no_run_for_this_sha_is_not_a_green_one(self, gh):
        gh.reply("actions/runs", {"workflow_runs": []})
        assert prf.fetch_ci("o/r", HEAD) == prf.CIState(None, None)
        assert prf.fetch_ci("o/r", "") == prf.CIState(None, None)

    def test_the_repo_and_pr_lookups_survive_a_gh_failure(self, gh, monkeypatch):
        """A rejected lookup is None, not a guess.

        `repo_slug` falls through `gh` to the environment to the git remote, so
        failing only the `gh` call would still find this checkout's own repo —
        the autouse guard clears GITHUB_REPOSITORY and the remote is stubbed
        away here, leaving nothing to answer with.
        """
        gh.reply("repo view", "", code=1)
        gh.reply("pr view", "", code=1)
        monkeypatch.setattr(
            prf.transport, "_run", lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "no remote")
        )
        assert prf.repo_slug() is None
        assert prf.current_pr() is None


class TestPosting:
    def test_the_status_carries_the_context_the_ruleset_waits_on(self, gh):
        prf.post_status("o/r", HEAD, prf.Verdict("failure", "x" * 300), "https://run")
        sent = " ".join(gh.sent(f"statuses/{HEAD}")[0])
        assert f"context={prf.STATUS_CONTEXT}" in sent
        assert "state=failure" in sent
        assert "target_url=https://run" in sent
        assert "x" * (prf.DESCRIPTION_LIMIT + 1) not in sent

    def test_a_failed_post_is_reported_rather_than_swallowed(self, gh, capsys):
        gh.reply("statuses", "", code=1)
        assert prf.post_status("o/r", HEAD, prf.Verdict("success", "fine")) is False
        assert f"POST repos/o/r/statuses/{HEAD} failed" in capsys.readouterr().err

    def test_a_clear_verdict_never_creates_a_comment(self, gh):
        """Only ever edited into an existing sticky. A PR with nothing to say
        about its review feedback should not acquire a comment saying so."""
        prf.upsert_sticky("o/r", 123, snapshot(), prf.Verdict("success", "no open review feedback"))
        assert gh.sent("issues/123/comments") == []

    def test_a_failing_verdict_creates_one(self, gh):
        prf.upsert_sticky("o/r", 123, snapshot(), prf.Verdict("failure", "1 unanswered review item"))
        assert len(gh.sent("issues/123/comments")) == 1

    def test_an_existing_sticky_is_edited_in_place(self, gh):
        snap = snapshot(comments=(comment(f"{prf.STICKY_MARKER}\nold", ident=55),))
        prf.upsert_sticky("o/r", 123, snap, prf.Verdict("success", "no open review feedback"))
        assert gh.sent("issues/comments/55")
        assert gh.sent("issues/123/comments") == []

    def test_a_review_body_is_never_mistaken_for_the_sticky_comment(self, gh):
        """Review ids and issue-comment ids come from different spaces, so
        PATCHing the issue-comments endpoint with a review id edits something
        unrelated."""
        snap = snapshot(comments=(comment(f"{prf.STICKY_MARKER}\nold", ident=55, kind="review"),))
        prf.upsert_sticky("o/r", 123, snap, prf.Verdict("failure", "1 unanswered review item"))
        assert gh.sent("issues/comments/55") == []
        assert len(gh.sent("issues/123/comments")) == 1

    def test_the_report_names_every_open_item(self):
        items = (prf.OpenItem("thread", "T1", "unresolved thread on a.py:3"),)
        out = prf.render_report(snapshot(), prf.Verdict("failure", "1 unanswered review item", items))
        assert "PR #123" in out and HEAD[:7] in out and "unresolved thread on a.py:3" in out


class TestMain:
    @pytest.fixture(autouse=True)
    def _gh_on_path(self, monkeypatch):
        """`gh` present and answering, which is what the `gh` fixture already
        fakes — pinned here too so `main`'s "no transport at all" guard is only
        exercised by the test that means to."""
        monkeypatch.setattr(prf.transport.shutil, "which", lambda _: "/usr/bin/gh")

    def _repo(self, gh):
        gh.reply("repo view", {"nameWithOwner": "o/r"})

    def test_open_feedback_exits_nonzero(self, gh, capsys):
        self._repo(gh)
        _seed_pr(gh)
        gh.replies.insert(0, ("actions/runs", 0, json.dumps({"workflow_runs": []})))
        assert prf.main(["--pr", "123"]) == 0  # pending, not a failure
        gh.replies.insert(
            0,
            (
                "issues/123/comments",
                0,
                json.dumps(
                    [{"id": 1, "user": {"login": REVIEWER}, "body": REVIEW_MARK.format(n=2), "created_at": None}]
                ),
            ),
        )
        assert prf.main(["--pr", "123"]) == 1
        assert "unanswered" in capsys.readouterr().out

    def test_json_is_machine_readable(self, gh, capsys):
        self._repo(gh)
        _seed_pr(gh)
        assert prf.main(["--pr", "123", "--json"]) in (0, 1)
        payload = json.loads(capsys.readouterr().out)
        assert payload["number"] == 123 and payload["head_sha"] == HEAD and "state" in payload

    def test_status_mode_succeeds_whatever_the_verdict(self, gh):
        """A red *step* would read as a broken workflow rather than as unanswered
        review, and would be the first thing anyone disabled. The status blocks."""
        self._repo(gh)
        _seed_pr(gh)
        assert prf.main(["--pr", "123", "--status"]) == 0
        assert gh.sent(f"statuses/{HEAD}")

    def test_a_required_check_never_stays_silent(self, gh):
        """A required status that was never posted is indistinguishable in the UI
        from this workflow not existing, and blocks the merge with nothing to do."""
        self._repo(gh)
        _seed_pr(gh)
        # The comments read fails, so the snapshot is None and the fallback path
        # re-reads the head SHA on its own — over REST's spelling, `head.sha`.
        gh.replies.insert(0, ("issues/123/comments", 1, ""))
        gh.reply("repos/o/r/pulls/123", {"head": {"sha": HEAD}})
        assert prf.main(["--pr", "123", "--status"]) == 2
        posted = " ".join(gh.sent(f"statuses/{HEAD}")[0])
        # No `ref` in this payload, so the lane could not be determined either —
        # and "we could not tell" is not "it is yours". Pending, fail-closed.
        assert "state=pending" in posted

    def test_an_unreadable_local_pr_is_not_left_pending(self, gh):
        """The same failure, on a branch this gate does not enforce on.

        Pending blocks a *required* check exactly as hard as red, so one `gh`
        hiccup would wedge a person's own PR behind a gate that is supposed to be
        advisory for them — with no finding to fix and no comment saying why. The
        lane survives a failed snapshot read because it needs only the head ref,
        which the one-shot `pulls/{n}` fallback already fetches.
        """
        self._repo(gh)
        _seed_pr(gh)
        gh.replies.insert(0, ("issues/123/comments", 1, ""))
        gh.reply("repos/o/r/pulls/123", {"head": {"sha": HEAD, "ref": "feature/nice-thing"}, "labels": []})
        assert prf.main(["--pr", "123", "--status"]) == 2
        posted = " ".join(gh.sent(f"statuses/{HEAD}")[0])
        assert "state=success" in posted

    def test_an_unreadable_machine_pr_is_still_held(self, gh):
        """The mirror: the enforced lane must not fall open on a failed read."""
        self._repo(gh)
        _seed_pr(gh)
        gh.replies.insert(0, ("issues/123/comments", 1, ""))
        gh.reply("repos/o/r/pulls/123", {"head": {"sha": HEAD, "ref": "cowork/standup-x"}, "labels": []})
        assert prf.main(["--pr", "123", "--status"]) == 2
        posted = " ".join(gh.sent(f"statuses/{HEAD}")[0])
        assert "state=pending" in posted

    def test_no_transport_at_all_says_how_to_get_one(self, monkeypatch, capsys):
        """No `gh` *and* no token. Either alone is fine now, and saying so is the
        point: the routine session that runs this has only the second, and the
        old message sent it to `brew install gh`."""
        monkeypatch.setattr(prf.transport.shutil, "which", lambda _: None)
        assert prf.main(["--pr", "1"]) == 2
        err = capsys.readouterr().err
        assert "brew install gh" in err and "GH_TOKEN" in err

    def test_a_token_alone_is_enough_to_start(self, monkeypatch, gh, capsys):
        """The production shape: a token, no CLI. It must get past the guard —
        it used to exit 2 here and post nothing, which reads as a gate that was
        never asked to run."""
        monkeypatch.setattr(prf.transport.shutil, "which", lambda _: None)
        monkeypatch.setenv("GH_TOKEN", "t")
        monkeypatch.setattr(prf.transport, "gh_available", lambda: False)
        monkeypatch.setattr(prf, "repo_slug", lambda: None)
        assert prf.main(["--pr", "1"]) == 2
        err = capsys.readouterr().err
        assert "brew install gh" not in err
        assert "could not resolve the repo" in err

    def test_no_pr_and_none_on_this_branch_is_an_error_not_a_pass(self, gh, capsys):
        self._repo(gh)
        gh.reply("pr view", "", code=1)
        assert prf.main([]) == 2
        assert "no PR given" in capsys.readouterr().err


class TestAnUnreadablePRIsNotACleanOne:
    """The failure this whole gate exists to prevent, in its newest form.

    A routine session's GitHub egress refuses GraphQL (403, recorded in
    `tests/fixtures/cowork_github_access_live.json`), and review threads plus
    `reviewDecision` exist in v4 and nowhere in v3. So there the gate reads
    *nothing* — and the one thing it must never do is let that look like a PR
    with nothing to answer.
    """

    def test_the_proxy_refusal_is_named_with_its_remedy(self):
        prf.LAST_FAILURE = (
            "graphql: HTTP 403 on POST /graphql: This GraphQL query is not enabled for this session "
            "— only the pinned set of PR-review operations is served."
        )
        reason = prf.unreadable_reason()
        assert "NOTHING was determined" in reason
        assert "Do not read this as a clean PR" in reason
        assert "pr-feedback.yml" in reason

    def test_an_ordinary_failure_is_not_dressed_up_as_the_proxy(self):
        """A 403 from a token-scope problem is a different fault with a different
        remedy, so the proxy wording must not be reached for every failure."""
        prf.LAST_FAILURE = "graphql: HTTP 401 on POST /graphql: Bad credentials"
        reason = prf.unreadable_reason()
        assert "Bad credentials" in reason
        assert "NOTHING was determined" not in reason

    def test_an_unreadable_pr_says_so_on_stderr_and_exits_non_zero(self, monkeypatch, capsys):
        monkeypatch.setattr(prf.transport, "gh_available", lambda: False)
        monkeypatch.setenv("GH_TOKEN", "t")
        monkeypatch.setattr(prf, "repo_slug", lambda: "owner/name")
        monkeypatch.setattr(prf, "fetch_snapshot", lambda number, slug: None)
        prf.LAST_FAILURE = "graphql: this GraphQL query is not enabled for this session"

        assert prf.main(["--pr", "7"]) == 2

        err = capsys.readouterr().err
        assert "could not read PR #7" in err
        assert "Do not read this as a clean PR" in err


class TestTheLocalLaneStillCarriesAPerson:
    """The advisory lane withdraws the *machine* reviewer, and only it.

    "Nobody on the other end to weigh a finding" is the whole argument for going
    advisory, and it is false by construction when the finding is a human's. The
    first cut of the lane short-circuited on the whole item list, which took an
    unresolved review thread and a `CHANGES_REQUESTED` review with it — on the
    one lane where the reviewer is a real person waiting for an answer, and
    against `definition-of-done.md`'s promise in as many words that those are
    never capped.
    """

    def test_an_unresolved_human_thread_blocks_a_local_branch(self):
        snap = snapshot(head_ref="feature/some-work", threads=(thread(),))
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "failure"
        assert verdict.reason != "advisory"

    def test_a_changes_requested_review_blocks_a_local_branch(self):
        snap = snapshot(head_ref="feature/some-work", review_decision="CHANGES_REQUESTED")
        assert prf.classify(snap, NOW).state == "failure"

    def test_review_findings_alone_stay_advisory(self):
        snap = snapshot(head_ref="feature/some-work", comments=(review(3),))
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "success"
        assert verdict.reason == "advisory"

    def test_a_human_thread_wins_over_advisory_findings(self):
        """Both present: the blocking half decides, and only it is listed."""
        snap = snapshot(head_ref="feature/some-work", comments=(review(3),), threads=(thread(),))
        verdict = prf.classify(snap, NOW)
        assert verdict.state == "failure"
        assert all(item.kind != "producer" for item in verdict.items)

    def test_it_still_cannot_be_pending_on_a_local_branch(self):
        """The load-bearing property survives the change: a local branch may be
        red for a person, never `pending` for a machine that did not speak."""
        snap = snapshot(head_ref="feature/some-work", comments=(), threads=(thread(),))
        assert prf.classify(snap, NOW).state != "pending"


class TestADismissalIsNeverAnAccount:
    """`answered=N` from the PR's own author is a bare marker with a number.

    The bare shape was refused and the counted one was not, though both claim no
    work at all — nothing for the reviewer's next read of the diff to check. On
    the unattended lane, with the review capped at two rounds, that let an agent
    close out every superseded round without ever claiming a fix.
    """

    def _superseded(self):
        return (review(3, minutes_ago=90, ident=1), review(0, minutes_ago=10, ident=2))

    def test_answered_only_from_the_author_does_not_account(self):
        reply = account(fixed=0, answered=3, author=AUTHOR)
        assert prf.classify(snapshot(comments=(*self._superseded(), reply)), NOW).state == "failure"

    def test_one_claimed_fix_makes_the_whole_reply_admissible(self):
        reply = account(fixed=1, answered=2, author=AUTHOR)
        assert prf.classify(snapshot(comments=(*self._superseded(), reply)), NOW).state == "success"

    def test_a_maintainer_may_still_answer_the_whole_round(self):
        """The refusal is about the applicant holding the key, not about the shape."""
        reply = account(fixed=0, answered=3, author=MAINTAINER)
        assert prf.classify(snapshot(comments=(*self._superseded(), reply)), NOW).state == "success"


class TestNativeCodexAdvisory:
    @pytest.mark.parametrize("login", ["chatgpt-codex-connector", "chatgpt-codex-connector[bot]"])
    def test_codex_thread_is_visible_without_blocking_or_requiring_completion(self, login):
        state = snapshot(comments=(review(0),))
        thread = prf._thread(
            {
                "id": "codex-1",
                "isResolved": False,
                "isOutdated": False,
                "path": "code.py",
                "line": 3,
                "comments": {"nodes": [{"author": {"login": login}, "body": "[P1] defect"}]},
            }
        )
        state = prf.replace(state, threads=(thread,))
        verdict = prf.classify(state, NOW)
        assert verdict.state == "success"
        assert not verdict.items
        assert len(verdict.advisory_items) == 1
        assert "Codex (advisory)" in prf.render_report(state, verdict)
        assert "Codex review — advisory" in prf.sticky_body(state, verdict)
        assert verdict.as_dict()["advisory_items"][0]["key"] == "codex-1"

    @pytest.mark.parametrize("login", ["chatgpt-codex-connector", "chatgpt-codex-connector[bot]"])
    def test_human_participation_in_codex_thread_still_blocks(self, login):
        state = snapshot(comments=(review(0),))
        thread = prf.Thread("codex-1", False, False, "code.py", 3, (login, MAINTAINER), "defect")
        verdict = prf.classify(prf.replace(state, threads=(thread,)), NOW)
        assert verdict.state == "failure"
        assert not verdict.advisory_items

    def test_similarly_named_reviewer_still_blocks(self):
        state = snapshot(comments=(review(0),))
        thread = prf.Thread("human-1", False, False, "code.py", 3, ("chatgpt-codex-connector-human",), "defect")
        verdict = prf.classify(prf.replace(state, threads=(thread,)), NOW)
        assert verdict.state == "failure"
        assert not verdict.advisory_items

    def test_codex_clean_review_cannot_satisfy_required_claude_verdict(self):
        state = snapshot(comments=(comment("Looks good", author="chatgpt-codex-connector[bot]"),))
        assert prf.classify(state, NOW).state != "success"
