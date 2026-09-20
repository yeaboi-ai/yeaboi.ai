---
name: pr-feedback
description: Answer the unresolved review feedback on a PR and clear the pr-feedback gate
---

Work through the review feedback on a pull request until nothing is unanswered.
Arguments (optional): the user’s arguments — a PR number; defaults to the current branch's PR.

Five things comment on a PR here and, until the `pr-feedback` gate existed, nothing
ever read any of it back. This is the procedure that reads it back. It is the one
place that job is written down; `/ship` and `/babysit-prs`
all point here rather than restating it.

1. **Look** — `make pr-feedback PR=<n>` (or `uv run python scripts/pr_feedback.py --pr <n> --json`
   when you want to iterate). It prints every open item: unanswered findings per
   producer, unresolved human threads, and a requested-changes review.

   Three states are not failures and need no work: `pending` means the review has
   not run yet (wait for CI); a draft or a Dependabot PR is out of scope by design;
   and "Claude Review never posted a verdict" means the *workflow* did not run —
   check `gh run list --workflow "Claude Review"` before touching the code, because
   nothing in the diff will fix it.

2. **Read what was actually said** — `gh pr view <n> --comments`, and
   `gh api repos/{owner}/{repo}/pulls/<n>/comments --paginate` for inline review comments. The verdict tells
   you how many findings are open; only the comments tell you what they are. Never
   answer a finding you have not read.

3. **Decide fix or answer, per item.** Fix when the finding is right. Answer when it
   is wrong, already handled elsewhere, or genuinely out of scope for this PR. Both
   are legitimate. Fixing everything to turn a check green is the worse failure: it
   lands unrequested changes on a branch someone already reviewed, and it teaches
   the reviewer that findings are cheap.

4. **Fixes** — minimal change, `make test` + `make lint`, commit with the repo's
   conventions and push to the same branch.

5. **Reply — one comment per producer, covering every finding, whichever way it
   went.** A line each, saying what you did and why, ending with that producer's
   marker on its own line:

   ```
   <!-- addressed: claude-review fixed=2 answered=1 -->
   ```

   `fixed=` counts the ones you changed; `answered=` counts the ones you argued.
   Two numbers rather than one because they are checked by different things: the
   reviewer's next pass reads the diff and re-reports a fix that is not there,
   and nothing but a person checks a disagreement. A bare `<!-- addressed:
   claude-review -->` is the older shape and still means "all of them, answered"
   — except from the PR's own author on the unattended lane, where "answered" is
   the one thing that account may not say, so write the counts.

   **On an unattended PR a fix without this reply does not clear the gate.** The
   next verdict reporting `open=0` used to be the whole clearing mechanism, and it
   left the entire record of what a machine did about three findings as a number
   going down — which nobody can review. `scripts/pr_feedback.py` reports that
   round as `stopped being reported without anything saying what changed`, and the
   way through is one comment. It is never capped and never expires.

   **On a local branch none of it is required**, for the same reason nothing else
   on that lane is: you did the fixing and you are the one merging. That covers
   the *machine* reviewer only — a human's unresolved thread and a `Request
   changes` review hold the check on every lane.

   The marker only counts if the comment is **newer than the verdict it answers**, so
   post it after the review, not before. If you push again afterwards, the next review
   pass will have read your reply and should stop reporting the finding at all.

   **A machine may not dismiss its own review.** On an unattended PR, a
   `feature/issue-N-…` branch, or a triage or sentinel branch,
   `scripts/pr_feedback.py` discards an `answered=` claim from the PR's own author,
   so arguing a finding down there is inert and the gate stays red. `fixed=` from
   that same author is accepted, and the asymmetry is the point: a claimed fix is
   checked by the reviewer's next read, a claimed disagreement by nothing.
   Without it the applicant would be holding the key: the account that wrote the
   change also has write access. So on that lane the route through a finding you
   disagree with is to hand it back to a human — as a proposal, or via
   `feedback-override`, which is a human's call and is recorded on the PR either
   way. `AGENTS.md` prescribes `feature/<description>` for human branches, so if
   you happen to name one `feature/issue-…` you land in this rule too; rename the
   branch.

6. **Human threads** — reply in the thread, then resolve it:

   ```bash
   gh api graphql -f query='mutation($id:ID!){resolveReviewThread(input:{threadId:$id}){thread{isResolved}}}' -F id=<threadId>
   ```

   The thread ids are in the `--json` output. Reply first, always — on an
   unattended PR that order is enforced rather than trusted: a thread the PR's own
   author resolved with no comment from them in it comes back as an open item.
   Resolved threads still accept comments, so the fix is a reply, not an un-resolve.

7. **Confirm** — re-run step 1 and report the final verdict with the PR URL. The
   `--json` output carries a `ledger`: one line per review round, what it found,
   and who wrote back to it. That is the shortest honest answer to "was this
   actually worked through, or did the number just go down".

For several PRs at once, or to run this unattended, use `/babysit-prs fix` — it spawns
the `pr-responder` subagent (`.agents/reviewers/pr-responder.md`) per PR in its own worktree.

## Rules

- **Never resolve a thread you did not answer**, and never write an `<!-- addressed: -->`
  marker without the reply that justifies it. The marker means "there is a reason below".
  Closing feedback silently is the behaviour this gate exists to stop — and on the
  unattended lane both halves are now checked in code rather than trusted.
- **Never claim a `fixed=` you did not make.** It is the one field the PR's own
  author may write on the enforced lane, and it is allowed only because the next
  review pass verifies it. Inflating it does not buy a merge; it buys a round.
- **A machine never answers its own review.** On an unattended PR the only way to clear a
  finding is to fix it; disagreement is escalated to a human, never asserted. This one is
  enforced in code rather than trusted, because it is the single rule standing between an
  unattended merge and no review at all.
- **The `feedback-override` label is a last resort and a human's call** — for a gate that
  has genuinely gone wrong (a review that errored, a producer that changed format), not
  for feedback that is inconvenient. It is recorded in the sticky comment, so an
  overridden PR never looks like a clean one.
- If a finding is real but beyond this PR, say so in the reply and file it as an issue
  with the owning `workstream:` label rather than dropping it.

Read advisory Codex findings as well as the enforced verdict. They are returned in `advisory_items`; a green gate does not mean those findings were addressed.

Native Codex findings use GitHub review threads, not producer verdict markers. Reply in the
original thread after fixing or explaining the finding; do not invent a Codex `addressed` marker.
Never wait for a Codex completion marker.
