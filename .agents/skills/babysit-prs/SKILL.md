---
name: babysit-prs
description: Check all open PRs, surface CI failures, and spawn fix agents for red ones
---

Babysit the open pull requests so finished work doesn't pile up. Arguments (optional): the user’s arguments — `fix` to auto-spawn fix agents, otherwise report-only.

Use the selected assistant's subagent facility and pass the referenced reviewer procedure as
its instructions. If that facility is unavailable, run the tasks sequentially in their isolated
worktrees. Codex does not need to start Claude to follow these procedures.

1. **Survey** — `gh pr list --state open` then `gh pr checks <number>` for each. Build a table: PR, branch, CI status (green/red/pending), **feedback** (from `make pr-feedback PR=<n>`), review status, mergeable.

   The feedback column is the one that used to be missing. `gh pr checks` says whether the machines are happy; it says nothing about the review comments sitting on the PR, which is how work merged past them for months.

2. **Green PRs** — a PR is ready only when CI **and** `pr-feedback` are green. A PR with red feedback is not "ready to merge, pending review" — it is in bucket 4 below. Do NOT merge anything yourself unless the PR carries the `auto-merge` label (then `gh pr merge --auto --squash` is allowed; the required checks still hold it until the feedback is answered).

3. **Red PRs** — for each failing PR, fetch the failing run's log (`gh run view <run-id> --log-failed`) and summarize the root cause in one line.
   - If `fix` was passed: for each red PR, create a headless worktree (`make wt-headless NAME=fix-pr-<number>`) and spawn the `pr-fixer` subagent (defined in `.agents/reviewers/pr-fixer.md`) in the background, passing the branch name, the failure summary, the failing log excerpt, and the worktree path. Its procedure (reproduce → minimal fix → verify → push to same branch) lives in the agent definition. Track the agents, report when they finish, and remove each worktree (`make wt-one-rm`) once its PR is green.
   - Otherwise: just report the failures with their causes.

4. **PRs with unanswered feedback** — for each PR whose `pr-feedback` status is red, run `uv run python scripts/pr_feedback.py --pr <n> --json` and summarize what is open in one line per item.
   - If `fix` was passed: create a headless worktree (`make wt-headless NAME=feedback-pr-<number>`) and spawn the `pr-responder` subagent (defined in `.agents/reviewers/pr-responder.md`) in the background, passing the PR number, the branch name, the JSON verdict, and the worktree path. Its procedure — read what was said, decide fix-or-answer per item, reply then resolve — lives in the agent definition, and `/pr-feedback` is the same procedure done by hand. Track the agents, report when they finish, and remove each worktree (`make wt-one-rm`) once its gate is clear.
   - Otherwise: report the open items and leave them.

   Two states here are not work: `pending` means the review has not run yet, and "Claude Review never posted a verdict" means the workflow itself did not fire — check `gh run list --workflow "Claude Review"` rather than spawning an agent at a diff that is fine.

5. **Stale PRs** — flag PRs behind `main` by many commits or inactive for days; suggest `/sync-main` on their worktree.

6. **Report** — end with a compact status table and the list of actions taken or recommended.

Read advisory Codex findings as well as the enforced verdict. They are returned in `advisory_items`; a green gate does not mean those findings were addressed.
