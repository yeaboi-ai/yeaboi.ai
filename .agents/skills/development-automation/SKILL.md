---
name: development-automation
description: Run the repo's existing issue implementation, CI repair, security triage, feedback triage, flaky-test, dependency review, or version-classification procedure locally with either Claude or Codex. Use for an explicit request to run one of those development jobs.
---

Read `AGENTS.md` and `.agents/repo-notes.md`. The workflow files below are the source of truth
for each procedure, its input collection, eligibility, deduplication, limits and verification.
Read only the selected workflow, including its guards and its agent prompt. The hosted execution
adapter remains Claude; the procedure also runs from a local Codex or Claude session.

| Task | Procedure |
|---|---|
| Implement an approved issue | `.github/workflows/claude.yml`, `implement` job |
| Diagnose or repair a failed main CI run | `.github/workflows/ci-sentinel.yml` |
| Triage code-scanning alerts | `.github/workflows/codeql-triage.yml` |
| Triage incoming feedback issues | `.github/workflows/feedback-remediation.yml` |
| Investigate flaky tests | `.github/workflows/flaky-test-hunter.yml` |
| Review a dependency update | `.github/workflows/dependabot-auto.yml` |
| Classify a release and update changelog | `.github/workflows/auto-version.yml` |
| Inspect implementation retries | `.github/workflows/implement-reconcile.yml` |

Resolve event expressions from the specified issue/PR/run and current GitHub state before
acting. Never execute an unresolved `${{ ... }}` expression or copy a GitHub Actions shell
block as a local command. The workflow's tool/model arguments are adapter configuration,
not instructions to start Claude from inside Codex.

Keep all eligibility and approval gates: `claude-implement` is the legacy approval label,
not a requirement that Claude write the code. Verify the live issue is open and approved;
preserve labels, retries, branch namespaces, `semver:none`, deduplication and human merge rules.
Triage never applies the implementation approval label. Dry-run requests make no remote changes.

Use a single isolated worktree for a writing task (`make wt-headless`); never run two writers
against the same branch. Run the procedure's Make verification targets, and report an incomplete
run explicitly. Local execution does not authorize a schedule, a hosted runner, automatic merges,
or changes to a sibling repository. Use the actual assistant in attribution.

Codex uses its local ChatGPT login. Do not export that login to GitHub Actions or introduce an
API key. Hosted writer jobs stay on Claude under the subscription-only deployment policy.
