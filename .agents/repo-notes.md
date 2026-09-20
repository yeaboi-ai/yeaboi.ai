# repo-notes — yeaboi

The facts the shared `/ship` and `/sync-main` deliberately do not hardcode, because the front-end,
desktop, site and tooling repos answer them differently. The procedures live in the `yeaboi-devkit`
plugin; this is what they read.

## If `/ship` is missing

The plugin arrives through `.claude/settings.json` alone, and that path is not reliable: Claude
Code can drop the plugin's install record after the marketplace bumps its version, then orphan the
cached copy, and every session afterwards starts with no `/ship`, `/sync-main`, `/wt` or `/migrate`
and no visible error (only a `plugin-cache-miss` line in `claude --debug-file`). The fix:

```
claude plugin install yeaboi-devkit@yeaboi-tooling --scope user
```

then restart the session. A user-scope install covers every worktree, so it needs doing once.

## Commit

Commit with **`SKIP=unit-tests`**. That pre-commit hook is `make test-scoped`, which the Stop hook
already ran at the end of the last turn and which `/ship`'s gate is about to run in full. Three runs
of the same tests is how a gate becomes one people pass with `--no-verify`. gitleaks, ruff and
`ruff-format` still run — those catch something the gate does not.

Use the actual contributing assistant in attribution; omit it when unknown.

## Gate

`make ship-gate` = `lint` → `format-check` → `test` → `security` → `preflight`, fail-fast, in one
`make` invocation so `lint` resolves once for itself and for `security`.

`preflight` is the half `make test` cannot cover: `scripts/preflight.py` runs the optional CI jobs
this branch's diff needs — front-end bundles, the desktop app, golden evaluators, the wheel's
contents, cross-version compat, actionlint — decided by `scripts/test_scope.py`, printing
every job it skipped and why.

**A new capability needs three registry edits or `make test` fails**, each with a message naming the
exact edit: a `CAPABILITIES` row in `tests/unit/test_surface_parity.py`, a `FeatureTip` in
`src/yeaboi/ui/shared/_tips.py`, and — if it records runs — a saved-sessions hub. See AGENTS.md,
*REQUIRED: Surface Parity*.

Changed a board tuple, an accent or a timing the browser is checked against? `make web-types` and commit `contracts/web/` — **yeaboi-frontend** vendors that directory and its CI runs the other half of the check.

Changed `pyproject.toml`'s `requires-python` or a `[project.urls]` entry? `make site-contract` and
commit `contracts/site.json` — the **yeaboi-site** repo vendors it, and `test_site_contract.py`
fails until you do.

## After the push

**The branch is stale again about a minute later, by design.** `auto-version.yml` pushes a
`chore: bump version … [auto]` commit onto the PR *branch*, touching `pyproject.toml` and
`src/yeaboi/changelog_data.json` — and *not* `uv.lock`, which then disagrees with both. Any later
push from the worktree must `git pull --rebase` first, and must **never** force-push over that
commit.

## Conflict playbook

The rows below name a side by what it is, never by `--ours`/`--theirs`: those flags *invert* under
rebase, so "take the upstream side" is unambiguous and "take theirs" is a coin flip.

| Conflicting path | Resolution |
|---|---|
| `contracts/web/**` | Generated. Take either side, then `make web-types` (for `enums.json`/`ui.json`) or re-run `tests/unit/test_web_wire_shapes.py` (for `fixtures/`) and commit what it writes |
| `contracts/web/fixtures/**` | Take the upstream (`origin/main`) side, then `uv run pytest tests/unit/test_web_wire_shapes.py` regenerates them; commit what it wrote |
| `tests/unit/__snapshots__/*.ambr` | Take the upstream side, then `make snapshot-update` |
| `contracts/v1/routes_manifest.json` | Take upstream. It is generated in yeaboi-desktop and only carried here — resolve it there with `make gen-manifest` and bring the result back |
| `pyproject.toml` version line, `src/yeaboi/changelog_data.json` | Keep **`origin/main`'s** and drop your bump entirely. `auto-version.yml` re-bumps on the PR branch, and the changelog is prepend-only — which makes every pair of release-worthy PRs collide here by construction |
| `uv.lock` | Keep **`origin/main`'s**, then `uv lock`. Its root `version` line tracks `pyproject.toml` and `auto-version.yml` does *not* update it, so a disagreement there is expected, not a mistake to preserve |
| `CURRENT_SCHEMA_VERSION` | Renumber **yours** to max+1 in all three places: `src/yeaboi/sessions.py` and its inline history comment, and the two hard-coded `== N` assertions in `tests/unit/test_agentwatch_store.py` and `tests/unit/test_analysis_sessions.py`. Never keep both branches on one number — the plausible resolution is the wrong one, and the second migration then silently never runs on an existing database |
| `CAPABILITIES` (`tests/unit/test_surface_parity.py`) and `_FEATURE_TIPS` (`src/yeaboi/ui/shared/_tips.py`) | Keep **both** rows, in **both** files. They are two-way bound, so keeping one side reds six separate checks |
| `.tooling-rev` | Take whichever sha is newer in the tooling repo, then `make tooling-check`. Never merge the two lines |
| anything else | A genuine overlap — merge both intents and say in the report what you did and why |

## Unattended lane

The `pr-feedback` gate **enforces** on `cowork/…` and `feature/issue-…` branches, triage and sentinel
branches, and anything labelled `cowork` — where nobody is on the other end. Everywhere else it is
advisory and stays green. A human reviewer's unresolved thread or a `Request changes` review holds
the check on any branch.

`/pr-feedback` and `/babysit-prs`, and the `pr-fixer` / `pr-responder` agents, live in this repo's
`.agents/skills/` and `.agents/reviewers/`, with Claude adapters in `.claude/`: they drive `scripts/pr_feedback.py`, and the
`pull_request_target` workflow that runs it is not portable yet.

## Clips

This repo has two surfaces worth filming, and they use different backends.

**The TUI** — `kind: "tty"`, 140×40, the same shape `scripts/record_demo.py` drives:

```python
"cmd": [sys.executable, "-m", "yeaboi.cli", "--dry-run"],
"env": {"YEABOI_UPDATE_CHECK": "0", "YEABOI_NO_TUNNEL": "1", "YEABOI_TELEMETRY": "off",
        "LOG_LEVEL": "ERROR", "ANTHROPIC_API_KEY": "test-key-dry-run-only"},
"env_unset": ["YEABOI_HOME"],
```

`--dry-run` is a CLI flag, not an env var — there is no `YEABOI_DRY_RUN`. Seed a temp `HOME` with a
`~/.yeaboi/.env` so the setup wizard never opens mid-take, and keep `YEABOI_UPDATE_CHECK=0` so the
version row cannot repaint under the recorder.

Two rules the existing recorder learned the hard way, both of which apply to clips:

- **Drive on `await` markers, never `pause`.** Markers make a take quick on a fast machine and
  correct on a slow one; `pause` only holds a screen that is already up.
- **Never put a `key` step immediately after Escape.** `read_key` treats a lone `\x1b` as Escape
  only if no second byte arrives within ~100ms (`src/yeaboi/ui/shared/_input.py`), so a key sent
  straight after is swallowed into an escape sequence and the take silently goes elsewhere.
  `tests/unit/test_record_demo.py` pins this for the canonical demo; nothing pins it for a clip.

**The web boards** — `kind: "page"`, served from this repo's seeded dev servers:

| Target | Port | Notes |
|---|---|---|
| `make dev-board` | 5173 | retro; static until touched, so takes repeat |
| `make dev-poker` | 5273 | **live** — re-announces the crew every 2s |
| `make dev-deck` | 5373 | reporting slide deck |
| `make dev-editable` | 5473 | correctable standup doc |

All are in-memory and write nothing to `~/.yeaboi`, so a clip is safe against a real install. The
tokens are fixed (`dev-token`, `dev-admin`) precisely so a recording cannot invalidate the tab it is
filming. Prefer retro unless the clip is about poker — poker's heartbeat moves state mid-take.

`make demo` still drives `scripts/record_demo.py`, which is TUI-tuned and predates the shared
recorder. Clips use the shared one; the two coexist deliberately.
