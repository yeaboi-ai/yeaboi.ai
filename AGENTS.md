# Agent instructions

Shared guidance for Claude Code and Codex working in this repository.

## Project

Terminal-based AI Scrum Master agent built with LangGraph, LangChain, and Anthropic Claude (with OpenAI, Google, AWS Bedrock, and local Ollama as alternative providers). Two audiences behind one landing split: **Team** (decomposes projects into epics, user stories, tasks, and sprint plans; standups, retros, poker, performance, reporting) and **Solo** (the same, run for one person, plus Weekly Review and the `agentwatch` family — cost, recoverable spend, and security posture of the AI coding agents working across the SDLC, computed locally from Claude Code session logs).

**Solo is hidden in the UI at launch.** `config.solo_world_enabled()` reads `$YEABOI_SOLO` (default off) and is the only thing that gates it: with Solo off there is one world, so the landing split does not render and the desktop shows no world chooser. Nothing else is gated — the CLI, the MCP tools, the engines and the app's HTTP routes all keep working. `YEABOI_SOLO=1 make run-dry` (or `YEABOI_SOLO=1 make dev` in yeaboi-desktop, which passes it to the sidecar) brings the whole world back.

## Commands

```bash
make ship-gate            # The full local gate /ship runs: lint + format-check + test + security + preflight
make test                 # Unit + integration + contract tests (parallel unit lane, then the serial slow lane)
make test-fast            # The whole unit lane, in parallel (~50s)
make test-scoped          # Only the areas the working tree touches + the always-run guards
make test-slow            # Integration + contract only — what CI's second test job runs
make test-v               # Full suite verbose
make test-all             # Everything including golden evaluators
make lint                 # Lint with ruff
make format               # Format with ruff (writes)
make format-check         # What CI's required "Format check (ruff)" job runs (asserts)
make preflight            # Run only the optional CI jobs this branch's diff needs (BASE=origin/main)
make package-check        # uv build + assert the wheel declares yeaboi-web-assets
make run                  # Run the CLI (ARGS="--flag" to pass arguments)
make run-dry              # Run TUI with fake delays, no LLM calls
make eval                 # Run golden dataset evaluators
make contract             # Run contract tests (recorded API responses)
make smoke-test           # Live API smoke tests (requires real credentials)
make snapshot-update      # Update syrupy snapshot baselines after formatter changes
make budget-report        # Show prompt token counts for trend monitoring
make web-types            # Regenerate the contracts yeaboi-frontend vendors (enums + ui)
make dev-board            # Seeded retro board on :5173 for front-end development
make dev-poker            # Seeded planning-poker board on :5273
make dev-deck             # Seeded reporting slide deck on :5373
make graph                # Generate the agent graph PNG into the yeaboi-site checkout
make demo                 # Re-record the terminal demo into yeaboi-site (scripted, no interaction; needs agg)
make demo-render          # Re-render the GIF from the committed cast (theme/size tweaks, no re-record)
make site-contract        # Regenerate contracts/site.json — the facts the website vendors
make build                # Build sdist + wheel into dist/
make publish              # Publish to PyPI
make record               # Re-record VCR cassettes against real APIs
make clean                # Remove build artifacts and caches
```

Run a single test: `uv run pytest tests/unit/test_state.py -v`
Run a single test class: `uv run pytest tests/unit/test_state.py::TestPriority -v`

**CI runs the tests a change touches, not all of them.** `scripts/test_scope.py` maps changed paths
onto areas and `ci.yml`'s `scope` job feeds the result to every other job. Three rules make it safe: `ALWAYS` runs the ~30 guards that
scan the repo rather than importing a module; `GLOBAL` forces the whole suite for anything reached by
everything (`conftest.py`, `sessions.py`, `ui/shared/`, `pyproject.toml`); and **any path the registry
does not recognise runs everything**. The five required status checks never carry an `if:` — scoping
changes what they run, never whether they report. See `tests/unit/test_test_scope.py`, which fails the
build when a source file is claimed by no area or a test file is selected by nothing.

Terminal GIFs: `make demo` re-records `demo.cast.gz` + `demo.gif` from a scripted pty session (deterministic, no interaction; needs `agg`); `make demo-render` re-renders the GIF from the committed cast for theme/size tweaks. Both write into a **yeaboi-site checkout** — see *The website* below.

## Parallel Development (worktrees)

Each feature gets its own git worktree under `<main checkout>/.worktrees/<name>` with its own branch, `.env`, uv venv, port block and `~/.yeaboi` data home (`.worktree.env`, generated at creation; `make wt-repair NAME=…` retrofits an older tree). Hooks are one tracked `.githooks/pre-commit` reached through a relative `core.hooksPath`, so each worktree runs its own venv's. **Shared on purpose**: one `.git` (hence one stash stack — use `make stash`/`make unstash`, never a bare `pop`) and one set of credentials in `~/.yeaboi/.env`. Never develop two features in one checkout. A new worktree is cut from latest `origin/main`. Existing branch names require `REUSE=1`; reuse and refresh rebase onto `origin/main`, skipping dirty trees and aborting conflicts safely. Existing `.claude/worktrees/<name>` trees remain supported in place.

The venv and pre-commit half is this repo's `scripts/provision.sh`, which the shared worktree script runs inside the new tree; the rest lives in **[yeaboi-tooling](https://github.com/yeaboi-ai/yeaboi-tooling)** (see *Shared tooling* below).

```bash
make wt-new NAME=my-feature       # create worktree off latest origin/main + open VS Code with Claude and Codex launch tasks
make wt-headless NAME=my-feature  # same, WITHOUT VS Code (for background-agent work)
make wt-issue ISSUE=123           # worktree from the branch of GitHub issue 123 (linked branch / closing PR); HEADLESS=1 to skip VS Code
make wt-list                      # list worktrees (branch, clean/dirty, path)
make wt-rm NAME=my-feature        # remove worktree dir + branch
```

Slash commands: `/wt` (worktree ops from inside a session), `/sync-main` (rebase on latest main + re-verify), `/ship` (independent review → full tests → commit → push → PR), `/migrate` (fan out a mechanical migration across many files via parallel worktree agents) come from the shared plugin; `/pr-feedback` and `/babysit-prs` are local adapters over `.agents/skills/`. Codex uses the same procedures through those skills or `make agent TASK=...`.

### Shared tooling (`yeaboi-tooling`)

The development workflow is managed in one place for all six yeaboi repos, and arrives here in two halves:

- **The `yeaboi-devkit` Claude Code plugin** — `/ship`, `/sync-main`, `/wt`, `/migrate`, the `code-reviewer` / `test-writer` / `migrator` agents, and the shared hook scripts. Installed by `extraKnownMarketplaces` + `enabledPlugins` in `.claude/settings.json`; nothing to run.
- **A pinned clone** — `mk/common.mk` and the worktree scripts, cloned to a gitignored `.tooling/` at the sha in `.tooling-rev`. The block at the top of the `Makefile` syncs it at parse time, and only when the pin and the checkout disagree, so a fresh worktree provisions itself on its first `make` and the steady state costs no network. Bump with `make tooling-bump`.

**The plugin reaches this repo only through Make targets** — `lint`, `test`, `test-fast`, `test-scoped`, `ship-gate` — which is what lets one `/ship` also drive the front-end, desktop and site repos. The procedure is shared; **this repo's facts live in `.agents/repo-notes.md`**, which `/ship` and `/sync-main` read: which pre-commit hook to skip, what the gate covers, that `auto-version.yml` rewrites the branch after the push, and the rebase conflict playbook. Keep that file current — `tests/unit/test_ship_gate.py` and CI's `make tooling-check` are what notice when it or a target goes missing.

Never edit anything under `.tooling/`: it is a pinned checkout, `tooling-check` fails on a dirty one, and a fix made there is invisible to every other repo. Change it upstream and bump the pin.

### Verification loop

- **Every turn (automatic)**: a Stop hook runs `make lint` + `make test-scoped` whenever a turn ends with dirty source files, and a PostToolUse hook ruff-formats every edited `.py` file. Claude loads the shared scripts through the plugin; Codex uses `.codex/hooks.json` after project trust and `/hooks` approval.
- **At ship time (`/ship`)**: the branch is committed and **rebased onto `origin/main` first** — a gate run on a stale base proves something about a tree that will never exist — and then an independent fresh-context agent reviews `git diff origin/main...HEAD` (spec-fit + conventions) **concurrently** with `make ship-gate` (`lint` → `format-check` → `test` → `security` → `preflight`).
- **In CI**: `claude-review.yml` posts an async code + security review once the full CI suite has passed on a PR (non-blocking; `ci.yml` remains the merge gate).

### Orchestration conventions

When driving multiple features at once, work as an **orchestrator**: one main session, one background agent per feature, each in its own worktree (`make wt-headless`). The orchestrator kicks off agents, tracks them, reviews **final diffs** (not intermediate steps), and runs `/ship` per feature when green. Use `make test-fast` in the inner loop; the full `make test` runs at ship time.

## Front End (`yeaboi-frontend` → the `yeaboi-web-assets` wheel)

Every browser-facing page — the retro and poker live boards, the share gate, the reporting slide deck, the ship board and the ten static HTML exports — is built from TypeScript in **[yeaboi-frontend](https://github.com/yeaboi-ai/yeaboi-frontend)** with Vite. Nothing about a bundle is edited in this repo any more.

The bundles arrive as **`yeaboi-web-assets`**, an ordinary hard dependency in `pyproject.toml`. That is what still lets `pip install yeaboi` work with no Node and keeps `make test` pytest-only.

- **Python reaches them only through `web/assets.py`**, which *resolves* where they live: `$YEABOI_WEB_STATIC` (a sibling checkout's build, for developing the front end against a running board), then the installed `yeaboi_web_assets`. Both failures raise; there is no third fallback to be silently wrong with. A served document's headers and CSPs come only from `web/security.py`; the masthead, frame title and accents only from `web/brand.py`. No request handler writes its own headers, and no Python generates markup — every surface is React, and a payload carries text and numbers, never markup and never presentation (one documented exception, in the skill).
- **The favicon is the one browser asset that stays here.** It is not Vite output: `gen_duck_sprites.py` renders it from the website's duck art, so moving it would make regenerating it a three-repo errand.
- **Three artefacts cross the Python→TypeScript line, and all live in `contracts/web/`**, which the front end vendors by sha: `enums.json` (the server-validated tuples), `ui.json` (the accents and timings its own tests assert against), and `fixtures/` (the wire snapshots). **Changed a board tuple, an accent or a timing? Run `make web-types` and commit `contracts/web/`** — that repo picks it up with `make contracts-sync`.
- **Guards that read TypeScript moved with it.** The CSP/eval/`var(--x)`/breakpoint checks and the duck-sprite geometry now run in that repo's vitest suite; `tests/unit/test_web_contracts.py` is what stays, and it is deliberately its own file — the module those checks used to live in skipped itself whole when `frontend/` was absent.

Everything else — the CSPs and what makes an export inert, the export capability flags, the payload rules, and the two Python/TS wire guards — is in the **`web-frontend`** skill. Read it before touching `src/yeaboi/web/`, `contracts/web/`, or any exporter.

## Desktop app (`yeaboi-desktop`)

The sixth surface is an Electron shell over **`yeaboi app`**, the loopback HTTP backend in `src/yeaboi/app/` — and that backend is the whole of it that lives here. The shell itself is in **[yeaboi-desktop](https://github.com/yeaboi-ai/yeaboi-desktop)**, which installs a *released* wheel from PyPI rather than building this tree; nothing about the app, its icons or its installers is edited or built in this repo any more. That repo is **public**, and it is more than a shell: beside the wheel it stages a second bundled runtime built from its own vendored `backend/` (the planning-platform FastAPI service). Neither is built here.

Two artefacts cross the boundary, both in `contracts/v1/`, and that repo vendors the directory by sha:

- **`app_http.md`** — the wire. Source of truth here, pinned by `tests/unit/test_app_wire.py`. Change a key or a route shape and it is a contract change: update this file and that repo's `make contracts-sync` in the same breath.
- **`routes_manifest.json`** — the desktop's route surface, and the desktop half of the parity registry. It is **generated there** (from its `src/renderer/routes.json`) and **committed here**, because `test_surface_parity.py` and `test_tui_parity.py` read it to decide whether a capability reached the desktop. That repo's `make check-manifest` fails whenever its registries and this snapshot disagree, so a route added on one side alone is red on the other.

A desktop route change is therefore two PRs: theirs, then a small one here carrying the regenerated manifest. `scripts/gen_desktop_icons.py` moved with the app — it read the website's duck art and wrote nothing this repo kept.

## The website (`yeaboi-site`)

yeaboi.ai lives in **[yeaboi-site](https://github.com/yeaboi-ai/yeaboi-site)** — flat HTML with no build step, served off its `main` by GitHub Pages, so merging there is publishing. Nothing about a page, the docs, or `install.sh` is edited in this repo any more.

Two things still cross the boundary, in opposite directions:

- **This repo publishes `contracts/site.json`** — the facts the site states about the package: the Python floor in its structured data, the repo URL in its JSON-LD, the PyPI install target. It is generated from `pyproject.toml` by `make site-contract` and `tests/unit/test_site_contract.py` fails when it is stale. The site vendors it by sha; **change a URL or the floor here and the site is a `make contracts-sync` behind until somebody bumps it.**
- **The site holds artefacts this repo generates, and the master brand art this repo reads.** `make graph`, `make demo`/`demo-render` and the two sprite generators all resolve a yeaboi-site checkout via `scripts/_sibling_repos.py`: `$YEABOI_SITE`, else a sibling of the main checkout. The persona costumes add a **yeaboi-desktop** checkout (`$YEABOI_DESKTOP`, else a sibling) for `gen_mascot_sprites.py`, which traces that repo's brand PNGs. None of them runs on a PR — every output is committed and guarded — so the extra-checkout requirement is paid by whoever changes the product or the brand, never by CI.

## Code Style

- Python 3.10+, ruff for linting/formatting (line-length 120)
- Imports sorted by ruff (isort rules: stdlib, third-party, local)
- Tests in `tests/`, source in `src/yeaboi/`

### Comments

**Legible code first.** Before writing a comment, try to make it unnecessary: a
clearer name, a smaller function, an extracted constant, an earlier return. A
comment is what is left over when the code genuinely cannot say it itself.

When one is needed:

- **Short and concise** — a line or two, not a paragraph and not a section.
- **Say what the code does**, or what a caller must know to use it safely.
- **No war notes.** No history of the bug that was fixed, no record of what was
  tried first, no reasoning about alternatives considered, no narration of the
  problem being solved. That goes in the commit message or the PR.

Much of the existing tree predates this and reads the other way. Match the rule,
not the neighbours; trim a long comment when you are editing the code under it,
but do not sweep files you are not otherwise touching.

## REQUIRED: Learning-First Development

This is the developer's first AI agent. These are NOT optional — follow them on every implementation task.

1. **ALWAYS add `# See docs: <section name>` comments** when introducing a LangGraph or LangChain concept for the first time in a file. Cross-reference the relevant page at https://yeaboi.ai/docs/ (the source lives in the **yeaboi-site** repo) so the developer can look up the theory.
2. **ALWAYS explain LangGraph/LangChain concepts in code comments** on first use — what a reducer does, why `add_messages` exists, what `StateGraph` expects, what `bind_tools` does, etc. Do NOT assume familiarity with these frameworks. This is the one carve-out from the comment rule above: a sentence naming the concept plus the `# See docs:` pointer, not a tutorial in the source.
3. **ALWAYS explain architectural decisions** in your response — when choosing between approaches, state the trade-offs and why this approach was chosen.

Key docs sections to reference:
- "Architecture" (`architecture.html`) — four layers, three design principles, agent graph, TUI system
- "The ReAct Loop" (`architecture.html`) — Thought → Action → Observation pattern
- "Agentic Blueprint Reference" (`architecture.html`) — core graph setup, two core nodes, wiring, tools, memory, streaming
- "Prompt Construction" (`architecture.html`) — ARC framework, few-shot, chain-of-thought, flipped prompt
- "Session Management" (`session-management.html`) — SQLite persistence, --resume, session IDs
- "Guardrails" (`architecture.html`) — input guardrails (4 layers), output guardrails (4 layers), human-in-the-loop
- "Tools" (`tools.html`) — 35 tools, tool types, risk levels
- "Scrum Standards" (`scrum-standards.html`) — story format, acceptance criteria, story points, DoD, discipline tagging

## REQUIRED: Verification

After every code change, ALWAYS run:
1. `make test` — all tests must pass
2. `make lint` — must be clean

At ship time run `make ship-gate` instead: same two, plus `format-check` (a *required* CI check with
no other local twin), `security`, and the `preflight` jobs this diff needs.

Do NOT commit until both pass.

## REQUIRED: Observability & Test Coverage

Every new feature MUST include all three pillars before it can be considered complete:

1. **Logging** — every user action gets `logger.info()` (entry, exit, key decisions); every LLM call logs via `_llm_invoke()`/`track_usage()`; every external API call logs start + result; every error path logs at `warning`/`error` with context. Handler setup, log directories, and the never-log-per-frame rule live in the `logging` skill — Read it when adding logging.
2. **Log directory** — all paths come from `src/yeaboi/paths.py`; never hardcode `Path.home() / ".yeaboi"`. Each mode logs to its own directory under `~/.yeaboi/logs/` (see the `logging` skill).
3. **Tests** — every new function gets at least one unit test (happy path + error case); every `_build_*_screen` gets render tests; every LLM-dependent function gets mock tests (success, error fallback, code fences); every new state field gets serialization round-trip tests; secret/sensitive rendering must be tested for masking. Tests live in `tests/unit/` — one file per source module.

## REQUIRED: Surface Parity

yeaboi ships on **six surfaces**: the TUI, CLI flags/subcommands, the Python engines, the MCP server, the Claude Code plugin skills, and the desktop app. Features MUST NOT land TUI-only. This is machine-enforced by **two** registries, and they ask different questions:

- `tests/unit/test_surface_parity.py` — does this *capability* exist on every surface? A declarative registry plus discovery checks over engines, MCP tools, `_MODE_CARDS`, `build_parser()`, plugin skills, and the desktop route manifest (`contracts/v1/routes_manifest.json`). The desktop rollout is over: every row carries real routes or a reasoned `Exempt`, and a `desktop: scheduled milestone …` exemption can no longer be reintroduced.
- `tests/unit/test_tui_parity.py` — do the *constructs a mode is made of* reach the window? A slash command, a settings section, a keyboard gesture is none of them a capability, and each can land in the terminal alone with the first registry fully green. `TERMINAL_ONLY` names what the desktop deliberately does differently, with the reason.

The contract:

1. **New mode / feature → engine first.** Implement the pipeline as a headless engine (`src/yeaboi/<mode>/engine.py`, parse → fallback → format, frozen-dataclass artifacts). The TUI, CLI, and MCP are thin adapters over it.
2. **Propagate to every surface** (or record a reasoned exemption): an MCP tool in `src/yeaboi/mcp/tools_*.py`, a CLI flag/subcommand in `cli.py`, a TUI card + handler, and — for user-facing workflows — a plugin skill in `claude-plugin/yeaboi/skills/`.
3. **Register it.** Add/extend the capability row in `CAPABILITIES` (and `PARAM_PAIRS` for engine-backed MCP tools) in `tests/unit/test_surface_parity.py`. Until you do, `make test` fails with a message naming the exact edit.
   - **Also add a discoverability tip.** Every capability needs a `FeatureTip` in `src/yeaboi/ui/shared/_tips.py` (`_FEATURE_TIPS`), keyed by the capability name — with a `mode_key` when it owns a `_MODE_CARDS` card so the welcome-screen jump-into-feature key (`g`) lands on it. `TestTips` enforces this two-way; opt out with a `TIP_EXEMPT` entry (reason required). Flag a just-shipped feature with `is_new=True` and clear it a release or two later.
   - **Also add a saved-sessions screen.** A mode card that records runs MUST land on a saved-sessions hub (`_run_mode_hub` in `src/yeaboi/ui/mode_select/__init__.py`, registered in `SAVED_SESSION_HUBS`) so finished runs can be reopened, exported and deleted instead of being overwritten by the next run. `TestSavedSessions` enforces this two-way; opt out with a `SAVED_SESSIONS_EXEMPT` entry (reason required).
4. **New engine params must reach the MCP tool.** The param-parity check compares the engine signature against the tool schema; expose the new param or add it to `HIDDEN_PARAMS` with a reason. `db_path`/`today`/`on_progress`/`dry_run` are injection seams, always hidden.
5. **Deliberate absences use `Exempt("reason")`** — e.g. the retro live board is TUI-only by design. Exemptions are visible, reviewed gaps, not silent ones.
6. **Removals count too.** Every check is two-way set equality: deleting a tool/card/skill without updating the registry also fails.

The MCP server internals and the module map (including `mcp/`, `roadmap/`, `analysis/`, `agent/headless.py`) are in the `project-map` skill; per-mode blueprints (including Roadmap Intake) are in `mode-blueprints`.

## Project Structure (top level)

```
src/yeaboi/
  cli.py / config.py / paths.py      — entry point, env/config, all filesystem paths
  sessions.py / persistence.py       — SQLite session store, state serialization, schema versioning
  agent/                             — ScrumState, graph wiring, node functions, LLM factory, headless.py
  prompts/                           — one factory function per prompt (ARC framework)
  tools/                             — @tool-decorated integrations (GitHub, Jira, AzDO, Confluence, Notion, …)
  standup/ retro/ poker/ performance/ reporting/ roadmap/ analysis/  — standalone modes (shared blueprint)
  agentwatch/                        — the Agents family: usage/standup/security engines over local agent-session telemetry
  niko/                              — the global assistant: a read-only tool loop over every other mode's stores
  provenance/                        — tamper-evident decision chain + conflicts vocabulary (recorded by standup/performance)
  ship/                              — supervised story → PR pipeline: budget fuse, worktree isolation, agent driver, approval gate
  ceremonies/                        — the clock any mode runs on: OS-job installer, guards, delivery channels
  slack/                             — the inbound half of that clock: anchors, a closed grammar, the poller, the ledger
  pricing.py                         — the per-model LLM rate table (cache-aware); every cost estimate goes through it
  mcp/                               — stdio MCP server (yeaboi-mcp; 71 tools over the engines)
  repl/                              — legacy REPL for CLI-flag-driven flows
  ui/                                — full-screen TUI (mode_select, provider_select, session, shared)
  input_guardrails.py / output_guardrails.py / formatters.py / *_exporter.py / *_sync.py
tests/
  unit/ (one file per module; nodes/ split by node)  integration/  contract/  smoke/  golden/  fixtures/
```

Conventions: agent logic in `agent/`, prompts separate in `prompts/`, tools separate in `tools/`; re-export public APIs from `__init__.py`; `_`-prefixed files inside `repl/`/`ui/` subpackages are internal. The full annotated module map is in the `project-map` skill.

## Testing (essentials)

- One test file per source module; group related tests in classes; `monkeypatch` away filesystem/network/delays
- Test happy path + edge cases; node tests live in `tests/unit/nodes/`
- **Never modify `tests/integration/test_repl.py`** (uniquely coupled — monkeypatches 10+ names)
- Pytest markers: `slow`, `eval`, `vcr`, `smoke`
- Full testing conventions (fixtures, helpers, the pty TUI smoke test) are in the `agent-and-state` skill

## Detailed Conventions (lazy-loaded skills)

Deep reference lives in `.agents/skills/` and loads on demand in interactive sessions. In CI/headless contexts, Read the SKILL.md for any area your change touches:

| Skill | Load when touching… |
|---|---|
| `tui-standards` | `ui/`, any `_build_*_screen`, themes, shared components |
| `agent-and-state` | `agent/`, `prompts/`, `tools/`, state fields, `sessions.py`, tests |
| `mode-blueprints` | `standup/`, `retro/`, `performance/`, `reporting/`, `roadmap/`, or adding a new mode |
| `web-frontend` | `src/yeaboi/web/`, `contracts/web/`, any exporter, a share or live-board surface |
| `logging` | logging calls, log files, `logging_setup.py` |
| `ci-and-release` | `.github/workflows`, versioning, releasing, Dependabot, deployment |
| `project-map` | full module map, CLI flags/subcommands, env vars, app flow, the MCP server + plugin |

## Git Conventions

- **Commit messages**: lowercase imperative (e.g. "add streaming output", "fix import sorting")
- **Branch naming**: `feature/<description>` for feature work
- **PRs**: feature branches merge to `main` via pull request
- Attribute AI assistance to the assistant that actually contributed; never claim a different provider or model.

## Shared assistant workflow

Run `make agent-setup` on a fresh checkout to expose the pinned shared skills, then
`make agent-check` to check discovery and CLI availability. Claude commands and Codex skills
read the same procedures. Codex uses `.agents/skills/`; Claude's existing commands remain available.
Start either CLI with `make agent AGENT=claude` or `make agent AGENT=codex`.
Use `TASK=yeaboi-ship` (or another installed skill) and `ARGS="task context"` to select a procedure;
`make agent-run` runs it non-interactively using the CLI's local login.

`make wt-new NAME=x` creates a workspace set and offers both editor launch tasks.
`AGENT=claude` or `AGENT=codex` auto-starts exactly one. `HEADLESS=1` opens no editor.
New worktrees use `<main>/.worktrees/<name>`; existing `.claude/worktrees/<name>` trees remain in place
and are supported by the same commands. `make wt-one` / `make wt-one-rm` affect only this repo;
`make wt-new` / `make wt-rm` affect the workspace set. Use `REPOS="..."` to narrow a set.

Codex hooks require project trust and a review in `/hooks`. They use the same formatting,
scoped verification, and stash guard as Claude. Do not bypass hook trust or sandbox permissions.
Use the installed CLI's own local authentication; this setup expects Codex's ChatGPT login.

## Code Review Rules

- Flag consequential correctness, security, and compatibility defects in changed code; leave mechanical checks to CI.
- Check worktree and repository boundaries: preserve existing work, isolate ports/data, and avoid operating on the main checkout by mistake.
- Preserve this repo's generated-file, contract, and release invariants in `.agents/repo-notes.md`.
- Configure Claude and native Codex GitHub reviews independently. Codex feedback is advisory initially;
  it must remain visible in feedback reports without being treated as a required completion signal.
