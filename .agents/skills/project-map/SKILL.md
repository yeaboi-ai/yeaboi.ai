---
name: project-map
description: Full annotated module map of src/yeaboi/ (incl. the MCP server, roadmap, analysis, and agent/headless.py), app flow, the complete CLI flags + subcommands, all environment variables, and the MCP server internals + Claude Code plugin. Use when navigating unfamiliar modules, adding CLI flags/subcommands, changing env/config, or working on the MCP server or plugin.
---

# Project Map

## Project Structure

```
src/yeaboi/
  __init__.py           — Version (__version__), LangSmith noise suppression
  cli.py                — CLI entry point (argparse, 20 flags, headless mode, session mgmt)
  config.py             — Environment/config (API keys, LangSmith, proxy detection, beta-notice acks)
  beta.py               — Canonical wording for features shipping in beta (BETA_LABEL/BETA_TAG/BETA_RGB + the per-mode notice). **Import-free by contract** — every surface pulls from it, incl. mcp/tools_performance.py at module scope, so an import here would drag langchain into MCP boot; test_beta.py AST-scans it
  persistence.py        — Session persistence layer (checkpoint system)
  sessions.py           — SessionStore (SQLite), state serialization, schema versioning
  setup_wizard.py       — First-time setup flow (provider selection, API key validation)
  formatters.py         — Rich Table/Panel rendering (dark/light themes)
  html_exporter.py      — Export plans to self-contained HTML
  json_exporter.py      — Export plans to clean JSON (for CI/CD pipelines)
  prd_exporter.py       — Build/export a Product Requirements Document from a plan (one LLM call for prose, deterministic sections + honest fallback; `--output prd`, MCP plan_export/plan_publish, TUI export flow)
  markdown_convert.py   — Generated Markdown → Notion blocks / Confluence storage XHTML (pure, no SDK; nested lists, hard breaks, links, native Confluence task lists, ⚠ Notices → callout/warning panels, ![alt](path) images via caller-supplied upload maps)
  export_targets.py     — publish_to_notion/publish_to_confluence/publish_markdown (PublishResult; never raises); uploads referenced images (Notion File Upload API / Confluence attach_file), localize_images() for portable .md folders, and yeaboi branding (🤙 Notion page icon, `yeaboi` Confluence label, linked footer auto-appended); with no exports page configured, docs group under an auto-created "🤙 yeaboi" container page (find-or-create, session-cached, best-effort fallback to root/space root)
  charts.py             — velocity/delivered-work PNG charts for exports (optional `charts` extra = matplotlib, lazy-imported; every function returns None gracefully)
  jira_sync.py          — Batch Jira creation (idempotent, cascade, progress callbacks)
  azdevops_sync.py      — Batch Azure DevOps creation (idempotent, cascade, progress callbacks)
  sync_naming.py        — Pure board/iteration numbering: consensus name prefix, starting-number resolution (the -1 sentinel guard), advance-past-closed batches; shared by both syncs and the small-project intake
  questionnaire_io.py   — Import/export questionnaire templates as Markdown
  input_guardrails.py   — Input validation (length, injection, profanity, relevance)
  output_guardrails.py  — Output validation (story format, AC coverage, sprint capacity)
  agent/
    state.py            — ScrumState TypedDict, artifact dataclasses, enums
    graph.py            — Graph compilation and wiring (create_graph())
    nodes.py            — Node functions (intake, analyzer, generators, planner)
    llm.py              — LLM provider factory (Anthropic/OpenAI/Google/Bedrock/Ollama, lazy imports)
    headless.py         — run_planning_pipeline(): auto-driven graph loop (no UI), used by the MCP plan_generate
  prompts/
    system.py           — Base system prompt (Scrum Master persona)
    intake.py           — 30 questions, smart/standard modes, adaptive templates, validation
    analyzer.py         — Project analysis prompt
    feature_generator.py— Feature generation prompt
    story_writer.py     — Story writing prompt with few-shot examples
    task_decomposer.py  — Task decomposition prompt
    sprint_planner.py   — Sprint planning prompt
    standup.py          — Daily Standup summary prompt (ARC framework)
  standup/              — Daily Standup mode (headless-capable, OS-scheduled)
    __init__.py         — public API (run_standup, StandupStore)
    engine.py           — run_standup() pipeline (collect → confidence → LLM summary → deliver → record)
    collector.py        — fan-out recent-activity collection across all sources (graceful skip)
    confidence.py       — deterministic sprint-day + burn-down confidence
    sprint_context.py   — sprint dates/points from plan state + live Jira/AzDO progress
    delivery.py         — NotificationDelivery ABC + Terminal/Desktop/Slack/Email + deliver()
    interactive.py      — timed, TTY-aware scheduled run (prompts for update + confirm; headless fallback)
    errors.py           — StandupSourceError (surfaces source 401/403 as warnings)
    scheduler.py        — OS-native scheduling (launchd on macOS, crontab on Linux); lead-time aware
    render.py           — StandupReport → plaintext (Slack/email) + Rich (terminal/TUI)
    export.py           — StandupReport → Markdown + self-contained HTML (auto-saved every run; Export button)
    store.py            — StandupStore (standup_config/history/updates tables, schema v6)
  retro/                — Retro mode (collaborative browser board, loopback + Cloudflare tunnel)
    __init__.py         — public API (RetroBoard, RetroServer, RetroStore, board_to_report)
    board.py            — RetroBoard (threading.Lock-guarded live cards) + board_to_report()
    server.py           — RetroServer: stdlib ThreadingHTTPServer, token auth, loopback bind, set_public_url()
    page.py             — build_board_html(): self-contained dark browser page (4 grids, polling, XSS-safe)
    engine.py           — generate_action_items(): one LLM call (parse → fallback) from feedback cards
    tunnel.py           — optional Cloudflare quick tunnel (off-network joining); auto-downloads cloudflared, zero-setup
    export.py           — RetroReport → Markdown + self-contained HTML (Export button)
    store.py            — RetroStore (retro_history table, schema v7)
  performance/          — Performance mode (per-engineer 1:1 prep/completion + 6-month review)
    __init__.py         — public API (run_one_on_one_prep, complete_one_on_one, run_six_month_review, PerformanceStore, fetch_roster)
    roster.py           — fetch_roster(): engineer list from Jira/AzDO assignees (graceful [])
    activity.py         — gather_engineer_activity(): current+prior-sprint tickets for one engineer
    engine.py           — the 3 workflow pipelines (parse → fallback → format; one LLM call each)
    context.py          — gather_performance_context(): per-engineer signal → Planning/Analysis
    render.py           — Prep/Completion/Review → Rich + plaintext
    export.py           — Prep/Completion/Review → Markdown + HTML (paths.get_performance_export_dir)
    delivery.py         — 1:1 summary email via SMTP (reuses standup config.get_smtp_*)
    store.py            — PerformanceStore (one_on_ones/reviews/notes tables, schema v8)
    references/         — bundled default competency_framework.md (overridable via env)
  agentwatch/           — the Agents family: what the AI coding agents did, cost, and exposed
    __init__.py         — public API (run_agent_usage, run_agent_security, AgentWatchStore)
    collector.py        — refresh(): ingest local Claude Code/OpenClaw session JSONL → rollups + in-stream security signals with the context each match sat in and a ≤120-char redacted snippet (requestId-deduped, cursor-skipped, never stores a secret)
    engine.py           — the 3 pipelines (deterministic numbers → one LLM call for prose → artifact)
    security_checks.py  — deterministic scans of agent settings/MCP config; findings are (pattern, file, line) only
    security_verdict.py — what a match means: needs-decision / unsure / test-data / handled / info, from where it sat
    security_fixes.py   — the fix catalogue (why + buttons per pattern) and apply_fix(): guard hook, settings edit, PR, rotate
    replay.py           — the redacted transcript turns around one signal (the security page's replay)
    render.py           — the 3 artifacts → Rich (CLI + TUI)
    export.py           — the 3 artifacts → Markdown (no HTML yet — see the beta notice)
    store.py            — AgentWatchStore (agent_sessions keyed on source_path, findings, 3 report tables, schema v27)
  reporting/            — Reporting mode (business-friendly delivery report: last sprint / last month)
    __init__.py         — public API (run_delivery_report, ReportingStore, export_report, build_presentation_html)
    activity.py         — gather_delivered_work(): team-wide completed (Done/Closed) tickets over the period
    sprints.py          — quarter_bounds() + list_sprints() (tracker→plan fallback) for the quarter multi-select
    engine.py           — run_delivery_report() pipeline (gather → one LLM "design" call → parse → fallback)
    render.py           — DeliveryReport → Rich + plaintext (TUI detail view)
    export.py           — DeliveryReport → Markdown + HTML + slide deck (paths.get_reporting_export_dir)
    presentation.py     — deck_payload() + build_presentation_html(): the slide deck's content; renderer is yeaboi-frontend's src/deck
    store.py            — ReportingStore (reporting_history table, schema v9)
  solo/                 — the Solo world's own modules (schema v32)
    today.py            — build_today_snapshot(): the welcome strip's "where am I" snapshot (also GET /api/solo/today)
    engine.py           — run_weekly_review() + carried_actions(): a self-review of the week (gather → verdict → one LLM call → fallback)
    store.py            — WeeklyReviewStore (weekly_review_history table)
    render.py / export.py — WeeklyReview → Rich, and → Markdown (paths.get_solo_export_dir)
  roadmap/              — Roadmap Intake mode (proactive Planning: quarterly roadmap → ranked candidate projects)
    engine.py           — run_roadmap_analysis(): fetch source → LLM analysis → ranked projects (parse → fallback → format)
    store.py            — RoadmapStore (roadmaps list + roadmap_history run log, schema v10/v11)
    export.py           — RoadmapAnalysis → Markdown + HTML (paths.get_roadmap_export_dir)
  analysis/             — Team-analysis engine (headless pipeline behind the TUI Analysis mode)
    engine.py           — run_team_analysis(): fetch history → _run_parallel_analysis → save profile → insights/samples
  anonymize/            — Anonymize action (mask any mode's output for public sharing; post-processing, not a mode)
    engine.py           — run_anonymize(text): deterministic config-seeded mask → one invoke_json LLM pass → parse → seed-only fallback; never raises. Returns the (original→placeholder) replacement map.
    apply.py            — in-place masker: apply_replacements(text) / mask_lines(list[str]) / mask_artifact(frozen via asdict→_dict_to_*) / mask_obj — re-render each mode's OWN screen masked, not a separate view
    export.py           — AnonymizedOutput → Markdown + HTML (paths.get_anonymize_export_dir; embeds a small MD→HTML renderer)
  usage_export.py       — build_usage_text(): serialize the Usage dashboard dict to pasteable plaintext (Copy-to-clipboard)
  mcp/                  — MCP stdio server (yeaboi-mcp entry point, optional [mcp] extra)
    server.py           — create_app() + main(): FastMCP app, guarded mcp import, logs to logs/mcp/
    runtime.py          — {ok, llm_mode, warnings, data} envelope + run_engine()/run_readonly() dispatch
    sampling.py         — SamplingChatModel (LangChain model over MCP sampling) + sampling→provider→fallback chain
    tools_*.py          — one module per domain (planning, sessions, standup, reporting, performance, retro, team)
  tools/
    __init__.py         — get_tools() factory (lazy imports all tool modules)
    github.py           — GitHub repo/file/issues/readme (4 tools) + recent-activity helpers
    local_git.py        — local `git log` recent-commit reader (standup, no SDK/creds)
    azure_devops.py     — Azure DevOps repo/file/work items/board/velocity/create (9 tools)
    jira.py             — Jira board/velocity/sprint/epic/story (6 tools)
    confluence.py       — Confluence search/read/write (5 tools)
    notion.py           — Notion search/read/write (5 tools) + recent-pages helper (own token, not Atlassian auth)
    codebase.py         — Local repo scanning (3 tools)
    calendar_tools.py   — Bank holiday detection (1 tool)
    llm_tools.py        — LLM-powered estimation and AC generation (2 tools)
  repl/                 — Legacy REPL (used for CLI-flag-driven flows)
    __init__.py         — run_repl() entry point
    _intake_menu.py     — Intake mode selection
    _io.py              — Artifact rendering, file import/export, markdown export
    _mode_menu.py       — Mode selection menu
    _questionnaire.py   — Questionnaire UI (one-at-a-time flow)
    _review.py          — Review checkpoint UI (accept/edit/reject)
    _ui.py              — Pipeline progress, streaming, spinner, toolbar
  ui/                   — Full-screen TUI system
    splash.py           — Animated intro
    mode_select/        — Mode selection screens, project cards, project list
    provider_select/    — LLM/tool provider setup, verification
    session/            — Main session (phases, editor, pipeline, Jira export, dry-run)
    shared/             — Animations, ASCII font, components, mouse input
tests/
  unit/                 — Fast unit tests (one file per source module)
    nodes/              — Node tests split into ~9 files (analyzer, route, tasks, etc.)
  integration/          — Graph compilation, multi-node flows, CLI, REPL
  contract/             — Contract tests with recorded API responses (VCR cassettes)
  smoke/                — Live API smoke tests (requires credentials)
  golden/               — Golden dataset evaluators
  fixtures/             — Test data files (SCRUM.md, questionnaire-answers.md)
  _node_helpers.py      — Shared factory functions + JSON fixtures for node tests
```

Conventions:
- Agent logic lives in `agent/` — state, graph wiring, and node functions
- Prompts are separate from agent logic in `prompts/`
- Tools are separate in `tools/` — each tool gets a `@tool` decorator with a descriptive docstring
- Re-export public APIs from `__init__.py` (e.g. `from yeaboi.agent import ScrumState`)
- The `ui/` package is the full-screen TUI; `repl/` is the legacy REPL for CLI-flag-driven flows (`--quick`, `--full-intake`, `--questionnaire`, `--mode`)
- Files prefixed with `_` inside `repl/` and `ui/` subpackages are internal — not public API

## App Flow

Two paths:

1. **Interactive TUI**: `cli.py` → splash → mode selection TUI → provider selection → session TUI
2. **CLI-flag / headless**: `cli.py` → `run_repl()` (for `--quick`, `--full-intake`, `--questionnaire`, `--mode`) or `_run_headless()` (for `--non-interactive`)

Sessions can be listed (`--list-sessions`), resumed (`--resume`), and cleared (`--clear-sessions`). `--dry-run` runs the TUI with fake delays and no LLM calls. `--non-interactive` runs the full pipeline headlessly with `--description` as input and `--output {json,html,markdown,prd}` for format.

## CLI Flags

Key flags to know about when modifying the CLI:

| Flag | Description |
|------|-------------|
| `--non-interactive` | Headless mode (requires `--description`) |
| `--description TEXT` | Project description; `@file.txt` reads from file |
| `--output {markdown,json,html,prd}` | Output format (only with `--non-interactive` or `--export-only`); `prd` writes the Product Requirements Document |
| `--ac-format {gwt,bullets}` | Acceptance-criteria style override (sets `YEABOI_AC_FORMAT`; default: learned team profile, else Given/When/Then) |
| `--architecture-spike {auto,include,skip}` | Architecture validation spike (sets `YEABOI_ARCHITECTURE_SPIKE` when not `auto`; `auto` = confidence-driven) |
| `--team-size N` | Maps to intake Q6 |
| `--sprint-length {1,2,3,4}` | Maps to intake Q8 |
| `--quick` / `--full-intake` | Mutually exclusive intake modes |
| `--questionnaire PATH` | Import filled-in questionnaire |
| `--export-only` | Auto-accept all review checkpoints |
| `--resume [ID]` | Resume session (no arg = picker, `latest` = most recent) |
| `--theme {dark,light}` | Terminal colour theme |
| `--dry-run` | TUI with mock data, no LLM calls |
| `--setup` | Re-run first-time setup wizard |
| `--allow-path PATH` | Session-only filesystem sandbox grant (repeatable); persistent allowances via `YEABOI_ALLOWED_PATHS` |
| `--list-audio-devices` | List the microphones dictation can record from, then exit (rescans first, so a mic plugged in after launch shows up); marks the `VOICE_DEVICE` pick |
| `--install-voice` | Install the dictation packages + speech model into the running environment, then exit — the headless twin of the in-app offer (`voice_install.py`). No MCP tool by design |
| `--standup-run` | Headless: run a daily standup and deliver it (what the OS scheduler invokes) |
| `--standup-interactive` | With `--standup-run`: timed prompt for the user's update + confirm before generating (TTY-aware; headless fallback) |
| `--standup-session ID` | Session to run the standup for (default: most recent) |
| `--standup-output {terminal,desktop,slack,email,all}` | Override the session's saved delivery channels |
| `--standup-remind-transcript` | Headless: post a desktop reminder if standups went unchecked against their meetings (the scheduler's *second* job, fired after the standup) |

**Subcommands** (additive `add_subparsers(dest="command")` — every flat flag keeps working): `yeaboi report`, `yeaboi standup`, `yeaboi perf {roster,prep,complete,review,note}`, `yeaboi review {run,history,export}` (the Solo world's Weekly Review), `yeaboi analyze`. Thin `_cmd_*` handlers in `cli.py` over the shared engines; `--format json` keeps stdout machine-clean. Dispatched in `main()` before the flag-guard sequence. Tests in `tests/unit/test_cli_subcommands.py` (the flat-flag assertions in `tests/integration/test_cli.py` must stay untouched).

Validation rules in `main()`:
- `--non-interactive` requires `--description`
- `--output` requires `--non-interactive` or `--export-only`
- `--export-only` requires `--quick` or `--questionnaire`

## MCP Server

The `src/yeaboi/mcp/` package exposes yeaboi to AI coding agents (Claude Code, Cursor, Codex CLI, VS Code…) as a **stdio MCP server** — entry point `yeaboi-mcp`, optional extra `mcp = ["mcp>=1.9,<2"]` (the `<2` pin is load-bearing; SDK v2 is a breaking pre-release). See README section "MCP Server" for the user-facing docs.

**Design:**
- **26 tools across 8 `tools_*.py` modules**, registered by `server.create_app()` (incl. `tools_anonymize.py`'s `anonymize_text`). Tool bodies lazy-import engines (repo convention); `mcp` itself is imported only inside `create_app()`/`main()` so `import yeaboi.mcp.server` always succeeds without the extra. Exception: `tools_*.py` import `Context` at module level — FastMCP evaluates PEP 563 stringified hints against module globals.
- **Envelope** (`runtime.py`): every tool returns `{ok, llm_mode, warnings, data}`; exceptions become structured `{ok: false, error, hint}` payloads (auth-looking messages get an actionable hint). A tool call never crashes the server. `run_engine()` is the single dispatch point: worker thread (`anyio.to_thread`) + process-wide `threading.Lock` (engines aren't concurrency-safe).
- **LLM chain** (`sampling.py`): `resolve_llm_mode()` picks `sampling` (client advertises the capability) → `provider` (`is_llm_configured()`) → `fallback` (deterministic artifacts + warning). Sampling injects `SamplingChatModel` via `llm_override()` in `agent/llm.py` (a ContextVar that short-circuits `get_llm()`; propagates into anyio worker threads). The model's sync `_generate` bridges to the loop with `anyio.from_thread.run(...)` — only ever call it from a `run_engine` worker thread. `YEABOI_MCP_LLM=provider` forces mode 2; `YEABOI_MCP_MAX_TOKENS` caps sampling responses (default 8192).
- **Planning** runs through `agent/headless.py:run_planning_pipeline()` — a UI-free replica of run_repl's export-only auto-drive (confirm → continue → accept, capacity auto-accepted, `_predict_next_node` == "agent" → done). `repl/_ui.py` re-exports `_predict_next_node` from there.
- **stdio rule**: stdout carries JSON-RPC — nothing may `print()` to it (there's a capsys test). Logs: `~/.yeaboi/logs/mcp/mcp.log` via `attach_mode_handler("mcp")`.
- **Claude Code plugin** at `claude-plugin/` (repo doubles as a marketplace): `yeaboi/.claude-plugin/plugin.json` (no `version` field — commit-SHA versioning), `.mcp.json` at the plugin root (NOT inside `.claude-plugin/`), and five skills (`plan-sprint`, `standup`, `delivery-report`, `performance`, `team-analysis`). Validate with `claude plugin validate claude-plugin/yeaboi`. Tests: `tests/unit/test_mcp_server.py`, `test_mcp_sampling.py`, `test_headless_pipeline.py`, `test_claude_plugin.py`; surface coverage is enforced by `tests/unit/test_surface_parity.py` (see AGENTS.md "REQUIRED: Surface Parity").

## Environment Setup

- `ANTHROPIC_API_KEY` — required when using Anthropic (default provider)
- `OPENAI_API_KEY` — required when `LLM_PROVIDER=openai`
- `GOOGLE_API_KEY` — required when `LLM_PROVIDER=google`
- `AWS_REGION` — required when `LLM_PROVIDER=bedrock` (auto-detected from `~/.aws/config` when a profile is present)
- `AWS_PROFILE` — optional, auto-detected from `~/.aws/config` (looks for profiles with `credential_source` or `role_arn`)
- `LLM_PROVIDER` — `anthropic` (default), `openai`, `google`, `bedrock`, `ollama`, plus the OpenAI-wire vendors `xai`, `deepseek`, `moonshot`, `mistral`, `qwen`, `zai`
- `XAI_API_KEY` / `DEEPSEEK_API_KEY` / `MOONSHOT_API_KEY` / `MISTRAL_API_KEY` / `DASHSCOPE_API_KEY` / `ZAI_API_KEY` — one per OpenAI-wire vendor; each also takes an optional `<VENDOR>_BASE_URL` (see `llm_providers.py`)
- `LLM_MODEL` — optional model override for the selected provider
- `OLLAMA_BASE_URL` — optional, base URL of the local Ollama server (default `http://localhost:11434`); no API key needed when `LLM_PROVIDER=ollama`
- `OLLAMA_NUM_CTX` — optional, context window requested from the Ollama model (default 16384)
- `GITHUB_TOKEN`, `AZURE_DEVOPS_TOKEN` — optional, for repo context tools
- `AZURE_DEVOPS_ORG_URL`, `AZURE_DEVOPS_PROJECT`, `AZURE_DEVOPS_TEAM` — optional, for Azure DevOps board sync
- `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`, `JIRA_PROJECT_KEY` — optional, for Jira integration
- `CONFLUENCE_SPACE_KEY` — optional, the space to scope searches to; the Export buttons publish Confluence reports here too
- `CONFLUENCE_BASE_URL` / `CONFLUENCE_EMAIL` / `CONFLUENCE_API_TOKEN` — optional standalone Atlassian login for Confluence. Confluence reuses the Jira creds by default; these let it be configured **without** Jira (they win over `JIRA_*` when set — see `config.get_confluence_base_url`). The Docs setup step collects them inline when Jira wasn't configured.
- `NOTION_TOKEN` — optional, Notion integration token (independent doc tool; its own auth, not shared with Jira/Confluence). Enables the 5 `notion_*` tools + analysis/standup context.
- `NOTION_ROOT_PAGE_ID` — optional, default parent for created Notion pages; also gates the Notion source in the Daily Standup activity feed (the Confluence-space-key analog)
- `YEABOI_ALLOWED_PATHS` — optional, comma-separated filesystem whitelist for the sandbox (`fs_policy.py`). The app may only read/write inside `~/.yeaboi` (or `$YEABOI_HOME`), a small documented built-in allowlist (CWD `.env`/`SCRUM.md`/`scrum-docs`, `~/.aws/config` read-only, LaunchAgents + Application Support for standup scheduling, legacy `~/.scrum-agent`), and the paths listed here. Session-only grants: `--allow-path`. Editable in the TUI via Settings → Allowed Paths.
- `YEABOI_HOME` — optional, relocates the whole data tree (exports, logs, sessions DB, scrum-docs…; default `~/.yeaboi`). Resolved once at import time in `paths.py` (`_resolve_root()`); `.env` itself always stays at `~/.yeaboi/.env` (the bootstrap file that holds this var). Editable in the TUI via Settings → Data Dir, which offers to move the existing tree (`paths.move_data_tree`) and notes a restart is needed to fully apply.
- `NOTION_EXPORT_PARENT_PAGE_ID` — optional, a dedicated Notion page the Export buttons publish under; **blank groups exports under an auto-created "yeaboi" page (🤙 icon) inside `NOTION_ROOT_PAGE_ID`**. With neither set, Notion export shows a warning pointing at Setup (the Notion API can't create top-level pages).
- `CONFLUENCE_EXPORT_PARENT_PAGE_ID` — optional page Confluence exports nest under; blank groups them under an auto-created "🤙 yeaboi" page at the root of `CONFLUENCE_SPACE_KEY` (no space key → warning pointing at Setup).
- `ANONYMIZE_MASK_TERMS` — optional, comma-separated company-specific terms the Anonymize action always masks (e.g. `"YouLend,YL"`); seeds the deterministic pre-mask pass so they're redacted even when no LLM is available
- `YEABOI_AC_FORMAT` — optional, acceptance-criteria style override for planning (`gwt`/`given-when-then`/`gherkin` or `bullets`/`freeform`/`checklist`; `config.get_ac_format` normalises). Sits between the session's stored choice and the learned team profile in `resolve_ac_style` precedence.
- `YEABOI_ARCHITECTURE_SPIKE` — optional, `include`/`skip` forces the architecture validation spike decision (unset = ask in interactive runs, confidence rule in headless ones). Set by `--architecture-spike`.
- `STANDUP_USER_NAME` — optional, your display name for your own standup update (default: "Me")
- `STANDUP_GITHUB_REPO` — optional *legacy pin*: one GitHub repo (owner/repo) for Daily Standup code activity. No longer required — `GITHUB_TOKEN` alone yields coverage, and the organisations to scan are picked in the Standup code-scope step (each covers every active repo inside it). Setting it keeps the standup narrow to that one repository.
- `SLACK_WEBHOOK_URL` — optional, Slack incoming-webhook URL for Daily Standup delivery
- `STANDUP_SMTP_HOST` / `STANDUP_SMTP_PORT` / `STANDUP_SMTP_USER` / `STANDUP_SMTP_PASSWORD` / `STANDUP_SMTP_SENDER` / `STANDUP_EMAIL_RECIPIENTS` — optional, SMTP email delivery for Daily Standup
- `YEABOI_WEB_STATIC` — optional, a directory of built front-end bundles that overrides the ones Python would otherwise serve (`web/assets.py`, `_static_dir`). For developing **yeaboi-frontend** against a running board: point it at that checkout's `yeaboi_web_assets/static`. Read once at import, so a rebuild needs a restart, and a path that is not a directory raises rather than silently falling back. Unset, the bundles come from the installed `yeaboi-web-assets` wheel — a hard dependency, so a missing one raises rather than degrading.
- `YEABOI_FRONTEND` — optional, a yeaboi-frontend checkout for the generators that write into it (`gen_duck_sprites.py`); defaults to a sibling of the main checkout. See `scripts/_sibling_repos.py`.
- `YEABOI_DESKTOP` — optional, a yeaboi-desktop checkout with the persona brand art, read by `gen_mascot_sprites.py` for the costume traces; defaults to a sibling of the main checkout. Same resolver.
- `RETRO_PORT` — optional, base loopback port for the Retro collaboration server, which the tunnel forwards to (default 5173; walks upward if busy)
- `POKER_PORT` — same for the Poker board (default 5273; clear of retro's 5173..5193 walk range)
- `SHIP_PORT` — same for the Ship board (default 5473), the live view over a supervised ship run
- `YEABOI_OAUTH_PORT` — optional, the loopback port the music sign-in's callback listener binds (default 8643). **Fixed, never walked**: Spotify matches the registered Redirect URI exactly, so changing it means updating the URI in your own Spotify app too
- `YEABOI_SHIP_BOARD` — optional, `1`/`true`/`yes` opts the Ship board in (`config.get_ship_board_enabled()`, default off). Read-only: guests watch; approval stays in the TUI
- `YEABOI_NO_TUNNEL` — optional, `1`/`true`/`yes` stops the live boards opening a Cloudflare tunnel (`config.tunnels_disabled()`). The board still runs for the host on `127.0.0.1` but has nothing to share. Needed because the tunnel now auto-starts: without it `make run-dry` would download ~40 MB and publish a public URL
- `CLOUDFLARED_PATH` — optional, path to an existing `cloudflared` binary for Retro remote tunnels (else the app auto-downloads one to `~/.yeaboi/bin/`)
- `TUNNEL_TIMEOUT_MINUTES` — auto-expiry for every Cloudflare share tunnel (Retro, Poker, and the generic Share Online flow), in minutes (default: 60, 0 = disabled). Enforced once inside `CloudflareTunnel` itself (`retro/tunnel.py`), so all three call sites inherit it uniformly. On the Settings page (System tab, next to Session Prune Days).
- `YEABOI_SOLO` — offers the **Solo** world in the UI (default off). The launch ships Team-only, so with this unset there is one world: the landing split does not render, the loop opens on the mode menu, and the desktop hides its chooser too (it reads `solo_enabled` off `/api/meta/capabilities`). UI-only — the CLI, the MCP tools, the engines and the app's routes ignore it (`config.solo_world_enabled`).
- `YEABOI_LAST_CATEGORY` — **state, not configuration.** The landing split's last world (`solo`/`team`), written on every pick and *preselected* next launch — never auto-skipped. Reads as `team` while Solo is off, and a pre-merge `agents` migrates to `solo` (`config.get_last_category`).
- `YEABOI_UPDATE_CHECK` — optional, `0`/`false`/`off`/`no` disables the background PyPI update check (`update_check.start_background_check`). Used by `make demo` recordings so the version row can never repaint mid-capture; also the opt-out for air-gapped installs
- `PERFORMANCE_FRAMEWORK_PATH` — optional, path to a custom competency framework / review template for Performance mode's 6-month review (else the bundled `performance/references/competency_framework.md` default is used). 1:1 summary emails reuse the standup `STANDUP_SMTP_*` / `STANDUP_EMAIL_RECIPIENTS` settings.
- `BETA_NOTICES_ENABLED` — default on; `false` silences the one-line beta caveat the CLI prints to stderr before a beta subcommand runs (`yeaboi perf …`). Does **not** affect the TUI's one-time notice.
- `BETA_NOTICES_ACK` — **state, not configuration.** Comma-separated `_MODE_CARDS` keys whose one-time TUI beta notice has been dismissed; written automatically to `~/.yeaboi/.env` on Continue. Not surfaced on the Settings page (it would invite hand-editing).
- `YEABOI_FORCE_BETA_NOTICE` — re-show an already-acknowledged TUI beta notice. `1`/`true` for all modes, or a comma-separated list of mode keys. The only way to re-check a once-ever gate for demos, screenshots or review.
- `VOICE_MODEL` — local Whisper size for dictation (`tiny`, `base` (default), `small`, `medium`, `large-v3`, plus the `.en` variants)
- `VOICE_INSTALL_OFFER` — `on` (default) / `off`: whether double-tapping Space may offer to install dictation in place. `n` at the offer sets `off`; `YEABOI_FORCE_VOICE_OFFER=1` overrides it for one run
- `VOICE_EXTRA_INSTALLED` — set once yeaboi has installed dictation itself, so a `uv tool upgrade` that wipes it can be explained rather than silently regressed
- `VOICE_DEVICE` — microphone for dictation: a PortAudio index (`2`) or a case-insensitive name substring (`shure`). Empty = system default. Set it from Settings → Voice Input → Input Device (an arrow-selectable picker with a live level test), or discover names with `yeaboi --list-audio-devices`
- `SESSION_PRUNE_DAYS` — auto-prune sessions older than N days (default: 30, 0 = disabled)
- `LOG_LEVEL` — file logger level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` — optional, enables LangSmith tracing
- Copy `.env.example` to `.env` and fill in keys (`make env`)
- Never commit `.env` or API keys
