---
name: agent-and-state
description: Agent, prompt, tool, LLM-provider, and state-schema conventions — parse→fallback→format node pattern, frozen-dataclass defaults, serialization, schema versioning, full testing conventions. Use when modifying src/yeaboi/agent/, prompts/, tools/, sessions.py, adding/changing state fields, or writing tests.
---

# Agent, Prompt, Tool & State Conventions

## Node Conventions

Nodes are plain functions in `agent/nodes.py` taking `ScrumState` and returning a dict. Key patterns:

### Parse → Fallback → Format

Every generation node follows this three-helper pattern:
- `_parse_*_response(text)` — extract JSON from LLM response, handle markdown fences
- `_build_fallback_*()` — deterministic fallback artifacts when LLM fails (no LLM call)
- `_format_*()` — Rich rendering for REPL display

### Error handling

- `_is_llm_auth_or_billing_error(e)` checks if an exception is auth/billing — these are **re-raised** (user must fix credentials). All other LLM errors trigger **fallback artifacts** and a warning log.
- Rate-limit errors (429) are retried with exponential backoff in the REPL layer (`_handle_rate_limit`), not in nodes.

### Human-in-the-loop review

When a generation node produces output, it sets `pending_review` to the node name. The REPL intercepts the next user input and routes to `[1] Accept  [2] Edit  [3] Reject`. On edit, feedback is packed as `"{feedback}\n\n---PREVIOUS OUTPUT---\n{serialized}"` so the node can extract both the feedback and previous generation.

## Prompt Conventions

Prompts live in `prompts/` with a factory function per file:
- `get_system_prompt()`, `get_analyzer_prompt(questionnaire, ...)`, `get_feature_generator_prompt(analysis)`, etc.
- Each factory takes parameters (not the full state) and returns a string
- Prompts use the ARC framework: Ask (what to do), Requirements (constraints), Context (background)
- `# See docs: "..."` comments cross-reference theory sections

## Tool Conventions

### Registration

All tools are registered via `get_tools()` in `tools/__init__.py`. This single factory function imports all tool modules and returns a flat list of `BaseTool` instances.

### Lazy imports

Tool modules are imported **inside** `get_tools()`, not at module level. This is because tool dependencies (PyGithub, azure-devops, jira SDK) may not be installed. Lazy import means `from yeaboi.tools import get_tools` always succeeds; ImportError surfaces only when `get_tools()` is called.

### Adding a new tool

1. Create or extend a file in `tools/` with `@tool`-decorated functions
2. Import and append to the tools list in `get_tools()`
3. The docstring is critical — the LLM reads it to decide when to use the tool
4. Set risk level via the tool's position in the graph routing (auto-execute for read, human confirmation for write)

## LLM Provider Conventions

`agent/llm.py` provides `get_llm()` — a factory supporting Anthropic (default), OpenAI, Google, AWS Bedrock, Ollama (local), and the six OpenAI-wire vendors:
- Each provider is **lazy-imported** inside an if-branch so the module works even if optional packages aren't installed
- Provider selected via `LLM_PROVIDER` env var; model override via `LLM_MODEL`
- Default models: `claude-sonnet-4-6` (Anthropic), `gpt-4o` (OpenAI), `gemini-2.5-flash` (Google), `us.anthropic.claude-sonnet-4-6-v1:0` (Bedrock), `qwen3:8b` (Ollama)
- Install optional providers: `uv sync --extra openai` / `--extra google` / `--extra bedrock` / `--extra ollama`
- **`src/yeaboi/llm_providers.py` is the table for every OpenAI-wire vendor** (xAI/Grok, DeepSeek, Moonshot/Kimi, Mistral, Qwen, Z.ai/GLM). They all speak the OpenAI protocol at their own host, so they share **one** `ChatOpenAI(base_url=...)` branch and add no dependency — the `openai` extra covers them. Adding a seventh is a row in that table, not a branch in `get_llm()`: `_PROVIDER_DEFAULTS`, `_ANALYSIS_FAST_MODELS`, `_CLASSIFIER_MODELS`, the wizard cards and the four `provider_verification.py` chains all derive from it. Each takes `<VENDOR>_BASE_URL` to reach a regional host or gateway.
- **Bedrock** uses IAM credentials (no API key) — auto-detects AWS profile from `~/.aws/config` via `get_aws_profile()` in `config.py`. On Lightsail, uses `[profile assumed]` with `credential_source=Ec2InstanceMetadata`. The boto3 session is created with explicit profile + increased read timeout (300s) for cross-region inference profiles.
- **Ollama** is the keyless local provider (README: "Local Mode (Ollama)") — `OLLAMA_BASE_URL` (default `http://localhost:11434`), `OLLAMA_NUM_CTX` (default 16384; the server's 2-4k default silently truncates the big planning prompts). Reliability layer: `get_llm(json_mode=True)` turns on ChatOllama's constrained JSON decoding (`format="json"`, no-op for cloud providers), and `invoke_json()` in `agent/llm.py` wraps every JSON-parsed call site (planning nodes, mode engines, team_learning) with a one-shot "your JSON was invalid, fix it" re-ask for **all** providers. `invoke_json` calls `track_usage()` internally — callers must not track again. Never pass `json_mode=True` on prose paths (conversational agent, llm_tools, guardrail classifier). Local failures surface via `_local_llm_hint()` in `nodes.py` — the single decision point (used by `_should_reraise_llm_error`, the TUI's `_classify_api_error`, and all 4 mode engines) mapping four cases to actionable hints: langchain-ollama not installed → "uv sync --extra ollama", model not pulled (404) → "ollama pull <model>", read timeout → "try a smaller model", server down → "ollama serve". Cloud connection blips keep the graceful fallback. Prose paths strip qwen3 `<think>` blocks via `strip_think_tags()`; chat history is trimmed to the context window by `_trim_history_for_local()` (cut at HumanMessage boundaries only, never splits tool-call pairs); `warn_if_context_overflow()` logs when a prompt likely exceeds `OLLAMA_NUM_CTX`. The setup wizard verifies the langchain-ollama package + server + pulled model, and offers an in-app model download (`pull_ollama_model()` in `provider_verification.py`, streamed `POST /api/pull`).

## State Schema Conventions

- **ScrumState** is a `TypedDict` — this is the LangGraph convention for graph state
- `messages` is the only required field, using `Annotated[list[BaseMessage], add_messages]` for append semantics
- All other fields are optional (`total=False`) and populated progressively as nodes run
- **Frozen dataclasses** for artifacts (Feature, UserStory, Task, Sprint, ProjectAnalysis) — immutable once created, serializable via `asdict()`
- **Mutable dataclass** for QuestionnaireState — updated incrementally by the intake node
- Artifact lists use `Annotated[list[...], operator.add]` so nodes can return new items that get appended

### Adding new state fields

1. Add to `ScrumState` in `agent/state.py`
2. Add tests in `test_state.py`
3. Update `__init__.py` exports if public
4. If the field should persist across `--resume`, it serializes automatically (only `messages` is skipped)
5. If adding a field to a **frozen dataclass**, always provide a default value for backward compatibility with saved sessions (e.g., `title: str = ""`, `discipline: Discipline = Discipline.FULLSTACK`)

### Frozen dataclass backward compatibility

New fields on frozen dataclasses MUST have defaults so deserialization of old JSON doesn't fail:
```python
# Good — old sessions without this field still deserialize
title: str = ""
discipline: Discipline = Discipline.FULLSTACK
test_plan: str = ""

# Bad — breaks --resume for sessions saved before this field existed
title: str   # no default!
```

The `_dict_to_*()` functions in `sessions.py` use `.get()` for optional fields so missing keys don't raise KeyError.

## Session Persistence

### Serialization

`sessions.py` handles state serialization for `--resume`:
- `messages` is the only field skipped (not needed for resume; re-initialized to `[]`)
- Custom `_StateEncoder` handles: frozen dataclasses (`asdict()`), enums (`.value`), sets (→ lists), tuples (→ lists)
- Reconstruction functions: `_dict_to_analysis()`, `_dict_to_story()`, `_dict_to_task()`, `_dict_to_sprint()`, `_dict_to_questionnaire()` — each handles type conversion (enum parsing, tuple reconstruction)

### Schema versioning

- `CURRENT_SCHEMA_VERSION = 12` tracked in a `schema_info` table (v3=team_profiles, v4=session_mode, v5=token_usage, v6=standup config/history/updates, v7=retro history, v8=performance 1:1s/reviews/notes, v9=reporting history, v10=roadmap config/history, v11=multi-row roadmaps list seeded from the v10 singleton, v12=token_usage local-perf columns)
- On startup: if stored version > current → `schema_mismatch = True` (warn user); if stored version < current → run migrations
- Session IDs: internal `new-<8hex>-<YYYY-MM-DD>`, display `<project-slug>-<YYYY-MM-DD>`

## Testing Conventions

- One test file per source module: `repl.py` → `test_repl.py`, `state.py` → `test_state.py`
- Group related tests in classes: `TestGracefulExit`, `TestStreaming`, `TestPriority`
- Use `pytest` fixtures for shared setup (e.g. `_make_console()` for rich Console with StringIO buffer)
- Use `monkeypatch` to avoid filesystem writes, network calls, and delays in tests
- Test both happy path and edge cases (empty input, boundary values, immutability)
- Node tests live in `tests/unit/nodes/` — split into ~9 files by node (analyzer, route, tasks, etc.)
- `tests/integration/test_tui_smoke.py` boots the real TUI (`yeaboi --dry-run`) in a pseudo-terminal and quits it — the only test exercising the live raw-mode/alt-screen path. It guards dependency bumps (e.g. `rich`) that unit tests with StringIO consoles can't catch
- Shared node test helpers in `tests/_node_helpers.py`: `make_completed_questionnaire()`, `make_dummy_analysis()`, `make_sample_features()`, `make_sample_stories()`, `make_sample_sprints()`
- **Never modify `tests/integration/test_repl.py`** — it monkeypatches 10+ names in `yeaboi.repl` and is the only test file with this level of coupling. Future tests should avoid this pattern.
- Pytest markers: `slow` (graph compilation), `eval` (golden evaluators), `vcr` (contract tests), `smoke` (live API)
