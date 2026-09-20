---
name: logging
description: Central logging rules — logging_setup.py handlers, mode_log context manager, per-session logs, log directories from paths.py, LOG_LEVEL, the never-log-per-frame rule. Use when adding logging, creating new log files, or touching logging_setup.py or paths.py.
---

# Logging

**All handler setup lives in `src/yeaboi/logging_setup.py`** — one shared format, one fallback level (`WARNING`), rotation everywhere (2 MB, 3 backups). Never build a `FileHandler` inline; use the central module:
- `configure_logging()` — attaches the always-on main log (`~/.yeaboi/logs/tui/yeaboi.log`); called once early in `cli.main()`
- `with mode_log("<mode>"):` — routes all records to `~/.yeaboi/logs/<mode>/<mode>.log` while a mode page runs (standup, retro, performance, reporting, analysis). Idempotent + detaches on exception. The analysis branch uses explicit `attach_mode_handler`/`detach` (too large for a `with` block)
- `attach_session_log(session_id)` / `detach_session_log()` — per-planning-session log (`~/.yeaboi/logs/planning/{session-id}.log`); called via `persistence.attach_session_logger`
- `apply_level(level)` — retunes the `yeaboi` logger + every attached handler live (used by the Settings page)

Log files:
- **Main/TUI**: `~/.yeaboi/logs/tui/yeaboi.log` (always on)
- **Per mode**: `~/.yeaboi/logs/{standup,retro,performance,reporting,analysis,news}/<mode>.log` — active while that page runs; the scheduled headless standup (`--standup-run`) also writes `standup/standup.log`; the front page's feed refresh (`news/desk.py`) writes `news/news.log`
- **Planning sessions**: `~/.yeaboi/logs/planning/{session-id}.log` (deleted with the project, including `.log.N` rotation backups)
- **Desktop backend**: `~/.yeaboi/logs/app/app.log` — everything `yeaboi app` serves, for the whole life of the process. Stdout belongs to the handshake line, so the log is the only place the backend can say anything
- **Analysis text reports**: `team-analysis-{project}-{timestamp}.log` in `logs/analysis/` — a hand-written product artifact from `team_profile_exporter`, not a logging handler

Rules:
- Log level: `LOG_LEVEL` env var (`DEBUG`/`INFO`/`WARNING`/`ERROR`, default `WARNING`) — settable via `.env` or the Settings page **Log Level** cycle button (`config.set_log_level()` + `apply_level()`)
- **Never log in per-frame code**: `_build_*_screen` builders and render paths run every frame (~60 fps); `logger.info` belongs in key-handling branches of runner loops and one-shot functions only
- INFO = user actions / page open-close / generate / export / config changes; DEBUG = action-time detail; WARNING/ERROR = every failure path. Never log secrets, tokens, join codes, or user content — log ids, counts, names, paths
- All paths defined in `src/yeaboi/paths.py` — never hardcode `Path.home() / ".yeaboi"`
- LangSmith 429 rate-limit errors are auto-suppressed via a custom logging filter in `__init__.py`
- Token usage is tracked via `track_usage()` in `agent/llm.py` and persisted to `token_usage` table in SQLite
