"""Surface-parity registry — every capability must ship on every surface (or be exempted).

# See docs: "MCP Server" — the delivery surfaces

yeaboi has six delivery surfaces: the TUI, CLI flags/subcommands, the Python
engines, the MCP server, the Claude Code plugin skills, and the desktop app.
Features have a habit of landing TUI-only. This file is the enforcement: a
declarative registry of capabilities mapped to the surfaces that implement
them, plus discovery checks that FAIL when something new appears on one
surface without being registered (and therefore consciously propagated — or
consciously exempted — everywhere else).

Discovery strategy mirrors ``tests/unit/tools/test_tools_registry.py``:
AST-scan engine modules (no imports, no side effects), introspect the real
FastMCP app for the tool inventory, read ``_MODE_CARDS`` for TUI modes, and
``build_parser()`` for CLI flags. Two-way set equality everywhere it's
meaningful, so removals rot the registry as loudly as additions.

The param-parity checks are the sharpest edge: for each MCP tool that wraps an
engine, the engine's keyword surface must be exposed on the tool, hidden via a
reasoned ``HIDDEN_PARAMS`` entry, or be a universal injection seam
(``HIDDEN_ALWAYS``). A new engine param therefore breaks the build until the
MCP tool grows it too.

How to fix a failure: update ``CAPABILITIES``/``PARAM_PAIRS`` below, or record
an ``Exempt(reason)``/``HIDDEN_PARAMS`` entry — see AGENTS.md
"REQUIRED: Surface Parity".
"""

from __future__ import annotations

import ast
import pathlib
import re
from typing import NamedTuple

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "yeaboi"
PLUGIN_SKILLS_DIR = REPO_ROOT / "claude-plugin" / "yeaboi" / "skills"

_HOW_TO = (
    "Fix: update CAPABILITIES/PARAM_PAIRS in tests/unit/test_surface_parity.py, or record an "
    "Exempt(reason)/HIDDEN_PARAMS entry — see AGENTS.md 'REQUIRED: Surface Parity'."
)


class Exempt(NamedTuple):
    """A deliberate, reasoned absence of a capability on one surface."""

    reason: str


# ---------------------------------------------------------------------------
# The registry — one row per capability, one column per surface.
#
#   engines:   set[(module, function)] — headless pipeline entry points
#   mcp_tools: set[str]                — MCP tool names on the yeaboi-mcp server
#   tui_mode:  str                     — the _MODE_CARDS key
#   cli:       set[str]                — argparse flags / subcommands
#   skill:     str                     — claude-plugin/yeaboi/skills/<name>/
#   desktop:   set[str]                — desktop renderer routes/actions, checked
#                                        against contracts/v1/routes_manifest.json
#                                        (committed by the desktop renderer build).
#                                        "desktop: scheduled milestone" Exempts are
#                                        the rollout ledger — each milestone burns
#                                        its own down; M12 removes the last.
#
# Any column may instead hold Exempt("why this surface is deliberately absent").
# ---------------------------------------------------------------------------

CAPABILITIES: dict[str, dict] = {
    "planning": {
        "engines": {("yeaboi.agent.headless", "run_planning_pipeline")},
        "mcp_tools": {
            "plan_generate",
            "intake_questions",
            "plan_get",
            "plan_export",
            "plan_publish",
            "plan_sync",
            "plan_prior_art",
            "plan_prior_art_feedback",
        },
        "tui_mode": "project-planning",
        "cli": {
            "--non-interactive",
            "--description",
            "--output",
            "--team-size",
            "--sprint-length",
            "--quick",
            "--questionnaire",
            "--export-questionnaire",
            "--export-only",
            "--solo",
            "--mode",
            "--prior-art",
            "--ac-format",
            "--architecture-spike",
            "--context",
            "--project-label",
            "--tag",
        },
        "skill": "plan-sprint",
        # The desktop's planning hub, its composer and the room — the chat in
        # the middle, the living plan in a drawer — all over the chat routes.
        "desktop": {"/planning", "/planning/new", "/planning/:id"},
    },
    "sessions": {
        "engines": Exempt("thin SessionStore reads — no pipeline to extract"),
        "mcp_tools": {"sessions_list", "session_get", "session_delete"},
        "tui_mode": Exempt(
            "not a card — the mode menu is the sessions surface: every card that records runs lands on "
            "its own saved-runs hub (SAVED_SESSION_HUBS), and the cross-mode list is --list-sessions / sessions_list"
        ),
        "cli": {"--list-sessions", "--resume", "--clear-sessions"},
        "skill": Exempt("agents call the session tools directly — no guided workflow needed"),
        # The home: the mode menu over the cross-mode Recent list.
        "desktop": {"/home"},
    },
    "context": {
        # What a run may read: which producer modes, over which timeframe, under
        # which project label and tags. The engine resolves it once; every
        # mode's run engine takes the result as `context=` and every run
        # labels itself for the next one.
        "engines": {
            ("yeaboi.context.engine", "parse_context_spec"),
            ("yeaboi.context.engine", "resolve_scope"),
            ("yeaboi.context.engine", "preview_scope"),
            ("yeaboi.context.engine", "context_options"),
            ("yeaboi.context.engine", "set_session_labels"),
            ("yeaboi.context.engine", "get_session_labels"),
            ("yeaboi.context.engine", "list_session_labels"),
            ("yeaboi.context.engine", "label_run"),
        },
        "mcp_tools": {
            "context_options",
            "context_preview",
            "session_labels_get",
            "session_labels_set",
            "session_labels_list",
        },
        "tui_mode": Exempt("a sub-page every hub's '+ New' card and the planning intake open, not a card of its own"),
        "cli": {"--context", "--project-label", "--tag"},
        "skill": Exempt(
            "agents pass context/project_label/tags straight into each mode's run tool; every mode skill "
            "documents the knob"
        ),
        "desktop": {"dialog:context-scope"},
    },
    "standup": {
        "engines": {
            ("yeaboi.standup.engine", "run_standup"),
            ("yeaboi.standup.engine", "run_transcript_review"),
            ("yeaboi.standup.engine", "import_transcript"),
            ("yeaboi.standup.engine", "transcript_nudge"),
            ("yeaboi.standup.engine", "file_transcript_issues"),
        },
        "mcp_tools": {
            "standup_run",
            "standup_history",
            "standup_config_get",
            "standup_config_set",
            "standup_members",
            "standup_repositories",
            "standup_review",
            "standup_gaps",
            "standup_practice_feedback",
        },
        "tui_mode": "daily-standup",
        "cli": {
            "standup",
            "standup-review",
            "--standup-run",
            "--standup-session",
            "--standup-output",
            "--standup-interactive",
        },
        "skill": "standup",
        "desktop": {
            "/team/standup",
            "/team/standup/setup",
            "/team/standup/schedule",
            "/team/standup/review",
        },
    },
    "reporting": {
        "engines": {("yeaboi.reporting.engine", "run_delivery_report")},
        # reporting_history / reporting_export are read-only store/export wrappers
        # (no pipeline) — the saved-runs hub surfaces them; parity with retro-board.
        "mcp_tools": {"report_delivery", "reporting_history", "reporting_export"},
        "tui_mode": "reporting",
        "cli": {"report"},
        "skill": "delivery-report",
        "desktop": {"/team/reporting", "/team/reporting/new", "/team/reporting/style"},
    },
    "weekly-review": {
        # The Solo world's one capability of its own: a self-review of the week
        # over the user's own standups, delivered work and plan. carried_actions
        # is the headless carry-forward load (last review's open actions) the
        # TUI, MCP and desktop show for marking — parity with retro-board's
        # carried_action_items_for_session.
        "engines": {
            ("yeaboi.solo.engine", "run_weekly_review"),
            ("yeaboi.solo.engine", "carried_actions"),
        },
        # weekly_review_history / weekly_review_export are read-only store/export
        # wrappers (no pipeline) — parity with reporting_history / reporting_export.
        "mcp_tools": {"weekly_review_run", "weekly_review_history", "weekly_review_export"},
        "tui_mode": "weekly-review",
        "cli": {"review"},
        "skill": "weekly-review",
        "desktop": {"/solo/review", "/solo/review/report"},
    },
    "performance": {
        "engines": {
            ("yeaboi.performance.engine", "run_one_on_one_prep"),
            ("yeaboi.performance.engine", "complete_one_on_one"),
            ("yeaboi.performance.engine", "run_six_month_review"),
        },
        "mcp_tools": {
            "perf_roster",
            "perf_one_on_one_prep",
            "perf_one_on_one_complete",
            "perf_six_month_review",
            "perf_note_add",
        },
        "tui_mode": "performance",
        "cli": {"perf"},
        "skill": "performance",
        "desktop": {"/team/performance", "/team/performance/engineer"},
    },
    "scrum-poker": {
        # get_poker_perspective: the one LLM call (AI take on a revealed vote
        # spread); the live voting board itself is a real-time server the TUI
        # hosts and tunnels — like retro, it can't be a one-shot pipeline.
        "engines": {
            ("yeaboi.poker.engine", "get_poker_perspective"),
            # The one record site every host (the app supervisor, the TUI page)
            # calls, so a closed table is stored and labelled the same way.
            ("yeaboi.poker.engine", "record_poker_run"),
        },
        "mcp_tools": {"poker_history", "poker_export"},
        "tui_mode": "poker",
        "cli": {"poker"},  # history read-back + export; the live voting board stays TUI-hosted
        "skill": Exempt("live voting session is TUI-hosted by design; history stays readable via poker_history"),
        "desktop": {"/team/poker", "/team/poker/new", "/team/poker/board"},
    },
    "retro-board": {
        # carried_action_items_for_session: the headless carry-forward load (prior
        # retro's action items) the TUI/browser adapt for the review column.
        # history_providers/report_payload: the board's step-back through previous
        # retros — the same runs `retro_history` already reads, shaped as cards.
        "engines": {
            ("yeaboi.retro.engine", "generate_action_items"),
            ("yeaboi.retro.engine", "carried_action_items_for_session"),
            ("yeaboi.retro.engine", "history_providers"),
            ("yeaboi.retro.engine", "report_payload"),
            # The standup→retro edge: a scoped board seeds the selected
            # standups' blockers as review cards.
            ("yeaboi.retro.engine", "standup_blocker_cards"),
            ("yeaboi.retro.engine", "record_retro_run"),
        },
        "mcp_tools": {"retro_history", "retro_export"},  # carried data rides along in retro_history's report
        "tui_mode": "retro",
        "cli": {"retro"},  # history read-back + export; the live board itself stays TUI-hosted
        "skill": Exempt("live board is TUI-only by design; history stays readable via retro_history"),
        "desktop": {"/team/retro", "/team/retro/board"},
    },
    "team-learning": {
        "engines": Exempt("lives in tools/team_learning.py as @tool functions — covered by test_tools_registry"),
        "mcp_tools": {"team_profile_get", "team_compare_plan_to_actuals"},
        "tui_mode": Exempt("profiles are consumed inside the planning/analysis screens, no dedicated card"),
        "cli": {"--team-profile", "--retro"},  # --learn moved to team-analysis (drives its engine now)
        "skill": Exempt("no plugin skill yet — tracked gap"),
        "desktop": Exempt("consumed inside the planning/analysis panels, same as the TUI — no dedicated page"),
    },
    "team-analysis": {
        "engines": {
            ("yeaboi.analysis.engine", "run_team_analysis"),
            ("yeaboi.analysis.engine", "get_team_roster"),
            ("yeaboi.analysis.engine", "get_team_roster_result"),
        },
        "mcp_tools": {"team_analyze", "team_roster"},
        "tui_mode": "team-analysis",
        "cli": {"analyze", "--learn"},
        "skill": "team-analysis",
        "desktop": {"/team/analysis", "/team/analysis/new", "/team/analysis/results"},
    },
    "roadmap": {
        # Landed on main (TUI-only) before both this parity framework and the
        # MCP surface existed; the non-TUI surfaces are visible tracked gaps,
        # not silent ones — a follow-up should add a roadmap_analyze tool + CLI.
        "engines": {
            ("yeaboi.roadmap.engine", "run_roadmap_analysis"),
            ("yeaboi.roadmap.engine", "intake_mode_for"),
        },
        "mcp_tools": Exempt("no roadmap_analyze tool yet — tracked follow-up gap (newer than the MCP surface)"),
        "tui_mode": Exempt("a Planning intake card in _INTAKE_CARDS (Chat/Roadmap/Offline), not a mode card"),
        "cli": Exempt("interactive source picker + intake handoff; a headless roadmap path is a tracked gap"),
        "skill": Exempt("no plugin skill yet — tracked follow-up gap"),
        # The desktop opens an engine chat from a roadmap item — the one
        # roadmap path that is not TUI-only, and the gap the four rows above
        # still track on the other surfaces.
        "desktop": {"/planning/from-roadmap"},
    },
    "anonymize": {
        # Post-processing action, not a mode of its own: an "Anonymize" button on every
        # mode's result screen masks the already-rendered output. The engine + MCP tool
        # give it real headless reach; the TUI-card/CLI/skill surfaces are deliberate gaps.
        "engines": {("yeaboi.anonymize.engine", "run_anonymize")},
        "mcp_tools": {"anonymize_text"},
        "tui_mode": Exempt("an action button on every mode's result screen, not a _MODE_CARDS entry"),
        "cli": Exempt("headless callers anonymize via the anonymize_text MCP tool"),
        "skill": Exempt("post-processing action, not a guided workflow"),
        "desktop": {"action:anonymize"},
    },
    "artifact-editing": {
        # Reader-authored corrections to a generated artifact. The browser half
        # is an action on the existing Share Online screen; the engine and the
        # MCP tools give it real headless reach, so an agent can fix a wrong
        # name before the report goes out and gets the same validation, caps and
        # allowlist the teammate in the browser would have.
        "engines": {
            ("yeaboi.artifacts.engine", "artifact_fields"),
            ("yeaboi.artifacts.engine", "artifact_edit_history"),
            ("yeaboi.artifacts.engine", "apply_artifact_edits"),
        },
        "mcp_tools": {"artifact_fields", "artifact_edit_history", "artifact_edit_apply"},
        "tui_mode": Exempt("editing happens on the shared browser document during a Share Online session"),
        "cli": Exempt("a JSON-patch flag would be an unusable surface; headless callers use the MCP tools"),
        "skill": Exempt("correcting one field is a single MCP call, not a multi-step guided workflow"),
        "desktop": {"action:edit-artifact"},
    },
    "output-sharing": {
        "engines": Exempt("transport over already-generated HTML artifacts, not an artifact-generation pipeline"),
        "mcp_tools": Exempt(
            "a stdio tool cannot safely own an interactive host process and temporary tunnel lifecycle"
        ),
        "tui_mode": Exempt("a Share Online action on existing result screens, not a dedicated mode card"),
        "cli": Exempt("temporary shares intentionally remain visible and cancellable in the interactive TUI"),
        "skill": Exempt("local process and access-code ownership belongs to the human host in the TUI"),
        "desktop": {"dialog:share", "dialog:export"},
    },
    "usage": {
        "engines": Exempt("TUI utility page — reads the local token_usage table"),
        "mcp_tools": {"usage_get"},
        "tui_mode": "usage",
        "cli": Exempt("TUI utility page; headless callers read usage_get over MCP"),
        "skill": Exempt("TUI utility page"),
        "desktop": {"/usage"},
    },
    "settings": {
        "engines": {
            ("yeaboi.settings.engine", "get_settings"),
            ("yeaboi.settings.engine", "set_setting"),
            ("yeaboi.settings.engine", "set_allowed_paths"),
            ("yeaboi.settings.engine", "set_list_setting"),
            ("yeaboi.settings.engine", "set_data_dir"),
            ("yeaboi.settings.engine", "provider_catalog"),
            ("yeaboi.settings.engine", "verify_provider"),
            ("yeaboi.settings.engine", "verify_connection"),
            ("yeaboi.settings.engine", "discover_models"),
        },
        "mcp_tools": Exempt("MCP servers must not rewrite host credentials"),
        "tui_mode": "settings",
        "cli": {"--setup", "--theme", "--allow-path", "--list-audio-devices", "--install-voice", "--setup-access"},
        "skill": Exempt("TUI utility page"),
        "desktop": {"/settings/credentials", "/settings/sharing", "/settings/system", "/setup"},
    },
    # The read-only connector layer. ONE row for the whole layer, not one per
    # vendor: twelve rows would mean twelve tips, which is the crowding the
    # layer exists to remove.
    "connections": {
        "engines": {
            ("yeaboi.connectors.engine", "list_connections"),
            ("yeaboi.connectors.engine", "fetch_ops_events"),
            # The user-created half of the layer. No MCP write tools on
            # purpose: an MCP server must not rewrite host credentials, and
            # must not mint a descriptor that aims them either.
            ("yeaboi.connectors.engine", "create_custom_connection"),
            ("yeaboi.connectors.engine", "delete_custom_connection"),
            ("yeaboi.connectors.engine", "draft_custom_connection"),
        },
        "mcp_tools": {"connections_list", "connections_fetch"},
        "tui_mode": Exempt(
            "a Settings tab, not a mode card — connecting a vendor is configuration, "
            "and an eleventh card costs the welcome screen a row it does not have"
        ),
        "cli": {"connections", "webhooks"},
        "skill": Exempt(
            "connecting a vendor is a credential write at the user's own machine; an agent "
            "reads the catalog with connections_list and cannot add one"
        ),
        "desktop": {"/settings/connections"},
    },
    # Ceremonies are the clock other modes run on, not a mode of their own: the
    # engine fires a catalogued mode and delivers its output.
    "ceremonies": {
        "engines": {("yeaboi.ceremonies.engine", "run_ceremony")},
        # Read-only on this surface, for the same reason the standup's own
        # schedule has never been settable here: declaring one installs a
        # launchd/crontab job on the user's machine that outlives the session,
        # survives reboots and spends money unattended.
        "mcp_tools": {"ceremonies_list", "ceremonies_history"},
        # Not a mode card: the menu draws every card with no scrolling, and an
        # eleventh pushes the version/changelog/feedback row off screen at the
        # enforced 84x40 minimum (measured) — trading three affordances every
        # user has for one menu entry. Reached by `s` from the welcome screen,
        # beside the changelog and feedback keycaps it belongs with.
        "tui_mode": Exempt("a welcome-screen keycap (s), not a card — an 11th card breaks the 84x40 layout"),
        "cli": {"ceremonies"},
        "skill": "ceremonies",
        # A full page rather than a card: the 84x40 constraint that kept it off
        # the TUI menu does not exist here.
        "desktop": {"/ceremonies"},
    },
    # The inbound half of that clock: a team reacting or replying in Slack, read
    # back on a schedule and applied to the run the post was about.
    "slack-inbound": {
        "engines": {
            ("yeaboi.slack.engine", "apply_inbound_events"),
            ("yeaboi.slack.engine", "inbound_history"),
            ("yeaboi.slack.engine", "link_slack_member"),
        },
        # Read-only on MCP for a sharper reason than ceremonies': the allowlist
        # that authorises an event lives in the poller, so this engine INHERITS
        # authorisation rather than checking it. An apply tool would let any MCP
        # client fabricate an event and drive a verdict or a pause with no Slack
        # in the loop at all. `link_slack_member` is absent for a different
        # reason — it is safe, but it decides whose name goes on somebody else's
        # report, which is the one binding Slack did not attest.
        "mcp_tools": {"slack_inbound_history", "slack_identities_list"},
        "tui_mode": Exempt("a Slack column and a link hint on the Ceremonies page — no 11th card (84x40)"),
        "cli": {"slack"},
        "skill": "slack-inbound",
        "desktop": {"/ceremonies/slack"},
    },
    # ── The Agents family (agentwatch) — cards live on the Agents menu
    # (_AGENT_CARDS), a sibling list of _MODE_CARDS behind the landing split.
    # Every mode ships at full parity: no Exempt entries, because each mode's
    # engine/MCP/CLI/skill surfaces landed in the same phase commit that
    # created its card.
    "agent-usage": {
        "engines": {("yeaboi.agentwatch.engine", "run_agent_usage")},
        # agents_usage_history is a read-only store wrapper (no pipeline) —
        # parity with reporting_history / retro_history.
        "mcp_tools": {"agents_usage", "agents_usage_history"},
        "tui_mode": "agent-usage",
        "cli": {"agents"},
        "skill": "agents-usage",
        "desktop": {"/agents/usage"},
    },
    "agent-advisor": {
        # advisor.py, not engine.py: the mirrored engine surface is served by
        # the Go sidecar, and the advisor pipeline is deliberately Python-only
        # (see advisor.py's module docstring) — a separate module keeps the
        # dual-maintenance boundary visible.
        "engines": {("yeaboi.agentwatch.advisor", "run_agent_advisor")},
        "mcp_tools": {"agents_advisor_run", "agents_advisor_history"},
        "tui_mode": "agent-advisor",
        "cli": {"agents"},
        "skill": "agents-advisor",
        "desktop": {"/agents/advisor"},
    },
    "agent-security": {
        "engines": {
            ("yeaboi.agentwatch.engine", "run_agent_security"),
            ("yeaboi.agentwatch.engine", "rebuild_security_report"),
        },
        "mcp_tools": {
            "agents_security_scan",
            "agents_security_history",
            "agents_security_dismiss",
            "agents_security_replay",
            "agents_security_signals",
            "agents_security_fix",
            "agents_security_verdict",
        },
        "tui_mode": "agent-security",
        "cli": {"agents"},
        "skill": "agents-security",
        "desktop": {"/agents/security"},
    },
    "niko": {
        # The global assistant. Read-only by construction: its tool surface
        # (yeaboi/niko/tools.py) holds no write tool, which is why the same
        # engine is safe to expose as an MCP tool — see that module on why it
        # must never call back through the dispatcher.
        "engines": {("yeaboi.niko.engine", "ask")},
        "mcp_tools": {"niko_ask"},
        "tui_mode": Exempt(
            "the mascot himself plus a welcome-screen keycap (n), not a card — the menu draws "
            "every card and an eleventh pushes the version row off at the enforced 84x40 minimum"
        ),
        "cli": {"ask"},
        "skill": "niko",
        # Chrome rather than a page: Niko is a panel over every route, so it
        # claims an action pseudo-path the way anonymize and share do.
        "desktop": {"action:ask-niko"},
    },
    "provenance": {
        "engines": {
            ("yeaboi.provenance.engine", "run_provenance_audit"),
            ("yeaboi.provenance.engine", "trace_entity"),
        },
        "mcp_tools": {"provenance_audit", "provenance_trace"},
        "tui_mode": Exempt(
            "the chain records itself during standup/performance runs and its cards render inside "
            "those modes (the standup Conflicts card); the verify/trace surface is CLI/MCP-first — "
            "a dedicated TUI card is a tracked gap"
        ),
        "cli": {"provenance"},
        "skill": "provenance",
        # The audit + trace explorer — the first surface with a page for this.
        "desktop": {"/provenance"},
    },
    "ship": {
        # The supervised story → PR pipeline. run_ship is the one entry point;
        # the MCP surface is read-only by design — launching holds a live
        # subprocess for many minutes behind the server's engine lock, and the
        # approval gate is a human decision made at a terminal (the
        # output-sharing precedent).
        # resume_ship is the second entry point, not a second capability: it
        # re-enters the same pipeline for a run abandoned at the gate, which is
        # otherwise stranded (only the engine writes `approved` and opens the PR).
        # run_ship_batch is the third: one launch, N stacked story runs, for an
        # epic shipped as one PR per story (`ship run <EPIC> --split`).
        "engines": {
            ("yeaboi.ship.engine", "run_ship"),
            ("yeaboi.ship.engine", "resume_ship"),
            ("yeaboi.ship.engine", "run_ship_batch"),
        },
        "mcp_tools": {"ship_history", "ship_status"},
        "tui_mode": "ship",
        "cli": {"ship"},
        "skill": "ship",
        # Present, but story-level only: the desktop cannot yet target an epic
        # or a task, nor split an epic into stacked PRs. A route-set check
        # cannot see a narrowing inside a route, so it is named here and in
        # contracts/v1/app_http.md.
        "desktop": {"/team/ship", "/team/ship/run"},
    },
}

# Engine modules discovered by convention: every src/yeaboi/*/engine.py, plus
# the planning pipeline which (for LangGraph reasons) lives in agent/headless.py
# and the advisor pipeline which lives outside agentwatch/engine.py to stay off
# the Go-mirrored surface.
EXTRA_ENGINE_MODULES = {
    "yeaboi.agent.headless": SRC / "agent" / "headless.py",
    "yeaboi.agentwatch.advisor": SRC / "agentwatch" / "advisor.py",
}

# ---------------------------------------------------------------------------
# Param parity: MCP tool ↔ engine signature.
# ---------------------------------------------------------------------------

# Which engine entry point each engine-backed MCP tool wraps. Store-read tools
# (standup_history, retro_history, sessions_*, team_*) have no pipeline pair.
PARAM_PAIRS: dict[str, tuple[str, str]] = {
    "connections_list": ("yeaboi.connectors.engine", "list_connections"),
    "connections_fetch": ("yeaboi.connectors.engine", "fetch_ops_events"),
    "plan_generate": ("yeaboi.agent.headless", "run_planning_pipeline"),
    "context_options": ("yeaboi.context.engine", "context_options"),
    "context_preview": ("yeaboi.context.engine", "preview_scope"),
    "session_labels_set": ("yeaboi.context.engine", "set_session_labels"),
    "session_labels_list": ("yeaboi.context.engine", "list_session_labels"),
    "standup_run": ("yeaboi.standup.engine", "run_standup"),
    "standup_review": ("yeaboi.standup.engine", "run_transcript_review"),
    "report_delivery": ("yeaboi.reporting.engine", "run_delivery_report"),
    "weekly_review_run": ("yeaboi.solo.engine", "run_weekly_review"),
    "perf_one_on_one_prep": ("yeaboi.performance.engine", "run_one_on_one_prep"),
    "perf_one_on_one_complete": ("yeaboi.performance.engine", "complete_one_on_one"),
    "perf_six_month_review": ("yeaboi.performance.engine", "run_six_month_review"),
    "team_analyze": ("yeaboi.analysis.engine", "run_team_analysis"),
    "anonymize_text": ("yeaboi.anonymize.engine", "run_anonymize"),
    "agents_usage": ("yeaboi.agentwatch.engine", "run_agent_usage"),
    "agents_advisor_run": ("yeaboi.agentwatch.advisor", "run_agent_advisor"),
    "agents_security_scan": ("yeaboi.agentwatch.engine", "run_agent_security"),
    "provenance_audit": ("yeaboi.provenance.engine", "run_provenance_audit"),
    "provenance_trace": ("yeaboi.provenance.engine", "trace_entity"),
    "niko_ask": ("yeaboi.niko.engine", "ask"),
}

# Injection/test seams that are never exposed on any wire surface.
# ``on_run_id`` joins them for the same reason as ``on_progress``: it hands a
# caller-side callback the id of the run being started, which only a surface
# that owns the process can use. ``on_agent_line`` is the same shape — it
# streams the coding agent's live output to a caller that owns the process (the
# ship board), and is meaningless on a CLI flag or an MCP wire.
HIDDEN_ALWAYS = {"db_path", "today", "on_progress", "on_run_id", "on_agent_line", "dry_run"}

# Per-tool engine params deliberately not exposed on the MCP tool. Every entry
# needs a reason; a stale entry (param gone from the engine) fails the tests.
HIDDEN_PARAMS: dict[str, dict[str, str]] = {
    "connections_fetch": {
        "now": "clock injection seam for deterministic window tests — the same shape as `today`",
    },
    "niko_ask": {
        "on_event": "caller-side stream callback — same shape as on_progress, meaningless on a wire",
        "cancel": "a threading.Event only a surface that owns the process can set",
        "user_name": "who is asking; the window knows it from the identity file, an MCP host does not",
        "surface": "the adapter fixes it to 'terminal' — it only changes how navigate is described",
    },
    "plan_generate": {
        "questionnaire": "adapter — built from description/answers/project_context",
        "session_id": "plan_generate always mints a fresh session; the id is returned in data",
        "save_session": "MCP plans are always persisted — the session id IS the handle",
        "max_steps": "internal runaway-loop guard, not a user knob",
    },
    "team_analyze": {
        "progress": "injected adapter — the tool bridges it to ctx.report_progress notifications",
        "team_name": "AzDO team label; MCP auto-resolves it from the configured AZURE_DEVOPS_TEAM",
        "cancel_event": "in-process threading.Event cancel seam for the TUI worker; meaningless over the MCP wire",
    },
    "report_delivery": {
        "cancel_event": "in-process threading.Event cancel seam for the TUI worker; meaningless over the MCP wire",
    },
}

# Tool params with no engine counterpart — adapter inputs the tool assembles
# into the engine's arguments.
TOOL_ONLY_PARAMS: dict[str, set[str]] = {
    "plan_generate": {"description", "answers", "team_size", "sprint_length_weeks", "project_context"},
    # file_issues is an adapter over the SECOND engine entry point
    # (file_transcript_issues): run_transcript_review deliberately has no such
    # param, so the drafting path structurally cannot publish.
    "standup_review": {"file_issues"},
}

# ---------------------------------------------------------------------------
# Param parity: CLI subcommand ↔ engine signature (the same drift guard as
# PARAM_PAIRS, for the `yeaboi <command>` surface — CLI-only gaps shipped
# because only the MCP side was enforced).
# ---------------------------------------------------------------------------

# Which engine each headless subcommand drives ("perf prep" = nested path).
# The planning capability's CLI is the flat --non-interactive flag set, which
# predates subcommands and maps through QuestionnaireState — not pairable here.
CLI_PARAM_PAIRS: dict[str, tuple[str, str]] = {
    "report": ("yeaboi.reporting.engine", "run_delivery_report"),
    "standup": ("yeaboi.standup.engine", "run_standup"),
    "standup-review": ("yeaboi.standup.engine", "run_transcript_review"),
    "review run": ("yeaboi.solo.engine", "run_weekly_review"),
    "perf prep": ("yeaboi.performance.engine", "run_one_on_one_prep"),
    "perf complete": ("yeaboi.performance.engine", "complete_one_on_one"),
    "perf review": ("yeaboi.performance.engine", "run_six_month_review"),
    "analyze": ("yeaboi.analysis.engine", "run_team_analysis"),
    "agents cost": ("yeaboi.agentwatch.engine", "run_agent_usage"),
    "agents security": ("yeaboi.agentwatch.engine", "run_agent_security"),
    "ship run": ("yeaboi.ship.engine", "run_ship"),
    "ship resume": ("yeaboi.ship.engine", "resume_ship"),
}

# CLI dest → engine param renames (the CLI keeps short ergonomic flag names).
CLI_RENAMES: dict[str, dict[str, str]] = {
    "report": {"session": "session_id", "label": "period_label_override"},
    "standup": {"session": "session_id"},
    "review run": {"session": "session_id"},
    # --transcript/--date carry explicit dest= in cli.py, so only --session
    # needs a rename here.
    "standup-review": {"session": "session_id"},
    "perf prep": {"session": "session_id"},
    "perf complete": {"session": "session_id"},
    "perf review": {"session": "session_id", "months": "period_months"},
    "ship run": {"session": "session_id", "check": "check_command"},
    "ship resume": {"check": "check_command"},
    # --repo is the repository path the Agents reports scope to (exact-or-prefix
    # on the session's project directory, never a basename substring).
    "agents cost": {"repo": "project_path"},
    "analyze": {
        # NOT a project link: analysis's --project is the tracker key (Jira/AzDO).
        "project": "project_key",
        "sprints": "sprint_count",
        "depth": "analysis_depth",
        "window_days": "analysis_window_days",
        "features": "analysis_features",
        "samples": "generate_samples",
        "no_insights": "include_insights",  # inverted store_true flag
    },
}

# CLI dests with no engine counterpart — output/dispatch concerns.
CLI_ONLY_DESTS: dict[str, set[str]] = {
    # source/code_sources/documentation_sources are assembled into the engine's
    # `sources` dict (component → source list), mirroring analyze's components flags.
    "report": {"format", "strict", "source", "code_sources", "documentation_sources"},
    "standup": {
        "format",
        "strict",
        "schedule",
        "list_members",
        "set_context",
    },  # schedule/list-members/set-context are adapters, not run_standup params
    # file-issues drives the separate file_transcript_issues entry point;
    # list-gaps is a store read. `paths` is the bare positional form of
    # --transcript ("yeaboi standup-review meeting.vtt", and what a dragged file
    # produces) — the handler folds it into transcript_paths, and a lone "-" into
    # transcript_text.
    "standup-review": {"format", "strict", "file_issues", "list_gaps", "paths"},
    # --mark ID=STATUS pairs are assembled into the engine's carried_statuses dict.
    "review run": {"format", "strict", "mark"},
    "perf prep": {"strict"},
    "perf complete": {"strict"},
    "perf review": {"strict"},
    "agents cost": {"format", "strict"},
    # The dismissal verbs edit a hand-kept allowlist instead of running the scan.
    "agents security": {
        "format",
        "strict",
        "dismiss",
        "reason",
        "undismiss",
        "list_dismissed",
        "replay",
        "line",
        "signals",
        "fix",
        "fix_id",
        "repo",
        "mark_test_data",
        "list_fixes",
    },
    # --split picks the entry point (run_ship_batch) rather than a run_ship param.
    "ship run": {"format", "strict", "split"},
    "ship resume": {"format", "strict"},
    # delivery/code/docs/ops are assembled into the engine's `components` dict (component
    # → sub-source map); each flag names a component's sub-sources, not an engine param.
    "analyze": {
        "format",
        "strict",
        "delivery",
        "code",
        "docs",
        "ops",
        "github_owner",
        "azdo_code_project",
        "confluence_space",
        "notion_root",
    },
}

# Engine params deliberately without a CLI flag. Reasoned; staleness-checked.
CLI_HIDDEN: dict[str, dict[str, str]] = {
    "review run": {
        "carried_statuses": "assembled from repeated --mark ID=STATUS flags; a raw dict flag invites typos",
    },
    "report": {
        "cancel_event": "in-process threading.Event cancel seam for the TUI worker; the CLI cancels via Ctrl-C",
        "sources": "assembled from the --source/--code-sources/--documentation-sources flags",
    },
    "analyze": {
        "progress": "live shared-list progress feed for the TUI frame loop — the CLI prints a banner instead",
        "team_name": "AzDO team label; auto-resolved from the configured AZURE_DEVOPS_TEAM",
        "components": "assembled from per-component --delivery/--code/--docs sub-source flags",
        "analysis_scope": "assembled from the four provider-specific scope flags",
        "cancel_event": "in-process threading.Event cancel seam for the TUI worker; the CLI cancels via Ctrl-C",
    },
    "ship run": {
        "cancel_event": "in-process threading.Event cancel seam for the TUI worker; the CLI cancels via Ctrl-C",
        "driver": "AgentDriver injection seam for tests; every wire surface runs the real Claude Code driver",
        "base_ref": "batch plumbing — run_ship_batch stacks each story on the branch before it",
        "pr_base": "batch plumbing — the stacked PR's target branch, set by run_ship_batch",
        "batch_id": "batch bookkeeping stamped by run_ship_batch; --split is how a user asks for one",
        "batch_item_id": "batch bookkeeping stamped by run_ship_batch — the epic its members came from",
        "batch_index": "batch bookkeeping stamped by run_ship_batch",
        "batch_total": "batch bookkeeping stamped by run_ship_batch",
    },
    "ship resume": {
        "cancel_event": "in-process threading.Event cancel seam for the TUI worker; the CLI cancels via Ctrl-C",
        "driver": "AgentDriver injection seam for tests; every wire surface runs the real Claude Code driver",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _non_exempt(field: str) -> dict[str, object]:
    """capability → value for every capability whose *field* is not Exempt."""
    return {cap: row[field] for cap, row in CAPABILITIES.items() if not isinstance(row[field], Exempt)}


def _module_to_path(module: str) -> pathlib.Path:
    if module in EXTRA_ENGINE_MODULES:
        return EXTRA_ENGINE_MODULES[module]
    return SRC.parent / pathlib.Path(module.replace(".", "/")).with_suffix(".py")


def _public_defs(path: pathlib.Path) -> set[str]:
    """Top-level public function names in *path* via AST — no import, no side effects."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_")
    }


def _mcp_app():
    """The real FastMCP app (skips when the [mcp] extra isn't installed)."""
    pytest.importorskip("mcp", reason="mcp extra not installed")
    from yeaboi.mcp.server import create_app

    app = create_app()
    assert hasattr(app, "_tool_manager"), (
        "the mcp SDK renamed FastMCP._tool_manager — update the introspection in test_surface_parity.py"
    )
    return app


def _tool_params(app, name: str) -> set[str]:
    """The client-visible parameter names of an MCP tool (ctx already excluded)."""
    tool = app._tool_manager.get_tool(name)
    return set(tool.parameters.get("properties", {}))


def _engine_params(module: str, fn: str) -> set[str]:
    import importlib
    import inspect

    return set(inspect.signature(getattr(importlib.import_module(module), fn)).parameters)


# ---------------------------------------------------------------------------
# Registry hygiene
# ---------------------------------------------------------------------------


class TestRegistryHygiene:
    def test_every_row_has_all_surfaces(self):
        required = {"engines", "mcp_tools", "tui_mode", "cli", "skill", "desktop"}
        for cap, row in CAPABILITIES.items():
            assert set(row) == required, f"capability {cap!r} must declare exactly the surfaces {sorted(required)}"

    def test_exempt_reasons_are_meaningful(self):
        for cap, row in CAPABILITIES.items():
            for field, value in row.items():
                if isinstance(value, Exempt):
                    assert len(value.reason) > 10, f"{cap}.{field}: Exempt needs a real reason, got {value.reason!r}"

    def test_hidden_params_have_reasons(self):
        for tool, params in HIDDEN_PARAMS.items():
            assert tool in PARAM_PAIRS, f"HIDDEN_PARAMS names unknown tool {tool!r}"
            for param, reason in params.items():
                assert len(reason) > 10, f"{tool}.{param}: hidden param needs a real reason"


# ---------------------------------------------------------------------------
# 1. Engine discovery — every engine module + entry point is registered
# ---------------------------------------------------------------------------


class TestEngines:
    def test_engine_modules_registered(self):
        discovered = {f"yeaboi.{p.parent.name}.engine" for p in SRC.glob("*/engine.py")}
        discovered |= set(EXTRA_ENGINE_MODULES)
        registered = {mod for entries in _non_exempt("engines").values() for mod, _fn in entries}
        assert discovered == registered, (
            f"engine modules on disk vs registered in CAPABILITIES differ.\n"
            f"  unregistered new engines: {sorted(discovered - registered)}\n"
            f"  registered but missing on disk: {sorted(registered - discovered)}\n{_HOW_TO}"
        )

    def test_engine_entry_points_registered(self):
        registered_by_module: dict[str, set[str]] = {}
        for entries in _non_exempt("engines").values():
            for mod, fn in entries:
                registered_by_module.setdefault(mod, set()).add(fn)
        for mod, registered_fns in registered_by_module.items():
            public = _public_defs(_module_to_path(mod))
            assert public == registered_fns, (
                f"public entry points of {mod} vs CAPABILITIES differ.\n"
                f"  new unregistered functions: {sorted(public - registered_fns)}\n"
                f"  registered but gone: {sorted(registered_fns - public)}\n{_HOW_TO}"
            )


# ---------------------------------------------------------------------------
# 2. MCP tool inventory
# ---------------------------------------------------------------------------


class TestMcpTools:
    def test_tool_inventory_registered(self):
        app = _mcp_app()
        actual = {t.name for t in app._tool_manager.list_tools()}
        registered = {name for names in _non_exempt("mcp_tools").values() for name in names}
        assert actual == registered, (
            f"MCP server tools vs CAPABILITIES differ.\n"
            f"  new unregistered tools: {sorted(actual - registered)}\n"
            f"  registered but not on the server: {sorted(registered - actual)}\n{_HOW_TO}"
        )


# ---------------------------------------------------------------------------
# 3. TUI mode cards
# ---------------------------------------------------------------------------


class TestTuiModes:
    def test_mode_cards_registered(self):
        # The union of both category menus — Solo (_SOLO_MENU_CARDS, which holds
        # the Agents family) and Team (_MODE_CARDS) — must equal the registered
        # tui_mode column.
        from yeaboi.ui.mode_select.screens._screens import _MODE_CARDS, _SOLO_MENU_CARDS

        actual = {card["key"] for card in (*_SOLO_MENU_CARDS, *_MODE_CARDS)}
        registered = set(_non_exempt("tui_mode").values())
        assert actual == registered, (
            f"_SOLO_MENU_CARDS/_MODE_CARDS keys vs CAPABILITIES differ.\n"
            f"  new unregistered cards: {sorted(actual - registered)}\n"
            f"  registered but card removed: {sorted(registered - actual)}\n{_HOW_TO}"
        )

    def test_solo_and_team_menus_differ_by_exactly_the_world_only_modes(self):
        # Every shared card keeps one key (dispatch, hubs, tips and this
        # registry all key on it). The difference is exactly the modes each
        # world owns: Team's three room-shaped modes, and Solo's Weekly Review
        # plus the Agents family — a self-review has no roster to review, and a
        # solo engineer is who watches their own agents.
        from yeaboi.ui.mode_select.screens._screens import _MODE_CARDS, _SOLO_MENU_CARDS

        solo = {card["key"] for card in _SOLO_MENU_CARDS}
        team = {card["key"] for card in _MODE_CARDS}
        assert solo - team == {"weekly-review", "agent-usage", "agent-advisor", "agent-security"}, sorted(solo - team)
        assert team - solo == {"retro", "poker", "performance"}

    def test_settings_is_the_last_card_on_both_menus(self):
        # The Agents family is spliced into the Solo menu, and Settings is last
        # on every menu — the splice must never put a card after it.
        from yeaboi.ui.mode_select.screens._screens import _MODE_CARDS, _SOLO_MENU_CARDS

        assert _SOLO_MENU_CARDS[-1]["key"] == "settings"
        assert _MODE_CARDS[-1]["key"] == "settings"


# ---------------------------------------------------------------------------
# 3b. Discoverability tips — every capability surfaces a rotating tip, so tips
#     stay current as features land. Model: the same two-way set-equality as the
#     mode-card check above, once per surface.
#
#     Once per surface is the point: a tip names a gesture as often as it names a
#     feature, and a keycap is not a thing the desktop app has. Each tip carries
#     the surfaces it is true on (untagged means all of them), and each surface is
#     checked against the capabilities that actually reached it.
# ---------------------------------------------------------------------------

# Capabilities that deliberately have no tip on a surface they do reach. Both
# empty today — every capability is worth surfacing. Add a
# `key: "reason (>10 chars)"` entry to opt one out.
TIP_EXEMPT: dict[str, str] = {}
DESKTOP_TIP_EXEMPT: dict[str, str] = {}

_TIP_HOW_TO = (
    "Fix: add a FeatureTip for this capability in src/yeaboi/ui/shared/_tips.py "
    "(_FEATURE_TIPS) — tagged with the surface, or untagged if it is true on both — "
    "or record a TIP_EXEMPT / DESKTOP_TIP_EXEMPT entry in tests/unit/test_surface_parity.py."
)

# Terminal gestures that must never appear in a tip the desktop app is served.
# This is the guard that catches the real regression: a new tip written at a
# terminal and tagged for both surfaces without anyone opening the window.
_TERMINAL_GESTURES = ("--", "press ", "Ctrl+", "`yeaboi ", "double-tap")


class TestTips:
    def test_every_capability_has_a_tui_tip(self):
        from yeaboi.ui.shared._tips import _FEATURE_TIPS

        actual = {t.key for t in _FEATURE_TIPS if "tui" in t.surfaces}
        registered = set(CAPABILITIES) - set(TIP_EXEMPT)
        assert actual == registered, (
            f"terminal feature tips vs CAPABILITIES differ.\n"
            f"  capabilities with no tip: {sorted(registered - actual)}\n"
            f"  tips for an unknown/exempt capability: {sorted(actual - registered)}\n{_TIP_HOW_TO}"
        )

    def test_every_desktop_capability_has_a_desktop_tip(self):
        # The desktop column is itself checked against contracts/v1/routes_manifest.json
        # further down, so this reads "the capabilities the app actually ships".
        from yeaboi.ui.shared._tips import _FEATURE_TIPS

        actual = {t.key for t in _FEATURE_TIPS if "desktop" in t.surfaces}
        registered = set(_non_exempt("desktop")) - set(DESKTOP_TIP_EXEMPT)
        assert actual == registered, (
            f"desktop feature tips vs the desktop capabilities differ.\n"
            f"  capabilities the app has but no tip mentions: {sorted(registered - actual)}\n"
            f"  tips for something the app does not have: {sorted(actual - registered)}\n{_TIP_HOW_TO}"
        )

    def test_tip_exempt_reasons_are_meaningful(self):
        for name, exempt in (("TIP_EXEMPT", TIP_EXEMPT), ("DESKTOP_TIP_EXEMPT", DESKTOP_TIP_EXEMPT)):
            for cap, reason in exempt.items():
                assert cap in CAPABILITIES, f"{name} names unknown capability {cap!r}"
                assert len(reason) > 10, f"{name}[{cap!r}] needs a real reason, got {reason!r}"

    def test_every_tip_names_valid_surfaces(self):
        from yeaboi.surfaces import VALID_SURFACES
        from yeaboi.ui.shared._tips import get_tips

        for tip in get_tips():
            assert tip.surfaces, f"tip {tip.key!r} is tagged for no surface at all"
            unknown = set(tip.surfaces) - VALID_SURFACES
            assert not unknown, f"tip {tip.key!r} names unknown surface(s) {sorted(unknown)}"

    @pytest.mark.parametrize("voice", ["ready", "installable", "declined", "unsupported"])
    def test_desktop_tips_name_no_terminal_gesture(self, monkeypatch, voice):
        # Driven over every voice state: the voice tip is the one built at run
        # time, so leaving it on this machine's answer tests one branch of four.
        from yeaboi.ui.shared import _tips

        monkeypatch.setattr("yeaboi.voice.voice_state", lambda: voice)
        monkeypatch.setattr("yeaboi.voice.unsupported_blocker", lambda: "no libportaudio2")
        _tips.get_tips.cache_clear()
        for tip in _tips.tips_for_surface("desktop"):
            found = [g for g in _TERMINAL_GESTURES if g in tip.text]
            assert not found, (
                f"desktop tip {tip.key!r} names a terminal-only gesture {found}:\n"
                f"  {tip.text}\n"
                'Fix: reword it for the window, or tag this one surfaces=("tui",) and add a '
                "desktop sibling under the same key."
            )
        _tips.get_tips.cache_clear()

    def test_desktop_only_tips_name_a_real_route(self):
        # `desktop:<slug>` names `/slug` in the app's own route manifest. These
        # tips answer to no capability, so without this a renamed route leaves a
        # tip pointing at a page that no longer exists.
        import json

        from yeaboi.ui.shared._tips import _DESKTOP_TIPS

        paths = {r["path"] for r in json.loads(DESKTOP_MANIFEST.read_text(encoding="utf-8"))["routes"]}
        for tip in _DESKTOP_TIPS:
            prefix, _, slug = tip.key.partition(":")
            assert prefix == "desktop" and slug, f"desktop-only tip {tip.key!r} must be keyed `desktop:<slug>`"
            assert f"/{slug}" in paths, (
                f"tip {tip.key!r} names route /{slug}, which is not in contracts/v1/routes_manifest.json.\n"
                "Fix: rename the tip's key to the route it points at, or drop the tip."
            )
            assert tip.surfaces == ("desktop",), f"{tip.key!r} is desktop furniture and must be tagged so"

    def test_carded_capabilities_have_jump_targets(self):
        # Every capability that owns a mode card must have a tip whose mode_key
        # points at that exact card, so the jump-into-feature key can't rot.
        # Cards span every category menu (the `g` jump switches category).
        from yeaboi.ui.mode_select.screens._screens import _MODE_CARDS, _SOLO_MENU_CARDS
        from yeaboi.ui.shared._tips import _FEATURE_TIPS

        card_keys = {card["key"] for card in (*_SOLO_MENU_CARDS, *_MODE_CARDS)}
        for cap, tui_mode in _non_exempt("tui_mode").items():
            # "some tip", not "the tip": several capabilities carry more than one.
            tips = [t for t in _FEATURE_TIPS if t.key == cap and "tui" in t.surfaces]
            assert any(t.mode_key == tui_mode for t in tips), (
                f"capability {cap!r} has mode card {tui_mode!r} but no terminal tip points at it "
                f"(got {sorted({t.mode_key for t in tips})}) — jump-into-feature would miss.\n{_TIP_HOW_TO}"
            )
        # No tip may point at a non-existent card.
        for tip in _FEATURE_TIPS:
            assert tip.mode_key is None or tip.mode_key in card_keys, (
                f"tip {tip.key!r} jumps to unknown card {tip.mode_key!r}\n{_TIP_HOW_TO}"
            )


# ---------------------------------------------------------------------------
# 3c. Saved sessions — every mode card that records runs lands on a hub the
#     user can reopen finished work from. Model: the same two-way set-equality
#     as the tip check above, keyed on cards rather than capabilities.
# ---------------------------------------------------------------------------

# Cards that deliberately have no saved-sessions hub. A reason is required and is
# length-checked, like Exempt — "no time" must not pass for "nothing to save".
SAVED_SESSIONS_EXEMPT: dict[str, str] = {
    "team-analysis": "saved analyses ARE the card's landing list (_build_project_list_screen)",
    "project-planning": "saved plans ARE the card's landing list (load_projects unions projects.json "
    "and the session store's planning rows)",
    "performance": "artifacts are per-engineer — the hub opens from the roster's History action",
    "usage": "a live dashboard over the usage DB; every view already spans the whole history",
    "settings": "a live config editor — there is no completed session to re-open",
    "agent-usage": "opens instantly on the last saved report (AgentWatchStore.latest_report); "
    "list_reports exists and a browsable hub is queued follow-up work",
    "agent-advisor": "opens instantly on the last saved report (AgentWatchStore.latest_report); "
    "list_reports exists and a browsable hub is queued follow-up work",
    "agent-security": "opens instantly on the last saved report (AgentWatchStore.latest_report); "
    "list_reports exists and a browsable hub is queued follow-up work",
}

_SAVED_SESSIONS_HOW_TO = (
    "Fix: give the card a saved-sessions hub — wire its dispatch branch through "
    "_run_mode_hub(...) in src/yeaboi/ui/mode_select/__init__.py and register it in "
    "SAVED_SESSION_HUBS there — or record a SAVED_SESSIONS_EXEMPT entry in "
    "tests/unit/test_surface_parity.py saying why this card has nothing to re-open."
)


class TestSavedSessions:
    """A finished run the user cannot get back to is a run they lost.

    Five of the mode cards grew a saved-runs hub organically and Ship shipped
    without one, which is exactly the failure mode a registry check exists for:
    the requirement lived only in the habit of copying a neighbouring mode.
    """

    def _card_keys(self) -> set[str]:
        from yeaboi.ui.mode_select.screens._screens import _AGENT_CARDS, _MODE_CARDS, _SOLO_CARDS

        return {card["key"] for card in (*_SOLO_CARDS, *_MODE_CARDS, *_AGENT_CARDS)}

    def test_every_card_lands_on_a_saved_sessions_hub(self):
        from yeaboi.ui.mode_select import SAVED_SESSION_HUBS

        expected = self._card_keys() - set(SAVED_SESSIONS_EXEMPT)
        actual = set(SAVED_SESSION_HUBS)
        assert actual == expected, (
            f"saved-sessions hubs vs mode cards differ.\n"
            f"  cards with no hub: {sorted(expected - actual)}\n"
            f"  hubs for an unknown/exempt card: {sorted(actual - expected)}\n{_SAVED_SESSIONS_HOW_TO}"
        )

    def test_registered_hubs_are_callable(self):
        from yeaboi.ui.mode_select import SAVED_SESSION_HUBS

        for key, hub in SAVED_SESSION_HUBS.items():
            assert callable(hub), f"SAVED_SESSION_HUBS[{key!r}] is not callable"

    def test_exempt_reasons_are_meaningful(self):
        card_keys = self._card_keys()
        for key, reason in SAVED_SESSIONS_EXEMPT.items():
            assert key in card_keys, f"SAVED_SESSIONS_EXEMPT names unknown mode card {key!r}"
            assert len(reason) > 10, f"SAVED_SESSIONS_EXEMPT[{key!r}] needs a real reason, got {reason!r}"


# ---------------------------------------------------------------------------
# 4. CLI flags — presence check (argparse can't tell us which new flag is
#    "a capability", so discovery in the reverse direction rides on the
#    engine/TUI/MCP checks) + the --mode ⊆ _MODE_CARDS drift guard.
# ---------------------------------------------------------------------------


class TestCli:
    def test_registered_flags_exist(self):
        from yeaboi.cli import build_parser

        parser = build_parser()
        option_strings = {s for action in parser._actions for s in action.option_strings}
        subcommands = set()
        for action in parser._actions:
            if hasattr(action, "choices") and action.choices and not action.option_strings:
                subcommands |= set(action.choices)  # subparsers action
        available = option_strings | subcommands
        for cap, flags in _non_exempt("cli").items():
            missing = set(flags) - available
            assert not missing, (
                f"capability {cap!r} registers CLI entries the parser doesn't define: {sorted(missing)}\n{_HOW_TO}"
            )

    def test_mode_choices_subset_of_mode_cards(self):
        from yeaboi.cli import build_parser
        from yeaboi.ui.mode_select.screens._screens import _MODE_CARDS

        parser = build_parser()
        mode_action = next(a for a in parser._actions if "--mode" in a.option_strings)
        card_keys = {card["key"] for card in _MODE_CARDS}
        drift = set(mode_action.choices) - card_keys
        assert not drift, f"--mode offers choices with no _MODE_CARDS entry: {sorted(drift)} — cli.py drifted"


# ---------------------------------------------------------------------------
# 5. Plugin skills
# ---------------------------------------------------------------------------


class TestPluginSkills:
    def test_skill_dirs_registered(self):
        actual = {p.parent.name for p in PLUGIN_SKILLS_DIR.glob("*/SKILL.md")}
        registered = set(_non_exempt("skill").values())
        assert actual == registered, (
            f"claude-plugin skills vs CAPABILITIES differ.\n"
            f"  new unregistered skills: {sorted(actual - registered)}\n"
            f"  registered but no SKILL.md: {sorted(registered - actual)}\n{_HOW_TO}"
        )

    def test_skills_mention_their_capability_tools(self):
        for cap, row in CAPABILITIES.items():
            if isinstance(row["skill"], Exempt) or isinstance(row["mcp_tools"], Exempt):
                continue
            body = (PLUGIN_SKILLS_DIR / row["skill"] / "SKILL.md").read_text(encoding="utf-8")
            referenced = set(re.findall(r"`([a-z][a-z0-9_]+)`", body))
            assert referenced & row["mcp_tools"], (
                f"skill {row['skill']!r} never mentions any of capability {cap!r}'s MCP tools "
                f"{sorted(row['mcp_tools'])} — the skill can't be driving this capability"
            )


# ---------------------------------------------------------------------------
# 6. Desktop — renderer routes/actions vs the committed manifest.
#
# The desktop renderer's route registry lives in yeaboi-desktop and is
# code-generated there into contracts/v1/routes_manifest.json, which is
# committed HERE and vendored back — so this Python suite never needs Node, and
# that repo's `make check-manifest` is red whenever the two disagree. That is
# the manifest == registries half; here we assert manifest == CAPABILITIES,
# two-way. Entries prefixed "action:" / "dialog:" are non-route affordances
# (result-screen buttons, dialogs).
# ---------------------------------------------------------------------------

DESKTOP_MANIFEST = REPO_ROOT / "contracts" / "v1" / "routes_manifest.json"


class TestDesktop:
    def _manifest_entries(self) -> list[dict]:
        import json

        data = json.loads(DESKTOP_MANIFEST.read_text(encoding="utf-8"))
        # v2 grew the settings tabs and the chat's command registry, both
        # checked in test_tui_parity.py — the routes below are unchanged.
        assert data.get("schema_version") == 2, "routes_manifest.json schema_version must be 2"
        return data["routes"]

    def test_manifest_entries_are_well_formed(self):
        for entry in self._manifest_entries():
            assert set(entry) == {"path", "capability", "title"}, f"malformed manifest entry: {entry}"
            assert entry["path"].startswith(("/", "action:", "dialog:")), entry["path"]
            assert entry["title"], f"manifest entry {entry['path']!r} needs a title"

    def test_routes_registered(self):
        actual = {entry["path"] for entry in self._manifest_entries() if entry["capability"]}
        registered = {path for paths in _non_exempt("desktop").values() for path in paths}
        assert actual == registered, (
            f"desktop manifest vs CAPABILITIES differ.\n"
            f"  in manifest but unregistered: {sorted(actual - registered)}\n"
            f"  registered but not in manifest: {sorted(registered - actual)}\n{_HOW_TO}"
        )

    def test_route_capabilities_match(self):
        by_cap = _non_exempt("desktop")
        for entry in self._manifest_entries():
            cap = entry["capability"]
            if cap is None:
                continue  # pure chrome (home, what's-new) — owned by no capability
            assert cap in CAPABILITIES, f"manifest route {entry['path']!r} names unknown capability {cap!r}"
            assert entry["path"] in by_cap.get(cap, set()), (
                f"manifest route {entry['path']!r} claims capability {cap!r}, "
                f"but that row's desktop column does not list it\n{_HOW_TO}"
            )


# ---------------------------------------------------------------------------
# 7. Param parity — the engine's keyword surface must reach the MCP tool
# ---------------------------------------------------------------------------


class TestParamParity:
    def test_every_engine_backed_tool_is_paired(self):
        registered_tools = {name for names in _non_exempt("mcp_tools").values() for name in names}
        unknown = set(PARAM_PAIRS) - registered_tools
        assert not unknown, f"PARAM_PAIRS names tools not in CAPABILITIES: {sorted(unknown)}"

    def test_engine_params_reach_the_tool(self):
        app = _mcp_app()
        problems: list[str] = []
        for tool_name, (mod, fn) in PARAM_PAIRS.items():
            tool_params = _tool_params(app, tool_name)
            engine_params = _engine_params(mod, fn)
            hidden = set(HIDDEN_PARAMS.get(tool_name, {}))
            unexposed = engine_params - tool_params - HIDDEN_ALWAYS - hidden
            if unexposed:
                problems.append(
                    f"{tool_name}: engine {mod}.{fn} grew params the MCP tool doesn't expose: "
                    f"{sorted(unexposed)} — expose them in src/yeaboi/mcp/tools_*.py or add "
                    f"them to HIDDEN_PARAMS with a reason"
                )
        assert not problems, "\n".join(problems) + f"\n{_HOW_TO}"

    def test_tool_params_map_to_the_engine(self):
        app = _mcp_app()
        problems: list[str] = []
        for tool_name, (mod, fn) in PARAM_PAIRS.items():
            tool_params = _tool_params(app, tool_name)
            engine_params = _engine_params(mod, fn)
            phantom = tool_params - engine_params - TOOL_ONLY_PARAMS.get(tool_name, set())
            if phantom:
                problems.append(
                    f"{tool_name}: tool params with no engine counterpart (typo'd rename?): {sorted(phantom)}"
                )
        assert not problems, "\n".join(problems) + f"\n{_HOW_TO}"

    def test_hidden_params_still_exist_on_engines(self):
        for tool_name, hidden in HIDDEN_PARAMS.items():
            mod, fn = PARAM_PAIRS[tool_name]
            engine_params = _engine_params(mod, fn)
            stale = set(hidden) - engine_params
            assert not stale, (
                f"{tool_name}: HIDDEN_PARAMS lists params the engine {mod}.{fn} no longer has: "
                f"{sorted(stale)} — delete the stale exemptions"
            )

    def test_tool_only_params_are_real(self):
        app = _mcp_app()
        for tool_name, extras in TOOL_ONLY_PARAMS.items():
            tool_params = _tool_params(app, tool_name)
            stale = extras - tool_params
            assert not stale, f"{tool_name}: TOOL_ONLY_PARAMS lists params the tool doesn't have: {sorted(stale)}"


# ---------------------------------------------------------------------------
# 7. Param parity — the engine's keyword surface must reach the CLI subcommand
# ---------------------------------------------------------------------------


def _cli_subparser(path: str):
    """Resolve 'report' or 'perf prep' to its argparse sub-parser."""
    import argparse

    from yeaboi.cli import build_parser

    p = build_parser()
    for part in path.split():
        action = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
        p = action.choices[part]
    return p


def _cli_dests(path: str) -> set[str]:
    """The argument dests a subcommand defines (its own flags + positionals)."""
    import argparse

    return {
        a.dest
        for a in _cli_subparser(path)._actions
        if a.dest != "help" and not isinstance(a, argparse._SubParsersAction)
    }


class TestCliParamParity:
    def test_engine_params_reach_the_cli(self):
        problems: list[str] = []
        for path, (mod, fn) in CLI_PARAM_PAIRS.items():
            renames = CLI_RENAMES.get(path, {})
            mapped = {renames.get(d, d) for d in _cli_dests(path) - CLI_ONLY_DESTS[path]}
            engine_params = _engine_params(mod, fn)
            hidden = set(CLI_HIDDEN.get(path, {}))
            unexposed = engine_params - mapped - HIDDEN_ALWAYS - hidden
            if unexposed:
                problems.append(
                    f"yeaboi {path}: engine {mod}.{fn} has params the CLI doesn't expose: {sorted(unexposed)} — "
                    f"add flags in cli.py build_parser() or a CLI_HIDDEN entry with a reason"
                )
        assert not problems, "\n".join(problems) + f"\n{_HOW_TO}"

    def test_cli_dests_map_to_the_engine(self):
        problems: list[str] = []
        for path, (mod, fn) in CLI_PARAM_PAIRS.items():
            renames = CLI_RENAMES.get(path, {})
            mapped = {renames.get(d, d) for d in _cli_dests(path) - CLI_ONLY_DESTS[path]}
            phantom = mapped - _engine_params(mod, fn)
            if phantom:
                problems.append(
                    f"yeaboi {path}: CLI args with no engine counterpart (typo'd rename?): {sorted(phantom)}"
                )
        assert not problems, "\n".join(problems) + f"\n{_HOW_TO}"

    def test_cli_registry_entries_are_real(self):
        """Renames/CLI-only/hidden entries must not go stale as flags evolve."""
        assert set(CLI_ONLY_DESTS) == set(CLI_PARAM_PAIRS), "CLI_ONLY_DESTS must cover exactly CLI_PARAM_PAIRS"
        for path in CLI_PARAM_PAIRS:
            dests = _cli_dests(path)
            stale_renames = set(CLI_RENAMES.get(path, {})) - dests
            assert not stale_renames, (
                f"yeaboi {path}: CLI_RENAMES names dests that don't exist: {sorted(stale_renames)}"
            )
            stale_only = CLI_ONLY_DESTS[path] - dests
            assert not stale_only, f"yeaboi {path}: CLI_ONLY_DESTS names dests that don't exist: {sorted(stale_only)}"

    def test_cli_hidden_params_still_exist_on_engines(self):
        for path, hidden in CLI_HIDDEN.items():
            mod, fn = CLI_PARAM_PAIRS[path]
            stale = set(hidden) - _engine_params(mod, fn)
            assert not stale, (
                f"yeaboi {path}: CLI_HIDDEN lists params the engine {mod}.{fn} no longer has: "
                f"{sorted(stale)} — delete the stale exemptions"
            )
            for param, reason in hidden.items():
                assert len(reason) > 10, f"{path}.{param}: hidden param needs a real reason"
