#!/usr/bin/env python3
"""Decide which tests and which CI jobs a set of changed files actually needs.

Every one of the `ci.yml` jobs ran on every push, and the two test jobs
were the whole pipeline — 310s of unit tests and 408s of "integration" that was
the same 11,000 unit tests plus 572 more. A one-line docs edit rebuilt the
front-end bundles and ran the suite twice.

This maps changed paths onto **areas** and prints the pytest paths and job
flags for that change. `ci.yml` runs it once in a cheap job and feeds the rest.

Three rules carry the whole design, and the last two are the ones that make it
safe rather than merely fast:

* **An area claims its sources and their tests together.** The registry below
  is that claim in a form a program can act on.
* **`ALWAYS` runs whatever changed.** A third of the guard suite scans the repo
  rather than importing a module — surface parity, the workflow schema, the
  committed web bundles. Nothing imports them, so no
  dependency-derived selection can reach them, and a naive "changed X, run
  test_X" would drop every one.
* **Anything unrecognised runs everything.** A path this file has never heard of
  is a path whose blast radius is unknown, and the fail-safe direction for a
  test selector is to over-run. `tests/unit/test_test_scope.py` additionally
  fails the build when a source or test file is claimed by nothing, so "unknown"
  stays a transient state between a rename and its registry entry, rather than a
  quiet hole that grows.

Usage::

    git diff --name-only origin/main... | python3 scripts/test_scope.py --changed-files - --unit-paths
    python3 scripts/test_scope.py --working-tree --explain
    python3 scripts/test_scope.py --changed-files - --github-output
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Area:
    """One area's source paths and the tests that cover them.

    ``src`` entries are path prefixes (a trailing ``/`` means a directory) and
    ``tests`` entries are globs relative to the repo root. Both are matched
    against forward-slash paths as git prints them.
    """

    name: str
    src: tuple[str, ...]
    tests: tuple[str, ...] = ()
    # Areas whose tests must run too. Not a general dependency graph — just the
    # handful of one-way couplings where one area's code is the other's fixture.
    also: tuple[str, ...] = ()


# --- the areas ----------------------------------------------------------------
# `tests/unit/test_test_scope.py` asserts every source and test file is claimed,
# so an area added or renamed here fails there until the registry follows.
AREAS: tuple[Area, ...] = (
    Area(
        # The seam to yeaboi-tooling: the pinned sha, the copied bootstrap, and
        # what a fresh worktree of this repo installs. No `tests` of its own —
        # the shared halves are tested in the tooling repo, and what this repo
        # has to hold up (the Make targets, .claude/repo-notes.md, the pin being
        # a sha) is `tests/unit/test_ship_gate.py`, which is in ALWAYS. CI's
        # `make tooling-check` covers the rest at runtime.
        "tooling",
        src=(".tooling-rev", "scripts/tooling-sync.sh", "scripts/provision.sh", ".githooks/"),
    ),
    Area(
        # The seam to yeaboi-site: the facts the website advertises about this
        # package, derived from pyproject and vendored over there by sha. No
        # `tests` — tests/unit/test_site_contract.py is in ALWAYS, because the
        # fact that goes stale most easily (`requires-python`) lives in
        # pyproject.toml, which no site path implies.
        "site-contract",
        src=("contracts/site.json", "scripts/gen_site_contract.py"),
    ),
    Area(
        # The cross-mode recent-sessions list /api/sessions/recent reads
        # through, and the composer's @ picker beside it.
        "sessions",
        src=(
            "src/yeaboi/sessions_recent.py",
            "src/yeaboi/references.py",
            "src/yeaboi/app/routes_sessions.py",
        ),
        tests=(
            "tests/unit/test_mcp_server.py",
            "tests/unit/test_sessions_recent.py",
            # Planning's saved-runs list rides projects.json.
            "tests/unit/test_persistence_projects.py",
        ),
    ),
    Area(
        # What a run may read from other sessions: the scope grammar, the
        # sprint calendar, the label store and the resolver every mode calls.
        "context",
        src=(
            "src/yeaboi/context/",
            "src/yeaboi/app/routes_context.py",
            "src/yeaboi/mcp/tools_context.py",
            "src/yeaboi/ui/mode_select/_context.py",
            "src/yeaboi/ui/mode_select/screens/_screens_context.py",
        ),
        tests=("tests/unit/test_context_*.py", "tests/unit/test_mcp_server.py"),
    ),
    Area(
        # The Solo world's own modules: the welcome's Today snapshot and the
        # desktop route that serves it. The engines it reads stay in their areas.
        "solo",
        src=(
            "src/yeaboi/solo/",
            "src/yeaboi/app/routes_solo.py",
            "src/yeaboi/mcp/tools_solo.py",
            "src/yeaboi/prompts/weekly_review.py",
            "src/yeaboi/ui/mode_select/_solo.py",
            "src/yeaboi/ui/mode_select/screens/_screens_solo.py",
        ),
        tests=(
            "tests/unit/test_solo_*.py",
            "tests/unit/test_app_solo_routes.py",
            "tests/unit/prompts/test_weekly_review_prompt.py",
            "tests/unit/test_mcp_server.py",
        ),
    ),
    Area(
        "standup",
        src=("src/yeaboi/standup/", "src/yeaboi/mcp/tools_standup.py"),
        tests=("tests/unit/test_mcp_server.py", "tests/unit/test_standup_*.py", "tests/unit/prompts/test_standup_*.py"),
    ),
    Area(
        "retro",
        src=("src/yeaboi/retro/", "src/yeaboi/mcp/tools_retro.py"),
        tests=(
            "tests/unit/test_mcp_server.py",
            "tests/unit/test_retro_*.py",
        ),
    ),
    Area(
        "poker",
        src=("src/yeaboi/poker/", "src/yeaboi/mcp/tools_poker.py"),
        tests=(
            "tests/unit/test_mcp_server.py",
            "tests/unit/test_poker_*.py",
            "tests/unit/prompts/test_poker_prompt.py",
        ),
    ),
    Area(
        "reporting",
        src=("src/yeaboi/reporting/", "src/yeaboi/mcp/tools_reporting.py"),
        tests=(
            "tests/unit/test_mcp_server.py",
            "tests/unit/test_reporting_*.py",
        ),
    ),
    Area(
        "roadmap",
        src=("src/yeaboi/roadmap/",),
        tests=("tests/unit/test_roadmap_*.py", "tests/unit/prompts/test_roadmap_prompt.py"),
    ),
    Area(
        "performance",
        src=("src/yeaboi/performance/", "src/yeaboi/mcp/tools_performance.py", "src/yeaboi/beta.py"),
        tests=("tests/unit/test_mcp_server.py", "tests/unit/test_performance_*.py", "tests/unit/test_beta*.py"),
    ),
    Area(
        "analysis",
        src=(
            "src/yeaboi/analysis/",
            "src/yeaboi/team_profile.py",
            "src/yeaboi/team_profile_exporter.py",
            "src/yeaboi/team_roster.py",
            "src/yeaboi/tools/team_learning.py",
            "src/yeaboi/mcp/tools_team.py",
        ),
        tests=(
            "tests/unit/test_mcp_server.py",
            "tests/unit/test_analysis_*.py",
            "tests/unit/test_team_*.py",
            "tests/unit/test_doc_quality.py",
            "tests/unit/test_doc_scoring.py",
            "tests/unit/test_code_health.py",
            "tests/unit/test_coverage.py",
            "tests/unit/test_practices.py",
            "tests/unit/test_ai_usage.py",
            "tests/unit/test_repo_inventory.py",
            "tests/unit/test_prior_art.py",
        ),
    ),
    Area(
        "agents",
        src=(
            "src/yeaboi/agentwatch/",
            "src/yeaboi/pricing.py",
            "src/yeaboi/mcp/tools_agentwatch.py",
            "src/yeaboi/prompts/agentwatch.py",
            "src/yeaboi/ui/mode_select/_agents.py",
            "src/yeaboi/ui/mode_select/screens/_screens_agents.py",
            "src/yeaboi/ui/mode_select/screens/_screens_category.py",
        ),
        tests=(
            "tests/unit/test_mcp_server.py",
            "tests/unit/test_agentwatch_*.py",
            "tests/unit/test_pricing.py",
            "tests/unit/test_category_screen.py",
        ),
    ),
    Area(
        "planning",
        src=(
            "src/yeaboi/agent/",
            "src/yeaboi/prompts/",
            "src/yeaboi/ui/session/",
            # The plan's back half: a story from the sprint plan driven through
            # a supervised coding agent to a PR.
            "src/yeaboi/ship/",
            "src/yeaboi/questionnaire_io.py",
            # The issue-tracker registry every planning dispatch site reads,
            # and the tracker sync modules its entries resolve to.
            "src/yeaboi/trackers.py",
            "src/yeaboi/linear_sync.py",
            "src/yeaboi/trello_sync.py",
            "src/yeaboi/transcript.py",
            "src/yeaboi/json_exporter.py",
            "src/yeaboi/prd_exporter.py",
            "src/yeaboi/ollama_control.py",
            "src/yeaboi/mcp/tools_planning.py",
            "src/yeaboi/mcp/tools_sessions.py",
        ),
        tests=(
            "tests/unit/test_mcp_server.py",
            "tests/unit/nodes/*.py",
            "tests/unit/prompts/*.py",
            "tests/unit/test_trackers.py",
            "tests/unit/test_linear_sync.py",
            "tests/unit/test_trello_sync.py",
            "tests/unit/test_state.py",
            "tests/unit/test_sessions.py",
            "tests/unit/test_phases_*.py",
            "tests/unit/test_planning_*.py",
            "tests/unit/test_plan_*.py",
            "tests/unit/test_planner_*.py",
            "tests/unit/test_prior_art*.py",
            "tests/unit/test_chat_*.py",
            "tests/unit/test_headless_pipeline.py",
            "tests/unit/test_streaming.py",
            "tests/unit/test_llm.py",
            "tests/unit/test_token_budgets.py",
            "tests/unit/test_questionnaire_io.py",
            "tests/unit/test_transcript*.py",
            "tests/unit/test_json_exporter.py",
            "tests/unit/test_prd_exporter.py",
            "tests/unit/test_ollama_control.py",
            "tests/unit/test_duck_*.py",
            "tests/unit/test_session_handoff.py",
            # test_ship_gate.py (the Makefile / repo-notes guard) also matches
            # this glob; it lives in ALWAYS, and a double claim is harmless.
            "tests/unit/test_ship_*.py",
        ),
    ),
    Area(
        "integrations",
        src=(
            "src/yeaboi/tools/",
            "src/yeaboi/jira_sync.py",
            "src/yeaboi/azdevops_sync.py",
            "src/yeaboi/sync_naming.py",
            "src/yeaboi/export_targets.py",
            "src/yeaboi/exporting.py",
            "src/yeaboi/ticket_text.py",
            "src/yeaboi/markdown_convert.py",
        ),
        tests=(
            "tests/unit/tools/*.py",
            "tests/unit/test_jira_*.py",
            "tests/unit/test_azdevops_*.py",
            "tests/unit/test_sync_naming.py",
            "tests/unit/test_azure_devops.py",
            "tests/unit/test_export_targets.py",
            "tests/unit/test_exporting.py",
            "tests/unit/test_ticket_text.py",
            "tests/unit/test_markdown_convert.py",
            "tests/unit/test_local_git.py",
            "tests/unit/test_repo_tree_helpers.py",
        ),
    ),
    Area(
        "artifacts-sharing",
        src=("src/yeaboi/artifacts/", "src/yeaboi/anonymize/", "src/yeaboi/sharing/"),
        tests=(
            "tests/unit/test_artifacts_*.py",
            "tests/unit/test_sharing_*.py",
            "tests/unit/test_anonymize*.py",
            "tests/unit/test_output_share_*.py",
        ),
    ),
    Area(
        "security",
        src=(
            "src/yeaboi/input_guardrails.py",
            "src/yeaboi/output_guardrails.py",
            "src/yeaboi/fs_policy.py",
            "src/yeaboi/redaction.py",
            "src/yeaboi/claude_auth.py",
            "src/yeaboi/auth_state.py",
            "src/yeaboi/web/security.py",
            "src/yeaboi/sharing/access.py",
            "src/yeaboi/sharing/gate.py",
            # The Cloudflare Access tier: local JWT verification and the named
            # tunnel that carries it. `sharing/` as a whole belongs to
            # artifacts-sharing; these two are claimed by name here for the same
            # reason access.py and gate.py are — they are access control, and a
            # change to them must run the security lane.
            "src/yeaboi/sharing/identity.py",
            "src/yeaboi/sharing/access_tunnel.py",
            "src/yeaboi/sharing/access_setup.py",
        ),
        tests=(
            "tests/unit/guardrails/*.py",
            "tests/unit/test_fs_policy.py",
            "tests/unit/test_redaction.py",
            "tests/unit/test_claude_auth.py",
            "tests/unit/test_auth_state.py",
            "tests/unit/test_web_security.py",
            "tests/unit/test_export_xss.py",
            "tests/unit/test_consent.py",
            # The tunnel's blast radius: one loopback origin, an allowlisted
            # child environment. Named test_retro_* as well so it also runs
            # under the `retro` area, which owns retro/tunnel.py itself.
            "tests/unit/test_retro_tunnel_containment.py",
            "tests/unit/test_sharing_identity.py",
            "tests/unit/test_access_tunnel.py",
            "tests/unit/test_access_server_identity.py",
            "tests/unit/test_access_setup.py",
            "tests/unit/test_access_setup_wizard.py",
        ),
    ),
    Area(
        "tui-ux",
        src=(
            "scripts/tui_fuzz.py",
            "src/yeaboi/ui/",
            "src/yeaboi/repl/",
            "src/yeaboi/usage_export.py",
            "src/yeaboi/formatters.py",
            "src/yeaboi/clipboard.py",
            "src/yeaboi/voice.py",
            "src/yeaboi/voice_install.py",
            "src/yeaboi/music.py",
            "src/yeaboi/ambience.py",
            "src/yeaboi/os_open.py",
            # Provider credential verification (live pings, format checks) used
            # by both the setup wizard and the pre-mode LLM gate; its own tests
            # (test_provider_*.py) already live in this area.
            "src/yeaboi/provider_verification.py",
        ),
        tests=(
            "tests/unit/test_*screen*.py",
            "tests/unit/test_*page*.py",
            "tests/unit/test_shared_components.py",
            "tests/unit/test_row_ctx.py",
            "tests/unit/test_performance_rows.py",
            "tests/unit/test_llm_gate.py",
            "tests/unit/test_mascot*.py",
            "tests/unit/test_scene_backdrops.py",
            "tests/unit/test_sibling_repos.py",
            "tests/unit/test_ansi_font.py",
            "tests/unit/test_ascii_font.py",
            "tests/unit/test_animations.py",
            "tests/unit/test_scroll.py",
            "tests/unit/test_click.py",
            "tests/unit/test_input_raw_mode.py",
            "tests/unit/test_screensaver.py",
            "tests/unit/test_music*.py",
            "tests/unit/test_ambience.py",
            "tests/unit/test_tips*.py",
            "tests/unit/test_mode_cards.py",
            "tests/unit/test_mode_select_callsites.py",
            "tests/unit/test_*hub*.py",  # the saved-sessions hubs are a TUI surface
            "tests/unit/test_version_row.py",
            "tests/unit/test_formatters.py",
            "tests/unit/test_usage_*.py",
            "tests/unit/test_voice*.py",
            "tests/unit/test_clipboard.py",
            "tests/unit/test_copy_clipboard.py",
            "tests/unit/test_os_open.py",
            "tests/unit/test_attachments.py",
            "tests/unit/test_mayhem.py",
            "tests/unit/test_provider_*.py",
            "tests/unit/test_ui_error_classification.py",
            "tests/test_*.py",
        ),
    ),
    Area(
        "web-ux",
        src=(
            # Generated by Python, read by TypeScript in yeaboi-frontend, which
            # vendors this directory by sha. A change here selects the tests
            # that keep it fresh; the other repo's CI runs the --check.
            "contracts/web/",
            "src/yeaboi/web/",
            "src/yeaboi/html_theme.py",
            "src/yeaboi/html_exporter.py",
            "src/yeaboi/charts.py",
            "src/yeaboi/names.py",
            "scripts/gen_web_types.py",
            "scripts/gen_web_ui_contract.py",
            "scripts/dev_*",
        ),
        tests=(
            "tests/unit/test_web_*.py",
            "tests/unit/test_html_*.py",
            "tests/unit/test_charts.py",
            "tests/unit/test_export_picker.py",
        ),
    ),
    Area(
        # The desktop home's front page: the outlet registry, the parsers and
        # the paper, plus the one route that serves it. Named before platform,
        # which claims src/yeaboi/app/ as a whole.
        "news",
        src=("src/yeaboi/news/", "src/yeaboi/app/routes_news.py"),
        tests=("tests/unit/test_news_*.py", "tests/unit/test_app_news_routes.py"),
    ),
    Area(
        "platform",
        src=(
            "src/yeaboi/cli.py",
            # `python -m yeaboi` — the same CLI, and how the desktop's bundled
            # interpreter starts the backend.
            "src/yeaboi/__main__.py",
            # The desktop backend: the loopback API server `yeaboi app` binds.
            # Cross-mode shared infrastructure like mcp/ below — it dispatches
            # into every mode's tools without owning any of them.
            "src/yeaboi/app/",
            # The contracts other repos vendor: the wire `yeaboi app` serves
            # (app_http.md) and the desktop route manifest the parity suite reads.
            "contracts/",
            # The headless settings service the desktop settings pages write
            # through — allowlisted config writes, masked reads.
            "src/yeaboi/settings/",
            # The read-only connector layer. It rides with settings for the same
            # reason app/ does: it owns no mode, and every surface reads its
            # catalog through the settings engine it derives.
            "src/yeaboi/connectors/",
            # The full-screen catalog browser over that layer (Settings ▸
            # Connections ▸ Enter) — it renders nothing the engine did not say.
            "src/yeaboi/ui/catalog/",
            # The shapes those connectors return. No mode owns them — they are
            # the vocabulary a mode reads production in — so they ride here with
            # the connectors that produce them.
            "src/yeaboi/ops/",
            # The identities the desktop vendors, generated from those
            # descriptors — its icon table is checked against this file.
            "scripts/gen_connectors_contract.py",
            "src/yeaboi/telemetry.py",
            "src/yeaboi/feedback.py",
            # The privacy statement + egress disclosures every surface renders,
            # and the offline system check the app wire serves beside them.
            "src/yeaboi/privacy.py",
            "src/yeaboi/system_check.py",
            "src/yeaboi/setup_wizard.py",
            "src/yeaboi/update_check.py",
            "src/yeaboi/changelog.py",
            "src/yeaboi/changelog_data.json",
            # The tui/desktop/web vocabulary the changelog and the tips share.
            # Its other consumer's guards (test_tips, test_surface_parity) are
            # in ALWAYS, so they run whatever this selects.
            "src/yeaboi/surfaces.py",
            "src/yeaboi/mcp/",
            # Cross-mode shared infrastructure, like config/paths above: the
            # tamper-evident decision chain every mode records into.
            "src/yeaboi/provenance/",
            # And the clock any mode can run on. It owns the OS-job installer the
            # standup schedule was promoted out of, so a change here reaches a
            # mode it does not otherwise touch.
            "src/yeaboi/ceremonies/",
            "src/yeaboi/standup/scheduler.py",
            # The inbound half of that clock: what a team said back in Slack.
            # It rides with ceremonies because it anchors to their delivered
            # posts and answers with their store.
            "src/yeaboi/slack/",
            "src/yeaboi/tools/slack.py",
            # Niko, the global assistant. Cross-mode by construction: its tool
            # surface reads every other mode's store, so it belongs beside mcp/
            # rather than to any one mode. Its prompt is named here too —
            # `prompts/` as a whole belongs to planning, and a Niko prompt change
            # must run the niko tests, not only the planning ones.
            "src/yeaboi/niko/",
            "src/yeaboi/prompts/niko.py",
            # Its TUI halves too: `ui/` as a whole belongs to the UI area, whose
            # tests do not include test_niko_*.py, so a page-only change would
            # otherwise never run them.
            "src/yeaboi/ui/mode_select/_niko.py",
            "src/yeaboi/ui/mode_select/screens/_screens_niko.py",
            "claude-plugin/",
            "packaging/",
        ),
        tests=(
            "tests/unit/test_cli_*.py",
            "tests/unit/test_app_*.py",
            "tests/unit/test_settings_*.py",
            "tests/unit/test_connectors_*.py",
            "tests/unit/test_ops_*.py",
            "tests/unit/test_connections_*.py",
            "tests/unit/test_catalog_tui.py",
            "tests/unit/test_webhook_*.py",
            "tests/unit/test_provider_verify_ops.py",
            "tests/unit/test_provider_verify_cloud.py",
            "tests/unit/test_feedback.py",
            "tests/unit/test_privacy.py",
            "tests/unit/test_system_check.py",
            "tests/unit/test_setup_wizard.py",
            "tests/unit/test_update_*.py",
            "tests/unit/test_changelog*.py",
            "tests/unit/test_surfaces.py",
            "tests/unit/test_mcp_*.py",
            "tests/unit/test_provenance_*.py",
            "tests/unit/test_ceremonies_*.py",
            "tests/unit/test_slack_*.py",
            # The schedule wizard drives the promoted installer through the shim,
            # so it is the test that catches a broken promotion.
            "tests/unit/test_standup_schedule_wizard.py",
            # The standup wiring consumes the chain; a provenance change must
            # prove it did not break its first consumer.
            "tests/unit/test_standup_provenance_log.py",
            "tests/unit/test_niko_*.py",
        ),
    ),
)

# --- always, whatever changed -------------------------------------------------
# Guards with no import edge to the thing they guard: they read the repo — the
# workflows, the charters, the committed bundles, the packaged wheel, the
# generated docs — so nothing about a changed Python module implies them. They
# are also, collectively, fast.
#
# `tests/unit/test_test_scope.py` asserts every glob here matches at least one
# real file, because a guard glob that has silently stopped matching is exactly
# as good as no guard at all.
ALWAYS: tuple[str, ...] = (
    "tests/unit/test_surface_parity.py",
    # Its finer-grained twin: reads the terminal's own tables and the committed
    # desktop manifest, so no changed module implies it either.
    "tests/unit/test_tui_parity.py",
    "tests/unit/test_tips.py",
    "tests/unit/tools/test_tools_registry.py",
    "tests/unit/test_conftest_guards.py",
    # The gh guard's own reach test. Separate from `test_conftest_guards.py`, and
    # named `zz_` so it collects last, because the property it checks is *which
    # modules are loaded* — `scripts/` is not a package and two loaders disagree
    # about who owns `_gh_transport`, so the guard has more than one object to
    # patch. Run early it would skip; run last it sees the split. It is in ALWAYS
    # because it guards the repo rather than a module, and because a scoped run is
    # the case where the loaded set is smallest and the guard's reach is thinnest.
    "tests/unit/test_zz_gh_guard.py",
    "tests/unit/test_fixtures.py",
    "tests/unit/test_test_scope.py",
    # The tests for the GLOBAL modules above. Their subjects force a full run
    # anyway; running the tests themselves on every scoped run is what keeps a
    # change *elsewhere* from breaking `paths.py`'s contract unnoticed, and the
    # three of them together are well under a second.
    "tests/unit/test_config.py",
    "tests/unit/test_paths.py",
    "tests/unit/test_logging_setup.py",
    # committed artefacts the Python suite reads but never builds
    "tests/unit/test_web_assets.py",
    # The generated contracts yeaboi-frontend vendors. Deliberately not in an
    # area: the tuples they are built from live in half a dozen modules, and
    # a stale contract is invisible on this side of the boundary.
    "tests/unit/test_web_contracts.py",
    # contracts/site.json is derived from pyproject.toml and consumed by another
    # repo. The path that invalidates it — pyproject.toml — is GLOBAL, but a
    # hand-edit of the JSON itself is claimed by one small area, so the freshness
    # assertion belongs here rather than behind either trigger.
    "tests/unit/test_site_contract.py",
    # repo + CI metadata
    "tests/unit/test_workflow_schema.py",
    "tests/unit/test_workflow_concurrency.py",
    "tests/unit/test_auto_version_fallback.py",
    "tests/unit/test_publish_workflows.py",
    "tests/unit/test_codeql_triage.py",
    "tests/unit/test_claude_workflow.py",
    "tests/unit/test_implement_reconcile.py",
    "tests/unit/test_claude_plugin.py",
    "tests/unit/test_pr_feedback.py",
    "tests/unit/test_gh_transport.py",
    # The install commands README.md advertises — and README.md is INERT, so
    # without this entry a README-only change runs nothing. The installer itself
    # lives in yeaboi-site and is tested there.
    "tests/unit/test_readme_install.py",
    # AST-scans every .py in the repo for the two 3.11-only constructs the 3.10
    # floor bans. Any file can reintroduce them, so no path implies this test.
    "tests/unit/test_compat.py",
    # Its guard half AST-scans every module in src/ for a bare fromisoformat,
    # which any file can reintroduce.
    "tests/unit/test_timeparse.py",
    # Ten surfaces name the supported Python floor and nothing else connects
    # them — including an OG PNG that is rendered by hand.
    "tests/unit/test_python_floor.py",
    # The ship gate reads the Makefile, .claude/repo-notes.md and this file.
    # Nothing about a changed module implies it, and its whole subject is the
    # machinery that decides what a scoped run covers — so it has to run on
    # every one. (The /ship and /sync-main commands themselves moved to the
    # yeaboi-devkit plugin; their shape is guarded in the tooling repo.)
    "tests/unit/test_ship_gate.py",
    "tests/unit/test_record_demo.py",
    # version / release lockstep
    "tests/unit/test_release_*.py",
    "tests/unit/test_bump_version.py",
)

# --- force the whole suite ----------------------------------------------------
# Reached by everything, or capable of changing how everything is collected. The
# cost of a false positive here is one full run; the cost of a false negative is
# a green PR that broke something nothing looked at.
GLOBAL: tuple[str, ...] = (
    "pyproject.toml",
    "uv.lock",
    "Makefile",
    "tests/conftest.py",
    "tests/__init__.py",
    "tests/_pages.py",
    "tests/_node_helpers.py",
    "tests/fixtures/",
    ".github/workflows/ci.yml",
    "scripts/test_scope.py",
    "src/yeaboi/__init__.py",
    "src/yeaboi/_compat.py",  # StrEnum for the 3.10 floor — its members serialize into every artifact
    "src/yeaboi/timeparse.py",  # every stored and provider timestamp in the app is read through it
    "src/yeaboi/sessions.py",  # CURRENT_SCHEMA_VERSION — every store migrates off it
    "src/yeaboi/persistence.py",
    "src/yeaboi/paths.py",
    "src/yeaboi/config.py",
    # The provider table every surface derives from: the factory, the wizard
    # cards, the settings engine, verification and the guardrail classifier.
    "src/yeaboi/llm_providers.py",
    "src/yeaboi/logging_setup.py",
    "src/yeaboi/ui/shared/",  # every screen in the app builds out of it
    "src/yeaboi/web/assets.py",  # the only door to the committed bundles
)

# --- paths that cannot change Python behaviour --------------------------------
# Prose and agent configuration. Safe to treat as inert *because* the tests that
# read them — `test_claude_*`, `test_readme_install` — are in ALWAYS and run
# regardless. Without that they would belong in GLOBAL.
INERT: tuple[str, ...] = (
    ".claude/",
    "README.md",
    "SECURITY.md",
    "AGENTS.md",
    ".gitignore",
    ".github/dependabot.yml",
)


@dataclass(frozen=True)
class Job:
    """An optional `ci.yml` job and the paths that make it worth running."""

    name: str
    triggers: tuple[str, ...]


JOBS: tuple[Job, ...] = (
    Job("package", ("pyproject.toml", "packaging/")),
    Job("eval", ("src/yeaboi/prompts/", "src/yeaboi/agent/", "tests/golden/")),
    # The non-required matrix that runs the unit lane on 3.11–3.14. Gated on any
    # Python at all, not just the two shims: `unit` and `integration` now pin the
    # floor, so a narrower trigger would leave a change to an ordinary module
    # tested on 3.10 and nowhere else — less coverage above the floor than before
    # this job existed. It is non-required with `fail-fast: false`, so the cost of
    # the wide trigger is runner minutes, never merge risk.
    Job("compat", ("src/", "tests/", "pyproject.toml", "uv.lock")),
)

FULL_UNIT = ("tests/unit/", "tests/test_*.py")
FULL_SLOW = ("tests/integration/", "tests/contract/")


@dataclass
class Scope:
    """What a change needs. ``full`` means every rule fell through to safety."""

    full: bool = False
    reasons: list[str] = field(default_factory=list)
    areas: set[str] = field(default_factory=set)
    tests: set[str] = field(default_factory=set)
    jobs: set[str] = field(default_factory=set)
    slow: bool = False


def _claim(path: str, prefixes: tuple[str, ...]) -> int:
    """How specifically `prefixes` claims `path` — 0 when it does not.

    The score is the length of the longest pattern that matched, which makes an
    exact file beat the directory containing it. `area_for` uses that to pick a
    winner instead of taking the first area listed.
    """
    best = 0
    for prefix in prefixes:
        if prefix.endswith("/"):
            if path.startswith(prefix):
                best = max(best, len(prefix))
        elif path == prefix or fnmatch.fnmatch(path, prefix):
            best = max(best, len(prefix))
    return best


def _matches(path: str, prefixes: tuple[str, ...]) -> bool:
    """A path is claimed by a prefix (directory or exact file) or a glob."""
    return _claim(path, prefixes) > 0


def area_for(path: str) -> Area | None:
    """The area claiming this source path, or None. MOST SPECIFIC match wins.

    It used to be first-match-wins, which silently made two entries unreachable:
    `security` claims `sharing/access.py` and `sharing/gate.py` by name, but
    `artifacts-sharing` claims the whole `sharing/` directory and is listed
    first, so a change to the share access-control code ran the sharing tests
    and never the guardrail ones. Ordering is invisible in a diff and the
    failure is silent, so the tie-break is specificity rather than position;
    `test_test_scope.py` now also asserts every `Area.src` entry resolves back
    to its own area, which is what makes a future shadowing fail the build.
    """
    best: Area | None = None
    best_score = 0
    for area in AREAS:
        score = _claim(path, area.src)
        if score > best_score:
            best, best_score = area, score
    return best


def resolve(changed: list[str]) -> Scope:
    """The whole decision. Everything below feeds this; nothing re-decides it."""
    scope = Scope()
    if not changed:
        # An empty diff is not "nothing to test", it is "we could not read the
        # diff" often enough that guessing is not worth the minutes saved.
        scope.full = True
        scope.reasons.append("no changed files were reported — running everything")
        return scope

    for path in changed:
        path = path.strip()
        if not path:
            continue
        if _matches(path, GLOBAL):
            scope.full = True
            scope.reasons.append(f"{path} is global")
            continue
        for job in JOBS:
            if _matches(path, job.triggers):
                scope.jobs.add(job.name)
        if _matches(path, INERT):
            scope.reasons.append(f"{path} is prose/config — no tests implied")
            continue
        if path.startswith("tests/integration/") or path.startswith("tests/contract/"):
            scope.slow = True
            scope.reasons.append(f"{path} is a slow-lane test")
            continue
        if path.startswith("tests/"):
            # A changed test runs, whatever else claims it. Its own area comes
            # along below if some source file in the same change points there.
            scope.tests.add(path)
            claimed = [a for a in AREAS if _matches(path, a.tests)]
            for area in claimed:
                scope.areas.add(area.name)
            if not claimed and not _matches(path, ALWAYS):
                scope.reasons.append(f"{path} is claimed by no area — running everything")
                scope.full = True
            continue
        area = area_for(path)
        if area is None:
            scope.full = True
            scope.reasons.append(f"{path} is claimed by no area — running everything")
            continue
        scope.areas.add(area.name)
        scope.areas.update(area.also)
        scope.reasons.append(f"{path} → {area.name}")

    if not scope.full:
        for name in scope.areas:
            area = next(a for a in AREAS if a.name == name)
            scope.tests.update(area.tests)
        scope.tests.update(ALWAYS)
    return scope


def unit_paths(scope: Scope) -> list[str]:
    """pytest arguments for the unit lane, deduped and ordered for readability."""
    if scope.full:
        return list(FULL_UNIT)
    # Globs that match nothing make pytest exit 4 ("file or directory not
    # found"), so they are resolved here rather than handed over verbatim.
    resolved: set[str] = set()
    for pattern in scope.tests:
        if any(ch in pattern for ch in "*?["):
            resolved.update(str(p.relative_to(ROOT)) for p in ROOT.glob(pattern))
        elif (ROOT / pattern).exists():
            resolved.add(pattern)
    return sorted(resolved)


def slow_paths(scope: Scope) -> list[str]:
    """The integration+contract lane is all-or-nothing: 572 tests, ~100s."""
    return list(FULL_SLOW)


def changed_from_git(base: str | None, working_tree: bool) -> list[str]:
    """Changed paths, from the working tree or from a merge-base diff."""

    def _git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else ""

    if working_tree:
        paths: list[str] = []
        for line in _git("status", "--porcelain").splitlines():
            entry = line[3:] if len(line) > 3 else ""
            # Renames print `old -> new`, and BOTH sides matter: the source area
            # still has tests importing the path the file left, and they are
            # exactly the ones a move breaks. Taking only the destination made a
            # cross-area `git mv` run the new area's tests and nothing else,
            # while the totality guard stayed green because the file *is*
            # claimed — just by the wrong area.
            for side in entry.split(" -> "):
                paths.append(side.strip().strip('"'))
        return [p for p in paths if p]
    if base:
        merge_base = _git("merge-base", base, "HEAD").strip()
        if not merge_base:
            # A base we cannot resolve is not an empty diff — say nothing and let
            # `resolve` fall through to the full suite.
            return []
        # `--no-renames`: with detection on, git prints a move as the single
        # destination path, hiding the source area the move broke.
        return [p for p in _git("diff", "--name-only", "--no-renames", f"{merge_base}...HEAD").splitlines() if p]
    return []


def explain(scope: Scope) -> str:
    lines = []
    if scope.full:
        lines.append("scope: FULL — every test, every job")
    else:
        lines.append(f"scope: {len(scope.areas)} area(s): {', '.join(sorted(scope.areas)) or '(guards only)'}")
        lines.append(f"       {len(unit_paths(scope))} unit test path(s), always-run guards included")
        lines.append(f"       jobs: {', '.join(sorted(scope.jobs)) or '(none)'}")
    for reason in scope.reasons[:25]:
        lines.append(f"  · {reason}")
    if len(scope.reasons) > 25:
        lines.append(f"  · … and {len(scope.reasons) - 25} more")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decide which tests and CI jobs a change needs.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--changed-files", help="file with one path per line, or - for stdin")
    source.add_argument("--working-tree", action="store_true", help="use `git status --porcelain`")
    source.add_argument("--base", help="diff against the merge-base with this ref")
    parser.add_argument("--unit-paths", action="store_true", help="pytest args for the unit lane")
    parser.add_argument("--slow-paths", action="store_true", help="pytest args for integration+contract")
    parser.add_argument("--jobs", action="store_true", help="JSON of optional-job booleans")
    parser.add_argument("--github-output", action="store_true", help="write every output to $GITHUB_OUTPUT")
    parser.add_argument("--explain", action="store_true", help="human-readable reasoning")
    args = parser.parse_args(argv)

    if args.changed_files:
        raw = sys.stdin.read() if args.changed_files == "-" else Path(args.changed_files).read_text()
        changed = [line.strip() for line in raw.splitlines() if line.strip()]
    else:
        changed = changed_from_git(args.base, args.working_tree)

    scope = resolve(changed)

    if args.explain:
        print(explain(scope), file=sys.stderr if args.unit_paths else sys.stdout)
    if args.unit_paths:
        print(" ".join(unit_paths(scope)))
    if args.slow_paths:
        print(" ".join(slow_paths(scope)))
    if args.jobs:
        print(json.dumps({job.name: scope.full or job.name in scope.jobs for job in JOBS}))
    if args.github_output:
        target = os.environ.get("GITHUB_OUTPUT")
        if not target:
            print("[test-scope] --github-output needs $GITHUB_OUTPUT", file=sys.stderr)
            return 2
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(f"full={'true' if scope.full else 'false'}\n")
            handle.write(f"unit_paths={' '.join(unit_paths(scope))}\n")
            handle.write(f"slow_paths={' '.join(slow_paths(scope))}\n")
            for job in JOBS:
                handle.write(f"{job.name}={'true' if scope.full or job.name in scope.jobs else 'false'}\n")
        print(explain(scope))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
