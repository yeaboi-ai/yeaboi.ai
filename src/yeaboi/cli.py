"""CLI entry point for yeaboi.

Module-level imports here are deliberately limited to the stdlib and yeaboi's
lightweight config/paths modules: `yeaboi --version`/`--help` (and every
subcommand's fixed overhead) pay for everything imported at the top of this
file before argparse even runs. Anything that pulls rich, prompt_toolkit, or
the langchain/anthropic stack is imported at its call site instead — the same
convention as beta.py, enforced by tests/unit/test_cli_startup.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from yeaboi import __version__, fs_policy, paths
from yeaboi.beta import AGENTWATCH_BETA_NOTICE, BETA_TAG, PERFORMANCE_BETA_NOTICE, WEEKLY_REVIEW_BETA_NOTICE
from yeaboi.config import (
    detect_proxy,
    disable_langsmith_tracing,
    is_langsmith_enabled,
    load_user_config,
)

if TYPE_CHECKING:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

# Default filename for exported questionnaire templates
DEFAULT_QUESTIONNAIRE_FILENAME = "scrum-questionnaire.md"


# Default DB path — inside the user config directory alongside history/config.
# Matches the path used by SessionStore in run_repl(). Single-sourced via
# paths.ROOT_DIR so the config-dir location (~/.yeaboi) lives in one place.
_SESSIONS_DB_DIR = paths.ROOT_DIR


def _summarise_scrum_md(console: Console, path: Path) -> None:
    """Print a brief pre-flight summary of the SCRUM.md file.
    Shows line count, URL count, and detected ## sections so users can
    confirm the right file was picked up before the analysis runs.
    """
    try:
        content = path.read_text()
    except OSError:
        console.print("[dim]  SCRUM.md detected — your project context will be included in the analysis.[/dim]")
        return

    lines = content.count("\n") + 1
    url_count = len(re.findall(r"https?://", content))
    sections = [ln.lstrip("#").strip() for ln in content.splitlines() if ln.startswith("## ")]

    stats = f"{lines} lines" + (f", {url_count} URL{'s' if url_count != 1 else ''}" if url_count else "")
    console.print(f"[dim]  SCRUM.md detected ({stats})[/dim]")
    if sections:
        console.print(f"[dim]    Sections: {' · '.join(sections)}[/dim]")


def _build_welcome_panel() -> Panel:
    """Build the branded welcome panel with version and quick-start hint.

    # See docs: "Architecture" — the CLI layer is the outermost layer,
    # responsible for user-facing chrome like the welcome screen.
    """
    from rich.panel import Panel
    from rich.text import Text

    body = Text.from_markup(
        f"[bold cyan]yeaboi.ai[/bold cyan]  [dim]v{__version__}[/dim]\n"
        "[white]Best friend to engineers and agents[/white]\n\n"
        "[dim]Describe your project to get started, or type [cyan]help[/cyan] for commands.[/dim]"
    )
    return Panel(body, border_style="cyan", padding=(1, 2))


# ---------------------------------------------------------------------------
# Session listing / picker helpers
# ---------------------------------------------------------------------------


def _build_sessions_table(sessions: list[dict], display_names: dict[str, str] | None = None) -> Table:
    """Build a Rich Table of saved sessions.

    Used by both --list-sessions and the interactive --resume picker.

    Args:
        sessions: List of session metadata dicts.
        display_names: Optional ``{session_id: unique_name}`` mapping from
            ``make_unique_display_names()``. When provided, the Project column
            shows the collision-free display name instead of the raw project_name.
    """
    from rich.table import Table

    table = Table(title="Saved sessions", show_lines=False, padding=(0, 1))
    table.add_column("#", style="bold", width=3)
    table.add_column("Project", style="cyan")
    table.add_column("Date", style="dim")
    table.add_column("Last Step", style="green")
    table.add_column("Session ID", style="dim")
    for i, meta in enumerate(sessions, 1):
        sid = meta.get("session_id", "")
        if display_names and sid in display_names:
            project = display_names[sid]
        else:
            project = meta.get("project_name") or "(unnamed)"
        date_str = meta.get("created_at", "")[:10]
        last_node = meta.get("last_node_completed") or "-"
        table.add_row(str(i), project, date_str, last_node, sid)
    return table


def _print_sessions_table(console: Console) -> None:
    """Print a table of all saved sessions and exit.

    Used by --list-sessions. Opens its own SessionStore so it works
    independently from the REPL.
    """
    from yeaboi.sessions import SessionStore, make_unique_display_names

    _SESSIONS_DB_DIR.mkdir(parents=True, exist_ok=True)
    db_path = _SESSIONS_DB_DIR / "sessions.db"
    with SessionStore(db_path) as store:
        sessions = store.list_sessions()
    if not sessions:
        console.print("[hint]No saved sessions found.[/hint]")
        return
    unique_names = make_unique_display_names(sessions)
    console.print(_build_sessions_table(sessions, display_names=unique_names))


def _clear_sessions(console: Console) -> None:
    """Interactively delete saved sessions.

    Shows a numbered list plus an [A] All option. The user picks a session
    number to delete one, or 'a'/'all' to wipe everything.
    """
    from prompt_toolkit import PromptSession

    from yeaboi.sessions import SessionStore, make_unique_display_names

    _SESSIONS_DB_DIR.mkdir(parents=True, exist_ok=True)
    db_path = _SESSIONS_DB_DIR / "sessions.db"
    with SessionStore(db_path) as store:
        sessions = store.list_sessions()
        if not sessions:
            console.print("[hint]No saved sessions found.[/hint]")
            return
        unique_names = make_unique_display_names(sessions)
        console.print(_build_sessions_table(sessions, display_names=unique_names))
        console.print()
        console.print(
            "[hint]Enter a session number to delete, [command]all[/command] to clear everything, "
            "or [command]q[/command] to cancel.[/hint]"
        )
        pick_session: PromptSession[str] = PromptSession()
        while True:
            try:
                raw = pick_session.prompt("Clear> ")
            except (KeyboardInterrupt, EOFError):
                return
            raw = raw.strip().lower()
            if raw in ("q", "quit", "cancel"):
                return
            if raw in ("a", "all"):
                count = store.delete_all_sessions()
                console.print(f"[success]Deleted all {count} session{'s' if count != 1 else ''}.[/success]")
                return
            try:
                idx = int(raw)
            except ValueError:
                console.print("[warning]Please enter a number, 'all', or 'q' to cancel.[/warning]")
                continue
            if 1 <= idx <= len(sessions):
                picked = sessions[idx - 1]
                sid = picked["session_id"]
                name = unique_names.get(sid, sid)
                store.delete_session(sid)
                console.print(f"[success]Deleted session: {name}[/success]")
                return
            console.print(f"[warning]Please pick a number between 1 and {len(sessions)}.[/warning]")


def _resolve_resume(console: Console, resume_arg: str) -> tuple[dict | None, str | None]:
    """Resolve --resume into a (graph_state, session_id) tuple.

    Args:
        console: Rich Console for output.
        resume_arg: The value of args.resume — "__pick__" for interactive,
            "latest" for most recent, or a specific session ID.

    Returns:
        (graph_state, session_id) on success, (None, None) on failure/cancel.

    # See docs: "Memory & State" — session persistence, --resume
    """
    from prompt_toolkit import PromptSession

    from yeaboi.sessions import SessionStore, make_display_name, make_unique_display_names

    _SESSIONS_DB_DIR.mkdir(parents=True, exist_ok=True)
    db_path = _SESSIONS_DB_DIR / "sessions.db"
    with SessionStore(db_path) as store:
        if resume_arg == "latest":
            sid = store.get_latest_session_id()
            if sid is None:
                console.print("[warning]No saved sessions found.[/warning]")
                return None, None
            state = store.load_state(sid)
            if state is None:
                console.print(f"[warning]Session {sid} has no saved state or is corrupt.[/warning]")
                return None, None
            meta = store.get_session(sid)
            name = make_display_name(meta) if meta else sid
            console.print(f"[success]Loading session:[/success] {name}")
            return state, sid

        if resume_arg == "__pick__":
            sessions = store.list_sessions()
            if not sessions:
                console.print("[hint]No saved sessions found.[/hint]")
                return None, None
            unique_names = make_unique_display_names(sessions)
            console.print(_build_sessions_table(sessions, display_names=unique_names))
            console.print()
            pick_session: PromptSession[str] = PromptSession()
            while True:
                try:
                    raw = pick_session.prompt("Pick a session number (or 'q' to cancel): ")
                except (KeyboardInterrupt, EOFError):
                    return None, None
                raw = raw.strip().lower()
                if raw in ("q", "quit", "cancel"):
                    return None, None
                # Try numeric index first, then match by display name.
                try:
                    idx = int(raw)
                except ValueError:
                    # Match against display names (case-insensitive, partial match)
                    matches = [(i, s) for i, s in enumerate(sessions) if raw in unique_names[s["session_id"]].lower()]
                    if len(matches) == 1:
                        idx = matches[0][0] + 1  # convert to 1-based
                    elif len(matches) > 1:
                        console.print(f"[warning]'{raw}' matches multiple sessions. Be more specific.[/warning]")
                        continue
                    else:
                        console.print("[warning]No match. Enter a number or session name (or 'q' to cancel).[/warning]")
                        continue
                if 1 <= idx <= len(sessions):
                    picked = sessions[idx - 1]
                    sid = picked["session_id"]
                    state = store.load_state(sid)
                    if state is None:
                        console.print(f"[warning]Session {sid} has no saved state or is corrupt.[/warning]")
                        return None, None
                    name = make_display_name(picked)
                    console.print(f"[success]Loading session:[/success] {name}")
                    return state, sid
                console.print(f"[warning]Please pick a number between 1 and {len(sessions)}.[/warning]")

        # Specific session ID
        state = store.load_state(resume_arg)
        if state is None:
            console.print(f"[warning]Session '{resume_arg}' not found or has no saved state.[/warning]")
            return None, None
        meta = store.get_session(resume_arg)
        name = make_display_name(meta) if meta else resume_arg
        console.print(f"[success]Loading session:[/success] {name}")
        return state, resume_arg


def _context_spec(spec: str):
    """argparse type for --context: 'all' | 'none' | 'inherit' | a scope spec."""
    from yeaboi.context.scope import parse_context_spec

    try:
        return parse_context_spec(spec)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _add_context_flags(parser: argparse.ArgumentParser, *, scope: bool = True) -> None:
    """The three flags every run takes: what it reads, and how it is labelled."""
    if scope:
        parser.add_argument(
            "--context",
            default=None,
            type=_context_spec,
            metavar="SPEC",
            help="Which other sessions this run may read: 'all', 'none', or e.g. "
            "'standup,retro:1@2sprints project=apollo tags=a,b' (default: the mode's saved or last-used scope)",
        )
    parser.add_argument(
        "--project-label",
        dest="project_label",
        default="",
        metavar="NAME",
        help="Free-text project label recorded on this run (autocompletes on the other surfaces)",
    )
    parser.add_argument(
        "--tag",
        dest="tags",
        action="append",
        default=[],
        metavar="TAG",
        help="Tag recorded on this run beside its default tags (repeatable)",
    )


def _context_kwargs(args: argparse.Namespace) -> dict:
    """The engine kwargs the context flags map onto — only the ones the user set."""
    out: dict = {}
    if getattr(args, "context", None) is not None:
        out["context"] = args.context
    if getattr(args, "project_label", ""):
        out["project_label"] = args.project_label
    if getattr(args, "tags", None):
        out["tags"] = list(args.tags)
    return out


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="yeaboi",
        description=(
            "yeaboi.ai — best friend to engineers and agents. Runs your team's scrum, and watches your AI agents work."
        ),
        epilog=(
            "examples:\n"
            "  yeaboi                        interactive mode (recommended)\n"
            "  yeaboi --quick                quick intake (2 questions only)\n"
            "  yeaboi --questionnaire q.md   import pre-filled questionnaire\n"
            "  yeaboi --export-only --quick  non-interactive, auto-accept all\n"
            "  yeaboi --resume               resume last session (interactive picker)\n"
            "  yeaboi --resume latest         resume most recent session\n"
            "  yeaboi --list-sessions         list all saved sessions\n"
            "  yeaboi --clear-sessions        delete saved sessions\n"
            '  yeaboi --non-interactive --description "Build a todo app"  headless mode\n'
            '  yeaboi --non-interactive --description "..." --output json  JSON to stdout'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        nargs="?",
        const="__pick__",
        default=None,
        help="Resume a previous session. Without an argument, shows an interactive session picker. "
        "Pass 'latest' to resume the most recent session, or a session ID to resume a specific one.",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        default=False,
        help="List all saved sessions and exit.",
    )
    parser.add_argument(
        "--clear-sessions",
        action="store_true",
        default=False,
        help="Interactively delete saved sessions (pick one or clear all) and exit.",
    )
    parser.add_argument(
        "--export-questionnaire",
        metavar="PATH",
        nargs="?",
        const=DEFAULT_QUESTIONNAIRE_FILENAME,
        default=None,
        help=f"Export a blank questionnaire template as Markdown (default: {DEFAULT_QUESTIONNAIRE_FILENAME}).",
    )
    parser.add_argument(
        "--questionnaire",
        metavar="PATH",
        default=None,
        help="Import a filled-in questionnaire Markdown file and jump to confirmation.",
    )

    # Intake mode flags — mutually exclusive.
    # Smart mode is the default when neither flag is given.
    # See docs: "Project Intake Questionnaire" — smart intake
    # The legacy --full-intake (30-question "standard" mode) has been removed —
    # smart intake is the single interactive path. --quick remains for power users.
    intake_group = parser.add_mutually_exclusive_group()
    intake_group.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Quick intake — only ask team size and tech stack, auto-fill everything else.",
    )

    parser.add_argument(
        "--export-only",
        action="store_true",
        default=False,
        help="Auto-accept all review checkpoints and exit after the full plan is generated. "
        "Combine with --quick or --questionnaire for fully non-interactive runs.",
    )

    # The Solo world from the command line: one developer, no roster.
    parser.add_argument(
        "--solo",
        action="store_true",
        default=False,
        help="Run as one developer: team questions default to you, standups and reports are yours alone.",
    )

    parser.add_argument(
        "--prior-art",
        action="append",
        default=None,
        metavar="REPO_KEY",
        help="Existing repository to build this plan on, as a key like 'github:acme/auth' "
        "(repeatable). Requires --non-interactive; an interactive run asks about prior art "
        "in the intake instead, so the flag is ignored there.",
    )

    parser.add_argument(
        "--ac-format",
        choices=["gwt", "bullets"],
        default=None,
        help="Acceptance-criteria style for generated stories: 'gwt' (Given/When/Then) or "
        "'bullets' (clear testable statements). Default: follow the learned team profile "
        "(or YEABOI_AC_FORMAT).",
    )

    parser.add_argument(
        "--architecture-spike",
        choices=["auto", "include", "skip"],
        default=None,
        help="Whether to add an architecture-validation spike when the analyzer's decision is "
        "open (2+ options): 'include'/'skip' force it, 'auto' adds it unless the analyzer's "
        "confidence is high. Default: ask interactively (auto in non-interactive runs).",
    )

    parser.add_argument(
        "--no-bell",
        action="store_true",
        default=False,
        help="Disable terminal bell after pipeline steps.",
    )

    parser.add_argument(
        "--theme",
        choices=["dark", "light"],
        default="dark",
        help="Terminal colour theme (default: dark). Use 'light' for white/cream backgrounds.",
    )

    parser.add_argument(
        "--mode",
        choices=["project-planning"],  # extend this list as new modes ship
        default=None,
        help="Skip the startup menu and launch directly into a specific mode.",
    )

    parser.add_argument(
        "--setup",
        action="store_true",
        default=False,
        help="Re-run the first-time setup wizard to update credentials.",
    )

    parser.add_argument(
        "--allow-path",
        metavar="PATH",
        action="append",
        default=[],
        help=(
            "Allow filesystem access to PATH for this run only (repeatable). "
            "yeaboi is sandboxed to ~/.yeaboi; persistent allowances live in "
            "YEABOI_ALLOWED_PATHS or Settings → Allowed Paths."
        ),
    )

    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        default=False,
        help="List the microphones yeaboi can record from, then exit. "
        "Set the one you want with VOICE_DEVICE (or Settings → Voice Input).",
    )

    parser.add_argument(
        "--setup-access",
        action="store_true",
        default=False,
        help="Set up the Cloudflare Access verified-users tier for shared boards, then exit. "
        "Runs the cloudflared sign-in, tunnel and DNS steps for you and stores the result; "
        "the Access application itself is still created in the Cloudflare dashboard. "
        "The same thing Settings offers inside the app — this is the non-TUI path.",
    )

    parser.add_argument(
        "--install-voice",
        action="store_true",
        default=False,
        help="Install the dictation packages and speech model into this environment, then exit. "
        "The same thing double-tapping Space offers inside the app — this is the non-TUI path "
        "for CI, dev containers and terminals the full-screen UI cannot drive.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run the TUI with mock data and fake delays — no LLM calls. For UI development.",
    )

    # ── Non-interactive / headless mode ──────────────────────────────────
    # Runs the full pipeline without user interaction. Requires --description.
    # Combine with --output to control output format (default: markdown).
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        default=False,
        help="Run the full pipeline headlessly (no user interaction). Requires --description.",
    )
    parser.add_argument(
        "--output",
        choices=["markdown", "json", "html", "prd"],
        default=None,
        help="Output format for the generated plan ('prd' writes a Product Requirements "
        "Document; one extra LLM call). Only valid with --non-interactive or --export-only.",
    )
    parser.add_argument(
        "--description",
        metavar="TEXT",
        default=None,
        help="Project description for headless mode. Use @file.txt to read from a file.",
    )
    parser.add_argument(
        "--team-size",
        metavar="N",
        type=int,
        default=None,
        help="Team size (maps to intake Q6). Only used with --non-interactive.",
    )
    parser.add_argument(
        "--sprint-length",
        metavar="WEEKS",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Sprint length in weeks (maps to intake Q8). Only used with --non-interactive.",
    )

    # What a headless plan may read, and how it is labelled — the same three
    # flags every run subcommand takes.
    _add_context_flags(parser)

    # ── Daily Standup flags ───────────────────────────────────────────────
    # --standup-run is what the OS scheduler (launchd/cron) invokes: it runs a
    # standup headlessly and delivers it. See docs: "Daily Standup".
    parser.add_argument(
        "--standup-run",
        action="store_true",
        default=False,
        help="Run a daily standup headlessly and deliver it (used by the OS scheduler).",
    )
    parser.add_argument(
        "--standup-session",
        metavar="SESSION_ID",
        default=None,
        help="Session to run the standup for. Defaults to the most recent session. Only used with --standup-run.",
    )
    parser.add_argument(
        "--standup-output",
        choices=["terminal", "desktop", "slack", "email", "all"],
        default=None,
        help="Override delivery channel(s) for --standup-run (default: the session's saved channels).",
    )
    parser.add_argument(
        "--standup-interactive",
        action="store_true",
        default=False,
        help="With --standup-run: prompt for your update + confirm (timed) before generating. "
        "What the scheduler opens in a terminal; falls back to headless when no TTY.",
    )
    # The second scheduled job: fires AFTER the standup, posts one desktop
    # notification if any standup went unchecked, and exits. No terminal, no LLM.
    parser.add_argument(
        "--standup-remind-transcript",
        action="store_true",
        default=False,
        help="Post a desktop reminder if standups went unchecked against their meetings (used by the OS scheduler).",
    )

    # ── Team learning flags ───────────────────────────────────────────────
    parser.add_argument(
        "--learn",
        action="store_true",
        default=False,
        help="Analyse historical Jira/AzDO sprint data and store a team calibration profile. "
        "Subsequent planning sessions use this profile to calibrate estimates.",
    )
    parser.add_argument(
        "--team-profile",
        action="store_true",
        default=False,
        help="Display the current stored team calibration profile and exit.",
    )
    parser.add_argument(
        "--retro",
        metavar="SESSION_ID",
        nargs="?",
        const="latest",
        default=None,
        help="Compare a past session's plan to actual Jira/AzDO outcomes. "
        "Pass a session ID or omit for the most recent session.",
    )

    # ── Subcommands — headless mode runners ───────────────────────────────
    # Additive: every flat flag above keeps working (the subparsers action is
    # optional, so bare `yeaboi` and `yeaboi --<flag>` parse unchanged).
    # See AGENTS.md "REQUIRED: Surface Parity" — each mode needs a CLI path;
    # these run the same engines the TUI and the MCP server use.
    subparsers = parser.add_subparsers(
        dest="command", metavar="{report,standup,standup-review,perf,retro,poker,review,analyze,agents}"
    )

    report_p = subparsers.add_parser("report", help="Generate a stakeholder delivery report (Reporting mode)")
    report_p.add_argument(
        "--period", choices=["last_week", "last_sprint", "last_month", "quarter", "window"], default="last_sprint"
    )
    report_p.add_argument("--session", default="", metavar="ID", help="Session to use (default: most recent)")
    report_p.add_argument(
        "--window-start", default="", metavar="YYYY-MM-DD", help="Explicit window start (quarter/window periods)"
    )
    report_p.add_argument("--window-end", default="", metavar="YYYY-MM-DD", help="Explicit window end")
    report_p.add_argument(
        "--sprint-names", default="", metavar="A,B", help="Comma-separated sprint names framing a quarter report"
    )
    report_p.add_argument("--label", default="", metavar="TEXT", help='Period label override (e.g. "Q3 2026")')
    report_p.add_argument("--jira-project", default="", metavar="KEY", help="Jira project key override")
    report_p.add_argument("--azdo-project", default="", metavar="NAME", help="Azure DevOps project override")
    report_p.add_argument(
        "--source",
        choices=["jira", "azdevops", "both"],
        default="",
        help="Ticketing source(s) for delivered work (default: every configured tracker)",
    )
    report_p.add_argument(
        "--code-sources",
        nargs="+",
        choices=["github", "azdevops"],
        default=None,
        metavar="SOURCE",
        help="Code hosts to pull supporting PR/commit context from (default: all configured)",
    )
    report_p.add_argument(
        "--documentation-sources",
        nargs="+",
        choices=["confluence", "notion"],
        default=None,
        metavar="SOURCE",
        help="Doc platforms to pull supporting doc-update context from (default: all configured)",
    )
    report_p.add_argument(
        "--solo",
        action="store_true",
        help="A one-person report: first-person narrative, never 'the team' (the Solo world)",
    )
    report_p.add_argument("--strict", action="store_true", help="Exit 3 on a degraded run (warnings/empty report)")
    _add_context_flags(report_p)
    report_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    report_p.add_argument(
        "--theme",
        default="midnight",
        metavar="NAME",
        help="Export palette: midnight/aurora/sunset/mono or a custom name from reporting_themes.json",
    )

    standup_p = subparsers.add_parser("standup", help="Run a Daily Standup (alias of --standup-run, more knobs)")
    standup_p.add_argument("--session", default="", metavar="ID", help="Session to use (default: most recent)")
    standup_p.add_argument(
        "--solo",
        action="store_true",
        help="A one-person standup: your own activity only, first-person summary (the Solo world)",
    )
    standup_p.add_argument("--deliver", action="store_true", help="Send to the configured channels (default: print)")
    _add_context_flags(standup_p)
    standup_p.add_argument(
        "--set-context",
        dest="set_context",
        default="",
        metavar="SPEC",
        help="Save the session's standup scope ('inherit' clears it) instead of running",
    )
    standup_p.add_argument(
        "--channels", nargs="+", choices=["terminal", "desktop", "slack", "email"], help="Override delivery channels"
    )
    standup_p.add_argument("--days", type=int, default=0, help="Activity look-back window override")
    standup_p.add_argument(
        "--tracker-sources",
        nargs="+",
        choices=["jira", "azure_devops"],
        help="Override saved standup tracker sources",
    )
    standup_p.add_argument(
        "--team-members",
        nargs="+",
        metavar="NAME",
        help="Override the saved authoritative team roster for this run",
    )
    standup_p.add_argument(
        "--code-sources",
        nargs="+",
        choices=["github", "azure_devops"],
        help="Override saved code providers for this run",
    )
    standup_p.add_argument(
        "--github-owners",
        nargs="+",
        metavar="OWNER",
        help="Override saved GitHub owner/organisation scope (covers every active repo inside each)",
    )
    standup_p.add_argument(
        "--github-repositories",
        nargs="+",
        metavar="OWNER/REPO",
        help="Override saved GitHub repository scope (exact repos, unioned with --github-owners)",
    )
    standup_p.add_argument(
        "--github-excluded-repositories",
        nargs="+",
        metavar="OWNER/REPO",
        help="Override saved GitHub repos to drop from an included owner's expansion",
    )
    standup_p.add_argument(
        "--azdo-projects",
        nargs="+",
        metavar="PROJECT",
        help="Override saved Azure DevOps project scope (all repositories in each project)",
    )
    standup_p.add_argument(
        "--azdo-repositories",
        nargs="+",
        metavar="PROJECT/REPO",
        help="Legacy Azure Repos override",
    )
    standup_p.add_argument(
        "--documentation-sources",
        nargs="+",
        choices=["confluence", "notion"],
        help="Override saved documentation providers for this run",
    )
    standup_p.add_argument(
        "--list-members",
        action="store_true",
        help="List discovered members for the selected/saved tracker sources and exit",
    )
    standup_p.add_argument(
        "--schedule",
        choices=["install", "remove", "status"],
        help="Manage the OS schedule (launchd/cron) that runs the standup daily, instead of running one now",
    )
    standup_p.add_argument(
        "--no-transcript-review",
        dest="review_transcripts",
        action="store_false",
        default=True,
        help="Skip the pre-standup review of any unreviewed meeting transcripts",
    )
    standup_p.add_argument("--strict", action="store_true", help="Exit 3 on a degraded run (warnings present)")
    standup_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    # ── standup-review ────────────────────────────────────────────────────
    # A sibling subcommand rather than `standup review`: `standup` is a leaf
    # parser and converting it into a parser-of-parsers would break the
    # existing `yeaboi standup --deliver` form.
    review_p = subparsers.add_parser(
        "standup-review",
        help="Review standup meeting transcripts to find what standup missed, and why",
    )
    review_p.add_argument("--session", default="", metavar="ID", help="Session to use (default: most recent)")
    # A bare `yeaboi standup-review meeting.vtt` is the shape people reach for
    # first, and it is what a dragged file produces.
    review_p.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="Transcript files to review (same as --transcript; '-' reads the transcript from stdin)",
    )
    review_p.add_argument(
        "--transcript",
        dest="transcript_paths",
        nargs="+",
        metavar="PATH",
        help="Review specific transcript files instead of sweeping the transcript folders",
    )
    review_p.add_argument(
        "--transcript-text",
        default="",
        metavar="TEXT",
        help="Review transcript text directly; it is saved to ~/.yeaboi/transcripts first",
    )
    review_p.add_argument(
        "--transcript-dir",
        default="",
        metavar="DIR",
        help="An extra transcript folder for this run (~/.yeaboi/transcripts is always swept)",
    )
    review_p.add_argument(
        "--date",
        dest="standup_date",
        default="",
        metavar="YYYY-MM-DD",
        help="Attribute transcripts to this standup date when their own date can't be inferred",
    )
    review_p.add_argument(
        "--max-transcripts",
        type=int,
        default=5,
        help="Cap on distinct standup dates reviewed (one AI call each)",
    )
    review_p.add_argument(
        "--include-reviewed",
        action="store_true",
        help="Re-review transcripts that have already been processed",
    )
    review_p.add_argument(
        "--file-issues",
        action="store_true",
        help="File the drafted gaps as GitHub issues (writes to a PUBLIC repo; off by default)",
    )
    review_p.add_argument(
        "--list-gaps",
        action="store_true",
        help="List past reviews and the gap→issue ledger instead of running a review",
    )
    review_p.add_argument("--strict", action="store_true", help="Exit 3 on a degraded run (warnings present)")
    review_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    perf_p = subparsers.add_parser(
        "perf",
        help=f"Performance mode {BETA_TAG}: 1:1 prep/completion, reviews, notes",
        description=PERFORMANCE_BETA_NOTICE,
    )
    perf_sub = perf_p.add_subparsers(dest="perf_command", metavar="{roster,prep,complete,review,note}", required=True)
    # Every child carries the same description: `yeaboi perf prep --help` is a
    # perfectly normal place to arrive without ever seeing the parent's help.
    perf_sub.add_parser(
        "roster",
        help="List the engineer roster from recent tracker assignees",
        description=PERFORMANCE_BETA_NOTICE,
    )
    prep_p = perf_sub.add_parser("prep", help="Prepare a 1:1 for an engineer", description=PERFORMANCE_BETA_NOTICE)
    prep_p.add_argument("engineer", help="Engineer name (see `yeaboi perf roster`)")
    prep_p.add_argument("--session", default="", metavar="ID", help="Session for team context (default: most recent)")
    prep_p.add_argument("--jira-project", default="", metavar="KEY", help="Jira project key override")
    prep_p.add_argument("--azdo-project", default="", metavar="NAME", help="Azure DevOps project override")
    prep_p.add_argument(
        "--deep-scan",
        action="store_true",
        help="Also live-scan the stretch no saved standup covered (slower, costs API calls)",
    )
    prep_p.add_argument("--strict", action="store_true", help="Exit 3 on a degraded run (warnings present)")
    _add_context_flags(prep_p)
    complete_p = perf_sub.add_parser(
        "complete", help="Complete a held 1:1 from its transcript", description=PERFORMANCE_BETA_NOTICE
    )
    complete_p.add_argument("engineer", help="Engineer name")
    complete_p.add_argument(
        "--transcript", required=True, metavar="TEXT", help="Transcript text; @file.txt reads from file"
    )
    complete_p.add_argument("--deliver", action="store_true", help="Email the summary via the configured SMTP")
    complete_p.add_argument("--session", default="", metavar="ID", help="Session for team context")
    complete_p.add_argument(
        "--images", nargs="+", default=[], metavar="PATH", help="Whiteboard/notes photos to attach to the summary"
    )
    complete_p.add_argument(
        "--recipients", nargs="+", default=None, metavar="EMAIL", help="Email recipients override (with --deliver)"
    )
    complete_p.add_argument("--strict", action="store_true", help="Exit 3 on a degraded run (warnings present)")
    _add_context_flags(complete_p, scope=False)
    review_p = perf_sub.add_parser(
        "review", help="Draft a periodic performance review", description=PERFORMANCE_BETA_NOTICE
    )
    review_p.add_argument("engineer", help="Engineer name")
    review_p.add_argument("--months", type=int, default=6, help="Review period in months (default 6)")
    review_p.add_argument("--session", default="", metavar="ID", help="Session for team context")
    review_p.add_argument("--jira-project", default="", metavar="KEY", help="Jira project key override")
    review_p.add_argument("--azdo-project", default="", metavar="NAME", help="Azure DevOps project override")
    review_p.add_argument(
        "--deep-scan",
        action="store_true",
        help="Also live-scan the stretch no saved standup covered (slower, costs API calls)",
    )
    review_p.add_argument("--strict", action="store_true", help="Exit 3 on a degraded run (warnings present)")
    _add_context_flags(review_p)
    note_p = perf_sub.add_parser("note", help="Record a note about an engineer", description=PERFORMANCE_BETA_NOTICE)
    note_p.add_argument("engineer", help="Engineer name")
    note_p.add_argument("--text", required=True, help="The note text")

    retro_p = subparsers.add_parser("retro", help="Read past retrospectives (the live board runs in the TUI)")
    retro_p.add_argument("--session", default="", metavar="ID", help="Session to read (default: most recent)")
    retro_p.add_argument("--limit", type=int, default=10, help="Number of past retros to show (default 10)")
    # dest stays "export"; the flag is spelled out because a bare "--export"
    # abbreviation-collides with the top-level --export-questionnaire/--export-only
    # under argparse's prefix matching (Python <3.14).
    retro_p.add_argument(
        "--export-latest",
        dest="export",
        action="store_true",
        help="Also export the latest retro to Markdown + HTML",
    )
    retro_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    poker_p = subparsers.add_parser("poker", help="Read past poker sessions (the live voting board runs in the TUI)")
    poker_p.add_argument("--session", default="", metavar="ID", help="Only show sessions recorded under this id")
    poker_p.add_argument("--limit", type=int, default=10, help="Number of past sessions to show (default 10)")
    # dest stays "export"; spelled out for the same argparse prefix-collision
    # reason as the retro subcommand above.
    poker_p.add_argument(
        "--export-latest",
        dest="export",
        action="store_true",
        help="Also export the latest poker session to Markdown + HTML",
    )
    poker_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    # ── review (Weekly Review — the Solo world's own mode) ─────────────────
    review_root = subparsers.add_parser(
        "review",
        help=f"Weekly self-review of your own week — the Solo world {BETA_TAG}",
        description=WEEKLY_REVIEW_BETA_NOTICE,
    )
    review_sub = review_root.add_subparsers(dest="review_command", metavar="{run,history,export}")
    review_run_p = review_sub.add_parser(
        "run", help="Review this week (or the week of --week-end)", description=WEEKLY_REVIEW_BETA_NOTICE
    )
    review_run_p.add_argument("--session", default="", metavar="ID", help="Session to scope by (default: newest)")
    review_run_p.add_argument(
        "--week-end",
        dest="week_end",
        default="",
        metavar="YYYY-MM-DD",
        help="A date inside the week to review — its Monday through this date (default: today)",
    )
    review_run_p.add_argument(
        "--mark",
        action="append",
        default=[],
        metavar="ID=STATUS",
        help="Mark one of last review's actions: done, dropped, pending or carried (repeatable; ids from history)",
    )
    review_run_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    review_run_p.add_argument("--strict", action="store_true", help="Exit 3 on a degraded run (warnings present)")
    _add_context_flags(review_run_p)
    review_hist_p = review_sub.add_parser("history", help="List past weekly reviews and last week's open actions")
    review_hist_p.add_argument("--session", default="", metavar="ID", help="Session to scope by (default: all)")
    review_hist_p.add_argument("--limit", type=int, default=12, help="Number of past reviews to show (default 12)")
    review_hist_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    review_exp_p = review_sub.add_parser("export", help="Export one weekly review to Markdown")
    review_exp_p.add_argument("--run-id", dest="run_id", type=int, default=0, help="Review to export (0 = latest)")

    # ── ask (Niko, the global assistant) ──────────────────────────────────
    ask_p = subparsers.add_parser(
        "ask",
        help="Ask Niko a question about yeaboi or your own data (read-only)",
    )
    # nargs="*" rather than "+" so `--list` needs no dummy question; the handler
    # rejects a genuinely empty ask.
    ask_p.add_argument("question", nargs="*", help='What to ask, e.g. yeaboi ask "what did my agents cost?"')
    ask_p.add_argument(
        "--conversation",
        default="",
        metavar="ID",
        help="Continue this conversation (see --list); default starts a new one",
    )
    ask_p.add_argument("--continue", dest="continue_latest", action="store_true", help="Continue the newest thread")
    ask_p.add_argument(
        "--route",
        default="",
        metavar="PATH",
        help="Where you are, e.g. /agents/usage — colours the answer",
    )
    # Spelled out rather than "--list": a bare --list is a strict prefix of the
    # top-level --list-sessions and --list-audio-devices, which argparse's
    # pre-scan rejects as ambiguous on Python <3.14 (the same trap retro's
    # --export-latest exists for).
    ask_p.add_argument(
        "--list-conversations",
        dest="list_only",
        action="store_true",
        help="List past conversations and exit",
    )
    ask_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    ceremonies_p = subparsers.add_parser(
        "ceremonies",
        help="Run a mode on a cadence — declare it once, the OS fires it",
    )
    ceremonies_sub = ceremonies_p.add_subparsers(dest="ceremonies_command", required=True)

    cer_list = ceremonies_sub.add_parser("list", help="Show this session's ceremonies and what they last did")
    cer_list.add_argument("--session", default="", metavar="ID", help="Session to read (default: most recent)")
    cer_list.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    cer_add = ceremonies_sub.add_parser("add", help="Declare a ceremony and install its job")
    cer_add.add_argument("name", help="A short name — lowercase letters, digits, dot, dash, underscore")
    cer_add.add_argument("--mode", required=True, help="Which mode runs (see `ceremonies modes`)")
    cer_add.add_argument("--at", default="09:00", metavar="HH:MM", help="Local time it should happen (default 09:00)")
    cer_add.add_argument("--weekdays", default="", metavar="SPEC", help="e.g. 1-5, 1,3,5 (default: the mode's own)")
    cer_add.add_argument(
        "--channels",
        default="terminal",
        metavar="LIST",
        help="Comma-separated: terminal, desktop, slack, email (default terminal)",
    )
    cer_add.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="A mode argument; repeatable (see `ceremonies modes`)",
    )
    cer_add.add_argument("--session", default="", metavar="ID", help="Session to attach it to")
    cer_add.add_argument(
        "--stale-after",
        type=int,
        default=120,
        metavar="MIN",
        help="Skip a scheduled run this many minutes late (0 disables; default 120)",
    )
    cer_add.add_argument(
        "--monthly-cap", type=float, default=0.0, metavar="USD", help="Skip scheduled runs past this month's spend"
    )
    cer_add.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    cer_rm = ceremonies_sub.add_parser("remove", help="Forget a ceremony and tear its job down")
    cer_rm.add_argument("name")
    cer_rm.add_argument("--session", default="", metavar="ID")
    cer_rm.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    for verb, blurb in (("pause", "Stop a ceremony firing, keeping it declared"), ("resume", "Start it again")):
        sub = ceremonies_sub.add_parser(verb, help=blurb)
        sub.add_argument("name")
        sub.add_argument("--session", default="", metavar="ID")
        sub.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    cer_skip = ceremonies_sub.add_parser(
        "skip",
        help="Skip the next occurrence only — the schedule and the job stay exactly as they are",
    )
    cer_skip.add_argument("name")
    cer_skip.add_argument(
        "--on",
        default="",
        metavar="YYYY-MM-DD",
        help="Skip this date's run instead of the next one; '' clears a pending skip",
    )
    cer_skip.add_argument("--clear", action="store_true", help="Cancel a pending skip")
    cer_skip.add_argument("--session", default="", metavar="ID")
    cer_skip.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    cer_run = ceremonies_sub.add_parser("run", help="Run one now (this is what the scheduled job invokes)")
    cer_run.add_argument("name")
    cer_run.add_argument("--session", default="", metavar="ID")
    cer_run.add_argument(
        "--scheduled",
        action="store_true",
        help="Arm the guards a fired run needs: staleness, the monthly cap, and pause",
    )
    cer_run.add_argument("--dry-run", action="store_true", help="Run the engine without LLM calls or delivery")
    cer_run.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    cer_hist = ceremonies_sub.add_parser("history", help="What the ceremonies actually did")
    cer_hist.add_argument("name", nargs="?", default="", help="Only this ceremony (default: all)")
    cer_hist.add_argument("--session", default="", metavar="ID")
    cer_hist.add_argument("--limit", type=int, default=20, help="Rows to show (default 20)")
    cer_hist.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    cer_modes = ceremonies_sub.add_parser("modes", help="Which modes can run on a cadence, and which cannot")
    cer_modes.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    connections_p = subparsers.add_parser(
        "connections",
        help="Read-only integrations: what is connected, and what it can see",
    )
    connections_sub = connections_p.add_subparsers(dest="connections_command", required=True)

    conn_list = connections_sub.add_parser("list", help="What is connected (--all for the whole catalog)")
    conn_list.add_argument("--family", default="", help="Only this family (e.g. observability)")
    conn_list.add_argument(
        "--all", action="store_true", help="The whole catalog: connectors you have not set up + built-in integrations"
    )
    conn_list.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    conn_show = connections_sub.add_parser("show", help="One connector: its fields and whether each is set")
    conn_show.add_argument("name", help="Connector name (see `connections list --all`)")
    conn_show.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    conn_verify = connections_sub.add_parser("verify", help="Live-check a connection (omit the name for all of them)")
    conn_verify.add_argument("name", nargs="?", default="", help="Connector name; default: every connected one")
    conn_verify.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    # No --token flag anywhere: a credential on argv lands in shell history and
    # the process table. Values are prompted for, exactly as `--setup` does.
    conn_add = connections_sub.add_parser("add", help="Connect one, prompting for each value")
    conn_add.add_argument("name", help="Connector name (see `connections list --all`)")

    conn_fetch = connections_sub.add_parser(
        "fetch", help="What a connection saw over a window — read-only, nothing is stored"
    )
    conn_fetch.add_argument("name", nargs="?", default="", help="Connector name; default: every connected one")
    conn_fetch.add_argument(
        "--since", default="14d", metavar="WINDOW", help="How far back to look: 14d, 48h, 2w (default 14d)"
    )
    conn_fetch.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    conn_create = connections_sub.add_parser(
        "create", help="Create your own connection — a generic API the catalog then carries"
    )
    conn_create.add_argument(
        "--from-json", dest="from_json", default="", metavar="FILE", help="A descriptor JSON file instead of prompts"
    )

    conn_wurl = connections_sub.add_parser(
        "webhook-url", help="Where a webhook connection's deliveries go — URL, header and secret"
    )
    conn_wurl.add_argument("name", help="Connection name (a custom webhook connection)")
    conn_wurl.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    conn_wtest = connections_sub.add_parser(
        "webhook-test", help="Send one synthetic signed delivery through the local receiver"
    )
    conn_wtest.add_argument("name", help="Connection name (a custom webhook connection)")

    conn_rm = connections_sub.add_parser("remove", help="Forget a connection's stored values")
    conn_rm.add_argument("name")
    conn_rm.add_argument("--yes", action="store_true", help="Do not ask for confirmation")

    webhooks_p = subparsers.add_parser(
        "webhooks",
        help="The inbound receiver for webhook connections (loopback; --share tunnels it)",
    )
    webhooks_sub = webhooks_p.add_subparsers(dest="webhooks_command", required=True)
    wh_serve = webhooks_sub.add_parser("serve", help="Run the receiver in the foreground (Ctrl+C stops)")
    wh_serve.add_argument("--port", type=int, default=0, help="Override the fixed port (default 8642)")
    wh_serve.add_argument(
        "--share", action="store_true", help="Also open a cloudflared quick tunnel (URL rotates per share)"
    )

    slack_p = subparsers.add_parser(
        "slack",
        help="Two-way Slack — read reactions and replies on what yeaboi posted",
    )
    slack_sub = slack_p.add_subparsers(dest="slack_command", required=True)

    slack_poll = slack_sub.add_parser("poll", help="Read Slack once and apply what is new (the scheduled job)")
    slack_poll.add_argument(
        "--scheduled",
        action="store_true",
        # Accepted and inert, deliberately. `ceremonies run` needs this flag
        # because an unattended fire raises questions a human at a terminal has
        # already answered; a poll raises none of them, so its overlap lock and
        # gap notice are unconditional. The installed job's argv carries it, so
        # removing it would break every job already on disk at upgrade.
        help="Accepted for symmetry with `ceremonies run`; a poll's guards are always armed",
    )
    slack_poll.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    slack_watch = slack_sub.add_parser("watch", help="Install, remove or inspect the recurring poll")
    watch_group = slack_watch.add_mutually_exclusive_group(required=True)
    watch_group.add_argument("--install", action="store_true", help="Install the recurring poll")
    watch_group.add_argument("--remove", action="store_true", help="Tear the poll down (nothing else)")
    watch_group.add_argument("--status", action="store_true", help="Report what is installed")
    slack_watch.add_argument(
        "--every",
        type=int,
        default=10,
        metavar="MIN",
        help="Minutes between polls; must divide 60 (default 10)",
    )
    slack_watch.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    slack_check = slack_sub.add_parser("check", help="Is two-way configured, and can it see the channel?")
    slack_check.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    slack_hist = slack_sub.add_parser("history", help="What Slack asked for, and what happened to it")
    slack_hist.add_argument("--limit", type=int, default=20, help="Rows to show (default 20)")
    slack_hist.add_argument("--pending", action="store_true", help="Only events claimed but never settled")
    slack_hist.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    slack_members = slack_sub.add_parser("members", help="List the workspace's people and their member ids")
    slack_members.add_argument("--match", default="", metavar="TEXT", help="Only those whose name or handle matches")
    slack_members.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    slack_link = slack_sub.add_parser("link", help="Say which team member a Slack user is (for attributing replies)")
    slack_link.add_argument("slack_user", nargs="?", default="", metavar="U0123456789", help="Slack member id")
    slack_link.add_argument("member", nargs="?", default="", help="Their name on this session's roster")
    slack_link.add_argument("--session", default="", metavar="ID", help="Session to link in (default: latest)")
    slack_link.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    slack_unlink = slack_sub.add_parser("unlink", help="Forget which team member a Slack user is")
    slack_unlink.add_argument("slack_user", metavar="U0123456789", help="Slack member id")
    slack_unlink.add_argument("--session", default="", metavar="ID", help="Session to unlink in (default: latest)")
    slack_unlink.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    agents_p = subparsers.add_parser(
        "agents",
        help=f"Agents mode {BETA_TAG}: your AI-native SDLC teammate (cost, recoverable spend, security posture)",
        description=AGENTWATCH_BETA_NOTICE,
    )
    agents_sub = agents_p.add_subparsers(dest="agents_command", metavar="{cost,advisor,security}", required=True)
    # Every child carries the same description — `yeaboi agents cost --help` is
    # a perfectly normal place to arrive without ever seeing the parent's help.
    cost_p = agents_sub.add_parser(
        "cost",
        help="What your agents cost: per-model/project/source breakdowns + daily trend",
        description=AGENTWATCH_BETA_NOTICE,
    )
    cost_p.add_argument("--window-days", type=int, default=30, metavar="N", help="Days to look back (default 30)")
    cost_p.add_argument("--project", default="", metavar="NAME", help="Filter by project directory name (substring)")
    cost_p.add_argument("--repo", default="", metavar="PATH", help="Only sessions under this repository path")
    cost_p.add_argument("--source", default="", choices=["", "claude_code"], help="Filter by telemetry source")
    cost_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    cost_p.add_argument("--strict", action="store_true", help="Exit 3 on a degraded run (warnings present)")
    advisor_p = agents_sub.add_parser(
        "advisor",
        help="How much of your agent spend is recoverable: Read waste, cache health, prompt-prefix churn",
        description=AGENTWATCH_BETA_NOTICE,
    )
    advisor_p.add_argument("--window-days", type=int, default=30, metavar="N", help="Days to look back (default 30)")
    advisor_p.add_argument("--repo", default="", metavar="PATH", help="Only sessions under this repository path")
    advisor_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    advisor_p.add_argument("--strict", action="store_true", help="Exit 3 on a degraded run (warnings present)")
    asec_p = agents_sub.add_parser(
        "security",
        help="Audit your agent setup: permissions, MCP servers, secrets exposure, risky commands",
        description=AGENTWATCH_BETA_NOTICE,
    )
    asec_p.add_argument("--deep", action="store_true", help="Re-scan every transcript, not just new/changed ones")
    asec_p.add_argument("--show-info", dest="include_info", action="store_true", help="List informational findings too")
    asec_p.add_argument(
        "--dismiss", default="", metavar="KEY", help="Dismiss one finding by its key (needs --reason); no scan runs"
    )
    asec_p.add_argument("--reason", default="", metavar="TEXT", help="Why the dismissed finding is expected")
    asec_p.add_argument("--undismiss", default="", metavar="KEY", help="Restore a dismissed finding; no scan runs")
    asec_p.add_argument("--list-dismissed", action="store_true", help="Print the dismissals on file; no scan runs")
    asec_p.add_argument("--replay", default="", metavar="KEY", help="Print the transcript turns around one finding")
    asec_p.add_argument("--line", type=int, default=0, metavar="N", help="With --replay: a specific matching line")
    asec_p.add_argument("--signals", default="", metavar="KEY", help="List every matching line behind one finding")
    asec_p.add_argument("--fix", default="", metavar="KEY", help="Apply a fix to one finding (needs --fix-id)")
    asec_p.add_argument("--fix-id", default="", metavar="ID", help="Which fix: one of the finding's fixes[].id")
    asec_p.add_argument("--repo", default="", metavar="PATH", help="With --fix: the repository a PR fix targets")
    asec_p.add_argument(
        "--mark-test-data", nargs="*", default=[], metavar="KEY", help="Set findings aside as test data; no scan runs"
    )
    asec_p.add_argument("--list-fixes", action="store_true", help="Print the fixes applied so far; no scan runs")
    asec_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    asec_p.add_argument("--strict", action="store_true", help="Exit 3 on a degraded run (warnings present)")

    provenance_desc = (
        "Every deterministic signal yeaboi surfaces (practice nudges, blocker flags, confidence "
        "adjustments, conflict cards, performance preps and reviews) is recorded in a tamper-evident "
        "hash chain with its evidence. This command verifies and reads that chain — deterministic, "
        "local, no LLM involved."
    )
    provenance_p = subparsers.add_parser(
        "provenance",
        help="Audit the tamper-evident decision chain behind yeaboi's signals",
        description=provenance_desc,
    )
    provenance_sub = provenance_p.add_subparsers(dest="provenance_command", metavar="{audit,trace}", required=True)
    # Every child carries the same description — `yeaboi provenance audit --help`
    # is a perfectly normal place to arrive without ever seeing the parent's help.
    paudit_p = provenance_sub.add_parser(
        "audit",
        help="Verify every chain link and summarise the recorded decisions",
        description=provenance_desc,
    )
    paudit_p.add_argument("--window-days", type=int, default=30, metavar="N", help="Days to look back (default 30)")
    paudit_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    paudit_p.add_argument(
        "--strict", action="store_true", help="Exit 3 on a broken chain or an empty one (warnings present)"
    )
    ptrace_p = provenance_sub.add_parser(
        "trace",
        help='The "why" trail behind one recorded decision, evidence included',
        description=provenance_desc,
    )
    ptrace_p.add_argument("entity_id", metavar="ENTITY", help="Entity id, as listed by `yeaboi provenance audit`")
    ptrace_p.add_argument("--depth", type=int, default=2, metavar="N", help="Evidence hops to follow (default 2)")
    ptrace_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    ship_desc = (
        "Hand an epic, story or task from your saved plan to a supervised coding agent (Claude Code headless): "
        "isolated worktree and branch, deterministic validation, a human approval gate at the terminal, "
        "and a PR only after approval. A user-global launch budget caps runs."
    )
    ship_p = subparsers.add_parser(
        "ship",
        help="Implement an epic, story or task from your plan via a supervised coding agent",
        description=ship_desc,
    )
    ship_sub = ship_p.add_subparsers(dest="ship_command", metavar="{run,resume,status,history}", required=True)
    srun_p = ship_sub.add_parser("run", help="Run one plan item through the pipeline", description=ship_desc)
    srun_p.add_argument(
        "item_id", metavar="ITEM", help="Epic, story or task id from the plan (e.g. F1, US-001, T-US-001-01)"
    )
    srun_p.add_argument(
        "--level",
        choices=["epic", "story", "task"],
        default="",
        help="Disambiguate a colliding id; inferred from the plan by default",
    )
    srun_p.add_argument(
        "--split",
        action="store_true",
        help="Epics only: one stacked PR per story instead of one PR for the whole epic",
    )
    srun_p.add_argument("--repo", default=".", metavar="PATH", help="Target git repository (default: current dir)")
    srun_p.add_argument("--session", default="", metavar="ID", help="Planning session id (default: latest)")
    srun_p.add_argument(
        "--check", default="", metavar="CMD", help="Validation command run in the worktree (e.g. 'make test')"
    )
    srun_p.add_argument("--timeout-minutes", type=int, default=30, metavar="N", help="Agent run timeout (default 30)")
    srun_p.add_argument("--dry-run", action="store_true", help="Canned run — no agent, no git, no network")
    srun_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    srun_p.add_argument("--strict", action="store_true", help="Exit 3 when the run did not end approved")
    sresume_p = ship_sub.add_parser(
        "resume",
        help="Finish a run left waiting at the approval gate",
        description=(
            "Re-attach to a run whose process died at the approval gate. Its branch, worktree and diff "
            "are still on disk; this re-asks the gate and, on approval, pushes and opens the PR."
        ),
    )
    sresume_p.add_argument("run_id", metavar="RUN", help="Run id from `yeaboi ship history`")
    sresume_p.add_argument(
        "--check",
        default="",
        metavar="CMD",
        help="Re-run this validation command before the gate (default: reuse the recorded verdict)",
    )
    sresume_p.add_argument(
        "--timeout-minutes", type=int, default=30, metavar="N", help="Agent run timeout (default 30)"
    )
    sresume_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    sresume_p.add_argument("--strict", action="store_true", help="Exit 3 when the run did not end approved")
    sstatus_p = ship_sub.add_parser("status", help="The latest run and the launch budget", description=ship_desc)
    sstatus_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    shistory_p = ship_sub.add_parser("history", help="Recent runs, newest first", description=ship_desc)
    shistory_p.add_argument("--limit", type=int, default=10, metavar="N", help="Runs to show (default 10)")
    shistory_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    analyze_p = subparsers.add_parser("analyze", help="Analyse team board history into a calibration profile")
    analyze_p.add_argument(
        "--source",
        choices=["jira", "azdevops", "both"],
        default="",
        help="Tracker: jira, azdevops, or both (default: auto-detect a single tracker)",
    )
    analyze_p.add_argument("--project", default="", metavar="KEY", help="Project key (default: configured)")
    analyze_p.add_argument("--sprints", type=int, default=8, help="Closed sprints to analyse (default 8)")
    analyze_p.add_argument(
        "--depth",
        choices=["quick", "deep"],
        default="deep",
        help="Analysis depth: deep provides exhaustive AI enrichment; quick is metrics-only (default deep)",
    )
    analyze_p.add_argument(
        "--window-days",
        type=int,
        default=120,
        help="Changed-content window shared by Code and Docs (default 120)",
    )
    analyze_p.add_argument(
        "--analysis-model",
        default=None,
        metavar="MODEL",
        help="Per-run model for structured Analysis tasks (final synthesis still uses the primary model)",
    )
    analyze_p.add_argument(
        "--features",
        nargs="+",
        choices=["delivery", "ai_footprint", "code_health", "documentation", "operational"],
        default=None,
        metavar="FEATURE",
        help="Analysis areas to run (default: all supported by the selected integrations)",
    )
    analyze_p.add_argument("--github-owner", action="append", default=None, metavar="OWNER")
    analyze_p.add_argument("--azdo-code-project", action="append", default=None, metavar="PROJECT")
    analyze_p.add_argument("--confluence-space", action="append", default=None, metavar="SPACE")
    analyze_p.add_argument("--notion-root", action="append", default=None, metavar="PAGE_ID")
    analyze_p.add_argument(
        "--samples",
        action="store_true",
        help="Also generate sample tickets (requires --depth deep; extra LLM calls)",
    )
    analyze_p.add_argument("--no-insights", action="store_true", help="Skip the coaching-insights LLM call")
    analyze_p.add_argument(
        "--no-ai-usage",
        dest="include_ai_usage",
        action="store_false",
        help="Skip the AI-adoption scan (commit/PR AI-tool markers)",
    )
    analyze_p.add_argument(
        "--no-doc-quality",
        dest="include_doc_quality",
        action="store_false",
        help="Skip the documentation usefulness and clarity scan",
    )
    # Each component runs over its OWN sub-sources (not the tracker): delivery ←
    # jira/azdevops boards, code ← github/azdo repos, docs ← confluence/notion.
    analyze_p.add_argument(
        "--delivery",
        nargs="+",
        choices=["jira", "azdevops"],
        default=None,
        metavar="TRACKER",
        help="Delivery (velocity/calibration) trackers to analyse. e.g. --delivery jira",
    )
    analyze_p.add_argument(
        "--code",
        nargs="+",
        choices=["github", "azdo"],
        default=None,
        metavar="HOST",
        help="Code hosts for the AI-usage scan. e.g. --code github azdo",
    )
    analyze_p.add_argument(
        "--docs",
        nargs="+",
        choices=["confluence", "notion"],
        default=None,
        metavar="PLATFORM",
        help="Doc platforms for the clarity/usefulness read. e.g. --docs confluence",
    )
    analyze_p.add_argument(
        "--ops",
        nargs="+",
        default=None,
        metavar="CONNECTOR",
        help=(
            "Monitoring connectors for the Operations rate. Validated against what is "
            "actually connected. e.g. --ops pagerduty sentry"
        ),
    )
    analyze_p.add_argument(
        "--members",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Selected members for delivery and code. Code analysis is empty without "
            "an explicit member scope; it never falls back to whole-team activity."
        ),
    )
    analyze_p.add_argument("--strict", action="store_true", help="Exit 3 on a degraded run (warnings present)")
    _add_context_flags(analyze_p)
    analyze_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    app_p = subparsers.add_parser(
        "app",
        help="Run the local desktop backend (spawned by the yeaboi desktop app)",
        description=(
            "Binds a loopback HTTP server over the same engines the TUI, CLI and MCP "
            "server use, and prints one YEABOI_APP_READY handshake line to stdout. "
            "Meant to be spawned by the desktop shell, not run by hand."
        ),
    )
    app_p.add_argument("--port", type=int, default=0, help="Port to bind on 127.0.0.1 (default 0 = ephemeral)")

    return parser


def _run_headless(args: argparse.Namespace) -> None:
    """Run the full pipeline headlessly — no TUI, no interactive input.

    Pre-populates a QuestionnaireState from CLI args (--description,
    --team-size, --sprint-length), then delegates to run_repl() with
    export_only=True and non_interactive=True.

    When --output json, Rich console output goes to stderr so only
    JSON is written to stdout.

    # See docs: "Architecture" — headless mode for CI/CD pipelines
    """
    from rich.console import Console

    from yeaboi.formatters import build_theme
    from yeaboi.questionnaire_io import build_questionnaire_from_answers
    from yeaboi.repl import run_repl

    output_format = args.output or "markdown"

    # When JSON output, redirect console to stderr so stdout is clean JSON
    if output_format == "json":
        console = Console(theme=build_theme(args.theme), file=sys.stderr)
    else:
        console = Console(theme=build_theme(args.theme))

    # Pre-populate questionnaire from CLI args
    answers: dict[int, str] = {}
    # Q1 gets the project description
    answers[1] = args.description
    if args.team_size is not None:
        answers[6] = str(args.team_size)
    if args.sprint_length is not None:
        answers[8] = str(args.sprint_length)

    # Load SCRUM.md from working directory if present — fills gaps the CLI
    # args didn't cover (e.g., tech stack, integrations, constraints).
    # Uses deterministic keyword extraction only (no LLM call) to stay fast.
    # CLI args always take priority over SCRUM.md.
    try:
        from yeaboi.agent.nodes import _keyword_extract_fallback, _load_user_context

        scrum_md_content, _ = _load_user_context()
        if scrum_md_content:
            scrum_extracted: dict[int, str] = {}
            _keyword_extract_fallback(scrum_md_content, scrum_extracted)
            # Merge: CLI args win over SCRUM.md
            for q_num, answer in scrum_extracted.items():
                if q_num not in answers:
                    answers[q_num] = answer
    except Exception:
        pass  # best-effort — never block headless mode

    questionnaire = build_questionnaire_from_answers(answers)

    run_repl(
        console=console,
        questionnaire=questionnaire,
        intake_mode="quick",
        export_only=True,
        bell=False,
        theme=args.theme,
        non_interactive=True,
        output_format=output_format,
        prior_art=args.prior_art,
        solo=bool(getattr(args, "solo", False)),
        **_context_kwargs(args),
    )


def _run_standup(args: argparse.Namespace) -> int:
    """Run a Daily Standup headlessly and deliver it. Returns a process exit code.

    This is what the OS scheduler (launchd plist / crontab entry) invokes at the
    configured time — even when the interactive app is closed. It resolves the
    target session (``--standup-session`` or the most recent), runs the engine,
    and delivers to the configured (or ``--standup-output``-overridden) channels.

    Exit codes: 0 = delivered, 2 = no session found, 1 = unexpected error.

    # See docs: "Daily Standup" — scheduling, headless run
    """
    from yeaboi.logging_setup import attach_mode_handler, configure_logging
    from yeaboi.paths import get_db_path
    from yeaboi.sessions import SessionStore

    # Route standup records to logs/standup/standup.log (rotating) alongside the
    # main TUI log, so scheduled runs are auditable. Level comes from LOG_LEVEL
    # (default WARNING) — set LOG_LEVEL=INFO in ~/.yeaboi/.env for run-by-run
    # audit detail. The process exits after the run, so no detach is needed.
    configure_logging()
    attach_mode_handler("standup")

    db_path = get_db_path()
    session_id = args.standup_session
    if not session_id or session_id == "latest":
        with SessionStore(db_path) as store:
            session_id = store.get_latest_session_id()
    if not session_id:
        print("Error: no session found to run a standup for.", file=sys.stderr)
        return 2

    # Resolve channel override: "all" expands to every channel.
    channels = None
    if args.standup_output:
        if args.standup_output == "all":
            from yeaboi.standup.delivery import ALL_CHANNELS

            channels = list(ALL_CHANNELS)
        else:
            channels = [args.standup_output]

    # Interactive scheduled run: prompt for the user's update + confirm (timed),
    # then generate + deliver. Falls back to headless when no TTY is attached.
    if getattr(args, "standup_interactive", False):
        from yeaboi.standup.interactive import run_interactive_standup

        return run_interactive_standup(session_id, channels=channels, solo=bool(getattr(args, "solo", False)))

    try:
        from yeaboi.standup.engine import run_standup

        report = run_standup(session_id, channels=channels, deliver=True, solo=bool(getattr(args, "solo", False)))
        warn = f" ({len(report.warnings)} notice(s))" if report.warnings else ""
        print(
            f"Standup delivered for session '{session_id}' (day {report.sprint_day}/{report.sprint_total_days}){warn}."
        )
        return 0
    except Exception as e:
        logging.getLogger(__name__).error("Standup run failed: %s", e, exc_info=True)
        print(f"Error: standup run failed: {e}", file=sys.stderr)
        return 1


def _run_transcript_reminder(args: argparse.Namespace) -> int:
    """Post one desktop reminder if standups went unchecked. Returns an exit code.

    The second job the OS scheduler installs, firing shortly AFTER the standup —
    the moment the recording has just landed and the meeting is still in mind.

    Deliberately tiny: no LLM, no collectors, no terminal window. It asks the
    same deterministic question the hub card asks, and if the answer is "nothing
    to say" it exits silently. A scheduled job that speaks only when it has
    something to report is one you leave installed.

    # See docs: "Daily Standup" — scheduling
    """
    from yeaboi.logging_setup import attach_mode_handler, configure_logging
    from yeaboi.paths import get_db_path
    from yeaboi.sessions import SessionStore

    configure_logging()
    attach_mode_handler("standup")

    session_id = args.standup_session
    if not session_id or session_id == "latest":
        with SessionStore(get_db_path()) as store:
            session_id = store.get_latest_session_id()
    if not session_id:
        print("Error: no session found to check transcripts for.", file=sys.stderr)
        return 2

    try:
        from yeaboi.standup.delivery import notify_desktop
        from yeaboi.standup.engine import transcript_nudge

        nudge = transcript_nudge(session_id)
        if not nudge:
            logging.getLogger(__name__).info("transcript reminder: nothing to say for %s", session_id)
            return 0
        notify_desktop("Standup transcript", nudge.message)
        print(nudge.message)
        return 0
    except Exception as e:
        logging.getLogger(__name__).error("Transcript reminder failed: %s", e, exc_info=True)
        print(f"Error: transcript reminder failed: {e}", file=sys.stderr)
        return 1


def _setup_access() -> int:
    """Headless twin of Settings → Share Mode → the Cloudflare Access wizard.

    Same engine, plain prints — the shape ``--install-voice`` established. Both
    entry points are human-initiated and one of them opens a browser, which is
    also why neither gets an MCP tool: an agent should not be able to obtain
    account-level Cloudflare authority on someone's behalf.

    Every fact is stored the moment it is known rather than at the end. These
    steps have side effects on a real Cloudflare account, and a tunnel created
    and then abandoned to a Ctrl-C is an orphan this command cannot see it made.
    """
    from yeaboi.sharing import access_setup

    def _ask(prompt: str, *, default: str = "") -> str:
        suffix = f" [{default}]" if default else ""
        try:
            answer = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            raise SystemExit(1) from None
        return answer or default

    state = access_setup.read_state()
    if not state.binary:
        print("Could not obtain cloudflared. Check the network and try again.")
        return 1

    print("Cloudflare Access setup")
    print("  cloudflared:", state.binary)
    print("  signed in:  ", "yes" if state.logged_in else "no")
    print("  PyJWT:      ", "installed" if state.jwt_installed else "missing")
    if state.missing_keys:
        print("  still to set:", ", ".join(state.missing_keys))
    print()

    # 1. Sign in.
    if not state.logged_in:
        print("Requires: a Cloudflare account, and a domain added to it.")
        print(f"  No domain on Cloudflare yet? Add one first: {access_setup.ADD_SITE_URL}")
        print("  (press Add a site — the free plan is enough — and update the")
        print("  nameservers at your registrar as the dashboard instructs).")
        print()
        print("Opening your browser to sign in to Cloudflare.")
        print("  Sign in, pick the domain you want boards served under from the")
        print("  zone list, and press Authorize — then come back here.")
        print("This writes ~/.cloudflared/cert.pem — an account-level credential.")
        outcome = access_setup.login(on_line=lambda phrase: print(f"  {phrase}"))
        print(f"  {outcome.message}")
        if not outcome.ok:
            return 1

    # A stored fact is a done step: re-runs resume at the first missing thing,
    # and changing a stored value is the Settings ▸ Sharing rows' (or .env's) job.
    missing = set(state.missing_keys)

    # 2. The tunnel is decided, not asked: the only one, or the one named
    # "yeaboi", created when none exists. A picker appears only on genuine
    # ambiguity. Settings ▸ Sharing edits the choice afterwards.
    if not missing & {"CLOUDFLARE_TUNNEL_ID", "CLOUDFLARE_TUNNEL_CREDENTIALS", "CLOUDFLARE_ACCESS_HOSTNAME"}:
        from yeaboi.config import access_hostname

        hostname = access_hostname()
        print(f"Tunnel and hostname already set — boards serve at https://{hostname}.")
    elif not missing & {"CLOUDFLARE_TUNNEL_ID", "CLOUDFLARE_TUNNEL_CREDENTIALS"}:
        # Only the hostname is missing. The stored tunnel is kept — re-resolving
        # could pick a different one and silently overwrite the stored id.
        from yeaboi.config import access_tunnel_id

        tunnel_id = access_tunnel_id()
        print(f"Using your stored tunnel ({tunnel_id}).")
        domain = _ask("Your domain on Cloudflare (e.g. acme.com)")
        hostname = access_setup.boards_hostname(domain)
        if not access_setup.valid_hostname(hostname):
            print("  That is not a domain name.")
            return 1
        print(f"Boards will be served at https://{hostname} (change via CLOUDFLARE_ACCESS_HOSTNAME).")
        outcome = access_setup.route_dns(tunnel_id, hostname, on_line=lambda phrase: print(f"  {phrase}"))
        print(f"  {outcome.message}")
        if not outcome.ok:
            if outcome.code != "DNS_EXISTS":
                return 1
            if _ask("Record exists — use this hostname anyway? (y/N)", default="n").lower() not in ("y", "yes"):
                return 1
        access_setup.save(CLOUDFLARE_ACCESS_HOSTNAME=hostname.strip().lower())
    else:
        tunnels, listed = access_setup.list_tunnels()
        if not listed.ok:
            print(f"  {listed.message}")
            return 1
        picked = access_setup.resolve_tunnel(tunnels)
        if picked is not None:
            print(f"Using your existing tunnel '{picked.name}' ({picked.id}).")
        elif tunnels:
            print("Several tunnels and none named 'yeaboi' — which should serve your boards?")
            for i, tunnel in enumerate(tunnels, 1):
                print(f"  {i}. {tunnel.name}  ({tunnel.id})")
            choice = _ask("Number", default="1")
            if not (choice.isdigit() and 1 <= int(choice) <= len(tunnels)):
                print("  That is not one of the numbers.")
                return 1
            picked = tunnels[int(choice) - 1]
        else:
            print(f"Creating a tunnel named '{access_setup.DEFAULT_TUNNEL_NAME}'…")
            picked, outcome = access_setup.create_tunnel(
                access_setup.DEFAULT_TUNNEL_NAME, on_line=lambda phrase: print(f"  {phrase}")
            )
            print(f"  {outcome.message}")
            if picked is None:
                return 1
        credentials = picked.credentials or access_setup.default_credentials(picked.id)

        access_setup.save(CLOUDFLARE_TUNNEL_ID=picked.id, CLOUDFLARE_TUNNEL_CREDENTIALS=credentials)

        # 3. Hostname + DNS route. Only the domain is asked; boards.<domain> is the
        # default nobody has to type, editable later via CLOUDFLARE_ACCESS_HOSTNAME.
        domain = _ask("Your domain on Cloudflare (e.g. acme.com)")
        hostname = access_setup.boards_hostname(domain)
        if not access_setup.valid_hostname(hostname):
            print("  That is not a domain name.")
            return 1
        print(f"Boards will be served at https://{hostname} (change via CLOUDFLARE_ACCESS_HOSTNAME).")
        outcome = access_setup.route_dns(picked.id, hostname, on_line=lambda phrase: print(f"  {phrase}"))
        print(f"  {outcome.message}")
        if not outcome.ok:
            if outcome.code != "DNS_EXISTS":
                return 1
            # Never overwritten silently: the record may point at this tunnel (a
            # re-run) or at something else. Ask rather than assume.
            if _ask("Record exists — use this hostname anyway? (y/N)", default="n").lower() not in ("y", "yes"):
                return 1
        access_setup.save(CLOUDFLARE_ACCESS_HOSTNAME=hostname.strip().lower())

    # 4. The one manual step.
    if not missing & {"CLOUDFLARE_ACCESS_TEAM", "CLOUDFLARE_ACCESS_AUD"}:
        print("Access application already configured — verifying.")
        team = aud = ""
    else:
        print()
        print("Now create the Access application in the Cloudflare dashboard:")
        print(f"  {access_setup.ACCESS_APP_ADD_URL}")
        print("  (Access controls → Applications → Add self-hosted, if the link does not land there)")
        print(f"  1. Name it anything; set the application domain to {hostname}.")
        print("  2. Policy: Action = Allow; under Include pick Emails and list your teammates'")
        print("     addresses — or 'Emails ending in' @your-company.com for everyone. Save it.")
        print("  3. Save the application, open its Overview tab, and copy the Audience (AUD) tag.")
        print()
        # The sign-in redirect carries the team name and the AUD once the
        # application exists — read them rather than ask, and offer the AUD as
        # a default to confirm against the Overview tab.
        team, aud_hint = access_setup.discover_app(hostname)
        if team:
            print(f"  Team name detected from {hostname}: {team}")
        if aud_hint:
            print("  An AUD tag was detected too — confirm it matches the application's Overview tab.")
        aud = _ask("The application's AUD tag", default=aud_hint)
        while not access_setup.valid_aud(aud):
            print("  That is not an AUD tag (64 hex characters, from the application's Overview tab).")
            aud = _ask("The application's AUD tag", default=aud_hint)
        if not team:
            team = _ask("Your Zero Trust team name")
        access_setup.save(CLOUDFLARE_ACCESS_TEAM=team, CLOUDFLARE_ACCESS_AUD=aud)

    # 5. PyJWT, then prove the whole thing works.
    if not access_setup.jwt_installed():
        print("Installing PyJWT…")
        outcome = access_setup.install_jwt(on_line=lambda phrase: print(f"  {phrase}"))
        print(f"  {outcome.message}")
        if not outcome.ok:
            return 1

    # The switch is written only once the tier is proven — see the wizard twin.
    verdict = access_setup.verify(assume_mode=True)
    if verdict.ok:
        from yeaboi.config import SHARE_MODE_ACCESS

        access_setup.save(YEABOI_SHARE_MODE=SHARE_MODE_ACCESS)
    print()
    print(f"  {verdict.message}")
    if verdict.ok:
        print("Shared boards will now require a verified Cloudflare Access sign-in.")
        print("Change any of this later in Settings ▸ Sharing (host powers via CLOUDFLARE_ACCESS_ADMIN_EMAILS).")
    return 0 if verdict.ok else 1


def _install_voice() -> int:
    """Install dictation from the command line, printing plain progress.

    Deliberately not a Rich ``Live``: this runs where the TUI cannot, so the
    output has to survive a pipe, a CI log and a terminal with no cursor
    control. Returns a process exit code.

    There is no MCP tool for this on purpose — installing arbitrary packages
    into the host environment on an LLM's say-so is not a capability worth
    shipping. Both entry points are human-initiated.
    """
    from yeaboi import voice, voice_install

    log = logging.getLogger(__name__)
    log.info("Installing voice input from the CLI")

    if voice.is_voice_available()[0]:
        print("Dictation is already installed.")
    else:
        # ignore_verdict: typing this command *is* the retry. A stored failure
        # from a month ago (a wheel that had not landed yet, a mirror that had
        # not synced) must not be the reason the explicit escape hatch refuses.
        plan = voice_install.install_plan(ignore_verdict=True)
        if plan.blocked:
            print(f"Cannot install dictation here — {plan.blocked}")
            return 1
        print(f"Installing dictation: {plan.display_command}")
        echoed: list[str] = []
        ok, message = voice_install.install_packages(lambda phrase: _echo_once(echoed, phrase))
        if not ok:
            print(message)
            return 1
        from yeaboi.config import mark_voice_extra_installed

        mark_voice_extra_installed()
        print("Packages installed.")
        if plan.follow_up:
            print(f"To keep dictation across upgrades: {plan.follow_up}")

    from yeaboi.config import get_voice_model

    size = get_voice_model()
    if voice_install.model_is_cached(size):
        print(f"Speech model '{size}' is already downloaded.")
    else:
        print(f"Downloading the '{size}' speech model to {voice_install.model_cache_dir()}…")
        ok, message = voice_install.download_model(size, _echo_progress)
        if not ok:
            # A missing model is a warning, not a failure: it downloads lazily on
            # the first dictation exactly as it always did. Fall through to the
            # probe rather than returning — the exit code still has to report
            # whether dictation can actually run here.
            print(message)
        else:
            print("\nSpeech model downloaded.")

    ready, reason = voice.probe_voice_backend(force=True)
    print("Dictation is ready — double-tap Space in any text field." if ready else f"Not ready — {reason}")
    return 0 if ready else 1


def _echo_once(echoed: list[str], phrase: str) -> None:
    """Print an installer phrase, skipping consecutive repeats.

    narrate() keeps emitting the same phrase for every line of a package's
    download, which on a terminal reads as a stutter rather than progress.
    """
    if phrase and (not echoed or echoed[-1] != phrase):
        echoed.append(phrase)
        print(f"  {phrase}")


def _echo_progress(status: str, fraction: float | None) -> None:
    """One-line download progress that degrades to nothing on a dumb pipe."""
    if fraction is None:
        return
    print(f"\r  {int(fraction * 100):3d}%  {status}", end="", flush=True)


def _list_audio_devices() -> None:
    """Print the available microphones — the diagnostic for "it can\'t hear me".

    Rescans first: PortAudio caches its device list at init, so a mic plugged in
    after this process started would otherwise be missing from the very listing
    the user ran to check whether it was detected.
    """
    from yeaboi import voice

    log = logging.getLogger(__name__)
    log.info("Listing audio input devices")
    available, reason = voice.is_voice_available()
    if not available:
        log.info("Audio device listing skipped: voice unavailable — %s", reason)
        print(f"Voice input unavailable — {reason}")
        return

    voice.refresh_devices()
    devices = voice.list_input_devices()
    if not devices:
        log.warning("Audio device listing found no input devices")
        print("No microphones found.")
        return

    configured = voice.get_voice_device()
    selected = voice.resolve_device(configured)
    print("Input devices:")
    for device in devices:
        tags = []
        if device["is_default"]:
            tags.append("system default")
        if selected is not None and device["index"] == selected:
            tags.append(f"selected via VOICE_DEVICE={configured}")
        suffix = f"  ({', '.join(tags)})" if tags else ""
        print(
            f"  {device['index']:>2}  {device['name']:<34} {device['channels']} ch  {device['samplerate']} Hz{suffix}"
        )
    if selected is None:
        print("\nUsing the system default. Choose another with VOICE_DEVICE=<name or index>.")
    log.info("Listed %d audio input device(s); VOICE_DEVICE=%r resolved to %s", len(devices), configured, selected)


def _json_dump(obj: object) -> str:
    """JSON for CLI --format json output — flattens frozen-dataclass artifacts."""
    import dataclasses
    import json

    def _default(o):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        return str(o)

    return json.dumps(obj, indent=2, default=_default)


def _resolve_cli_session(session_id: str) -> str | None:
    """Blank/latest → the most recent saved session id (None when none exist).

    An explicit id must exist — a typo'd --session used to pass through
    verbatim and silently produce empty-context output.
    """
    from yeaboi.paths import get_db_path
    from yeaboi.sessions import SessionStore

    with SessionStore(get_db_path()) as store:
        if session_id and session_id != "latest":
            known = [s["session_id"] for s in store.list_sessions()]
            if session_id not in known:
                listing = ", ".join(known) if known else "none saved yet"
                raise ValueError(f"session {session_id!r} not found — available: {listing}")
            return session_id
        return store.get_latest_session_id()


def _cmd_app(args: argparse.Namespace, console) -> int:
    """`yeaboi app` — run the desktop backend server.

    stdout is reserved for the YEABOI_APP_READY handshake line (the same
    discipline as the MCP server's stdio rule), so the Rich console is unused
    and all diagnostics go to ~/.yeaboi/logs/app/.
    """
    from yeaboi.app.run import run_app
    from yeaboi.logging_setup import attach_mode_handler

    attach_mode_handler("app")
    return run_app(port=args.port)


def _cmd_ask(args: argparse.Namespace, console) -> int:
    """`yeaboi ask` — one Niko turn, in the terminal.

    The answer streams as it is written, the same events the desktop panel
    renders; tool calls print as one dim line each so the reader can see which
    numbers the answer stands on.
    """
    import json

    from yeaboi.logging_setup import mode_log
    from yeaboi.niko import engine
    from yeaboi.niko.store import NikoStore

    to_json = args.format == "json"

    with mode_log("niko"):
        if args.list_only:
            with NikoStore() as store:
                rows = [
                    {
                        "id": c.id,
                        "title": c.title,
                        "messages": c.message_count,
                        "updated_at": c.updated_at,
                    }
                    for c in store.conversations(limit=20)
                ]
            if to_json:
                print(json.dumps({"conversations": rows}, indent=2))
            elif not rows:
                console.print("[dim]No Niko conversations yet — ask something.[/dim]")
            else:
                for row in rows:
                    console.print(f"[dim]{row['updated_at']}[/dim]  {row['id'][:8]}  {row['title'] or '(untitled)'}")
            return 0

        if not " ".join(args.question).strip():
            print('Error: ask Niko something, e.g. yeaboi ask "what should I look at?"', file=sys.stderr)
            return 2

        conversation_id = args.conversation
        if args.continue_latest and not conversation_id:
            with NikoStore() as store:
                newest = store.conversations(limit=1)
            conversation_id = newest[0].id if newest else ""

        # Whether the last thing printed was a streamed token, so a tool line
        # starts on its own row instead of running on from mid-sentence.
        mid_line = [False]

        def on_event(event) -> None:
            # JSON mode keeps stdout machine-clean: the streamed answer would
            # corrupt the single object a caller is parsing.
            if to_json:
                return
            if isinstance(event, engine.Token):
                console.print(event.text, end="", highlight=False, markup=False)
                mid_line[0] = True
                return
            if mid_line[0]:
                console.print()
                mid_line[0] = False
            if isinstance(event, engine.ToolStarted):
                console.print(f"[dim]· reading {event.name}…[/dim]")
            elif isinstance(event, engine.Navigate):
                console.print(f"[dim]· that lives at {event.route}[/dim]")

        answer = engine.ask(
            " ".join(args.question),
            conversation_id=conversation_id,
            route=args.route,
            surface="terminal",
            on_event=on_event,
        )

    if to_json:
        from yeaboi.mcp.runtime import to_jsonable

        print(json.dumps(to_jsonable(answer), indent=2))
        return 0

    console.print()
    for warning in answer.warnings:
        console.print(f"[yellow]! {warning}[/yellow]")
    console.print(f"[dim]conversation {answer.conversation_id[:8]} — continue with `yeaboi ask --continue`[/dim]")
    return 0


def _run_subcommand(args: argparse.Namespace) -> int:
    """Dispatch `yeaboi <command>` headless runners. Returns a process exit code.

    Thin adapters over the same engines the TUI and MCP server use
    (AGENTS.md "REQUIRED: Surface Parity").
    """
    from rich.console import Console

    to_json = getattr(args, "format", "text") == "json"
    # JSON mode keeps stdout machine-clean: human chatter goes to stderr.
    console = Console(stderr=to_json)
    handlers = {
        "report": _cmd_report,
        "standup": _cmd_standup,
        "standup-review": _cmd_standup_review,
        "perf": _cmd_perf,
        "retro": _cmd_retro,
        "poker": _cmd_poker,
        "review": _cmd_review,
        "analyze": _cmd_analyze,
        "agents": _cmd_agents,
        "provenance": _cmd_provenance,
        "ship": _cmd_ship,
        "ceremonies": _cmd_ceremonies,
        "slack": _cmd_slack,
        "connections": _cmd_connections,
        "webhooks": _cmd_webhooks,
        "app": _cmd_app,
        "ask": _cmd_ask,
    }
    try:
        return handlers[args.command](args, console)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _print_perf(console, artifact, kind: str) -> None:
    """Print a performance artifact through the same renderer the TUI page uses.

    One artifact, one rendering: the CLI used to have its own Rich formatter that
    duplicated the layout and hardcoded the mode accent as a literal, with a
    comment admitting it had to be kept in sync by hand.
    """
    from rich.console import Group

    from yeaboi.ui.shared._components import PERFORMANCE_THEME
    from yeaboi.ui.shared._performance_rows import performance_detail_rows

    rows, _heights = performance_detail_rows(artifact, kind=kind, theme=PERFORMANCE_THEME, width=console.width)
    console.print(Group(*rows))


def _strict_exit(strict: bool, warnings, empty: bool = False) -> int:
    """--strict maps a degraded run (warnings, or an empty result) to exit 3 —
    so CI can tell a real report from a deterministic fallback. Default runs
    keep exit 0 with warnings on stderr."""
    if strict and (warnings or empty):
        detail = f"{len(list(warnings))} warning(s)" + (", empty result" if empty else "")
        print(f"strict: degraded run ({detail}) — exit 3", file=sys.stderr)
        return 3
    return 0


def _print_beta_notice(notice: str) -> None:
    """Print a beta-maturity caveat before a beta subcommand runs.

    stderr, not stdout: the artifact these commands print is routinely piped or
    redirected, and a caveat inside the file would be worse than no caveat. It
    also matches the ``⚠ {warning}`` lines the handlers already emit. Scripted
    callers turn it off with ``BETA_NOTICES_ENABLED=false``.

    Deliberately NOT routed through the engine's warnings tuple: ``--strict``
    maps any warning to exit 3, which would make every strict run fail forever.
    """
    from yeaboi.config import is_beta_notice_enabled

    if is_beta_notice_enabled():
        print(f"⚠ {notice}", file=sys.stderr)


def _cmd_report(args: argparse.Namespace, console: Console) -> int:
    from yeaboi.reporting.engine import run_delivery_report
    from yeaboi.reporting.render import format_report_rich

    console.print(f"[bold cyan]Generating {args.period.replace('_', ' ')} delivery report...[/bold cyan]")
    sprint_names = tuple(s.strip() for s in args.sprint_names.split(",") if s.strip())
    # Assemble the engine's sources dict only when a source flag was given —
    # None means "every configured source" (the engine's auto default).
    sources: dict | None = None
    if args.source or args.code_sources is not None or args.documentation_sources is not None:
        sources = {}
        if args.source:
            sources["delivery"] = ["jira", "azdevops"] if args.source == "both" else [args.source]
        if args.code_sources is not None:
            sources["code"] = list(args.code_sources)
        if args.documentation_sources is not None:
            sources["docs"] = list(args.documentation_sources)
    report = run_delivery_report(
        args.period,
        session_id=_resolve_cli_session(args.session) or "",
        jira_project=args.jira_project,
        azdo_project=args.azdo_project,
        window_start=args.window_start,
        window_end=args.window_end,
        sprint_names=sprint_names,
        period_label_override=args.label,
        theme=args.theme,
        sources=sources,
        solo=args.solo,
        **_context_kwargs(args),
    )
    for warning in report.warnings:
        print(f"⚠ {warning}", file=sys.stderr)
    if args.format == "json":
        print(_json_dump(report))
    else:
        from yeaboi.paths import REPORTING_EXPORTS_DIR

        console.print(format_report_rich(report))
        console.print(
            f"[dim]Exports (Markdown + HTML + slides, and .pptx when python-pptx is installed): "
            f"{REPORTING_EXPORTS_DIR}[/dim]"
        )
    return _strict_exit(args.strict, report.warnings, empty=not report.delivered_items)


def _cmd_standup(args: argparse.Namespace, console: Console) -> int:
    # Route this run's records to ~/.yeaboi/logs/standup/ like every other
    # standup entry point (AGENTS.md "Observability" — each mode logs to its own
    # directory). Only the --standup-run scheduler path did this before, so a
    # `yeaboi standup` run left nothing behind in the standup log.
    from yeaboi.logging_setup import mode_log

    with mode_log("standup"):
        return _cmd_standup_inner(args, console)


def _cmd_standup_inner(args: argparse.Namespace, console: Console) -> int:
    from yeaboi.standup.engine import run_standup
    from yeaboi.standup.render import format_standup_rich

    session_id = _resolve_cli_session(args.session)
    if not session_id:
        print("Error: no session found to run a standup for.", file=sys.stderr)
        return 2
    if args.schedule:
        return _cmd_standup_schedule(args, console, session_id)
    if args.set_context:
        from yeaboi.paths import get_db_path
        from yeaboi.standup.store import StandupStore

        scope = _context_spec(args.set_context)
        with StandupStore(get_db_path()) as store:
            store.set_context_scope(session_id, scope)
        console.print(f"Standup scope for {session_id}: {scope.to_spec() if scope is not None else 'inherit'}")
        return 0
    if args.list_members:
        from yeaboi.config import get_azure_devops_project, get_jira_project_key
        from yeaboi.paths import get_db_path
        from yeaboi.standup.roster import default_tracker_sources, discover_team_members
        from yeaboi.standup.store import StandupStore

        with StandupStore(get_db_path()) as store:
            config = store.load_config(session_id) or {}
        jira_project = get_jira_project_key() or ""
        azdo_project = get_azure_devops_project() or ""
        sources = (
            args.tracker_sources
            or (config.get("tracker_sources") if config.get("roster_configured") else None)
            or default_tracker_sources(
                jira_project=jira_project,
                azdo_project=azdo_project,
            )
        )
        for member in discover_team_members(
            sources,
            jira_project=jira_project,
            azdo_project=azdo_project,
        ):
            console.print(member)
        return 0
    report = run_standup(
        session_id,
        deliver=args.deliver,
        days=args.days or None,
        channels=args.channels,
        tracker_sources=args.tracker_sources,
        team_members=args.team_members,
        code_sources=args.code_sources,
        github_owners=args.github_owners,
        github_repositories=args.github_repositories,
        github_excluded_repositories=args.github_excluded_repositories,
        azdo_projects=args.azdo_projects,
        azdo_repositories=args.azdo_repositories,
        documentation_sources=args.documentation_sources,
        review_transcripts=args.review_transcripts,
        solo=args.solo,
        **_context_kwargs(args),
    )
    for warning in report.warnings:
        print(f"⚠ {warning}", file=sys.stderr)
    if args.format == "json":
        print(_json_dump(report))
    else:
        console.print(format_standup_rich(report))
    return _strict_exit(args.strict, report.warnings)


def _cmd_standup_schedule(args: argparse.Namespace, console: Console, session_id: str) -> int:
    """`yeaboi standup --schedule install|remove|status` — manage the OS-native daily job.

    Uses the session's saved standup config (time/weekdays/lead) — set it via the
    TUI Configure screen or the MCP standup_config_set tool.
    """
    import json

    from yeaboi.standup.scheduler import get_schedule_status, install_schedule, remove_schedule

    logging.getLogger("yeaboi.cli").info("standup schedule %s: session=%s", args.schedule, session_id)
    if args.schedule == "status":
        status = get_schedule_status(session_id)
        if args.format == "json":
            print(json.dumps(status, indent=2))
        else:
            state = "installed" if status.get("installed") else "not installed"
            suffix = f" ({status.get('path')})" if status.get("path") else ""
            console.print(f"Schedule for [bold]{session_id}[/bold] [{status.get('platform', '?')}]: {state}{suffix}")
        return 0
    if args.schedule == "remove":
        console.print(remove_schedule(session_id))
        return 0
    # install — from the saved config (defaults mirror standup/store.py)
    from yeaboi.paths import get_db_path
    from yeaboi.standup.store import StandupStore

    with StandupStore(get_db_path()) as store:
        config = store.load_config(session_id) or {}
    standup_time = config.get("time") or "10:00"
    weekdays = config.get("weekdays") or "1-5"
    lead_minutes = int(config.get("lead_minutes", 10))
    console.print(install_schedule(session_id, standup_time, weekdays, lead_minutes))
    return 0


def _format_review_text(review, console: Console) -> None:
    """Print a transcript review as scannable text."""
    console.print(f"[bold]Transcript review — {review.standup_date or 'unknown date'}[/bold]")
    if review.sources:
        console.print("Read: " + ", ".join(s.filename for s in review.sources))
    if review.accuracy_note:
        console.print(review.accuracy_note)
    if review.gaps:
        console.print("\n[bold]Gaps in standup itself[/bold] (drafted as GitHub issues)")
        for gap in review.gaps:
            # Parentheses, not brackets: Rich reads [high/high] as markup and
            # silently swallows it.
            console.print(f"  • {gap.title}  ({gap.priority} priority, {gap.confidence} confidence)  {gap.fingerprint}")
            if gap.root_cause:
                console.print(f"    {gap.root_cause}")
    if review.config_suggestions:
        console.print("\n[bold]Fix in your configuration[/bold] (never filed)")
        for gap in review.config_suggestions:
            console.print(f"  • {gap.title}")
            if gap.remedy:
                console.print(f"    → {gap.remedy}")
    if not review.gaps and not review.config_suggestions:
        console.print("\nNo gaps found — the report matched what the team said.")


def _resolve_review_inputs(args: argparse.Namespace) -> tuple[list[str] | None, str]:
    """Fold the positional paths, ``-`` and ``--transcript-text`` into engine args.

    Returns ``(transcript_paths, transcript_text)``.

    Stdin is resolved HERE rather than in the engine: ``sweep_and_review`` does a
    bare ``Path(p)`` on everything it is handed, so a literal ``"-"`` would reach
    it as a filename. Every path is run through ``normalize_dropped_path`` because
    a file dragged from Finder arrives quoted (Terminal) or backslash-escaped
    (iTerm2), and would otherwise fail as "not found".
    """
    from yeaboi.standup.transcripts import normalize_dropped_path

    raw = [*(args.transcript_paths or []), *(getattr(args, "paths", None) or [])]
    text = args.transcript_text or ""

    paths: list[str] = []
    for entry in raw:
        if entry == "-":
            if not text:
                # A bare `-` on an interactive terminal means the user expected a
                # pipe and did not give one. Reading anyway blocks forever with a
                # blank screen and no hint, which reads as a hung command.
                if sys.stdin.isatty():
                    raise SystemExit(
                        "standup-review -: nothing is piped in. "
                        "Pipe a transcript (cat standup.txt | yeaboi standup-review -), "
                        "or pass the file path instead."
                    )
                text = sys.stdin.read()
            continue
        cleaned = normalize_dropped_path(entry)
        if cleaned:
            paths.append(cleaned)
    return (paths or None), text


def _cmd_standup_review(args: argparse.Namespace, console: Console) -> int:
    """`yeaboi standup-review` — audit standup reports against meeting transcripts."""
    from yeaboi.logging_setup import mode_log

    with mode_log("standup"):
        return _cmd_standup_review_inner(args, console)


def _cmd_standup_review_inner(args: argparse.Namespace, console: Console) -> int:
    from yeaboi.paths import get_db_path
    from yeaboi.standup.engine import file_transcript_issues, run_transcript_review, transcript_nudge
    from yeaboi.standup.store import StandupStore

    logging.getLogger(__name__).info("standup-review (list_gaps=%s file=%s)", args.list_gaps, args.file_issues)
    session_id = _resolve_cli_session(args.session)
    if not session_id:
        print("Error: no session found to review transcripts for.", file=sys.stderr)
        return 2

    if args.list_gaps:
        # The dedup ledger, so it is possible to SEE dedup working rather than
        # having to trust it.
        with StandupStore(get_db_path()) as store:
            reviews = store.get_reviews(session_id, limit=30)
            ledger = store.get_gap_issues(limit=50)
        if args.format == "json":
            print(_json_dump({"reviews": reviews, "gap_issues": ledger}))
            return 0
        console.print(f"[bold]{len(reviews)} review(s)[/bold]")
        for row in reviews:
            console.print(f"  {row['standup_date'] or '?'}  run={row['run_id'] or '-'}  {row['status']}")
        console.print(f"\n[bold]{len(ledger)} tracked gap(s)[/bold]")
        for entry in ledger:
            issue = f"#{entry['issue_number']}" if entry["issue_number"] else entry["state"]
            console.print(f"  {entry['fingerprint']}  {issue}  ×{entry['occurrences']}  {entry['title']}")
        nudge = transcript_nudge(session_id)
        if nudge:
            console.print(f"\n[bold]{len(nudge.missed_dates)} unchecked standup(s)[/bold]")
            console.print("  " + ", ".join(nudge.missed_dates[:14]))
            console.print(f"  {nudge.message}")
        return 0

    paths, transcript_text = _resolve_review_inputs(args)

    review = run_transcript_review(
        session_id,
        transcript_paths=paths,
        transcript_text=transcript_text,
        transcript_dir=args.transcript_dir,
        standup_date=args.standup_date,
        max_transcripts=args.max_transcripts,
        include_reviewed=args.include_reviewed,
    )
    if transcript_text and review.sources:
        # Name what landed: an import the user cannot see is an import they
        # cannot tell went to the right day.
        imported = review.sources[0]
        print(f"Imported {imported.filename} — covers {imported.covered_date} ({imported.attribution})")
    for warning in review.warnings:
        print(f"⚠ {warning}", file=sys.stderr)

    filing = None
    if args.file_issues:
        if not review.gaps:
            print("Nothing to file — no gaps in standup itself were diagnosed.", file=sys.stderr)
        else:
            filing = file_transcript_issues(review.review_id, session_id=session_id)
            for warning in filing.warnings:
                print(f"⚠ {warning}", file=sys.stderr)

    if args.format == "json":
        print(_json_dump({"review": review, "filing": filing} if filing else review))
    else:
        _format_review_text(review, console)
        if filing:
            console.print(f"\nFiled {filing.filed}, commented {filing.commented}, skipped {filing.skipped}.")
            for link in filing.links:
                if link.issue_url:
                    console.print(f"  {link.issue_url}")
        elif review.gaps:
            console.print("\nRun again with --file-issues to file these on GitHub.")

    warnings = list(review.warnings) + list(filing.warnings if filing else [])
    return _strict_exit(args.strict, warnings)


def _cmd_perf(args: argparse.Namespace, console: Console) -> int:
    logging.getLogger(__name__).info("perf %s (beta)", args.perf_command)
    # One call site ahead of the branch covers all five subcommands.
    _print_beta_notice(PERFORMANCE_BETA_NOTICE)

    if args.perf_command == "roster":
        from yeaboi.performance.roster import fetch_roster

        engineers = fetch_roster()
        if not engineers:
            console.print("[yellow]No engineers found — is a tracker (Jira/AzDO) configured?[/yellow]")
            return 2
        for eng in engineers:
            console.print(f"  • {getattr(eng, 'name', eng)}")
        return 0

    if args.perf_command == "prep":
        from yeaboi.performance.engine import run_one_on_one_prep

        prep = run_one_on_one_prep(
            args.engineer,
            session_id=_resolve_cli_session(args.session) or "",
            jira_project=args.jira_project,
            azdo_project=args.azdo_project,
            deep_scan=args.deep_scan,
            **_context_kwargs(args),
        )
        for warning in prep.warnings:
            print(f"⚠ {warning}", file=sys.stderr)
        _print_perf(console, prep, "prep")
        return _strict_exit(args.strict, prep.warnings)

    if args.perf_command == "complete":
        transcript = args.transcript
        if transcript.startswith("@"):
            try:
                path = fs_policy.resolve_and_check(transcript[1:], mode="read", context="perf complete @transcript")
            except fs_policy.SandboxViolationError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
            if not path.exists():
                print(f"Error: transcript file not found: {path}", file=sys.stderr)
                return 1
            transcript = path.read_text().strip()
        from yeaboi.performance.engine import complete_one_on_one

        record = complete_one_on_one(
            args.engineer,
            transcript,
            session_id=_resolve_cli_session(args.session) or "",
            deliver=args.deliver,
            recipients=args.recipients,
            images=tuple(args.images),
            **_context_kwargs(args),
        )
        for warning in record.warnings:
            print(f"⚠ {warning}", file=sys.stderr)
        _print_perf(console, record, "completion")
        return _strict_exit(args.strict, record.warnings)

    if args.perf_command == "review":
        from yeaboi.performance.engine import run_six_month_review

        review = run_six_month_review(
            args.engineer,
            session_id=_resolve_cli_session(args.session) or "",
            jira_project=args.jira_project,
            azdo_project=args.azdo_project,
            period_months=args.months,
            deep_scan=args.deep_scan,
            **_context_kwargs(args),
        )
        for warning in review.warnings:
            print(f"⚠ {warning}", file=sys.stderr)
        _print_perf(console, review, "review")
        return _strict_exit(args.strict, review.warnings)

    # note
    from yeaboi.paths import get_db_path
    from yeaboi.performance.store import PerformanceStore

    with PerformanceStore(get_db_path()) as store:
        store.add_note(args.engineer, args.text)
    console.print(f"[green]Note recorded for {args.engineer}[/green]")
    return 0


def _cmd_provenance(args: argparse.Namespace, console: Console) -> int:
    """The provenance audit headless: same engine the MCP tools use
    (AGENTS.md "REQUIRED: Surface Parity")."""
    import json
    from dataclasses import asdict

    logging.getLogger(__name__).info("provenance %s", args.provenance_command)

    if args.provenance_command == "audit":
        from yeaboi.provenance.engine import run_provenance_audit
        from yeaboi.provenance.render import format_audit_rich

        report = run_provenance_audit(window_days=args.window_days)
        for warning in report.warnings:
            print(f"⚠ {warning}", file=sys.stderr)
        if args.format == "json":
            print(json.dumps(asdict(report), indent=2))
        else:
            console.print(format_audit_rich(report))
        return _strict_exit(args.strict, report.warnings, empty=report.total_records == 0)

    # trace
    from yeaboi.provenance.engine import trace_entity
    from yeaboi.provenance.render import format_trace_rich

    trace = trace_entity(args.entity_id, depth=args.depth)
    for warning in trace.warnings:
        print(f"⚠ {warning}", file=sys.stderr)
    if args.format == "json":
        print(json.dumps(asdict(trace), indent=2))
    else:
        console.print(format_trace_rich(trace))
    return 0 if trace.found else 1


def _cmd_connections(args: argparse.Namespace, console: Console) -> int:
    """`yeaboi connections …` — the read-only integration catalog."""
    import json
    import os

    from rich.markup import escape

    from yeaboi.connectors import legacy, registry
    from yeaboi.connectors.engine import list_connections

    to_json = getattr(args, "format", "text") == "json"
    command = args.connections_command

    if command == "list":
        payload = list_connections(family=args.family, connected_only=not args.all, include_legacy=args.all)
        if to_json:
            print(json.dumps(payload, indent=2))
            return 0
        rows = payload["connectors"]
        if not rows:
            console.print("[dim]Nothing connected yet. `yeaboi connections list --all` shows what you can add.[/dim]")
            return 0
        for row in rows:
            mark = "[green]connected[/green]" if row["connected"] else "[dim]not connected[/dim]"
            console.print(f"{row['glyph']} [bold]{row['label']}[/bold]  [dim]{row['family_label']}[/dim]  {mark}")
            # A list of vendor names is a list of things to go look up. Say what
            # each one is for, on the line under it.
            if row["summary"]:
                console.print(f"   [dim]{row['summary']}[/dim]")
            if row["managed_by"] == "credentials":
                console.print("   [dim]set up via: yeaboi --setup (Settings ▸ Credentials)[/dim]")
        return 0

    if command == "create":
        return _connections_create(args, console)

    connector = registry.by_key(args.name) if getattr(args, "name", "") else None
    legacy_entry = legacy.by_key(args.name) if connector is None and getattr(args, "name", "") else None
    if getattr(args, "name", "") and connector is None and legacy_entry is None:
        known = ", ".join([c.key for c in registry.all_connectors()] + [c.key for c in registry.legacy_entries()])
        print(f"Error: unknown connector {args.name!r}. Known: {known}", file=sys.stderr)
        return 1

    if command in ("webhook-url", "webhook-test"):
        from yeaboi.connectors.webhooks.server import connection_url, send_test_delivery

        if command == "webhook-test":
            outcome = send_test_delivery(args.name)
            console.print(escape(str(outcome["message"])))
            return 0 if outcome["ok"] else 1
        info = connection_url(args.name)
        if info is None:
            print(f"Error: {args.name!r} is not a webhook connection.", file=sys.stderr)
            return 1
        if to_json:
            print(json.dumps(info, indent=2))
            return 0
        console.print(f"[bold]{escape(info['key'])}[/bold]")
        console.print(f"  local:  {escape(info['url'])}")
        if info["tunnel_url"]:
            console.print(f"  tunnel: {escape(info['tunnel_url'])}  [dim](rotates per share — for testing)[/dim]")
        console.print(f"  header: {escape(info['header'])}  [dim]({info['verify']})[/dim]")
        console.print(f"  secret: {escape(info['secret'])}")
        if not info["running"]:
            console.print("  [yellow]The receiver is not running — start it with `yeaboi webhooks serve`.[/yellow]")
        if info["last_received_at"]:
            console.print(f"  last delivery: {escape(info['last_received_at'])}")
        else:
            console.print("  [dim]waiting for the first delivery[/dim]")
        return 0

    if legacy_entry is not None and command in ("add", "remove", "fetch"):
        # A built-in integration's credentials belong to Credentials/setup —
        # they are shared with features this command must not silently unplug.
        print(
            f"{legacy_entry.label} is set up via `yeaboi --setup` or Settings ▸ Credentials, "
            f"not `yeaboi connections {command}`.",
            file=sys.stderr,
        )
        return 1

    if command == "show":
        payload = list_connections(connected_only=False, include_legacy=True)
        shown_key = (connector or legacy_entry).key
        row = next(r for r in payload["connectors"] if r["key"] == shown_key)
        if to_json:
            print(json.dumps(row, indent=2))
            return 0
        entry = connector or legacy_entry
        console.print(f"{row['glyph']} [bold]{row['label']}[/bold]  [dim]{row['family_label']}[/dim]")
        if row["detail"]:
            from rich.padding import Padding

            # Padding, not a literal indent: the detail wraps, and only the first
            # wrapped line would carry a leading-space indent.
            console.print(Padding(f"[dim]{row['detail']}[/dim]", (0, 0, 1, 2)))
        chosen = registry.chosen_method(entry)
        if chosen is not None:
            console.print(f"  [bold]{chosen.label}[/bold]  [dim]{chosen.summary}[/dim]")
            if chosen.warning:
                console.print(f"  [yellow]⚠  {chosen.warning}[/yellow]")
        shown = {f.env for f in entry.fields_for(chosen.key)} if chosen else None
        for field in row["fields"]:
            if shown is not None and field["env"] not in shown:
                continue
            state = "[green]set[/green]" if field["is_set"] else "[dim]not set[/dim]"
            need = "" if field["required"] else " [dim](optional)[/dim]"
            console.print(f"  {field['label']}: {state}{need}  [dim]{field['env']}[/dim]")
        for method in entry.auth_methods:
            if chosen is not None and method.key == chosen.key:
                continue
            console.print(f"  [dim]or {method.key}: {method.summary}[/dim]")
            if method.warning:
                console.print(f"  [dim]   ⚠  {method.warning}[/dim]")
        if row["docs_url"]:
            console.print(f"  [dim]docs: {row['docs_url']}[/dim]")
        if row["managed_by"] == "credentials":
            console.print("  [dim]set up via: yeaboi --setup (Settings ▸ Credentials)[/dim]")
        return 0

    if command == "verify":
        from yeaboi.settings.engine import verify_connection

        if legacy_entry is not None and legacy_entry.key not in legacy.LEGACY_VERIFY_KINDS:
            print(f"{legacy_entry.label} has no live probe — check it under Settings ▸ System.", file=sys.stderr)
            return 1
        if connector is not None and not connector.verify:
            print(f"{connector.label} has no live probe — it receives deliveries instead.", file=sys.stderr)
            return 1
        if connector or legacy_entry:
            targets = [(connector or legacy_entry).key]
        else:
            # A probe-less connection (a webhook custom receives deliveries) is
            # skipped rather than reported as a failure.
            targets = [key for key in registry.connected() if getattr(registry.by_key(key), "verify", "")]
        if not targets:
            console.print("[dim]Nothing connected to verify.[/dim]")
            return 0
        results = []
        for key in targets:
            try:
                outcome = verify_connection(key, {})
            except ValueError as exc:
                outcome = {"ok": False, "message": str(exc)}
            results.append({"connector": key, **outcome})
        if to_json:
            print(json.dumps(results, indent=2))
        else:
            for result in results:
                tick = "[green]ok[/green]" if result["ok"] else "[red]failed[/red]"
                # A probe's message is data, not markup: an install hint saying
                # yeaboi[cloud] would otherwise lose the extra to a style tag.
                console.print(f"{result['connector']}: {tick} — {escape(str(result['message']))}")
        return 0 if all(r["ok"] for r in results) else 1

    if command == "fetch":
        from yeaboi.connectors.engine import fetch_ops_events

        try:
            payload = fetch_ops_events(connector.key if connector else "", since=args.since)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if to_json:
            print(json.dumps(payload, indent=2))
            return 0
        if not payload["sources"]:
            console.print("[dim]Nothing connected has anything to gather yet.[/dim]")
            return 0
        window = payload["window"]
        console.print(f"[dim]{window['start'][:10]} → {window['end'][:10]}[/dim]")
        for source in payload["sources"]:
            if source["ok"]:
                console.print(f"  [green]{source['label']}[/green]: {source['count']} event(s)")
            else:
                console.print(f"  [red]{source['label']}[/red]: {escape(str(source['error']))}")
        for signal in payload["signals"]:
            worst = f" [dim]worst: {signal['severity']}[/dim]" if signal["severity"] else ""
            console.print(
                f"{signal['kind']} via {signal['source']}: [bold]{signal['count']}[/bold] "
                f"({signal['resolved']} resolved){worst}"
            )
            for sample in signal["samples"]:
                console.print(f"   [dim]{sample}[/dim]")
        return 0 if all(s["ok"] for s in payload["sources"]) else 1

    if command == "add":
        from getpass import getpass

        from yeaboi.config import apply_config_value

        console.print(f"{connector.mark} [bold]{connector.label}[/bold]")
        if connector.docs_url:
            console.print(f"[dim]{connector.docs_url}[/dim]")

        # The method comes first, because it decides which of the questions
        # below are even asked. A connector with one way in skips this entirely.
        method = connector.default_method
        if connector.auth_methods:
            for option in connector.auth_methods:
                tag = " [green](recommended)[/green]" if option.recommended else ""
                console.print(f"  [bold]{option.key}[/bold]{tag} — {option.summary}")
                if option.warning:
                    console.print(f"  [yellow]  ⚠  {option.warning}[/yellow]")
            picked = input(f"How to connect [{method.key}]: ").strip() or method.key
            chosen = connector.method(picked)
            if chosen is None:
                known = ", ".join(m.key for m in connector.auth_methods)
                print(f"Error: unknown method {picked!r}. Known: {known}", file=sys.stderr)
                return 1
            method = chosen
            apply_config_value(connector.auth_env, method.key)
            os.environ[connector.auth_env] = method.key
            if method.setup_url:
                console.print(f"[dim]Set it up: {method.setup_url}[/dim]")

        for field in connector.fields_for(method.key) if method else connector.fields:
            if connector.auth_env and field.env == connector.auth_env:
                continue
            # A sign-in's fields are minted by the flow, never typed.
            if field.action == "signin":
                if field.secret:
                    console.print(f"[dim]{field.label}: sign in from the desktop's Music page[/dim]")
                continue
            # An external ID is yeaboi's to mint, not the user's to invent: it
            # is what stops a confused deputy assuming the role, so it has to be
            # unguessable. Shown once, for pasting into the trust policy.
            if field.env == "AWS_EXTERNAL_ID" and not os.environ.get(field.env, "").strip():
                from yeaboi.connectors.aws import new_external_id

                generated = new_external_id()
                apply_config_value(field.env, generated)
                os.environ[field.env] = generated
                console.print(f"  External ID (paste into the role's trust policy): [bold]{generated}[/bold]")
                continue
            hint = f" [{field.default}]" if field.default else ""
            label = f"{field.label}{'' if field.required else ' (optional)'}{hint}: "
            if field.help_scope:
                console.print(f"[dim]{field.help_scope}[/dim]")
            # Secrets are never echoed, and never reach argv.
            value = (getpass(label) if field.secret else input(label)).strip()
            if not value and field.default:
                value = field.default
            if not value:
                if field.required:
                    print(f"Error: {connector.label} needs {field.label}", file=sys.stderr)
                    return 1
                continue
            apply_config_value(field.env, value)
        console.print(f"[green]Saved.[/green] Verify it with: yeaboi connections verify {connector.key}")
        return 0

    if command == "remove":
        from yeaboi.config import apply_config_value

        is_custom = connector.key.startswith("custom_")
        what = "definition and stored values" if is_custom else "stored values"
        if not args.yes:
            reply = input(f"Forget the {connector.label} {what}? [y/N] ").strip().lower()
            if reply not in ("y", "yes"):
                console.print("[dim]Left alone.[/dim]")
                return 0
        if is_custom:
            from yeaboi.connectors.engine import delete_custom_connection

            delete_custom_connection(connector.key)
            console.print(f"[green]{connector.label} removed — definition and values.[/green]")
            return 0
        for field in connector.fields:
            apply_config_value(field.env, "")
            os.environ.pop(field.env, None)
        console.print(f"[green]{connector.label} disconnected.[/green]")
        return 0

    return 1


def _connections_create(args: argparse.Namespace, console: Console) -> int:
    """`yeaboi connections create` — a descriptor from prompts or a JSON file.

    Descriptor only: credentials are entered afterwards with `connections add`,
    so they take the masked path this command has no business owning.
    """
    import json

    from rich.markup import escape

    from yeaboi.connectors.custom import spec_from_dict
    from yeaboi.connectors.engine import create_custom_connection
    from yeaboi.connectors.spec import FAMILIES
    from yeaboi.connectors.validation import AUTH_SCHEMES
    from yeaboi.ops.events import EVENT_KINDS

    if args.from_json:
        try:
            raw = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"Error: could not read {args.from_json}: {exc}", file=sys.stderr)
            return 1
    else:
        console.print(
            "[bold]Create a connection[/bold] — a read-only API, an inbound webhook or an MCP server "
            "the catalog then carries."
        )
        label = input("Service name: ").strip()
        if not label:
            print("Error: a name is required", file=sys.stderr)
            return 1
        default_key = "custom_" + "".join(ch if ch.isalnum() else "_" for ch in label.lower()).strip("_")
        key = input(f"Key [{default_key}]: ").strip() or default_key
        families = ", ".join(FAMILIES)
        family = input(f"Family ({families}) [observability]: ").strip() or "observability"
        summary = input("One-line summary: ").strip()
        glyph = input("Icon (one emoji) [🔌]: ").strip() or "\U0001f50c"
        accent = input("Accent rgb(r,g,b) [rgb(120,160,200)]: ").strip() or "rgb(120,160,200)"
        docs_url = input("Docs URL (https, optional): ").strip()
        raw = {
            "key": key,
            "label": label,
            "family": family,
            "summary": summary,
            "glyph": glyph,
            "accent": accent,
            "docs_url": docs_url,
        }
        kind = input("Kind (api, webhook, mcp) [api]: ").strip() or "api"
        raw["kind"] = kind
        if kind == "webhook":
            # Inbound-only: what matters is how a delivery authenticates and
            # how its rows become events; yeaboi mints the secret itself.
            verify_mode = input("Delivery auth (token, hmac) [token]: ").strip() or "token"
            kinds = ", ".join(EVENT_KINDS)
            event_kind = input(f"Event kind ({kinds}) [alert]: ").strip() or "alert"
            title_path = input("Title path (dot path to a delivery's name, e.g. incident.name): ").strip()
            raw.update({"webhook_verify": verify_mode, "events": {"kind": event_kind, "title_path": title_path}})
        elif kind != "mcp":
            # The HTTP shape belongs to the api kind; an MCP connection has
            # nothing to ask — its Server URL and token are entered afterwards
            # like any other credential.
            schemes = ", ".join(AUTH_SCHEMES)
            auth_scheme = input(f"Auth scheme ({schemes}) [bearer]: ").strip() or "bearer"
            header_name = ""
            if auth_scheme == "header":
                header_name = input("Header name: ").strip()
            probe_path = input("Probe path (an authenticated GET, e.g. /v1/me) [/]: ").strip() or "/"
            raw.update({"auth_scheme": auth_scheme, "header_name": header_name, "probe_path": probe_path})
            extras = []
            while len(extras) < 4 and input("Add an extra credential/config field? (y/N): ").strip().lower() == "y":
                extra_label = input("  Field label (e.g. Application Key): ").strip()
                suffix = input("  Env suffix (UPPER_SNAKE, e.g. APP_KEY): ").strip()
                secret = (input("  Secret? (Y/n): ").strip().lower() or "y") != "n"
                extra_header = input("  Sent as request header (blank for none): ").strip()
                extras.append(
                    {"label": extra_label, "env_suffix": suffix, "secret": secret, "header_name": extra_header}
                )
            if extras:
                raw["extra_fields"] = extras

    try:
        row = create_custom_connection(raw)
    except ValueError as exc:
        # The validator's lines are data — a hostname or an emoji must not
        # become a Rich style tag.
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    console.print(f"[green]Created.[/green] {escape(str(row['label']))} is in the catalog.")
    if row.get("webhook_secret"):
        console.print("Delivery secret (shown once — `yeaboi connections webhook-url` shows it again):")
        console.print(escape(str(row["webhook_secret"])))
    else:
        console.print(f"Enter its credentials with: yeaboi connections add {spec_from_dict(raw).key}")
    return 0


def _cmd_webhooks(args: argparse.Namespace, console: Console) -> int:
    """`yeaboi webhooks serve` — the loopback receiver, foreground."""
    import time as _time

    from yeaboi.connectors.webhooks.server import start_server, start_share, stop_server

    if args.webhooks_command != "serve":
        return 1
    try:
        status = start_server(args.port or None)
    except OSError as exc:
        # A fixed port is the point — a walk would silently break pasted URLs.
        print(f"Error: could not bind 127.0.0.1:{args.port or 'default'} — {exc}", file=sys.stderr)
        return 1
    console.print(f"[green]Receiver listening[/green] on 127.0.0.1:{status['port']} (loopback only)")
    if args.share:
        url = start_share()
        if url:
            console.print(f"Tunnel: {url}  [dim](rotates per share and expires — for testing a sender)[/dim]")
        else:
            console.print("[yellow]Could not open the tunnel — the local receiver still runs.[/yellow]")
    console.print("[dim]`yeaboi connections webhook-url <name>` prints each connection's URL. Ctrl+C stops.[/dim]")
    try:
        while True:
            _time.sleep(1)
    except KeyboardInterrupt:
        stop_server()
        console.print("\n[dim]Receiver stopped.[/dim]")
        return 0


def _cmd_slack(args: argparse.Namespace, console: Console) -> int:
    """`yeaboi slack …` — the inbound half of the Slack channel."""
    import json

    from yeaboi import config
    from yeaboi.slack import allowlist as allow

    to_json = getattr(args, "format", "text") == "json"
    command = args.slack_command

    if command == "check":
        ready, why = config.slack_two_way_ready()
        report: dict = {
            "configured": ready,
            "reason": why,
            "allowlist": allow.describe(),
            "ack_reaction": config.get_slack_ack_reaction() or "off",
        }
        if ready:
            from yeaboi.tools import slack as slack_api

            identity = slack_api.auth_test()
            report["identity"] = (
                {"team": identity.data.get("team", ""), "bot": identity.data.get("user", "")}
                if identity.ok
                else slack_api.error_message(identity)
            )
            report["reachable"] = identity.ok
            if identity.ok:
                # auth.test passes for a bot nobody ever invited, so identity
                # alone cannot say "on". Read the channel the poll will read.
                probe = slack_api.history(config.get_slack_channel_id(), limit=1)
                report["readable"] = probe.ok
                if not probe.ok:
                    report["channel"] = slack_api.error_message(probe)
        if to_json:
            print(json.dumps(report, indent=2))
        else:
            console.print(f"Two-way: {'[green]on[/green]' if ready else f'[yellow]off[/yellow] — {why}'}")
            console.print(f"Allowlist: {report['allowlist']}")
            console.print(f"Ack reaction: {report['ack_reaction']}")
            if "identity" in report:
                console.print(f"Slack says: {report['identity']}")
            if "channel" in report:
                console.print(f"[yellow]Channel: {report['channel']}[/yellow]")
            elif report.get("readable"):
                console.print("Channel: readable")
        return 0 if ready and report.get("reachable", True) and report.get("readable", True) else 1

    if command == "watch":
        from yeaboi.ceremonies import scheduler

        if args.status:
            status = scheduler.slack_poll_status()
            if to_json:
                print(json.dumps(status, indent=2))
            elif status.get("installed"):
                console.print(f"Polling every {status.get('interval_min', 0)} min ({status.get('platform', '?')})")
            else:
                console.print("The Slack poll is not installed.")
            return 0
        message = scheduler.remove_slack_poll() if args.remove else scheduler.install_slack_poll(minutes=args.every)
        if to_json:
            print(json.dumps({"message": message}, indent=2))
        else:
            console.print(message)
        # An install this refuses (no token, an interval cron cannot express) is
        # a request that could not be honoured, not a crash.
        return 1 if message.startswith(("Not installing", "Failed")) or "not a usable interval" in message else 0

    if command == "history":
        from yeaboi.slack.store import SlackStore

        with SlackStore() as store:
            rows = store.unsettled(limit=args.limit) if args.pending else store.history(limit=args.limit)
        if to_json:
            print(json.dumps({"events": rows}, indent=2))
        elif not rows:
            console.print("Nothing from Slack yet." if not args.pending else "Nothing left unfinished.")
        else:
            for row in rows:
                outcome = row.get("outcome") or "in flight"
                console.print(
                    f"  {row.get('claimed_at', '')[:16]}  {outcome:<13} {row.get('intent', '') or '-':<8} "
                    f"{row.get('slack_user', '')}  {row.get('reason', '')}"
                )
        return 0

    if command == "members":
        from yeaboi.tools import slack as slack_api
        from yeaboi.tools.slack import SlackResponse

        ready, why = config.slack_two_way_ready()
        if not ready:
            console.print(f"[yellow]{why}[/yellow]")
            return 1
        # `paginate` hands its callback the cursor positionally; `users_list`
        # takes it by keyword, so the lambda is the adapter between them.
        people, error = slack_api.paginate(lambda cursor: slack_api.users_list(cursor=cursor), "members")
        if error:
            console.print(f"[yellow]{slack_api.error_message(SlackResponse(ok=False, error=error))}[/yellow]")
            return 1
        needle = args.match.strip().lower()
        rows = []
        for person in people:
            # `users.list` returns bots, apps and the workspace's own Slackbot
            # alongside people. Every one of them would be a member id you can
            # never usefully link or allow, so none of them are offered.
            if person.get("deleted") or person.get("is_bot") or person.get("id") == "USLACKBOT":
                continue
            profile = person.get("profile") or {}
            label = profile.get("real_name") or person.get("real_name") or person.get("name") or ""
            handle = person.get("name") or ""
            if needle and needle not in f"{label} {handle}".lower():
                continue
            rows.append({"id": person.get("id", ""), "name": label, "handle": handle})
        if to_json:
            print(json.dumps({"members": rows}, indent=2))
        elif not rows:
            console.print("Nobody matched." if needle else "The workspace returned no people.")
        else:
            for row in rows:
                console.print(f"  {row['id']:<12} {row['name']}  [dim]@{row['handle']}[/dim]")
            console.print('\nUse an id in SLACK_ALLOWED_MEMBER_IDS, or `yeaboi slack link <id> "<name>"`.')
        return 0

    if command in ("link", "unlink"):
        from yeaboi.mcp.tools_sessions import resolve_session_id
        from yeaboi.slack.engine import link_slack_member
        from yeaboi.slack.identity import IdentityError

        try:
            session_id = resolve_session_id(args.session)
        except Exception:  # noqa: BLE001 — no saved session is a normal empty state
            session_id = ""
        try:
            result = link_slack_member(
                session_id,
                args.slack_user,
                getattr(args, "member", ""),
                unlink=command == "unlink",
            )
        except IdentityError as exc:
            # Named rather than silently dropped, for the allowlist's reason: a
            # link that did not happen leaves somebody believing their
            # corrections carry their name when they carry an id.
            if to_json:
                print(json.dumps({"error": str(exc)}, indent=2))
            else:
                console.print(f"[yellow]{exc}[/yellow]")
            return 1
        if to_json:
            print(json.dumps(result, indent=2))
        elif "identities" in result:
            rows = result["identities"]
            if not rows:
                console.print(
                    "Nothing linked — replies from Slack are attributed to the raw member id, which still works."
                )
            for row in rows:
                console.print(f"  {row.get('slack_user', ''):<12} {row.get('member', '')}")
        elif command == "unlink":
            console.print("Unlinked." if result.get("unlinked") else "Nothing was linked to that id.")
        else:
            console.print(f"Linked {result['linked']}.")
        return 0

    # poll
    from yeaboi.slack.engine import apply_inbound_events

    result = apply_inbound_events()
    if to_json:
        print(json.dumps(result, indent=2))
    else:
        console.print(f"{result['outcome']}: {result['detail'] or '-'}")
        if result["events_applied"]:
            console.print(f"  applied {result['events_applied']} of {result['events_seen']} seen")
    # A declined poll is not a failed one — the same rule the ceremonies runner
    # follows, so a cron job that could not act does not page anybody.
    return 1 if result["outcome"] == "failed" else 0


def _cmd_ceremonies(args: argparse.Namespace, console: Console) -> int:
    """Declare, inspect and fire recurring runs.

    ``ceremonies run --scheduled`` is what the installed OS job invokes, so this
    handler is on the unattended path: it prints, exits with a code, and never
    prompts. Every write goes through the store (which validates) and then the
    scheduler, in that order — a job installed for a ceremony the store refused
    would be one nothing can describe or remove.
    """
    import json
    from dataclasses import asdict

    from yeaboi.agent.state import Ceremony
    from yeaboi.ceremonies import catalog, render, scheduler
    from yeaboi.ceremonies.store import CeremonyStore

    command = args.ceremonies_command
    to_json = getattr(args, "format", "text") == "json"
    logging.getLogger(__name__).info("ceremonies %s", command)

    if command == "modes":
        if to_json:
            print(
                json.dumps(
                    {
                        "schedulable": [asdict(m) for m in catalog.CATALOG],
                        "refused": catalog.UNSCHEDULABLE,
                    },
                    indent=2,
                )
            )
        else:
            console.print(render.format_modes_rich())
        return 0

    from yeaboi.mcp.tools_sessions import resolve_session_id

    session_id = resolve_session_id(getattr(args, "session", ""))

    with CeremonyStore() as store:
        if command == "list":
            for gone in scheduler.reap_dead_jobs():
                print(f"⚠ removed a scheduled job that could never run again: {gone}", file=sys.stderr)
            declared = store.list(session_id)
            # The store and the operating system are two different things, and
            # the gap between them is invisible until something does not fire.
            # Three states, not two: "installed but paused" is a bug, whereas
            # "declared but not installed" is exactly what pause is for.
            installed = set(scheduler.installed_ceremonies(session_id))
            known = {c.name for c in declared}
            expected = {c.name for c in declared if c.enabled}
            drift = [
                f"a job is installed for {orphan!r}, which is not declared here" for orphan in sorted(installed - known)
            ]
            drift += [
                f"{zombie!r} is paused but its job is still installed"
                for zombie in sorted((installed & known) - expected)
            ]
            drift += [
                f"{missing!r} is declared but has no scheduled job — re-add it"
                for missing in sorted(expected - installed)
            ]
            if to_json:
                # Drift and the OS's own view ride along, because a scripted
                # caller is exactly who cannot notice a morning going quiet.
                print(
                    json.dumps(
                        {
                            "ceremonies": [asdict(c) for c in declared],
                            "installed_jobs": sorted(installed),
                            "drift": drift,
                        },
                        indent=2,
                    )
                )
                return 0
            last = {c.name: store.last_run(session_id, c.name) for c in declared}
            console.print(render.format_ceremonies_rich(declared, last))
            for line in drift:
                console.print(f"[yellow]![/yellow] {line}")
            return 0

        if command == "history":
            runs = store.runs(session_id, getattr(args, "name", ""), limit=args.limit)
            if to_json:
                print(json.dumps([asdict(r) for r in runs], indent=2))
            else:
                console.print(render.format_history_rich(runs))
            return 0

        if command == "add":
            mode = catalog.lookup(args.mode)
            if mode is None:
                print(f"Error: {catalog.refuse_reason(args.mode)}", file=sys.stderr)
                return 2
            declared_args: list[tuple[str, str]] = []
            for pair in args.arg:
                key, sep, value = pair.partition("=")
                if not sep:
                    print(f"Error: --arg expects KEY=VALUE, got {pair!r}", file=sys.stderr)
                    return 2
                declared_args.append((key.strip(), value.strip()))
            channels = tuple(c.strip() for c in args.channels.split(",") if c.strip())
            ceremony = store.save(
                Ceremony(
                    session_id=session_id,
                    name=args.name,
                    mode=mode.key,
                    args=tuple(declared_args),
                    weekdays=args.weekdays or mode.default_weekdays,
                    at=args.at,
                    channels=channels,
                    stale_after_min=args.stale_after,
                    monthly_cap_usd=args.monthly_cap,
                )
            )
            message = scheduler.install_ceremony(session_id, ceremony.name, ceremony.at, ceremony.weekdays)
            if to_json:
                print(json.dumps({"ceremony": asdict(ceremony), "scheduler": message}, indent=2))
            else:
                console.print(f"[green]✓[/green] {ceremony.name} — {render.cadence_label(ceremony)}")
                console.print(f"  → {', '.join(ceremony.channels)}")
                console.print(f"  {message}", style="dim")
            return 0

        if command == "remove":
            message = scheduler.remove_ceremony(session_id, args.name)
            removed = store.remove(session_id, args.name)
            if to_json:
                print(json.dumps({"removed": removed, "scheduler": message}, indent=2))
            elif removed:
                console.print(f"[green]✓[/green] removed {args.name}. {message}")
            else:
                # Still report the scheduler's half: an orphaned job with no
                # declaration is exactly the state worth telling someone about.
                console.print(f"No ceremony named {args.name!r}. {message}")
            return 0 if removed else 1

        if command in ("pause", "resume"):
            enabled = command == "resume"
            ceremony = store.set_enabled(session_id, args.name, enabled)
            if ceremony is None:
                print(f"Error: no ceremony named {args.name!r}", file=sys.stderr)
                return 1
            if enabled:
                message = scheduler.install_ceremony(session_id, ceremony.name, ceremony.at, ceremony.weekdays)
            else:
                # Pause removes the JOB and keeps the declaration: a paused
                # ceremony that still fires is the thing users report as a bug.
                message = scheduler.remove_ceremony(session_id, ceremony.name)
            if to_json:
                print(json.dumps({"ceremony": asdict(ceremony), "scheduler": message}, indent=2))
            else:
                console.print(f"[green]✓[/green] {ceremony.name} {command}d. {message}")
            return 0

        if command == "skip":
            from yeaboi.ceremonies.scheduler import next_occurrence

            existing = store.get(session_id, args.name)
            if existing is None:
                print(f"Error: no ceremony named {args.name!r}", file=sys.stderr)
                return 1
            # Resolved to a date here rather than stored as a flag: launchd
            # coalesces a missed slot into a fire the next morning, and a flag
            # would be spent on the occurrence the user already saw.
            occurrence = "" if args.clear else (args.on or next_occurrence(existing))
            if not occurrence and not args.clear:
                print(f"Error: could not work out {args.name!r}'s next run — pass --on YYYY-MM-DD", file=sys.stderr)
                return 1
            try:
                ceremony = store.set_skip_next(session_id, args.name, occurrence)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
            # The job is deliberately left installed: a one-day intent must not
            # churn a plist, and a crash mid-reinstall would kill the ceremony.
            if to_json:
                print(json.dumps({"ceremony": asdict(ceremony), "skipping": occurrence}, indent=2))
            elif occurrence:
                console.print(f"[green]✓[/green] {ceremony.name} will skip its {occurrence} run, then carry on.")
            else:
                console.print(f"[green]✓[/green] {ceremony.name} is no longer skipping a run.")
            return 0

    # run — outside the store context: the engine opens its own connection.
    from yeaboi.ceremonies.engine import CeremonyNotFoundError, run_ceremony

    try:
        run = run_ceremony(
            args.name,
            session_id=session_id,
            scheduled=args.scheduled,
            dry_run=args.dry_run,
            on_progress=None if to_json else lambda step: console.print(f"  {step}", style="dim"),
        )
    except CeremonyNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if to_json:
        print(json.dumps(asdict(run), indent=2))
    else:
        console.print(render.format_history_rich([run]))
    # A declined run is not a failed one: the guards did their job, and a
    # scheduled wrapper treating "skipped because the laptop was asleep" as a
    # crash would fill the logs with alarms about working behaviour.
    return 1 if run.outcome == "failed" else 0


def _cmd_ship(args: argparse.Namespace, console: Console) -> int:
    """The ship pipeline headless: same engine the TUI page uses
    (AGENTS.md "REQUIRED: Surface Parity"). The approval gate prompts on the
    terminal and resolves through the same ShipStore CAS as the TUI screen."""
    import json
    from dataclasses import asdict

    from yeaboi.beta import SHIP_BETA_NOTICE

    logging.getLogger(__name__).info("ship %s (beta)", args.ship_command)
    _print_beta_notice(SHIP_BETA_NOTICE)

    if args.ship_command == "history":
        from yeaboi.ship.render import format_history_rich
        from yeaboi.ship.store import ShipStore, listing_dict

        with ShipStore() as store:
            runs = store.list_runs(limit=args.limit)
        if args.format == "json":
            # `listing_dict` drops the stored patch: capped per run is not
            # capped per response, and a listing is polled in a loop. The
            # single run `ship run --format json` prints keeps it.
            print(json.dumps([listing_dict(r) for r in runs], indent=2))
        else:
            console.print(format_history_rich(runs))
        return 0

    if args.ship_command == "status":
        from yeaboi.ship import budget
        from yeaboi.ship.render import format_budget_rich, format_run_rich
        from yeaboi.ship.store import ShipStore, listing_dict

        with ShipStore() as store:
            runs = store.list_runs(limit=1)
        posture = budget.status()
        if args.format == "json":
            print(json.dumps({"latest": listing_dict(runs[0]) if runs else None, "budget": asdict(posture)}, indent=2))
        else:
            if runs:
                console.print(format_run_rich(runs[0]))
            else:
                console.print("No ship runs yet.")
            console.print(format_budget_rich(posture))
        return 0

    if args.ship_command == "resume":
        return _ship_resume(args, console)

    # run
    return _ship_run(args, console)


def _ship_run(args: argparse.Namespace, console: Console) -> int:
    """`yeaboi ship run` — engine on a worker thread, the gate answered here."""

    from yeaboi import fs_policy
    from yeaboi.ship import engine, worktree

    repo = str(Path(args.repo).expanduser().resolve())
    if not args.dry_run:
        try:
            # Resolve the toplevel BEFORE the consent check: every write lands
            # there (`git worktree add` into <toplevel>/.git, the later push),
            # and fs_policy containment is `is_relative_to` — so granting a
            # subdirectory would let yeaboi write outside the approved root.
            repo = str(worktree.resolve_repo(repo))
        except worktree.WorktreeError as exc:
            print(f"✗ {exc}", file=sys.stderr)
            return 2
        try:
            fs_policy.resolve_and_check(repo, mode="write", context="ship: run a coding agent against this repository")
        except PermissionError as exc:
            print(f"✗ {exc}", file=sys.stderr)
            return 2
        if not sys.stdin.isatty():
            # The gate is a human decision made at a terminal; without one the
            # run would hang forever awaiting an approval nobody can give.
            print("✗ ship run needs an interactive terminal — the approval gate prompts here.", file=sys.stderr)
            return 2

    # The engine hands its run id back the moment it exists. Anything else in
    # the store belongs to ANOTHER session (a TUI in another terminal, a stale
    # process) — prompting for one of those would let this user approve, and
    # push, a diff they are not looking at.
    mine: list[str] = [""]

    if args.split and args.dry_run:
        print("✗ --split has no dry run — a batch is a sequence of real runs.", file=sys.stderr)
        return 2

    batch: list = []

    def _work(cancel, on_run_id):
        if args.split:
            members = engine.run_ship_batch(
                args.item_id,
                repo,
                level=args.level,
                session_id=args.session,
                check_command=args.check,
                timeout_minutes=args.timeout_minutes,
                on_run_id=on_run_id,
                cancel_event=cancel,
            )
            batch.extend(members)
            # The last member that actually ran. A stopped batch ends in
            # unstarted rows, which were never persisted and describe nothing.
            started = [m for m in members if m.run_id]
            return started[-1] if started else (members[-1] if members else None)
        return engine.run_ship(
            args.item_id,
            repo,
            level=args.level,
            session_id=args.session,
            check_command=args.check,
            timeout_minutes=args.timeout_minutes,
            dry_run=args.dry_run,
            on_run_id=on_run_id,
            cancel_event=cancel,
        )

    return _ship_report(args, console, _ship_supervise(console, _work, mine=mine), batch=batch)


def _print_batch(console: Console, members: list) -> None:
    """One line per story: what shipped, what stopped it, what was never reached."""
    shipped = sum(1 for m in members if m.status == "approved")
    console.print(f"\nBatch: {shipped} of {len(members)} stories shipped")
    for member in members:
        detail = member.pr_url or (member.warnings[-1] if member.warnings else "")
        console.print(f"  {member.batch_index}. {member.item_id:22} {member.status:18} {detail}")


def _ship_resume(args: argparse.Namespace, console: Console) -> int:
    """`yeaboi ship resume` — finish a run abandoned at the gate.

    No repository consent prompt: the repo was granted when the run was launched,
    the worktree already exists, and resume writes nowhere new.
    """
    from yeaboi.ship import engine

    if not sys.stdin.isatty():
        print("✗ ship resume needs an interactive terminal — the approval gate prompts here.", file=sys.stderr)
        return 2

    def _work(cancel, _on_run_id):
        return engine.resume_ship(
            args.run_id,
            check_command=args.check,
            timeout_minutes=args.timeout_minutes,
            cancel_event=cancel,
        )

    # The run id is known up front here, unlike a fresh run — resume is addressed
    # at exactly one run, so the gate loop can watch it from the first tick.
    return _ship_report(args, console, _ship_supervise(console, _work, mine=[args.run_id]))


def _ship_supervise(console: Console, run_engine, *, mine: list[str]):
    """Drive one engine call on a worker thread, answering its gate at this terminal.

    ``run_engine(cancel, on_run_id)`` makes the call; ``mine`` carries the run id —
    pre-seeded by resume, filled by the callback for a fresh run — so this loop only
    ever opens the gate for its own run. Returns the ShipRun, or None if the engine
    reported nothing.
    """
    import threading
    import time

    from yeaboi.ship.render import format_run_rich
    from yeaboi.ship.store import ShipStore

    result_box: list = [None]
    cancel = threading.Event()

    def _work() -> None:
        result_box[0] = run_engine(cancel, lambda run_id: mine.__setitem__(0, run_id))

    worker = threading.Thread(target=_work, daemon=True)
    worker.start()
    try:
        with ShipStore() as store:
            while worker.is_alive():
                worker.join(timeout=0.5)
                if not worker.is_alive() or cancel.is_set():
                    continue  # a cancelled run winds down on its own; stop prompting
                if not mine[0]:
                    continue  # the id has not been minted yet; nothing can be ours
                # Fetched by id, never scanned out of a newest-first page: with
                # concurrency raised, runs started after ours would push it off
                # the end and its gate would never render here. An open gate is
                # one this loop has not answered yet — resolving stamps
                # gate_resolution and a rework clears it, so a reopened gate
                # prompts again by construction.
                run = store.get_run(mine[0])
                if run is not None and run.status == "awaiting_approval" and not run.gate_resolution:
                    # The gate is the only control before a push; it shows the
                    # patch, not a file count.
                    console.print(format_run_rich(run, show_diff=True))
                    try:
                        answer = input("Approve and open a PR? [y]es / [n]o with feedback / [c]ancel run: ")
                    except EOFError:
                        answer = "c"
                    answer = answer.strip().lower()
                    if answer in ("y", "yes"):
                        store.resolve_gate(run.run_id, "approved")
                    elif answer in ("n", "no"):
                        try:
                            comment = input("Feedback for the agent's rework: ").strip()
                        except EOFError:
                            comment = ""
                        store.resolve_gate(run.run_id, "rejected", comment)
                    else:
                        cancel.set()
                time.sleep(0.5)
    except KeyboardInterrupt:
        cancel.set()
        print("Cancelling the run — the agent is stopped and nothing is pushed…", file=sys.stderr)
        worker.join(timeout=60)
    worker.join(timeout=5)
    return result_box[0]


def _ship_report(args: argparse.Namespace, console: Console, run, *, batch: list | None = None) -> int:
    """Print a finished run and pick the exit code. Shared by run and resume.

    ``batch`` is every member of a ``--split`` run; nothing but the members'
    own rows may reach stdout before the JSON document, or it stops parsing.
    """
    import json
    from dataclasses import asdict

    from yeaboi.ship.render import format_run_rich

    if run is None:
        print("✗ the run did not report a result — see logs", file=sys.stderr)
        return 1
    for warning in run.warnings:
        print(f"⚠ {warning}", file=sys.stderr)
    if args.format == "json":
        from yeaboi.ship.store import listing_dict

        payload = asdict(run)
        payload["story_id"] = payload["item_id"]  # the legacy mirror every ship payload carries
        if batch:
            payload["batch"] = [listing_dict(m) for m in batch]
        print(json.dumps(payload, indent=2))
    else:
        if batch:
            _print_batch(console, batch)
        console.print(format_run_rich(run))
    if args.strict and run.status != "approved":
        print(f"strict: run ended {run.status} — exit 3", file=sys.stderr)
        return 3
    return 0


def _cmd_agents_security_actions(args: argparse.Namespace, console: Console) -> int:
    """The security verbs that act on the last scan — replay, signals, fixes — never a new one."""
    import json
    from dataclasses import asdict

    from yeaboi.agentwatch import security_fixes
    from yeaboi.agentwatch.engine import rebuild_security_report

    if args.list_fixes:
        from yeaboi.agentwatch.store import AgentWatchStore
        from yeaboi.paths import get_db_path

        with AgentWatchStore(get_db_path()) as store:
            rows = store.list_fixes()
        if args.format == "json":
            print(json.dumps(rows, indent=2))
        elif not rows:
            console.print("No fixes applied yet.")
        else:
            for row in rows:
                console.print(f"{row['applied_at'][:19]}  {row['fix_id']:<14} {row['pattern']:<26} {row['outcome']}")
        return 0
    if args.mark_test_data:
        from yeaboi.agentwatch import dismissals

        for key in args.mark_test_data:
            dismissals.dismiss(key, reason="test data: fixture or example text", by=os.environ.get("USER", ""))
            console.print(f"[green]✓[/green] {key} marked as test data")
        return 0
    if args.fix:
        outcome = security_fixes.apply_fix(args.fix, args.fix_id, reason=args.reason, repo=args.repo)
        if args.format == "json":
            print(json.dumps(asdict(outcome), indent=2))
        elif outcome.ok:
            console.print(f"[green]✓[/green] {outcome.detail}" + (f"\n  {outcome.pr_url}" if outcome.pr_url else ""))
        else:
            print(f"✗ {outcome.detail}", file=sys.stderr)
        return 0 if outcome.ok else 1
    report = rebuild_security_report(include_info=True, record=False)
    key = args.replay or args.signals
    finding = next((f for f in report.findings if f.key == key), None)
    if finding is None:
        print(f"✗ no finding {key!r} in the latest scan — run `yeaboi agents security` first", file=sys.stderr)
        return 1
    if args.signals:
        from yeaboi.agentwatch.store import AgentWatchStore
        from yeaboi.paths import get_db_path

        with AgentWatchStore(get_db_path()) as store:
            rows = store.list_findings_for_key(
                category=finding.category,
                pattern=finding.pattern,
                source_path=finding.location,
                context=finding.context,
            )
        if args.format == "json":
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                console.print(
                    f"line {row['line_no']:<7} {str(row.get('at') or '')[:19]:<20} {row.get('snippet') or ''}"
                )
        return 0
    from yeaboi.agentwatch import replay as replay_mod
    from yeaboi.agentwatch.render import format_replay_rich

    try:
        result = replay_mod.replay(finding.location, args.line or finding.line_no, pattern=finding.pattern)
    except replay_mod.ReplayError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps(asdict(result), indent=2))
    else:
        console.print(format_replay_rich(result))
    return 0


def _cmd_agents_dismissals(args: argparse.Namespace, console: Console) -> int:
    """The security dismissal verbs: a hand-kept allowlist, never a scan."""
    import json

    from yeaboi.agentwatch import dismissals

    if args.dismiss:
        try:
            entry = dismissals.dismiss(args.dismiss, reason=args.reason, by=os.environ.get("USER", ""))
        except ValueError as exc:
            print(f"✗ {exc}", file=sys.stderr)
            return 2
        console.print(f"[green]✓[/green] dismissed {entry.key} — {entry.reason}")
        return 0
    if args.undismiss:
        if dismissals.undismiss(args.undismiss):
            console.print(f"[green]✓[/green] restored {args.undismiss}")
            return 0
        print(f"✗ no dismissal on file for {args.undismiss}", file=sys.stderr)
        return 1
    rows = dismissals.load()
    if args.format == "json":
        print(json.dumps([r.__dict__ for r in rows], indent=2))
    elif not rows:
        console.print("No dismissed findings.")
    else:
        for row in rows:
            expiry = f" (until {row.expires})" if row.expires else ""
            console.print(f"• {row.key}{expiry} — {row.reason}")
    return 0


def _cmd_agents(args: argparse.Namespace, console: Console) -> int:
    """The Agents family headless: same engines the TUI cards and MCP tools use
    (AGENTS.md "REQUIRED: Surface Parity")."""
    logging.getLogger(__name__).info("agents %s (beta)", args.agents_command)
    _print_beta_notice(AGENTWATCH_BETA_NOTICE)

    if args.agents_command == "cost":
        import json
        from dataclasses import asdict

        from yeaboi.agentwatch.engine import run_agent_usage
        from yeaboi.agentwatch.render import format_usage_rich

        report = run_agent_usage(
            window_days=args.window_days, project=args.project, source=args.source, project_path=args.repo
        )
        for warning in report.warnings:
            print(f"⚠ {warning}", file=sys.stderr)
        if args.format == "json":
            print(json.dumps(asdict(report), indent=2))
        else:
            console.print(format_usage_rich(report))
        return _strict_exit(args.strict, report.warnings, empty=report.session_count == 0)

    if args.agents_command == "advisor":
        import json
        from dataclasses import asdict

        from yeaboi.agentwatch.advisor import run_agent_advisor
        from yeaboi.agentwatch.render import format_advisor_rich

        report = run_agent_advisor(window_days=args.window_days, project_path=args.repo)
        for warning in report.warnings:
            print(f"⚠ {warning}", file=sys.stderr)
        if args.format == "json":
            print(json.dumps(asdict(report), indent=2))
        else:
            console.print(format_advisor_rich(report))
        return _strict_exit(args.strict, report.warnings, empty=report.session_count == 0)

    if args.agents_command == "security":
        import json
        from dataclasses import asdict

        from yeaboi.agentwatch.engine import run_agent_security
        from yeaboi.agentwatch.render import format_security_rich

        if args.list_dismissed or args.dismiss or args.undismiss:
            return _cmd_agents_dismissals(args, console)
        if args.replay or args.signals or args.fix or args.mark_test_data or args.list_fixes:
            return _cmd_agents_security_actions(args, console)
        report = run_agent_security(deep=args.deep, include_info=args.include_info)
        for warning in report.warnings:
            print(f"⚠ {warning}", file=sys.stderr)
        if args.format == "json":
            print(json.dumps(asdict(report), indent=2))
        else:
            console.print(format_security_rich(report, expanded=("needs-decision", "unsure", "test-data", "handled")))
        return _strict_exit(args.strict, report.warnings)

    return 1


def _cmd_retro(args: argparse.Namespace, console: Console) -> int:
    """Read-back of past retro boards. The live collaborative board needs a TTY
    host and stays in the TUI (see the surface-parity registry)."""
    import json

    from yeaboi.paths import get_db_path
    from yeaboi.retro.store import RetroStore

    session_id = _resolve_cli_session(args.session)
    if not session_id:
        print("Error: no session found to read retros for.", file=sys.stderr)
        return 2
    with RetroStore(get_db_path()) as store:
        history = store.get_history(session_id, limit=args.limit)
        # Load the latest report for the carried-items summary (and for --export).
        latest = store.get_latest_report(session_id)
    exported: dict = {}
    if args.export:
        if latest is None:
            print("Error: no retro recorded yet — run a retro board from the TUI first.", file=sys.stderr)
            return 2
        from yeaboi.retro.export import export_retro

        exported = {k: str(v) for k, v in export_retro(latest, history=history).items()}
    # Summarise the latest retro's carried-over action items by status.
    carried = list(latest.carried_action_items) if latest else []
    carried_summary = ""
    if carried:
        from collections import Counter

        from yeaboi.retro.board import CARRIED_STATUS_LABELS

        counts = Counter((c.status or "pending") for c in carried)
        carried_summary = " · ".join(
            f"{n} {CARRIED_STATUS_LABELS.get(st, st).lower()}" for st, n in counts.most_common()
        )
    if args.format == "json":
        print(
            json.dumps(
                {
                    "session_id": session_id,
                    "history": history,
                    "exported": exported,
                    "carried_action_items": [{"text": c.text, "status": c.status or "pending"} for c in carried],
                },
                indent=2,
            )
        )
        return 0
    if not history:
        console.print("[yellow]No retros recorded for this session yet — run one from the TUI Retro page.[/yellow]")
        return 0
    for row in history:
        console.print(
            f"  • {row['retro_date'] or row['run_at'][:10]}  {row['project_name'] or session_id}"
            f"  — {row['card_count']} cards"
        )
    if carried_summary:
        console.print(f"  Last sprint's actions: {carried_summary}")
    for kind, path in exported.items():
        console.print(f"  Exported {kind}: {path}")
    return 0


def _parse_marks(marks: list[str]) -> dict[str, str]:
    """``ID=STATUS`` pairs from repeated --mark flags; a malformed one is an error."""
    from yeaboi.solo.engine import ACTION_STATUSES

    parsed: dict[str, str] = {}
    for raw in marks:
        action_id, sep, status = raw.partition("=")
        action_id, status = action_id.strip(), status.strip().lower()
        if not sep or not action_id or status not in ACTION_STATUSES:
            raise ValueError(f"--mark expects ID=STATUS with STATUS one of {', '.join(ACTION_STATUSES)} — got {raw!r}")
        parsed[action_id] = status
    return parsed


def _cmd_review(args: argparse.Namespace, console: Console) -> int:
    """`yeaboi review {run,history,export}` — the Solo world's Weekly Review."""
    command = getattr(args, "review_command", None)
    if command is None:
        print("Usage: yeaboi review {run,history,export}", file=sys.stderr)
        return 2
    if command == "run":
        return _cmd_review_run(args, console)
    if command == "history":
        return _cmd_review_history(args, console)
    return _cmd_review_export(args, console)


def _cmd_review_run(args: argparse.Namespace, console: Console) -> int:
    from yeaboi.solo.engine import run_weekly_review
    from yeaboi.solo.render import format_review_rich

    _print_beta_notice(WEEKLY_REVIEW_BETA_NOTICE)
    try:
        marks = _parse_marks(args.mark)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    console.print("[bold cyan]Reviewing your week...[/bold cyan]")
    review = run_weekly_review(
        session_id=_resolve_cli_session(args.session) or "",
        week_end=args.week_end,
        carried_statuses=marks or None,
        **_context_kwargs(args),
    )
    for warning in review.warnings:
        print(f"⚠ {warning}", file=sys.stderr)
    if args.format == "json":
        print(_json_dump(review))
    else:
        from yeaboi.paths import SOLO_EXPORTS_DIR

        console.print(format_review_rich(review))
        console.print(f"[dim]Exports (Markdown): {SOLO_EXPORTS_DIR}[/dim]")
    return _strict_exit(args.strict, review.warnings, empty=not (review.summary or review.went_well))


def _cmd_review_history(args: argparse.Namespace, console: Console) -> int:
    import json

    from yeaboi.paths import get_db_path
    from yeaboi.solo.engine import carried_actions
    from yeaboi.solo.store import WeeklyReviewStore

    path = get_db_path()
    with WeeklyReviewStore(path) as store:
        history = store.get_all_history(limit=args.limit)
    carried = carried_actions(db_path=path)
    if args.format == "json":
        print(
            json.dumps(
                {
                    "history": history,
                    "carried": [{"id": a.id, "text": a.text, "status": a.status} for a in carried],
                },
                indent=2,
            )
        )
        return 0
    if not history:
        console.print("[yellow]No weekly reviews yet — run `yeaboi review run` at the end of a week.[/yellow]")
        return 0
    for row in history:
        console.print(
            f"  • #{row['id']}  {row['week_label']}  {row['project_name'] or 'all projects'}"
            f"  — {row['action_count']} action(s)"
        )
    if carried:
        console.print("  Open from last week (mark with `review run --mark ID=done`):")
        for action in carried:
            console.print(f"    {action.id}  {action.text}")
    return 0


def _cmd_review_export(args: argparse.Namespace, console: Console) -> int:
    from yeaboi.paths import get_db_path
    from yeaboi.solo.export import export_weekly_review
    from yeaboi.solo.store import WeeklyReviewStore

    with WeeklyReviewStore(get_db_path()) as store:
        review = store.get_run_by_id(args.run_id) if args.run_id else store.get_latest_report()
    if review is None:
        print("Error: no weekly review recorded yet — run `yeaboi review run` first.", file=sys.stderr)
        return 2
    paths = export_weekly_review(review)
    console.print(f"  Exported markdown: {paths['markdown']}")
    return 0


def _cmd_poker(args: argparse.Namespace, console: Console) -> int:
    """Read-back of past poker sessions. The live voting board needs a TTY host
    and stays in the TUI (see the surface-parity registry)."""
    import json

    from yeaboi.paths import get_db_path
    from yeaboi.poker.store import PokerStore

    with PokerStore(get_db_path()) as store:
        # Poker sessions often run under auto-created quick sessions, so the
        # default listing is cross-session; --session narrows it.
        rows = store.get_all_history(200)
        if args.session:
            rows = [r for r in rows if r.get("session_id") == args.session]
        rows = rows[: args.limit]
        latest = store.get_run_by_id(rows[0]["id"]) if rows else None
    exported: dict = {}
    if args.export:
        if latest is None:
            print("Error: no poker session recorded yet — run one from the TUI Poker page.", file=sys.stderr)
            return 2
        from yeaboi.poker.export import export_poker

        with PokerStore(get_db_path()) as _store:
            run_history = _store.get_history(latest.session_id, limit=30) if latest.session_id else []
        exported = {k: str(v) for k, v in export_poker(latest, history=run_history).items()}
    if args.format == "json":
        print(json.dumps({"history": rows, "exported": exported}, indent=2))
        return 0
    if not rows:
        console.print("[yellow]No poker sessions recorded yet — run one from the TUI Poker page.[/yellow]")
        return 0
    for row in rows:
        scope = " · ".join(p for p in (row.get("source"), row.get("scope_label")) if p)
        console.print(
            f"  • {row['poker_date'] or row['run_at'][:10]}  {scope or row.get('project_name') or '—'}"
            f"  — {row['estimated_count']}/{row['ticket_count']} estimated"
        )
    for kind, path in exported.items():
        console.print(f"  Exported {kind}: {path}")
    return 0


def _cmd_analyze(args: argparse.Namespace, console: Console) -> int:
    from rich.table import Table

    from yeaboi.analysis import run_team_analysis

    # Each component runs over its own sub-sources; --members applies to whichever
    # delivery tracker(s) run (the engine reads the per-tracker entry) and to code authors.
    components: dict[str, list[str]] = {}
    if args.delivery:
        components["delivery"] = args.delivery
    if args.code:
        components["code"] = args.code
    if args.docs:
        components["docs"] = args.docs
    if args.ops:
        components["ops"] = args.ops
    members = {"jira": args.members, "azdevops": args.members} if args.members else None
    analysis_scope = {
        provider: values
        for provider, values in {
            "github": getattr(args, "github_owner", None),
            "azdo": getattr(args, "azdo_code_project", None),
            "confluence": getattr(args, "confluence_space", None),
            "notion": getattr(args, "notion_root", None),
        }.items()
        if values
    }

    # An explicit components map is authoritative for delivery, so --source only takes
    # effect when delivery is left to the default. Warn rather than silently ignore it.
    if components and "delivery" not in components and args.source:
        print(
            f"⚠ --source {args.source} ignored: no --delivery selected (code/docs are global scans)",
            file=sys.stderr,
        )

    console.print("[bold cyan]Analysing team history...[/bold cyan]")
    result = run_team_analysis(
        source=args.source,
        project_key=args.project,
        sprint_count=args.sprints,
        generate_samples=args.samples,
        include_insights=not args.no_insights,
        include_ai_usage=args.include_ai_usage,
        include_doc_quality=args.include_doc_quality,
        analysis_depth=args.depth,
        analysis_window_days=getattr(args, "window_days", 120),
        analysis_scope=analysis_scope or None,
        analysis_model=getattr(args, "analysis_model", None),
        analysis_features=getattr(args, "features", None),
        components=components or None,
        members=members,
        **_context_kwargs(args),
    )
    for warning in result["warnings"]:
        print(f"⚠ {warning}", file=sys.stderr)
    if args.format == "json":
        print(_json_dump(result))
        return _strict_exit(args.strict, result["warnings"])
    # Delivery — one section per tracker, under a "From Jira"/"From Azure DevOps"
    # banner so it's clear which velocity numbers came from which tracker.
    _source_names = {"jira": "From Jira", "azdevops": "From Azure DevOps"}
    for tracker, sub in result.get("delivery", {}).items():
        console.rule(f"[bold cyan]{_source_names.get(tracker, tracker)}[/bold cyan]")
        _print_profile_summary(console, sub)
    comparison = result.get("comparison") or []
    if comparison:
        table = Table(title="Delivery — side by side", show_header=True)
        table.add_column("Metric", style="bold")
        table.add_column("Jira")
        table.add_column("Azure DevOps")
        for label, jira_val, azdo_val in comparison:
            table.add_row(label, jira_val, azdo_val)
        console.print(table)
    # Code + Docs — single global scans, printed once.
    if result.get("code"):
        _print_code_summary(console, result["code"])
    if result.get("docs"):
        _print_docs_summary(console, result["docs"])
    return _strict_exit(args.strict, result["warnings"])


def _print_profile_summary(console: Console, sub: dict) -> None:
    """Print one delivery tracker's summary (saved profile + top coaching insights)."""
    profile = sub["profile"]
    console.print(f"[green]Team profile saved for {profile.source}/{profile.project_key}[/green]")
    console.print(
        f"  Analysed [bold]{profile.sample_sprints}[/bold] sprints, [bold]{profile.sample_stories}[/bold] stories"
    )
    console.print(f"  Avg velocity: [bold]{profile.velocity_avg:.0f} ± {profile.velocity_stddev:.0f}[/bold] pts/sprint")
    insights = sub.get("insights") or {}
    for category in ("start", "stop", "keep", "try"):
        for item in insights.get(category, [])[:2]:
            console.print(f"  [bold]{category.upper()}[/bold]: {item.get('title', '')}")


def _print_code_summary(console: Console, code: dict) -> None:
    """Print selected-user Code activity, changed-file, and coverage summary."""
    sig = code.get("signal")
    examples = code.get("examples") or {}
    enabled = set(examples.get("enabled_features") or ("ai_footprint", "code_health"))
    console.rule("[bold cyan]Code — selected-user analysis[/bold cyan]")
    if "ai_footprint" in enabled:
        from yeaboi.analysis.ai_usage import footprint_small_sample

        fp = getattr(sig, "footprint_pct", 0.0)
        scanned = getattr(sig, "scanned_commits", 0) + getattr(sig, "scanned_prs", 0)
        marked = getattr(sig, "ai_commits", 0) + getattr(sig, "ai_prs", 0)
        if sig is not None and footprint_small_sample(sig):
            console.print(
                f"  AI-marked: [bold]{marked} of {scanned}[/bold] commits/PRs "
                "(small sample — % suppressed; lower bound)"
            )
        else:
            console.print(f"  AI footprint: [bold]{fp:.0f}%[/bold] of {scanned} commits/PRs (lower bound)")
    health = examples.get("repository_health") or {}
    if "code_health" in enabled:
        console.print(
            f"  Changed files: [bold]{health.get('files_analysed', 0)}[/bold] analysed · "
            f"{health.get('repositories_touched', 0)} repositories touched · {health.get('findings', 0)} findings"
        )
    selected = examples.get("selected_users") or []
    unmatched = examples.get("unmatched_users") or []
    if selected:
        console.print(f"  Selected users: {', '.join(selected)}")
    if unmatched:
        console.print(f"  Unmatched users: [yellow]{', '.join(unmatched)}[/yellow]")
    coverage = examples.get("coverage_report") or {}
    console.print(
        f"  Coverage: [bold]{str(coverage.get('status', 'complete')).upper()}[/bold] · "
        f"{coverage.get('succeeded', 0)}/{coverage.get('eligible', 0)} eligible assets succeeded"
    )
    for action in (examples.get("action_plan") or [])[:5]:
        console.print(f"  [bold]{str(action.get('priority', '')).upper()}[/bold]: {action.get('title', '')}")


def _print_docs_summary(console: Console, docs: dict) -> None:
    """Print the global Docs (clarity) scan summary."""
    sig = docs.get("signal")
    examples = docs.get("examples") or {}
    console.rule("[bold cyan]Docs — clarity[/bold cyan]")
    from yeaboi.analysis.doc_quality import doc_small_sample

    window = examples.get("window_days")
    small = " (small sample)" if sig is not None and doc_small_sample(sig) else ""
    console.print(
        f"  Clarity: [bold]{getattr(sig, 'avg_clarity', 0):.0f}/100[/bold] · "
        f"Usefulness: [bold]{getattr(sig, 'avg_usefulness', 0):.0f}/100[/bold] · "
        f"{getattr(sig, 'pages_scanned', 0)} pages" + (f" · last {window} days" if window else "") + small
    )
    coverage = examples.get("coverage_report") or {}
    console.print(
        f"  Coverage: [bold]{str(coverage.get('status', 'complete')).upper()}[/bold] · "
        f"{coverage.get('succeeded', 0)}/{coverage.get('eligible', 0)} eligible assets succeeded"
    )
    for action in (examples.get("action_plan") or [])[:5]:
        console.print(f"  [bold]{str(action.get('priority', '')).upper()}[/bold]: {action.get('title', '')}")


def _run_learn(console: Console) -> None:
    """Run the full team analysis via the analysis engine and print a summary.

    Delegates to analysis/engine.py:run_team_analysis — the same pipeline the
    TUI Analysis mode and the MCP team_analyze tool use — so the CLI stores the
    identical rich profile (with examples) in the real sessions DB.
    """
    from rich.table import Table

    console.print("[bold cyan]Analysing team history...[/bold cyan]")
    try:
        from yeaboi.analysis import run_team_analysis

        result = run_team_analysis(include_insights=False)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    for warning in result["warnings"]:
        console.print(f"[yellow]⚠ {warning}[/yellow]")

    # --learn auto-detects a single tracker → one delivery profile.
    delivery = result.get("delivery") or {}
    if not delivery:
        console.print("[yellow]No delivery profile produced (no tracker configured).[/yellow]")
        return
    profile = next(iter(delivery.values()))["profile"]

    console.print(f"[green]Team profile saved for {profile.source}/{profile.project_key}[/green]")
    console.print(
        f"  Analysed [bold]{profile.sample_sprints}[/bold] sprints, [bold]{profile.sample_stories}[/bold] stories"
    )
    console.print(f"  Avg velocity: [bold]{profile.velocity_avg:.0f} ± {profile.velocity_stddev:.0f}[/bold] pts/sprint")
    console.print(f"  Estimation accuracy: [bold]{profile.estimation_accuracy_pct:.0f}%[/bold]")
    console.print(f"  Sprint completion rate: [bold]{profile.sprint_completion_rate:.0f}%[/bold]")

    if profile.point_calibrations:
        table = Table(title="Story Point Calibration", show_header=True)
        table.add_column("Points", style="bold")
        table.add_column("Avg Cycle Time")
        table.add_column("Samples")
        table.add_column("Overshoot %")
        for cal in profile.point_calibrations:
            if cal.sample_count > 0:
                table.add_row(
                    str(cal.point_value),
                    f"{cal.avg_cycle_time_days:.1f} days",
                    str(cal.sample_count),
                    f"{cal.overshoot_pct:.0f}%",
                )
        console.print(table)


def _run_team_profile(console: Console) -> None:
    """Display the current stored team calibration profile."""

    from yeaboi.paths import get_db_path
    from yeaboi.team_profile import TeamProfileStore

    db_path = get_db_path()
    if not db_path.exists():
        console.print("[yellow]No team profiles found. Run --learn first.[/yellow]")
        return

    with TeamProfileStore(db_path) as store:
        profiles = store.list_profiles()

    if not profiles:
        console.print("[yellow]No team profiles found. Run --learn first.[/yellow]")
        return

    for profile in profiles:
        console.print(
            f"\n[bold cyan]{profile.team_id}[/bold cyan] "
            f"({profile.sample_sprints} sprints, {profile.sample_stories} stories)"
        )
        console.print(f"  Velocity: {profile.velocity_avg:.0f} ± {profile.velocity_stddev:.0f} pts/sprint")
        console.print(f"  Estimation accuracy: {profile.estimation_accuracy_pct:.0f}%")
        console.print(f"  Sprint completion rate: {profile.sprint_completion_rate:.0f}%")
        if profile.point_calibrations:
            console.print("  [dim]Point calibrations:[/dim]")
            for cal in profile.point_calibrations:
                if cal.sample_count > 0:
                    console.print(
                        f"    {cal.point_value} pt → {cal.avg_cycle_time_days:.1f} day avg "
                        f"({cal.sample_count} samples, {cal.overshoot_pct:.0f}% overshoot)"
                    )


def _run_retro(console: Console, session_id: str) -> None:
    """Run compare_plan_to_actuals and display the result."""
    import json

    from yeaboi.tools.team_learning import compare_plan_to_actuals

    console.print(f"[bold cyan]Comparing plan to actuals for session: {session_id}[/bold cyan]")
    try:
        result = compare_plan_to_actuals.invoke({"session_id": session_id})
        data = json.loads(result)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    if "error" in data:
        console.print(f"[red]{data['error']}[/red]")
        return

    console.print(f"  Session: [bold]{data.get('session_id', session_id)}[/bold]")
    console.print(f"  Planned stories: {data.get('planned_story_count', 0)}")
    console.print(f"  Planned sprints: {data.get('planned_sprint_count', 0)}")
    console.print(f"  Planned points: {data.get('planned_total_points', 0)}")
    console.print(f"  Tracker: {data.get('tracker', 'none')}")
    console.print(f"  Matched stories: {data.get('matched_stories', 0)}")
    if "note" in data:
        console.print(f"  [yellow]{data['note']}[/yellow]")


def _seed_allowed_paths_from_standup() -> None:
    """One-time sandbox grandfathering for pre-sandbox standup configs.

    Standup repo paths configured before the filesystem sandbox existed would
    silently stop being scanned. Exactly once — while YEABOI_ALLOWED_PATHS has
    never been set (unset, not empty) — copy any configured repo_path values
    into the whitelist and log it. Users who later edit or clear the whitelist
    are never re-seeded.
    """
    if os.getenv("YEABOI_ALLOWED_PATHS") is not None:
        return
    db = paths.DB_PATH
    if not db.exists():
        return
    try:
        import sqlite3

        with sqlite3.connect(str(db)) as conn:
            rows = conn.execute("SELECT DISTINCT repo_path FROM standup_config WHERE repo_path != ''").fetchall()
    except Exception:  # noqa: BLE001 — missing table/old schema: nothing to seed
        return
    repo_paths = [r[0] for r in rows if r and r[0]]
    if not repo_paths:
        return
    from yeaboi.config import set_allowed_paths

    set_allowed_paths(repo_paths)
    logging.getLogger(__name__).info(
        "sandbox: seeded YEABOI_ALLOWED_PATHS from existing standup repo paths: %s", repo_paths
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the yeaboi CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Migrate the config tree from the pre-rebrand ~/.scrum-agent dir BEFORE any
    # read/mkdir of ~/.yeaboi. Must run ahead of load_user_config() (which mkdirs
    # the config dir) and ahead of the headless/standup flows that return early,
    # otherwise those paths would create an empty ~/.yeaboi and skip migration.
    paths.migrate_root_dir()

    # Load ~/.yeaboi/.env before any credential reads.
    # override=False means shell env vars and project .env always take precedence.
    load_user_config()

    # --ac-format rides the YEABOI_AC_FORMAT env override that resolve_ac_style
    # already reads — one seam serves the TUI, REPL and headless paths alike.
    if getattr(args, "ac_format", None):
        os.environ["YEABOI_AC_FORMAT"] = args.ac_format
    # Same seam for the architecture spike ("auto" = the built-in behaviour,
    # so only a forced include/skip needs the override).
    if getattr(args, "architecture_spike", None) in ("include", "skip"):
        os.environ["YEABOI_ARCHITECTURE_SPIKE"] = args.architecture_spike
    # A subscription token expires and says nothing about when, so check it once
    # per launch — off the startup path, since nothing on the first screen waits
    # on the answer and the duck picks the warning up on a later frame. A no-op
    # unless subscription auth is actually configured.
    from yeaboi.auth_state import probe_in_background

    probe_in_background()

    # ── --list-audio-devices: print the mic table and exit ───────────────────
    # After load_user_config() so the currently-configured VOICE_DEVICE can be
    # marked, and before anything interactive so it never triggers the wizard.
    if args.list_audio_devices:
        _list_audio_devices()
        return

    # ── --setup-access: configure the verified-users tier and exit ───────────
    # Logging first, for the same reason --install-voice does it: cloudflared's
    # raw child output goes to the log at DEBUG, and on this surface the log is
    # the only diagnostic when a Cloudflare step refuses.
    if args.setup_access:
        from yeaboi.logging_setup import configure_logging

        configure_logging()
        raise SystemExit(_setup_access())

    # ── --install-voice: set dictation up headlessly and exit ────────────────
    # Logging is configured first, unlike the exits above it: this is the CI and
    # dev-container surface, where the log file is the only diagnostic and the
    # installer's raw child output (logged at DEBUG) is the whole point of it.
    # Without this the records go to a handler-less root logger and vanish.
    if args.install_voice:
        from yeaboi.logging_setup import configure_logging

        configure_logging()
        raise SystemExit(_install_voice())

    # ── Filesystem sandbox: session-scoped --allow-path grants ───────────────
    # Applied before any flow that might touch user-supplied paths, so every
    # denial message's "--allow-path" remedy actually works for the same
    # command re-run. Session-only: nothing is persisted (see fs_policy.py).
    for allowed in args.allow_path:
        fs_policy.grant_session(allowed)
    _seed_allowed_paths_from_standup()

    # ── Validation for --non-interactive mode ────────────────────────────────
    if args.non_interactive and not args.description:
        print("Error: --non-interactive requires --description", file=sys.stderr)
        sys.exit(1)

    if args.output and not args.non_interactive and not args.export_only:
        print("Error: --output is only valid with --non-interactive or --export-only", file=sys.stderr)
        sys.exit(1)

    # Resolve --description @file.txt → read file contents
    if args.description and args.description.startswith("@"):
        try:
            desc_path = fs_policy.resolve_and_check(args.description[1:], mode="read", context="--description @file")
        except fs_policy.SandboxViolationError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        if not desc_path.exists():
            print(f"Error: description file not found: {desc_path}", file=sys.stderr)
            sys.exit(1)
        args.description = desc_path.read_text().strip()

    # ── Subcommand dispatch (yeaboi report/standup/perf/analyze) ─────────────
    # Headless mode runners over the shared engines — no TUI, no splash.
    if getattr(args, "command", None):
        from yeaboi.logging_setup import configure_logging

        configure_logging()
        sys.exit(_run_subcommand(args))

    # ── Daily Standup headless flow ──────────────────────────────────────────
    # What the OS scheduler (launchd/cron) invokes: run a standup and deliver it,
    # with no TUI/splash. Runs before the interactive setup below.
    if args.standup_run:
        sys.exit(_run_standup(args))

    if args.standup_remind_transcript:
        sys.exit(_run_transcript_reminder(args))

    # ── Non-interactive headless flow ────────────────────────────────────────
    # Runs the full pipeline without any TUI, splash, or interactive input.
    if args.non_interactive:
        from yeaboi.logging_setup import configure_logging

        configure_logging()
        _run_headless(args)
        return

    # ── Migrate legacy file structure ────────────────────────────────────────
    from yeaboi.paths import migrate_legacy_paths

    migrate_legacy_paths()

    # ── File-based logging ────────────────────────────────────────────────────
    # Writes to ~/.yeaboi/logs/tui/yeaboi.log so developers can diagnose issues
    # without interfering with the TUI display. Rotates at 2 MB. Log level is
    # controlled by LOG_LEVEL in .env (default: WARNING; DEBUG for diagnostics)
    # and can be changed live from the Settings page.
    from yeaboi.logging_setup import configure_logging

    configure_logging()

    # Interactive-path imports — deferred so the headless flows above (and
    # --version/--help) never pay for rich or the setup/splash machinery.
    from rich.console import Console

    from yeaboi.formatters import build_theme
    from yeaboi.persistence import migrate_history_file
    from yeaboi.setup_wizard import is_first_run, run_setup_wizard
    from yeaboi.ui.splash import show_splash

    # Create the console with the requested theme so semantic style names
    # ([command], [hint], [success], etc.) resolve correctly throughout the
    # REPL. Must be created after arg parsing so args.theme is available.
    console = Console(theme=build_theme(args.theme))

    # Rename legacy history file (~/.scrum-agent/history → repl-history).
    # See docs: "Memory & State" — clearer naming for the REPL history file.
    migrate_history_file()

    # See docs: "Architecture" — splash replaces the static welcome panel.
    # The animated intro runs before any interactive UI (wizard / mode select).
    # Skipped when this process was relaunched by the ctrl+U in-app update: the
    # user just watched the upgrade land and doesn't need the ~2s intro again.
    # select_mode's Live(screen=True) enters the alternate screen on its own, so
    # the only thing lost is the animation.
    from yeaboi import update_check

    if not update_check.is_fresh_restart():
        show_splash(console)

    # ── First-run setup wizard ────────────────────────────────────────────────
    # Triggers when ~/.scrum-agent/.env is absent (first run) or --setup is passed.
    # If the user cancels the wizard, exit early — the agent can't run without
    # a configured provider (a cloud API key, or a local Ollama server).
    # Runs immediately after splash — both use fullscreen Live, so no console
    # prints happen in between (avoids visible flicker on alt-screen exit).
    if args.setup or is_first_run():
        completed = run_setup_wizard(console)
        if not completed:
            return

    # Determine early whether we'll use the old REPL or the fullscreen TUI.
    # The TUI path keeps alt-screen active from splash → select_mode seamlessly.
    # The old REPL path needs to exit alt-screen and print info to the terminal.
    use_old_repl = args.mode is not None or args.quick or args.questionnaire is not None

    team_learning_flag = args.learn or args.team_profile or args.retro is not None
    if use_old_repl or args.resume is not None or args.list_sessions or args.clear_sessions or team_learning_flag:
        # Leave alt-screen before printing to the normal terminal
        if console.is_alt_screen:
            console.set_alt_screen(False)

        # Informational prints — only shown for non-TUI paths
        scrum_md = Path(os.getcwd()) / "SCRUM.md"
        if scrum_md.is_file():
            _summarise_scrum_md(console, scrum_md)
        else:
            console.print(
                "[dim]  Tip: create a [cyan]SCRUM.md[/cyan] in this directory to add project notes, "
                "URLs, and design decisions that the agent will read automatically. "
                "See [cyan]SCRUM.md.example[/cyan] for a template.[/dim]"
            )

        if is_langsmith_enabled():
            proxy = detect_proxy()
            if proxy:
                disable_langsmith_tracing()
                console.print(f"[yellow]Warning: proxy detected ({proxy}) — LangSmith tracing auto-disabled[/yellow]")
            else:
                console.print("[dim]LangSmith tracing enabled[/dim]")

    # ── --list-sessions: print all sessions and exit ──────────────────────────
    if args.list_sessions:
        _print_sessions_table(console)
        return

    # ── --clear-sessions: interactive delete and exit ─────────────────────────
    if args.clear_sessions:
        _clear_sessions(console)
        return

    # ── --learn: analyse team history and store calibration profile ───────────
    if args.learn:
        _run_learn(console)
        return

    # ── --team-profile: display stored calibration profile ───────────────────
    if args.team_profile:
        _run_team_profile(console)
        return

    # ── --retro: compare plan to actuals ─────────────────────────────────────
    if args.retro is not None:
        _run_retro(console, args.retro)
        return

    # --export-questionnaire: write a blank template and exit (no REPL)
    if args.export_questionnaire is not None:
        from yeaboi.questionnaire_io import export_questionnaire_md

        path = export_questionnaire_md(None, Path(args.export_questionnaire))
        console.print(f"[green]Questionnaire template exported to {path}[/green]")
        return

    # ── --resume: load saved session and skip mode menu ───────────────────────
    # Phase 8B: when --resume is passed, load the saved state and go directly
    # to run_repl() with the resume_state — no mode selection needed since
    # resumed sessions are always project-planning.
    # See docs: "Memory & State" — session persistence, --resume
    if args.resume is not None:
        resume_state, resume_session_id = _resolve_resume(console, args.resume)
        if resume_state is None:
            return  # user cancelled or no sessions
        from yeaboi.repl import run_repl

        run_repl(
            console=console,
            bell=not args.no_bell,
            theme=args.theme,
            resume_state=resume_state,
            resume_session_id=resume_session_id,
        )
        return  # skip mode menu — resume goes straight to REPL

    # --questionnaire: import a filled file, build state, pass to REPL
    questionnaire = None
    if args.questionnaire is not None:
        qpath = Path(args.questionnaire)
        if not qpath.exists():
            console.print(f"[red]Error: file not found: {qpath}[/red]")
            sys.exit(1)
        try:
            from yeaboi.questionnaire_io import build_questionnaire_from_answers, parse_questionnaire_md

            parsed = parse_questionnaire_md(qpath)
            questionnaire = build_questionnaire_from_answers(parsed)
            console.print(f"[green]Loaded {len(parsed)} answers from {qpath}[/green]")
        except (ValueError, fs_policy.SandboxViolationError) as e:
            # A sandbox denial (path outside the whitelist) prints the same
            # clean message as --description @file / perf @transcript, not a traceback.
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    # Determine intake mode from flags.
    # Default is None — triggers the interactive mode selection menu in the REPL.
    # CLI flags bypass the menu for power users who know what they want.
    # See docs: "Project Intake Questionnaire" — smart intake
    if args.quick:
        intake_mode = "quick"
    else:
        intake_mode = None

    # --export-only: validate that a non-interactive intake source is provided.
    # Without --quick or --questionnaire the intake requires interactive input,
    # which defeats the purpose of --export-only.
    if args.export_only and not args.quick and questionnaire is None:
        console.print("[red]Error: --export-only requires --quick or --questionnaire to supply intake answers.[/red]")
        sys.exit(1)

    # ── Top-level mode selection ──────────────────────────────────────────────
    # Full-screen mode selector with ASCII art titles and typewriter descriptions.
    #
    # Three paths:
    #   1. --mode flag → bypass UI, use old REPL (backwards compat for scripted runs)
    #   2. --quick/--questionnaire → bypass UI, use old REPL
    #   3. Interactive → full-screen TUI mode selector, which launches the TUI
    #      session (run_session) for smart intake inside its Live context
    #
    # See docs: "Architecture" — mode selection is a CLI-layer concern.
    if use_old_repl:
        from yeaboi.repl import run_repl

        startup_mode = args.mode or "project-planning"
        if startup_mode == "project-planning":
            run_repl(
                console=console,
                questionnaire=questionnaire,
                intake_mode=intake_mode,
                export_only=args.export_only,
                bell=not args.no_bell,
                theme=args.theme,
                solo=args.solo,
            )
        else:
            console.print(f"\n[warning]Unknown mode '{startup_mode}'.[/warning]")
    else:
        # Interactive TUI flow — select_mode() launches the full session when
        # the user picks Smart or Full intake. It returns None when done.
        # Alt-screen stays active from splash through select_mode to avoid
        # flicker; clean it up when the TUI exits.
        # Mouse tracking captures scroll-wheel events so they scroll within
        # the app instead of the terminal's own scrollback buffer.
        import atexit

        from yeaboi.ui.mode_select import select_mode
        from yeaboi.ui.shared._input import (
            disable_mouse_tracking,
            enable_mouse_tracking,
            enter_raw_mode,
            exit_raw_mode,
        )

        def _terminal_cleanup() -> None:
            """Safety net — ensure terminal is restored even on unhandled crash."""
            try:
                disable_mouse_tracking()
            except Exception:
                pass
            try:
                exit_raw_mode()
            except Exception:
                pass

        atexit.register(_terminal_cleanup)
        # Hold cbreak+no-echo for the whole TUI so mouse-report bytes arriving
        # between key reads can't echo as garbage during a fast wheel scroll.
        enter_raw_mode()
        enable_mouse_tracking()
        _tui_error: Exception | None = None
        _kb_interrupt = False
        try:
            mode_result = select_mode(console, dry_run=args.dry_run)
        except KeyboardInterrupt:
            mode_result = None
            _kb_interrupt = True
        except Exception as _exc:
            logging.getLogger(__name__).exception("Unhandled exception in TUI")
            _tui_error = _exc
            mode_result = None
        finally:
            disable_mouse_tracking()
            exit_raw_mode()
            # Stop any background music daemon so it doesn't outlive the app.
            from yeaboi import music

            music.shutdown()
            if console.is_alt_screen:
                console.set_alt_screen(False)
        # Surface a friendly, one-line message (never a raw traceback) now that
        # the terminal is restored. See _classify_api_error for the mapping.
        if _tui_error is not None:
            from yeaboi.ui.session._utils import _classify_api_error

            console.print(f"[red]{_classify_api_error(_tui_error)}[/red]")
            console.print("[dim]See ~/.scrum-agent/logs/tui/yeaboi.log for details.[/dim]")
        # Ctrl-C unwinds past the menu's own quit popup, so offer the same
        # stop-Ollama courtesy here — but only now that the finally above has
        # restored the terminal (raw mode off, alt-screen closed), so a plain
        # console.input() echoes normally. The esc/q path handles its own
        # prompt in-TUI and never sets _kb_interrupt, so this can't double-fire.
        if _kb_interrupt:
            try:
                from yeaboi.ollama_control import should_offer_ollama_stop, stop_ollama_server

                if sys.stdin.isatty() and should_offer_ollama_stop():
                    if console.input("Stop the local Ollama server? [y/N] ").strip().lower() == "y":
                        _stopped, _msg = stop_ollama_server()
                        console.print(f"[dim]{_msg}[/dim]")
            except Exception:
                pass
        # The ctrl+U update flow asks for a relaunch onto the version it just
        # installed. It has to happen HERE: os.execv replaces the process image
        # without running atexit handlers, so it's only safe now that the finally
        # above has taken the terminal out of raw mode, stopped mouse tracking and
        # left the alternate screen. restart_in_place only returns on failure.
        if update_check.restart_requested():
            if not update_check.restart_in_place():
                console.print("[dim]Update installed — restart yeaboi to use the new version.[/dim]")
            return
        if mode_result is None:
            return
        # mode_result is non-None only for offline import (questionnaire path)
        startup_mode, ui_intake, questionnaire_path = mode_result
        if ui_intake is not None and intake_mode is None:
            intake_mode = ui_intake
        if questionnaire_path and questionnaire is None:
            qpath = Path(questionnaire_path)
            try:
                from yeaboi.questionnaire_io import build_questionnaire_from_answers, parse_questionnaire_md

                parsed = parse_questionnaire_md(qpath)
                questionnaire = build_questionnaire_from_answers(parsed)
                console.print(f"[green]Loaded {len(parsed)} answers from {qpath}[/green]")
            except ValueError as e:
                console.print(f"[red]Error: {e}[/red]")
                sys.exit(1)
        # Import flow falls through to REPL for review
        if startup_mode == "project-planning":
            from yeaboi.repl import run_repl

            run_repl(
                console=console,
                questionnaire=questionnaire,
                intake_mode=intake_mode,
                export_only=args.export_only,
                bell=not args.no_bell,
                theme=args.theme,
                solo=args.solo,
            )


if __name__ == "__main__":
    main()
