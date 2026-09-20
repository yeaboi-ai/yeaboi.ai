"""Configuration and environment variable handling."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


# ---------------------------------------------------------------------------
# User config directory (~/.yeaboi/)
# ---------------------------------------------------------------------------


def restrict_permissions(path: Path, *, mode: int) -> None:
    """Best-effort ``chmod`` so on-disk secrets aren't group/other readable.

    The config dir (``0o700``) and ``.env`` (``0o600``) hold API keys and tokens
    in plaintext, so they must not be readable by other local accounts. This is a
    POSIX concept: on Windows ``chmod`` can't express these bits and the call is a
    harmless no-op. Never raises — a filesystem that rejects ``chmod`` must not
    break a config write.
    """
    try:
        path.chmod(mode)
    except OSError as e:  # pragma: no cover - platform/filesystem dependent
        logger.debug("could not chmod %s to %o: %s", path, mode, e)


def get_config_dir() -> Path:
    """Return ~/.yeaboi/, creating it if necessary.

    Kept as a live ``Path.home()`` computation (rather than importing
    ``paths.ROOT_DIR``) so it stays test-monkeypatchable and matches
    ``paths.ROOT_DIR``. Callers must ensure ``paths.migrate_root_dir()`` has run
    first (done at the top of ``cli.main``) so a pre-rebrand ~/.scrum-agent tree
    is moved over before this mkdir creates an empty ~/.yeaboi.
    """
    d = Path.home() / ".yeaboi"  # lens-exempt: paths-through-paths-py — see the docstring above
    d.mkdir(exist_ok=True)
    # Repair perms on every call (cheap): the dir holds .env + sessions.db, both secret-bearing.
    restrict_permissions(d, mode=0o700)
    return d


def set_config_value(key: str, value: str) -> Path:
    """Persist a single ``key=value`` to ~/.yeaboi/.env and lock the file to 0o600.

    Central choke point for every ``set_key``-based setter so the credential file
    is re-hardened after each write (``dotenv.set_key`` may recreate it at
    umask-default perms). Does not touch ``os.environ`` — callers still do that.
    """
    from dotenv import set_key

    config_file = get_config_file()
    set_key(str(config_file), key, value)
    restrict_permissions(config_file, mode=0o600)
    return config_file


def apply_config_value(key: str, value: str) -> Path:
    """Persist ``key=value`` AND mirror it into ``os.environ`` for this process.

    :func:`set_config_value` only writes the .env file, so a running session keeps
    its old ``os.environ`` — anything reading config (including the Settings page,
    which re-reads the environment) would show the previous value until a restart.
    Every other setter in this module pairs the write with an ``os.environ`` update;
    this is that pair as one reusable call. An empty ``value`` clears the variable.
    """
    config_file = set_config_value(key, value)
    if value:
        os.environ[key] = value
    else:
        os.environ.pop(key, None)
    # A credential that changed invalidates whatever the last probe of it said.
    # Here rather than in settings.engine because the TUI catalog, `yeaboi
    # connect`, Access setup and OAuth rotation all write through this call and
    # none of them go near that one.
    from yeaboi.connectors import verify_status

    verify_status.forget_for_env(key)
    return config_file


def get_config_file() -> Path:
    """Return path to ~/.yeaboi/.env."""
    return get_config_dir() / ".env"


def get_sessions_db() -> Path:
    """Return path to ~/.yeaboi/sessions.db (SQLite session store).

    Legacy location — most stores use ``paths.get_db_path()`` (data/sessions.db)
    instead; this getter is still used by ceremony history and performance
    context. Both apply the same 0o600 hardening as the .env file.
    """
    db = get_config_dir() / "sessions.db"
    if db.exists():
        restrict_permissions(db, mode=0o600)
    return db


def load_user_config() -> None:
    """Load ~/.yeaboi/.env without overriding existing env vars.

    Called once at CLI startup before any credential reads.
    dotenv's override=False means shell env vars and project .env always win
    — safe for CI/CD and developer overrides.
    """
    config_path = get_config_file()
    logger.info("Loading user config from %s", config_path)
    load_dotenv(config_path, override=False)
    logger.debug(
        "API keys — ANTHROPIC_API_KEY: %s, OPENAI_API_KEY: %s, GOOGLE_API_KEY: %s",
        "set" if os.getenv("ANTHROPIC_API_KEY") else "missing",
        "set" if os.getenv("OPENAI_API_KEY") else "missing",
        "set" if os.getenv("GOOGLE_API_KEY") else "missing",
    )
    logger.debug(
        "Integrations — GITHUB_TOKEN: %s, JIRA_API_TOKEN: %s, AZURE_DEVOPS_TOKEN: %s",
        "set" if os.getenv("GITHUB_TOKEN") else "missing",
        "set" if os.getenv("JIRA_API_TOKEN") else "missing",
        "set" if os.getenv("AZURE_DEVOPS_TOKEN") else "missing",
    )


def get_anthropic_api_key() -> str:
    """Return the Anthropic API key or raise if not set."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise OSError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
    return key


def get_anthropic_subscription_token() -> str:
    """The Claude subscription token to authenticate with, or ``""`` for key auth.

    Both halves must agree before this returns anything: the user has to have
    picked subscription auth in Settings *and* have a token stored. Returning the
    token on its own presence would silently hijack a working API key for anyone
    who happens to have the Claude Code CLI logged in.
    """
    if os.getenv("ANTHROPIC_AUTH_MODE", "").strip().lower() != "subscription":
        return ""
    return os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "").strip()


def is_langsmith_enabled() -> bool:
    """Check whether LangSmith tracing is enabled."""
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


def is_tips_enabled() -> bool:
    """Return True if on-screen discoverability tips should be shown (default on).

    Controls the rotating welcome-screen tip banner and the inline voice hints on
    text-entry screens. Any value other than "false" (case-insensitive) keeps tips
    on, so an unset var means enabled — the feature should be visible by default.
    """
    return os.getenv("TIPS_ENABLED", "true").strip().lower() != "false"


def is_beta_notice_enabled() -> bool:
    """Return True if CLI runs should print the beta-maturity caveat (default on).

    Mirrors :func:`is_tips_enabled`: any value other than "false" keeps it on, so
    an unset var means enabled — a maturity caveat has to be opt-out, never
    opt-in. Scripted and cron callers that capture stderr can turn it off.
    """
    return os.getenv("BETA_NOTICES_ENABLED", "true").strip().lower() != "false"


def set_tips_enabled(enabled: bool) -> None:
    """Persist the tips on/off preference to ~/.scrum-agent/.env and apply it now.

    Uses dotenv's set_key so only this one key is updated — save_config() rewrites
    the whole file and would drop any keys not passed to it. os.environ is updated
    too so the running session reflects the change immediately (no reload needed).
    """
    value = "true" if enabled else "false"
    # set_config_value creates the file if missing, preserves existing keys, and re-locks it to 0o600.
    config_file = set_config_value("TIPS_ENABLED", value)
    os.environ["TIPS_ENABLED"] = value
    logger.info("Tips %s (persisted to %s)", "enabled" if enabled else "disabled", config_file)


# The Solo world's off switch. Read here and nowhere else: the CLI, the MCP
# tools, the engines and the app's HTTP routes ignore it — only what a user can
# see and click is gated.
SOLO_WORLD_ENV = "YEABOI_SOLO"


def solo_world_enabled() -> bool:
    """Whether the UI offers the Solo world, and with it the landing split.

    Opt-in: an unset or unrecognised value hides the world.
    """
    return _env_truthy(SOLO_WORLD_ENV)


# The landing split's categories. Persisted so the next launch preselects
# (never auto-skips) the category the user worked in last.
LAST_CATEGORY_KEY = "YEABOI_LAST_CATEGORY"
_VALID_CATEGORIES = ("solo", "team")
# Values written by older releases, mapped on read; rewritten on the next set.
# "agents" was its own world before the Agents family moved into Solo.
_LEGACY_CATEGORIES = {"humans": "team", "agents": "solo"}


def get_last_category() -> str:
    """Return the last-chosen landing category ("solo"/"team", default team).

    Preselection only — the category screen always shows. Unknown values fall
    back to "team" so a hand-edited .env can't wedge the landing screen, and so
    does "solo" while the Solo world is off: there is one world, and it is Team.
    """
    value = os.getenv(LAST_CATEGORY_KEY, "team").strip().lower()
    value = _LEGACY_CATEGORIES.get(value, value)
    if value not in _VALID_CATEGORIES:
        return "team"
    return value if value == "team" or solo_world_enabled() else "team"


def set_last_category(category: str) -> None:
    """Persist the landing-category choice (mirrors :func:`set_tips_enabled`)."""
    if category not in _VALID_CATEGORIES:
        return
    config_file = set_config_value(LAST_CATEGORY_KEY, category)
    os.environ[LAST_CATEGORY_KEY] = category
    logger.info("Landing category set to %s (persisted to %s)", category, config_file)


# Which world the running session is in. Process-local and unpersisted, unlike
# the preselection above: the world is chosen on the landing split every launch,
# and a stale one silently reshaping next week's runs is worse than re-picking.
_solo_mode: bool = False


def is_solo_mode() -> bool:
    """True while this session is in the Solo world (set by the landing split)."""
    return _solo_mode


def set_solo_mode(solo: bool) -> None:
    """Record which world this session is in."""
    global _solo_mode
    if solo != _solo_mode:
        logger.info("solo mode %s", "on" if solo else "off")
    _solo_mode = solo


LAST_DOOR_KEY = "YEABOI_LAST_DOOR"
_VALID_DOORS = ("projects", "sessions")


def get_last_door() -> str:
    """Return the last-chosen door ("projects"/"sessions", default sessions).

    Preselection only — the door screen always shows after the landing split.
    Unknown values fall back to "sessions" so a hand-edited .env can't wedge it.
    """
    value = os.getenv(LAST_DOOR_KEY, "sessions").strip().lower()
    return value if value in _VALID_DOORS else "sessions"


def set_last_door(door: str) -> None:
    """Persist the door choice (mirrors :func:`set_last_category`)."""
    if door not in _VALID_DOORS:
        return
    config_file = set_config_value(LAST_DOOR_KEY, door)
    os.environ[LAST_DOOR_KEY] = door
    logger.info("Door set to %s (persisted to %s)", door, config_file)


# The scope a mode's runs last read under, one .env key per mode
# (YEABOI_CONTEXT_STANDUP, …) holding the context/scope.py JSON twin. The
# context page opens on it; a run started without an explicit scope inherits it.
CONTEXT_SCOPE_KEY_PREFIX = "YEABOI_CONTEXT_"


def _context_scope_key(mode: str) -> str:
    return CONTEXT_SCOPE_KEY_PREFIX + "".join(ch if ch.isalnum() else "_" for ch in mode.strip().upper())


def get_last_context_scope(mode: str) -> dict | None:
    """The persisted scope dict for ``mode``, or ``None`` when unset or unreadable."""
    import json

    raw = os.getenv(_context_scope_key(mode), "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        logger.warning("Ignoring unreadable %s", _context_scope_key(mode))
        return None
    return parsed if isinstance(parsed, dict) else None


def set_last_context_scope(mode: str, scope: dict | None) -> None:
    """Persist ``mode``'s scope dict (``None`` clears it), mirroring :func:`set_last_category`."""
    import json

    key = _context_scope_key(mode)
    # Pinned sessions belong to one run, never to the next one's default.
    remembered = ({**scope, "sessions": []} if "sessions" in scope else dict(scope)) if scope else None
    value = json.dumps(remembered, sort_keys=True) if remembered else ""
    config_file = set_config_value(key, value)
    os.environ[key] = value
    logger.info("Context scope for %s %s (persisted to %s)", mode, "set" if value else "cleared", config_file)


# The sprint grid a "last N sprints" window falls back to when neither a
# tracker nor a plan defines it. Both are Settings › Advanced fields.
SPRINT_LENGTH_KEY = "YEABOI_SPRINT_LENGTH_WEEKS"
SPRINT_ANCHOR_KEY = "YEABOI_SPRINT_ANCHOR_DATE"
DEFAULT_SPRINT_LENGTH_WEEKS = 2


def get_sprint_length_weeks() -> int:
    """Sprint length in weeks (default 2; anything unreadable or below 1 reads as the default)."""
    raw = os.getenv(SPRINT_LENGTH_KEY, "").strip()
    try:
        value = int(raw) if raw else DEFAULT_SPRINT_LENGTH_WEEKS
    except ValueError:
        return DEFAULT_SPRINT_LENGTH_WEEKS
    return value if value >= 1 else DEFAULT_SPRINT_LENGTH_WEEKS


def get_sprint_anchor_date() -> str:
    """A known sprint start as an ISO date, or ``""`` when unset or malformed."""
    from yeaboi.timeparse import parse_date

    raw = os.getenv(SPRINT_ANCHOR_KEY, "").strip()
    if not raw:
        return ""
    try:
        return parse_date(raw).isoformat()
    except (TypeError, ValueError):
        return ""


def is_duck_enabled() -> bool:
    """Return True if the corner duck's speech bubble may show lines (default on).

    Mirrors :func:`is_tips_enabled`: any value other than "false" keeps the
    bubble on, so an unset var means enabled. Only the bubble is gated — the
    duck himself (bob, quack, shades) always stays.
    """
    return os.getenv("DUCK_ENABLED", "true").strip().lower() != "false"


def set_duck_enabled(enabled: bool) -> None:
    """Persist the duck-bubble on/off preference to ~/.scrum-agent/.env and apply it now."""
    value = "true" if enabled else "false"
    config_file = set_config_value("DUCK_ENABLED", value)
    os.environ["DUCK_ENABLED"] = value
    logger.info("Duck bubble %s (persisted to %s)", "enabled" if enabled else "disabled", config_file)


def is_pet_enabled() -> bool:
    """Return True if the desktop duck pet should be on screen (default off).

    Deliberately not a Settings-page field like :func:`is_duck_enabled`: it
    configures a window the terminal cannot draw, so it is offered where it can
    be seen — the desktop tray and the desktop's own ambience controls. Off by
    default; an always-on-top duck is opt-in.
    """
    return os.getenv("PET_ENABLED", "false").strip().lower() == "true"


def set_pet_enabled(enabled: bool) -> None:
    """Persist the desktop-pet on/off preference to ~/.scrum-agent/.env."""
    value = "true" if enabled else "false"
    config_file = set_config_value("PET_ENABLED", value)
    os.environ["PET_ENABLED"] = value
    logger.info("Desktop pet %s (persisted to %s)", "enabled" if enabled else "disabled", config_file)


def get_saver_style() -> str:
    """Return the persisted idle-screensaver style key (defaults to "duck-yard").

    Unvalidated on the way out, like :func:`get_music_channel`: the catalogue
    lives in :mod:`yeaboi.ambience`, which clamps an unrecognised key to the
    default. The one value every surface understands without the catalogue is
    "off".
    """
    return os.getenv("SAVER_STYLE", "duck-yard").strip().lower() or "duck-yard"


def set_saver_style(style: str) -> None:
    """Persist the idle-screensaver style to ~/.scrum-agent/.env and apply it now."""
    value = style.strip().lower()
    config_file = set_config_value("SAVER_STYLE", value)
    os.environ["SAVER_STYLE"] = value
    logger.info("Screensaver style set to %s (persisted to %s)", value, config_file)


def is_music_enabled() -> bool:
    """Return True if background music was left enabled (default off).

    Only records the on/off preference so the status bar can reflect it; playback
    itself is never auto-started (that would be surprise noise). Mirrors
    :func:`is_tips_enabled`.
    """
    return os.getenv("MUSIC_ENABLED", "false").strip().lower() == "true"


def set_music_enabled(enabled: bool) -> None:
    """Persist the music on/off preference to ~/.scrum-agent/.env and apply it now."""
    value = "true" if enabled else "false"
    config_file = set_config_value("MUSIC_ENABLED", value)
    os.environ["MUSIC_ENABLED"] = value
    logger.info("Music %s (persisted to %s)", "enabled" if enabled else "disabled", config_file)


def get_music_channel() -> int:
    """Return the persisted music channel index (defaults to 0)."""
    try:
        return int(os.getenv("MUSIC_CHANNEL", "0").strip())
    except ValueError:
        return 0


def set_music_channel(idx: int) -> None:
    """Persist the selected music channel index to ~/.scrum-agent/.env."""
    value = str(int(idx))
    config_file = set_config_value("MUSIC_CHANNEL", value)
    os.environ["MUSIC_CHANNEL"] = value
    logger.info("Music channel set to %s (persisted to %s)", value, config_file)


# Modes that ship in beta show a one-time notice the first time they're opened.
# The acknowledgement is a comma-separated list of _MODE_CARDS keys rather than a
# boolean per mode, so the next beta mode costs one card key instead of a new env
# var and a new pair of functions.
BETA_ACK_KEY = "BETA_NOTICES_ACK"
FORCE_BETA_NOTICE_ENV = "YEABOI_FORCE_BETA_NOTICE"


def beta_notices_acked() -> set[str]:
    """Return the set of mode keys whose beta notice has been acknowledged."""
    raw = os.getenv(BETA_ACK_KEY, "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def is_beta_notice_seen(mode_key: str) -> bool:
    """Return True if ``mode_key``'s one-time beta notice has already been shown.

    ``YEABOI_FORCE_BETA_NOTICE`` overrides the acknowledgement — either truthy
    (all modes) or a comma-separated list of specific keys. A once-ever gate is
    otherwise impossible to re-check by hand after the first click, which makes
    it impossible to demo, screenshot, or eyeball during review.
    """
    forced = os.getenv(FORCE_BETA_NOTICE_ENV, "").strip().lower()
    if forced:
        if forced in {"1", "true", "yes", "on"}:
            return False
        if mode_key in {part.strip() for part in forced.split(",") if part.strip()}:
            return False
    return mode_key in beta_notices_acked()


def mark_beta_notice_seen(mode_key: str) -> None:
    """Record that ``mode_key``'s beta notice has been acknowledged.

    Deliberately not :func:`apply_config_value` (which does the same pair): that
    helper writes to disk *first*, so a failed write would skip the ``os.environ``
    update too. Here the order is inverted — ``os.environ`` is set even when the
    disk write fails (read-only home, odd permissions), because a user who just
    dismissed the notice must not see it again in the same session. Re-showing
    it on the next restart is the lesser failure.
    """
    acked = beta_notices_acked()
    if mode_key in acked:
        return
    value = ",".join(sorted(acked | {mode_key}))
    os.environ[BETA_ACK_KEY] = value
    try:
        config_file = set_config_value(BETA_ACK_KEY, value)
    except OSError as exc:
        logger.warning("Could not persist beta notice acknowledgement for %s: %s", mode_key, exc)
        return
    logger.info("Beta notice acknowledged for %s (persisted to %s)", mode_key, config_file)


# The in-app dictation install offer. Two tiers of "no": Esc declines for the
# session (in-memory, in the UI module), `n` declines for good — which is this.
VOICE_OFFER_KEY = "VOICE_INSTALL_OFFER"
FORCE_VOICE_OFFER_ENV = "YEABOI_FORCE_VOICE_OFFER"
VOICE_INSTALLED_KEY = "VOICE_EXTRA_INSTALLED"


def is_voice_install_offer_enabled() -> bool:
    """True when double-tapping Space may offer to install dictation.

    Defaults on. ``YEABOI_FORCE_VOICE_OFFER`` overrides a permanent decline, for
    the same reason :func:`is_beta_notice_seen` has its override: a once-ever
    gate is otherwise impossible to demo, screenshot or review after the first
    dismissal.
    """
    if os.getenv(FORCE_VOICE_OFFER_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return os.getenv(VOICE_OFFER_KEY, "").strip().lower() not in {"0", "off", "false", "no"}


def set_voice_install_offer(enabled: bool) -> None:
    """Persist whether the dictation install offer may appear.

    Same inverted ordering as :func:`mark_beta_notice_seen`: ``os.environ`` first,
    disk second, because a user who just said "never" must not be asked again in
    this session even if the config file cannot be written.
    """
    value = "on" if enabled else "off"
    os.environ[VOICE_OFFER_KEY] = value
    try:
        config_file = set_config_value(VOICE_OFFER_KEY, value)
    except OSError as exc:
        logger.warning("Could not persist the voice install offer setting: %s", exc)
        return
    logger.info("Voice install offer set to %s (persisted to %s)", value, config_file)


def voice_extra_was_installed() -> bool:
    """True if yeaboi has installed the dictation packages here before.

    An in-place ``uv pip install`` is not recorded in uv's tool receipt, so a
    later ``uv tool upgrade`` rebuilds the venv and silently drops dictation.
    This flag is what lets the offer say *"an upgrade removed dictation"* rather
    than starting the conversation over as if nothing had happened.
    """
    return os.getenv(VOICE_INSTALLED_KEY, "").strip().lower() in {"1", "true", "yes", "on"}


def mark_voice_extra_installed() -> None:
    """Record that the dictation packages were installed from inside the app."""
    os.environ[VOICE_INSTALLED_KEY] = "1"
    try:
        set_config_value(VOICE_INSTALLED_KEY, "1")
    except OSError as exc:
        logger.warning("Could not persist the voice-installed marker: %s", exc)


# Proxy environment variables to check (both uppercase and lowercase conventions).
_PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def detect_proxy() -> str | None:
    """Return the first proxy URL found in environment variables, or None."""
    for var in _PROXY_ENV_VARS:
        value = os.getenv(var)
        if value:
            logger.debug("Proxy detected via %s", var)
            return value
    logger.debug("No proxy detected")
    return None


def get_github_token() -> str | None:
    """Return the GitHub PAT, or None if not set (tools work for public repos without a token)."""
    return os.getenv("GITHUB_TOKEN") or None


def get_azure_devops_token() -> str | None:
    """Return the Azure DevOps PAT, or None if not set."""
    return os.getenv("AZURE_DEVOPS_TOKEN") or None


def get_azure_devops_org_url() -> str | None:
    """Return the Azure DevOps organization URL (e.g. https://dev.azure.com/myorg), or None if not set.

    Normalised on read: whitespace/trailing slashes stripped and a missing scheme
    defaulted to https://. A bare "dev.azure.com/org" value otherwise reaches the
    SDK's URL joining and surfaces as MissingSchema errors with a doubled host
    ("dev.azure.com/org/dev.azure.com/org/_apis") on every AzDO surface.
    """
    raw = (os.getenv("AZURE_DEVOPS_ORG_URL") or "").strip().rstrip("/")
    if not raw:
        return None
    if "://" not in raw:
        raw = f"https://{raw}"
    return raw


def get_azure_devops_project() -> str | None:
    """Return the Azure DevOps project name, or None if not set."""
    return os.getenv("AZURE_DEVOPS_PROJECT") or None


def get_azure_devops_team() -> str | None:
    """Return the Azure DevOps team name, or None if not set.

    Defaults to "{project} Team" when not explicitly set (AzDO's default team naming).
    """
    team = os.getenv("AZURE_DEVOPS_TEAM")
    if team:
        return team
    project = get_azure_devops_project()
    if project:
        return f"{project} Team"
    return None


def get_jira_base_url() -> str | None:
    """Return the Jira Cloud base URL (e.g. https://org.atlassian.net), or None if not set."""
    return os.getenv("JIRA_BASE_URL") or None


def get_jira_email() -> str | None:
    """Return the Atlassian account email used for Jira basic auth, or None if not set."""
    return os.getenv("JIRA_EMAIL") or None


def get_jira_token() -> str | None:
    """Return the Jira API token, or None if not set."""
    return os.getenv("JIRA_API_TOKEN") or None


def get_jira_project_key() -> str | None:
    """Return the default Jira project key (e.g. 'MYPROJ'), or None if not set."""
    return os.getenv("JIRA_PROJECT_KEY") or None


# YEABOI_AC_FORMAT alias table — normalized to the canonical style names in
# agent/state.py (AC_STYLES). Unknown values normalize to "" (no override);
# a config getter never raises.
_AC_FORMAT_ALIASES: dict[str, str] = {
    "gwt": "gwt",
    "given-when-then": "gwt",
    "given/when/then": "gwt",
    "gherkin": "gwt",
    "bullets": "bullets",
    "bullet": "bullets",
    "freeform": "bullets",
    "free-form": "bullets",
    "checklist": "bullets",
}


def get_ac_format() -> str:
    """Return the acceptance-criteria style override from YEABOI_AC_FORMAT.

    "" when unset or unrecognized — the planner then follows the learned team
    profile (see resolve_ac_style in agent/state.py).
    """
    raw = (os.getenv("YEABOI_AC_FORMAT") or "").strip().lower()
    return _AC_FORMAT_ALIASES.get(raw, "")


def get_confluence_base_url() -> str | None:
    """Return the Confluence Cloud base URL, or None if not set.

    Confluence shares Atlassian identity with Jira, so it historically reused the
    Jira creds. To let Confluence be configured standalone (without Jira issue
    tracking), a dedicated CONFLUENCE_BASE_URL may be set; it falls back to
    JIRA_BASE_URL so existing Jira+Confluence setups keep working unchanged.
    """
    return os.getenv("CONFLUENCE_BASE_URL") or get_jira_base_url()


def get_confluence_email() -> str | None:
    """Return the Atlassian email for Confluence basic auth, or None if not set.

    Falls back to JIRA_EMAIL (same Atlassian account) — see get_confluence_base_url.
    """
    return os.getenv("CONFLUENCE_EMAIL") or get_jira_email()


def get_confluence_token() -> str | None:
    """Return the Atlassian API token for Confluence, or None if not set.

    Falls back to JIRA_API_TOKEN (same Atlassian account) — see get_confluence_base_url.
    """
    return os.getenv("CONFLUENCE_API_TOKEN") or get_jira_token()


def get_confluence_space_key() -> str | None:
    """Return the default Confluence space key (e.g. 'MYSPACE'), or None if not set."""
    return os.getenv("CONFLUENCE_SPACE_KEY") or None


def get_linear_api_key() -> str | None:
    """Return the Linear personal API key, or None if not set."""
    return os.getenv("LINEAR_API_KEY") or None


def get_linear_team_key() -> str | None:
    """Return the Linear team key to work in (e.g. 'ENG'), or None if not set.

    Optional: a workspace with one team needs no key at all.
    """
    return os.getenv("LINEAR_TEAM_KEY") or None


def get_trello_api_key() -> str | None:
    """Return the Trello API key (names the app; the token grants access), or None."""
    return os.getenv("TRELLO_API_KEY") or None


def get_trello_token() -> str | None:
    """Return the Trello API token, or None if not set."""
    return os.getenv("TRELLO_TOKEN") or None


def get_trello_board_id() -> str | None:
    """Return the Trello board to plan against, or None if not set.

    Optional: an account with one open board needs no id at all.
    """
    return os.getenv("TRELLO_BOARD_ID") or None


def get_anonymize_mask_terms() -> tuple[str, ...]:
    """Return extra company-specific terms to always mask in Anonymize mode.

    Read from ANONYMIZE_MASK_TERMS as a comma-separated list (e.g. "YouLend,YL,Acme").
    These seed the deterministic pre-mask pass in anonymize/engine.py so the obvious
    company identifiers are redacted even when the LLM is unavailable. Blank/whitespace
    entries are dropped; returns () when unset.
    """
    raw = os.getenv("ANONYMIZE_MASK_TERMS", "")
    return tuple(term.strip() for term in raw.split(",") if term.strip())


def get_notion_token() -> str | None:
    """Return the Notion integration token, or None if not set.

    Unlike Confluence (which reuses Jira's Atlassian auth), Notion has its own
    OAuth2 integration token — the only credential its SDK needs.
    """
    return os.getenv("NOTION_TOKEN") or None


def get_notion_root_page_id() -> str | None:
    """Return the optional Notion root page/database ID, or None if not set.

    Notion has no "space key" — search spans whatever pages the integration is
    granted. This optional ID scopes the default parent for page creation and the
    standup recent-pages feed (analogous to CONFLUENCE_SPACE_KEY).
    """
    return os.getenv("NOTION_ROOT_PAGE_ID") or None


# ---------------------------------------------------------------------------
# Storage & export destinations
# ---------------------------------------------------------------------------
# One YEABOI_HOME override relocates the whole data tree (exports, logs, DB…);
# Notion/Confluence exports publish to the destinations collected in provider
# setup, falling back to the tool's natural default (Notion root page /
# Confluence space root) when no dedicated exports page is set.


def _set_env_value(key: str, value: str) -> None:
    """Persist an env var to ~/.yeaboi/.env and apply it to the live process."""
    from dotenv import set_key

    config_file = get_config_file()
    set_key(str(config_file), key, value)
    os.environ[key] = value
    logger.info("%s persisted to %s", key, config_file)


def get_data_dir() -> str:
    """Return the configured data home (YEABOI_HOME), or '' when using ~/.yeaboi.

    The actual resolution lives in paths.py (import time); this getter exists
    for the Settings page display and the Data Dir edit flow.
    """
    return os.getenv("YEABOI_HOME", "") or ""


def set_data_dir(value: str) -> None:
    """Persist the data home override ('' clears back to ~/.yeaboi).

    Written to the pinned bootstrap ~/.yeaboi/.env (see paths.ENV_FILE) —
    module-level path constants are baked at import, so a restart fully applies.
    """
    _set_env_value("YEABOI_HOME", value.strip())


def get_allowed_paths() -> tuple[str, ...]:
    """Return the user's filesystem whitelist (YEABOI_ALLOWED_PATHS).

    Comma-separated directory/file paths the sandbox (fs_policy.py) allows
    beyond the data home. Shares _csv_config's limitation: paths containing a
    comma can't be expressed. Applies to reads and writes alike.
    """
    return _csv_config("YEABOI_ALLOWED_PATHS")


def set_allowed_paths(values: list[str] | tuple[str, ...]) -> None:
    """Persist the whitelist (deduplicated, order-preserving) and apply live."""
    deduped: list[str] = []
    for raw in values:
        value = raw.strip()
        if value and value not in deduped:
            deduped.append(value)
    _set_env_value("YEABOI_ALLOWED_PATHS", ",".join(deduped))


def add_allowed_path(value: str) -> None:
    """Append one path to the whitelist ('Always allow' in the consent popup)."""
    set_allowed_paths([*get_allowed_paths(), value])


def get_notion_export_parent_page_id() -> str | None:
    """Return the Notion page ID exports publish under, or None when unavailable.

    The dedicated exports page (NOTION_EXPORT_PARENT_PAGE_ID, optional in the
    Notion setup step) wins; otherwise falls back to NOTION_ROOT_PAGE_ID —
    where the publisher groups docs under an auto-created "yeaboi" container
    page (🤙 icon). The Notion API can't create top-level pages, so None
    (neither set) blocks Notion export with a warning pointing at Setup.
    """
    return os.getenv("NOTION_EXPORT_PARENT_PAGE_ID") or get_notion_root_page_id()


def get_confluence_export_parent_page_id() -> str | None:
    """Return the optional Confluence parent page ID for exports.

    Blank means exports group under an auto-created "🤙 yeaboi" container page
    at the root of the space (see export_targets._ensure_confluence_brand_parent).
    """
    return os.getenv("CONFLUENCE_EXPORT_PARENT_PAGE_ID") or None


# ---------------------------------------------------------------------------
# Daily Standup configuration
# ---------------------------------------------------------------------------
# Non-secret standup settings (schedule time, channels) live in the SQLite
# standup_config table, keyed by session. Secrets and single-value integration
# creds live here in .env, same as the other integrations. get_standup_* getters
# read env; the two secret-bearing setters use dotenv.set_key like set_tips_enabled.


def get_standup_github_repo() -> str:
    """Return the GitHub repo (owner/repo or URL) to scan for standup code activity."""
    return os.getenv("STANDUP_GITHUB_REPO", "") or ""


def _csv_config(name: str) -> tuple[str, ...]:
    """Return a stable, de-duplicated comma-separated configuration value."""
    values: list[str] = []
    for raw in os.getenv(name, "").split(","):
        value = raw.strip()
        if value and value not in values:
            values.append(value)
    return tuple(values)


def get_team_analysis_github_owners() -> tuple[str, ...]:
    """GitHub owners/orgs whose repositories form the analysis estate.

    Falls back to the owner of ``STANDUP_GITHUB_REPO`` for compatibility.
    """
    configured = _csv_config("TEAM_ANALYSIS_GITHUB_OWNERS")
    if configured:
        return configured
    repo = get_standup_github_repo()
    return (repo.split("/", 1)[0],) if "/" in repo else ()


def get_team_analysis_azdo_projects() -> tuple[str, ...]:
    configured = _csv_config("TEAM_ANALYSIS_AZDO_PROJECTS")
    if configured:
        return configured
    project = get_azure_devops_project() or ""
    return (project,) if project else ()


def get_team_analysis_confluence_spaces() -> tuple[str, ...]:
    configured = _csv_config("TEAM_ANALYSIS_CONFLUENCE_SPACES")
    if configured:
        return configured
    space = get_confluence_space_key() or ""
    return (space,) if space else ()


def get_team_analysis_notion_roots() -> tuple[str, ...]:
    configured = _csv_config("TEAM_ANALYSIS_NOTION_ROOTS")
    if configured:
        return configured
    root = get_notion_root_page_id() or ""
    return (root,) if root else ()


def get_team_analysis_enrichment_timeout_seconds() -> int:
    """Maximum seconds for one Analysis-mode AI enrichment request."""
    raw = os.getenv("TEAM_ANALYSIS_ENRICHMENT_TIMEOUT_SECONDS", "120")
    try:
        return max(10, min(int(raw), 600))
    except ValueError:
        return 120


def get_team_analysis_fast_model() -> str | None:
    """Optional model override for lightweight Analysis-mode enrichment calls.

    Unset selects the provider-specific fast default. ``default``/``off`` keeps
    the user's primary model instead.
    """
    raw = os.getenv("TEAM_ANALYSIS_FAST_MODEL", "").strip()
    return raw or None


def get_team_analysis_llm_target_seconds() -> int:
    """Target wall-clock time for the LLM portion of a Deep analysis run."""
    raw = os.getenv("TEAM_ANALYSIS_LLM_TARGET_SECONDS", "600")
    try:
        return max(60, min(int(raw), 7200))
    except ValueError:
        return 600


def get_team_analysis_llm_max_concurrency() -> int:
    """Maximum concurrent cloud LLM requests used by Analysis mode."""
    raw = os.getenv("TEAM_ANALYSIS_LLM_MAX_CONCURRENCY", "6")
    try:
        return max(1, min(int(raw), 12))
    except ValueError:
        return 6


def get_team_analysis_doc_request_timeout_seconds() -> int:
    """Maximum seconds for one documentation-provider request in Analysis."""
    raw = os.getenv("TEAM_ANALYSIS_DOC_REQUEST_TIMEOUT_SECONDS", "30")
    try:
        return max(5, min(int(raw), 120))
    except ValueError:
        return 30


def get_team_analysis_doc_max_concurrency() -> int:
    """Maximum concurrent documentation body reads per provider."""
    raw = os.getenv("TEAM_ANALYSIS_DOC_MAX_CONCURRENCY", "8")
    try:
        return max(1, min(int(raw), 16))
    except ValueError:
        return 8


def get_team_analysis_code_max_concurrency() -> int:
    """Maximum concurrent code-provider reads used by Analysis mode."""
    raw = os.getenv("TEAM_ANALYSIS_CODE_MAX_CONCURRENCY", "6")
    try:
        return max(1, min(int(raw), 16))
    except ValueError:
        return 6


def get_team_analysis_tracker_max_concurrency() -> int:
    """Maximum concurrent Jira/Azure-DevOps per-story reads during sprint-history fetch.

    Deliberately lower than the code knob's default: Jira Cloud's cost-based rate
    limiting throttles bursts sooner than GitHub does, and a throttled per-story
    call degrades that story's data silently (empty comments / missing changelog).
    """
    raw = os.getenv("TEAM_ANALYSIS_TRACKER_MAX_CONCURRENCY", "4")
    try:
        return max(1, min(int(raw), 12))
    except ValueError:
        return 4


def get_team_analysis_max_change_lookups() -> int:
    """Maximum per-run code-change metadata lookups (each cache miss costs one API call).

    Applied to cache misses only — warm re-runs resolve from the SQLite sha cache
    and keep full coverage; a capped cold run discloses the truncation in the
    coverage notes and prefers the newest changes.
    """
    raw = os.getenv("TEAM_ANALYSIS_MAX_CHANGE_LOOKUPS", "500")
    try:
        return max(50, min(int(raw), 5000))
    except ValueError:
        return 500


def get_retro_server_port() -> int:
    """Return the base port for the Retro collaboration server (default 5173).

    The server walks upward from this port if it is busy (see retro/server.py).
    """
    try:
        return int(os.getenv("RETRO_PORT", "5173"))
    except ValueError:
        return 5173


def get_poker_server_port() -> int:
    """Return the base port for the Poker collaboration server (default 5273).

    5273 sits clear of retro's 5173..5193 walk range so both modes can run at
    once. The server walks upward from this port if it is busy (poker/server.py).
    """
    try:
        return int(os.getenv("POKER_PORT", "5273"))
    except ValueError:
        return 5273


def get_deck_server_port() -> int:
    """Return the port for the Reporting deck dev server (default 5373).

    5373 sits clear of retro's 5173..5193 and poker's 5273..5293 walk ranges.
    Unlike those two this server does not walk: a busy port is a hard failure,
    which is right once each worktree has its own block.
    """
    try:
        return int(os.getenv("DECK_PORT", "5373"))
    except ValueError:
        return 5373


def get_ship_server_port() -> int:
    """Return the base port for the Ship board server (default 5473).

    5473 sits clear of retro's 5173..5193 and poker's 5273..5293 walk ranges so
    a ship board can run alongside either. The server walks upward from this
    port if it is busy (ship/server.py).
    """
    try:
        return int(os.getenv("SHIP_PORT", "5473"))
    except ValueError:
        return 5473


def get_ship_board_enabled() -> bool:
    """True when a ship run should open its live, shareable web board.

    Opt-in for now (``YEABOI_SHIP_BOARD=1``). The board is read-only and safe,
    but it turns on the driver's ``stream-json`` path and, unless
    :func:`tunnels_disabled`, brings up a Cloudflare tunnel — both are new
    behaviour a plain terminal run should not get by surprise. Flip it on to
    watch a run from a browser and share it with teammates.
    """
    return os.getenv("YEABOI_SHIP_BOARD", "").strip().lower() in ("1", "true", "yes")


def tunnels_disabled() -> bool:
    """True when the live boards must not open a Cloudflare tunnel.

    The boards now start one the moment they open, which is right for a real
    ceremony and wrong for everything else: ``make run-dry`` advertises "fake
    delays, no LLM calls", and merely opening the Retro card under it would
    download a ~40 MB binary and publish a URL to the internet. Someone on a
    locked-down or offline network wants the same opt-out.

    Set ``YEABOI_NO_TUNNEL=1`` and the board still runs — the host reaches it on
    ``127.0.0.1`` — it simply has nothing to hand out, and says so.
    """
    return os.getenv("YEABOI_NO_TUNNEL", "").strip().lower() in ("1", "true", "yes")


def cloudflared_strict() -> bool:
    """True when only the pinned, checksum-verified cloudflared may run.

    ``ensure_cloudflared`` resolves four sources, and two of them carry no
    guarantee: ``CLOUDFLARED_PATH`` and a ``cloudflared`` already on ``PATH`` are
    whatever the machine offers, so the pinned SHA-256 that protects the download
    never sees them. Trusting them is the right default — a user who installed
    cloudflared themselves should be able to use it — but it means the supply
    chain the download path is careful about can be sidestepped by anything that
    can write to ``PATH`` or the environment.

    Set ``YEABOI_CLOUDFLARED_STRICT=1`` and both are refused; only the managed
    copy under ``~/.yeaboi/bin`` runs, and it is re-verified on every launch.
    """
    return os.getenv("YEABOI_CLOUDFLARED_STRICT", "").strip().lower() in ("1", "true", "yes")


def get_agentwatch_fresh_minutes() -> int:
    """How long a saved Agents report counts as fresh, in minutes (default 60).

    An Agents page opens on its last saved report and re-runs the engine —
    a full transcript scan plus one LLM call — only when that report is older
    than this. ``0`` re-runs on every open, which is what the pages used to do
    and what made "it keeps running in the background" true.
    """
    raw = os.getenv("YEABOI_AGENTWATCH_FRESH_MINUTES", "60")
    try:
        minutes = int(raw)
    except ValueError:
        return 60
    return max(0, min(minutes, 24 * 60))


def get_tunnel_timeout_minutes() -> int:
    """Auto-expiry for Cloudflare share tunnels, in minutes (default 60).

    A share left open forever is a real exposure window — code-gated, but
    reachable from the internet for as long as the host's TUI screen stays
    open. ``0`` switches the timeout off entirely, matching how
    ``SESSION_PRUNE_DAYS`` uses ``0`` to mean "never" (see
    ``get_session_prune_days``). Clamped to at most 24h so a typo can't leave
    a tunnel open for weeks.
    """
    raw = os.getenv("TUNNEL_TIMEOUT_MINUTES", "60")
    try:
        minutes = int(raw)
    except ValueError:
        return 60
    return max(0, min(minutes, 1440))


# The two share tiers. ``quick`` is the zero-setup default: a random
# ``*.trycloudflare.com`` hostname, no Cloudflare account, a join code as the
# only boundary. ``access`` is the opt-in tier: the host's own named tunnel on
# a hostname in a zone they control, fronted by Cloudflare Access, with every
# tunnel-borne request carrying a JWT this process verifies locally.
SHARE_MODE_QUICK = "quick"
SHARE_MODE_ACCESS = "access"

#: Env keys that only mean anything in the Access tier. Setting *any* of them is
#: read as "the host asked for the tier" — see ``sharing.identity.preflight``,
#: which is why a half-finished config is a named error rather than a silent
#: fall back to a public quick tunnel.
ACCESS_ENV_KEYS: tuple[str, ...] = (
    "CLOUDFLARE_TUNNEL_ID",
    "CLOUDFLARE_TUNNEL_CREDENTIALS",
    "CLOUDFLARE_ACCESS_HOSTNAME",
    "CLOUDFLARE_ACCESS_HOSTNAME_RETRO",
    "CLOUDFLARE_ACCESS_HOSTNAME_POKER",
    "CLOUDFLARE_ACCESS_HOSTNAME_SHARE",
    "CLOUDFLARE_ACCESS_HOSTNAME_SHIP",
    "CLOUDFLARE_ACCESS_TEAM",
    "CLOUDFLARE_ACCESS_AUD",
    "CLOUDFLARE_ACCESS_ADMIN_EMAILS",
)


def share_mode() -> str:
    """Which share tier the boards use: ``quick`` (default) or ``access``.

    One switch, deliberately explicit. The tier is never inferred from "the
    Access variables happen to be set", because the failure mode of guessing
    wrong is publishing a board on a quick tunnel while the host believes they
    are behind their identity provider. An unrecognised value reads as
    ``quick``: the tier that promises less cannot be entered by a typo.
    """
    raw = os.getenv("YEABOI_SHARE_MODE", "").strip().lower()
    return SHARE_MODE_ACCESS if raw == SHARE_MODE_ACCESS else SHARE_MODE_QUICK


def access_mode_enabled() -> bool:
    """True when the Access tier is switched on."""
    return share_mode() == SHARE_MODE_ACCESS


def share_tier_prompted() -> bool:
    """True once the first-share tier question no longer needs asking.

    The TUI asks exactly once, right before the first share, whether links
    should be open to anyone (the default) or to verified users only. Any
    resolution of that screen persists ``YEABOI_SHARE_TIER_PROMPTED=1``; an
    explicit ``YEABOI_SHARE_MODE`` answers the question too, however it was
    set. Settings ▸ Sharing and ``--setup-access`` remain the doors back in.
    """
    if os.getenv("YEABOI_SHARE_MODE", "").strip():
        return True
    return os.getenv("YEABOI_SHARE_TIER_PROMPTED", "").strip().lower() in ("1", "true", "yes")


def access_tunnel_id() -> str:
    """The named tunnel's UUID or name (``CLOUDFLARE_TUNNEL_ID``).

    This is a *locally*-managed tunnel — credentials file plus an ingress file
    yeaboi generates — not the dashboard's ``--token`` form. The reason is the
    port: every server here picks its port at bind time (``sharing/server.py``
    binds port 0, the boards walk a range on conflict), and a remotely-managed
    tunnel takes its ingress from the dashboard, where the port would have to
    be written down in advance. A locally-managed tunnel lets the ingress name
    the port we actually got.
    """
    return os.getenv("CLOUDFLARE_TUNNEL_ID", "").strip()


def access_credentials_file() -> str:
    """Path to the named tunnel's credentials JSON (``CLOUDFLARE_TUNNEL_CREDENTIALS``).

    Written by ``cloudflared tunnel create`` — usually
    ``~/.cloudflared/<uuid>.json``. yeaboi only *references* it from the ingress
    file it generates; the secret is never read into this process, never copied,
    and never logged.
    """
    return os.path.expanduser(os.getenv("CLOUDFLARE_TUNNEL_CREDENTIALS", "").strip())


def access_hostname(surface: str = "") -> str:
    """The stable hostname this surface is served at, e.g. ``retro.team.example.com``.

    ``CLOUDFLARE_ACCESS_HOSTNAME_{RETRO,POKER,SHARE}`` wins over the shared
    ``CLOUDFLARE_ACCESS_HOSTNAME``. The per-surface override is not a nicety:
    one named tunnel accepts many simultaneous connectors (that is how
    Cloudflare does HA), so a host who opens a retro board *and* a poker board
    on one hostname gets two connectors advertising ingress for it, and
    teammates land on whichever answers first. See
    ``sharing.access_tunnel.claim_hostname``, which refuses the second board
    rather than letting that happen silently.
    """
    if surface:
        specific = os.getenv(f"CLOUDFLARE_ACCESS_HOSTNAME_{surface.upper()}", "").strip()
        if specific:
            return specific
    return os.getenv("CLOUDFLARE_ACCESS_HOSTNAME", "").strip()


def access_team() -> str:
    """The Cloudflare Access team name, used to build the issuer and JWKS URLs.

    Accepts either the bare team name or the full
    ``https://<team>.cloudflareaccess.com`` — hosts copy whichever their
    dashboard shows them, and getting this wrong fails closed in a way that
    looks like "nobody can log in", so it is worth being forgiving here.
    """
    raw = os.getenv("CLOUDFLARE_ACCESS_TEAM", "").strip()
    raw = raw.removeprefix("https://").removeprefix("http://").rstrip("/")
    return raw.removesuffix(".cloudflareaccess.com")


def access_aud() -> str:
    """The Access application's AUD tag — what a token's ``aud`` claim must equal.

    This is the claim that binds a token to *this* application. Without it any
    valid token from the same Cloudflare team — issued for a different app, to a
    different audience — would verify, so the verifier refuses to be built
    without one.
    """
    return os.getenv("CLOUDFLARE_ACCESS_AUD", "").strip()


def access_admin_emails() -> frozenset[str]:
    """Verified emails granted host powers, lowercased (``CLOUDFLARE_ACCESS_ADMIN_EMAILS``).

    Comma-separated. In the Access tier this replaces the admin secret that
    otherwise travels in the host link's query string — where it reaches
    Cloudflare's edge access log and stays a static bearer for the life of the
    screen. Membership is tested by exact set equality, never substring: an
    ``in`` test would make ``ada@example.com`` match ``ada@example.com.evil.net``.
    """
    raw = os.getenv("CLOUDFLARE_ACCESS_ADMIN_EMAILS", "")
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


def get_slack_webhook_url() -> str:
    """Return the Slack incoming-webhook URL for standup delivery, or '' if unset."""
    return os.getenv("SLACK_WEBHOOK_URL", "") or ""


def set_slack_webhook_url(url: str) -> None:
    """Persist the Slack webhook URL to ~/.yeaboi/.env and apply it now."""
    config_file = set_config_value("SLACK_WEBHOOK_URL", url)
    os.environ["SLACK_WEBHOOK_URL"] = url
    logger.info("Slack webhook URL persisted to %s", config_file)


# ── Slack two-way ──────────────────────────────────────────────────────────
#
# A webhook answers a POST with the literal body ``ok`` and no message id, so
# yeaboi cannot identify its own message — which makes a reaction on it
# unreadable by construction. Reading anything back therefore needs a bot
# token, and that is the whole reason these exist beside the webhook rather
# than replacing it. With no token the webhook path is untouched.


def get_slack_bot_token() -> str:
    """Return the Slack bot token (``xoxb-…``) for two-way, or '' if unset."""
    return os.getenv("SLACK_BOT_TOKEN", "") or ""


def set_slack_bot_token(token: str) -> None:
    """Persist the Slack bot token to ~/.yeaboi/.env and apply it now."""
    config_file = set_config_value("SLACK_BOT_TOKEN", token)
    os.environ["SLACK_BOT_TOKEN"] = token
    logger.info("Slack bot token persisted to %s", config_file)


def get_slack_channel_id() -> str:
    """Return the Slack channel id (``C…``/``G…``) posts land in, or ''."""
    return os.getenv("SLACK_CHANNEL_ID", "") or ""


def set_slack_channel_id(channel: str) -> None:
    """Persist the Slack channel id to ~/.yeaboi/.env and apply it now."""
    config_file = set_config_value("SLACK_CHANNEL_ID", channel)
    os.environ["SLACK_CHANNEL_ID"] = channel
    logger.info("Slack channel id persisted to %s", config_file)


def get_slack_allowed_member_ids() -> str:
    """Return the raw allowlist of Slack member ids, or '' if unset.

    Raw rather than parsed, because the parse fails loudly and this getter must
    not: ``slack.allowlist`` owns the one place that decides what a malformed
    entry means (nobody is authorised).
    """
    return os.getenv("SLACK_ALLOWED_MEMBER_IDS", "") or ""


def set_slack_allowed_member_ids(ids: str) -> None:
    """Persist the Slack allowlist to ~/.yeaboi/.env and apply it now."""
    config_file = set_config_value("SLACK_ALLOWED_MEMBER_IDS", ids)
    os.environ["SLACK_ALLOWED_MEMBER_IDS"] = ids
    logger.info("Slack allowlist persisted to %s", config_file)


def get_slack_ack_reaction() -> str:
    """The emoji yeaboi adds to a message it has acted on ('' = off).

    Off by default because it needs ``reactions:write``, and the two-way lane is
    a *read* feature — a token that can write into a team channel is the scope
    an administrator is most likely to refuse. The reaction is a courtesy for
    humans and is never read back as a record.
    """
    return (os.getenv("SLACK_ACK_REACTION", "") or "").strip().strip(":")


def slack_two_way_ready() -> tuple[bool, str]:
    """(ready, why-not) for the two-way path — the one predicate every surface asks.

    Gated on the token AND the channel, never the token alone. Posting via
    ``chat.postMessage`` changes the *visible sender* — a webhook posts as its
    configured app, a bot posts as the bot user and must be invited to the
    channel — so a token pasted for some unrelated reason must not silently
    change how a team's daily standup looks, or break it with
    ``not_in_channel``.
    """
    if not get_slack_bot_token():
        return False, "no SLACK_BOT_TOKEN — a webhook cannot read, so two-way needs a bot token"
    if not get_slack_channel_id():
        return False, "no SLACK_CHANNEL_ID — set the channel the bot posts in and reads back from"
    return True, ""


def get_smtp_host() -> str:
    """Return the SMTP host for standup email delivery, or '' if unset."""
    return os.getenv("STANDUP_SMTP_HOST", "") or ""


def get_smtp_port() -> int:
    """Return the SMTP port (default 587)."""
    try:
        return int(os.getenv("STANDUP_SMTP_PORT", "587") or "587")
    except ValueError:
        return 587


def get_smtp_user() -> str:
    """Return the SMTP username, or '' if unset."""
    return os.getenv("STANDUP_SMTP_USER", "") or ""


def get_smtp_password() -> str:
    """Return the SMTP password, or '' if unset."""
    return os.getenv("STANDUP_SMTP_PASSWORD", "") or ""


def set_smtp_password(password: str) -> None:
    """Persist the SMTP password to ~/.scrum-agent/.env and apply it now."""
    config_file = set_config_value("STANDUP_SMTP_PASSWORD", password)
    os.environ["STANDUP_SMTP_PASSWORD"] = password
    logger.info("SMTP password persisted to %s", config_file)


def get_smtp_sender() -> str:
    """Return the From address for standup emails (defaults to the SMTP user)."""
    return os.getenv("STANDUP_SMTP_SENDER", "") or get_smtp_user()


def get_standup_email_recipients() -> list[str]:
    """Return the standup email recipient list, parsed from a comma-separated env var."""
    raw = os.getenv("STANDUP_EMAIL_RECIPIENTS", "") or ""
    return [addr.strip() for addr in raw.split(",") if addr.strip()]


def get_standup_user_name() -> str:
    """Return the display name for the current user's self-reported standup update.

    Reads STANDUP_USER_NAME; defaults to "Me" so a solo user still gets a sensible
    label without configuration.
    """
    return os.getenv("STANDUP_USER_NAME", "").strip() or "Me"


def get_performance_framework_path() -> str:
    """Return an optional path to a custom competency framework / review template.

    Reads PERFORMANCE_FRAMEWORK_PATH. When set, the 6-month review uses this file's
    contents in place of the bundled default framework (performance/references/
    competency_framework.md), so a lead can drop in their org's HR template.
    Returns "" when unset.
    """
    return os.getenv("PERFORMANCE_FRAMEWORK_PATH", "").strip()


# ---------------------------------------------------------------------------
# LLM provider configuration
# ---------------------------------------------------------------------------


def get_llm_provider() -> str:
    """Return the active LLM provider name (lowercase).

    Set LLM_PROVIDER in .env to switch providers. Defaults to 'anthropic'.
    Supported values: 'anthropic', 'openai', 'google', 'bedrock', 'ollama',
    plus the OpenAI-wire vendors in llm_providers.py ('xai', 'deepseek',
    'moonshot', 'mistral', 'qwen', 'zai').
    """
    return os.getenv("LLM_PROVIDER", "anthropic").lower()


def get_llm_model() -> str | None:
    """Return the model ID override from LLM_MODEL env var, or None to use the provider default."""
    return os.getenv("LLM_MODEL") or None


def get_bedrock_region() -> str:
    """Return the AWS region for Bedrock API calls.

    Reads AWS_REGION, then AWS_DEFAULT_REGION from env. Defaults to 'us-east-1'.
    """
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"


def get_aws_profile() -> str | None:
    """Return the AWS profile to use for Bedrock API calls.

    Reads AWS_PROFILE from env. If not set, auto-detects from ~/.aws/config
    by finding the first profile with a region or role_arn configured.
    Returns None if only [default] is available (boto3 handles that automatically).
    """
    profile = os.getenv("AWS_PROFILE")
    if profile:
        return profile

    # Auto-detect: parse ~/.aws/config for non-default profiles
    try:
        config_path = Path.home() / ".aws" / "config"
        if config_path.exists():
            import configparser

            cfg = configparser.ConfigParser()
            cfg.read(config_path)
            for section in cfg.sections():
                # AWS config sections are [default] or [profile <name>]
                if section.startswith("profile "):
                    profile_name = section.removeprefix("profile ").strip()
                    if cfg.has_option(section, "role_arn") or cfg.has_option(section, "credential_source"):
                        return profile_name
    except Exception:
        pass

    return None


def get_openai_api_key() -> str | None:
    """Return the OpenAI API key, or None if not set."""
    return os.getenv("OPENAI_API_KEY") or None


def get_google_api_key() -> str | None:
    """Return the Google AI API key, or None if not set."""
    return os.getenv("GOOGLE_API_KEY") or None


def get_provider_api_key(provider: str) -> str | None:
    """Return the API key for any OpenAI-wire vendor, or None if not set.

    One accessor rather than a getter per vendor — the key env lives on the
    provider's row in llm_providers.py. Returns None for providers that are
    not on that table (anthropic, google, bedrock, ollama have their own).
    """
    from yeaboi.llm_providers import spec

    found = spec(provider)
    return (os.getenv(found.key_env) or None) if found else None


def get_ollama_base_url() -> str:
    """Return the base URL of the local Ollama server.

    Ollama runs entirely on the user's machine — no API key, no cloud account.
    Override with OLLAMA_BASE_URL for a non-default port or a server elsewhere
    on the network. Trailing slashes are stripped so URL joining is predictable.
    """
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def get_ollama_num_ctx() -> int:
    """Return the context window (tokens) requested from the Ollama model.

    Ollama's server default context (2-4k tokens) is smaller than the biggest
    assembled prompts in the planning pipeline (~5k tokens plus optional repo /
    Confluence context). A too-small context silently truncates the prompt,
    which destroys the JSON output the pipeline parses — so we default to 16384
    and let power users tune it via OLLAMA_NUM_CTX (e.g. lower it on low-RAM
    machines, raise it for very large projects).
    """
    try:
        return int(os.getenv("OLLAMA_NUM_CTX", "16384"))
    except ValueError:
        return 16384


def is_llm_configured() -> tuple[bool, str]:
    """Return (ok, message) for whether the selected LLM provider has credentials.

    Cheap, no network call — just checks the env var the active provider needs.
    Callers (e.g. the standup engine) use this to surface a clear "set your API
    key" message instead of silently degrading. Bedrock uses IAM, so a configured
    AWS region/profile counts as ready.
    """
    provider = get_llm_provider()
    if provider == "anthropic":
        # Either credential counts: a subscription token authenticates as a
        # bearer and needs no key at all (see get_llm).
        ok = bool(os.getenv("ANTHROPIC_API_KEY") or get_anthropic_subscription_token())
        return (ok, "ANTHROPIC_API_KEY not set, and no Claude subscription signed in")
    if provider == "openai":
        return (bool(get_openai_api_key()), "OPENAI_API_KEY not set")
    if provider == "google":
        return (bool(get_google_api_key()), "GOOGLE_API_KEY not set")
    if provider == "bedrock":
        ok = bool(os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or get_aws_profile())
        return (ok, "AWS credentials/region not configured for Bedrock")
    if provider == "ollama":
        # Local provider — no credentials to check. Server reachability is
        # verified at call time (a down server surfaces an actionable error).
        return (True, "")
    from yeaboi.llm_providers import spec

    vendor = spec(provider)
    if vendor is not None:
        return (bool(get_provider_api_key(provider)), f"{vendor.key_env} not set")
    return (bool(os.getenv("ANTHROPIC_API_KEY")), f"No API key configured for provider '{provider}'")


def get_voice_model() -> str:
    """Return the local Whisper model size for voice input.

    Transcription runs on-device via faster-whisper, so this is a model *size*,
    not a cloud model name. Override with the VOICE_MODEL env var. Valid values:
    ``tiny``, ``base`` (default), ``small``, ``medium``, ``large-v3`` (and the
    English-only ``.en`` variants). Larger = more accurate but slower and a
    bigger one-time download.
    """
    return os.getenv("VOICE_MODEL") or "base"


def get_voice_device() -> str:
    """Return the configured microphone preference for voice input.

    Reads ``VOICE_DEVICE``: either a PortAudio device index (``"2"``) or a
    case-insensitive substring of the device name (``"shure"``). Empty means
    "use the system default input", which is how voice behaved before the
    setting existed. Resolution to an actual index happens in
    :func:`yeaboi.voice.resolve_device` — this stays a plain string read so
    config never imports the optional audio stack.
    """
    return (os.getenv("VOICE_DEVICE") or "").strip()


def get_session_prune_days() -> int:
    """Return the number of days after which old sessions are pruned.

    Phase 8C: reads ``SESSION_PRUNE_DAYS`` env var.
    Default is 30. Set to 0 to disable pruning. Invalid values fall back to 30.
    """
    raw = os.getenv("SESSION_PRUNE_DAYS", "30")
    try:
        value = int(raw)
        return value if value >= 0 else 30
    except ValueError:
        return 30


def get_log_level() -> str:
    """Return the configured log level for the file logger.

    Reads ``LOG_LEVEL`` from .env. Defaults to ``WARNING``.
    Valid values: DEBUG, INFO, WARNING, ERROR.
    Invalid values fall back to WARNING.
    """
    raw = os.getenv("LOG_LEVEL", "WARNING").upper()
    if raw in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        return raw
    return "WARNING"


# Levels settable from the Settings page cycle button. CRITICAL stays readable
# from .env for power users but is deliberately not part of the cycle.
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def set_log_level(level: str) -> None:
    """Persist LOG_LEVEL to ~/.yeaboi/.env and apply it to the running process.

    Uses dotenv's set_key so only this one key is updated (preserves other
    keys/comments). os.environ is updated too so get_log_level() reflects the
    change immediately. Callers that want live handlers retuned should also
    call yeaboi.logging_setup.apply_level().

    Raises ValueError for levels outside VALID_LOG_LEVELS.
    """
    level = level.upper()
    if level not in VALID_LOG_LEVELS:
        raise ValueError(f"invalid log level: {level}")
    # Route through set_config_value so the shared .env stays locked to 0o600.
    config_file = set_config_value("LOG_LEVEL", level)
    os.environ["LOG_LEVEL"] = level
    logger.info("Log level set to %s (persisted to %s)", level, config_file)


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def is_team_analysis_jira_dev_links_enabled() -> bool:
    """When true, team analysis calls Jira dev-status for linked PRs/repos (extra API calls)."""
    return _env_truthy("TEAM_ANALYSIS_JIRA_DEV_LINKS")


def is_team_analysis_azdo_pr_search_enabled() -> bool:
    """When true, team analysis scans AzDO Git PRs for work item links / branch names (extra API calls)."""
    return _env_truthy("TEAM_ANALYSIS_AZDO_BRANCH_SEARCH")


def get_team_analysis_azdo_pr_search_max_repos() -> int:
    """Max Git repos to scan per analysis when branch/PR search is enabled (1–50, default 10)."""
    raw = os.getenv("TEAM_ANALYSIS_AZDO_PR_SEARCH_MAX_REPOS", "10")
    try:
        return max(1, min(int(raw), 50))
    except ValueError:
        return 10


def get_team_analysis_azdo_pr_search_top() -> int:
    """Max pull requests per repo per status when PR search is enabled (10–200, default 75)."""
    raw = os.getenv("TEAM_ANALYSIS_AZDO_PR_SEARCH_PRS_PER_REPO", "75")
    try:
        return max(10, min(int(raw), 200))
    except ValueError:
        return 75


def get_team_analysis_azdo_repo_allowlist() -> frozenset[str] | None:
    """Optional comma-separated repo names (lowercase) to limit PR search; None = all repos up to max."""
    raw = os.getenv("TEAM_ANALYSIS_AZDO_REPO_ALLOWLIST", "").strip()
    if not raw:
        return None
    return frozenset(x.strip().lower() for x in raw.split(",") if x.strip())


def disable_langsmith_tracing() -> None:
    """Disable LangSmith by unsetting LANGSMITH_TRACING in the current process.

    LangSmith reads LANGSMITH_TRACING from os.environ at runtime, so removing
    it prevents the SDK from attempting to send traces for the rest of the process.
    """
    logger.info("Disabling LangSmith tracing for this process")
    os.environ.pop("LANGSMITH_TRACING", None)
