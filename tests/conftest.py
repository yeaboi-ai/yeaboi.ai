"""Top-level test configuration.

Provides VCR.py (pytest-recording) settings for contract tests:
- Cassettes: stored per-module at <test_dir>/cassettes/<module_name>/
- Token scrubbing: strips Authorization headers, API keys, and PATs
  from recorded cassettes so they're safe to commit.

See README: "Testing — Contract Tests" for background on VCR.py replay.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import webbrowser
from pathlib import Path

import pytest

# Render tests use the same terminal in an editor, CI, or an assistant shell.
# Individual tests can still override these values with monkeypatch.
os.environ.pop("NO_COLOR", None)
os.environ["TERM"] = "xterm-256color"
os.environ["COLUMNS"] = "140"
os.environ["LINES"] = "40"

# At module scope, and deliberately not a fixture: `web/assets.py` resolves the
# bundle directory once, at import, so by the time any fixture runs the choice
# has been made. A developer who exports this to serve a Vite `dist/` would
# otherwise have the whole suite assert against bundles that are not the
# committed ones — passing or failing for a reason nothing in the output names.
os.environ.pop("YEABOI_WEB_STATIC", None)

# Same reason, one line further down: `paths.ROOT_DIR` is resolved at import, so
# this cannot be a fixture either. Without it the suite writes the developer's
# REAL ~/.yeaboi/data/sessions.db, and two worktrees running `make test-fast` at
# once corrupt each other's ledger through it.
#
# Derived from this file's path rather than from the worktree marker, so it holds
# for a non-editable install and a raw source tree too. `setdefault`, so CI or a
# make recipe can still choose. Credentials are untouched on purpose:
# ~/.yeaboi/.env is pinned outside YEABOI_HOME and is shared by design.
os.environ.setdefault("YEABOI_HOME", str(Path(__file__).resolve().parent.parent / ".pytest-home"))


@pytest.fixture(scope="session", autouse=True)
def _sandbox_allows_test_dirs(tmp_path_factory):
    """Whitelist pytest temp dirs + repo fixtures in the filesystem sandbox.

    The app is sandboxed to ~/.yeaboi (fs_policy.py); tests exercise exports,
    imports, and repo reads against pytest tmp dirs and tests/fixtures/, which
    would otherwise all be denied. Session-scoped (a plain MonkeyPatch, since
    the monkeypatch fixture is function-scoped) so class/module-scoped fixtures
    are covered too. Every per-test tmp_path lives under the session basetemp.
    Tests that verify denial behaviour (test_fs_policy.py) override this via
    their own monkeypatch of YEABOI_ALLOWED_PATHS.
    """
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    basetemp = tmp_path_factory.getbasetemp()
    fixtures_dir = Path(__file__).parent / "fixtures"
    # CWD too: the (unmodifiable) REPL integration suite exports scrum-plan.*
    # into the repo root it runs from.
    mp.setenv("YEABOI_ALLOWED_PATHS", f"{basetemp},{fixtures_dir},{Path.cwd()}")
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _no_env_leak():
    """`os.environ` is restored after every test, whatever the test did to it.

    Thirteen setters in ``config.py`` write straight to ``os.environ`` on
    purpose — ``set_tips_enabled``, ``set_duck_enabled``, the beta ack, the log
    level and the rest — so a preference change takes effect in the running
    session without a reload. A test that calls one without putting the key under
    ``monkeypatch`` therefore leaks it into every test that runs *after* it, in
    whatever order the collector happened to pick.

    That is invisible until the order changes, and it changed twice at once here:
    ``tests/*.py`` joined the unit lane, and the lane went parallel.
    ``test_set_tips_enabled_preserves_other_keys`` left ``TIPS_ENABLED=false``
    behind, and four welcome-screen tests then rendered a screen with no tip
    strip and failed on a missing row — passing alone, failing in the suite, with
    nothing in either failure naming the environment.

    Restoring wholesale rather than fixing the two callers, because the callers
    are not the bug: writing to ``os.environ`` is what those functions are *for*,
    there are thirteen of them, and the next one added would reintroduce this
    with no test to catch it. Cost is one dict copy per test.
    """
    before = os.environ.copy()
    yield
    if os.environ != before:
        os.environ.clear()
        os.environ.update(before)


class RealBrowserBlocked(BaseException):
    """Raised when a test reaches a real ``webbrowser`` call.

    Deliberately a ``BaseException``, not an ``Exception``: all three production
    call sites wrap their ``webbrowser.open`` in ``except Exception`` and degrade
    to a "copy this URL" branch, so an ``Exception`` here would be swallowed and
    the guard would silently reroute the test instead of failing it.
    """


@pytest.fixture(autouse=True)
def _reset_solo_mode():
    """Clear the process-local Solo/Team world flag between tests.

    ``yeaboi.config`` holds it in a module global, so a test that enters the
    Solo world leaks it into every test that runs after it in the same worker.
    """
    from yeaboi import config

    config.set_solo_mode(False)
    yield
    config.set_solo_mode(False)


@pytest.fixture(autouse=True)
def _no_real_browser(monkeypatch):
    """No test may open a real browser tab.

    Three production paths call webbrowser (standup/gap_issues.py, feedback.py, and the
    TUI mode_select handler); a test that forgets to patch one hijacks the developer's
    browser on every `make test-fast`. Tests that legitimately exercise those paths patch
    webbrowser themselves — they share this MonkeyPatch instance, so their setattr lands
    after ours and wins.

    Patching the webbrowser module once covers every call site: each importer holds a
    reference to the same module object, including the function-local ``import webbrowser``
    in mode_select (resolved from ``sys.modules`` at call time).

    Function-scoped on purpose, unlike the session-scoped sandbox fixture above: a test
    overriding this guard is the intended path, and that only works when it shares the
    ``monkeypatch`` instance. Nothing reaches webbrowser from a higher-scoped fixture today.
    """

    def _blocked(url, *args, **kwargs):
        raise RealBrowserBlocked(f"test tried to open a real browser: {url}")

    # ``get`` too: ``webbrowser.get(...).open(url)`` would otherwise bypass the stubs.
    for name in ("open", "open_new", "open_new_tab", "get"):
        monkeypatch.setattr(webbrowser, name, _blocked)


# ---------------------------------------------------------------------------
# Sensitive headers / query params to scrub from recorded cassettes
# ---------------------------------------------------------------------------
_SCRUBBED_HEADERS = [
    "Authorization",
    "X-Api-Key",
    "Private-Token",
    "Cookie",
    "Set-Cookie",
]

_SCRUBBED_QUERY_PARAMS = [
    "api_key",
    "token",
    "access_token",
]


def _scrub_response(response: dict) -> dict:
    """Remove sensitive headers from recorded responses."""
    headers = response.get("headers", {})
    for header in _SCRUBBED_HEADERS:
        headers.pop(header, None)
        headers.pop(header.lower(), None)
    return response


def _scrub_request(request):
    """Remove sensitive headers and query params from recorded requests.

    VCR.py's Request.query is a read-only property derived from the URI,
    so we scrub sensitive query params by rewriting the URI directly.
    """
    from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

    for header in _SCRUBBED_HEADERS:
        if header in request.headers:
            request.headers[header] = "SCRUBBED"
        if header.lower() in request.headers:
            request.headers[header.lower()] = "SCRUBBED"

    # Scrub sensitive query parameters by rewriting the URI
    parsed = urlparse(request.uri)
    if parsed.query:
        params = parse_qs(parsed.query, keep_blank_values=True)
        changed = False
        for key in _SCRUBBED_QUERY_PARAMS:
            if key in params:
                params[key] = ["SCRUBBED"]
                changed = True
        if changed:
            request.uri = urlunparse(parsed._replace(query=urlencode(params, doseq=True)))
    return request


# ---------------------------------------------------------------------------
# pytest-recording VCR configuration fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vcr_config():
    """VCR.py configuration applied to all @pytest.mark.vcr tests.

    pytest-recording picks up this fixture automatically. Key settings:
    - before_record_request / before_record_response: scrub tokens
    - decode_compressed_response: store readable JSON, not gzipped blobs
    - filter_headers: belt-and-suspenders scrubbing on record
    """
    return {
        "before_record_request": _scrub_request,
        "before_record_response": _scrub_response,
        "decode_compressed_response": True,
        "filter_headers": _SCRUBBED_HEADERS,
        # Match on endpoint identity, not exact query strings. PyJira adds
        # defaults like maxResults=50 that vary by version — we care about
        # the response shape, not the exact query params sent.
        "match_on": ["method", "scheme", "host", "port", "path"],
    }


class RealPackageInstallBlocked(BaseException):
    """Raised when a test reaches a real dictation-install or model-download spawn.

    A ``BaseException`` for the same reason as :class:`RealBrowserBlocked`:
    ``install_packages`` and ``download_model`` both wrap their spawn in
    ``except OSError`` and degrade to a "could not start" message, and the
    ``_run_install`` worker now catches ``Exception`` — so a plain exception here
    would be swallowed and the guard would quietly report a failed install
    instead of failing the test.
    """


@pytest.fixture(autouse=True)
def _isolated_connection_status(tmp_path, monkeypatch):
    """Each test gets its own connection-status store.

    ``verify_connection`` records a probe of the stored credentials, so without
    this one test's recorded outcome is the next one's starting state — and the
    presence check that normally expires a row cannot tell the two apart when
    both set the same env.
    """
    from yeaboi.connectors import verify_status

    monkeypatch.setattr(verify_status, "_store_path", lambda: tmp_path / "connection_status.json")
    verify_status.invalidate()
    yield
    verify_status.invalidate()


@pytest.fixture(autouse=True)
def _no_real_package_install(monkeypatch):
    """No test may spawn a real package manager or model download.

    This is not hypothetical. ``TestDoubleTapInDescriptionLoop`` patched
    ``is_voice_available`` but not the strict ``probe_voice_backend``, so on a
    machine *without* the voice extra — which is every CI runner — the double-tap
    fell through to the install offer, its leftover ``"enter"`` keystroke accepted
    it, and the job ran a real ``uv pip install`` plus a 145 MB model fetch until
    the runner was killed. It passed locally, where the extra is installed and the
    offer never appears.

    Blocks ``voice_install._popen``, the module's single spawn seam, rather than
    ``subprocess.Popen`` — patching the latter would patch the shared module for
    every other test in the suite. Tests that legitimately drive the installer
    patch the same name and share this MonkeyPatch instance, so their setattr
    lands after ours and wins.
    """

    def _blocked(argv, *args, **kwargs):
        raise RealPackageInstallBlocked(f"test tried to spawn a real installer: {argv}")

    monkeypatch.setattr("yeaboi.voice_install._popen", _blocked)


class RealTunnelSpawnBlocked(BaseException):
    """A test reached a real ``cloudflared``. ``BaseException`` for the same
    reason as the two guards around it: the tunnel code's whole contract is to
    degrade gracefully, so a plain ``Exception`` would be swallowed by one of
    its broad handlers and reported as "the tunnel did not come up" rather than
    as the missing monkeypatch it is.
    """


@pytest.fixture(autouse=True)
def _no_real_tunnel_spawn(monkeypatch):
    """No test may spawn the real, network-reaching cloudflared.

    Blocks the two spawn seams (``retro.tunnel._popen`` and
    ``sharing.access_tunnel._run``) rather than ``subprocess.Popen``/``run``,
    so the shared modules stay untouched for every other test.

    Only a *real* cloudflared is refused — the app-managed pinned copy, or one
    found on ``PATH``. The tunnel tests drive fake shell scripts that are also
    named ``cloudflared`` (in ``tmp_path``), and those are exactly what a unit
    test should be running, so they pass straight through.
    """
    real_popen = subprocess.Popen
    real_run = subprocess.run

    def _real_binaries() -> set[Path]:
        found: set[Path] = set()
        for candidate in (shutil.which("cloudflared"), os.getenv("CLOUDFLARED_PATH")):
            if candidate:
                found.add(Path(candidate).resolve())
        managed = Path.home() / ".yeaboi" / "bin" / "cloudflared"
        if managed.exists():
            found.add(managed.resolve())
        return found

    def _guard(delegate):
        def _spawn(argv, *args, **kwargs):
            try:
                target = Path(str(argv[0])).resolve()
            except (OSError, IndexError, TypeError):
                return delegate(argv, *args, **kwargs)
            if target in _real_binaries():
                raise RealTunnelSpawnBlocked(f"test tried to spawn the real cloudflared: {argv}")
            return delegate(argv, *args, **kwargs)

        return _spawn

    monkeypatch.setattr("yeaboi.retro.tunnel._popen", _guard(real_popen))
    monkeypatch.setattr("yeaboi.sharing.access_tunnel._run", _guard(real_run))
    monkeypatch.setattr("yeaboi.sharing.access_setup._popen", _guard(real_popen))


class RealGitHubWriteBlocked(BaseException):
    """A test reached the real `gh` CLI. Never caught — see the fixture below.

    ``BaseException`` for the same reason ``RealPackageInstallBlocked`` is: this
    fires inside code whose whole contract is to degrade gracefully, and a broad
    ``except Exception`` on the way out would swallow it and report a failed `gh`
    call instead of failing the test.
    """


@pytest.fixture(autouse=True)
def _no_real_gh_calls(monkeypatch):
    """No test may shell out to the real `gh` CLI.

    This is not hypothetical. ``TestMigrateProposalsApply`` stubbed
    ``cowork_setup._api`` — the REST seam — and asserted on the calls. That was
    complete for as long as ``_reclassify`` had only a REST branch. The moment it
    grew a ``gh`` branch (``TRANSPORT`` defaults to ``"gh"``), those same tests
    took the unstubbed path and ran real commands against the real repository:
    ``gh issue edit 7 --add-label cowork:queued`` plus four ``gh issue comment``
    calls, landing on a PR merged months earlier. They had to be undone by hand.

    A test asserting on writes is exactly a test that will make them if a seam
    moves, and "stub the right seam" is not a property anything checks. Blocking
    the single process seam is, and it fails loudly rather than mutating anything.

    Blocked at the **process spawn** inside ``_gh_transport``, not at ``gh()``
    itself. That is deliberate and is the only level that works for everyone:
    ``test_gh_transport.py`` calls ``transport.gh`` directly on purpose — proving a
    missing binary degrades to 127 rather than a traceback — and stubs
    ``transport._run`` beneath it. Both levels are legitimate, both share this
    MonkeyPatch instance, and a stub at either level lands after ours and wins.
    What is left over is precisely the case with no stub at all, which is the one
    that reaches the network.

    **There can be more than one ``_gh_transport``, and patching the wrong one is
    silent.** ``scripts/`` is not a package, and the two loaders disagree about who
    owns the name: a script does a plain ``import _gh_transport``, binding whatever
    object exists at its load time, while ``test_gh_transport.py`` builds a *fresh*
    module off the file path and assigns it over ``sys.modules["_gh_transport"]``.
    Collection is alphabetical, so in a full-suite run the registry entry is the
    fresh object and a plain-importing script still holds the original — patching
    only the registry leaves that module unguarded, and the guard's own proof
    passes when run on one file because there the two happen to coincide.

    So every distinct transport object reachable from the loaded scripts modules is
    patched, deduped by identity. If none was imported there is nothing to block
    and this is a no-op.
    """
    import sys as _sys

    reachable = []
    for name in ("_gh_transport", "pr_feedback"):
        module = _sys.modules.get(name)
        if module is None:
            continue
        candidate = module if name == "_gh_transport" else getattr(module, "transport", None)
        if candidate is not None and not any(candidate is seen for seen in reachable):
            reachable.append(candidate)

    for transport in reachable:
        real = transport._run

        def _blocked(argv, *args, _real=real, **kwargs):
            # `gh` only. Everything else through this seam is `git remote get-url
            # origin` — a local, read-only lookup that cannot reach GitHub and that
            # several tests legitimately let run. Blocking it too would fail seven
            # tests that never did anything wrong, and a guard with collateral like
            # that gets loosened rather than kept.
            if argv and argv[0] == "gh":
                raise RealGitHubWriteBlocked(f"test tried to spawn the real gh CLI: {argv}")
            return _real(argv, *args, **kwargs)

        monkeypatch.setattr(transport, "_run", _blocked)

        def _blocked_http(request, *args, **kwargs):
            # The REST half. `_run` covers the `gh` CLI; this covers the transport
            # a routine session actually uses, which authenticates from GH_TOKEN in
            # the environment — so on any machine that exports one, an unstubbed
            # call here reaches the real repository and writes to it. That is the
            # same accident `_run` was named for, through the seam that did not
            # exist yet when it was.
            #
            # Unconditional, unlike `_run`'s `gh`-only test: there is no local,
            # read-only call through this seam to spare. Every one of them is
            # api.github.com.
            method = getattr(request, "get_method", lambda: "?")()
            url = getattr(request, "full_url", request)
            raise RealGitHubWriteBlocked(f"test tried to reach the real GitHub API: {method} {url}")

        monkeypatch.setattr(transport, "_urlopen", _blocked_http)
