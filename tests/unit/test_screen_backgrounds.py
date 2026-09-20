"""Every TUI page paints its own mode-tinted background (no terminal bleed-through).

Two layers of protection:
- Render tests: representative screens emit the truecolor background SGR escape
  (``48;2;R;G;B``) for their mode's ``Theme.bg`` when captured with a truecolor
  console, so the whole terminal shows the mode's colour.
- An AST scan: no full-screen ``Panel`` under ``src/yeaboi/ui`` may bypass
  ``build_page_panel`` (which is what applies the background style).
"""

from __future__ import annotations

import ast
import io
import pathlib
import re

import pytest
from rich.console import Console
from rich.text import Text

from yeaboi.ui.shared._components import (
    ANALYSIS_THEME,
    CHANGELOG_THEME,
    FEEDBACK_THEME,
    NEUTRAL_BG,
    PERFORMANCE_THEME,
    PLANNING_THEME,
    POKER_THEME,
    REPORTING_THEME,
    RETRO_THEME,
    SETTINGS_THEME,
    STANDUP_THEME,
    USAGE_THEME,
    build_page_panel,
    standup_title,
)

UI_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "yeaboi" / "ui"


def _render(renderable, *, width: int = 100, height: int = 30) -> str:
    """Render with escapes emitted so background SGR codes appear in the capture."""
    console = Console(
        file=io.StringIO(),
        width=width,
        height=height,
        force_terminal=True,
        color_system="truecolor",
        legacy_windows=False,
        highlight=False,
    )
    console.print(renderable)
    return console.file.getvalue()


def _bg_escape(rgb: str) -> str:
    """Truecolor background SGR fragment for an "rgb(r,g,b)" style string."""
    r, g, b = re.fullmatch(r"rgb\((\d+),(\d+),(\d+)\)", rgb).groups()
    return f"48;2;{r};{g};{b}"


class TestThemeBackgrounds:
    """Each mode's Theme carries a dark tint and build_page_panel emits it."""

    MODE_THEMES = {
        "analysis": ANALYSIS_THEME,
        "planning": PLANNING_THEME,
        "usage": USAGE_THEME,
        "standup": STANDUP_THEME,
        "retro": RETRO_THEME,
        "poker": POKER_THEME,
        "performance": PERFORMANCE_THEME,
        "reporting": REPORTING_THEME,
    }

    @pytest.mark.parametrize("name", sorted(MODE_THEMES))
    def test_mode_theme_bg_renders(self, name):
        theme = self.MODE_THEMES[name]
        out = _render(build_page_panel(Text("hello"), theme=theme, height=10))
        assert _bg_escape(theme.bg) in out

    def test_mode_backgrounds_are_unified(self):
        # Per-mode background tints were dropped for one consistent backdrop across
        # every screen: all modes now share the neutral base. Accents stay distinct.
        bgs = [t.bg for t in self.MODE_THEMES.values()]
        assert set(bgs) == {NEUTRAL_BG}
        accents = [t.accent for t in self.MODE_THEMES.values()]
        assert len(set(accents)) == len(accents)  # each mode keeps its own accent hue

    def test_mode_tints_are_dark(self):
        # Foreground styles are designed for dark backgrounds; keep tints deep.
        for theme in self.MODE_THEMES.values():
            r, g, b = (int(v) for v in re.findall(r"\d+", theme.bg))
            assert max(r, g, b) <= 40, theme.bg

    def test_neutral_pages_use_neutral_base(self):
        for theme in (SETTINGS_THEME, CHANGELOG_THEME, FEEDBACK_THEME):
            assert theme.bg == NEUTRAL_BG
        out = _render(build_page_panel(Text("hello"), height=10))
        assert _bg_escape(NEUTRAL_BG) in out


class TestRepresentativeScreens:
    """Real screen builders route through build_page_panel and emit their tint."""

    def test_mode_home_uses_neutral_base(self):
        from yeaboi.ui.mode_select.screens._screens import _build_mode_screen

        out = _render(_build_mode_screen(0, width=100, height=30))
        assert _bg_escape(NEUTRAL_BG) in out

    def test_analysis_setup_screen_uses_analysis_tint(self):
        from yeaboi.ui.mode_select.screens._screens_secondary import _build_analysis_depth_screen

        out = _render(_build_analysis_depth_screen(0, width=100, height=30))
        assert _bg_escape(ANALYSIS_THEME.bg) in out

    def test_run_hub_uses_mode_theme(self):
        from yeaboi.ui.mode_select.screens._run_hub_screen import _build_run_hub_screen

        out = _render(_build_run_hub_screen([], 0, title_fn=standup_title, theme=STANDUP_THEME))
        assert _bg_escape(STANDUP_THEME.bg) in out

    def test_run_hub_without_theme_falls_back_to_neutral(self):
        from yeaboi.ui.mode_select.screens._run_hub_screen import _build_run_hub_screen

        out = _render(_build_run_hub_screen([], 0, title_fn=standup_title))
        assert _bg_escape(NEUTRAL_BG) in out

    def test_provider_select_frame_uses_neutral_base(self):
        from yeaboi.ui.provider_select.screens._screens import _build_screen_frame

        out = _render(
            _build_screen_frame(subtitle="s", step=0, body_items=[Text("body")], body_height=1, width=100, height=30)
        )
        assert _bg_escape(NEUTRAL_BG) in out

    def test_splash_frame_uses_neutral_base(self):
        from yeaboi.ui.splash import _build_splash_frame

        out = _render(_build_splash_frame(["YEABOI"], width=100, height=30))
        assert _bg_escape(NEUTRAL_BG) in out

    def test_screensaver_uses_neutral_base(self):
        from yeaboi.ui.shared._screensaver import build_screensaver

        out = _render(build_screensaver(width=80, height=24, elapsed=0.0))
        assert _bg_escape(NEUTRAL_BG) in out

    def test_screensaver_tiny_terminal_uses_neutral_base(self):
        from yeaboi.ui.shared._screensaver import build_screensaver

        out = _render(build_screensaver(width=18, height=6, elapsed=0.0), width=18, height=6)
        assert _bg_escape(NEUTRAL_BG) in out

    def test_beta_notice_uses_the_gated_modes_theme(self):
        from yeaboi.ui.shared._beta_notice import _build_beta_notice_screen

        out = _render(_build_beta_notice_screen(mode_key="performance", width=100, height=30))
        assert _bg_escape(PERFORMANCE_THEME.bg) in out


class TestNoRawFullScreenPanels:
    """AST guard: full-screen Panels must be built via build_page_panel.

    A "full-screen" Panel is one with a ``height=`` kwarg, no ``width=`` kwarg
    and ``expand`` not explicitly False — it fills the terminal, so an unstyled
    one would show the user's terminal background through padding and blank
    rows. Adding a new screen? Return ``build_page_panel(...)`` with the mode's
    Theme (see .claude/skills/tui-standards).
    """

    # Files allowed to construct a styled full-screen Panel directly.
    ALLOWED = {
        "shared/_components.py",  # build_page_panel itself
        "splash.py",  # pre-Live boot screen, styles itself with NEUTRAL_BG
    }

    def test_no_unstyled_full_screen_panel(self):
        offenders: list[str] = []
        for path in sorted(UI_ROOT.rglob("*.py")):
            rel = path.relative_to(UI_ROOT).as_posix()
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                if name != "Panel":
                    continue
                kw = {k.arg: k.value for k in node.keywords if k.arg}
                expand_false = isinstance(kw.get("expand"), ast.Constant) and kw["expand"].value is False
                if "height" not in kw or "width" in kw or expand_false:
                    continue  # inner card / chip / popup — inherits the page bg
                if "style" in kw and rel in self.ALLOWED:
                    continue
                offenders.append(f"{rel}:{node.lineno}")
        assert not offenders, (
            "Full-screen Panel(s) bypass build_page_panel — return "
            f"build_page_panel(..., theme=<mode theme>) instead: {offenders}"
        )


class TestTruecolorConsoles:
    """A test asserting an rgb SGR fragment must pin its console's color system.

    Rich picks a color system from ``COLORTERM``/``TERM`` when ``color_system``
    is left to auto-detection. Dev shells export ``COLORTERM=truecolor``; CI
    runners do not. So a console built with ``force_terminal=True`` and nothing
    else renders 24-bit locally and downgrades to 8-colour on CI — a test that
    asserts ``"34;158;122"`` passes on every machine it was written on and
    fails only after it is pushed.

    It is worse than one red test, because Rich caches the rendered escape on
    the ``Style`` object (``Style._ansi``) and ``Style.parse`` memoises Styles
    globally: the *first* 8-colour render of a shared style poisons every later
    truecolor render of it in the same process. One unpinned console took nine
    tests in this file down with it — tests that pin truecolor correctly and
    have nothing to do with the console that broke them.
    """

    TESTS_ROOT = pathlib.Path(__file__).resolve().parents[1]

    # An rgb SGR fragment as it appears in an assertion: "34;158;122" (a colour)
    # or "48;2;16;16;20" (a background). Two or more semicolon-joined numbers.
    RGB_FRAGMENT = re.compile(r"\"\d{1,3};\d{1,3};[\d;]*\d{1,3}\"")

    # `tests/integration/` is exempt, and deliberately: it is a separate CI job in
    # its own process, so it cannot poison the unit lane, and AGENTS.md forbids
    # editing `test_repl.py` at all.
    EXEMPT_DIRS = ("integration",)

    def _lane_files(self):
        """Every file the unit lane runs — `tests/unit/**` plus `tests/*.py`."""
        for path in sorted(self.TESTS_ROOT.rglob("test_*.py")):
            rel = path.relative_to(self.TESTS_ROOT)
            if rel.parts[0] in self.EXEMPT_DIRS:
                continue
            yield path, rel.as_posix()

    def test_no_unit_lane_console_leaves_its_color_system_to_the_environment(self):
        """The carve-out below is not safe for the *poisoning*, only for the assert.

        A file that asserts no rgb fragment can still render one of the shared
        styles through an 8-colour console and fix `Style._ansi` for the whole
        worker — which is exactly how nine tests here went red from a change to
        the music bar in `tests/test_mode_select.py`, a file that asserts no
        colour at all and imports nothing from this one. Under `--dist loadfile`
        which files share a worker depends on the runner's CPU count, so the
        failure is not even reproducible on the machine that caused it.

        So the rule for the unit lane is unconditional: force the terminal and you
        pin the colour system, whatever you assert.
        """
        offenders: list[str] = []
        for path, rel in self._lane_files():
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call):
                    continue
                if getattr(node.func, "id", getattr(node.func, "attr", "")) != "Console":
                    continue
                kw = {k.arg: k.value for k in node.keywords if k.arg}
                forced = isinstance(kw.get("force_terminal"), ast.Constant) and kw["force_terminal"].value is True
                if forced and "color_system" not in kw:
                    offenders.append(f"{rel}:{node.lineno}")
        assert not offenders, (
            "Console(force_terminal=True) with no color_system in the unit lane. Rich memoises "
            "Styles globally and caches the rendered escape on them, so an 8-colour render here "
            'poisons every truecolor assertion in the same worker: add color_system="truecolor". '
            f"{offenders}"
        )

    def test_rgb_asserting_files_pin_color_system(self):
        offenders: list[str] = []
        for path in sorted(self.TESTS_ROOT.rglob("test_*.py")):
            source = path.read_text()
            if not self.RGB_FRAGMENT.search(source):
                continue  # asserts glyphs or layout, not colour — anything goes
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.Call):
                    continue
                if getattr(node.func, "id", getattr(node.func, "attr", "")) != "Console":
                    continue
                kw = {k.arg: k.value for k in node.keywords if k.arg}
                # force_terminal=False emits no escapes at all, so there is
                # nothing for a colour system to downgrade — only True matters.
                forced = isinstance(kw.get("force_terminal"), ast.Constant) and kw["force_terminal"].value is True
                if forced and "color_system" not in kw:
                    rel = path.relative_to(self.TESTS_ROOT).as_posix()
                    offenders.append(f"{rel}:{node.lineno}")
        assert not offenders, (
            "Console(force_terminal=True) with no color_system in a file that asserts "
            'rgb SGR fragments — add color_system="truecolor" so the render does not '
            f"depend on COLORTERM being set (it is unset on CI): {offenders}"
        )
