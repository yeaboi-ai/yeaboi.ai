---
name: tui-standards
description: TUI component standards — shared primitives in ui/shared/_components.py, the mandatory Panel page structure, themes, buttons, scrollbars, viewport math. Use when creating or modifying any TUI screen, any _build_*_screen function, or code under src/yeaboi/ui/.
---

# TUI Component Standards

All TUI screens MUST use the shared component system in `src/yeaboi/ui/shared/_components.py`. Do NOT duplicate rendering logic.

## Shared Primitives (use these, don't rewrite)

| Component | Function | Purpose |
|-----------|----------|---------|
| `Theme` | `ANALYSIS_THEME`, `PLANNING_THEME`, `USAGE_THEME`, `SETTINGS_THEME` | Colour palette per mode (incl. `bg` page tint) |
| Page panel | `build_page_panel(content, theme=, height=)` | The full-screen root Panel every page returns |
| Buttons | `build_action_buttons(actions, selected)` | Consistent button row (Accept/Edit/Export/Back etc.) |
| Scrollbar | `build_scrollbar(viewport_h, total, offset, max_scroll)` | Right-side scroll indicator |
| Progress | `build_progress_dots(stages, current, theme=)` | Stage indicator (● ● ○ ○ ○) |
| Viewport | `calc_viewport(height, header_h=, action_h=)` | Viewport height calculation |
| Titles | `planning_title()`, `analysis_title()`, `usage_title()`, `settings_title()` | ASCII art headers |
| Popup | `build_popup(message, width=, border_style=)` | Confirmation dialogs |
| Badge | `build_badge(label, rgb=BETA_RGB, dim=)` | Inverse-video status chip (BETA, COMING SOON). Takes an rgb **tuple**, not a colour string — `dim` does arithmetic on the channels, and `COLOR_RGB` only knows the mode accents. Never draw a chip with box glyphs: the mode-card click hit-testing finds title rows by scanning for `█▀▄` |
| Padding | `PAD` constant | Left indent for visual balance |

**Beta modes.** A mode whose output isn't verified yet keeps `"available": True`
in `_MODE_CARDS` (that flag gates Enter, the click handler *and* the welcome
screen's `g` jump key — flipping it removes the feature rather than labelling
it) and adds `"badge": BETA_LABEL`. Three markers must move together, and
`test_beta_surfaces.py::TestBetaMarkersAgree` enforces it: the card badge, a
`_BETA_MODES` entry in `ui/shared/_beta_notice.py` (the one-time entry notice),
and `FeatureTip(is_beta=True)`. Where both could render, **BETA beats NEW**.

## Page Structure (every `_build_*_screen` function MUST follow)

```
build_page_panel(content, theme=<MODE_THEME>, height=height)   # NEVER a raw full-screen Panel(...)
  ├── Text("")                    # blank
  ├── title                       # ASCII art from *_title()
  ├── Text("")                    # blank
  ├── subtitle / progress dots    # context line
  ├── Text("")                    # blank
  ├── viewport_renderable         # scrollable content (with optional scrollbar)
  ├── Text("")                    # blank
  ├── btn_top                     # from build_action_buttons()
  ├── btn_mid                     #
  └── btn_bot                     #
```

## Rules

1. **DRY** — Never inline button rendering, scrollbar math, or viewport calculations. Always use shared functions.
2. **Themes** — Never hardcode colour values (`"rgb(100,180,100)"`). Use `theme.accent`, `theme.muted`, etc. from the appropriate Theme constant.
3. **New pages** — Adding a new mode/page requires: a Theme constant (including a dark `bg` tint of the accent hue), a `*_title()` function, a colour entry in `COLOR_RGB`, an entry in `_MODE_CARDS` (if it's a main menu item), and a `FeatureTip` in `ui/shared/_tips.py` keyed by the capability (with `mode_key` = the card key so the welcome-screen `g` key jumps in; `is_new=True` for a release or two), and — if the mode records runs — a **saved-sessions hub** registered in `SAVED_SESSION_HUBS` (`ui/mode_select/__init__.py`), which is what the card lands on. `TestTips` and `TestSavedSessions` enforce this.
4. **Consistency** — All pages use the same Panel structure (title → subtitle → viewport → buttons). No exceptions.
5. **Page background** — Every page paints its own background so all users see the same TUI regardless of terminal theme. `build_page_panel` applies `theme.bg` (a dark tint of the mode accent; `NEUTRAL_BG` for home/setup/settings pages) as the Panel style, and Rich cascades it onto every child segment — content rows, spacers, scroll filler, padding. Never return a raw full-screen `Panel` from a builder; `tests/unit/test_screen_backgrounds.py` fails the build if one appears. `MusicLive._stamp` back-fills `NEUTRAL_BG` on any unstyled Panel as a safety net, and non-Panel surfaces (screensaver, splash) set their own `on NEUTRAL_BG` style. Keep tints dark (channel values ≤ 40) — foreground styles assume dark backgrounds.
6. **Saved sessions** — A mode that records runs MUST land on a saved-sessions hub, not on its own live page: a finished run the user cannot get back to is a run they lost. Build it with `_run_mode_hub(...)` (`ui/mode_select/__init__.py`) — inject `load_runs` → `RunSummary`s, `make_detail` (or `open_snapshot` when the snapshot needs its own buttons), `files_export`, `get_document`, `delete_run`, `run_new` — and register the card key in `SAVED_SESSION_HUBS`. Render the snapshot through the mode's **own** screen builder in a snapshot mode, so a saved run looks like the live view rather than flat text. `tests/unit/test_surface_parity.py::TestSavedSessions` fails the build for a card with neither a hub nor a `SAVED_SESSIONS_EXEMPT` reason.
7. **Scrollbar** — Content that can overflow MUST use `build_scrollbar()`. Use `always_show=True` for pages where the track should always be visible.
8. **Buttons** — Register new button labels in `_BTN_COLORS` dict in `_components.py` with accent/grey colour tuples.
9. **No `_PAD` aliases** — Import `PAD` directly from `yeaboi.ui.shared._components`. Legacy `_PAD = PAD` aliases exist but should not be added to new files.
10. **Never log in per-frame code** — `_build_*_screen` builders and render paths run every frame (~60 fps); `logger.info` belongs in key-handling branches of runner loops and one-shot functions only (see the `logging` skill).
