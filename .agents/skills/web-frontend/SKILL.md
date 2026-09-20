---
name: web-frontend
description: Browser-surface conventions — the yeaboi-web-assets boundary, the contracts/web/ artifacts the front-end repo vendors, the assets.py/brand.py/security.py split, CSP and export inertness, payload rules, and the two Python/TS wire guards. Use when touching src/yeaboi/web/, contracts/web/, any exporter, or a share/board surface.
---

# Web Front End (`yeaboi-web-assets`, built in **yeaboi-frontend**)

Every browser-facing page — the retro and poker live boards, the share gate, the reporting slide
deck, the ship board, and the static HTML exports — is built from TypeScript in
[yeaboi-frontend](https://github.com/yeaboi-ai/yeaboi-frontend) with Vite. **No bundle is edited or
built in this repo.**

They arrive as the **`yeaboi-web-assets`** wheel, an ordinary hard dependency in `pyproject.toml`.
That is what still lets `pip install yeaboi` work with no Node and keeps `make test` pytest-only.

## What lives where

| Here | There |
|---|---|
| `web/assets.py`, `brand.py`, `security.py` — the whole Python boundary | every `.tsx`, `.css`, the Vite config, the bundles |
| `contracts/web/` — generated, vendored by that repo | `src/types/enums.ts`, rendered from `enums.json` |
| the favicon (`web/favicon.png`) | the duck sprites (`src/assets/duck/`) |
| `test_web_contracts.py`, `test_web_assets.py`, `test_web_wire_shapes.py`, `test_web_request_keys.py` | the CSP/eval/`var(--x)`/breakpoint guards, the sprite geometry, the accent and throttle halves |

**Changed something the browser depends on?** `make web-types`, commit `contracts/web/`, then
`make contracts-sync` in that checkout. Its CI runs the other half of every check.

## Self-contained bundles

Bundles must stay self-contained: no CDN, no external `<link>`, no `eval`/`new Function`, no
dynamic `import()`, classic IIFE not ESM. Exports open over `file://` (where a `type="module"`
script does not execute at all) and tunnel pages run under a strict CSP.

`tests/unit/test_web_assets.py` enforces all of this **against the installed package** — so what is
checked is what actually shipped, not what some tree happened to build. The front-end repo asserts
the same rules on its own `dist/` before publishing, one release earlier. Two carve-outs, both
narrow and both asserted rather than assumed: every document carries one `<link rel="icon">` whose
href is a `data:` URI (use `assert_self_contained()` from `tests/_pages.py`, which allows exactly
that one), and the footer credit is an `<a>` to the project site — a place to go, not something the
page loads, exempted by blanking a *single* occurrence so a second one still fails.

## The Python boundary

Python reaches the bundles only through `src/yeaboi/web/assets.py` (`read_asset`, `json_island`,
`render_page`). Never read a bundle by path — where they live is *resolved*, not fixed.
`_static_dir()` tries `$YEABOI_WEB_STATIC` (a sibling yeaboi-frontend build, for developing the
front end against a running board), then the installed `yeaboi_web_assets`. Both failures raise:
a set-but-wrong override would otherwise look like the front end simply not rebuilding, and a
missing package is a broken install with nothing left to fall back to. Resolution happens once at
import, so a rebuild needs a restart. The favicon does **not** follow that path: it is not Vite
output, `gen_duck_sprites.py` renders it from the website's duck art, and it stays package data of
`yeaboi`.

Two sibling leaf modules own the other halves
of that boundary, and a surface that re-implements either is the drift this layout exists to stop:
**`web/brand.py`** is the only place that builds a masthead payload (`build_chrome`), spells the
terminal-frame title (`frame_title`), maps a mode to its accent (`accent_mode` — including
`roadmap → planning` and `anonymize → analysis`) or names the byline (`DEFAULT_FOOTER`); and
**`web/security.py`** is the only place a served document's headers and CSPs come from
(`send_document`, `DOCUMENT_HEADERS`, `BOARD_CSP`/`GATE_CSP`/`ARTIFACT_CSP`/`EDIT_CSP`). No
request handler writes its own headers.

## Exports are inert unless a server is behind them

`ARTIFACT_CSP` sets `connect-src 'none'`, so a written file or a finished share physically cannot
make a request. Two shares talk back, and both are served under the *same* `EDIT_CSP` — identical
to `ARTIFACT_CSP` but for `connect-src 'self'`, pinned by a test that diffs the two policies whole:
an **editable** artifact (`OutputShareServer(editable=…)`), and a **correctable** standup, whose
reader answers a practice signal (`ShareDocument.corrections`, set only when the TUI passes
`session_id`+`run_id`). One policy rather than one each, because they differ in what they send and
not at all in what they may reach.

**The correctable half currently has no host**: both standup share paths went editable, and one
document cannot have two writers — an editable share replays its own edit log, a practice vote
rewrites the run beneath it. The path, its route and its tests stay because carrying a verdict
*through* the edit log (a third op beside `OP_NOTE`/`OP_FIELD`) is what would let both live on one
document; signals are answered from the TUI's Practices action until then.

`export/actions.ts` (edits) and `export/vote.ts` (verdicts) are the only network code in the export
bundle; gate any new control on the payload's capability flag — `edit`, `correctable` — or written
exports render a button that does nothing. Post via `mutate('/api/…', {…})` with a literal path and
body, and read via `payload.get("…")`, so `test_web_request_keys.py` keeps seeing the route.

## Payloads

- Server-validated tuples (grids, statuses, emojis, avatars, deck values) come from
  the front end's generated `types/enums.ts` — see *Generated artifacts* below. **Never also ship them in a boot
  payload**: the island would win at runtime, so a stale bundle would render a board that disagrees
  with what the server accepts. Payloads carry only what a codegen cannot pin.
- **Every browser surface is React** — both live boards, the reporting slide deck, the share gate,
  and all ten static exports. Their Python files are the shell plus a boot island, and no Python
  generates markup or a stylesheet any more: `html_theme` kept `escape`, `safe_url`, the
  trend-series normalisation, image embedding and `export_page`, and lost `EXPORT_CSS`, `html_page`
  and the dozen markup primitives. An exporter builds a payload of text and numbers; a component
  draws it.
- **No markup crosses the wire, and no presentation either** — the payload sends the word
  (`"high"`, `"done"`) or the number, never the colour. The one documented exception is the team
  profile's `Cell.tone`, because its thresholds are per-column *and* directional (80% completion is
  good, 80% spillover is not); it is still a word, and `Profile.tsx` gates it against `TONES`
  before it reaches a `var(--…)`.

## Generated artifacts (`contracts/web/`)

Three artifacts are written here and read by TypeScript there. They live in `contracts/web/` so the
front-end repo vendors them by sha (`make contracts-sync`, pinned in its `.contracts-rev`) instead
of importing Python that is not there.

- **`enums.json`** — `scripts/gen_web_types.py` writes the data (names, values, docs); that repo's
  `scripts/gen-enums.mjs` renders `src/types/enums.ts` from it. Every TypeScript decision — `as
  const`, how a type alias is spelled, JSDoc layout — belongs to the `.mjs`; the JSON is data.
  Lookup tables travel as `[key, value]` **pairs**, not objects, because JS hoists integer-like
  keys and `BLOCK_GLYPHS` would come back with its digits first.
- **`ui.json`** — `scripts/gen_web_ui_contract.py` writes the accents `brand.py` names and the
  server-side timings the browser must respect. Two guards used to read `.css` and `.ts` from
  Python; now the values travel and the assertions are made on the side that owns the file. Each
  end asserts the other still carries what it compares against, so neither half can quietly lapse.
- **`fixtures/`** — the wire snapshots, written by `test_web_wire_shapes.py`. See below.

**`make web-types` regenerates the first two.** The check is split across two repos and neither
alone is enough: `tests/unit/test_web_contracts.py` asserts the contracts are fresh (in `ALWAYS`,
so nothing can dodge it), and the front end's CI runs `gen-enums --check` plus its own assertions.
That file is deliberately separate from anything that reads front-end sources — the module these
checks used to live in skipped itself whole when `frontend/` was absent, which was fine while it
was a sibling directory and silently fatal the moment it was not.

## The two wire guards

One per direction. `test_web_wire_shapes.py` drives real boards through a real round, writes the
snapshots to `contracts/web/fixtures/`, and the front end's `wire.ts` asserts each one `satisfies`
its interface in `types/board.ts` — so a dropped response field fails its `npm run typecheck`.
`test_web_request_keys.py` parses the request bodies out of each `actions.ts` and requires every key
to be one the handler reads — that direction fails *silently* (`payload.get(key, default)` just
returns the default), which is how a 60-second duel turn became 90 with nothing reported. The deck's
payload rides the same response-direction guard — an export is a file, so a dropped field surfaces
months later as a blank slide with no server and no log to look at.
