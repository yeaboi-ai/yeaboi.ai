"""Documentation quality — is the team's written knowledge clear, and how does AI show up in it?

# See README: "Architecture" — engines are UI-free pipelines; this is a sub-analysis
# of team-analysis mode (AGENTS.md "REQUIRED: Surface Parity" — the TUI/CLI/MCP are
# thin adapters over ``analysis/engine.py:run_team_analysis``, which calls into here).

What this does
--------------
Reads every recently changed page in configured Notion & Confluence containers
(pairing the ``*_recent_pages`` metadata helpers with the never-raise
``*_read_page_text`` body readers), then per page computes:

- a **clarity score** (deterministic, readability-based; 0–100, higher = clearer), and
- a **usefulness score** (purpose, structure, ownership, and actionability), plus
- an **explicit AI-marker** check (a pasted "Generated with Claude" style disclosure).

It aggregates those into a :class:`DocQualitySignal` and coaches the lead on writing
clearer docs and using AI effectively in them (start / stop / keep / try).

Honesty contract
----------------
Clarity and usefulness are heuristics, not objective truth. We do not guess
whether prose was AI-authored. Explicit AI disclosures remain a genuine lower
bound and are rendered separately.

Error contract
--------------
Everything here is best-effort and NEVER raises: a missing SDK/credential or a
failing platform contributes zero and is recorded as a coverage gap.
``run_doc_quality`` wraps the whole thing so the analysis pipeline can call it
unguarded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from yeaboi.analysis.ai_usage import _classify_ai_markers
from yeaboi.team_profile import DocQualitySignal

logger = logging.getLogger(__name__)

# A page counts as AI-assisted only when a marker appears in an AUTHORSHIP
# context ("Generated with…", a Co-Authored-By line) — a page that merely
# documents AI tooling (pasting copilot@github.com or an anthropic URL as an
# example) is *about* AI, not written by it.
_AI_DISCLOSURE_CONTEXT = re.compile(
    r"\b(generated|written|drafted|created|produced|co-authored)\s+(with|by)\b|co-authored-by:",
    re.IGNORECASE,
)


# Below this many pages the clarity/usefulness averages are examples, not a trend.
_MIN_DOC_SAMPLE = 5


def doc_small_sample(signal: DocQualitySignal) -> bool:
    """True when too few pages were scanned for the averages to be a trend.

    Shared by every surface (TUI, CLI, exporters — Surface Parity) so they agree
    on when to frame the scores as examples rather than a stable estate average.
    """
    return signal.pages_scanned < _MIN_DOC_SAMPLE


def _has_ai_disclosure(text: str) -> bool:
    """True when the page carries an explicit AI-authorship disclosure (lower bound)."""
    if not text:
        return False
    return bool(_classify_ai_markers(text)) and bool(_AI_DISCLOSURE_CONTEXT.search(text))


# Default look-back window. Collectors page through every eligible page in each
# configured container; callers can override this window for a run.
_SCAN_DAYS = 90
_READ_CHARS = 100_000  # explicit safety ceiling; any hit is reported as partial coverage
_DOC_READ_ATTEMPTS = 2
_DOC_READ_WORKERS = {"confluence": 8, "notion": 2}
_DOC_INITIAL_WORKERS = {"confluence": 4, "notion": 2}
_DOC_RETRY_BASE_SECONDS = 0.25
_DOC_CACHE_TASK = "documentation_page_score"
# v3: structure-aware extraction (headings/lists/tables/code fences from the
# Confluence/Notion readers), prose-only Flesch, wider owner detection, and the
# AI-disclosure gate — cached v2 page scores were built from structure-less text
# and must not be reused.
#
# This version is the cache's ONLY invalidation lever: the key carries no
# engine version, so any scoring change MUST bump this constant too, or the
# stale rows are served forever.
_DOC_SCORING_VERSION = "deterministic-v3"
_EMPTY_BODY = "empty page body"

# Clarity score bands (0–100, higher = clearer). Aligned to Flesch reading-ease:
# ~60 is "plain English", below ~40 is "difficult".
_CLEAR_MIN = 60.0
_UNCLEAR_MAX = 40.0


def _doc_request_timeout_seconds() -> int:
    from yeaboi.config import get_team_analysis_doc_request_timeout_seconds

    return get_team_analysis_doc_request_timeout_seconds()


def _doc_read_workers(provider: str) -> int:
    from yeaboi.config import get_team_analysis_doc_max_concurrency

    configured = get_team_analysis_doc_max_concurrency()
    return min(_DOC_READ_WORKERS[provider], configured)


def _report_doc_progress(progress: list | None, detail: str) -> None:
    """Update the Docs lifecycle row without creating ambiguous completed steps."""
    if progress is None:
        return
    from yeaboi.analysis.progress import append_component_progress

    append_component_progress(
        progress,
        component_id="docs:documentation",
        label="Assessing documentation quality",
        status="running",
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Per-page heuristics — pure, deterministic, no I/O (the core unit-test seams)
# ---------------------------------------------------------------------------


def _count_syllables(word: str) -> int:
    """Rough syllable count for the Flesch approximation — vowel groups, silent-e trim."""
    groups = re.findall(r"[aeiouy]+", word.lower())
    n = len(groups)
    if word.lower().endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def _clarity_metrics(text: str) -> dict:
    """Deterministic readability metrics for one page.

    Returns a dict with ``word_count``, ``sentence_count``, ``avg_sentence_words``,
    ``long_sentence_pct``, ``heading_count``, ``has_lists`` and a **clarity** score
    (0–100, higher = clearer) from a Flesch reading-ease approximation plus a small
    structure bonus (headings/lists aid a doc's clarity). Pure — no I/O.
    """
    # Score readability on PROSE only: fenced code (see the extractors' ``` fences)
    # is full of long identifiers that tank a Flesch score, so an engineering
    # runbook full of examples would otherwise read as "unclear". Structure markers
    # are also counted on the prose — a "#" comment inside a code fence is not a
    # heading. Having code examples is kept as a neutral signal, not a penalty.
    prose = re.sub(r"(?s)```.*?```", " ", text)
    has_code_blocks = prose != text
    sentences = [s for s in re.split(r"[.!?]+(?:\s|$)", prose) if s.strip()]
    words = re.findall(r"[A-Za-z']+", prose)
    n_sentences = len(sentences)
    n_words = len(words)
    if n_words == 0 or n_sentences == 0:
        return {
            "word_count": n_words,
            "sentence_count": n_sentences,
            "avg_sentence_words": 0.0,
            "long_sentence_pct": 0.0,
            "heading_count": 0,
            "has_lists": False,
            "has_code_blocks": has_code_blocks,
            "clarity": 0.0,
        }

    avg_sentence_words = n_words / n_sentences
    long_sentences = sum(1 for s in sentences if len(s.split()) > 25)
    long_sentence_pct = round(long_sentences / n_sentences * 100, 1)
    syllables = sum(_count_syllables(w) for w in words)

    heading_count = len(re.findall(r"(?m)^\s{0,3}#{1,6}\s", prose))
    has_lists = bool(re.search(r"(?m)^\s*(?:[-*•]|\d+[.)])\s", prose))

    # Flesch Reading Ease — higher = easier to read. Clamp to 0–100.
    flesch = 206.835 - 1.015 * avg_sentence_words - 84.6 * (syllables / n_words)
    clarity = flesch
    # Small structure bonus: a doc with headings/lists reads more clearly than a wall.
    if heading_count:
        clarity += 4
    if has_lists:
        clarity += 3
    clarity = max(0.0, min(100.0, clarity))

    return {
        "word_count": n_words,
        "sentence_count": n_sentences,
        "avg_sentence_words": round(avg_sentence_words, 1),
        "long_sentence_pct": long_sentence_pct,
        "heading_count": heading_count,
        "has_lists": has_lists,
        "has_code_blocks": has_code_blocks,
        "clarity": round(clarity, 1),
    }


def _usefulness_metrics(text: str) -> dict:
    """Measure whether a page is structured, owned, and usable for action."""
    lower = text.lower()
    clarity = _clarity_metrics(text)
    # Owner lines survive extraction in several shapes: "Owner: Jane", a bolded
    # "**Owner** - Jane", or a table row "Owner | Jane" (see _strip_html_tags).
    owned = bool(re.search(r"(?im)^\s*[*_]{0,2}(owner|maintainer|contact|responsible)[*_]{0,2}\s*[:\-|]", text))
    actionable = bool(
        re.search(
            r"\b(run|execute|deploy|rollback|verify|check|decide|decision|next step|procedure|troubleshoot|resolve)\b",
            lower,
        )
    )
    has_purpose = bool(re.search(r"\b(purpose|goal|overview|summary|tl;dr|why)\b", lower))
    structured = bool(clarity["heading_count"] or clarity["has_lists"])
    score = 20.0
    score += 20 if has_purpose else 0
    score += 20 if structured else 0
    score += 20 if actionable else 0
    score += 20 if owned else 0
    return {
        "usefulness": score,
        "owned": owned,
        "actionable": actionable,
        "structured": structured,
        "has_purpose": has_purpose,
    }


def _analyse_page_asset(page: dict) -> dict:
    """Score one page into the complete derived record persisted by the cache."""
    text = str(page.get("text", ""))
    clarity = _clarity_metrics(text)
    useful = _usefulness_metrics(text)
    return {
        "title": str(page.get("title", "Untitled"))[:80],
        "platform": page.get("platform", ""),
        "clarity": clarity["clarity"],
        "usefulness": useful["usefulness"],
        "owned": useful["owned"],
        "actionable": useful["actionable"],
        "structured": useful["structured"],
        "has_code_blocks": clarity["has_code_blocks"],
        "marked": _has_ai_disclosure(text),
        "url": page.get("url", ""),
        "key": page.get("key", ""),
        "container": page.get("container", ""),
        "version": page.get("version") or page.get("timestamp", ""),
    }


def _aggregate_doc_assets(assets: list[dict]) -> DocQualitySignal:
    """Aggregate fresh and version-matched cached assets without needing bodies."""
    if not assets:
        return DocQualitySignal()
    platforms: list[str] = []
    per_platform: dict[str, int] = {}
    clear = mixed = unclear = owned = actionable = structured = ai_marked = 0
    scored: list[tuple[float, float, str]] = []
    for asset in assets:
        platform = str(asset.get("platform", ""))
        if platform and platform not in platforms:
            platforms.append(platform)
        per_platform[platform] = per_platform.get(platform, 0) + 1
        clarity = float(asset.get("clarity", 0))
        usefulness = float(asset.get("usefulness", 0))
        clear += int(clarity >= _CLEAR_MIN)
        unclear += int(clarity < _UNCLEAR_MAX)
        mixed += int(_UNCLEAR_MAX <= clarity < _CLEAR_MIN)
        owned += int(bool(asset.get("owned")))
        actionable += int(bool(asset.get("actionable")))
        structured += int(bool(asset.get("structured")))
        ai_marked += int(bool(asset.get("marked")))
        scored.append((clarity, usefulness, str(asset.get("title", "Untitled"))))
    flagged: list[tuple[str, str]] = []
    seen: set[str] = set()
    for clarity, _usefulness, title in sorted(scored, key=lambda item: item[0]):
        if clarity < _CLEAR_MIN and title not in seen:
            flagged.append((title, f"clarity {clarity:.0f}/100 — dense or long-winded"))
            seen.add(title)
    for _clarity, usefulness, title in sorted(scored, key=lambda item: item[1]):
        if usefulness < 60 and title not in seen:
            flagged.append((title, f"usefulness {usefulness:.0f}/100 — missing purpose, ownership, or actions"))
            seen.add(title)

    def _sorted_pairs(values: dict[str, int]) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(values.items(), key=lambda item: (-item[1], item[0])))

    return DocQualitySignal(
        pages_scanned=len(assets),
        platforms_scanned=tuple(platforms),
        avg_clarity=round(sum(float(a.get("clarity", 0)) for a in assets) / len(assets), 1),
        avg_usefulness=round(sum(float(a.get("usefulness", 0)) for a in assets) / len(assets), 1),
        clear_pages=clear,
        mixed_pages=mixed,
        unclear_pages=unclear,
        owned_pages=owned,
        actionable_pages=actionable,
        structured_pages=structured,
        ai_marked_pages=ai_marked,
        per_platform=_sorted_pairs(per_platform),
        flagged_pages=tuple(flagged),
        is_ai_estimate=False,
    )


def _doc_cache_key(meta: dict) -> str:
    version = str(meta.get("version") or meta.get("timestamp", "")).strip()
    if not version:
        return ""
    raw = json.dumps(
        {
            "provider": meta.get("platform", ""),
            "container": meta.get("container", ""),
            "page_id": meta.get("key", ""),
            "version": version,
            "scoring": _DOC_SCORING_VERSION,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _write_doc_cache(scoreable: list[dict], assets: list[dict], db_path) -> None:
    """Persist the freshly scored assets, one cache row per cache-miss page.

    ``scoreable``/``assets`` are the seam's aligned in/out lists. This replaces
    the old in-worker checkpoint write, so it is one batch after scoring; cached
    entries hold only derived assets — never page bodies. Best-effort, like
    every other touch of this cache.
    """
    if db_path is None:
        return
    # A misaligned pair would write asset i+1 under page i's cache key and
    # serve that wrong score until _DOC_SCORING_VERSION bumps — but a raise
    # here would escape into run_doc_quality's blanket handler and zero the
    # whole Documentation component, so a mismatch skips the write instead.
    if len(scoreable) != len(assets):
        logger.warning(
            "Documentation cache write skipped — %d pages vs %d assets",
            len(scoreable),
            len(assets),
        )
        return
    fresh = [
        (page, asset)
        for page, asset in zip(scoreable, assets, strict=False)
        if not isinstance(page.get("asset"), dict) and isinstance(asset, dict)
    ]
    if not fresh:
        return
    from yeaboi.team_profile import TeamProfileStore

    try:
        store = TeamProfileStore(Path(db_path))
    except Exception:
        logger.debug("Documentation cache store unavailable", exc_info=True)
        return
    try:
        for page, asset in fresh:
            cache_key = _doc_cache_key(page)
            if not cache_key:
                continue
            try:
                store.save_analysis_enrichment(_DOC_CACHE_TASK, cache_key, _DOC_SCORING_VERSION, {"asset": asset})
            except Exception:
                logger.debug("Documentation cache write failed", exc_info=True)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Data gathering — graceful, best-effort fan-out (mirrors analysis/ai_usage.py)
# ---------------------------------------------------------------------------


def _discover_confluence_pages(
    space_key: str = "",
    days: int = _SCAN_DAYS,
    progress_callback=None,
) -> list[dict]:
    """Discover changed Confluence pages without Standup's editor-history calls."""
    from yeaboi.tools.confluence import confluence_recent_pages

    try:
        return confluence_recent_pages(
            space_key=space_key,
            days=days,
            include_version_history=False,
            request_timeout_seconds=_doc_request_timeout_seconds(),
            count_first=True,
            progress_callback=progress_callback,
            raise_on_error=True,
            return_metadata=True,
        )
    except TypeError:
        # Compatibility with older/mocked connector signatures.
        try:
            return confluence_recent_pages(space_key=space_key, days=days)
        except TypeError:
            return confluence_recent_pages(days=days)


def _discover_notion_pages(
    root_id: str = "",
    days: int = _SCAN_DAYS,
    progress_callback=None,
) -> list[dict]:
    """Discover changed Notion pages before any page bodies are read."""
    from yeaboi.tools.notion import notion_recent_pages

    try:
        return notion_recent_pages(
            root_id=root_id,
            days=days,
            request_timeout_seconds=_doc_request_timeout_seconds(),
            progress_callback=progress_callback,
            raise_on_error=True,
        )
    except TypeError:
        return notion_recent_pages(days=days)


def _provider_readers() -> dict[str, object]:
    """Build provider readers with Analysis-specific request deadlines."""
    from yeaboi.tools.confluence import _make_confluence_client, confluence_read_page_text
    from yeaboi.tools.notion import _make_notion_client, notion_read_page_text

    local = threading.local()

    def _confluence(page_id: str):
        client = getattr(local, "confluence_client", None)
        if client is None:
            try:
                client = _make_confluence_client(_doc_request_timeout_seconds())
            except TypeError:
                client = _make_confluence_client()
            local.confluence_client = client
        try:
            return confluence_read_page_text(
                page_id=page_id,
                max_chars=_READ_CHARS,
                request_timeout_seconds=_doc_request_timeout_seconds(),
                _client=client,
            )
        except TypeError:
            return confluence_read_page_text(page_id=page_id, max_chars=_READ_CHARS)

    def _notion(page_id: str):
        client = getattr(local, "notion_client", None)
        if client is None:
            try:
                client = _make_notion_client(_doc_request_timeout_seconds())
            except TypeError:
                client = _make_notion_client()
            local.notion_client = client
        try:
            return notion_read_page_text(
                page_id,
                max_chars=_READ_CHARS,
                request_timeout_seconds=_doc_request_timeout_seconds(),
                _client=client,
            )
        except TypeError:
            return notion_read_page_text(page_id, max_chars=_READ_CHARS)

    return {"confluence": _confluence, "notion": _notion}


def _is_transient_doc_error(error: str) -> bool:
    text = error.lower()
    return any(
        marker in text
        for marker in (
            "429",
            "rate limit",
            "timeout",
            "timed out",
            "connection",
            "temporar",
            "500",
            "502",
            "503",
            "504",
        )
    )


def _read_one_page(meta: dict, reader) -> dict:
    """Read one discovered page, retrying only bounded transient failures."""
    doc = None
    for attempt in range(_DOC_READ_ATTEMPTS):
        try:
            doc = reader(str(meta.get("key", "")))
        except Exception as exc:  # a connector seam may raise despite its contract
            doc = {"text": "", "truncated": False, "error": str(exc)}
        error = str(doc.get("error", "")) if isinstance(doc, dict) else "invalid page response"
        if not error or attempt + 1 >= _DOC_READ_ATTEMPTS or not _is_transient_doc_error(error):
            break
        retry_after = doc.get("retry_after") if isinstance(doc, dict) else None
        try:
            delay = max(0.0, min(float(retry_after), 5.0))
        except (TypeError, ValueError):
            delay = _DOC_RETRY_BASE_SECONDS * (2**attempt)
        if delay:
            time.sleep(delay)

    text = str(doc.get("text", "")) if isinstance(doc, dict) else ""
    read_error = str(doc.get("error", "")) if isinstance(doc, dict) else "invalid page response"
    if not text.strip() and not read_error:
        read_error = _EMPTY_BODY
    return {
        **meta,
        "title": meta.get("title") or (doc.get("title", "") if isinstance(doc, dict) else "") or "Untitled",
        "text": text,
        "truncated": bool(doc.get("truncated", False)) if isinstance(doc, dict) else False,
        "read_error": read_error,
    }


class _AdaptiveDocGate:
    """Provider-local concurrency gate that backs off after transient failures."""

    def __init__(self, initial: int, maximum: int) -> None:
        self._limit = max(1, initial)
        self._maximum = max(self._limit, maximum)
        self._active = 0
        self._success_streak = 0
        self._condition = threading.Condition()

    def acquire(self) -> None:
        with self._condition:
            while self._active >= self._limit:
                self._condition.wait()
            self._active += 1

    def release(self, *, transient_failure: bool) -> None:
        with self._condition:
            self._active -= 1
            if transient_failure:
                self._limit = max(1, self._limit // 2)
                self._success_streak = 0
            else:
                self._success_streak += 1
                if self._success_streak >= self._limit and self._limit < self._maximum:
                    self._limit += 1
                    self._success_streak = 0
            self._condition.notify_all()


def _read_page_inventory(
    inventory: list[dict],
    readers: dict[str, object],
    progress: list[str] | None = None,
    db_path=None,
    *,
    _return_metrics: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    """Read every inventory item through bounded provider-specific worker pools.

    Provider groups run concurrently, but results are rebuilt in discovery order.
    """
    if not inventory:
        empty_metrics = {"cache_lookup_seconds": 0.0, "body_read_seconds": 0.0}
        return ([], empty_metrics) if _return_metrics else []

    cache_started = time.monotonic()
    results: list[dict | None] = [None] * len(inventory)
    cache_store = None
    cache_hits = 0
    if db_path is not None:
        from yeaboi.team_profile import TeamProfileStore

        cache_store = TeamProfileStore(Path(db_path))
        for index, meta in enumerate(inventory):
            cache_key = _doc_cache_key(meta)
            if not cache_key:
                continue
            try:
                cached = cache_store.load_analysis_enrichment(_DOC_CACHE_TASK, cache_key, _DOC_SCORING_VERSION)
            except Exception:
                logger.debug("Documentation cache lookup failed", exc_info=True)
                cached = None
            if isinstance(cached, dict) and isinstance(cached.get("asset"), dict):
                results[index] = {
                    **meta,
                    "asset": cached["asset"],
                    "cache_status": "hit",
                    "text": "",
                    "truncated": False,
                    "read_error": "",
                }
                cache_hits += 1
    cache_lookup_seconds = time.monotonic() - cache_started

    provider_indices: dict[str, list[int]] = {}
    for index, meta in enumerate(inventory):
        if results[index] is not None:
            continue
        provider_indices.setdefault(str(meta.get("platform", "")), []).append(index)

    total = len(inventory)
    completed = cache_hits
    failed = 0
    counter_lock = threading.Lock()
    _report_doc_progress(
        progress,
        f"Documentation cache: {cache_hits} unchanged · {total - cache_hits} bodies to fetch",
    )

    def _read_provider(provider: str, indices: list[int]) -> None:
        nonlocal completed, failed
        reader = readers[provider]
        workers = min(_doc_read_workers(provider), len(indices))
        gate = _AdaptiveDocGate(min(_DOC_INITIAL_WORKERS[provider], workers), workers)

        def _gated_read(index: int) -> dict:
            # Reading only — scoring happens in one batch behind the
            # ``analysis.score_docs`` seam after every body is in (see
            # ``run_doc_quality``), which also moved the score-cache write
            # there: an interrupted run no longer checkpoints partial scores,
            # and a raise while scoring is no longer isolated to one failed
            # page — it fails the whole component via run_doc_quality's guard.
            gate.acquire()
            page: dict = {}
            try:
                page = _read_one_page(inventory[index], reader)
                return page
            finally:
                error = str(page.get("read_error", ""))
                gate.release(transient_failure=bool(error and _is_transient_doc_error(error)))

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"docs-{provider}") as executor:
            futures = {executor.submit(_gated_read, index): index for index in indices}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    page = future.result()
                except Exception as exc:  # defensive; _read_one_page already catches
                    page = {**inventory[index], "text": "", "truncated": False, "read_error": str(exc)}
                results[index] = page
                with counter_lock:
                    completed += 1
                    read_error = str(page.get("read_error", ""))
                    failed += int(bool(read_error and read_error != _EMPTY_BODY))
                    _report_doc_progress(
                        progress,
                        f"Reading documentation: {completed}/{total} · {failed} failed",
                    )

    body_read_started = time.monotonic()
    try:
        groups = [(provider, indices) for provider, indices in provider_indices.items() if provider in readers]
        if groups:
            with ThreadPoolExecutor(max_workers=len(groups), thread_name_prefix="docs-provider") as executor:
                futures = [executor.submit(_read_provider, provider, indices) for provider, indices in groups]
                for future in futures:
                    future.result()
        ordered = [page for page in results if page is not None]
        metrics = {
            "cache_lookup_seconds": round(cache_lookup_seconds, 3),
            "body_read_seconds": round(time.monotonic() - body_read_started, 3),
        }
        return (ordered, metrics) if _return_metrics else ordered
    finally:
        if cache_store is not None:
            cache_store.close()


def collect_doc_pages(
    source: str,
    project_key: str,
    sub_sources: list[str] | None = None,
    *,
    window_days: int = _SCAN_DAYS,
    analysis_scope: dict[str, list[str]] | None = None,
    progress: list[str] | None = None,
    db_path=None,
    _return_coverage: bool = False,
) -> tuple[list[dict], list[str], list[str]] | tuple[list[dict], list[str], list[str], dict]:
    """Fan out over Confluence + Notion for recently-changed pages with their body text.

    Returns ``(pages, platforms_scanned, coverage_notes)``. Every platform is
    best-effort and lazily imported (optional SDKs); a missing credential/SDK or a
    failing platform contributes zero and is added to ``coverage_notes`` so absent
    coverage is visible rather than silent. Never raises. ``source``/``project_key``
    are accepted for signature parity with the other sub-analyses; doc platforms are
    resolved purely from their own config. ``sub_sources`` restricts which platforms
    to read (subset of ``{"confluence", "notion"}``; None = both).
    """
    from yeaboi.analysis.coverage import CoverageTracker, coverage_notes
    from yeaboi.config import (
        get_confluence_base_url,
        get_confluence_token,
        get_notion_token,
        get_team_analysis_confluence_spaces,
        get_team_analysis_notion_roots,
    )

    def _want(tag: str) -> bool:
        return sub_sources is None or tag in sub_sources

    collection_started = time.monotonic()
    inventory: list[dict] = []
    platforms_scanned: list[str] = []
    coverage: list[str] = []
    tracker = CoverageTracker("docs", window_days)
    scope = analysis_scope or {}
    seen_pages: set[tuple[str, str]] = set()

    def _add_discovered(raw: list[dict], provider: str, container: str) -> None:
        for meta in raw:
            page_id = str(meta.get("key", "")).strip()
            if not page_id:
                tracker.add(provider, container, str(meta.get("title", "unknown page")), "failed", "missing page id")
                continue
            identity = (provider, page_id)
            if identity in seen_pages:
                continue
            seen_pages.add(identity)
            inventory.append(
                {
                    "title": meta.get("title", "") or "Untitled",
                    "author": meta.get("author", ""),
                    "url": meta.get("url", ""),
                    "key": page_id,
                    "timestamp": meta.get("timestamp", ""),
                    "version": meta.get("version") or meta.get("timestamp", ""),
                    "platform": provider,
                    "container": container,
                }
            )

    discovery_jobs: list[tuple[str, str, object]] = []

    def _unique(values: list[str]) -> list[str]:
        return list(dict.fromkeys(str(value).strip() for value in values))

    def _discovery_callback(provider: str, container: str):
        label = "Confluence space" if provider == "confluence" else "Notion root"

        def _callback(discovered: int, expected: int | None, batch: int) -> None:
            if expected is not None and batch == 0:
                detail = f"{label} {container}: {expected:,} pages found"
            elif expected is not None:
                batches = max(1, (expected + 99) // 100)
                detail = f"Discovering {label} {container}: {discovered:,}/{expected:,} · batch {batch}/{batches}"
            else:
                detail = f"Discovering {label} {container}: {discovered:,} found · batch {batch}"
            _report_doc_progress(progress, detail)

        return _callback

    def _run_confluence_discovery(space: str, callback) -> list[dict]:
        try:
            return _discover_confluence_pages(space, window_days, callback)
        except TypeError:
            try:
                return _discover_confluence_pages(space, window_days)
            except TypeError:
                return _discover_confluence_pages()

    def _run_notion_discovery(root: str, callback) -> list[dict]:
        try:
            return _discover_notion_pages(root, window_days, callback)
        except TypeError:
            try:
                return _discover_notion_pages(root, window_days)
            except TypeError:
                return _discover_notion_pages()

    # Queue every configured container, then discover containers concurrently.
    if _want("confluence") and get_confluence_token() and get_confluence_base_url():
        spaces = _unique(scope.get("confluence") or list(get_team_analysis_confluence_spaces()) or [""])
        for space in spaces:
            container = space or "all-accessible"
            discovery_jobs.append(
                (
                    "confluence",
                    container,
                    lambda value=space, cb=_discovery_callback("confluence", container): _run_confluence_discovery(
                        value, cb
                    ),
                )
            )
    elif _want("confluence"):
        tracker.add(
            "confluence",
            "unconfigured",
            "page estate",
            "inaccessible",
            "CONFLUENCE_API_TOKEN / base URL not set",
        )

    if _want("notion") and get_notion_token():
        roots = _unique(scope.get("notion") or list(get_team_analysis_notion_roots()) or [""])
        for root in roots:
            container = root or "integration-visible"
            discovery_jobs.append(
                (
                    "notion",
                    container,
                    lambda value=root, cb=_discovery_callback("notion", container): _run_notion_discovery(value, cb),
                )
            )
    elif _want("notion"):
        tracker.add("notion", "unconfigured", "page estate", "inaccessible", "NOTION_TOKEN not set")

    discovery_results: list[tuple[str, str, object | None, Exception | None]] = [
        ("", "", None, None) for _ in discovery_jobs
    ]
    if discovery_jobs:
        with ThreadPoolExecutor(
            max_workers=min(4, len(discovery_jobs)),
            thread_name_prefix="docs-discovery",
        ) as executor:
            futures = {
                executor.submit(discover): index
                for index, (_provider, _container, discover) in enumerate(discovery_jobs)
            }
            for future in as_completed(futures):
                index = futures[future]
                provider, container, _discover = discovery_jobs[index]
                try:
                    discovery_results[index] = (provider, container, future.result(), None)
                except Exception as exc:
                    discovery_results[index] = (provider, container, None, exc)

    expected_inventory = 0
    for provider, container, raw, error in discovery_results:
        if error is not None:
            tracker.add(provider, container, "page estate", "inaccessible", str(error))
            continue
        discovery_complete = bool(getattr(raw, "complete", True))
        discovery_error = str(getattr(raw, "error", "") or "")
        expected_total = getattr(raw, "expected_total", None)
        if isinstance(expected_total, int):
            expected_inventory += expected_total
        raw_pages = getattr(raw, "items", raw)
        if provider not in platforms_scanned:
            platforms_scanned.append(provider)
        if not raw_pages and discovery_complete:
            coverage.append(f"{provider}: no pages changed in the last {window_days} days")
        _add_discovered(raw_pages or [], provider, container)
        if not discovery_complete:
            tracker.add(
                provider,
                container,
                "remaining page estate",
                "inaccessible",
                discovery_error or "page discovery did not complete",
            )

    discovery_seconds = time.monotonic() - collection_started
    _report_doc_progress(progress, f"Discovered {len(inventory)} documentation pages")

    read_started = time.monotonic()
    pages, read_metrics = _read_page_inventory(
        inventory,
        _provider_readers(),
        progress,
        db_path=db_path,
        _return_metrics=True,
    )
    read_seconds = time.monotonic() - read_started
    readable_pages: list[dict] = []
    for page in pages:
        read_error = str(page.get("read_error", ""))
        empty_body = read_error == _EMPTY_BODY
        status = (
            "cached"
            if page.get("cache_status") == "hit"
            else "unchanged"
            if empty_body
            else "failed"
            if read_error
            else "truncated"
            if page.get("truncated")
            else "succeeded"
        )
        tracker.add(
            str(page.get("platform", "")),
            str(page.get("container", "")),
            str(page.get("key") or page.get("title")),
            status,
            read_error or ("page hit safety ceiling" if page.get("truncated") else ""),
            eligible=not empty_body,
        )
        # A page is readable when it carries a version-matched cached asset or a
        # body for the scoring seam — fresh reads are no longer scored in the
        # worker, so "has an asset" alone would drop every cache miss.
        if isinstance(page.get("asset"), dict) or str(page.get("text", "")).strip():
            readable_pages.append(page)

    coverage_blob = tracker.as_dict()
    coverage_blob["stage_timings"] = {
        "discovery_seconds": round(discovery_seconds, 3),
        "read_seconds": round(read_seconds, 3),
        **read_metrics,
    }
    coverage_blob["expected"] = max(len(inventory), expected_inventory)
    coverage_blob["cached"] = sum(page.get("cache_status") == "hit" for page in pages)
    coverage_blob["cache_misses"] = sum(page.get("cache_status") != "hit" for page in pages)
    coverage_blob["bodies_fetched"] = sum(
        page.get("cache_status") != "hit" and bool(str(page.get("text", "")).strip()) for page in pages
    )
    coverage.extend(coverage_notes(coverage_blob))
    if _return_coverage:
        return readable_pages, platforms_scanned, coverage, coverage_blob
    return readable_pages, platforms_scanned, coverage


def _doc_findings(assets: list[dict]) -> list[dict]:
    findings: list[dict] = []
    for asset in assets:
        title = str(asset.get("title", "Untitled"))
        scope = f"{asset.get('platform', '')}:{title}"
        base = {
            "link": asset.get("url", ""),
            "affected_scope": [scope],
            "owner_role": "Documentation owner",
            "confidence": "high",
        }
        if float(asset.get("clarity", 0)) < _CLEAR_MIN:
            findings.append(
                {
                    **base,
                    "id": f"{scope}:clarity",
                    "category": "clarity",
                    "title": "Rewrite dense documentation",
                    "detail": (
                        "Lead with the outcome, shorten sentences, and split the page "
                        "with descriptive headings and lists."
                    ),
                    "priority": "high",
                    "impact": "Makes operational knowledge faster to understand and use.",
                    "evidence": f"{title} scored {asset.get('clarity', 0):.0f}/100 for clarity.",
                    "next_steps": [
                        "Rewrite the summary and longest sections.",
                        "Have a target reader validate the instructions.",
                    ],
                    "effort": "small",
                    "completion_check": (
                        "A target reader can identify the purpose and required action without author help."
                    ),
                }
            )
        if float(asset.get("usefulness", 0)) < 60:
            findings.append(
                {
                    **base,
                    "id": f"{scope}:usefulness",
                    "category": "usefulness",
                    "title": "Add purpose, ownership, and actions",
                    "detail": (
                        "State why the page exists, who maintains it, and the concrete "
                        "procedure or decision it supports."
                    ),
                    "priority": "high",
                    "impact": "Turns descriptive prose into maintainable, actionable team knowledge.",
                    "evidence": f"{title} scored {asset.get('usefulness', 0):.0f}/100 for usefulness.",
                    "next_steps": ["Add purpose and owner fields.", "Add verified steps, decisions, or next actions."],
                    "effort": "small",
                    "completion_check": "The page names an owner and provides a verifiable action or decision.",
                }
            )
    return findings


def _prioritize_doc_actions(findings: list[dict]) -> list[dict]:
    order = {"high": 0, "medium": 1, "low": 2}
    grouped: dict[tuple[str, str], list[dict]] = {}
    for finding in findings:
        grouped.setdefault((str(finding.get("category")), str(finding.get("title"))), []).append(finding)
    actions: list[dict] = []
    for group in grouped.values():
        action = dict(group[0])
        scopes = sorted({scope for item in group for scope in item.get("affected_scope", [])})
        action["affected_scope"] = scopes
        action["breadth"] = len(scopes)
        if len(scopes) > 1:
            action["evidence"] += f" Affects {len(scopes)} pages."
        actions.append(action)
    return sorted(actions, key=lambda a: (order.get(str(a.get("priority")), 9), -int(a.get("breadth", 1))))


def run_doc_quality(
    source: str,
    project_key: str,
    sub_sources: list[str] | None = None,
    *,
    window_days: int = _SCAN_DAYS,
    analysis_scope: dict[str, list[str]] | None = None,
    progress: list[str] | None = None,
    db_path=None,
) -> tuple[DocQualitySignal, dict]:
    """Orchestrate the doc-quality scan: collect recent pages → score → aggregate.

    Returns ``(signal, examples_blob)``. ``examples_blob`` carries the aggregated
    summary, one structured asset per readable page (titles/scores only — never
    bodies), and coverage notes. Wholly best-effort — any failure yields an empty
    signal and a coverage note, never an exception (the pipeline calls this unguarded).
    """
    logger.info("run_doc_quality: source=%s project=%s", source, project_key)
    run_started = time.monotonic()
    try:
        try:
            collected = collect_doc_pages(
                source,
                project_key,
                sub_sources,
                window_days=window_days,
                analysis_scope=analysis_scope,
                progress=progress,
                db_path=db_path,
                _return_coverage=True,
            )
        except TypeError:
            collected = collect_doc_pages(source, project_key, sub_sources)
        if len(collected) == 3:
            pages, platforms_scanned, coverage = collected
            coverage_blob = {
                "component": "docs",
                "status": "complete",
                "window_days": window_days,
                "discovered": len(pages),
                "eligible": len(pages),
                "attempted": len(pages),
                "succeeded": len(pages),
                "failed": 0,
                "unchanged": 0,
                "inaccessible": 0,
                "truncated": 0,
                "per_container": {},
                "assets": [],
            }
        else:
            pages, platforms_scanned, coverage, coverage_blob = collected
        from yeaboi.analysis.aggregate import (
            build_score_docs_inputs,
            doc_signal_from_wire,
            score_docs,
            scoreable_doc_pages,
        )

        scoreable = scoreable_doc_pages(pages)
        _report_doc_progress(progress, f"Assembling quality results for {len(scoreable)} documentation pages")
        score_started = time.monotonic()
        inputs = build_score_docs_inputs(pages=pages)
        scored = score_docs(inputs)
        signal = doc_signal_from_wire(scored["signal"])
        assets = scored["assets"]
        _write_doc_cache(scoreable, assets, db_path)

        samples = assets
        findings = scored["findings"]
        action_plan = scored["action_plan"]
        score_seconds = time.monotonic() - score_started
        stage_timings = {
            "discovery_seconds": 0.0,
            "read_seconds": 0.0,
            **coverage_blob.get("stage_timings", {}),
        }
        stage_timings.update(
            {
                "score_seconds": round(score_seconds, 3),
                "total_seconds": round(time.monotonic() - run_started, 3),
            }
        )
        blob: dict = {
            "summary": scored["summary"],
            "samples": samples,
            "assets": samples,
            "coverage": coverage,
            "coverage_report": coverage_blob,
            "stage_timings": stage_timings,
            "window_days": window_days,
            "findings": findings,
            "action_plan": action_plan,
        }
        # The seam always computes the (deterministic) coaching insights; only a
        # run that actually read pages earns them in the blob — the same gate the
        # team-learning caller used to apply after the fact.
        if signal.pages_scanned > 0 and coverage_blob.get("status") not in {"failed", "no_data"}:
            blob["insights"] = scored["insights"]
        else:
            blob["insights"] = {}
        logger.info(
            "run_doc_quality: pages=%d avg_clarity=%.0f usefulness=%.0f marked=%d platforms=%s",
            signal.pages_scanned,
            signal.avg_clarity,
            signal.avg_usefulness,
            signal.ai_marked_pages,
            ",".join(platforms_scanned) or "none",
        )
        return signal, blob
    except Exception:  # pragma: no cover - collect/aggregate already guard
        logger.exception("run_doc_quality failed; returning empty signal")
        return DocQualitySignal(), {"summary": {}, "samples": [], "coverage": ["doc-quality scan failed"]}


# ---------------------------------------------------------------------------
# Coaching insights — start / stop / keep / try (mirrors ai_usage insights)
# ---------------------------------------------------------------------------


def _fallback_doc_quality_insights(signal: DocQualitySignal, samples: list[dict] | None = None) -> dict:
    """Return deterministic, evidence-linked coaching from all page results."""
    from yeaboi.tools.team_learning import _INSIGHT_MAX_ITEMS, _insight_item

    samples = samples or []
    findings = _doc_findings(samples)
    actions = _prioritize_doc_actions(findings)
    if actions:
        items = []
        for action in actions[:_INSIGHT_MAX_ITEMS]:
            item = _insight_item(
                str(action.get("title", "")),
                str(action.get("detail", "")),
                str(action.get("evidence", "")),
            )
            if action.get("link"):
                item["link"] = action["link"]
            items.append(item)
        return {
            "start": items,
            "stop": [
                _insight_item(
                    "Stop publishing ownerless guidance",
                    "Every operational page should name a maintainer and a concrete validation step.",
                    f"{max(0, signal.pages_scanned - signal.owned_pages)} page(s) lack an owner signal",
                )
            ],
            "keep": [
                _insight_item(
                    "Keep actionable pages current",
                    "Preserve the pages that already combine clear structure with executable guidance.",
                    f"{signal.actionable_pages} actionable page(s) found",
                )
            ],
            "try": [
                _insight_item(
                    "Use a shared documentation template",
                    "Start pages with purpose, owner, last-reviewed date, procedure, and verification.",
                    f"Average usefulness {signal.avg_usefulness:.0f}/100",
                )
            ],
        }
    return {
        "start": [
            _insight_item(
                "Set a documentation quality baseline",
                "Use purpose, owner, procedure, and verification fields for every shared page.",
                f"{signal.pages_scanned} page(s) scanned",
            )
        ],
        "stop": [
            _insight_item(
                "Stop relying on implicit ownership",
                "Name a maintainer so readers know who can verify and update the page.",
                "Ownership is assessed explicitly in the new documentation score.",
            )
        ],
        "keep": [
            _insight_item(
                "Keep clear documentation patterns",
                "Continue using concise sections and concrete procedures.",
                f"Average clarity {signal.avg_clarity:.0f}/100",
            )
        ],
        "try": [
            _insight_item(
                "Review documentation with a target reader",
                "Ask someone other than the author to execute or explain the documented process.",
                f"Average usefulness {signal.avg_usefulness:.0f}/100",
            )
        ],
    }


def generate_doc_quality_insights(signal: DocQualitySignal, examples: dict) -> dict:
    """Compatibility seam for deterministic, evidence-based documentation advice."""
    return _fallback_doc_quality_insights(signal, examples.get("samples", []))
