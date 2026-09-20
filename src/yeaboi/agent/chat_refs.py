"""What a turn points at, rendered for the model — the refs and files behind the composer's ``@`` and ``+``.

A reference is ``{kind, label, id?, mode?, source?, subject?, url?}``: another
plan, another mode's run, an item from an integration, or a link. It rides
the turn the way an image does — a ``[ref #N]`` chip in the text, the whole
list beside it, the surviving chips deciding which travel — and is resolved
here into a short bounded block of text the latest human message carries at
invoke time. Files (``[file #N]``) are read and inlined the same way. Stored
history keeps the person's own words; see ``agent/nodes.py:_attach_chat_context``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

REF_CHIP_RE = re.compile(r"\[ref #(\d+)\]")
REF_KINDS: tuple[str, ...] = ("plan", "run", "integration", "link")

MAX_REFS = 6
MAX_REF_LABEL = 120
MAX_REF_CHARS = 1_500
MAX_REFS_CHARS = 6_000
MAX_FILE_CHARS = 12_000
MAX_FILES_CHARS = 30_000

GONE = "(no longer available)"
_TRUNCATED = "… [truncated]"


def ref_chip_text(index: int) -> str:
    """The placeholder chip for the ``index``-th reference (1-based)."""
    return f"[ref #{index}]"


def referenced_refs(text: str, refs: list[dict]) -> list[dict]:
    """The refs whose ``[ref #N]`` chip survives in ``text`` — the images rule, for references."""
    if not refs:
        return []
    indices = {int(m) for m in REF_CHIP_RE.findall(text)}
    return [ref for i, ref in enumerate(refs, start=1) if i in indices]


def validate_ref(raw: object) -> dict:
    """One reference, checked and normalised; ``ValueError`` names what is wrong."""
    from yeaboi.references import SOURCES
    from yeaboi.sessions_recent import MODES

    if not isinstance(raw, Mapping):
        raise ValueError("a ref must be an object")
    kind = str(raw.get("kind", "") or "").strip().lower()
    if kind not in REF_KINDS:
        raise ValueError(f"unknown ref kind {kind!r} — one of {', '.join(REF_KINDS)}")
    ref: dict = {"kind": kind}
    for key in ("label", "id", "mode", "source", "subject", "url"):
        value = raw.get(key, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise ValueError(f"ref {key} must be a string")
        ref[key] = " ".join(value.split()) if key == "label" else value.strip()
    if len(ref["label"]) > MAX_REF_LABEL:
        raise ValueError(f"a ref label is at most {MAX_REF_LABEL} characters")
    if kind == "plan" and not ref["id"]:
        raise ValueError("a plan ref needs an id")
    if kind == "run":
        ref["mode"] = ref["mode"].lower()
        if ref["mode"] not in MODES:
            raise ValueError(f"unknown ref mode {ref['mode']!r} — one of {', '.join(MODES)}")
    if kind == "integration":
        ref["source"] = ref["source"].lower()
        if ref["source"] not in SOURCES:
            raise ValueError(f"unknown ref source {ref['source']!r} — one of {', '.join(SOURCES)}")
        if not (ref["id"] or ref["subject"]):
            raise ValueError("an integration ref needs an id or a subject")
    if kind == "link" and not re.match(r"^https?://\S+$", ref["url"], re.IGNORECASE):
        raise ValueError("a link ref needs an http(s) url")
    return ref


def validate_refs(raw: object) -> list[dict]:
    """A body's ``refs`` list, checked; an absent or null list is empty."""
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        raise ValueError("refs must be a list")
    if len(raw) > MAX_REFS:
        raise ValueError(f"at most {MAX_REFS} refs on a turn")
    return [validate_ref(item) for item in raw]


# ------------------------------------------------------------- resolution


def resolve_ref(ref: Mapping, *, db_path: Path | None = None) -> str:
    """The text the model sees for one reference; never raises, a lost target reads as gone."""
    kind = ref.get("kind", "")
    try:
        if kind == "plan" or (kind == "run" and ref.get("mode") == "planning" and ref.get("id")):
            text = _resolve_plan(str(ref.get("id", "")), db_path=db_path)
        elif kind == "run":
            text = _resolve_run(str(ref.get("mode", "")), str(ref.get("id", "")), db_path=db_path)
        elif kind == "integration":
            text = _resolve_integration(ref)
        else:
            text = f"{ref.get('label') or ref.get('url')} — {ref.get('url', '')}".strip(" —")
    except Exception:  # noqa: BLE001 — one unreadable target must not stop the turn
        logger.warning("chat refs: %s ref %r could not be resolved", kind, ref.get("label"), exc_info=True)
        text = ""
    if not text:
        logger.warning("chat refs: %s ref %r resolved to nothing", kind, ref.get("label"))
        return f"{ref.get('label') or kind} {GONE}"
    logger.info("chat refs: resolved %s ref %r (%d chars)", kind, ref.get("label"), len(text))
    return _clip(text, MAX_REF_CHARS)


def _resolve_plan(session_id: str, *, db_path: Path | None) -> str:
    from yeaboi.agent.plan_view import plan_view
    from yeaboi.paths import get_db_path
    from yeaboi.sessions import SessionStore

    path = Path(db_path or get_db_path())
    if not path.exists():
        return ""
    with SessionStore(path) as store:
        meta = store.get_session(session_id)
        state = store.load_state(session_id)
    if not state:
        return ""
    view = plan_view(state)
    analysis = view.get("analysis") or {}
    meta = meta or {}
    title = meta.get("title") or analysis.get("name") or meta.get("project_name") or session_id
    lines = [f"Plan: {title} (stage {view.get('stage', '')})"]
    if analysis.get("description"):
        lines.append(f"About: {analysis['description']}")
    if analysis.get("goals"):
        lines.append("Goals: " + "; ".join(str(g) for g in analysis["goals"][:5]))
    counts = view.get("counts") or {}
    if any(counts.values()):
        lines.append(", ".join(f"{n} {k}" for k, n in counts.items() if n))
    for sprint in (view.get("sprints") or [])[:6]:
        goal = sprint.get("goal", "") if isinstance(sprint, Mapping) else ""
        name = sprint.get("name", "") if isinstance(sprint, Mapping) else ""
        if name or goal:
            lines.append(f"- {name}: {goal}".rstrip(": "))
    return "\n".join(lines)


def _resolve_run(mode: str, run_id: str, *, db_path: Path | None) -> str:
    from yeaboi.sessions_recent import recent_sessions

    rows = recent_sessions(mode=mode, limit=0 if run_id else 1, db_path=db_path)
    if run_id:
        rows = [r for r in rows if run_id in (r.run_id, r.session_id)]
    if not rows:
        return ""
    row = rows[0]
    lines = [f"{mode.capitalize()} run: {row.title}"]
    if row.subtitle:
        lines.append(row.subtitle)
    when = (row.created_at or row.last_modified)[:10]
    if when:
        lines.append(f"On: {when}")
    if row.project_label:
        lines.append(f"Project: {row.project_label}")
    if row.tags:
        lines.append("Tags: " + ", ".join(row.tags))
    return "\n".join(lines)


def _resolve_integration(ref: Mapping) -> str:
    from yeaboi.references import read

    source = str(ref.get("source", ""))
    subject = str(ref.get("subject", "")) or str(ref.get("id", "")).partition(":")[2]
    sheet = read(source, subject or str(ref.get("label", "")))
    row = next((r for r in sheet.items if r.id == ref.get("id") or (subject and r.subject == subject)), None)
    if row is None:
        if sheet.warning:
            logger.warning("chat refs: %s — using the ref's own words", sheet.warning)
        parts = [ref.get("label") or subject, ref.get("url", "")]
        return " — ".join(p for p in parts if p)
    return " — ".join(p for p in (row.label, row.detail, row.url) if p)


# --------------------------------------------------------------- rendering


def render_context_block(
    refs: Iterable[Mapping] = (), files: Iterable[str] = (), *, db_path: Path | None = None
) -> list[str]:
    """The paragraphs a turn's refs and files add to the message the model sees."""
    from yeaboi.input_guardrails import check_prompt_injection

    out: list[str] = []
    spent = 0
    for index, ref in enumerate(refs, start=1):
        text = resolve_ref(ref, db_path=db_path)
        if spent + len(text) > MAX_REFS_CHARS:
            text = _clip(text, max(0, MAX_REFS_CHARS - spent))
        spent += len(text)
        if text:
            out.append(f"Reference {index} ({ref.get('kind', '')}): {text}")
    spent = 0
    for index, path in enumerate(files, start=1):
        name = Path(path).name
        try:
            body = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.warning("chat refs: file %s could not be read", name)
            out.append(f"File {index} ({name}): could not be read")
            continue
        body = _clip(body.strip(), min(MAX_FILE_CHARS, max(0, MAX_FILES_CHARS - spent)))
        spent += len(body)
        out.append(f"File {index} ({name}):\n{body}")
    for paragraph in out:
        if check_prompt_injection(paragraph):
            logger.warning("chat refs: a reference or file looks like a prompt injection — passed through")
            break
    logger.info("chat refs: rendered %d paragraph(s)", len(out))
    return out


def _clip(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    return text if len(text) <= limit else text[: max(0, limit - len(_TRUNCATED))] + _TRUNCATED
