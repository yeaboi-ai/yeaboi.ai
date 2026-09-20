"""What a run may read from other sessions — the scope, as pure data.

A :class:`ContextScope` names the producer modes a run may read (``sources``),
the timeframe (``window``), the project labels and tags the sessions must
carry, a per-source "newest N" cap (``limits``), and the sessions pinned by
name (``sessions``) — those are always in scope, whatever the rest says.
``None`` is today's unscoped behaviour byte-for-byte; every narrowing is opt-in.

Two twins of the same value: a JSON dict (HTTP bodies, ``ScrumState``) and a
one-line spec string (CLI, MCP). Both round-trip through this module. Nothing
here touches a store — resolution lives in :mod:`yeaboi.context.resolve`.
"""

from __future__ import annotations

import logging
import re
import shlex
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace

logger = logging.getLogger(__name__)

#: The producer modes a run may read, in the order surfaces list them.
SOURCES: tuple[str, ...] = ("plan", "standup", "retro", "poker", "performance", "analysis", "reporting", "review")

SOURCE_LABELS: dict[str, str] = {
    "plan": "Sprint plans",
    "standup": "Standups",
    "retro": "Retros",
    "poker": "Poker sessions",
    "performance": "1:1s and reviews",
    "analysis": "Analysis profiles",
    "reporting": "Delivery reports",
    "review": "Weekly reviews",
}

#: One line per source, for chips and help text.
SOURCE_HINTS: dict[str, str] = {
    "plan": "sprint framing and roster",
    "standup": "blockers, confidence trend and cadence",
    "retro": "action items, themes and carry-over",
    "poker": "agreed estimates",
    "performance": "open 1:1 actions and review focus",
    "analysis": "team calibration and AC style",
    "reporting": "what shipped last period",
    "review": "last week's actions",
}

WINDOW_KINDS: tuple[str, ...] = ("all", "sprints", "month", "quarter", "year", "custom")

WINDOW_LABELS: dict[str, str] = {
    "all": "Everything",
    "sprints": "Last sprints",
    "month": "Last month",
    "quarter": "Last quarter",
    "year": "Last year",
    "custom": "Custom range",
}

#: Source token → the ``mode`` its label rows (and a pinned session) carry.
SOURCE_MODES: dict[str, str] = {name: ("planning" if name == "plan" else name) for name in SOURCES}
#: The inverse: a session's mode → the source it is read under.
MODE_SOURCES: dict[str, str] = {mode: source for source, mode in SOURCE_MODES.items()}

#: The most sessions a scope pins by name.
MAX_PINNED_SESSIONS = 50

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SPRINTS = re.compile(r"^(\d+)\s*sprints?$")
_SOURCE_TOKEN = re.compile(r"^([a-z]+)(?::(\d+))?$")


def _valid_sources_text() -> str:
    return ", ".join(SOURCES)


@dataclass(frozen=True)
class Window:
    """The timeframe a scope reads over. ``kind`` is one of :data:`WINDOW_KINDS`."""

    kind: str = "all"
    count: int = 0  # sprints only; 0 reads as 1
    start: str = ""  # custom only, ISO date
    end: str = ""  # custom only, ISO date; "" = today

    def __post_init__(self) -> None:
        if self.kind not in WINDOW_KINDS:
            raise ValueError(f"unknown window kind {self.kind!r} — one of {', '.join(WINDOW_KINDS)}")
        if self.count < 0:
            raise ValueError("window count cannot be negative")
        for label, value in (("start", self.start), ("end", self.end)):
            if value and not _ISO_DATE.match(value):
                raise ValueError(f"window {label} must be an ISO date (YYYY-MM-DD), got {value!r}")

    @property
    def bounded(self) -> bool:
        return self.kind != "all"

    def label(self) -> str:
        """A short human label: ``last 2 sprints``, ``last month``, ``2026-06-01 to 2026-08-31``."""
        if self.kind == "all":
            return "everything"
        if self.kind == "sprints":
            n = max(1, self.count)
            return "last sprint" if n == 1 else f"last {n} sprints"
        if self.kind == "custom":
            return f"{self.start or '…'} to {self.end or 'today'}"
        return WINDOW_LABELS[self.kind].lower()

    def to_dict(self) -> dict:
        out: dict = {"kind": self.kind}
        if self.kind == "sprints":
            out["count"] = max(1, self.count)
        if self.kind == "custom":
            out["start"] = self.start
            out["end"] = self.end
        return out

    @classmethod
    def from_dict(cls, data: Mapping | None) -> Window:
        """Tolerant: an unknown kind or a bad date reads as ``all`` with a warning."""
        if not isinstance(data, Mapping):
            return cls()
        kind = str(data.get("kind", "all") or "all").strip().lower()
        if kind not in WINDOW_KINDS:
            logger.warning("Window.from_dict: unknown kind %r — reading everything", kind)
            return cls()
        try:
            count = int(data.get("count", 0) or 0)
        except (TypeError, ValueError):
            count = 0
        try:
            return cls(
                kind=kind,
                count=max(0, count),
                start=str(data.get("start", "") or ""),
                end=str(data.get("end", "") or ""),
            )
        except ValueError as exc:
            logger.warning("Window.from_dict: %s — reading everything", exc)
            return cls()

    def to_spec(self) -> str:
        if self.kind == "all":
            return "all"
        if self.kind == "sprints":
            n = max(1, self.count)
            return f"{n}sprint" if n == 1 else f"{n}sprints"
        if self.kind == "custom":
            return f"{self.start}..{self.end}" if self.end else f"{self.start}.." if self.start else "all"
        return self.kind


@dataclass(frozen=True)
class SessionRef:
    """One session pinned by name: always in scope, whatever the window or labels say.

    ``mode`` is the session's own mode (``planning``, ``standup``, …); the
    history stores key a run by ``run_id`` (a performance row's carries a
    colon, ``1on1:12``), the session stores by ``session_id``.
    """

    mode: str
    session_id: str = ""
    run_id: str = ""

    def __post_init__(self) -> None:
        if self.mode not in MODE_SOURCES:
            raise ValueError(f"unknown session mode {self.mode!r} — one of {', '.join(MODE_SOURCES)}")
        if not (self.session_id or self.run_id):
            raise ValueError("a pinned session needs a session_id or a run_id")

    @property
    def source(self) -> str:
        return MODE_SOURCES[self.mode]

    @property
    def key(self) -> str:
        """The id the resolver reads the run under — the store row's own id first."""
        return self.run_id or self.session_id

    def to_dict(self) -> dict:
        return {"mode": self.mode, "session_id": self.session_id, "run_id": self.run_id}

    def to_spec(self) -> str:
        return f"{self.mode}:{self.session_id}:{self.run_id}" if self.run_id else f"{self.mode}:{self.session_id}"

    @classmethod
    def from_spec(cls, text: str) -> SessionRef:
        """``mode:session_id[:run_id]`` — split twice from the left, so a run id keeps its colons."""
        parts = text.strip().split(":", 2)
        if len(parts) < 2:
            raise ValueError(f"a pinned session is mode:session_id[:run_id], got {text!r}")
        mode, session_id = parts[0].strip().lower(), parts[1].strip()
        run_id = parts[2].strip() if len(parts) == 3 else ""
        return cls(mode=mode, session_id=session_id, run_id=run_id)


@dataclass(frozen=True)
class ContextScope:
    """The sessions a run may read. Every field narrows; the default narrows nothing."""

    sources: frozenset[str] | None = None  # None = every source; frozenset() = incognito
    window: Window = field(default_factory=Window)
    projects: tuple[str, ...] = ()  # project labels, any of (OR)
    tags: tuple[str, ...] = ()  # tags, all of (AND)
    limits: tuple[tuple[str, int], ...] = ()  # (source, newest N) caps
    sessions: tuple[SessionRef, ...] = ()  # pinned by name; always read

    def wants(self, source: str) -> bool:
        return self.sources is None or source in self.sources or bool(self.pinned(source))

    def pinned(self, source: str) -> tuple[str, ...]:
        """The keys pinned under ``source``, in pin order."""
        return tuple(dict.fromkeys(ref.key for ref in self.sessions if ref.source == source))

    @property
    def incognito(self) -> bool:
        return self.sources is not None and not self.sources and not self.sessions

    @property
    def narrows(self) -> bool:
        """Whether resolving this scope can change any read at all."""
        return (
            self.sources is not None
            or self.window.bounded
            or bool(self.projects or self.tags or self.limits or self.sessions)
        )

    def limit_for(self, source: str) -> int:
        for name, cap in self.limits:
            if name == source:
                return cap
        return 0

    def to_dict(self) -> dict:
        return {
            "sources": None if self.sources is None else sorted(self.sources, key=SOURCES.index),
            "window": self.window.to_dict(),
            "projects": list(self.projects),
            "tags": list(self.tags),
            "limits": {name: cap for name, cap in self.limits},
            "sessions": [ref.to_dict() for ref in self.sessions],
        }

    @classmethod
    def from_dict(cls, data: Mapping | None) -> ContextScope:
        """The JSON twin, read tolerantly.

        Unknown keys are ignored and unknown sources dropped with a warning. A
        ``sources`` list whose every entry is unknown reads as all-on, never
        incognito: only an explicitly empty list switches everything off.
        """
        if not isinstance(data, Mapping):
            return cls()
        raw_sources = data.get("sources")
        sources: frozenset[str] | None = None
        if isinstance(raw_sources, (list, tuple, set, frozenset)):
            tokens = {str(item).strip().lower() for item in raw_sources if str(item).strip()}
            known = tokens & set(SOURCES)
            if tokens - known:
                logger.warning("ContextScope.from_dict: dropping unknown source(s) %s", sorted(tokens - known))
            if raw_sources and not known and tokens:
                logger.warning("ContextScope.from_dict: no known source in %s — reading everything", sorted(tokens))
                sources = None
            elif not raw_sources:
                sources = frozenset()
            else:
                sources = frozenset(known)
        elif raw_sources is not None:
            logger.warning("ContextScope.from_dict: sources must be a list or null, got %r", type(raw_sources).__name__)
        limits: list[tuple[str, int]] = []
        raw_limits = data.get("limits")
        if isinstance(raw_limits, Mapping):
            for name, cap in raw_limits.items():
                if str(name) not in SOURCES:
                    continue
                try:
                    value = int(cap)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    limits.append((str(name), value))
        return cls(
            sources=sources,
            window=Window.from_dict(data.get("window")),
            projects=_clean_strings(data.get("projects")),
            tags=_clean_strings(data.get("tags")),
            limits=tuple(sorted(limits, key=lambda pair: SOURCES.index(pair[0]))),
            sessions=_clean_sessions(data.get("sessions")),
        )

    def to_spec(self) -> str:
        """The one-line grammar twin; ``parse_context_spec`` reads it back."""
        if self.incognito:
            return "none"
        if self.sources is None:
            head = "all"
            caps = ",".join(f"{name}:{cap}" for name, cap in self.limits if cap > 0)
        elif not self.sources:
            head = "none"
            caps = ""
        else:
            head = ",".join(
                f"{name}:{self.limit_for(name)}" if self.limit_for(name) else name
                for name in SOURCES
                if name in self.sources
            )
            caps = ""
        if self.window.bounded:
            head = f"{head}@{self.window.to_spec()}"
        parts = [head]
        if caps:
            # "all" carries no per-source token, so the caps ride as their own clause.
            parts.append(caps)
        if self.projects:
            parts.append("project=" + ",".join(_quote(label) for label in self.projects))
        if self.tags:
            parts.append("tags=" + ",".join(_quote(tag) for tag in self.tags))
        if self.sessions:
            parts.append("session=" + ",".join(ref.to_spec() for ref in self.sessions))
        return " ".join(parts)


def _clean_sessions(value: object) -> tuple[SessionRef, ...]:
    """The pins a dict carries, read tolerantly: a bad entry is dropped with a warning."""
    if not isinstance(value, (list, tuple)):
        return ()
    out: dict[tuple[str, str], SessionRef] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        try:
            ref = SessionRef(
                mode=str(item.get("mode", "") or "").strip().lower(),
                session_id=str(item.get("session_id", "") or "").strip(),
                run_id=str(item.get("run_id", "") or "").strip(),
            )
        except ValueError as exc:
            logger.warning("ContextScope.from_dict: dropping pinned session: %s", exc)
            continue
        out.setdefault((ref.mode, ref.key), ref)
        if len(out) >= MAX_PINNED_SESSIONS:
            break
    return tuple(out.values())


def pin_sessions(scope: ContextScope | None, refs: Iterable[SessionRef]) -> ContextScope:
    """``scope`` with ``refs`` pinned (an absent scope becomes one that pins them alone)."""
    base = scope or ContextScope()
    merged: dict[tuple[str, str], SessionRef] = {(r.mode, r.key): r for r in base.sessions}
    for ref in refs:
        merged.setdefault((ref.mode, ref.key), ref)
    return replace(base, sessions=tuple(merged.values())[:MAX_PINNED_SESSIONS])


def _clean_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Iterable):
        return ()
    seen: dict[str, None] = {}
    for item in value:
        text = str(item).strip()
        if text:
            seen.setdefault(text, None)
    return tuple(seen)


def _quote(label: str) -> str:
    return f'"{label}"' if any(ch in label for ch in " ,\"'") else label


def wants(scope: ContextScope | None, source: str) -> bool:
    """Whether ``scope`` allows ``source``; an absent scope allows everything."""
    return scope is None or scope.wants(source)


def incognito(scope: ContextScope | None) -> bool:
    """Whether ``scope`` switches every source off; an absent scope never does."""
    return scope is not None and scope.incognito


def coerce_scope(value: ContextScope | Mapping | str | None) -> ContextScope | None:
    """The one entry every engine calls: a scope, its dict, its spec, or nothing.

    A spec string that does not parse raises ``ValueError`` — the surfaces turn
    that into a 400 or a CLI error, never a silent "read nothing".
    """
    if value is None or isinstance(value, ContextScope):
        return value
    if isinstance(value, str):
        if value.lstrip().startswith("{"):
            # The JSON twin travels as a string on graph state and in config
            # columns; a dict inside a string is still the dict twin.
            import json

            try:
                parsed = json.loads(value)
            except ValueError as exc:
                raise ValueError(f"context scope is not valid JSON: {exc}") from None
            return ContextScope.from_dict(parsed) if isinstance(parsed, dict) else None
        return parse_context_spec(value)
    if isinstance(value, Mapping):
        return ContextScope.from_dict(value)
    raise TypeError(f"a context scope is a ContextScope, a dict, a spec string or None — not {type(value).__name__}")


def parse_context_spec(spec: str) -> ContextScope | None:
    """Parse the one-line grammar::

        SPEC    := "" | inherit | none | CLAUSE (WS CLAUSE)*
        CLAUSE  := SOURCES ["@" WINDOW] | window=WINDOW | project=LABELS | tags=TAGS | session=PINS
        SOURCES := all | none | TOKEN ("," TOKEN)*  TOKEN := SOURCE [":" N]
        WINDOW  := all | N sprint(s) | month | quarter | year | DATE ".." [DATE] | DATE
        PINS    := PIN ("," PIN)*                   PIN := MODE ":" SESSION_ID [":" RUN_ID]

    ``""``/``inherit`` → ``None`` (the caller's default applies); ``none`` is
    incognito. An unknown source raises ``ValueError`` naming the valid ones —
    a typo must never read as "that source is switched off".
    """
    text = spec.strip()
    if text.lower() in ("", "inherit"):
        return None
    if text.lower() == "none":
        return ContextScope(sources=frozenset())
    try:
        clauses = shlex.split(text)
    except ValueError as exc:
        raise ValueError(f"could not read context spec {spec!r}: {exc}") from None
    explicit: set[str] = set()
    saw_sources = False
    read_all = False
    window: Window | None = None
    projects: list[str] = []
    tags: list[str] = []
    limits: dict[str, int] = {}
    pins: list[SessionRef] = []
    explicit_none = False
    for clause in clauses:
        key, sep, value = clause.partition("=")
        if sep and key.lower() in ("window", "project", "projects", "tag", "tags", "session", "sessions"):
            word = key.lower()
            if word == "window":
                window = _parse_window(value)
            elif word.startswith("project"):
                projects.extend(_split_labels(value))
            elif word.startswith("session"):
                pins.extend(SessionRef.from_spec(part) for part in _split_labels(value))
            else:
                tags.extend(_split_labels(value))
            continue
        if clause.partition("@")[0].strip().lower() == "none":
            explicit_none = True
            saw_sources = True
            if "@" in clause:
                window = _parse_window(clause.partition("@")[2])
            continue
        head, at, tail = clause.partition("@")
        if at:
            window = _parse_window(tail)
        parsed, caps = _parse_sources(head)
        saw_sources = True
        if parsed is None:
            read_all = True
        else:
            explicit.update(parsed)
        limits.update(caps)
    sources = None if (read_all or not saw_sources) else explicit
    if explicit_none and not explicit:
        sources = set()
    scope_sources = None if sources is None else frozenset(sources)
    unique: dict[tuple[str, str], SessionRef] = {(r.mode, r.key): r for r in pins}
    return ContextScope(
        sources=scope_sources,
        window=window or Window(),
        projects=tuple(dict.fromkeys(projects)),
        tags=tuple(dict.fromkeys(tags)),
        limits=tuple(sorted(limits.items(), key=lambda pair: SOURCES.index(pair[0]))),
        sessions=tuple(unique.values())[:MAX_PINNED_SESSIONS],
    )


def _parse_sources(text: str) -> tuple[list[str] | None, dict[str, int]]:
    """``all`` → ``(None, {})``; ``standup,retro:1`` → ``(["standup","retro"], {"retro": 1})``."""
    word = text.strip().lower()
    if not word or word == "all":
        return None, {}
    names: list[str] = []
    caps: dict[str, int] = {}
    for token in word.split(","):
        token = token.strip()
        if not token:
            continue
        match = _SOURCE_TOKEN.match(token)
        if not match or match.group(1) not in SOURCES:
            raise ValueError(f"unknown context source {token.split(':')[0]!r} — valid: {_valid_sources_text()}")
        name, cap = match.group(1), match.group(2)
        names.append(name)
        if cap and int(cap) > 0:
            caps[name] = int(cap)
    return list(dict.fromkeys(names)), caps


def _parse_window(text: str) -> Window:
    word = text.strip().lower()
    if not word or word == "all":
        return Window()
    if word in ("month", "quarter", "year"):
        return Window(kind=word)
    if word == "sprint":
        return Window(kind="sprints", count=1)
    match = _SPRINTS.match(word)
    if match:
        return Window(kind="sprints", count=max(1, int(match.group(1))))
    start, dots, end = word.partition("..")
    if dots or _ISO_DATE.match(word):
        for value in (start, end):
            if value and not _ISO_DATE.match(value):
                raise ValueError(f"window dates must be ISO (YYYY-MM-DD), got {value!r}")
        return Window(kind="custom", start=start, end=end)
    raise ValueError(
        f"unknown window {text!r} — use all, <N>sprints, month, quarter, year, or YYYY-MM-DD[..YYYY-MM-DD]"
    )


def _split_labels(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]
