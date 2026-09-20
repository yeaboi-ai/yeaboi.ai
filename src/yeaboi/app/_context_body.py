"""The three context keys every run body may carry, read once.

``context`` (a scope object or its spec string), ``project_label`` and
``tags`` mean the same thing on every run route, so one reader validates
them and every handler forwards what it returns.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from yeaboi.app.router import HTTPError
from yeaboi.context.labels import MAX_TAG_LENGTH, normalize_tag
from yeaboi.context.scope import ContextScope, coerce_scope

logger = logging.getLogger(__name__)

MAX_PROJECT_LABEL = 80
MAX_TAGS = 20
MAX_INTEGRATIONS = 20


def read_context(payload: Mapping) -> tuple[ContextScope | None, str, tuple[str, ...]]:
    """``(scope, project_label, tags)`` from a request body; 400 when malformed.

    An absent or blank ``context`` is ``None`` — the run reads as it does today.
    """
    raw = payload.get("context")
    scope: ContextScope | None = None
    if raw not in (None, ""):
        try:
            scope = coerce_scope(raw)
        except (ValueError, TypeError) as exc:
            raise HTTPError(400, f"context: {exc}") from None
    label = payload.get("project_label", "")
    if not isinstance(label, str):
        raise HTTPError(400, "project_label must be a string")
    label = " ".join(label.split())
    if len(label) > MAX_PROJECT_LABEL:
        raise HTTPError(400, f"project_label is longer than {MAX_PROJECT_LABEL} characters")
    raw_tags = payload.get("tags") or []
    if not isinstance(raw_tags, list) or not all(isinstance(t, str) for t in raw_tags):
        raise HTTPError(400, "tags must be a list of strings")
    if len(raw_tags) > MAX_TAGS:
        raise HTTPError(400, f"at most {MAX_TAGS} tags")
    for tag in raw_tags:
        if len(tag.strip()) > MAX_TAG_LENGTH:
            raise HTTPError(400, f"a tag is at most {MAX_TAG_LENGTH} characters")
    tags = tuple(dict.fromkeys(t for t in (normalize_tag(tag) for tag in raw_tags) if t))
    if scope is not None or label or tags:
        logger.info("context body: scope=%s label=%r tags=%d", scope.to_spec() if scope else "-", label, len(tags))
    return scope, label, tags


def context_kwargs(payload: Mapping) -> dict:
    """The engine kwargs for a run body — only the keys the body actually carried.

    A blank body forwards nothing; the engine then reads under the mode's
    saved or last-used scope on this machine, unscoped only when there is none.
    """
    scope, label, tags = read_context(payload)
    out: dict = {}
    if scope is not None:
        out["context"] = scope
    if label:
        out["project_label"] = label
    if tags:
        out["tags"] = tags
    return out


def known_integrations() -> tuple[str, ...]:
    """Every connection key a plan may name — the catalog's and the legacy entries'."""
    from yeaboi.connectors import registry

    return tuple(dict.fromkeys(c.key for c in (*registry.all_connectors(), *registry.legacy_entries())))


def read_integrations(payload: Mapping) -> list[str] | None:
    """The ``integrations`` a run body names, or None when the key is absent or null.

    A list is the only integrations the plan may consult, ``[]`` is none;
    every key must be a known connection (connected or not), else 400.
    """
    if "integrations" not in payload or payload.get("integrations") is None:
        return None
    raw = payload["integrations"]
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise HTTPError(400, "integrations must be a list of connection keys")
    if len(raw) > MAX_INTEGRATIONS:
        raise HTTPError(400, f"at most {MAX_INTEGRATIONS} integrations")
    known = known_integrations()
    keys = list(dict.fromkeys(item.strip().lower() for item in raw if item.strip()))
    for key in keys:
        if key not in known:
            raise HTTPError(400, f"unknown integration {key!r} — one of {', '.join(known)}")
    logger.info("integrations body: n=%d", len(keys))
    return keys
