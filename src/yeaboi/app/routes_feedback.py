"""Native routes for the in-app feedback form.

``feedback.py`` is not an engine and has no MCP tool, deliberately: filing an
issue on a public repository under the user's own GitHub token is not something
an arbitrary tool client should be able to do on their behalf. It is offered
where a person is looking at the form they wrote.

Four calls. Polish is one LLM call that rewrites the draft and returns it for
review — it never submits. Submit is the one that writes, and it never raises: a
token path failure comes back as a browser URL the person can finish by hand.

Attachments arrive as base64 in JSON, like ``routes_chat.attach``: the proxy the
desktop reaches this server through sends JSON bodies only, and there is no
multipart parser here. ``attach`` stores the bytes and hands back a path; the
submit and polish calls then carry paths, and every path they carry is checked
to live inside the feedback attachments directory. That check is load-bearing —
``polish_feedback`` reads these files and sends their contents to an LLM, so an
unchecked path would read any file on the machine.
"""

from __future__ import annotations

import logging

from yeaboi.app.router import HTTPError, Request, Response, json_response

logger = logging.getLogger(__name__)

_MAX_TITLE = 250
_MAX_DESCRIPTION = 20_000

#: Where every feedback attachment lives. One scope for the whole form: these
#: files outlive no session and belong to no project.
_SCOPE = "feedback"


def options(app, request: Request) -> Response:
    """``GET /api/feedback/options`` — the vocabularies, the caps, and the route.

    ``has_github_token`` is what lets the form say which of the two paths Submit
    will take before it is pressed, rather than after.
    """
    import platform

    from yeaboi import __version__
    from yeaboi.changelog import AREA_COLORS
    from yeaboi.config import get_github_token
    from yeaboi.feedback import (
        FEEDBACK_AREAS,
        FEEDBACK_IMAGE_MIMES,
        FEEDBACK_REPO,
        FEEDBACK_TEXT_MIMES,
        FEEDBACK_TYPES,
        MAX_ATTACHMENTS,
        MAX_TEXT_ATTACHMENT_BYTES,
        max_image_bytes,
    )

    return json_response(
        {
            "types": list(FEEDBACK_TYPES),
            "areas": list(FEEDBACK_AREAS),
            "repo": FEEDBACK_REPO,
            "area_colors": {area: AREA_COLORS[area] for area in FEEDBACK_AREAS if area in AREA_COLORS},
            "version": __version__,
            "platform": f"{platform.system()} {platform.machine()}",
            "has_github_token": bool(get_github_token()),
            "image_mimes": list(FEEDBACK_IMAGE_MIMES),
            "text_mimes": list(FEEDBACK_TEXT_MIMES),
            "max_image_bytes": max_image_bytes(),
            "max_text_bytes": MAX_TEXT_ATTACHMENT_BYTES,
            "max_attachments": MAX_ATTACHMENTS,
        }
    )


def attach(app, request: Request) -> Response:
    """``POST /api/feedback/attachments`` — keep one screenshot or log file.

    Body ``{"name": "app.log", "mime": "text/plain", "data": "<base64>"}``. The
    reply's ``path`` is what a later submit or polish call sends back.
    """
    import base64
    import binascii
    import uuid

    from yeaboi.feedback import (
        MAX_TEXT_ATTACHMENT_BYTES,
        attachment_extension,
        attachment_filename,
        feedback_attachment_kind,
        max_image_bytes,
        safe_attachment_name,
    )
    from yeaboi.paths import get_attachments_dir

    payload = request.json()
    mime = str(payload.get("mime", ""))
    kind = feedback_attachment_kind(mime)
    if kind is None:
        raise HTTPError(400, f"unsupported file type {mime!r} — attach a PNG, a JPEG, or a text file")
    try:
        data = base64.b64decode(str(payload.get("data", "")), validate=True)
    except (binascii.Error, ValueError):
        raise HTTPError(400, "the file must be base64") from None
    if not data:
        raise HTTPError(400, "no file was sent")

    ceiling = max_image_bytes() if kind == "image" else MAX_TEXT_ATTACHMENT_BYTES
    if len(data) > ceiling:
        raise HTTPError(
            413,
            f"Too large ({len(data) / (1024 * 1024):.1f} MB, max {ceiling / (1024 * 1024):.1f} MB)",
        )

    # The stored name is the one the issue body shows, so it keeps the
    # reporter's own — a maintainer reading `app-3f2a91bc.log` is the point.
    name = safe_attachment_name(payload.get("name"), f"attachment{attachment_extension(mime)}")
    path = get_attachments_dir(_SCOPE) / attachment_filename(name, mime, uuid.uuid4().hex[:8])
    try:
        path.write_bytes(data)
    except OSError as exc:
        logger.error("failed to save feedback attachment to %s: %s", path, exc)
        raise HTTPError(500, "Could not save the attachment") from None

    logger.info("feedback attachment saved: kind=%s bytes=%d mime=%s", kind, len(data), mime)
    reply = {"path": str(path), "name": name, "kind": kind, "bytes": len(data)}
    if kind == "text":
        reply["lines"] = len(data.decode("utf-8", "replace").splitlines())
    return json_response(reply)


def submit(app, request: Request) -> Response:
    """``POST /api/feedback`` — file the issue (API token) or hand back a URL."""
    from yeaboi.feedback import submit_feedback
    from yeaboi.mcp.runtime import to_jsonable

    kind, area, title, description = _draft(request)
    images, texts = _attachment_paths(request.json())
    result = submit_feedback(kind, area, title, description, images, texts)
    logger.info("feedback submitted: type=%s area=%s ok=%s via=%s", kind, area, result.ok, result.via)
    return json_response(to_jsonable(result))


def polish(app, request: Request) -> Response:
    """``POST /api/feedback/polish`` — one LLM rewrite of the draft, for review.

    ``polished`` is null when no LLM is configured or the call failed; ``status``
    says why, and the draft the person wrote is what stands. Never an error
    status: falling back to the original is the designed outcome, not a failure.
    """
    from yeaboi.feedback import polish_feedback

    kind, area, title, description = _draft(request)
    images, texts = _attachment_paths(request.json())
    polished, status = polish_feedback(kind, area, title, description, images, texts)
    return json_response(
        {
            "polished": {"title": polished[0], "description": polished[1]} if polished else None,
            "status": status,
        }
    )


def _draft(request: Request) -> tuple[str, str, str, str]:
    """The four fields both calls take, validated against their vocabularies."""
    from yeaboi.feedback import FEEDBACK_AREAS, FEEDBACK_TYPES

    payload = request.json()
    kind = str(payload.get("kind", "")).strip()
    area = str(payload.get("area", "")).strip()
    title = str(payload.get("title", "")).strip()
    description = str(payload.get("description", "")).strip()
    if kind not in FEEDBACK_TYPES:
        raise HTTPError(400, f"unknown feedback type {kind!r} — one of {', '.join(FEEDBACK_TYPES)}")
    if area not in FEEDBACK_AREAS:
        raise HTTPError(400, f"unknown feedback area {area!r} — one of {', '.join(FEEDBACK_AREAS)}")
    if not title:
        raise HTTPError(400, "a feedback report needs a title")
    if not description:
        raise HTTPError(400, "a feedback report needs a description")
    return kind, area, title[:_MAX_TITLE], description[:_MAX_DESCRIPTION]


def _attachment_paths(payload: dict) -> tuple[list[str], list[str]]:
    """The attachment paths this report carries, confined to the attachments directory.

    A path from anywhere else is refused rather than ignored: polish reads these
    files and sends them to an LLM, and the issue body prints them.
    """
    from pathlib import Path

    from yeaboi.feedback import ACCEPTED_SUFFIXES, MAX_ATTACHMENTS
    from yeaboi.paths import get_attachments_dir
    from yeaboi.redaction import log_safe

    root = get_attachments_dir(_SCOPE).resolve()
    kept: list[list[str]] = [[], []]
    seen: set[str] = set()
    total = 0
    for index, key in enumerate(("image_paths", "text_paths")):
        raw = payload.get(key) or []
        if not isinstance(raw, list):
            raise HTTPError(400, f"{key} must be a list of attachment paths")
        want = "image" if key == "image_paths" else "text"
        for entry in raw:
            try:
                # resolve() before relative_to(): it is what defeats a `..` and
                # a symlink planted inside the directory alike.
                resolved = Path(str(entry)).resolve()
                resolved.relative_to(root)
            except (OSError, ValueError):
                logger.warning("feedback: refused an attachment path outside %s: %s", root, log_safe(entry))
                raise HTTPError(400, "an attachment path must be one this app returned") from None
            if ACCEPTED_SUFFIXES.get(resolved.suffix.lower()) != want:
                logger.warning("feedback: refused %s in %s", log_safe(resolved.name), key)
                raise HTTPError(400, f"{resolved.name} is not a {want} attachment")
            total += 1
            if total > MAX_ATTACHMENTS:
                raise HTTPError(400, f"too many attachments — {MAX_ATTACHMENTS} at most")
            # A file the person removed from disk between attaching and sending
            # is dropped, not an error: the report is still worth filing.
            if str(resolved) in seen or not resolved.is_file():
                continue
            seen.add(str(resolved))
            kept[index].append(str(resolved))
    return kept[0], kept[1]
