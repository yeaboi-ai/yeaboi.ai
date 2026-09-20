"""The planning room routes — the planning conversation over HTTP.

One turn is a chunked NDJSON stream: each line is a typed event from
:mod:`yeaboi.agent.chat_session`, terminated by ``done``, ``cancelled`` or
``error``. Loopback has no proxy buffering, so a stream is simply the right
shape — the long-poll rationale that shaped the board servers does not apply
here.

A turn runs on a worker thread and the generator drains its queue, because
``ChatSession`` calls back synchronously and a generator cannot yield from
inside a callback. The turn lock and the operation entry are released in the
generator's ``finally``, so a disconnected client frees them too.

Slash input never reaches the model: a ``/``-prefixed text is refused here,
because the verbs run on the client (``GET /api/chat/commands`` lists them).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from collections.abc import Callable, Iterator

from yeaboi.agent.chat_session import (
    SECTION_KINDS,
    Action,
    AskQuestion,
    Assistant,
    AwaitChoice,
    AwaitConfirm,
    AwaitReview,
    Done,
    Notice,
    Progress,
    SectionChanged,
    ShowArtifact,
    Token,
    UserSaid,
    parked_gate,
    replay,
)
from yeaboi.agent.plan_view import pipeline_progress, plan_view, section_status
from yeaboi.app._context_body import read_context, read_integrations
from yeaboi.app.chats import ChatBusyError, LiveChat, UnknownChatError, described_as
from yeaboi.app.router import HTTPError, Request, Response, json_response
from yeaboi.mcp.runtime import to_jsonable
from yeaboi.sessions import plan_title

logger = logging.getLogger(__name__)

#: Every line type a turn may stream. The vocabulary is the contract.
WIRE_TYPES: tuple[str, ...] = (
    "op",
    "token",
    "assistant",
    "user",
    "question",
    "await_confirm",
    "await_review",
    "await_choice",
    "artifact",
    "progress",
    "section",
    "notice",
    "action",
    "done",
    "cancelled",
    "error",
)

#: The session view's keys, in order. ``project_id`` duplicates ``session_id``
#: for one release, until the desktop reads the new name.
SESSION_VIEW_KEYS: tuple[str, ...] = (
    "session_id",
    "project_id",
    "title",
    "project_label",
    "tags",
    "integrations",
    "stage",
    "intake_mode",
    "opening",
    "created_at",
    "last_modified",
    "transcript",
    "question",
    "progress",
    "pending",
    "sections",
)

#: The stages that want ``advance`` rather than a reply.
ADVANCE_STAGES = ("pipeline", "epic")

MAX_TITLE = 120
DEFAULT_LIST_LIMIT = 50

#: The stream is over. Never reaches the wire.
_END = object()

#: What the window may paste. Mirrors ``_attachments._EXT_FOR_MIME``, which is
#: keyed the same way but private to the terminal's clipboard path.
_EXT_FOR_IMAGE = {"image/png": ".png", "image/jpeg": ".jpg"}

ATTACHMENT_KINDS = ("image", "text")


# ------------------------------------------------------------- the sessions


def create(app, request: Request) -> Response:
    """``POST /api/chat/sessions`` — open a conversation on a description."""
    payload = request.json()
    description = str(payload.get("description", "")).strip()
    if not description:
        raise HTTPError(400, "description is required")
    intake_mode = str(payload.get("intake_mode", ""))
    if intake_mode not in ("", "small_project", "smart"):
        raise HTTPError(400, "intake_mode must be 'small_project' or 'smart'")
    solo = bool(payload.get("solo", False))
    if solo:
        logger.info("Chat create: solo intake")
    profile_id = str(payload.get("analysis_profile_id", "") or "").strip()
    if profile_id and not _profile_exists(profile_id):
        raise HTTPError(400, f"no analysis profile {profile_id!r}")
    title = _read_title(payload)
    from yeaboi.context.resolve import scope_for

    scope, project_label, tags = read_context(payload)
    integrations = read_integrations(payload)
    refs = _read_refs(payload)
    # The rule every run shares: an absent scope inherits the last one used.
    scope = scope_for("planning", scope)
    chat = app.chats.create(
        description,
        intake_mode=intake_mode,
        solo=solo,
        analysis_profile_id=profile_id,
        context_scope=scope.to_dict() if scope is not None else None,
        project_label=project_label,
        title=title,
        integrations=integrations,
        refs=refs,
    )
    app.chats.save(chat)
    _label(app, chat, project_label=project_label, tags=tags, scope=scope, defaults=True)
    if refs and app.chats.link_sessions(chat, refs) is not None:
        app.chats.save(chat)
    return json_response(_view(app, chat), code=201)


def list_sessions(app, request: Request) -> Response:
    """``GET /api/chat/sessions`` — every plan, newest first; ``?limit=&project_label=&tag=``."""
    raw_limit = str(request.query.get("limit", "")).strip()
    try:
        limit = int(raw_limit) if raw_limit else DEFAULT_LIST_LIMIT
    except ValueError:
        raise HTTPError(400, "limit must be a number") from None
    if limit < 0:
        raise HTTPError(400, "limit must be zero or more")
    project_label = " ".join(str(request.query.get("project_label", "")).split())
    tag = str(request.query.get("tag", "")).strip().lower()
    rows = app.chats.list_rows(limit=limit, project_label=project_label, tag=tag)
    return json_response({"sessions": rows})


def commands(app, request: Request) -> Response:
    """``GET /api/chat/commands`` — the slash verbs the window runs itself."""
    from yeaboi.ui.session.chat._commands import wire_commands

    return json_response({"commands": wire_commands()})


def get(app, request: Request) -> Response:
    """``GET /api/chat/sessions/{session_id}`` — the whole conversation, replayed."""
    return json_response(_view(app, _chat(app, request)))


def update(app, request: Request) -> Response:
    """``POST /api/chat/sessions/{session_id}/update`` — rename, relabel or rescope a plan.

    Only the keys present change. ``tags`` replaces the list (the picker
    sends every tag it shows, the fixed ones included); a blank
    ``project_label`` clears the label; ``context`` null or blank clears the
    scope.
    """
    from dataclasses import replace

    from yeaboi.context.scope import coerce_scope

    payload = request.json()
    chat = _chat(app, request)
    scope, project_label, tags = read_context(payload)
    integrations = read_integrations(payload)
    touched = {key for key in ("context", "project_label", "tags", "integrations") if key in payload}
    # A running turn ends by replacing the state wholesale; a write that
    # slipped in beside it would be lost, so the update waits its turn.
    _hold_turn(chat)
    try:
        if "title" in payload:
            app.chats.rename(chat, _read_title(payload))
        if touched:
            state = chat.session.state
            if "integrations" in touched:
                if integrations is None:
                    state.pop("session_integrations", None)
                else:
                    state["session_integrations"] = list(integrations)
            if "context" in touched:
                raw_context = payload.get("context")
                if scope is not None and isinstance(raw_context, dict) and "sessions" not in raw_context:
                    # A picker that predates pins must not drop them.
                    try:
                        stored = coerce_scope(state.get("context_scope") or None)
                    except (TypeError, ValueError):
                        stored = None
                    if stored is not None and stored.sessions:
                        scope = replace(scope, sessions=stored.sessions)
                if scope is None:
                    state.pop("context_scope", None)
                else:
                    state["context_scope"] = json.dumps(scope.to_dict())
            if "project_label" in touched:
                if project_label:
                    state["project_label"] = project_label
                else:
                    state.pop("project_label", None)
            app.chats.save(chat)
            _label(
                app,
                chat,
                project_label=project_label if "project_label" in touched else None,
                tags=tags if "tags" in touched else None,
                scope=scope if "context" in touched else None,
                clear_scope="context" in touched and scope is None,
            )
    finally:
        chat.turn.release()
    logger.info("Chat updated: session=%s keys=%s", chat.session_id, sorted(set(payload) & {"title", *touched}))
    meta = app.chats.meta(chat.session_id)
    labels = app.chats.labels(chat.session_id)
    return json_response(
        {
            "session_id": chat.session_id,
            "title": meta["title"],
            "project_label": labels["project_label"],
            "tags": labels["tags"],
            "context": labels["scope"],
            "integrations": chat.session.state.get("session_integrations"),
        }
    )


def delete(app, request: Request) -> Response:
    """``POST /api/chat/sessions/{session_id}/delete`` — the plan, its versions, labels and files."""
    session_id = request.params.get("session_id", "")
    try:
        deleted = app.chats.delete(session_id)
    except ChatBusyError:
        raise HTTPError(409, "a turn is running for this conversation — try again when it lands") from None
    if not deleted:
        raise HTTPError(404, f"no conversation {session_id!r}")
    return json_response({"deleted": True, "session_id": session_id})


# ---------------------------------------------------------------- the turns


def send(app, request: Request) -> Response:
    """``POST /api/chat/sessions/{session_id}/send`` — one turn, streamed as NDJSON.

    The first line names the operation id, so the client can cancel the turn
    through ``POST /api/ops/{op_id}/cancel`` before the reply lands.

    ``images``, ``files`` and ``refs`` are the composer's whole lists, in
    order. Which of them actually travel is decided here from the surviving
    ``[image #N]``, ``[file #N]`` and ``[ref #N]`` chips, so deleting a chip
    detaches its attachment on this surface exactly as it does in the
    terminal — one implementation of the rule, not two.
    """
    from yeaboi.agent.chat_refs import referenced_refs
    from yeaboi.ui.session.chat._commands import is_slash_verb
    from yeaboi.ui.shared._attachments import referenced_files, referenced_images

    payload = request.json()
    text = str(payload.get("text", ""))
    if is_slash_verb(text):
        raise HTTPError(400, "slash commands run on the client — see GET /api/chat/commands")
    attachments = [str(name) for name in payload.get("images") or []]
    images = referenced_images(text, attachments) if attachments else []
    refs = referenced_refs(text, _read_refs(payload))
    chat = _chat(app, request)
    files = _confined_files(referenced_files(text, _read_paths(payload, "files")), chat.session_id)
    if chat.session.awaiting in ADVANCE_STAGES:
        raise HTTPError(409, "the plan is being built — POST …/advance runs the next step")
    logger.info(
        "Chat turn start: session=%s len=%d images=%d files=%d refs=%d",
        chat.session_id,
        len(text),
        len(images),
        len(files),
        len(refs),
    )

    def run(on_event, cancel):
        if refs:
            app.chats.link_sessions(chat, refs)
        return chat.session.reply(text, on_event, images=images, files=files, refs=refs, cancel=cancel)

    return _stream(app, chat, run)


def advance(app, request: Request) -> Response:
    """``POST /api/chat/sessions/{session_id}/advance`` — one no-input step of the build."""
    chat = _chat(app, request)
    if chat.session.awaiting not in ADVANCE_STAGES:
        raise HTTPError(409, "nothing to run — the conversation is waiting for a reply")
    logger.info("Chat advance start: session=%s stage=%s", chat.session_id, chat.session.awaiting)
    return _stream(app, chat, lambda on_event, cancel: chat.session.advance(on_event, cancel=cancel))


def _hold_turn(chat: LiveChat) -> None:
    """Take the conversation's turn lock now or answer 409; the caller releases it."""
    if not chat.turn.acquire(blocking=False):
        raise HTTPError(409, "a turn is running for this conversation — try again when it lands")


def _stream(app, chat: LiveChat, run: Callable) -> Response:
    if not chat.turn.acquire(blocking=False):
        raise HTTPError(409, "a turn is already running for this conversation")
    try:
        op = app.ops.create()
    except Exception:
        chat.turn.release()
        raise
    return Response(
        content_type="application/x-ndjson",
        stream=_lines(_turn(app, chat, op, run)),
        headers=(("X-Accel-Buffering", "no"),),
    )


def _turn(app, chat: LiveChat, op, run: Callable) -> Iterator[dict]:
    from yeaboi.mcp.runtime import _ENGINE_LOCK

    events: queue.Queue = queue.Queue()
    failure: list[BaseException | None] = [None]

    def worker() -> None:
        try:
            # Engines are one-at-a-time process-wide; a chat turn is one of
            # them. Never fork this lock — a second one serialises nothing.
            with _ENGINE_LOCK:
                run(events.put, op.cancel)
        except BaseException as exc:  # noqa: BLE001 — reported on the stream below
            failure[0] = exc
        finally:
            events.put(_END)

    thread = threading.Thread(target=worker, name="chat-turn", daemon=True)
    thread.start()
    try:
        yield {"type": "op", "op_id": op.op_id}
        while (event := events.get()) is not _END:
            yield _wire(event, chat)
        thread.join()
        if failure[0] is not None:
            yield _error_line(failure[0])
        else:
            app.chats.save(chat)
    finally:
        app.ops.remove(op.op_id)
        chat.turn.release()


# ------------------------------------------------------------ the questions


def questions(app, request: Request) -> Response:
    """``GET /api/chat/sessions/{session_id}/questions`` — this run's question plan.

    Every question the run touches: the essential gaps still to ask, plus
    anything already answered. Not the 30-question bank — extraction, SCRUM.md
    and defaults answer most of it silently, and listing all thirty would
    misdescribe the conversation the person is actually having.

    One payload serves three affordances the terminal keeps apart: the
    ``/questions`` checklist, the ``/form`` questionnaire, and the answer
    browser behind a bare ``/edit``.
    """
    from yeaboi.agent.state import TOTAL_QUESTIONS, QuestionnaireState
    from yeaboi.prompts.intake import QUESTION_SHORT_LABELS
    from yeaboi.ui.session.chat._question_view import planned_question_sets

    chat = _chat(app, request)
    qs = chat.session.state.get("questionnaire")
    if not isinstance(qs, QuestionnaireState):
        # Pre-graph: the questionnaire only exists after the first invoke.
        return json_response({"questions": [], "total": TOTAL_QUESTIONS, "completed": False, "derived": False})

    sets = planned_question_sets(qs)
    remaining = set(sets[0]) if sets else set()
    answered = {number for number, answer in qs.answers.items() if answer}
    rows = [
        {
            "number": number,
            "label": QUESTION_SHORT_LABELS.get(number, f"Question {number}"),
            "answer": qs.answers.get(number, ""),
            "remaining": number in remaining,
            "skipped": number in qs.skipped_questions,
        }
        for number in sorted(remaining | answered)
    ]
    return json_response(
        {
            "questions": rows,
            "total": TOTAL_QUESTIONS,
            "completed": bool(qs.completed),
            # False when the gap derivation failed — the client then says so
            # rather than presenting a short list as the whole plan.
            "derived": sets is not None,
        }
    )


def size(app, request: Request) -> Response:
    """``POST /api/chat/sessions/{session_id}/size`` — switch the plan size mid-run.

    The answers survive the switch; the pipeline artifacts do not, because
    they were generated for the other mode. Returns ``changed: false`` when
    the conversation is already that size, so the caller says so rather than
    running a pointless turn.
    """
    from yeaboi.agent.nodes import apply_size_switch

    payload = request.json()
    mode = str(payload.get("mode", ""))
    if mode not in ("small_project", "smart"):
        raise HTTPError(400, "mode must be 'small_project' or 'smart'")
    chat = _chat(app, request)
    if chat.session.dry_run:
        raise HTTPError(409, "Size switching is not available in dry-run")
    _hold_turn(chat)
    try:
        state = chat.session.state
        if state.get("_intake_mode") == mode:
            return json_response({"changed": False, "mode": mode})
        if state.get("questionnaire") is None:
            # Pre-intake there is nothing to reset — record the preference and the
            # size exchange honours it.
            state["_intake_mode"] = mode
            app.chats.save(chat)
            return json_response({"changed": True, "mode": mode, "reopened": False})
        apply_size_switch(state, mode)
        # The prior-art step re-runs under the new mode, so its old card has no
        # data left to render from.
        state.pop("_prior_art_preview", None)
        app.chats.save(chat)
    finally:
        chat.turn.release()
    logger.info("Chat size switched: session=%s mode=%s", chat.session_id, mode)
    return json_response({"changed": True, "mode": mode, "reopened": True})


def attach(app, request: Request) -> Response:
    """``POST /api/chat/sessions/{session_id}/attachments`` — keep one pasted image or text file.

    The window reads the clipboard itself (the terminal cannot), so what
    arrives here is bytes rather than a paste event. Everything downstream is
    the terminal's: the same size ceiling, the same attachments directory, and
    the same ``[image #N]`` chip, which is what makes the image detachable by
    deleting text. ``kind: "text"`` keeps a small text file the same way,
    behind a ``[file #N]`` chip.
    """
    import base64
    import binascii
    import uuid

    from yeaboi.paths import get_attachments_dir
    from yeaboi.ui.shared._attachments import MAX_IMAGE_BYTES, chip_text

    chat = _chat(app, request)
    payload = request.json()
    kind = str(payload.get("kind", "image") or "image")
    if kind not in ATTACHMENT_KINDS:
        raise HTTPError(400, f"unknown attachment kind {kind!r} — one of {', '.join(ATTACHMENT_KINDS)}")
    if kind == "text":
        return _attach_text(chat, payload)
    mime = str(payload.get("mime", "image/png"))
    if mime not in _EXT_FOR_IMAGE:
        raise HTTPError(400, f"unsupported image type {mime!r} — paste a PNG or a JPEG")
    try:
        data = base64.b64decode(str(payload.get("image", "")), validate=True)
    except (binascii.Error, ValueError):
        raise HTTPError(400, "image must be base64") from None
    if not data:
        raise HTTPError(400, "no image was sent")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPError(413, f"Image too large ({len(data) / (1024 * 1024):.1f} MB, max 4.5 MB)")

    index = int(payload.get("index", 1))
    path = get_attachments_dir(chat.session_id) / f"img-{uuid.uuid4().hex[:8]}{_EXT_FOR_IMAGE[mime]}"
    try:
        path.write_bytes(data)
    except OSError as exc:
        logger.error("failed to save pasted image to %s: %s", path, exc)
        raise HTTPError(500, "Could not save pasted image") from None
    logger.info("image pasted: session=%s bytes=%d mime=%s", chat.session_id, len(data), mime)
    return json_response({"path": str(path), "chip": chip_text(index)})


def _attach_text(chat: LiveChat, payload: dict) -> Response:
    """One text file, kept under the session's attachments directory behind a ``[file #N]`` chip."""
    import uuid
    from pathlib import Path

    from yeaboi.feedback import safe_attachment_name
    from yeaboi.paths import get_attachments_dir
    from yeaboi.ui.shared._attachments import MAX_TEXT_FILE_BYTES, TEXT_FILE_SUFFIXES, file_chip_text

    text = payload.get("text")
    if not isinstance(text, str):
        raise HTTPError(400, "text must be a string")
    if not text.strip():
        raise HTTPError(400, "no text was sent")
    name = safe_attachment_name(payload.get("name"), "file.txt")
    suffix = Path(name).suffix.lower()
    if suffix not in TEXT_FILE_SUFFIXES:
        raise HTTPError(400, f"unsupported file type {suffix or name!r} — one of {', '.join(TEXT_FILE_SUFFIXES)}")
    data = text.encode("utf-8")
    if len(data) > MAX_TEXT_FILE_BYTES:
        raise HTTPError(413, f"File too large ({len(data) / 1024:.0f} KB, max {MAX_TEXT_FILE_BYTES // 1024} KB)")
    index = int(payload.get("index", 1))
    path = get_attachments_dir(chat.session_id) / f"file-{uuid.uuid4().hex[:8]}-{name}"
    try:
        path.write_bytes(data)
    except OSError as exc:
        logger.error("failed to save attached file to %s: %s", path, exc)
        raise HTTPError(500, "Could not save the file") from None
    logger.info("file attached: session=%s name=%s bytes=%d", chat.session_id, name, len(data))
    return json_response(
        {"path": str(path), "chip": file_chip_text(index), "kind": "text", "name": name, "bytes": len(data)}
    )


# ----------------------------------------------------------------- the plan


def plan(app, request: Request) -> Response:
    """``GET /api/chat/sessions/{session_id}/plan`` — every section as plain data."""
    chat = _chat(app, request)
    view = plan_view(chat.session.state, versions=app.chats.version_counts(chat.session_id))
    return json_response({"session_id": chat.session_id, **view})


def plan_versions(app, request: Request) -> Response:
    """``GET /api/chat/sessions/{session_id}/plan/versions`` — the accepted snapshots, oldest first."""
    chat = _chat(app, request)
    section = str(request.query.get("section", "")).strip()
    if section and section not in SECTION_KINDS:
        raise HTTPError(400, f"section must be one of {', '.join(SECTION_KINDS)}")
    return json_response({"versions": app.chats.versions(chat.session_id, section)})


def plan_version(app, request: Request) -> Response:
    """``GET /api/chat/sessions/{session_id}/plan/versions/{section}/{version}`` — one snapshot."""
    chat = _chat(app, request)
    section = request.params.get("section", "")
    if section not in SECTION_KINDS:
        raise HTTPError(400, f"section must be one of {', '.join(SECTION_KINDS)}")
    try:
        number = int(request.params.get("version", ""))
    except ValueError:
        raise HTTPError(400, "version must be a number") from None
    row = app.chats.version(chat.session_id, section, number)
    if row is None:
        raise HTTPError(404, f"no version {number} of {section}")
    return json_response(row)


# ---------------------------------------------------------------- internals


def _error_line(error: BaseException) -> dict:
    from yeaboi.agent.streaming import ChatStreamCancelledError

    if isinstance(error, ChatStreamCancelledError):
        return {"type": "cancelled"}
    # The one place SDK exceptions become human text — never str(exc), which
    # for a JIRAError is its entire HTTP response.
    from yeaboi.ui.session._utils import _classify_api_error

    message = _classify_api_error(error) if isinstance(error, Exception) else "The turn stopped unexpectedly."
    logger.error("Chat turn failed: %s", message)
    return {"type": "error", "message": message}


def _lines(objects: Iterator[dict]) -> Iterator[bytes]:
    for obj in objects:
        yield (json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def _wire(event, chat: LiveChat) -> dict:
    """One chat event as its wire object. The type tag is the contract."""
    if isinstance(event, Token):
        return {"type": "token", "text": event.text}
    if isinstance(event, Assistant):
        return {"type": "assistant", "text": event.text}
    if isinstance(event, UserSaid):
        return {"type": "user", "text": event.text}
    if isinstance(event, AskQuestion):
        return {"type": "question", "text": event.text, "number": event.number}
    if isinstance(event, AwaitConfirm):
        return {"type": "await_confirm", "kind": event.kind, "prompt": event.prompt}
    if isinstance(event, ShowArtifact):
        return {"type": "artifact", "kind": event.kind}
    if isinstance(event, AwaitReview):
        return {"type": "await_review", "node": event.node, "kind": event.kind, "prompt": event.prompt}
    if isinstance(event, AwaitChoice):
        options = [{"key": key, "label": label} for key, label in event.options]
        return {"type": "await_choice", "kind": event.kind, "prompt": event.prompt, "options": options}
    if isinstance(event, Progress):
        return {
            "type": "progress",
            "node": event.node,
            "step": event.step,
            "total": event.total,
            "status": event.status,
        }
    if isinstance(event, SectionChanged):
        return {"type": "section", "kind": event.kind, "status": event.status, "version": event.version}
    if isinstance(event, Notice):
        return {"type": "notice", "text": event.text}
    if isinstance(event, Action):
        return {"type": "action", "name": event.name, "detail": event.detail}
    if isinstance(event, Done):
        return {"type": "done", "stage": chat.session.awaiting}
    raise TypeError(f"no wire shape for chat event {type(event).__name__}")


def _view(app, chat: LiveChat) -> dict:
    """The whole conversation as the renderer draws it, keys in :data:`SESSION_VIEW_KEYS` order."""
    from yeaboi.ui.session.chat._question_view import derive_question_view

    state = chat.session.state
    meta = app.chats.meta(chat.session_id)
    labels = app.chats.labels(chat.session_id)
    counts = app.chats.version_counts(chat.session_id)
    gate = parked_gate(state)
    values = {
        "session_id": chat.session_id,
        "project_id": chat.session_id,
        "title": plan_title(
            chat.title or meta["title"],
            getattr(state.get("project_analysis"), "project_name", "") or "",
            described_as(state),
        ),
        "project_label": labels["project_label"],
        "tags": labels["tags"],
        # None = unrestricted (a plan from before the key, or a client that sent none).
        "integrations": state.get("session_integrations"),
        "stage": chat.session.awaiting,
        "intake_mode": state.get("_intake_mode", ""),
        # Non-empty only until the description has been sent as the first turn.
        "opening": state.get("_chat_opening", ""),
        "created_at": meta["created_at"],
        "last_modified": meta["last_modified"],
        "transcript": [_wire(item, chat) for item in replay(state)],
        "question": to_jsonable(derive_question_view(state)),
        "progress": pipeline_progress(state),
        # The gate a reopened window redraws its buttons from — never from prose.
        "pending": _wire(gate, chat) if gate is not None else None,
        "sections": [
            {"kind": kind, "status": section_status(state, kind), "version": int(counts.get(kind, 0))}
            for kind in SECTION_KINDS
        ],
    }
    return {key: values[key] for key in SESSION_VIEW_KEYS}


def _chat(app, request: Request) -> LiveChat:
    session_id = request.params.get("session_id", "")
    try:
        return app.chats.open(session_id)
    except UnknownChatError:
        raise HTTPError(404, f"no conversation {session_id!r}") from None


def _read_title(payload: dict) -> str:
    title = payload.get("title", "")
    if not isinstance(title, str):
        raise HTTPError(400, "title must be a string")
    title = " ".join(title.split())
    if len(title) > MAX_TITLE:
        raise HTTPError(400, f"title is longer than {MAX_TITLE} characters")
    return title


def _label(app, chat: LiveChat, **kwargs) -> None:
    # A label failure must not fail the plan.
    try:
        app.chats.set_labels(chat, **kwargs)
    except Exception:  # noqa: BLE001 — logged, the conversation goes on
        logger.warning("Labels for %s were not written", chat.session_id, exc_info=True)


def _read_refs(payload: dict) -> list[dict]:
    """The body's ``refs``, validated; 400 names the first bad one."""
    from yeaboi.agent.chat_refs import validate_refs

    try:
        return validate_refs(payload.get("refs"))
    except ValueError as exc:
        raise HTTPError(400, f"refs: {exc}") from None


def _read_paths(payload: dict, key: str) -> list[str]:
    raw = payload.get(key) or []
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise HTTPError(400, f"{key} must be a list of attachment paths")
    return raw


def _confined_files(paths: list[str], session_id: str) -> list[str]:
    """The attached text files a turn may read: under this session's attachments directory, of an allowed type.

    A path from anywhere else is dropped with a warning rather than read —
    the model would see its contents.
    """
    from pathlib import Path

    from yeaboi.paths import get_attachments_dir
    from yeaboi.redaction import log_safe
    from yeaboi.ui.shared._attachments import TEXT_FILE_SUFFIXES

    if not paths:
        return []
    root = get_attachments_dir(session_id).resolve()
    kept: list[str] = []
    for entry in paths:
        try:
            resolved = Path(entry).resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            logger.warning("chat: dropped a file outside %s: %s", root, log_safe(entry))
            continue
        if resolved.suffix.lower() not in TEXT_FILE_SUFFIXES:
            logger.warning("chat: dropped %s — not a text attachment", log_safe(resolved.name))
            continue
        if not resolved.is_file():
            logger.warning("chat: dropped %s — no longer there", log_safe(resolved.name))
            continue
        kept.append(str(resolved))
    return kept


def _profile_exists(profile_id: str) -> bool:
    """True when a saved analysis profile carries this id."""
    from yeaboi.paths import get_db_path
    from yeaboi.team_profile import TeamProfileStore

    db_path = get_db_path()
    if not db_path.exists():
        return False
    try:
        with TeamProfileStore(db_path) as store:
            return any(profile.team_id == profile_id for profile in store.list_profiles())
    except Exception:  # noqa: BLE001 — an unreadable store has no such profile
        logger.warning("Analysis profiles could not be read", exc_info=True)
        return False
