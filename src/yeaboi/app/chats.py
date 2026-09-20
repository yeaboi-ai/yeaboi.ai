"""Live planning conversations — one :class:`ChatSession` per session id.

Sessions live in the backend, not in the renderer: a reloaded window rejoins
the conversation it left, and the graph is compiled once for the process
rather than once per turn. Persistence is the shared session store
(``sessions.py``), the same rows ``plan_get``/``plan_export``/``plan_sync``
and the recent-sessions list read — a plan started here is one plan
everywhere. Conversations the old file store holds still open, read-only.
The label a plan carries (its project label, tags and the scope it reads
under) lives in the context package's label store beside it.

The graph factory and every store access are injected so the whole surface
can be tested without an LLM or a home directory.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from functools import partial

from yeaboi.agent.chat_session import ChatSession, last_completed_node, stage_of, start_state
from yeaboi.sessions import plan_title

logger = logging.getLogger(__name__)

#: The section counts a list row carries.
COUNT_KINDS = ("features", "stories", "tasks", "sprints")


@dataclass
class LiveChat:
    """One conversation plus the lock that keeps its turns single-file."""

    session_id: str
    session: ChatSession
    turn: threading.Lock = field(default_factory=threading.Lock)
    title: str = ""  # a user-given title, written with the first save
    closed: bool = False  # deleted while open — nothing about it is written again
    stale: bool = False  # a tracker sync rewrote the row mid-turn — merge its keys before saving


#: The state keys a tracker sync writes; what a stale chat takes back from the store.
_SYNC_KEY = re.compile(r"^(jira|azdevops|linear|trello)_")


def _compile_graph():
    from yeaboi.agent.graph import create_graph

    return create_graph()


def _store():
    from yeaboi.paths import get_db_path
    from yeaboi.sessions import SessionStore

    return SessionStore(get_db_path())


def _label_store():
    from yeaboi.context.labels import LabelStore
    from yeaboi.paths import get_db_path

    return LabelStore(get_db_path())


def _new_id() -> str:
    from yeaboi.sessions import make_session_id

    return make_session_id()


def _remove_attachments(session_id: str) -> None:
    from yeaboi.paths import ATTACHMENTS_DIR, PLANNING_LOGS_DIR, _safe_key

    folder = ATTACHMENTS_DIR / _safe_key(session_id, "misc")
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    if PLANNING_LOGS_DIR.exists():
        for log_file in PLANNING_LOGS_DIR.glob(f"{session_id}.log*"):
            log_file.unlink(missing_ok=True)


def described_as(state: dict | None) -> str:
    """The description a plan was opened on: the analysis's, else the opening line, else messages[0]."""
    if not state:
        return ""
    text = state.get("project_description") or state.get("_chat_opening") or ""
    if text:
        return str(text)
    for message in state.get("messages") or []:
        if getattr(message, "type", "") == "human":
            content = getattr(message, "content", "")
            return content if isinstance(content, str) else ""
    return ""


class UnknownChatError(LookupError):
    """No conversation with that id is open or stored."""


class ChatBusyError(RuntimeError):
    """A turn is running, so the conversation cannot be removed yet."""


class ChatSupervisor:
    """The open conversations. Thread-safe; one compiled graph for all of them.

    ``store_factory`` opens the session store and ``label_store_factory`` the
    label store; the finer seams (``loader``, ``saver``, ``meta_saver``,
    ``version_recorder``) default to the store and exist for tests that keep
    state in a dict.
    """

    def __init__(
        self,
        *,
        graph_factory=_compile_graph,
        store_factory=_store,
        label_store_factory=_label_store,
        loader: Callable[[str], dict | None] | None = None,
        saver: Callable[[str, dict], None] | None = None,
        id_factory=_new_id,
        meta_saver: Callable[..., None] | None = None,
        version_recorder: Callable[[str, str, dict], int] | None = None,
        attachment_remover: Callable[[str], None] = _remove_attachments,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._graph_factory = graph_factory
        self._store = store_factory
        self._labels = label_store_factory
        self._loader = loader or self._load
        self._saver = saver or self._save
        self._id_factory = id_factory
        self._meta_saver = meta_saver or self._save_meta
        self._version_recorder = version_recorder or self._record_version
        self._attachment_remover = attachment_remover
        self._today = today
        self._chats: dict[str, LiveChat] = {}
        self._lock = threading.Lock()
        self._graph = None
        self._graph_lock = threading.Lock()

    # ------------------------------------------------------------ the store

    def _load(self, session_id: str) -> dict | None:
        with self._store() as store:
            state = store.load_state(session_id)
        if state is not None:
            return state
        # A conversation from before the store move. Read-only: its next save
        # lands in the store, where every other reader already looks.
        from yeaboi.persistence import load_graph_state

        return load_graph_state(session_id)

    def _save(self, session_id: str, state: dict) -> None:
        with self._store() as store:
            store.create_session(session_id, mode="planning")  # INSERT OR IGNORE — adopts a file-store chat
            store.save_state(session_id, state)
            analysis = state.get("project_analysis")
            name = getattr(analysis, "project_name", "") or ""
            if name:
                store.update_project_name(session_id, name)
            node = last_completed_node(state)
            if node:
                store.update_last_node(session_id, node)

    def _save_meta(self, session_id: str, *, title: str) -> None:
        with self._store() as store:
            store.update_session_meta(session_id, title=title)

    def _record_version(self, session_id: str, section: str, payload: dict) -> int:
        with self._store() as store:
            return store.record_plan_version(session_id, section, payload)

    def graph(self):
        """The compiled planning graph — built once, on first use."""
        with self._graph_lock:
            if self._graph is None:
                logger.info("Compiling the planning graph for the app")
                self._graph = self._graph_factory()
            return self._graph

    def _session(self, session_id: str, state: dict) -> ChatSession:
        # No typewriter: a socket wants the reply once, from the state, not
        # paced eight characters at a time.
        return ChatSession(
            self.graph(),
            state,
            typewriter=False,
            on_version=partial(self._version_recorder, session_id),
        )

    # ------------------------------------------------------- conversations

    def create(
        self,
        description: str,
        *,
        intake_mode: str = "",
        solo: bool = False,
        analysis_profile_id: str = "",
        context_scope: dict | None = None,
        project_label: str = "",
        title: str = "",
        integrations: list[str] | None = None,
        refs: list[dict] | None = None,
    ) -> LiveChat:
        """Open a new conversation seeded with the greeting and the description.

        ``refs`` are read into the intake now — the first turn must not repeat them.
        """
        session_id = self._id_factory()
        state = start_state(
            description,
            intake_mode=intake_mode,
            solo=solo,
            analysis_profile_id=analysis_profile_id,
            context_scope=context_scope,
            project_label=project_label,
            integrations=integrations,
        )
        if refs:
            from yeaboi.agent.chat_refs import render_context_block

            state["pasted_context"] = render_context_block(refs)
        chat = LiveChat(session_id, self._session(session_id, state), title=title)
        with self._lock:
            self._chats[session_id] = chat
        logger.info("Chat created: session=%s", session_id)
        return chat

    def open(self, session_id: str) -> LiveChat:
        """The live conversation for an id, resuming it from disk if needed."""
        with self._lock:
            chat = self._chats.get(session_id)
            if chat is not None:
                return chat
        state = self._loader(session_id)
        if state is None:
            raise UnknownChatError(session_id)
        chat = LiveChat(session_id, self._session(session_id, state))
        with self._lock:
            # Another thread may have resumed the same id first — one live
            # session per conversation, or two turns would fork the state.
            chat = self._chats.setdefault(session_id, chat)
        logger.info("Chat resumed: session=%s", session_id)
        return chat

    def save(self, chat: LiveChat) -> None:
        if chat.closed:
            logger.info("Chat %s was deleted mid-turn — not saved", chat.session_id)
            return
        if chat.stale:
            self._take_sync_keys(chat)
        self._saver(chat.session_id, chat.session.state)
        if chat.title:
            # After the state, so the row exists; once, so a later rename sticks.
            self._meta_saver(chat.session_id, title=chat.title)
            chat.title = ""

    def rename(self, chat: LiveChat, title: str) -> None:
        """A user-given title, written now (the row exists once saved)."""
        self._meta_saver(chat.session_id, title=title)
        chat.title = ""

    def _take_sync_keys(self, chat: LiveChat) -> None:
        """Overlay the tracker keys a sync wrote while this turn ran, so the save keeps them."""
        stored = self._loader(chat.session_id) or {}
        for key, value in stored.items():
            if _SYNC_KEY.match(key) and value:
                chat.session.state[key] = value
        chat.stale = False
        logger.info("Chat %s took the synced tracker keys before saving", chat.session_id)

    def close(self, session_id: str) -> None:
        """Forget a conversation (it stays on disk and can be reopened)."""
        with self._lock:
            self._chats.pop(session_id, None)

    def evict(self, session_id: str = "") -> None:
        """Drop a conversation (blank: every one) after its row was rewritten on disk.

        A conversation mid-turn cannot be dropped — the worker still owns it —
        so it is marked stale and takes the rewritten keys when its turn saves.
        """
        with self._lock:
            chats = [self._chats[session_id]] if session_id in self._chats else []
            if not session_id:
                chats = list(self._chats.values())
        for chat in chats:
            if chat.turn.acquire(blocking=False):
                try:
                    self.close(chat.session_id)
                finally:
                    chat.turn.release()
            else:
                chat.stale = True
                logger.info("Chat %s is mid-turn — marked stale rather than evicted", chat.session_id)

    def close_all(self) -> None:
        with self._lock:
            self._chats.clear()

    def delete(self, session_id: str) -> bool:
        """Remove a conversation everywhere: the store, its labels, attachments and log.

        Raises :class:`ChatBusyError` while a turn is running — the worker
        would otherwise write the row straight back.
        """
        with self._lock:
            chat = self._chats.get(session_id)
        if chat is not None:
            if not chat.turn.acquire(blocking=False):
                raise ChatBusyError(f"a turn is running for {session_id}")
            try:
                chat.closed = True
            finally:
                chat.turn.release()
        self.close(session_id)
        with self._store() as store:
            deleted = store.delete_session(session_id)
        if not deleted:
            return False
        try:
            with self._labels() as labels:
                labels.delete("planning", session_id)
        except Exception:  # noqa: BLE001 — the row is gone; a stale label is harmless
            logger.warning("Labels for %s were not removed", session_id, exc_info=True)
        try:
            self._attachment_remover(session_id)
        except OSError:
            logger.warning("Attachments for %s were not removed", session_id, exc_info=True)
        logger.info("Chat deleted: session=%s", session_id)
        return True

    # ------------------------------------------------------------ metadata

    def meta(self, session_id: str) -> dict:
        """The row's title and dates; blank fields for a conversation not yet saved."""
        with self._store() as store:
            row = store.get_session(session_id)
        if row is None:
            return {"title": "", "created_at": "", "last_modified": "", "last_node_completed": ""}
        return {
            "title": row.get("title") or "",
            "created_at": row.get("created_at") or "",
            "last_modified": row.get("last_modified") or "",
            "last_node_completed": row.get("last_node_completed") or "",
        }

    def labels(self, session_id: str) -> dict:
        """``{project_label, tags, scope}`` for a plan; blanks when it carries none."""
        try:
            with self._labels() as labels:
                row = labels.get_labels("planning", session_id)
        except Exception:  # noqa: BLE001 — a label read must not break the view
            logger.warning("Labels for %s could not be read", session_id, exc_info=True)
            row = None
        if row is None:
            return {"project_label": "", "tags": [], "scope": None}
        return {"project_label": row.project, "tags": list(row.tags), "scope": row.scope}

    def link_sessions(self, chat: LiveChat, refs: list[dict]):
        """Pin the plans and runs ``refs`` name on the plan's scope, in state and on its label row.

        Returns the scope that now holds them, or None when nothing was pinnable.
        A label failure never fails the turn.
        """
        from yeaboi.context.scope import MODE_SOURCES, SessionRef, coerce_scope, pin_sessions

        pins: list[SessionRef] = []
        for ref in refs:
            kind = ref.get("kind")
            mode = "planning" if kind == "plan" else ref.get("mode", "")
            key = ref.get("id", "")
            if kind not in ("plan", "run") or not key:
                continue
            if mode not in MODE_SOURCES:
                logger.info("Chat %s: a %s run is not pinnable (no context source), skipped", chat.session_id, mode)
                continue
            if mode in ("planning", "analysis"):
                pins.append(SessionRef(mode=mode, session_id=key))
            else:
                pins.append(SessionRef(mode=mode, run_id=key))
        if not pins:
            return None
        state = chat.session.state
        try:
            stored = coerce_scope(state.get("context_scope") or None)
        except (TypeError, ValueError):
            logger.warning("Chat %s: unreadable context_scope replaced while pinning", chat.session_id)
            stored = None
        scope = pin_sessions(stored, pins)
        state["context_scope"] = json.dumps(scope.to_dict())
        try:
            self.set_labels(chat, scope=scope)
        except Exception:  # noqa: BLE001 — logged, the turn goes on
            logger.warning("Pins for %s were not written to the label row", chat.session_id, exc_info=True)
        logger.info("Chat %s: pinned %d session(s) (%d on the scope)", chat.session_id, len(pins), len(scope.sessions))
        return scope

    def set_labels(
        self,
        chat: LiveChat,
        *,
        project_label: str | None = None,
        tags=None,
        scope=None,
        defaults: bool = False,
        clear_scope: bool = False,
    ) -> dict:
        """Write a plan's labels.

        ``project_label`` None keeps the label and ``""`` clears it; ``tags``
        None leaves the tags alone and a list replaces them; ``defaults`` adds
        the tags every plan gets; ``clear_scope`` records that the plan reads
        unscoped now.
        """
        from yeaboi.context.labels import default_tags

        all_tags = list(tags or ())
        if defaults:
            state = chat.session.state
            all_tags += default_tags(
                "planning",
                today=self._today(),
                world="solo" if state.get("solo") else "team",
                plan_size=state.get("_intake_mode", ""),
            )
        with self._labels() as labels:
            labels.set_labels(
                "planning",
                chat.session_id,
                project=project_label,
                tags=all_tags,
                scope=scope,
                merge_tags=tags is None,
                clear_scope=clear_scope,
            )
        return self.labels(chat.session_id)

    def list_rows(self, *, limit: int = 50, project_label: str = "", tag: str = "") -> list[dict]:
        """The planning hub's rows, newest first, each with its stage and counts."""
        with self._store() as store:
            rows = store.list_sessions(mode="planning")
        try:
            with self._labels() as labels:
                by_id = {row.session_id: row for row in labels.list_labels(mode="planning", limit=0)}
        except Exception:  # noqa: BLE001 — the list renders without labels
            logger.warning("Plan labels could not be listed", exc_info=True)
            by_id = {}
        out: list[dict] = []
        for row in rows:
            label = by_id.get(row["session_id"])
            row_label = label.project if label else ""
            row_tags = list(label.tags) if label else []
            if project_label and row_label.casefold() != project_label.casefold():
                continue
            if tag and tag not in row_tags:
                continue
            state = self._peek(row["session_id"], row.get("session_state_raw") or "")
            out.append(
                {
                    "session_id": row["session_id"],
                    "title": plan_title(
                        row.get("title") or "",
                        row.get("project_name") or "",
                        described_as(state),
                    ),
                    "project_name": row.get("project_name") or "",
                    "project_label": row_label,
                    "tags": row_tags,
                    "stage": stage_of(state) if state is not None else "",
                    "created_at": row.get("created_at") or "",
                    "last_modified": row.get("last_modified") or "",
                    "last_node_completed": row.get("last_node_completed") or "",
                    "counts": {kind: len(state.get(kind) or []) if state else 0 for kind in COUNT_KINDS},
                }
            )
            if limit and len(out) >= limit:
                break
        logger.info("Chat list: %d row(s) (label=%r tag=%r)", len(out), project_label, tag)
        return out

    def _peek(self, session_id: str, raw: str) -> dict | None:
        """The state a list row is summarised from: the live one, else the row's own blob minus its transcript."""
        from yeaboi.sessions import state_from_raw

        with self._lock:
            chat = self._chats.get(session_id)
        if chat is not None:
            return chat.session.state
        if not raw:
            return None
        try:
            data = json.loads(raw)
            data.pop("messages", None)  # the transcript is the bulk of the blob and says nothing a row shows
            return state_from_raw(data)
        except Exception:  # noqa: BLE001 — one unreadable blob must not empty the list
            logger.warning("Plan %s could not be read for the list", session_id, exc_info=True)
            return None

    # ------------------------------------------------------------ versions

    def versions(self, session_id: str, section: str = "") -> list[dict]:
        with self._store() as store:
            return store.list_plan_versions(session_id, section)

    def version(self, session_id: str, section: str, version: int) -> dict | None:
        with self._store() as store:
            return store.get_plan_version(session_id, section, version)

    def version_counts(self, session_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.versions(session_id):
            counts[row["section"]] = max(counts.get(row["section"], 0), int(row["version"]))
        return counts
