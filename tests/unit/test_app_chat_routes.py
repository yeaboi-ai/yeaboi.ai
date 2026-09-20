"""The /api/chat routes — socketless, over AppServer.handle().

The conversation's own behaviour lives in test_chat_session.py; here the
subject is the wire: the session view, the NDJSON turn (its line order, its
op id, its terminators), and the locks that keep two turns off one state.
"""

from __future__ import annotations

import json
import pathlib
import threading

import pytest
from langchain_core.messages import AIMessage

from yeaboi.agent.state import QuestionnaireState
from yeaboi.app.chats import ChatSupervisor, UnknownChatError
from yeaboi.app.router import parse_request
from yeaboi.app.server import AppServer

TOKEN = "test-token"


class FakeGraph:
    """Answers every turn with one scripted reply, recording the invokes."""

    def __init__(self):
        self.invocations: list[dict] = []
        self.gate = threading.Event()
        self.gate.set()

    def invoke(self, state: dict) -> dict:
        self.invocations.append(state)
        self.gate.wait(timeout=5)
        qs = QuestionnaireState(intake_mode="smart", current_question=2)
        return {**state, "questionnaire": qs, "messages": [*state["messages"], AIMessage(content="How many of you?")]}


@pytest.fixture
def graph():
    return FakeGraph()


@pytest.fixture
def app(graph, tmp_path):
    """State in a dict; the row-level stores (titles, versions, labels) on a throwaway db."""
    from yeaboi.context.labels import LabelStore
    from yeaboi.sessions import SessionStore

    saved: dict[str, dict] = {}
    db = tmp_path / "sessions.db"
    chats = ChatSupervisor(
        graph_factory=lambda: graph,
        store_factory=lambda: SessionStore(db),
        label_store_factory=lambda: LabelStore(db),
        loader=saved.get,
        saver=saved.__setitem__,
        id_factory=lambda: "proj-1",
    )
    server = AppServer(token=TOKEN, chats=chats)
    server.saved = saved  # the tests assert on what was persisted
    server.db = db
    return server


@pytest.fixture
def store_app(graph, tmp_path, monkeypatch):
    """Everything on the session store — what the room's list, rename and delete read."""
    db = tmp_path / "sessions.db"
    monkeypatch.setattr("yeaboi.paths.get_db_path", lambda: db)
    ids = iter(f"new-{n:08x}-2026-09-12" for n in range(1, 100))
    removed: list[str] = []
    chats = ChatSupervisor(graph_factory=lambda: graph, id_factory=lambda: next(ids), attachment_remover=removed.append)
    server = AppServer(token=TOKEN, chats=chats)
    server.db = db
    server.removed = removed
    return server


def request(app: AppServer, method: str, path: str, payload: dict | None = None, *, authed: bool = True):
    headers = {"Authorization": f"Bearer {TOKEN}"} if authed else {}
    body = json.dumps(payload).encode() if payload is not None else b""
    return app.handle(parse_request(method, path, headers, body))


def open_chat(app, description="a booking app for barbers", **kw):
    resp = request(app, "POST", "/api/chat/sessions", {"description": description, "intake_mode": "smart", **kw})
    assert resp.code == 201, resp.body
    return json.loads(resp.body)


def turn(app, project_id="proj-1", text="four engineers"):
    resp = request(app, "POST", f"/api/chat/sessions/{project_id}/send", {"text": text})
    assert resp.code == 200, resp.body
    assert resp.content_type == "application/x-ndjson"
    return [json.loads(line) for line in b"".join(resp.stream).decode().splitlines()]


class TestCreate:
    def test_requires_auth(self, app):
        assert request(app, "POST", "/api/chat/sessions", {"description": "x"}, authed=False).code == 401

    def test_a_description_is_required(self, app):
        assert request(app, "POST", "/api/chat/sessions", {"description": "   "}).code == 400

    def test_an_unknown_size_is_refused(self, app):
        assert request(app, "POST", "/api/chat/sessions", {"description": "x", "intake_mode": "huge"}).code == 400

    def test_the_view_opens_on_the_greeting(self, app):
        view = open_chat(app)
        assert view["project_id"] == "proj-1"
        assert view["stage"] == "intake"
        assert [item["type"] for item in view["transcript"]] == ["assistant", "assistant"]
        # The description is messages[0], never a preamble echo — it would
        # otherwise appear twice the moment the first turn lands.
        assert not any("barbers" in item["text"] for item in view["transcript"])

    def test_the_description_is_owed_as_the_first_turn(self, app):
        # It has to reach the graph as messages[0]; until it does, the view
        # carries it so the client knows to send it.
        assert open_chat(app)["opening"] == "a booking app for barbers"

    def test_the_opening_is_spent_once_it_has_been_sent(self, app):
        open_chat(app)
        turn(app, text="a booking app for barbers")
        assert json.loads(request(app, "GET", "/api/chat/sessions/proj-1").body)["opening"] == ""

    def test_a_new_conversation_is_persisted_immediately(self, app):
        open_chat(app)
        assert "proj-1" in app.saved


class TestSessionView:
    def test_an_unknown_conversation_is_a_404(self, app):
        assert request(app, "GET", "/api/chat/sessions/nope").code == 404

    def test_the_view_replays_the_turn(self, app):
        open_chat(app)
        turn(app)
        view = json.loads(request(app, "GET", "/api/chat/sessions/proj-1").body)
        kinds = [item["type"] for item in view["transcript"]]
        assert kinds == ["assistant", "assistant", "user", "assistant"]
        assert view["transcript"][-1]["text"] == "How many of you?"
        assert view["question"]["current_question"] == 2


class TestTurnStream:
    def test_the_op_id_leads_and_done_terminates(self, app):
        open_chat(app)
        lines = turn(app)
        assert lines[0]["type"] == "op" and lines[0]["op_id"]
        assert lines[-1] == {"stage": "intake", "type": "done"}

    def test_a_deterministic_reply_lands_once_as_the_question_line(self, app):
        # No typewriter over the wire: the text is on the question line, not
        # paced out as tokens first.
        open_chat(app)
        lines = turn(app)
        assert [line for line in lines if line["type"] == "token"] == []
        question = next(line for line in lines if line["type"] == "question")
        assert question["number"] == 2 and "How many of you?" in question["text"]

    def test_the_text_reaches_the_graph_and_the_state_is_saved(self, app, graph):
        open_chat(app)
        turn(app, text="four engineers")
        assert graph.invocations[-1]["messages"][-1].content == "four engineers"
        assert app.saved["proj-1"]["messages"]

    def test_a_provider_failure_lands_as_one_classified_line(self, app, graph, monkeypatch):
        open_chat(app)
        monkeypatch.setattr(graph, "invoke", lambda _state: (_ for _ in ()).throw(RuntimeError("boom")))
        lines = turn(app)
        assert lines[-1]["type"] == "error"
        # One classified human line, not a traceback and not a raw SDK dump.
        assert lines[-1]["message"].startswith("Unexpected error")
        assert len([line for line in lines if line["type"] == "error"]) == 1

    def test_a_second_turn_is_refused_while_one_is_running(self, app, graph):
        open_chat(app)
        graph.gate.clear()  # park the first turn inside the graph
        first = request(app, "POST", "/api/chat/sessions/proj-1/send", {"text": "one"})
        lines = iter(first.stream)
        assert json.loads(next(lines))["type"] == "op"  # the turn is in flight
        assert request(app, "POST", "/api/chat/sessions/proj-1/send", {"text": "two"}).code == 409
        graph.gate.set()
        b"".join(lines)  # drain, releasing the turn lock

    def test_the_turn_lock_and_op_are_released_after_the_stream(self, app):
        open_chat(app)
        lines = turn(app)
        op_id = lines[0]["op_id"]
        assert app.ops.get(op_id) is None
        assert request(app, "POST", "/api/chat/sessions/proj-1/send", {"text": "again"}).code == 200


class TestSupervisor:
    def test_one_live_session_per_conversation(self, app):
        open_chat(app)
        assert app.chats.open("proj-1") is app.chats.open("proj-1")

    def test_a_closed_conversation_resumes_from_disk(self, app):
        open_chat(app)
        turn(app)
        app.chats.close("proj-1")
        resumed = app.chats.open("proj-1")
        assert resumed.session.state["messages"]


class TestQuestionPlan:
    def test_a_pre_graph_conversation_has_no_plan_yet(self, app):
        # The questionnaire only exists after the first invoke, so there is
        # nothing to list — an empty plan, not a 404.
        open_chat(app)
        plan = json.loads(request(app, "GET", "/api/chat/sessions/proj-1/questions").body)
        assert plan == {"questions": [], "total": 30, "completed": False, "derived": False}

    def test_it_lists_what_the_run_answered_and_still_asks(self, app, monkeypatch):
        open_chat(app)
        turn(app)
        chat = app.chats.open("proj-1")
        chat.session.state["questionnaire"].answers = {1: "a booking app", 6: "four"}
        monkeypatch.setattr(
            "yeaboi.ui.session.chat._question_view.planned_question_sets",
            lambda qs: ([7], {1, 6}),
        )
        plan = json.loads(request(app, "GET", "/api/chat/sessions/proj-1/questions").body)
        assert [row["number"] for row in plan["questions"]] == [1, 6, 7]
        assert plan["questions"][0]["answer"] == "a booking app"
        # 7 is a gap: planned, not yet answered.
        assert plan["questions"][2]["remaining"] and plan["questions"][2]["answer"] == ""
        assert plan["questions"][0]["label"] and plan["derived"]

    def test_a_failed_derivation_says_so_rather_than_shrinking_the_plan(self, app, monkeypatch):
        # Falling back silently would present the answers as the whole plan.
        open_chat(app)
        turn(app)
        app.chats.open("proj-1").session.state["questionnaire"].answers = {1: "a booking app"}
        monkeypatch.setattr(
            "yeaboi.ui.session.chat._question_view.planned_question_sets",
            lambda qs: None,
        )
        plan = json.loads(request(app, "GET", "/api/chat/sessions/proj-1/questions").body)
        assert plan["derived"] is False
        assert [row["number"] for row in plan["questions"]] == [1]

    def test_an_unknown_conversation_is_a_404(self, app):
        assert request(app, "GET", "/api/chat/sessions/nope/questions").code == 404


class TestSizeSwitch:
    def test_an_unknown_size_is_refused(self, app):
        open_chat(app)
        assert request(app, "POST", "/api/chat/sessions/proj-1/size", {"mode": "huge"}).code == 400

    def test_switching_to_the_size_it_already_is_changes_nothing(self, app):
        open_chat(app)
        body = json.loads(request(app, "POST", "/api/chat/sessions/proj-1/size", {"mode": "smart"}).body)
        assert body == {"changed": False, "mode": "smart"}

    def test_before_the_intake_it_only_records_the_preference(self, app):
        open_chat(app)
        body = json.loads(request(app, "POST", "/api/chat/sessions/proj-1/size", {"mode": "small_project"}).body)
        assert body["changed"] and body["reopened"] is False
        assert app.chats.open("proj-1").session.state["_intake_mode"] == "small_project"

    def test_a_real_switch_keeps_the_answers(self, app):
        open_chat(app)
        turn(app)
        state = app.chats.open("proj-1").session.state
        state["questionnaire"].answers = {1: "a booking app"}
        body = json.loads(request(app, "POST", "/api/chat/sessions/proj-1/size", {"mode": "small_project"}).body)
        assert body["changed"] and body["reopened"]
        assert state["questionnaire"].answers == {1: "a booking app"}
        assert state["_intake_mode"] == "small_project"

    def test_the_switch_is_persisted(self, app):
        open_chat(app)
        request(app, "POST", "/api/chat/sessions/proj-1/size", {"mode": "small_project"})
        assert "proj-1" in app.saved


class TestAttachments:
    def _post(self, app, **payload):
        return request(app, "POST", "/api/chat/sessions/proj-1/attachments", payload)

    def test_a_pasted_image_is_kept_and_chipped(self, app, tmp_path, monkeypatch):
        import base64

        monkeypatch.setattr("yeaboi.paths.get_attachments_dir", lambda scope: tmp_path)
        open_chat(app)
        body = json.loads(self._post(app, image=base64.b64encode(b"PNGDATA").decode(), index=2).body)
        assert body["chip"] == "[image #2]"
        assert pathlib.Path(body["path"]).read_bytes() == b"PNGDATA"
        assert pathlib.Path(body["path"]).suffix == ".png"

    def test_a_jpeg_keeps_its_own_extension(self, app, tmp_path, monkeypatch):
        import base64

        monkeypatch.setattr("yeaboi.paths.get_attachments_dir", lambda scope: tmp_path)
        open_chat(app)
        body = json.loads(self._post(app, image=base64.b64encode(b"JPG").decode(), mime="image/jpeg", index=1).body)
        assert pathlib.Path(body["path"]).suffix == ".jpg"

    def test_a_pdf_is_not_an_image(self, app):
        open_chat(app)
        assert self._post(app, image="AAAA", mime="application/pdf", index=1).code == 400

    def test_garbage_is_refused_rather_than_written(self, app):
        open_chat(app)
        assert self._post(app, image="not base64!!", index=1).code == 400
        assert self._post(app, image="", index=1).code == 400

    def test_an_oversized_image_is_refused_at_paste_time(self, app):
        import base64

        from yeaboi.ui.shared._attachments import MAX_IMAGE_BYTES

        open_chat(app)
        big = base64.b64encode(b"x" * (MAX_IMAGE_BYTES + 1)).decode()
        resp = self._post(app, image=big, index=1)
        assert resp.code == 413 and "4.5 MB" in resp.body.decode()


class TestImagesFollowTheirChips:
    """Deleting an ``[image #N]`` chip detaches its image — the terminal's rule."""

    @staticmethod
    def _sent(graph) -> list[str]:
        """The images the last invoke carried, whichever slot the stage puts them in."""
        state = graph.invocations[-1]
        return list(state.get("chat_images") or state.get("pasted_images") or [])

    def test_only_the_chipped_attachments_travel(self, app, graph):
        open_chat(app)
        turn(app)
        resp = request(
            app,
            "POST",
            "/api/chat/sessions/proj-1/send",
            {"text": "look at [image #2]", "images": ["/tmp/a.png", "/tmp/b.png"]},
        )
        b"".join(resp.stream)
        assert self._sent(graph) == ["/tmp/b.png"]

    def test_an_attachment_with_no_surviving_chip_is_dropped(self, app, graph):
        open_chat(app)
        turn(app)
        resp = request(
            app,
            "POST",
            "/api/chat/sessions/proj-1/send",
            {"text": "never mind", "images": ["/tmp/a.png"]},
        )
        b"".join(resp.stream)
        assert self._sent(graph) == []


class TestTextAttachments:
    def _post(self, app, **payload):
        return request(app, "POST", "/api/chat/sessions/proj-1/attachments", {"kind": "text", **payload})

    def test_a_text_file_is_kept_and_chipped(self, app, tmp_path, monkeypatch):
        monkeypatch.setattr("yeaboi.paths.get_attachments_dir", lambda scope: tmp_path)
        open_chat(app)
        body = json.loads(self._post(app, name="../notes.md", text="# hi\n", index=2).body)
        assert body["chip"] == "[file #2]" and body["kind"] == "text" and body["name"] == "notes.md"
        assert pathlib.Path(body["path"]).read_text() == "# hi\n"
        assert pathlib.Path(body["path"]).name.endswith("-notes.md")

    def test_default_kind_is_still_image(self, app, tmp_path, monkeypatch):
        import base64

        monkeypatch.setattr("yeaboi.paths.get_attachments_dir", lambda scope: tmp_path)
        open_chat(app)
        resp = request(
            app,
            "POST",
            "/api/chat/sessions/proj-1/attachments",
            {"image": base64.b64encode(b"PNG").decode(), "index": 1},
        )
        assert json.loads(resp.body)["chip"] == "[image #1]"

    def test_a_bad_suffix_is_a_400(self, app):
        open_chat(app)
        assert self._post(app, name="tool.exe", text="x", index=1).code == 400
        assert self._post(app, name="noext", text="x", index=1).code == 400

    def test_over_200kb_is_a_413(self, app):
        from yeaboi.ui.shared._attachments import MAX_TEXT_FILE_BYTES

        open_chat(app)
        assert self._post(app, name="big.log", text="x" * (MAX_TEXT_FILE_BYTES + 1), index=1).code == 413

    def test_an_unknown_kind_or_a_blank_text_is_a_400(self, app):
        open_chat(app)
        assert request(app, "POST", "/api/chat/sessions/proj-1/attachments", {"kind": "pdf"}).code == 400
        assert self._post(app, name="a.md", text="   ", index=1).code == 400
        assert self._post(app, name="a.md", text=3, index=1).code == 400


class TestFilesFollowTheirChips:
    @staticmethod
    def _sent(graph) -> list[str]:
        state = graph.invocations[-1]
        return list(state.get("chat_context") or state.get("pasted_context") or [])

    def _send(self, app, text, files):
        resp = request(app, "POST", "/api/chat/sessions/proj-1/send", {"text": text, "files": files})
        assert resp.code == 200, resp.body
        b"".join(resp.stream)

    def test_only_chipped_files_travel(self, app, graph, tmp_path, monkeypatch):
        monkeypatch.setattr("yeaboi.paths.get_attachments_dir", lambda scope: tmp_path)
        open_chat(app)
        a, b = tmp_path / "a.md", tmp_path / "b.md"
        a.write_text("alpha")
        b.write_text("beta")
        self._send(app, "read [file #2]", [str(a), str(b)])
        assert self._sent(graph) == ["File 1 (b.md):\nbeta"]

    def test_a_path_outside_the_attachments_dir_is_refused(self, app, graph, tmp_path, monkeypatch):
        monkeypatch.setattr("yeaboi.paths.get_attachments_dir", lambda scope: tmp_path / "inside")
        (tmp_path / "inside").mkdir()
        open_chat(app)
        outside = tmp_path / "secret.md"
        outside.write_text("nope")
        self._send(app, "[file #1] [file #2]", [str(outside), str(tmp_path / "inside" / "x.exe")])
        assert self._sent(graph) == []

    def test_files_must_be_a_list(self, app):
        open_chat(app)
        assert request(app, "POST", "/api/chat/sessions/proj-1/send", {"text": "x", "files": "a"}).code == 400


class TestRefs:
    LINK = {"kind": "link", "label": "spec", "url": "https://example.com/spec"}

    @staticmethod
    def _context(graph) -> list[str]:
        state = graph.invocations[-1]
        return list(state.get("chat_context") or state.get("pasted_context") or [])

    def _send(self, app, text, refs):
        resp = request(app, "POST", "/api/chat/sessions/proj-1/send", {"text": text, "refs": refs})
        assert resp.code == 200, resp.body
        b"".join(resp.stream)

    def test_only_chipped_refs_travel(self, app, graph):
        open_chat(app)
        other = {"kind": "link", "label": "other", "url": "https://example.com/other"}
        self._send(app, "see [ref #2]", [other, self.LINK])
        assert self._context(graph) == ["Reference 1 (link): spec — https://example.com/spec"]

    def test_refs_ride_the_intake_channel_during_intake(self, app, graph):
        open_chat(app)
        self._send(app, "[ref #1]", [self.LINK])
        assert "pasted_context" in graph.invocations[-1] and "chat_context" not in graph.invocations[-1]

    def test_a_bad_kind_and_too_many_are_400s(self, app):
        from yeaboi.agent.chat_refs import MAX_REFS

        open_chat(app)
        bad = request(app, "POST", "/api/chat/sessions/proj-1/send", {"text": "x", "refs": [{"kind": "video"}]})
        assert bad.code == 400 and "kind" in json.loads(bad.body)["error"]
        many = request(
            app, "POST", "/api/chat/sessions/proj-1/send", {"text": "x", "refs": [self.LINK] * (MAX_REFS + 1)}
        )
        assert many.code == 400

    def test_refs_on_create_are_read_into_the_intake_once(self, app, graph):
        open_chat(app, refs=[self.LINK])
        assert app.saved["proj-1"]["pasted_context"] == ["Reference 1 (link): spec — https://example.com/spec"]
        turn(app)
        assert graph.invocations[-1]["pasted_context"] == ["Reference 1 (link): spec — https://example.com/spec"]


class TestLinkedSessions:
    PLAN = {"kind": "plan", "label": "Apollo", "id": "new-9"}
    RUN = {"kind": "run", "mode": "standup", "id": "4", "label": "standup"}

    def test_a_plan_ref_is_recorded_on_the_scope_and_the_label_row(self, app):
        from yeaboi.context.labels import LabelStore

        open_chat(app)
        resp = request(
            app, "POST", "/api/chat/sessions/proj-1/send", {"text": "[ref #1] [ref #2]", "refs": [self.PLAN, self.RUN]}
        )
        b"".join(resp.stream)
        scope = json.loads(app.chats.open("proj-1").session.state["context_scope"])
        assert scope["sessions"] == [
            {"mode": "planning", "session_id": "new-9", "run_id": ""},
            {"mode": "standup", "session_id": "", "run_id": "4"},
        ]
        with LabelStore(app.db) as labels:
            assert labels.get_labels("planning", "proj-1").scope["sessions"][0]["session_id"] == "new-9"

    def test_a_latest_run_and_a_link_pin_nothing(self, app):
        open_chat(app)
        refs = [{"kind": "run", "mode": "retro", "label": "retro"}, TestRefs.LINK]
        resp = request(app, "POST", "/api/chat/sessions/proj-1/send", {"text": "[ref #1] [ref #2]", "refs": refs})
        b"".join(resp.stream)
        assert "context_scope" not in app.chats.open("proj-1").session.state

    def test_an_update_without_sessions_keeps_the_pins(self, app):
        open_chat(app, refs=[self.PLAN])
        assert json.loads(app.saved["proj-1"]["context_scope"])["sessions"][0]["session_id"] == "new-9"
        body = json.loads(
            request(app, "POST", "/api/chat/sessions/proj-1/update", {"context": {"sources": ["retro"]}}).body
        )
        assert body["context"]["sources"] == ["retro"]
        assert body["context"]["sessions"][0]["session_id"] == "new-9"

    def test_an_explicit_empty_list_clears_them(self, app):
        open_chat(app, refs=[self.PLAN])
        body = json.loads(request(app, "POST", "/api/chat/sessions/proj-1/update", {"context": {"sessions": []}}).body)
        assert body["context"]["sessions"] == []


class TestIntegrations:
    def test_absent_key_leaves_the_state_unrestricted(self, app):
        view = open_chat(app)
        assert view["integrations"] is None
        assert "session_integrations" not in app.saved["proj-1"]

    def test_a_list_is_kept_and_shown(self, app, graph):
        view = open_chat(app, integrations=["jira", "github"])
        assert view["integrations"] == ["jira", "github"]
        assert app.saved["proj-1"]["session_integrations"] == ["jira", "github"]
        turn(app)
        assert graph.invocations[-1]["session_integrations"] == ["jira", "github"]
        assert open_chat(app, integrations=[])["integrations"] == []

    def test_an_unknown_key_is_a_400(self, app):
        resp = request(app, "POST", "/api/chat/sessions", {"description": "x", "integrations": ["fax"]})
        assert resp.code == 400 and "fax" in json.loads(resp.body)["error"]
        assert not app.saved

    def test_update_replaces_and_null_clears(self, app):
        open_chat(app, integrations=["jira"])
        body = json.loads(request(app, "POST", "/api/chat/sessions/proj-1/update", {"integrations": ["notion"]}).body)
        assert body["integrations"] == ["notion"]
        assert app.saved["proj-1"]["session_integrations"] == ["notion"]
        body = json.loads(request(app, "POST", "/api/chat/sessions/proj-1/update", {"integrations": None}).body)
        assert body["integrations"] is None
        assert "session_integrations" not in app.saved["proj-1"]
        assert json.loads(request(app, "GET", "/api/chat/sessions/proj-1").body)["integrations"] is None


class TestSoloConversations:
    def test_solo_true_seeds_the_state_key(self, app):
        open_chat(app, solo=True)
        assert app.saved["proj-1"]["solo"] is True

    def test_the_default_is_a_team_conversation(self, app):
        open_chat(app)
        assert "solo" not in app.saved["proj-1"]


class TestSupervisorOnSessionStore:
    """The default loader/saver/recorder — the session store, not the file store."""

    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        db = tmp_path / "sessions.db"
        monkeypatch.setattr("yeaboi.paths.get_db_path", lambda: db)
        return db

    def _supervisor(self, graph):
        return ChatSupervisor(graph_factory=lambda: graph, id_factory=lambda: "new-aaaa1111-2026-09-11")

    def test_create_and_save_write_one_planning_row_with_its_title(self, db, graph):
        from yeaboi.sessions import SessionStore

        chats = self._supervisor(graph)
        chat = chats.create("a booking app", intake_mode="smart", title="Barbers", project_label="apollo")
        assert chat.session.state["project_label"] == "apollo"
        chats.save(chat)
        with SessionStore(db) as store:
            (row,) = store.list_sessions(mode="planning")
            assert row["session_id"] == chat.session_id and row["title"] == "Barbers"
            assert store.load_state(chat.session_id)["_chat_opening"] == "a booking app"

    def test_a_closed_conversation_reopens_from_the_store(self, db, graph):
        chats = self._supervisor(graph)
        chat = chats.create("a booking app", intake_mode="smart")
        chat.session.state["solo"] = True
        chats.save(chat)
        chats.close(chat.session_id)
        again = chats.open(chat.session_id)
        assert again is not chat and again.session.state["solo"] is True

    def test_an_accepted_section_lands_in_plan_versions(self, db, graph):
        from yeaboi.sessions import SessionStore

        chats = self._supervisor(graph)
        chat = chats.create("a booking app", intake_mode="smart")
        chats.save(chat)
        chat.session.state.update(pending_review="story_writer", stories=[])
        chat.session.reply("accept", lambda _e: None)
        with SessionStore(db) as store:
            assert [v["section"] for v in store.list_plan_versions(chat.session_id)] == ["stories"]

    def test_a_file_store_conversation_still_opens(self, db, graph, monkeypatch):
        monkeypatch.setattr(
            "yeaboi.persistence.load_graph_state", lambda pid: {"messages": [], "_intake_mode": "smart"}
        )
        chat = self._supervisor(graph).open("uuid-from-before")
        assert chat.session.state["_intake_mode"] == "smart"

    def test_an_unknown_id_is_unknown_everywhere(self, db, graph, monkeypatch):
        monkeypatch.setattr("yeaboi.persistence.load_graph_state", lambda pid: None)
        with pytest.raises(UnknownChatError):
            self._supervisor(graph).open("nope")

    def test_the_project_name_and_last_node_follow_the_state(self, db, graph):
        from tests._node_helpers import make_dummy_analysis
        from yeaboi.sessions import SessionStore

        chats = self._supervisor(graph)
        chat = chats.create("a booking app", intake_mode="smart")
        chat.session.state["project_analysis"] = make_dummy_analysis()
        chats.save(chat)
        with SessionStore(db) as store:
            row = store.get_session(chat.session_id)
        assert row["project_name"] == make_dummy_analysis().project_name
        assert row["last_node_completed"] == "project_analyzer"


# ---------------------------------------------------------------------------
# The planning room: list, update, delete, plan, versions, commands, advance.
# ---------------------------------------------------------------------------


class PipelineGraph:
    """A graph whose one step produces the analysis and parks on its gate."""

    def __init__(self):
        self.invocations: list[dict] = []

    def invoke(self, state: dict) -> dict:
        from tests._node_helpers import make_dummy_analysis

        self.invocations.append(state)
        return {
            **state,
            "project_analysis": make_dummy_analysis(),
            "pending_review": "project_analyzer",
            "messages": [*state["messages"], AIMessage(content="## Analysis\nA booking app.")],
        }


def _stories(count: int) -> list:
    from yeaboi.agent.state import Priority, StoryPointValue, UserStory

    return [
        UserStory(
            id=f"S{n}",
            feature_id="F1",
            persona="client",
            goal="book",
            benefit="time",
            acceptance_criteria=(),
            story_points=StoryPointValue.ONE,
            priority=Priority.LOW,
        )
        for n in range(count)
    ]


def _completed_questionnaire() -> QuestionnaireState:
    qs = QuestionnaireState(intake_mode="smart", current_question=31)
    qs.completed = True
    qs.answers = {1: "A booking app"}
    return qs


class TestSessionViewShape:
    def test_the_keys_are_the_pinned_ones_in_order(self, app):
        from yeaboi.app.routes_chat import SESSION_VIEW_KEYS

        view = open_chat(app)
        assert set(view) == set(SESSION_VIEW_KEYS)
        assert view["session_id"] == view["project_id"] == "proj-1"
        assert view["pending"] is None and view["intake_mode"] == "smart"
        assert view["progress"]["step"] == 1 and view["progress"]["total"] == 6
        assert [s["kind"] for s in view["sections"]][:2] == ["intake", "analysis"]
        assert {s["status"] for s in view["sections"]} == {"empty"}

    def test_the_title_falls_back_to_the_description_then_the_analysis(self, store_app):
        from yeaboi.agent.state import ProjectAnalysis

        view = open_chat(store_app, description="A booking app for barbers. Payments later.")
        assert view["title"] == "A booking app for barbers."
        sid = view["session_id"]
        chat = store_app.chats.open(sid)
        chat.session.state["project_analysis"] = ProjectAnalysis(
            project_name="Barbershop",
            project_description="",
            project_type="greenfield",
            goals=(),
            end_users=(),
            target_state="",
            tech_stack=(),
            integrations=(),
            constraints=(),
            sprint_length_weeks=2,
            risks=(),
            out_of_scope=(),
            assumptions=(),
            target_sprints=3,
        )
        view = json.loads(request(store_app, "GET", f"/api/chat/sessions/{sid}").body)
        assert view["title"] == "Barbershop"
        request(store_app, "POST", f"/api/chat/sessions/{sid}/update", {"title": "Mine"})
        view = json.loads(request(store_app, "GET", f"/api/chat/sessions/{sid}").body)
        assert view["title"] == "Mine"

    def test_a_parked_gate_is_the_pending_line(self, app):
        open_chat(app)
        chat = app.chats.open("proj-1")
        chat.session.state.update(pending_review="story_writer", stories=[])
        view = json.loads(request(app, "GET", "/api/chat/sessions/proj-1").body)
        assert view["stage"] == "review"
        assert view["pending"]["type"] == "await_review" and view["pending"]["kind"] == "stories"
        assert view["transcript"][-1] == view["pending"]
        assert {s["kind"]: s["status"] for s in view["sections"]}["stories"] == "awaiting_review"


class TestCreateSeedsProfileAndContext:
    def test_the_three_context_keys_reach_the_state_and_the_labels(self, app):
        from yeaboi.context.labels import LabelStore

        open_chat(app, context="standup,retro:1@2sprints", project_label="Apollo", tags=["Q3"])
        state = app.saved["proj-1"]
        assert state["project_label"] == "Apollo"
        assert set(json.loads(state["context_scope"])["sources"]) == {"retro", "standup"}
        with LabelStore(app.db) as labels:
            row = labels.get_labels("planning", "proj-1")
        assert row.project == "Apollo"
        assert {"q3", "mode:planning", "world:team", "size:large"} <= set(row.tags)
        assert row.scope["window"]["kind"] == "sprints"
        view = json.loads(request(app, "GET", "/api/chat/sessions/proj-1").body)
        assert view["project_label"] == "Apollo" and "q3" in view["tags"]

    def test_a_bad_context_is_refused_before_anything_is_created(self, app):
        resp = request(app, "POST", "/api/chat/sessions", {"description": "x", "context": "stanup"})
        assert resp.code == 400 and "standup" in json.loads(resp.body)["error"]
        assert not app.saved

    def test_an_analysis_profile_must_exist(self, app, monkeypatch):
        monkeypatch.setattr("yeaboi.app.routes_chat._profile_exists", lambda pid: pid == "jira-PROJ-1")
        resp = request(app, "POST", "/api/chat/sessions", {"description": "x", "analysis_profile_id": "nope"})
        assert resp.code == 400
        view = open_chat(app, analysis_profile_id="jira-PROJ-1")
        assert view["session_id"] == "proj-1"
        assert app.saved["proj-1"]["analysis_profile_id"] == "jira-PROJ-1"

    def test_a_title_is_kept(self, store_app):
        view = open_chat(store_app, title="  Barbers  ")
        assert view["title"] == "Barbers"
        again = json.loads(request(store_app, "GET", f"/api/chat/sessions/{view['session_id']}").body)
        assert again["title"] == "Barbers" and again["created_at"]

    def test_a_label_failure_does_not_fail_the_plan(self, app, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("labels down")

        monkeypatch.setattr(app.chats, "set_labels", boom)
        assert open_chat(app, project_label="Apollo")["session_id"] == "proj-1"

    def test_no_context_inherits_the_scope_last_used_for_planning(self, app, monkeypatch):
        monkeypatch.setattr("yeaboi.config.get_last_context_scope", lambda mode: {"sources": ["retro"]})
        open_chat(app)
        assert json.loads(app.saved["proj-1"]["context_scope"])["sources"] == ["retro"]
        monkeypatch.setattr("yeaboi.config.get_last_context_scope", lambda mode: None)
        open_chat(app)  # the same fixture id — a fresh state
        assert "context_scope" not in app.saved["proj-1"]


class TestSendRefusesSlash:
    def test_slash_input_never_reaches_the_model(self, app, graph):
        open_chat(app)
        resp = request(app, "POST", "/api/chat/sessions/proj-1/send", {"text": "/help"})
        assert resp.code == 400 and "commands" in json.loads(resp.body)["error"]
        assert graph.invocations == []

    def test_a_build_stage_wants_advance(self, app):
        open_chat(app)
        app.chats.open("proj-1").session.state["questionnaire"] = _completed_questionnaire()
        resp = request(app, "POST", "/api/chat/sessions/proj-1/send", {"text": "hello"})
        assert resp.code == 409 and "advance" in json.loads(resp.body)["error"]


class TestAdvance:
    @pytest.fixture
    def graph(self):
        return PipelineGraph()

    def test_it_runs_one_step_and_parks_on_the_gate(self, app, graph):
        open_chat(app)
        app.chats.open("proj-1").session.state["questionnaire"] = _completed_questionnaire()
        resp = request(app, "POST", "/api/chat/sessions/proj-1/advance")
        assert resp.code == 200 and resp.content_type == "application/x-ndjson"
        lines = [json.loads(line) for line in b"".join(resp.stream).decode().splitlines()]
        kinds = [line["type"] for line in lines]
        assert kinds[0] == "op" and kinds[-1] == "done"
        assert kinds[1:3] == ["progress", "section"]
        assert lines[1]["node"] == "project_analyzer" and lines[1]["status"] == "running"
        assert "await_review" in kinds and "artifact" in kinds
        assert lines[-1]["stage"] == "review"
        assert len(graph.invocations) == 1
        assert app.saved["proj-1"]["pending_review"] == "project_analyzer"

    def test_it_is_refused_while_a_reply_is_owed(self, app, graph):
        open_chat(app)
        resp = request(app, "POST", "/api/chat/sessions/proj-1/advance")
        assert resp.code == 409
        assert graph.invocations == []

    def test_an_unknown_conversation_is_a_404(self, app):
        assert request(app, "POST", "/api/chat/sessions/nope/advance").code == 404


class TestList:
    def test_the_label_filter_ignores_case(self, store_app):
        open_chat(store_app, project_label="Apollo")
        rows = json.loads(request(store_app, "GET", "/api/chat/sessions?project_label=apollo").body)["sessions"]
        assert [row["project_label"] for row in rows] == ["Apollo"]

    def test_plans_list_newest_first_with_their_stage_and_counts(self, store_app):
        first = open_chat(store_app, title="First")
        second = open_chat(store_app, title="Second")
        chat = store_app.chats.open(second["session_id"])
        chat.session.state.update(pending_review="story_writer", stories=_stories(2))
        store_app.chats.save(chat)
        rows = json.loads(request(store_app, "GET", "/api/chat/sessions").body)["sessions"]
        assert [row["title"] for row in rows] == ["Second", "First"]
        assert rows[0]["stage"] == "review" and rows[0]["counts"]["stories"] == 2
        assert rows[1]["session_id"] == first["session_id"] and rows[1]["stage"] == "intake"
        assert set(rows[0]) == {
            "session_id",
            "title",
            "project_name",
            "project_label",
            "tags",
            "stage",
            "created_at",
            "last_modified",
            "last_node_completed",
            "counts",
        }

    def test_limit_label_and_tag_narrow_the_list(self, store_app):
        open_chat(store_app, title="Apollo plan", project_label="Apollo", tags=["q3"])
        open_chat(store_app, title="Other")
        by_limit = json.loads(request(store_app, "GET", "/api/chat/sessions?limit=1").body)["sessions"]
        assert len(by_limit) == 1
        by_label = json.loads(request(store_app, "GET", "/api/chat/sessions?project_label=Apollo").body)["sessions"]
        assert [row["title"] for row in by_label] == ["Apollo plan"]
        by_tag = json.loads(request(store_app, "GET", "/api/chat/sessions?tag=q3").body)["sessions"]
        assert [row["title"] for row in by_tag] == ["Apollo plan"]
        assert request(store_app, "GET", "/api/chat/sessions?limit=x").code == 400

    def test_a_plan_without_a_title_is_named_from_its_description(self, store_app):
        open_chat(store_app, description="A booking app for barbers. Payments later.")
        rows = json.loads(request(store_app, "GET", "/api/chat/sessions").body)["sessions"]
        assert rows[0]["title"] == "A booking app for barbers."

    def test_an_empty_store_is_an_empty_list(self, store_app):
        assert json.loads(request(store_app, "GET", "/api/chat/sessions").body) == {"sessions": []}


class TestUpdate:
    def test_only_the_keys_present_change(self, store_app):
        sid = open_chat(store_app, title="Old", project_label="Apollo", tags=["q3"])["session_id"]
        resp = request(store_app, "POST", f"/api/chat/sessions/{sid}/update", {"title": "New"})
        body = json.loads(resp.body)
        assert resp.code == 200 and body["title"] == "New" and body["project_label"] == "Apollo"
        assert "q3" in body["tags"]
        view = json.loads(request(store_app, "GET", f"/api/chat/sessions/{sid}").body)
        assert view["title"] == "New"

    def test_tags_replace_and_a_blank_context_clears_the_scope(self, store_app):
        sid = open_chat(store_app, context="standup@month", tags=["q3"])["session_id"]
        body = json.loads(
            request(
                store_app,
                "POST",
                f"/api/chat/sessions/{sid}/update",
                {"tags": ["mode:planning", "Team A"], "context": "", "project_label": "Borealis"},
            ).body
        )
        assert body["tags"] == ["mode:planning", "team-a"] and body["context"] is None
        assert body["project_label"] == "Borealis"
        state = store_app.chats.open(sid).session.state
        assert "context_scope" not in state and state["project_label"] == "Borealis"

    def test_a_new_scope_reaches_the_state(self, store_app):
        sid = open_chat(store_app)["session_id"]
        body = json.loads(
            request(store_app, "POST", f"/api/chat/sessions/{sid}/update", {"context": "retro:1@quarter"}).body
        )
        assert body["context"]["sources"] == ["retro"] and body["context"]["window"]["kind"] == "quarter"
        assert json.loads(store_app.chats.open(sid).session.state["context_scope"])["limits"] == {"retro": 1}

    def test_malformed_updates_are_400s(self, store_app):
        sid = open_chat(store_app)["session_id"]
        assert request(store_app, "POST", f"/api/chat/sessions/{sid}/update", {"title": 3}).code == 400
        assert request(store_app, "POST", f"/api/chat/sessions/{sid}/update", {"context": "stanup"}).code == 400
        assert request(store_app, "POST", "/api/chat/sessions/nope/update", {"title": "x"}).code == 404

    def test_a_blank_project_label_clears_it_and_an_absent_one_keeps_it(self, store_app):
        sid = open_chat(store_app, project_label="Apollo")["session_id"]
        kept = json.loads(request(store_app, "POST", f"/api/chat/sessions/{sid}/update", {"title": "T"}).body)
        assert kept["project_label"] == "Apollo"
        cleared = json.loads(request(store_app, "POST", f"/api/chat/sessions/{sid}/update", {"project_label": ""}).body)
        assert cleared["project_label"] == ""
        assert "project_label" not in store_app.chats.open(sid).session.state
        view = json.loads(request(store_app, "GET", f"/api/chat/sessions/{sid}").body)
        assert view["project_label"] == ""


class TestDelete:
    def test_the_plan_its_versions_and_files_go(self, store_app):
        from yeaboi.sessions import SessionStore

        sid = open_chat(store_app, project_label="Apollo")["session_id"]
        chat = store_app.chats.open(sid)
        chat.session.state.update(pending_review="story_writer", stories=[])
        turn(store_app, sid, "accept")
        with SessionStore(store_app.db) as store:
            assert store.list_plan_versions(sid)
        resp = request(store_app, "POST", f"/api/chat/sessions/{sid}/delete")
        assert json.loads(resp.body) == {"deleted": True, "session_id": sid}
        assert request(store_app, "GET", f"/api/chat/sessions/{sid}").code == 404
        with SessionStore(store_app.db) as store:
            assert store.list_plan_versions(sid) == [] and store.get_session(sid) is None
        assert store_app.removed == [sid]
        assert json.loads(request(store_app, "GET", "/api/chat/sessions").body) == {"sessions": []}

    def test_an_unknown_plan_is_a_404(self, store_app):
        assert request(store_app, "POST", "/api/chat/sessions/nope/delete").code == 404

    def test_a_running_turn_refuses_the_delete(self, store_app):
        sid = open_chat(store_app)["session_id"]
        chat = store_app.chats.open(sid)
        assert chat.turn.acquire(blocking=False)  # the worker holds it for the whole turn
        try:
            resp = request(store_app, "POST", f"/api/chat/sessions/{sid}/delete")
            assert resp.code == 409 and "turn is running" in json.loads(resp.body)["error"]
            assert request(store_app, "GET", f"/api/chat/sessions/{sid}").code == 200
        finally:
            chat.turn.release()
        assert request(store_app, "POST", f"/api/chat/sessions/{sid}/delete").code == 200

    def test_a_deleted_chat_is_never_saved_again(self, store_app):
        from yeaboi.sessions import SessionStore

        sid = open_chat(store_app)["session_id"]
        chat = store_app.chats.open(sid)
        assert store_app.chats.delete(sid)
        store_app.chats.save(chat)  # the worker's late save after the row went
        with SessionStore(store_app.db) as store:
            assert store.get_session(sid) is None


class TestPlan:
    def test_the_plan_view_carries_every_section(self, app):
        open_chat(app)
        turn(app)
        view = json.loads(request(app, "GET", "/api/chat/sessions/proj-1/plan").body)
        assert view["session_id"] == "proj-1" and view["stage"] == "intake"
        assert [s["kind"] for s in view["sections"]] == [
            "intake",
            "analysis",
            "epic",
            "features",
            "stories",
            "tasks",
            "sprints",
        ]
        assert view["sections"][0]["status"] == "generating"
        assert view["intake"]["phases"][0]["label"] == "Project Context"
        assert view["counts"] == {"features": 0, "stories": 0, "tasks": 0, "sprints": 0}

    def test_an_unknown_conversation_is_a_404(self, app):
        assert request(app, "GET", "/api/chat/sessions/nope/plan").code == 404


class TestVersions:
    def test_an_accept_records_a_version_the_routes_serve(self, store_app):
        sid = open_chat(store_app)["session_id"]
        chat = store_app.chats.open(sid)
        chat.session.state.update(pending_review="story_writer", stories=[])
        lines = turn(store_app, sid, "accept")
        section = next(line for line in lines if line["type"] == "section")
        assert section == {"type": "section", "kind": "stories", "status": "accepted", "version": 1}
        listed = json.loads(request(store_app, "GET", f"/api/chat/sessions/{sid}/plan/versions").body)["versions"]
        assert [(v["section"], v["version"]) for v in listed] == [("stories", 1)]
        one = json.loads(request(store_app, "GET", f"/api/chat/sessions/{sid}/plan/versions/stories/1").body)
        assert one["section"] == "stories" and one["version"] == 1 and one["payload"] == {"items": []}
        plan = json.loads(request(store_app, "GET", f"/api/chat/sessions/{sid}/plan").body)
        assert {s["kind"]: s["version"] for s in plan["sections"]}["stories"] == 1
        only = request(store_app, "GET", f"/api/chat/sessions/{sid}/plan/versions?section=intake")
        assert json.loads(only.body) == {"versions": []}

    def test_a_missing_version_and_a_bad_section_are_refused(self, store_app):
        sid = open_chat(store_app)["session_id"]
        assert request(store_app, "GET", f"/api/chat/sessions/{sid}/plan/versions/stories/9").code == 404
        assert request(store_app, "GET", f"/api/chat/sessions/{sid}/plan/versions/budget/1").code == 400
        assert request(store_app, "GET", f"/api/chat/sessions/{sid}/plan/versions/stories/x").code == 400
        assert request(store_app, "GET", f"/api/chat/sessions/{sid}/plan/versions?section=budget").code == 400


class TestCommands:
    def test_the_window_gets_the_verbs_it_runs(self, app):
        body = json.loads(request(app, "GET", "/api/chat/commands").body)
        names = {row["name"] for row in body["commands"]}
        assert {"help", "export", "skip", "finish", "edit", "small", "large"} <= names
        assert not names & {"image", "paste", "voice", "quit", "duck"}
        assert all(row["availability"] and row["help"] for row in body["commands"])

    def test_it_needs_auth(self, app):
        assert request(app, "GET", "/api/chat/commands", authed=False).code == 401


class TestPlanSyncEvictsTheLiveChat:
    def test_a_sync_closes_the_named_conversation(self, app):
        from yeaboi.app.routes_meta import _after_tool

        open_chat(app)
        _after_tool(app, "plan_get", {"session_id": "proj-1"})
        assert "proj-1" in app.chats._chats
        _after_tool(app, "plan_sync", {"session_id": "proj-1"})
        assert "proj-1" not in app.chats._chats

    def test_a_sync_of_the_newest_plan_closes_every_conversation(self, app):
        from yeaboi.app.routes_meta import _after_tool

        open_chat(app)
        _after_tool(app, "plan_sync", {})
        assert app.chats._chats == {}


class TestDescribedAs:
    def test_the_analysis_description_wins_then_the_opening_then_the_first_turn(self):
        from langchain_core.messages import AIMessage, HumanMessage

        from yeaboi.app.chats import described_as

        assert described_as(None) == ""
        assert described_as({"project_description": "analysed", "_chat_opening": "opening"}) == "analysed"
        assert described_as({"_chat_opening": "opening", "messages": [HumanMessage(content="first")]}) == "opening"
        assert described_as({"messages": [AIMessage(content="hi"), HumanMessage(content="first")]}) == "first"
        assert described_as({"messages": [HumanMessage(content=[{"type": "text", "text": "x"}])]}) == ""


class TestWritesWaitForTheTurn:
    def test_a_running_turn_refuses_the_update(self, store_app):
        sid = open_chat(store_app)["session_id"]
        chat = store_app.chats.open(sid)
        assert chat.turn.acquire(blocking=False)
        try:
            resp = request(store_app, "POST", f"/api/chat/sessions/{sid}/update", {"title": "Barbers"})
            assert resp.code == 409 and "turn is running" in json.loads(resp.body)["error"]
        finally:
            chat.turn.release()
        resp = request(store_app, "POST", f"/api/chat/sessions/{sid}/update", {"title": "Barbers"})
        assert resp.code == 200 and json.loads(resp.body)["title"] == "Barbers"

    def test_a_running_turn_refuses_the_size_switch(self, app):
        open_chat(app)
        chat = app.chats.open("proj-1")
        assert chat.turn.acquire(blocking=False)
        try:
            resp = request(app, "POST", "/api/chat/sessions/proj-1/size", {"mode": "small_project"})
            assert resp.code == 409
        finally:
            chat.turn.release()
        assert request(app, "POST", "/api/chat/sessions/proj-1/size", {"mode": "small_project"}).code == 200


class TestSlashVerbsOnly:
    def test_a_known_verb_is_refused_but_a_path_is_a_message(self, app):
        open_chat(app)
        for verb in ("/finish", "/quit", "  /Help me"):
            resp = request(app, "POST", "/api/chat/sessions/proj-1/send", {"text": verb})
            assert resp.code == 400, verb
        lines = turn(app, "proj-1", "/api/health is the route the shell polls")
        assert lines[0]["type"] == "op" and lines[-1]["type"] == "done"


class TestListReadsTheRowItself:
    def test_no_second_query_per_row(self, store_app, monkeypatch):
        from yeaboi.sessions import SessionStore

        sid = open_chat(store_app)["session_id"]
        store_app.chats.close(sid)  # not live: the row's own blob is what the list summarises

        def boom(self, session_id):
            raise AssertionError("the list must not load the state a second time")

        monkeypatch.setattr(SessionStore, "load_state", boom)
        rows = json.loads(request(store_app, "GET", "/api/chat/sessions").body)["sessions"]
        assert [row["session_id"] for row in rows] == [sid]
        assert rows[0]["stage"] and rows[0]["counts"] == {"features": 0, "stories": 0, "tasks": 0, "sprints": 0}


class TestPlanSyncDuringATurn:
    def test_a_mid_turn_chat_is_marked_stale_and_takes_the_synced_keys_on_save(self, app):
        from yeaboi.app.routes_meta import _after_tool

        open_chat(app)
        chat = app.chats.open("proj-1")
        assert chat.turn.acquire(blocking=False)
        try:
            _after_tool(app, "plan_sync", {"session_id": "proj-1"})
            assert "proj-1" in app.chats._chats and chat.stale
            _after_tool(app, "plan_sync", {})
            assert "proj-1" in app.chats._chats
        finally:
            chat.turn.release()
        app.saved["proj-1"] = {**app.saved["proj-1"], "jira_epic_key": "PROJ-1", "jira_story_keys": {"S1": "PROJ-2"}}
        app.chats.save(chat)
        assert chat.session.state["jira_epic_key"] == "PROJ-1"
        assert chat.session.state["jira_story_keys"] == {"S1": "PROJ-2"}
        assert not chat.stale
        assert app.saved["proj-1"]["jira_epic_key"] == "PROJ-1"
