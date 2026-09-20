"""Wire-shape pins for the desktop backend contract (contracts/v1/app_http.md).

The desktop shell parses these shapes; a changed key is a broken shell. Keep
this file and the contract doc in step — the last test greps the doc for every
route so the prose cannot silently fall behind the table.
"""

from __future__ import annotations

import json
from pathlib import Path

from yeaboi.app.handshake import READY_PREFIX, Handshake, ready_line

CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "v1" / "app_http.md"


class TestHandshakeWire:
    def test_ready_line_shape_is_pinned(self):
        line = ready_line(Handshake(url="u", token="t", pid=1, schema=2, version="v"))
        assert line == 'YEABOI_APP_READY {"pid":1,"schema":2,"token":"t","url":"u","version":"v"}'

    def test_prefix_is_pinned(self):
        assert READY_PREFIX == "YEABOI_APP_READY "


class TestEnvelopeWire:
    def test_success_envelope_keys(self):
        from yeaboi.mcp.runtime import envelope

        assert set(envelope({})) == {"ok", "llm_mode", "warnings", "data"}

    def test_error_envelope_keys(self):
        from yeaboi.mcp.runtime import error_envelope

        payload = error_envelope(ValueError("x"))
        assert {"ok", "llm_mode", "error"} <= set(payload)
        assert set(payload["error"]) == {"type", "message"}

    def test_router_error_shape(self):
        from yeaboi.app.router import Request, Router

        resp = Router().dispatch(Request(method="GET", path="/api/none", authed=True))
        assert json.loads(resp.body) == {"error": "not found"}


class TestSseWire:
    def test_data_frame_shape(self):
        from yeaboi.app.events import EventBus

        bus = EventBus()
        stream = bus.sse_stream()
        next(stream)
        bus.publish("progress", op_id="x", tool="t", progress=1.0, total=None, message=None)
        frame = next(stream)
        stream.close()
        payload = json.loads(frame[len(b"data: ") : -2])
        assert {"type", "seq", "ts", "op_id", "tool", "progress", "total", "message"} == set(payload)


class TestAmbienceWire:
    """The desktop hand-maintains a TypeScript mirror of this payload."""

    def test_saver_keys_are_pinned(self, monkeypatch):
        from pathlib import Path as _Path

        from yeaboi import ambience, config

        monkeypatch.setattr(config, "set_config_value", lambda _k, _v: _Path("/tmp/.env"))
        monkeypatch.delenv("SAVER_STYLE", raising=False)
        assert set(ambience.state()["saver"]) == {"idle_seconds", "style", "styles"}

    def test_off_is_always_offerable(self):
        # Both surfaces honour it; a catalogue without it takes away the only
        # way to turn the screensaver off.
        from yeaboi import ambience

        assert "off" in ambience.SAVER_STYLES
        assert ambience.DEFAULT_SAVER_STYLE in ambience.SAVER_STYLES

    def test_the_front_page_style_is_named_on_both_sides(self):
        # The desktop draws this one in React rather than on the canvas, and
        # keys its own catalogue by this exact string.
        from yeaboi import ambience

        assert ambience.SAVER_STYLES["front-page"] == "Front Page"


class TestContractDoc:
    def test_every_route_is_documented(self):
        from yeaboi.app.registry import ROUTES

        text = CONTRACT.read_text(encoding="utf-8")
        for route in ROUTES:
            assert f"`{route.path}`" in text, f"{route.path} missing from contracts/v1/app_http.md"

    def test_doc_pins_the_ready_prefix(self):
        assert "YEABOI_APP_READY" in CONTRACT.read_text(encoding="utf-8")


class TestSessionsWire:
    """The cross-mode sessions row the desktop's Sessions page and project pages parse."""

    def test_row_keys_are_pinned(self):
        from dataclasses import fields

        from yeaboi.sessions_recent import RecentSession

        assert [f.name for f in fields(RecentSession)] == [
            "session_id",
            "run_id",
            "mode",
            "title",
            "created_at",
            "last_modified",
            "subtitle",
            "kind",
            "project_label",
            "tags",
            "engineer",
        ]

    def test_mode_vocabulary_is_pinned(self):
        from yeaboi.sessions_recent import MODES

        assert MODES == (
            "planning",
            "analysis",
            "standup",
            "retro",
            "reporting",
            "ship",
            "review",
            "poker",
            "performance",
            "roadmap",
            "agent-usage",
            "agent-advisor",
            "agent-security",
        )

    def test_doc_has_the_section(self):
        text = CONTRACT.read_text(encoding="utf-8")
        assert "## Sessions and references" in text


class TestPlanningRoomWire:
    """The planning room's shapes: the session view, the turn's line types, the plan view."""

    def test_session_view_keys_are_pinned(self):
        from yeaboi.app.routes_chat import SESSION_VIEW_KEYS

        assert SESSION_VIEW_KEYS == (
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

    def test_line_types_are_pinned(self):
        from yeaboi.app.routes_chat import WIRE_TYPES

        assert set(WIRE_TYPES) == {
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
        }

    def test_every_line_type_is_documented(self):
        from yeaboi.app.routes_chat import WIRE_TYPES

        text = CONTRACT.read_text(encoding="utf-8")
        for line_type in WIRE_TYPES:
            assert f"| `{line_type}` |" in text, f"line type {line_type} missing from the contract"

    def test_plan_sections_and_statuses_are_pinned(self):
        from yeaboi.agent.chat_session import SECTION_KINDS
        from yeaboi.agent.plan_view import STATUSES

        assert SECTION_KINDS == ("intake", "analysis", "epic", "features", "stories", "tasks", "sprints")
        assert STATUSES == ("empty", "generating", "awaiting_review", "accepted")

    def test_doc_has_the_sections(self):
        text = CONTRACT.read_text(encoding="utf-8")
        assert "## Planning room routes" in text
        assert "### Plan view" in text and "### Plan versions" in text


class TestContextWire:
    """The context scope and the labels row the desktop's picker and drawers parse."""

    def test_labels_row_keys_are_pinned(self):
        from yeaboi.context.labels import SessionLabels

        assert list(SessionLabels("standup", "s1").to_dict()) == [
            "mode",
            "session_id",
            "run_id",
            "project_label",
            "tags",
            "scope",
            "created_at",
            "updated_at",
        ]

    def test_scope_dict_keys_are_pinned(self):
        from yeaboi.context.scope import ContextScope

        assert list(ContextScope().to_dict()) == ["sources", "window", "projects", "tags", "limits", "sessions"]

    def test_options_keys_are_pinned(self):
        from dataclasses import fields

        from yeaboi.context.engine import ContextOptions

        assert [f.name for f in fields(ContextOptions)] == [
            "sources",
            "windows",
            "projects",
            "tags",
            "calendar",
            "default",
            "defaults",
        ]

    def test_doc_has_the_section(self):
        assert "## Context scope and labels" in CONTRACT.read_text(encoding="utf-8")


class TestReferencesWire:
    """The @ picker's row, pinned field for field against the contract."""

    def test_item_keys_are_pinned(self):
        from dataclasses import fields

        from yeaboi.references import Reference, ReferenceSheet

        assert [f.name for f in fields(Reference)] == ["id", "subject", "label", "detail", "url"]
        assert [f.name for f in fields(ReferenceSheet)] == ["source", "source_label", "items", "warning"]


class TestNewsWire:
    """The front page's item shape, pinned field for field against the contract."""

    def test_item_keys_are_pinned(self):
        from dataclasses import fields

        from yeaboi.news.parse import NewsItem

        assert [f.name for f in fields(NewsItem)] == [
            "id",
            "title",
            "url",
            "source_id",
            "source_name",
            "published",
            "summary",
            "image_url",
            "kind",
            "topic",
            "persona",
            "column",
        ]

    def test_paper_keys_are_pinned(self):
        from dataclasses import fields

        from yeaboi.news.paper import Paper, SourceStatus

        assert [f.name for f in fields(Paper)] == ["schema", "generated_at", "stale", "lead", "sections", "sources"]
        assert [f.name for f in fields(SourceStatus)] == [
            "id",
            "name",
            "home_url",
            "column",
            "ok",
            "fetched_at",
            "error",
            "item_count",
        ]
