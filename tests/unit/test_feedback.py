"""Unit tests for the feedback engine (GitHub issue creation + AI polish, all mocked)."""

import json
import urllib.parse
from pathlib import Path

import pytest

from yeaboi import feedback
from yeaboi.feedback import (
    FEEDBACK_AREAS,
    FEEDBACK_REPO,
    FEEDBACK_TYPES,
    MAX_INLINE_TEXT_CHARS,
    FeedbackResult,
    attachment_filename,
    build_issue_body,
    build_issue_url,
    feedback_attachment_kind,
    issue_labels,
    issue_title,
    polish_feedback,
    read_text_tail,
    submit_feedback,
)


class _FakeResp:
    def __init__(self, content):
        self.content = content
        self.response_metadata = {}


class TestConstants:
    def test_areas_match_changelog_vocab(self):
        from yeaboi.changelog import VALID_AREAS

        assert set(FEEDBACK_AREAS) == VALID_AREAS

    def test_types(self):
        assert FEEDBACK_TYPES == ("Bug", "Feature", "Improvement", "Other")


class TestSafeAttachmentName:
    def test_keeps_the_basename_only(self):
        assert feedback.safe_attachment_name("../../etc/passwd", "f.txt") == "passwd"
        assert feedback.safe_attachment_name("C:\\Users\\me\\app.log", "f.txt") == "app.log"

    def test_blank_falls_back(self):
        assert feedback.safe_attachment_name("", "f.txt") == "f.txt"
        assert feedback.safe_attachment_name(None, "f.txt") == "f.txt"
        assert len(feedback.safe_attachment_name("x" * 300, "f.txt")) == 120


class TestIssueTitle:
    def test_prefixes_type(self):
        assert issue_title("Bug", "crash on resize") == "[Bug] crash on resize"

    def test_clamps_length(self):
        assert len(issue_title("Feature", "x" * 500)) == 250


class TestIssueLabels:
    def test_labels(self):
        assert issue_labels("Bug", "standup") == ["type:bug", "area:standup"]


class TestBuildIssueBody:
    def test_metadata_line_and_footer(self):
        from yeaboi import __version__

        body = build_issue_body("Bug", "standup", "it broke")
        assert "**Type:** Bug · **Area:** standup" in body
        assert f"yeaboi v{__version__}" in body
        assert "it broke" in body

    def test_chip_replacement(self):
        body = build_issue_body("Bug", "general", "see [image #1] and [image #2]")
        assert "(screenshot 1)" in body
        assert "(screenshot 2)" in body
        assert "[image #" not in body

    def test_screenshots_section_with_paths(self):
        body = build_issue_body("Bug", "general", "desc", ["/tmp/a.png", "/tmp/b.png"])
        assert "### Screenshots (2)" in body
        assert "`/tmp/a.png`" in body
        assert "drag these files" in body

    def test_no_screenshots_section_when_empty(self):
        body = build_issue_body("Bug", "general", "desc", [])
        assert "Screenshots" not in body
        assert "drag these files" not in body

    def test_home_paths_relativized(self):
        home = str(Path.home())
        body = build_issue_body("Bug", "general", "desc", [f"{home}/.yeaboi/attachments/feedback/img.png"])
        assert home not in body
        assert "`~/.yeaboi/attachments/feedback/img.png`" in body


class TestTextAttachments:
    """Logs are the one attachment that genuinely reaches GitHub — inlined."""

    def test_kind_vocabulary(self):
        assert feedback_attachment_kind("image/png") == "image"
        assert feedback_attachment_kind("text/plain") == "text"
        assert feedback_attachment_kind("application/zip") is None

    def test_contents_inlined_in_a_collapsed_block(self, tmp_path):
        log = tmp_path / "app.log"
        log.write_text("first line\nboom: it broke\n")
        body = build_issue_body("Bug", "general", "desc", None, [str(log)])
        assert "### Logs and files (1)" in body
        assert "<summary>app.log — 2 lines</summary>" in body
        assert "boom: it broke" in body

    def test_published_contents_are_redacted(self, tmp_path, monkeypatch):
        # The one thing a report publishes that the reporter did not type, and
        # it goes to a public repository.
        # A value-layer secret rather than a token-shaped one, so the fixture
        # itself is not a credential gitleaks has to be told to ignore.
        monkeypatch.setenv("GITHUB_TOKEN", "not-a-real-credential-0123456789")
        log = tmp_path / "app.log"
        log.write_text("auth failed with not-a-real-credential-0123456789\n")
        body = build_issue_body("Bug", "general", "desc", None, [str(log)])
        assert "not-a-real-credential" not in body
        assert "[REDACTED]" in body

    def test_published_contents_lose_the_home_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        log = tmp_path / "app.log"
        log.write_text(f"could not open {tmp_path}/projects/secret.db\n")
        body = build_issue_body("Bug", "general", "desc", None, [str(log)])
        assert str(tmp_path) not in body
        assert "~/projects/secret.db" in body

    def test_the_inline_share_stays_under_github_s_body_limit(self, tmp_path):
        # Six files at the per-file ceiling plus a full description would clear
        # 65_536, which the token path answers with a 422.
        paths = []
        for i in range(6):
            log = tmp_path / f"{i}.log"
            log.write_text("x" * MAX_INLINE_TEXT_CHARS)
            paths.append(str(log))
        body = build_issue_body("Bug", "general", "d" * 20_000, None, paths)
        assert len(body) < 65_536

    def test_only_the_tail_survives_a_long_log(self, tmp_path):
        log = tmp_path / "big.log"
        log.write_text("x" * (MAX_INLINE_TEXT_CHARS + 500) + "\nTHE END\n")
        body = build_issue_body("Bug", "general", "desc", None, [str(log)])
        assert "THE END" in body
        assert len(body) < MAX_INLINE_TEXT_CHARS + 2_000
        assert "last " in body

    def test_backticks_inside_a_log_cannot_escape_the_fence(self, tmp_path):
        log = tmp_path / "md.log"
        log.write_text("```\nnot the end of the block\n```\n")
        body = build_issue_body("Bug", "general", "desc", None, [str(log)])
        assert "````" in body
        assert "not the end of the block" in body

    def test_browser_body_names_the_files_instead_of_carrying_them(self, tmp_path):
        log = tmp_path / "app.log"
        log.write_text("boom: it broke\n")
        body = build_issue_body("Bug", "general", "desc", None, [str(log)], inline_text=False)
        assert "boom: it broke" not in body
        assert f"`{log}`" in body
        assert "drag them onto this issue" in body

    def test_home_paths_relativized_in_the_named_form(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        log = tmp_path / "app.log"
        log.write_text("boom\n")
        body = build_issue_body("Bug", "general", "desc", None, [str(log)], inline_text=False)
        assert str(tmp_path) not in body
        assert "`~/app.log`" in body

    def test_no_section_when_there_are_none(self):
        assert "Logs and files" not in build_issue_body("Bug", "general", "desc", None, [])

    def test_an_unreadable_file_does_not_sink_the_report(self, tmp_path):
        body = build_issue_body("Bug", "general", "desc", None, [str(tmp_path / "gone.log")])
        assert "### Logs and files (1)" in body
        assert "desc" in body

    def test_read_text_tail_reports_whether_it_cut(self, tmp_path):
        log = tmp_path / "a.log"
        log.write_text("abcdef")
        assert read_text_tail(str(log), 100) == ("abcdef", False)
        assert read_text_tail(str(log), 3) == ("def", True)


class TestAttachmentFilename:
    """The stored name is what the issue shows, so it keeps the reporter's."""

    def test_keeps_the_name_and_makes_it_unique(self):
        assert attachment_filename("app.log", "text/plain", "3f2a91bc") == "app-3f2a91bc.log"

    def test_keeps_a_png_a_png(self):
        assert attachment_filename("poker room.png", "image/png", "aa11") == "poker-room-aa11.png"

    def test_a_suffix_that_disagrees_with_the_mime_is_replaced(self):
        assert attachment_filename("shot.png", "text/plain", "aa11") == "shot-aa11.txt"

    def test_a_name_that_survives_nothing_still_produces_a_file(self):
        assert attachment_filename("../../", "text/plain", "aa11") == "file-aa11.txt"

    def test_nothing_outside_the_safe_alphabet_survives(self):
        # The stored name is interpolated into the issue body's HTML.
        stored = attachment_filename('<img src=x onerror="alert(1)">.log', "text/plain", "aa11")
        assert "<" not in stored and '"' not in stored
        assert stored.endswith("-aa11.log")


class TestBuildIssueUrl:
    def test_url_shape_and_encoding(self):
        url = build_issue_url("Bug", "standup", "crash & burn", "line one\nline two")
        assert url.startswith(f"https://github.com/{FEEDBACK_REPO}/issues/new?")
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert params["title"] == ["[Bug] crash & burn"]
        assert params["body"] == ["line one\nline two"]
        assert params["labels"] == ["type:bug,area:standup"]

    def test_long_body_truncated(self):
        url = build_issue_url("Bug", "general", "t", "word " * 5000)
        assert len(url) <= feedback._MAX_URL_CHARS
        body = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["body"][0]
        assert "truncated" in body

    def test_unicode_survives(self):
        url = build_issue_url("Feature", "planning", "émoji 🚀", "déscription")
        params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert params["title"] == ["[Feature] émoji 🚀"]


class _FakeIssue:
    number = 42
    html_url = f"https://github.com/{FEEDBACK_REPO}/issues/42"


class _FakeRepo:
    def __init__(self):
        self.kwargs = None

    def create_issue(self, **kwargs):
        self.kwargs = kwargs
        return _FakeIssue()


class TestSubmitFeedbackApi:
    def test_token_path_creates_issue(self, monkeypatch):
        fake_repo = _FakeRepo()
        monkeypatch.setattr("yeaboi.config.get_github_token", lambda: "ghp_x")
        monkeypatch.setattr(
            "yeaboi.tools.github._get_github_client",
            lambda: type("G", (), {"get_repo": lambda self, slug: fake_repo})(),
        )
        result = submit_feedback("Bug", "standup", "crash", "it broke")
        assert result == FeedbackResult(ok=True, via="api", url=_FakeIssue.html_url, message="Issue #42 created!")
        assert fake_repo.kwargs["title"] == "[Bug] crash"
        assert fake_repo.kwargs["labels"] == ["type:bug", "area:standup"]
        assert "it broke" in fake_repo.kwargs["body"]

    def test_api_error_returns_browser_fallback(self, monkeypatch):
        monkeypatch.setattr("yeaboi.config.get_github_token", lambda: "ghp_x")

        def _boom():
            raise RuntimeError("api down")

        monkeypatch.setattr("yeaboi.tools.github._get_github_client", _boom)
        result = submit_feedback("Bug", "standup", "crash", "it broke")
        assert result.ok is False
        assert result.via == "api"
        assert result.url.startswith(f"https://github.com/{FEEDBACK_REPO}/issues/new?")
        assert "browser" in result.message


class TestSubmitFeedbackBrowser:
    def test_no_token_opens_browser(self, monkeypatch):
        opened = {}
        monkeypatch.setattr("yeaboi.config.get_github_token", lambda: None)
        monkeypatch.setattr("webbrowser.open", lambda url: opened.setdefault("url", url) or True)
        result = submit_feedback("Feature", "planning", "dark mode", "please")
        assert result.ok is True
        assert result.via == "browser"
        assert opened["url"] == result.url

    def test_browser_open_false_asks_manual_copy(self, monkeypatch):
        monkeypatch.setattr("yeaboi.config.get_github_token", lambda: None)
        monkeypatch.setattr("webbrowser.open", lambda url: False)
        result = submit_feedback("Feature", "planning", "dark mode", "please")
        assert result.ok is False
        assert "copy" in result.message.lower()
        assert result.url.startswith("https://github.com/")

    def test_screenshot_hint_in_browser_message(self, monkeypatch):
        monkeypatch.setattr("yeaboi.config.get_github_token", lambda: None)
        monkeypatch.setattr("webbrowser.open", lambda url: True)
        result = submit_feedback("Bug", "general", "t", "see [image #1]", ["/tmp/a.png"])
        assert "Drag" in result.message


class TestParsePolishResponse:
    def test_plain_json(self):
        raw = json.dumps({"title": "T", "description": "D"})
        assert feedback._parse_polish_response(raw) == ("T", "D")

    def test_fenced_json(self):
        raw = '```json\n{"title": "T", "description": "D"}\n```'
        assert feedback._parse_polish_response(raw) == ("T", "D")

    def test_garbage_returns_none(self):
        assert feedback._parse_polish_response("nope") is None

    def test_missing_fields_returns_none(self):
        assert feedback._parse_polish_response('{"title": "only"}') is None

    def test_non_dict_returns_none(self):
        assert feedback._parse_polish_response('["a"]') is None


class TestPolishFeedback:
    def test_not_configured_keeps_original(self, monkeypatch):
        monkeypatch.setattr("yeaboi.config.is_llm_configured", lambda: (False, "ANTHROPIC_API_KEY not set"))
        polished, msg = polish_feedback("Bug", "standup", "t", "d")
        assert polished is None
        assert "unavailable" in msg.lower()

    def test_happy_path(self, monkeypatch):
        monkeypatch.setattr("yeaboi.config.is_llm_configured", lambda: (True, ""))
        monkeypatch.setattr("yeaboi.agent.llm.get_llm", lambda **k: object())
        monkeypatch.setattr("yeaboi.agent.llm.track_usage", lambda resp: None)
        captured = {}

        def _fake_invoke(llm, prompt, image_paths=None):
            captured["prompt"] = prompt
            captured["images"] = image_paths
            return _FakeResp('{"title": "Better", "description": "Clearer"}')

        monkeypatch.setattr("yeaboi.agent.llm.invoke_with_images", _fake_invoke)
        polished, msg = polish_feedback("Bug", "standup", "t", "d", ["/tmp/a.png"])
        assert polished == ("Better", "Clearer")
        assert "polished" in msg.lower()
        assert captured["images"] == ["/tmp/a.png"]
        assert "standup" in captured["prompt"]

    def test_track_usage_called(self, monkeypatch):
        monkeypatch.setattr("yeaboi.config.is_llm_configured", lambda: (True, ""))
        monkeypatch.setattr("yeaboi.agent.llm.get_llm", lambda **k: object())
        tracked = {}
        monkeypatch.setattr("yeaboi.agent.llm.track_usage", lambda resp: tracked.setdefault("resp", resp))
        monkeypatch.setattr(
            "yeaboi.agent.llm.invoke_with_images",
            lambda llm, prompt, image_paths=None: _FakeResp('{"title": "T", "description": "D"}'),
        )
        polish_feedback("Bug", "general", "t", "d")
        assert "resp" in tracked

    def test_unusable_response_keeps_original(self, monkeypatch):
        monkeypatch.setattr("yeaboi.config.is_llm_configured", lambda: (True, ""))
        monkeypatch.setattr("yeaboi.agent.llm.get_llm", lambda **k: object())
        monkeypatch.setattr("yeaboi.agent.llm.track_usage", lambda resp: None)
        monkeypatch.setattr("yeaboi.agent.llm.invoke_with_images", lambda llm, p, image_paths=None: _FakeResp("junk"))
        polished, msg = polish_feedback("Bug", "general", "t", "d")
        assert polished is None
        assert "keeping your original" in msg.lower()

    def test_auth_error_not_reraised(self, monkeypatch):
        monkeypatch.setattr("yeaboi.config.is_llm_configured", lambda: (True, ""))
        monkeypatch.setattr("yeaboi.agent.llm.get_llm", lambda **k: object())
        # The real classifier isinstance-checks provider SDK exception classes;
        # force it True to exercise the auth/billing branch.
        monkeypatch.setattr("yeaboi.agent.nodes._is_llm_auth_or_billing_error", lambda exc: True)

        def _boom(llm, prompt, image_paths=None):
            raise RuntimeError("invalid x-api-key")

        monkeypatch.setattr("yeaboi.agent.llm.invoke_with_images", _boom)
        polished, msg = polish_feedback("Bug", "general", "t", "d")
        assert polished is None
        assert "unavailable" in msg.lower()

    def test_generic_error_not_reraised(self, monkeypatch):
        monkeypatch.setattr("yeaboi.config.is_llm_configured", lambda: (True, ""))
        monkeypatch.setattr("yeaboi.agent.llm.get_llm", lambda **k: object())

        def _boom(llm, prompt, image_paths=None):
            raise RuntimeError("network down")

        monkeypatch.setattr("yeaboi.agent.llm.invoke_with_images", _boom)
        polished, msg = polish_feedback("Bug", "general", "t", "d")
        assert polished is None
        assert "failed" in msg.lower()


class TestPolishWithAttachments:
    def test_an_attached_log_reaches_the_prompt(self, tmp_path, monkeypatch):
        monkeypatch.setattr("yeaboi.config.is_llm_configured", lambda: (True, ""))
        monkeypatch.setattr("yeaboi.agent.llm.get_llm", lambda **k: object())
        monkeypatch.setattr("yeaboi.agent.llm.track_usage", lambda _r: None)
        seen = {}

        def _invoke(llm, prompt, image_paths=None):
            seen["prompt"] = prompt
            return _FakeResp(json.dumps({"title": "T", "description": "D"}))

        monkeypatch.setattr("yeaboi.agent.llm.invoke_with_images", _invoke)
        log = tmp_path / "app.log"
        log.write_text("boom: it broke\n")
        polished, _ = polish_feedback("Bug", "general", "t", "d", None, [str(log)])
        assert polished == ("T", "D")
        assert "boom: it broke" in seen["prompt"]

    def test_a_log_that_cannot_be_read_is_dropped_not_fatal(self, tmp_path, monkeypatch):
        monkeypatch.setattr("yeaboi.config.is_llm_configured", lambda: (True, ""))
        monkeypatch.setattr("yeaboi.agent.llm.get_llm", lambda **k: object())
        monkeypatch.setattr("yeaboi.agent.llm.track_usage", lambda _r: None)
        seen = {}

        def _invoke(llm, prompt, image_paths=None):
            seen["prompt"] = prompt
            return _FakeResp(json.dumps({"title": "T", "description": "D"}))

        monkeypatch.setattr("yeaboi.agent.llm.invoke_with_images", _invoke)
        polished, _ = polish_feedback("Bug", "general", "t", "d", None, [str(tmp_path / "gone.log")])
        assert polished == ("T", "D")
        assert "ATTACHED FILE (last lines" not in seen["prompt"]


class TestFeedbackResultDataclass:
    def test_defaults(self):
        r = FeedbackResult()
        assert r.ok is False
        assert r.via == ""

    def test_frozen(self):
        r = FeedbackResult()
        with pytest.raises(AttributeError):
            r.ok = True
