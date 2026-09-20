"""Tests for the shared Ctrl+V image-attachment helper (ui/shared/_attachments.py)."""

from __future__ import annotations

import pytest

from yeaboi.ui.shared._attachments import (
    MAX_IMAGE_BYTES,
    MAX_TEXT_FILE_BYTES,
    TEXT_FILE_SUFFIXES,
    UNSUPPORTED_MESSAGE,
    chip_text,
    file_chip_text,
    handle_ctrl_v,
    referenced_files,
    referenced_images,
    unsupported_notice,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@pytest.fixture
def attach_dir(tmp_path, monkeypatch):
    """Redirect ~/.yeaboi/attachments to tmp_path so tests never touch $HOME."""
    monkeypatch.setattr("yeaboi.paths.ATTACHMENTS_DIR", tmp_path)
    return tmp_path


class TestHandleCtrlV:
    def test_happy_path_saves_file_and_returns_chip(self, attach_dir, monkeypatch):
        monkeypatch.setattr("yeaboi.clipboard.read_clipboard_image", lambda: (PNG_BYTES, "image/png"))
        notices = []
        attachments: list[str] = []

        chip = handle_ctrl_v(attachments, scope_id="proj-1", set_notice=notices.append)

        assert chip == "[image #1]"
        assert len(attachments) == 1
        saved = attach_dir / "proj-1"
        files = list(saved.glob("img-*.png"))
        assert len(files) == 1
        assert files[0].read_bytes() == PNG_BYTES
        assert attachments[0] == str(files[0])
        assert notices == []

    def test_jpeg_gets_jpg_extension(self, attach_dir, monkeypatch):
        monkeypatch.setattr("yeaboi.clipboard.read_clipboard_image", lambda: (b"\xff\xd8\xff", "image/jpeg"))
        attachments: list[str] = []
        handle_ctrl_v(attachments, scope_id="p", set_notice=lambda _: None)
        assert attachments[0].endswith(".jpg")

    def test_second_paste_numbers_chip_two(self, attach_dir, monkeypatch):
        monkeypatch.setattr("yeaboi.clipboard.read_clipboard_image", lambda: (PNG_BYTES, "image/png"))
        attachments: list[str] = []
        handle_ctrl_v(attachments, scope_id="p", set_notice=lambda _: None)
        chip = handle_ctrl_v(attachments, scope_id="p", set_notice=lambda _: None)
        assert chip == "[image #2]"
        assert len(attachments) == 2

    def test_no_image_on_clipboard_notices_and_returns_none(self, attach_dir, monkeypatch):
        monkeypatch.setattr("yeaboi.clipboard.read_clipboard_image", lambda: None)
        notices = []
        attachments: list[str] = []
        assert handle_ctrl_v(attachments, scope_id="p", set_notice=notices.append) is None
        assert attachments == []
        assert notices and "No image on clipboard" in notices[0]

    def test_oversize_image_rejected(self, attach_dir, monkeypatch):
        big = b"\x00" * (MAX_IMAGE_BYTES + 1)
        monkeypatch.setattr("yeaboi.clipboard.read_clipboard_image", lambda: (big, "image/png"))
        notices = []
        attachments: list[str] = []
        assert handle_ctrl_v(attachments, scope_id="p", set_notice=notices.append) is None
        assert attachments == []
        assert notices and "too large" in notices[0]


class TestReferencedImages:
    def test_surviving_chip_kept(self):
        assert referenced_images("see [image #1] here", ["/a.png"]) == ["/a.png"]

    def test_deleted_chip_detaches(self):
        assert referenced_images("chip was deleted", ["/a.png"]) == []

    def test_partial_chip_does_not_match(self):
        assert referenced_images("broken [image #1", ["/a.png"]) == []

    def test_out_of_range_index_ignored(self):
        assert referenced_images("[image #9]", ["/a.png"]) == []

    def test_duplicate_chips_deduped(self):
        assert referenced_images("[image #1] and again [image #1]", ["/a.png"]) == ["/a.png"]

    def test_order_follows_attachments_list(self):
        text = "[image #2] before [image #1]"
        assert referenced_images(text, ["/a.png", "/b.png"]) == ["/a.png", "/b.png"]

    def test_empty_attachments_short_circuits(self):
        assert referenced_images("[image #1]", []) == []


def test_chip_text_format():
    assert chip_text(3) == "[image #3]"


def test_unsupported_notice_sends_standard_message():
    notices = []
    unsupported_notice(notices.append)
    assert notices == [UNSUPPORTED_MESSAGE]


class TestReferencedFiles:
    """``[file #N]`` chips pick the text files a turn sends — the images rule, for files."""

    def test_only_the_chipped_files_travel(self):
        files = ["/a/notes.md", "/a/log.txt"]
        assert referenced_files("read [file #2]", files) == ["/a/log.txt"]
        assert referenced_files("[file #1] [file #2] [file #1]", files) == files

    def test_no_chip_no_file(self):
        assert referenced_files("nothing", ["/a/notes.md"]) == []
        assert referenced_files("[file #3]", ["/a/notes.md"]) == []
        assert referenced_files("[file #1]", []) == []

    def test_an_image_chip_is_not_a_file_chip(self):
        assert referenced_files("[image #1]", ["/a/notes.md"]) == []

    def test_chip_and_limits(self):
        assert file_chip_text(2) == "[file #2]"
        assert MAX_TEXT_FILE_BYTES == 200 * 1024
        assert TEXT_FILE_SUFFIXES == (".md", ".txt", ".csv", ".json", ".log")
