"""Shared Ctrl+V image-attachment handling for TUI textboxes.

Every LLM-connected input loop wires image paste the same way:

1. keep a local ``attachments: list[str]`` of saved image file paths,
2. on ``key == "ctrl+v"`` call :func:`handle_ctrl_v` and insert the returned
   chip (``[image #N]``) into the buffer exactly like typed text,
3. at submit, call :func:`referenced_images` to keep only the attachments whose
   chip still survives in the text — deleting a chip detaches its image.

The chip is deliberately *plain printable text*: backspace, Ctrl+W, cursor
movement, and multi-line editing all work on it with zero changes to any input
loop, and the model sees "``[image #2]`` shows the login page" adjacent to image
block #2 in the multimodal prompt (see ``agent/llm.py:build_multimodal_content``).

Non-LLM fields (API tokens, file paths) call :func:`unsupported_notice` instead,
so Ctrl+V is never silently swallowed anywhere.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable

from yeaboi.paths import get_attachments_dir

logger = logging.getLogger(__name__)

# Matches surviving chips in the buffer text at submit time. N is the 1-based
# position in that textbox's local attachments list.
CHIP_RE = re.compile(r"\[image #(\d+)\]")

# Strictest provider inline-image limit (Anthropic: 5 MB/image) minus headroom
# for base64 expansion — enforced at paste time so a failing LLM call later
# never surprises the user.
MAX_IMAGE_BYTES = int(4.5 * 1024 * 1024)

_EXT_FOR_MIME = {"image/png": ".png", "image/jpeg": ".jpg"}

UNSUPPORTED_MESSAGE = "Images are not supported in this field"

# A text file attached to a turn rides the same convention as an image: a
# `[file #N]` chip in the text, a path list beside it, the chip deciding
# which travel. What the window may attach, and how big.
FILE_CHIP_RE = re.compile(r"\[file #(\d+)\]")
MAX_TEXT_FILE_BYTES = 200 * 1024
TEXT_FILE_SUFFIXES: tuple[str, ...] = (".md", ".txt", ".csv", ".json", ".log")


def chip_text(index: int) -> str:
    """Return the placeholder chip for the ``index``-th attachment (1-based)."""
    return f"[image #{index}]"


def handle_ctrl_v(
    attachments: list[str],
    *,
    scope_id: str,
    set_notice: Callable[[str], None],
) -> str | None:
    """Read an image from the OS clipboard, save it, and append its path.

    On success the image is written under ``~/.yeaboi/attachments/<scope_id>/``,
    its path is appended to ``attachments``, and the chip text to insert into the
    buffer is returned. On failure ``None`` is returned and a user-facing reason
    has been sent to ``set_notice`` ("No image on clipboard", "Image too large…").
    """
    from yeaboi.clipboard import read_clipboard_image

    result = read_clipboard_image()
    if result is None:
        set_notice("No image on clipboard (copy a screenshot, then Ctrl+V)")
        return None

    data, mime = result
    if len(data) > MAX_IMAGE_BYTES:
        set_notice(f"Image too large ({len(data) / (1024 * 1024):.1f} MB, max 4.5 MB)")
        logger.warning("pasted image rejected: %d bytes exceeds %d limit", len(data), MAX_IMAGE_BYTES)
        return None

    ext = _EXT_FOR_MIME.get(mime, ".png")
    path = get_attachments_dir(scope_id) / f"img-{uuid.uuid4().hex[:8]}{ext}"
    try:
        path.write_bytes(data)
    except OSError as exc:
        set_notice("Could not save pasted image")
        logger.error("failed to save pasted image to %s: %s", path, exc)
        return None

    attachments.append(str(path))
    logger.info("image pasted: %s (%d bytes, %s)", path, len(data), mime)
    return chip_text(len(attachments))


def referenced_images(text: str, attachments: list[str]) -> list[str]:
    """Return the attachment paths whose ``[image #N]`` chip survives in ``text``.

    Deleting a chip from the buffer detaches its image at send time. Out-of-range
    indices are ignored, duplicates deduped; order follows the attachments list.
    """
    if not attachments:
        return []
    indices = {int(m) for m in CHIP_RE.findall(text)}
    return [path for i, path in enumerate(attachments, start=1) if i in indices]


def file_chip_text(index: int) -> str:
    """Return the placeholder chip for the ``index``-th attached text file (1-based)."""
    return f"[file #{index}]"


def referenced_files(text: str, files: list[str]) -> list[str]:
    """Return the file paths whose ``[file #N]`` chip survives in ``text`` — the images rule, for files."""
    if not files:
        return []
    indices = {int(m) for m in FILE_CHIP_RE.findall(text)}
    return [path for i, path in enumerate(files, start=1) if i in indices]


def unsupported_notice(set_notice: Callable[[str], None]) -> None:
    """Standard Ctrl+V response for fields whose content never reaches the LLM."""
    set_notice(UNSUPPORTED_MESSAGE)
