"""The install commands README.md advertises.

`README.md` is the PyPI project page — `pyproject.toml` sets `readme =
"README.md"` — so it is where whoever hits pip's `Requires-Python` error lands
next. The command it shows first has to be the one that cannot fail.

The installer itself, and the landing page that offers the same command, live in
the yeaboi-site repo and are tested there. What is left here is the half that
reads this repo's files, and the string the two repos have to agree on.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # 3.10 — tomllib landed in 3.11; the `dev` extra supplies the backport.
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"

# The command the install work exists to make the headline. yeaboi-site asserts
# the same literal against install.sh and the landing page's copy buttons; any
# drift between the two repos is a bug in the funnel, not a formatting nit.
CURL_COMMAND = "curl -LsSf https://yeaboi.ai/install.sh | sh"


class TestDocumentedCommands:
    def test_readme_leads_with_the_curl_command(self):
        body = README.read_text(encoding="utf-8")
        quick_start = body.index("## 🚀 Quick Start")
        first_block = body.index("```bash", quick_start)
        assert CURL_COMMAND in body[first_block : first_block + 400]

    def test_no_bare_pipx_install_is_advertised(self):
        """`pipx install yeaboi` is the exact command that sends users to upgrade Python.

        pipx uses the interpreter it is running under and will not fetch one
        unless asked, so it may only appear with --python or --fetch-missing-python.
        """
        for path in (README, ROOT / "AGENTS.md"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if "pipx install" not in line:
                    continue
                assert "--python" in line or "--fetch-missing-python" in line, (
                    f"{path.name}: bare `pipx install` fails on an old Python — {line.strip()!r}"
                )

    def test_the_advertised_floor_matches_the_package(self):
        """The README states a Python version; it must be the one pip enforces.

        The same specifier reaches the website through contracts/site.json, so
        this is the assertion that keeps the whole funnel on one floor.
        """
        floor = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["requires-python"]
        minimum = floor.lstrip(">=").split(",")[0].strip()
        body = README.read_text(encoding="utf-8")
        assert f"Python {minimum}+" in body, (
            f"README does not advertise 'Python {minimum}+', which is what pyproject requires ({floor})"
        )
