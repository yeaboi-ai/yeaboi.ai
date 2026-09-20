"""One floor, named in ten places, asserted here to be the same number.

Lowering the supported Python touched package metadata, two packaging manifests,
the installer, the README badge, three docs pages, an Open Graph PNG and the CI
matrix. Nothing but this file connects them, and the PNG in particular is rendered
by hand — `make site-og` — so a stale version there is invisible until someone
reads a search result.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:  # 3.10 — tomllib landed in 3.11; the `dev` extra supplies the backport.
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"

REQUIRES_PYTHON = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["requires-python"]
FLOOR = REQUIRES_PYTHON.lstrip(">=").split(",")[0].strip()
FLOOR_TUPLE = tuple(int(part) for part in FLOOR.split("."))

# The versions above the floor that the non-required `compat` job covers. The
# floor itself is covered by `unit`, which is required — see ci.yml.
CEILING = "3.14"


class TestPackageMetadata:
    def test_the_floor_is_a_lower_bound_not_a_pin(self):
        assert REQUIRES_PYTHON.startswith(">="), REQUIRES_PYTHON

    def test_classifiers_span_the_floor_to_the_ceiling(self):
        meta = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
        declared = [
            c.rsplit(" :: ", 1)[-1] for c in meta["classifiers"] if c.startswith("Programming Language :: Python :: 3.")
        ]
        expected = [f"3.{minor}" for minor in range(FLOOR_TUPLE[1], int(CEILING.split(".")[1]) + 1)]
        assert declared == expected, "classifiers must list every supported version, in order"

    @pytest.mark.parametrize(
        "manifest",
        ["packaging/scrum-agent-shim/pyproject.toml"],
    )
    def test_the_packaging_manifests_agree(self, manifest):
        """A shim that floors higher than the package makes `uv lock` unsolvable
        for everyone — the resolver refuses before any test can run."""
        text = (ROOT / manifest).read_text(encoding="utf-8")
        found = re.search(r'^requires-python\s*=\s*"([^"]+)"', text, re.MULTILINE)
        assert found, f"{manifest} declares no requires-python"
        assert found.group(1) == REQUIRES_PYTHON

    def test_the_lock_was_regenerated(self):
        lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
        assert f'requires-python = "{REQUIRES_PYTHON}"' in lock, "run `uv lock` and commit the result"


class TestUserFacingSurfaces:
    """The website's surfaces — install.sh, the docs pages, the Open Graph card —
    live in the yeaboi-site repo and are asserted there against the floor this
    repo publishes in contracts/site.json."""

    def test_the_contract_publishes_the_same_specifier(self):
        """The one hop the website's own guards depend on."""
        contract = json.loads((ROOT / "contracts" / "site.json").read_text(encoding="utf-8"))
        assert contract["requires_python"] == REQUIRES_PYTHON, "run `make site-contract` and commit the result"

    @pytest.mark.parametrize("page", ["README.md", "AGENTS.md"])
    def test_no_page_names_a_version_below_the_floor(self, page):
        """A doc that still says 3.11+ turns away people the change just admitted."""
        text = (ROOT / page).read_text(encoding="utf-8")
        stale = {f"3.{minor}+" for minor in range(FLOOR_TUPLE[1] + 1, int(CEILING.split(".")[1]) + 1)}
        found = sorted(marker for marker in stale if f"Python {marker}" in text)
        assert not found, f"{page} advertises {found}; the floor is {FLOOR}"


class TestCi:
    """The required jobs run the floor; the matrix covers everything above it."""

    @staticmethod
    def _workflow() -> dict:
        return yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))

    @pytest.mark.parametrize("job", ["unit", "integration"])
    def test_the_required_jobs_run_on_the_floor(self, job):
        steps = self._workflow()["jobs"][job]["steps"]
        commands = [step.get("run", "") for step in steps]
        assert any(f"uv python install {FLOOR}" == cmd.strip() for cmd in commands), f"{job} must install {FLOOR}"
        # Without the pin, uv picks whatever the runner preinstalled and the job
        # silently proves nothing about the floor.
        assert any(f"--python {FLOOR}" in cmd for cmd in commands), f"{job} must pin --python {FLOOR}"
        assert any("sys.version_info[:2]" in cmd for cmd in commands), (
            f"{job} must assert the interpreter it actually got"
        )

    @pytest.mark.parametrize("job", ["unit", "integration"])
    def test_the_required_jobs_stay_unconditional(self, job):
        """Required contexts attached to a skippable job block every PR forever."""
        assert "if" not in self._workflow()["jobs"][job]

    def test_the_matrix_covers_every_version_above_the_floor(self):
        compat = self._workflow()["jobs"]["compat"]
        expected = [f"3.{minor}" for minor in range(FLOOR_TUPLE[1] + 1, int(CEILING.split(".")[1]) + 1)]
        assert compat["strategy"]["matrix"]["python"] == expected

    def test_the_makefile_target_covers_the_same_versions(self):
        """`make preflight` runs `test-compat` for this job; a Makefile list that
        drifts from the matrix means the local gate proves less than CI."""
        text = (ROOT / "Makefile").read_text(encoding="utf-8")
        found = re.search(r"^COMPAT_PYTHONS \?= (.+)$", text, re.MULTILINE)
        assert found, "Makefile no longer defines COMPAT_PYTHONS"
        matrix = self._workflow()["jobs"]["compat"]["strategy"]["matrix"]["python"]
        assert found.group(1).split() == matrix

    def test_the_matrix_job_is_not_required_and_does_not_hide_failures(self):
        compat = self._workflow()["jobs"]["compat"]
        # It carries an `if:` — which is exactly why it must never be a required
        # context, and why the matrix lives here rather than on `unit`.
        assert "if" in compat
        assert compat["strategy"]["fail-fast"] is False
        assert "matrix.python" in compat["name"], "a matrix job needs a per-version name"
