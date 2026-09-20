"""Guards for the ship gate: the Makefile targets, the preflight map, the repo's notes.

Everything here is a repo-reading guard rather than a module test, which is why it
is registered in ``scripts/test_scope.py``'s ``ALWAYS`` — nothing about a changed
source file implies it.

``/ship`` and ``/sync-main`` themselves are no longer in this repo — they moved to
the ``yeaboi-devkit`` plugin, and their shape is guarded there. What stayed is the
half that is about *this* repo: the targets the shared procedure calls, and the
facts it reads out of ``.claude/repo-notes.md``.

Three of these encode a bug that actually shipped:

* ``/ship`` handed ``code-reviewer`` ``git diff main...HEAD``. Local ``main`` in a
  worktree is routinely several commits behind ``origin/main``, so the reviewer
  received other people's already-merged PRs and none of the branch's work — and a
  review of the wrong diff reports clean.
* ``make test`` ran 10,383 unit tests single-threaded while ``make test-fast`` ran
  the identical set with ``-n auto``. 310 of the gate's 408 seconds were that.
* ``make security``'s SAST line was ``ruff check src/ tests/``, byte for byte the
  whole of ``make lint``, so the gate linted twice.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _preflight():
    """Import scripts/preflight.py. `scripts/` is not a package; this is the seam."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import preflight  # noqa: PLC0415

    return preflight


MAKEFILE = ROOT / "Makefile"
COMMANDS = sorted((ROOT / ".claude" / "commands").glob("*.md"))
AGENTS = sorted((ROOT / ".claude" / "agents").glob("*.md"))


def _recipe(target: str) -> tuple[str, str]:
    """Return (prerequisites, recipe body) for a Makefile target."""
    text = MAKEFILE.read_text()
    match = re.search(rf"^{re.escape(target)}:(.*?)$\n((?:[\t#].*\n|\n(?=[\t#]))*)", text, re.MULTILINE)
    assert match, f"Makefile has no target {target!r}"
    prereqs = match.group(1).split("##")[0].strip()
    return prereqs, match.group(2)


class TestTheGateIsFast:
    """`make test` must not re-run the unit lane serially."""

    def test_test_delegates_to_the_parallel_and_serial_lanes(self):
        prereqs, body = _recipe("test")
        assert prereqs.split() == ["test-fast", "test-slow"], (
            "`make test` should be `test: test-fast test-slow` so the unit lane runs under "
            f"PYTEST_PARALLEL and the flags cannot drift from the lane variables. Got: {prereqs!r}"
        )
        assert "pytest" not in body, (
            "`make test` grew its own pytest invocation again — that is how the serial unit lane "
            "came back the first time. Delegate to test-fast/test-slow instead."
        )

    def test_the_unit_lane_is_parallel_and_the_slow_lane_is_not(self):
        _, fast = _recipe("test-fast")
        _, slow = _recipe("test-slow")
        assert "$(PYTEST_PARALLEL)" in fast
        # tests/integration/test_repl.py monkeypatches ten-plus names and AGENTS.md
        # forbids editing it, so the slow lane stays serial deliberately.
        assert "$(PYTEST_PARALLEL)" not in slow

    def test_make_may_not_reorder_the_gate(self):
        assert ".NOTPARALLEL:" in MAKEFILE.read_text(), (
            "`test` and `ship-gate` order their prerequisites deliberately, and two pytest "
            "processes in one worktree invent failures. `make -j` must not reorder them."
        )

    def test_security_does_not_repeat_lint(self):
        prereqs, body = _recipe("security")
        assert "lint" in prereqs.split(), "`security` should take `lint` as a prerequisite, not repeat its command"
        assert "ruff check" not in body, (
            "`make security`'s SAST step is `make lint`, byte for byte. Running it again makes the "
            "ship gate lint twice for nothing."
        )


class TestTheGateIsComplete:
    """`make ship-gate` must cover what CI checks, including the non-pytest half."""

    def test_ship_gate_runs_every_half_of_the_gate(self):
        prereqs, _ = _recipe("ship-gate")
        assert set(prereqs.split()) >= {"lint", "format-check", "test", "security", "preflight"}, (
            f"ship-gate is missing part of the gate. Got: {prereqs!r}"
        )

    def test_format_check_asserts_rather_than_writes(self):
        """CI's `Format check (ruff)` is a required check and had no local twin."""
        _, body = _recipe("format-check")
        assert "--check" in body
        _, write = _recipe("format")
        assert "--check" not in write, "`make format` writes; `make format-check` asserts. Do not merge them."

    def test_the_wheel_assertion_is_not_duplicated_in_ci(self):
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "make package-check" in ci, "ci.yml's package job should call the same script `make package-check` does"
        assert "zipfile.ZipFile" not in ci, (
            "the wheel assertion is inlined in ci.yml again — that is the version nobody can run locally"
        )


class TestPreflightCoversEveryJob:
    def test_every_scope_job_has_a_target(self):
        """A job the selector knows about and preflight does not would simply never run locally.

        Same shape as `test_test_scope.py`'s totality checks: a selector's failure
        mode is silence, so the totality check is the feature.
        """
        sys.path.insert(0, str(ROOT / "scripts"))
        from preflight import JOB_TARGETS  # noqa: PLC0415
        from test_scope import JOBS  # noqa: PLC0415

        assert {job.name for job in JOBS} == set(JOB_TARGETS), (
            "scripts/test_scope.py's JOBS and scripts/preflight.py's JOB_TARGETS disagree. "
            "A job in one and not the other is a CI check `make preflight` silently never runs."
        )

    def test_every_preflight_target_exists_in_the_makefile(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from preflight import JOB_TARGETS  # noqa: PLC0415

        text = MAKEFILE.read_text()
        for job, targets in JOB_TARGETS.items():
            for target in targets:
                assert re.search(rf"^{re.escape(target)}:", text, re.MULTILINE), (
                    f"preflight maps job {job!r} to `make {target}`, which the Makefile does not define"
                )

    def test_preflight_names_every_job_it_skipped(self, monkeypatch, capsys):
        """Never silently narrow — a run that says nothing reads as 'covered everything'.

        This asserted only that the string "[preflight]" appeared, which the
        unconditional `base … · N changed path(s)` header prints. Deleting the
        entire skip loop left it green — an assertion-free test of exactly the kind
        this branch's `assertion-free-tests` lens exists to find.
        """
        preflight = _preflight()
        monkeypatch.setattr(
            preflight,
            "decide",
            lambda changed: ({"package": True, "eval": False, "compat": False}, ""),
        )
        monkeypatch.setattr(preflight, "changed_paths", lambda base: ["pyproject.toml"])

        assert preflight.main(["--base", "origin/main", "--list"]) == 0
        out = capsys.readouterr().out
        assert "running: package" in out
        for job in ("eval", "compat"):
            assert f"skipped {job} —" in out, f"preflight ran without {job} and never said so"

    def test_a_missing_toolchain_is_reported_not_failed(self, monkeypatch, capsys):
        """An unattended sandbox has no Node; a job needing one is skipped, not failed.

        Failing the unattended lane on an environment fact rather than on the diff
        is an outage, not a gate. CI has the toolchains.

        JOB_TOOLCHAIN is empty now that the Electron shell lives in
        yeaboi-desktop — every remaining job is Python. The mechanism still has
        to work, so this pins it against a fake entry rather than deleting the
        test along with its last real user.
        """
        preflight = _preflight()
        monkeypatch.setitem(preflight.JOB_TARGETS, "fake", ("fake-target",))
        monkeypatch.setitem(preflight.JOB_TOOLCHAIN, "fake", "npm")
        monkeypatch.setattr(
            preflight, "decide", lambda changed: (dict.fromkeys(preflight.JOB_TARGETS, False) | {"fake": True}, "")
        )
        monkeypatch.setattr(preflight, "changed_paths", lambda base: ["anything.py"])
        monkeypatch.setattr(preflight.shutil, "which", lambda name: None)

        assert preflight.main(["--base", "origin/main", "--list"]) == 0
        out = capsys.readouterr().out
        assert "skipped fake — npm is not on PATH" in out

    def test_job_selection_sees_uncommitted_work(self):
        """`--base` in test_scope.py is committed-only; the ship gate runs before the commit.

        The source arguments there are a mutually-exclusive group, so a `--base`
        call cannot also read the working tree. preflight passes the union over
        `--changed-files -` instead; a partially-committed branch would otherwise
        skip web/package for anything living only in the working tree.
        """
        source = (ROOT / "scripts" / "preflight.py").read_text()
        assert '"--changed-files", "-"' in source, (
            "preflight must hand test_scope.py the path list it computed, not a --base ref — "
            "--base cannot see uncommitted work, and the gate runs before the commit"
        )
        assert '"--base", base' not in source

    def test_an_unreadable_scope_runs_everything(self, monkeypatch):
        sys.path.insert(0, str(ROOT / "scripts"))
        import preflight  # noqa: PLC0415

        class _Broken:
            returncode = 0
            stdout = "not json at all"

        monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: _Broken())
        jobs, note = preflight.decide("origin/main")
        assert all(jobs.values()), "a selector we cannot read must run every job, not none"
        assert note, "and it must say why"


class TestNoCommandTrustsLocalMain:
    """`git diff main...HEAD` in a worktree reviews somebody else's merged PRs."""

    STALE = re.compile(r"(?<!origin/)\bmain\.\.\.|git (diff|rebase|merge) main\b|\.\.\.main\b")

    @pytest.mark.parametrize("path", COMMANDS + AGENTS, ids=lambda p: p.name)
    def test_the_base_ref_is_always_remote(self, path: Path):
        offenders = [
            f"{path.relative_to(ROOT)}:{n}: {line.strip()}"
            for n, line in enumerate(path.read_text().splitlines(), 1)
            if self.STALE.search(line)
        ]
        assert not offenders, (
            "these reference local `main`, which in a worktree is routinely several commits behind "
            "origin/main. Use `origin/main`:\n" + "\n".join(offenders)
        )


class TestWorktreeIsolation:
    """What a worktree owns alone, and what it deliberately shares.

    Worktrees share one common gitdir, so a hook installed into `.git/hooks`
    is installed for all of them, pinned to whichever venv wrote it last. These
    pin the arrangement that replaced it — nothing else in the suite implies it,
    because it lives in git config and a shell script.
    """

    HOOK = ROOT / ".githooks" / "pre-commit"
    PROVISION = ROOT / "scripts" / "provision.sh"

    def test_the_hook_is_tracked_and_executable(self):
        """A generated hook leaves an unprovisioned worktree running nothing at all."""
        assert self.HOOK.is_file(), "the per-worktree pre-commit hook is missing"
        assert self.HOOK.stat().st_mode & 0o111, f"chmod +x {self.HOOK.relative_to(ROOT)}"

    def test_the_hook_resolves_the_venv_from_the_tree_being_committed(self):
        text = self.HOOK.read_text()
        assert "--show-toplevel" in text, "the hook must find the worktree it is running in"
        assert "/.venv/bin/pre-commit" in text
        assert "/Users/" not in text, "a baked absolute path is the bug this replaced"

    def test_provisioning_points_git_at_the_per_worktree_hooks(self):
        text = self.PROVISION.read_text()
        assert "core.hooksPath .githooks" in text, (
            "the path is relative on purpose: git chdirs to the working-tree root before "
            "running a hook, so one shared setting gives every worktree its own"
        )
        # Prose may name it; nothing may run it — it writes into the SHARED
        # .git/hooks, which is exactly what broke.
        runs = [ln for ln in text.splitlines() if "pre-commit install" in ln and not ln.lstrip().startswith("#")]
        assert not runs, runs

    def test_the_repair_target_sets_it_too(self):
        """The main checkout never runs provision.sh."""
        _, body = _recipe("pre-commit")
        assert "core.hooksPath .githooks" in body

    def test_the_generated_port_block_is_never_committed(self):
        ignored = (ROOT / ".gitignore").read_text()
        assert "/.worktree.env" in ignored, "a worktree's ports would follow it into every diff"

    def test_credentials_stay_machine_global(self):
        """Data is per-worktree; API keys are the user's, not the branch's."""
        text = (ROOT / "src" / "yeaboi" / "paths.py").read_text()
        assert "ENV_FILE = DEFAULT_ROOT_DIR" in text, (
            "the bootstrap .env must not move with YEABOI_HOME — it is what can set it"
        )


class TestTheSharedTooling:
    """The plugin's commands speak to this repo only through Make and repo-notes."""

    NOTES = ROOT / ".claude" / "repo-notes.md"

    def test_the_pin_is_a_sha(self):
        rev = (ROOT / ".tooling-rev").read_text().strip()
        assert re.fullmatch(r"[0-9a-f]{40}", rev), (
            f".tooling-rev must hold one full commit sha of yeaboi-tooling, got {rev!r}"
        )

    def test_the_bootstrap_runs_before_the_include(self):
        text = MAKEFILE.read_text()
        assert "scripts/tooling-sync.sh" in text
        assert text.index("scripts/tooling-sync.sh") < text.index("include $(TOOLING)/mk/common.mk"), (
            "the include would fail on a fresh worktree, where .tooling/ does not exist yet"
        )

    def test_the_makefile_still_defines_every_shared_target(self):
        """`make tooling-check` asserts this at runtime; this catches it without a clone."""
        text = MAKEFILE.read_text()
        for target in ("lint", "test", "test-fast", "test-scoped", "ship-gate"):
            assert re.search(rf"^{re.escape(target)}:", text, re.MULTILINE), (
                f"the devkit plugin's commands and hooks call `make {target}` on every repo"
            )

    def test_the_notes_carry_the_conflict_playbook(self):
        playbook = self.NOTES.read_text()
        for path in ("contracts/web", "uv.lock", "CURRENT_SCHEMA_VERSION", "changelog_data.json"):
            assert path in playbook, f"the rebase playbook says nothing about {path}"
        assert "make web-types" in playbook, "a conflicted contract is regenerated, never chosen"

    def test_the_notes_carry_what_ship_asks_for(self):
        """/ship reads these out of the repo rather than hardcoding one repo's facts."""
        notes = self.NOTES.read_text()
        assert "SKIP=unit-tests" in notes, "the commit step needs to know which test hook to skip"
        assert "auto-version" in notes, (
            "the push step needs to know CI rewrites this branch — a force-push over that commit "
            "is how the version bump gets lost"
        )
        assert "make ship-gate" in notes

    @pytest.mark.parametrize("gone", ["go/internal", "src/yeaboi/web/static", "frontend/", "desktop/"])
    def test_the_playbook_does_not_name_a_deleted_tree(self, gone):
        """Each of these was once a place a rebase could conflict, and is now in
        another repo or nowhere. A playbook naming one sends the next person to
        resolve a file that does not exist."""
        assert gone not in self.NOTES.read_text()
