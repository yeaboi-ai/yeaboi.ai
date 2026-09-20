"""The registry that decides which tests CI runs must never lose a test.

`scripts/test_scope.py` is a test *selector*, and a selector's failure mode is
silence: it drops a file, the file stops running, and every check stays green.
Nothing else in the repo would notice — that is the whole reason the always-run
set exists, and it is the reason this file is stricter than it looks.

Two-way totality is the load-bearing assertion, in the style
`test_surface_parity.py` established. Every source file must be claimed by an
area or by the global set; every test file must be reachable from some area or
from `ALWAYS`. A new module with no charter, a renamed test, a glob that has
quietly stopped matching — all three fail here, in the run that introduces them,
rather than by a regression shipping months later.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Every key `--github-output` writes. `test_workflow_schema.py` asserts ci.yml's
# fail-safe fallback writes exactly these, so the two cannot drift.
GITHUB_OUTPUT_KEYS = frozenset({"full", "unit_paths", "slow_paths", "package", "eval", "compat"})

# Loaded by path: scripts/ is not a package, and the module must be importable
# without installing anything (CI runs it before `uv sync`).
_spec = importlib.util.spec_from_file_location("test_scope", ROOT / "scripts" / "test_scope.py")
assert _spec and _spec.loader
scope_mod = importlib.util.module_from_spec(_spec)
sys.modules["test_scope"] = scope_mod
_spec.loader.exec_module(scope_mod)


def _rel(paths):
    return sorted(str(p.relative_to(ROOT)) for p in paths)


def source_files() -> list[str]:
    return _rel(ROOT.glob("src/yeaboi/**/*.py"))


def collected_test_files() -> list[str]:
    found = list(ROOT.glob("tests/unit/**/*.py")) + list(ROOT.glob("tests/test_*.py"))
    return [p for p in _rel(found) if "__snapshots__" not in p and not p.endswith("__init__.py")]


class TestTotality:
    """Both directions, because a selector can fail by omission at either end."""

    @pytest.mark.parametrize("path", source_files())
    def test_every_source_file_is_claimed(self, path):
        claimed = scope_mod.area_for(path) is not None or scope_mod._matches(path, scope_mod.GLOBAL)
        assert claimed, (
            f"{path} is claimed by no area and is not global — add it to an Area's `src` in "
            "scripts/test_scope.py, or to GLOBAL "
            "if a change to it can reach everything"
        )

    @pytest.mark.parametrize("path", collected_test_files())
    def test_every_test_file_is_reachable(self, path):
        reachable = scope_mod._matches(path, scope_mod.ALWAYS) or any(
            scope_mod._matches(path, area.tests) for area in scope_mod.AREAS
        )
        assert reachable, (
            f"{path} is selected by nothing — a scoped CI run would never execute it. Add it to an "
            "Area's `tests` in scripts/test_scope.py, or to ALWAYS if it guards the repo rather "
            "than a module"
        )


class TestTheGlobsStillMatch:
    """A pattern that matches nothing is exactly as good as no pattern at all,
    and it looks identical in a diff."""

    @pytest.mark.parametrize("pattern", scope_mod.ALWAYS)
    def test_every_always_glob_matches_something(self, pattern):
        assert list(ROOT.glob(pattern)), f"ALWAYS pattern {pattern!r} matches no file"

    @pytest.mark.parametrize(
        "name,pattern",
        [(area.name, pattern) for area in scope_mod.AREAS for pattern in area.tests],
    )
    def test_every_area_glob_matches_something(self, name, pattern):
        assert list(ROOT.glob(pattern)), f"{name}: test pattern {pattern!r} matches no file"

    @pytest.mark.parametrize("path", scope_mod.GLOBAL)
    def test_every_global_path_exists(self, path):
        target = ROOT / path.rstrip("/")
        assert target.exists(), f"GLOBAL entry {path!r} does not exist — a rename left it dead"

    @pytest.mark.parametrize("prefix", [t for job in scope_mod.JOBS for t in job.triggers])
    def test_every_job_trigger_exists(self, prefix):
        target = ROOT / prefix.rstrip("/")
        assert target.exists(), f"job trigger {prefix!r} does not exist — a rename left it dead"


class TestFailingSafe:
    """Every path that cannot be classified must widen the run, never narrow it."""

    def test_an_unknown_path_runs_everything(self):
        assert scope_mod.resolve(["some/new/thing.txt"]).full is True

    def test_an_empty_diff_runs_everything(self):
        """An empty list is far more often a diff we failed to read than a change
        that touched nothing, and the two are indistinguishable here."""
        assert scope_mod.resolve([]).full is True

    def test_a_global_path_runs_everything(self):
        assert scope_mod.resolve(["tests/conftest.py"]).full is True
        assert scope_mod.resolve(["src/yeaboi/sessions.py"]).full is True

    def test_a_new_unclaimed_test_file_runs_everything(self):
        assert scope_mod.resolve(["tests/unit/test_brand_new_thing.py"]).full is True

    def test_the_full_scope_names_every_job(self):
        jobs = scope_mod.resolve(["some/new/thing.txt"])
        for job in scope_mod.JOBS:
            assert jobs.full or job.name in jobs.jobs


class TestTheAlwaysSetIsAlwaysThere:
    def test_a_narrow_change_still_runs_the_guards(self):
        selected = set(scope_mod.unit_paths(scope_mod.resolve(["src/yeaboi/standup/engine.py"])))
        assert "tests/unit/test_surface_parity.py" in selected
        assert "tests/unit/test_web_assets.py" in selected

    def test_a_prose_only_change_still_runs_the_guards(self):
        """`.claude/` is treated as inert, which is only safe because the tests
        that read it are in ALWAYS. If they ever leave it, that prefix has to
        move to GLOBAL in the same commit."""
        scope = scope_mod.resolve([".claude/repo-notes.md"])
        selected = set(scope_mod.unit_paths(scope))
        assert not scope.full
        assert any("test_claude_plugin" in path or "test_ship_gate" in path for path in selected)

    def test_it_does_not_drag_in_an_unrelated_area(self):
        selected = set(scope_mod.unit_paths(scope_mod.resolve(["src/yeaboi/standup/engine.py"])))
        assert not any("test_poker" in path for path in selected)
        assert not any("test_reporting" in path for path in selected)


class TestSelectionIsRunnable:
    """Whatever comes out has to be something pytest will accept.

    A glob that resolves to nothing makes pytest exit 4 — "file or directory not
    found" — which reads in CI as a broken job rather than as an empty selection.
    """

    @pytest.mark.parametrize(
        "changed",
        [
            ["src/yeaboi/standup/engine.py"],
            ["src/yeaboi/poker/board.py"],
            ["contracts/site.json"],
            ["contracts/web/enums.json"],
            ["src/yeaboi/ui/mode_select/screens/_screens.py"],
        ],
    )
    def test_every_selected_path_exists(self, changed):
        for path in scope_mod.unit_paths(scope_mod.resolve(changed)):
            assert (ROOT / path).exists(), f"{path} was selected but does not exist"

    def test_a_scoped_selection_is_a_subset_of_the_full_one(self):
        scoped = set(scope_mod.unit_paths(scope_mod.resolve(["src/yeaboi/poker/board.py"])))
        every = set(collected_test_files())
        assert scoped <= every, sorted(scoped - every)

    def test_the_slow_lane_is_all_or_nothing(self):
        """572 tests and about 100s — cheap enough that splitting it would buy
        less than the risk of getting the split wrong."""
        assert scope_mod.slow_paths(scope_mod.resolve(["src/yeaboi/poker/board.py"])) == list(scope_mod.FULL_SLOW)


class TestEveryRegistryEntryIsReachable:
    """A pattern that matches nothing looks exactly like one that works.

    The totality guards above ask "is every file claimed by something". These
    ask the mirror question — "does every claim reach a file" — which is the
    half that caught two real bugs: `security` named
    `src/yeaboi/sharing/access.py` and `sharing/gate.py`, but `artifacts-sharing`
    claims the whole `sharing/` directory and was listed first, so under
    first-match-wins those two entries were dead and a change to the share
    access-control code never ran a single guardrail test.
    """

    def test_every_src_entry_resolves_to_its_own_area(self):
        """The anti-shadowing check. `area_for` is most-specific-wins now, so a
        directory can no longer swallow a file another area names outright."""
        shadowed = []
        for area in scope_mod.AREAS:
            for entry in area.src:
                probe = f"{entry}__probe__.py" if entry.endswith("/") else entry
                won = scope_mod.area_for(probe)
                if won is None or won.name != area.name:
                    shadowed.append(f"{area.name} claims {entry}, but it resolves to {won.name if won else None}")
        assert not shadowed, "unreachable registry entries:\n  " + "\n  ".join(shadowed)

    def test_every_src_entry_matches_something_in_the_tree(self):
        missing = []
        for area in scope_mod.AREAS:
            for entry in area.src:
                if entry.endswith("/"):
                    if not (ROOT / entry).is_dir():
                        missing.append(f"{area.name}: {entry}")
                elif "*" in entry:
                    if not list(ROOT.glob(entry)):
                        missing.append(f"{area.name}: {entry}")
                elif not (ROOT / entry).exists():
                    missing.append(f"{area.name}: {entry}")
        assert not missing, "registry entries matching no file:\n  " + "\n  ".join(missing)

    def test_the_inert_list_matches_something(self):
        missing = [
            entry
            for entry in scope_mod.INERT
            if not (ROOT / entry).is_dir() and not (ROOT / entry).exists() and not list(ROOT.glob(entry))
        ]
        assert not missing, f"INERT entries matching nothing: {missing}"


class TestTheMcpServerTestFollowsItsTools:
    """`mcp/tools_<mode>.py` belongs to the mode, per AGENTS.md — but the test
    that drives every tool end-to-end through `create_app` lives in one file
    owned by `platform`. Without this coupling a tools change ran the mode's
    tests and never the server's, and `test_surface_parity.py` (in ALWAYS) only
    catches schema drift, not behaviour."""

    def _tool_modules(self):
        return sorted(p for p in (ROOT / "src" / "yeaboi" / "mcp").glob("tools_*.py"))

    def test_there_is_at_least_one_tools_module(self):
        assert self._tool_modules(), "the glob below would pass vacuously"

    def test_a_tools_change_runs_the_mcp_server_test(self):
        missed = []
        for module in self._tool_modules():
            rel = module.relative_to(ROOT).as_posix()
            selected = scope_mod.unit_paths(scope_mod.resolve([rel]))
            if not any("test_mcp_server.py" in path for path in selected):
                missed.append(rel)
        assert not missed, f"these select no MCP server test: {missed}"


class TestARenameReportsBothSides:
    """`git diff --name-only` prints a move as its destination only, so a
    cross-area `git mv` selected the new area and never the old one — whose
    tests still import the path the file left, and are exactly the ones a move
    breaks. The totality guard stayed green throughout, because the file *is*
    claimed; just by the wrong area."""

    def test_the_merge_base_diff_disables_rename_detection(self):
        source = (ROOT / "scripts" / "test_scope.py").read_text(encoding="utf-8")
        assert '"--no-renames"' in source, "a rename would report only its destination"

    def test_the_ci_diff_disables_rename_detection_too(self):
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "--no-renames" in workflow, "ci.yml resolves the diff itself and needs the same flag"

    def test_a_working_tree_rename_reports_both_paths(self, monkeypatch):
        monkeypatch.setattr(
            scope_mod.subprocess,
            "run",
            lambda *a, **k: type(
                "R", (), {"returncode": 0, "stdout": "R  src/yeaboi/standup/a.py -> src/yeaboi/analysis/a.py\n"}
            )(),
        )
        paths = scope_mod.changed_from_git(None, working_tree=True)
        assert paths == ["src/yeaboi/standup/a.py", "src/yeaboi/analysis/a.py"]

    def test_a_cross_area_move_selects_both_areas(self):
        scope = scope_mod.resolve(["src/yeaboi/standup/a.py", "src/yeaboi/analysis/a.py"])
        assert {"standup", "analysis"} <= scope.areas


class TestTheCiEntryPoints:
    """`changed_from_git`, `explain` and `main` are what CI actually calls, and
    they had no tests at all. Given that `unit` (a required context) depends on
    this script's exit code, an uncaught exception in `main` is the thing that
    skips a required check."""

    def test_the_working_tree_parser_handles_every_porcelain_shape(self, monkeypatch):
        porcelain = " M src/yeaboi/poker/board.py\n?? scripts/new_thing.py\nA  tests/unit/test_x.py\n"
        monkeypatch.setattr(
            scope_mod.subprocess,
            "run",
            lambda *a, **k: type("R", (), {"returncode": 0, "stdout": porcelain})(),
        )
        assert scope_mod.changed_from_git(None, working_tree=True) == [
            "src/yeaboi/poker/board.py",
            "scripts/new_thing.py",
            "tests/unit/test_x.py",
        ]

    def test_a_failed_git_call_reports_nothing_which_means_everything(self, monkeypatch):
        monkeypatch.setattr(
            scope_mod.subprocess,
            "run",
            lambda *a, **k: type("R", (), {"returncode": 128, "stdout": ""})(),
        )
        assert scope_mod.changed_from_git("origin/main", working_tree=False) == []
        assert scope_mod.resolve([]).full is True

    def test_an_unresolvable_base_is_not_an_empty_diff(self, monkeypatch):
        monkeypatch.setattr(
            scope_mod.subprocess,
            "run",
            lambda *a, **k: type("R", (), {"returncode": 0, "stdout": "\n"})(),
        )
        assert scope_mod.changed_from_git("origin/main", working_tree=False) == []

    def test_explain_names_the_areas_and_survives_a_full_scope(self):
        scoped = scope_mod.explain(scope_mod.resolve(["src/yeaboi/poker/board.py"]))
        assert "poker" in scoped
        assert "FULL" in scope_mod.explain(scope_mod.resolve([]))

    def test_main_writes_every_github_output_key(self, tmp_path, monkeypatch, capsys):
        changed = tmp_path / "changed.txt"
        changed.write_text("src/yeaboi/poker/board.py\n", encoding="utf-8")
        out = tmp_path / "out.txt"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))
        assert scope_mod.main(["--changed-files", str(changed), "--github-output"]) == 0
        written = dict(line.split("=", 1) for line in out.read_text(encoding="utf-8").splitlines())
        assert set(written) == GITHUB_OUTPUT_KEYS
        assert written["full"] == "false"
        assert "tests/unit/test_poker_board.py" in written["unit_paths"]

    def test_main_refuses_github_output_with_no_env(self, tmp_path, monkeypatch):
        changed = tmp_path / "changed.txt"
        changed.write_text("src/yeaboi/poker/board.py\n", encoding="utf-8")
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        assert scope_mod.main(["--changed-files", str(changed), "--github-output"]) == 2

    def test_main_reads_stdin(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("src/yeaboi/poker/board.py\n"))
        assert scope_mod.main(["--changed-files", "-", "--unit-paths"]) == 0
        assert "test_poker" in capsys.readouterr().out
