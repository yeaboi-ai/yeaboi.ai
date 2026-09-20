"""Tests for paths.py — export-dir helpers, the YEABOI_HOME root override, and move_data_tree."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from yeaboi import paths

_HELPERS = [
    (paths.get_analysis_export_dir, "analysis"),
    (paths.get_planning_export_dir, "planning"),
    (paths.get_standup_export_dir, "standup"),
    (paths.get_retro_export_dir, "retro"),
    (paths.get_performance_export_dir, "performance"),
    (paths.get_reporting_export_dir, "reporting"),
    (paths.get_solo_export_dir, "solo"),
]


class TestExportDirHelpers:
    @pytest.mark.parametrize(("helper", "subdir"), _HELPERS)
    def test_defaults_under_constant(self, helper, subdir, monkeypatch, tmp_path):
        # Monkeypatch the module constant (the pattern existing suites rely on)
        const = f"{subdir.upper()}_EXPORTS_DIR"
        monkeypatch.setattr(paths, const, tmp_path / subdir)
        d = helper("MyProj")
        assert d == tmp_path / subdir / "myproj"
        assert d.is_dir()

    def test_performance_empty_key_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setattr(paths, "PERFORMANCE_EXPORTS_DIR", tmp_path / "performance")
        assert paths.get_performance_export_dir("").name == "engineer"

    def test_reporting_empty_key_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setattr(paths, "REPORTING_EXPORTS_DIR", tmp_path / "reporting")
        assert paths.get_reporting_export_dir("").name == "report"

    def test_changelog_seen_path_under_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(paths, "CHANGELOG_SEEN_FILE", tmp_path / "data" / "changelog_seen.json")
        p = paths.get_changelog_seen_path()
        assert p == tmp_path / "data" / "changelog_seen.json"
        assert p.parent.is_dir()  # data dir created, file itself may not exist yet

    def test_reporting_prefs_path_under_data_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(paths, "REPORTING_PREFS_FILE", tmp_path / "data" / "reporting_prefs.json")
        p = paths.get_reporting_prefs_path()
        assert p == tmp_path / "data" / "reporting_prefs.json"
        assert p.parent.is_dir()  # data dir created, file itself may not exist yet


class TestAgentwatchLogDir:
    """The Agents family's log directory (AGENTS.md: every mode logs to its own)."""

    def test_creates_under_the_constant(self, monkeypatch, tmp_path):
        monkeypatch.setattr(paths, "AGENTWATCH_LOGS_DIR", tmp_path / "logs" / "agentwatch")
        d = paths.get_agentwatch_log_dir()
        assert d == tmp_path / "logs" / "agentwatch"
        assert d.is_dir()

    def test_idempotent(self, monkeypatch, tmp_path):
        monkeypatch.setattr(paths, "AGENTWATCH_LOGS_DIR", tmp_path / "logs" / "agentwatch")
        assert paths.get_agentwatch_log_dir() == paths.get_agentwatch_log_dir()

    def test_lives_under_the_logs_root(self):
        assert paths.LOGS_DIR in paths.AGENTWATCH_LOGS_DIR.parents


class TestSoloDirs:
    """The Solo world's own export and log directories."""

    def test_export_empty_key_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setattr(paths, "SOLO_EXPORTS_DIR", tmp_path / "solo")
        assert paths.get_solo_export_dir("").name == "review"

    def test_log_dir_creates_under_the_constant(self, monkeypatch, tmp_path):
        monkeypatch.setattr(paths, "SOLO_LOGS_DIR", tmp_path / "logs" / "solo")
        d = paths.get_solo_log_dir()
        assert d == tmp_path / "logs" / "solo" and d.is_dir()
        assert paths.get_solo_log_dir() == d

    def test_lives_under_the_roots(self):
        assert paths.LOGS_DIR in paths.SOLO_LOGS_DIR.parents
        assert paths.EXPORTS_DIR in paths.SOLO_EXPORTS_DIR.parents


class TestTranscriptsDir:
    """The managed standup-transcript drop folder."""

    def test_creates_under_root(self, monkeypatch, tmp_path):
        monkeypatch.setattr(paths, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
        d = paths.get_transcripts_dir()
        assert d == tmp_path / "transcripts"
        assert d.is_dir()

    def test_idempotent(self, monkeypatch, tmp_path):
        monkeypatch.setattr(paths, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
        assert paths.get_transcripts_dir() == paths.get_transcripts_dir()

    def test_lives_inside_the_yeaboi_root(self):
        # Being under ROOT_DIR is the whole reason this folder needs no
        # path-consent prompt — fs_policy already allows the tree.
        assert paths.TRANSCRIPTS_DIR.parent == paths.ROOT_DIR


class TestSafeKey:
    """_safe_key must never let a key escape its export root."""

    def test_normal_key_lowercased(self):
        assert paths._safe_key("MyProj", "project") == "myproj"

    def test_empty_key_falls_back(self):
        assert paths._safe_key("", "project") == "project"
        assert paths._safe_key("   ", "project") == "project"

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("../evil", "evil"),
            ("../../etc/passwd", "etc-passwd"),
            ("a/b", "a-b"),
            ("a\\b", "a-b"),
            ("..", "project"),
            ("./.", "project"),
            ("/absolute/path", "absolute-path"),
        ],
    )
    def test_traversal_neutralized(self, key, expected):
        assert paths._safe_key(key, "project") == expected

    def test_export_helper_cannot_escape(self, monkeypatch, tmp_path):
        monkeypatch.setattr(paths, "ANALYSIS_EXPORTS_DIR", tmp_path / "exports" / "analysis")
        d = paths.get_analysis_export_dir("../../outside")
        assert d.is_relative_to(tmp_path / "exports" / "analysis")


class TestGetDbPathPermissions:
    """get_db_path() hardens the data dir (0o700) and DB file (0o600)."""

    @pytest.fixture(autouse=True)
    def _isolated_db(self, monkeypatch, tmp_path):
        monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(paths, "DB_PATH", tmp_path / "data" / "sessions.db")
        monkeypatch.setattr(paths, "LEGACY_DB_PATH", tmp_path / "sessions.db")

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_creates_restricted_db_and_dir(self):
        db = paths.get_db_path()
        assert (db.stat().st_mode & 0o777) == 0o600
        assert (db.parent.stat().st_mode & 0o777) == 0o700

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_repairs_existing_lax_db(self, tmp_path):
        (tmp_path / "data").mkdir()
        db = tmp_path / "data" / "sessions.db"
        db.touch(mode=0o644)
        db.chmod(0o644)
        paths.get_db_path()
        assert (db.stat().st_mode & 0o777) == 0o600

    def test_legacy_rename_still_works(self, tmp_path):
        (tmp_path / "sessions.db").write_bytes(b"")
        db = paths.get_db_path()
        assert db == tmp_path / "data" / "sessions.db"
        assert db.exists()
        assert not (tmp_path / "sessions.db").exists()


class TestResolveRoot:
    """YEABOI_HOME relocates the whole data tree (resolved once at import time)."""

    @pytest.fixture
    def no_checkout_marker(self, monkeypatch):
        """Ignore the worktree this suite is running inside.

        These cases are about the env-var/default half of the resolution; the
        checkout marker gets its own class below.
        """
        monkeypatch.setattr(paths, "_checkout_home", lambda: None)

    def test_default_is_home_yeaboi(self, monkeypatch, no_checkout_marker):
        monkeypatch.delenv("YEABOI_HOME", raising=False)
        assert paths._resolve_root() == Path.home() / ".yeaboi"

    def test_override_used_when_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("YEABOI_HOME", str(tmp_path / "custom"))
        assert paths._resolve_root() == tmp_path / "custom"

    def test_tilde_expansion(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("YEABOI_HOME", "~/yb-data")
        assert paths._resolve_root() == tmp_path / "yb-data"

    def test_blank_override_ignored(self, monkeypatch, no_checkout_marker):
        monkeypatch.setenv("YEABOI_HOME", "   ")
        assert paths._resolve_root() == Path.home() / ".yeaboi"

    def test_env_file_pinned_to_default_home(self):
        # The bootstrap .env holds YEABOI_HOME itself, so it never moves.
        assert paths.ENV_FILE == paths.DEFAULT_ROOT_DIR / ".env"
        assert paths.DEFAULT_ROOT_DIR == Path.home() / ".yeaboi"


class TestCheckoutHome:
    """A worktree pins its own data home, so parallel worktrees do not share one.

    `wt.sh` writes `.worktree.env` beside the source tree; `paths` is imported
    before anything can load a dotenv, so this is the only place early enough
    to read it.
    """

    def _marker(self, monkeypatch, tmp_path, body: str) -> None:
        """Point _checkout_home() at a fake checkout root holding *body*."""
        root = tmp_path / "checkout"
        (root / "src" / "yeaboi").mkdir(parents=True)
        (root / ".worktree.env").write_text(body)
        monkeypatch.setattr(paths, "__file__", str(root / "src" / "yeaboi" / "paths.py"))

    def test_the_marker_moves_the_root(self, monkeypatch, tmp_path):
        self._marker(monkeypatch, tmp_path, "export YEABOI_HOME=/tmp/wt-home\n")
        assert paths._checkout_home() == Path("/tmp/wt-home")

    def test_a_bare_assignment_works_too(self, monkeypatch, tmp_path):
        self._marker(monkeypatch, tmp_path, "YEABOI_HOME=/tmp/bare\n")
        assert paths._checkout_home() == Path("/tmp/bare")

    def test_an_explicit_env_var_still_wins(self, monkeypatch, tmp_path):
        self._marker(monkeypatch, tmp_path, "export YEABOI_HOME=/tmp/wt-home\n")
        monkeypatch.setenv("YEABOI_HOME", str(tmp_path / "explicit"))
        assert paths._resolve_root() == tmp_path / "explicit"

    def test_other_keys_are_ignored(self, monkeypatch, tmp_path):
        self._marker(monkeypatch, tmp_path, "export RETRO_PORT=20100\nexport YEABOI_WT_SLOT=1\n")
        assert paths._checkout_home() is None

    def test_no_marker_means_no_opinion(self, monkeypatch, tmp_path):
        root = tmp_path / "wheel-ish"
        (root / "src" / "yeaboi").mkdir(parents=True)
        monkeypatch.setattr(paths, "__file__", str(root / "src" / "yeaboi" / "paths.py"))
        assert paths._checkout_home() is None

    def test_an_unreadable_marker_never_raises(self, monkeypatch, tmp_path):
        # This runs at import, before logging exists: a bad file must degrade
        # to the default, not make the package unimportable.
        root = tmp_path / "checkout"
        (root / "src" / "yeaboi").mkdir(parents=True)
        (root / ".worktree.env").mkdir()  # a directory where a file belongs
        monkeypatch.setattr(paths, "__file__", str(root / "src" / "yeaboi" / "paths.py"))
        assert paths._checkout_home() is None

    def test_credentials_do_not_move_with_the_data(self, monkeypatch, tmp_path):
        """The whole point: one set of API keys still serves every worktree."""
        monkeypatch.delenv("YEABOI_HOME", raising=False)  # conftest pins one for the suite
        self._marker(monkeypatch, tmp_path, "export YEABOI_HOME=/tmp/wt-home\n")
        assert paths._resolve_root() == Path("/tmp/wt-home")
        assert paths.ENV_FILE == Path.home() / ".yeaboi" / ".env"


class TestMoveDataTree:
    def _make_src(self, monkeypatch, tmp_path) -> Path:
        src = tmp_path / "old-home"
        src.mkdir()
        monkeypatch.setenv("YEABOI_HOME", str(src))
        return src

    def test_moves_children(self, monkeypatch, tmp_path):
        src = self._make_src(monkeypatch, tmp_path)
        (src / "data").mkdir()
        (src / "data" / "sessions.db").write_text("db")
        (src / "repl-history").write_text("hist")
        dest = tmp_path / "new-home"

        ok, msg = paths.move_data_tree(dest)
        assert ok
        assert (dest / "data" / "sessions.db").read_text() == "db"
        assert (dest / "repl-history").exists()
        assert not (src / "data").exists()
        assert "2 item(s)" in msg

    def test_env_file_is_skipped(self, monkeypatch, tmp_path):
        src = self._make_src(monkeypatch, tmp_path)
        (src / ".env").write_text("SECRET=1")
        dest = tmp_path / "new-home"

        ok, _ = paths.move_data_tree(dest)
        assert ok
        assert (src / ".env").exists()
        assert not (dest / ".env").exists()

    def test_existing_destination_child_skipped(self, monkeypatch, tmp_path):
        src = self._make_src(monkeypatch, tmp_path)
        (src / "exports").mkdir()
        (src / "exports" / "a.md").write_text("src")
        dest = tmp_path / "new-home"
        (dest / "exports").mkdir(parents=True)

        ok, msg = paths.move_data_tree(dest)
        assert ok
        assert (src / "exports" / "a.md").exists()  # left in place, not clobbered
        assert "skipped" in msg

    def test_missing_source_is_noop(self, monkeypatch, tmp_path):
        monkeypatch.setenv("YEABOI_HOME", str(tmp_path / "never-created"))
        ok, msg = paths.move_data_tree(tmp_path / "new-home")
        assert ok
        assert "No existing data" in msg

    def test_same_location_is_noop(self, monkeypatch, tmp_path):
        src = self._make_src(monkeypatch, tmp_path)
        (src / "data").mkdir()
        ok, msg = paths.move_data_tree(src)
        assert ok
        assert (src / "data").exists()
        assert "nothing to move" in msg

    def test_default_source_when_no_override(self, monkeypatch, tmp_path):
        # With YEABOI_HOME unset the source is ~/.yeaboi (here: a faked HOME).
        monkeypatch.delenv("YEABOI_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(paths, "DEFAULT_ROOT_DIR", tmp_path / ".yeaboi")
        (tmp_path / ".yeaboi").mkdir()
        (tmp_path / ".yeaboi" / "scrum-docs").mkdir()
        dest = tmp_path / "elsewhere"

        ok, _ = paths.move_data_tree(dest)
        assert ok
        assert (dest / "scrum-docs").exists()


class TestRunDir:
    """Where the generated tunnel ingress files live."""

    def test_it_is_created_owner_only(self, tmp_path, monkeypatch):
        import stat

        from yeaboi import paths

        monkeypatch.setattr(paths, "ROOT_DIR", tmp_path / ".yeaboi")
        monkeypatch.setattr(paths, "RUN_DIR", tmp_path / ".yeaboi" / "run")
        run_dir = paths.get_run_dir()
        assert run_dir.is_dir()
        # The ingress files name the credentials path; group/other have no
        # business reading them.
        assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700

    def test_it_is_idempotent(self, tmp_path, monkeypatch):
        from yeaboi import paths

        monkeypatch.setattr(paths, "ROOT_DIR", tmp_path / ".yeaboi")
        monkeypatch.setattr(paths, "RUN_DIR", tmp_path / ".yeaboi" / "run")
        assert paths.get_run_dir() == paths.get_run_dir()

    def test_the_bin_dir_is_owner_only_too(self, tmp_path, monkeypatch):
        """It holds the cloudflared binary and its recorded digest — another
        local user able to rewrite either runs their code on the next share."""
        import stat

        from yeaboi import paths

        monkeypatch.setattr(paths, "ROOT_DIR", tmp_path / ".yeaboi")
        monkeypatch.setattr(paths, "BIN_DIR", tmp_path / ".yeaboi" / "bin")
        bin_dir = paths.get_bin_dir()
        assert bin_dir.is_dir()
        assert stat.S_IMODE(bin_dir.stat().st_mode) == 0o700


class TestConnectionStatusPath:
    def test_it_sits_in_the_data_dir_and_creates_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(paths, "CONNECTION_STATUS_FILE", tmp_path / "data" / "connection_status.json")
        got = paths.get_connection_status_path()
        assert got.name == "connection_status.json"
        assert got.parent.is_dir()
