"""W4 D2 ops tests — deletes, export, and the H9 retention decoupling.

Deletion is directory-granularity and refuses to run unconfirmed
(structural gate in history_ops, independent of any UI). Export is the
only off-corpus path and never deletes. The kill-switch-off sweep test
proves retention is decoupled from the meeting feature flag (H9).
"""

from __future__ import annotations

import os
import sys
import time

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app.services import history_ops
from assistant_app.services import meeting_store as ms
from assistant_app.services.meeting import run_startup_cleanup


class FakeCfg:
    audio_dir = ""
    language = "english"


class FakeSweepCfg(FakeCfg):
    """Meeting-config surface for the retention sweep (H9)."""

    enabled = False  # the kill-switch is OFF — cleanup must run anyway
    retention_hours = 1.0


def make_cfg(tmp_path) -> FakeCfg:
    cfg = FakeCfg()
    cfg.audio_dir = str(tmp_path / "corpus")
    return cfg


def make_meeting(cfg, text="an utterance"):
    paths = ms.open_meeting(cfg, time.time())
    writer = ms.MeetingTranscriptWriter(paths, "english")
    writer.append_utterance(1, text, 100.0)
    return paths


class TestDeleteMeeting:
    def test_refuses_without_confirmation(self, tmp_path):
        cfg = make_cfg(tmp_path)
        make_meeting(cfg)
        with pytest.raises(history_ops.HistoryRefusalError, match="confirmation"):
            history_ops.delete_meeting(cfg, "whatever", confirm=False)

    def test_deletes_the_whole_directory_only(self, tmp_path):
        cfg = make_cfg(tmp_path)
        doomed = make_meeting(cfg, "delete me")
        survivor = make_meeting(cfg, "keep me")

        assert history_ops.delete_meeting(cfg, doomed.meeting_id, confirm=True) is True

        assert not os.path.isdir(doomed.meeting_dir)  # transcript + everything else, gone
        assert os.path.isdir(survivor.meeting_dir)
        assert os.path.isdir(ms.meeting_root(cfg))  # the corpus root is config-owned storage

    def test_double_delete_is_idempotent(self, tmp_path):
        cfg = make_cfg(tmp_path)
        paths = make_meeting(cfg)
        assert history_ops.delete_meeting(cfg, paths.meeting_id, confirm=True) is True
        assert history_ops.delete_meeting(cfg, paths.meeting_id, confirm=True) is False

    def test_path_traversal_is_a_lookup_error_not_a_delete(self, tmp_path):
        cfg = make_cfg(tmp_path)
        make_meeting(cfg)
        with pytest.raises(LookupError):
            history_ops.delete_meeting(cfg, "../../etc", confirm=True)
        assert os.path.isdir(ms.meeting_root(cfg))


class TestDeleteAll:
    def test_refuses_without_confirmation(self, tmp_path):
        cfg = make_cfg(tmp_path)
        make_meeting(cfg)
        with pytest.raises(history_ops.HistoryRefusalError, match="confirmation"):
            history_ops.delete_all_meetings(cfg, confirm=False)

    def test_removes_every_meeting_but_keeps_the_root(self, tmp_path):
        cfg = make_cfg(tmp_path)
        make_meeting(cfg, "one")
        make_meeting(cfg, "two")

        removed = history_ops.delete_all_meetings(cfg, confirm=True)

        assert removed == 2
        assert os.listdir(ms.meeting_root(cfg)) == []

    def test_empty_or_missing_corpus_removes_nothing(self, tmp_path):
        assert history_ops.delete_all_meetings(make_cfg(tmp_path), confirm=True) == 0


class TestExportMeeting:
    def test_copies_the_directory_and_keeps_the_corpus(self, tmp_path):
        cfg = make_cfg(tmp_path)
        paths = make_meeting(cfg, "export me")
        dest = str(tmp_path / "exports")

        out = history_ops.export_meeting(cfg, paths.meeting_id, dest)

        assert out == os.path.join(dest, paths.meeting_id)
        assert os.path.isdir(out)  # transcript copied with the directory
        assert os.path.exists(paths.transcript_path)  # corpus copy untouched
        with open(os.path.join(out, os.path.basename(paths.transcript_path)), encoding="utf-8") as fh:
            assert "export me" in fh.read()

    def test_never_overwrites_an_existing_destination(self, tmp_path):
        cfg = make_cfg(tmp_path)
        paths = make_meeting(cfg, "export me")
        dest = str(tmp_path / "exports")
        history_ops.export_meeting(cfg, paths.meeting_id, dest)
        with pytest.raises(FileExistsError):
            history_ops.export_meeting(cfg, paths.meeting_id, dest)

    def test_unknown_meeting_is_a_lookup_error(self, tmp_path):
        with pytest.raises(LookupError):
            history_ops.export_meeting(make_cfg(tmp_path), "nope", str(tmp_path / "dest"))

    def test_empty_destination_is_rejected(self, tmp_path):
        cfg = make_cfg(tmp_path)
        paths = make_meeting(cfg)
        with pytest.raises(ValueError, match="destination"):
            history_ops.export_meeting(cfg, paths.meeting_id, "   ")


class TestRetentionRunsRegardlessOfKillSwitch:
    def test_sweep_removes_expired_dirs_with_meeting_disabled(self, tmp_path):
        """H9: retention is one mechanism, decoupled from meeting.enabled.
        With the kill-switch OFF, expired artifacts must still be swept —
        otherwise disabling the feature orphans old dirs forever."""
        cfg = FakeSweepCfg()
        cfg.audio_dir = str(tmp_path / "corpus")
        paths = make_meeting(cfg)
        old = time.time() - 10 * 3600.0
        os.utime(paths.meeting_dir, (old, old))  # directory mtime governs the tree

        removed = run_startup_cleanup(cfg)

        assert removed == 1
        assert not os.path.isdir(paths.meeting_dir)

    def test_zero_retention_keeps_everything_with_meeting_disabled(self, tmp_path):
        cfg = FakeSweepCfg()
        cfg.audio_dir = str(tmp_path / "corpus")
        cfg.retention_hours = 0.0  # 0 = forever, preserved default
        paths = make_meeting(cfg)
        assert run_startup_cleanup(cfg) == 0
        assert os.path.isdir(paths.meeting_dir)
