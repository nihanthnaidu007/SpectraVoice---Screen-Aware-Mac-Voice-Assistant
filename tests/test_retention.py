"""Shared retention tests (W3 D4.3): one mechanism, two consumers (R2)."""

import os
import sys
import time
from unittest import mock

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app.utils import retention


def age_entry(path: str, hours: float) -> None:
    old = time.time() - hours * 3600
    os.utime(path, (old, old))


class TestPruneExpired:
    def test_removes_only_expired_files(self, tmp_path):
        keep = tmp_path / "keep.wav"
        stale = tmp_path / "stale.wav"
        keep.write_bytes(b"x")
        stale.write_bytes(b"x")
        age_entry(str(stale), hours=48)
        removed = retention.prune_expired(str(tmp_path), retention_hours=24.0)
        assert removed == 1 and keep.exists() and not stale.exists()

    def test_zero_retention_keeps_everything(self, tmp_path):
        stale = tmp_path / "old.wav"
        stale.write_bytes(b"x")
        age_entry(str(stale), hours=10_000)
        assert retention.prune_expired(str(tmp_path), 0.0) == 0
        assert stale.exists()

    def test_missing_directory_is_not_an_error(self, tmp_path):
        assert retention.prune_expired(str(tmp_path / "nope"), 24.0) == 0

    def test_directories_pruned_as_units(self, tmp_path):
        """Meeting directories age as a unit — transcript, clips, summary together."""
        meeting = tmp_path / "20250101-000000"
        (meeting / "clips").mkdir(parents=True)
        (meeting / "transcript.jsonl").write_text("{}\n")
        age_entry(str(meeting), hours=72)
        fresh = tmp_path / "20260917-000000"
        fresh.mkdir()
        removed = retention.prune_expired(str(tmp_path), 24.0)
        assert removed == 1
        assert not meeting.exists() and fresh.exists()

    def test_one_unreadable_entry_does_not_block_the_sweep(self, tmp_path):
        keep = tmp_path / "keep.wav"
        bad = tmp_path / "bad.wav"
        keep.write_bytes(b"x")
        bad.write_bytes(b"x")
        age_entry(str(bad), hours=48)
        age_entry(str(keep), hours=48)
        real_remove = os.remove

        def flaky_remove(path):
            if path.endswith("bad.wav"):
                raise OSError("device busy")
            real_remove(path)

        with mock.patch("assistant_app.utils.retention.os.remove", flaky_remove):
            removed = retention.prune_expired(str(tmp_path), 24.0)
        assert removed == 1  # keep.wav went, bad.wav was skipped with a log
        assert bad.exists() and not keep.exists()

    def test_dictation_dir_is_a_supported_consumer(self, tmp_path):
        """The dictation persist_audio directory uses the same mechanism."""
        stale = tmp_path / "dictation_audio" / "dictation_1.wav"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"x")
        age_entry(str(stale), hours=72)
        assert retention.prune_expired(str(tmp_path / "dictation_audio"), 24.0) == 1
        assert not stale.exists()
