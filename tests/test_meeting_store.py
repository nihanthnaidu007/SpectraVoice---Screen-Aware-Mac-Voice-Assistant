"""Meeting store tests (W3 D1.4/D4.3): JSONL layout, crash-safe flush, gaps.

The crash-mid-meeting guarantee is exercised here at the store level: every
append flushes before returning, so "crash" (abandon the writer, read the
file) never loses a record that had been returned to the caller.
"""

import json
import os
import sys
import time
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app.services import meeting_store as ms


class FakeCfg:
    audio_dir = ""
    language = "english"


def make_paths(tmp_path, monkeypatch) -> ms.MeetingPaths:
    """A REAL meeting directory via open_meeting — meta line already written."""
    monkeypatch.chdir(tmp_path)
    return ms.open_meeting(FakeCfg(), time.time())


class TestOpenMeeting:
    def test_creates_directory_and_meta_line(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        started = time.time()
        paths = ms.open_meeting(FakeCfg(), started)
        assert os.path.isdir(paths.meeting_dir)
        records = ms.read_transcript(paths.transcript_path)
        assert len(records) == 1
        meta = records[0]
        assert meta["type"] == "meta"
        assert meta["schema_version"] == ms.SCHEMA_VERSION
        assert meta["language"] == "english"
        assert abs(meta["started_at"] - started) < 1.0

    def test_honors_audio_dir(self, tmp_path):
        cfg = FakeCfg()
        cfg.audio_dir = str(tmp_path / "custom")
        paths = ms.open_meeting(cfg, time.time())
        assert paths.meeting_dir.startswith(str(tmp_path / "custom"))

    def test_second_meeting_same_second_gets_unique_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        started = time.time()
        first = ms.open_meeting(FakeCfg(), started)
        second = ms.open_meeting(FakeCfg(), started)
        assert first.meeting_dir != second.meeting_dir
        assert ms.read_transcript(second.transcript_path)[0]["type"] == "meta"


class TestTranscriptWriter:
    def test_utterance_and_gap_records_round_trip(self, tmp_path, monkeypatch):
        paths = make_paths(tmp_path, monkeypatch)
        writer = ms.MeetingTranscriptWriter(paths, "english")
        writer.append_utterance(1, "hello team", 1000.0)
        writer.append_gap(2, 1001.0, 1004.0, ms.GAP_PAUSED)
        writer.append_utterance(3, "second utterance", 1005.0)
        records = ms.read_transcript(paths.transcript_path)
        assert [r["type"] for r in records] == ["meta", "utterance", "gap", "utterance"]
        assert records[1]["text"] == "hello team"
        assert records[1]["spoken_at"] == 1000.0
        assert "spoken_at_iso" in records[1]
        assert records[2]["reason"] == ms.GAP_PAUSED

    def test_every_append_is_flushed_to_disk_before_returning(self, tmp_path, monkeypatch):
        """Crash simulation: append, then read the file back with a FRESH
        handle (no in-memory state) — the record must already be on disk."""
        paths = make_paths(tmp_path, monkeypatch)
        writer = ms.MeetingTranscriptWriter(paths, "english")
        writer.append_utterance(1, "survives a crash", 1000.0)
        # Simulated crash: no writer.close(), no cleanup — just read from disk.
        with open(paths.transcript_path, encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        assert any(r.get("text") == "survives a crash" for r in lines)

    def test_concurrent_appends_do_not_interleave(self, tmp_path, monkeypatch):
        import threading

        paths = make_paths(tmp_path, monkeypatch)
        writer = ms.MeetingTranscriptWriter(paths, "english")

        def spam(tag):
            for i in range(50):
                writer.append_utterance(i, f"{tag}-{i}", 1000.0 + i)

        threads = [threading.Thread(target=spam, args=(t,)) for t in ("a", "b", "c")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        records = ms.read_transcript(paths.transcript_path)
        assert len(records) == 1 + 150  # meta + all appends, none interleaved/corrupt
        assert all(r["type"] in {"meta", "utterance"} for r in records)


class TestSummaryFiles:
    def test_writes_markdown_and_json(self, tmp_path, monkeypatch):
        paths = make_paths(tmp_path, monkeypatch)
        ms.write_summary_files(paths, "# Summary\n\nDid things.", {"summarizer": "local", "chunks": 2})
        assert os.path.exists(paths.summary_md_path)
        md = Path(paths.summary_md_path).read_text(encoding="utf-8")
        assert "Did things." in md
        payload = json.loads(Path(paths.summary_json_path).read_text(encoding="utf-8"))
        assert payload["type"] == "summary"
        assert payload["summarizer"] == "local"
        assert payload["meeting_id"] == paths.meeting_id
        assert "written_at" in payload


class TestReadTranscript:
    def test_skips_malformed_lines(self, tmp_path):
        transcript = tmp_path / "transcript.jsonl"
        with open(transcript, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"type": "meta"}) + "\n")
            handle.write("NOT JSON AT ALL\n")
            handle.write(json.dumps({"type": "utterance", "text": "ok"}) + "\n")
        records = ms.read_transcript(str(transcript))
        assert [r["type"] for r in records] == ["meta", "utterance"]

    def test_missing_file_is_empty(self, tmp_path):
        assert ms.read_transcript(str(tmp_path / "nope.jsonl")) == []


class TestMeetingDirName:
    def test_timestamp_format(self):
        ts = time.mktime(time.strptime("2026-09-17 10:15:00", "%Y-%m-%d %H:%M:%S"))
        assert ms.meeting_dir_name(ts) == "20260917-101500"


class TestPruneMeetings:
    def test_removes_expired_meeting_directories(self, tmp_path):
        root = tmp_path / "meetings"
        old = root / "20250101-000000"
        recent = root / "20260917-000000"
        for d in (old, recent):
            (d / "clips").mkdir(parents=True)
            (d / "transcript.jsonl").write_text("{}\n")
        old_time = time.time() - 48 * 3600
        os.utime(old, (old_time, old_time))
        removed = ms.prune_meetings(str(root), retention_hours=24.0)
        assert removed == 1
        assert not old.exists() and recent.exists()

    def test_zero_retention_keeps_everything(self, tmp_path):
        root = tmp_path / "meetings"
        (root / "20250101-000000").mkdir(parents=True)
        assert ms.prune_meetings(str(root), 0.0) == 0
        assert (root / "20250101-000000").exists()
