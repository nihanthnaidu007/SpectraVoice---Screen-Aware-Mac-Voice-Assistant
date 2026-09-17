"""MeetingController tests (W3 D1.4): fidelity, crash-safety, consent, routing.

The core invariants, tested against a fake transcriber (no audio stack):

- **crash-mid-meeting flush survival** — every utterance Whisper returned is
  on disk even if the process dies before stop().
- **no silent loss** — overflow, transcription failure, and pauses become
  structured gap markers, never dropped records.
- **consent** — no kill-switch, no recording; per-clip re-check stops the
  recording when the switch flips mid-meeting.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app.core.consent import RecordingConsent
from assistant_app.services import meeting_store as ms
from assistant_app.services.meeting import MeetingController, run_startup_cleanup
from assistant_app.utils.config import MeetingConfig


class FakeClip:
    """Stands in for speech_recognition.AudioData (get_wav_data -> WAV bytes)."""

    def __init__(self, payload: bytes = b"RIFF-fake"):
        self._payload = payload

    def get_wav_data(self) -> bytes:
        return self._payload


def make_controller(tmp_path, monkeypatch, transcribe=None, consent=None, **kwargs) -> MeetingController:
    monkeypatch.chdir(tmp_path)  # meetings land in tmp_path/logs/meetings
    cfg = MeetingConfig(enabled=True, audio_dir="")
    consent = consent or RecordingConsent(feature_enabled=True)
    consent.arm()
    return MeetingController(
        cfg,
        transcribe=transcribe or (lambda audio: "hello world"),
        consent=consent,
        **kwargs,
    )


def read_records(meeting_dir: str) -> list[dict]:
    return ms.read_transcript(os.path.join(meeting_dir, "transcript.jsonl"))


def wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestConsentGating:
    def test_start_refused_while_kill_switch_engaged(self, tmp_path, monkeypatch):
        consent = RecordingConsent(feature_enabled=False)  # kill-switch ON (engaged)
        controller = make_controller(tmp_path, monkeypatch, consent=consent)
        assert controller.start() is False
        assert controller.recording is False
        assert controller.paths is None

    def test_start_refused_without_explicit_arm(self, tmp_path, monkeypatch):
        consent = RecordingConsent(feature_enabled=True)  # enabled but NOT armed
        monkeypatch.chdir(tmp_path)
        cfg = MeetingConfig(enabled=True)
        controller = MeetingController(cfg, transcribe=lambda a: "x", consent=consent)
        assert controller.start() is False
        assert controller.recording is False

    def test_start_succeeds_when_consent_granted(self, tmp_path, monkeypatch):
        controller = make_controller(tmp_path, monkeypatch)
        assert controller.start() is True
        assert controller.recording is True
        # The transcript exists and is parseable from the instant start returns.
        records = read_records(controller.paths.meeting_dir)
        assert records[0]["type"] == "meta"

    def test_kill_switch_flipped_mid_meeting_stops_clips(self, tmp_path, monkeypatch):
        consent = RecordingConsent(feature_enabled=True)
        controller = make_controller(tmp_path, monkeypatch, consent=consent)
        assert controller.start() is True
        consent.set_feature_enabled(False)  # the kill-switch flips
        assert controller.on_clip(FakeClip(), spoken_at=1000.0) is False
        records = read_records(controller.paths.meeting_dir)
        assert [r["type"] for r in records] == ["meta"]  # nothing recorded


class TestTranscriptionFidelity:
    def test_utterance_flushed_without_stop(self, tmp_path, monkeypatch):
        """Crash-mid-meeting: on_clip → Whisper → flush. No stop() — the
        record must already be on disk (supervisor could kill us now)."""
        controller = make_controller(tmp_path, monkeypatch)
        controller.start()
        controller.on_clip(FakeClip(), spoken_at=1000.0)
        assert wait_until(lambda: controller.utterance_count == 1)
        records = read_records(controller.paths.meeting_dir)
        assert records[-1]["type"] == "utterance"
        assert records[-1]["text"] == "hello world"
        assert records[-1]["spoken_at"] == 1000.0
        # Still no stop(): the crash-survival point of this test.

    def test_hallucination_artifact_not_recorded(self, tmp_path, monkeypatch):
        controller = make_controller(tmp_path, monkeypatch, transcribe=lambda a: "[BLANK_AUDIO]")
        controller.start()
        controller.on_clip(FakeClip(), spoken_at=1000.0)
        assert wait_until(lambda: controller.gap_count == 0 and controller.utterance_count == 0)
        time.sleep(0.1)
        records = read_records(controller.paths.meeting_dir)
        assert [r["type"] for r in records] == ["meta"]

    def test_transcription_failure_becomes_gap_marker(self, tmp_path, monkeypatch):
        def failing(audio):
            raise RuntimeError("whisper exploded")

        controller = make_controller(tmp_path, monkeypatch, transcribe=failing)
        controller.start()
        controller.on_clip(FakeClip(), spoken_at=1000.0)
        assert wait_until(lambda: controller.gap_count == 1)
        records = read_records(controller.paths.meeting_dir)
        gap = records[-1]
        assert gap["type"] == "gap"
        assert gap["reason"] == ms.GAP_TRANSCRIBE_FAILED

    def test_queue_overflow_becomes_gap_marker(self, tmp_path, monkeypatch):
        """Whisper slower than speech: the queue fills, the clip is not
        transcribed — but a gap marker with the reason lands in the transcript."""
        gate = threading.Event()

        def slow_transcribe(audio):
            gate.wait(timeout=5)
            return "late"

        controller = make_controller(tmp_path, monkeypatch, transcribe=slow_transcribe)
        cfg_depth_gate = controller._queue  # depth 256 default — shrink for the test
        cfg_depth_gate.maxsize = 1
        controller.start()
        # Worker is stuck in slow_transcribe after the first clip; fill the queue.
        controller.on_clip(FakeClip(), spoken_at=1000.0)
        assert wait_until(lambda: controller._queue.qsize() == 1)
        controller.on_clip(FakeClip(), spoken_at=1001.0)  # queued
        controller.on_clip(FakeClip(), spoken_at=1002.0)  # overflow
        assert wait_until(lambda: controller.overflow_count == 1)
        records = read_records(controller.paths.meeting_dir)
        assert any(r["type"] == "gap" and r["reason"] == ms.GAP_OVERFLOW for r in records)
        gate.set()

    def test_no_silent_loss_by_construction(self, tmp_path, monkeypatch):
        """Every clip must end as an utterance or a gap — count them."""
        def flaky(audio):
            raise RuntimeError("no")

        controller = make_controller(tmp_path, monkeypatch, transcribe=flaky)
        controller.start()
        for i in range(5):
            controller.on_clip(FakeClip(), spoken_at=1000.0 + i)
        assert wait_until(lambda: controller.gap_count == 5)
        records = read_records(controller.paths.meeting_dir)
        accounted = sum(1 for r in records if r["type"] in {"utterance", "gap"})
        assert accounted == 5

    def test_pending_clips_drained_as_gaps_at_stop(self, tmp_path, monkeypatch):
        gate = threading.Event()

        def slow_transcribe(audio):
            gate.wait(timeout=5)
            return "never in time"

        controller = make_controller(tmp_path, monkeypatch, transcribe=slow_transcribe, stop_timeout=0.3)
        controller.start()
        controller.on_clip(FakeClip(), spoken_at=1000.0)
        result = controller.stop()  # does not wait 5s — stop_timeout bounds it
        assert result is not None
        assert result["gaps"] >= 1
        records = read_records(result["meeting_dir"])
        assert any(r.get("reason") == ms.GAP_NOT_TRANSCRIBED for r in records)
        gate.set()


class TestPauseResume:
    def test_paused_clips_become_gap_markers(self, tmp_path, monkeypatch):
        controller = make_controller(tmp_path, monkeypatch)
        controller.start()
        controller.pause()
        assert controller.on_clip(FakeClip(), spoken_at=1000.0) is True
        records = read_records(controller.paths.meeting_dir)
        assert records[-1]["type"] == "gap" and records[-1]["reason"] == ms.GAP_PAUSED

    def test_resume_restores_transcription(self, tmp_path, monkeypatch):
        controller = make_controller(tmp_path, monkeypatch)
        controller.start()
        controller.pause()
        controller.on_clip(FakeClip(), spoken_at=1000.0)
        controller.resume()
        controller.on_clip(FakeClip(), spoken_at=1001.0)
        assert wait_until(lambda: controller.utterance_count == 1)
        records = read_records(controller.paths.meeting_dir)
        assert any(r["type"] == "utterance" for r in records)


class TestLifecycle:
    def test_stop_returns_session_result_and_ends_recording(self, tmp_path, monkeypatch):
        controller = make_controller(tmp_path, monkeypatch)
        controller.start()
        controller.on_clip(FakeClip(), spoken_at=1000.0)
        assert wait_until(lambda: controller.utterance_count == 1)
        result = controller.stop()
        assert result is not None
        assert result["utterances"] == 1
        assert result["summary_written"] is False  # no summarizer injected
        assert controller.recording is False
        assert controller.stop() is None  # second stop is a no-op

    def test_start_while_recording_refused(self, tmp_path, monkeypatch):
        controller = make_controller(tmp_path, monkeypatch)
        assert controller.start() is True
        assert controller.start() is False

    def test_status_updates_and_transcript_callback(self, tmp_path, monkeypatch):
        statuses: list[str] = []
        transcripts: list[str] = []
        controller = make_controller(
            tmp_path,
            monkeypatch,
            on_status=statuses.append,
            on_transcript=lambda text, at: transcripts.append(text),
        )
        controller.start()
        controller.on_clip(FakeClip(), spoken_at=1000.0)
        assert wait_until(lambda: len(transcripts) == 1)
        assert transcripts == ["hello world"]
        assert any("recording" in s for s in statuses)
        controller.stop()
        assert any("ended" in s for s in statuses)

    def test_status_line_shapes(self, tmp_path, monkeypatch):
        controller = make_controller(tmp_path, monkeypatch)
        assert controller.status_line() == "Meeting: idle"
        controller.start()
        assert controller.status_line().startswith("Meeting: recording")
        controller.pause()
        assert controller.status_line() == "Meeting: paused"


class TestSummaries:
    def test_summary_files_written_on_stop(self, tmp_path, monkeypatch):
        def summarize(utterances):
            return "# Meeting Summary\n\nDiscussed the roadmap.", {"chunks": 1}

        controller = make_controller(tmp_path, monkeypatch, summarize=summarize)
        controller.start()
        controller.on_clip(FakeClip(), spoken_at=1000.0)
        assert wait_until(lambda: controller.utterance_count == 1)
        result = controller.stop()
        assert result["summary_written"] is True
        assert os.path.exists(os.path.join(result["meeting_dir"], "summary.md"))
        payload = json.loads(Path(result["meeting_dir"], "summary.json").read_text(encoding="utf-8"))
        assert payload["summarizer_status"] == "success"

    def test_summary_failure_preserves_transcript_and_writes_note(self, tmp_path, monkeypatch):
        def broken_summarize(utterances):
            raise RuntimeError("ollama is down")

        controller = make_controller(tmp_path, monkeypatch, summarize=broken_summarize)
        controller.start()
        controller.on_clip(FakeClip(), spoken_at=1000.0)
        assert wait_until(lambda: controller.utterance_count == 1)
        result = controller.stop()
        # The transcript is intact...
        records = read_records(result["meeting_dir"])
        assert any(r["type"] == "utterance" for r in records)
        # ...and the failure is VISIBLE as a note, never silent.
        note = Path(result["meeting_dir"], "summary.md").read_text(encoding="utf-8")
        assert "failed" in note
        payload = json.loads(Path(result["meeting_dir"], "summary.json").read_text(encoding="utf-8"))
        assert payload["summarizer_status"] == "failed"


class TestStartupCleanup:
    def test_run_startup_cleanup_applies_retention(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = MeetingConfig(enabled=True, retention_hours=24.0)
        stale = tmp_path / "logs" / "meetings" / "20250101-000000"
        stale.mkdir(parents=True)
        old = time.time() - 48 * 3600
        os.utime(stale, (old, old))
        assert run_startup_cleanup(cfg) == 1
        assert not stale.exists()
