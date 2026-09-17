"""Meeting mode: consent-gated, crash-safe meeting transcription (W3 D1).

Mirror of the DictationController seam, with different fidelity guarantees.
Dictation optimizes latency (queue depth 2, evict-oldest on overflow — losing
an "um" is fine). A meeting must never silently lose an utterance, so:

- transcription runs on a **non-dropping worker** with a deep queue
  (``meeting.queue_depth``). If the queue still fills (Whisper slower than
  speech for minutes on end), the clip becomes a **gap marker** in the
  transcript — audio existed, was not transcribed, reason attached. Silent
  loss is impossible by construction.
- **every** transcript record is flushed to disk before the append returns
  (spec R6): a supervisor restart loses at most the in-flight clip, never a
  session. The transcript file exists from the instant start() succeeds.
- meeting clips are consumed here and **never** reach the screen-aware
  assistant pipeline (spec R8) — the assistant's audio callback branches to
  ``on_clip`` before dictation handling and before SmartVoiceDetector.

Consent posture (D2): recording requires the RecordingConsent gate to allow
it — config kill-switch ON **and** an explicit per-meeting start. pause()/
resume() mark the pause with gap markers rather than discarding audio, so a
resumed transcript can show what was skipped. TTS muting is the assistant's
job (it owns TTS); this module only reports state.

Summaries (D3) are injected: ``summarize(utterances) -> (markdown, metadata)``.
The production wiring builds the local-first map-reduce summarizer on the
LLMProvider seam; this module never imports an LLM provider at module scope.
Summary output is files only — nothing here ever speaks.

Deliberately dependency-light (stdlib + logging_config + the dictation
hallucination helper) so unit tests run without SpeechRecognition, PyAudio,
or Whisper installed.
"""

import os
import queue
import threading
import time
import wave
from collections.abc import Callable

from assistant_app.services import meeting_store
from assistant_app.services.dictation import is_likely_hallucination
from assistant_app.utils.logging_config import get_logger

# Transcription retry policy: one retry, then a gap marker. Whisper failures
# are rare and usually transient (model contention); more retries would back
# the fidelity queue up behind a broken clip.
_TRANSCRIBE_ATTEMPTS = 2


class MeetingController:
    """Owns one meeting recording session: capture queue, transcript, summaries.

    Args:
        cfg: ``utils.config.MeetingConfig`` (language, queue_depth, storage…).
        transcribe: ``callable(audio) -> str`` — blocking Whisper call with the
            meeting language applied. Injected by the assistant wiring (it owns
            the recognizer); tests inject a fake.
        consent: the shared ``RecordingConsent`` gate. Recording runs only
            while ``consent.recording_allowed``; every clip re-checks it.
        on_status: status strings for the HUD/log ("Recording meeting", …).
        on_transcript: called with ``(text, spoken_at)`` per transcribed
            utterance (the HUD shows a live tail; the assistant may count).
        summarize: ``callable(list[dict]) -> (markdown, metadata)`` — map-reduce
            summarizer invoked at stop; None skips summarization (files note it).
        stop_timeout: seconds to wait for the worker's final clip at stop().
    """

    def __init__(
        self,
        cfg,
        transcribe: Callable[[object], str],
        consent,
        on_status: Callable[[str], None] | None = None,
        on_transcript: Callable[[str, float], None] | None = None,
        summarize: Callable[[list[dict]], tuple[str, dict]] | None = None,
        logger=None,
        stop_timeout: float = 10.0,
    ):
        self.cfg = cfg
        self._transcribe = transcribe
        self._consent = consent
        self._on_status = on_status or (lambda status: self.logger.info(f"🏛️ {status}"))
        self._on_transcript = on_transcript or (lambda text, spoken_at: None)
        self._summarize = summarize
        self.logger = logger or get_logger(__name__)
        self._stop_timeout = stop_timeout

        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=cfg.queue_depth)
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._writer: meeting_store.MeetingTranscriptWriter | None = None
        self._paths: meeting_store.MeetingPaths | None = None
        self._recording = False
        self._paused = False
        self._started_at: float | None = None
        self._seq = 0  # transcript-wide record counter (utterances + gaps share it)
        self._utterance_count = 0
        self._gap_count = 0
        self._overflow_count = 0

    # === lifecycle ===

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._recording

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def utterance_count(self) -> int:
        with self._lock:
            return self._utterance_count

    @property
    def gap_count(self) -> int:
        with self._lock:
            return self._gap_count

    @property
    def overflow_count(self) -> int:
        """Clips that hit a full queue — always 0 unless Whisper fell behind."""
        with self._lock:
            return self._overflow_count

    @property
    def paths(self) -> meeting_store.MeetingPaths | None:
        """Artifact paths for the live/last meeting (None before the first start)."""
        with self._lock:
            return self._paths

    def start(self) -> bool:
        """Begin recording. Refuses without recording consent (kill-switch or
        missing explicit start) and while already recording."""
        with self._lock:
            if self._recording:
                self.logger.warning("⚠️ Meeting start ignored — already recording")
                return False
        if not self._consent.recording_allowed:
            self.logger.warning(
                "🚫 Meeting start refused — recording requires the config kill-switch "
                "(meeting.enabled: true) AND an explicit per-meeting start"
            )
            self._emit_status("Meeting refused (consent)")
            return False

        started_at = time.time()
        paths = meeting_store.open_meeting(self.cfg, started_at)
        writer = meeting_store.MeetingTranscriptWriter(paths, self.cfg.language)
        with self._lock:
            self._paths = paths
            self._writer = writer
            self._recording = True
            self._paused = False
            self._started_at = started_at
            self._seq = 0
            self._utterance_count = 0
            self._gap_count = 0
            self._overflow_count = 0
            self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run_worker, daemon=True, name="Meeting-Transcribe")
        self._worker.start()
        self.logger.info(f"🏛️ Meeting recording started → {paths.meeting_dir}")
        self._emit_status()
        return True

    def stop(self) -> dict | None:
        """End the recording, drain, and summarize. Returns the session result
        (or None when not recording). The transcript is complete the moment any
        pending work is written; summary failures never touch it."""
        with self._lock:
            if not self._recording:
                return None
            paths = self._paths
            self._recording = False
            self._paused = False
            self._stop_event.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=self._stop_timeout)
        # A clip still mid-transcription after the join timeout is accounted
        # for here: the worker suppresses its own outcome once stop() began
        # (stop_event), so exactly one gap marker covers it — never silence.
        with self._lock:
            in_flight_spoken_at = self._in_flight_spoken_at
            self._in_flight_spoken_at = None
        if in_flight_spoken_at is not None:
            self._record_gap(in_flight_spoken_at, time.time(), meeting_store.GAP_NOT_TRANSCRIBED)
        # Anything still queued after the worker finished: recorded as a gap,
        # never discarded silently.
        self._drain_pending(reason=meeting_store.GAP_NOT_TRANSCRIBED)

        assert paths is not None
        result = {
            "meeting_dir": paths.meeting_dir,
            "meeting_id": paths.meeting_id,
            "transcript_path": paths.transcript_path,
            "utterances": self.utterance_count,
            "gaps": self.gap_count,
            "overflows": self.overflow_count,
            "duration_seconds": round(time.time() - (self._started_at or time.time()), 3),
            "summary_written": False,
        }
        if self._summarize is not None:
            self._write_summary(paths, result)
        self.logger.info(
            f"🏛️ Meeting recording stopped — {result['utterances']} utterance(s), "
            f"{result['gaps']} gap(s) → {paths.meeting_dir}"
        )
        self._emit_status("Meeting ended")
        return result

    def shutdown(self) -> None:
        """Stop any active recording (assistant shutdown path)."""
        self.stop()

    def pause(self) -> None:
        """Pause transcription; arriving clips become 'paused' gap markers so
        the transcript shows exactly what was skipped."""
        with self._lock:
            if not self._recording or self._paused:
                return
            self._paused = True
        self.logger.info("⏸️ Meeting paused (speech is marked as a gap, not recorded into the transcript)")
        self._emit_status()

    def resume(self) -> None:
        with self._lock:
            if not self._recording or not self._paused:
                return
            self._paused = False
        self.logger.info("▶️ Meeting resumed")
        self._emit_status()

    # === audio entry point (assistant audio callback, R8 routing) ===

    def on_clip(self, audio, spoken_at: float | None = None) -> bool:
        """Consume one utterance clip. Returns True when queued for transcription.

        Never raises and never blocks the audio thread beyond a bounded put:
        overflow becomes a gap marker. Clips arriving while the meeting is
        paused are marked as gaps (the transcript must show they existed).
        """
        spoken_at = time.time() if spoken_at is None else spoken_at
        if not self.recording:
            return False  # race with stop — the assistant re-checks routing itself
        if not self._consent.recording_allowed:
            # Kill-switch flipped (or arming revoked) mid-meeting: stop consuming.
            self.logger.warning("🚫 Meeting clip ignored — recording consent is no longer granted")
            return False
        if self.paused:
            self._record_gap(spoken_at, spoken_at, meeting_store.GAP_PAUSED)
            return True
        self._persist_clip(audio, spoken_at)
        try:
            self._queue.put_nowait((audio, spoken_at))
            return True
        except queue.Full:
            with self._lock:
                self._overflow_count += 1
            self.logger.warning(
                "⚠️ Meeting transcription queue full — writing a gap marker "
                "(the utterance was captured but not transcribed)"
            )
            self._record_gap(spoken_at, spoken_at, meeting_store.GAP_OVERFLOW)
            return False

    # === worker ===

    def _run_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                audio, spoken_at = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._lock:
                self._in_flight_spoken_at = spoken_at
            self._transcribe_and_record(audio, spoken_at)
            # Clear only on the normal path: when stop() interrupted us, the
            # marker must survive for stop()'s gap accounting.
            if not self._stop_event.is_set():
                with self._lock:
                    self._in_flight_spoken_at = None

    def _transcribe_and_record(self, audio, spoken_at: float) -> None:
        """Blocking transcription + incremental flush. Failure after the retry
        becomes a gap marker — the clip is accounted for either way."""
        if self._stop_event.is_set():
            return  # stop() owns this clip's accounting (a not_transcribed gap)
        text: str | None = None
        last_error: Exception | None = None
        for attempt in range(_TRANSCRIBE_ATTEMPTS):
            try:
                text = self._transcribe(audio)
                break
            except Exception as exc:  # Whisper may raise anything; retry, then persist a gap
                last_error = exc
                self.logger.warning(f"⚠️ Meeting transcription attempt {attempt + 1} failed: {exc}")
        if text is None:
            self.logger.error(f"❌ Meeting transcription failed after {_TRANSCRIBE_ATTEMPTS} attempts: {last_error}")
            self._record_gap(spoken_at, time.time(), meeting_store.GAP_TRANSCRIBE_FAILED)
            return
        cleaned = text.strip()
        if not cleaned or is_likely_hallucination(cleaned):
            return  # Whisper-on-silence artifact — not content, nothing lost
        self._append_utterance(cleaned, spoken_at)

    def _append_utterance(self, text: str, spoken_at: float) -> None:
        writer = self._writer
        with self._lock:
            seq = self._seq
            self._seq += 1
            self._utterance_count += 1
        if writer is not None:
            writer.append_utterance(seq, text, spoken_at)  # flushed on return (R6)
        self._on_transcript(text, spoken_at)

    def _record_gap(self, started_at: float, ended_at: float, reason: str) -> None:
        writer = self._writer
        with self._lock:
            seq = self._seq
            self._seq += 1
            self._gap_count += 1
        if writer is not None:
            writer.append_gap(seq, started_at, ended_at, reason)

    def _drain_pending(self, reason: str) -> None:
        while True:
            try:
                _audio, spoken_at = self._queue.get_nowait()
            except queue.Empty:
                return
            self._record_gap(spoken_at, time.time(), reason)

    def _persist_clip(self, audio, spoken_at: float) -> None:
        """Write the utterance clip when meeting.persist_audio is set (local only)."""
        if not self.cfg.persist_audio:
            return
        wav_bytes = None
        if hasattr(audio, "get_wav_data"):
            wav_bytes = audio.get_wav_data()
        elif isinstance(audio, (bytes, bytearray)):
            wav_bytes = bytes(audio)
        else:
            return
        paths = self.paths
        if paths is None:
            return
        try:
            os.makedirs(paths.clips_dir, exist_ok=True)
            path = os.path.join(paths.clips_dir, f"clip_{int(spoken_at * 1000)}.wav")
            with wave.open(path, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(wav_bytes)
        except OSError as exc:
            self.logger.warning(f"⚠️ Could not persist meeting clip: {exc}")

    # === summaries (D3 — files, never spoken) ===

    def _write_summary(self, paths: meeting_store.MeetingPaths, result: dict) -> None:
        assert self._summarize is not None
        records = meeting_store.read_transcript(paths.transcript_path)
        spoken = meeting_store.utterances(records)
        try:
            summary_text, metadata = self._summarize(spoken)
            status = "success"
        except Exception as exc:  # summarizer is injectable, may raise anything
            self.logger.error(f"❌ Meeting summary failed: {exc}")
            summary_text = (
                "# Meeting Summary\n\n"
                f"⚠️ Summary generation failed: {exc}\n\n"
                "The full transcript is intact alongside this note — regenerate the "
                "summary from transcript.jsonl when the summarizer is available."
            )
            metadata = {"summarizer_status": "failed", "error": str(exc)}
            status = "failed"
        metadata = {"summarizer_status": status, **metadata}
        try:
            meeting_store.write_summary_files(paths, summary_text, metadata)
            result["summary_written"] = status == "success"
        except OSError as exc:
            self.logger.error(f"❌ Could not write meeting summary files: {exc}")

    # === status ===

    def status_line(self) -> str:
        if not self.recording:
            return "Meeting: idle"
        if self.paused:
            return "Meeting: paused"
        elapsed = max(0, int(time.time() - (self._started_at or time.time())))
        return (
            f"Meeting: recording | {elapsed // 60:02d}:{elapsed % 60:02d} | "
            f"{self.utterance_count} utterances"
            + (f" | {self.gap_count} gaps" if self.gap_count else "")
        )

    def _emit_status(self, status: str | None = None) -> None:
        self._on_status(self.status_line() if status is None else status)


def run_startup_cleanup(cfg) -> int:
    """Delete meeting artifacts past their retention TTL (D4, called at launch)."""
    return meeting_store.prune_meetings(meeting_store.meeting_root(cfg), cfg.retention_hours)
