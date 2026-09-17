"""Off-thread Whisper transcription.

The audio callback used to run ``recognizer.recognize_whisper`` inline, holding
that thread for the full model-dependent transcription time (0.3-1 s for
``base``, many seconds for ``large``) and letting bursts of utterances spawn
parallel Whisper runs that contend for CPU and memory.

``TranscriptionWorker`` moves transcription onto a single consumer thread:
``submit()`` returns immediately (the recognizer loop regains control), runs
are serialized in arrival order, and a bounded queue drops stale audio when
speech arrives faster than it can be transcribed.

Deliberately dependency-light (stdlib + logging_config only) so unit tests can
exercise it without SpeechRecognition, PyAudio, or Whisper installed.
"""

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from assistant_app.utils.logging_config import get_logger


@dataclass
class TranscriptionJob:
    """One utterance awaiting transcription."""

    audio: Any  # speech_recognition.AudioData
    spoken_at: float  # time.time() when the utterance was captured
    submitted_at: float = 0.0  # time.time() when submit() was called


@dataclass
class TranscriptionStats:
    """Counters surfaced in the heartbeat log."""

    submitted: int = 0
    completed: int = 0
    failed: int = 0
    dropped: int = 0
    last_queue_wait: float = 0.0  # seconds the newest completed job waited in queue


class TranscriptionWorker:
    """Serialize blocking transcription on a dedicated thread.

    Args:
        recognize: ``callable(audio) -> text`` (blocking; runs Whisper).
        on_result: called with ``(text, job)`` after successful transcription.
        on_error: called with ``(exc, job)`` when recognize raises. UnknownValueError
            (no speech found) arrives here too — treat it as a normal no-op.
        on_drop: called with ``(job,)`` when an utterance is dropped from a full queue.
        max_queue: maximum pending jobs before the oldest pending job is dropped.
    """

    def __init__(
        self,
        recognize: Callable[[Any], str],
        on_result: Callable[[str, TranscriptionJob], None],
        on_error: Callable[[Exception, TranscriptionJob], None],
        on_drop: Callable[[TranscriptionJob], None] | None = None,
        max_queue: int = 2,
    ):
        self._recognize = recognize
        self._on_result = on_result
        self._on_error = on_error
        self._on_drop = on_drop
        self._queue: queue.Queue[TranscriptionJob] = queue.Queue(maxsize=max_queue)
        self._stats = TranscriptionStats()
        self._stats_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.logger = get_logger(__name__)

    def submit(self, audio: Any, spoken_at: float | None = None) -> bool:
        """Queue an utterance for transcription. Returns True unless the job was dropped."""
        now = time.time()
        job = TranscriptionJob(audio=audio, spoken_at=now if spoken_at is None else spoken_at, submitted_at=now)
        with self._stats_lock:
            self._stats.submitted += 1
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            return self._evict_and_retry(job)

    def _evict_and_retry(self, job: TranscriptionJob) -> bool:
        """Queue full: evict the oldest pending job (newest audio is most relevant)."""
        try:
            evicted = self._queue.get_nowait()
            self._notify_drop(evicted)
        except queue.Empty:
            # Worker drained the queue between the Full and now — retry fits.
            pass
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            # Defensive: another producer refilled the slot first.
            self._notify_drop(job)
            return False

    def start(self) -> "TranscriptionWorker":
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="Whisper-Transcription")
            self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the worker after the in-flight job finishes (Whisper cannot be interrupted mid-call)."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    @property
    def stats(self) -> TranscriptionStats:
        with self._stats_lock:
            return TranscriptionStats(**vars(self._stats))

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            queue_wait = time.time() - job.submitted_at
            with self._stats_lock:
                self._stats.last_queue_wait = queue_wait

            try:
                text = self._recognize(job.audio)
            except Exception as e:
                # A recognize failure must not kill the worker thread.
                with self._stats_lock:
                    self._stats.failed += 1
                self._safe_call(self._on_error, e, job, context=f"transcription failed: {e}")
                continue

            with self._stats_lock:
                self._stats.completed += 1
            self._safe_call(self._on_result, text, job, context="on_result handler failed")

    def _safe_call(self, handler: Callable, *args: Any, context: str) -> None:
        try:
            handler(*args)
        except Exception as handler_error:
            # Never let a handler bug kill the transcription thread.
            self.logger.error(f"❌ {context}: {handler_error}")

    def _notify_drop(self, job: TranscriptionJob) -> None:
        with self._stats_lock:
            self._stats.dropped += 1
        self.logger.warning("🔇 Dropping stale utterance (queue full)")
        if self._on_drop is not None:
            try:
                self._on_drop(job)
            except Exception as e:
                self.logger.error(f"❌ on_drop handler failed: {e}")
