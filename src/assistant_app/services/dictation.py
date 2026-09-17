"""Dictation mode: speech -> cleaned text -> typed at the cursor.

System-wide dictation for SpectraVoice (Wave 1). Reuses the existing stack —
local Whisper transcription runs through the same TranscriptionWorker the
assistant already uses, and insertion goes through the ToolExecutor
``type_text`` handler so dry-run, the audit log, and PyAutoGUI's FAILSAFE
safety-stop keep applying to dictation. No cloud transcription path exists
anywhere in this module: Whisper runs on-device and transcripts stay local.

Two activation modes:
- **push_to_talk**: hold the configured hotkey while speaking; the utterance
  clip that arrives after release is cleaned and typed.
- **vad**: continuous; consecutive recognizer clips inside
  ``vad_silence_timeout`` are joined into one utterance, a longer silence gap
  ends the utterance and triggers insertion. The VADStateMachine below is that
  silence/onset decision made explicit and unit-testable.

Per-app styles and snippets come from config (the single settings source).
Dictation audio is kept in memory for the session only unless
``dictation.persist_audio`` is enabled (then clips are written to
``dictation.audio_dir`` as WAV and never leave the machine).

Deliberately dependency-light (stdlib + logging_config only) so unit tests run
without SpeechRecognition, PyAudio, Whisper, or pynput installed.
"""

import os
import re
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field

from assistant_app.utils.logging_config import get_logger

# Cleanup styles: "standard" removes fillers and spoken punctuation; "minimal"
# only expands snippets (terminals want text verbatim — a dropped "um" in a
# command would change semantics, but spoken punctuation should still expand).
DICTATION_STYLES = ("standard", "minimal")
ACTIVATION_MODES = ("push_to_talk", "vad")

# Whisper hallucination artifacts from silence/noise (subset of the assistant's
# SmartVoiceDetector checks — dictation needs a light guard, not the LLM-query
# noise filter, which would reject short legitimate dictation).
_HALLUCINATION_PATTERN = re.compile(r"^[\[(].*[\])]$")


def is_likely_hallucination(text: str) -> bool:
    """True when a transcript is a Whisper-on-silence artifact.

    Bracketed placeholders like ``[BLANK_AUDIO]`` / ``(键盘音)`` — Whisper
    hallucinating a description of silence instead of words. Public so the
    meeting recorder (W3) applies exactly the same guard as dictation.
    """
    return bool(text) and bool(_HALLUCINATION_PATTERN.match(text.strip()))


def expand_snippets(text: str, snippets: dict[str, str]) -> str:
    """Replace spoken snippet phrases ("comma", "new line", ...) in a transcript.

    Matched case-insensitively on word boundaries so "comma" inside "command"
    is never rewritten.
    """
    cleaned = text
    for phrase, replacement in sorted(snippets.items(), key=lambda kv: -len(kv[0])):
        if not phrase:
            continue
        cleaned = re.sub(
            rf"\b{re.escape(phrase.casefold())}\b",
            replacement,
            cleaned,
            flags=re.IGNORECASE,
        )
    return cleaned


def strip_fillers(text: str, filler_words: list[str]) -> str:
    """Remove filler words ("um", "uh", ...) from a transcript, word-boundary safe."""
    cleaned = text
    for filler in filler_words:
        if not filler:
            continue
        cleaned = re.sub(rf"\b{re.escape(filler)}\b\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def normalize_spacing(text: str) -> str:
    """Collapse the whitespace left behind by cleanup; glue orphan punctuation.

    A space is re-inserted after , ; : ! ? when it lands between word
    characters. A period between word characters is left alone so tokens
    like Node.js and 3.5 survive Whisper's own punctuation.
    """
    text = re.sub(r"\s+([,.;:!?])", r"\1", re.sub(r"\s{2,}", " ", text)).strip()
    return re.sub(r"([,;:!?])(?=[A-Za-z0-9])", r"\1 ", text)


@dataclass
class TextCleaner:
    """Transcript cleanup pipeline: snippets -> filler removal -> spacing.

    ``style="minimal"`` keeps filler words (terminal-facing dictation must not
    alter verbatim text beyond snippet expansion); ``style="standard"`` is for
    prose targets.
    """

    snippets: dict[str, str] = field(default_factory=dict)
    filler_words: list[str] = field(default_factory=list)
    filler_removal: bool = True
    style: str = "standard"

    def clean(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        cleaned = expand_snippets(text.strip(), self.snippets)
        if self.filler_removal and self.style == "standard":
            cleaned = strip_fillers(cleaned, self.filler_words)
        return normalize_spacing(cleaned)


class VADState:
    """Explicit states for the dictation VAD (kept as plain strings for logs)."""

    IDLE = "idle"
    SPEAKING = "speaking"
    TRAILING = "trailing"


class VADStateMachine:
    """Utterance-boundary machine for continuous dictation.

    Feed it one event per recognizer clip or silence tick and it decides when
    an utterance is ready:

    - ``on_speech(now)`` — a speech burst arrived; transitions IDLE/TRAILING ->
      SPEAKING. Returns True only on the IDLE -> SPEAKING onset (a *new*
      utterance), so callers can flush or reset context accordingly.
    - ``on_silence(now)`` — no speech; SPEAKING -> TRAILING when the configured
      silence gap has passed, and TRAILING -> ready once the full timeout has
      elapsed. Returns True exactly when the utterance is complete.

    Trailing state exists so a short pause inside one utterance does not
    produce an insertion mid-sentence: readiness requires the full
    ``silence_timeout`` of quiet since the last speech event.
    """

    def __init__(self, silence_timeout: float = 1.2):
        if silence_timeout <= 0:
            raise ValueError("silence_timeout must be > 0")
        self.silence_timeout = silence_timeout
        self.state = VADState.IDLE
        self._last_speech_at: float | None = None

    @property
    def speaking(self) -> bool:
        return self.state in (VADState.SPEAKING, VADState.TRAILING)

    def on_speech(self, now: float) -> bool:
        """Speech onset or continuation. True only on IDLE -> SPEAKING onset."""
        was_idle = self.state == VADState.IDLE
        self.state = VADState.SPEAKING
        self._last_speech_at = now
        return was_idle

    def on_silence(self, now: float) -> bool:
        """Quiet-time advance. True when the utterance has ended (ready)."""
        if self.state == VADState.IDLE or self._last_speech_at is None:
            return False
        quiet_for = now - self._last_speech_at
        if quiet_for < self.silence_timeout:
            self.state = VADState.TRAILING
            return False
        self.state = VADState.IDLE
        self._last_speech_at = None
        return True

    def reset(self) -> None:
        """Force back to IDLE (used after an insertion or a cancellation)."""
        self.state = VADState.IDLE
        self._last_speech_at = None


class DictationController:
    """Owns the dictation session: arming, cleanup, and chunked insertion.

    The controller never transcribes audio itself — the assistant feeds it the
    transcript from the shared TranscriptionWorker. ``typer`` is the insertion
    function (the ToolExecutor type_text path in production); ``frontmost_app``
    resolves the focused application for per-app styles; both are injectable
    for tests. ``on_status`` receives lightweight status strings (W1 uses the
    console/log; the menu-bar HUD is Wave 2).

    Cancellation: ``cancel()`` (called by a new push-to-talk press during
    insertion, or session stop) prevents the remaining chunks from being typed.
    Text chunks already typed stay on screen — a clean cancel is defined as no
    duplication and a consistent controller state, not an undo of keystrokes.
    """

    def __init__(
        self,
        cfg,  # utils.config.DictationConfig
        typer: Callable[..., bool] | None = None,  # (text, press_enter) -> bool
        frontmost_app: Callable[[], str | None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_dictated: Callable[[str], None] | None = None,
        logger=None,
        stale_release_after: float = 3.0,
    ):
        self.cfg = cfg
        self._typer = typer or self._default_typer
        self._frontmost_app = frontmost_app or self._get_frontmost_app
        self._on_status = on_status or (lambda status: self.logger.info(f"📝 {status}"))
        self._on_dictated = on_dictated or (lambda text: None)
        self.logger = logger or get_logger(__name__)
        self._stale_release_after = stale_release_after

        self._style_overrides = {k.casefold(): v for k, v in (cfg.app_styles or {}).items()}
        self._lock = threading.Lock()
        self._active_cancel: threading.Event | None = None
        self._insert_thread: threading.Thread | None = None

        # Activation state
        self.activation = cfg.activation if cfg.activation in ACTIVATION_MODES else "push_to_talk"
        self._armed = False
        self._release_pending = False
        self._released_at: float | None = None
        self._inserting = False

        # VAD-mode utterance assembly: transcript fragments from consecutive
        # clips inside the silence window, joined on utterance end.
        self._vad = VADStateMachine(silence_timeout=cfg.vad_silence_timeout)
        self._pending_fragments: list[str] = []

    # === lifecycle ===

    @property
    def armed(self) -> bool:
        """True while the next utterance clip should be dictated (not LLM'd)."""
        with self._lock:
            return self._armed

    @property
    def inserting(self) -> bool:
        with self._lock:
            return self._inserting

    def start(self) -> None:
        """Begin the dictation session for the configured activation mode.

        VAD mode starts listening immediately; push-to-talk stays unarmed
        until the hotkey is held (that is what hold-to-speak means).
        """
        with self._lock:
            self._armed = self.activation == "vad"
            self._release_pending = False
            self._vad.reset()
        self._emit_status()
        if not self._armed:
            self.logger.info(
                "🎙️ Dictation push-to-talk: hold the hotkey to speak "
                "(activation mode toggle: hotkey or config)"
            )

    def stop(self) -> None:
        with self._lock:
            self._armed = False
            self._release_pending = False
            self._pending_fragments.clear()
        self.cancel()
        self.logger.info("🎙️ Dictation stopped")

    def cancel(self) -> None:
        """Abort the in-progress insertion (no duplication, state consistent).

        Per-insertion events: the event belongs to the running insert, so a
        later insertion starts fresh and cannot resurrect a finished cancel.
        """
        with self._lock:
            event = self._active_cancel
        if event is not None:
            event.set()

    def shutdown(self) -> None:
        self.stop()

    # === hotkey callbacks (wired by the assistant from the hotkey listener) ===

    def on_mode_toggle(self) -> None:
        """Cycle push_to_talk <-> vad (mode-toggle hotkey or CLI re-invocation)."""
        with self._lock:
            self.activation = "vad" if self.activation == "push_to_talk" else "push_to_talk"
            self._armed = self.activation == "vad"
            self._release_pending = False
            self._pending_fragments.clear()
            self._vad.reset()
        self.logger.info(f"🔁 Dictation activation mode → {self.activation}")
        self._emit_status()

    def on_ptt_press(self) -> None:
        """Push-to-talk press: cancel any running insertion, arm a fresh utterance.

        Pressing while text is still being typed cancels the rest of it (the
        keystrokes already sent remain — they cannot be un-typed) and restarts
        capture cleanly.
        """
        with self._lock:
            self._armed = True
            self._release_pending = False
        self.cancel()
        self.logger.info("🎙️ Push-to-talk engaged")

    def on_ptt_release(self) -> None:
        """Release: keep armed until the post-release clip arrives (or goes stale)."""
        with self._lock:
            if not self._armed:
                return
            self._release_pending = True
            self._released_at = time.time()
        self.logger.info("🎙️ Push-to-talk released — waiting for the utterance clip")

    # === transcript / audio entry points (called from the assistant loop) ===

    def handle_transcript(self, text: str) -> None:
        """Route one transcribed clip through dictation.

        Insertion is asynchronous (see _insert_async); success is reported via
        the ``on_dictated`` callback. PTT mode inserts the first post-release
        clip and disarms. VAD mode accumulates clip transcripts and inserts
        the joined utterance once the silence window closes.
        """
        with self._lock:
            if not self._armed:
                return
            now = time.time()
            stale = self._release_pending and now - self._released_at > self._stale_release_after
            if stale:
                # Release happened long ago and no real clip followed — disarm
                # instead of typing an unrelated later utterance.
                self._armed = False
                self._release_pending = False
                self.logger.info("🎙️ Push-to-talk window expired without speech")
                return

        transcript = text.strip()
        if not transcript or _HALLUCINATION_PATTERN.match(transcript):
            return

        if self.activation == "push_to_talk":
            with self._lock:
                self._armed = False
                self._release_pending = False
            # Insertion runs off the transcription thread so a long insert
            # never blocks the next utterance's Whisper run.
            self._insert_async(text)
            return

        # VAD mode: buffer the fragment; insertion waits for the silence gap.
        self._vad.on_speech(now)
        self._pending_fragments.append(text)
        return

    def poll_vad(self) -> str | None:
        """Advance the VAD clock; returns the typed text when an utterance ends.

        Called from the assistant's heartbeat loop (1 Hz) — cheap and bounded.
        """
        if self.activation != "vad" or not self._pending_fragments:
            return None
        if not self._vad.on_silence(time.time()):
            return None
        fragments = list(self._pending_fragments)
        self._pending_fragments.clear()
        self._vad.reset()
        self._insert_async(self._clean_for_frontmost_app(" ".join(fragments)))
        return None

    # === cleanup + insertion ===

    def insert(self, text: str) -> str | None:
        """Synchronously clean and type text (test entry point / direct use)."""
        return self._insert_cleaned(self._clean_for_frontmost_app(text))

    def _insert_async(self, text: str) -> None:
        """Clean + insert on a dedicated thread; see handle_transcript for why
        (also keeps the osascript frontmost-app probe off the audio threads)."""

        def run() -> None:
            self._insert_cleaned(self._clean_for_frontmost_app(text))

        self._insert_thread = threading.Thread(target=run, daemon=True, name="Dictation-Insert")
        self._insert_thread.start()

    def wait_for_insertion(self, timeout: float = 5.0) -> None:
        """Join the most recent insertion thread (tests / clean shutdown)."""
        thread = self._insert_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _clean_for_frontmost_app(self, text: str) -> str:
        style = self.cfg.style
        app = None
        try:
            app = self._frontmost_app()
        except Exception as e:  # a failed app probe must not kill dictation
            self.logger.debug(f"Frontmost-app lookup failed ({e}); using default style")
        if app:
            override = self._style_overrides.get(app.casefold())
            if override in DICTATION_STYLES:
                style = override
        cleaner = TextCleaner(
            snippets=dict(self.cfg.snippets or {}),
            filler_words=list(self.cfg.filler_words or []),
            filler_removal=self.cfg.filler_removal,
            style=style,
        )
        return cleaner.clean(text)

    def _insert_cleaned(self, text: str) -> str | None:
        """Type cleaned text in chunks, honoring cancellation between chunks."""
        if not text:
            return None
        cancel = threading.Event()
        with self._lock:
            self._active_cancel = cancel
            self._inserting = True
        self._emit_status("Dictating")
        try:
            chunks = self._chunk_for_insertion(text)
            for index, chunk in enumerate(chunks):
                if cancel.is_set():
                    self.logger.info("🛑 Dictation insertion cancelled")
                    return None
                press_enter = bool(self.cfg.insert_enter) and index == len(chunks) - 1
                if not self._safe_type(chunk, press_enter):
                    self.logger.error("❌ Dictation insertion failed (safety-stop or automation error)")
                    return None
            self._on_dictated(text)
            return text
        finally:
            with self._lock:
                self._inserting = False
            self._emit_status("Dictation idle")

    def _safe_type(self, chunk: str, press_enter: bool = False) -> bool:
        try:
            return bool(self._typer(chunk, press_enter))
        except Exception as e:  # automation layer may raise (FAILSAFE, permissions)
            self.logger.error(f"❌ Typing failed: {e}")
            return False

    def _chunk_for_insertion(self, text: str) -> list[str]:
        """Split long transcripts into sentence-sized chunks so a cancel lands
        within one chunk instead of after a full paragraph of keystrokes."""
        parts = re.split(r"(?<=[.!?])\s+", text)
        parts = [p for p in parts if p.strip()]
        if len(parts) <= 1:
            return [text]
        merged: list[str] = []
        for part in parts:
            if merged and len(merged[-1]) + len(part) + 1 <= 240:
                merged[-1] = f"{merged[-1]} {part}"
            else:
                merged.append(part)
        return merged

    @staticmethod
    def _default_typer(text: str, press_enter: bool = False) -> bool:
        """Insert via the ToolExecutor type_text path (PyAutoGUI/osascript with
        dry-run, audit logging, and the FAILSAFE safety-stop)."""
        from assistant_app.services.spectravoice_assistant import get_active_tool_executor

        executor = get_active_tool_executor()
        if executor is None:
            raise RuntimeError("Dictation has no ToolExecutor bound — cannot type")
        result = executor.execute("type_text", {"text": text, "press_enter": press_enter})
        return result.success

    @staticmethod
    def _get_frontmost_app() -> str | None:
        """Name of the focused macOS application (per-app style lookup)."""
        import subprocess

        try:
            result = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to get name of first application process whose frontmost is true',
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            app = result.stdout.strip()
            return app or None
        except Exception:
            return None

    def _emit_status(self, status: str | None = None) -> None:
        self._on_status(self.status_line() if status is None else status)

    def status_line(self) -> str:
        mode = "hold-to-speak" if self.activation == "push_to_talk" else "continuous (VAD)"
        state = "inserting" if self.inserting else ("listening" if self.armed else "idle")
        return f"Dictation: {mode} | {state}"

    # === audio retention (D2 privacy) ===

    def record_audio(self, audio, timestamp: float | None = None) -> None:
        """Retain the utterance clip only when explicitly configured.

        Default (persist_audio=False) this is a no-op: buffers die with the
        session. When persist_audio is true, clips are written as WAV files
        under audio_dir (local disk only — never uploaded, never sent to any
        cloud endpoint).
        """
        if not self.cfg.persist_audio:
            return
        wav_bytes = None
        if hasattr(audio, "get_wav_data"):
            wav_bytes = audio.get_wav_data()
        elif isinstance(audio, (bytes, bytearray)):
            wav_bytes = bytes(audio)
        else:
            return
        self._write_wav(wav_bytes, timestamp if timestamp is not None else time.time())

    def _write_wav(self, wav_bytes: bytes, timestamp: float) -> None:
        directory = self.cfg.audio_dir or os.path.join("logs", "dictation_audio")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"dictation_{int(timestamp * 1000)}.wav")
        try:
            with wave.open(path, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(wav_bytes)
            self.logger.info(f"💾 Dictation audio persisted to {path}")
        except OSError as e:
            self.logger.warning(f"⚠️ Could not persist dictation audio: {e}")
