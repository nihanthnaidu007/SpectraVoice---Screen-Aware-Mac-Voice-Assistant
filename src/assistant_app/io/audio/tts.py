"""Text-to-Speech Service - OpenAI TTS streamed to PyAudio playback.

Synthesis streams: playback starts as soon as the first pre-buffered PCM
chunks arrive over the network instead of waiting for the full response
(previously the entire utterance — hard-capped at 500 chars — was synthesized
before a single sample played).

Supports barge-in interruption via the stop() method.
Runs playback in a background thread to allow concurrent listening.

Thread Safety:
- _state_lock: Protects _state and _should_stop (single source of truth)
- _playback_lock: Protects _current_player for safe interruption
- _lock: Ensures only one speak operation at a time

The stop() method can be safely called from any thread at any time.
PyAudio is imported lazily so this module stays importable (and unit-testable)
on machines without PortAudio installed.
"""

import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

import openai

from assistant_app.utils.logging_config import get_logger

try:
    from pyaudio import paInt16  # 16-bit signed int samples
except ImportError:  # PortAudio not installed (CI, unit tests) — only needed at playback time
    # Part of PortAudio's stable ABI (portaudio.h), not a pyaudio implementation detail.
    paInt16 = 8


class TTSState(Enum):
    """TTS playback state for orchestrator tracking."""
    IDLE = "idle"
    SYNTHESIZING = "synthesizing"  # Waiting for API response
    PLAYING = "playing"  # Audio playback in progress
    STOPPING = "stopping"  # Stop requested, winding down


class TextToSpeech:
    """
    Thread-safe Text-to-Speech service with barge-in support.
    
    Thread Safety Guarantees:
    - stop() can be called from any thread at any time
    - State transitions are atomic and protected by _state_lock
    - Only one speak operation runs at a time (_lock)
    
    Barge-in Latency:
    - Interrupt response time: ~170ms (CHUNK_SIZE/SAMPLE_RATE)
    - State check frequency: Every chunk write
    - Uses threading.Event for fast cross-thread signaling
    
    Utterance Tracking:
    - Each speak() call gets a unique utterance_id
    - Detailed logging: TTS start (id, length) and TTS end (id, reason)
    """
    
    SAMPLE_RATE = 24000
    CHANNELS = 1
    # Larger chunks = smoother playback with minimal buffer underruns
    CHUNK_SIZE = 4096  # ~170ms chunks - smooth playback while still responsive
    # Pre-buffer before starting playback to prevent initial stuttering
    PRE_BUFFER_CHUNKS = 3  # Buffer ~500ms of audio before starting playback
    
    def __init__(self, voice: str = "shimmer", hd_quality: bool = True, output_device: int | None = None,
                 speech_rate: float = 1.0):
        self.voice = voice
        self.hd_quality = hd_quality  # Use tts-1-hd for smoother, higher quality voice
        self.output_device = output_device  # PyAudio output device index; None = system default
        self.speech_rate = speech_rate  # Playback speed multiplier (OpenAI TTS 0.25-4.0)
        self._output_device_failed = False  # Sticky: configured device failed once, use default
        self.logger = get_logger(__name__)
        
        # === THREAD-SAFE STATE ===
        # Single source of truth for TTS state
        self._state_lock = threading.Lock()  # Protects _state
        self._state = TTSState.IDLE
        
        # Fast interrupt signaling using Event (faster than polling a flag)
        self._stop_event = threading.Event()  # Set to signal stop, clear to allow playback
        
        # Separate locks for different concerns
        self._lock = threading.Lock()  # Ensures single speak operation
        self._playback_lock = threading.Lock()  # Protects _current_player
        
        self._audio_context = None
        self._current_player = None  # Track current player for interruption
        self._playback_thread: threading.Thread | None = None  # Background playback thread
        self._on_interrupted: Callable[[], None] | None = None  # Callback when interrupted
        self._on_complete: Callable[[bool], None] | None = None  # Callback when done
        
        # === UTTERANCE TRACKING ===
        self._utterance_counter = 0
        self._current_utterance_id: int | None = None
        
        # === BARGE-IN METRICS ===
        self._barge_in_count = 0
        self._last_barge_in_time: float = 0.0
        
    def _get_audio_context(self):
        if self._audio_context is None:
            # Lazy import: keeps this module importable without PortAudio (CI, unit tests).
            from pyaudio import PyAudio

            self._audio_context = PyAudio()
        return self._audio_context

    def _open_player(self):
        """Open a 16-bit PCM output stream, honoring the configured output device.

        Falls back to the system default device (once, then sticky) when the
        configured output device index is missing or fails — e.g. a Bluetooth
        headset that disconnected mid-session.
        """
        def default_stream():
            return self._get_audio_context().open(
                format=paInt16,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                output=True,
                frames_per_buffer=8192,
            )

        if self.output_device is None or self._output_device_failed:
            return default_stream()

        try:
            device_info = self._get_audio_context().get_device_info_by_index(self.output_device)
            self.logger.info(
                f"🔊 Using configured output device [{self.output_device}]: {device_info.get('name', '?')}"
            )
        except OSError as e:
            self.logger.warning(f"⚠️ Configured output device {self.output_device} not found ({e}); using system default")
            self._output_device_failed = True
            return default_stream()

        try:
            return self._get_audio_context().open(
                format=paInt16,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                output=True,
                output_device_index=self.output_device,
                frames_per_buffer=8192,
            )
        except OSError as e:
            self.logger.warning(f"⚠️ Output device {self.output_device} failed ({e}); falling back to system default")
            self._output_device_failed = True
            return default_stream()

        
    def speak(self, text: str) -> bool:
        """
        Speak the given text synchronously (blocking). Can be interrupted by calling stop().
        
        Args:
            text: Text to speak
            
        Returns:
            True if speech completed normally, False if interrupted
        """
        if not text or not text.strip():
            return True
        
        # Clear stop event before starting
        self._stop_event.clear()
        
        with self._lock:
            return self._speak_impl(text)
    
    def speak_async(self, text: str) -> None:
        """
        Speak the given text asynchronously (non-blocking).
        Returns immediately while audio plays in background.
        Use stop() to interrupt, and set_on_complete() for completion callback.
        
        This is the preferred method for barge-in support as it doesn't block
        the calling thread, allowing continued voice detection.
        
        Args:
            text: Text to speak
        """
        if not text or not text.strip():
            if self._on_complete:
                self._on_complete(False)
            return
        
        # Stop any existing playback first (fast path)
        if self.is_playing:
            self.stop()
            # Wait briefly for previous playback to stop (reduced from 0.5s)
            self._wait_for_stop(timeout=0.2)
        
        # Clear stop event before starting new speech
        self._stop_event.clear()
        
        # Start playback in background thread
        self._playback_thread = threading.Thread(
            target=self._async_speak_worker,
            args=(text,),
            daemon=True,
            name="TTS-Playback"
        )
        self._playback_thread.start()
    
    def _async_speak_worker(self, text: str) -> None:
        """Background worker thread for async speech."""
        with self._lock:
            was_interrupted = not self._speak_impl(text)
        
        # Call completion callback
        if self._on_complete:
            try:
                self._on_complete(was_interrupted)
            except Exception as e:
                self.logger.debug(f"Error in on_complete callback: {e}")
    
    def _wait_for_stop(self, timeout: float = 1.0) -> bool:
        """Wait for current playback to stop. Returns True if stopped."""
        start = time.time()
        while self.is_playing and (time.time() - start) < timeout:
            time.sleep(0.01)
        return not self.is_playing
    
    def stop(self) -> None:
        """
        Immediately stop any ongoing speech playback (barge-in support).
        
        Thread Safety: Can be safely called from any thread at any time.
        This is the single path to stop TTS and flip the state.
        
        Latency: ~21ms maximum (one chunk duration)
        """
        # Fast check without lock first
        if self._state == TTSState.IDLE:
            return
        
        # Signal stop immediately via Event (fastest cross-thread notification)
        self._stop_event.set()
        
        # Update state atomically
        with self._state_lock:
            if self._state == TTSState.IDLE:
                return
            self._barge_in_count += 1
            self._last_barge_in_time = time.time()
            self._state = TTSState.STOPPING
        
        self.logger.info(f"🛑 User speech detected during TTS, stopping TTS (barge-in #{self._barge_in_count})...")
        
        # Force-stop the current player immediately
        with self._playback_lock:
            if self._current_player:
                try:
                    self._current_player.stop_stream()
                except Exception:
                    pass
    
    def set_on_interrupted(self, callback: Callable[[], None] | None) -> None:
        """Set callback to be called when speech is interrupted."""
        self._on_interrupted = callback
    
    def set_on_complete(self, callback: Callable[[bool], None] | None) -> None:
        """
        Set callback to be called when speech completes (or is interrupted).
        
        Args:
            callback: Function that takes a bool (True if was_interrupted)
        """
        self._on_complete = callback
    
    def _speak_impl(self, text: str) -> bool:
        """
        Internal speech implementation with interrupt support.
        
        Returns:
            True if completed normally, False if interrupted
        """
        player = None
        was_interrupted = False
        end_reason = "completed"
        
        # Generate unique utterance ID for tracking
        self._utterance_counter += 1
        utterance_id = self._utterance_counter
        self._current_utterance_id = utterance_id
        text_length = len(text)
        
        try:
            self.logger.info(f"🔊 TTS start (id={utterance_id}, length={text_length} chars)")
            
            # === THREAD-SAFE STATE TRANSITION ===
            with self._state_lock:
                self._state = TTSState.SYNTHESIZING
            
            # Check for early interrupt (before API call) - use Event for speed
            if self._stop_event.is_set():
                end_reason = "cancelled_before_start"
                self.logger.info(f"⏹️ TTS end (id={utterance_id}, reason={end_reason})")
                return False
            
            # Full text — the 500-char cap is gone; synthesis streams, so long
            # responses start playing while later segments are still generated.
            tts_model = "tts-1-hd" if self.hd_quality else "tts-1"
            
            was_interrupted, bytes_received, bytes_played, player = self._stream_to_player(text, tts_model, utterance_id)
            
            if not was_interrupted:
                if bytes_received == 0:
                    end_reason = "empty_audio"
                    self.logger.warning(f"⚠️ TTS end (id={utterance_id}, reason={end_reason})")
                    return True
                audio_duration_ms = (bytes_received / 2) / self.SAMPLE_RATE * 1000  # 16-bit = 2 bytes/sample
                self.logger.info(
                    f"✅ TTS end (id={utterance_id}, reason=completed, received={bytes_received} bytes, "
                    f"played={bytes_played}, ~{audio_duration_ms:.0f}ms)"
                )
            
            return not was_interrupted

        except openai.APIError as e:
            end_reason = f"api_error: {e}"
            self.logger.error(f"❌ TTS end (id={utterance_id}, reason={end_reason})")
            return False
        except OSError as e:
            if self._stop_event.is_set():
                # Expected during interrupt
                end_reason = "barge_in_oserror"
                self.logger.info(f"⏹️ TTS end (id={utterance_id}, reason={end_reason})")
                was_interrupted = True
                return False
            if "PortAudio" in str(e) or "-9986" in str(e) or "-50" in str(e):
                end_reason = f"audio_device_error: {e}"
                self.logger.warning(f"⚠️ TTS end (id={utterance_id}, reason={end_reason})")
                self._reinit_audio()
            else:
                end_reason = f"os_error: {e}"
                self.logger.error(f"❌ TTS end (id={utterance_id}, reason={end_reason})")
            return False
        except Exception as e:
            end_reason = f"exception: {type(e).__name__}: {e}"
            self.logger.error(f"❌ TTS end (id={utterance_id}, reason={end_reason})")
            return False
        finally:
            # === THREAD-SAFE STATE CLEANUP ===
            with self._state_lock:
                self._state = TTSState.IDLE
            # Clear stop event for next speak call
            self._stop_event.clear()
            self._current_utterance_id = None
            
            # Clear player reference
            with self._playback_lock:
                self._current_player = None
            
            if player:
                try:
                    player.stop_stream()
                    player.close()
                except Exception:
                    pass
            
            # Call interrupted callback if applicable
            if was_interrupted and self._on_interrupted:
                try:
                    self._on_interrupted()
                except Exception:
                    pass
    
    def _stream_to_player(self, text: str, tts_model: str, utterance_id: int) -> tuple[bool, int, int, Any]:
        """Stream synthesis and play PCM chunks as they arrive.

        Opens the player once ~500ms is pre-buffered (or at end-of-stream for
        short responses), so speech starts before synthesis completes.

        Returns:
            (was_interrupted, bytes_received, bytes_played, player) — player is
            returned so _speak_impl's finally block can close it.
        """
        player = None
        pre_buffer: list[bytes] = []
        bytes_received = 0
        bytes_played = 0
        was_interrupted = False

        def play_chunk(dst, chunk: bytes) -> bool:
            nonlocal was_interrupted
            try:
                dst.write(chunk)
                return True
            except OSError as e:
                was_interrupted = True
                if self._stop_event.is_set():
                    self.logger.info(f"⏹️ TTS end (id={utterance_id}, reason=barge_in_oserror)")
                else:
                    self.logger.warning(f"⚠️ TTS end (id={utterance_id}, reason=playback_error: {e})")
                return False

        def start_playback() -> None:
            nonlocal player, pre_buffer, bytes_played
            player = self._open_player()
            with self._playback_lock:
                self._current_player = player
            with self._state_lock:
                self._state = TTSState.PLAYING
            self.logger.debug(f"🔊 TTS playback started (id={utterance_id})")
            for buffered in pre_buffer:
                if not play_chunk(player, buffered):
                    break
                bytes_played += len(buffered)
            pre_buffer = []

        stream_ctx = openai.audio.speech.with_streaming_response.create(
            model=tts_model,
            voice=self.voice,
            response_format="pcm",
            input=text,
            speed=self.speech_rate,
        )

        with stream_ctx as response:
            if self._stop_event.is_set():
                was_interrupted = True
                self.logger.info(f"⏹️ TTS end (id={utterance_id}, reason=cancelled_after_synthesis)")
                return was_interrupted, bytes_received, bytes_played, player

            for chunk in response.iter_bytes(chunk_size=self.CHUNK_SIZE):
                if not chunk:
                    continue
                bytes_received += len(chunk)
                if self._stop_event.is_set():
                    was_interrupted = True
                    self.logger.info(f"⏹️ TTS end (id={utterance_id}, reason=barge_in)")
                    break
                if player is None:
                    pre_buffer.append(chunk)
                    if len(pre_buffer) >= self.PRE_BUFFER_CHUNKS:
                        start_playback()
                        if was_interrupted:
                            break
                else:
                    if not play_chunk(player, chunk):
                        break
                    bytes_played += len(chunk)

            # Short responses: fewer chunks arrived than the pre-buffer target.
            if player is None and pre_buffer and not was_interrupted:
                start_playback()

        return was_interrupted, bytes_received, bytes_played, player

    def _reinit_audio(self) -> None:
        try:
            if self._audio_context:
                self._audio_context.terminate()
        except Exception:
            pass
        self._audio_context = None
        
        time.sleep(0.2)
        
        try:
            from pyaudio import PyAudio

            self._audio_context = PyAudio()
            self.logger.debug("🔄 Audio context reinitialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to reinitialize audio: {e}")
    
    @property
    def is_playing(self) -> bool:
        """True if TTS is currently synthesizing or playing audio. Thread-safe."""
        with self._state_lock:
            return self._state in (TTSState.SYNTHESIZING, TTSState.PLAYING, TTSState.STOPPING)
    
    @property
    def state(self) -> TTSState:
        """Get current TTS state for detailed tracking. Thread-safe."""
        with self._state_lock:
            return self._state
    
    @property
    def barge_in_count(self) -> int:
        """Total number of barge-in interruptions since startup."""
        return self._barge_in_count

    def cleanup(self) -> None:
        try:
            if self._audio_context:
                self._audio_context.terminate()
                self._audio_context = None
        except Exception:
            pass
