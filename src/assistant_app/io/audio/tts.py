"""Text-to-Speech Service - OpenAI TTS with PyAudio playback.

Supports barge-in interruption via the stop() method.
Runs playback in a background thread to allow concurrent listening.

Thread Safety:
- _state_lock: Protects _state and _should_stop (single source of truth)
- _playback_lock: Protects _current_player for safe interruption
- _lock: Ensures only one speak operation at a time

The stop() method can be safely called from any thread at any time.
"""

import threading
import time
from collections.abc import Callable
from enum import Enum

import openai
from pyaudio import PyAudio, paInt16

from assistant_app.utils.logging_config import get_logger


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
    
    def __init__(self, voice: str = "shimmer", hd_quality: bool = True):
        self.voice = voice
        self.hd_quality = hd_quality  # Use tts-1-hd for smoother, higher quality voice
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
        
    def _get_audio_context(self) -> PyAudio:
        if self._audio_context is None:
            self._audio_context = PyAudio()
        return self._audio_context
        
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
            
            text_to_speak = text[:500] if len(text) > 500 else text
            
            # Use tts-1-hd for smoother, more natural sounding voice
            tts_model = "tts-1-hd" if self.hd_quality else "tts-1"
            
            response = openai.audio.speech.create(
                model=tts_model,
                voice=self.voice,
                response_format="pcm",
                input=text_to_speak,
            )
            
            # Check for interrupt after API call - use Event for speed
            if self._stop_event.is_set():
                end_reason = "cancelled_after_synthesis"
                self.logger.info(f"⏹️ TTS end (id={utterance_id}, reason={end_reason})")
                return False
            
            audio_data = response.content
            
            if not audio_data:
                end_reason = "empty_audio"
                self.logger.warning(f"⚠️ TTS end (id={utterance_id}, reason={end_reason})")
                return True
            
            audio_bytes = len(audio_data)
            audio_duration_ms = (audio_bytes / 2) / self.SAMPLE_RATE * 1000  # 16-bit = 2 bytes per sample
            
            # Now playing audio - thread-safe state transition
            with self._state_lock:
                self._state = TTSState.PLAYING
            self.logger.debug(f"🔊 TTS playback started (id={utterance_id}, audio={audio_bytes} bytes, ~{audio_duration_ms:.0f}ms)")
            
            audio_ctx = self._get_audio_context()
            # Use larger buffer for smoother playback (8192 frames = ~340ms buffer)
            player = audio_ctx.open(
                format=paInt16,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                output=True,
                frames_per_buffer=8192,
            )
            
            # Store player reference for external interruption
            with self._playback_lock:
                self._current_player = player
            
            # Pre-buffer audio chunks before starting playback for smooth start
            chunks_played = 0
            total_chunks = (len(audio_data) + self.CHUNK_SIZE - 1) // self.CHUNK_SIZE
            
            # Collect chunks into a list first
            all_chunks = []
            for i in range(0, len(audio_data), self.CHUNK_SIZE):
                chunk = audio_data[i:i + self.CHUNK_SIZE]
                if chunk:
                    all_chunks.append(chunk)
            
            # Play audio in chunks, checking for interrupt periodically (not every chunk)
            # This reduces overhead and improves smoothness
            interrupt_check_interval = 2  # Check every 2 chunks (~340ms)
            
            for idx, chunk in enumerate(all_chunks):
                # === INTERRUPT CHECK (periodic, not every chunk) ===
                if idx % interrupt_check_interval == 0 and self._stop_event.is_set():
                    end_reason = "barge_in"
                    self.logger.info(f"⏹️ TTS end (id={utterance_id}, reason={end_reason}, played={chunks_played}/{total_chunks} chunks)")
                    was_interrupted = True
                    break
                
                try:
                    player.write(chunk)
                    chunks_played += 1
                except OSError as e:
                    # Stream was stopped externally
                    end_reason = f"playback_error: {e}"
                    self.logger.warning(f"⚠️ TTS end (id={utterance_id}, reason={end_reason})")
                    was_interrupted = True
                    break
            
            if not was_interrupted:
                end_reason = "completed"
                self.logger.info(f"✅ TTS end (id={utterance_id}, reason={end_reason}, played={chunks_played}/{total_chunks} chunks)")
            
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
    
    def _reinit_audio(self) -> None:
        try:
            if self._audio_context:
                self._audio_context.terminate()
        except Exception:
            pass
        self._audio_context = None
        
        time.sleep(0.2)
        
        try:
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
