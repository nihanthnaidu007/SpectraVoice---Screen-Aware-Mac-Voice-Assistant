"""SpectraVoice Controller - Main orchestration service."""

import atexit
import os
import signal
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from speech_recognition import Microphone, Recognizer, UnknownValueError

from assistant_app.utils.logging_config import get_logger
from assistant_app.io.vision.screen_capture import ScreenCapture
from assistant_app.io.audio.voice_detector import SmartVoiceDetector
from assistant_app.io.audio.tts import TextToSpeech
from assistant_app.io.indicator import start_indicator, update_status, stop_indicator
from assistant_app.core.assistant import Assistant

# GUI mode imports (lazy loaded)
if TYPE_CHECKING:
    import cv2
    import numpy as np

os.environ.setdefault('PYTHONWARNINGS', 'ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Greeting messages (avoid importing random for just this)
_GREETINGS = (
    "Hi! I'm ready to help.",
    "Hello! What can I do for you?",
    "Hey! Ready when you are.",
    "Hi there! How can I assist you?",
)


@dataclass
class PerformanceMetrics:
    transcription_time: float = 0.0
    api_response_time: float = 0.0
    tts_time: float = 0.0
    total_latency: float = 0.0
    interaction_count: int = 0
    
    def reset(self) -> None:
        self.transcription_time = self.api_response_time = self.tts_time = self.total_latency = 0.0


class SpectraVoiceAssistant:
    """
    SpectraVoice with barge-in support.
    
    Barge-in allows users to interrupt the assistant while it's speaking.
    The assistant will stop speaking and respond to the new query.
    
    Noise Protection (Step 6):
    - Only valid speech (passes SmartVoiceDetector) triggers barge-in
    - Minimum confidence threshold prevents noise from interrupting
    - Echo filtering prevents assistant from hearing itself
    """
    
    MODE_CONFIG = {"terminal": (80, 0.8, "base", 800), "gui": (80, 0.8, "base", 800), "minimal": (60, 0.6, "tiny", 300)}
    
    # === BARGE-IN CONFIGURATION ===
    # Tuned for responsive interruption while filtering pure noise
    BARGE_IN_MIN_CONFIDENCE = 0.40  # Low enough to catch real speech mid-TTS
    BARGE_IN_MIN_WORDS = 1  # Single word like "stop" or "hey" should interrupt
    BARGE_IN_MIN_CHARS = 3  # Even short commands like "hey" or "no" count
    
    def __init__(
        self, 
        mode: str = "terminal", 
        debug: bool = False, 
        voice: str = "shimmer", 
        whisper_model: str | None = None,
        llm_config = None,  # LLMConfig from assistant_app.llm (deprecated)
        llm_provider = None,  # Pre-validated LLMProvider instance
        enable_barge_in: bool = True,  # Enable barge-in (interrupt TTS with new speech)
    ):
        self.mode, self.running = mode, True
        self.debug = debug
        self.logger = get_logger(__name__)
        self.metrics = PerformanceMetrics()
        self.enable_barge_in = enable_barge_in
        self._barge_in_count = 0  # Track barge-in occurrences
        
        quality, scale, default_whisper, max_tokens = self.MODE_CONFIG.get(mode, self.MODE_CONFIG["terminal"])
        self.whisper_model = whisper_model or default_whisper
        self.voice = voice
        
        self.screen_capture = ScreenCapture(quality=quality, scale_factor=scale)
        
        # Initialize assistant with provider or config
        if llm_provider is not None:
            self.assistant = Assistant(max_tokens=max_tokens, provider=llm_provider)
        else:
            self.assistant = Assistant(max_tokens=max_tokens, llm_config=llm_config)
        
        self.tts = TextToSpeech(voice=voice)
        self.voice_detector = SmartVoiceDetector()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(stop_indicator)
        
        # Connect screen capture to tool executor for click verification
        if self.assistant.tool_executor:
            self.assistant.tool_executor.set_screen_capture(self.screen_capture)
        
        # Set up TTS callbacks for barge-in and completion tracking
        self.tts.set_on_interrupted(self._on_tts_interrupted)
        self.tts.set_on_complete(self._on_tts_complete)
        
    def _signal_handler(self, signum, frame) -> None:
        self.logger.info("🛑 Shutting down...")
        self.running = False
        stop_indicator()
    
    def _on_tts_interrupted(self) -> None:
        """Callback when TTS is interrupted by barge-in."""
        self._barge_in_count += 1
        self.logger.debug(f"📊 Barge-in count: {self._barge_in_count}")
        # Update indicator when interrupted
        update_status("Listening")
    
    def _on_tts_complete(self, was_interrupted: bool) -> None:
        """Callback when TTS finishes (either normally or interrupted)."""
        if not was_interrupted:
            # Mark TTS complete for echo filtering
            self.voice_detector.mark_tts_complete()
        # Update indicator back to listening
        update_status("Listening")

    def _audio_callback(self, recognizer, audio) -> None:
        """
        Callback for background voice recognition.
        
        This runs in a background thread from speech_recognition.
        With async TTS, this callback returns quickly, allowing continuous listening
        even while the assistant is speaking.
        
        Barge-in Logic (Step 4):
        - If TTS is playing and valid speech is detected → stop TTS
        - Process new speech as a new query (old response abandoned)
        
        Noise Protection (Step 6):
        - Requires minimum confidence (BARGE_IN_MIN_CONFIDENCE)
        - Requires minimum word count (BARGE_IN_MIN_WORDS)
        - Echo filtering prevents assistant from hearing itself
        """
        try:
            total_start = time.time()
            self.metrics.reset()
            
            trans_start = time.time()
            prompt = recognizer.recognize_whisper(audio, model=self.whisper_model, language="english")
            self.metrics.transcription_time = time.time() - trans_start
            
            # === BARGE-IN CHECK ===
            # Check if TTS is playing BEFORE validating speech
            is_tts_playing = self.tts.is_playing
            
            is_valid, reason, confidence = self.voice_detector.is_valid(prompt)
            
            if not is_valid:
                # Log TTS echo rejections at info level for debugging
                if reason == "tts_echo":
                    self.logger.info(f"🔇 Rejected TTS echo: '{prompt[:50]}...'")
                elif reason not in ("empty", "self_echo", "echo_cooldown"):
                    self.logger.debug(f"🔇 Ignored ({reason}): '{prompt}'")
                return
            
            # === STRICT NOISE PROTECTION FOR BARGE-IN ===
            # Only trigger barge-in for HIGH-confidence, multi-word, substantial speech
            # This prevents random noise/hallucinations from cancelling TTS
            word_count = len(prompt.split())
            char_count = len(prompt)
            
            # All conditions must be met for barge-in
            barge_in_conditions = {
                "tts_playing": is_tts_playing,
                "enabled": self.enable_barge_in,
                "confidence": confidence >= self.BARGE_IN_MIN_CONFIDENCE,
                "words": word_count >= self.BARGE_IN_MIN_WORDS,
                "chars": char_count >= self.BARGE_IN_MIN_CHARS,
            }
            should_barge_in = all(barge_in_conditions.values())
            
            # === BARGE-IN TRIGGER ===
            if should_barge_in:
                self._barge_in_count += 1
                self.logger.info(f"🗣️ Barge-in #{self._barge_in_count}: User interrupted assistant")
                self.logger.info(f"📝 Handling new query after barge-in: '{prompt[:60]}...'")
                self.tts.stop()
                # NO DELAY - proceed immediately to handle new query
            elif is_tts_playing and self.enable_barge_in:
                # Speech detected but didn't meet ALL barge-in thresholds
                failed = [k for k, v in barge_in_conditions.items() if not v and k not in ("tts_playing", "enabled")]
                self.logger.debug(
                    f"🔇 Speech during TTS rejected for barge-in (conf={confidence:.0%}, words={word_count}, chars={char_count}, failed={failed})"
                )
                # Don't process this as a query - let TTS continue
                return
            
            # Normalize input to correct mishearings
            normalized_prompt = self.voice_detector.normalize_input(prompt)
            if normalized_prompt != prompt:
                self.logger.debug(f"🔧 Normalized: '{prompt}' → '{normalized_prompt}'")
                prompt = normalized_prompt
            
            self.logger.info(f"🎤 User ({confidence:.0%}): {prompt}")
            
            # Update indicator to show we're processing
            update_status("Thinking")
            
            image_data = self.screen_capture.get_encoded()
            if not image_data:
                self.logger.error("❌ No screen data available")
                update_status("Listening")
                return
            
            self.logger.info("👁️ Analyzing screen...")
            
            api_start = time.time()
            response = self.assistant.process(prompt, image_data)
            self.metrics.api_response_time = time.time() - api_start
            
            if response:
                self.logger.info(f"🤖 Assistant: {response}")
                
                # Record metrics before starting async TTS
                self.metrics.total_latency = time.time() - total_start
                self.metrics.interaction_count += 1
                
                self.logger.debug(
                    f"⏱️ Performance: Trans={self.metrics.transcription_time:.2f}s, "
                    f"API={self.metrics.api_response_time:.2f}s, "
                    f"Total (pre-TTS)={self.metrics.total_latency:.2f}s"
                )
                
                # === ASYNC TTS - NON-BLOCKING ===
                # Start speaking in background thread so we can continue listening
                # The callback returns immediately, allowing new speech detection
                # TTS completion/interruption is handled via callbacks
                update_status("Speaking")
                
                # Record what we're about to say for echo detection
                # This prevents the mic from picking up our own TTS and triggering barge-in
                self.voice_detector.mark_tts_start(response)
                
                self.tts.speak_async(response)
                
                # NOTE: We return here immediately!
                # The _on_tts_complete callback will handle:
                # - Marking TTS complete for echo filtering
                # - Updating indicator back to "Listening"
            else:
                self.logger.warning("❌ No response generated")
                update_status("Listening")
                
        except UnknownValueError:
            pass
        except Exception as e:
            self.logger.error(f"❌ Audio error: {e}")
            update_status("Listening")
    
    def run(self) -> None:
        self.logger.info(f"🚀 SpectraVoice Starting ({self.mode.upper()} mode)")
        self.logger.info(f"🎙️ Voice: {self.voice} | Whisper: {self.whisper_model}")
        print("=" * 50)
        
        self.logger.info("🎤 Setting up microphone...")
        recognizer = Recognizer()
        # Higher energy threshold = less sensitive to background noise
        # This reduces false triggers from ambient sounds
        recognizer.energy_threshold = 5000  # Raised from 4000 to reduce noise sensitivity
        recognizer.dynamic_energy_threshold = False
        recognizer.pause_threshold = 0.8  # Seconds of silence before considering speech complete
        
        microphone = Microphone()
        with microphone as source:
            self.logger.info("🔧 Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
        
        self.logger.info("📸 Starting screen capture...")
        self.screen_capture.start()
        
        self.logger.info("👂 Starting voice recognition...")
        stop_listening = recognizer.listen_in_background(microphone, self._audio_callback)
        
        # Start on-screen status indicator (non-blocking)
        start_indicator("Listening")
        
        self.logger.info("✅ READY! SpectraVoice is listening...")
        print("👁️ I can see your screen and help with anything")
        print("🎤 Speak naturally - I'm always listening")
        if self.enable_barge_in:
            print("🗣️ Barge-in enabled - interrupt me anytime!")
        print("🛑 Press Ctrl+C to quit")
        print("=" * 50)
        
        self._speak_greeting()
        
        try:
            if self.mode == "gui":
                self._run_gui_mode()
            else:
                self._run_terminal_mode()
        finally:
            self._cleanup(stop_listening)
    
    def _speak_greeting(self) -> None:
        """Analyze screen silently first, then speak greeting."""
        self.logger.info("👁️ Pre-analyzing screen context...")
        
        # Wait for screen capture to produce at least one frame
        image_data = None
        for _ in range(20):  # up to 2 seconds
            image_data = self.screen_capture.get_encoded()
            if image_data:
                break
            time.sleep(0.1)
        
        if image_data:
            try:
                self.assistant.process(
                    "Note what's on screen silently. Remember it for context. Reply with just 'Ready' and nothing else.",
                    image_data
                )
                self.logger.info("✅ Screen context loaded")
            except Exception as e:
                self.logger.debug(f"⚠️ Pre-analysis skipped: {e}")
        else:
            self.logger.warning("⚠️ Screen capture not ready, skipping pre-analysis")
        
        greeting = _GREETINGS[int(time.time()) % len(_GREETINGS)]
        
        self.logger.info(f"🤖 {greeting}")
        try:
            self.voice_detector.mark_tts_start(greeting)
            self.tts.speak(greeting)
            self.voice_detector.mark_tts_complete()
        except Exception as e:
            self.logger.warning(f"⚠️ Could not speak greeting: {e}")
    
    def _cleanup(self, stop_listening) -> None:
        self.logger.info("🔄 Cleaning up...")
        
        # Stop the status indicator
        stop_indicator()
        
        for cleanup in [lambda: stop_listening(wait_for_stop=False), self.screen_capture.stop, self.tts.cleanup]:
            try:
                cleanup()
            except Exception:
                pass
        if self.mode == "gui":
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
        self.logger.info("👋 SpectraVoice stopped")

    def _run_terminal_mode(self) -> None:
        """Run in terminal mode with periodic heartbeat."""
        last_heartbeat = time.time()
        while self.running:
            time.sleep(1)
            if time.time() - last_heartbeat >= 30:
                self.logger.info(f"💓 SpectraVoice active... (interactions: {self.metrics.interaction_count})")
                last_heartbeat = time.time()
    
    def _run_gui_mode(self) -> None:
        """Run in GUI mode with visual status window."""
        import cv2
        import numpy as np
        import math
        
        status_window = np.zeros((250, 500, 3), dtype=np.uint8)
        
        while self.running:
            display = status_window.copy()
            
            cv2.putText(display, "SpectraVoice", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "Status: LISTENING", (50, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"Mode: {self.mode.upper()}", (50, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(display, "Speak naturally to interact", (50, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display, "Press 'q' or ESC to quit", (50, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            
            pulse = int(50 + 30 * abs(math.sin(time.time() * 3)))
            cv2.circle(display, (420, 50), 15, (0, pulse, 0), -1)
            
            cv2.imshow("SpectraVoice", display)
            
            if cv2.waitKey(100) in [27, ord("q")]:
                self.running = False
                break
