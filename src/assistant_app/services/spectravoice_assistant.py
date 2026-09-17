"""SpectraVoice Controller - Main orchestration service."""

import atexit
import os
import signal
import time
import warnings
from dataclasses import dataclass, replace

from speech_recognition import Microphone, Recognizer, UnknownValueError

from assistant_app.core.assistant import Assistant
from assistant_app.core.consent import PrivacyConsent
from assistant_app.io.audio.tts import TextToSpeech
from assistant_app.io.audio.voice_detector import SmartVoiceDetector
from assistant_app.io.hotkeys import DictationHotkeyListener, parse_hotkey
from assistant_app.io.indicator import start_indicator, stop_indicator, update_status
from assistant_app.io.vision.screen_capture import ScreenCapture
from assistant_app.services.dictation import DictationController
from assistant_app.services.transcription import TranscriptionWorker
from assistant_app.utils.config import ConfigManager, get_config_manager
from assistant_app.utils.logging_config import get_logger

# GUI mode imports (lazy loaded)

os.environ.setdefault('PYTHONWARNINGS', 'ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Greeting messages (avoid importing random for just this)
_GREETINGS = (
    "Hi! I'm ready to help.",
    "Hello! What can I do for you?",
    "Hey! Ready when you are.",
    "Hi there! How can I assist you?",
)

# The active assistant's ToolExecutor, for components that must type at the
# cursor through the same automation layer (with its safety-stop) as tools —
# currently only dictation. Set in __init__, read by dictation's default typer.
_active_tool_executor = None


def get_active_tool_executor():
    """ToolExecutor of the running assistant, or None (e.g. in unit tests)."""
    return _active_tool_executor


@dataclass
class PerformanceMetrics:
    transcription_time: float = 0.0
    api_response_time: float = 0.0
    tts_time: float = 0.0
    total_latency: float = 0.0
    handback_time: float = 0.0  # time the audio callback held the recognizer thread
    interaction_count: int = 0
    
    def reset(self) -> None:
        self.transcription_time = self.api_response_time = self.tts_time = 0.0
        self.total_latency = self.handback_time = 0.0


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
    
    def __init__(
        self, 
        mode: str = "terminal", 
        debug: bool = False, 
        voice: str | None = None, 
        whisper_model: str | None = None,
        llm_config = None,  # LLMConfig from assistant_app.llm (deprecated)
        llm_provider = None,  # Pre-validated LLMProvider instance
        enable_barge_in: bool | None = None,  # None = use config value
        config_manager: ConfigManager | None = None,  # None = global instance
        dictation_enabled: bool | None = None,  # None = use config value
        dictation_activation: str | None = None,  # None = use config value
    ):
        # Config is the single source of runtime settings (config.yaml + env
        # overrides); explicit CLI-derived arguments passed in here win.
        self.config_manager = config_manager or get_config_manager()
        cfg = self.config_manager.config
        profile = cfg.mode_profile(mode)
        
        self.mode, self.running = mode, True
        self.debug = debug
        self.logger = get_logger(__name__)
        self.metrics = PerformanceMetrics()
        
        # === BARGE-IN CONFIGURATION (config `barge_in:` section) ===
        # Tuned for responsive interruption while filtering pure noise
        self.enable_barge_in = cfg.barge_in.enabled if enable_barge_in is None else enable_barge_in
        self._barge_in_count = 0  # Track barge-in occurrences
        self._barge_min_confidence = cfg.barge_in.min_confidence  # Low enough to catch real speech mid-TTS
        self._barge_min_words = cfg.barge_in.min_words  # Single word like "stop" should interrupt
        self._barge_min_chars = cfg.barge_in.min_chars  # Even short commands like "no" count
        
        self.whisper_model = whisper_model or profile.whisper_model
        self.voice = voice or cfg.voice.tts_voice
        self.mic_config = cfg.microphone
        
        self.screen_capture = ScreenCapture(
            quality=profile.screen_quality,
            scale_factor=profile.screen_scale,
            refresh_interval=cfg.screen.refresh_interval,
            cache_duration=cfg.screen.cache_duration,
        )
        
        # Privacy: screen content leaves the machine only with explicit consent
        # (SPECTRAVOICE_SCREEN_CONSENT, default OFF) and while not paused.
        self.privacy_consent = PrivacyConsent.from_env()
        
        # Initialize assistant with provider or config
        if llm_provider is not None:
            self.assistant = Assistant(max_tokens=profile.max_tokens, provider=llm_provider, privacy_consent=self.privacy_consent)
        else:
            self.assistant = Assistant(max_tokens=profile.max_tokens, llm_config=llm_config, privacy_consent=self.privacy_consent)
        
        self.tts = TextToSpeech(
            voice=self.voice,
            hd_quality=cfg.tts.hd_quality,
            output_device=cfg.tts.output_device,
            speech_rate=cfg.voice.speech_rate,
        )
        self.voice_detector = SmartVoiceDetector()
        self._transcription_worker: TranscriptionWorker | None = None
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(stop_indicator)
        
        # Connect screen capture to tool executor for click verification
        if self.assistant.tool_executor:
            self.assistant.tool_executor.set_screen_capture(self.screen_capture)
        
        # Set up TTS callbacks for barge-in and completion tracking
        self.tts.set_on_interrupted(self._on_tts_interrupted)
        self.tts.set_on_complete(self._on_tts_complete)

        # === DICTATION (W1) ===
        # Local Whisper transcripts are routed here INSTEAD of the LLM whenever
        # dictation is armed. Insertion reuses this assistant's ToolExecutor
        # (type_text) so dry-run, audit logging, and the PyAutoGUI FAILSAFE
        # safety-stop apply to dictation exactly as to any other automation.
        global _active_tool_executor
        _active_tool_executor = self.assistant.tool_executor
        self.dictation_enabled = (
            cfg.dictation.enabled if dictation_enabled is None else dictation_enabled
        )
        self._dictation_hotkey_listener: DictationHotkeyListener | None = None
        self.dictation: DictationController | None = None
        if self.dictation_enabled:
            dict_cfg = cfg.dictation
            if dictation_activation is not None and dictation_activation != dict_cfg.activation:
                # CLI flag wins — copy, never mutate the shared config object
                dict_cfg = replace(dict_cfg, activation=dictation_activation)
            self.dictation = DictationController(
                dict_cfg,
                on_status=lambda status: self.logger.info(f"📝 {status}"),
                on_dictated=self._on_dictated_text,
            )
        
    def _on_dictated_text(self, text: str) -> None:
        """DictationController success callback (insert thread)."""
        self.metrics.interaction_count += 1
        self.logger.info(f"📝 Dictated: {text}")

    def _signal_handler(self, signum, frame) -> None:
        self.logger.info("🛑 Shutting down...")
        self.running = False
        stop_indicator()
    
    def pause_screen_sharing(self) -> None:
        """Runtime privacy toggle: stop sending screen content to the LLM."""
        self.privacy_consent.pause()
        self.logger.info("🔒 Screen sharing paused — queries run without vision")
    
    def resume_screen_sharing(self) -> None:
        """Clear the runtime pause (only effective if consent is granted)."""
        self.privacy_consent.resume()
        if self.privacy_consent.screen_upload_allowed:
            self.logger.info("🔓 Screen sharing resumed")
        else:
            self.logger.info("🔒 Pause cleared — screen sharing still off (no consent)")
    
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
        
        Runs in a fresh thread from speech_recognition's listener. It only hands
        the utterance to the transcription pipeline and returns: with the
        TranscriptionWorker, recognize_whisper no longer holds this thread
        (before W1 it blocked here for the full model-dependent transcription
        time). Set `microphone.inline_transcription: true` in config.yaml to
        restore the pre-W1 inline path.
        """
        cb_start = time.time()
        self.metrics.reset()
        
        # Dictation audio retention: default policy keeps clips in memory for
        # the session only (persist_audio=false → this call discards them).
        if self.dictation is not None and self.dictation.armed:
            self.dictation.record_audio(audio, cb_start)
        
        if self.mic_config.inline_transcription:
            self._transcribe_and_handle(recognizer, audio, cb_start)
            return
        
        worker = self._transcription_worker
        if worker is None or not worker.submit(audio, spoken_at=cb_start):
            self.logger.warning("🔇 Transcription queue full — dropping utterance")
            return
        self.metrics.handback_time = time.time() - cb_start
    
    def _transcribe_and_handle(self, recognizer, audio, spoken_at: float) -> None:
        """Inline path (legacy): transcribe on the calling thread, then handle."""
        try:
            prompt = self._recognize(recognizer, audio)
        except UnknownValueError:
            return  # no speech found in the clip — normal
        except Exception as e:
            self.logger.error(f"❌ Transcription failed: {e}")
            return
        self._handle_prompt(prompt, spoken_at)
    
    def _recognize(self, recognizer, audio) -> str:
        """Run Whisper on the audio. Raises on recognition failure.
        
        Pure transcription — transcript dispatch (_handle_prompt) belongs to the
        caller, so the inline path and the TranscriptionWorker share one entry
        point without handling the transcript twice.
        """
        trans_start = time.time()
        try:
            return recognizer.recognize_whisper(audio, model=self.whisper_model, language="english")
        finally:
            self.metrics.transcription_time = time.time() - trans_start
    
    def _handle_prompt(self, prompt: str, spoken_at: float) -> None:
        """Validate a transcript and run the LLM + TTS response pipeline.
        
        Barge-in Logic (Step 4):
        - If TTS is playing and valid speech is detected → stop TTS
        - Process new speech as a new query (old response abandoned)
        
        Noise Protection (Step 6):
        - Requires minimum confidence (barge_in.min_confidence)
        - Requires minimum word count (barge_in.min_words)
        - Echo filtering prevents assistant from hearing itself
        """
        try:
            total_start = spoken_at

            # === DICTATION ROUTING (W1) ===
            # When the dictation session is armed, this utterance is typed at
            # the cursor instead of running the screen-aware LLM flow. Checked
            # before SmartVoiceDetector filtering: its min-word/noise rules
            # would reject short legitimate dictation, and the dictation
            # pipeline applies its own light hallucination guard.
            if self.dictation is not None and self.dictation.armed:
                self.dictation.handle_transcript(prompt)
                return

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
                "confidence": confidence >= self._barge_min_confidence,
                "words": word_count >= self._barge_min_words,
                "chars": char_count >= self._barge_min_chars,
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
                    f"Handback={self.metrics.handback_time * 1000:.1f}ms, "
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
                
        except Exception as e:
            self.logger.error(f"❌ Audio error: {e}")
            update_status("Listening")

    
    def _open_microphone(self, recognizer: Recognizer) -> Microphone:
        """Open the configured input device, falling back to the system default on failure."""
        mic = self.mic_config
        try:
            microphone = Microphone(device_index=mic.input_device)
            if mic.input_device is not None:
                self.logger.info(f"🎙️ Using configured input device index {mic.input_device}")
            with microphone as source:
                self.logger.info("🔧 Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=mic.ambient_noise_seconds)
            return microphone
        except Exception as e:
            if mic.input_device is None:
                raise
            self.logger.error(
                f"❌ Configured input device {mic.input_device} unavailable ({e}); falling back to system default"
            )
            microphone = Microphone()
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=mic.ambient_noise_seconds)
            return microphone
    
    def _on_transcription_error(self, exc: Exception, _job) -> None:
        """TranscriptionWorker error callback."""
        if isinstance(exc, UnknownValueError):
            return  # no speech found in the audio clip — normal
        self.logger.error(f"❌ Transcription failed: {exc}")
    
    def run(self) -> None:
        self.logger.info(f"🚀 SpectraVoice Starting ({self.mode.upper()} mode)")
        self.logger.info(f"🎙️ Voice: {self.voice} | Whisper: {self.whisper_model}")
        print("=" * 50)
        
        self.logger.info("🎤 Setting up microphone...")
        recognizer = Recognizer()
        mic = self.mic_config
        recognizer.energy_threshold = mic.energy_threshold
        recognizer.dynamic_energy_threshold = mic.dynamic_energy_threshold
        recognizer.pause_threshold = mic.pause_threshold
        
        microphone = self._open_microphone(recognizer)
        
        self.logger.info("📸 Starting screen capture...")
        self.screen_capture.start()
        
        # Move Whisper off the audio-callback thread: transcription runs on a
        # dedicated worker while the recognizer loop keeps listening.
        if not mic.inline_transcription:
            self._transcription_worker = TranscriptionWorker(
                recognize=lambda audio: self._recognize(recognizer, audio),
                on_result=lambda text, job: self._handle_prompt(text, job.spoken_at),
                on_error=self._on_transcription_error,
            ).start()
        
        self.logger.info("👂 Starting voice recognition...")
        stop_listening = recognizer.listen_in_background(microphone, self._audio_callback)
        
        # Start dictation (controller + optional global hotkeys) before the
        # indicator so the status line reflects the dictation session.
        if self.dictation is not None:
            self.dictation.start()
            self._dictation_hotkey_listener = self._start_dictation_hotkeys()
        
        # Start on-screen status indicator (non-blocking)
        start_indicator("Listening")
        
        self.logger.info("✅ READY! SpectraVoice is listening...")
        if self.privacy_consent.screen_upload_allowed:
            print("👁️ Screen sharing is ON — screenshots are sent to your LLM provider with each query")
        else:
            print("🔒 Screen sharing is OFF — I will answer without seeing your screen")
            print("   (opt in by setting SPECTRAVOICE_SCREEN_CONSENT=1 in .env)")
        print("🎤 Speak naturally - I'm always listening")
        if self.enable_barge_in:
            print("🗣️ Barge-in enabled - interrupt me anytime!")
        if self.dictation is not None:
            print(f"📝 Dictation ON — {self.dictation.status_line()}")
            print("   Speech is transcribed locally and typed at your cursor; nothing leaves this machine")
            print("   Audio is not kept after this session (set dictation.persist_audio: true in config.yaml to change)")
            hotkeys_cfg = self.config_manager.config.hotkeys
            if self.dictation.activation == "push_to_talk":
                print(f"🎤 Push-to-talk: hold the '{hotkeys_cfg.push_to_talk}' key while speaking")
            else:
                print("🎤 Continuous dictation: speak, pause ~1-2s, and the text is inserted")
            print(f"🔁 Cycle activation modes with the dictation_mode hotkey ({hotkeys_cfg.dictation_mode})")
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
        
        cleanups = [
            lambda: stop_listening(wait_for_stop=False),
            lambda: self._transcription_worker.stop() if self._transcription_worker else None,
            lambda: self._dictation_hotkey_listener.stop() if self._dictation_hotkey_listener else None,
            lambda: self.dictation.shutdown() if self.dictation else None,
            self.screen_capture.stop,
            self.tts.cleanup,
        ]
        for cleanup in cleanups:
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

    def _start_dictation_hotkeys(self) -> DictationHotkeyListener | None:
        """Wire push-to-talk / mode-toggle hotkeys; degrade gracefully to
        flag-only control when pynput or permissions are unavailable."""
        hotkeys = self.config_manager.config.hotkeys
        ptt = parse_hotkey(hotkeys.push_to_talk)
        mode = parse_hotkey(hotkeys.dictation_mode)
        if not ptt:
            self.logger.warning("⚠️ No valid push_to_talk hotkey configured — push-to-talk unavailable")
            return None
        listener = DictationHotkeyListener(self.dictation, ptt, mode)
        listener_ok = listener.start()
        return listener if listener_ok else None

    def _run_terminal_mode(self) -> None:
        """Run in terminal mode with periodic heartbeat."""
        last_heartbeat = time.time()
        while self.running:
            time.sleep(1)
            # VAD dictation advances on this 1 Hz clock: buffered fragments
            # become an insertion once the silence window closes.
            if self.dictation is not None:
                try:
                    self.dictation.poll_vad()
                except Exception as e:  # heartbeat must never kill the loop
                    self.logger.error(f"❌ Dictation VAD poll failed: {e}")
            if time.time() - last_heartbeat >= 30:
                stats = self._transcription_worker.stats if self._transcription_worker else None
                extra = f", transcribed: {stats.completed}, dropped: {stats.dropped}" if stats else ""
                self.logger.info(f"💓 SpectraVoice active... (interactions: {self.metrics.interaction_count}{extra})")
                last_heartbeat = time.time()
    
    def _run_gui_mode(self) -> None:
        """Run in GUI mode with visual status window."""
        import math

        import cv2
        import numpy as np
        
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
