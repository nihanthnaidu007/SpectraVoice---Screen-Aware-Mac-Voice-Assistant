"""Configuration Management System with YAML/JSON support."""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from assistant_app.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class VoiceConfig:
    """Voice-related configuration."""
    tts_voice: str = "shimmer"
    whisper_model: str = "base"
    language: str = "en"
    speech_rate: float = 1.0


@dataclass
class ScreenConfig:
    """Screen capture configuration."""
    quality: int = 80
    scale_factor: float = 0.8
    refresh_interval: float = 0.2
    cache_duration: float = 1.0


@dataclass
class LLMProviderConfig:
    """LLM Provider configuration."""
    provider: str = "auto"  # "cloud", "local", or "auto" (fall through to the interactive/GUI selector)
    cloud_model: str = "gpt-5"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-r1:7b"
    max_tokens: int = 800
    temperature: float = 0.7
    timeout: int = 60
    enable_vision: bool = True
    enable_tools: bool = True


@dataclass
class APIConfig:
    """API configuration (legacy, use LLMProviderConfig)."""
    openai_model: str = "gpt-5"
    max_tokens: int = 800
    temperature: float = 0.7
    timeout: int = 30
    max_retries: int = 3


@dataclass
class HotkeyConfig:
    """Hotkey configuration."""
    mute_toggle: str = "cmd+shift+m"
    push_to_talk: str = "space"
    pause_resume: str = "cmd+shift+p"
    quit: str = "cmd+shift+q"
    # Tap to cycle dictation activation (push_to_talk <-> vad)
    dictation_mode: str = "cmd+shift+d"
    # Start/stop the (explicitly started) meeting recording
    meeting_toggle: str = "cmd+shift+e"


@dataclass
class DictationConfig:
    """Dictation mode settings (W1): speech -> cleaned text -> typed at cursor.

    Local-first: transcripts never leave the device (Whisper runs locally) and
    audio is session-only unless persist_audio is enabled. Styles and snippets
    live here — config.yaml stays the single settings source.
    """
    enabled: bool = False
    # "push_to_talk" (hold the hotkey) or "vad" (continuous, silence-gap ended)
    activation: str = "push_to_talk"
    # Cleanup style: "standard" (filler removal for prose) or "minimal"
    # (verbatim text for terminals — only snippets expand)
    style: str = "standard"
    # Per-app style overrides, lowercased app name -> style
    app_styles: dict[str, str] = field(
        default_factory=lambda: {"terminal": "minimal", "iterm": "minimal", "iterm2": "minimal"}
    )
    filler_removal: bool = True
    filler_words: list[str] = field(
        default_factory=lambda: ["um", "uh", "er", "ah", "hmm", "like", "you know", "i mean", "sort of", "kind of"]
    )
    # Spoken phrase -> typed expansion
    snippets: dict[str, str] = field(
        default_factory=lambda: {
            "period": ".",
            "comma": ",",
            "question mark": "?",
            "exclamation mark": "!",
            "colon": ":",
            "semicolon": ";",
            "open paren": "(",
            "close paren": ")",
            "new line": "\n",
            "new paragraph": "\n\n",
        }
    )
    # Seconds of silence that end a VAD-mode utterance
    vad_silence_timeout: float = 1.2
    # Insert an Enter keystroke after each dictation insert
    insert_enter: bool = False
    # Privacy: dictation audio dies with the session unless explicitly enabled
    persist_audio: bool = False
    # Where persisted clips go (local disk only — never uploaded)
    audio_dir: str = ""
    # Retention (W3): persisted clips older than this many hours are deleted at
    # startup; 0 keeps them forever. Shares the W3 retention mechanism with
    # meeting artifacts (one mechanism, two consumers).
    retention_hours: float = 0.0


@dataclass
class MeetingConfig:
    """Meeting recording settings (W3): transparent, consent-gated capture + local summaries.

    Default-OFF in both layers: ``enabled`` is the config kill-switch, and even
    when it is on, a recording starts ONLY via an explicit per-meeting action
    (the ``--meeting`` flag, the HUD item, or the hotkey) — no stealth capture,
    ever, and the recording state stays visible while capture is live. Meeting
    is a flag-gated feature section (the dictation precedent), not a fourth
    top-level ``mode``.

    Summaries are local-first: the cloud (OpenAI) summarizer exists in code but
    requires BOTH ``summarizer: "cloud"`` AND the separately-consented
    ``cloud_consent`` flag (off by default) — third-party speech never leaves
    the machine unless the user makes that active, explicit choice.
    """

    # Kill-switch: when false, meeting recording is impossible (every start
    # path refuses). When true, recordings still need the per-meeting start.
    enabled: bool = False
    # Whisper ``language`` parameter for meeting transcription (R10): meetings
    # are where multilingual content realistically shows up, so this is a knob
    # instead of dictation's hardcoded "english".
    language: str = "english"
    # Fidelity transcription queue depth (not the dictation depth of 2, which
    # evicts the oldest utterance under load — silent transcript loss). Any
    # residual drop at this depth persists a gap marker in the transcript.
    queue_depth: int = 256
    # Keep per-utterance WAV clips alongside the transcript (local-only).
    persist_audio: bool = False
    # Local-only meeting directory; "" resolves to logs/meetings
    audio_dir: str = ""
    # Retention: meeting artifacts older than this many hours are deleted at
    # startup; 0 keeps them forever (the user decides; nothing is destroyed
    # silently by default).
    retention_hours: float = 0.0
    # Summarizer: "local" (Ollama, default) or "cloud" (OpenAI). The cloud path
    # additionally requires ``cloud_consent`` — two independent switches.
    summarizer: str = "local"
    # Separate, explicit consent for sending third-party speech to the cloud.
    cloud_consent: bool = False
    # Map-reduce chunk size in characters (per-chunk prompt input budget).
    chunk_chars: int = 6000


@dataclass
class HistoryConfig:
    """History surface settings (W4): the privacy dashboard / search / CLI knobs.

    Deliberately minimal (H7): the settings window auto-renders every field,
    so only genuine user knobs live here — each with implemented behavior.
    Search itself has no knobs (scan-on-query is the locked decision, not a
    setting), and retention stays on the meeting/dictation sections (one
    shared mechanism, W3 D4).
    """

    # Default export destination ("" = ask every time). Export is the only
    # off-device path; a configured default just pre-fills the destination —
    # the user still chooses (dashboard panel / CLI DIR argument overrides).
    export_dir: str = ""


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: str = "logs/assistant.log"
    max_size_mb: int = 10
    backup_count: int = 5
    format: str = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"


@dataclass
class SafetyConfig:
    """Safety configuration."""
    dry_run: bool = False
    block_dangerous_commands: bool = True
    require_confirmation: bool = False


@dataclass
class ModeProfile:
    """Per-mode runtime profile (screen quality/scale, Whisper model, max tokens)."""
    screen_quality: int = 80
    screen_scale: float = 0.8
    whisper_model: str = "base"
    max_tokens: int = 800


# Fallback when config.yaml has no `modes:` section — preserves the per-mode
# behavior SpectraVoiceAssistant previously hardcoded in MODE_CONFIG.
DEFAULT_MODE_PROFILES: dict[str, ModeProfile] = {
    "terminal": ModeProfile(80, 0.8, "base", 800),
    "gui": ModeProfile(80, 0.8, "base", 800),
    "minimal": ModeProfile(60, 0.6, "tiny", 300),
}


@dataclass
class BargeInConfig:
    """Barge-in (interrupt while speaking) thresholds."""
    enabled: bool = True
    min_confidence: float = 0.40
    min_words: int = 1
    min_chars: int = 3


@dataclass
class MicrophoneConfig:
    """Microphone input settings. input_device: PyAudio device index or null for system default."""
    input_device: int | None = None
    energy_threshold: int = 5000
    dynamic_energy_threshold: bool = False
    pause_threshold: float = 0.8
    ambient_noise_seconds: float = 1.0
    # Transcribe inside the audio callback (pre-W1 behavior) instead of the
    # off-thread TranscriptionWorker. Kept for A/B measurement and fallback.
    inline_transcription: bool = False


@dataclass
class TTSConfig:
    """TTS output settings. output_device: PyAudio device index or null for system default."""
    output_device: int | None = None
    hd_quality: bool = True


@dataclass
class SupervisorConfig:
    """Auto-restart supervision settings.

    When enabled, the CLI runs the assistant as a supervised child process and
    restarts it after an unhandled crash. max_restarts crashes within
    window_seconds give up instead of looping forever; backoff grows from
    backoff_seconds up to backoff_max_seconds between restarts.
    """
    enabled: bool = False
    max_restarts: int = 5
    window_seconds: float = 60.0
    backoff_seconds: float = 1.0
    backoff_max_seconds: float = 30.0


@dataclass
class AssistantConfig:
    """Main configuration container."""
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    llm: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    api: APIConfig = field(default_factory=APIConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)
    microphone: MicrophoneConfig = field(default_factory=MicrophoneConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    dictation: DictationConfig = field(default_factory=DictationConfig)
    meeting: MeetingConfig = field(default_factory=MeetingConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    modes: dict[str, ModeProfile] = field(default_factory=dict)
    mode: str = "terminal"
    debug: bool = False

    def mode_profile(self, mode: str) -> ModeProfile:
        """Resolve the runtime profile for a mode: config `modes:` entry, else built-in default."""
        return (
            self.modes.get(mode)
            or DEFAULT_MODE_PROFILES.get(mode)
            or DEFAULT_MODE_PROFILES["terminal"]
        )


class ConfigManager:
    """
    Configuration manager with YAML/JSON support.
    
    Features:
    - Load from YAML or JSON file
    - Environment variable overrides
    - Default values for all settings
    - Validation of configuration
    - Hot-reload capability
    """
    
    DEFAULT_CONFIG_PATHS = [
        "config.yaml",
        "config.yml", 
        "config.json",
        ".assistant/config.yaml",
    ]
    
    ENV_PREFIX = "VA_"  # SpectraVoice environment variable prefix
    

    ENV_MAPPINGS = {  # env override table: ENV_VAR -> (section, field | None)
        # Voice
        f"{ENV_PREFIX}VOICE_TTS_VOICE": ("voice", "tts_voice"),
        f"{ENV_PREFIX}VOICE_WHISPER_MODEL": ("voice", "whisper_model"),
        f"{ENV_PREFIX}VOICE_LANGUAGE": ("voice", "language"),
        
        # Screen
        f"{ENV_PREFIX}SCREEN_QUALITY": ("screen", "quality"),
        f"{ENV_PREFIX}SCREEN_SCALE": ("screen", "scale_factor"),
        
        # API
        f"{ENV_PREFIX}API_MODEL": ("api", "openai_model"),
        f"{ENV_PREFIX}API_MAX_TOKENS": ("api", "max_tokens"),
        f"{ENV_PREFIX}API_TEMPERATURE": ("api", "temperature"),
        
        # Logging
        f"{ENV_PREFIX}LOG_LEVEL": ("logging", "level"),
        f"{ENV_PREFIX}LOG_FILE": ("logging", "file"),
        
        # Barge-in
        f"{ENV_PREFIX}BARGE_IN_ENABLED": ("barge_in", "enabled"),
        f"{ENV_PREFIX}BARGE_IN_MIN_CONFIDENCE": ("barge_in", "min_confidence"),
        
        # Microphone / TTS devices
        f"{ENV_PREFIX}MICROPHONE_INPUT_DEVICE": ("microphone", "input_device"),
        f"{ENV_PREFIX}MICROPHONE_ENERGY_THRESHOLD": ("microphone", "energy_threshold"),
        f"{ENV_PREFIX}TTS_OUTPUT_DEVICE": ("tts", "output_device"),

        # Supervisor (auto-restart)
        f"{ENV_PREFIX}SUPERVISOR_ENABLED": ("supervisor", "enabled"),
        f"{ENV_PREFIX}SUPERVISOR_MAX_RESTARTS": ("supervisor", "max_restarts"),

        # Dictation (W1)
        f"{ENV_PREFIX}DICTATION_ENABLED": ("dictation", "enabled"),
        f"{ENV_PREFIX}DICTATION_ACTIVATION": ("dictation", "activation"),
        f"{ENV_PREFIX}DICTATION_PERSIST_AUDIO": ("dictation", "persist_audio"),
        f"{ENV_PREFIX}DICTATION_RETENTION_HOURS": ("dictation", "retention_hours"),

        # Meeting recording (W3) — kill-switch + posture; a recording still
        # requires the explicit per-meeting start on top of these.
        f"{ENV_PREFIX}MEETING_ENABLED": ("meeting", "enabled"),
        f"{ENV_PREFIX}MEETING_LANGUAGE": ("meeting", "language"),
        f"{ENV_PREFIX}MEETING_SUMMARIZER": ("meeting", "summarizer"),
        f"{ENV_PREFIX}MEETING_RETENTION_HOURS": ("meeting", "retention_hours"),

        # History surface (W4) — dashboard/search/export knobs. Deletion has
        # NO env mapping on purpose: destructive actions want explicit intent,
        # and there is nothing to configure about them.
        f"{ENV_PREFIX}HISTORY_EXPORT_DIR": ("history", "export_dir"),
        
        # Mode
        f"{ENV_PREFIX}MODE": ("mode", None),
        f"{ENV_PREFIX}DEBUG": ("debug", None),
    }
    def __init__(self, config_path: str | Path | None = None):
        self.config_path = self._find_config_file(config_path)
        self._config: AssistantConfig | None = None
        self._raw_config: dict = {}
        self._reload_callbacks: list[Callable[[AssistantConfig], None]] = []
        self.load()
    
    def _find_config_file(self, config_path: str | Path | None) -> Path | None:
        """Find configuration file from path or default locations."""
        if config_path:
            path = Path(config_path)
            if path.exists():
                return path
            logger.warning(f"Config file not found: {config_path}")
        
        # Search default locations
        for default_path in self.DEFAULT_CONFIG_PATHS:
            path = Path(default_path)
            if path.exists():
                logger.info(f"📋 Found config: {path}")
                return path
        
        return None
    
    def load(self) -> AssistantConfig:
        """Load configuration from file with environment overrides."""
        # Start with defaults
        config_dict: dict[str, Any] = {}
        
        # Load from file if exists
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    if self.config_path.suffix in ('.yaml', '.yml'):
                        config_dict = yaml.safe_load(f) or {}
                    else:
                        import json
                        config_dict = json.load(f)
                logger.info(f"✅ Loaded config from {self.config_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load config: {e}")
                config_dict = {}
        
        self._raw_config = config_dict
        
        # Apply environment variable overrides
        config_dict = self._apply_env_overrides(config_dict)
        
        # Build configuration object
        self._config = self._build_config(config_dict)
        
        return self._config
    
    def _apply_env_overrides(self, config: dict) -> dict:
        """
        Apply environment variable overrides.
        
        Environment variables use format: VA_SECTION_KEY
        Example: VA_VOICE_TTS_VOICE=nova
        """
        env_mappings = self.ENV_MAPPINGS
        
        for env_var, (section, key) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                if key is None:
                    # Top-level config
                    config[section] = self._parse_env_value(value)
                else:
                    # Nested config
                    if section not in config:
                        config[section] = {}
                    config[section][key] = self._parse_env_value(value)
                logger.debug(f"📝 Env override: {env_var}")
        
        return config
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Boolean
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        if value.lower() in ('false', '0', 'no', 'off'):
            return False
        
        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        return value
    
    def _build_config(self, config_dict: dict) -> AssistantConfig:
        """Build AssistantConfig from dictionary, warning about unrecognized keys."""
        from dataclasses import fields as dc_fields

        def section(name: str, dc_cls: type) -> dict:
            raw = config_dict.get(name, {})
            if not isinstance(raw, dict):
                logger.warning(f"⚠️ Config section '{name}' must be a mapping, got {type(raw).__name__} — using defaults")
                return {}
            known = {f.name for f in dc_fields(dc_cls)}
            unknown = set(raw) - known
            if unknown:
                logger.warning(f"⚠️ Ignoring unknown config keys in '{name}': {sorted(unknown)}")
            return {k: v for k, v in raw.items() if k in known}

        raw_modes = config_dict.get('modes', {})
        modes: dict[str, ModeProfile] = {}
        if isinstance(raw_modes, dict):
            for mode_name, values in raw_modes.items():
                if not isinstance(values, dict):
                    logger.warning(f"⚠️ Ignoring mode profile '{mode_name}': must be a mapping")
                    continue
                fallback = DEFAULT_MODE_PROFILES.get(str(mode_name), DEFAULT_MODE_PROFILES["terminal"])
                try:
                    modes[str(mode_name)] = ModeProfile(
                        screen_quality=int(values.get("screen_quality", fallback.screen_quality)),
                        screen_scale=float(values.get("screen_scale", fallback.screen_scale)),
                        whisper_model=str(values.get("whisper_model", fallback.whisper_model)),
                        max_tokens=int(values.get("max_tokens", fallback.max_tokens)),
                    )
                except (TypeError, ValueError) as e:
                    logger.warning(f"⚠️ Ignoring mode profile '{mode_name}': {e}")

        return AssistantConfig(
            voice=VoiceConfig(**section('voice', VoiceConfig)),
            screen=ScreenConfig(**section('screen', ScreenConfig)),
            llm=LLMProviderConfig(**section('llm', LLMProviderConfig)),
            api=APIConfig(**section('api', APIConfig)),
            hotkeys=HotkeyConfig(**section('hotkeys', HotkeyConfig)),
            logging=LoggingConfig(**section('logging', LoggingConfig)),
            safety=SafetyConfig(**section('safety', SafetyConfig)),
            barge_in=BargeInConfig(**section('barge_in', BargeInConfig)),
            microphone=MicrophoneConfig(**section('microphone', MicrophoneConfig)),
            tts=TTSConfig(**section('tts', TTSConfig)),
            supervisor=SupervisorConfig(**section('supervisor', SupervisorConfig)),
            dictation=DictationConfig(**section('dictation', DictationConfig)),
            meeting=MeetingConfig(**section('meeting', MeetingConfig)),
            history=HistoryConfig(**section('history', HistoryConfig)),
            modes=modes,
            mode=config_dict.get('mode', 'terminal'),
            debug=config_dict.get('debug', False),
        )
    
    @property
    def config(self) -> AssistantConfig:
        """Get current configuration."""
        if self._config is None:
            self.load()
        return self._config  # type: ignore
    
    def reload(self) -> AssistantConfig:
        """Reload configuration from file and notify registered consumers.

        W2 makes the reload path functional: callbacks registered via
        on_reload() run after the fresh config is in place (settings UI
        hot-apply, HUD). Callback failures are isolated — one bad consumer
        cannot break the reload or its siblings.
        """
        logger.info("🔄 Reloading configuration...")
        config = self.load()
        self.notify_config_changed(config)
        return config

    def on_reload(self, callback: Callable[[AssistantConfig], None]) -> None:
        """Register a callback invoked after every successful reload()."""
        self._reload_callbacks.append(callback)

    def notify_config_changed(self, config: AssistantConfig) -> None:
        """Dispatch config-change notifications with failure isolation."""
        for callback in tuple(self._reload_callbacks):
            try:
                callback(config)
            except Exception:
                logger.exception("Config reload callback failed — continuing")

    @classmethod
    def env_mapping_table(cls) -> dict[tuple[str, str], str]:
        """{(section, field): ENV_VAR} reverse lookup for the settings UI.
        Top-level keys (mode, debug) use section ""."""
        table: dict[tuple[str, str], str] = {}
        for env_var, (section, key) in cls.ENV_MAPPINGS.items():
            if key is None:
                table[("", section)] = env_var
            else:
                table[(section, key)] = env_var
        return table
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation key.
        
        Example: config.get('voice.tts_voice')
        """
        parts = key.split('.')
        value: Any = self._raw_config
        
        try:
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return default
            return value if value is not None else default
        except (KeyError, TypeError):
            return default
    
    def validate(self) -> list[str]:
        """
        Validate configuration and return list of issues.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        cfg = self.config
        
        # Voice validation
        valid_voices = {'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'}
        if cfg.voice.tts_voice not in valid_voices:
            issues.append(f"Invalid TTS voice: {cfg.voice.tts_voice}. Valid: {valid_voices}")
        
        valid_whisper = {'tiny', 'base', 'small', 'medium', 'large'}
        if cfg.voice.whisper_model not in valid_whisper:
            issues.append(f"Invalid Whisper model: {cfg.voice.whisper_model}. Valid: {valid_whisper}")
        
        # Screen validation
        if not 1 <= cfg.screen.quality <= 100:
            issues.append(f"Screen quality must be 1-100, got: {cfg.screen.quality}")
        
        if not 0.1 <= cfg.screen.scale_factor <= 1.0:
            issues.append(f"Scale factor must be 0.1-1.0, got: {cfg.screen.scale_factor}")
        
        # API validation
        if cfg.api.max_tokens < 100:
            issues.append(f"Max tokens too low: {cfg.api.max_tokens}")
        
        if not 0 <= cfg.api.temperature <= 2:
            issues.append(f"Temperature must be 0-2, got: {cfg.api.temperature}")
        
        # LLM provider validation
        if cfg.llm.provider not in {'auto', 'cloud', 'local'}:
            issues.append(f"Invalid llm.provider: {cfg.llm.provider}. Valid: auto, cloud, local")
        
        # Speech rate validation (OpenAI TTS supports 0.25-4.0)
        if not 0.25 <= cfg.voice.speech_rate <= 4.0:
            issues.append(f"Speech rate must be 0.25-4.0, got: {cfg.voice.speech_rate}")
        
        # Logging validation
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if cfg.logging.level.upper() not in valid_levels:
            issues.append(f"Invalid log level: {cfg.logging.level}. Valid: {valid_levels}")
        
        # Barge-in validation
        if not 0 <= cfg.barge_in.min_confidence <= 1:
            issues.append(f"Barge-in min_confidence must be 0-1, got: {cfg.barge_in.min_confidence}")
        
        if cfg.barge_in.min_words < 1:
            issues.append(f"Barge-in min_words must be >= 1, got: {cfg.barge_in.min_words}")
        
        if cfg.barge_in.min_chars < 1:
            issues.append(f"Barge-in min_chars must be >= 1, got: {cfg.barge_in.min_chars}")
        
        # Microphone validation
        if cfg.microphone.energy_threshold <= 0:
            issues.append(f"Microphone energy_threshold must be > 0, got: {cfg.microphone.energy_threshold}")
        
        if cfg.microphone.pause_threshold < 0.1:
            issues.append(f"Microphone pause_threshold must be >= 0.1s, got: {cfg.microphone.pause_threshold}")
        
        if cfg.microphone.input_device is not None and cfg.microphone.input_device < 0:
            issues.append(f"Microphone input_device must be a non-negative index or null, got: {cfg.microphone.input_device}")
        
        if cfg.microphone.ambient_noise_seconds < 0:
            issues.append(f"Microphone ambient_noise_seconds must be >= 0, got: {cfg.microphone.ambient_noise_seconds}")

        # Dictation validation
        if cfg.dictation.activation not in {'push_to_talk', 'vad'}:
            issues.append(f"Invalid dictation.activation: {cfg.dictation.activation}. Valid: push_to_talk, vad")
        if cfg.dictation.style not in {'standard', 'minimal'}:
            issues.append(f"Invalid dictation.style: {cfg.dictation.style}. Valid: standard, minimal")
        for app_name, app_style in cfg.dictation.app_styles.items():
            if app_style not in {'standard', 'minimal'}:
                issues.append(f"Invalid dictation.app_styles['{app_name}']: {app_style}. Valid: standard, minimal")
        if cfg.dictation.vad_silence_timeout <= 0:
            issues.append(f"Dictation vad_silence_timeout must be > 0, got: {cfg.dictation.vad_silence_timeout}")
        if cfg.dictation.retention_hours < 0:
            issues.append(f"Dictation retention_hours must be >= 0 (0 = keep forever), got: {cfg.dictation.retention_hours}")

        # Meeting validation (W3)
        if not cfg.meeting.language.strip():
            issues.append("Meeting language must be a non-empty Whisper language name or code")
        if cfg.meeting.queue_depth < 1:
            issues.append(f"Meeting queue_depth must be >= 1, got: {cfg.meeting.queue_depth}")
        if cfg.meeting.retention_hours < 0:
            issues.append(f"Meeting retention_hours must be >= 0 (0 = keep forever), got: {cfg.meeting.retention_hours}")
        if cfg.meeting.summarizer not in {'local', 'cloud'}:
            issues.append(f"Invalid meeting.summarizer: {cfg.meeting.summarizer}. Valid: local, cloud")
        if cfg.meeting.summarizer == 'cloud' and not cfg.meeting.cloud_consent:
            issues.append(
                "Meeting summarizer is 'cloud' but meeting.cloud_consent is false — sending "
                "third-party speech to the cloud requires the explicit consent flag "
                "(or set summarizer back to 'local')"
            )
        if cfg.meeting.chunk_chars < 200:
            issues.append(f"Meeting chunk_chars must be >= 200, got: {cfg.meeting.chunk_chars}")

        # History validation (W4) — minimal: the section holds one path knob.
        if cfg.history.export_dir != cfg.history.export_dir.strip():
            issues.append("History export_dir must not have leading/trailing whitespace")
        
        # TTS validation
        if cfg.tts.output_device is not None and cfg.tts.output_device < 0:
            issues.append(f"TTS output_device must be a non-negative index or null, got: {cfg.tts.output_device}")

        # Supervisor validation
        if cfg.supervisor.max_restarts < 0:
            issues.append(f"Supervisor max_restarts must be >= 0, got: {cfg.supervisor.max_restarts}")
        if cfg.supervisor.window_seconds <= 0:
            issues.append(f"Supervisor window_seconds must be > 0, got: {cfg.supervisor.window_seconds}")
        if cfg.supervisor.backoff_seconds < 0:
            issues.append(f"Supervisor backoff_seconds must be >= 0, got: {cfg.supervisor.backoff_seconds}")
        if cfg.supervisor.backoff_max_seconds < 0:
            issues.append(f"Supervisor backoff_max_seconds must be >= 0, got: {cfg.supervisor.backoff_max_seconds}")
        
        # Mode validation
        valid_modes = {'terminal', 'gui', 'minimal'}
        if cfg.mode not in valid_modes:
            issues.append(f"Invalid mode: {cfg.mode}. Valid: {valid_modes}")
        
        return issues
    
    def to_dict(self) -> dict:
        """Export configuration as dictionary."""
        cfg = self.config
        return {
            'mode': cfg.mode,
            'debug': cfg.debug,
            'voice': {
                'tts_voice': cfg.voice.tts_voice,
                'whisper_model': cfg.voice.whisper_model,
                'language': cfg.voice.language,
                'speech_rate': cfg.voice.speech_rate,
            },
            'screen': {
                'quality': cfg.screen.quality,
                'scale_factor': cfg.screen.scale_factor,
                'refresh_interval': cfg.screen.refresh_interval,
                'cache_duration': cfg.screen.cache_duration,
            },
            'llm': {
                'provider': cfg.llm.provider,
                'cloud_model': cfg.llm.cloud_model,
                'ollama_url': cfg.llm.ollama_url,
                'ollama_model': cfg.llm.ollama_model,
                'max_tokens': cfg.llm.max_tokens,
                'temperature': cfg.llm.temperature,
                'timeout': cfg.llm.timeout,
                'enable_vision': cfg.llm.enable_vision,
                'enable_tools': cfg.llm.enable_tools,
            },
            'api': {
                'openai_model': cfg.api.openai_model,
                'max_tokens': cfg.api.max_tokens,
                'temperature': cfg.api.temperature,
                'timeout': cfg.api.timeout,
                'max_retries': cfg.api.max_retries,
            },
            'hotkeys': {
                'mute_toggle': cfg.hotkeys.mute_toggle,
                'push_to_talk': cfg.hotkeys.push_to_talk,
                'pause_resume': cfg.hotkeys.pause_resume,
                'quit': cfg.hotkeys.quit,
                'dictation_mode': cfg.hotkeys.dictation_mode,
                'meeting_toggle': cfg.hotkeys.meeting_toggle,
            },
            'logging': {
                'level': cfg.logging.level,
                'file': cfg.logging.file,
                'max_size_mb': cfg.logging.max_size_mb,
                'backup_count': cfg.logging.backup_count,
                'format': cfg.logging.format,
            },
            'safety': {
                'dry_run': cfg.safety.dry_run,
                'block_dangerous_commands': cfg.safety.block_dangerous_commands,
                'require_confirmation': cfg.safety.require_confirmation,
            },
            'barge_in': {
                'enabled': cfg.barge_in.enabled,
                'min_confidence': cfg.barge_in.min_confidence,
                'min_words': cfg.barge_in.min_words,
                'min_chars': cfg.barge_in.min_chars,
            },
            'microphone': {
                'input_device': cfg.microphone.input_device,
                'energy_threshold': cfg.microphone.energy_threshold,
                'dynamic_energy_threshold': cfg.microphone.dynamic_energy_threshold,
                'pause_threshold': cfg.microphone.pause_threshold,
                'ambient_noise_seconds': cfg.microphone.ambient_noise_seconds,
                'inline_transcription': cfg.microphone.inline_transcription,
            },
            'tts': {
                'output_device': cfg.tts.output_device,
                'hd_quality': cfg.tts.hd_quality,
            },
            'supervisor': {
                'enabled': cfg.supervisor.enabled,
                'max_restarts': cfg.supervisor.max_restarts,
                'window_seconds': cfg.supervisor.window_seconds,
                'backoff_seconds': cfg.supervisor.backoff_seconds,
                'backoff_max_seconds': cfg.supervisor.backoff_max_seconds,
            },
            'dictation': {
                'enabled': cfg.dictation.enabled,
                'activation': cfg.dictation.activation,
                'style': cfg.dictation.style,
                'app_styles': dict(cfg.dictation.app_styles),
                'filler_removal': cfg.dictation.filler_removal,
                'filler_words': list(cfg.dictation.filler_words),
                'snippets': dict(cfg.dictation.snippets),
                'vad_silence_timeout': cfg.dictation.vad_silence_timeout,
                'insert_enter': cfg.dictation.insert_enter,
                'persist_audio': cfg.dictation.persist_audio,
                'audio_dir': cfg.dictation.audio_dir,
                'retention_hours': cfg.dictation.retention_hours,
            },
            'meeting': {
                'enabled': cfg.meeting.enabled,
                'language': cfg.meeting.language,
                'queue_depth': cfg.meeting.queue_depth,
                'persist_audio': cfg.meeting.persist_audio,
                'audio_dir': cfg.meeting.audio_dir,
                'retention_hours': cfg.meeting.retention_hours,
                'summarizer': cfg.meeting.summarizer,
                'cloud_consent': cfg.meeting.cloud_consent,
                'chunk_chars': cfg.meeting.chunk_chars,
            },
            'history': {
                'export_dir': cfg.history.export_dir,
            },
            'modes': {
                name: {
                    'screen_quality': profile.screen_quality,
                    'screen_scale': profile.screen_scale,
                    'whisper_model': profile.whisper_model,
                    'max_tokens': profile.max_tokens,
                }
                for name, profile in cfg.modes.items()
            },
        }
    
    def save(self, path: str | Path | None = None) -> None:
        """Save current configuration to file."""
        save_path = Path(path) if path else self.config_path
        if not save_path:
            save_path = Path('config.yaml')
        
        config_dict = self.to_dict()
        
        with open(save_path, 'w') as f:
            if save_path.suffix == '.json':
                import json
                json.dump(config_dict, f, indent=2)
            else:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"💾 Configuration saved to {save_path}")


# Global config manager instance
_config_manager: ConfigManager | None = None


def get_config() -> AssistantConfig:
    """Get the global configuration."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.config


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def init_config(config_path: str | Path | None = None) -> ConfigManager:
    """Initialize configuration from a specific path."""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager
