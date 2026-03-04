"""Configuration Management System with YAML/JSON support."""

import os
from pathlib import Path
from dataclasses import dataclass, field
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
    provider: str = "cloud"  # "cloud" or "local"
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
class AssistantConfig:
    """Main configuration container."""
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    llm: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    api: APIConfig = field(default_factory=APIConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    mode: str = "terminal"
    debug: bool = False


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
    
    def __init__(self, config_path: str | Path | None = None):
        self.config_path = self._find_config_file(config_path)
        self._config: AssistantConfig | None = None
        self._raw_config: dict = {}
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
        env_mappings = {
            # Voice
            f"{self.ENV_PREFIX}VOICE_TTS_VOICE": ("voice", "tts_voice"),
            f"{self.ENV_PREFIX}VOICE_WHISPER_MODEL": ("voice", "whisper_model"),
            f"{self.ENV_PREFIX}VOICE_LANGUAGE": ("voice", "language"),
            
            # Screen
            f"{self.ENV_PREFIX}SCREEN_QUALITY": ("screen", "quality"),
            f"{self.ENV_PREFIX}SCREEN_SCALE": ("screen", "scale_factor"),
            
            # API
            f"{self.ENV_PREFIX}API_MODEL": ("api", "openai_model"),
            f"{self.ENV_PREFIX}API_MAX_TOKENS": ("api", "max_tokens"),
            f"{self.ENV_PREFIX}API_TEMPERATURE": ("api", "temperature"),
            
            # Logging
            f"{self.ENV_PREFIX}LOG_LEVEL": ("logging", "level"),
            f"{self.ENV_PREFIX}LOG_FILE": ("logging", "file"),
            
            # Mode
            f"{self.ENV_PREFIX}MODE": ("mode", None),
            f"{self.ENV_PREFIX}DEBUG": ("debug", None),
        }
        
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
        """Build AssistantConfig from dictionary."""
        return AssistantConfig(
            voice=VoiceConfig(**config_dict.get('voice', {})),
            screen=ScreenConfig(**config_dict.get('screen', {})),
            llm=LLMProviderConfig(**config_dict.get('llm', {})),
            api=APIConfig(**config_dict.get('api', {})),
            hotkeys=HotkeyConfig(**config_dict.get('hotkeys', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            safety=SafetyConfig(**config_dict.get('safety', {})),
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
        """Reload configuration from file."""
        logger.info("🔄 Reloading configuration...")
        return self.load()
    
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
        
        # Logging validation
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if cfg.logging.level.upper() not in valid_levels:
            issues.append(f"Invalid log level: {cfg.logging.level}. Valid: {valid_levels}")
        
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
            },
            'logging': {
                'level': cfg.logging.level,
                'file': cfg.logging.file,
                'max_size_mb': cfg.logging.max_size_mb,
                'backup_count': cfg.logging.backup_count,
            },
            'safety': {
                'dry_run': cfg.safety.dry_run,
                'block_dangerous_commands': cfg.safety.block_dangerous_commands,
                'require_confirmation': cfg.safety.require_confirmation,
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
