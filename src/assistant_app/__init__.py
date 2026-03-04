"""
SpectraVoice – Modular Services Package
=======================================
Clean, modular architecture for the SpectraVoice screen-aware assistant.
"""

from .utils.logging_config import setup_logging, get_logger
from .io.vision.screen_capture import ScreenCapture
from .io.audio.voice_detector import SmartVoiceDetector
from .io.audio.tts import TextToSpeech
from .core.conversation import ConversationMemory
from .core.assistant import Assistant
from .tools.web_search import WebSearch, SearchResult, SearchCategory
from .services.spectravoice_assistant import SpectraVoiceAssistant, PerformanceMetrics

__all__ = [
    "setup_logging",
    "get_logger",
    "ScreenCapture",
    "SmartVoiceDetector", 
    "ConversationMemory",
    "Assistant",
    "TextToSpeech",
    "WebSearch",
    "SearchResult",
    "SearchCategory",
    "SpectraVoiceAssistant",
    "PerformanceMetrics",
]

__version__ = "1.2.0"
