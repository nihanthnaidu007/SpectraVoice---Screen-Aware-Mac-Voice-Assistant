"""
Audio I/O Package
=================
Audio input/output modules for voice detection and text-to-speech.
"""

from .tts import TextToSpeech
from .voice_detector import SmartVoiceDetector

__all__ = ["SmartVoiceDetector", "TextToSpeech"]
