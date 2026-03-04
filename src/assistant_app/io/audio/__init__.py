"""
Audio I/O Package
=================
Audio input/output modules for voice detection and text-to-speech.
"""

from .voice_detector import SmartVoiceDetector
from .tts import TextToSpeech

__all__ = ["SmartVoiceDetector", "TextToSpeech"]
