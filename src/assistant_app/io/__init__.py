"""
I/O Package
===========
Input/Output modules for audio and vision processing.
"""

from .audio import SmartVoiceDetector, TextToSpeech
from .vision import ScreenCapture

__all__ = ["SmartVoiceDetector", "TextToSpeech", "ScreenCapture"]
