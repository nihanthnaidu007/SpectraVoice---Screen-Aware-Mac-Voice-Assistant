"""
Tools Package
=============
External tool integrations for SpectraVoice.
"""

from .mouse_controller import MouseButton, MouseController, MovementStyle, get_mouse_controller
from .safety import RiskLevel, SafetyValidator, create_confirmation_callback
from .smart_click import SmartClicker, SmartClickResult, get_smart_clicker
from .tool_executor import TOOL_SCHEMAS, SafetyConfig, ToolExecutor, ToolResult
from .web_search import SearchCategory, SearchResult, WebSearch

__all__ = [
    "TOOL_SCHEMAS",
    "MouseButton",
    "MouseController",
    "MovementStyle",
    "RiskLevel",
    "SafetyConfig",
    "SafetyValidator",
    "SearchCategory",
    "SearchResult",
    "SmartClickResult",
    "SmartClicker",
    "ToolExecutor",
    "ToolResult",
    "WebSearch",
    "create_confirmation_callback",
    "get_mouse_controller",
    "get_smart_clicker",
]
