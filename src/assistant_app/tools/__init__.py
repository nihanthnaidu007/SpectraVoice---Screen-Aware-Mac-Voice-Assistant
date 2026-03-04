"""
Tools Package
=============
External tool integrations for SpectraVoice.
"""

from .web_search import WebSearch, SearchResult, SearchCategory
from .tool_executor import ToolExecutor, ToolResult, SafetyConfig, TOOL_SCHEMAS
from .safety import SafetyValidator, RiskLevel, create_confirmation_callback
from .mouse_controller import (
    MouseController,
    MouseButton,
    MovementStyle,
    get_mouse_controller
)
from .smart_click import SmartClicker, SmartClickResult, get_smart_clicker

__all__ = [
    # Web Search
    "WebSearch", 
    "SearchResult", 
    "SearchCategory",
    # Tool Executor
    "ToolExecutor",
    "ToolResult", 
    "SafetyConfig",
    "TOOL_SCHEMAS",
    # Safety
    "SafetyValidator",
    "RiskLevel",
    "create_confirmation_callback",
    # Mouse Control
    "MouseController",
    "MouseButton",
    "MovementStyle",
    "get_mouse_controller",
    # Smart Click
    "SmartClicker",
    "SmartClickResult",
    "get_smart_clicker",
]
