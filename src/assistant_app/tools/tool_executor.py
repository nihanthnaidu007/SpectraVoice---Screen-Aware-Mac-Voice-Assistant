"""Tool Executor Service - OpenAI Function Calling for macOS automation."""

import json
import subprocess
import os
import time
from typing import Any, Callable
from dataclasses import dataclass, field

from assistant_app.utils.logging_config import get_logger
from assistant_app.tools.mouse_controller import (
    MouseController, MouseButton, MovementStyle, ScreenRegion,
    get_mouse_controller
)
from assistant_app.tools.smart_click import SmartClicker, get_smart_clicker


# OpenAI Function Schemas for GPT-5
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "open_application",
            "description": "Open a macOS application by name. Use this when the user wants to launch or open an app.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the application to open (e.g., 'Chrome', 'Safari', 'Finder', 'Notes', 'Terminal')"
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text at the current cursor/focus position. IMPORTANT: You must FIRST use click_element to click on the target input field/search bar/text area before using type_text. If you use type_text without clicking on the right element first, the text will go to the wrong place or nowhere.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to type into the active window"
                    },
                    "press_enter": {
                        "type": "boolean",
                        "description": "Whether to press Enter after typing the text",
                        "default": False
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "click_element",
            "description": "Click on any screen element (buttons, links, search bars, video thumbnails, icons, input fields, etc.) at specified coordinates. You can see the screen — identify the element's position and click it. ALWAYS use this before type_text to focus the target input. For opening desktop icons, use clicks=2.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "X coordinate - aim for CENTER of the target element"
                    },
                    "y": {
                        "type": "integer",
                        "description": "Y coordinate - aim for CENTER of the target element"
                    },
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button: left (default), right (context menu), middle",
                        "default": "left"
                    },
                    "clicks": {
                        "type": "integer",
                        "enum": [1, 2, 3],
                        "description": "1=single click, 2=double click (to OPEN items), 3=triple click",
                        "default": 1
                    },
                    "element_description": {
                        "type": "string",
                        "description": "What you are clicking on (e.g., 'Chrome icon', 'OK button')"
                    },
                    "verify": {
                        "type": "boolean",
                        "description": "Verify the click had an effect by checking if screen changed (default: true)",
                        "default": True
                    }
                },
                "required": ["x", "y"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_mouse",
            "description": "Move the mouse cursor to a specific position without clicking. Use for hovering or positioning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "description": "Target X coordinate"
                    },
                    "y": {
                        "type": "integer",
                        "description": "Target Y coordinate"
                    },
                    "relative": {
                        "type": "boolean",
                        "description": "If true, x and y are offsets from current position",
                        "default": False
                    },
                    "movement": {
                        "type": "string",
                        "enum": ["instant", "smooth", "natural"],
                        "description": "Movement style (default: smooth)",
                        "default": "smooth"
                    },
                    "duration": {
                        "type": "number",
                        "description": "Movement duration in seconds (default: 0.25)",
                        "default": 0.25
                    }
                },
                "required": ["x", "y"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drag_mouse",
            "description": "Drag the mouse from current position (or start position) to end position. Useful for drag-and-drop, selecting text, moving windows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "end_x": {
                        "type": "integer",
                        "description": "End X coordinate for the drag"
                    },
                    "end_y": {
                        "type": "integer",
                        "description": "End Y coordinate for the drag"
                    },
                    "start_x": {
                        "type": "integer",
                        "description": "Start X coordinate (optional, uses current position if not specified)"
                    },
                    "start_y": {
                        "type": "integer",
                        "description": "Start Y coordinate (optional, uses current position if not specified)"
                    },
                    "button": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Mouse button to use for dragging (default: left)",
                        "default": "left"
                    },
                    "duration": {
                        "type": "number",
                        "description": "Drag duration in seconds (default: 0.5)",
                        "default": 0.5
                    }
                },
                "required": ["end_x", "end_y"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_mouse_position",
            "description": "Get the current mouse cursor position on screen.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "click_at_cursor",
            "description": "Click at the CURRENT mouse cursor position. Use this when user says they placed/moved cursor to a target and wants you to click there. This is more accurate than estimating coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "clicks": {
                        "type": "integer",
                        "enum": [1, 2, 3],
                        "description": "1=single click, 2=double click (to OPEN), 3=triple click",
                        "default": 1
                    },
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "description": "Mouse button to click",
                        "default": "left"
                    },
                    "element_description": {
                        "type": "string",
                        "description": "What the user is clicking on"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files on the computer by name, type, or content. Use this when the user wants to find files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query - file name, partial name, or keywords"
                    },
                    "file_type": {
                        "type": "string",
                        "description": "Filter by file type/extension (e.g., 'pdf', 'docx', 'py', 'png')",
                        "default": None
                    },
                    "search_path": {
                        "type": "string",
                        "description": "Directory path to search in (defaults to home directory)",
                        "default": None
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a safe terminal command. Only non-destructive commands are allowed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The terminal command to execute (e.g., 'ls', 'pwd', 'date', 'whoami')"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum time in seconds to wait for command completion",
                        "default": 30
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "keyboard_shortcut",
            "description": "Execute a keyboard shortcut combination. Use this for common actions like copy, paste, save, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keys to press together (e.g., ['command', 'c'] for copy, ['command', 'v'] for paste)"
                    }
                },
                "required": ["keys"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll the screen or active window in any direction (up, down, left, right).",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                        "description": "Direction to scroll"
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Amount to scroll (number of scroll units, default: 3)",
                        "default": 3
                    },
                    "x": {
                        "type": "integer",
                        "description": "X coordinate to scroll at (optional, uses current position if not specified)"
                    },
                    "y": {
                        "type": "integer",
                        "description": "Y coordinate to scroll at (optional, uses current position if not specified)"
                    }
                },
                "required": ["direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": "Take a screenshot of the current screen and save it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name for the screenshot file (without extension)",
                        "default": "screenshot"
                    },
                    "region": {
                        "type": "object",
                        "description": "Optional region to capture (x, y, width, height)",
                        "properties": {
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                            "width": {"type": "integer"},
                            "height": {"type": "integer"}
                        }
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Open a URL in the default web browser. For YouTube searches, use 'https://www.youtube.com/results?search_query=your+search+terms'. For Google searches, use 'https://www.google.com/search?q=your+search+terms'. This is the MOST RELIABLE way to search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to open (e.g., 'https://youtube.com', 'https://google.com')"
                    }
                },
                "required": ["url"]
            }
        }
    }
]


# Application name mappings for common variations
APP_NAME_MAPPINGS = {
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "firefox": "Firefox",
    "safari": "Safari",
    "finder": "Finder",
    "notes": "Notes",
    "terminal": "Terminal",
    "iterm": "iTerm",
    "iterm2": "iTerm",
    "vscode": "Visual Studio Code",
    "code": "Visual Studio Code",
    "visual studio code": "Visual Studio Code",
    "sublime": "Sublime Text",
    "sublime text": "Sublime Text",
    "slack": "Slack",
    "discord": "Discord",
    "spotify": "Spotify",
    "mail": "Mail",
    "messages": "Messages",
    "imessage": "Messages",
    "calendar": "Calendar",
    "preview": "Preview",
    "photos": "Photos",
    "music": "Music",
    "itunes": "Music",
    "system preferences": "System Preferences",
    "settings": "System Preferences",
    "word": "Microsoft Word",
    "excel": "Microsoft Excel",
    "powerpoint": "Microsoft PowerPoint",
    "pages": "Pages",
    "numbers": "Numbers",
    "keynote": "Keynote",
    "xcode": "Xcode",
    "cursor": "Cursor",
    "notion": "Notion",
    "obsidian": "Obsidian",
    "zoom": "zoom.us",
    "teams": "Microsoft Teams",
}


# Dangerous commands that should be blocked
BLOCKED_COMMANDS = [
    "rm -rf",
    "rm -r",
    "rmdir",
    "sudo",
    "mkfs",
    "dd if=",
    ":(){ :|:& };:",
    "> /dev/",
    "chmod -R 777",
    "chown -R",
    "mv /*",
    "cp /*",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "killall",
    "pkill -9",
    "fork bomb",
]


# Safe commands whitelist (commands allowed to run)
SAFE_COMMAND_PREFIXES = [
    "ls",
    "pwd",
    "date",
    "whoami",
    "echo",
    "cat",
    "head",
    "tail",
    "grep",
    "find",
    "which",
    "where",
    "wc",
    "sort",
    "uniq",
    "df",
    "du",
    "ps",
    "top -l 1",
    "uptime",
    "hostname",
    "uname",
    "sw_vers",
    "system_profiler",
    "diskutil list",
    "networksetup",
    "ifconfig",
    "curl",
    "wget",
    "open",
    "osascript",
    "say",
    "screencapture",
    "defaults read",
    "python",
    "python3",
    "pip list",
    "pip3 list",
    "npm list",
    "node -v",
    "git status",
    "git log",
    "git branch",
    "git diff",
    "brew list",
    "brew info",
]


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    message: str
    data: Any = None
    error: str | None = None


@dataclass
class SafetyConfig:
    """Safety configuration for tool execution."""
    require_confirmation: bool = False
    blocked_commands: list[str] = field(default_factory=lambda: BLOCKED_COMMANDS.copy())
    safe_command_prefixes: list[str] = field(default_factory=lambda: SAFE_COMMAND_PREFIXES.copy())
    max_command_timeout: int = 60
    enable_logging: bool = True
    dry_run: bool = False


# Button string to MouseButton enum mapping (shared across handlers)
BUTTON_MAP = {
    "left": MouseButton.LEFT,
    "right": MouseButton.RIGHT,
    "middle": MouseButton.MIDDLE
}


class ToolExecutor:
    """Executes tools based on OpenAI function calls."""
    
    def __init__(self, safety_config: SafetyConfig | None = None, screen_capture=None):
        self.logger = get_logger(__name__)
        self.safety = safety_config or SafetyConfig()
        self._pyautogui = None
        self._mouse_controller: MouseController | None = None
        self._smart_clicker: SmartClicker | None = None
        self._screen_capture = screen_capture
        self._tool_handlers: dict[str, Callable] = {
            "open_application": self._handle_open_application,
            "type_text": self._handle_type_text,
            "click_element": self._handle_click_element,
            "click_at_cursor": self._handle_click_at_cursor,
            "move_mouse": self._handle_move_mouse,
            "drag_mouse": self._handle_drag_mouse,
            "get_mouse_position": self._handle_get_mouse_position,
            "search_files": self._handle_search_files,
            "run_command": self._handle_run_command,
            "keyboard_shortcut": self._handle_keyboard_shortcut,
            "scroll": self._handle_scroll,
            "take_screenshot": self._handle_take_screenshot,
            "open_url": self._handle_open_url,
        }
        self._execution_log: list[dict] = []
    
    @property
    def pyautogui(self):
        """Lazy load pyautogui to avoid import errors when not needed."""
        if self._pyautogui is None:
            try:
                import pyautogui
                pyautogui.FAILSAFE = True  # Move mouse to corner to abort
                pyautogui.PAUSE = 0.1  # Small pause between actions
                self._pyautogui = pyautogui
            except ImportError:
                self.logger.error("pyautogui not installed. Run: pip install pyautogui")
                raise
        return self._pyautogui
    
    @property
    def mouse(self) -> MouseController:
        """Get the mouse controller instance."""
        if self._mouse_controller is None:
            self._mouse_controller = get_mouse_controller()
        return self._mouse_controller
    
    @property
    def smart_clicker(self) -> SmartClicker:
        """Get the smart clicker instance with screen capture for verification."""
        if self._smart_clicker is None:
            self._smart_clicker = get_smart_clicker(self._screen_capture)
        elif self._screen_capture is not None:
            self._smart_clicker.screen_capture = self._screen_capture
        return self._smart_clicker
    
    def set_screen_capture(self, screen_capture) -> None:
        """Set the screen capture instance for click verification."""
        self._screen_capture = screen_capture
        if self._smart_clicker is not None:
            self._smart_clicker.screen_capture = screen_capture
    
    @staticmethod
    def get_tool_schemas() -> list[dict]:
        """Get the OpenAI function schemas for all available tools."""
        return TOOL_SCHEMAS.copy()
    
    def execute(self, function_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with given arguments."""
        self.logger.info(f"🔧 Executing tool: {function_name}")
        self.logger.debug(f"   Arguments: {arguments}")
        
        if function_name not in self._tool_handlers:
            return ToolResult(
                success=False,
                message=f"Unknown tool: {function_name}",
                error=f"Tool '{function_name}' is not registered"
            )
        
        # Log execution attempt
        log_entry = {
            "timestamp": time.time(),
            "tool": function_name,
            "arguments": arguments,
            "dry_run": self.safety.dry_run
        }
        
        if self.safety.dry_run:
            self.logger.info(f"🔒 DRY RUN: Would execute {function_name} with {arguments}")
            log_entry["result"] = "dry_run"
            self._execution_log.append(log_entry)
            return ToolResult(
                success=True,
                message=f"[DRY RUN] Would execute {function_name}",
                data={"would_execute": function_name, "arguments": arguments}
            )
        
        try:
            handler = self._tool_handlers[function_name]
            result = handler(**arguments)
            log_entry["result"] = "success" if result.success else "failure"
            log_entry["message"] = result.message
            self._execution_log.append(log_entry)
            return result
        except Exception as e:
            self.logger.error(f"❌ Tool execution failed: {e}")
            log_entry["result"] = "error"
            log_entry["error"] = str(e)
            self._execution_log.append(log_entry)
            return ToolResult(
                success=False,
                message=f"Tool execution failed: {e}",
                error=str(e)
            )
    
    def execute_from_response(self, tool_calls: list) -> list[dict]:
        """Execute tools from OpenAI response tool_calls."""
        results = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                results.append({
                    "tool_call_id": tool_call.id,
                    "output": json.dumps({
                        "success": False,
                        "error": f"Invalid JSON arguments: {e}"
                    })
                })
                continue
            
            result = self.execute(function_name, arguments)
            results.append({
                "tool_call_id": tool_call.id,
                "output": json.dumps({
                    "success": result.success,
                    "message": result.message,
                    "data": result.data,
                    "error": result.error
                })
            })
        
        return results
    
    # Tool Handler Implementations
    
    def _handle_open_application(self, name: str) -> ToolResult:
        """Open a macOS application by name."""
        # Normalize app name
        app_name = APP_NAME_MAPPINGS.get(name.lower(), name)
        
        try:
            # Use macOS 'open' command
            result = subprocess.run(
                ["open", "-a", app_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ Opened application: {app_name}")
                return ToolResult(
                    success=True,
                    message=f"Successfully opened {app_name}",
                    data={"app_name": app_name}
                )
            else:
                error_msg = result.stderr.strip() or "Application not found"
                self.logger.warning(f"⚠️ Failed to open {app_name}: {error_msg}")
                return ToolResult(
                    success=False,
                    message=f"Could not open {app_name}: {error_msg}",
                    error=error_msg
                )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                message=f"Timeout opening {app_name}",
                error="Command timed out"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error opening {app_name}: {e}",
                error=str(e)
            )
    
    def _handle_type_text(self, text: str, press_enter: bool = False) -> ToolResult:
        """Type text into the active/focused element. Assumes the correct element is already focused (clicked on)."""
        try:
            time.sleep(0.3)
            
            # Use osascript for reliable Unicode text entry on macOS
            escaped = text.replace('\\', '\\\\').replace('"', '\\"')
            subprocess.run(
                ["osascript", "-e", f'tell application "System Events" to keystroke "{escaped}"'],
                capture_output=True, text=True, timeout=10
            )
            
            if press_enter:
                time.sleep(0.1)
                self.pyautogui.press('return')
            
            self.logger.info(f"✅ Typed text ({len(text)} chars)")
            return ToolResult(
                success=True,
                message=f"Successfully typed text ({len(text)} characters)",
                data={"text_length": len(text), "pressed_enter": press_enter}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error typing text: {e}",
                error=str(e)
            )
    
    def _handle_click_element(
        self, 
        x: int, 
        y: int, 
        button: str = "left",
        clicks: int = 1,
        element_description: str = "",
        verify: bool = True
    ) -> ToolResult:
        """
        Click at specified screen coordinates with smart verification.
        
        Uses the SmartClicker for human-like clicking with optional
        verification that the click had an effect.
        """
        try:
            # Map button string to MouseButton enum
            mouse_button = BUTTON_MAP.get(button.lower(), MouseButton.LEFT)
            
            # Use smart clicker for verified, human-like clicking
            result = self.smart_clicker.click_at(
                x=x,
                y=y,
                button=mouse_button,
                clicks=clicks,
                description=element_description,
                verify=verify and self._screen_capture is not None,
                max_attempts=2 if verify else 1
            )
            
            if result.success:
                return ToolResult(
                    success=True,
                    message=result.message,
                    data={
                        "x": x, 
                        "y": y, 
                        "button": button,
                        "clicks": clicks,
                        "element": element_description,
                        "verified": result.verified,
                        "screen_changed": result.screen_changed,
                        "attempts": result.attempts
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    message=result.message,
                    error=result.message
                )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error clicking: {e}",
                error=str(e)
            )
    
    def _handle_move_mouse(
        self,
        x: int,
        y: int,
        relative: bool = False,
        movement: str = "smooth",
        duration: float = 0.25
    ) -> ToolResult:
        """Move mouse cursor to position without clicking."""
        try:
            # Map movement style
            style_map = {
                "instant": MovementStyle.INSTANT,
                "smooth": MovementStyle.SMOOTH,
                "natural": MovementStyle.NATURAL,
                "linear": MovementStyle.LINEAR,
                "bezier": MovementStyle.BEZIER
            }
            move_style = style_map.get(movement.lower(), MovementStyle.SMOOTH)
            
            if relative:
                final_pos = self.mouse.move_relative(x, y, duration, move_style)
                self.logger.info(f"✅ Moved mouse by ({x}, {y}) relative")
            else:
                final_pos = self.mouse.move_to(x, y, duration, move_style)
                self.logger.info(f"✅ Moved mouse to ({x}, {y})")
            
            return ToolResult(
                success=True,
                message=f"Moved mouse to ({final_pos.x}, {final_pos.y})",
                data={
                    "x": final_pos.x,
                    "y": final_pos.y,
                    "movement": movement,
                    "relative": relative
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error moving mouse: {e}",
                error=str(e)
            )
    
    def _handle_drag_mouse(
        self,
        end_x: int,
        end_y: int,
        start_x: int | None = None,
        start_y: int | None = None,
        button: str = "left",
        duration: float = 0.5
    ) -> ToolResult:
        """Drag mouse from one position to another."""
        try:
            mouse_button = BUTTON_MAP.get(button.lower(), MouseButton.LEFT)
            
            result = self.mouse.drag_to(
                end_x=end_x,
                end_y=end_y,
                start_x=start_x,
                start_y=start_y,
                button=mouse_button,
                duration=duration
            )
            
            if result.success:
                return ToolResult(
                    success=True,
                    message=result.message,
                    data={
                        "end_x": end_x,
                        "end_y": end_y,
                        "start_x": start_x,
                        "start_y": start_y,
                        "button": button
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    message=result.message,
                    error=result.message
                )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error dragging: {e}",
                error=str(e)
            )
    
    def _handle_get_mouse_position(self) -> ToolResult:
        """Get current mouse cursor position."""
        try:
            pos = self.mouse.get_position()
            screen_info = self.mouse.get_screen_info()
            
            return ToolResult(
                success=True,
                message=f"Mouse is at ({pos.x}, {pos.y})",
                data={
                    "x": pos.x,
                    "y": pos.y,
                    "screen_width": screen_info["width"],
                    "screen_height": screen_info["height"]
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error getting mouse position: {e}",
                error=str(e)
            )
    
    def _handle_click_at_cursor(
        self,
        clicks: int = 1,
        button: str = "left",
        element_description: str = ""
    ) -> ToolResult:
        """
        Click at the current mouse cursor position.
        
        This is more accurate than estimating coordinates when the user
        has manually positioned their cursor on a target.
        """
        try:
            # Get current position
            pos = self.mouse.get_position()
            
            # Map button string to MouseButton enum
            mouse_button = BUTTON_MAP.get(button.lower(), MouseButton.LEFT)
            
            # Use smart clicker at current position (no movement needed)
            result = self.smart_clicker.click_at(
                x=pos.x,
                y=pos.y,
                button=mouse_button,
                clicks=clicks,
                description=element_description or "current cursor position",
                verify=self._screen_capture is not None,
                max_attempts=1  # No retries - user positioned it
            )
            
            click_type = {1: "single", 2: "double", 3: "triple"}.get(clicks, "")
            
            if result.success:
                return ToolResult(
                    success=True,
                    message=f"{click_type.capitalize()} clicked at cursor position ({pos.x}, {pos.y})" + 
                            (f" on {element_description}" if element_description else ""),
                    data={
                        "x": pos.x,
                        "y": pos.y,
                        "button": button,
                        "clicks": clicks,
                        "element": element_description,
                        "screen_changed": result.screen_changed
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    message=result.message,
                    error=result.message
                )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error clicking at cursor: {e}",
                error=str(e)
            )
    
    def _handle_search_files(
        self, 
        query: str, 
        file_type: str | None = None,
        search_path: str | None = None,
        max_results: int = 10
    ) -> ToolResult:
        """Search for files using macOS mdfind (Spotlight)."""
        try:
            # Build mdfind command
            search_dir = search_path or os.path.expanduser("~")
            
            # Build query for mdfind
            mdfind_query = f"kMDItemDisplayName == '*{query}*'wc"
            if file_type:
                mdfind_query += f" && kMDItemFSName == '*.{file_type}'wc"
            
            result = subprocess.run(
                ["mdfind", "-onlyin", search_dir, mdfind_query],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            files = result.stdout.strip().split('\n')[:max_results]
            files = [f for f in files if f]  # Remove empty strings
            
            if files:
                self.logger.info(f"✅ Found {len(files)} files matching '{query}'")
                return ToolResult(
                    success=True,
                    message=f"Found {len(files)} file(s) matching '{query}'",
                    data={"files": files, "count": len(files)}
                )
            else:
                return ToolResult(
                    success=True,
                    message=f"No files found matching '{query}'",
                    data={"files": [], "count": 0}
                )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                message="File search timed out",
                error="Search timeout"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error searching files: {e}",
                error=str(e)
            )
    
    def _handle_run_command(self, command: str, timeout: int = 30) -> ToolResult:
        """Execute a terminal command with safety checks."""
        # Safety check: block dangerous commands
        command_lower = command.lower()
        for blocked in self.safety.blocked_commands:
            if blocked.lower() in command_lower:
                self.logger.warning(f"🚫 Blocked dangerous command: {command}")
                return ToolResult(
                    success=False,
                    message=f"Command blocked for safety: contains '{blocked}'",
                    error="Command blocked by safety filter"
                )
        
        # Safety check: verify command starts with safe prefix
        is_safe = any(command_lower.startswith(safe.lower()) for safe in self.safety.safe_command_prefixes)
        if not is_safe:
            self.logger.warning(f"⚠️ Command not in safe list: {command}")
            return ToolResult(
                success=False,
                message=f"Command '{command.split()[0]}' is not in the allowed commands list",
                error="Command not whitelisted"
            )
        
        # Apply timeout limit
        timeout = min(timeout, self.safety.max_command_timeout)
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.path.expanduser("~")
            )
            
            output = result.stdout.strip()
            if result.returncode == 0:
                self.logger.info(f"✅ Command executed: {command[:50]}...")
                return ToolResult(
                    success=True,
                    message=f"Command executed successfully",
                    data={"output": output, "return_code": result.returncode}
                )
            else:
                error_output = result.stderr.strip() or "Command failed"
                return ToolResult(
                    success=False,
                    message=f"Command failed: {error_output}",
                    error=error_output,
                    data={"output": output, "return_code": result.returncode}
                )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                message=f"Command timed out after {timeout} seconds",
                error="Command timeout"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error executing command: {e}",
                error=str(e)
            )
    
    def _handle_keyboard_shortcut(self, keys: list[str]) -> ToolResult:
        """Execute a keyboard shortcut."""
        try:
            # Normalize key names for macOS
            key_mapping = {
                "cmd": "command",
                "ctrl": "control",
                "opt": "option",
                "alt": "option",
                "win": "command",
            }
            
            normalized_keys = [key_mapping.get(k.lower(), k.lower()) for k in keys]
            
            # Execute hotkey
            self.pyautogui.hotkey(*normalized_keys)
            
            shortcut_str = "+".join(normalized_keys)
            self.logger.info(f"✅ Executed shortcut: {shortcut_str}")
            return ToolResult(
                success=True,
                message=f"Successfully executed keyboard shortcut: {shortcut_str}",
                data={"keys": normalized_keys}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error executing keyboard shortcut: {e}",
                error=str(e)
            )
    
    def _handle_scroll(
        self, 
        direction: str, 
        amount: int = 3,
        x: int | None = None,
        y: int | None = None
    ) -> ToolResult:
        """Scroll the screen in any direction."""
        try:
            direction_lower = direction.lower()
            
            # Use mouse controller for scrolling
            if direction_lower == "up":
                result = self.mouse.scroll_up(amount, x, y)
            elif direction_lower == "down":
                result = self.mouse.scroll_down(amount, x, y)
            elif direction_lower == "left":
                result = self.mouse.scroll_left(amount, x, y)
            elif direction_lower == "right":
                result = self.mouse.scroll_right(amount, x, y)
            else:
                return ToolResult(
                    success=False,
                    message=f"Invalid scroll direction: {direction}",
                    error="Direction must be up, down, left, or right"
                )
            
            if result.success:
                return ToolResult(
                    success=True,
                    message=f"Successfully scrolled {direction}",
                    data={"direction": direction, "amount": amount, "x": x, "y": y}
                )
            else:
                return ToolResult(
                    success=False,
                    message=result.message,
                    error=result.message
                )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error scrolling: {e}",
                error=str(e)
            )
    
    def _handle_take_screenshot(
        self, 
        filename: str = "screenshot",
        region: dict | None = None
    ) -> ToolResult:
        """Take a screenshot."""
        try:
            # Create screenshots directory
            screenshots_dir = os.path.expanduser("~/Desktop/Screenshots")
            os.makedirs(screenshots_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(screenshots_dir, f"{filename}_{timestamp}.png")
            
            if region:
                screenshot = self.pyautogui.screenshot(
                    region=(region["x"], region["y"], region["width"], region["height"])
                )
            else:
                screenshot = self.pyautogui.screenshot()
            
            screenshot.save(filepath)
            
            self.logger.info(f"✅ Screenshot saved: {filepath}")
            return ToolResult(
                success=True,
                message=f"Screenshot saved to {filepath}",
                data={"filepath": filepath}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error taking screenshot: {e}",
                error=str(e)
            )
    
    def _handle_open_url(self, url: str) -> ToolResult:
        """Open a URL in the default browser."""
        try:
            # Ensure URL has protocol
            if not url.startswith(("http://", "https://")):
                url = f"https://{url}"
            
            result = subprocess.run(
                ["open", url],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ Opened URL: {url}")
                return ToolResult(
                    success=True,
                    message=f"Successfully opened {url}",
                    data={"url": url}
                )
            else:
                return ToolResult(
                    success=False,
                    message=f"Failed to open URL: {result.stderr}",
                    error=result.stderr
                )
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error opening URL: {e}",
                error=str(e)
            )
    
    def get_execution_log(self) -> list[dict]:
        """Get the execution history log."""
        return self._execution_log.copy()
    
    def clear_execution_log(self) -> None:
        """Clear the execution history log."""
        self._execution_log.clear()
