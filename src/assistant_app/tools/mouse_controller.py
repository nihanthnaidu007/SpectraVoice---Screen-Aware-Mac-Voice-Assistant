"""Robust Mouse Controller - Advanced mouse control for macOS automation."""

import time
import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from assistant_app.utils.logging_config import get_logger


class MouseButton(Enum):
    """Mouse button types."""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class MovementStyle(Enum):
    """Mouse movement animation styles."""
    INSTANT = "instant"       # Teleport to position
    LINEAR = "linear"         # Straight line movement
    SMOOTH = "smooth"         # Ease in/out curve
    NATURAL = "natural"       # Human-like with slight curve
    BEZIER = "bezier"         # Bezier curve movement


class ScreenRegion(Enum):
    """Predefined screen regions for relative positioning."""
    CENTER = "center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    TOP_CENTER = "top_center"
    BOTTOM_CENTER = "bottom_center"
    LEFT_CENTER = "left_center"
    RIGHT_CENTER = "right_center"


@dataclass
class MousePosition:
    """Represents a mouse position with optional metadata."""
    x: int
    y: int
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def distance_to(self, other: 'MousePosition') -> float:
        """Calculate distance to another position."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def as_tuple(self) -> tuple[int, int]:
        """Return as (x, y) tuple."""
        return (self.x, self.y)


@dataclass
class ClickResult:
    """Result of a mouse click operation."""
    success: bool
    message: str
    position: MousePosition | None = None
    button: MouseButton = MouseButton.LEFT
    click_count: int = 1


class MouseController:
    """Advanced mouse controller with smooth movements and precise clicking."""
    
    # Default timing configurations
    DEFAULT_MOVE_DURATION = 0.25  # seconds
    DEFAULT_CLICK_DELAY = 0.05    # delay between clicks
    DEFAULT_DOUBLE_CLICK_INTERVAL = 0.1
    MIN_MOVE_DURATION = 0.01
    MAX_MOVE_DURATION = 2.0
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._pyautogui = None
        self._screen_size: tuple[int, int] | None = None
        self._position_history: list[MousePosition] = []
        self._max_history = 50
        
    @property
    def pyautogui(self):
        """Lazy load pyautogui."""
        if self._pyautogui is None:
            try:
                import pyautogui
                pyautogui.FAILSAFE = True
                pyautogui.PAUSE = 0.02  # Minimal pause for responsiveness
                self._pyautogui = pyautogui
            except ImportError:
                self.logger.error("pyautogui not installed")
                raise
        return self._pyautogui
    
    @property
    def screen_size(self) -> tuple[int, int]:
        """Get screen dimensions (width, height)."""
        if self._screen_size is None:
            self._screen_size = self.pyautogui.size()
        return self._screen_size
    
    @property
    def screen_width(self) -> int:
        return self.screen_size[0]
    
    @property
    def screen_height(self) -> int:
        return self.screen_size[1]
    
    # ==================== Position Methods ====================
    
    def get_position(self) -> MousePosition:
        """Get current mouse position."""
        x, y = self.pyautogui.position()
        pos = MousePosition(x=x, y=y)
        self._record_position(pos)
        return pos
    
    def _record_position(self, pos: MousePosition) -> None:
        """Record position in history."""
        self._position_history.append(pos)
        if len(self._position_history) > self._max_history:
            self._position_history.pop(0)
    
    def get_position_history(self) -> list[MousePosition]:
        """Get recent position history."""
        return self._position_history.copy()
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if coordinates are within screen bounds."""
        return 0 <= x <= self.screen_width and 0 <= y <= self.screen_height
    
    def clamp_to_screen(self, x: int, y: int) -> tuple[int, int]:
        """Clamp coordinates to screen bounds."""
        clamped_x = max(0, min(x, self.screen_width - 1))
        clamped_y = max(0, min(y, self.screen_height - 1))
        return (clamped_x, clamped_y)
    
    # ==================== Movement Methods ====================
    
    def move_to(
        self, 
        x: int, 
        y: int, 
        duration: float = DEFAULT_MOVE_DURATION,
        style: MovementStyle = MovementStyle.SMOOTH
    ) -> MousePosition:
        """
        Move mouse to absolute coordinates.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Movement duration in seconds
            style: Movement animation style
        
        Returns:
            Final MousePosition
        """
        # Clamp to screen bounds
        x, y = self.clamp_to_screen(x, y)
        duration = max(self.MIN_MOVE_DURATION, min(duration, self.MAX_MOVE_DURATION))
        
        if style == MovementStyle.INSTANT:
            self.pyautogui.moveTo(x, y, duration=0)
        elif style == MovementStyle.LINEAR:
            self.pyautogui.moveTo(x, y, duration=duration, tween=self.pyautogui.linear)
        elif style == MovementStyle.SMOOTH:
            self.pyautogui.moveTo(x, y, duration=duration, tween=self.pyautogui.easeInOutQuad)
        elif style == MovementStyle.NATURAL:
            self._move_natural(x, y, duration)
        elif style == MovementStyle.BEZIER:
            self._move_bezier(x, y, duration)
        else:
            self.pyautogui.moveTo(x, y, duration=duration)
        
        final_pos = self.get_position()
        self.logger.debug(f"🖱️ Moved to ({x}, {y}) [{style.value}]")
        return final_pos
    
    def move_relative(
        self, 
        dx: int, 
        dy: int, 
        duration: float = DEFAULT_MOVE_DURATION,
        style: MovementStyle = MovementStyle.SMOOTH
    ) -> MousePosition:
        """
        Move mouse relative to current position.
        
        Args:
            dx: Horizontal offset (positive = right, negative = left)
            dy: Vertical offset (positive = down, negative = up)
            duration: Movement duration
            style: Movement style
        
        Returns:
            Final MousePosition
        """
        current = self.get_position()
        new_x = current.x + dx
        new_y = current.y + dy
        return self.move_to(new_x, new_y, duration, style)
    
    def move_to_region(
        self, 
        region: ScreenRegion,
        offset_x: int = 0,
        offset_y: int = 0,
        duration: float = DEFAULT_MOVE_DURATION
    ) -> MousePosition:
        """
        Move mouse to a predefined screen region.
        
        Args:
            region: Target screen region
            offset_x: Additional X offset from region center
            offset_y: Additional Y offset from region center
            duration: Movement duration
        
        Returns:
            Final MousePosition
        """
        x, y = self._get_region_coordinates(region)
        return self.move_to(x + offset_x, y + offset_y, duration)
    
    def _get_region_coordinates(self, region: ScreenRegion) -> tuple[int, int]:
        """Get coordinates for a screen region."""
        w, h = self.screen_width, self.screen_height
        margin = 50  # Margin from edges
        
        regions = {
            ScreenRegion.CENTER: (w // 2, h // 2),
            ScreenRegion.TOP_LEFT: (margin, margin),
            ScreenRegion.TOP_RIGHT: (w - margin, margin),
            ScreenRegion.BOTTOM_LEFT: (margin, h - margin),
            ScreenRegion.BOTTOM_RIGHT: (w - margin, h - margin),
            ScreenRegion.TOP_CENTER: (w // 2, margin),
            ScreenRegion.BOTTOM_CENTER: (w // 2, h - margin),
            ScreenRegion.LEFT_CENTER: (margin, h // 2),
            ScreenRegion.RIGHT_CENTER: (w - margin, h // 2),
        }
        return regions.get(region, (w // 2, h // 2))
    
    def _move_natural(self, x: int, y: int, duration: float) -> None:
        """Move with natural human-like motion (slight curve)."""
        start = self.get_position()
        
        # Add a slight random curve point
        mid_x = (start.x + x) // 2 + int((x - start.x) * 0.1)
        mid_y = (start.y + y) // 2 + int((y - start.y) * 0.15)
        
        # Move through curve point then to target
        self.pyautogui.moveTo(mid_x, mid_y, duration=duration * 0.4, tween=self.pyautogui.easeOutQuad)
        self.pyautogui.moveTo(x, y, duration=duration * 0.6, tween=self.pyautogui.easeInOutQuad)
    
    def _move_bezier(self, x: int, y: int, duration: float) -> None:
        """Move along a bezier curve for smooth motion."""
        start = self.get_position()
        steps = max(10, int(duration * 60))  # 60 steps per second
        
        # Control points for bezier curve
        ctrl1_x = start.x + (x - start.x) * 0.3
        ctrl1_y = start.y
        ctrl2_x = start.x + (x - start.x) * 0.7
        ctrl2_y = y
        
        step_duration = duration / steps
        
        for i in range(steps + 1):
            t = i / steps
            # Cubic bezier formula
            bx = int((1-t)**3 * start.x + 3*(1-t)**2*t * ctrl1_x + 3*(1-t)*t**2 * ctrl2_x + t**3 * x)
            by = int((1-t)**3 * start.y + 3*(1-t)**2*t * ctrl1_y + 3*(1-t)*t**2 * ctrl2_y + t**3 * y)
            self.pyautogui.moveTo(bx, by, duration=0)
            time.sleep(step_duration)
    
    # ==================== Click Methods ====================
    
    def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: MouseButton = MouseButton.LEFT,
        clicks: int = 1,
        interval: float = DEFAULT_DOUBLE_CLICK_INTERVAL,
        move_duration: float = DEFAULT_MOVE_DURATION
    ) -> ClickResult:
        """
        Perform mouse click(s).
        
        Args:
            x: X coordinate (None = current position)
            y: Y coordinate (None = current position)
            button: Which mouse button to click
            clicks: Number of clicks (1=single, 2=double, 3=triple)
            interval: Time between multiple clicks
            move_duration: Time to move to position before clicking
        
        Returns:
            ClickResult with operation details
        """
        try:
            # Move to position if specified
            if x is not None and y is not None:
                x, y = self.clamp_to_screen(x, y)
                self.move_to(x, y, duration=move_duration, style=MovementStyle.SMOOTH)
            
            current_pos = self.get_position()
            
            # Perform clicks
            self.pyautogui.click(
                clicks=clicks,
                interval=interval,
                button=button.value
            )
            
            click_type = {1: "single", 2: "double", 3: "triple"}.get(clicks, f"{clicks}x")
            self.logger.info(f"✅ {click_type.capitalize()} {button.value} click at ({current_pos.x}, {current_pos.y})")
            
            return ClickResult(
                success=True,
                message=f"Successfully performed {click_type} {button.value} click",
                position=current_pos,
                button=button,
                click_count=clicks
            )
        except Exception as e:
            self.logger.error(f"❌ Click failed: {e}")
            return ClickResult(
                success=False,
                message=f"Click failed: {e}",
                button=button,
                click_count=0
            )
    
    def left_click(self, x: int | None = None, y: int | None = None) -> ClickResult:
        """Perform a left click."""
        return self.click(x, y, MouseButton.LEFT, clicks=1)
    
    def right_click(self, x: int | None = None, y: int | None = None) -> ClickResult:
        """Perform a right click (context menu)."""
        return self.click(x, y, MouseButton.RIGHT, clicks=1)
    
    def middle_click(self, x: int | None = None, y: int | None = None) -> ClickResult:
        """Perform a middle click."""
        return self.click(x, y, MouseButton.MIDDLE, clicks=1)
    
    def double_click(self, x: int | None = None, y: int | None = None) -> ClickResult:
        """Perform a double click."""
        return self.click(x, y, MouseButton.LEFT, clicks=2)
    
    def triple_click(self, x: int | None = None, y: int | None = None) -> ClickResult:
        """Perform a triple click (select paragraph/line)."""
        return self.click(x, y, MouseButton.LEFT, clicks=3)
    
    def click_and_hold(
        self, 
        x: int | None = None, 
        y: int | None = None,
        button: MouseButton = MouseButton.LEFT,
        duration: float = 1.0
    ) -> ClickResult:
        """
        Click and hold the mouse button for a duration.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button to hold
            duration: How long to hold in seconds
        
        Returns:
            ClickResult
        """
        try:
            if x is not None and y is not None:
                self.move_to(x, y)
            
            current_pos = self.get_position()
            
            self.pyautogui.mouseDown(button=button.value)
            time.sleep(duration)
            self.pyautogui.mouseUp(button=button.value)
            
            self.logger.info(f"✅ Click and hold ({duration}s) at ({current_pos.x}, {current_pos.y})")
            
            return ClickResult(
                success=True,
                message=f"Held {button.value} button for {duration}s",
                position=current_pos,
                button=button
            )
        except Exception as e:
            self.pyautogui.mouseUp(button=button.value)  # Safety release
            return ClickResult(success=False, message=f"Click and hold failed: {e}")
    
    # ==================== Drag Methods ====================
    
    def drag_to(
        self,
        end_x: int,
        end_y: int,
        start_x: int | None = None,
        start_y: int | None = None,
        button: MouseButton = MouseButton.LEFT,
        duration: float = 0.5
    ) -> ClickResult:
        """
        Drag from current/start position to end position.
        
        Args:
            end_x: Destination X coordinate
            end_y: Destination Y coordinate
            start_x: Start X (None = current position)
            start_y: Start Y (None = current position)
            button: Mouse button to use for dragging
            duration: Drag duration
        
        Returns:
            ClickResult with drag details
        """
        try:
            # Move to start if specified
            if start_x is not None and start_y is not None:
                self.move_to(start_x, start_y)
            
            start_pos = self.get_position()
            end_x, end_y = self.clamp_to_screen(end_x, end_y)
            
            self.pyautogui.drag(
                end_x - start_pos.x,
                end_y - start_pos.y,
                duration=duration,
                button=button.value
            )
            
            end_pos = self.get_position()
            self.logger.info(f"✅ Dragged from ({start_pos.x}, {start_pos.y}) to ({end_pos.x}, {end_pos.y})")
            
            return ClickResult(
                success=True,
                message=f"Dragged from ({start_pos.x}, {start_pos.y}) to ({end_x}, {end_y})",
                position=end_pos,
                button=button
            )
        except Exception as e:
            self.pyautogui.mouseUp()  # Safety release
            return ClickResult(success=False, message=f"Drag failed: {e}")
    
    def drag_relative(
        self,
        dx: int,
        dy: int,
        button: MouseButton = MouseButton.LEFT,
        duration: float = 0.5
    ) -> ClickResult:
        """
        Drag by a relative offset from current position.
        
        Args:
            dx: Horizontal drag distance
            dy: Vertical drag distance
            button: Mouse button to use
            duration: Drag duration
        
        Returns:
            ClickResult
        """
        try:
            start_pos = self.get_position()
            
            self.pyautogui.drag(dx, dy, duration=duration, button=button.value)
            
            end_pos = self.get_position()
            self.logger.info(f"✅ Dragged by ({dx}, {dy})")
            
            return ClickResult(
                success=True,
                message=f"Dragged by offset ({dx}, {dy})",
                position=end_pos,
                button=button
            )
        except Exception as e:
            self.pyautogui.mouseUp()
            return ClickResult(success=False, message=f"Drag failed: {e}")
    
    # ==================== Scroll Methods ====================
    
    def scroll(
        self,
        amount: int,
        x: int | None = None,
        y: int | None = None,
        horizontal: bool = False
    ) -> ClickResult:
        """
        Scroll the mouse wheel.
        
        Args:
            amount: Scroll amount (positive = up/right, negative = down/left)
            x: X coordinate to scroll at (None = current)
            y: Y coordinate to scroll at (None = current)
            horizontal: If True, scroll horizontally
        
        Returns:
            ClickResult
        """
        try:
            if x is not None and y is not None:
                self.move_to(x, y, duration=0.1)
            
            current_pos = self.get_position()
            
            if horizontal:
                self.pyautogui.hscroll(amount)
                direction = "right" if amount > 0 else "left"
            else:
                self.pyautogui.scroll(amount)
                direction = "up" if amount > 0 else "down"
            
            self.logger.info(f"✅ Scrolled {direction} by {abs(amount)}")
            
            return ClickResult(
                success=True,
                message=f"Scrolled {direction} by {abs(amount)}",
                position=current_pos
            )
        except Exception as e:
            return ClickResult(success=False, message=f"Scroll failed: {e}")
    
    def scroll_up(self, amount: int = 3, x: int | None = None, y: int | None = None) -> ClickResult:
        """Scroll up."""
        return self.scroll(abs(amount), x, y)
    
    def scroll_down(self, amount: int = 3, x: int | None = None, y: int | None = None) -> ClickResult:
        """Scroll down."""
        return self.scroll(-abs(amount), x, y)
    
    def scroll_left(self, amount: int = 3, x: int | None = None, y: int | None = None) -> ClickResult:
        """Scroll left (horizontal)."""
        return self.scroll(-abs(amount), x, y, horizontal=True)
    
    def scroll_right(self, amount: int = 3, x: int | None = None, y: int | None = None) -> ClickResult:
        """Scroll right (horizontal)."""
        return self.scroll(abs(amount), x, y, horizontal=True)
    
    # ==================== Utility Methods ====================
    
    def wait(self, seconds: float) -> None:
        """Wait for specified seconds."""
        time.sleep(seconds)
    
    def reset_to_center(self) -> MousePosition:
        """Move mouse to screen center."""
        return self.move_to_region(ScreenRegion.CENTER)
    
    def nudge(self, direction: str, pixels: int = 10) -> MousePosition:
        """
        Nudge mouse in a direction by specified pixels.
        
        Args:
            direction: 'up', 'down', 'left', 'right'
            pixels: Number of pixels to move
        
        Returns:
            New MousePosition
        """
        offsets = {
            "up": (0, -pixels),
            "down": (0, pixels),
            "left": (-pixels, 0),
            "right": (pixels, 0),
        }
        dx, dy = offsets.get(direction.lower(), (0, 0))
        return self.move_relative(dx, dy, duration=0.1, style=MovementStyle.INSTANT)
    
    def circle_motion(
        self,
        center_x: int | None = None,
        center_y: int | None = None,
        radius: int = 50,
        duration: float = 1.0
    ) -> None:
        """
        Move mouse in a circle (useful for highlighting areas).
        
        Args:
            center_x: Center X (None = current position)
            center_y: Center Y (None = current position)
            radius: Circle radius in pixels
            duration: Time to complete circle
        """
        if center_x is None or center_y is None:
            pos = self.get_position()
            center_x = pos.x
            center_y = pos.y
        
        steps = 36  # 10 degrees per step
        step_duration = duration / steps
        
        for i in range(steps + 1):
            angle = (2 * math.pi * i) / steps
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            x, y = self.clamp_to_screen(x, y)
            self.pyautogui.moveTo(x, y, duration=0)
            time.sleep(step_duration)
    
    def get_screen_info(self) -> dict:
        """Get screen information."""
        return {
            "width": self.screen_width,
            "height": self.screen_height,
            "current_position": self.get_position().as_tuple(),
            "history_length": len(self._position_history)
        }


# Singleton instance for easy access
_mouse_controller: MouseController | None = None


def get_mouse_controller() -> MouseController:
    """Get the singleton mouse controller instance."""
    global _mouse_controller
    if _mouse_controller is None:
        _mouse_controller = MouseController()
    return _mouse_controller
