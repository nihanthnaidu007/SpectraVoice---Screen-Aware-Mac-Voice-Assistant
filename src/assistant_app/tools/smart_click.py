"""Smart Click System - Accurate visual targeting with verification."""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from assistant_app.tools.mouse_controller import (
    get_mouse_controller, 
    MouseController,
    MouseButton, 
    MovementStyle,
    ClickResult
)
from assistant_app.utils.logging_config import get_logger

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class ClickTarget:
    """Represents a target to click on."""
    description: str
    x: int
    y: int
    confidence: float = 0.0
    width: int = 50  # Estimated target width
    height: int = 50  # Estimated target height
    
    @property
    def center(self) -> tuple[int, int]:
        """Get center point of target."""
        return (self.x, self.y)
    
    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Get bounding box (x1, y1, x2, y2)."""
        half_w = self.width // 2
        half_h = self.height // 2
        return (
            self.x - half_w,
            self.y - half_h,
            self.x + half_w,
            self.y + half_h
        )


@dataclass  
class SmartClickResult:
    """Result of a smart click operation."""
    success: bool
    message: str
    clicked_at: tuple[int, int] | None = None
    attempts: int = 1
    verified: bool = False
    screen_changed: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for tool result data."""
        return {
            "success": self.success,
            "clicked_at": self.clicked_at,
            "attempts": self.attempts,
            "verified": self.verified,
            "screen_changed": self.screen_changed
        }


class SmartClicker:
    """
    Human-like clicking with visual verification.
    
    Strategy:
    1. Move to target area with smooth motion
    2. Pause briefly (like a human would)
    3. Click
    4. Verify the action had an effect (screen changed)
    5. Retry with adjusted coordinates if needed
    """
    
    # Timing settings (human-like behavior)
    PRE_CLICK_PAUSE = 0.08       # Brief pause before clicking
    POST_CLICK_PAUSE = 0.25      # Wait for UI to respond
    MOVE_DURATION = 0.2          # Mouse movement duration
    VERIFICATION_WAIT = 0.5      # Extra wait for verification screenshot
    
    # Retry settings
    DEFAULT_MAX_ATTEMPTS = 2
    RETRY_OFFSET = 5             # Pixels to adjust on retry
    
    def __init__(self, screen_capture=None):
        """
        Initialize SmartClicker.
        
        Args:
            screen_capture: Optional ScreenCapture instance for verification
        """
        self.logger = get_logger(__name__)
        self._mouse: MouseController | None = None
        self._screen_capture = screen_capture
    
    @property
    def mouse(self) -> MouseController:
        """Lazy load mouse controller."""
        if self._mouse is None:
            self._mouse = get_mouse_controller()
        return self._mouse
    
    @property
    def screen_capture(self):
        """Get screen capture instance."""
        return self._screen_capture
    
    @screen_capture.setter
    def screen_capture(self, value):
        """Set screen capture instance."""
        self._screen_capture = value
    
    def click_at(
        self,
        x: int,
        y: int,
        button: MouseButton = MouseButton.LEFT,
        clicks: int = 1,
        description: str = "",
        verify: bool = False,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS
    ) -> SmartClickResult:
        """
        Click at coordinates with optional verification.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            button: Mouse button to use
            clicks: Number of clicks (1=single, 2=double, 3=triple)
            description: What we're trying to click (for logging)
            verify: Whether to verify click had an effect
            max_attempts: Max retry attempts if verification fails
        
        Returns:
            SmartClickResult with details of the operation
        """
        original_x, original_y = x, y
        attempts = 0
        last_error = ""
        
        while attempts < max_attempts:
            attempts += 1
            
            try:
                # 1. Capture initial state for verification
                initial_screenshot = None
                if verify and self.screen_capture:
                    try:
                        initial_screenshot = self.screen_capture.capture()
                    except Exception as e:
                        self.logger.debug(f"Could not capture initial screenshot: {e}")
                
                # 2. Move to target with smooth, human-like motion
                target_desc = f"'{description}'" if description else f"({x}, {y})"
                self.logger.info(f"🎯 Moving to {target_desc}")
                
                self.mouse.move_to(
                    x, y, 
                    duration=self.MOVE_DURATION, 
                    style=MovementStyle.SMOOTH
                )
                
                # 3. Brief pause (human-like hesitation before clicking)
                time.sleep(self.PRE_CLICK_PAUSE)
                
                # 4. Perform the click
                click_result = self.mouse.click(
                    button=button,
                    clicks=clicks
                )
                
                if not click_result.success:
                    last_error = click_result.message
                    self.logger.warning(f"⚠️ Click failed: {last_error}")
                    continue
                
                # 5. Wait for UI to respond
                time.sleep(self.POST_CLICK_PAUSE)
                
                # 6. Verify if requested
                screen_changed = False
                if verify and self.screen_capture and initial_screenshot:
                    time.sleep(self.VERIFICATION_WAIT - self.POST_CLICK_PAUSE)
                    screen_changed = self._verify_screen_changed(initial_screenshot)
                    
                    if not screen_changed and attempts < max_attempts:
                        # Screen didn't change - might have missed target
                        self.logger.warning(
                            f"⚠️ Screen unchanged after click (attempt {attempts}/{max_attempts})"
                        )
                        # Adjust coordinates slightly for retry
                        x = original_x + (self.RETRY_OFFSET * attempts)
                        y = original_y + (self.RETRY_OFFSET * attempts)
                        continue
                
                # Success!
                click_type = {1: "clicked", 2: "double-clicked", 3: "triple-clicked"}.get(clicks, "clicked")
                button_name = button.value if button != MouseButton.LEFT else ""
                
                msg_parts = [click_type.capitalize()]
                if button_name:
                    msg_parts.insert(0, button_name)
                msg_parts.append(f"at ({original_x}, {original_y})")
                if description:
                    msg_parts.append(f"on {description}")
                
                message = " ".join(msg_parts)
                
                if verify:
                    if screen_changed:
                        message += " - screen changed (action likely succeeded)"
                    else:
                        message += " - verify the intended action occurred"
                
                self.logger.info(f"✅ {message}")
                
                return SmartClickResult(
                    success=True,
                    message=message,
                    clicked_at=(original_x, original_y),
                    attempts=attempts,
                    verified=verify,
                    screen_changed=screen_changed
                )
                
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"❌ Click error: {e}")
                if attempts >= max_attempts:
                    break
        
        # All attempts failed
        return SmartClickResult(
            success=False,
            message=f"Failed to click after {attempts} attempts: {last_error}",
            clicked_at=None,
            attempts=attempts,
            verified=False
        )
    
    def _verify_screen_changed(self, initial_screenshot: 'Image.Image') -> bool:
        """
        Verify that the screen changed after clicking.
        
        Args:
            initial_screenshot: Screenshot taken before clicking
        
        Returns:
            True if screen appears to have changed
        """
        if not self.screen_capture:
            return True  # Can't verify, assume success
        
        try:
            # Capture current screen
            current_screenshot = self.screen_capture.capture()
            
            if current_screenshot is None:
                return True  # Can't capture, assume success
            
            # Compare screenshots
            return self._images_differ(initial_screenshot, current_screenshot)
            
        except Exception as e:
            self.logger.debug(f"Verification failed: {e}")
            return True  # On error, assume success
    
    def _images_differ(self, img1: 'Image.Image', img2: 'Image.Image') -> bool:
        """
        Check if two images are different.
        
        Uses a simple but effective comparison that detects
        most UI changes (windows opening, selections, etc.)
        """
        try:
            # Quick size check
            if img1.size != img2.size:
                return True
            
            # Convert to same mode for comparison
            if img1.mode != img2.mode:
                img2 = img2.convert(img1.mode)
            
            # Compare raw bytes (fast)
            bytes1 = img1.tobytes()
            bytes2 = img2.tobytes()
            
            if bytes1 == bytes2:
                return False
            
            # Images differ - check if difference is significant
            # (ignore tiny changes like cursor blink)
            diff_count = sum(1 for a, b in zip(bytes1, bytes2) if a != b)
            total_bytes = len(bytes1)
            diff_ratio = diff_count / total_bytes
            
            # Consider changed if more than 0.1% of pixels differ
            significant_change = diff_ratio > 0.001
            
            if significant_change:
                self.logger.debug(f"Screen changed: {diff_ratio:.2%} pixels different")
            
            return significant_change
            
        except Exception as e:
            self.logger.debug(f"Image comparison failed: {e}")
            return True  # Assume changed on error
    
    def double_click_to_open(
        self,
        x: int,
        y: int,
        item_name: str = "",
        verify: bool = True
    ) -> SmartClickResult:
        """
        Double-click to open an item (like a desktop icon or file).
        
        Args:
            x: Target X coordinate (center of icon)
            y: Target Y coordinate (center of icon)
            item_name: Name of item being opened (for logging)
            verify: Whether to verify the action succeeded
        
        Returns:
            SmartClickResult
        """
        description = f"'{item_name}'" if item_name else "item"
        self.logger.info(f"📂 Double-clicking to open {description}")
        
        return self.click_at(
            x=x,
            y=y,
            button=MouseButton.LEFT,
            clicks=2,
            description=item_name or "item",
            verify=verify,
            max_attempts=2
        )
    
    def right_click_context_menu(
        self,
        x: int,
        y: int,
        item_name: str = ""
    ) -> SmartClickResult:
        """
        Right-click to open context menu.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            item_name: Name of item (for logging)
        
        Returns:
            SmartClickResult
        """
        description = f"'{item_name}'" if item_name else "item"
        self.logger.info(f"📋 Right-clicking {description} for context menu")
        
        return self.click_at(
            x=x,
            y=y,
            button=MouseButton.RIGHT,
            clicks=1,
            description=item_name or "context menu",
            verify=True,  # Context menus should cause screen change
            max_attempts=1  # Don't retry right-clicks
        )
    
    def click_button(
        self,
        x: int,
        y: int,
        button_name: str = ""
    ) -> SmartClickResult:
        """
        Click a UI button.
        
        Args:
            x: Button center X coordinate
            y: Button center Y coordinate
            button_name: Name of the button (for logging)
        
        Returns:
            SmartClickResult
        """
        self.logger.info(f"🔘 Clicking button: {button_name or 'button'}")
        
        return self.click_at(
            x=x,
            y=y,
            button=MouseButton.LEFT,
            clicks=1,
            description=button_name or "button",
            verify=True,
            max_attempts=2
        )


# Singleton instance
_smart_clicker: SmartClicker | None = None


def get_smart_clicker(screen_capture=None) -> SmartClicker:
    """
    Get the singleton SmartClicker instance.
    
    Args:
        screen_capture: Optional ScreenCapture to use for verification
    
    Returns:
        SmartClicker instance
    """
    global _smart_clicker
    if _smart_clicker is None:
        _smart_clicker = SmartClicker(screen_capture)
    elif screen_capture is not None:
        _smart_clicker.screen_capture = screen_capture
    return _smart_clicker
