"""On-Screen Status Indicator - Small always-on-top UI showing assistant state.

This module provides a lightweight, non-blocking visual indicator that shows
when the voice assistant is active. It runs as a subprocess to properly
handle macOS GUI requirements.

Features:
- Small, unobtrusive window at screen edge (default: top-right)
- Always-on-top positioning (floats above ALL windows including fullscreen)
- Status updates (Live, Listening, Thinking, etc.)
- Pulsing status dot for visual feedback
- Native macOS implementation using Cocoa/AppKit
- Clean startup/shutdown without affecting main loop

Configuration:
- ASSISTANT_INDICATOR_ENABLED=true|false (default: true)
- ASSISTANT_INDICATOR_POSITION=top-right|top-left|bottom-right|bottom-left (default: top-right)

Usage:
    from assistant_app.io.indicator import start_indicator, update_status, stop_indicator
    
    handle = start_indicator("Live")
    update_status("Listening")
    stop_indicator()
"""

import os
import sys
import subprocess
import threading
import time
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from assistant_app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Configuration defaults
INDICATOR_ENABLED_VAR = "ASSISTANT_INDICATOR_ENABLED"
INDICATOR_POSITION_VAR = "ASSISTANT_INDICATOR_POSITION"
DEFAULT_ENABLED = True
DEFAULT_POSITION = "top-right"


@dataclass
class IndicatorConfig:
    """Configuration for the status indicator."""
    enabled: bool = True
    position: str = "top-right"  # top-right, top-left, bottom-right, bottom-left


@dataclass 
class IndicatorHandle:
    """Handle for controlling the indicator subprocess."""
    is_running: bool = False
    _process: Optional[subprocess.Popen] = None
    _status_file: Optional[Path] = None


# Global indicator handle (singleton pattern for simplicity)
_indicator_handle: Optional[IndicatorHandle] = None
_lock = threading.Lock()


def _get_config() -> IndicatorConfig:
    """Get indicator configuration from environment."""
    config = IndicatorConfig()
    
    # Check enabled flag
    enabled_str = os.environ.get(INDICATOR_ENABLED_VAR, "").lower()
    if enabled_str in ("false", "0", "no", "off"):
        config.enabled = False
    elif enabled_str in ("true", "1", "yes", "on"):
        config.enabled = True
    
    # Check position
    position = os.environ.get(INDICATOR_POSITION_VAR, DEFAULT_POSITION).lower()
    if position in ("top-right", "top-left", "bottom-right", "bottom-left"):
        config.position = position
    
    return config


# Inline subprocess script for the indicator window
_INDICATOR_SCRIPT = '''
"""Status Indicator Subprocess - Native macOS floating window."""
import sys
import os
import time
import math

def main():
    position = sys.argv[1] if len(sys.argv) > 1 else "top-right"
    initial_status = sys.argv[2] if len(sys.argv) > 2 else "Live"
    status_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        from AppKit import (
            NSApplication, NSWindow, NSView, NSColor, NSFont,
            NSBackingStoreBuffered, NSWindowStyleMaskBorderless,
            NSStatusWindowLevel, NSScreen, NSMakeRect, NSBezierPath,
            NSMutableParagraphStyle, NSFontAttributeName, 
            NSForegroundColorAttributeName, NSParagraphStyleAttributeName,
            NSAttributedString, NSRunLoop, NSDate
        )
        from Foundation import NSPoint
        import objc
    except ImportError:
        print("PyObjC not available", file=sys.stderr)
        sys.exit(1)

    # Window dimensions
    WIDTH, HEIGHT = 150, 40
    PADDING = 15
    CORNER_RADIUS = 12.0
    
    # Colors (RGB 0-1)
    COLORS = {
        "bg": (0.1, 0.1, 0.18, 0.95),
        "listening": (0.0, 1.0, 0.53),
        "thinking": (1.0, 0.67, 0.0),
        "idle": (0.53, 0.53, 0.53),
        "text": (1.0, 1.0, 1.0),
    }
    
    def get_status_color(status):
        s = status.lower()
        if s in ("listening", "live", "ready", "active"):
            return COLORS["listening"]
        elif s in ("thinking", "processing", "analyzing"):
            return COLORS["thinking"]
        return COLORS["idle"]

    # Create custom view class using objc
    class StatusView(NSView):
        def initWithFrame_(self, frame):
            self = objc.super(StatusView, self).initWithFrame_(frame)
            if self:
                self._status = initial_status
                self._pulse_time = 0.0
            return self
        
        def setStatus_(self, status):
            self._status = status
            self.setNeedsDisplay_(True)
        
        def getStatus(self):
            return self._status
        
        def updatePulse(self):
            self._pulse_time += 0.05
            self.setNeedsDisplay_(True)
        
        def drawRect_(self, rect):
            # Background
            bg = NSColor.colorWithCalibratedRed_green_blue_alpha_(*COLORS["bg"])
            bg.setFill()
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                rect, CORNER_RADIUS, CORNER_RADIUS
            )
            path.fill()
            
            # Pulsing dot
            pulse = 0.7 + 0.3 * abs(math.sin(self._pulse_time * 2))
            dot_color = get_status_color(self._status)
            pulsed = NSColor.colorWithCalibratedRed_green_blue_alpha_(
                min(1.0, dot_color[0] * pulse + 0.2 * (1 - pulse)),
                min(1.0, dot_color[1] * pulse + 0.2 * (1 - pulse)),
                min(1.0, dot_color[2] * pulse + 0.2 * (1 - pulse)),
                1.0
            )
            pulsed.setFill()
            dot_rect = NSMakeRect(12, rect.size.height / 2 - 5, 10, 10)
            NSBezierPath.bezierPathWithOvalInRect_(dot_rect).fill()
            
            # Text
            text = f"🤖 {self._status}"
            font = NSFont.boldSystemFontOfSize_(13)
            text_color = NSColor.colorWithCalibratedRed_green_blue_alpha_(*COLORS["text"], 1.0)
            para = NSMutableParagraphStyle.alloc().init()
            para.setAlignment_(0)
            attrs = {
                NSFontAttributeName: font,
                NSForegroundColorAttributeName: text_color,
                NSParagraphStyleAttributeName: para
            }
            attr_str = NSAttributedString.alloc().initWithString_attributes_(text, attrs)
            text_rect = NSMakeRect(28, rect.size.height / 2 - 8, rect.size.width - 35, 20)
            attr_str.drawInRect_(text_rect)
        
        def mouseDown_(self, event):
            self._drag_start = event.locationInWindow()
        
        def mouseDragged_(self, event):
            win = self.window()
            curr = event.locationInWindow()
            frame = win.frame()
            dx = curr.x - self._drag_start.x
            dy = curr.y - self._drag_start.y
            win.setFrameOrigin_(NSPoint(frame.origin.x + dx, frame.origin.y + dy))

    # Initialize app
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(1)  # Accessory app (no dock icon)
    
    # Calculate position
    screen = NSScreen.mainScreen()
    sf = screen.visibleFrame()
    
    if "right" in position:
        x = sf.origin.x + sf.size.width - WIDTH - PADDING
    else:
        x = sf.origin.x + PADDING
    
    if "top" in position:
        y = sf.origin.y + sf.size.height - HEIGHT - PADDING
    else:
        y = sf.origin.y + PADDING + 50

    # Create window
    rect = NSMakeRect(x, y, WIDTH, HEIGHT)
    window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        rect, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
    )
    window.setLevel_(NSStatusWindowLevel + 1)
    window.setOpaque_(False)
    window.setBackgroundColor_(NSColor.clearColor())
    window.setHasShadow_(True)
    window.setMovableByWindowBackground_(True)
    window.setCollectionBehavior_(1 << 0 | 1 << 4)
    
    view = StatusView.alloc().initWithFrame_(rect)
    window.setContentView_(view)
    window.orderFrontRegardless()
    
    # Run loop with status file checking
    last_status = initial_status
    try:
        while True:
            # Process events
            event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                0xFFFFFFFF, NSDate.dateWithTimeIntervalSinceNow_(0.05),
                "kCFRunLoopDefaultMode", True
            )
            if event:
                app.sendEvent_(event)
            
            # Update animation
            view.updatePulse()
            
            # Check status file for updates
            if status_file and os.path.exists(status_file):
                try:
                    with open(status_file, "r") as f:
                        content = f.read().strip()
                        if content == "STOP":
                            break
                        elif content and content != last_status:
                            view.setStatus_(content)
                            last_status = content
                except:
                    pass
            
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        window.close()

if __name__ == "__main__":
    main()
'''


def start_indicator(initial_status: str = "Live") -> Optional[IndicatorHandle]:
    """Start the on-screen status indicator.
    
    Args:
        initial_status: Initial status text to display (default: "Live")
    
    Returns:
        IndicatorHandle if started successfully, None if disabled or failed
    
    Example:
        handle = start_indicator("Live")
        if handle:
            update_status("Listening")
            # ... later ...
            stop_indicator()
    """
    global _indicator_handle
    
    config = _get_config()
    
    if not config.enabled:
        logger.info("📍 Status indicator disabled via config")
        return None
    
    with _lock:
        # Stop existing indicator if any
        if _indicator_handle is not None and _indicator_handle.is_running:
            stop_indicator()
        
        # Create new handle
        handle = IndicatorHandle()
        _indicator_handle = handle
        
        try:
            # Create temp file for IPC
            import tempfile
            status_file = Path(tempfile.mktemp(suffix="_indicator_status.txt"))
            status_file.write_text(initial_status)
            handle._status_file = status_file
            
            # Write script to temp file
            script_file = Path(tempfile.mktemp(suffix="_indicator.py"))
            script_file.write_text(_INDICATOR_SCRIPT)
            
            # Start subprocess (same process group so it dies with parent)
            process = subprocess.Popen(
                [sys.executable, str(script_file), config.position, initial_status, str(status_file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            handle._process = process
            handle._script_file = script_file
            
            # Give it a moment to start
            time.sleep(0.3)
            
            # Check if process is running
            if process.poll() is None:
                handle.is_running = True
                logger.info(f"📍 Status indicator started ({config.position})")
            else:
                # Process exited, read stderr
                stderr = process.stderr.read().decode() if process.stderr else ""
                logger.warning(f"⚠️ Indicator failed to start: {stderr}")
                _cleanup_files(handle)
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to start indicator: {e}")
            return None
    
    return handle


def _cleanup_files(handle: IndicatorHandle) -> None:
    """Clean up temporary files."""
    try:
        if handle._status_file and handle._status_file.exists():
            handle._status_file.unlink()
    except Exception:
        pass
    try:
        if hasattr(handle, '_script_file') and handle._script_file.exists():
            handle._script_file.unlink()
    except Exception:
        pass


def update_status(status: str) -> None:
    """Update the status text displayed in the indicator.
    
    Args:
        status: New status text (e.g., "Listening", "Thinking", "Live")
    
    Example:
        update_status("Listening")  # Shows "🤖 Listening" with green dot
        update_status("Thinking")   # Shows "🤖 Thinking" with orange dot
    """
    global _indicator_handle
    
    with _lock:
        if _indicator_handle is not None and _indicator_handle.is_running:
            try:
                if _indicator_handle._status_file:
                    _indicator_handle._status_file.write_text(status)
            except Exception:
                pass


def stop_indicator() -> None:
    """Stop and clean up the status indicator.
    
    Safe to call even if indicator wasn't started or already stopped.
    
    Example:
        stop_indicator()  # Cleans up the indicator window
    """
    global _indicator_handle
    
    with _lock:
        if _indicator_handle is not None:
            handle = _indicator_handle
            _indicator_handle = None
            
            # Kill the process immediately
            if handle._process and handle._process.poll() is None:
                try:
                    handle._process.kill()
                    handle._process.wait(timeout=0.5)
                except Exception:
                    pass
            
            # Clean up files
            _cleanup_files(handle)
            
            handle.is_running = False
            logger.info("📍 Status indicator stopped")


def is_indicator_running() -> bool:
    """Check if the indicator is currently running.
    
    Returns:
        True if indicator is running and visible, False otherwise
    """
    with _lock:
        return _indicator_handle is not None and _indicator_handle.is_running


# Convenience context manager for automatic cleanup
class IndicatorContext:
    """Context manager for automatic indicator lifecycle management.
    
    Example:
        with IndicatorContext("Live") as indicator:
            # Do assistant work...
            indicator.update("Listening")
        # Indicator automatically stopped on exit
    """
    
    def __init__(self, initial_status: str = "Live"):
        self.initial_status = initial_status
        self.handle = None
    
    def __enter__(self) -> "IndicatorContext":
        self.handle = start_indicator(self.initial_status)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        stop_indicator()
        return False
    
    def update(self, status: str) -> None:
        """Update the indicator status."""
        update_status(status)
