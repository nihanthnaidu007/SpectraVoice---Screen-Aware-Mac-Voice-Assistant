"""Screen Capture Service - Threaded capture with caching."""

import base64
import time
from threading import Event, Lock, Thread

import cv2
import numpy as np

from assistant_app.utils.logging_config import get_logger


class ScreenCapture:
    def __init__(self, quality: int = 80, scale_factor: float = 0.8,
                 refresh_interval: float = 0.2, cache_duration: float = 1.0):
        self.quality = quality
        self.scale_factor = scale_factor
        self.refresh_interval = refresh_interval
        self.cache_duration = cache_duration
        self.screenshot = None
        self.encoded_cache = None
        self.cache_timestamp = 0
        self.running = False
        self.lock = Lock()
        self.stop_event = Event()
        self.logger = get_logger(__name__)
        self._capture_errors = 0
        
    def start(self) -> "ScreenCapture":
        if self.running:
            return self
        self.running = True
        self.stop_event.clear()
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self
    
    def _capture_loop(self) -> None:
        from PIL import Image, ImageGrab
        
        last_capture = 0
        interval = self.refresh_interval
        
        while self.running and not self.stop_event.is_set():
            if time.time() - last_capture < interval:
                time.sleep(0.05)
                continue
            
            if self.stop_event.is_set():
                break
                
            try:
                screenshot = ImageGrab.grab()
                
                if screenshot is None:
                    continue
                
                if self.scale_factor < 1.0:
                    w = int(screenshot.width * self.scale_factor)
                    h = int(screenshot.height * self.scale_factor)
                    screenshot = screenshot.resize((w, h), Image.LANCZOS)
                
                screenshot_bgr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                
                with self.lock:
                    self.screenshot = screenshot_bgr
                    self.encoded_cache = None
                
                self._capture_errors = 0
                last_capture = time.time()
                
            except Exception as e:
                self._capture_errors += 1
                if self._capture_errors <= 3:
                    self.logger.warning(f"⚠️ Capture error: {e}")
                time.sleep(0.5)
    
    def get_encoded(self) -> bytes | None:
        with self.lock:
            if self.screenshot is None:
                return None
                
            if self.encoded_cache and time.time() - self.cache_timestamp < self.cache_duration:
                return self.encoded_cache
            
            try:
                _, buffer = cv2.imencode('.jpg', self.screenshot, 
                                          [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                self.encoded_cache = base64.b64encode(buffer)
                self.cache_timestamp = time.time()
                return self.encoded_cache
            except Exception as e:
                self.logger.error(f"❌ Encoding error: {e}")
                return None
    
    def stop(self) -> None:
        self.running = False
        self.stop_event.set()
        
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)
