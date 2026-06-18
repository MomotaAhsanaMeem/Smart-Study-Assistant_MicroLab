import subprocess
import threading
import time
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class PiCameraStream:
    """A highly robust, threaded camera reader utilizing Raspberry Pi 5's libcamera backend."""
    def __init__(self, width=1280, height=720, framerate=30):
        # Higher resolution for OCR
        self.width = width
        self.height = height
        self.frame_size = int(width * height * 1.5)
        self.framerate = framerate
        self.process = None
        self.latest_frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        command = [
            "rpicam-vid",
            "-t", "0",
            "--codec", "yuv420",
            "--width", str(self.width),
            "--height", str(self.height),
            "--framerate", str(self.framerate),
            "-o", "-",
            "--nopreview",
            "--denoise", "cdn_hq" # High quality denoise for clear text
        ]
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**7)
        except FileNotFoundError:
            return None
            
        self.running = True
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        
        start_wait = time.time()
        while self.latest_frame is None and time.time() - start_wait < 3.0:
            time.sleep(0.1)
            
        return self

    def _update(self):
        while self.running and self.process and self.process.poll() is None:
            try:
                raw_frame = self.process.stdout.read(self.frame_size)
                if len(raw_frame) != self.frame_size:
                    time.sleep(0.01)
                    continue
                    
                yuv_data = np.frombuffer(raw_frame, dtype=np.uint8).reshape((int(self.height * 1.5), self.width))
                bgr_frame = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420)
                
                with self.lock:
                    self.latest_frame = bgr_frame
            except Exception as e:
                logger.error(f"Camera Error: {e}")
                break

    def read(self):
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def isOpened(self):
        return self.running and self.process is not None and self.process.poll() is None

    def release(self):
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except:
                self.process.kill()
