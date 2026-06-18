"""
Focus Tracking System
This module provides an industry-standard, object-oriented approach to track 
user focus using MediaPipe and OpenCV. It is designed to be robust, handle edge cases, 
and provide structured logging.
"""

import cv2
import mediapipe as mp
import time
import numpy as np
import platform
import logging
import sys
import os
import subprocess
import threading
import shutil
from enum import Enum

# Configure structured logging for production environments
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Status(Enum):
    FOCUSED = "FOCUSED"
    DISTRACTED = "DISTRACTED"
    NO_FACE = "NO_FACE"

class Config:
    """Configuration class to easily tune parameters"""
    CAMERA_INDEX = 0  # Default webcam
    
    # Head pose thresholds (in degrees)
    MAX_YAW = 15.0    # Look right
    MIN_YAW = -15.0   # Look left
    MAX_PITCH = 20.0  # Look down (e.g., at phone)
    MIN_PITCH = -10.0 # Look up
    
    # Alert mechanism
    DISTRACTION_TIME_THRESHOLD_SEC = 30.0  # Time before alert is triggered

class PiCameraStream:
    """A highly robust, threaded camera reader utilizing Raspberry Pi 5's libcamera backend via rpicam-vid."""
    def __init__(self, width=640, height=480, framerate=30):
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
            "--denoise", "cdn_off"
        ]
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**7)
        except FileNotFoundError:
            return None # rpicam-vid not found
            
        self.running = True
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        
        # Give it a moment to spin up and produce the first frame
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
                logger.error(f"Error reading from PiCameraStream: {e}")
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
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


class FocusTracker:
    def __init__(self, config=Config()):
        self.config = config
        
        # Determine if we are running in a headless environment (e.g. SSH without X11)
        self.headless = not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
        if self.headless:
            logger.info("No display detected. Running in headless mode (no video output).")
        else:
            logger.info("Display detected. Video UI will be shown.")
            
        # Initialize MediaPipe Face Mesh securely
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            logger.critical(f"Failed to initialize MediaPipe FaceMesh: {e}")
            sys.exit(1)
            
        # Tracking states
        self.distracted_start_time = None
        self.current_status = Status.NO_FACE
        self.last_alert_time = 0
        
        # Specific landmark indices for 3D Head Pose
        # Nose tip(1), Left Eye(33), Right Eye(263), Chin(152), Left Mouth(61), Right Mouth(291)
        self.target_indices = [1, 33, 263, 152, 61, 291]
        
        self.cap = None

    def _play_alert_sound(self):
        """Cross-platform alert sound implementation"""
        # Throttle sound to avoid spamming the audio buffer every frame
        current_time = time.time()
        if current_time - self.last_alert_time < 5.0: 
            return
            
        self.last_alert_time = current_time
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(1000, 500)
            else:
                # Terminal bell as fallback for Linux/Mac/Raspberry Pi
                sys.stdout.write('\a')
                sys.stdout.flush()
        except Exception as e:
            logger.error(f"Failed to play alert sound: {e}")

    def _get_head_pose(self, face_landmarks, img_w, img_h):
        """Calculates Head Pose (Yaw, Pitch, Roll) using solvePnP"""
        face_2d = []
        face_3d = []
        
        # Validate indices to prevent index out of bounds
        if len(face_landmarks.landmark) < max(self.target_indices):
            return None, None

        for idx in self.target_indices:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
            
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        
        focal_length = 1 * img_w
        cam_matrix = np.array([
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        try:
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            if not success:
                return None, None
                
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            
            pitch = angles[0] * 360
            yaw = angles[1] * 360
            
            return pitch, yaw
        except cv2.error as e:
            logger.error(f"OpenCV solvePnP error: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error in head pose estimation: {e}")
            return None, None

    def start_tracking(self):
        """Main execution loop with hardware-accelerated Pi 5 capture logic"""
        
        # Attempt L9-engineer optimal approach: rpicam-vid stream
        if shutil.which("rpicam-vid"):
            logger.info("Attempting to initialize hardware-accelerated PiCameraStream...")
            self.cap = PiCameraStream(width=640, height=480, framerate=30).start()
            if self.cap and self.cap.isOpened():
                logger.info("PiCameraStream initialized successfully.")
            else:
                if self.cap:
                    self.cap.release()
                self.cap = None
        
        # Fallback to standard V4L2 indices if rpicam-vid fails or is missing
        if self.cap is None or not self.cap.isOpened():
            logger.warning("Falling back to standard V4L2 indices...")
            for idx in [0, 1, 2, 4, 6]:
                logger.info(f"Testing camera index {idx}...")
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if cap.isOpened():
                    # Validate we can actually read a frame before committing
                    success, _ = cap.read()
                    if success:
                        self.config.CAMERA_INDEX = idx
                        self.cap = cap
                        logger.info(f"Successfully connected to V4L2 camera at index {idx}.")
                        break
                    else:
                        cap.release()
                        
        if self.cap is None or not self.cap.isOpened():
            logger.error("Could not open any camera. Please ensure camera is connected and not busy.")
            return

        logger.info("Focus tracking system started.")
        consecutive_failures = 0

        try:
            while True:
                success, image = self.cap.read()
                if not success:
                    consecutive_failures += 1
                    logger.warning(f"Failed to grab frame (Attempt {consecutive_failures}/5).")
                    if consecutive_failures >= 5:
                        logger.error("Camera connection lost. Reconnecting...")
                        self.cap.release()
                        time.sleep(1.0)
                        
                        # Reconnect logic
                        if isinstance(self.cap, PiCameraStream):
                            self.cap = PiCameraStream(width=640, height=480, framerate=30).start()
                        else:
                            self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX, cv2.CAP_V4L2)
                        consecutive_failures = 0
                    else:
                        time.sleep(0.5)
                    continue

                consecutive_failures = 0

                img_h, img_w, _ = image.shape
                
                # Optimizations for MediaPipe (pass by reference, read-only)
                image.flags.writeable = False
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(image_rgb)
                image.flags.writeable = True

                color = (0, 255, 255) # Yellow for NO_FACE

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Draw the face mesh and points for real-time visualization
                        if not self.headless:
                            self.mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                            )
                            self.mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                            )

                        pitch, yaw = self._get_head_pose(face_landmarks, img_w, img_h)
                        
                        if pitch is not None and yaw is not None:
                            # Evaluate posture
                            if (yaw < self.config.MIN_YAW or yaw > self.config.MAX_YAW or 
                                pitch < self.config.MIN_PITCH or pitch > self.config.MAX_PITCH):
                                self.current_status = Status.DISTRACTED
                                color = (0, 0, 255) # Red
                            else:
                                self.current_status = Status.FOCUSED
                                color = (0, 255, 0) # Green
                        else:
                            self.current_status = Status.NO_FACE
                else:
                    self.current_status = Status.NO_FACE
                    color = (0, 255, 255) # Yellow

                # Alert Logic (Timer processing)
                if self.current_status in [Status.DISTRACTED, Status.NO_FACE]:
                    if self.distracted_start_time is None:
                        self.distracted_start_time = time.time()
                    else:
                        elapsed = time.time() - self.distracted_start_time
                        if elapsed > self.config.DISTRACTION_TIME_THRESHOLD_SEC:
                            cv2.putText(image, "ALERT: PLEASE FOCUS!", (30, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            self._play_alert_sound()
                else:
                    # Reset timer if FOCUSED
                    self.distracted_start_time = None

                # UI Updates
                if not self.headless:
                    cv2.putText(image, f"Status: {self.current_status.value}", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    
                    try:
                        cv2.imshow('Student Focus Tracker', image)
                        # Graceful exit via UI (ESC key)
                        if cv2.waitKey(5) & 0xFF == 27:
                            logger.info("User requested shutdown via UI.")
                            break
                    except Exception as e:
                        logger.error(f"UI Error: {e}. Switching to headless mode.")
                        self.headless = True
                else:
                    # In headless mode, we still need a way to throttle the loop slightly 
                    # if the camera stream is extremely fast, though PiCameraStream rate-limits via framerate.
                    pass

        except KeyboardInterrupt:
            logger.info("Force closed via keyboard interrupt.")
        except Exception as e:
            logger.error(f"Unexpected error during tracking loop: {e}", exc_info=True)
        finally:
            self._cleanup()

    def _cleanup(self):
        """Safely release hardware and software resources"""
        logger.info("Cleaning up resources...")
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        try:
            self.face_mesh.close()
        except:
            pass
        cv2.destroyAllWindows()
        logger.info("System shutdown complete.")

if __name__ == "__main__":
    tracker = FocusTracker()
    tracker.start_tracking()
