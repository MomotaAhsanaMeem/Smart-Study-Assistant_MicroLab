"""
Hand Gesture Lamp Controller
This module provides an industry-standard, object-oriented approach to control a lamp
using hand gestures via MediaPipe and OpenCV. It is completely independent from the focus tracker.
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
from abc import ABC, abstractmethod
from collections import deque
import hardware_config

try:
    from gpiozero import AngularServo, PWMLED
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

try:
    import smbus2
    SMBUS_AVAILABLE = True
except ImportError:
    SMBUS_AVAILABLE = False

# Configure structured logging for production environments
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LampState(Enum):
    ON = "ON"
    OFF = "OFF"

class Gesture(Enum):
    OPEN_HAND = "OPEN_HAND"
    CLOSED_FIST = "CLOSED_FIST"
    PINCH = "PINCH"
    POINT = "POINT"
    PEACE = "PEACE"
    THUMBS_UP = "THUMBS_UP"
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"

class SystemState(Enum):
    IDLE = "IDLE"      # Low-power state, camera closed, listening for wake word
    ACTIVE = "ACTIVE"  # Active state, camera open, gesture control active

class LampInterface(ABC):
    @abstractmethod
    def turn_on(self):
        pass

    @abstractmethod
    def turn_off(self):
        pass

    @abstractmethod
    def get_state(self) -> LampState:
        pass

    @abstractmethod
    def set_brightness(self, level: int):
        pass

    @abstractmethod
    def get_brightness(self) -> int:
        pass

    @abstractmethod
    def set_pan_angle(self, angle: float):
        pass

    @abstractmethod
    def get_pan_angle(self) -> int:
        pass

    @abstractmethod
    def idle(self):
        pass

    def move_pan_direct(self, angle: float):
        """Set pan angle directly (no 20-deg stepping). Used by buttons and dashboard."""
        angle = max(0.0, min(180.0, angle))
        self._pan_angle = angle

class MockLamp(LampInterface):
    """A simulated lamp for development and testing."""
    def __init__(self):
        self._state = LampState.OFF
        self._brightness = 50.0 # Default 50%
        self._pan_angle = 100.0 # Default center segment (0-180 in 20 deg steps)
        logger.info(f"MockLamp initialized in state {self._state.value} with brightness {self._brightness}%")

    def turn_on(self):
        if self._state != LampState.ON:
            self._state = LampState.ON
            logger.info("Lamp turned ON.")

    def turn_off(self):
        if self._state != LampState.OFF:
            self._state = LampState.OFF
            logger.info("Lamp turned OFF.")

    def get_state(self) -> LampState:
        return self._state

    def set_brightness(self, level: int):
        # Clamp between 0 and 100
        level = max(0, min(100, level))
        # Reduce spamming logs by only logging significant changes or just accepting the value quietly
        if abs(self._brightness - level) > 5:
            self._brightness = level
            if self._state == LampState.ON:
                logger.info(f"Lamp brightness set to {self._brightness}%")
        else:
             self._brightness = level

    def get_brightness(self) -> int:
        return int(round(self._brightness))

    def set_pan_angle(self, angle: float):
        # Clamp target angle between 0 and 180
        angle = max(0.0, min(180.0, angle))
        
        # Calculate deviation from the current motor segment position
        diff = angle - self._pan_angle
        
        # Shift motor by 20-degree steps if hand moves 20 degrees or more
        if diff >= 20.0:
            steps = int(diff // 20)
            new_angle = self._pan_angle + (steps * 20)
            self._pan_angle = max(0.0, min(180.0, new_angle))
            if self._state == LampState.ON:
                logger.info(f"Lamp pan angle set to {int(self._pan_angle)} deg")
        elif diff <= -20.0:
            steps = int(abs(diff) // 20)
            new_angle = self._pan_angle - (steps * 20)
            self._pan_angle = max(0.0, min(180.0, new_angle))
            if self._state == LampState.ON:
                logger.info(f"Lamp pan angle set to {int(self._pan_angle)} deg")

    def get_pan_angle(self) -> int:
        return int(round(self._pan_angle))

    def idle(self):
        pass

    def move_pan_direct(self, angle: float):
        """Set pan angle directly (no 20-deg stepping). Used by buttons and dashboard."""
        angle = max(0.0, min(180.0, angle))
        if abs(self._pan_angle - angle) > 0.1:
            self._pan_angle = angle
            if self._state == LampState.ON:
                logger.info(f"Lamp pan angle set to {int(self._pan_angle)} deg (direct)")

class ServoLamp(LampInterface):
    """A lamp controller that uses an actual servo motor for panning and a MOSFET PWM for brightness."""
    def __init__(self, servo_pin=hardware_config.SERVO_PIN, brightness_pin=hardware_config.LAMP_BRIGHTNESS_PIN):
        self._state = LampState.OFF
        self._brightness = 50.0 # Default 50%
        self._pan_angle = 100.0 # Default center segment (0-180 in 20 deg steps)
        self.servo = None
        self.lamp_pwm = None
        self._servo_attached = True # Track attachment state manually to avoid gpiozero KeyError
        self.last_move_time = time.time() # Track when the motor last shifted/moved to allow self-detaching on idle
        self._watchdog_running = False
        
        if GPIO_AVAILABLE:
            try:
                # Initialize servo on the specified pin
                # min_pulse_width and max_pulse_width adjusted for standard 180 deg SG90
                self.servo = AngularServo(servo_pin, min_angle=0, max_angle=180, min_pulse_width=0.0005, max_pulse_width=0.0024)
                self.servo.angle = self._pan_angle
                
                # Initialize lamp PWM output
                self.lamp_pwm = PWMLED(brightness_pin)
                self.lamp_pwm.value = 0.0
                
                logger.info(f"ServoLamp initialized with Servo on GPIO {servo_pin} and LED/Lamp PWM on GPIO {brightness_pin}")
                
                # Start background servo watchdog thread to handle automatic detaching
                self._watchdog_running = True
                self._watchdog_thread = threading.Thread(target=self._servo_watchdog, daemon=True)
                self._watchdog_thread.start()
            except Exception as e:
                logger.error(f"Failed to initialize servo or PWM LED: {e}")
        else:
            logger.warning("gpiozero not available. Running without hardware servo/lamp control.")

    def _servo_watchdog(self):
        """Background thread that automatically detaches the servo after movement completes to prevent jitter."""
        while self._watchdog_running:
            if self.servo and self._servo_attached and (time.time() - self.last_move_time > 0.8):
                try:
                    self.servo.detach()
                    self._servo_attached = False
                    logger.info("Servo watchdog: Servo detached automatically to prevent vibration.")
                except Exception as e:
                    logger.error(f"Servo watchdog error detaching: {e}")
            time.sleep(0.1)

    def turn_on(self):
        if self._state != LampState.ON:
            self._state = LampState.ON
            logger.info("Lamp turned ON.")
            if self.servo:
                self.servo.angle = self._pan_angle
                self._servo_attached = True
                self.last_move_time = time.time()
            if self.lamp_pwm:
                self.lamp_pwm.value = self._brightness / 100.0

    def turn_off(self):
        if self._state != LampState.OFF:
            self._state = LampState.OFF
            logger.info("Lamp turned OFF.")
            if self.servo:
                self.servo.detach()
                self._servo_attached = False
            if self.lamp_pwm:
                self.lamp_pwm.value = 0.0

    def get_state(self) -> LampState:
        return self._state

    def set_brightness(self, level: int):
        level = max(0, min(100, level))
        if abs(self._brightness - level) > 5:
            self._brightness = level
            if self._state == LampState.ON:
                logger.info(f"Lamp brightness set to {self._brightness}%")
        else:
             self._brightness = level
        if self._state == LampState.ON and self.lamp_pwm:
            self.lamp_pwm.value = self._brightness / 100.0

    def get_brightness(self) -> int:
        return int(round(self._brightness))

    def set_pan_angle(self, angle: float):
        # Clamp target angle between 0 and 180
        angle = max(0.0, min(180.0, angle))
        
        # Calculate deviation from the current motor segment position
        diff = angle - self._pan_angle
        
        # Shift motor by 20-degree steps if hand moves 20 degrees or more
        if diff >= 20.0:
            steps = int(diff // 20)
            new_angle = self._pan_angle + (steps * 20)
            self._pan_angle = max(0.0, min(180.0, new_angle))
            if self._state == LampState.ON:
                logger.info(f"Lamp pan angle set to {int(self._pan_angle)} deg")
                if self.servo:
                    self.servo.angle = self._pan_angle
                    self._servo_attached = True
                    self.last_move_time = time.time()
        elif diff <= -20.0:
            steps = int(abs(diff) // 20)
            new_angle = self._pan_angle - (steps * 20)
            self._pan_angle = max(0.0, min(180.0, new_angle))
            if self._state == LampState.ON:
                logger.info(f"Lamp pan angle set to {int(self._pan_angle)} deg")
                if self.servo:
                    self.servo.angle = self._pan_angle
                    self._servo_attached = True
                    self.last_move_time = time.time()

    def get_pan_angle(self) -> int:
        return int(round(self._pan_angle))

    def idle(self):
        # Kept for compatibility. The background watchdog thread now handles this automatically.
        pass

    def move_pan_direct(self, angle: float):
        """Set pan angle directly (no 20-deg stepping). Used by buttons and dashboard."""
        angle = max(0.0, min(180.0, angle))
        self._pan_angle = angle
        if self._state == LampState.ON and self.servo:
            self.servo.angle = self._pan_angle
            self._servo_attached = True
            self.last_move_time = time.time()
            logger.info(f"Lamp pan angle set to {int(self._pan_angle)} deg (direct)")

class SmartLightingSensor:
    """Reads LDR analog values via PCF8591 I2C ADC module for auto-brightness."""
    def __init__(self, bus_num=hardware_config.I2C_BUS_NUMBER, address=hardware_config.PCF8591_ADDRESS):
        self.address = address
        self.bus = None
        self.available = False
        if SMBUS_AVAILABLE:
            try:
                self.bus = smbus2.SMBus(bus_num)
                # Test connection by writing control byte (0x40 enables AOUT, selects AIN0)
                self.bus.write_byte(self.address, 0x40)
                self.bus.read_byte(self.address)
                self.available = True
                logger.info(f"PCF8591 Smart Lighting Sensor detected at I2C {hex(address)}")
            except Exception as e:
                logger.warning(f"PCF8591 Sensor not detected or error: {e}")
        else:
            logger.warning("smbus2 not available, cannot initialize Smart Lighting Sensor.")

    def read_light_level(self):
        if not self.available:
            return None
        try:
            self.bus.write_byte(self.address, 0x40) # Select AIN0
            self.bus.read_byte(self.address) # Dummy read to clear old value
            value = self.bus.read_byte(self.address)
            return value
        except Exception as e:
            logger.error(f"Error reading LDR: {e}")
            return None

class PiCameraStream:
    """A highly robust, threaded camera reader utilizing Raspberry Pi 5's libcamera backend."""
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

class HandTracker:
    """Wrapper around MediaPipe Hands to isolate the CV logic."""
    def __init__(self, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, bgr_image):
        image_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True
        return results

    def close(self):
        self.hands.close()

class GestureRecognizer:
    """Interprets MediaPipe landmarks into logical gestures and continuous values."""
    def __init__(self, debounce_frames=10):
        self.debounce_frames = debounce_frames
        self.gesture_buffer = deque(maxlen=debounce_frames)
        self.distance_buffer = deque(maxlen=8) # Moving average for brightness (wider = smoother)
        self.point_x_buffer = deque(maxlen=15) # Increased buffer size for smoother moving average
        self.tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky

    def _is_finger_open(self, hand_landmarks, finger_tip_id):
        if finger_tip_id == 4: # Thumb — distance-based extension check
            tip = hand_landmarks.landmark[4]
            mcp = hand_landmarks.landmark[2]
            wrist = hand_landmarks.landmark[0]
            tip_dist = np.hypot(tip.x - wrist.x, tip.y - wrist.y)
            mcp_dist = np.hypot(mcp.x - wrist.x, mcp.y - wrist.y)
            return tip_dist > mcp_dist + 0.05
        else: # Other fingers — tip further from wrist than PIP joint
            tip = hand_landmarks.landmark[finger_tip_id]
            pip = hand_landmarks.landmark[finger_tip_id - 2]
            wrist = hand_landmarks.landmark[0]
            tip_dist = np.hypot(tip.x - wrist.x, tip.y - wrist.y)
            pip_dist = np.hypot(pip.x - wrist.x, pip.y - wrist.y)
            return tip_dist > pip_dist

    def _is_thumb_pointing_up(self, hand_landmarks):
        """Y-coordinate check: is the thumb tip physically above the knuckle line?

        This is the ONLY reliable way to distinguish THUMBS_UP from CLOSED_FIST.
        Both gestures have 4 curled fingers; the difference is whether the thumb
        sticks UPWARD (tip.y well above index MCP) or wraps sideways (same level).

        MediaPipe Y=0 is the TOP of the frame, so a smaller Y means higher up.
        We require the thumb tip to be above the INDEX MCP (knuckle) by a margin
        to avoid false positives when the thumb rests flat on the side of the fist.
        """
        thumb_tip   = hand_landmarks.landmark[4]   # thumb fingertip
        thumb_ip    = hand_landmarks.landmark[3]   # thumb IP joint (one below tip)
        index_mcp   = hand_landmarks.landmark[5]   # index knuckle — reference height
        middle_mcp  = hand_landmarks.landmark[9]   # middle knuckle — secondary reference

        # The thumb tip must be clearly ABOVE (lower Y) the average knuckle line
        avg_knuckle_y = (index_mcp.y + middle_mcp.y) / 2.0

        # Primary check: tip is above the knuckle line by at least 0.05 (normalized)
        tip_above_knuckles = thumb_tip.y < (avg_knuckle_y - 0.05)

        # Secondary check: the thumb segment is pointing upward (IP joint above MCP)
        thumb_pointing_up = thumb_tip.y < thumb_ip.y

        return tip_above_knuckles and thumb_pointing_up

    def recognize(self, hand_landmarks):
        """Returns the recognized Gesture, smoothed pinch distance, and smoothed point x."""
        if not hand_landmarks:
            self.gesture_buffer.append(Gesture.NONE)
            return Gesture.NONE, None, None
            
        fingers_open = [self._is_finger_open(hand_landmarks, tip_id) for tip_id in self.tip_ids]
        
        # Calculate pinch distance (Thumb tip = 4, Index tip = 8)
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        pinch_dist = np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
        
        # Point X (Index tip x)
        point_x = index_tip.x
        
        # Smoothing buffers
        self.distance_buffer.append(pinch_dist)
        avg_pinch_dist = sum(self.distance_buffer) / len(self.distance_buffer)
        
        self.point_x_buffer.append(point_x)
        avg_point_x = sum(self.point_x_buffer) / len(self.point_x_buffer)
        
        # Determine instantaneous gesture
        current_gesture = Gesture.UNKNOWN
        
        # ---- Thumb disambiguation ----
        # Use Y-coordinate geometry to tell THUMBS_UP apart from CLOSED_FIST.
        # Distance-based thumb extension alone is NOT reliable because in a fist
        # the thumb tip can still appear 'extended' sideways in 2D projection.
        thumb_up = self._is_thumb_pointing_up(hand_landmarks)

        if fingers_open[0] and all(fingers_open[1:]):
            # All 5 fingers open (including thumb) -> OPEN_HAND
            current_gesture = Gesture.OPEN_HAND
        elif thumb_up and not any(fingers_open[1:]):
            # Thumb pointing UPWARD + all 4 fingers curled -> unambiguous THUMBS_UP
            current_gesture = Gesture.THUMBS_UP
        elif not any(fingers_open[1:]) and not thumb_up:
            # 4 fingers curled + thumb NOT pointing up (sideways / tucked) -> CLOSED_FIST
            current_gesture = Gesture.CLOSED_FIST
        elif fingers_open[1] and fingers_open[2] and not fingers_open[3] and not fingers_open[4]:
            # Index and Middle open, Ring and Pinky closed -> PEACE (Pan Control)
            current_gesture = Gesture.PEACE
        elif fingers_open[1] and not any(fingers_open[2:]):
            # Index open, Middle, ring, and pinky are closed -> POINT or PINCH mode
            # Check the pinch distance to distinguish point from pinch
            if avg_pinch_dist < 0.07:
                current_gesture = Gesture.PINCH
            else:
                current_gesture = Gesture.POINT
            
        # Add to debounce buffer for ON/OFF stability
        self.gesture_buffer.append(current_gesture)
        
        # For discrete actions (ON/OFF), require settled state
        if len(self.gesture_buffer) >= self.debounce_frames and all(g == self.gesture_buffer[-1] for g in self.gesture_buffer):
            settled_gesture = self.gesture_buffer[-1]
        elif current_gesture in [Gesture.PINCH, Gesture.PEACE, Gesture.POINT]:
            # We allow continuous controls to be immediate so they adjust smoothly
            settled_gesture = current_gesture
        else:
            settled_gesture = Gesture.UNKNOWN
            
        return settled_gesture, avg_pinch_dist, avg_point_x

def find_usb_microphone_index():
    """Dynamically locates the best available audio input device (Bluetooth, USB, default, or other)."""
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        bt_devices = []
        usb_devices = []
        default_devices = []
        other_devices = []
        
        for i in range(p.get_device_count()):
            try:
                info = p.get_device_info_by_index(i)
                name = info.get('name', '')
                name_lower = name.lower()
                max_input_channels = info.get('maxInputChannels', 0)
                
                if max_input_channels > 0:
                    # Categorize by device type keywords
                    if any(kw in name_lower for kw in ['bluetooth', 'bluez', 'headset', 'handsfree', 'sony', 'wf-']):
                        bt_devices.append((i, name))
                    elif any(kw in name_lower for kw in ['usb', 'pnp', 'microphone', 'mic']):
                        usb_devices.append((i, name))
                    elif any(kw in name_lower for kw in ['default', 'pulse', 'pipewire']):
                        default_devices.append((i, name))
                    else:
                        other_devices.append((i, name))
            except Exception as e:
                continue
                
        p.terminate()
        
        if default_devices:
            logger.info(f"Dynamically detected Default/PipeWire Input Device: '{default_devices[0][1]}' at index {default_devices[0][0]}")
            return default_devices[0][0]
        elif bt_devices:
            logger.info(f"Dynamically detected Bluetooth Input Device: '{bt_devices[0][1]}' at index {bt_devices[0][0]}")
            return bt_devices[0][0]
        elif usb_devices:
            logger.info(f"Dynamically detected USB Input Device: '{usb_devices[0][1]}' at index {usb_devices[0][0]}")
            return usb_devices[0][0]
        elif other_devices:
            logger.info(f"Dynamically detected Input Device: '{other_devices[0][1]}' at index {other_devices[0][0]}")
            return other_devices[0][0]
            
    except Exception as e:
        logger.warning(f"Error while searching for audio devices: {e}")
        
    logger.warning("No specific audio input device detected. Using system default input device (None).")
    return None

class VoiceCommandListener:
    """Runs a background thread using SpeechRecognition to capture audio and detect wake commands."""
    def __init__(self, command_callback):
        self.command_callback = command_callback
        self.running = False
        self.thread = None
        self.device_index = find_usb_microphone_index()
        
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 1500
            self.recognizer.dynamic_energy_threshold = True
        except ImportError:
            logger.critical("speech_recognition library is not available.")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        logger.info("VoiceCommandListener background listener started.")

    def stop(self):
        self.running = False

    def _listen_loop(self):
        import speech_recognition as sr
        
        # Calibrate once on startup to prevent repeated calibration logs
        calibrated = False
        
        while self.running:
            try:
                # Initialize microphone with detected index
                with sr.Microphone(device_index=self.device_index) as source:
                    if not calibrated:
                        logger.info("🔊 [VOICE] Calibrating microphone for background noise... Please remain quiet.")
                        self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                        calibrated = True
                        
                    logger.info("🎙️ [VOICE] Listening for commands (e.g., 'activate lamp')...")
                    audio = self.recognizer.listen(source, timeout=5.0, phrase_time_limit=3.0)
                
                if not self.running:
                    break
                
                logger.info("⚡ [VOICE] Audio captured, transcribing speech...")
                text = self.recognizer.recognize_google(audio).lower()
                logger.info(f"📝 [VOICE] Heard: '{text}'")
                
                # Pass text to the application to handle specific commands
                self.command_callback(text)
            except sr.WaitTimeoutError:
                # Timeout occurred, loop again
                continue
            except sr.UnknownValueError:
                logger.warning("❓ [VOICE] Captured audio, but could not understand speech. Please speak clearly.")
                continue
            except sr.RequestError as e:
                logger.error(f"❌ [VOICE] Google Speech API request error (check internet connection): {e}")
                time.sleep(5.0)
            except Exception as e:
                logger.error(f"Background voice listener error: {e}")
                time.sleep(2.0)

class LampControllerApp:
    def __init__(self):
        self.headless = not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
        if self.headless:
            logger.info("No display detected. Running in headless mode.")
        else:
            logger.info("Display detected. Video UI will be shown.")
            
        self.lamp = ServoLamp(servo_pin=hardware_config.SERVO_PIN, brightness_pin=hardware_config.LAMP_BRIGHTNESS_PIN)
        self.tracker = HandTracker()
        # Require 15 consecutive frames for a solid ON/OFF to prevent flickering
        self.recognizer = GestureRecognizer(debounce_frames=15) 
        self.cap = None
        
        # Brightness mapping range for pinch distance 
        # (normalized MediaPipe coordinates are typically 0.0 to 1.0)
        self.min_pinch_dist = 0.03
        self.max_pinch_dist = 0.20
        
        self.smart_sensor = SmartLightingSensor(bus_num=hardware_config.I2C_BUS_NUMBER, address=hardware_config.PCF8591_ADDRESS)
        self.manual_override_time = 0.0
        self.smart_mode_enabled = True # Enabled by default
        self.prev_gesture = Gesture.NONE
        
        # Voice Activation & Sleep State Machine
        self.state = SystemState.IDLE
        self.last_gesture_time = time.time()
        self.voice_listener = VoiceCommandListener(self.handle_voice_command)
        self.voice_listener.start()

    def handle_voice_command(self, text):
        """Callback to handle specific commands parsed from voice text."""
        logger.info(f"Handling voice command: {text}")
        
        # Wake / gesture activation
        if "activate gesture" in text or "start gesture" in text or "start camera" in text or "wake" in text:
            if self.state == SystemState.IDLE:
                logger.info("Voice command heard: activating lamp and gesture tracking!")
                self.state = SystemState.ACTIVE
                self.lamp.turn_on()
                self.last_gesture_time = time.time()
                
        # Basic ON/OFF
        elif "lamp on" in text or "turn on" in text or "turn the lamp on" in text:
            self.lamp.turn_on()
            self.last_gesture_time = time.time()
        elif "lamp off" in text or "turn off" in text or "turn the lamp off" in text:
            self.lamp.turn_off()
            if self.state == SystemState.ACTIVE:
                self._cleanup_camera()
                self.state = SystemState.IDLE
                
        # Brightness control
        if "brightness" in text:
            import re
            match = re.search(r'\d+', text)
            if match:
                brightness = int(match.group())
                self.lamp.set_brightness(brightness)
                self.manual_override_time = time.time()
            elif "up" in text or "increase" in text or "higher" in text:
                self.lamp.set_brightness(min(100, self.lamp.get_brightness() + 20))
                self.manual_override_time = time.time()
            elif "down" in text or "decrease" in text or "lower" in text:
                self.lamp.set_brightness(max(0, self.lamp.get_brightness() - 20))
                self.manual_override_time = time.time()
                
        # Pan control
        if "pan" in text or "turn" in text or "rotate" in text or "look" in text:
            if "left" in text:
                self.lamp.set_pan_angle(max(0.0, self.lamp.get_pan_angle() - 30.0))
            elif "right" in text:
                self.lamp.set_pan_angle(min(180.0, self.lamp.get_pan_angle() + 30.0))
            elif "center" in text:
                self.lamp.set_pan_angle(90.0)
                
        # Smart mode toggle
        if "smart mode on" in text or "enable smart mode" in text:
            self.smart_mode_enabled = True
            logger.info("Smart Auto-Brightness Mode toggled: ON via voice")
        elif "smart mode off" in text or "disable smart mode" in text:
            self.smart_mode_enabled = False
            logger.info("Smart Auto-Brightness Mode toggled: OFF via voice")

    def _init_camera(self):
        """Initializes the camera stream when system transitions to ACTIVE."""
        if shutil.which("rpicam-vid"):
            logger.info("Attempting to initialize hardware-accelerated PiCameraStream...")
            self.cap = PiCameraStream(width=640, height=480, framerate=30).start()
            if self.cap and self.cap.isOpened():
                logger.info("PiCameraStream initialized successfully.")
            else:
                if self.cap:
                    self.cap.release()
                self.cap = None
                
        if self.cap is None or not self.cap.isOpened():
            logger.warning("Falling back to standard V4L2 indices...")
            for idx in [0, 1, 2, 4, 6]:
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if cap.isOpened():
                    success, _ = cap.read()
                    if success:
                        self.cap = cap
                        logger.info(f"Successfully connected to V4L2 camera at index {idx}.")
                        break
                    else:
                        cap.release()

    def _cleanup_camera(self):
        """Safely releases the camera and closes UI windows on entering IDLE."""
        logger.info("Releasing camera and UI resources...")
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
            self.cap = None
        cv2.destroyAllWindows()
        logger.info("Camera resources fully released.")

    def start(self):
        # Print Command Guide once at startup
        print("="*60)
        print("💡 GESTURE & VOICE LAMP CONTROLLER COMMAND GUIDE")
        print("="*60)
        print("\n🎙️  VOICE COMMANDS (Always Listening):")
        print("  • Wake / Gesture:   'activate gesture', 'start gesture', 'start camera', 'wake'")
        print("  • Power Controls:   'lamp on', 'turn on', 'turn the lamp on'")
        print("                      'lamp off', 'turn off', 'turn the lamp off'")
        print("  • Brightness:       'brightness [0-100]' (e.g., 'brightness 80')")
        print("                      'brightness up', 'increase brightness', 'brightness higher'")
        print("                      'brightness down', 'decrease brightness', 'brightness lower'")
        print("  • Pan / Direction:  'pan left', 'turn left', 'rotate left', 'look left'")
        print("                      'pan right', 'turn right', 'rotate right', 'look right'")
        print("                      'pan center', 'turn center', 'rotate center', 'look center'")
        print("  • Smart Mode:       'smart mode on', 'enable smart mode'")
        print("                      'smart mode off', 'disable smart mode'")
        print("\n🖐️  GESTURE CONTROLS (Active Mode):")
        print("  • Turn ON:          OPEN HAND (Show all fingers)")
        print("  • Turn OFF & Sleep: CLOSED FIST (Make a fist)")
        print("  • Brightness:       PINCH (Pinch thumb & index finger; slide hand left/right)")
        print("                      *Note: Brightness gestures only work when Smart Mode is OFF.")
        print("  • Pan / Direction:  PEACE or POINT (Show peace sign or point; move left/right)")
        print("  • Toggle Smart:     THUMBS UP (Enable/disable auto-brightness)")
        print("="*60 + "\n")

        logger.info("Lamp Controller system started in IDLE mode. Say 'activate lamp' to begin.")
        consecutive_failures = 0
        
        try:
            while True:
                if self.state == SystemState.IDLE:
                    # Sleep in IDLE state to reduce CPU consumption to 0%
                    time.sleep(0.1)
                    continue

                # Open camera stream if entering ACTIVE state
                if self.cap is None or not self.cap.isOpened():
                    self._init_camera()
                    if self.cap is None or not self.cap.isOpened():
                        logger.error("Failed to reinitialize camera. Returning to IDLE.")
                        self.state = SystemState.IDLE
                        self.lamp.turn_off()
                        continue
                    self.last_gesture_time = time.time()

                success, image = self.cap.read()
                if not success:
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        logger.error("Camera connection lost. Returning to IDLE.")
                        self._cleanup_camera()
                        self.state = SystemState.IDLE
                        self.lamp.turn_off()
                    else:
                        time.sleep(0.1)
                    continue
                consecutive_failures = 0
                
                # Optimizations for MediaPipe
                results = self.tracker.process_frame(image)
                
                gesture_text = "No Hand"
                
                if results.multi_hand_landmarks:
                    primary_hand = results.multi_hand_landmarks[0]
                    
                    if not self.headless:
                        self.tracker.mp_drawing.draw_landmarks(
                            image,
                            primary_hand,
                            self.tracker.mp_hands.HAND_CONNECTIONS,
                            self.tracker.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.tracker.mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                    gesture, pinch_dist, point_x = self.recognizer.recognize(primary_hand)
                    
                    # Update inactivity timer on valid gestures
                    if gesture not in [Gesture.NONE, Gesture.UNKNOWN]:
                        self.last_gesture_time = time.time()

                    if gesture == Gesture.THUMBS_UP and self.prev_gesture != Gesture.THUMBS_UP:
                        self.smart_mode_enabled = not self.smart_mode_enabled
                        mode_status = "ON" if self.smart_mode_enabled else "OFF"
                        logger.info(f"Smart Auto-Brightness Mode toggled: {mode_status}")
                    
                    self.prev_gesture = gesture

                    if gesture == Gesture.OPEN_HAND:
                        self.lamp.turn_on()
                        gesture_text = "Action: Turn ON"
                    elif gesture == Gesture.CLOSED_FIST:
                        self.lamp.turn_off()
                        gesture_text = "Action: Turn OFF"
                        logger.info("Closed fist detected: returning to IDLE sleep mode.")
                        
                        # Briefly render OFF state on frame before closing UI window
                        if not self.headless:
                            status_color = (0, 0, 255)
                            mode_text = "Smart: ON" if self.smart_mode_enabled else "Smart: OFF"
                            cv2.putText(image, f"Lamp: OFF | {mode_text}", (20, 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                            cv2.putText(image, gesture_text, (20, image.shape[0] - 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.imshow('Gesture Lamp Controller', image)
                            cv2.waitKey(1000)
                        
                        self._cleanup_camera()
                        self.state = SystemState.IDLE
                        continue
                    elif gesture == Gesture.PINCH and self.lamp.get_state() == LampState.ON:
                        if not self.smart_mode_enabled:
                            # Map pinch distance to brightness
                            dist = max(self.min_pinch_dist, min(self.max_pinch_dist, pinch_dist))
                            percent = ((dist - self.min_pinch_dist) / (self.max_pinch_dist - self.min_pinch_dist)) * 100
                            self.lamp.set_brightness(int(percent))
                            gesture_text = f"Action: Brightness {int(percent)}%"
                            self.manual_override_time = time.time()
                        else:
                            gesture_text = "Pinch Disabled (Smart Mode ON)"
                    elif gesture in [Gesture.PEACE, Gesture.POINT] and self.lamp.get_state() == LampState.ON:
                        # Map point_x (0.0 - 1.0) to pan angle (0 - 180)
                        clamped_x = max(0.0, min(1.0, point_x))
                        # Invert x so that moving hand left corresponds to panning left
                        angle = (1.0 - clamped_x) * 180.0
                        self.lamp.set_pan_angle(angle)
                        gesture_text = f"Action: Pan {int(round(angle))} deg"
                    else:
                        gesture_text = "Tracking..."
                else:
                    self.recognizer.recognize(None)

                # Let the lamp object manage its own detaching on idle (0.8s inactivity)
                if self.lamp.get_state() == LampState.ON:
                    self.lamp.idle()
                    
                # Smart Lighting Logic
                # Only adjust if the lamp is ON, smart mode is enabled, and the user hasn't manually overridden it via PINCH in the last 10 seconds
                if self.smart_mode_enabled and self.lamp.get_state() == LampState.ON and (time.time() - self.manual_override_time > 10.0):
                    light_level = self.smart_sensor.read_light_level()
                    if light_level is not None:
                        # Typically for PCF8591 with LDR:
                        # High value (near 255) = dark room. Low value (near 0) = bright room.
                        # We map dark room -> 100% brightness, bright room -> 0% brightness
                        # We use a moving average in the lamp controller or just set it directly
                        smart_brightness = int((light_level / 255.0) * 100)
                        self.lamp.set_brightness(smart_brightness)
                        
                        # Only show smart mode in UI if we are actively tracking but not overriding
                        if not results.multi_hand_landmarks:
                            gesture_text = f"Smart Auto-Brightness: {smart_brightness}%"
                            
                # UI Rendering
                if not self.headless:
                    status_color = (0, 255, 0) if self.lamp.get_state() == LampState.ON else (0, 0, 255)
                    mode_text = "Smart: ON" if self.smart_mode_enabled else "Smart: OFF"
                    
                    cv2.putText(image, f"Lamp: {self.lamp.get_state().value} | {mode_text}", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                                
                    if self.lamp.get_state() == LampState.ON:
                        cv2.putText(image, f"Bright: {self.lamp.get_brightness()}% | Pan: {self.lamp.get_pan_angle()} deg", (20, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                                    
                    cv2.putText(image, gesture_text, (20, image.shape[0] - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                                
                    cv2.imshow('Gesture Lamp Controller', image)
                    if cv2.waitKey(5) & 0xFF == 27: # ESC
                        logger.info("User requested shutdown via UI.")
                        break

                # Auto-timeout: return to IDLE sleep mode after 60 seconds of inactivity
                if self.state == SystemState.ACTIVE and (time.time() - self.last_gesture_time > 60.0):
                    logger.info("Inactivity auto-timeout (60 seconds) reached: returning to IDLE mode.")
                    self.lamp.turn_off()
                    self._cleanup_camera()
                    self.state = SystemState.IDLE

        except KeyboardInterrupt:
            logger.info("Force closed via keyboard interrupt.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Cleaning up resources...")
        self.voice_listener.stop()
        self._cleanup_camera()
        self.tracker.close()
        logger.info("System shutdown complete.")

if __name__ == "__main__":
    app = LampControllerApp()
    app.start()
