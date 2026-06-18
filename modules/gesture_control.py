"""
Gesture-based Lamp Control Module
==================================
Wraps HandTracker and GestureRecognizer from the standalone lamp_controller.py.
Processes camera frames supplied by the SystemManager and controls the shared lamp.
"""

import time
import re
import logging
import cv2

from modules.base import FeatureModule
from lamp_controller import (
    HandTracker,
    GestureRecognizer,
    Gesture,
    LampState,
)

logger = logging.getLogger("GestureControl")


class GestureControlModule(FeatureModule):
    """Processes camera frames for hand-gesture–based lamp control."""

    def __init__(self, lamp, smart_sensor):
        self.lamp = lamp
        self.smart_sensor = smart_sensor

        # Re-use the proven classes from lamp_controller.py
        self.tracker = HandTracker()
        self.recognizer = GestureRecognizer(debounce_frames=15)

        # Module state
        self.active = False
        # BUG-FIX: default to False so pinch works immediately on startup.
        # User can enable smart mode via thumbs-up gesture, voice, or dashboard.
        self.smart_mode_enabled = False
        # BUG-FIX: seed to now() so the LDR sensor doesn't override pinch
        # on the very first frames (old value of 0.0 was epoch → always > 10 s).
        self.manual_override_time = time.time()
        self.prev_gesture = Gesture.NONE
        self.last_gesture_time = time.time()
        self.current_gesture = Gesture.NONE

        # Pinch-to-brightness mapping range
        self.min_pinch_dist = 0.03
        self.max_pinch_dist = 0.20

    # ----------------------------------------------------------------
    #  Activate / Deactivate
    # ----------------------------------------------------------------

    def activate(self):
        self.active = True
        self.last_gesture_time = time.time()
        logger.info("Gesture control activated.")

    def deactivate(self):
        self.active = False
        logger.info("Gesture control deactivated.")

    # ----------------------------------------------------------------
    #  Frame Processing  (called by SystemManager camera loop)
    # ----------------------------------------------------------------

    def process_frame(self, frame):
        """Process one camera frame. Returns an event dict or None.

        Possible events:
            {'type': 'lamp_on'}
            {'type': 'lamp_off_and_sleep'}
            {'type': 'gesture_timeout'}
        """
        if not self.active:
            return None

        results = self.tracker.process_frame(frame)
        self._last_results = results
        event = None

        if results.multi_hand_landmarks:
            primary_hand = results.multi_hand_landmarks[0]
            gesture, pinch_dist, point_x = self.recognizer.recognize(primary_hand)

            # Reset inactivity timer on any meaningful gesture
            if gesture not in (Gesture.NONE, Gesture.UNKNOWN):
                self.last_gesture_time = time.time()

            self.current_gesture = gesture

            # --- Thumbs-up toggles smart auto-brightness mode (edge-triggered) ---
            # NOTE: prev_gesture still holds the PREVIOUS frame's value here — correct.
            if gesture == Gesture.THUMBS_UP and self.prev_gesture != Gesture.THUMBS_UP:
                self.smart_mode_enabled = not self.smart_mode_enabled
                logger.info(f"Smart mode: {'ON' if self.smart_mode_enabled else 'OFF'}")

            # ---- FIX: all action checks use prev_gesture BEFORE we update it ----

            # --- Open hand → Turn ON (edge-triggered: fires only on transition) ---
            if gesture == Gesture.OPEN_HAND and self.prev_gesture != Gesture.OPEN_HAND:
                self.lamp.turn_on()
                event = {"type": "lamp_on"}

            # --- Closed fist → Turn OFF + sleep (edge-triggered) ---
            elif gesture == Gesture.CLOSED_FIST and self.prev_gesture != Gesture.CLOSED_FIST:
                self.lamp.turn_off()
                event = {"type": "lamp_off_and_sleep"}

            # --- Pinch → Smooth brightness control (only when smart mode is off) ---
            elif gesture == Gesture.PINCH and self.lamp.get_state() == LampState.ON:
                # Flush stale frames from a previous gesture to prevent brightness spike on entry
                if self.prev_gesture != Gesture.PINCH:
                    self.recognizer.distance_buffer.clear()

                if not self.smart_mode_enabled and pinch_dist is not None:
                    dist = max(self.min_pinch_dist, min(self.max_pinch_dist, pinch_dist))
                    pct = ((dist - self.min_pinch_dist) /
                           (self.max_pinch_dist - self.min_pinch_dist)) * 100
                    new_brightness = int(pct)
                    # Dead-zone: only update if change is ≥ 2% to suppress hand-tremor jitter
                    if abs(new_brightness - self.lamp.get_brightness()) >= 2:
                        self.lamp.set_brightness(new_brightness)
                        self.manual_override_time = time.time()

            # --- Peace / Point → Pan angle ---
            elif gesture in (Gesture.PEACE, Gesture.POINT) and self.lamp.get_state() == LampState.ON:
                if point_x is not None:
                    clamped_x = max(0.0, min(1.0, point_x))
                    angle = (1.0 - clamped_x) * 180.0
                    self.lamp.set_pan_angle(angle)

            # ---- FIX: update prev_gesture AFTER all checks so edge-detection works ----
            self.prev_gesture = gesture

        else:
            # No hand detected — flush buffers and reset prev so next detection fires correctly
            self.recognizer.recognize(None)
            self.current_gesture = Gesture.NONE
            self.prev_gesture = Gesture.NONE  # Reset so next OPEN_HAND / CLOSED_FIST fires fresh

        # --- Smart auto-brightness via LDR sensor ---
        # Guard with sensor.available so a missing/disconnected sensor can't override manual pinch
        if (self.smart_mode_enabled
                and self.lamp.get_state() == LampState.ON
                and self.smart_sensor.available
                and (time.time() - self.manual_override_time > 10.0)):
            light = self.smart_sensor.read_light_level()
            if light is not None:
                # Invert: dark room (high raw value) → needs more brightness
                smart_brightness = int(((255 - light) / 255.0) * 100)
                self.lamp.set_brightness(smart_brightness)

        # Servo idle management (prevent jitter)
        if self.lamp.get_state() == LampState.ON:
            self.lamp.idle()

        # Auto-timeout after 60 s of inactivity → sleep
        if time.time() - self.last_gesture_time > 60.0:
            event = {"type": "gesture_timeout"}

        return event

    # ----------------------------------------------------------------
    #  Frame Annotation (for web MJPEG feed)
    # ----------------------------------------------------------------

    def annotate_frame(self, frame):
        """Draw hand landmarks and lamp status overlay for web display."""
        display = frame.copy()
        h, w = display.shape[:2]

        if hasattr(self, '_last_results') and self._last_results and self._last_results.multi_hand_landmarks:
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            mp_hands = mp.solutions.hands

            for hand_landmarks in self._last_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        # --- Status overlay ---
        lamp_on = self.lamp.get_state() == LampState.ON
        lamp_color = (0, 220, 80) if lamp_on else (0, 60, 255)
        smart_txt = "Smart: ON" if self.smart_mode_enabled else "Smart: OFF"

        cv2.rectangle(display, (0, 0), (w, 44), (0, 0, 0), -1)
        cv2.putText(display, f"Lamp: {'ON' if lamp_on else 'OFF'}  |  {smart_txt}",
                    (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, lamp_color, 2, cv2.LINE_AA)

        # Current gesture
        gesture_txt = f"Gesture: {self.current_gesture.value}"
        cv2.putText(display, gesture_txt, (14, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if lamp_on:
            info = f"Bright: {self.lamp.get_brightness()}%  |  Pan: {int(self.lamp.get_pan_angle())} deg"
            cv2.putText(display, info, (14, h - 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 230), 1, cv2.LINE_AA)

        return display

    # ----------------------------------------------------------------
    #  FeatureModule interface
    # ----------------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "active": self.active,
            "gesture": self.current_gesture.value,
            "smart_mode": self.smart_mode_enabled,
            "lamp_on": self.lamp.get_state() == LampState.ON,
            "brightness": self.lamp.get_brightness(),
            "pan_angle": self.lamp.get_pan_angle(),
        }

    def handle_voice_command(self, text: str) -> bool:
        # --- Power ---
        if any(kw in text for kw in ["lamp on", "turn on", "light on", "turn the lamp on"]):
            self.lamp.turn_on()
            self.last_gesture_time = time.time()
            return True
        if any(kw in text for kw in ["lamp off", "turn off", "light off", "turn the lamp off"]):
            self.lamp.turn_off()
            return True

        # --- Brightness ---
        if "brightness" in text:
            match = re.search(r"\d+", text)
            if match:
                self.lamp.set_brightness(int(match.group()))
                self.manual_override_time = time.time()
                return True
            if any(kw in text for kw in ["up", "increase", "higher"]):
                self.lamp.set_brightness(min(100, self.lamp.get_brightness() + 20))
                self.manual_override_time = time.time()
                return True
            if any(kw in text for kw in ["down", "decrease", "lower"]):
                self.lamp.set_brightness(max(0, self.lamp.get_brightness() - 20))
                self.manual_override_time = time.time()
                return True

        # --- Pan ---
        if any(kw in text for kw in ["pan", "rotate", "look"]):
            if "left" in text:
                self.lamp.set_pan_angle(max(0, self.lamp.get_pan_angle() - 30))
                return True
            if "right" in text:
                self.lamp.set_pan_angle(min(180, self.lamp.get_pan_angle() + 30))
                return True
            if "center" in text or "centre" in text:
                self.lamp.set_pan_angle(90)
                return True

        # --- Smart mode ---
        if "smart mode on" in text or "enable smart" in text:
            self.smart_mode_enabled = True
            return True
        if "smart mode off" in text or "disable smart" in text:
            self.smart_mode_enabled = False
            return True

        return False

    def cleanup(self):
        self.tracker.close()
