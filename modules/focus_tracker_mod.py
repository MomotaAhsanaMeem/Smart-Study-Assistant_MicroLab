"""
Focus Tracker Module
=====================
Uses MediaPipe Face Mesh to estimate head pose and determine whether
the user is FOCUSED, DISTRACTED, or absent (NO_FACE).

Designed to be activated/deactivated by the SystemManager —
e.g. automatically turned on when a Pomodoro study session starts.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import logging

from modules.base import FeatureModule

logger = logging.getLogger("FocusTracker")


class FocusTrackerModule(FeatureModule):
    """Processes camera frames to detect user focus via head-pose estimation."""

    def __init__(self):
        # Module state
        self.active = False
        self.status = "NO_FACE"  # FOCUSED | DISTRACTED | NO_FACE
        self.distracted_start_time = None
        self.distracted_seconds = 0.0
        self.distraction_count = 0
        self.last_alert_time = 0.0

        # Callback for cross-feature alerts (set by SystemManager)
        self.on_distraction_alert = None

        # Head-pose thresholds (degrees)
        self.max_yaw = 15.0
        self.min_yaw = -15.0
        self.max_pitch = 20.0
        self.min_pitch = -10.0
        self.distraction_threshold_sec = 30.0

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Landmark indices used for solvePnP head-pose:
        # Nose(1), L-Eye(33), R-Eye(263), Chin(152), L-Mouth(61), R-Mouth(291)
        self._target_indices = [1, 33, 263, 152, 61, 291]

        # Store last results for frame annotation (web feed)
        self._last_results = None

        # MediaPipe drawing utilities
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

    # ----------------------------------------------------------------
    #  Activate / Deactivate
    # ----------------------------------------------------------------

    def activate(self):
        self.active = True
        self.distracted_start_time = None
        self.distracted_seconds = 0.0
        logger.info("Focus tracker activated.")

    def deactivate(self):
        self.active = False
        self.distracted_start_time = None
        self.distracted_seconds = 0.0
        logger.info("Focus tracker deactivated.")

    # ----------------------------------------------------------------
    #  Frame Processing  (called by SystemManager camera loop)
    # ----------------------------------------------------------------

    def process_frame(self, frame):
        """Analyse a BGR frame and update internal status."""
        if not self.active:
            return

        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.face_mesh.process(frame_rgb)
        self._last_results = results

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            pitch, yaw = self._get_head_pose(face, w, h)

            if pitch is not None and yaw is not None:
                if (yaw < self.min_yaw or yaw > self.max_yaw
                        or pitch < self.min_pitch or pitch > self.max_pitch):
                    self.status = "DISTRACTED"
                else:
                    self.status = "FOCUSED"
            else:
                self.status = "NO_FACE"
        else:
            self.status = "NO_FACE"

        # ----------- distraction alert logic -----------
        if self.status in ("DISTRACTED", "NO_FACE"):
            if self.distracted_start_time is None:
                self.distracted_start_time = time.time()
            else:
                self.distracted_seconds = time.time() - self.distracted_start_time
                if self.distracted_seconds > self.distraction_threshold_sec:
                    if time.time() - self.last_alert_time >= 5.0:
                        self.distraction_count += 1
                        self.last_alert_time = time.time()
                        self._play_alert()
                        if self.on_distraction_alert:
                            self.on_distraction_alert()
        else:
            self.distracted_start_time = None
            self.distracted_seconds = 0.0

    # ----------------------------------------------------------------
    #  Head pose estimation (solvePnP)
    # ----------------------------------------------------------------

    def _get_head_pose(self, face_landmarks, img_w, img_h):
        """Returns (pitch, yaw) in degrees, or (None, None)."""
        if len(face_landmarks.landmark) < max(self._target_indices):
            return None, None

        face_2d, face_3d = [], []
        for idx in self._target_indices:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal = 1.0 * img_w
        cam_matrix = np.array([
            [focal, 0,     img_h / 2],
            [0,     focal, img_w / 2],
            [0,     0,     1],
        ], dtype=np.float64)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        try:
            ok, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            if not ok:
                return None, None
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            return angles[0] * 360, angles[1] * 360  # pitch, yaw
        except Exception:
            return None, None

    # ----------------------------------------------------------------
    #  Frame Annotation (for web MJPEG feed)
    # ----------------------------------------------------------------

    def annotate_frame(self, frame):
        """Draw face mesh and status overlay on the frame for web display."""
        display = frame.copy()
        h, w = display.shape[:2]

        if self._last_results and self._last_results.multi_face_landmarks:
            for face_landmarks in self._last_results.multi_face_landmarks:
                # Draw tessellation (subtle mesh)
                self._mp_drawing.draw_landmarks(
                    image=display,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._mp_drawing_styles
                        .get_default_face_mesh_tesselation_style(),
                )
                # Draw contours (eyes, lips, face oval)
                self._mp_drawing.draw_landmarks(
                    image=display,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._mp_drawing_styles
                        .get_default_face_mesh_contours_style(),
                )

        # --- Status overlay ---
        if self.status == "FOCUSED":
            color = (0, 220, 80)     # green
            label = "FOCUSED"
        elif self.status == "DISTRACTED":
            color = (0, 60, 255)     # red (BGR)
            label = "DISTRACTED"
        else:
            color = (0, 200, 230)    # yellow
            label = "NO FACE"

        # Background bar for text
        cv2.rectangle(display, (0, 0), (w, 44), (0, 0, 0), -1)
        cv2.putText(display, label, (14, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        # Distraction timer (if active)
        if self.distracted_seconds > 0:
            txt = f"Distracted: {self.distracted_seconds:.1f}s  |  Alerts: {self.distraction_count}"
            cv2.putText(display, txt, (14, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        return display

    # ----------------------------------------------------------------
    #  Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _play_alert():
        pass

    # ----------------------------------------------------------------
    #  FeatureModule interface
    # ----------------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "active": self.active,
            "status": self.status,
            "distracted_seconds": round(self.distracted_seconds, 1),
            "distraction_count": self.distraction_count,
        }

    def handle_voice_command(self, text: str) -> bool:
        if any(kw in text for kw in ["focus on", "start focus", "track focus", "enable focus"]):
            self.activate()
            return True
        if any(kw in text for kw in ["focus off", "stop focus", "disable focus"]):
            self.deactivate()
            return True
        return False

    def cleanup(self):
        try:
            self.face_mesh.close()
        except Exception:
            pass
