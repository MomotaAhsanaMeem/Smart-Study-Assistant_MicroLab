"""
Shared Hardware Layer
=====================
Provides thread-safe, singleton-style access to hardware that must be
shared across feature modules: camera and microphone.

The camera is opened/closed on demand by the SystemManager.
The voice engine runs a single background listener thread.
"""

import threading
import time
import logging
import shutil
import cv2
import numpy as np
import subprocess
import sys

logger = logging.getLogger("SharedHardware")


# =========================================================================
#  Shared Camera — wraps PiCameraStream or V4L2 with thread-safe reads
# =========================================================================

class SharedCamera:
    """Thread-safe camera that can be opened/closed on demand."""

    def __init__(self, width=640, height=480, framerate=30):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.frame_size = int(width * height * 1.5)  # YUV420

        self._process = None       # rpicam-vid subprocess
        self._v4l2_cap = None      # OpenCV V4L2 capture
        self._latest_frame = None
        self._lock = threading.Lock()
        self._running = False
        self._reader_thread = None
        self._is_pi_camera = False

    # ---- public API ------------------------------------------------

    def open(self) -> bool:
        """Open the camera (tries Pi camera, then V4L2 fallback). Returns True on success."""
        if self._running:
            return True

        # --- Try rpicam-vid (Raspberry Pi 5 hardware-accelerated) ---
        if shutil.which("rpicam-vid"):
            logger.info("Opening PiCameraStream (rpicam-vid)...")
            cmd = [
                "rpicam-vid", "-t", "0", "--codec", "yuv420",
                "--width", str(self.width), "--height", str(self.height),
                "--framerate", str(self.framerate),
                "-o", "-", "--nopreview", "--denoise", "cdn_off",
            ]
            try:
                self._process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**7
                )
                self._is_pi_camera = True
                self._running = True
                self._reader_thread = threading.Thread(target=self._read_pi_camera, daemon=True)
                self._reader_thread.start()

                # Wait up to 3 s for the first frame
                t0 = time.time()
                while self._latest_frame is None and time.time() - t0 < 3.0:
                    time.sleep(0.1)

                if self._latest_frame is not None:
                    logger.info("PiCameraStream opened successfully.")
                    return True
                else:
                    self.close()  # timed-out, fall through to V4L2
            except FileNotFoundError:
                pass

        # --- Fallback: standard V4L2 via OpenCV ---
        logger.warning("Trying V4L2 camera fallback...")
        for idx in [0, 1, 2, 4, 6]:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    self._v4l2_cap = cap
                    self._is_pi_camera = False
                    self._running = True
                    self._reader_thread = threading.Thread(target=self._read_v4l2, daemon=True)
                    self._reader_thread.start()
                    logger.info(f"V4L2 camera opened at index {idx}.")
                    return True
                cap.release()

        logger.error("Failed to open any camera.")
        return False

    def read(self):
        """Return (success: bool, frame: ndarray | None)."""
        with self._lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy()
            return False, None

    def is_open(self) -> bool:
        return self._running

    def close(self):
        """Release camera resources."""
        self._running = False
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._v4l2_cap:
            self._v4l2_cap.release()
            self._v4l2_cap = None
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self._latest_frame = None
        logger.info("Camera closed.")

    # ---- internal reader threads -----------------------------------

    def _read_pi_camera(self):
        while self._running and self._process and self._process.poll() is None:
            try:
                raw = self._process.stdout.read(self.frame_size)
                if len(raw) != self.frame_size:
                    time.sleep(0.01)
                    continue
                yuv = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (int(self.height * 1.5), self.width)
                )
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                with self._lock:
                    self._latest_frame = bgr
            except Exception as e:
                logger.error(f"PiCameraStream read error: {e}")
                break

    def _read_v4l2(self):
        while self._running and self._v4l2_cap and self._v4l2_cap.isOpened():
            ok, frame = self._v4l2_cap.read()
            if ok:
                with self._lock:
                    self._latest_frame = frame
            else:
                time.sleep(0.05)


# =========================================================================
#  Shared Voice Engine — three-layer AI pipeline
#
#  Layer 1 — STT  : Vosk (offline) → fallback: Google Speech Recognition
#  Layer 2 — NLU  : FunctionGemma 270M (LLM function calling, local GGUF)
#  Layer 3 — Route: SystemManager dispatcher → fallback: keyword matching
# =========================================================================

class SharedVoiceEngine:
    """
    Runs ONE background listener thread on the microphone.

    Pipeline
    --------
    [mic] → STT (Vosk / Google) → FunctionGemma → SystemManager dispatcher
                                                  → keyword fallback

    The engine exposes two callbacks:
      • command_callback(text)       — original keyword-match router (fallback)
      • function_dispatch_callback   — set by SystemManager for AI-resolved calls
    """

    def __init__(self, command_callback, function_dispatch_callback=None):
        """
        Parameters
        ----------
        command_callback : callable(str)
            Called with raw transcript text for keyword-match fallback routing.
        function_dispatch_callback : callable(FunctionCall) | None
            If set, called first with the FunctionGemma parsed result.
            If the call.name == 'unknown', falls through to command_callback.
        """
        self.command_callback          = command_callback
        self.function_dispatch_callback = function_dispatch_callback
        self.running                   = False
        self.thread                    = None
        self.device_index              = self._find_microphone()

        # ── Layer 1: STT engines ──
        self._vosk_stt    = None   # Vosk offline STT (preferred)
        self._sr_engine   = None   # speech_recognition Google STT (fallback)
        self._stt_backend = "none" # "vosk" | "google" | "none"
        self._init_stt()

        # ── Layer 2: VoiceNLU Engine (Primary) ──
        self._nlu_engine = None
        self._init_voice_nlu()

        # ── Layer 2b: FunctionGemma (Experimental) ──
        self._fg_engine = None
        self._init_functiongemma()

    # ----------------------------------------------------------------
    #  Initialisation helpers
    # ----------------------------------------------------------------

    def _init_stt(self) -> None:
        """Try Vosk first; fall back to Google Speech Recognition."""
        try:
            from modules.vosk_stt import VoskSTT
            stt = VoskSTT(device_index=self.device_index)
            if stt.available:
                self._vosk_stt    = stt
                self._stt_backend = "vosk"
                logger.info("🎙️ STT backend: Vosk (offline)")
                return
        except Exception as e:
            logger.debug(f"Vosk STT init error: {e}")

        # Vosk unavailable — try speech_recognition (Google)
        try:
            import speech_recognition as sr
            self._sr_engine = sr.Recognizer()
            self._sr_engine.energy_threshold         = 1500
            self._sr_engine.dynamic_energy_threshold = True
            self._stt_backend = "google"
            logger.info("🎙️ STT backend: Google Speech Recognition (online)")
        except ImportError:
            logger.warning(
                "No STT backend available."
                " Install vosk & pyaudio (run ./install_functiongemma.sh) or"
                " 'pip install SpeechRecognition pyaudio'."
            )

    def _init_voice_nlu(self) -> None:
        """Load the fast deterministic VoiceNLU engine."""
        try:
            from modules.voice_nlu import VoiceNLU
            self._nlu_engine = VoiceNLU()
            logger.info("🧠 VoiceNLU Engine loaded (Primary).")
        except Exception as e:
            logger.warning(f"VoiceNLU init error: {e}")

    def _init_functiongemma(self) -> None:
        """Load FunctionGemma 270M; degrade gracefully if model missing."""
        try:
            from modules.function_gemma import FunctionGemmaEngine
            self._fg_engine = FunctionGemmaEngine()
            if self._fg_engine.available:
                logger.info("🤖 FunctionGemma loaded (Experimental Fallback).")
                # Warmup in background so it doesn't block start()
                threading.Thread(
                    target=self._fg_engine.warmup,
                    daemon=True,
                    name="FunctionGemma-Warmup",
                ).start()
            else:
                logger.info(
                    "🤖 FunctionGemma not available (model missing). "
                    "VoiceNLU handles all intents. "
                    "Run ./install_functiongemma.sh if you want to test the LLM."
                )
        except Exception as e:
            logger.warning(f"FunctionGemma init error: {e}")

    # ---- microphone detection --------------------------------------

    @staticmethod
    def _find_microphone():
        """Dynamically locate the best audio input device."""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            categories = {"default": [], "bluetooth": [], "usb": [], "other": []}

            for i in range(p.get_device_count()):
                try:
                    info = p.get_device_info_by_index(i)
                    name = info.get("name", "").lower()
                    if info.get("maxInputChannels", 0) > 0:
                        if any(kw in name for kw in ["default", "pulse", "pipewire"]):
                            categories["default"].append((i, info.get("name")))
                        elif any(kw in name for kw in ["bluetooth", "bluez", "headset", "handsfree"]):
                            categories["bluetooth"].append((i, info.get("name")))
                        elif any(kw in name for kw in ["usb", "pnp", "mic", "microphone"]):
                            categories["usb"].append((i, info.get("name")))
                        else:
                            categories["other"].append((i, info.get("name")))
                except Exception:
                    continue

            p.terminate()

            for cat in ["default", "bluetooth", "usb", "other"]:
                if categories[cat]:
                    idx, nm = categories[cat][0]
                    logger.info(f"Audio input: '{nm}' (index {idx})")
                    return idx
        except Exception as e:
            logger.warning(f"Error finding microphone: {e}")

        logger.warning("No audio input device found — using system default (None).")
        return None

    # ---- start / stop ----------------------------------------------

    def start(self):
        if self._stt_backend == "none":
            logger.error("No STT backend available — voice engine not started.")
            return
        self.running = True
        self.thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="VoiceEngine",
        )
        self.thread.start()
        logger.info(f"🎙️ Voice engine started (STT: {self._stt_backend}).")

    def stop(self):
        self.running = False
        if self._vosk_stt is not None:
            self._vosk_stt.cleanup()
        if self._fg_engine is not None:
            self._fg_engine.unload()

    # ---- background listener loop ----------------------------------

    def _listen_loop(self):
        """Main loop — listens forever, runs the three-layer pipeline."""
        if self._stt_backend == "vosk":
            self._vosk_loop()
        else:
            self._google_loop()

    # ── Vosk loop (offline, preferred) ──────────────────────────────

    def _vosk_loop(self):
        logger.info("🎙️ Vosk listen loop running…")
        while self.running:
            try:
                text = self._vosk_stt.listen_once(timeout=6.0, phrase_time_limit=4.0)
                if not self.running:
                    break
                if text:
                    self._dispatch(text)
            except Exception as e:
                logger.error(f"Vosk loop error: {e}")
                time.sleep(1.0)

    # ── Google STT loop (online, fallback) ──────────────────────────

    def _google_loop(self):
        import speech_recognition as sr
        calibrated = False
        logger.info("🎙️ Google STT listen loop running…")

        while self.running:
            try:
                with sr.Microphone(device_index=self.device_index) as source:
                    if not calibrated:
                        logger.info("🔊 Calibrating microphone for ambient noise…")
                        self._sr_engine.adjust_for_ambient_noise(source, duration=1.5)
                        calibrated = True
                    audio = self._sr_engine.listen(source, timeout=5.0, phrase_time_limit=4.0)

                if not self.running:
                    break

                text = self._sr_engine.recognize_google(audio).lower()
                self._dispatch(text)

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                logger.error(f"Google STT API error: {e}")
                time.sleep(5.0)
            except Exception as e:
                logger.error(f"Google STT loop error: {e}")
                time.sleep(2.0)

    # ── Three-layer dispatch ─────────────────────────────────────────

    def _dispatch(self, text: str) -> None:
        """
        Run the three-layer pipeline for a recognised transcript.

        1. Log the raw transcript.
        2. VoiceNLU → structured function call (Fast & Deterministic).
        3. FunctionGemma → fallback (if loaded and VoiceNLU failed).
        4. If function_dispatch_callback set AND call.name != 'unknown' → call it.
        5. Otherwise fall through to command_callback (keyword matching).
        """
        logger.info(f"🎙️ Voice heard: '{text}'")

        # ── Layer 2: VoiceNLU (Primary) ──
        if self._nlu_engine is not None and self.function_dispatch_callback is not None:
            call = self._nlu_engine.parse(text)
            if call.name != "unknown":
                try:
                    self.function_dispatch_callback(call)
                except Exception as e:
                    logger.error(f"function_dispatch_callback error (VoiceNLU): {e}")
                return   # consumed by VoiceNLU

        # ── Layer 2b: FunctionGemma NLU (Experimental Fallback) ──
        if (
            self._fg_engine is not None
            and self._fg_engine.available
            and self.function_dispatch_callback is not None
        ):
            call = self._fg_engine.parse(text)
            if call.name != "unknown":
                try:
                    self.function_dispatch_callback(call)
                except Exception as e:
                    logger.error(f"function_dispatch_callback error: {e}")
                return   # consumed by FunctionGemma

        # ── Layer 3: keyword fallback ──
        try:
            self.command_callback(text)
        except Exception as e:
            logger.error(f"command_callback error: {e}")
