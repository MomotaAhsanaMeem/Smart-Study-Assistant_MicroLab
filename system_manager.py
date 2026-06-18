"""
System Manager — Central Orchestrator
=======================================
The "brain" of the MicroLab Smart Lamp System.

Responsibilities:
  • Initialises all shared hardware (camera, mic, servo, sensors)
  • Initialises and owns every feature module
  • Routes voice commands to the correct module
  • Manages camera priority (gesture > focus > ocr)
  • Implements cross-feature automation rules:
      – Pomodoro STUDY  → lamp 100 %, focus tracker ON
      – Pomodoro BREAK  → lamp  30 %, focus tracker OFF
      – Session end     → flash lamp alert
      – Distraction     → flash lamp alert
  • Serves the Flask-SocketIO web dashboard on port 5000
"""

import os
import time
import sys
import threading
import logging
from datetime import datetime
from collections import deque

from flask import Flask, render_template, jsonify, Response, send_file, send_from_directory
from flask_socketio import SocketIO

from shared_hardware import SharedCamera, SharedVoiceEngine
from modules.gesture_control import GestureControlModule
from modules.focus_tracker_mod import FocusTrackerModule
from modules.pomodoro import PomodoroModule
from modules.environment import EnvironmentModule
from modules.ocr_scanner import OCRScannerModule
from modules.button_control import ButtonControlModule

import hardware_config

# Import lamp hardware classes from the standalone controller
from lamp_controller import ServoLamp, MockLamp, SmartLightingSensor, LampState

import cv2

logger = logging.getLogger("SystemManager")

# ---------------------------------------------------------------------------
#  Flask + SocketIO setup
# ---------------------------------------------------------------------------
_base = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(_base, "web", "templates"),
    static_folder=os.path.join(_base, "web", "static"),
)
app.config["SECRET_KEY"] = "microlab-smart-lamp-2026"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ===========================================================================
class SystemManager:
    """Central orchestrator that glues all modules + hardware together."""

    def __init__(self):
        logger.info("=" * 60)
        logger.info("  MICROLAB SMART LAMP SYSTEM — Initialising…")
        logger.info("=" * 60)

        # Event log for the dashboard (most-recent 100 entries)
        self.event_log = deque(maxlen=100)
        self._log("System initialising…")

        # Camera mode: None | 'gesture' | 'focus' | 'ocr'
        self.camera_mode = None
        self._running = False

        # Thread-safe display frame for MJPEG web feed
        self._display_frame = None
        self._frame_lock = threading.Lock()

        # ---- Shared hardware ----
        self.camera = SharedCamera()
        self.lamp = self._create_lamp()
        self.smart_sensor = SmartLightingSensor(
            bus_num=hardware_config.I2C_BUS_NUMBER,
            address=hardware_config.PCF8591_ADDRESS,
        )

        # ---- OCR state: keep a history of all generated PDFs ----
        self._pdf_history = []   # list of {"filename": str, "path": str, "time": str}
        self._load_pdf_history()

        # ---- Feature modules ----
        self.gesture    = GestureControlModule(self.lamp, self.smart_sensor)
        self.focus      = FocusTrackerModule()
        self.pomodoro   = PomodoroModule()
        self.environment = EnvironmentModule()
        self.ocr        = OCRScannerModule()
        self.buttons    = ButtonControlModule(self.lamp, self.ocr)

        # ---- Cross-feature wiring ----
        self.pomodoro.on_study_start  = self._on_study_start
        self.pomodoro.on_break_start  = self._on_break_start
        self.pomodoro.on_session_end  = self._on_session_end
        self.focus.on_distraction_alert = self._on_distraction_alert
        self.buttons.on_enter_ocr_mode = self._on_button_ocr_mode
        self.buttons.on_ocr_capture    = self._on_button_ocr_capture
        self.ocr.on_pdf_ready          = self._on_ocr_pdf_ready

        # ---- Voice engine (last, after all modules) ----
        self.voice = SharedVoiceEngine(
            command_callback=self._handle_voice,
            function_dispatch_callback=self._dispatch_function_call,
        )

        # ---- Flask routes ----
        self._register_routes()

        self._log("All modules initialised ✓")
        logger.info("System initialisation complete.")

    # ------------------------------------------------------------------
    #  Lamp factory
    # ------------------------------------------------------------------
    def _create_lamp(self):
        try:
            lamp = ServoLamp(
                servo_pin=hardware_config.SERVO_PIN,
                brightness_pin=hardware_config.LAMP_BRIGHTNESS_PIN,
            )
            self._log("Servo lamp hardware connected.")
            return lamp
        except Exception as e:
            logger.warning(f"Servo lamp unavailable ({e}), using MockLamp.")
            self._log("Using mock lamp (no hardware).")
            return MockLamp()

    def _load_pdf_history(self):
        """Scan the current directory for previously generated PDFs to persist history."""
        try:
            cwd = os.getcwd()
            for filename in os.listdir(cwd):
                if filename.startswith("Scanned_Notes_") and filename.endswith(".pdf"):
                    pdf_path = os.path.join(cwd, filename)
                    mtime = os.path.getmtime(pdf_path)
                    ts = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
                    self._pdf_history.append({
                        "filename": filename,
                        "path": pdf_path,
                        "time": ts,
                        "mtime": mtime
                    })
            # Sort chronologically by mtime
            self._pdf_history.sort(key=lambda x: x["mtime"])
            # Clean up the internal mtime key
            for entry in self._pdf_history:
                entry.pop("mtime", None)
            if self._pdf_history:
                logger.info(f"Loaded {len(self._pdf_history)} existing PDFs from disk")
        except Exception as e:
            logger.error(f"Failed to load PDF history: {e}")


    # ------------------------------------------------------------------
    #  Event log helper
    # ------------------------------------------------------------------
    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = {"time": ts, "message": msg}
        self.event_log.append(entry)
        try:
            socketio.emit("log_event", entry)
        except Exception:
            pass

    # ==================================================================
    #  Cross-Feature Automation
    # ==================================================================

    def _on_study_start(self):
        """Pomodoro → STUDY: full brightness + activate focus tracker."""
        self._log("📚 Study session started!")
        self.lamp.turn_on()
        self.lamp.set_brightness(100)
        self.focus.activate()
        self._set_camera_mode("focus")

    def _on_break_start(self):
        """Pomodoro → BREAK: dim lamp + deactivate focus tracker."""
        self._log("☕ Break time! Lamp dimmed to 30 %.")
        self.focus.deactivate()
        self.lamp.set_brightness(30)
        self._set_camera_mode(None)

    def _on_session_end(self, next_mode):
        """Pomodoro session ended → flash lamp as alert."""
        self._log(f"⏱️ Session ended! Next: {next_mode}")
        self._flash_lamp()

    def _on_distraction_alert(self):
        """Focus tracker detected sustained distraction → flash lamp and beep."""
        self._log("⚠️ Focus alert — you're distracted!")
        self._flash_lamp()
        self._play_beep(duration=1.0)

    def _play_beep(self, duration=1.0):
        """Plays a short beep on the hardware buzzer, falling back to system bell."""
        def _beep():
            if self.pomodoro and self.pomodoro.buzzer:
                try:
                    self.pomodoro.buzzer.value = 0.2
                    time.sleep(duration)
                    self.pomodoro.buzzer.value = 0.0
                    return
                except Exception as e:
                    logger.error(f"Error playing hardware buzzer beep: {e}")
            
            # Fallback for terminal bell
            try:
                sys.stdout.write("\a")
                sys.stdout.flush()
            except Exception:
                pass
        threading.Thread(target=_beep, daemon=True).start()

    def _flash_lamp(self):
        """Rapidly toggle brightness to create a visual alert."""
        def _flash():
            original = self.lamp.get_brightness()
            for _ in range(3):
                self.lamp.set_brightness(100)
                time.sleep(0.3)
                self.lamp.set_brightness(10)
                time.sleep(0.3)
            self.lamp.set_brightness(original)
        threading.Thread(target=_flash, daemon=True).start()

    # ==================================================================
    #  Camera Mode Management
    # ==================================================================

    def _set_camera_mode(self, mode):
        """Switch the camera to serve a different module (or close it)."""
        if mode == self.camera_mode:
            return

        # --- Close camera ---
        if mode is None:
            if self.camera_mode == "gesture":
                self.gesture.deactivate()
            elif self.camera_mode == "focus":
                self.focus.deactivate()
            if self.camera.is_open():
                self.camera.close()
            self.camera_mode = None
            self._log("Camera closed.")
            return

        # --- Open camera if needed ---
        if not self.camera.is_open():
            if not self.camera.open():
                logger.error("Failed to open camera.")
                self._log("❌ Camera failed to open!")
                return

        # --- Deactivate previous owner ---
        if self.camera_mode == "gesture":
            self.gesture.deactivate()
        elif self.camera_mode == "focus":
            self.focus.deactivate()

        # --- Activate new owner ---
        self.camera_mode = mode
        if mode == "gesture":
            self.gesture.activate()
            self._log("🖐️ Gesture control activated.")
        elif mode == "focus":
            self.focus.activate()
            self._log("👁️ Focus tracking activated.")
        elif mode == "ocr":
            self._log("📄 OCR scanner camera ready.")

    # ==================================================================
    #  AI Function Dispatcher (FunctionGemma 270M)
    # ==================================================================

    def _dispatch_function_call(self, call) -> None:
        """
        Execute a structured FunctionCall produced by FunctionGemma 270M.

        This is the primary voice-command handler when the AI model is loaded.
        The keyword-match router (_handle_voice) only fires as a fallback when
        FunctionGemma returns 'unknown'.

        Parameters
        ----------
        call : FunctionCall
            Parsed function call with .name (str) and .args (dict).
        """
        name = call.name
        args = call.args

        self._log(f'🤖 AI: {name}({args if args else ""})')
        logger.info(f"AI dispatch: {name}({args})  [{call.latency_ms:.0f}ms]")

        # ---- Lamp control ----
        if name == "lamp_on":
            self.lamp.turn_on()
            self._log("💡 Lamp ON")

        elif name == "lamp_off":
            self.lamp.turn_off()
            self._log("💡 Lamp OFF")

        elif name == "set_brightness":
            value = args.get("value", 50)
            self.lamp.set_brightness(value)
            self.gesture.manual_override_time = time.time()
            self._log(f"💡 Brightness → {value}%")

        elif name == "increase_brightness":
            new_val = min(100, self.lamp.get_brightness() + 20)
            self.lamp.set_brightness(new_val)
            self.gesture.manual_override_time = time.time()
            self._log(f"💡 Brightness → {new_val}% (+20)")

        elif name == "decrease_brightness":
            new_val = max(0, self.lamp.get_brightness() - 20)
            self.lamp.set_brightness(new_val)
            self.gesture.manual_override_time = time.time()
            self._log(f"💡 Brightness → {new_val}% (-20)")

        elif name == "set_pan_angle":
            angle = args.get("angle", 90)
            self.lamp.move_pan_direct(angle)
            self._log(f"🔄 Pan → {angle}°")

        elif name == "pan_left":
            new_angle = max(0, self.lamp.get_pan_angle() - 30)
            self.lamp.set_pan_angle(new_angle)
            self._log(f"⬅️ Pan left → {new_angle}°")

        elif name == "pan_right":
            new_angle = min(180, self.lamp.get_pan_angle() + 30)
            self.lamp.set_pan_angle(new_angle)
            self._log(f"➡️ Pan right → {new_angle}°")

        elif name == "pan_center":
            self.lamp.set_pan_angle(90)
            self._log("🔄 Pan → center (90°)")

        elif name == "enable_smart_mode":
            self.gesture.smart_mode_enabled = True
            self._log("🧠 Smart auto-brightness: ON")

        elif name == "disable_smart_mode":
            self.gesture.smart_mode_enabled = False
            self._log("🧠 Smart auto-brightness: OFF (manual)")

        # ---- Gesture control ----
        elif name == "activate_gesture_control":
            self._set_camera_mode("gesture")
            self.lamp.turn_on()
            self._log("👋 Gesture control activated")

        elif name == "deactivate_gesture_control":
            if self.camera_mode == "gesture":
                self._set_camera_mode(None)
            self._log("👋 Gesture control deactivated")

        # ---- Focus tracker ----
        elif name == "activate_focus_tracker":
            self.focus.activate()
            self._set_camera_mode("focus")
            self._log("👁️ Focus tracker activated")

        elif name == "deactivate_focus_tracker":
            self.focus.deactivate()
            if self.camera_mode == "focus":
                self._set_camera_mode(None)
            self._log("👁️ Focus tracker deactivated")

        # ---- Pomodoro ----
        elif name == "start_pomodoro":
            self.pomodoro.start_timer()
            self._log("⏱️ Pomodoro started")

        elif name == "pause_pomodoro":
            self.pomodoro.pause_timer()
            self._log("⏱️ Pomodoro paused")

        elif name == "reset_pomodoro":
            self.pomodoro.reset_timer()
            self._log("⏱️ Pomodoro reset")

        # ---- OCR ----
        elif name == "start_scan":
            if self.ocr.start_scan():
                self._set_camera_mode("ocr")
            self._log("📄 OCR scan mode started")

        elif name == "capture_page":
            if self.camera_mode == "ocr":
                ok, frame = self.camera.read()
                if ok:
                    self.ocr.capture_page(frame)
                    n = len(self.ocr.captured_images)
                    self._log(f"📄 Page {n} captured")
                    socketio.emit("page_captured", {"index": n - 1, "total": n})

        elif name == "finish_scan":
            self.ocr.process_captures()
            self._set_camera_mode(None)
            self.buttons.exit_ocr_mode()
            self._log("📄 Processing scan with Gemini AI…")

        elif name == "cancel_scan":
            self.ocr.stop_scan()
            self._set_camera_mode(None)
            self.buttons.exit_ocr_mode()
            self._log("📄 Scan cancelled")

        # ---- Environment ----
        elif name == "get_environment_status":
            state = self.environment.get_state()
            temp = state.get("temp")
            hum  = state.get("humidity")
            if temp is not None and hum is not None:
                self._log(f"🌡️ Temp: {temp}°C  |  Humidity: {hum}%")
            else:
                self._log("🌡️ Environment sensor not available")

        else:
            logger.warning(f"Unhandled function call: {name}")

    # ==================================================================
    #  Voice Command Router (keyword fallback)
    # ==================================================================

    def _handle_voice(self, text):
        """Route a recognised voice command to the right module."""
        self._log(f'🎙️ Voice: "{text}"')

        # ---- System-level commands (highest priority) ----
        if any(kw in text for kw in ["activate gesture", "start gesture",
                                      "start camera", "wake"]):
            self._set_camera_mode("gesture")
            self.lamp.turn_on()
            return

        if any(kw in text for kw in ["scan document", "start scan"]):
            if self.ocr.start_scan():
                self._set_camera_mode("ocr")
            return

        if "capture" in text and self.camera_mode == "ocr":
            ok, frame = self.camera.read()
            if ok:
                self.ocr.capture_page(frame)
                self._log(f"📄 Page captured ({len(self.ocr.captured_images)})")
            return

        if any(kw in text for kw in ["process scan", "finish scan", "done scanning"]):
            self.ocr.process_captures()
            self._set_camera_mode(None)
            self.buttons.exit_ocr_mode()
            return

        # ---- Delegate to feature modules (first match wins) ----
        for module in [self.pomodoro, self.gesture, self.focus, self.ocr]:
            if module.handle_voice_command(text):
                return

    # ==================================================================
    #  Background Threads
    # ==================================================================

    def _camera_loop(self):
        """Reads camera frames and dispatches them to the active module."""
        while self._running:
            if self.camera_mode is None or not self.camera.is_open():
                # Clear display frame when camera is off
                with self._frame_lock:
                    self._display_frame = None
                time.sleep(0.1)
                continue

            ok, frame = self.camera.read()
            if not ok:
                time.sleep(0.05)
                continue

            if self.camera_mode == "gesture":
                event = self.gesture.process_frame(frame)

                # Annotate frame for web feed
                try:
                    annotated = self.gesture.annotate_frame(frame)
                    with self._frame_lock:
                        self._display_frame = annotated
                except Exception:
                    with self._frame_lock:
                        self._display_frame = frame

                if event:
                    etype = event.get("type")
                    if etype == "lamp_off_and_sleep":
                        self._log("✊ Fist → lamp off, sleeping.")
                        self._set_camera_mode(None)
                    elif etype == "gesture_timeout":
                        self._log("💤 Gesture timeout → sleeping.")
                        self.lamp.turn_off()
                        self._set_camera_mode(None)
                    elif etype == "lamp_on":
                        self._log("🖐️ Open hand → lamp on!")

            elif self.camera_mode == "focus":
                self.focus.process_frame(frame)

                # Annotate frame for web feed
                try:
                    annotated = self.focus.annotate_frame(frame)
                    with self._frame_lock:
                        self._display_frame = annotated
                except Exception:
                    with self._frame_lock:
                        self._display_frame = frame

            elif self.camera_mode == "ocr":
                # OCR captures on-demand, but still show raw feed
                with self._frame_lock:
                    self._display_frame = frame

            time.sleep(0.01)  # ~100 fps cap to avoid CPU saturation

    def _broadcast_loop(self):
        """Emit full system state to all connected web clients every 500 ms."""
        while self._running:
            try:
                socketio.emit("state_update", self._full_state())
            except Exception:
                pass
            time.sleep(0.5)

    def _full_state(self):
        return {
            "lamp":        self.gesture.get_state(),
            "pomodoro":    self.pomodoro.get_state(),
            "focus":       self.focus.get_state(),
            "environment": self.environment.get_state(),
            "ocr":         self.ocr.get_state(),
            "camera_mode": self.camera_mode,
            "events":      list(self.event_log),
        }

    # ==================================================================
    #  Flask + SocketIO Routes
    # ==================================================================

    def _register_routes(self):

        @app.route("/")
        def _index():
            return render_template("index.html")

        @app.route("/api/state")
        def _api_state():
            return jsonify(self._full_state())

        # ---- Live camera MJPEG stream ----
        @app.route("/video_feed")
        def _video_feed():
            def generate():
                while True:
                    with self._frame_lock:
                        frame = self._display_frame
                    if frame is not None:
                        ret, jpeg = cv2.imencode(
                            '.jpg', frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 65]
                        )
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n'
                                   + jpeg.tobytes() + b'\r\n')
                    time.sleep(0.05)   # ~20 fps to the browser
            return Response(
                generate(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        # ---- OCR: serve captured page thumbnail by index ----
        @app.route("/ocr/page/<int:idx>")
        def _ocr_page(idx):
            images = self.ocr.captured_images
            if idx < 0 or idx >= len(images):
                return Response("Not found", status=404)
            ret, jpeg = cv2.imencode('.jpg', images[idx],
                                     [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ret:
                return Response("Encode error", status=500)
            from io import BytesIO
            return send_file(BytesIO(jpeg.tobytes()), mimetype='image/jpeg')

        # ---- OCR: download a generated PDF by filename ----
        @app.route("/ocr/download/<filename>")
        def _ocr_download(filename):
            # Safety: only serve files recorded in our history
            for entry in self._pdf_history:
                if entry["filename"] == filename:
                    directory = os.path.dirname(entry["path"])
                    return send_from_directory(
                        directory, filename, as_attachment=True
                    )
            return Response("Not found", status=404)

        # ---- OCR: list of all generated PDFs ----
        @app.route("/api/pdfs")
        def _api_pdfs():
            return jsonify(self._pdf_history)

        # ---- Lamp controls ----
        @socketio.on("lamp_on")
        def _lamp_on():
            self.lamp.turn_on()
            self._log("💡 Lamp ON (dashboard)")

        @socketio.on("lamp_off")
        def _lamp_off():
            self.lamp.turn_off()
            self._log("💡 Lamp OFF (dashboard)")

        @socketio.on("lamp_brightness")
        def _brightness(data):
            self.lamp.set_brightness(int(data.get("value", 50)))
            self.gesture.manual_override_time = time.time()

        @socketio.on("lamp_pan")
        def _pan(data):
            angle = float(data.get("value", 90))
            self.lamp.move_pan_direct(angle)

        @socketio.on("smart_mode_toggle")
        def _smart():
            self.gesture.smart_mode_enabled = not self.gesture.smart_mode_enabled
            st = "ON" if self.gesture.smart_mode_enabled else "OFF"
            self._log(f"🧠 Smart mode: {st}")

        # ---- Gesture controls ----
        @socketio.on("gesture_activate")
        def _gesture_on():
            self._set_camera_mode("gesture")
            self.lamp.turn_on()

        @socketio.on("gesture_deactivate")
        def _gesture_off():
            if self.camera_mode == "gesture":
                self._set_camera_mode(None)

        # ---- Pomodoro controls ----
        @socketio.on("pomodoro_start")
        def _pomo_start():
            self.pomodoro.start_timer()
            self._log("⏱️ Pomodoro started (dashboard)")

        @socketio.on("pomodoro_pause")
        def _pomo_pause():
            self.pomodoro.pause_timer()
            self._log("⏱️ Pomodoro paused")

        @socketio.on("pomodoro_reset")
        def _pomo_reset():
            self.pomodoro.reset_timer()
            self._log("⏱️ Pomodoro reset")

        # ---- Focus controls ----
        @socketio.on("focus_toggle")
        def _focus():
            if self.focus.active:
                self.focus.deactivate()
                if self.camera_mode == "focus":
                    self._set_camera_mode(None)
                self._log("👁️ Focus tracker disabled")
            else:
                self.focus.activate()
                self._set_camera_mode("focus")
                self._log("👁️ Focus tracker enabled")

        # ---- OCR controls ----
        @socketio.on("ocr_start")
        def _ocr_start():
            if self.ocr.start_scan():
                self._set_camera_mode("ocr")

        @socketio.on("ocr_capture")
        def _ocr_capture():
            if self.camera_mode == "ocr" and self.camera.is_open():
                ok, frame = self.camera.read()
                if ok:
                    self.ocr.capture_page(frame)
                    n = len(self.ocr.captured_images)
                    self._log(f"📄 Page captured ({n} total)")
                    # Push thumbnail index to all browser clients
                    socketio.emit("page_captured", {"index": n - 1, "total": n})

        @socketio.on("ocr_process")
        def _ocr_process():
            self.ocr.process_captures()
            self._set_camera_mode(None)
            self.buttons.exit_ocr_mode()
            self._log("📄 Processing with Gemini AI…")

        @socketio.on("ocr_cancel")
        def _ocr_cancel():
            self.ocr.stop_scan()
            self._set_camera_mode(None)
            self.buttons.exit_ocr_mode()
            self._log("📄 Scan cancelled")

    # ==================================================================
    #  Start / Shutdown
    # ==================================================================

    def start(self):
        """Boot the entire system (blocks in the main thread)."""
        self._running = True

        # Background services
        self.voice.start()
        self.environment.start()

        threading.Thread(target=self._camera_loop, daemon=True).start()
        threading.Thread(target=self._broadcast_loop, daemon=True).start()

        host = hardware_config.WEB_SERVER_HOST
        port = hardware_config.WEB_SERVER_PORT

        # Detect AI pipeline status for banner display
        stt_backend = getattr(self.voice, "_stt_backend", "unknown")
        fg_available = (
            getattr(self.voice, "_fg_engine", None) is not None
            and getattr(self.voice._fg_engine, "available", False)
        )

        print("=" * 60)
        print("  🔮 MICROLAB SMART LAMP SYSTEM")
        print("=" * 60)
        print(f"  Dashboard  → http://0.0.0.0:{port}")
        print(f"  STT Engine → {'Vosk (offline)' if stt_backend == 'vosk' else 'Google STT (online)' if stt_backend == 'google' else 'None'}")
        print("  AI Voice   → VoiceNLU Smart Engine ✅ (Deterministic Intent Parsing)")
        print("─" * 60)
        print("  🎙️  Just speak naturally! Examples:")
        print("       \"make the light brighter\"")
        print("       \"start my focus session\"")
        print("       \"rotate the lamp to the left\"")
        print("       \"scan this document\"")
        print("       \"what's the temperature in here\"")
        print("=" * 60 + "\n")

        self._log("✅ System fully started!")

        # Flask-SocketIO blocks here (main thread)
        socketio.run(
            app,
            host=host,
            port=port,
            debug=False,
            allow_unsafe_werkzeug=True,
        )

    # ------------------------------------------------------------------
    #  Button Callbacks
    # ------------------------------------------------------------------

    def _on_button_ocr_mode(self):
        """Both buttons held → enter OCR scan mode."""
        if self.ocr.start_scan():
            self._set_camera_mode("ocr")
            self._log("📄 OCR scan mode activated (buttons).")

    def _on_button_ocr_capture(self):
        """Lamp button pressed in OCR mode → capture a page."""
        if self.camera_mode == "ocr" and self.camera.is_open():
            ok, frame = self.camera.read()
            if ok:
                self.ocr.capture_page(frame)
                n = len(self.ocr.captured_images)
                self._log(f"📄 Page captured ({n} total) — button.")
                socketio.emit("page_captured", {"index": n - 1, "total": n})

    def _on_ocr_pdf_ready(self, pdf_path: str):
        """Called by OCRScannerModule when a PDF is successfully generated."""
        filename = os.path.basename(pdf_path)
        ts = datetime.now().strftime("%H:%M:%S")
        entry = {"filename": filename, "path": pdf_path, "time": ts}
        self._pdf_history.append(entry)
        self._log(f"📄 PDF ready: {filename}")
        socketio.emit("pdf_ready", entry)

    def shutdown(self):
        """Gracefully release all resources."""
        logger.info("Shutting down…")
        self._running = False
        self.voice.stop()
        self.buttons.cleanup()
        self.gesture.cleanup()
        self.focus.cleanup()
        self.pomodoro.cleanup()
        self.environment.cleanup()
        self.ocr.cleanup()
        self.camera.close()
        logger.info("System shutdown complete.")
