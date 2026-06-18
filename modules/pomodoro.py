"""
Pomodoro Timer Module
======================
A headless (no Tkinter) Pomodoro timer that runs entirely via
background threads, exposing state for the web dashboard and
voice/button control.

Cross-feature callbacks:
    on_study_start  → SystemManager turns lamp to 100 %, activates focus tracker
    on_break_start  → SystemManager dims lamp to 30 %, deactivates focus tracker
    on_session_end  → SystemManager flashes lamp alert
"""

import time
import sys
import threading
import logging

from modules.base import FeatureModule
import hardware_config

logger = logging.getLogger("Pomodoro")

try:
    from gpiozero import Button, PWMOutputDevice
    GPIO_AVAILABLE = True
except (ImportError, NotImplementedError, Exception):
    GPIO_AVAILABLE = False

STUDY_TIME = 25 * 60   # 25 minutes
BREAK_TIME = 5 * 60    # 5 minutes


class PomodoroModule(FeatureModule):
    """Pomodoro timer with hardware buzzer/button and cross-feature callbacks."""

    def __init__(self):
        self.state = "IDLE"           # IDLE | STUDY | BREAK
        self.time_left = STUDY_TIME
        self.is_running = False
        self.current_mode = "STUDY"   # what the NEXT session will be
        self._lock = threading.Lock()
        self._thread = None

        # Cross-feature callbacks (set by SystemManager)
        self.on_study_start = None
        self.on_break_start = None
        self.on_session_end = None    # receives next_mode as arg

        # Hardware: buzzer + physical button
        self.buzzer = None
        self.button = None
        if GPIO_AVAILABLE:
            try:
                self.buzzer = PWMOutputDevice(hardware_config.POMODORO_BUZZER_PIN, active_high=False, frequency=2300)
                self.button = Button(
                    hardware_config.POMODORO_BUTTON_PIN,
                    pull_up=True, bounce_time=0.1,
                )
                self.button.when_pressed = self.toggle
                logger.info("Pomodoro hardware (buzzer + button) initialised.")
            except Exception as e:
                logger.error(f"Pomodoro GPIO init error: {e}")

    # ----------------------------------------------------------------
    #  Timer Controls
    # ----------------------------------------------------------------

    def start_timer(self):
        with self._lock:
            if self.is_running:
                return
            self.is_running = True
            self.state = self.current_mode
            logger.info(f"Pomodoro started: {self.state}")

            if self.state == "STUDY" and self.on_study_start:
                self.on_study_start()
            elif self.state == "BREAK" and self.on_break_start:
                self.on_break_start()

            # Spin up the countdown thread (if not already alive)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._countdown, daemon=True)
                self._thread.start()

    def pause_timer(self):
        with self._lock:
            if self.is_running:
                self.is_running = False
                self.state = "IDLE"
                logger.info("Pomodoro paused.")

    def reset_timer(self):
        with self._lock:
            self.is_running = False
            self.state = "IDLE"
            self.current_mode = "STUDY"
            self.time_left = STUDY_TIME
            logger.info("Pomodoro reset.")

    def toggle(self):
        """Toggle start/pause — bound to the physical button."""
        if self.is_running:
            self.pause_timer()
        else:
            self.start_timer()

    # ----------------------------------------------------------------
    #  Internal countdown
    # ----------------------------------------------------------------

    def _countdown(self):
        while True:
            time.sleep(1)
            with self._lock:
                if not self.is_running:
                    return
                if self.time_left > 0:
                    self.time_left -= 1
                else:
                    self._handle_session_end()
                    return

    def _handle_session_end(self):
        """Called (under lock) when time_left hits zero."""
        self.is_running = False
        self._trigger_alarm()

        if self.current_mode == "STUDY":
            logger.info("Study session complete → switching to Break.")
            self.current_mode = "BREAK"
            self.time_left = BREAK_TIME
        else:
            logger.info("Break complete → switching to Study.")
            self.current_mode = "STUDY"
            self.time_left = STUDY_TIME

        self.state = "IDLE"

        if self.on_session_end:
            self.on_session_end(self.current_mode)

    def _trigger_alarm(self):
        def _beep():
            if self.buzzer:
                for _ in range(3):
                    self.buzzer.value = 0.2
                    time.sleep(0.5)
                    self.buzzer.value = 0.0
                    time.sleep(0.5)
            else:
                for _ in range(3):
                    sys.stdout.write("\a")
                    sys.stdout.flush()
                    time.sleep(1)
        threading.Thread(target=_beep, daemon=True).start()

    # ----------------------------------------------------------------
    #  FeatureModule interface
    # ----------------------------------------------------------------

    def get_state(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "time_left": self.time_left,
                "is_running": self.is_running,
                "current_mode": self.current_mode,
                "time_display": f"{self.time_left // 60:02d}:{self.time_left % 60:02d}",
            }

    def handle_voice_command(self, text: str) -> bool:
        if any(kw in text for kw in ["start timer", "start pomodoro", "begin study", "resume timer"]):
            self.start_timer()
            return True
        if any(kw in text for kw in ["pause timer", "pause pomodoro", "stop timer"]):
            self.pause_timer()
            return True
        if any(kw in text for kw in ["reset timer", "reset pomodoro", "restart timer"]):
            self.reset_timer()
            return True
        return False

    def cleanup(self):
        self.is_running = False
        if self.buzzer:
            try:
                self.buzzer.close()
            except Exception:
                pass
        if self.button:
            try:
                self.button.close()
            except Exception:
                pass
