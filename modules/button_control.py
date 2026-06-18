"""
Physical Push Button Control Module
=====================================
Two push buttons for hands-free lamp and servo control.

  LAMP BUTTON (GPIO 22):
    • Hold ≥0.8s     → Toggle lamp ON (starts at 20%) / OFF  [fires while still holding]
    • Short click    → Cycle brightness: 20 → 40 → 60 → 80 → 100 → 20 → …
    • In OCR mode: short click → capture page

  ROTATE BUTTON (GPIO 23):
    • Short click    → Rotate right +20°
    • Hold ≥0.8s     → Rotate left −20°, then repeats every 0.5s while still held

  COMBO (both held ≥1.5s) → Enter OCR document scan mode  [fires while still holding]

Design
------
All long-press logic is driven by a single background watcher thread that polls
every 50 ms.  Actions fire the moment the hold threshold is met — while the button
is still physically pressed — not on release.

Release only handles the short-click case: if the long-press action has NOT already
fired (i.e. button released before the threshold), it's treated as a short click.
"""

import time
import threading
import logging

from modules.base import FeatureModule
from lamp_controller import LampState
import hardware_config

logger = logging.getLogger("ButtonControl")

try:
    from gpiozero import Button
    GPIO_AVAILABLE = True
except (ImportError, NotImplementedError, Exception):
    GPIO_AVAILABLE = False

# --------------------------------------------------------------------------
# Timing constants (seconds)
# --------------------------------------------------------------------------
LONG_PRESS_THRESHOLD = 0.8   # how long to hold before long-press fires
COMBO_HOLD_THRESHOLD = 1.5   # how long both must be held for OCR combo
ROTATE_REPEAT_INTERVAL = 0.5 # interval between repeated rotate-left steps

# Brightness cycle levels (%)
BRIGHTNESS_STEPS = [20, 40, 60, 80, 100]


class ButtonControlModule(FeatureModule):
    """Handles two physical push buttons for lamp power/brightness and servo pan."""

    def __init__(self, lamp, ocr):
        self.lamp = lamp
        self.ocr  = ocr

        # Callbacks wired by SystemManager
        self.on_enter_ocr_mode = None   # () → opens camera + starts OCR
        self.on_ocr_capture    = None   # () → grabs frame and stores it

        # ---- Shared state (read by watcher thread, written by press/release) ----
        self._ocr_mode        = False
        self._brightness_index = 0          # index into BRIGHTNESS_STEPS

        # Per-button press timestamps (0.0 = button is not pressed)
        self._lamp_press_time   = 0.0
        self._rotate_press_time = 0.0

        # Flags to prevent double-action:
        #   True  = long-press already fired while holding → release does nothing
        #   False = long-press not yet fired → release triggers short-click
        self._lamp_long_fired   = False
        self._rotate_long_fired = False
        self._combo_fired       = False

        # Timestamp of the last rotate-left repeat (for continuous rotation)
        self._rotate_last_repeat = 0.0

        # ---- Hardware ----
        self._lamp_btn   = None
        self._rotate_btn = None
        self._running    = False

        if GPIO_AVAILABLE:
            try:
                self._lamp_btn = Button(
                    hardware_config.LAMP_BUTTON_PIN,
                    pull_up=True, bounce_time=0.05,
                )
                self._rotate_btn = Button(
                    hardware_config.ROTATE_BUTTON_PIN,
                    pull_up=True, bounce_time=0.05,
                )

                # Press → record timestamp; Release → handle short-click only
                self._lamp_btn.when_pressed   = self._on_lamp_pressed
                self._lamp_btn.when_released  = self._on_lamp_released
                self._rotate_btn.when_pressed  = self._on_rotate_pressed
                self._rotate_btn.when_released = self._on_rotate_released

                # Single watcher thread handles ALL long-press + combo logic
                self._running = True
                threading.Thread(
                    target=self._watcher, daemon=True, name="ButtonWatcher"
                ).start()

                logger.info(
                    f"ButtonControl initialised: Lamp=GPIO {hardware_config.LAMP_BUTTON_PIN}, "
                    f"Rotate=GPIO {hardware_config.ROTATE_BUTTON_PIN}"
                )
            except Exception as e:
                logger.error(f"ButtonControl GPIO init error: {e}")
        else:
            logger.warning("gpiozero not available — button control disabled.")

    # -----------------------------------------------------------------------
    #  Background watcher — fires long-press actions while buttons are held
    # -----------------------------------------------------------------------

    def _watcher(self):
        """
        Polls every 50 ms.  For each button that is currently held:
          • Once hold duration ≥ LONG_PRESS_THRESHOLD and action not yet fired → fire immediately.
          • For rotate-left: also repeats every ROTATE_REPEAT_INTERVAL while still held.
          • Combo: both held ≥ COMBO_HOLD_THRESHOLD → enter OCR mode.
        """
        while self._running:
            now = time.time()

            # ---- Combo check (highest priority — suppress individual actions) ----
            lamp_held   = self._lamp_press_time > 0
            rotate_held = self._rotate_press_time > 0

            # ---- Combo check — both buttons held (highest priority) ----
            if lamp_held and rotate_held and not self._combo_fired:
                earliest = min(self._lamp_press_time, self._rotate_press_time)
                if now - earliest >= COMBO_HOLD_THRESHOLD:
                    self._combo_fired       = True
                    self._lamp_long_fired   = True   # suppress individual actions
                    self._rotate_long_fired = True
                    self._handle_combo()

            # ---- Lamp long-press ----
            # SKIPPED when both buttons are held — that is the combo sequence.
            # Fires once as soon as threshold is met while lamp button is held alone.
            if (lamp_held
                    and not rotate_held           # <-- key guard: not a combo attempt
                    and not self._lamp_long_fired
                    and (now - self._lamp_press_time) >= LONG_PRESS_THRESHOLD):
                self._lamp_long_fired = True
                self._handle_lamp_long()

            # ---- Rotate long-press + continuous repeat ----
            # SKIPPED when both buttons are held — that is the combo sequence.
            if rotate_held and not lamp_held and not self._combo_fired:  # <-- key guard
                held_duration = now - self._rotate_press_time

                if not self._rotate_long_fired:
                    # First fire: threshold just met
                    if held_duration >= LONG_PRESS_THRESHOLD:
                        self._rotate_long_fired  = True
                        self._rotate_last_repeat = now
                        self._handle_rotate_left()
                else:
                    # Subsequent fires: repeat every ROTATE_REPEAT_INTERVAL
                    if (now - self._rotate_last_repeat) >= ROTATE_REPEAT_INTERVAL:
                        self._rotate_last_repeat = now
                        self._handle_rotate_left()

            time.sleep(0.05)

    # -----------------------------------------------------------------------
    #  Long-press action handlers
    # -----------------------------------------------------------------------

    def _handle_combo(self):
        """Both buttons held → enter OCR scan mode."""
        logger.info("Button combo: Entering OCR document scan mode.")
        self._ocr_mode = True
        if self.on_enter_ocr_mode:
            self.on_enter_ocr_mode()

    def _handle_lamp_long(self):
        """Lamp held ≥ threshold → toggle ON / OFF immediately."""
        if self.lamp.get_state() == LampState.ON:
            self.lamp.turn_off()
            logger.info("Lamp button held: Lamp OFF.")
        else:
            self.lamp.turn_on()
            self.lamp.set_brightness(BRIGHTNESS_STEPS[0])
            self._brightness_index = 0
            logger.info("Lamp button held: Lamp ON at 20%.")

    def _handle_rotate_left(self):
        """Rotate-left step — only when lamp is ON."""
        if self.lamp.get_state() != LampState.ON:
            return
        new_angle = max(0.0, self.lamp.get_pan_angle() - 20.0)
        self.lamp.move_pan_direct(new_angle)
        logger.info(f"Rotate button held: Pan left → {int(new_angle)}°.")

    # -----------------------------------------------------------------------
    #  Press callbacks — just record the timestamp
    # -----------------------------------------------------------------------

    def _on_lamp_pressed(self):
        self._lamp_press_time   = time.time()
        self._lamp_long_fired   = False

    def _on_rotate_pressed(self):
        self._rotate_press_time  = time.time()
        self._rotate_long_fired  = False
        self._rotate_last_repeat = 0.0

    # -----------------------------------------------------------------------
    #  Release callbacks — short-click only (long-press already handled above)
    # -----------------------------------------------------------------------

    def _on_lamp_released(self):
        press_time = self._lamp_press_time
        long_fired = self._lamp_long_fired

        # Clear press state
        self._lamp_press_time = 0.0

        # Reset combo flag when both buttons are released
        if self._combo_fired and self._rotate_press_time == 0:
            self._combo_fired = False

        # If long-press already fired (or combo), do nothing on release
        if long_fired:
            return

        # --- Short click ---
        if self._ocr_mode:
            # In OCR mode: short click captures a page
            logger.info("Lamp button click (OCR mode): Capturing page.")
            if self.on_ocr_capture:
                self.on_ocr_capture()
        else:
            # Normal mode: cycle brightness (lamp must be ON)
            if self.lamp.get_state() == LampState.ON:
                self._brightness_index = (self._brightness_index + 1) % len(BRIGHTNESS_STEPS)
                new_brightness = BRIGHTNESS_STEPS[self._brightness_index]
                self.lamp.set_brightness(new_brightness)
                logger.info(f"Lamp button click: Brightness → {new_brightness}%.")

    def _on_rotate_released(self):
        long_fired = self._rotate_long_fired

        # Clear press state
        self._rotate_press_time  = 0.0
        self._rotate_last_repeat = 0.0

        # Reset combo flag when both buttons are released
        if self._combo_fired and self._lamp_press_time == 0:
            self._combo_fired = False

        # If long-press already fired (or combo), do nothing on release
        if long_fired:
            return

        # --- Short click → rotate RIGHT ---
        if self.lamp.get_state() != LampState.ON:
            return
        new_angle = min(180.0, self.lamp.get_pan_angle() + 20.0)
        self.lamp.move_pan_direct(new_angle)
        logger.info(f"Rotate button click: Pan right → {int(new_angle)}°.")

    # -----------------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------------

    def exit_ocr_mode(self):
        """Called by SystemManager when OCR scan completes or is cancelled."""
        self._ocr_mode = False
        logger.info("Button control: Exited OCR mode.")

    # -----------------------------------------------------------------------
    #  FeatureModule interface
    # -----------------------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "ocr_mode":       self._ocr_mode,
            "brightness_step": BRIGHTNESS_STEPS[self._brightness_index],
        }

    def handle_voice_command(self, text: str) -> bool:
        return False   # buttons don't respond to voice

    def cleanup(self):
        self._running = False
        for btn in (self._lamp_btn, self._rotate_btn):
            if btn:
                try:
                    btn.close()
                except Exception:
                    pass
        logger.info("ButtonControl cleaned up.")
