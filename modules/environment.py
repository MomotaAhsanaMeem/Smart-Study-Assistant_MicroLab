"""
Environment Monitoring Module
==============================
Wraps the existing EnvironmentMonitor (DHT11) in a background thread,
keeping the latest readings + history available for the web dashboard.
"""

import time
import threading
import logging
import collections
from datetime import datetime

from modules.base import FeatureModule

logger = logging.getLogger("Environment")


class EnvironmentModule(FeatureModule):
    """Reads temperature/humidity in a background thread and exposes state."""

    def __init__(self):
        self.monitor = None
        self.temp = None
        self.humidity = None
        self.alerts = []

        self.temp_range = (15, 28)
        self.hum_range = (30, 65)

        # Rolling history for sparkline charts
        self.history_len = 50
        self.timestamps = collections.deque(maxlen=self.history_len)
        self.temp_history = collections.deque(maxlen=self.history_len)
        self.hum_history = collections.deque(maxlen=self.history_len)

        self._running = False
        self._thread = None

        # Attempt to initialise the DHT11 sensor
        try:
            import board
            import adafruit_dht
            import hardware_config
            from environment_monitor import EnvironmentMonitor

            self.monitor = EnvironmentMonitor(
                pin=hardware_config.DHT11_PIN,
                temp_range=self.temp_range,
                hum_range=self.hum_range,
                sensor_type=adafruit_dht.DHT11,
            )
            logger.info("DHT11 environment sensor initialised.")
        except Exception as e:
            logger.warning(f"Environment sensor not available: {e}")

    # ----------------------------------------------------------------
    #  Start / Stop background reading
    # ----------------------------------------------------------------

    def start(self):
        if self.monitor and not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            logger.info("Environment monitoring started (2.5 s interval).")

    def stop(self):
        self._running = False

    # ----------------------------------------------------------------
    #  Internal read loop
    # ----------------------------------------------------------------

    def _read_loop(self):
        while self._running:
            try:
                temp, hum = self.monitor.read_sensor(retries=5)
                if temp is not None and hum is not None:
                    self.temp = temp
                    self.humidity = hum
                    self.alerts = self.monitor.check_safety(temp, hum)

                    ts = datetime.now().strftime("%H:%M:%S")
                    self.timestamps.append(ts)
                    self.temp_history.append(temp)
                    self.hum_history.append(hum)
            except Exception as e:
                logger.error(f"Sensor read error: {e}")

            time.sleep(2.5)

    # ----------------------------------------------------------------
    #  Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _status_label(value, lo, hi):
        if value is None:
            return "UNKNOWN"
        if value > hi:
            return "HIGH"
        if value < lo:
            return "LOW"
        return "OPTIMAL"

    # ----------------------------------------------------------------
    #  FeatureModule interface
    # ----------------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "temp": round(self.temp, 1) if self.temp is not None else None,
            "humidity": round(self.humidity, 1) if self.humidity is not None else None,
            "alerts": self.alerts,
            "temp_range": self.temp_range,
            "hum_range": self.hum_range,
            "temp_status": self._status_label(self.temp, *self.temp_range),
            "hum_status": self._status_label(self.humidity, *self.hum_range),
            "temp_history": list(self.temp_history),
            "hum_history": list(self.hum_history),
            "timestamps": list(self.timestamps),
        }

    def handle_voice_command(self, text: str) -> bool:
        if any(kw in text for kw in ["temperature", "humidity", "environment", "weather"]):
            if self.temp is not None:
                logger.info(f"Environment: {self.temp:.1f}°C, {self.humidity:.1f}%")
            return True
        return False

    def cleanup(self):
        self._running = False
        if self.monitor:
            try:
                self.monitor.cleanup()
            except Exception:
                pass
