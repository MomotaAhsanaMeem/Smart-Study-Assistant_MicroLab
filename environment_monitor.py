import time
import logging
import board
import adafruit_dht
import hardware_config

# Set up logging for professional output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)

class EnvironmentMonitor:
    """
    A robust Object-Oriented class for monitoring temperature and humidity 
    using DHT11/DHT22 sensors.
    """
    def __init__(self, pin=hardware_config.DHT11_PIN, temp_range=(15, 30), hum_range=(30, 60), sensor_type=adafruit_dht.DHT11):
        """
        Initialize the Environment Monitor.
        
        :param pin: The GPIO pin the sensor data line is connected to (e.g., board.D4).
        :param temp_range: Tuple of (min_temp, max_temp) in Celsius.
        :param hum_range: Tuple of (min_humidity, max_humidity) in percentage.
        :param sensor_type: The DHT sensor class to use (default: adafruit_dht.DHT11).
        """
        self.logger = logging.getLogger("EnvMonitor")
        self.temp_range = temp_range
        self.hum_range = hum_range
        self.pin = pin
        self.sensor_type = sensor_type
        
        self.logger.info(f"Initializing sensor on pin {pin}...")
        # Use pulseio=False on newer Raspberry Pi boards (like Pi 5) for better compatibility
        self.sensor = self.sensor_type(self.pin, use_pulseio=False)

    def read_sensor(self, retries=5):
        """
        Reads temperature and humidity from the sensor.
        DHT sensors are notoriously timing-critical on Linux. This method will retry 
        up to `retries` times before giving up on the current reading cycle.
        """
        for attempt in range(retries):
            try:
                temperature = self.sensor.temperature
                humidity = self.sensor.humidity
                
                # Check for valid readings (sometimes the sensor returns None instead of throwing)
                if temperature is not None and humidity is not None:
                    return temperature, humidity
                    
            except RuntimeError as e:
                # DHT sensors often fail to read due to strict timing requirements on Linux.
                # We log this as DEBUG instead of WARNING so it doesn't spam the user console.
                self.logger.debug(f"Sensor read error (attempt {attempt+1}/{retries}): {e}")
                
                # HARDWARE RESET: Sometimes the GPIO pin buffer gets locked up.
                # Re-initializing the sensor forces a clean state.
                try:
                    self.sensor.exit()
                except Exception:
                    pass
                
                time.sleep(2.1) # DHT11 requires at least 2 seconds between reads
                self.sensor = self.sensor_type(self.pin, use_pulseio=False)
                
            except Exception as e:
                self.logger.error(f"Unexpected error reading sensor: {e}", exc_info=True)
                return None, None
                
        self.logger.error("Failed to read from DHT sensor after maximum retries.")
        return None, None

    def check_safety(self, temperature, humidity):
        """
        Checks if the current readings are within the defined safe ranges.
        Returns a list of alert messages if any range is exceeded.
        """
        alerts = []
        
        if temperature is not None:
            if temperature < self.temp_range[0]:
                alerts.append(f"Temperature TOO LOW: {temperature:.1f}°C (Min safe: {self.temp_range[0]}°C)")
            elif temperature > self.temp_range[1]:
                alerts.append(f"Temperature TOO HIGH: {temperature:.1f}°C (Max safe: {self.temp_range[1]}°C)")
                
        if humidity is not None:
            if humidity < self.hum_range[0]:
                alerts.append(f"Humidity TOO LOW: {humidity:.1f}% (Min safe: {self.hum_range[0]}%)")
            elif humidity > self.hum_range[1]:
                alerts.append(f"Humidity TOO HIGH: {humidity:.1f}% (Max safe: {self.hum_range[1]}%)")
                
        return alerts

    def trigger_alert(self, alerts):
        """
        Handles alerts when safe ranges are exceeded. 
        Designed to be overridden or expanded for hardware integration (buzzers, LEDs, Emails).
        """
        for alert in alerts:
            self.logger.critical(f"⚠️ ALERT TRIGGERED ⚠️ -> {alert}")
            # TODO: Combine with your lamp_controller.py to flash the lamp red!
            # TODO: Or trigger a GPIO buzzer here.

    def cleanup(self):
        """Safely release the GPIO resources used by the sensor."""
        self.logger.info("Cleaning up sensor resources...")
        self.sensor.exit()

    def run(self, interval=2.5):
        """
        Main continuous monitoring loop.
        
        :param interval: Time in seconds to wait between readings. 
                         (DHT11 requires at least 1-2 seconds between reads)
        """
        self.logger.info(f"Starting Environment Monitor.")
        self.logger.info(f"Safety Limits -> Temp: {self.temp_range}°C, Hum: {self.hum_range}%")
        
        try:
            while True:
                temp, hum = self.read_sensor()
                
                if temp is not None and hum is not None:
                    self.logger.info(f"Current State: Temp: {temp:.1f}°C, Humidity: {hum:.1f}%")
                    
                    alerts = self.check_safety(temp, hum)
                    if alerts:
                        self.trigger_alert(alerts)
                
                # Sleep before next read
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("Monitoring manually stopped by user.")
        finally:
            self.cleanup()

if __name__ == "__main__":
    # --- HARDWARE SETUP INSTRUCTIONS ---
    # 1. DHT11 'S' (Signal) pin -> Raspberry Pi GPIO 4 (Pin 7)
    # 2. DHT11 Middle pin (VCC) -> Raspberry Pi 3.3V (Pin 1 or 17)
    # 3. DHT11 '-' (GND) pin    -> Raspberry Pi GND (Pin 6 or 9)
    # -----------------------------------
    
    try:
        # Instantiate the monitor. Default uses DHT11 on GPIO 4.
        monitor = EnvironmentMonitor(
            pin=hardware_config.DHT11_PIN,           # Adjust this if you connect 'S' to a different GPIO
            temp_range=(18, 28),    # Ideal Server Room Temp (C)
            hum_range=(30, 65),     # Alert if outside 30% - 65% humidity
            sensor_type=adafruit_dht.DHT11
        )
        # DHT11 sensors are slow; 2.5 seconds is a safe polling interval
        monitor.run(interval=2.5)
        
    except NotImplementedError:
        logging.error("GPIO access error. Ensure you are running with sufficient permissions, "
                      "or that your Raspberry Pi pin configurations are supported.")
