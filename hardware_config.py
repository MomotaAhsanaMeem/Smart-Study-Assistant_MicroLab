import board

# ==========================================
# CENTRALIZED HARDWARE PIN CONFIGURATION
# ==========================================

# 1. Environment Monitor (DHT11 Sensor)
# 1. Environment Monitor (DHT11 Sensor)
# The data pin connected to the DHT11 sensor ('S' pin).
# CRITICAL PIN MAPPING: 
# If you plugged the wire into "Physical Pin 7" on the Pi, that is "GPIO 4" in software!
# Therefore, you MUST use `board.D4` here, NOT `board.D7`. 
# (`board.D7` would be Physical Pin 26).
DHT11_PIN = board.D4

# 2. Lamp Controller (Servo Motor)
# The GPIO pin connected to the servo motor signal wire.
# Default: 18 (GPIO 18, Physical Pin 12)
SERVO_PIN = 18

# 2.5 Lamp Brightness (MOSFET Gate / PWM)
# The GPIO pin connected to the MOSFET gate to control the lamp brightness.
# Default: 12 (GPIO 12, Physical Pin 32)
LAMP_BRIGHTNESS_PIN = 12


# 3. Smart Lighting (PCF8591 I2C ADC for LDR)
# I2C communication uses a bus, not a single pin. 
# The Raspberry Pi's hardware I2C Bus 1 is permanently mapped to:
#   - SDA: GPIO 2 (Physical Pin 3)
#   - SCL: GPIO 3 (Physical Pin 5)
I2C_BUS_NUMBER = 1
PCF8591_ADDRESS = 0x48

# 4. Pomodoro Timer
POMODORO_BUTTON_PIN = 17
POMODORO_BUZZER_PIN = 27

# 6. Physical Push Buttons (Lamp + Rotate)
# Wiring: Button between GPIO pin and GND (internal pull-up used, no external resistor needed)
LAMP_BUTTON_PIN = 22      # GPIO 22, Physical Pin 15
ROTATE_BUTTON_PIN = 23    # GPIO 23, Physical Pin 16

# 5. Web Dashboard (Unified System)
WEB_SERVER_HOST = "0.0.0.0"   # Listen on all interfaces (accessible via WiFi)
WEB_SERVER_PORT = 5000

