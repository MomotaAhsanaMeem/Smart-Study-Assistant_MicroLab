#!/usr/bin/env python3
import time
import sys
import hardware_config

try:
    from gpiozero import PWMLED
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("❌ Error: gpiozero library is not installed in the python environment.")
    sys.exit(1)

# Initialize lamp brightness using configuration
brightness_pin = hardware_config.LAMP_BRIGHTNESS_PIN

print("=========================================================")
print("💡  LAMP BRIGHTNESS / PWM TEST SCRIPT  💡")
print("=========================================================")
print(f"Initializing lamp PWM control on GPIO pin {brightness_pin} (Physical Pin 32)...")
print("=========================================================\n")

try:
    # Initialize the PWM LED device
    lamp = PWMLED(brightness_pin)
    
    print("🌟 [TEST] Setting brightness to 100% (Full Brightness)...")
    lamp.value = 1.0
    input("Press Enter to reduce brightness to 50%...")
    
    print("\n🌗 [TEST] Dimming brightness to 50% (Medium Brightness)...")
    lamp.value = 0.5
    input("Press Enter to reduce brightness to 10%...")
    
    print("\n🌘 [TEST] Dimming brightness to 10% (Low Brightness)...")
    lamp.value = 0.1
    input("Press Enter to turn the lamp OFF...")
    
    print("\n🌑 [TEST] Turning the lamp OFF...")
    lamp.value = 0.0
    time.sleep(0.5)
    
    print("\n✨ Lamp brightness test completed successfully!")

except KeyboardInterrupt:
    print("\n🛑 Test interrupted by user. Safely turning off the lamp...")
    try:
        lamp.value = 0.0
    except Exception:
        pass
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    print("\nTroubleshooting tips:")
    print("1. If you get permission/GPIO errors, ensure the 'micro' user has permission to write to /dev/gpiomem.")
    print("2. Check physical wiring:")
    print("   - MOSFET Gate -> GPIO 12 (Physical Pin 32)")
    print("   - Load (Lamp Negative) -> MOSFET Drain")
    print("   - External Power VCC -> Lamp Positive")
    print("   - Common Ground -> Pi GND & MOSFET Source & Power Supply Ground")
