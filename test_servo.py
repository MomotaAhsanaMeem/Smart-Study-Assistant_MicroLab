import time
import sys
import os
import hardware_config

try:
    from gpiozero import AngularServo
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Error: gpiozero library is not installed in the python environment.")
    sys.exit(1)

# Initialize servo using the configuration
servo_pin = hardware_config.SERVO_PIN
print(f"Initializing servo motor on GPIO pin {servo_pin} (Physical Pin 12)...")

try:
    # min_pulse_width and max_pulse_width adjusted for standard 180 deg SG90
    servo = AngularServo(
        servo_pin, 
        min_angle=0, 
        max_angle=180, 
        min_pulse_width=0.0005, 
        max_pulse_width=0.0024
    )
    
    print("Moving servo to 0 degrees...")
    servo.angle = 0
    input("Press Enter to move to 90 degrees (Center)...")
    
    print("Moving servo to 90 degrees (Center)...")
    servo.angle = 90
    input("Press Enter to move to 180 degrees...")
    
    print("Moving servo to 180 degrees...")
    servo.angle = 180
    input("Press Enter to move back to 90 degrees (Center)...")
    
    print("Moving servo back to 90 degrees (Center)...")
    servo.angle = 90
    input("Press Enter to detach servo and finish...")
    
    print("Detaching servo...")
    servo.detach()
    print("Servo test completed successfully!")

except Exception as e:
    print(f"\nError occurred: {e}")
    print("\nTroubleshooting tips:")
    print("1. If you get permission/GPIO errors, ensure the 'micro' user has permission to write to /dev/gpiomem.")
    print("2. Check physical wiring:")
    print("   - PWM Signal (Orange/Yellow wire) -> GPIO 18 (Physical Pin 12)")
    print("   - Power VCC (Red wire) -> 5V Pin")
    print("   - Ground GND (Black/Brown wire) -> Ground Pin")
