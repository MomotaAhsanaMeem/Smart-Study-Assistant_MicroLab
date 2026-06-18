"""
MicroLab Smart Lamp System — Single Entry Point
================================================
Starts the entire unified system with one command:
    python main.py   (or:  make run-system)

This boots:
  • Shared hardware (camera, microphone, servo, sensors)
  • All feature modules (gesture, focus, pomodoro, environment, OCR)
  • Web dashboard on http://0.0.0.0:5000
  • Voice command listener
"""

import os
import sys
import signal
import logging

def load_dotenv():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for path in [os.path.join(base_dir, ".env"), os.path.join(base_dir, "..", ".env"), ".env", "../.env"]:
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            parts = line.split("=", 1)
                            if len(parts) == 2:
                                k = parts[0].strip()
                                v = parts[1].strip()
                                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                                    v = v[1:-1]
                                os.environ[k] = v
                break
            except Exception:
                pass

load_dotenv()

# Configure root logger ONCE, before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from system_manager import SystemManager


def main():
    system = SystemManager()

    # Graceful shutdown on Ctrl-C / kill
    def _signal_handler(sig, frame):
        print("\n⛔ Shutdown signal received...")
        system.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Blocks in the main thread (Flask-SocketIO event loop)
    system.start()


if __name__ == "__main__":
    main()
