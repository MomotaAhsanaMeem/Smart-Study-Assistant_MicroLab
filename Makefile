# Variables
CONDA_ENV = focus_env
PYTHON = ./$(CONDA_ENV)/bin/python
MICROMAMBA = ./bin/micromamba

# Display configuration (inherits from environment if set)
DISPLAY ?= :0
export DISPLAY

# Default to wayland-1 only for the local physical screen (:0)
ifeq ($(DISPLAY),:0)
  WAYLAND_DISPLAY ?= wayland-1
  export WAYLAND_DISPLAY
endif

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = python

# Default target
all: help

# Help instructions
help:
	@echo "========================================================="
	@echo "AI PROJECT COMMAND CENTER"
	@echo "========================================================="
	@echo "Available commands:"
	@echo ""
	@echo "  🔮 UNIFIED SYSTEM:"
	@echo "  make run-system : Start the complete Smart Lamp System (all features + web dashboard)"
	@echo ""
	@echo "  🧩 STANDALONE FEATURES:"
	@echo "  make run        : Start the Real-time Focus Tracker"
	@echo "  make run-lamp   : Start the Gesture Lamp & Servo Controller"
	@echo "  make test-servo : Run a sweep test to verify the servo motor connection"
	@echo "  make test-brightness : Sweep the lamp brightness (100% -> 50% -> 10% -> OFF)"
	@echo "  make test-mic   : Test the connected USB microphone and speech recognition"
	@echo "  make run-ocr    : Start the AI Document Scanner & PDF Generator"
	@echo "  make run-env    : Start the DHT11 Environment Monitor"
	@echo "  make run-dashboard : Start the GUI Environment Dashboard"
	@echo "  make run-pomodoro  : Start the Pomodoro Study Timer"
	@echo ""
	@echo "  🛠️  UTILITIES:"
	@echo "  make stop       : Safely kill all background Python/Camera runs"
	@echo "  make install    : Setup the environment and download dependencies"
	@echo "  make clean      : Wipe the environment and cached files"
	@echo "========================================================="

$(CONDA_ENV):
	@if [ ! -f $(MICROMAMBA) ]; then \
		echo "Downloading Micromamba..."; \
		mkdir -p bin; \
		curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj bin/micromamba; \
	fi
	@if [ ! -d $(CONDA_ENV) ]; then \
		echo "Creating Micromamba environment (Python 3.11 + MediaPipe + PortAudio)..."; \
		$(MICROMAMBA) create -q -p ./$(CONDA_ENV) python=3.11 portaudio -c conda-forge -y; \
		$(PYTHON) -m pip install mediapipe opencv-python numpy; \
	fi

install: $(CONDA_ENV)
	$(PYTHON) -m pip install -q google-genai fpdf2 pillow gpiozero rpi-lgpio adafruit-blinka adafruit-circuitpython-dht matplotlib smbus2 pyaudio speechrecognition

# Run the focus tracker using the Micromamba environment
run: install
	@echo "Starting Focus Tracker using Micromamba environment..."
	$(PYTHON) focus_tracker.py

# Run the lamp controller
run-lamp: install
	@echo "Starting Lamp Controller..."
	$(PYTHON) lamp_controller.py

# Test the servo motor connection and movement
test-servo: install
	@echo "Starting Servo Sweep Test..."
	$(PYTHON) test_servo.py

# Test the lamp brightness levels
test-brightness: install
	@echo "Starting Lamp Brightness Sweep Test..."
	$(PYTHON) test_brightness.py

# Test the USB microphone connection and speech recognition
test-mic: install
	@echo "Starting Microphone Test..."
	$(PYTHON) test_microphone.py

# Run the AI OCR Document Scanner
run-ocr: install
	@echo "Starting AI OCR Scanner..."
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python GEMINI_API_KEY=$$GEMINI_API_KEY python main.py
# Run the Environment Monitor
run-env: install
	@echo "Starting Environment Monitor..."
	$(PYTHON) environment_monitor.py

# Run the Environment Monitor UI Dashboard
run-dashboard: install
	@echo "Starting Environment GUI Dashboard..."
	$(PYTHON) env_dashboard.py

# Run the Pomodoro Study Timer
run-pomodoro: install
	@echo "Starting Pomodoro Study Timer..."
	$(PYTHON) pomodoro_timer.py

# ============================================
# UNIFIED SMART LAMP SYSTEM (All-in-One)
# ============================================
run-system: install
	@echo "Installing web dashboard dependencies..."
	$(PYTHON) -m pip install -q flask flask-socketio
	@echo "Starting MicroLab Smart Lamp Unified System..."
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python GEMINI_API_KEY="AIzaSyBzOHjnRfZmisTf95a8MT_PAouPrJ6ZJnM" $(PYTHON) main.py

# Stop all Python runs and camera streams
stop:
	@echo "Stopping all Python runs and camera streams..."
	-pkill -u $$(whoami) -f "main.py"
	-pkill -u $$(whoami) -f "focus_tracker.py"
	-pkill -u $$(whoami) -f "lamp_controller.py"
	-pkill -u $$(whoami) -f "test_microphone.py"
	-pkill -u $$(whoami) -f "app.py"
	-pkill -u $$(whoami) -f "environment_monitor.py"
	-pkill -u $$(whoami) -f "env_dashboard.py"
	-pkill -u $$(whoami) -f "pomodoro_timer.py"
	-pkill -u $$(whoami) -f "rpicam-vid"
	-pkill -u $$(whoami) -f "python"

# Clean up temporary files and environment
clean:
	@echo "Cleaning up..."
	rm -rf $(CONDA_ENV)
	rm -rf bin/
	find . -type d -name "__pycache__" -exec rm -rf {} +

.PHONY: all install run run-lamp test-servo test-brightness test-mic run-ocr run-env run-dashboard run-pomodoro run-system stop clean
