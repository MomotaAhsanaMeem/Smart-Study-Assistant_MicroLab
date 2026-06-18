import tkinter as tk
from tkinter import font
import time
import threading
import logging
import os
import hardware_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PomodoroTimer")

try:
    from gpiozero import Button, PWMOutputDevice
    GPIO_AVAILABLE = True
except (ImportError, NotImplementedError, Exception) as e:
    GPIO_AVAILABLE = False
    logger.warning(f"gpiozero not available or failed to load ({e}). Hardware button/buzzer disabled.")

# Study configurations (in seconds for easy testing, normally 25*60 and 5*60)
# Default is standard Pomodoro, but we can fast-forward for testing
STUDY_TIME = 25 * 60
BREAK_TIME = 5 * 60

class PomodoroState:
    IDLE = "IDLE"
    STUDY = "STUDY"
    BREAK = "BREAK"

def find_usb_microphone_index():
    """Dynamically locates the best available audio input device."""
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        bt_devices = []
        usb_devices = []
        default_devices = []
        other_devices = []
        
        for i in range(p.get_device_count()):
            try:
                info = p.get_device_info_by_index(i)
                name = info.get('name', '').lower()
                max_input_channels = info.get('maxInputChannels', 0)
                
                if max_input_channels > 0:
                    if any(kw in name for kw in ['bluetooth', 'bluez', 'headset', 'handsfree']):
                        bt_devices.append((i, info.get('name')))
                    elif any(kw in name for kw in ['usb', 'pnp', 'mic']):
                        usb_devices.append((i, info.get('name')))
                    elif any(kw in name for kw in ['default', 'pulse', 'pipewire']):
                        default_devices.append((i, info.get('name')))
                    else:
                        other_devices.append((i, info.get('name')))
            except Exception:
                continue
                
        p.terminate()
        
        if default_devices: return default_devices[0][0]
        if bt_devices: return bt_devices[0][0]
        if usb_devices: return usb_devices[0][0]
        if other_devices: return other_devices[0][0]
    except Exception as e:
        logger.warning(f"Error while searching for audio devices: {e}")
    return None

class VoiceCommandListener:
    """Runs a background thread to listen for timer voice commands."""
    def __init__(self, command_callback):
        self.command_callback = command_callback
        self.running = False
        self.thread = None
        self.device_index = find_usb_microphone_index()
        
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 1500
            self.recognizer.dynamic_energy_threshold = True
        except ImportError:
            logger.critical("speech_recognition library is not available. Voice disabled.")
            self.recognizer = None

    def start(self):
        if not self.recognizer: return
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        logger.info("Voice listener started.")

    def stop(self):
        self.running = False

    def _listen_loop(self):
        import speech_recognition as sr
        calibrated = False
        
        while self.running:
            try:
                with sr.Microphone(device_index=self.device_index) as source:
                    if not calibrated:
                        logger.info("Calibrating microphone for ambient noise...")
                        self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                        calibrated = True
                    logger.info("Listening for timer commands (e.g., 'start timer', 'pause', 'reset')...")
                    audio = self.recognizer.listen(source, timeout=5.0, phrase_time_limit=3.0)
                
                if not self.running: break
                
                text = self.recognizer.recognize_google(audio).lower()
                logger.info(f"Heard: '{text}'")
                self.command_callback(text)
                
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except Exception as e:
                logger.error(f"Voice listener error: {e}")
                time.sleep(2.0)

class PomodoroTimerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pomodoro Study Timer")
        self.root.geometry("400x350")
        self.root.configure(bg="#2C3E50")
        
        self.state = PomodoroState.IDLE
        self.time_left = STUDY_TIME
        self.is_running = False
        self.current_mode = PomodoroState.STUDY # tracks if next session is study or break
        
        # Hardware Integration
        self.buzzer = None
        self.button = None
        if GPIO_AVAILABLE:
            try:
                self.buzzer = PWMOutputDevice(hardware_config.POMODORO_BUZZER_PIN, active_high=False, frequency=2300)
                self.button = Button(hardware_config.POMODORO_BUTTON_PIN, pull_up=True, bounce_time=0.1)
                self.button.when_pressed = self.toggle_timer
                logger.info("Hardware Button and Buzzer initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize GPIO pins: {e}")

        # Voice Integration
        self.voice_listener = VoiceCommandListener(self.handle_voice_command)
        self.voice_listener.start()
        
        self.setup_ui()
        self.update_timer()

    def setup_ui(self):
        self.title_font = font.Font(family="Helvetica", size=24, weight="bold")
        self.time_font = font.Font(family="Helvetica", size=60, weight="bold")
        self.btn_font = font.Font(family="Helvetica", size=14)

        self.status_label = tk.Label(self.root, text="READY TO STUDY", font=self.title_font, bg="#2C3E50", fg="#ECF0F1")
        self.status_label.pack(pady=20)

        self.time_label = tk.Label(self.root, text=self.format_time(self.time_left), font=self.time_font, bg="#2C3E50", fg="#E74C3C")
        self.time_label.pack(pady=10)

        btn_frame = tk.Frame(self.root, bg="#2C3E50")
        btn_frame.pack(pady=20)

        self.start_btn = tk.Button(btn_frame, text="Start", font=self.btn_font, command=self.start_timer, bg="#27AE60", fg="white", width=8)
        self.start_btn.grid(row=0, column=0, padx=10)

        self.pause_btn = tk.Button(btn_frame, text="Pause", font=self.btn_font, command=self.pause_timer, bg="#F39C12", fg="white", width=8)
        self.pause_btn.grid(row=0, column=1, padx=10)

        self.reset_btn = tk.Button(btn_frame, text="Reset", font=self.btn_font, command=self.reset_timer, bg="#C0392B", fg="white", width=8)
        self.reset_btn.grid(row=0, column=2, padx=10)

    def format_time(self, seconds):
        mins, secs = divmod(seconds, 60)
        return f"{mins:02d}:{secs:02d}"

    def update_ui(self):
        self.time_label.config(text=self.format_time(self.time_left))
        if self.state == PomodoroState.IDLE:
            self.status_label.config(text="PAUSED / IDLE", fg="#BDC3C7")
        elif self.state == PomodoroState.STUDY:
            self.status_label.config(text="STUDYING", fg="#E74C3C")
            self.time_label.config(fg="#E74C3C")
        elif self.state == PomodoroState.BREAK:
            self.status_label.config(text="BREAK TIME", fg="#2ECC71")
            self.time_label.config(fg="#2ECC71")

    def toggle_timer(self):
        """Called by the physical button"""
        if self.is_running:
            self.pause_timer()
        else:
            self.start_timer()

    def handle_voice_command(self, text):
        if "start" in text or "begin" in text or "resume" in text:
            logger.info("Voice command triggered: START")
            self.root.after(0, self.start_timer)
        elif "pause" in text or "stop" in text or "wait" in text:
            logger.info("Voice command triggered: PAUSE")
            self.root.after(0, self.pause_timer)
        elif "reset" in text or "restart" in text:
            logger.info("Voice command triggered: RESET")
            self.root.after(0, self.reset_timer)

    def start_timer(self):
        if not self.is_running:
            self.is_running = True
            self.state = self.current_mode
            self.update_ui()
            logger.info(f"Timer started: {self.state}")

    def pause_timer(self):
        if self.is_running:
            self.is_running = False
            self.state = PomodoroState.IDLE
            self.update_ui()
            logger.info("Timer paused.")

    def reset_timer(self):
        self.is_running = False
        self.state = PomodoroState.IDLE
        self.current_mode = PomodoroState.STUDY
        self.time_left = STUDY_TIME
        self.update_ui()
        logger.info("Timer reset.")

    def trigger_alarm(self):
        """Runs the buzzer in a separate thread so it doesn't block the UI"""
        def beep():
            if self.buzzer:
                for _ in range(3):
                    self.buzzer.value = 0.2
                    time.sleep(0.5)
                    self.buzzer.value = 0.0
                    time.sleep(0.5)
            else:
                # Fallback terminal bell if no buzzer
                import sys
                for _ in range(3):
                    sys.stdout.write('\a')
                    sys.stdout.flush()
                    time.sleep(1)
        threading.Thread(target=beep, daemon=True).start()

    def handle_session_end(self):
        self.is_running = False
        self.trigger_alarm()
        
        if self.current_mode == PomodoroState.STUDY:
            logger.info("Study session complete! Switching to Break.")
            self.current_mode = PomodoroState.BREAK
            self.time_left = BREAK_TIME
        else:
            logger.info("Break complete! Back to Study.")
            self.current_mode = PomodoroState.STUDY
            self.time_left = STUDY_TIME
            
        self.state = PomodoroState.IDLE
        self.update_ui()

    def update_timer(self):
        if self.is_running and self.time_left > 0:
            self.time_left -= 1
            self.update_ui()
        elif self.is_running and self.time_left <= 0:
            self.handle_session_end()
            
        # Schedule the next update
        self.root.after(1000, self.update_timer)

    def on_close(self):
        logger.info("Shutting down Pomodoro Timer...")
        self.is_running = False
        self.voice_listener.stop()
        if self.buzzer: self.buzzer.close()
        if self.button: self.button.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PomodoroTimerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
