import tkinter as tk
from tkinter import font
import threading
import time
import collections
from datetime import datetime
import board
import adafruit_dht
import hardware_config
from environment_monitor import EnvironmentMonitor

import logging
# Suppress Matplotlib's annoying categorical string INFO logs
logging.getLogger('matplotlib.category').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Matplotlib for Professional Graphing
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ==========================================
# PRO UI THEME CONFIGURATION
# ==========================================
BG_COLOR = "#0D1117"       # GitHub Dark Dimmed Background
PANEL_COLOR = "#161B22"    # Slightly lighter panel background
TEXT_COLOR = "#C9D1D9"     # Off-white text
SAFE_COLOR = "#238636"     # Success Green (Optimal)
WARN_COLOR = "#DA3633"     # Alert Red (Exceeded / High)
LOW_COLOR = "#8957E5"      # Purple (Warning / Low)
HIGHLIGHT_COLOR = "#58A6FF" # Accent Blue
TEXT_MUTED = "#8B949E"     # Greyed out labels

class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enterprise Environment Telemetry Dashboard")
        self.root.geometry("1024x720")
        self.root.configure(bg=BG_COLOR)
        
        # Data History (stores up to 50 historical readings for the live charts)
        self.history_len = 50
        self.timestamps = collections.deque(maxlen=self.history_len)
        self.temp_history = collections.deque(maxlen=self.history_len)
        self.hum_history = collections.deque(maxlen=self.history_len)
        
        # Hardware Initialization
        self.monitor = None
        self.temp_range = (15, 28)
        self.hum_range = (30, 65)
        
        try:
            self.monitor = EnvironmentMonitor(
                pin=hardware_config.DHT11_PIN, 
                temp_range=self.temp_range,
                hum_range=self.hum_range,
                sensor_type=adafruit_dht.DHT11
            )
        except Exception as e:
            print(f"HARDWARE ERROR: {e}")

        self.setup_ui()
        
        # State Variables
        self.latest_temp = None
        self.latest_hum = None
        self.latest_alerts = []
        self.flash_state = False
        self.needs_chart_redraw = False
        
        # Start Background Sensor Thread
        self.running = True
        if self.monitor:
            self.sensor_thread = threading.Thread(target=self.read_sensor_loop)
            self.sensor_thread.daemon = True
            self.sensor_thread.start()

        # Start GUI Update Loop
        self.update_gui()
        
    def setup_ui(self):
        """Constructs the professional dashboard layout."""
        # Custom Fonts
        self.font_title = font.Font(family="Helvetica", size=24, weight="bold")
        self.font_value = font.Font(family="Helvetica", size=60, weight="bold")
        self.font_label = font.Font(family="Helvetica", size=16, weight="bold")
        self.font_status = font.Font(family="Helvetica", size=18, weight="bold")
        self.font_alert = font.Font(family="Helvetica", size=14, weight="bold")

        # Main Layout Frame
        self.main_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.main_frame.pack(expand=True, fill="both", padx=20, pady=15)

        # Header Title
        self.header = tk.Label(self.main_frame, text="ENVIRONMENTAL TELEMETRY", font=self.font_title, bg=BG_COLOR, fg=HIGHLIGHT_COLOR)
        self.header.pack(pady=(0, 15))

        # Top Panels Container (For Current Values)
        self.top_frame = tk.Frame(self.main_frame, bg=BG_COLOR)
        self.top_frame.pack(fill="x", pady=(0, 10))

        # ---------------- TEMPERATURE PANEL ----------------
        self.temp_frame = tk.Frame(self.top_frame, bg=PANEL_COLOR, highlightbackground="#30363D", highlightthickness=2)
        self.temp_frame.pack(side="left", expand=True, fill="both", padx=(0, 10))
        
        tk.Label(self.temp_frame, text="LIVE TEMPERATURE", font=self.font_label, bg=PANEL_COLOR, fg=TEXT_MUTED).pack(pady=(15, 0))
        self.temp_val = tk.Label(self.temp_frame, text="--.-°C", font=self.font_value, bg=PANEL_COLOR, fg=TEXT_COLOR)
        self.temp_val.pack(pady=5)
        self.temp_status = tk.Label(self.temp_frame, text="ANALYZING...", font=self.font_status, bg=PANEL_COLOR, fg=HIGHLIGHT_COLOR)
        self.temp_status.pack(pady=(0, 15))

        # ---------------- HUMIDITY PANEL ----------------
        self.hum_frame = tk.Frame(self.top_frame, bg=PANEL_COLOR, highlightbackground="#30363D", highlightthickness=2)
        self.hum_frame.pack(side="right", expand=True, fill="both", padx=(10, 0))
        
        tk.Label(self.hum_frame, text="LIVE HUMIDITY", font=self.font_label, bg=PANEL_COLOR, fg=TEXT_MUTED).pack(pady=(15, 0))
        self.hum_val = tk.Label(self.hum_frame, text="--.-%", font=self.font_value, bg=PANEL_COLOR, fg=TEXT_COLOR)
        self.hum_val.pack(pady=5)
        self.hum_status = tk.Label(self.hum_frame, text="ANALYZING...", font=self.font_status, bg=PANEL_COLOR, fg=HIGHLIGHT_COLOR)
        self.hum_status.pack(pady=(0, 15))

        # ---------------- MATPLOTLIB CHARTS PANEL ----------------
        self.chart_frame = tk.Frame(self.main_frame, bg=PANEL_COLOR, highlightbackground="#30363D", highlightthickness=2)
        self.chart_frame.pack(expand=True, fill="both", pady=10)
        
        # Setup Figure
        self.fig = Figure(figsize=(10, 3.5), dpi=100, facecolor=PANEL_COLOR)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.fig.subplots_adjust(left=0.08, bottom=0.20, right=0.95, top=0.85, wspace=0.25)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill="both", padx=5, pady=5)

        # ---------------- BOTTOM ALERT BAR ----------------
        self.alert_frame = tk.Frame(self.main_frame, bg=PANEL_COLOR, highlightbackground="#30363D", highlightthickness=2)
        self.alert_frame.pack(fill="x", pady=(10, 0))
        
        init_text = "INITIALIZING SENSORS & ACCUMULATING DATA..." if self.monitor else "HARDWARE ERROR DETECTED"
        init_color = HIGHLIGHT_COLOR if self.monitor else WARN_COLOR
        self.alert_label = tk.Label(self.alert_frame, text=init_text, font=self.font_alert, bg=PANEL_COLOR, fg=init_color)
        self.alert_label.pack(pady=12)

    def draw_charts(self):
        """Draws the real-time Matplotlib charts for both metrics."""
        if not self.timestamps:
            return
            
        # Clear plots for redraw
        self.ax1.clear()
        self.ax2.clear()
        
        # Apply dark theme styling to axes
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor(PANEL_COLOR)
            ax.tick_params(colors=TEXT_MUTED, labelsize=9)
            for spine in ax.spines.values():
                spine.set_color('#30363D')
            ax.grid(color='#30363D', linestyle='--', alpha=0.6)

        # === PLOT 1: TEMPERATURE ===
        self.ax1.plot(self.timestamps, self.temp_history, color=HIGHLIGHT_COLOR, linewidth=2.5, marker='o', markersize=4)
        self.ax1.set_title("Temperature History (°C)", color=TEXT_COLOR, fontsize=12, pad=12, weight="bold")
        
        # Shaded Safe Region Band
        self.ax1.axhspan(self.temp_range[0], self.temp_range[1], facecolor=SAFE_COLOR, alpha=0.15)
        # Warning/Danger lines
        self.ax1.axhline(self.temp_range[1], color=WARN_COLOR, linestyle='--', alpha=0.8, label="Max Limit")
        self.ax1.axhline(self.temp_range[0], color=LOW_COLOR, linestyle='--', alpha=0.8, label="Min Limit")

        # === PLOT 2: HUMIDITY ===
        self.ax2.plot(self.timestamps, self.hum_history, color="#E3B341", linewidth=2.5, marker='o', markersize=4)
        self.ax2.set_title("Humidity History (%)", color=TEXT_COLOR, fontsize=12, pad=12, weight="bold")
        
        # Shaded Safe Region Band
        self.ax2.axhspan(self.hum_range[0], self.hum_range[1], facecolor=SAFE_COLOR, alpha=0.15)
        # Warning/Danger lines
        self.ax2.axhline(self.hum_range[1], color=WARN_COLOR, linestyle='--', alpha=0.8)
        self.ax2.axhline(self.hum_range[0], color=LOW_COLOR, linestyle='--', alpha=0.8)

        # Format X-axis timestamps (prevent crowding by only showing every Nth label)
        step = max(1, len(self.timestamps) // 5)
        
        self.ax1.set_xticks(list(range(len(self.timestamps)))[::step])
        self.ax1.set_xticklabels(list(self.timestamps)[::step], rotation=30)
        
        self.ax2.set_xticks(list(range(len(self.timestamps)))[::step])
        self.ax2.set_xticklabels(list(self.timestamps)[::step], rotation=30)
        
        self.canvas.draw()

    def read_sensor_loop(self):
        """Runs continuously in the background to fetch data via the robust retry mechanism."""
        while self.running:
            temp, hum = self.monitor.read_sensor(retries=5)
            if temp is not None and hum is not None:
                self.latest_temp = temp
                self.latest_hum = hum
                self.latest_alerts = self.monitor.check_safety(temp, hum)
                
                # Append to Data History
                current_time = datetime.now().strftime("%H:%M:%S")
                self.timestamps.append(current_time)
                self.temp_history.append(temp)
                self.hum_history.append(hum)
                
                # Flag to inform the main thread a redraw is needed
                self.needs_chart_redraw = True
                
            time.sleep(2.5)

    def get_status_config(self, value, ranges):
        """Returns the appropriate text label and color depending on if value is high, low, or optimal."""
        if value > ranges[1]:
            return "STATUS: EXCEEDED (HIGH)", WARN_COLOR
        elif value < ranges[0]:
            return "STATUS: WARNING (LOW)", LOW_COLOR
        else:
            return "STATUS: OPTIMAL", SAFE_COLOR

    def update_gui(self):
        """Safely updates all UI components from the main Tkinter thread."""
        if self.latest_temp is not None and self.latest_hum is not None:
            # 1. Update Massive Numbers
            self.temp_val.config(text=f"{self.latest_temp:.1f}°C")
            self.hum_val.config(text=f"{self.latest_hum:.1f}%")

            # 2. Set Temperature Status Label (High/Low/Optimal)
            t_status_text, t_color = self.get_status_config(self.latest_temp, self.temp_range)
            self.temp_status.config(text=t_status_text, fg=t_color)
            self.temp_val.config(fg=t_color)
            self.temp_frame.config(highlightbackground=t_color if t_color != SAFE_COLOR else "#30363D")

            # 3. Set Humidity Status Label (High/Low/Optimal)
            h_status_text, h_color = self.get_status_config(self.latest_hum, self.hum_range)
            self.hum_status.config(text=h_status_text, fg=h_color)
            self.hum_val.config(fg=h_color)
            self.hum_frame.config(highlightbackground=h_color if h_color != SAFE_COLOR else "#30363D")

            # 4. Redraw Matplotlib Charts (Only if new data came in)
            if self.needs_chart_redraw:
                self.draw_charts()
                self.needs_chart_redraw = False

            # 5. Handle Global Alert Banner
            if self.latest_alerts:
                self.flash_state = not self.flash_state
                alert_text = "   |   ".join(self.latest_alerts)
                
                # Flash between solid red block and dark panel
                if self.flash_state:
                    self.alert_frame.config(bg=WARN_COLOR)
                    self.alert_label.config(text=f"⚠️ {alert_text} ⚠️", bg=WARN_COLOR, fg="#FFFFFF")
                else:
                    self.alert_frame.config(bg=PANEL_COLOR)
                    self.alert_label.config(text=f"⚠️ {alert_text} ⚠️", bg=PANEL_COLOR, fg=WARN_COLOR)
            else:
                self.alert_frame.config(bg=PANEL_COLOR)
                self.alert_label.config(text="✓ ALL SYSTEMS OPERATING WITHIN OPTIMAL PARAMETERS", bg=PANEL_COLOR, fg=SAFE_COLOR)

        # Schedule next UI update
        self.root.after(1000, self.update_gui)

    def on_closing(self):
        """Cleanup logic when window is closed."""
        self.running = False
        if self.monitor:
            self.monitor.cleanup()
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
