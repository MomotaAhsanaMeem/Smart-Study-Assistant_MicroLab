/* ============================================================
   MicroLab Smart Lamp — Dashboard Client
   Real-time WebSocket communication with the SystemManager.
   ============================================================ */

// --- Socket.IO connection ---
const socket = io();
let currentState = {};

// Slider-dragging flags (prevent server updates from fighting the user's thumb)
const sliderActive = { brightness: false, pan: false };


/* ============================================================
   CONNECTION STATUS
   ============================================================ */
socket.on("connect", () => {
    document.getElementById("status-dot").classList.add("connected");
    document.getElementById("connection-text").textContent = "Connected";

    // Fetch existing PDFs on connect
    fetch('/api/pdfs')
        .then(r => r.json())
        .then(data => {
            const list = document.getElementById("pdf-list");
            if (list) list.innerHTML = "";
            const empty = document.getElementById("pdf-empty");
            if (empty && data.length === 0) {
                empty.style.display = "block";
            }
            data.forEach(entry => addPdfEntry(entry));
        })
        .catch(err => console.error("Error fetching PDFs:", err));
});

socket.on("disconnect", () => {
    document.getElementById("status-dot").classList.remove("connected");
    document.getElementById("connection-text").textContent = "Disconnected";
});


/* ============================================================
   STATE UPDATES  (received ~2× per second)
   ============================================================ */
socket.on("state_update", (state) => {
    currentState = state;
    updateLamp(state.lamp);
    updatePomodoro(state.pomodoro);
    updateFocus(state.focus);
    updateCameraFeed(state.camera_mode, state.focus);
    updateEnvironment(state.environment);
    updateOCR(state.ocr);
});

socket.on("log_event", addLogEntry);

// --- OCR real-time events ---
socket.on("page_captured", (data) => {
    addThumbnail(data.index, data.total);
});

socket.on("pdf_ready", (data) => {
    addPdfEntry(data);
});


/* ============================================================
   UPDATE FUNCTIONS
   ============================================================ */

function updateLamp(lamp) {
    if (!lamp) return;

    // Indicator dot
    const ind = document.getElementById("lamp-indicator");
    ind.className = "lamp-indicator" + (lamp.lamp_on ? " on" : "");

    // ON / OFF button highlighting
    document.getElementById("btn-lamp-on").className  = "btn btn-on"  + (lamp.lamp_on  ? " active" : "");
    document.getElementById("btn-lamp-off").className = "btn btn-off" + (!lamp.lamp_on ? " active" : "");

    // Brightness slider (skip if user is dragging)
    if (!sliderActive.brightness) {
        document.getElementById("brightness-slider").value = lamp.brightness;
        document.getElementById("brightness-val").textContent = lamp.brightness + "%";
    }

    // Pan slider
    if (!sliderActive.pan) {
        document.getElementById("pan-slider").value = lamp.pan_angle;
        document.getElementById("pan-val").textContent = lamp.pan_angle + "°";
    }

    // Smart mode toggle
    document.getElementById("smart-toggle").checked = lamp.smart_mode;

    // Gesture status
    const gestBtn = document.getElementById("btn-gesture");
    const gestSt  = document.getElementById("gesture-status");
    if (lamp.active) {
        gestBtn.className = "btn btn-gesture active";
        gestBtn.textContent = "🖐️ Deactivate Gestures";
        gestSt.textContent  = lamp.gesture || "Tracking…";
        gestSt.style.color  = "var(--green)";
    } else {
        gestBtn.className = "btn btn-gesture";
        gestBtn.textContent = "🖐️ Activate Gestures";
        gestSt.textContent  = "Inactive";
        gestSt.style.color  = "var(--text-3)";
    }
}


function updatePomodoro(p) {
    if (!p) return;
    const card = document.getElementById("pomodoro-card");

    document.getElementById("timer-time").textContent = p.time_display;
    document.getElementById("timer-state").textContent =
        p.is_running ? p.state : (p.state === "IDLE" ? "READY" : p.state);

    // SVG ring progress
    const C = 2 * Math.PI * 52;   // circumference
    const total = p.current_mode === "STUDY" ? 25 * 60 : 5 * 60;
    const progress = p.time_left / total;
    const offset = C * (1 - progress);
    document.getElementById("timer-progress").style.strokeDashoffset = offset;

    // State colour
    card.className = "card pomodoro-card";
    const tt = document.getElementById("timer-time");
    const rp = document.getElementById("timer-progress");
    if (p.state === "STUDY") {
        card.classList.add("study");
        tt.style.color = "var(--red)";
        rp.style.stroke = "var(--red)";
    } else if (p.state === "BREAK") {
        card.classList.add("break");
        tt.style.color = "var(--green)";
        rp.style.stroke = "var(--green)";
    } else {
        tt.style.color = "var(--text-1)";
        rp.style.stroke = "";   // falls back to gradient
    }
}


function updateFocus(f) {
    if (!f) return;
    const dot   = document.getElementById("focus-dot");
    const label = document.getElementById("focus-label");
    document.getElementById("focus-toggle").checked = f.active;

    if (!f.active) {
        dot.className = "focus-dot";
        label.textContent = "Inactive";
        label.style.color = "var(--text-3)";
    } else if (f.status === "FOCUSED") {
        dot.className = "focus-dot focused";
        label.textContent = "FOCUSED";
        label.style.color = "var(--green)";
    } else if (f.status === "DISTRACTED") {
        dot.className = "focus-dot distracted";
        label.textContent = "DISTRACTED";
        label.style.color = "var(--red)";
    } else {
        dot.className = "focus-dot noface";
        label.textContent = "NO FACE";
        label.style.color = "var(--yellow)";
    }

    document.getElementById("distraction-time").textContent  = f.distracted_seconds + "s";
    document.getElementById("distraction-count").textContent = f.distraction_count;
}


// --- Camera Feed ---
let _feedActive = false;   // track if feed img src is set

function updateCameraFeed(cameraMode, focus) {
    const container = document.getElementById("camera-feed-container");
    const img       = document.getElementById("camera-feed");
    const label     = document.getElementById("camera-feed-label");

    const shouldShow = (cameraMode === "focus" || cameraMode === "gesture" || cameraMode === "ocr");

    if (shouldShow) {
        container.style.display = "";

        // Start the MJPEG stream if not already running
        if (!_feedActive) {
            img.src = "/video_feed?" + Date.now();   // cache-bust
            _feedActive = true;
        }

        // Dynamic border colour
        container.className = "camera-feed-container";
        if (cameraMode === "focus" && focus) {
            if (focus.status === "FOCUSED") {
                container.classList.add("focused-border");
                label.textContent = "FOCUS · LIVE";
            } else if (focus.status === "DISTRACTED") {
                container.classList.add("distracted-border");
                label.textContent = "DISTRACTED · LIVE";
            } else {
                label.textContent = "LIVE";
            }
        } else if (cameraMode === "gesture") {
            container.classList.add("gesture-border");
            label.textContent = "GESTURE · LIVE";
        } else {
            label.textContent = "LIVE";
        }
    } else {
        container.style.display = "none";
        if (_feedActive) {
            img.src = "";      // stop the MJPEG connection
            _feedActive = false;
        }
    }
}


function updateEnvironment(e) {
    if (!e) return;

    if (e.temp !== null && e.temp !== undefined) {
        document.getElementById("temp-value").textContent = e.temp.toFixed(1);
        const ts = document.getElementById("temp-status");
        ts.textContent = e.temp_status;
        ts.className = "gauge-status " + e.temp_status.toLowerCase();
    }

    if (e.humidity !== null && e.humidity !== undefined) {
        document.getElementById("hum-value").textContent = e.humidity.toFixed(1);
        const hs = document.getElementById("hum-status");
        hs.textContent = e.hum_status;
        hs.className = "gauge-status " + e.hum_status.toLowerCase();
    }

    const alertEl = document.getElementById("env-alert");
    if (e.alerts && e.alerts.length > 0) {
        alertEl.textContent = "⚠️ " + e.alerts.join(" | ");
        alertEl.className = "env-alert alert";
    } else if (e.temp !== null && e.temp !== undefined) {
        alertEl.textContent = "✓ All systems within optimal parameters";
        alertEl.className = "env-alert ok";
    }
}


function updateOCR(o) {
    if (!o) return;
    document.getElementById("ocr-status").textContent = o.status_message || o.state;
    document.getElementById("ocr-pages").textContent  = "Pages: " + o.pages_captured;

    const start   = document.getElementById("btn-ocr-start");
    const capture = document.getElementById("btn-ocr-capture");
    const process = document.getElementById("btn-ocr-process");
    const cancel  = document.getElementById("btn-ocr-cancel");

    if (o.state === "SCANNING") {
        start.style.display   = "none";
        capture.style.display = "";
        process.style.display = (o.pages_captured > 0) ? "" : "none";
        cancel.style.display  = "";
    } else if (o.state === "PROCESSING") {
        start.style.display   = "none";
        capture.style.display = "none";
        process.style.display = "none";
        cancel.style.display  = "none";
    } else {
        start.style.display   = "";
        capture.style.display = "none";
        process.style.display = "none";
        cancel.style.display  = "none";
    }

    if (!o.available) {
        start.disabled = true;
        start.textContent = "📷 No API Key";
    }
}


/* ============================================================
   OCR HELPERS — Thumbnails and PDF list
   ============================================================ */

function addThumbnail(index, total) {
    const strip = document.getElementById("ocr-thumbnails");
    // Avoid duplicate entries on reconnect / re-render
    if (document.getElementById("ocr-thumb-" + index)) return;

    const wrap = document.createElement("div");
    wrap.className = "ocr-thumb-wrap";
    wrap.id = "ocr-thumb-" + index;

    const label = document.createElement("span");
    label.className = "ocr-thumb-label";
    label.textContent = "Page " + (index + 1);

    const img = document.createElement("img");
    img.className = "ocr-thumb";
    img.alt = "Page " + (index + 1);
    // cache-bust so the browser always fetches the latest capture
    img.src = "/ocr/page/" + index + "?t=" + Date.now();

    wrap.appendChild(img);
    wrap.appendChild(label);
    strip.appendChild(wrap);
    strip.scrollLeft = strip.scrollWidth;   // auto-scroll to the new thumb
}

function addPdfEntry(entry) {
    const list  = document.getElementById("pdf-list");
    const empty = document.getElementById("pdf-empty");
    if (empty) empty.style.display = "none";

    // Avoid duplicates
    if (document.getElementById("pdf-" + entry.filename)) return;

    const li = document.createElement("li");
    li.className = "pdf-entry";
    li.id = "pdf-" + entry.filename;

    li.innerHTML =
        `<div class="pdf-entry-info">
            <span class="pdf-icon">📄</span>
            <div class="pdf-entry-details">
                <span class="pdf-name">${entry.filename}</span>
                <span class="pdf-time">${entry.time}</span>
            </div>
        </div>
        <a class="btn btn-download" href="/ocr/download/${entry.filename}" download="${entry.filename}">
            ⬇ Download
        </a>`;

    list.appendChild(li);
}


/* ============================================================
   LOG
   ============================================================ */
function addLogEntry(entry) {
    const container = document.getElementById("log-container");
    const div = document.createElement("div");
    div.className = "log-entry";
    div.innerHTML = '<span class="log-ts">' + entry.time + "</span>" + entry.message;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;

    // Cap at 80 visible entries
    while (container.children.length > 80) {
        container.removeChild(container.firstChild);
    }
}


/* ============================================================
   CONTROL FUNCTIONS  (emit SocketIO events)
   ============================================================ */

function lampOn()  { socket.emit("lamp_on");  }
function lampOff() { socket.emit("lamp_off"); }

function setBrightness(v) {
    document.getElementById("brightness-val").textContent = v + "%";
    socket.emit("lamp_brightness", { value: v });
}

function setPan(v) {
    document.getElementById("pan-val").textContent = v + "°";
    socket.emit("lamp_pan", { value: v });
}

function toggleSmart() { socket.emit("smart_mode_toggle"); }

function toggleGesture() {
    if (currentState.lamp && currentState.lamp.active) {
        socket.emit("gesture_deactivate");
    } else {
        socket.emit("gesture_activate");
    }
}

function pomodoroStart() { socket.emit("pomodoro_start"); }
function pomodoroPause() { socket.emit("pomodoro_pause"); }
function pomodoroReset() { socket.emit("pomodoro_reset"); }

function toggleFocus() { socket.emit("focus_toggle"); }

function ocrStart()   {
    // Clear old thumbnails when starting a fresh scan
    document.getElementById("ocr-thumbnails").innerHTML = "";
    socket.emit("ocr_start");
}
function ocrCapture() { socket.emit("ocr_capture");  }
function ocrProcess() { socket.emit("ocr_process");  }
function ocrCancel()  {
    // Clear thumbnails on cancel too
    document.getElementById("ocr-thumbnails").innerHTML = "";
    socket.emit("ocr_cancel");
}


/* ============================================================
   SLIDER DRAG TRACKING
   Prevent server state updates from snapping the slider
   while the user is actively dragging it.
   ============================================================ */
["brightness-slider", "pan-slider"].forEach((id) => {
    const key = id.startsWith("b") ? "brightness" : "pan";
    const el  = document.getElementById(id);

    el.addEventListener("mousedown",  () => { sliderActive[key] = true; });
    el.addEventListener("touchstart", () => { sliderActive[key] = true; }, { passive: true });
    el.addEventListener("mouseup",    () => { sliderActive[key] = false; });
    el.addEventListener("touchend",   () => { sliderActive[key] = false; });
});

/* ============================================================
   USER MANUAL ACCORDION
   ============================================================ */
function toggleAccordion(button) {
    const accordion = button.parentElement;
    const content = accordion.querySelector('.accordion-content');
    
    // Toggle active class on accordion container
    accordion.classList.toggle('active');
    
    // Toggle max-height for smooth transition
    if (accordion.classList.contains('active')) {
        content.style.maxHeight = content.scrollHeight + "px";
    } else {
        content.style.maxHeight = null;
    }
}
