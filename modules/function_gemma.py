"""
FunctionGemma 270M — Local AI Voice Command Router
====================================================
Loads FunctionGemma 270M (GGUF format) via llama-cpp-python and converts
natural-language transcripts into structured function calls that the
MicroLab SystemManager can execute directly.

Architecture
------------
  [STT transcript]  →  FunctionGemmaEngine.parse()  →  FunctionCall(name, args)
                                                          ↓
                                                    SystemManager dispatcher
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("FunctionGemma")

# ──────────────────────────────────────────────────────────────────────────────
#  TOOL DECLARATIONS & FEW-SHOT HISTORY (OPTIMIZED FOR 270M MODEL CONTEXT)
# ──────────────────────────────────────────────────────────────────────────────

DECLARATIONS = [
    # Lamp power
    "declaration:lamp_on{description:<escape>Turn lamp on<escape>,parameters:{}}",
    "declaration:lamp_off{description:<escape>Turn lamp off<escape>,parameters:{}}",
    # Brightness
    "declaration:set_brightness{description:<escape>Set brightness percentage (0-100)<escape>,parameters:{properties:{value:{type:<escape>INTEGER<escape>}},required:[<escape>value<escape>]}}",
    "declaration:increase_brightness{description:<escape>Make lamp brighter<escape>,parameters:{}}",
    "declaration:decrease_brightness{description:<escape>Make lamp dimmer<escape>,parameters:{}}",
    # Pan
    "declaration:set_pan_angle{description:<escape>Rotate lamp to angle (0-180)<escape>,parameters:{properties:{angle:{type:<escape>INTEGER<escape>}},required:[<escape>angle<escape>]}}",
    "declaration:pan_left{description:<escape>Rotate lamp left<escape>,parameters:{}}",
    "declaration:pan_right{description:<escape>Rotate lamp right<escape>,parameters:{}}",
    "declaration:pan_center{description:<escape>Center lamp position<escape>,parameters:{}}",
    # Smart mode
    "declaration:enable_smart_mode{description:<escape>Enable auto brightness<escape>,parameters:{}}",
    "declaration:disable_smart_mode{description:<escape>Disable auto brightness<escape>,parameters:{}}",
    # Gesture Control
    "declaration:activate_gesture_control{description:<escape>Start gesture control camera<escape>,parameters:{}}",
    "declaration:deactivate_gesture_control{description:<escape>Stop gesture control camera<escape>,parameters:{}}",
    # Focus Tracker
    "declaration:activate_focus_tracker{description:<escape>Start focus tracker camera<escape>,parameters:{}}",
    "declaration:deactivate_focus_tracker{description:<escape>Stop focus tracker camera<escape>,parameters:{}}",
    # Pomodoro
    "declaration:start_pomodoro{description:<escape>Start Pomodoro timer<escape>,parameters:{}}",
    "declaration:pause_pomodoro{description:<escape>Pause Pomodoro timer<escape>,parameters:{}}",
    "declaration:reset_pomodoro{description:<escape>Reset Pomodoro timer<escape>,parameters:{}}",
    # OCR
    "declaration:start_scan{description:<escape>Start document scan camera<escape>,parameters:{}}",
    "declaration:capture_page{description:<escape>Capture document page<escape>,parameters:{}}",
    "declaration:finish_scan{description:<escape>Generate scan PDF<escape>,parameters:{}}",
    "declaration:cancel_scan{description:<escape>Cancel document scan<escape>,parameters:{}}",
    # Environment
    "declaration:get_environment_status{description:<escape>Get room temp/humidity<escape>,parameters:{}}",
]

FEW_SHOTS = [
    ("turn lamp on", "lamp_on{}"),
    ("turn lamp off", "lamp_off{}"),
    ("make it brighter", "increase_brightness{}"),
    ("dim the light", "decrease_brightness{}"),
    ("set brightness to 75%", "set_brightness{value:75}"),
    ("rotate left", "pan_left{}"),
    ("rotate right", "pan_right{}"),
    ("center position", "pan_center{}"),
    ("set angle to 120", "set_pan_angle{angle:120}"),
    ("smart mode on", "enable_smart_mode{}"),
    ("smart mode off", "disable_smart_mode{}"),
    ("get room temperature", "get_environment_status{}"),
]

ALLOWED_FUNCTIONS = {
    "lamp_on", "lamp_off", "set_brightness", "increase_brightness", "decrease_brightness",
    "set_pan_angle", "pan_left", "pan_right", "pan_center", "enable_smart_mode", "disable_smart_mode",
    "activate_gesture_control", "deactivate_gesture_control", "activate_focus_tracker", "deactivate_focus_tracker",
    "start_pomodoro", "pause_pomodoro", "reset_pomodoro", "start_scan", "capture_page", "finish_scan", "cancel_scan",
    "get_environment_status"
}

# Translate common small-model hallucinations to correct API names
FUNCTION_ALIASES = {
    "turn_on_light": "lamp_on",
    "turn_off_light": "lamp_off",
    "turn_on_lamp": "lamp_on",
    "turn_off_lamp": "lamp_off",
    "switch_on_light": "lamp_on",
    "switch_off_light": "lamp_off",
    "switch_on_lamp": "lamp_on",
    "switch_off_lamp": "lamp_off",
    "dim_light": "decrease_brightness",
    "brighten_light": "increase_brightness",
    "switch_off_auto_brightness": "disable_smart_mode",
    "stop_automatic_mode": "disable_smart_mode",
    "turn_off_smart_mode": "disable_smart_mode",
    "turn_on_smart_mode": "enable_smart_mode",
    "stop_gesture_control": "deactivate_gesture_control",
    "start_gesture_control": "activate_gesture_control",
    "disable_gesture_control": "deactivate_gesture_control",
    "enable_gesture_control": "activate_gesture_control",
    "stop_focus_tracking": "deactivate_focus_tracker",
    "start_focus_tracking": "activate_focus_tracker",
    "disable_focus_tracker": "deactivate_focus_tracker",
    "enable_focus_tracker": "activate_focus_tracker",
    "activate_focus": "activate_focus_tracker",
    "deactivate_focus": "deactivate_focus_tracker",
    "stop_pomodoro": "pause_pomodoro",
    "restart_pomodoro": "reset_pomodoro",
    "discard_document_scan": "cancel_scan",
    "stop_document_scan": "cancel_scan",
}

# Semantic keyword validation to eliminate false positives
FUNCTION_VALIDATION = {
    "lamp_on": ["on", "enable", "start", "switch", "wake", "up", "light", "lamp"],
    "lamp_off": ["off", "disable", "stop", "kill", "sleep", "light", "lamp"],
    "set_brightness": ["brightness", "bright", "dim", "percent", "%", "value", "level"],
    "increase_brightness": ["brighter", "increase", "more", "up", "light", "bright"],
    "decrease_brightness": ["dimmer", "decrease", "less", "down", "dim"],
    "set_pan_angle": ["angle", "degree", "rotate", "point", "pan", "angle"],
    "pan_left": ["left", "turn", "rotate", "pan", "look"],
    "pan_right": ["right", "turn", "rotate", "pan", "look"],
    "pan_center": ["center", "centre", "forward", "ahead", "reset", "middle"],
    "enable_smart_mode": ["smart", "auto", "sensor", "automatic"],
    "disable_smart_mode": ["manual", "smart", "auto", "sensor", "automatic"],
    "activate_gesture_control": ["gesture", "hand", "camera", "control"],
    "deactivate_gesture_control": ["gesture", "hand", "camera", "control"],
    "activate_focus_tracker": ["focus", "track", "watch", "camera"],
    "deactivate_focus_tracker": ["focus", "track", "watch", "camera"],
    "start_pomodoro": ["pomodoro", "timer", "study", "session", "begin", "start"],
    "pause_pomodoro": ["pomodoro", "timer", "study", "session", "pause", "stop", "hold"],
    "reset_pomodoro": ["pomodoro", "timer", "study", "session", "reset", "restart", "clear"],
    "start_scan": ["scan", "ocr", "document", "camera", "page"],
    "capture_page": ["capture", "photo", "snap", "page", "take"],
    "finish_scan": ["finish", "pdf", "generate", "process", "done"],
    "cancel_scan": ["cancel", "discard", "abort", "stop", "scan"],
    "get_environment_status": ["temp", "humidity", "hot", "cold", "weather", "room", "status"],
}

# ──────────────────────────────────────────────────────────────────────────────
#  Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FunctionCall:
    """A parsed function call from FunctionGemma."""
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    raw_output: str = ""        # raw model output for debugging
    latency_ms: float = 0.0    # inference latency in ms


UNKNOWN_CALL = FunctionCall(name="unknown", args={})

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "functiongemma-270m-q4_k_m.gguf",
)

_ALT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "google_functiongemma-270m-it-Q4_K_M.gguf",
)

# ──────────────────────────────────────────────────────────────────────────────
#  FunctionGemmaEngine
# ──────────────────────────────────────────────────────────────────────────────

class FunctionGemmaEngine:
    """
    Loads FunctionGemma 270M (GGUF) and converts voice transcript text
    into structured FunctionCall objects using standard tool-calling formats.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_threads: Optional[int] = None,
        max_tokens: int = 64,
        temperature: float = 0.1,
    ):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        if not os.path.isfile(self.model_path) and os.path.isfile(_ALT_MODEL_PATH):
            self.model_path = _ALT_MODEL_PATH
            logger.info(f"Using alternate model path: {self.model_path}")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.available = False
        self._llm = None

        if n_threads is None:
            import os as _os
            n_threads = _os.cpu_count() or 4
        self.n_threads = n_threads

        self._load_model()

    def _load_model(self) -> None:
        if not os.path.isfile(self.model_path):
            logger.warning(
                f"FunctionGemma model not found at: {self.model_path}\n"
                f"  → Voice control will fall back to keyword matching."
            )
            return

        try:
            from llama_cpp import Llama

            logger.info(f"Loading FunctionGemma 270M from: {self.model_path}")
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_threads=self.n_threads,
                n_gpu_layers=0,
                use_mmap=True,
                use_mlock=False,
                verbose=False,
            )
            self.available = True
            logger.info("✅ FunctionGemma 270M loaded and ready.")

        except ImportError:
            logger.warning("llama-cpp-python is not installed. Falling back to keyword matching.")
        except Exception as e:
            logger.error(f"Failed to load FunctionGemma model: {e}")

    def parse(self, transcript: str) -> FunctionCall:
        if not self.available or self._llm is None:
            return UNKNOWN_CALL

        transcript = transcript.strip()
        if not transcript:
            return UNKNOWN_CALL

        prompt = self._build_prompt(transcript)

        try:
            t0 = time.perf_counter()
            response = self._llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["<end_function_call>", "\n"],
                echo=False,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            raw = response["choices"][0]["text"].strip()
            logger.debug(f"FunctionGemma raw output ({latency_ms:.0f} ms): {raw!r}")

            name, args = self._parse_output(raw, transcript)
            call = FunctionCall(name=name, args=args, raw_output=raw, latency_ms=latency_ms)

            if call.name != "unknown":
                logger.info(
                    f"🤖 FunctionGemma [{latency_ms:.0f}ms]: "
                    f'"{transcript}" → {call.name}({call.args})'
                )
            else:
                logger.info(f"🤖 FunctionGemma: no function matched or validation failed for: \"{transcript}\"")

            return call

        except Exception as e:
            logger.error(f"FunctionGemma inference error: {e}")
            return UNKNOWN_CALL

    def _build_prompt(self, transcript: str) -> str:
        decl_str = "".join([f"<start_function_declaration>{d}<end_function_declaration>" for d in DECLARATIONS])
        prompt = (
            f"<start_of_turn>developer\n"
            f"You are a model that can do function calling with the following functions\n"
            f"{decl_str}<end_of_turn>\n"
        )
        for q, a in FEW_SHOTS:
            prompt += (
                f"<start_of_turn>user\n"
                f"{q}<end_of_turn>\n"
                f"<start_of_turn>model\n"
                f"<start_function_call>call:{a}<end_function_call><end_of_turn>\n"
            )
        prompt += (
            f"<start_of_turn>user\n"
            f"{transcript}<end_of_turn>\n"
            f"<start_of_turn>model\n"
            f"<start_function_call>call:"
        )
        return prompt

    def _parse_output(self, raw: str, transcript: str) -> tuple[str, dict[str, Any]]:
        # Reconstruct prefix in case model completes from function name onwards
        reconstructed = "call:" + raw.strip()
        match = re.search(r"call:([a-zA-Z0-9_-]+)\{(.*)\}", reconstructed)
        if not match:
            match = re.search(r"call:([a-zA-Z0-9_-]+)\{(.*)\}", raw)
            if not match:
                return "unknown", {}

        name = match.group(1)
        args_str = match.group(2)
        args = self._parse_args(args_str)

        # Alias resolution
        if name in FUNCTION_ALIASES:
            name = FUNCTION_ALIASES[name]

        # Whitelist validation
        if name not in ALLOWED_FUNCTIONS:
            return "unknown", {}

        # Semantic keyword validation
        if name in FUNCTION_VALIDATION:
            keywords = FUNCTION_VALIDATION[name]
            transcript_lower = transcript.lower()
            if not any(kw in transcript_lower for kw in keywords):
                logger.warning(
                    f"Validation rejected AI prediction '{name}' for text '{transcript}'"
                )
                return "unknown", {}

        # Parameter boundary validations
        if name == "set_brightness" and "value" in args:
            try:
                args["value"] = max(0, min(100, int(args["value"])))
            except (TypeError, ValueError):
                args["value"] = 50

        if name == "set_pan_angle" and "angle" in args:
            try:
                args["angle"] = max(0, min(180, int(args["angle"])))
            except (TypeError, ValueError):
                args["angle"] = 90

        return name, args

    def _parse_args(self, args_str: str) -> dict[str, Any]:
        args = {}
        pattern = r"([a-zA-Z0-9_]+):(?:<escape>(.*?)<escape>|([^,{}]+))"
        for match in re.finditer(pattern, args_str):
            k = match.group(1)
            if match.group(2) is not None:
                v = match.group(2)
            else:
                v = match.group(3).strip()
                try:
                    if "." in v:
                        v = float(v)
                    else:
                        v = int(v)
                except ValueError:
                    pass
            args[k] = v
        return args

    def warmup(self) -> None:
        if not self.available:
            return
        logger.info("Warming up FunctionGemma model…")
        _ = self.parse("turn lamp on")
        logger.info("FunctionGemma warmup complete.")

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            self.available = False
            logger.info("FunctionGemma model unloaded.")


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    engine = FunctionGemmaEngine()
    if not engine.available:
        print("\n❌ Model not loaded.\n")
        sys.exit(1)

    engine.warmup()
    call = engine.parse("turn on the lamp")
    print(f"Result: {call.name} ({call.args})")
