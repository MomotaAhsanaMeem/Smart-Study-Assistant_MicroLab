"""
Vosk Offline STT — Drop-in replacement for Google Speech Recognition
======================================================================
Provides fully offline speech-to-text on Raspberry Pi 5 using the Vosk
library and a small ~50 MB English model.

Why Vosk?
  • 100% offline — no internet required at runtime
  • Low latency: ~200–400 ms on RPi 5 for short commands
  • Low RAM: ~80 MB at runtime
  • Designed for embedded / edge devices

Model Download
  • Handled automatically by install_functiongemma.sh
  • Expected path: models/vosk-model-small-en-us-0.15/
  • Manual: https://alphacephei.com/vosk/models

Usage
-----
  stt = VoskSTT()
  if stt.available:
      text = stt.listen_once(timeout=5.0)   # blocks until speech or timeout
      print(text)   # e.g. "turn on the lamp"
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Optional

logger = logging.getLogger("VoskSTT")

# ──────────────────────────────────────────────────────────────────────────────
#  Default model path
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "vosk-model-small-en-us-0.15",
)


# ──────────────────────────────────────────────────────────────────────────────
#  VoskSTT
# ──────────────────────────────────────────────────────────────────────────────

class VoskSTT:
    """
    Offline speech-to-text using Vosk + PyAudio.

    Parameters
    ----------
    model_path : str
        Path to the Vosk model directory.
    device_index : int | None
        PyAudio input device index. None = system default.
    sample_rate : int
        Audio sample rate in Hz. Must match the Vosk model (16000).
    chunk_size : int
        PyAudio read chunk size in frames.
    """

    SAMPLE_RATE = 16000   # Vosk models require 16 kHz
    CHANNELS    = 1       # mono
    DTYPE       = 8       # paInt16 (PyAudio constant)

    def __init__(
        self,
        model_path: Optional[str] = None,
        device_index: Optional[int] = None,
        chunk_size: int = 4000,
    ):
        self.model_path   = model_path or DEFAULT_MODEL_PATH
        self.device_index = device_index
        self.chunk_size   = chunk_size
        self.available    = False

        self._model      = None
        self._recognizer = None
        self._pyaudio    = None

        self._load()

    # ----------------------------------------------------------------
    #  Initialisation
    # ----------------------------------------------------------------

    def _load(self) -> None:
        """Load Vosk model and set up PyAudio. Sets self.available."""
        # Check model directory
        if not os.path.isdir(self.model_path):
            logger.warning(
                f"Vosk model not found at: {self.model_path}\n"
                f"  → Run  ./install_functiongemma.sh  to download it.\n"
                f"  → STT will fall back to Google Speech Recognition."
            )
            return

        try:
            import vosk  # type: ignore
            import pyaudio  # type: ignore

            logger.info(f"Loading Vosk model from: {self.model_path}")
            self._model      = vosk.Model(self.model_path)
            self._recognizer = vosk.KaldiRecognizer(self._model, self.SAMPLE_RATE)
            self._recognizer.SetMaxAlternatives(0)
            self._recognizer.SetWords(False)

            self._pyaudio = pyaudio.PyAudio()
            self.available = True

            logger.info("✅ Vosk offline STT ready.")

        except ImportError as e:
            logger.warning(
                f"Vosk or PyAudio not installed ({e}).\n"
                f"  → Run  ./install_functiongemma.sh  to install them.\n"
                f"  → STT will fall back to Google Speech Recognition."
            )
        except Exception as e:
            logger.error(f"Vosk initialisation failed: {e}")

    # ----------------------------------------------------------------
    #  Public API
    # ----------------------------------------------------------------

    def listen_once(
        self,
        timeout: float = 6.0,
        phrase_time_limit: float = 4.0,
        energy_threshold: int = 800,
    ) -> Optional[str]:
        """
        Block until a phrase is detected and fully transcribed, or timeout.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for speech to START.
        phrase_time_limit : float
            Maximum seconds of active speech before forcing a final result.
        energy_threshold : int
            RMS energy level required to consider audio as "speech onset".

        Returns
        -------
        str | None
            The transcribed text (lowercased), or None on timeout / error.
        """
        if not self.available:
            return None

        import pyaudio  # type: ignore

        stream = None
        try:
            stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
            )

            # ---- Phase 1: wait for speech onset ----
            start_wait = time.monotonic()
            speech_started = False
            phrase_started = 0.0

            self._recognizer.Reset()   # clear any leftover state

            while True:
                elapsed_wait = time.monotonic() - start_wait

                if elapsed_wait > timeout and not speech_started:
                    return None   # silence timeout

                if speech_started and (time.monotonic() - phrase_started) > phrase_time_limit:
                    break  # phrase time limit hit — finalise

                data = stream.read(self.chunk_size, exception_on_overflow=False)

                # Detect speech onset via simple energy gate
                import struct
                samples = struct.unpack_from(f"<{len(data)//2}h", data)
                rms = (sum(s * s for s in samples) / max(len(samples), 1)) ** 0.5

                if not speech_started:
                    if rms > energy_threshold:
                        speech_started = True
                        phrase_started = time.monotonic()
                        logger.debug(f"Speech onset detected (RMS={rms:.0f})")
                else:
                    # If energy drops very low for > 0.8s, phrase is done
                    if rms < energy_threshold * 0.4:
                        if time.monotonic() - phrase_started > 0.8:
                            break

                self._recognizer.AcceptWaveform(data)

            # ---- Force final result ----
            self._recognizer.AcceptWaveform(b"\x00" * self.chunk_size * 2)
            final_json = self._recognizer.FinalResult()
            result = json.loads(final_json)
            text = result.get("text", "").strip().lower()

            return text if text else None

        except Exception as e:
            logger.error(f"Vosk listen error: {e}")
            return None

        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass

    def cleanup(self) -> None:
        """Release PyAudio resources."""
        if self._pyaudio is not None:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None
        self.available = False
        logger.info("Vosk STT cleaned up.")


# ──────────────────────────────────────────────────────────────────────────────
#  Standalone self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    stt = VoskSTT()

    if not stt.available:
        print("\n❌  Vosk not available. Run  ./install_functiongemma.sh  first.\n")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("  Vosk Offline STT — Microphone Test")
    print("=" * 50)
    print("  Speak a command when prompted. Press Ctrl-C to exit.\n")

    count = 0
    while True:
        count += 1
        print(f"  🎙️  [{count}] Listening (max 6s)…", end="", flush=True)
        text = stt.listen_once(timeout=6.0)
        if text:
            print(f"\r  ✅  [{count}] Heard: \"{text}\"                    ")
        else:
            print(f"\r  ⏱️  [{count}] Timeout / silence.                 ")
