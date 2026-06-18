"""
OCR Scanner Module
===================
On-demand document scanning triggered by the voice command "scan document"
or the dashboard button.  Uses the existing Gemini LLM client and PDF builder.

Flow:
  1. User says "scan document" → SystemManager opens camera in OCR mode
  2. User says "capture" (or clicks button) → snapshot is stored
  3. Repeat for multiple pages
  4. User says "process scan" → images sent to Gemini AI → PDF generated
"""

import os
import time
import threading
import logging

from modules.base import FeatureModule

logger = logging.getLogger("OCRScanner")


class OCRScannerModule(FeatureModule):
    """On-demand OCR scanning with Gemini AI backend."""

    def __init__(self):
        self.state = "IDLE"   # IDLE | SCANNING | PROCESSING | DONE | ERROR
        self.captured_images = []
        self.output_pdf_path = ""
        self.status_message = ""
        self._processing_thread = None
        self.on_pdf_ready = None   # callback(pdf_path) set by SystemManager

        self.llm_client = None
        self.pdf_builder_cls = None

        try:
            from ocr_system.llm_client import GeminiOCRClient
            from ocr_system.pdf_builder import PDFBuilder

            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                self.llm_client = GeminiOCRClient(api_key=api_key)
                self.pdf_builder_cls = PDFBuilder
                logger.info("OCR Scanner initialised (Gemini API key found).")
            else:
                logger.warning("GEMINI_API_KEY not set — OCR Scanner disabled.")
        except Exception as e:
            logger.warning(f"OCR Scanner init error: {e}")

    # ----------------------------------------------------------------
    #  Scanning workflow
    # ----------------------------------------------------------------

    def start_scan(self) -> bool:
        """Begin a new scanning session. Returns False if OCR is unavailable."""
        if self.llm_client is None:
            logger.error("OCR Scanner not available (no API key).")
            return False
        self.state = "SCANNING"
        self.captured_images = []
        self.output_pdf_path = ""
        self.status_message = "Scanning mode active — capture pages."
        logger.info("OCR scanning started.")
        return True

    def capture_page(self, frame):
        """Store a camera frame as a scanned page."""
        if self.state != "SCANNING":
            return False
        self.captured_images.append(frame.copy())
        count = len(self.captured_images)
        self.status_message = f"Captured page {count}."
        logger.info(f"OCR: page {count} captured.")
        return True

    def process_captures(self) -> bool:
        """Send captured images to Gemini AI for OCR + PDF generation."""
        if self.state != "SCANNING" or len(self.captured_images) == 0:
            return False
        self.state = "PROCESSING"
        self.status_message = "Processing with Gemini AI…"
        self._processing_thread = threading.Thread(target=self._process, daemon=True)
        self._processing_thread.start()
        return True

    def stop_scan(self):
        """Cancel the current scanning session."""
        self.state = "IDLE"
        self.captured_images = []
        self.status_message = ""

    # ----------------------------------------------------------------
    #  Background processing
    # ----------------------------------------------------------------

    def _process(self):
        try:
            from ocr_system.pdf_builder import PDFBuilder

            organized_text = self.llm_client.process_images(self.captured_images)
            filename = f"Scanned_Notes_{int(time.time())}.pdf"
            builder = PDFBuilder()
            pdf_path = builder.build_document(organized_text, filename)

            if pdf_path:
                self.output_pdf_path = pdf_path
                self.state = "DONE"
                self.status_message = f"PDF saved: {pdf_path}"
                logger.info(f"OCR complete → {pdf_path}")
                if self.on_pdf_ready:
                    self.on_pdf_ready(pdf_path)
            else:
                self.state = "ERROR"
                self.status_message = "Failed to generate PDF."
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            self.state = "ERROR"
            self.status_message = f"Error: {e}"

    # ----------------------------------------------------------------
    #  FeatureModule interface
    # ----------------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "state": self.state,
            "pages_captured": len(self.captured_images),
            "status_message": self.status_message,
            "output_pdf": self.output_pdf_path,
            "available": self.llm_client is not None,
        }

    def handle_voice_command(self, text: str) -> bool:
        if any(kw in text for kw in ["scan document", "start scan", "open scanner"]):
            self.start_scan()
            return True
        if "capture" in text and self.state == "SCANNING":
            # Actual frame capture is done by SystemManager (it has the camera)
            return True
        if any(kw in text for kw in ["process scan", "finish scan", "done scanning"]):
            self.process_captures()
            return True
        if any(kw in text for kw in ["stop scan", "cancel scan"]):
            self.stop_scan()
            return True
        return False

    def cleanup(self):
        self.state = "IDLE"
        self.captured_images = []
