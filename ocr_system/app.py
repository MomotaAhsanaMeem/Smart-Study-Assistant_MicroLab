import cv2
import os
import time
import logging
import sys
import shutil
import threading
import numpy as np

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

from camera import PiCameraStream
from llm_client import GeminiOCRClient
from pdf_builder import PDFBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ScannerState:
    SCANNING = "SCANNING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    ERROR = "ERROR"

class OCRScannerApp:
    def __init__(self):
        self.headless = not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
        api_key = os.environ.get('GEMINI_API_KEY')
        
        self.llm_client = GeminiOCRClient(api_key=api_key)
        self.pdf_builder = PDFBuilder()
        self.cap = None
        self.captured_images = []
        self.state = ScannerState.SCANNING
        self.output_pdf_path = ""
        self.status_message = "Press 'C' to Capture, 'ENTER' to Process"

    def start(self):
        if shutil.which("rpicam-vid"):
            logger.info("Initializing hardware-accelerated PiCameraStream for OCR...")
            self.cap = PiCameraStream(width=1280, height=720, framerate=30).start()
            
        if self.cap is None or not self.cap.isOpened():
            logger.warning("Falling back to standard V4L2...")
            for idx in [0, 1, 2, 4, 6]:
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                # Set high resolution for OCR
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                if cap.isOpened():
                    success, _ = cap.read()
                    if success:
                        self.cap = cap
                        break
                    else:
                        cap.release()

        if self.cap is None or not self.cap.isOpened():
            logger.error("Could not open any camera.")
            return

        logger.info("OCR Scanner running. Use UI to capture documents.")
        
        try:
            self._run_loop()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        finally:
            self.cleanup()

    def _run_loop(self):
        while True:
            if self.state == ScannerState.SCANNING:
                success, frame = self.cap.read()
                if not success:
                    time.sleep(0.1)
                    continue
                
                display_frame = frame.copy()
                
                if not self.headless:
                    # Draw UI overlay
                    cv2.putText(display_frame, f"Captured Pages: {len(self.captured_images)}", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                
                    # Add clear on-screen instructions
                    cv2.putText(display_frame, "INSTRUCTIONS:", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, "1. Align your document inside the yellow box.", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, "2. Press 'C' to capture the current page.", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, "3. Press 'ENTER' to send all captures to Gemini LLM.", (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv2.putText(display_frame, self.status_message, (30, display_frame.shape[0] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    
                    # Draw a document alignment guide
                    h, w = display_frame.shape[:2]
                    cv2.rectangle(display_frame, (int(w*0.1), int(h*0.1)), (int(w*0.9), int(h*0.9)), (255, 255, 0), 2)
                    
                    cv2.imshow("AI OCR Document Scanner", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27: # ESC
                        break
                    elif key == ord('c') or key == ord('C'):
                        self.captured_images.append(frame.copy())
                        self.status_message = f"Captured page {len(self.captured_images)}!"
                        logger.info(f"Page {len(self.captured_images)} captured.")
                    elif key == 13: # ENTER
                        if len(self.captured_images) == 0:
                            self.status_message = "No images captured! Press C first."
                        else:
                            self.state = ScannerState.PROCESSING
                            self.status_message = "Processing with AI... Please wait."
                            logger.info("Starting AI processing thread...")
                            threading.Thread(target=self._process_documents_thread).start()
                            
            elif self.state == ScannerState.PROCESSING:
                if not self.headless:
                    loading_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(loading_frame, "Processing documents with Gemini LLM...", (250, 360),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    cv2.putText(loading_frame, "This may take 10-30 seconds depending on page count.", (280, 420),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow("AI OCR Document Scanner", loading_frame)
                    if cv2.waitKey(100) & 0xFF == 27:
                        break
                        
            elif self.state == ScannerState.DONE:
                if not self.headless:
                    done_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(done_frame, "SUCCESS!", (550, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    cv2.putText(done_frame, f"Saved to: {self.output_pdf_path}", (200, 400),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(done_frame, "Press ESC to exit, or 'R' to scan more.", (350, 500),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                    
                    cv2.imshow("AI OCR Document Scanner", done_frame)
                    key = cv2.waitKey(100) & 0xFF
                    if key == 27:
                        break
                    elif key == ord('r') or key == ord('R'):
                        self.captured_images.clear()
                        self.pdf_builder = PDFBuilder() # Reset builder
                        self.status_message = "Press 'C' to Capture, 'ENTER' to Process"
                        self.state = ScannerState.SCANNING

    def _process_documents_thread(self):
        try:
            organized_text = self.llm_client.process_images(self.captured_images)
            filename = f"Scanned_Notes_{int(time.time())}.pdf"
            pdf_path = self.pdf_builder.build_document(organized_text, filename)
            
            if pdf_path:
                self.output_pdf_path = pdf_path
                self.state = ScannerState.DONE
            else:
                self.state = ScannerState.ERROR
        except Exception as e:
            logger.error(f"Processing thread error: {e}")
            self.state = ScannerState.ERROR

    def cleanup(self):
        logger.info("Shutting down scanner...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = OCRScannerApp()
    app.start()
