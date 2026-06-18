from fpdf import FPDF
import datetime
import logging
import os

logger = logging.getLogger(__name__)

class PDFBuilder:
    def __init__(self):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.add_page()
        
    def build_document(self, text: str, output_filename: str):
        logger.info(f"Generating PDF document: {output_filename}")
        
        # Header
        self.pdf.set_font("Helvetica", 'B', 18)
        self.pdf.cell(0, 10, "AI OCR Transcribed Document", ln=True, align='C')
        
        # Subheader / Timestamp
        self.pdf.set_font("Helvetica", 'I', 10)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.pdf.cell(0, 10, f"Scanned & Organized on {timestamp}", ln=True, align='C')
        self.pdf.line(10, 30, 200, 30)
        self.pdf.ln(10)
        
        # Body
        self.pdf.set_font("Helvetica", size=12)
        
        # Encode safely to latin-1 to avoid FPDF Helvetica font errors with weird LLM unicode chars
        safe_text = text.encode('latin-1', 'replace').decode('latin-1')
        
        self.pdf.multi_cell(0, 7, safe_text)
        
        # Save
        try:
            output_path = os.path.join(os.getcwd(), output_filename)
            self.pdf.output(output_path)
            logger.info(f"PDF successfully saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save PDF: {e}")
            return None
