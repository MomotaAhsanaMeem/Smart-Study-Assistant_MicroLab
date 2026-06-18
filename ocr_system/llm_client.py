from google import genai
from PIL import Image
import cv2
import logging

logger = logging.getLogger(__name__)

class GeminiOCRClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Gemini API key is required.")
        
        # Use the modern google.genai SDK
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-3-flash-preview'
        logger.info(f"GeminiOCRClient initialized successfully with model {self.model_name}.")

    def process_images(self, cv2_images: list) -> str:
        logger.info(f"Uploading {len(cv2_images)} images to Gemini for OCR analysis...")
        
        # Convert BGR (OpenCV) to RGB (PIL)
        pil_images = []
        for img in cv2_images:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb_img))

        prompt = """
        You are an expert document transcriber and structural editor.
        I have provided images of notes, book pages, or handwriting.
        
        Please do the following:
        1. Read and transcribe all text found in these images accurately.
        2. Fix any obvious spelling or grammatical errors (especially if it's messy handwriting).
        3. Organize the transcribed text logically.
        4. Provide a clear Title at the top.
        5. Use bullet points and paragraphs where appropriate.
        
        IMPORTANT FORMATTING RULES:
        - Return ONLY the organized text.
        - DO NOT use markdown symbols like asterisks (*) or hashes (#).
        - Use ALL CAPS for section headers.
        - Use standard dashes (-) for bullet points.
        - Add empty lines between paragraphs to make it readable.
        This formatting is strict because the output will be piped directly into a basic PDF renderer.
        """
        
        try:
            # Send prompt + images using the modern SDK
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt] + pil_images
            )
            logger.info("OCR analysis complete.")
            return response.text
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            return f"Error occurred during LLM processing: {str(e)}"
