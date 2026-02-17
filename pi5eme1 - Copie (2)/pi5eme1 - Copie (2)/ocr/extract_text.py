# ocr/extract_text.py
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import os

DEFAULT_POPPLER_PATH = os.getenv("POPPLER_PATH")
DEFAULT_TESSERACT_CMD = os.getenv("TESSERACT_CMD")

if DEFAULT_TESSERACT_CMD and os.path.exists(DEFAULT_TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_CMD

def extract_text_from_image(image_path, lang="fra+eng"):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, lang=lang)

def extract_text_from_pdf(pdf_path, poppler_path=None, lang="fra+eng"):
    poppler = poppler_path or DEFAULT_POPPLER_PATH
    if not poppler or not os.path.exists(poppler):
        raise FileNotFoundError(f"Poppler non trouvé : {poppler}")
    pages = convert_from_path(pdf_path, poppler_path=poppler)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page, lang=lang) + "\n"
    return text
