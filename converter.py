import pytesseract
from pdf2image import convert_from_path
import fitz
import sys
import glob
import os
import cv2
import numpy as np
from PIL import Image

# Set TESSDATA_PREFIX for safety (adjust if needed based on Tesseract version)
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/tessdata'  # For Tesseract 5

def preprocess_image(img_pil):
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # Get grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Noise removal
    denoised = cv2.medianBlur(gray, 5)
    
    # Thresholding
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Opening - remove small noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Convert back to PIL
    processed_pil = Image.fromarray(opening)
    return processed_pil

def convert_pdf(input_path, output_path, dpi=500):
    print("Converting PDF to images...")
    images = convert_from_path(input_path, dpi=dpi)

    print("Running OCR...")
    snippets = []
    for i, img in enumerate(images):
        print(f"Page {i+1}")
        # Preprocess the image for better OCR accuracy
        processed_img = preprocess_image(img)
        # Use Bengali as primary + English for OCR, with PSM 6 for block of text
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(processed_img, extension='pdf', lang='ben+eng', config='--psm 6')
        snippets.append(pdf_bytes)

    print("Merging...")
    doc = fitz.open()
    for bytes_pdf in snippets:
        temp = fitz.open("pdf", bytes_pdf)
        doc.insert_pdf(temp)
        temp.close()

    doc.save(output_path)
    doc.close()
    print(f"Done: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # For single-file mode, optionally add a check to skip if input looks like an output
        if sys.argv[1].lower().endswith('_ocr.pdf'):
            print(f"Skipping {sys.argv[1]}: Appears to be an OCR output file.")
            sys.exit(0)
        convert_pdf(sys.argv[1], sys.argv[2])
    else:
        pdf_files = glob.glob('*.pdf')
        input_files = [f for f in pdf_files if not f.lower().endswith('_ocr.pdf')]
        for pdf_file in input_files:
            output_path = pdf_file.replace('.pdf', '_ocr.pdf')
            convert_pdf(pdf_file, output_path)
        if not pdf_files:
            print("No PDF files found in the current directory.")
        elif not input_files:
            print("No input PDFs found; only OCR outputs present.")
