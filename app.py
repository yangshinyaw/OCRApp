from flask import Flask, request, render_template, send_from_directory, url_for
import os
from dotenv import load_dotenv
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2

# Load environment variables from .env file
load_dotenv()

# Get the current working directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Set the paths for tesseract.exe and tessdata within the Tesseract-OCR folder
tesseract_cmd = os.path.join(project_root, 'Tesseract-OCR', 'tesseract.exe')
tessdata_prefix = os.path.join(project_root, 'Tesseract-OCR', 'tessdata')

# Debugging prints
print(f"Tesseract CMD: {tesseract_cmd}")
print(f"Tessdata Prefix: {tessdata_prefix}")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

# Configure Tesseract for pytesseract
if not os.path.isfile(tesseract_cmd):
    app.logger.error(f"Tesseract executable not found at: {tesseract_cmd}")
else:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

if not os.path.isdir(tessdata_prefix):
    app.logger.error(f"Tesseract data directory not found at: {tessdata_prefix}")
else:
    os.environ['TESSDATA_PREFIX'] = tessdata_prefix

# Create directories if they do not exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def enlarge_image(image_path, scale_factor=2):
    try:
        image = Image.open(image_path)
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        enlarged_image = image.resize(new_size, Image.LANCZOS)
        enlarged_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'enlarged_image.png')
        enlarged_image.save(enlarged_image_path)
        app.logger.info(f"Enlarged image saved to: {enlarged_image_path}")
        return enlarged_image_path
    except Exception as e:
        app.logger.error(f"Error in enlarge_image: {e}")
        return None

def enhance_image(image_path):
    try:
        image = Image.open(image_path)
        image = image.convert('L')  # Convert to grayscale
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.5)
        image = image.filter(ImageFilter.MedianFilter())
        enhanced_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'enhanced_image.png')
        image.save(enhanced_image_path)
        app.logger.info(f"Enhanced image saved to: {enhanced_image_path}")
        return enhanced_image_path
    except Exception as e:
        app.logger.error(f"Error in enhance_image: {e}")
        return None

def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
        preprocessed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'preprocessed_image.png')
        cv2.imwrite(preprocessed_image_path, binary_image)
        app.logger.info(f"Preprocessed image saved to: {preprocessed_image_path}")
        return preprocessed_image_path
    except Exception as e:
        app.logger.error(f"Error in preprocess_image: {e}")
        return None

def ocr_image(image_path):
    try:
        enlarged_image_path = enlarge_image(image_path)
        if not enlarged_image_path:
            return "", "", "", ""
        enhanced_image_path = enhance_image(enlarged_image_path)
        if not enhanced_image_path:
            return "", "", "", ""
        preprocessed_image_path = preprocess_image(enhanced_image_path)
        if not preprocessed_image_path:
            return "", "", "", ""

        custom_config = r'--psm 6'
        app.logger.info(f"Preprocessed Image Path: {preprocessed_image_path}")

        text = pytesseract.image_to_string(Image.open(preprocessed_image_path), config=custom_config, lang='eng')
        app.logger.info(f"Extracted Text: {text}")
        return text, enlarged_image_path, enhanced_image_path, preprocessed_image_path
    except pytesseract.TesseractNotFoundError as e:
        app.logger.error(f"Tesseract Not Found: {e}")
        return "", "", "", ""
    except Exception as e:
        app.logger.error(f"Error in ocr_image: {e}")
        return "", "", "", ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            app.logger.error("No file part in request")
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            app.logger.error("No selected file")
            return "No selected file", 400
        if file:
            app.logger.info(f"Received file: {file.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            extracted_text, enlarged_image_path, enhanced_image_path, preprocessed_image_path = ocr_image(file_path)
            app.logger.info(f"Extracted Text: {extracted_text}")
            return render_template('result.html', text=extracted_text,
                                   filename=file.filename,
                                   enlarged_image=url_for('processed_file', filename='enlarged_image.png'),
                                   enhanced_image=url_for('processed_file', filename='enhanced_image.png'),
                                   preprocessed_image=url_for('processed_file', filename='preprocessed_image.png'))
    except Exception as e:
        app.logger.error(f"Error in upload_file: {e}")
        return str(e), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False)
