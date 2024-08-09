from flask import Flask, request, render_template, send_from_directory
import os
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def enlarge_image(image_path, scale_factor=2):
    try:
        image = Image.open(image_path)
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        enlarged_image = image.resize(new_size, Image.LANCZOS)
        enlarged_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'enlarged_image.png')
        enlarged_image.save(enlarged_image_path)
        return enlarged_image_path
    except Exception as e:
        return str(e)

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
        return enhanced_image_path
    except Exception as e:
        return str(e)

def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
        preprocessed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'preprocessed_image.png')
        cv2.imwrite(preprocessed_image_path, binary_image)
        return preprocessed_image_path
    except Exception as e:
        return str(e)

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
        custom_config = r'--oem 3 --psm 3 '
        text = pytesseract.image_to_string(Image.open(preprocessed_image_path), config=custom_config, lang='eng')
        return text, enlarged_image_path, enhanced_image_path, preprocessed_image_path
    except pytesseract.TesseractNotFoundError as e:
        return "", "", "", ""
    except Exception as e:
        return "", "", "", ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        extracted_text, enlarged_image_path, enhanced_image_path, preprocessed_image_path = ocr_image(file_path)
        return render_template('result.html', text=extracted_text,
                               enlarged_image=enlarged_image_path,
                               enhanced_image=enhanced_image_path,
                               preprocessed_image=preprocessed_image_path)
    return "Something went wrong", 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
