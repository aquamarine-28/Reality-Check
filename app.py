import cv2 # OpenCV for image manipulation
from flask import Flask, request, render_template, redirect, url_for # Flask framework for handling web requests
from werkzeug.utils import secure_filename # Security utils for file names
from keras.applications.imagenet_utils import decode_predictions # Keras utilities for ImageNet
import os # Operating system interfaces
k2 = os.environ.get('API_KEY_MI')
import shutil # High-level file operations
import ultralytics  # YOLO package from Ultralytics
import json # JSON encoder and decoder
from ultralytics import YOLO # YOLO model from Ultralytics
from PIL import Image, __version__ as PIL_VERSION # Python Imaging Library
import numpy as np # Numerical Python for array operations
import yaml # YAML parser and emitter

# Function to load class labels from a YAML file
def load_class_labels(yaml_file = 'C:\\Users\\horiz\\Downloads\\data.yaml'): #Change to where the data.yml is stored
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
        return data['names'] # 'names' should be a key in your YAML file containing class labels
# Load class labels into a variable for later use
class_labels = load_class_labels()
# Pre-flight checks by Ultralytics YOLO
ultralytics.checks()
app = Flask(__name__)

# Configuration
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join('C:\\Users\\horiz\\Downloads\\best.pt')  # Change to the path where the model is stored
print(f"Does the model file exist at '{MODEL_PATH}'? {'Yes' if os.path.exists(MODEL_PATH) else 'No'}")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit
# ensuring the uphold folder exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your model
model = YOLO(MODEL_PATH)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def letterbox_image(image, new_shape=(416, 416), color=(128, 128, 128)):
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Image has zero size: height or width are 0.")

    # Compute scaling factor
    scale = min(new_shape[0] / height, new_shape[1] / width)
    new_unpad = (int(width * scale), int(height * scale))

    # Assertion to ensure new dimensions are not zero
    assert new_unpad[0] > 0 and new_unpad[1] > 0, "New dimensions must be greater than zero."

    # Debugging output
    print(f"Resizing to new dimensions: {new_unpad}")

    # Proceed to resize
    resized_image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Prepare the canvas
    canvas = np.full((new_shape[1], new_shape[0], 3), color, dtype=np.uint8)
    top, left = ((new_shape[1] - new_unpad[1]) // 2, (new_shape[0] - new_unpad[0]) // 2)
    canvas[top:top+new_unpad[1], left:left+new_unpad[0]] = resized_image
    return canvas


def preprocess(image_path):
    print(f"[DEBUG] Attempting to load and resize image from: {image_path}")

    # Load the image using Pillow
    try:
        with Image.open(image_path) as img:
            print("[INFO] Image loaded successfully.")
        # Resize the image using Pillow's LANCZOS resampling
        img_resized = img.resize((416, 416), Image.LANCZOS)
        print(f"[INFO] Image resized to {(416, 416)} using Pillow.")
    except IOError as e:
        error_message = f"Unable to load image. Error: {e}"
        print(f"[ERROR] {error_message}")
        raise FileNotFoundError(error_message)

    # Convert PIL image to numpy array if further processing is needed
    image_np = np.array(img_resized)
    if image_np.ndim == 2:  # Convert grayscale to RGB if needed
        image_np = np.stack([image_np]*3, axis=-1)
    return image_np

def load_image(image_path, target_size=(416, 416,)):
    try:
        img = Image.open(image_path).convert('RGB')
        # Use the correct resampling method based on Pillow version
        resample = Image.Resampling.LANCZOS if hasattr(Image.Resampling, 'LANCZOS') else Image.ANTIALIAS
        img = img.resize(target_size[:2], resample)
        img = np.array(img) / 255.0
        return img.reshape((1, *img.shape))
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def process_yolo_output(yhat, classes):
    if not results:
        return ["No detections"]
    result = results[0]  # Assuming there's at least one Results object in the list
    formatted_results = []

    if result.boxes:  # Ensure there are bounding boxes
        # Process each detection in the first image's results
        for box in result.boxes.xyxy[0]:  # xyxy is the attribute for bounding box coordinates
            x1, y1, x2, y2, conf, class_id = box.cpu().numpy()  # Convert tensor to numpy array
            class_id = int(class_id)  # Convert class_id to integer
            label = classes[class_id]  # Get the label corresponding to the class_id
            formatted_results.append(f"{label}: {conf:.2f}%")  # Append formatted string to results list
    else:
        return ["No boxes found"]  # Return if no bounding boxes are found

    return formatted_results

def predict_image(image_path):
    results = None
    try:
        image = load_image(image_path)
        if image is None:
            raise ValueError("Failed to load image properly.")
        results = model.predict(image)
        if not results:
            return render_template('index.html', prediction="No results found")
        predictions = process_yolo_output(results, class_labels)
        return render_template('index.html', prediction=' '.join(predictions))
    except Exception as e:
        print(f"Error in processing the image: {e}")
        return render_template('error.html', error=str(e))



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and prediction."""
    filepath = None  # Initialize filepath to None
    if request.method == 'POST':
        file = request.files.get('file')  # Safer way to access files
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)  # Save the file
                prediction_result = predict_image(filepath)  # Process the image
            except Exception as e:
                print(f"Error in processing image: {e}")
                return render_template('error.html', error=str(e))
            finally:
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)  # Ensure file is removed after processing
                return render_template('result.html', result=prediction_result)
        else:
            return redirect(request.url)  # Redirect if file is not present or not allowed
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
