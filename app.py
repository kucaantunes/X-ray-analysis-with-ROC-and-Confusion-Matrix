import os
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Folder to save uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained model (e.g., ResNet)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(device)
model.fc = torch.nn.Linear(model.fc.in_features, 3)  # Adjusting last layer for 3 classes

# Provide the correct path to your model.pth file
model_path = os.path.join(os.getcwd(), "model.pth")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))  # Load your trained model here
else:
    print(f"Model file not found: {model_path}")
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define labels
labels = ["Normal", "COVID-19", "Pneumonia"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    print(f"File saved at: {file_path}")

    try:
        # Perform classification
        image_tensor = preprocess_image(file_path)
        probabilities, prediction = classify_image(image_tensor)
        probabilities_dict = {label: prob for label, prob in zip(labels, probabilities)}

        # Generate ROC curve and confusion matrix
        roc_curve_path = generate_roc_curve(probabilities)
        confusion_matrix_path = generate_confusion_matrix()

        return render_template('results.html', original_image=filename, prediction=prediction, probabilities=probabilities_dict, roc_curve=roc_curve_path, confusion_matrix=confusion_matrix_path)
    except Exception as e:
        print(f"Error during classification: {e}")
        return jsonify({"error": "Error during classification"}), 500

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def classify_image(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        prediction = np.argmax(probabilities)

    return probabilities, labels[prediction]

def generate_roc_curve(probabilities):
    # Mocking ROC curve data for demonstration
    fpr = np.array([0.0, 0.1, 0.2, 0.3, 1.0])
    tpr = np.array([0.0, 0.4, 0.6, 0.8, 1.0])
    roc_auc = 0.85

    # Create an image with white background
    img = Image.new('RGB', (800, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw ROC curve
    draw.line([(x * 800, y * 600) for x, y in zip(fpr, tpr)], fill=(255, 0, 0), width=3)
    draw.line([(0, 0), (800, 600)], fill=(0, 0, 0), width=2)

    # Add labels
    font = ImageFont.load_default()
    draw.text((350, 550), "False Positive Rate", font=font, fill=(0, 0, 0))
    draw.text((10, 10), "True Positive Rate", font=font, fill=(0, 0, 0))
    draw.text((300, 30), f"ROC curve (area = {roc_auc:.2f})", font=font, fill=(0, 0, 0))

    roc_curve_path = os.path.join(app.config['UPLOAD_FOLDER'], "roc_curve.png")
    img.save(roc_curve_path)
    return roc_curve_path

def generate_confusion_matrix():
    # Mocking Confusion Matrix data for demonstration
    cm = np.array([[5, 2, 1],
                   [1, 6, 2],
                   [0, 1, 8]])

    # Calculate metrics
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tn = np.sum(cm) - (fp + fn + tp)
    
    accuracy = np.sum(tp) / np.sum(cm)
    precision = np.mean(tp / (tp + fp))
    recall = np.mean(tp / (tp + fn))
    f1_score = 2 * (precision * recall) / (precision + recall)

    metrics_text = (f"Accuracy: {accuracy:.2f}\n"
                    f"Precision: {precision:.2f}\n"
                    f"Recall: {recall:.2f}\n"
                    f"F1 Score: {f1_score:.2f}\n"
                    f"TP: {tp}\n"
                    f"TN: {tn}\n"
                    f"FP: {fp}\n"
                    f"FN: {fn}")

    # Create an image with white background
    img = Image.new('RGB', (800, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw the grid and labels
    cell_size = 200
    for i in range(4):
        draw.line([(i * cell_size, 0), (i * cell_size, 600)], fill=(0, 0, 0), width=2)
        draw.line([(0, i * cell_size), (800, i * cell_size)], fill=(0, 0, 0), width=2)
    
    # Add the labels and values
    font = ImageFont.load_default()
    for i in range(3):
        for j in range(3):
            value = cm[i, j]
            draw.text((j * cell_size + 90, i * cell_size + 90), str(value), font=font, fill=(0, 0, 0))

    # Add metrics text
    draw.text((10, 650), metrics_text, font=font, fill=(0, 0, 0))

    confusion_matrix_path = os.path.join(app.config['UPLOAD_FOLDER'], "confusion_matrix.png")
    img.save(confusion_matrix_path)
    return confusion_matrix_path

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=3001)
