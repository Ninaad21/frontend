from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import hashlib
import timm
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "your_secret_key_here"

# In-memory user store (use a database in production)
users = {}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Load model
model_path = "vit_model_state.pth"
num_classes = 2
class_labels = ["Fresh", "Stale"]
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username in users and users[username] == hash_password(password):
        session['user'] = username
        return redirect(url_for('index'))
    flash('Invalid credentials')
    return redirect(url_for('login_page'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm = request.form['confirm_password']
        if password != confirm:
            flash('Passwords do not match')
            return redirect(url_for('register'))
        if username in users:
            flash('User already exists')
            return redirect(url_for('register'))
        users[username] = hash_password(password)
        flash('Registration successful! Please log in.')
        return redirect(url_for('login_page'))
    return render_template('register.html')

@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('login_page'))
    return render_template('index.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully')
    return redirect(url_for('login_page'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(file).convert('RGB')
        img = transform(img).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            label = class_labels[predicted.item()]

        return jsonify({'label': label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
