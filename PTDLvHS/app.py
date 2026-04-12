import os
import json
from datetime import datetime
from flask import Flask, render_template, request
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms

from src.model import EmbeddingModel
from src.predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
HISTORY_FILE = "static/history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========================
# LOAD DATA
# ========================
class_names = np.load("class_names.npy", allow_pickle=True)
num_classes = len(class_names)

model = EmbeddingModel(num_classes=num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

embeddings = np.load("embeddings.npy")
labels = np.load("labels.npy")
full_labels = np.load("full_labels.npy", allow_pickle=True)

# ========================
# TRANSFORM
# ========================
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

# ========================
# HISTORY
# ========================
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

def save_history(data):
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)

    history.append(data)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# ========================
# ROUTE
# ========================
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    images = []

    if request.method == "POST":
        files = request.files.getlist("files")

        for file in files:
            if file:
                filename = datetime.now().strftime("%H%M%S_") + file.filename
                path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(path)

                images.append(path)

                img = Image.open(path).convert("RGB")
                img = transform(img).unsqueeze(0)

                res = predict_image(
                    model,
                    img,
                    embeddings,
                    labels,
                    class_names,
                    full_labels
                )

                results.append(res)

        save_history({
            "time": str(datetime.now()),
            "images": images,
            "results": results
        })

    # 🔥 FIX QUAN TRỌNG
    pairs = list(zip(images, results))

    return render_template("index.html", pairs=pairs)

# ========================
# RUN
# ========================
if __name__ == "__main__":
    app.run(debug=True)