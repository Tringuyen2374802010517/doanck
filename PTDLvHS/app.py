from flask import Flask, render_template, request
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms

from src.model import EmbeddingModel
from src.predict import predict_image

app = Flask(__name__)

# ========================
# LOAD MODEL
# ========================
model = EmbeddingModel()
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# ========================
# LOAD DATABASE
# ========================
embeddings = np.load("embeddings.npy")
labels = np.load("labels.npy")
class_names = np.load("class_names.npy", allow_pickle=True)
full_labels = np.load("full_labels.npy", allow_pickle=True)

# ========================
# IMAGE TRANSFORM
# ========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ========================
# HOME PAGE
# ========================
@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            img = Image.open(file).convert("RGB")
            img = transform(img).unsqueeze(0)

            results = predict_image(
                model,
                img,
                embeddings,
                labels,
                class_names,
                full_labels,
                top_k=3
            )

    return render_template("index.html", results=results)

# ========================
# RUN APP
# ========================
if __name__ == "__main__":
    app.run(debug=True)