from flask import Flask, render_template, request
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms

from src.model import EmbeddingModel
from src.predict import predict_image

app = Flask(__name__)

model = EmbeddingModel()
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

embeddings = np.load("embeddings.npy")
labels = np.load("labels.npy")
class_names = np.load("class_names.npy")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

@app.route("/", methods=["GET","POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["file"]
        img = Image.open(file).convert("RGB")
        img = transform(img).unsqueeze(0)

        result = predict_image(model, img, embeddings, labels, class_names)

    return render_template("index.html", result=result)

app.run(debug=True)