import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import datasets, transforms
from model import EmbeddingModel
import os

# ========================
# DEVICE
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========================
# PATH
# ========================
BASE_DIR = "/content/doanck/PTDLvHS/data"
train_dir = os.path.join(BASE_DIR, "train")

# CSV gốc chứa label đầy đủ
csv_path = "/content/ePillID_data/all_labels.csv"

# ========================
# TRANSFORM
# ========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ========================
# DATASET
# ========================
dataset = datasets.ImageFolder(train_dir, transform=transform)

# ========================
# LOAD CSV
# ========================
df = pd.read_csv(csv_path)

# map tên file ảnh -> label đầy đủ
filename_to_label = dict(zip(df["images"], df["label"]))

# ========================
# MODEL
# ========================
model = EmbeddingModel().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ========================
# BUILD DATABASE
# ========================
embeddings = []
labels = []
full_labels = []

for img_path, label in dataset.samples:
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img).cpu().numpy()

    embeddings.append(emb)
    labels.append(label)

    filename = os.path.basename(img_path)
    full_label = filename_to_label.get(filename, "Unknown")
    full_labels.append(full_label)

embeddings = np.vstack(embeddings)

# ========================
# SAVE
# ========================
np.save("embeddings.npy", embeddings)
np.save("labels.npy", labels)
np.save("class_names.npy", dataset.classes)
np.save("full_labels.npy", full_labels)

print("✅ Done building database!")
print("Total embeddings:", len(embeddings))