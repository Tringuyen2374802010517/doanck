import torch
import numpy as np
from torchvision import datasets, transforms
from model import EmbeddingModel
import os

# ========================
# DEVICE
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========================
# PATH (QUAN TRỌNG)
# ========================
BASE_DIR = "/content/doanck/PTDLvHS/data"
train_dir = os.path.join(BASE_DIR, "train")

# ========================
# TRANSFORM
# ========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ========================
# DATASET (CHUẨN)
# ========================
dataset = datasets.ImageFolder(train_dir, transform=transform)

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

for img, label in dataset:
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img).cpu().numpy()

    embeddings.append(emb)
    labels.append(label)

embeddings = np.vstack(embeddings)

# ========================
# SAVE
# ========================
np.save("embeddings.npy", embeddings)
np.save("labels.npy", labels)
np.save("class_names.npy", dataset.classes)

print("✅ Done building database!")
print("Total embeddings:", len(embeddings))