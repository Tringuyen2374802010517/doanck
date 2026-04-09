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

# thư mục train đã tách ảnh
train_dir = os.path.join(BASE_DIR, "train")

# file CSV đúng theo cấu trúc thư mục của bạn
csv_path = os.path.join(BASE_DIR, "ePillID_data", "all_labels.csv")

# ========================
# CHECK PATH
# ========================
print("Train exists:", os.path.exists(train_dir))
print("CSV exists:", os.path.exists(csv_path))

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Không tìm thấy thư mục train: {train_dir}")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")

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

model_path = "/content/doanck/PTDLvHS/best_model.pth"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ========================
# BUILD DATABASE
# ========================
embeddings = []
labels = []
full_labels = []

for img_path, label in dataset.samples:
    try:
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(img).cpu().numpy()

        embeddings.append(emb)
        labels.append(label)

        filename = os.path.basename(img_path)
        full_label = filename_to_label.get(filename, "Unknown")
        full_labels.append(full_label)

    except Exception as e:
        print(f"Lỗi ảnh {img_path}: {e}")

# ========================
# SAVE
# ========================
if len(embeddings) == 0:
    raise ValueError("Không tạo được embedding nào.")

embeddings = np.vstack(embeddings)

save_dir = "/content/doanck/PTDLvHS"

np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
np.save(os.path.join(save_dir, "labels.npy"), np.array(labels))
np.save(os.path.join(save_dir, "class_names.npy"), np.array(dataset.classes))
np.save(os.path.join(save_dir, "full_labels.npy"), np.array(full_labels))

print("✅ Done building database!")
print("Total embeddings:", len(embeddings))