import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import os

from model import EmbeddingModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/content/doanck/PTDLvHS/data"
train_dir = os.path.join(BASE_DIR,"train")

tf = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor()
])

ds = datasets.ImageFolder(train_dir,tf)

model = EmbeddingModel(len(ds.classes)).to(device)
model.load_state_dict(torch.load("best_model.pth",map_location=device))
model.eval()

embeddings = []
labels = []
full_labels = []

for path, label in ds.samples:
    img = Image.open(path).convert("RGB")
    img = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img)

    emb = emb.cpu().numpy()
    emb = emb / np.linalg.norm(emb)

    embeddings.append(emb)
    labels.append(label)

    # 🔥 FIX: thêm full label (tên file)
    filename = os.path.basename(path)
    full_labels.append(filename)

embeddings = np.vstack(embeddings)

np.save("embeddings.npy", embeddings)
np.save("labels.npy", labels)
np.save("class_names.npy", ds.classes)
np.save("full_labels.npy", full_labels)  # 🔥 FIX

print("DONE BUILD DATABASE")