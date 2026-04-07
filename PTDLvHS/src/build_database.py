import torch
import numpy as np
from torchvision import datasets, transforms
from model import EmbeddingModel

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("data", transform=transform)

model = EmbeddingModel()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

embeddings = []
labels = []

for img, label in dataset:
    img = img.unsqueeze(0)

    with torch.no_grad():
        emb = model(img).numpy()

    embeddings.append(emb)
    labels.append(label)

embeddings = np.vstack(embeddings)

np.save("embeddings.npy", embeddings)
np.save("labels.npy", labels)
np.save("class_names.npy", dataset.classes)