import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from model import EmbeddingModel
from dataset import TripletDataset

# ========================
# DEVICE
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========================
# TRANSFORM
# ========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor()
])

# ========================
# DATASET (CHIA TRAIN / VAL)
# ========================
train_dataset = TripletDataset("data/train", transform=transform)
val_dataset   = TripletDataset("data/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)

# ========================
# MODEL
# ========================
model = EmbeddingModel().to(device)



# ========================
# LOSS + OPTIMIZER
# ========================
criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ========================
# TRAIN CONFIG
# ========================
epochs = 40
best_val_loss = float("inf")

train_losses = []
val_losses = []

# ========================
# TRAIN LOOP
# ========================
for epoch in range(epochs):
    model.train()
    total_loss = 0

    print(f"\nEpoch {epoch+1}/{epochs}")

    for i, (a, p, n) in enumerate(tqdm(train_loader)):
        a, p, n = a.to(device), p.to(device), n.to(device)

        emb_a = model(a)
        emb_p = model(p)
        emb_n = model(n)

        loss = criterion(emb_a, emb_p, emb_n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

    # ========================
    # VALIDATION
    # ========================
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for a, p, n in val_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)

            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)

            loss = criterion(emb_a, emb_p, emb_n)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # ========================
    # SAVE BEST MODEL
    # ========================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model!")

# ========================
# PLOT GRAPH
# ========================
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.savefig("loss.png")
plt.show()