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
# PATH
# ========================
BASE_DIR = "/content/doanck/PTDLvHS/data"

train_path = os.path.join(BASE_DIR, "train")
val_path   = os.path.join(BASE_DIR, "val")

print("Train exists:", os.path.exists(train_path))
print("Val exists:", os.path.exists(val_path))

# ========================
# DEVICE
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========================
# TRANSFORM (TÁCH RIÊNG)
# ========================
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ========================
# DATASET
# ========================
train_dataset = TripletDataset(train_path, transform=train_transform)
val_dataset   = TripletDataset(val_path, transform=val_transform)

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
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

# ========================
# ACCURACY FUNCTION
# ========================
def triplet_accuracy(a, p, n):
    dist_ap = torch.norm(a - p, dim=1)
    dist_an = torch.norm(a - n, dim=1)
    return (dist_ap < dist_an).float().mean().item()

# ========================
# EARLY STOPPING
# ========================
patience = 5
counter = 0

# ========================
# TRAIN CONFIG
# ========================
epochs = 15
best_val_loss = float("inf")

train_losses = []
val_losses = []
train_accs = []
val_accs = []

# ========================
# TRAIN LOOP
# ========================
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_acc = 0

    print(f"\nEpoch {epoch+1}/{epochs}")

    for a, p, n in tqdm(train_loader):
        a, p, n = a.to(device), p.to(device), n.to(device)

        emb_a = model(a)
        emb_p = model(p)
        emb_n = model(n)

        loss = criterion(emb_a, emb_p, emb_n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        acc = triplet_accuracy(emb_a, emb_p, emb_n)
        total_acc += acc

    train_loss = total_loss / len(train_loader)
    train_acc  = total_acc / len(train_loader)

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # ========================
    # VALIDATION
    # ========================
    model.eval()
    val_loss = 0
    val_acc_total = 0

    with torch.no_grad():
        for a, p, n in val_loader:
            a, p, n = a.to(device), p.to(device), n.to(device)

            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)

            loss = criterion(emb_a, emb_p, emb_n)
            val_loss += loss.item()

            acc = triplet_accuracy(emb_a, emb_p, emb_n)
            val_acc_total += acc

    val_loss /= len(val_loader)
    val_acc  = val_acc_total / len(val_loader)

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Train Acc : {train_acc:.4f} | Val Acc : {val_acc:.4f}")

    # ========================
    # SAVE + EARLY STOPPING
    # ========================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Saved best model!")
        counter = 0
    else:
        counter += 1
        print(f"No improvement: {counter}/{patience}")
        if counter >= patience:
            print("⏹ Early stopping")
            break

# ========================
# SMOOTH FUNCTION
# ========================
def smooth_curve(values, weight=0.8):
    smoothed = []
    last = values[0]
    for v in values:
        s = last * weight + (1 - weight) * v
        smoothed.append(s)
        last = s
    return smoothed

# ========================
# PLOT LOSS
# ========================
plt.figure(figsize=(8,5))
plt.plot(smooth_curve(train_losses), label="Train Loss")
plt.plot(smooth_curve(val_losses), label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.savefig("loss.png")
plt.show()

# ========================
# PLOT ACCURACY
# ========================
plt.figure(figsize=(8,5))
plt.plot(smooth_curve(train_accs), label="Train Accuracy")
plt.plot(smooth_curve(val_accs), label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid()
plt.savefig("accuracy.png")
plt.show()