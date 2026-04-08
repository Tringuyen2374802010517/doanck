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
device = torch.device("cuda")
print("Using device:", device)
print("GPU:", torch.cuda.get_device_name(0))

# ========================
# TRANSFORM
# ========================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True)

# ========================
# MODEL
# ========================
model = EmbeddingModel().to(device)

# ========================
# LOSS + OPTIMIZER
# ========================
criterion = nn.TripletMarginLoss(margin=1.2)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-5,
    weight_decay=1e-4,
    betas=(0.9, 0.99)  # 🔥 giúp curve mượt hơn
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2,
)

# ========================
# AMP (NEW API FIX WARNING)
# ========================
scaler = torch.amp.GradScaler("cuda")

# ========================
# ACCURACY
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
# CONFIG
# ========================
epochs = 20
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

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)

            loss = criterion(emb_a, emb_p, emb_n)

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

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

    scheduler.step(val_loss)

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
# SMOOTH (EMA XỊN)
# ========================
def smooth_curve(values, weight=0.92):
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return smoothed

# ========================
# PLOT LOSS (RAW + SMOOTH)
# ========================
plt.figure(figsize=(8,5))

plt.plot(train_losses, alpha=0.3, linestyle='--', label="Train Loss (raw)")
plt.plot(val_losses, alpha=0.3, linestyle='--', label="Val Loss (raw)")

plt.plot(smooth_curve(train_losses), linewidth=2, label="Train Loss (smooth)")
plt.plot(smooth_curve(val_losses), linewidth=2, label="Val Loss (smooth)")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.savefig("loss.png")
plt.show()

# ========================
# PLOT ACC (RAW + SMOOTH)
# ========================
plt.figure(figsize=(8,5))

plt.plot(train_accs, alpha=0.3, linestyle='--', label="Train Acc (raw)")
plt.plot(val_accs, alpha=0.3, linestyle='--', label="Val Acc (raw)")

plt.plot(smooth_curve(train_accs), linewidth=2, label="Train Acc (smooth)")
plt.plot(smooth_curve(val_accs), linewidth=2, label="Val Acc (smooth)")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid()
plt.savefig("accuracy.png")
plt.show()