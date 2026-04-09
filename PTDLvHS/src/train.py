# train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import EmbeddingModel
from dataset import TripletDataset

# =====================
# DEVICE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# =====================
# PATH
# =====================
BASE_DIR = "/content/doanck/PTDLvHS/data"
train_dir = os.path.join(BASE_DIR, "train")
val_dir = os.path.join(BASE_DIR, "val")

print("Train exists:", os.path.exists(train_dir))
print("Val exists:", os.path.exists(val_dir))

# =====================
# TRANSFORM
# =====================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =====================
# DATASET
# =====================
train_dataset = TripletDataset(train_dir, transform=train_transform, length=10000)
val_dataset = TripletDataset(val_dir, transform=val_transform, length=4000)

num_classes = len(train_dataset.classes)
print("Number of classes:", num_classes)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# =====================
# MODEL
# =====================
model = EmbeddingModel(num_classes=num_classes).to(device)

# =====================
# LOSS
# =====================
criterion_triplet = nn.TripletMarginLoss(margin=0.5)
criterion_cls = nn.CrossEntropyLoss()

# =====================
# OPTIMIZER
# =====================
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

scaler = torch.amp.GradScaler("cuda")

# =====================
# TRAIN CONFIG
# =====================
num_epochs = 40
best_val_loss = float("inf")
patience = 8
counter = 0

train_losses = []
val_losses = []
train_accs = []
val_accs = []

# =====================
# TRAIN LOOP
# =====================
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for anchor, positive, negative, label in tqdm(train_loader):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            a_emb, a_logits = model(anchor)
            p_emb, _ = model(positive)
            n_emb, _ = model(negative)

            loss_triplet = criterion_triplet(a_emb, p_emb, n_emb)
            loss_cls = criterion_cls(a_logits, label)

            loss = loss_triplet + 0.5 * loss_cls

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        d_pos = torch.norm(a_emb - p_emb, dim=1)
        d_neg = torch.norm(a_emb - n_emb, dim=1)

        correct += (d_pos < d_neg).sum().item()
        total += anchor.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # =====================
    # VALIDATION
    # =====================
    model.eval()
    val_running_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for anchor, positive, negative, label in tqdm(val_loader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            label = label.to(device)

            a_emb, a_logits = model(anchor)
            p_emb, _ = model(positive)
            n_emb, _ = model(negative)

            loss_triplet = criterion_triplet(a_emb, p_emb, n_emb)
            loss_cls = criterion_cls(a_logits, label)

            loss = loss_triplet + 0.5 * loss_cls
            val_running_loss += loss.item()

            d_pos = torch.norm(a_emb - p_emb, dim=1)
            d_neg = torch.norm(a_emb - n_emb, dim=1)

            val_correct += (d_pos < d_neg).sum().item()
            val_total += anchor.size(0)

    val_loss = val_running_loss / len(val_loader)
    val_acc = val_correct / val_total

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Train Acc : {train_acc:.4f} | Val Acc : {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model!")
        counter = 0
    else:
        counter += 1
        print(f"No improvement: {counter}/{patience}")

    if counter >= patience:
        print("Early stopping")
        break

# =====================
# PLOT LOSS
# =====================
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# =====================
# PLOT ACCURACY
# =====================
plt.figure(figsize=(8, 5))
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid(True)
plt.show()