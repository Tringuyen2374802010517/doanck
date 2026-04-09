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

train_dir = os.path.join(BASE_DIR, "small_epillid")
val_dir = os.path.join(BASE_DIR, "small_epillid")

print("Train exists:", os.path.exists(train_dir))
print("Val exists:", os.path.exists(val_dir))

# =====================
# TRANSFORM
# =====================
train_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

# =====================
# DATASET
# =====================
train_dataset = TripletDataset(train_dir, transform=train_transform, length=500)
val_dataset = TripletDataset(val_dir, transform=val_transform, length=100)

num_classes = len(train_dataset.classes)
print("Number of classes:", num_classes)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
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
criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.1)

# =====================
# OPTIMIZER
# =====================
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-5,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)

scaler = torch.amp.GradScaler("cuda")

# =====================
# TRAIN CONFIG
# =====================
num_epochs = 40
best_val_acc = 0

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

        _, preds = torch.max(a_logits, 1)
        correct += (preds == label).sum().item()
        total += label.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # VALIDATION
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

            _, preds = torch.max(a_logits, 1)
            val_correct += (preds == label).sum().item()
            val_total += label.size(0)

    val_loss = val_running_loss / len(val_loader)
    val_acc = val_correct / val_total

    scheduler.step(val_acc)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Train Acc : {train_acc:.4f} | Val Acc : {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model!")

# =====================
# SMOOTH
# =====================
def smooth_curve(values, factor=0.9):
    smoothed = []
    for v in values:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + v * (1 - factor))
        else:
            smoothed.append(v)
    return smoothed

# =====================
# PLOT
# =====================
plt.figure(figsize=(10, 5))
plt.plot(smooth_curve(train_losses), label="Train")
plt.plot(smooth_curve(val_losses), label="Validation")
plt.title("Loss")
plt.legend()
plt.grid()
plt.savefig("loss_curve.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(smooth_curve(train_accs), label="Train")
plt.plot(smooth_curve(val_accs), label="Validation")
plt.title("Accuracy")
plt.legend()
plt.grid()
plt.savefig("accuracy_curve.png")
plt.close()

print("Done")