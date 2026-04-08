import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import EmbeddingModel
from dataset import TripletDataset

# ========================
# PATH
# ========================
BASE_DIR = "/content/doanck/PTDLvHS/data"
train_path = os.path.join(BASE_DIR, "train")
val_path = os.path.join(BASE_DIR, "val")

print("Train exists:", os.path.exists(train_path))
print("Val exists:", os.path.exists(val_path))

# ========================
# DEVICE
# ========================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: cuda")
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except:
        print("Không lấy được tên GPU")
else:
    device = torch.device("cpu")
    print("Using device: cpu")

# ========================
# TRANSFORM
# ========================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.15, 0.15, 0.15),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ========================
# DATASET
# ========================
train_dataset = TripletDataset(train_path, transform=train_transform)
val_dataset = TripletDataset(val_path, transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# ========================
# MODEL
# ========================
model = EmbeddingModel().to(device)

# ========================
# LOSS + OPTIMIZER
# ========================
criterion = nn.TripletMarginLoss(margin=1.0)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-5,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.3,
    patience=2,
)

# ========================
# MIXED PRECISION
# ========================
scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

# ========================
# ACCURACY FUNCTION
# ========================
def triplet_accuracy(a, p, n):
    dist_ap = torch.norm(a - p, dim=1)
    dist_an = torch.norm(a - n, dim=1)
    return (dist_ap < dist_an).float().mean().item()

# ========================
# SMOOTH FUNCTION
# ========================
def smooth_curve(values, weight=0.95):
    smoothed = []
    last = values[0]
    for v in values:
        s = last * weight + (1 - weight) * v
        smoothed.append(s)
        last = s
    return smoothed

# ========================
# EARLY STOPPING
# ========================
patience = 5
counter = 0
best_val_loss = float("inf")

# ========================
# TRAIN CONFIG
# ========================
epochs = 35
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

    print(f"\nEpoch {epoch + 1}/{epochs}")

    for a, p, n in tqdm(train_loader):
        a = a.to(device)
        p = p.to(device)
        n = n.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                emb_a = model(a)
                emb_p = model(p)
                emb_n = model(n)
                loss = criterion(emb_a, emb_p, emb_n)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)
            loss = criterion(emb_a, emb_p, emb_n)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_acc += triplet_accuracy(emb_a, emb_p, emb_n)

    train_loss = total_loss / len(train_loader)
    train_acc = total_acc / len(train_loader)

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # ========================
    # VALIDATION
    # ========================
    model.eval()
    total_val_loss = 0
    total_val_acc = 0

    with torch.no_grad():
        for a, p, n in val_loader:
            a = a.to(device)
            p = p.to(device)
            n = n.to(device)

            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)

            loss = criterion(emb_a, emb_p, emb_n)

            total_val_loss += loss.item()
            total_val_acc += triplet_accuracy(emb_a, emb_p, emb_n)

    val_loss = total_val_loss / len(val_loader)
    val_acc = total_val_acc / len(val_loader)

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Train Acc : {train_acc:.4f} | Val Acc : {val_acc:.4f}")

    scheduler.step(val_loss)

    # ========================
    # SAVE MODEL + EARLY STOP
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
# PLOT LOSS
# ========================
plt.figure(figsize=(8, 5))
plt.plot(smooth_curve(train_losses), label="Train Loss")
plt.plot(smooth_curve(val_losses), label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss.png")
plt.show()

# ========================
# PLOT ACCURACY
# ========================
plt.figure(figsize=(8, 5))
plt.plot(smooth_curve(train_accs), label="Train Accuracy")
plt.plot(smooth_curve(val_accs), label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid(True)
plt.savefig("accuracy.png")
plt.show()
