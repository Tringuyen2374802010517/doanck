import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
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
# DEVICE (tự check GPU)
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ CUDA not available. Running on CPU.")

# ========================
# TRANSFORM (vừa đủ để train đẹp + ổn)
# ========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.15, 0.15, 0.15),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ========================
# DATASET
# ========================
train_dataset = TripletDataset(train_path, transform=train_transform)
val_dataset   = TripletDataset(val_path, transform=val_transform)

# batch size tự động theo thiết bị
batch_size = 96 if torch.cuda.is_available() else 16

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available()
)

# ========================
# MODEL
# ========================
model = EmbeddingModel().to(device)

# ========================
# LOSS
# ========================
criterion = nn.TripletMarginLoss(margin=1.2)

# ========================
# OPTIMIZER
# ========================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=1e-4,
    betas=(0.9, 0.99)
)

# ========================
# TRAIN CONFIG
# ========================
epochs = 30
patience = 8
counter = 0
best_val_loss = float("inf")

# ========================
# SCHEDULER
# ========================
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)

# ========================
# AMP (chỉ bật khi có GPU)
# ========================
scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

# ========================
# ACCURACY FUNCTION
# ========================
def triplet_accuracy(anchor, positive, negative):
    dist_ap = torch.norm(anchor - positive, dim=1)
    dist_an = torch.norm(anchor - negative, dim=1)
    return (dist_ap < dist_an).float().mean().item()

# ========================
# TRAIN LOOP
# ========================
for epoch in range(epochs):
    model.train()

    total_train_loss = 0
    total_train_acc = 0

    print(f"\nEpoch {epoch + 1}/{epochs}")

    for a, p, n in tqdm(train_loader):
        a = a.to(device)
        p = p.to(device)
        n = n.to(device)

        optimizer.zero_grad()

        # ===== GPU (AMP) =====
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

        # ===== CPU =====
        else:
            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)

            loss = criterion(emb_a, emb_p, emb_n)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        total_train_loss += loss.item()
        total_train_acc += triplet_accuracy(emb_a, emb_p, emb_n)

    train_loss = total_train_loss / len(train_loader)
    train_acc  = total_train_acc / len(train_loader)

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
    val_acc  = total_val_acc / len(val_loader)

    # ========================
    # UPDATE LR
    # ========================
    scheduler.step()

    # ========================
    # LOG
    # ========================
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Train Acc : {train_acc:.4f} | Val Acc : {val_acc:.4f}")
    print(f"LR        : {optimizer.param_groups[0]['lr']:.8f}")

    # ========================
    # SAVE BEST MODEL
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

print("\n🎉 Training completed!")
print("Best model saved as: best_model.pth")