import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import EmbeddingModel
from dataset import TripletDataset

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== PATH =====
BASE_DIR = "/content/doanck/PTDLvHS/data"
train_dir = os.path.join(BASE_DIR, "train")
val_dir = os.path.join(BASE_DIR, "val")

# ===== TRANSFORM (ỔN ĐỊNH HƠN) =====
train_tf = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1,0.1,0.1),
    transforms.ToTensor()
])

val_tf = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor()
])

# ===== DATASET =====
train_ds = TripletDataset(train_dir, train_tf, length=8000)
val_ds = TripletDataset(val_dir, val_tf, length=1000)

print("Train classes:", len(train_ds.classes))
print("Val classes:", len(val_ds.classes))

# ===== DATALOADER =====
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, num_workers=2)

# ===== MODEL =====
model = EmbeddingModel(len(train_ds.classes)).to(device)

# ===== LOSS =====
triplet = nn.TripletMarginLoss(margin=1.0)
ce = nn.CrossEntropyLoss(label_smoothing=0.1)

# ===== OPTIM (GIẢM LR) =====
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# ===== SCHEDULER (MƯỢT CURVE) =====
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

# ===== TRAIN CONFIG =====
EPOCHS = 40
best_val = 0

# ===== HISTORY =====
train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []

print("🔥 START TRAINING...")

# ===== TRAIN LOOP =====
for epoch in range(EPOCHS):
    print(f"\n===== EPOCH {epoch+1}/{EPOCHS} =====")

    # ===== TRAIN =====
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for a,p,n,label in tqdm(train_loader):
        a,p,n,label = a.to(device),p.to(device),n.to(device),label.to(device)

        optimizer.zero_grad()

        a_emb, logits = model(a, label)
        p_emb = model(p)
        n_emb = model(n)

        loss_triplet = triplet(a_emb, p_emb, n_emb)
        loss_cls = ce(logits, label)

        loss = loss_cls + 0.5 * loss_triplet

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(logits, 1)
        correct += (preds == label).sum().item()
        total += label.size(0)

    train_acc = correct / total
    train_loss = total_loss / len(train_loader)

    # ===== VALID =====
    model.eval()
    v_loss = 0
    v_correct = 0
    v_total = 0

    with torch.no_grad():
        for a,p,n,label in val_loader:
            a,p,n,label = a.to(device),p.to(device),n.to(device),label.to(device)

            a_emb, logits = model(a, label)
            p_emb = model(p)
            n_emb = model(n)

            loss = ce(logits, label) + 0.5 * triplet(a_emb, p_emb, n_emb)

            v_loss += loss.item()

            _, preds = torch.max(logits, 1)
            v_correct += (preds == label).sum().item()
            v_total += label.size(0)

    val_acc = v_correct / v_total
    val_loss = v_loss / len(val_loader)

    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

    # ===== SAVE HISTORY =====
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    # ===== SAVE BEST =====
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("🔥 SAVE BEST MODEL")

    # ===== STEP LR =====
    scheduler.step()

# ===== SMOOTH FUNCTION =====
def smooth(y, box=5):
    return np.convolve(y, np.ones(box)/box, mode='same')

# ===== PLOT =====
plt.figure()
plt.plot(smooth(train_acc_list), label="Train Acc")
plt.plot(smooth(val_acc_list), label="Val Acc")
plt.legend()
plt.title("Accuracy")
plt.savefig("accuracy.png")

plt.figure()
plt.plot(smooth(train_loss_list), label="Train Loss")
plt.plot(smooth(val_loss_list), label="Val Loss")
plt.legend()
plt.title("Loss")
plt.savefig("loss.png")

print("\n✅ DONE TRAIN + CHART SMOOTH")