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

# ===== TRANSFORM =====
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(0, translate=(0.1,0.1)),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor()
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ===== DATA =====
train_ds = TripletDataset(train_dir, train_tf, length=8000)
val_ds = TripletDataset(val_dir, val_tf, length=2000)

train_loader = DataLoader(train_ds, batch_size=48, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=48)

# ===== MODEL =====
model = EmbeddingModel(len(train_ds.classes)).to(device)

# ===== LOSS =====
triplet = nn.TripletMarginLoss(margin=1.0)
ce = nn.CrossEntropyLoss(label_smoothing=0.1)

def contrastive_loss(a, p, n):
    pos = torch.norm(a - p, dim=1)
    neg = torch.norm(a - n, dim=1)
    return torch.mean(pos) + torch.mean(torch.relu(1.0 - neg))

# ===== OPTIM =====
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)

# ===== SAVE LIST =====
train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []

best_val = 0
EPOCHS = 80

# ===== TRAIN LOOP =====
for epoch in range(EPOCHS):
    print(f"\n===== EPOCH {epoch+1}/{EPOCHS} =====")

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for a,p,n,label in tqdm(train_loader):
        a,p,n,label = a.to(device),p.to(device),n.to(device),label.to(device)

        opt.zero_grad()

        a_emb, logits = model(a, label)
        p_emb = model(p)
        n_emb = model(n)

        loss_triplet = triplet(a_emb, p_emb, n_emb)
        loss_con = contrastive_loss(a_emb, p_emb, n_emb)
        loss_cls = ce(logits, label)

        # 🔥 MULTI-HEAD LOSS (chuẩn paper)
        loss = loss_triplet + loss_con + loss_cls

        loss.backward()
        opt.step()

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

            loss = (
                triplet(a_emb,p_emb,n_emb)
                + contrastive_loss(a_emb,p_emb,n_emb)
                + ce(logits,label)
            )

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

    scheduler.step()

    # ===== SAVE BEST =====
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("🔥 SAVE BEST MODEL")

# ===== SMOOTH FUNCTION =====
def smooth(y, box=5):
    return np.convolve(y, np.ones(box)/box, mode='same')

# ===== PLOT ACCURACY =====
plt.figure()
plt.plot(smooth(train_acc_list), label="Train Acc")
plt.plot(smooth(val_acc_list), label="Val Acc")
plt.legend()
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("accuracy.png")

# ===== PLOT LOSS =====
plt.figure()
plt.plot(smooth(train_loss_list), label="Train Loss")
plt.plot(smooth(val_loss_list), label="Val Loss")
plt.legend()
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss.png")

print("\n✅ DONE TRAIN + SAVED CHART")