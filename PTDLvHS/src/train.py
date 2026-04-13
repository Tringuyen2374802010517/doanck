import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import EmbeddingModel
from dataset import TripletDataset

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== PATH =====
BASE_DIR = "/content/doanck/PTDLvHS/data"
train_dir = os.path.join(BASE_DIR, "train")
val_dir = os.path.join(BASE_DIR, "val")

# ===== TRANSFORM (GIỐNG PAPER - 224) =====
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor()
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ===== DATA =====
train_ds = TripletDataset(train_dir, train_tf, length=5000)
val_ds = TripletDataset(val_dir, val_tf, length=1000)

print("Train classes:", len(train_ds.classes))
print("Val classes:", len(val_ds.classes))

# 🔥 batch lớn hơn (giống paper)
train_loader = DataLoader(train_ds, batch_size=48, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=48)

# ===== MODEL =====
model = EmbeddingModel(len(train_ds.classes)).to(device)

# ===== LOSS =====
triplet = nn.TripletMarginLoss(margin=1.0)
ce = nn.CrossEntropyLoss(label_smoothing=0.1)

# 🔥 Contrastive loss (giả lập paper)
def contrastive_loss(a, p, n):
    pos = torch.norm(a - p, dim=1)
    neg = torch.norm(a - n, dim=1)
    return torch.mean(pos) + torch.mean(torch.relu(1.0 - neg))

# ===== OPTIMIZER =====
opt = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

# 🔥 scheduler giống paper
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3)

best_val_acc = 0
EPOCHS = 40

# ===== TRAIN LOOP =====
for epoch in range(EPOCHS):
    print(f"\n========= EPOCH {epoch+1}/{EPOCHS} =========")

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader)

    for a,p,n,label in pbar:
        a,p,n,label = a.to(device),p.to(device),n.to(device),label.to(device)

        opt.zero_grad()

        # ===== FORWARD =====
        a_emb,a_log = model(a, label)
        p_emb = model(p)
        n_emb = model(n)

        # ===== LOSSES =====
        loss_triplet = triplet(a_emb,p_emb,n_emb)
        loss_cls = ce(a_log,label)
        loss_con = contrastive_loss(a_emb, p_emb, n_emb)

        # 🔥 BALANCE GIỐNG PAPER
        loss = loss_triplet + loss_con + 0.1 * loss_cls

        loss.backward()
        opt.step()

        total_loss += loss.item()

        # ===== ACC =====
        _, preds = torch.max(a_log, 1)
        correct += (preds == label).sum().item()
        total += label.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.2f}",
            "acc": f"{correct/total:.3f}"
        })

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

            a_emb,a_log = model(a, label)
            p_emb = model(p)
            n_emb = model(n)

            loss = (
                triplet(a_emb,p_emb,n_emb)
                + contrastive_loss(a_emb,p_emb,n_emb)
                + 0.1 * ce(a_log,label)
            )

            v_loss += loss.item()

            _, preds = torch.max(a_log, 1)
            v_correct += (preds == label).sum().item()
            v_total += label.size(0)

    val_acc = v_correct / v_total
    val_loss = v_loss / len(val_loader)

    print(f"\nTrain Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

    # ===== SAVE BEST =====
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("🔥 Saved BEST model")

    # ===== LR SCHEDULER =====
    scheduler.step(val_loss)

print("\n🎉 TRAIN DONE")