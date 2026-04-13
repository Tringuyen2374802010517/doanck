import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import EmbeddingModel
from dataset import TripletDataset

# ===== SMOOTH =====
def smooth_curve(values, factor=0.9):
    smoothed = []
    for v in values:
        if smoothed:
            smoothed.append(smoothed[-1]*factor + v*(1-factor))
        else:
            smoothed.append(v)
    return smoothed

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== PATH =====
BASE_DIR = "/content/doanck/PTDLvHS/data"
train_dir = os.path.join(BASE_DIR, "train")
val_dir = os.path.join(BASE_DIR, "val")

# ===== AUGMENTATION (STRONGER) =====
train_tf = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.4,0.4,0.4),
    transforms.ToTensor()
])

val_tf = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ToTensor()
])

# ===== DATA =====
train_ds = TripletDataset(train_dir, train_tf, length=5000)
val_ds = TripletDataset(val_dir, val_tf, length=1000)

print("Train classes:", len(train_ds.classes))
print("Val classes:", len(val_ds.classes))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# ===== MODEL =====
model = EmbeddingModel(len(train_ds.classes)).to(device)

# ===== LOSS =====
triplet = nn.TripletMarginLoss(margin=0.3)   # 🔥 giảm margin
ce = nn.CrossEntropyLoss(label_smoothing=0.1)

# ===== OPTIMIZER =====
opt = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-5   # 🔥 giảm LR
)

# ===== LOG =====
train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_acc = 0

EPOCHS = 40

# ===== TRAIN LOOP =====
for epoch in range(EPOCHS):
    print(f"\n================ EPOCH {epoch+1}/{EPOCHS} ================")

    # ===== TRAIN =====
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader)

    for a,p,n,label in pbar:
        a,p,n,label = a.to(device),p.to(device),n.to(device),label.to(device)

        opt.zero_grad()

        a_emb,a_log = model(a, label)
        p_emb = model(p)
        n_emb = model(n)

        loss_triplet = triplet(a_emb,p_emb,n_emb)
        loss_cls = ce(a_log,label)

        # 🔥 FIX CHÍNH
        loss = loss_triplet + 0.7 * loss_cls

        loss.backward()
        opt.step()

        total_loss += loss.item()

        _, preds = torch.max(a_log, 1)
        correct += (preds == label).sum().item()
        total += label.size(0)

        # ===== LIVE LOG =====
        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "acc": f"{correct/total:.3f}"
        })

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

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

            loss = triplet(a_emb,p_emb,n_emb) + 0.7 * ce(a_log,label)
            v_loss += loss.item()

            _, preds = torch.max(a_log, 1)
            v_correct += (preds == label).sum().item()
            v_total += label.size(0)

    val_loss = v_loss / len(val_loader)
    val_acc = v_correct / v_total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"\n📊 RESULT:")
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

    # ===== SAVE BEST =====
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("🔥 Saved BEST model!")

# ===== PLOT =====
plt.plot(smooth_curve(train_losses), label="Train Loss")
plt.plot(smooth_curve(val_losses), label="Val Loss")
plt.legend()
plt.savefig("loss_curve.png")

plt.figure()
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.savefig("accuracy_curve.png")

print("\n🎉 TRAIN DONE")