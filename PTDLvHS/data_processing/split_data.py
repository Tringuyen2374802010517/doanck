import os
import random
import shutil

BASE_DIR = "/content/doanck/PTDLvHS/data"

src = os.path.join(BASE_DIR, "processed")
train_dir = os.path.join(BASE_DIR, "train")
val_dir = os.path.join(BASE_DIR, "val")

split_ratio = 0.8

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

classes = sorted(os.listdir(src))

valid_classes = []

# ===== LỌC CLASS =====
for cls in classes:
    cls_path = os.path.join(src, cls)

    if not os.path.isdir(cls_path):
        continue

    images = [
        img for img in os.listdir(cls_path)
        if img.lower().endswith(('.png','.jpg','.jpeg'))
    ]

    # 🔥 CHỈ GIỮ CLASS >= 10 ẢNH
    if len(images) >= 10:
        valid_classes.append(cls)

print("🔥 Total valid classes:", len(valid_classes))

# ===== SPLIT =====
for cls in valid_classes:
    cls_path = os.path.join(src, cls)
    images = os.listdir(cls_path)

    random.shuffle(images)

    split_idx = int(len(images) * split_ratio)

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    if len(train_imgs) == 0 or len(val_imgs) == 0:
        continue

    train_cls_dir = os.path.join(train_dir, cls)
    val_cls_dir = os.path.join(val_dir, cls)

    os.makedirs(train_cls_dir, exist_ok=True)
    os.makedirs(val_cls_dir, exist_ok=True)

    for img in train_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(train_cls_dir, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(val_cls_dir, img)
        )

    print(f"✅ {cls}: {len(train_imgs)} train | {len(val_imgs)} val")

print("\n🎉 DONE SPLIT DATASET!")