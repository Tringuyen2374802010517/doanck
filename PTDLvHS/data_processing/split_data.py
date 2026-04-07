import os
import random
import shutil

# ===== CONFIG =====
src = "data/processed"   # folder đã tạo từ CSV
train_dir = "data/train"
val_dir = "data/val"
split_ratio = 0.8        # 80% train, 20% val

# ===== CREATE DIR =====
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# ===== SPLIT =====
for cls in os.listdir(src):
    cls_path = os.path.join(src, cls)

    # bỏ qua nếu không phải folder
    if not os.path.isdir(cls_path):
        continue

    images = [img for img in os.listdir(cls_path)
              if os.path.isfile(os.path.join(cls_path, img))]

    # bỏ class nếu quá ít ảnh
    if len(images) < 2:
        print(f"⚠️ Skip class {cls} (not enough images)")
        continue

    random.shuffle(images)

    split_idx = int(len(images) * split_ratio)

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # tạo folder class
    train_cls_dir = os.path.join(train_dir, cls)
    val_cls_dir = os.path.join(val_dir, cls)

    os.makedirs(train_cls_dir, exist_ok=True)
    os.makedirs(val_cls_dir, exist_ok=True)

    # copy train
    for img in train_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(train_cls_dir, img)
        )

    # copy val
    for img in val_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(val_cls_dir, img)
        )

    print(f"✅ {cls}: {len(train_imgs)} train | {len(val_imgs)} val")

print("\n🎉 DONE SPLIT DATASET!")