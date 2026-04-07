import pandas as pd
import os
import shutil

# ===== PATH =====
BASE_DIR = "/content/doanck/PTDLvHS/data/ePillID_data"

csv_path = os.path.join(BASE_DIR, "all_labels.csv")

# 👉 GIỮ dòng này (theo ý bạn)
img_dir = os.path.join(BASE_DIR, "classification_data/segmented_nih_pills_224")

# 👉 dùng thật (fcn)
img_dir_fcn = os.path.join(BASE_DIR, "classification_data/fcn_mix_weight")

# 👉 folder chứa ảnh thật
img_dirs = [
    os.path.join(img_dir_fcn, "dc_224"),
    os.path.join(img_dir_fcn, "dr_224"),
]

out_dir = "/content/doanck/PTDLvHS/data/processed"

os.makedirs(out_dir, exist_ok=True)

# ===== LOAD CSV =====
df = pd.read_csv(csv_path)

print("Total rows:", len(df))

# ===== BUILD DATASET =====
count = 0
missing = 0

for _, row in df.iterrows():
    img_name = row["images"]
    label = str(row["label_code_id"])

    src_path = None

    # 👉 tìm trong dc_224 + dr_224
    for d in img_dirs:
        temp_path = os.path.join(d, img_name)
        if os.path.exists(temp_path):
            src_path = temp_path
            break

    if src_path is None:
        missing += 1
        continue

    cls_dir = os.path.join(out_dir, label)
    os.makedirs(cls_dir, exist_ok=True)

    shutil.copy(src_path, os.path.join(cls_dir, img_name))
    count += 1

print("✅ Copied:", count, "images")
print("❌ Missing:", missing, "images")