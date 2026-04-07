import pandas as pd
import os
import shutil

# ===== PATH =====
BASE_DIR = "/content/doanck/PTDLvHS/data/ePillID_data"

csv_path = os.path.join(BASE_DIR, "all_labels.csv")
img_dir = os.path.join(BASE_DIR, "classification_data/segmented_nih_pills_224")
out_dir = "/content/doanck/PTDLvHS/data/processed"

os.makedirs(out_dir, exist_ok=True)

# ===== LOAD CSV =====
df = pd.read_csv(csv_path)

print("Total rows:", len(df))

# ===== BUILD DATASET =====
count = 0

for _, row in df.iterrows():
    img_name = row["images"]              # ví dụ: 100.jpg
    label = str(row["label_code_id"])     # class

    src_path = os.path.join(img_dir, img_name)

    if not os.path.exists(src_path):
        continue

    cls_dir = os.path.join(out_dir, label)
    os.makedirs(cls_dir, exist_ok=True)

    dst_path = os.path.join(cls_dir, img_name)

    shutil.copy(src_path, dst_path)
    count += 1

print("✅ Copied:", count, "images")