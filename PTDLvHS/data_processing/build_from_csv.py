import pandas as pd
import os
import shutil

# ===== PATH =====
BASE_DIR = "/content/doanck/PTDLvHS/data/ePillID_data"

csv_path = os.path.join(BASE_DIR, "all_labels.csv")

# 👉 Đọc cả 2 folder chứa ảnh
img_dirs = [
    os.path.join(BASE_DIR, "classification_data/fcn_mix_weight/dc_224"),
    os.path.join(BASE_DIR, "classification_data/fcn_mix_weight/dr_224"),
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
    img_name = row["images"]              # ví dụ: 100.jpg
    label = str(row["label_code_id"])     # class

    # ===== TÌM ẢNH TRONG dc_224 + dr_224 =====
    src_path = None

    for d in img_dirs:
        temp_path = os.path.join(d, img_name)
        if os.path.exists(temp_path):
            src_path = temp_path
            break

    # ===== KHÔNG TÌM THẤY =====
    if src_path is None:
        missing += 1
        continue

    # ===== TẠO FOLDER CLASS =====
    cls_dir = os.path.join(out_dir, label)
    os.makedirs(cls_dir, exist_ok=True)

    # ===== COPY ẢNH =====
    dst_path = os.path.join(cls_dir, img_name)
    shutil.copy(src_path, dst_path)

    count += 1

# ===== RESULT =====
print("✅ Copied:", count, "images")
print("❌ Missing:", missing, "images")