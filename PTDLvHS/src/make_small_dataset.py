import os
import random
import shutil

# ==========================
# ĐƯỜNG DẪN GỐC DATASET
# ==========================
src_root = "/content/ePillID_data"   # thư mục gốc sau unzip
dst_root = "/content/small_epillid"

# số folder muốn lấy
num_classes = 10

# số ảnh mỗi folder
images_per_class = 50

# random cố định để ra giống nhau
random.seed(42)

# ==========================
# XÓA THƯ MỤC CŨ NẾU CÓ
# ==========================
if os.path.exists(dst_root):
    shutil.rmtree(dst_root)

os.makedirs(dst_root)

# ==========================
# LẤY DANH SÁCH CLASS
# ==========================
all_classes = [
    cls for cls in os.listdir(src_root)
    if os.path.isdir(os.path.join(src_root, cls))
]

all_classes.sort()

# chọn 10 class đầu
selected_classes = all_classes[:num_classes]

print("Selected classes:", selected_classes)

# ==========================
# COPY 50 ẢNH / CLASS
# ==========================
for cls in selected_classes:
    src_class = os.path.join(src_root, cls)
    dst_class = os.path.join(dst_root, cls)

    os.makedirs(dst_class)

    images = [
        img for img in os.listdir(src_class)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(images) < images_per_class:
        print(f"{cls} không đủ ảnh")
        continue

    selected_images = random.sample(images, images_per_class)

    for img in selected_images:
        shutil.copy(
            os.path.join(src_class, img),
            os.path.join(dst_class, img)
        )

    print(f"Copied {cls}")
    
print("Done!")