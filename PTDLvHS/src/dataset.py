import os
import random
from PIL import Image
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(self, root, transform=None, length=1000):
        self.root = root
        self.transform = transform
        self.length = length

        # ===== load class =====
        self.classes = sorted(os.listdir(root))
        self.class_to_images = {}

        for c in self.classes:
            folder = os.path.join(root, c)

            # chỉ lấy file ảnh
            imgs = [img for img in os.listdir(folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # chỉ giữ class có >= 2 ảnh
            if len(imgs) >= 2:
                self.class_to_images[c] = imgs

        # cập nhật lại classes sau khi lọc
        self.classes = list(self.class_to_images.keys())

        print("Total valid classes:", len(self.classes))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # ===== anchor class =====
        c = random.choice(self.classes)
        imgs = self.class_to_images[c]

        # ===== anchor + positive =====
        a_name, p_name = random.sample(imgs, 2)

        # ===== negative class =====
        neg_c = random.choice(self.classes)
        while neg_c == c:
            neg_c = random.choice(self.classes)

        n_name = random.choice(self.class_to_images[neg_c])

        # ===== load ảnh =====
        a_path = os.path.join(self.root, c, a_name)
        p_path = os.path.join(self.root, c, p_name)
        n_path = os.path.join(self.root, neg_c, n_name)

        try:
            a_img = Image.open(a_path).convert("RGB")
            p_img = Image.open(p_path).convert("RGB")
            n_img = Image.open(n_path).convert("RGB")
        except:
            # fallback nếu ảnh lỗi
            return self.__getitem__(random.randint(0, self.length - 1))

        # ===== transform =====
        if self.transform:
            a_img = self.transform(a_img)
            p_img = self.transform(p_img)
            n_img = self.transform(n_img)

        label = self.classes.index(c)

        return a_img, p_img, n_img, label