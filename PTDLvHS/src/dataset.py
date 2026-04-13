import os
import random
from PIL import Image
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(self, root, transform=None, length=2000):
        self.root = root
        self.transform = transform
        self.length = length

        self.classes = sorted(os.listdir(root))
        self.class_to_images = {}

        for c in self.classes:
            folder = os.path.join(root, c)
            imgs = [img for img in os.listdir(folder)
                    if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if len(imgs) >= 2:
                self.class_to_images[c] = imgs

        self.classes = list(self.class_to_images.keys())
        print("Total valid classes:", len(self.classes))

        # 🔥 CACHE ẢNH (QUAN TRỌNG)
        self.cache = {}

    def __len__(self):
        return self.length

    def load_image(self, path):
        if path not in self.cache:
            try:
                img = Image.open(path).convert("RGB")
                self.cache[path] = img
            except:
                return None
        return self.cache[path]

    def __getitem__(self, idx):
        while True:
            # ===== Anchor class =====
            c = random.choice(self.classes)
            imgs = self.class_to_images[c]

            if len(imgs) < 2:
                continue

            a_name, p_name = random.sample(imgs, 2)

            # ===== Negative =====
            neg_c = random.choice(self.classes)
            while neg_c == c:
                neg_c = random.choice(self.classes)

            n_name = random.choice(self.class_to_images[neg_c])

            a_path = os.path.join(self.root, c, a_name)
            p_path = os.path.join(self.root, c, p_name)
            n_path = os.path.join(self.root, neg_c, n_name)

            a_img = self.load_image(a_path)
            p_img = self.load_image(p_path)
            n_img = self.load_image(n_path)

            if a_img is None or p_img is None or n_img is None:
                continue

            if self.transform:
                a_img = self.transform(a_img)
                p_img = self.transform(p_img)
                n_img = self.transform(n_img)

            label = self.classes.index(c)

            return a_img, p_img, n_img, label