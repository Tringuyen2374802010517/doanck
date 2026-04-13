import os
import random
from PIL import Image
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(self, root, transform=None, length=5000):
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

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # ===== Anchor class =====
        c = random.choice(self.classes)
        imgs = self.class_to_images[c]

        a_name, p_name = random.sample(imgs, 2)

        # 🔥 semi-hard negative (gần hơn random)
        neg_c = random.choice(self.classes)
        while neg_c == c:
            neg_c = random.choice(self.classes)

        n_name = random.choice(self.class_to_images[neg_c])

        a_path = os.path.join(self.root, c, a_name)
        p_path = os.path.join(self.root, c, p_name)
        n_path = os.path.join(self.root, neg_c, n_name)

        a_img = Image.open(a_path).convert("RGB")
        p_img = Image.open(p_path).convert("RGB")
        n_img = Image.open(n_path).convert("RGB")

        if self.transform:
            a_img = self.transform(a_img)
            p_img = self.transform(p_img)
            n_img = self.transform(n_img)

        label = self.classes.index(c)

        return a_img, p_img, n_img, label