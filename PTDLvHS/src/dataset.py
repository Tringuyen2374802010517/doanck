# dataset.py

import os
import random
from PIL import Image
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, root, transform=None, length=10000):
        self.root = root
        self.transform = transform
        self.length = length

        self.class_to_images = {}
        self.classes = []

        folders = sorted(os.listdir(root))

        for idx, cls in enumerate(folders):
            path = os.path.join(root, cls)
            if not os.path.isdir(path):
                continue

            images = [
                os.path.join(path, img)
                for img in os.listdir(path)
                if img.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            if len(images) > 0:
                self.class_to_images[idx] = images
                self.classes.append(idx)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            c = random.choice(self.classes)
            if len(self.class_to_images[c]) >= 2:
                break

        neg_c = random.choice(self.classes)
        while neg_c == c:
            neg_c = random.choice(self.classes)

        a_path, p_path = random.sample(self.class_to_images[c], 2)
        n_path = random.choice(self.class_to_images[neg_c])

        a = Image.open(a_path).convert("RGB")
        p = Image.open(p_path).convert("RGB")
        n = Image.open(n_path).convert("RGB")

        if self.transform:
            a = self.transform(a)
            p = self.transform(p)
            n = self.transform(n)

        return a, p, n, c