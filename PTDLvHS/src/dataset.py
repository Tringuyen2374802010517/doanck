import os
import random
from PIL import Image
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.class_to_images = {}
        self.samples = []

        for cls in os.listdir(root):
            path = os.path.join(root, cls)
            images = os.listdir(path)

            full_paths = [os.path.join(path, img) for img in images]

            if len(full_paths) >= 2:
                self.class_to_images[cls] = full_paths

                for img in full_paths:
                    self.samples.append((cls, img))

        self.classes = list(self.class_to_images.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_class, anchor_path = self.samples[idx]

        # positive
        positive_path = anchor_path
        while positive_path == anchor_path:
            positive_path = random.choice(self.class_to_images[anchor_class])

        # negative
        negative_class = random.choice(self.classes)
        while negative_class == anchor_class:
            negative_class = random.choice(self.classes)

        negative_path = random.choice(self.class_to_images[negative_class])

        a = Image.open(anchor_path).convert("RGB")
        p = Image.open(positive_path).convert("RGB")
        n = Image.open(negative_path).convert("RGB")

        if self.transform:
            a = self.transform(a)
            p = self.transform(p)
            n = self.transform(n)

        return a, p, n