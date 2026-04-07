import os
import random
from PIL import Image
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.class_to_images = {}

        for cls in os.listdir(root):
            path = os.path.join(root, cls)
            images = os.listdir(path)
            self.class_to_images[cls] = [
                os.path.join(path, img) for img in images
            ]

        self.classes = list(self.class_to_images.keys())

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        while True:
            c = random.choice(self.classes)
            if len(self.class_to_images[c]) >= 2:
                break

        neg_c = random.choice(self.classes)
        while neg_c == c:
            neg_c = random.choice(self.classes)

        a, p = random.sample(self.class_to_images[c], 2)
        n = random.choice(self.class_to_images[neg_c])

        a = Image.open(a).convert("RGB")
        p = Image.open(p).convert("RGB")
        n = Image.open(n).convert("RGB")

        if self.transform:
            a = self.transform(a)
            p = self.transform(p)
            n = self.transform(n)

        return a, p, n