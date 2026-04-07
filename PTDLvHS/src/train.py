import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import EmbeddingModel
from dataset import TripletDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])

dataset = TripletDataset("data", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = EmbeddingModel().to(device)

criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 40
best_loss = float("inf")

for epoch in range(epochs):
    total_loss = 0

    for a, p, n in loader:
        a, p, n = a.to(device), p.to(device), n.to(device)

        emb_a = model(a)
        emb_p = model(p)
        emb_n = model(n)

        loss = criterion(emb_a, emb_p, emb_n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg = total_loss / len(loader)
    print(f"Epoch {epoch+1}: {avg:.4f}")

    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), "best_model.pth")