import torch

def accuracy(logits, labels):
    _, preds = torch.max(logits, 1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total