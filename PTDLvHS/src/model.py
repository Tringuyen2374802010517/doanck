import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 128)

    def forward(self, x):
        return F.normalize(self.backbone(x))