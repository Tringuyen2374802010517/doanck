import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model('resnet50', pretrained=True)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.embedding = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x