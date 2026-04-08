import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Backbone
        self.backbone = timm.create_model('resnet50', pretrained=True)

        # Freeze 70% layer đầu (giảm overfit)
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False

        in_features = self.backbone.fc.in_features

        # Head xịn hơn
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x, p=2, dim=1)  # cực quan trọng
        return x