import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Backbone
        self.backbone = timm.create_model("resnet50", pretrained=True)

        # Freeze phần đầu
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False

        in_features = self.backbone.fc.in_features

        # bỏ fc cũ
        self.backbone.fc = nn.Identity()

        # ===== Head 1: shape =====
        self.head1 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64)
        )

        # ===== Head 2: texture =====
        self.head2 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64)
        )

        # ===== Head 3: imprint =====
        self.head3 = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        feat = self.backbone(x)

        e1 = self.head1(feat)
        e2 = self.head2(feat)
        e3 = self.head3(feat)

        # concat 3 head
        emb = torch.cat([e1, e2, e3], dim=1)

        # normalize
        emb = F.normalize(emb, p=2, dim=1)

        return emb