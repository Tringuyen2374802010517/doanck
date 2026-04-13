import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ===== ArcFace =====
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, x, labels):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        # tránh NaN
        cosine = torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7)

        theta = torch.acos(cosine)
        target = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        output = cosine * (1 - one_hot) + target * one_hot
        return output * self.s


# ===== Model =====
class EmbeddingModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # ===== Backbone =====
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=True,
            num_classes=0
        )

        in_features = self.backbone.num_features

        # 🔥 FIX QUAN TRỌNG: KHÔNG freeze hết
        # chỉ freeze layer rất nông để giữ feature cơ bản
        for name, param in self.backbone.named_parameters():
            if "blocks.0" in name or "blocks.1" in name:
                param.requires_grad = False

        # ===== Embedding Head =====
        self.embedding = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128)
        )

        # ===== ArcFace =====
        self.arcface = ArcFace(128, num_classes)

    def forward(self, x, labels=None):
        # ===== Extract feature =====
        feat = self.backbone(x)

        # ===== Embedding =====
        emb = self.embedding(feat)

        # normalize vector (quan trọng cho metric learning)
        emb = F.normalize(emb, p=2, dim=1)

        # ===== Train mode =====
        if labels is not None:
            logits = self.arcface(emb, labels)
            return emb, logits

        # ===== Inference mode =====
        return emb