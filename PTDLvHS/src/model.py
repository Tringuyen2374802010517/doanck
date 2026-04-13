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
        cosine = torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7)

        theta = torch.acos(cosine)
        target = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        output = cosine * (1 - one_hot) + target * one_hot
        return output * self.s


# ===== MODEL =====
class EmbeddingModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 🔥 Backbone mạnh hơn
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=True,
            num_classes=0
        )

        in_features = self.backbone.num_features

        # ===== Embedding Head =====
        self.embedding = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128)
        )

        self.arcface = ArcFace(128, num_classes)

    def forward(self, x, labels=None):
        feat = self.backbone(x)
        emb = self.embedding(feat)
        emb = F.normalize(emb, p=2, dim=1)

        if labels is not None:
            logits = self.arcface(emb, labels)
            return emb, logits

        return emb