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
        theta = torch.acos(torch.clamp(cosine, -1+1e-7, 1-1e-7))
        target = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1,1), 1)

        output = cosine*(1-one_hot) + target*one_hot
        return output * self.s


# ===== Model =====
class EmbeddingModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=True,
            num_classes=0
        )

        in_features = self.backbone.num_features

        # 🔥 FREEZE BACKBONE (GIẢM OVERFIT)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ===== Embedding head =====
        self.embedding = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128)
        )

        # ===== ArcFace =====
        self.arcface = ArcFace(128, num_classes)

    def forward(self, x, labels=None):
        feat = self.backbone(x)

        emb = self.embedding(feat)
        emb = F.normalize(emb, p=2, dim=1)

        if labels is not None:
            logits = self.arcface(emb, labels)
            return emb, logits

        return emb