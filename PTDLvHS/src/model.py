import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EmbeddingModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # =====================
        # BACKBONE: EfficientNet-B3
        # =====================
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=True,
            num_classes=0
        )

        in_features = self.backbone.num_features

        # =====================
        # Freeze phần đầu, chỉ train block cuối
        # =====================
        for name, param in self.backbone.named_parameters():
            if "blocks.5" not in name and "blocks.6" not in name:
                param.requires_grad = False

        # =====================
        # Multi-head embedding
        # =====================
        def make_head():
            return nn.Sequential(
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 64)
            )

        self.head1 = make_head()
        self.head2 = make_head()
        self.head3 = make_head()

        self.classifier = nn.Linear(192, num_classes)

    def forward(self, x):
        feat = self.backbone(x)

        e1 = self.head1(feat)
        e2 = self.head2(feat)
        e3 = self.head3(feat)

        emb = torch.cat([e1, e2, e3], dim=1)
        emb = F.normalize(emb, p=2, dim=1)

        logits = self.classifier(emb)

        return emb, logits