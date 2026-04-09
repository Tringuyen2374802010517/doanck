import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EmbeddingModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = timm.create_model("resnet50", pretrained=True)

        # Freeze all except layer4
        for name, param in self.backbone.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        def make_head():
            return nn.Sequential(
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
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