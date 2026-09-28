import torch
import torch.nn as nn
import timm


class Discriminator(nn.Module):
    """
    EfficientNet-B4 Discriminator
    Input:  (B, 3, 224, 224)
    Output: (B, 1) → confidence score [0=fake, 1=real]
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=pretrained, num_classes=0
        )
        in_features = self.backbone.num_features  # 1792
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class VideoDiscriminator(nn.Module):
    """
    Temporal Discriminator for video clips
    Input:  (B, N, 3, H, W)
    Output: (B, 1)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=pretrained, num_classes=0
        )
        in_features = self.backbone.num_features  # 1792

        self.temporal = nn.LSTM(
            input_size=in_features,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 1),
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        feats = self.backbone(x)           # (B*N, features)
        feats = feats.view(B, N, -1)       # (B, N, features)
        _, (hidden, _) = self.temporal(feats)
        return self.classifier(hidden.squeeze(0))  # (B, 1) 
