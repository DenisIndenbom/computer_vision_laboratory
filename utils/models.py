from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16

from .gradient import grad_reverse


class ResNetDANN(nn.Module):
    def __init__(
        self, num_classes: int, lambda_: float = 1.0, weights: ResNet50_Weights | None = None
    ):
        super().__init__()

        backbone = resnet50(weights=weights)
        self.features = nn.Sequential(OrderedDict(list(backbone.named_children())[:-2]))
        self.avgpool = backbone.avgpool
        num_features = backbone.fc.in_features

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.avgpool(self.features(x)).flatten(1)
        class_out = self.classifier(feats)
        domain_out = self.domain_classifier(grad_reverse(feats, self.lambda_))

        return class_out, domain_out


class ViTB16DANN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        lambda_: float = 1.0,
        weights: ViT_B_16_Weights | None = None,
    ):
        super().__init__()

        backbone = vit_b_16(weights=weights)
        self.features = backbone
        self.features.heads = nn.Sequential(nn.Identity())
        num_features = backbone.hidden_dim

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x)

        class_out = self.classifier(feats)
        domain_out = self.domain_classifier(grad_reverse(feats, self.lambda_))

        return class_out, domain_out
