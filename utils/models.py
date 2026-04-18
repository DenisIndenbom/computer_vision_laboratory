import torch

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from .gradient import grad_reverse


class ResNetDANN(nn.Module):
    def __init__(
        self, num_classes: int, lambda_: float = 1.0, weights: ResNet50_Weights | None = None
    ):
        super().__init__()

        backbone = resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = backbone.avgpool
        num_features = backbone.fc.in_features

        self.classifier = nn.Linear(num_features, num_classes)
        self.domain_classifier = nn.Linear(num_features, 1)
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.avgpool(self.features(x)).flatten(1)
        class_out = self.classifier(feats)
        domain_out = self.domain_classifier(grad_reverse(feats, self.lambda_))

        return class_out, domain_out
