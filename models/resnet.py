import torch.nn as nn
import torchvision.models as models
from utils.seed import set_seed

set_seed(42)

class ResNet32Classifier(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.backbone = models.resnet18(
            weights=None
        )

        # CIFAR modification
        self.backbone.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.backbone.maxpool = nn.Identity()

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):

        return self.backbone(x)