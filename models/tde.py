import torch
import torch.nn as nn


# ============================================================
# TDE Classifier
# ============================================================

class TDEClassifier(nn.Module):

    def __init__(
        self,
        feat_dim,
        num_classes,
        alpha=1.0,
        momentum=0.9
    ):

        super().__init__()

        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.alpha = alpha
        self.momentum = momentum

        # ----------------------------------------------------
        # classifier
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            feat_dim,
            num_classes
        )

        # ----------------------------------------------------
        # running mean buffer
        # ----------------------------------------------------

        self.register_buffer(
            "running_mean",
            torch.zeros(feat_dim)
        )

        self.register_buffer(
            "initialized",
            torch.tensor(False)
        )

    # ========================================================
    # Update running mean
    # ========================================================

    @torch.no_grad()
    def update_running_mean(
        self,
        features
    ):

        batch_mean = features.mean(dim=0)

        # ----------------------------------------------------
        # initialize first batch
        # ----------------------------------------------------

        if not self.initialized:

            self.running_mean.copy_(
                batch_mean
            )

            self.initialized.fill_(True)

            return

        # ----------------------------------------------------
        # EMA update
        # ----------------------------------------------------

        self.running_mean.mul_(
            self.momentum
        ).add_(
            batch_mean * (1.0 - self.momentum)
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        features,
        training=True
    ):

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        if training:

            self.update_running_mean(
                features.detach()
            )

            logits = self.classifier(
                features
            )

            return logits

        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        corrected_features = (
            features
            -
            self.alpha *
            self.running_mean.unsqueeze(0)
        )

        logits = self.classifier(
            corrected_features
        )

        return logits

    # ========================================================
    # Diagnostics
    # ========================================================

    @torch.no_grad()
    def get_running_mean_norm(self):

        return torch.norm(
            self.running_mean
        ).item()


# ============================================================
# Generic TDE Wrapper
# ============================================================

class TDEModel(nn.Module):

    def __init__(
        self,
        backbone,
        feat_dim,
        num_classes,
        alpha=1.0,
        momentum=0.9
    ):

        super().__init__()

        self.backbone = backbone

        self.tde_classifier = TDEClassifier(
            feat_dim=feat_dim,
            num_classes=num_classes,
            alpha=alpha,
            momentum=momentum
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x
    ):

        features = self.backbone(x)

        logits = self.tde_classifier(
            features,
            training=self.training
        )

        return logits

    # ========================================================
    # Feature extraction
    # ========================================================

    @torch.no_grad()
    def extract_features(
        self,
        x
    ):

        return self.backbone(x)


# ============================================================
# Example MLP Feature Extractor
# ============================================================

class MLPFeatureExtractor(nn.Module):

    def __init__(
        self,
        input_dim=784,
        hidden_dim=256,
        feat_dim=128
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                feat_dim
            )
        )

    def forward(
        self,
        x
    ):

        x = x.view(
            x.size(0),
            -1
        )

        return self.net(x)
    
    # ============================================================
# ResNet Feature Extractor
# ============================================================

import torchvision.models as models


class ResNetFeatureExtractor(nn.Module):

    def __init__(
        self,
        feat_dim=128
    ):

        super().__init__()

        # ----------------------------------------------------
        # Base ResNet18
        #
        # Paper uses ResNet32, but ResNet18 is acceptable
        # lightweight substitute for reproduction.
        # ----------------------------------------------------

        backbone = models.resnet18(
            weights=None
        )

        # ----------------------------------------------------
        # CIFAR modifications
        # ----------------------------------------------------

        backbone.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        backbone.maxpool = nn.Identity()

        # ----------------------------------------------------
        # Remove classifier
        # ----------------------------------------------------

        in_features = backbone.fc.in_features

        backbone.fc = nn.Identity()

        self.backbone = backbone

        # ----------------------------------------------------
        # Projection head
        # ----------------------------------------------------

        self.projector = nn.Sequential(

            nn.Linear(
                in_features,
                feat_dim
            ),

            nn.ReLU(),

            nn.Linear(
                feat_dim,
                feat_dim
            )
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x
    ):

        features = self.backbone(x)

        features = self.projector(
            features
        )

        return features