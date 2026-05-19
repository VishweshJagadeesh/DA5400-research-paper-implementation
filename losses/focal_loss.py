import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(
        self,
        gamma=2.0,
        alpha=None,
        reduction="mean"
    ):

        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits,
        targets
    ):

        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)

        focal_loss = (
            (1 - pt) ** self.gamma
        ) * ce_loss

        # optional class weighting

        if self.alpha is not None:

            alpha_t = self.alpha[
                targets
            ]

            focal_loss = (
                alpha_t * focal_loss
            )

        if self.reduction == "mean":
            return focal_loss.mean()

        elif self.reduction == "sum":
            return focal_loss.sum()

        return focal_loss