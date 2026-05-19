import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedLoss(nn.Module):

    def __init__(
        self,
        samples_per_class,
        beta=0.9999,
        reduction="mean"
    ):

        super().__init__()

        effective_num = (
            1.0 -
            np.power(
                beta,
                samples_per_class
            )
        )

        weights = (
            (1.0 - beta) /
            effective_num
        )

        weights = (
            weights /
            np.sum(weights)
        ) * len(samples_per_class)

        self.weights = torch.tensor(
            weights,
            dtype=torch.float32
        )

        self.reduction = reduction

    def forward(
        self,
        logits,
        targets
    ):

        weights = self.weights.to(
            logits.device
        )

        return F.cross_entropy(
            logits,
            targets,
            weight=weights,
            reduction=self.reduction
        )