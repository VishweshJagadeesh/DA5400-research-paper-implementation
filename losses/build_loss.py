import torch.nn as nn

from losses.focal_loss import FocalLoss
from losses.class_balanced_loss import (
    ClassBalancedLoss
)


# ============================================================
# Loss Factory
# ============================================================


def build_loss(
    loss_name,
    samples_per_class=None,
    gamma=2.0,
    beta=0.9999
):

    # --------------------------------------------------------
    # Cross Entropy
    # --------------------------------------------------------

    if loss_name == "ce":

        return nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Focal Loss
    # --------------------------------------------------------

    elif loss_name == "focal":

        return FocalLoss(
            gamma=gamma
        )

    # --------------------------------------------------------
    # Class Balanced Loss
    # --------------------------------------------------------

    elif loss_name == "cb":

        if samples_per_class is None:

            raise ValueError(
                "samples_per_class must be provided "
                "for ClassBalancedLoss"
            )

        # --------------------------------------------
        # handle dict/list formats
        # --------------------------------------------

        if isinstance(samples_per_class, dict):

            samples_per_class = list(
                samples_per_class.values()
            )

        return ClassBalancedLoss(
            samples_per_class=samples_per_class,
            beta=beta
        )

    # --------------------------------------------------------
    # Unknown
    # --------------------------------------------------------

    else:

        raise ValueError(
            f"Unknown loss: {loss_name}"
        )