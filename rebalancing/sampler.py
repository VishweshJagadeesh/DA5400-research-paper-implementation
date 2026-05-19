import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


class DifficultySampler:
    """
    Converts difficulty values into sampling weights.

    Paper:

        w_i =
            D_i / sum(D)

    where:
        D_i = instance difficulty
    """

    def __init__(
        self,
        replacement=True,
        power=1.0,
        min_weight=1e-8
    ):

        self.replacement = replacement

        # optional sharpening
        #
        # power > 1:
        #   emphasizes hard samples
        #
        # power < 1:
        #   smooths distribution

        self.power = power

        self.min_weight = min_weight

    # ========================================================
    # Normalize difficulties
    # ========================================================

    def compute_weights(
        self,
        difficulties
    ):

        difficulties = np.asarray(
            difficulties,
            dtype=np.float64
        )

        difficulties = np.clip(
            difficulties,
            self.min_weight,
            None
        )

        # optional temperature sharpening

        difficulties = difficulties ** self.power

        weights = (
            difficulties /
            difficulties.sum()
        )

        return weights

    # ========================================================
    # Create sampler
    # ========================================================

    def build_sampler(
        self,
        difficulties,
        num_samples=None
    ):

        weights = self.compute_weights(
            difficulties
        )

        weights_tensor = torch.DoubleTensor(
            weights
        )

        if num_samples is None:
            num_samples = len(weights)

        sampler = WeightedRandomSampler(
            weights=weights_tensor,
            num_samples=num_samples,
            replacement=self.replacement
        )

        return sampler

    # ========================================================
    # Diagnostics
    # ========================================================

    def summarize_weights(
        self,
        difficulties
    ):

        weights = self.compute_weights(
            difficulties
        )

        return {
            "mean": float(np.mean(weights)),
            "std": float(np.std(weights)),
            "min": float(np.min(weights)),
            "max": float(np.max(weights)),
            "entropy": float(
                -np.sum(
                    weights *
                    np.log(weights + 1e-12)
                )
            )
        }