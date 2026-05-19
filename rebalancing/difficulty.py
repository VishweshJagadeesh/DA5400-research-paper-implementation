import numpy as np
import torch


class DifficultyTracker:
    """
    Tracks instance difficulty across epochs.

    Implements Eq. (4), Eq. (6), Eq. (7)
    from the paper.

    Difficulty:

        D_i =
            (c + sum(du_i))
            ----------------
            (c + sum(dl_i))

    where:
        du -> unlearning trend accumulation
        dl -> learning trend accumulation
    """

    def __init__(
        self,
        dataset_size: int,
        num_classes: int,
        c: float = 1.0,
        eps: float = 1e-8
    ):

        self.dataset_size = dataset_size
        self.num_classes = num_classes

        self.c = c
        self.eps = eps

        # ----------------------------------------
        # cumulative unlearning / learning
        # ----------------------------------------

        self.du_sum = np.zeros(
            dataset_size,
            dtype=np.float64
        )

        self.dl_sum = np.zeros(
            dataset_size,
            dtype=np.float64
        )

        # ----------------------------------------
        # previous probability distributions
        #
        # initialized as uniform distribution
        # following paper
        # ----------------------------------------

        self.prev_probs = np.full(
            (dataset_size, num_classes),
            1.0 / num_classes,
            dtype=np.float64
        )

    # ========================================================
    # Eq. 6 + Eq. 7
    # ========================================================

    def compute_du_dl(
        self,
        prev_probs,
        curr_probs,
        label
    ):

        prev_probs = np.clip(
            prev_probs,
            self.eps,
            1.0
        )

        curr_probs = np.clip(
            curr_probs,
            self.eps,
            1.0
        )

        du = 0.0
        dl = 0.0

        # ----------------------------------------------------
        # TRUE CLASS TERM
        # ----------------------------------------------------

        delta = (
            curr_probs[label]
            - prev_probs[label]
        )

        ratio = np.log(
            curr_probs[label]
            / prev_probs[label]
        )

        contribution = delta * ratio

        if delta < 0:
            du += contribution
        else:
            dl += contribution

        # ----------------------------------------------------
        # NON-TRUE CLASS TERMS
        # ----------------------------------------------------

        for j in range(self.num_classes):

            if j == label:
                continue

            delta = (
                curr_probs[j]
                - prev_probs[j]
            )

            ratio = np.log(
                curr_probs[j]
                / prev_probs[j]
            )

            contribution = delta * ratio

            # opposite logic
            # increasing wrong-class probability
            # => unlearning

            if delta > 0:
                du += contribution
            else:
                dl += contribution

        return du, dl

    # ========================================================
    # Update statistics after epoch inference
    # ========================================================

    @torch.no_grad()
    def update(
        self,
        indices,
        probs,
        labels
    ):
        """
        Parameters
        ----------
        indices : tensor/list
            sample indices

        probs : tensor [B, C]
            current predicted probabilities

        labels : tensor [B]
            ground-truth labels
        """

        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()

        for idx, curr_prob, label in zip(
            indices,
            probs,
            labels
        ):

            prev_prob = self.prev_probs[idx]

            du, dl = self.compute_du_dl(
                prev_prob,
                curr_prob,
                label
            )

            self.du_sum[idx] += du
            self.dl_sum[idx] += dl

            self.prev_probs[idx] = curr_prob

    # ========================================================
    # Difficulty computation
    # ========================================================

    def get_difficulties(self):

        difficulties = (
            self.c + self.du_sum
        ) / (
            self.c + self.dl_sum
        )

        difficulties = np.nan_to_num(
            difficulties,
            nan=1.0,
            posinf=1.0,
            neginf=1.0
        )

        return difficulties

    # ========================================================
    # Diagnostics
    # ========================================================

    def get_statistics(self):

        difficulties = self.get_difficulties()

        return {
            "mean": float(np.mean(difficulties)),
            "std": float(np.std(difficulties)),
            "min": float(np.min(difficulties)),
            "max": float(np.max(difficulties)),
        }

    # ========================================================
    # Reset
    # ========================================================

    def reset(self):

        self.du_sum.fill(0.0)
        self.dl_sum.fill(0.0)

        self.prev_probs.fill(
            1.0 / self.num_classes
        )