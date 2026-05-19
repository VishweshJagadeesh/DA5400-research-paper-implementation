# ============================================================
# metrics.py
# ============================================================

import numpy as np
import torch


# ============================================================
# Overall accuracy
# ============================================================

def accuracy(
    logits,
    labels
):

    preds = logits.argmax(dim=1)

    correct = (
        preds == labels
    ).sum().item()

    total = labels.size(0)

    return 100.0 * correct / total


# ============================================================
# Per-class accuracy
# ============================================================

def per_class_accuracy(
    preds,
    labels,
    num_classes
):

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    class_acc = {}

    for c in range(num_classes):

        mask = labels == c

        total = mask.sum()

        if total == 0:

            class_acc[c] = 0.0

            continue

        correct = (
            preds[mask] == labels[mask]
        ).sum()

        class_acc[c] = (
            100.0 * correct / total
        )

    return class_acc


# ============================================================
# Group accuracy
#
# Example:
# majority vs minority
# ============================================================

def group_accuracy(
    preds,
    labels,
    groups
):
    """
    groups:
        {
            "majority": [0,1,2],
            "minority": [3,4]
        }
    """

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    results = {}

    for group_name, class_ids in groups.items():

        mask = np.isin(
            labels,
            class_ids
        )

        total = mask.sum()

        if total == 0:

            results[group_name] = 0.0

            continue

        correct = (
            preds[mask] == labels[mask]
        ).sum()

        results[group_name] = (
            100.0 * correct / total
        )

    return results


# ============================================================
# Confusion matrix
# ============================================================

def confusion_matrix(
    preds,
    labels,
    num_classes
):

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    matrix = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64
    )

    for p, y in zip(preds, labels):

        matrix[y, p] += 1

    return matrix


# ============================================================
# Long-tail split
#
# Automatically split classes into:
# head / medium / tail
# ============================================================

def split_long_tail_classes(
    class_counts,
    head_ratio=0.3,
    tail_ratio=0.3
):
    """
    class_counts:
        dict[class_id] -> count
    """

    sorted_items = sorted(
        class_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    classes = [
        c for c, _ in sorted_items
    ]

    n = len(classes)

    head_end = int(n * head_ratio)
    tail_start = int(n * (1 - tail_ratio))

    head = classes[:head_end]

    medium = classes[
        head_end:tail_start
    ]

    tail = classes[tail_start:]

    return {
        "head": head,
        "medium": medium,
        "tail": tail
    }


# ============================================================
# Difficulty statistics
# ============================================================

def difficulty_statistics(
    difficulties
):

    difficulties = np.asarray(
        difficulties
    )

    return {

        "mean":
            float(np.mean(difficulties)),

        "std":
            float(np.std(difficulties)),

        "min":
            float(np.min(difficulties)),

        "max":
            float(np.max(difficulties)),

        "median":
            float(np.median(difficulties))
    }


# ============================================================
# Entropy of sampling distribution
# ============================================================

def sampling_entropy(
    weights
):

    weights = np.asarray(weights)

    weights = weights / weights.sum()

    eps = 1e-12

    entropy = -np.sum(
        weights * np.log(weights + eps)
    )

    return float(entropy)