import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rebalancing.difficulty import DifficultyTracker
from rebalancing.sampler import DifficultySampler

from utils.seed import set_seed

set_seed(42)
# ============================================================
# Trainer
# ============================================================

class RebalanceTrainer:

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        optimizer,
        device,
        num_classes,
        criterion=None,
        use_rebalancing=True,
        batch_size=128,
        inference_batch_size=256,
        num_workers=0,
        difficulty_c=1.0,
        sampler_power=1.0
    ):

        self.model = model.to(device)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.optimizer = optimizer

        self.device = device

        self.batch_size = batch_size
        self.inference_batch_size = (
            inference_batch_size
        )

        self.num_workers = num_workers

        # ----------------------------------------------------
        # loss
        # ----------------------------------------------------

        if criterion is None:

            self.criterion = nn.CrossEntropyLoss()

        else:

            self.criterion = criterion
        
        self.use_rebalancing = use_rebalancing

        # ----------------------------------------------------
        # difficulty tracker
        # ----------------------------------------------------

        self.difficulty_tracker = (
            DifficultyTracker(
                dataset_size=len(train_dataset),
                num_classes=num_classes,
                c=difficulty_c
            )
        )

        # ----------------------------------------------------
        # sampler
        # ----------------------------------------------------

        self.sampler_builder = (
            DifficultySampler(
                replacement=True,
                power=sampler_power
            )
        )

    # ========================================================
    # Build weighted train loader
    # ========================================================

    def build_train_loader(self):

        # --------------------------------------------------------
        # Standard ERM loader
        # --------------------------------------------------------

        if not self.use_rebalancing:

            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available()
            )

            return loader

        # --------------------------------------------------------
        # Rebalancing loader
        # --------------------------------------------------------

        difficulties = (
            self.difficulty_tracker
            .get_difficulties()
        )

        sampler = (
            self.sampler_builder
            .build_sampler(difficulties)
        )

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        return loader
    # ========================================================
    # Build inference loader
    # ========================================================

    def build_inference_loader(self):

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return loader

    # ========================================================
    # Validation loader
    # ========================================================

    def build_val_loader(self):

        loader = DataLoader(
            self.val_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return loader

    # ========================================================
    # Train one epoch
    # ========================================================

    def train_one_epoch(
        self,
        epoch
    ):

        self.model.train()

        train_loader = self.build_train_loader()

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch}"
        )

        for imgs, labels, indices in progress:

            imgs = imgs.to(
                self.device,
                non_blocking=True
            )

            labels = labels.to(
                self.device,
                non_blocking=True
            )

            # ----------------------------------------
            # forward
            # ----------------------------------------

            logits = self.model(imgs)

            loss = self.criterion(
                logits,
                labels
            )

            # ----------------------------------------
            # backward
            # ----------------------------------------

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            # ----------------------------------------
            # metrics
            # ----------------------------------------

            running_loss += (
                loss.item() * imgs.size(0)
            )

            preds = logits.argmax(dim=1)

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

            progress.set_postfix({
                "loss":
                    running_loss / total,
                "acc":
                    100.0 * correct / total
            })

        epoch_loss = running_loss / total

        epoch_acc = (
            100.0 * correct / total
        )

        return {
            "loss": epoch_loss,
            "acc": epoch_acc
        }

    # ========================================================
    # Full-trainset inference
    #
    # Required by paper:
    # update difficulties AFTER epoch
    # ========================================================

    @torch.no_grad()
    def update_difficulties(self):

        self.model.eval()

        inference_loader = (
            self.build_inference_loader()
        )

        progress = tqdm(
            inference_loader,
            desc="Updating Difficulties"
        )

        for imgs, labels, indices in progress:

            imgs = imgs.to(
                self.device,
                non_blocking=True
            )

            logits = self.model(imgs)

            probs = torch.softmax(
                logits,
                dim=1
            )

            self.difficulty_tracker.update(
                indices=indices,
                probs=probs,
                labels=labels
            )

    # ========================================================
    # Validation
    # ========================================================

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        val_loader = self.build_val_loader()

        running_loss = 0.0

        correct = 0
        total = 0

        for imgs, labels, _ in val_loader:

            imgs = imgs.to(
                self.device,
                non_blocking=True
            )

            labels = labels.to(
                self.device,
                non_blocking=True
            )

            logits = self.model(imgs)

            loss = self.criterion(
                logits,
                labels
            )

            running_loss += (
                loss.item() * imgs.size(0)
            )

            preds = logits.argmax(dim=1)

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

        val_loss = running_loss / total

        val_acc = (
            100.0 * correct / total
        )

        return {
            "loss": val_loss,
            "acc": val_acc
        }

    # ========================================================
    # Diagnostics
    # ========================================================

    def difficulty_statistics(self):

        return (
            self.difficulty_tracker
            .get_statistics()
        )

    # ========================================================
    # Main training loop
    # ========================================================

    def fit(
        self,
        epochs
    ):

        history = []

        for epoch in range(1, epochs + 1):

            # ----------------------------------------
            # 1. train epoch using weighted sampler
            # ----------------------------------------

            train_metrics = (
                self.train_one_epoch(epoch)
            )

            # ----------------------------------------
            # 2. infer full training set
            # 3. update difficulties
            # ----------------------------------------

            if self.use_rebalancing:
                self.update_difficulties()

            # ----------------------------------------
            # 4. validate
            # ----------------------------------------

            val_metrics = self.validate()

            # ----------------------------------------
            # difficulty diagnostics
            # ----------------------------------------

            diff_stats = (
                self.difficulty_statistics()
            )

            result = {

                "epoch": epoch,

                "train_loss":
                    train_metrics["loss"],

                "train_acc":
                    train_metrics["acc"],

                "val_loss":
                    val_metrics["loss"],

                "val_acc":
                    val_metrics["acc"],

                "difficulty_mean":
                    diff_stats["mean"],

                "difficulty_std":
                    diff_stats["std"],

                "difficulty_min":
                    diff_stats["min"],

                "difficulty_max":
                    diff_stats["max"]
            }

            history.append(result)

            # ----------------------------------------
            # logging
            # ----------------------------------------

            print(
                f"\nEpoch {epoch}"
            )

            print(
                f"Train Loss: "
                f"{result['train_loss']:.4f}"
            )

            print(
                f"Train Acc: "
                f"{result['train_acc']:.2f}%"
            )

            print(
                f"Val Loss: "
                f"{result['val_loss']:.4f}"
            )

            print(
                f"Val Acc: "
                f"{result['val_acc']:.2f}%"
            )

            print(
                f"Difficulty Mean: "
                f"{result['difficulty_mean']:.4f}"
            )

            print(
                f"Difficulty Std: "
                f"{result['difficulty_std']:.4f}"
            )

        return history