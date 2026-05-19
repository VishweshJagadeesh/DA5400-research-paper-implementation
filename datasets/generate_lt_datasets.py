import warnings
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision import transforms

from utils.seed import set_seed

set_seed(42)
# ============================================================
# Utilities
# ============================================================

def get_img_num_per_cls(
    cls_num: int,
    imb_factor: float,
    img_max: int
):
    """
    Generate exponentially decayed class counts.

    Parameters
    ----------
    cls_num : int
        Number of classes

    imb_factor : float
        max_class_count / min_class_count

    img_max : int
        Largest class size
    """

    img_num_per_cls = []

    for cls_idx in range(cls_num):

        num = img_max * (
            (1 / imb_factor) ** (
                cls_idx / (cls_num - 1)
            )
        )

        img_num_per_cls.append(int(num))

    return img_num_per_cls


def solve_img_max_for_target_total(
    cls_num: int,
    imb_factor: float,
    target_total: int
):
    """
    Solve img_max such that:

        sum(get_img_num_per_cls(...)) ~= target_total

    Using binary search because integer rounding
    prevents a clean closed-form inversion.
    """

    low = 1
    high = target_total * 10

    best_img_max = low

    while low <= high:

        mid = (low + high) // 2

        counts = get_img_num_per_cls(
            cls_num=cls_num,
            imb_factor=imb_factor,
            img_max=mid
        )

        total = sum(counts)

        if total <= target_total:
            best_img_max = mid
            low = mid + 1
        else:
            high = mid - 1

    return best_img_max


# ============================================================
# Long-Tailed CIFAR
# ============================================================

class ImbalancedCIFAR10(CIFAR10):

    cls_num = 10

    def __init__(
        self,
        root,
        imb_factor=100,
        train=True,
        transform=None,
        target_transform=None,
        download=True,
        seed=42
    ):

        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download
        )

        np.random.seed(seed)

        self.imb_factor = imb_factor

        if train:
            self.gen_imbalanced_data()

    def gen_imbalanced_data(self):

        img_max = len(self.data) // self.cls_num

        img_num_per_cls = get_img_num_per_cls(
            cls_num=self.cls_num,
            imb_factor=self.imb_factor,
            img_max=img_max
        )

        new_data = []
        new_targets = []

        targets_np = np.array(self.targets)

        self.num_per_cls_dict = {}

        for cls_idx, img_num in enumerate(img_num_per_cls):

            self.num_per_cls_dict[cls_idx] = img_num

            idx = np.where(targets_np == cls_idx)[0]

            np.random.shuffle(idx)

            if img_num > len(idx):

                warnings.warn(
                    f"Class {cls_idx}: requested "
                    f"{img_num} samples but only "
                    f"{len(idx)} available. Capping."
                )

                img_num = len(idx)

            selected_idx = idx[:img_num]

            new_data.append(self.data[selected_idx])

            new_targets.extend([cls_idx] * img_num)

        self.data = np.vstack(new_data)
        self.targets = new_targets

    def get_cls_num_list(self):

        return [
            self.num_per_cls_dict[i]
            for i in range(self.cls_num)
        ]

    def get_imbalance_ratio(self):

        cls_counts = self.get_cls_num_list()

        return max(cls_counts) / min(cls_counts)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        return img, target, idx
    def __repr__(self):

        return (
            f"{self.__class__.__name__}("
            f"classes={self.cls_num}, "
            f"imb_factor={self.imb_factor}, "
            f"samples={len(self.data)})"
        )


class ImbalancedCIFAR100(ImbalancedCIFAR10, CIFAR100):
    """
    Long-tailed CIFAR-100.

    Inherits imbalanced sampling logic from ImbalancedCIFAR10 and
    all dataset attributes (URLs, checksums, meta, file lists) from
    CIFAR100 via Python's MRO. This avoids hardcoding torchvision
    internals that can change across library versions.

    MRO: ImbalancedCIFAR100 -> ImbalancedCIFAR10 -> CIFAR100 -> CIFAR10
    Method resolution order ensures:
      - __init__ and gen_imbalanced_data come from ImbalancedCIFAR10
      - base_folder, url, train_list, test_list, meta come from CIFAR100
    """

    cls_num = 100


# ============================================================
# Long-Tailed Binary MNIST
# ============================================================

class LongTailedBinaryMNIST(Dataset):
    """
    Binary classification:

        even -> 0
        odd  -> 1

    Hidden sub-classes:

        Even digits: 0,2,4,6,8
        Odd digits : 1,3,5,7,9

    The dataset explicitly creates:

    1. Class-level imbalance
    2. Sub-class imbalance within each class
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        download=True,
        class_imbalance_factor=100,
        subclass_imbalance_factor=10,
        seed=42
    ):

        self.mnist = MNIST(
            root=root,
            train=train,
            transform=None,
            download=download
        )

        self.transform = transform

        np.random.seed(seed)

        self.class_imbalance_factor = (
            class_imbalance_factor
        )

        self.subclass_imbalance_factor = (
            subclass_imbalance_factor
        )

        if train:
            self.create_long_tail()
        else:

            # --------------------------------------------
            # Test set intentionally remains:
            #
            # 1. Unmodified
            # 2. Unshuffled
            #
            # to preserve standard evaluation protocol.
            # --------------------------------------------

            self.images = self.mnist.data.numpy()

            self.digit_labels = (
                self.mnist.targets.numpy()
            )

            self.targets = np.array([
                0 if d % 2 == 0 else 1
                for d in self.digit_labels
            ])

    def create_long_tail(self):

        data = self.mnist.data.numpy()
        targets = self.mnist.targets.numpy()

        even_digits = [0, 2, 4, 6, 8]
        odd_digits = [1, 3, 5, 7, 9]

        new_images = []
        new_targets = []
        new_digit_labels = []

        # ----------------------------------------------------
        # Derive valid maximum count dynamically
        # ----------------------------------------------------

        digit_counts = {
            d: int((targets == d).sum())
            for d in range(10)
        }

        even_img_max = min(
            digit_counts[d]
            for d in even_digits
        )

        # ----------------------------------------------------
        # Construct even subclass distribution
        # ----------------------------------------------------

        even_counts = get_img_num_per_cls(
            cls_num=len(even_digits),
            imb_factor=self.subclass_imbalance_factor,
            img_max=even_img_max
        )

        target_even_total = sum(even_counts)

        # ----------------------------------------------------
        # Desired odd total:
        #
        # even_total / odd_total
        # ~= class_imbalance_factor
        # ----------------------------------------------------

        target_odd_total = max(
            1,
            int(
                target_even_total /
                self.class_imbalance_factor
            )
        )

        # ----------------------------------------------------
        # Solve odd img_max
        # ----------------------------------------------------

        odd_img_max = solve_img_max_for_target_total(
            cls_num=len(odd_digits),
            imb_factor=self.subclass_imbalance_factor,
            target_total=target_odd_total
        )

        odd_counts = get_img_num_per_cls(
            cls_num=len(odd_digits),
            imb_factor=self.subclass_imbalance_factor,
            img_max=odd_img_max
        )

        # ----------------------------------------------------
        # Verify actual class-level imbalance
        # ----------------------------------------------------

        actual_ratio = (
            sum(even_counts) / sum(odd_counts)
        )

        print(
            f"[INFO] Target class imbalance: "
            f"{self.class_imbalance_factor:.2f}"
        )

        print(
            f"[INFO] Actual class imbalance: "
            f"{actual_ratio:.2f}"
        )

        # ----------------------------------------------------
        # Generate even samples
        # ----------------------------------------------------

        for digit, img_num in zip(
            even_digits,
            even_counts
        ):

            idx = np.where(targets == digit)[0]

            np.random.shuffle(idx)

            if img_num > len(idx):

                warnings.warn(
                    f"Digit {digit}: requested "
                    f"{img_num} samples but only "
                    f"{len(idx)} available. Capping."
                )

                img_num = len(idx)

            selected_idx = idx[:img_num]

            imgs = data[selected_idx]

            labels = np.zeros(
                img_num,
                dtype=np.int64
            )

            digit_labels = np.full(
                img_num,
                digit,
                dtype=np.int64
            )

            new_images.append(imgs)
            new_targets.append(labels)
            new_digit_labels.append(digit_labels)

        # ----------------------------------------------------
        # Generate odd samples
        # ----------------------------------------------------

        for digit, img_num in zip(
            odd_digits,
            odd_counts
        ):

            idx = np.where(targets == digit)[0]

            np.random.shuffle(idx)

            if img_num > len(idx):

                warnings.warn(
                    f"Digit {digit}: requested "
                    f"{img_num} samples but only "
                    f"{len(idx)} available. Capping."
                )

                img_num = len(idx)

            selected_idx = idx[:img_num]

            imgs = data[selected_idx]

            labels = np.ones(
                img_num,
                dtype=np.int64
            )

            digit_labels = np.full(
                img_num,
                digit,
                dtype=np.int64
            )

            new_images.append(imgs)
            new_targets.append(labels)
            new_digit_labels.append(digit_labels)

        # ----------------------------------------------------
        # Final storage
        # ----------------------------------------------------

        self.images = np.concatenate(
            new_images,
            axis=0
        )

        self.targets = np.concatenate(
            new_targets,
            axis=0
        )

        self.digit_labels = np.concatenate(
            new_digit_labels,
            axis=0
        )

    def __len__(self):

        return len(self.targets)

    def __getitem__(self, idx):

        img = Image.fromarray(
            self.images[idx],
            mode="L"
        )

        target = int(self.targets[idx])

        if self.transform:
            img = self.transform(img)

        return img, target,idx

    def get_cls_num_list(self):

        return {
            "even": int((self.targets == 0).sum()),
            "odd": int((self.targets == 1).sum())
        }

    def get_digit_num_list(self):

        return {
            "even": {
                d: int(
                    (self.digit_labels == d).sum()
                )
                for d in [0, 2, 4, 6, 8]
            },
            "odd": {
                d: int(
                    (self.digit_labels == d).sum()
                )
                for d in [1, 3, 5, 7, 9]
            }
        }

    def get_imbalance_ratio(self):

        cls_counts = list(
            self.get_cls_num_list().values()
        )

        return max(cls_counts) / min(cls_counts)

    def __repr__(self):

        return (
            f"{self.__class__.__name__}("
            f"class_imbalance_factor="
            f"{self.class_imbalance_factor}, "
            f"subclass_imbalance_factor="
            f"{self.subclass_imbalance_factor}, "
            f"samples={len(self.images)})"
        )


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":

    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
    ])

    # ========================================================
    # Long-Tailed CIFAR-10
    # ========================================================

    cifar10_lt = ImbalancedCIFAR10(
        root="./data",
        imb_factor=100,
        train=True,
        transform=transform_cifar,
    )

    print(cifar10_lt)

    print("CIFAR10 class counts:")
    print(cifar10_lt.get_cls_num_list())

    print()

    # ========================================================
    # Long-Tailed CIFAR-100
    # ========================================================

    cifar100_lt = ImbalancedCIFAR100(
        root="./data",
        imb_factor=100,
        train=True,
        transform=transform_cifar,
    )

    print(cifar100_lt)

    print("First 10 CIFAR100 class counts:")
    print(cifar100_lt.get_cls_num_list()[:10])

    print()

    # ========================================================
    # Long-Tailed Binary MNIST
    # ========================================================

    mnist_lt = LongTailedBinaryMNIST(
        root="./data",
        train=True,
        transform=transform_mnist,
        class_imbalance_factor=100,
        subclass_imbalance_factor=10
    )

    print(mnist_lt)

    print()

    print("Binary class counts:")
    print(mnist_lt.get_cls_num_list())

    print()

    print("Digit-level counts:")
    print(mnist_lt.get_digit_num_list())

    print()

    print("Actual imbalance ratio:")
    print(mnist_lt.get_imbalance_ratio())