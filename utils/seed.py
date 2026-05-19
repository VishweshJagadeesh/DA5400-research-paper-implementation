import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):

    # --------------------------------------------------------
    # python
    # --------------------------------------------------------

    random.seed(seed)

    # --------------------------------------------------------
    # numpy
    # --------------------------------------------------------

    np.random.seed(seed)

    # --------------------------------------------------------
    # pytorch
    # --------------------------------------------------------

    torch.manual_seed(seed)

    # --------------------------------------------------------
    # cuda
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # --------------------------------------------------------
    # deterministic behavior
    # --------------------------------------------------------

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --------------------------------------------------------
    # hash seed
    # --------------------------------------------------------

    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[INFO] Global seed set to {seed}")