"""Reproducibility utilities for MedDriftGuard."""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set fixed random seeds across standard Python, OS environment, and NumPy.
    Ensures scientific reproducibility for experimental evaluation runs.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # Optional torch seeding if torch is installed
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
