"""Shared RNG seeding helpers."""

from __future__ import annotations

import random
import secrets

import numpy as np
import torch


def seed_everything(seed: int | None = None) -> int:
    """Seed Python, NumPy, Torch CPU, and CUDA RNGs.

    Args:
        seed: Seed to apply. If ``None``, generate a random 32-bit seed.

    Returns:
        The applied seed.
    """
    if seed is None:
        seed = secrets.randbits(32)
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
