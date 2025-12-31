"""Shared pytest fixtures for python_ai tests."""
from __future__ import annotations

import pytest
import torch
import numpy as np

from python_ai.gomoku_env import GomokuEnv, BOARD_SIZE
from python_ai.model import PolicyValueNet


@pytest.fixture
def env() -> GomokuEnv:
    """Fresh GomokuEnv instance."""
    return GomokuEnv()


@pytest.fixture
def small_model() -> PolicyValueNet:
    """Small model for fast tests (channels=8, blocks=2)."""
    return PolicyValueNet(channels=8, blocks=2)


@pytest.fixture
def device() -> torch.device:
    """CPU device for deterministic tests."""
    return torch.device("cpu")


@pytest.fixture
def seed() -> int:
    """Fixed seed for reproducibility."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed
