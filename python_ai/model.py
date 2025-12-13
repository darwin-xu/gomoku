"""Policy-Value network and utilities (ResNet backbone)."""
from __future__ import annotations

import importlib.util
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, Any, Dict

from .gomoku_env import BOARD_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        return F.relu(out)


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 128, blocks: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.input_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            x = block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value.squeeze(-1)


@dataclass
class DeviceConfig:
    device: torch.device
    use_mps: bool


def get_device() -> DeviceConfig:
    if torch.backends.mps.is_available():
        return DeviceConfig(device=torch.device("mps"), use_mps=True)
    return DeviceConfig(device=torch.device("cpu"), use_mps=False)


def save_checkpoint(
    model: PolicyValueNet,
    path: str,
    *,
    optimizer_state: Optional[Dict[str, Any]] = None,
    training_state: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "config": {
            "channels": int(getattr(model, "channels", 128)),
            "blocks": int(getattr(model, "blocks", 8)),
        },
    }
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    if training_state is not None:
        payload["training"] = training_state
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: Optional[torch.device] = None) -> PolicyValueNet:
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        cfg = state.get("config", {}) if isinstance(state.get("config", {}), dict) else {}
        channels = int(cfg.get("channels", 128))
        blocks = int(cfg.get("blocks", 8))
        model = PolicyValueNet(channels=channels, blocks=blocks)
        model.load_state_dict(state["state_dict"])
        return model

    # Backward compatibility: older checkpoints were raw state_dict
    model = PolicyValueNet()
    model.load_state_dict(state)
    return model


def export_coreml(model_path: str, coreml_path: str) -> None:
    """Export a trained PyTorch model to CoreML for ANE-friendly inference."""
    if sys.platform != "darwin":
        raise RuntimeError("CoreML export is only supported on macOS")
    if sys.version_info >= (3, 13):
        raise RuntimeError(
            f"CoreML export is not supported on Python {sys.version_info.major}.{sys.version_info.minor}. "
            "Use Python 3.11/3.12 for coremltools-based export."
        )
    if importlib.util.find_spec("coremltools") is None:
        raise RuntimeError("coremltools is not installed; cannot export CoreML")
    # Avoid importing coremltools if its native extension isn't present for this Python.
    if importlib.util.find_spec("coremltools.libcoremlpython") is None:
        raise RuntimeError(
            "coremltools native library is missing (coremltools.libcoremlpython). "
            "This typically means coremltools does not ship wheels for your Python version."
        )

    import coremltools as ct

    dummy = torch.zeros(1, 3, BOARD_SIZE, BOARD_SIZE)
    model = load_checkpoint(model_path, map_location=torch.device("cpu"))
    model.eval()
    traced = torch.jit.trace(model, dummy)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=dummy.shape, name="input")],
        compute_units=ct.ComputeUnit.ALL,
    )
    mlmodel.save(coreml_path)
