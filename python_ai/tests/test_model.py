"""Tests for PolicyValueNet model."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from python_ai.gomoku_env import BOARD_SIZE
from python_ai.model import (
    PolicyValueNet,
    ResidualBlock,
    get_device,
    save_checkpoint,
    load_checkpoint,
)


class TestResidualBlock:
    """ResidualBlock unit tests."""

    def test_residual_output_shape(self) -> None:
        """ResidualBlock preserves spatial dimensions."""
        block = ResidualBlock(channels=32)
        x = torch.randn(2, 32, 15, 15)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self) -> None:
        """ResidualBlock adds input to output (skip connection)."""
        block = ResidualBlock(channels=16)
        block.eval()
        x = torch.randn(1, 16, 15, 15)

        # Zero out conv weights to isolate residual
        with torch.no_grad():
            block.conv1.weight.zero_()
            block.conv2.weight.zero_()
            block.bn1.weight.fill_(1)
            block.bn1.bias.zero_()
            block.bn2.weight.fill_(1)
            block.bn2.bias.zero_()

        out = block(x)
        # Output should be relu(x + 0) = relu(x)
        expected = torch.relu(x)
        assert torch.allclose(out, expected, atol=1e-5)


class TestPolicyValueNet:
    """PolicyValueNet tests."""

    def test_forward_shapes(self, small_model: PolicyValueNet) -> None:
        """Forward pass produces correct output shapes."""
        batch_size = 4
        x = torch.randn(batch_size, 3, BOARD_SIZE, BOARD_SIZE)

        policy, value = small_model(x)

        assert policy.shape == (batch_size, BOARD_SIZE * BOARD_SIZE)
        assert value.shape == (batch_size,)

    def test_value_range(self, small_model: PolicyValueNet) -> None:
        """Value head outputs in [-1, 1] due to tanh."""
        x = torch.randn(10, 3, BOARD_SIZE, BOARD_SIZE)
        _, value = small_model(x)

        assert (value >= -1).all()
        assert (value <= 1).all()

    def test_policy_logits_finite(self, small_model: PolicyValueNet) -> None:
        """Policy logits are finite (no NaN/inf)."""
        x = torch.randn(5, 3, BOARD_SIZE, BOARD_SIZE)
        policy, _ = small_model(x)

        assert torch.isfinite(policy).all()

    def test_model_configs(self) -> None:
        """Model stores its configuration."""
        model = PolicyValueNet(channels=64, blocks=4)
        assert model.channels == 64
        assert model.blocks == 4

    def test_default_config(self) -> None:
        """Default configuration is 128 channels, 8 blocks."""
        model = PolicyValueNet()
        assert model.channels == 128
        assert model.blocks == 8

    def test_parameter_count_varies_with_config(self) -> None:
        """Larger configs have more parameters."""
        small = PolicyValueNet(channels=32, blocks=2)
        large = PolicyValueNet(channels=128, blocks=8)

        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())

        assert large_params > small_params

    def test_gradients_flow(self, small_model: PolicyValueNet) -> None:
        """Gradients flow through both heads."""
        x = torch.randn(2, 3, BOARD_SIZE, BOARD_SIZE)
        policy, value = small_model(x)

        loss = policy.sum() + value.sum()
        loss.backward()

        # Check gradients exist for input layer
        assert small_model.input_conv.weight.grad is not None
        assert small_model.input_conv.weight.grad.abs().sum() > 0


class TestCheckpointing:
    """Checkpoint save/load tests."""

    def test_save_load_roundtrip(self, small_model: PolicyValueNet) -> None:
        """Model can be saved and loaded identically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_checkpoint(small_model, str(path))

            loaded = load_checkpoint(str(path))

            # Compare state dicts
            for key in small_model.state_dict():
                assert torch.equal(
                    small_model.state_dict()[key],
                    loaded.state_dict()[key],
                )

    def test_checkpoint_preserves_config(self) -> None:
        """Checkpoint preserves channels/blocks config."""
        model = PolicyValueNet(channels=64, blocks=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_checkpoint(model, str(path))

            loaded = load_checkpoint(str(path))

            assert loaded.channels == 64
            assert loaded.blocks == 4

    def test_checkpoint_with_optimizer_state(self, small_model: PolicyValueNet) -> None:
        """Checkpoint can include optimizer state."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)

        # Do a step to create state
        x = torch.randn(1, 3, BOARD_SIZE, BOARD_SIZE)
        policy, value = small_model(x)
        (policy.sum() + value.sum()).backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_checkpoint(
                small_model,
                str(path),
                optimizer_state=optimizer.state_dict(),
                training_state={"episode": 42},
            )

            # Load raw checkpoint to verify structure
            raw = torch.load(str(path))
            assert "optimizer" in raw
            assert "training" in raw
            assert raw["training"]["episode"] == 42

    def test_checkpoint_with_training_state(self, small_model: PolicyValueNet) -> None:
        """Training state is preserved in checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_checkpoint(
                small_model,
                str(path),
                training_state={"episode": 100},
            )

            raw = torch.load(str(path))
            assert raw["training"]["episode"] == 100

    def test_load_on_different_device(self, small_model: PolicyValueNet) -> None:
        """Can load checkpoint to specific device."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_checkpoint(small_model, str(path))

            loaded = load_checkpoint(str(path), map_location=torch.device("cpu"))

            # Verify on CPU
            assert next(loaded.parameters()).device == torch.device("cpu")


class TestDeviceConfig:
    """Device configuration tests."""

    def test_get_device_returns_config(self) -> None:
        """get_device returns DeviceConfig."""
        cfg = get_device()
        assert hasattr(cfg, "device")
        assert hasattr(cfg, "use_mps")
        assert isinstance(cfg.device, torch.device)
        assert isinstance(cfg.use_mps, bool)

    def test_device_is_valid(self) -> None:
        """Returned device is usable."""
        cfg = get_device()
        # Should be able to create tensor on device
        t = torch.zeros(1, device=cfg.device)
        assert t.device.type == cfg.device.type


class TestBackwardCompatibility:
    """Tests for loading old-style checkpoints."""

    def test_load_raw_state_dict(self) -> None:
        """Can load old-style raw state_dict checkpoint."""
        model = PolicyValueNet(channels=128, blocks=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "old_model.pt"
            # Save in old format (raw state_dict)
            torch.save(model.state_dict(), str(path))

            # Should still load
            loaded = load_checkpoint(str(path))
            assert loaded.channels == 128
            assert loaded.blocks == 8
