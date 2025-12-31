"""Tests for checkpoint inspection."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from python_ai.model import PolicyValueNet, save_checkpoint
from python_ai.inspect import inspect_checkpoint, _try_parse_checkpoint, CheckpointInfo


class TestInspectCheckpoint:
    """Tests for inspect_checkpoint function."""

    def test_inspect_new_format(self) -> None:
        """Inspect new-style checkpoint with full metadata."""
        model = PolicyValueNet(channels=64, blocks=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_checkpoint(
                model,
                str(path),
                optimizer_state={"dummy": "state"},
                training_state={"episode": 100},
            )

            info = inspect_checkpoint(path)

            assert info["format"] == "checkpoint"
            assert info["channels"] == 64
            assert info["blocks"] == 4
            assert info["episode"] == 100
            assert info["has_optimizer"] is True
            assert info["param_count"] > 0

    def test_inspect_old_format(self) -> None:
        """Inspect old-style raw state_dict checkpoint."""
        model = PolicyValueNet(channels=128, blocks=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "old_model.pt"
            torch.save(model.state_dict(), str(path))

            info = inspect_checkpoint(path)

            assert info["format"] == "state_dict"
            assert info["channels"] == 128
            assert info["blocks"] == 8
            assert info["episode"] is None
            assert info["has_optimizer"] is False

    def test_inspect_no_optimizer(self) -> None:
        """Checkpoint without optimizer state."""
        model = PolicyValueNet(channels=32, blocks=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_checkpoint(model, str(path))

            info = inspect_checkpoint(path)

            assert info["has_optimizer"] is False

    def test_inspect_no_training_state(self) -> None:
        """Checkpoint without training state."""
        model = PolicyValueNet(channels=32, blocks=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_checkpoint(model, str(path))

            info = inspect_checkpoint(path)

            assert info["episode"] is None

    def test_inspect_file_not_found(self) -> None:
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            inspect_checkpoint(Path("/nonexistent/model.pt"))

    def test_inspect_param_count(self) -> None:
        """Parameter count is accurate."""
        model = PolicyValueNet(channels=32, blocks=2)
        expected_params = sum(p.numel() for p in model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_checkpoint(model, str(path))

            info = inspect_checkpoint(path)

            assert info["param_count"] == expected_params

    def test_inspect_dtype(self) -> None:
        """Reports parameter dtype."""
        model = PolicyValueNet(channels=32, blocks=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_checkpoint(model, str(path))

            info = inspect_checkpoint(path)

            assert info["dtype"] == "torch.float32"

    def test_inspect_replay_sidecar_present(self) -> None:
        """Detects replay sidecar when present."""
        model = PolicyValueNet(channels=32, blocks=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            replay_path = Path(tmpdir) / "model.pt.replay.npz"

            save_checkpoint(model, str(path))

            # Create dummy replay file
            np.savez(str(replay_path), dummy=np.array([1, 2, 3]))

            info = inspect_checkpoint(path)

            assert info["replay_sidecar"] == str(replay_path)

    def test_inspect_replay_sidecar_absent(self) -> None:
        """Reports None when replay sidecar missing."""
        model = PolicyValueNet(channels=32, blocks=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            save_checkpoint(model, str(path))

            info = inspect_checkpoint(path)

            assert info["replay_sidecar"] is None


class TestTryParseCheckpoint:
    """Tests for _try_parse_checkpoint helper."""

    def test_parse_new_format(self) -> None:
        """Parse new-style checkpoint dict."""
        model = PolicyValueNet(channels=64, blocks=4)
        obj = {
            "state_dict": model.state_dict(),
            "config": {"channels": 64, "blocks": 4},
            "training": {"episode": 50},
            "optimizer": {"state": "data"},
        }

        parsed_model, info = _try_parse_checkpoint(obj)

        assert isinstance(parsed_model, PolicyValueNet)
        assert info.kind == "checkpoint"
        assert info.channels == 64
        assert info.blocks == 4
        assert info.episode == 50
        assert info.has_optimizer is True

    def test_parse_old_format(self) -> None:
        """Parse old-style raw state_dict."""
        model = PolicyValueNet()
        obj = model.state_dict()

        parsed_model, info = _try_parse_checkpoint(obj)

        assert isinstance(parsed_model, PolicyValueNet)
        assert info.kind == "state_dict"

    def test_parse_missing_config_uses_defaults(self) -> None:
        """Missing config uses default values."""
        model = PolicyValueNet(channels=128, blocks=8)
        obj = {
            "state_dict": model.state_dict(),
            # no config key
        }

        parsed_model, info = _try_parse_checkpoint(obj)

        assert info.channels == 128
        assert info.blocks == 8

    def test_parse_invalid_raises(self) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError):
            _try_parse_checkpoint({"invalid": "data"})

        with pytest.raises(ValueError):
            _try_parse_checkpoint([1, 2, 3])

        with pytest.raises(ValueError):
            _try_parse_checkpoint("string")
