"""Inspect a saved checkpoint and print useful metadata.

Examples:
    python -m python_ai.inspect --model-path python_ai/checkpoints/overnight.pt

Notes:
- New-style checkpoints created by python_ai.train include:
  - state_dict
  - config (channels/blocks)
  - optimizer (optional)
  - training (episode)
- Older checkpoints may be raw state_dict only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from python_ai.model import PolicyValueNet


@dataclass
class CheckpointInfo:
    kind: str
    channels: Optional[int]
    blocks: Optional[int]
    episode: Optional[int]
    has_optimizer: bool
    param_count: int


def _try_parse_checkpoint(obj: Any) -> Tuple[PolicyValueNet, CheckpointInfo]:
    # New-style checkpoint: dict with state_dict
    if isinstance(obj, dict) and "state_dict" in obj:
        cfg = obj.get("config", {}) if isinstance(obj.get("config", {}), dict) else {}
        channels = int(cfg.get("channels", 128)) if "channels" in cfg else 128
        blocks = int(cfg.get("blocks", 8)) if "blocks" in cfg else 8

        model = PolicyValueNet(channels=channels, blocks=blocks)
        model.load_state_dict(obj["state_dict"])

        training = obj.get("training") if isinstance(obj.get("training"), dict) else None
        episode = int(training.get("episode")) if training and "episode" in training else None

        has_optimizer = isinstance(obj.get("optimizer"), dict)
        param_count = sum(p.numel() for p in model.parameters())
        return model, CheckpointInfo(
            kind="checkpoint",
            channels=channels,
            blocks=blocks,
            episode=episode,
            has_optimizer=has_optimizer,
            param_count=int(param_count),
        )

    # Old-style: raw state_dict
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        model = PolicyValueNet()
        try:
            model.load_state_dict(obj)  # type: ignore[arg-type]
            param_count = sum(p.numel() for p in model.parameters())
            return model, CheckpointInfo(
                kind="state_dict",
                channels=getattr(model, "channels", None),
                blocks=getattr(model, "blocks", None),
                episode=None,
                has_optimizer=False,
                param_count=int(param_count),
            )
        except Exception:
            # fall through
            pass

    raise ValueError("Unrecognized checkpoint format")


def inspect_checkpoint(model_path: Path) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))

    obj = torch.load(str(model_path), map_location=torch.device("cpu"))
    model, info = _try_parse_checkpoint(obj)

    # Also report replay sidecar if it exists.
    replay_path = Path(str(model_path) + ".replay.npz")

    out: Dict[str, Any] = {
        "path": str(model_path),
        "format": info.kind,
        "channels": info.channels,
        "blocks": info.blocks,
        "episode": info.episode,
        "has_optimizer": bool(info.has_optimizer),
        "param_count": int(info.param_count),
        "dtype": str(next(model.parameters()).dtype) if info.param_count else None,
        "replay_sidecar": str(replay_path) if replay_path.exists() else None,
    }
    return out


def _print(out: Dict[str, Any]) -> None:
    print(f"Path: {out['path']}")
    print(f"Format: {out['format']}")
    print(f"Config: channels={out['channels']} blocks={out['blocks']}")
    if out.get("episode") is None:
        print("Training: episode=unknown (not stored in this checkpoint)")
    else:
        print(f"Training: episode={int(out['episode'])}")
    print(f"Optimizer state: {'yes' if out.get('has_optimizer') else 'no'}")
    print(f"Parameters: {int(out['param_count']):,}")
    if out.get("dtype") is not None:
        print(f"DType: {out['dtype']}")
    if out.get("replay_sidecar"):
        print(f"Replay sidecar: {out['replay_sidecar']}")
    else:
        print("Replay sidecar: (none found)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect a Gomoku policy-value checkpoint")
    p.add_argument("--model-path", required=True, help="Path to .pt checkpoint")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = inspect_checkpoint(Path(args.model_path).expanduser().resolve())
    _print(out)


if __name__ == "__main__":
    main()
