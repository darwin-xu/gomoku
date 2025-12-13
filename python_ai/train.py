"""Self-play training loop for Gomoku policy-value net."""
from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

# Allow running as a script: python python_ai/train.py
if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from python_ai.gomoku_env import BOARD_SIZE
from python_ai.model import PolicyValueNet, get_device, save_checkpoint, load_checkpoint, export_coreml
from python_ai.self_play import Sample, apply_symmetry, play_self_game_mcts


class ReplayDataset(Dataset):
    def __init__(self, samples: List[Sample], augment: bool) -> None:
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        state = torch.from_numpy(s.state)  # (3,15,15)
        pi = torch.from_numpy(s.pi)        # (225,)
        z = torch.tensor(s.z, dtype=torch.float32)
        if self.augment:
            sym = random.randint(0, 7)
            state, pi = apply_symmetry(state, pi, sym)
        return state, pi, z


def make_dataloader(samples: List[Sample], batch_size: int, augment: bool) -> DataLoader:
    ds = ReplayDataset(samples, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=getattr(make_dataloader, "_num_workers", 0),
        persistent_workers=getattr(make_dataloader, "_num_workers", 0) > 0,
    )


def train(args: argparse.Namespace) -> None:
    if args.torch_threads and args.torch_threads > 0:
        try:
            torch.set_num_threads(int(args.torch_threads))
            torch.set_num_interop_threads(max(1, int(args.torch_threads) // 2))
            print(f"Torch CPU threads: {int(args.torch_threads)}")
        except Exception as e:
            print(f"Warning: failed to set torch threads ({e})")

    device_cfg = get_device()
    device = device_cfg.device
    print(f"Using device: {device} (MPS={device_cfg.use_mps})")

    # Configure DataLoader workers (0 = main process)
    make_dataloader._num_workers = int(args.dataloader_workers) if args.dataloader_workers else 0  # type: ignore[attr-defined]

    start_episode = 1
    optimizer_state = None

    if args.resume and args.model_path.exists():
        raw = torch.load(str(args.model_path), map_location=device)
        if isinstance(raw, dict) and "state_dict" in raw:
            cfg = raw.get("config", {}) if isinstance(raw.get("config", {}), dict) else {}
            channels = int(cfg.get("channels", args.channels))
            blocks = int(cfg.get("blocks", args.blocks))
            model = PolicyValueNet(channels=channels, blocks=blocks)
            model.load_state_dict(raw["state_dict"])
            training_state = raw.get("training", {}) if isinstance(raw.get("training", {}), dict) else {}
            last_episode = int(training_state.get("episode", 0))
            start_episode = max(1, last_episode + 1)
            optimizer_state = raw.get("optimizer") if isinstance(raw.get("optimizer"), dict) else None
        else:
            # Backward compatibility: old file is raw state_dict
            model = load_checkpoint(str(args.model_path), map_location=device)
        print(f"Loaded model from {args.model_path} (resuming at episode {start_episode})")
    else:
        model = PolicyValueNet(channels=args.channels, blocks=args.blocks)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.resume and optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("Loaded optimizer state from checkpoint")
        except Exception as e:
            print(f"Warning: failed to load optimizer state ({e}); continuing with fresh optimizer")

    replay: Deque[Sample] = deque(maxlen=args.replay_size)

    for episode in range(start_episode, args.episodes + 1):
        model.eval()
        # Temperature schedule: explore more early in the game and early in training.
        temperature = args.temperature
        if args.temperature_decay and episode > 1:
            temperature = max(args.min_temperature, args.temperature * (args.temperature_decay ** (episode - 1)))

        episode_samples: List[Sample] = []
        for _ in range(max(1, int(args.games_per_episode))):
            game_samples = play_self_game_mcts(
                model=model,
                device=device,
                simulations=args.simulations,
                temperature=temperature,
                c_puct=args.c_puct,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_frac=args.dirichlet_frac,
            )
            episode_samples.extend(game_samples)
        replay.extend(episode_samples)

        winner = 0
        if episode_samples:
            # All samples share the same outcome; infer from first target
            # (z is from perspective of player_to_move at that state).
            # Not perfect for reporting, but good enough for stats.
            winner = 0

        warmup = len(replay) < args.min_replay

        avg_policy = float("nan")
        avg_value = float("nan")

        if not warmup:
            model.train()
            batch = random.sample(replay, k=min(len(replay), args.batch_size * args.batches_per_episode))
            loader = make_dataloader(batch, args.batch_size, augment=args.augment)

            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_batches = 0

            for states, target_pi, target_z in loader:
                states = states.to(device)
                target_pi = target_pi.to(device)
                target_z = target_z.to(device)

                optimizer.zero_grad()
                policy_logits, values = model(states)

                log_probs = F.log_softmax(policy_logits, dim=-1)
                policy_loss = -(target_pi * log_probs).sum(dim=-1).mean()
                value_loss = F.mse_loss(values, target_z)
                loss = policy_loss + args.value_loss_weight * value_loss
                loss.backward()
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_batches += 1

            avg_policy = total_policy_loss / max(1, total_batches)
            avg_value = total_value_loss / max(1, total_batches)

        status = "warmup" if warmup else "train"
        policy_str = "n/a" if warmup else f"{avg_policy:.4f}"
        value_str = "n/a" if warmup else f"{avg_value:.4f}"
        print(
            f"Episode {episode}: status={status}, moves={len(episode_samples)}, games={int(args.games_per_episode)}, sims={args.simulations}, temp={temperature:.3f}, "
            f"policy_loss={policy_str}, value_loss={value_str}, replay={len(replay)}/{args.min_replay}"
        )

        if episode % args.save_every == 0:
            save_checkpoint(
                model,
                str(args.model_path),
                optimizer_state=optimizer.state_dict(),
                training_state={"episode": int(episode)},
            )
            print(f"Saved model to {args.model_path}")
            if args.coreml_path:
                export_coreml(str(args.model_path), str(args.coreml_path))
                print(f"Exported CoreML model to {args.coreml_path}")

    # final save
    save_checkpoint(
        model,
        str(args.model_path),
        optimizer_state=optimizer.state_dict(),
        training_state={"episode": int(args.episodes)},
    )
    print(f"Training finished. Model saved to {args.model_path}")
    if args.coreml_path:
        export_coreml(str(args.model_path), str(args.coreml_path))
        print(f"CoreML model saved to {args.coreml_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-play training for Gomoku")
    parser.add_argument("--model-path", type=str, default="./python_ai/checkpoints/policy_value.pt")
    parser.add_argument("--coreml-path", type=str, default="", help="Optional CoreML export path (.mlpackage)")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches-per-episode", type=int, default=8)
    parser.add_argument("--games-per-episode", type=int, default=1, help="How many self-play games to generate per episode")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--min-replay", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-decay", type=float, default=0.995)
    parser.add_argument("--min-temperature", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-frac", type=float, default=0.25)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=0, help="CPU threads for PyTorch ops (0 = default)")
    parser.add_argument("--dataloader-workers", type=int, default=0, help="DataLoader worker processes (0 = main process)")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from pathlib import Path

    args.model_path = Path(args.model_path)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    if args.coreml_path:
        from pathlib import Path
        args.coreml_path = Path(args.coreml_path)
        args.coreml_path.parent.mkdir(parents=True, exist_ok=True)

    train(args)
