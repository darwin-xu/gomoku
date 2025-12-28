"""Self-play training loop for Gomoku policy-value net."""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple, Optional

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
from python_ai.self_play import Sample, apply_symmetry, play_self_game_mcts, play_self_game_mcts_with_trace
from python_ai.telemetry import make_telemetry


def _default_replay_path(model_path: Path) -> Path:
    # Keep the original suffix (e.g., .pt) and append a sidecar extension.
    return Path(str(model_path) + ".replay.npz")


def save_replay_buffer(replay: Deque[Sample], path: Path) -> None:
    if not replay:
        return
    states = np.stack([s.state for s in replay], axis=0).astype(np.float32, copy=False)
    pis = np.stack([s.pi for s in replay], axis=0).astype(np.float32, copy=False)
    zs = np.asarray([s.z for s in replay], dtype=np.float32)
    players = np.asarray([s.player_to_move for s in replay], dtype=np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, states=states, pis=pis, zs=zs, players=players)


def load_replay_buffer(path: Path, maxlen: int) -> Deque[Sample]:
    replay: Deque[Sample] = deque(maxlen=maxlen)
    if not path.exists():
        return replay
    data = np.load(path)
    states = data["states"]
    pis = data["pis"]
    zs = data["zs"]
    players = data.get("players")
    if players is None:
        players = np.ones((states.shape[0],), dtype=np.int8)
    count = int(min(states.shape[0], maxlen))
    start = int(states.shape[0] - count)
    for i in range(start, start + count):
        replay.append(Sample(
            state=states[i],
            pi=pis[i],
            z=float(zs[i]),
            player_to_move=int(players[i]),
        ))
    return replay


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


def _policy_snapshot_from_sample(sample: Sample) -> dict:
    state = sample.state
    board_size = int(state.shape[1])
    board = np.zeros((board_size, board_size), dtype=np.int8)
    board[state[0] > 0.5] = int(sample.player_to_move)
    board[state[1] > 0.5] = int(-sample.player_to_move)
    return {
        "board_size": board_size,
        "board": board.reshape(-1).astype(np.int8).tolist(),
        "policy": sample.pi.astype(np.float32).tolist(),
        "player_to_move": int(sample.player_to_move),
    }


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

    dashboard_url = str(getattr(args, "dashboard_url", "") or "").strip()
    if not dashboard_url and bool(getattr(args, "telemetry", False)):
        # Backward compatibility: previously training started its own telemetry server.
        # Now --telemetry means "report to a dashboard", using host/port if provided.
        host = str(getattr(args, "telemetry_host", "127.0.0.1") or "127.0.0.1")
        port = int(getattr(args, "telemetry_port", 8765) or 8765)
        dashboard_url = f"http://{host}:{port}"

    telemetry = make_telemetry(
        enabled=bool(dashboard_url),
        dashboard_url=dashboard_url if dashboard_url else None,
        job_id=str(getattr(args, "dashboard_job_id", "") or str(getattr(args, "job_id", "") or "")) or None,
    )

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

    replay_path: Path = args.replay_path if args.replay_path is not None else _default_replay_path(args.model_path)
    if args.resume:
        replay = load_replay_buffer(replay_path, maxlen=args.replay_size)
        if replay:
            print(f"Loaded replay buffer: {len(replay)} samples from {replay_path}")
        else:
            replay = deque(maxlen=args.replay_size)
    else:
        replay = deque(maxlen=args.replay_size)

    for episode in range(start_episode, args.episodes + 1):
        ep_t0 = time.perf_counter()
        model.eval()
        # Temperature schedule: explore more early in the game and early in training.
        temperature = args.temperature
        if args.temperature_decay and episode > 1:
            temperature = max(args.min_temperature, args.temperature * (args.temperature_decay ** (episode - 1)))

        episode_samples: List[Sample] = []
        policy_snapshot: Optional[dict] = None
        self_play_trace: Optional[dict] = None
        rich_telemetry = bool(getattr(telemetry, "cfg", None) and telemetry.cfg.enabled)

        for _ in range(max(1, int(args.games_per_episode))):
            need_trace = rich_telemetry and self_play_trace is None
            if need_trace:
                game_samples, trace = play_self_game_mcts_with_trace(
                    model=model,
                    device=device,
                    simulations=args.simulations,
                    temperature=temperature,
                    c_puct=args.c_puct,
                    dirichlet_alpha=args.dirichlet_alpha,
                    dirichlet_frac=args.dirichlet_frac,
                )
                if trace:
                    self_play_trace = trace
            else:
                game_samples = play_self_game_mcts(
                    model=model,
                    device=device,
                    simulations=args.simulations,
                    temperature=temperature,
                    c_puct=args.c_puct,
                    dirichlet_alpha=args.dirichlet_alpha,
                    dirichlet_frac=args.dirichlet_frac,
                )
            if policy_snapshot is None and game_samples:
                policy_snapshot = _policy_snapshot_from_sample(game_samples[0])

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

        ep_sec = time.perf_counter() - ep_t0
        completed_in_this_run = (episode - start_episode) + 1
        if completed_in_this_run == 1:
            avg_ep_sec = ep_sec
        else:
            # Running average without storing all times.
            prev = getattr(train, "_avg_ep_sec", ep_sec)
            avg_ep_sec = prev + (ep_sec - prev) / float(completed_in_this_run)
        train._avg_ep_sec = avg_ep_sec  # type: ignore[attr-defined]

        est_1000_sec = avg_ep_sec * 1000.0
        est_1000_hr = est_1000_sec / 3600.0

        point = {
            "episode": int(episode),
            "status": status,
            "moves": int(len(episode_samples)),
            "games": int(args.games_per_episode),
            "simulations": int(args.simulations),
            "temperature": float(temperature),
            "policy_loss": None if warmup else float(avg_policy),
            "value_loss": None if warmup else float(avg_value),
            "replay_size": int(len(replay)),
            "min_replay": int(args.min_replay),
            "episode_sec": float(ep_sec),
            "avg_ep_sec": float(avg_ep_sec),
            "est_1000_ep_hr": float(est_1000_hr),
        }
        if policy_snapshot is not None:
            point["policy_snapshot"] = policy_snapshot
        if self_play_trace is not None:
            point["self_play_trace"] = self_play_trace

        telemetry.log(point)
        print(
            f"Episode {episode}: status={status}, moves={len(episode_samples)}, games={int(args.games_per_episode)}, sims={args.simulations}, temp={temperature:.3f}, "
            f"policy_loss={policy_str}, value_loss={value_str}, replay={len(replay)}/{args.min_replay}, "
            f"episode_sec={ep_sec:.2f}, avg_ep_sec={avg_ep_sec:.2f}, est_1000_ep_hr={est_1000_hr:.2f}"
        )

        if episode % args.save_every == 0:
            save_checkpoint(
                model,
                str(args.model_path),
                optimizer_state=optimizer.state_dict(),
                training_state={"episode": int(episode)},
            )
            print(f"Saved model to {args.model_path}")
            if replay_path:
                try:
                    save_replay_buffer(replay, replay_path)
                    print(f"Saved replay buffer: {len(replay)} samples to {replay_path}")
                except Exception as e:
                    print(f"Warning: failed to save replay buffer ({type(e).__name__}: {e})")
            if args.coreml_path:
                try:
                    export_coreml(str(args.model_path), str(args.coreml_path))
                    print(f"Exported CoreML model to {args.coreml_path}")
                except Exception as e:
                    print(
                        "CoreML export failed (training still succeeded). "
                        f"Error: {type(e).__name__}: {e}"
                    )
                    print(
                        "Hint: CoreML export depends on coremltools compatibility with your Python and PyTorch versions. "
                        "If you need CoreML, try Python 3.11/3.12 and a coremltools-supported PyTorch version (often <= 2.7)."
                    )

    # final save
    save_checkpoint(
        model,
        str(args.model_path),
        optimizer_state=optimizer.state_dict(),
        training_state={"episode": int(args.episodes)},
    )
    print(f"Training finished. Model saved to {args.model_path}")
    if replay_path:
        try:
            save_replay_buffer(replay, replay_path)
            print(f"Saved replay buffer: {len(replay)} samples to {replay_path}")
        except Exception as e:
            print(f"Warning: failed to save replay buffer ({type(e).__name__}: {e})")
    if args.coreml_path:
        try:
            export_coreml(str(args.model_path), str(args.coreml_path))
            print(f"CoreML model saved to {args.coreml_path}")
        except Exception as e:
            print(
                "CoreML export failed (training still succeeded). "
                f"Error: {type(e).__name__}: {e}"
            )
            print(
                "Hint: CoreML export depends on coremltools compatibility with your Python and PyTorch versions. "
                "If you need CoreML, try Python 3.11/3.12 and a coremltools-supported PyTorch version (often <= 2.7)."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AlphaZero-style self-play training for Gomoku (15×15, 5-in-a-row).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Basic run with default settings
  python -m python_ai.train --model-path checkpoints/my_model.pt --episodes 500

  # Stronger, less random self-play
  python -m python_ai.train --simulations 128 --temperature 0.6 \\
      --temperature-decay 0.99 --min-temperature 0.2 \\
      --dirichlet-alpha 0.15 --dirichlet-frac 0.15

  # Large-scale training with dashboard telemetry
  python -m python_ai.train --episodes 50000 --replay-size 200000 \\
      --min-replay 10000 --games-per-episode 4 --augment \\
      --dashboard-url http://127.0.0.1:8787 --dashboard-job-id run1
""",
    )

    # ── Model & Checkpointing ────────────────────────────────────────────────
    ckpt = parser.add_argument_group("Model & Checkpointing")
    ckpt.add_argument(
        "--model-path",
        type=str,
        default="./python_ai/checkpoints/policy_value.pt",
        help="Path to save/load neural network weights. If --resume is set and file exists, "
             "training continues from it; otherwise a fresh network is created.",
    )
    ckpt.add_argument(
        "--resume",
        action="store_true",
        help="Load weights, optimizer state, and episode counter from --model-path and continue. "
             "Without this flag training starts from scratch, overwriting any existing file.",
    )
    ckpt.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Checkpoint frequency in episodes. Lower = more rollback points but more I/O.",
    )
    ckpt.add_argument(
        "--coreml-path",
        type=str,
        default="",
        help="Optional .mlpackage path. After each save the model is exported to CoreML for "
             "on-device inference (macOS/iOS). Requires coremltools.",
    )
    ckpt.add_argument(
        "--replay-path",
        type=str,
        default="",
        help="Sidecar file storing the experience buffer (default: <model-path>.replay.npz). "
             "Use a custom path to share replay across runs or checkpoint it separately.",
    )

    # ── Network Architecture ─────────────────────────────────────────────────
    arch = parser.add_argument_group("Network Architecture")
    arch.add_argument(
        "--channels",
        type=int,
        default=128,
        help="Width of the residual tower (feature-map depth). More channels → more capacity "
             "but slower training/inference. 64–256 typical.",
    )
    arch.add_argument(
        "--blocks",
        type=int,
        default=8,
        help="Number of residual blocks. Deeper networks learn more complex patterns; 6–12 common for 15×15.",
    )

    # ── Self-Play (Data Generation) ──────────────────────────────────────────
    sp = parser.add_argument_group("Self-Play (Data Generation)")
    sp.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Total training iterations. Each episode: generate games → sample from replay → gradient steps.",
    )
    sp.add_argument(
        "--games-per-episode",
        type=int,
        default=1,
        help="Self-play games generated per episode. More games = more diversity per update but longer episodes.",
    )
    sp.add_argument(
        "--simulations",
        type=int,
        default=64,
        help="MCTS rollouts per move. Higher = stronger, sharper policies; 64–256 balances speed vs quality.",
    )
    sp.add_argument(
        "--c-puct",
        type=float,
        default=1.5,
        help="Exploration constant in PUCT formula. Higher encourages following the prior; "
             "lower trusts value estimates more. 1–2 typical.",
    )
    sp.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for move selection from visit counts. At τ=1 moves are proportional "
             "to visits; as τ→0 it becomes greedy. High early → diversity; low late → quality.",
    )
    sp.add_argument(
        "--temperature-decay",
        type=float,
        default=0.995,
        help="Multiplicative decay applied each episode: τ_{e+1} = τ_e × decay. Gradually sharpens play.",
    )
    sp.add_argument(
        "--min-temperature",
        type=float,
        default=0.1,
        help="Floor for temperature schedule. Prevents fully greedy play which can hurt diversity late in training.",
    )
    sp.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.3,
        help="Shape parameter of Dirichlet noise added to the root prior. Smaller α → spikier noise "
             "(more aggressive exploration). For 15×15 boards 0.15–0.3 common.",
    )
    sp.add_argument(
        "--dirichlet-frac",
        type=float,
        default=0.25,
        help="Blend weight: P'(a) = (1−ε)P(a) + ε·Dir(α). 0.25 = 25%% noise; increase for more opening variety.",
    )

    # ── Replay Buffer & Training ─────────────────────────────────────────────
    train_grp = parser.add_argument_group("Replay Buffer & Training")
    train_grp.add_argument(
        "--replay-size",
        type=int,
        default=50000,
        help="Max samples in the circular buffer. Larger buffers keep older data longer; "
             "tune to balance recency vs coverage.",
    )
    train_grp.add_argument(
        "--min-replay",
        type=int,
        default=5000,
        help="Warmup threshold. No gradient updates until buffer has at least this many samples, "
             "preventing early overfitting to few games.",
    )
    train_grp.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Samples per mini-batch. Larger batches stabilize gradients but need more memory.",
    )
    train_grp.add_argument(
        "--batches-per-episode",
        type=int,
        default=8,
        help="Gradient steps per episode. Controls how much you train on existing data vs generating new data.",
    )
    train_grp.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate. Start around 1e-3; decay or lower if loss plateaus/diverges.",
    )
    train_grp.add_argument(
        "--value-loss-weight",
        type=float,
        default=1.0,
        help="Coefficient on MSE value loss vs cross-entropy policy loss: L = L_π + w·L_v. "
             "Increase if value head lags; decrease if policy collapses.",
    )
    train_grp.add_argument(
        "--augment",
        action="store_true",
        help="Apply random dihedral symmetry (8 transformations) to each sample. Effectively 8× data; highly recommended.",
    )

    # ── Performance Tuning ───────────────────────────────────────────────────
    perf = parser.add_argument_group("Performance Tuning")
    perf.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="CPU threads for intra-op parallelism (0 = PyTorch default). Limit on shared machines or if GPU is primary.",
    )
    perf.add_argument(
        "--dataloader-workers",
        type=int,
        default=0,
        help="Parallel processes loading batches. >0 can speed up CPU-bound augmentation; 0 keeps everything in main process.",
    )

    # ── Telemetry / Dashboard ────────────────────────────────────────────────
    tele = parser.add_argument_group("Telemetry / Dashboard")
    tele.add_argument(
        "--dashboard-url",
        type=str,
        default="",
        help="Base URL of a running dashboard (e.g. http://127.0.0.1:8787). Training POSTs metrics each episode for live plotting.",
    )
    tele.add_argument(
        "--dashboard-job-id",
        type=str,
        default="",
        help="Identifier grouping telemetry points on the dashboard. Useful when multiple runs report to the same server.",
    )

    # ── Deprecated / Backward Compatibility ───────────────────────────────────
    deprecated = parser.add_argument_group("Deprecated (Backward Compatibility)")
    deprecated.add_argument(
        "--telemetry",
        action="store_true",
        help="(Deprecated) Legacy flag; enables posting to --telemetry-host/--telemetry-port. Prefer --dashboard-url.",
    )
    deprecated.add_argument(
        "--telemetry-host",
        type=str,
        default="127.0.0.1",
        help="(Deprecated) Host for legacy telemetry mode.",
    )
    deprecated.add_argument(
        "--telemetry-port",
        type=int,
        default=8765,
        help="(Deprecated) Port for legacy telemetry mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from pathlib import Path

    args.model_path = Path(args.model_path)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    if args.replay_path:
        args.replay_path = Path(args.replay_path)
    else:
        args.replay_path = None
    if args.coreml_path:
        from pathlib import Path
        args.coreml_path = Path(args.coreml_path)
        args.coreml_path.parent.mkdir(parents=True, exist_ok=True)

    train(args)
