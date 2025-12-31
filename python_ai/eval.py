"""Arena evaluation: pit two checkpoints against each other.

This gives a much better read on playing strength than training losses.

Example:
    python -m python_ai.eval --a checkpoints/policy_value.pt --b checkpoints/older.pt --games 200 --sims 64
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from python_ai.gomoku_env import BOARD_SIZE, GomokuEnv
from python_ai.mcts import MCTS
from python_ai.model import get_device, load_checkpoint


@dataclass
class ArenaResult:
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0

    def total(self) -> int:
        return self.a_wins + self.b_wins + self.draws


class MCTSAgent:
    def __init__(
        self,
        model_path: Path,
        *,
        device: torch.device,
        sims: int,
        c_puct: float,
    ) -> None:
        self.device = device
        self.model = load_checkpoint(str(model_path), map_location=device)
        self.model.to(device)
        self.model.eval()
        self.mcts = MCTS(
            model=self.model,
            device=device,
            c_puct=c_puct,
            dirichlet_alpha=0.3,
            dirichlet_frac=0.0,  # inference
        )
        self.sims = int(sims)

    def choose_action(self, env: GomokuEnv) -> int:
        action, _ = self.mcts.run(env, simulations=self.sims, temperature=1e-3)
        return int(action)


@dataclass
class GameRecord:
    """Full record of a single game for later review."""
    model_a: str
    model_b: str
    a_is_black: bool
    moves: list  # List of {"move_num": int, "row": int, "col": int, "player": int, "agent": str}
    winner: int  # 1 = A wins, -1 = B wins, 0 = draw
    winner_name: str  # "A", "B", or "draw"
    total_moves: int
    board_size: int = BOARD_SIZE

    def to_dict(self) -> Dict:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "a_is_black": self.a_is_black,
            "moves": self.moves,
            "winner": self.winner,
            "winner_name": self.winner_name,
            "total_moves": self.total_moves,
            "board_size": self.board_size,
        }


def _play_game(a: MCTSAgent, b: MCTSAgent, *, a_is_black: bool) -> int:
    """Return winner label: 1 means A wins, -1 means B wins, 0 draw."""
    env = GomokuEnv()

    # Map env.current_player (1 black, -1 white) to agent.
    # a_is_black: A plays as +1.
    while env.winner is None:
        if env.current_player == 1:
            agent = a if a_is_black else b
        else:
            agent = b if a_is_black else a

        action = agent.choose_action(env)
        row, col = divmod(action, BOARD_SIZE)
        ok = env.step(row, col)
        if not ok:
            # Should never happen; treat as loss for the acting side.
            return -1 if agent is a else 1

    if env.winner == 0:
        return 0

    # env.winner is the player id (+1 black / -1 white) who won.
    if (env.winner == 1 and a_is_black) or (env.winner == -1 and not a_is_black):
        return 1
    return -1


def _play_game_with_record(
    a: MCTSAgent,
    b: MCTSAgent,
    *,
    a_is_black: bool,
    model_a_name: str,
    model_b_name: str,
) -> Tuple[int, GameRecord]:
    """Play a game and return (winner_label, GameRecord) with full move history."""
    env = GomokuEnv()
    moves = []
    move_num = 0

    while env.winner is None:
        current_player = env.current_player  # +1 black, -1 white
        if current_player == 1:
            agent = a if a_is_black else b
            agent_name = "A" if a_is_black else "B"
        else:
            agent = b if a_is_black else a
            agent_name = "B" if a_is_black else "A"

        action = agent.choose_action(env)
        row, col = divmod(action, BOARD_SIZE)
        ok = env.step(row, col)

        moves.append({
            "move_num": move_num,
            "row": row,
            "col": col,
            "player": current_player,
            "agent": agent_name,
        })
        move_num += 1

        if not ok:
            winner = -1 if agent is a else 1
            winner_name = "B" if winner == -1 else "A"
            return winner, GameRecord(
                model_a=model_a_name,
                model_b=model_b_name,
                a_is_black=a_is_black,
                moves=moves,
                winner=winner,
                winner_name=winner_name,
                total_moves=len(moves),
            )

    if env.winner == 0:
        winner = 0
        winner_name = "draw"
    elif (env.winner == 1 and a_is_black) or (env.winner == -1 and not a_is_black):
        winner = 1
        winner_name = "A"
    else:
        winner = -1
        winner_name = "B"

    return winner, GameRecord(
        model_a=model_a_name,
        model_b=model_b_name,
        a_is_black=a_is_black,
        moves=moves,
        winner=winner,
        winner_name=winner_name,
        total_moves=len(moves),
    )


def eval_arena(args: argparse.Namespace) -> None:
    if args.seed is not None:
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if args.device:
        device = torch.device(str(args.device))
        use_mps = str(device) == "mps"
    else:
        device_cfg = get_device()
        device = device_cfg.device
        use_mps = device_cfg.use_mps

    a_path = Path(args.a).resolve()
    b_path = Path(args.b).resolve()

    if not a_path.exists():
        raise SystemExit(f"Model A not found: {a_path}")
    if not b_path.exists():
        raise SystemExit(f"Model B not found: {b_path}")

    print(f"Device: {device} (MPS={use_mps})")
    print(f"A: {a_path}")
    print(f"B: {b_path}")
    print(f"Games: {int(args.games)} (alternating colors)")
    print(f"MCTS sims: {int(args.sims)}  c_puct: {float(args.c_puct)}")

    agent_a = MCTSAgent(a_path, device=device, sims=int(args.sims), c_puct=float(args.c_puct))
    agent_b = MCTSAgent(b_path, device=device, sims=int(args.sims), c_puct=float(args.c_puct))

    res = ArenaResult()
    for i in range(int(args.games)):
        a_is_black = (i % 2 == 0)
        winner = _play_game(agent_a, agent_b, a_is_black=a_is_black)
        if winner == 1:
            res.a_wins += 1
        elif winner == -1:
            res.b_wins += 1
        else:
            res.draws += 1

        if (i + 1) % int(args.print_every) == 0:
            total = res.total()
            print(
                f"{total}/{int(args.games)}: "
                f"A {res.a_wins}  B {res.b_wins}  D {res.draws}  "
                f"A_winrate={(res.a_wins / total):.3f}"
            )

    total = res.total()
    print("\nFinal")
    print(f"A wins: {res.a_wins}")
    print(f"B wins: {res.b_wins}")
    print(f"Draws : {res.draws}")
    print(f"A winrate: {(res.a_wins / max(1, total)):.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arena eval: pit two Gomoku checkpoints")
    p.add_argument("--a", required=True, help="Path to model A checkpoint (.pt)")
    p.add_argument("--b", required=True, help="Path to model B checkpoint (.pt)")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="", help="Force device: cpu or mps (default: auto)")
    p.add_argument("--print-every", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.device:
        args.device = None
    eval_arena(args)
