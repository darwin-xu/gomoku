"""AlphaZero-style self-play generation (MCTS -> policy targets)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from python_ai.gomoku_env import GomokuEnv, BOARD_SIZE
from python_ai.mcts import MCTS
from python_ai.model import PolicyValueNet


@dataclass
class Sample:
    state: np.ndarray        # (3, 15, 15) float32
    pi: np.ndarray           # (225,) float32
    z: float                 # value target
    player_to_move: int      # 1 or -1


def apply_symmetry(state: torch.Tensor, pi: torch.Tensor, sym: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply one of the 8 dihedral symmetries to (state, pi)."""
    k = sym % 4
    flip = sym >= 4

    state_t = torch.rot90(state, k=k, dims=(1, 2))
    pi_grid = torch.rot90(pi.view(BOARD_SIZE, BOARD_SIZE), k=k, dims=(0, 1))

    if flip:
        state_t = torch.flip(state_t, dims=(2,))
        pi_grid = torch.flip(pi_grid, dims=(1,))

    return state_t, pi_grid.reshape(-1)


def play_self_game_mcts(
    model: PolicyValueNet,
    device: torch.device,
    simulations: int,
    temperature: float,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_frac: float,
    restrict_move_distance: int | None = 2,
) -> List[Sample]:
    env = GomokuEnv()
    mcts = MCTS(
        model=model,
        device=device,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_frac=dirichlet_frac,
        restrict_move_distance=restrict_move_distance,
    )

    samples: List[Sample] = []
    while env.winner is None:
        player_to_move = env.current_player
        action, visits = mcts.run(env, simulations=simulations, temperature=temperature)
        pi = visits / (visits.sum() + 1e-8)

        samples.append(Sample(
            state=env.encode(player_to_move).astype(np.float32),
            pi=pi.astype(np.float32),
            z=0.0,
            player_to_move=player_to_move,
        ))

        row, col = divmod(action, BOARD_SIZE)
        env.step(row, col)

    for i, s in enumerate(samples):
        if env.winner == 0:
            z = 0.0
        else:
            z = 1.0 if env.winner == s.player_to_move else -1.0
        samples[i] = Sample(state=s.state, pi=s.pi, z=z, player_to_move=s.player_to_move)
    return samples
