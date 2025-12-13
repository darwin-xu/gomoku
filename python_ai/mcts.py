"""Lightweight PUCT MCTS for Gomoku."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from python_ai.gomoku_env import GomokuEnv, BOARD_SIZE
from python_ai.model import PolicyValueNet


@dataclass
class Node:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[int, "Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class MCTS:
    def __init__(
        self,
        model: PolicyValueNet,
        device: torch.device,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_frac: float = 0.25,
    ) -> None:
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac

    def run(self, env: GomokuEnv, simulations: int, temperature: float) -> Tuple[int, np.ndarray]:
        root = self._expand_root(env)
        for _ in range(simulations):
            self._simulate(env.clone(), root)

        visits = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visits

        if temperature <= 1e-3:
            action = int(np.argmax(visits))
        else:
            probs = visits ** (1.0 / max(temperature, 1e-3))
            probs = probs / (probs.sum() + 1e-8)
            action = int(np.random.choice(len(probs), p=probs))
        return action, visits

    def _expand_root(self, env: GomokuEnv) -> Node:
        root = Node(prior=1.0)
        logits, value = self._predict(env)

        mask = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=bool)
        for (r, c) in env.valid_moves():
            mask[r * BOARD_SIZE + c] = True

        priors = self._masked_softmax(logits, mask)

        # Dirichlet noise for root exploration
        noise = np.random.dirichlet([self.dirichlet_alpha] * mask.sum())
        idx = 0
        for a in range(BOARD_SIZE * BOARD_SIZE):
            if not mask[a]:
                continue
            mixed = (1 - self.dirichlet_frac) * priors[a] + self.dirichlet_frac * noise[idx]
            root.children[a] = Node(prior=mixed)
            idx += 1
        root.value_sum = value
        root.visits = 1
        return root

    def _simulate(self, env: GomokuEnv, node: Node) -> float:
        if env.winner is not None:
            # Value from the perspective of env.current_player (side to move).
            # With GomokuEnv semantics, if winner != 0 then the previous player won,
            # so current player is the loser.
            if env.winner == 0:
                return 0.0
            return -1.0

        if not node.children:
            logits, value = self._predict(env)
            mask = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=bool)
            for (r, c) in env.valid_moves():
                mask[r * BOARD_SIZE + c] = True
            priors = self._masked_softmax(logits, mask)
            for a in range(BOARD_SIZE * BOARD_SIZE):
                if mask[a]:
                    node.children[a] = Node(prior=priors[a])
            node.visits = 1
            node.value_sum = value
            return value

        best_action, child = self._select_child(node)
        row, col = divmod(best_action, BOARD_SIZE)
        env.step(row, col)
        value = -self._simulate(env, child)  # value is from current player's view, flip sign

        child.value_sum += value
        child.visits += 1
        return value

    def _select_child(self, node: Node) -> Tuple[int, Node]:
        total_visits = math.sqrt(sum(child.visits for child in node.children.values()) + 1e-8)
        best_score = -1e9
        best_action = -1
        best_child: Optional[Node] = None
        for action, child in node.children.items():
            prior = child.prior
            q = child.value
            u = self.c_puct * prior * (total_visits / (1 + child.visits))
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child  # type: ignore

    def _predict(self, env: GomokuEnv) -> Tuple[np.ndarray, float]:
        state = torch.from_numpy(env.encode(env.current_player)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.model(state)
        return logits[0].detach().cpu().numpy(), float(value.item())

    @staticmethod
    def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
        logits = logits.copy()
        logits[~mask] = -1e9
        logits = logits - logits.max()
        exp = np.exp(logits)
        exp[~mask] = 0.0
        z = exp.sum()
        if z <= 0:
            return np.zeros_like(logits)
        return exp / z
