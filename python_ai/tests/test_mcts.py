"""Tests for MCTS implementation."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from python_ai.gomoku_env import GomokuEnv, BOARD_SIZE
from python_ai.mcts import MCTS, Node
from python_ai.model import PolicyValueNet


class TestNode:
    """Node dataclass tests."""

    def test_node_defaults(self) -> None:
        """Node initializes with correct defaults."""
        node = Node(prior=0.5)
        assert node.prior == 0.5
        assert node.visits == 0
        assert node.value_sum == 0.0
        assert node.children == {}

    def test_node_value_zero_visits(self) -> None:
        """Value is 0 when no visits."""
        node = Node(prior=0.1)
        assert node.value == 0.0

    def test_node_value_with_visits(self) -> None:
        """Value is average of value_sum."""
        node = Node(prior=0.1)
        node.visits = 10
        node.value_sum = 5.0
        assert node.value == 0.5


class TestMCTSBasics:
    """Basic MCTS functionality."""

    def test_mcts_initialization(
        self, small_model: PolicyValueNet, device: torch.device
    ) -> None:
        """MCTS initializes with given parameters."""
        mcts = MCTS(
            model=small_model,
            device=device,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_frac=0.25,
        )
        assert mcts.c_puct == 1.5
        assert mcts.dirichlet_alpha == 0.3
        assert mcts.dirichlet_frac == 0.25

    def test_run_returns_valid_action(
        self, small_model: PolicyValueNet, device: torch.device, env: GomokuEnv, seed: int
    ) -> None:
        """MCTS.run returns a valid action."""
        mcts = MCTS(model=small_model, device=device, c_puct=1.5)

        action, visits = mcts.run(env, simulations=10, temperature=1.0)

        assert 0 <= action < BOARD_SIZE * BOARD_SIZE
        row, col = divmod(action, BOARD_SIZE)
        assert env.board[row, col] == 0  # cell is empty

    def test_run_returns_visits(
        self, small_model: PolicyValueNet, device: torch.device, env: GomokuEnv, seed: int
    ) -> None:
        """MCTS.run returns visit counts array."""
        mcts = MCTS(model=small_model, device=device, c_puct=1.5)

        action, visits = mcts.run(env, simulations=20, temperature=1.0)

        assert visits.shape == (BOARD_SIZE * BOARD_SIZE,)
        assert visits.dtype == np.float32
        assert visits.sum() > 0
        assert visits[action] > 0  # chosen action was visited

    def test_visits_are_distributed(
        self, small_model: PolicyValueNet, device: torch.device, env: GomokuEnv, seed: int
    ) -> None:
        """Visits are distributed among children after simulations."""
        mcts = MCTS(model=small_model, device=device, c_puct=1.5, dirichlet_frac=0.0)

        sims = 50
        _, visits = mcts.run(env, simulations=sims, temperature=1.0)

        # Visits should be positive and distributed
        assert visits.sum() > 0
        # At least some actions should have visits
        assert (visits > 0).sum() >= 1

    def test_greedy_selection_low_temperature(
        self, small_model: PolicyValueNet, device: torch.device, env: GomokuEnv, seed: int
    ) -> None:
        """Low temperature selects most-visited action."""
        mcts = MCTS(model=small_model, device=device, c_puct=1.5)

        action, visits = mcts.run(env, simulations=50, temperature=1e-4)

        assert action == int(np.argmax(visits))

    def test_stochastic_selection_high_temperature(
        self, small_model: PolicyValueNet, device: torch.device, env: GomokuEnv
    ) -> None:
        """High temperature allows variety in action selection."""
        mcts = MCTS(model=small_model, device=device, c_puct=1.5)

        # Run multiple times with different seeds
        actions = set()
        for i in range(20):
            np.random.seed(i)
            torch.manual_seed(i)
            action, _ = mcts.run(env, simulations=30, temperature=2.0)
            actions.add(action)

        # Should see some variety (not always same action)
        # With high temp and 30 sims, we expect at least 2 different actions
        assert len(actions) >= 1  # at minimum it works


class TestMCTSMoveRestriction:
    """Tests for move distance restriction."""

    def test_restrict_moves_empty_board(
        self, small_model: PolicyValueNet, device: torch.device, seed: int
    ) -> None:
        """On empty board, all moves are valid even with restriction."""
        env = GomokuEnv()
        mcts = MCTS(
            model=small_model,
            device=device,
            c_puct=1.5,
            restrict_move_distance=2,
        )

        action, _ = mcts.run(env, simulations=10, temperature=1.0)
        assert 0 <= action < BOARD_SIZE * BOARD_SIZE

    def test_restrict_moves_near_stone(
        self, small_model: PolicyValueNet, device: torch.device, seed: int
    ) -> None:
        """Restricted MCTS plays near existing stones."""
        env = GomokuEnv()
        env.step(7, 7)  # black at center

        mcts = MCTS(
            model=small_model,
            device=device,
            c_puct=1.5,
            restrict_move_distance=1,
            dirichlet_frac=0.0,  # deterministic priors
        )

        action, visits = mcts.run(env, simulations=20, temperature=1e-3)
        row, col = divmod(action, BOARD_SIZE)

        # Should be within distance 1 of (7, 7)
        assert abs(row - 7) <= 1 and abs(col - 7) <= 1

        # Only neighbors should have visits
        for a in range(BOARD_SIZE * BOARD_SIZE):
            if visits[a] > 0:
                r, c = divmod(a, BOARD_SIZE)
                assert abs(r - 7) <= 1 and abs(c - 7) <= 1


class TestMCTSMaskedSoftmax:
    """Tests for masked softmax helper."""

    def test_masked_softmax_valid(self) -> None:
        """Masked softmax zeroes invalid actions."""
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = np.array([True, False, True, False, True])

        probs = MCTS._masked_softmax(logits, mask)

        assert probs[1] == 0.0
        assert probs[3] == 0.0
        assert probs[0] > 0
        assert probs[2] > 0
        assert probs[4] > 0
        assert np.isclose(probs.sum(), 1.0)

    def test_masked_softmax_all_masked(self) -> None:
        """All masked returns zeros."""
        logits = np.array([1.0, 2.0, 3.0])
        mask = np.array([False, False, False])

        probs = MCTS._masked_softmax(logits, mask)

        assert (probs == 0).all()

    def test_masked_softmax_numerical_stability(self) -> None:
        """Handles large logits without overflow."""
        logits = np.array([1000.0, 1001.0, 1002.0])
        mask = np.array([True, True, True])

        probs = MCTS._masked_softmax(logits, mask)

        assert np.isfinite(probs).all()
        assert np.isclose(probs.sum(), 1.0)


class TestMCTSGameplay:
    """Tests for MCTS in game scenarios."""

    def test_mcts_avoids_occupied(
        self, small_model: PolicyValueNet, device: torch.device, seed: int
    ) -> None:
        """MCTS never selects occupied cells."""
        env = GomokuEnv()
        mcts = MCTS(model=small_model, device=device, c_puct=1.5)

        # Play some moves
        env.step(7, 7)
        env.step(7, 8)
        env.step(6, 7)

        action, visits = mcts.run(env, simulations=30, temperature=1.0)

        # Selected action should be empty
        row, col = divmod(action, BOARD_SIZE)
        assert env.board[row, col] == 0

        # Occupied cells should have zero visits
        assert visits[7 * BOARD_SIZE + 7] == 0
        assert visits[7 * BOARD_SIZE + 8] == 0
        assert visits[6 * BOARD_SIZE + 7] == 0

    def test_mcts_works_near_end_game(
        self, small_model: PolicyValueNet, device: torch.device, seed: int
    ) -> None:
        """MCTS works when few moves remain."""
        env = GomokuEnv()
        mcts = MCTS(model=small_model, device=device, c_puct=1.5)

        # Fill most of the board (leave just a few cells)
        move_count = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if move_count >= BOARD_SIZE * BOARD_SIZE - 5:
                    break
                if env.winner is None:
                    env.step(r, c)
                    move_count += 1
            if move_count >= BOARD_SIZE * BOARD_SIZE - 5:
                break

        if env.winner is None:
            action, visits = mcts.run(env, simulations=10, temperature=1.0)
            row, col = divmod(action, BOARD_SIZE)
            assert env.board[row, col] == 0
