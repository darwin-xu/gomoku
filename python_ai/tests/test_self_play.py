"""Tests for self-play generation."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from python_ai.gomoku_env import BOARD_SIZE
from python_ai.model import PolicyValueNet
from python_ai.self_play import (
    Sample,
    apply_symmetry,
    play_self_game_mcts,
    play_self_game_mcts_with_trace,
)


class TestSample:
    """Sample dataclass tests."""

    def test_sample_fields(self) -> None:
        """Sample has expected fields."""
        state = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        pi = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        sample = Sample(state=state, pi=pi, z=0.5, player_to_move=1)

        assert sample.state.shape == (3, BOARD_SIZE, BOARD_SIZE)
        assert sample.pi.shape == (BOARD_SIZE * BOARD_SIZE,)
        assert sample.z == 0.5
        assert sample.player_to_move == 1


class TestSymmetry:
    """Tests for dihedral symmetry transformations."""

    def test_symmetry_identity(self) -> None:
        """Symmetry 0 is identity."""
        state = torch.randn(3, BOARD_SIZE, BOARD_SIZE)
        pi = torch.randn(BOARD_SIZE * BOARD_SIZE)

        state_t, pi_t = apply_symmetry(state, pi, sym=0)

        assert torch.equal(state_t, state)
        assert torch.equal(pi_t, pi)

    def test_symmetry_rotation_90(self) -> None:
        """Symmetry 1 is 90° rotation."""
        state = torch.zeros(3, BOARD_SIZE, BOARD_SIZE)
        state[0, 0, 0] = 1.0  # mark corner

        pi = torch.zeros(BOARD_SIZE * BOARD_SIZE)
        pi[0] = 1.0  # top-left

        state_t, pi_t = apply_symmetry(state, pi, sym=1)

        # After 90° CCW rotation, top-left goes to bottom-left
        assert state_t[0, BOARD_SIZE - 1, 0] == 1.0
        assert pi_t[(BOARD_SIZE - 1) * BOARD_SIZE + 0] == 1.0

    def test_symmetry_rotation_180(self) -> None:
        """Symmetry 2 is 180° rotation."""
        state = torch.zeros(3, BOARD_SIZE, BOARD_SIZE)
        state[0, 0, 0] = 1.0

        pi = torch.zeros(BOARD_SIZE * BOARD_SIZE)
        pi[0] = 1.0

        state_t, pi_t = apply_symmetry(state, pi, sym=2)

        # 180° rotation: top-left goes to bottom-right
        assert state_t[0, BOARD_SIZE - 1, BOARD_SIZE - 1] == 1.0
        assert pi_t[BOARD_SIZE * BOARD_SIZE - 1] == 1.0

    def test_symmetry_flip(self) -> None:
        """Symmetries 4-7 include horizontal flip."""
        state = torch.zeros(3, BOARD_SIZE, BOARD_SIZE)
        state[0, 0, 0] = 1.0

        pi = torch.zeros(BOARD_SIZE * BOARD_SIZE)
        pi[0] = 1.0

        state_t, pi_t = apply_symmetry(state, pi, sym=4)

        # sym=4: identity rotation + flip → top-left goes to top-right
        assert state_t[0, 0, BOARD_SIZE - 1] == 1.0
        assert pi_t[BOARD_SIZE - 1] == 1.0

    def test_symmetry_preserves_sum(self) -> None:
        """All symmetries preserve tensor sums."""
        state = torch.randn(3, BOARD_SIZE, BOARD_SIZE)
        pi = torch.randn(BOARD_SIZE * BOARD_SIZE)
        pi = torch.softmax(pi, dim=0)  # make it a distribution

        for sym in range(8):
            state_t, pi_t = apply_symmetry(state, pi, sym)
            assert torch.isclose(state_t.sum(), state.sum(), atol=1e-5)
            assert torch.isclose(pi_t.sum(), pi.sum(), atol=1e-5)

    def test_all_eight_symmetries_distinct(self) -> None:
        """All 8 symmetries produce different results for asymmetric input."""
        # Create asymmetric state
        state = torch.zeros(3, BOARD_SIZE, BOARD_SIZE)
        state[0, 0, 1] = 1.0  # asymmetric position
        state[0, 2, 3] = 2.0

        pi = torch.zeros(BOARD_SIZE * BOARD_SIZE)
        pi[1] = 1.0
        pi[BOARD_SIZE * 2 + 3] = 0.5

        results = []
        for sym in range(8):
            state_t, pi_t = apply_symmetry(state, pi, sym)
            results.append((state_t.numpy().tobytes(), pi_t.numpy().tobytes()))

        # All 8 should be different
        assert len(set(results)) == 8


class TestSelfPlay:
    """Tests for self-play game generation."""

    def test_play_game_produces_samples(
        self, small_model: PolicyValueNet, device: torch.device, seed: int
    ) -> None:
        """Self-play generates samples."""
        samples = play_self_game_mcts(
            model=small_model,
            device=device,
            simulations=10,
            temperature=1.0,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_frac=0.25,
        )

        assert len(samples) > 0
        assert len(samples) <= BOARD_SIZE * BOARD_SIZE  # max possible moves

    def test_sample_state_shapes(
        self, small_model: PolicyValueNet, device: torch.device, seed: int
    ) -> None:
        """Samples have correct state shapes."""
        samples = play_self_game_mcts(
            model=small_model,
            device=device,
            simulations=5,
            temperature=1.0,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_frac=0.25,
        )

        for s in samples:
            assert s.state.shape == (3, BOARD_SIZE, BOARD_SIZE)
            assert s.state.dtype == np.float32

    def test_sample_policy_valid_distribution(
        self, small_model: PolicyValueNet, device: torch.device, seed: int
    ) -> None:
        """Sample policies are valid probability distributions."""
        samples = play_self_game_mcts(
            model=small_model,
            device=device,
            simulations=10,
            temperature=1.0,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_frac=0.25,
        )

        for s in samples:
            assert s.pi.shape == (BOARD_SIZE * BOARD_SIZE,)
            assert (s.pi >= 0).all()
            assert np.isclose(s.pi.sum(), 1.0, atol=1e-5)

    def test_sample_values_in_range(
        self, small_model: PolicyValueNet, device: torch.device, seed: int
    ) -> None:
        """Sample z values are -1, 0, or 1."""
        samples = play_self_game_mcts(
            model=small_model,
            device=device,
            simulations=5,
            temperature=1.0,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_frac=0.25,
        )

        for s in samples:
            assert s.z in (-1.0, 0.0, 1.0)

    def test_sample_player_alternates(
        self, small_model: PolicyValueNet, device: torch.device, seed: int
    ) -> None:
        """Samples alternate between players."""
        samples = play_self_game_mcts(
            model=small_model,
            device=device,
            simulations=5,
            temperature=1.0,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_frac=0.25,
        )

        if len(samples) >= 2:
            for i in range(len(samples) - 1):
                assert samples[i].player_to_move != samples[i + 1].player_to_move

    def test_winner_perspective_correct(
        self, small_model: PolicyValueNet, device: torch.device, seed: int
    ) -> None:
        """Winner gets z=1, loser gets z=-1."""
        samples = play_self_game_mcts(
            model=small_model,
            device=device,
            simulations=5,
            temperature=1.0,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_frac=0.25,
        )

        # Count z values
        z_values = [s.z for s in samples]

        # If game has a winner (not draw), we should see both +1 and -1
        if 1.0 in z_values or -1.0 in z_values:
            assert 1.0 in z_values
            assert -1.0 in z_values


class TestSelfPlayWithTrace:
    """Tests for self-play with trace collection."""

    def test_returns_trace(
        self, small_model: PolicyValueNet, device: torch.device, seed: int
    ) -> None:
        """With trace enabled, returns game trace."""
        samples, trace = play_self_game_mcts_with_trace(
            model=small_model,
            device=device,
            simulations=5,
            temperature=1.0,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_frac=0.25,
        )

        assert trace is not None
        assert "board_size" in trace
        assert "winner" in trace
        assert "moves" in trace

    def test_trace_board_size(
        self, small_model: PolicyValueNet, device: torch.device, seed: int
    ) -> None:
        """Trace has correct board size."""
        _, trace = play_self_game_mcts_with_trace(
            model=small_model,
            device=device,
            simulations=5,
            temperature=1.0,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_frac=0.25,
        )

        assert trace["board_size"] == BOARD_SIZE

    def test_trace_moves_match_samples(
        self, small_model: PolicyValueNet, device: torch.device, seed: int
    ) -> None:
        """Trace has same number of moves as samples."""
        samples, trace = play_self_game_mcts_with_trace(
            model=small_model,
            device=device,
            simulations=5,
            temperature=1.0,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_frac=0.25,
        )

        assert len(trace["moves"]) == len(samples)

    def test_trace_move_format(
        self, small_model: PolicyValueNet, device: torch.device, seed: int
    ) -> None:
        """Trace moves have row, col, player."""
        _, trace = play_self_game_mcts_with_trace(
            model=small_model,
            device=device,
            simulations=5,
            temperature=1.0,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_frac=0.25,
        )

        for move in trace["moves"]:
            assert "row" in move
            assert "col" in move
            assert "player" in move
            assert 0 <= move["row"] < BOARD_SIZE
            assert 0 <= move["col"] < BOARD_SIZE
            assert move["player"] in (1, -1)

    def test_trace_winner_matches_outcome(
        self, small_model: PolicyValueNet, device: torch.device, seed: int
    ) -> None:
        """Trace winner matches sample z values."""
        samples, trace = play_self_game_mcts_with_trace(
            model=small_model,
            device=device,
            simulations=5,
            temperature=1.0,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_frac=0.25,
        )

        winner = trace["winner"]

        if winner == 0:
            # Draw: all z should be 0
            assert all(s.z == 0.0 for s in samples)
        else:
            # Winner exists
            for s in samples:
                if s.player_to_move == winner:
                    assert s.z == 1.0
                else:
                    assert s.z == -1.0
