"""Tests for GomokuEnv."""
from __future__ import annotations

import numpy as np
import pytest

from python_ai.gomoku_env import GomokuEnv, BOARD_SIZE, WIN_LENGTH


class TestBoardBasics:
    """Basic board operations."""

    def test_initial_state(self, env: GomokuEnv) -> None:
        """Fresh board is empty, black to move, no winner."""
        assert env.current_player == 1  # black first
        assert env.winner is None
        assert len(env.move_history) == 0
        assert (env.board == 0).all()

    def test_board_size(self, env: GomokuEnv) -> None:
        """Board is 15x15."""
        assert env.board.shape == (BOARD_SIZE, BOARD_SIZE)
        assert BOARD_SIZE == 15

    def test_valid_moves_initial(self, env: GomokuEnv) -> None:
        """All 225 positions are valid initially."""
        moves = env.valid_moves()
        assert len(moves) == BOARD_SIZE * BOARD_SIZE

    def test_clone_independence(self, env: GomokuEnv) -> None:
        """Cloned env is independent of original."""
        env.step(7, 7)
        clone = env.clone()
        clone.step(7, 8)

        assert env.board[7, 8] == 0
        assert clone.board[7, 8] != 0
        assert len(env.move_history) == 1
        assert len(clone.move_history) == 2


class TestMoves:
    """Move execution and validation."""

    def test_step_places_stone(self, env: GomokuEnv) -> None:
        """Step places current player's stone."""
        assert env.step(7, 7) is True
        assert env.board[7, 7] == 1  # black

    def test_step_alternates_players(self, env: GomokuEnv) -> None:
        """Players alternate after each move."""
        assert env.current_player == 1
        env.step(7, 7)
        assert env.current_player == -1
        env.step(7, 8)
        assert env.current_player == 1

    def test_step_invalid_occupied(self, env: GomokuEnv) -> None:
        """Cannot place on occupied cell."""
        env.step(7, 7)
        assert env.step(7, 7) is False
        assert env.current_player == -1  # still white's turn

    def test_step_invalid_out_of_bounds(self, env: GomokuEnv) -> None:
        """Cannot place outside board."""
        assert env.step(-1, 0) is False
        assert env.step(0, BOARD_SIZE) is False
        assert env.step(BOARD_SIZE, 0) is False

    def test_move_history_tracking(self, env: GomokuEnv) -> None:
        """Move history records all moves."""
        env.step(0, 0)
        env.step(1, 1)
        env.step(2, 2)

        assert len(env.move_history) == 3
        assert env.move_history[0].row == 0
        assert env.move_history[0].col == 0
        assert env.move_history[0].player == 1
        assert env.move_history[1].player == -1
        assert env.move_history[2].player == 1

    def test_valid_moves_decreases(self, env: GomokuEnv) -> None:
        """Valid moves decrease after each move."""
        initial = len(env.valid_moves())
        env.step(7, 7)
        assert len(env.valid_moves()) == initial - 1


class TestWinDetection:
    """Win condition detection."""

    def test_horizontal_win(self, env: GomokuEnv) -> None:
        """Detect horizontal 5-in-a-row."""
        # Black: 0,0  0,1  0,2  0,3  0,4
        # White: 1,0  1,1  1,2  1,3
        for i in range(4):
            env.step(0, i)  # black
            env.step(1, i)  # white
        env.step(0, 4)  # black wins

        assert env.winner == 1

    def test_vertical_win(self, env: GomokuEnv) -> None:
        """Detect vertical 5-in-a-row."""
        for i in range(4):
            env.step(i, 0)  # black
            env.step(i, 1)  # white
        env.step(4, 0)  # black wins

        assert env.winner == 1

    def test_diagonal_win(self, env: GomokuEnv) -> None:
        """Detect diagonal 5-in-a-row."""
        # Black diagonal: (0,0) (1,1) (2,2) (3,3) (4,4)
        for i in range(4):
            env.step(i, i)  # black
            env.step(i, i + 5)  # white (off diagonal)
        env.step(4, 4)  # black wins

        assert env.winner == 1

    def test_anti_diagonal_win(self, env: GomokuEnv) -> None:
        """Detect anti-diagonal 5-in-a-row."""
        # Black: (0,4) (1,3) (2,2) (3,1) (4,0)
        positions = [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
        for i in range(4):
            env.step(*positions[i])  # black
            env.step(10, i)  # white (elsewhere)
        env.step(*positions[4])  # black wins

        assert env.winner == 1

    def test_white_wins(self, env: GomokuEnv) -> None:
        """White can also win."""
        # Black: 0,0  0,1  0,2  0,3  (dummy) 5,5
        # White: 1,0  1,1  1,2  1,3  1,4
        for i in range(4):
            env.step(0, i)  # black
            env.step(1, i)  # white
        env.step(5, 5)  # black elsewhere
        env.step(1, 4)  # white wins

        assert env.winner == -1

    def test_no_winner_four_in_row(self, env: GomokuEnv) -> None:
        """Four in a row is not a win."""
        for i in range(4):
            env.step(0, i)
            if i < 3:
                env.step(1, i)

        assert env.winner is None

    def test_cannot_move_after_win(self, env: GomokuEnv) -> None:
        """No moves allowed after game ends."""
        for i in range(4):
            env.step(0, i)
            env.step(1, i)
        env.step(0, 4)  # black wins

        assert env.step(2, 0) is False


class TestDraw:
    """Draw detection."""

    def test_is_full(self, env: GomokuEnv) -> None:
        """Board full detection."""
        assert not env.is_full()

        # Fill board without winning (checkerboard-ish pattern)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if env.board[r, c] == 0 and env.winner is None:
                    env.step(r, c)

        # Board should be full or game ended
        assert env.is_full() or env.winner is not None


class TestEncoding:
    """State encoding for neural network."""

    def test_encode_shape(self, env: GomokuEnv) -> None:
        """Encoding produces (3, 15, 15) tensor."""
        encoded = env.encode(perspective=1)
        assert encoded.shape == (3, BOARD_SIZE, BOARD_SIZE)
        assert encoded.dtype == np.float32

    def test_encode_empty_board(self, env: GomokuEnv) -> None:
        """Empty board encodes correctly."""
        encoded = env.encode(perspective=1)

        # Channel 0: current player stones (none)
        assert encoded[0].sum() == 0
        # Channel 1: opponent stones (none)
        assert encoded[1].sum() == 0
        # Channel 2: color indicator (all 1s for black perspective)
        assert (encoded[2] == 1.0).all()

    def test_encode_with_stones(self, env: GomokuEnv) -> None:
        """Encoding reflects placed stones."""
        env.step(7, 7)  # black at center
        env.step(0, 0)  # white at corner

        # From black's perspective
        enc_black = env.encode(perspective=1)
        assert enc_black[0, 7, 7] == 1.0  # black stone in my-stones channel
        assert enc_black[1, 0, 0] == 1.0  # white stone in opponent channel

        # From white's perspective
        enc_white = env.encode(perspective=-1)
        assert enc_white[1, 7, 7] == 1.0  # black stone is now opponent
        assert enc_white[0, 0, 0] == 1.0  # white stone is now mine
        assert (enc_white[2] == 0.0).all()  # color indicator for white

    def test_encode_perspective_consistency(self, env: GomokuEnv) -> None:
        """Encoding is consistent with current player perspective."""
        env.step(7, 7)

        # Current player is now white
        enc = env.encode(env.current_player)
        assert enc[1, 7, 7] == 1.0  # black stone is opponent from white's view


class TestHeuristicMoves:
    """Heuristic move generation."""

    def test_heuristic_empty_board(self, env: GomokuEnv) -> None:
        """All moves valid on empty board."""
        moves = env.valid_moves_heuristic(distance=2)
        assert len(moves) == BOARD_SIZE * BOARD_SIZE

    def test_heuristic_restricts_distance(self, env: GomokuEnv) -> None:
        """Heuristic limits moves to near existing stones."""
        env.step(7, 7)

        moves = env.valid_moves_heuristic(distance=1)
        # Should only include 8 neighbors of (7,7) that are empty
        assert len(moves) <= 8
        for r, c in moves:
            assert abs(r - 7) <= 1 and abs(c - 7) <= 1
            assert (r, c) != (7, 7)

    def test_heuristic_distance_2(self, env: GomokuEnv) -> None:
        """Distance 2 includes larger neighborhood."""
        env.step(7, 7)

        moves = env.valid_moves_heuristic(distance=2)
        # 5x5 area minus center = 24 moves
        assert len(moves) == 24


class TestReset:
    """Board reset functionality."""

    def test_reset_clears_board(self, env: GomokuEnv) -> None:
        """Reset returns board to initial state."""
        env.step(7, 7)
        env.step(7, 8)
        env.reset()

        assert (env.board == 0).all()
        assert env.current_player == 1
        assert env.winner is None
        assert len(env.move_history) == 0
