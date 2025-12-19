"""Lightweight Gomoku environment for self-play training."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

BOARD_SIZE = 15
WIN_LENGTH = 5


@dataclass
class Move:
    row: int
    col: int
    player: int  # 1 for current player at move time, -1 for opponent perspective


class GomokuEnv:
    def __init__(self) -> None:
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = 1  # 1 for black, -1 for white
        self.move_history: List[Move] = []
        self.winner: Optional[int] = None

    def reset(self) -> None:
        self.board.fill(0)
        self.current_player = 1
        self.move_history.clear()
        self.winner = None

    def clone(self) -> "GomokuEnv":
        clone = GomokuEnv()
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        clone.move_history = list(self.move_history)
        clone.winner = self.winner
        return clone

    def valid_moves(self) -> List[Tuple[int, int]]:
        rows, cols = np.where(self.board == 0)
        return list(zip(rows.tolist(), cols.tolist()))

    def valid_moves_heuristic(self, distance: int = 2) -> List[Tuple[int, int]]:
        """Return valid moves that are within 'distance' of any existing stone."""
        occupied = self.board != 0
        if not np.any(occupied):
            return self.valid_moves()

        rows, cols = np.where(occupied)
        
        mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
        
        # Vectorized bounding box calculation
        r_min = np.maximum(0, rows - distance)
        r_max = np.minimum(BOARD_SIZE, rows + distance + 1)
        c_min = np.maximum(0, cols - distance)
        c_max = np.minimum(BOARD_SIZE, cols + distance + 1)
        
        for i in range(len(rows)):
            mask[r_min[i]:r_max[i], c_min[i]:c_max[i]] = True
            
        mask[occupied] = False
        
        valid_rows, valid_cols = np.where(mask)
        return list(zip(valid_rows.tolist(), valid_cols.tolist()))

    def is_full(self) -> bool:
        return not (self.board == 0).any()

    def step(self, row: int, col: int) -> bool:
        if self.winner is not None:
            return False
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return False
        if self.board[row, col] != 0:
            return False
        self.board[row, col] = self.current_player
        self.move_history.append(Move(row=row, col=col, player=self.current_player))
        if self._check_win(row, col, self.current_player):
            self.winner = self.current_player
            # Keep semantics consistent: current_player always means "side to move".
            self.current_player *= -1
        elif self.is_full():
            self.winner = 0  # draw
            self.current_player *= -1
        else:
            self.current_player *= -1
        return True

    def _check_win(self, row: int, col: int, player: int) -> bool:
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            if self._count_in_direction(row, col, player, dr, dc) + self._count_in_direction(row, col, player, -dr, -dc) + 1 >= WIN_LENGTH:
                return True
        return False

    def _count_in_direction(self, row: int, col: int, player: int, dr: int, dc: int) -> int:
        r, c = row + dr, col + dc
        count = 0
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r, c] == player:
            count += 1
            r += dr
            c += dc
        return count

    def encode(self, perspective: int) -> np.ndarray:
        """Encode board into 3 channels: current, opponent, bias."""
        current = (self.board == perspective).astype(np.float32)
        opponent = (self.board == -perspective).astype(np.float32)
        bias = np.full_like(current, fill_value=1.0 if perspective == 1 else 0.0)
        return np.stack([current, opponent, bias], axis=0)

    def last_player(self) -> int:
        if not self.move_history:
            return -1
        return -self.current_player
