#!/usr/bin/env python3

# By Darwin

import tkinter as tk
from tkinter import messagebox
import random
from collections import deque


class Gomoku:
    def __init__(self, size=15, win_length=5):
        self.size = size
        self.win_length = win_length
        self.board = [["." for _ in range(size)] for _ in range(size)]
        self.current_player = "X"
        self.game_over = False

    def make_move(self, x, y):
        if not self.game_over and self.board[x][y] == ".":
            self.board[x][y] = self.current_player
            if self.check_win(x, y):
                self.game_over = True
            self.current_player = "O" if self.current_player == "X" else "X"
            return True
        return False

    def check_win(self, x, y):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for i in range(1, self.win_length):
                nx, ny = x + i * dx, y + i * dy
                if (
                    0 <= nx < self.size
                    and 0 <= ny < self.size
                    and self.board[nx][ny] == self.current_player
                ):
                    count += 1
                else:
                    break
            for i in range(1, self.win_length):
                nx, ny = x - i * dx, y - i * dy
                if (
                    0 <= nx < self.size
                    and 0 <= ny < self.size
                    and self.board[nx][ny] == self.current_player
                ):
                    count += 1
                else:
                    break
            if count >= self.win_length:
                return True
        return False

    def getBoard(self):
        return self.board


class GomokuGUI:
    def __init__(self, root, game, player):
        self.root = root
        self.game = game
        self.player = player
        self.canvas_size = 600
        self.square_size = self.canvas_size // self.game.size
        self.canvas = tk.Canvas(
            self.root, width=self.canvas_size, height=self.canvas_size
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.draw_board()

    def draw_board(self):
        for i in range(self.game.size):
            self.canvas.create_line(
                (i + 0.5) * self.square_size,
                0.5 * self.square_size,
                (i + 0.5) * self.square_size,
                (self.game.size - 0.5) * self.square_size,
            )
            self.canvas.create_line(
                0.5 * self.square_size,
                (i + 0.5) * self.square_size,
                (self.game.size - 0.5) * self.square_size,
                (i + 0.5) * self.square_size,
            )

    def on_click(self, event):
        if not self.game.game_over:
            x, y = int(event.x / self.square_size), int(event.y / self.square_size)
            if self.game.make_move(x, y):
                self.draw_piece(x, y)
                if self.game.game_over:
                    winner = self.game.current_player
                    tk.messagebox.showinfo("Game Over", f"Player {winner} wins!")
                else:
                    x, y = self.player.nextMove(self.game.getBoard())
                    if self.game.make_move(x, y):
                        self.draw_piece(x, y)
                        if self.game.game_over:
                            winner = self.game.current_player
                            tk.messagebox.showinfo(
                                "Game Over", f"Player {winner} wins!"
                            )
                    else:
                        # Shouldn't happen, throw an error
                        raise ValueError("Invalid move")

    def draw_piece(self, x, y):
        color = "black" if self.game.current_player == "X" else "white"
        center_x = (x + 0.5) * self.square_size
        center_y = (y + 0.5) * self.square_size
        radius = self.square_size // 3
        self.canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            fill=color,
        )


class DummyPlayer:
    def __init__(self):
        self.piece = "O"
        self.directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
        self.scoreMap = {1: 10, 2: 100, 3: 1000, 4: 10000, 5: 100000}

    def addScore(self, board, scoreBoard, x, y, score):
        if 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] == ".":
            scoreBoard[x][y] += score
        else:
            print("Invalid position: ", x, y, score)
        return scoreBoard

    def scorePattern(self, pattern, scoreBoard):
        pieces = [i[0] for i in pattern]
        count = 0
        cur = "."
        for i in range(len(pieces)):
            if pieces[i] == ".":
                continue
            elif cur == "." or cur == pieces[i]:
                cur = pieces[i]
                count += 1
            else:
                count = -1
                break

        if count > 0:
            for p, x, y in pattern:
                if p == ".":
                    scoreBoard[x][y] += self.scoreMap[count]

    def traverseHorizontal(self, board, scoreBoard):
        for i in range(len(board)):
            pattern = deque()
            for j in range(len(board[0])):
                # Get the next five pieces into pattern
                pattern.append((board[i][j], i, j))
                if len(pattern) > 5:
                    pattern.popleft()
                if len(pattern) == 5:
                    self.scorePattern(pattern, scoreBoard)

    def traverseVertical(self, board, scoreBoard):
        for j in range(len(board[0])):
            pattern = deque()
            for i in range(len(board)):
                pattern.append((board[i][j], i, j))
                if len(pattern) > 5:
                    pattern.popleft()
                if len(pattern) == 5:
                    self.scorePattern(pattern, scoreBoard)

    def traverseDiagonal(self, board, scoreBoard):
        rows = len(board)
        cols = len(board[0])
        for d in range(rows + cols - 1):
            pattern = deque()
            i = 0 if d < cols else d - cols + 1
            j = d if d < cols else cols - 1
            while i < rows and j >= 0:
                pattern.append((board[i][j], i, j))
                if len(pattern) > 5:
                    pattern.popleft()
                if len(pattern) == 5:
                    self.scorePattern(pattern, scoreBoard)
                i += 1
                j -= 1

    def traverseAntiDiagonal(self, board, scoreBoard):
        rows = len(board)
        cols = len(board[0])
        for d in range(rows + cols - 1):
            pattern = deque()
            i = rows - d - 1 if d < rows else 0
            j = 0 if d < rows else d - rows + 1
            while i < rows and j < cols:
                pattern.append((board[i][j], i, j))
                if len(pattern) > 5:
                    pattern.popleft()
                if len(pattern) == 5:
                    self.scorePattern(pattern, scoreBoard)
                i += 1
                j += 1

    def findHighestScore(self, scoreBoard):
        maxScore = -1
        x, y = 0, 0
        for i in range(len(scoreBoard)):
            for j in range(len(scoreBoard[0])):
                if scoreBoard[i][j] > maxScore:
                    maxScore = scoreBoard[i][j]
                    x, y = i, j
        return x, y

    def nextMove(self, board):
        scoreBoard = [[0 for _ in range(len(board[0]))] for _ in range(len(board))]
        self.traverseHorizontal(board, scoreBoard)
        self.traverseVertical(board, scoreBoard)
        self.traverseDiagonal(board, scoreBoard)
        self.traverseAntiDiagonal(board, scoreBoard)

        print("Score Board:")
        for i in range(len(scoreBoard)):
            for j in range(len(scoreBoard[0])):
                print("%3.0f " % scoreBoard[j][i], end=" ")
            print()
        print

        return self.findHighestScore(scoreBoard)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Gomoku")
    game = Gomoku()
    player = DummyPlayer()
    gui = GomokuGUI(root, game, player)
    root.mainloop()
