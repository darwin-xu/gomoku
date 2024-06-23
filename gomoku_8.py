#!/usr/bin/env python3

# By Darwin

import tkinter as tk
from tkinter import messagebox
import random


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
        self.directions = [
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
            (0, -1),
            (-1, 0),
        ]
        self.directions1 = [(1, 0), (0, 1), (1, 1), (-1, 1)]
        self.scoreMap = {1: 10, 2: 100, 3: 1000, 4: 10000, 5: 100000}

    def scoreForOnePiece(self, board, scoreBoard):
        for x in range(len(board)):
            for y in range(len(board[0])):
                if board[x][y] == ".":
                    pass
                elif board[x][y] == self.piece:
                    # check in all directions
                    for i in range(len(self.directions)):
                        dx, dy = self.directions[i]
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < len(board) and 0 <= ny < len(board[0]):
                            if board[nx][ny] == ".":
                                scoreBoard[nx][ny] += 10
                else:
                    # check in all directions
                    for i in range(len(self.directions)):
                        dx, dy = self.directions[i]
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < len(board) and 0 <= ny < len(board[0]):
                            if board[nx][ny] == ".":
                                scoreBoard[nx][ny] += 8
        return scoreBoard

    def addScore(self, board, scoreBoard, x, y, score):
        if 0 <= x < len(board) and 0 <= y < len(board[0]) and board[x][y] == ".":
            scoreBoard[x][y] += score
        else:
            print("Invalid position: ", x, y, score)
        return scoreBoard

    def scoreForPieces(self, board, scoreBoard):
        for x in range(len(board)):
            for y in range(len(board[0])):
                if board[x][y] == ".":
                    pass
                else:
                    cur = board[x][y]
                    # check 4 directions
                    for i in range(len(self.directions1)):
                        nx, ny = x, y
                        dx, dy = self.directions1[i]
                        count = 1
                        for _ in range(5):
                            nx, ny = nx + dx, ny + dy
                            if 0 <= nx < len(board) and 0 <= ny < len(board[0]):
                                if board[nx][ny] == cur:
                                    count += 1
                                else:
                                    break
                            else:
                                break
                        print("log: ", x, y, dx, dy, count)
                        self.addScore(
                            board, scoreBoard, x - dx, y - dy, self.scoreMap[count]
                        )
                        self.addScore(board, scoreBoard, nx, ny, self.scoreMap[count])

        return scoreBoard

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
        scoreBoard = self.scoreForPieces(board, scoreBoard)

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
