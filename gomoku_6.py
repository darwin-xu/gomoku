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

    # def make_move(self, x, y, p):
    #     if not self.game_over and self.board[x][y] == "." and self.current_player == p:
    #         # switch player
    #         self.current_player = "O" if self.current_player == "X" else "X"
    #         self.board[x][y] = p
    #         if self.check_win(x, y):
    #             self.game_over = True
    #         return True
    #     return False

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
        self.player = player()
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
            #print("x = %d, y = %d" % (x, y))
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
        color = "black" if self.game.current_player == "O" else "white"
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
    def ratePoint(self, x, y, board):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        rate = 1
        for dx, dy in directions:
            for i in range(1, 5):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < 15 and 0 <= ny < 15 and board[nx][ny] == "X":
                    rate += (1+i)
               # if i==3: rate+=2
                else:
                    break
            for i in range(1, 5):
                nx, ny = x - i * dx, y - i * dy
                if 0 <= nx < 15 and 0 <= ny < 15 and board[nx][ny] == "X":
                    rate += (1+i)
                #if i==3: rate+=2
                else:
                    break
        for dx, dy in directions:
            for i in range(1, 5):
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < 15 and 0 <= ny < 15 and board[nx][ny] == "O":
                    rate += (1+i) 
                else:
                    break
            for i in range(1, 5):
                nx, ny = x - i * dx, y - i * dy
                if 0 <= nx < 15 and 0 <= ny < 15 and board[nx][ny] == "O":
                    rate += (1+i)
                else:
                    break
        return rate

    def rateBoard(self, board):
        ratingBoard = [[0 for _ in range(15)] for _ in range(15)]
        for i in range(15):
            for j in range(15):
                if board[i][j] == ".":
                    ratingBoard[i][j] = self.ratePoint(i, j, board)
        for i in range(15):
            for j in range(15):
                print("%5.2f, " % ratingBoard[j][i], end="")
            print()

        print()

        return ratingBoard

    def nextMove(self, board):
        # for i in range(15):
        #     for j in range(15):
        #         print("%s, " % board[j][i], end="")
        #     print()
        # print()
        
        
        max_rating = -1
        best_move = None
        rating_board = self.rateBoard(board)
        for i in range(15):
            for j in range(15):
                if board[i][j] == "." and rating_board[i][j] > max_rating:
                    #print("rating_board[%d][%d] = %f" % (i, j, rating_board[i][j]))
                    max_rating = rating_board[i][j]
                    best_move = (i, j)
        return best_move


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Gomoku")
    game = Gomoku()
    gui = GomokuGUI(root, game, DummyPlayer)
    root.mainloop()
