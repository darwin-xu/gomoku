import { Board } from './Board';
import { Position, PlayerColor, GameState, Move } from './types';

/**
 * Represents the Gomoku game logic
 */
export class Game {
    private board: Board;
    private currentPlayer: PlayerColor;
    private gameState: GameState;
    private moveHistory: Move[];

    constructor() {
        this.board = new Board();
        this.currentPlayer = PlayerColor.BLACK; // Black plays first
        this.gameState = GameState.IN_PROGRESS;
        this.moveHistory = [];
    }

    /**
     * Gets the current board
     */
    public getBoard(): Board {
        return this.board;
    }

    /**
     * Gets the current player
     */
    public getCurrentPlayer(): PlayerColor {
        return this.currentPlayer;
    }

    /**
     * Gets the current game state
     */
    public getGameState(): GameState {
        return this.gameState;
    }

    /**
     * Gets the move history
     */
    public getMoveHistory(): Move[] {
        return [...this.moveHistory];
    }

    /**
     * Makes a move at the specified position
     */
    public makeMove(position: Position): boolean {
        if (this.gameState !== GameState.IN_PROGRESS) {
            return false;
        }

        if (!this.board.placeStone(position, this.currentPlayer)) {
            return false;
        }

        // Record the move
        this.moveHistory.push({
            position,
            color: this.currentPlayer
        });

        // Check for win
        if (this.board.checkWin(position, this.currentPlayer)) {
            this.gameState =
                this.currentPlayer === PlayerColor.BLACK
                    ? GameState.BLACK_WIN
                    : GameState.WHITE_WIN;
            return true;
        }

        // Check for draw
        if (this.board.isFull()) {
            this.gameState = GameState.DRAW;
            return true;
        }

        // Switch player
        this.switchPlayer();
        return true;
    }

    /**
     * Switches the current player
     */
    private switchPlayer(): void {
        this.currentPlayer =
            this.currentPlayer === PlayerColor.BLACK
                ? PlayerColor.WHITE
                : PlayerColor.BLACK;
    }

    /**
     * Resets the game to initial state
     */
    public reset(): void {
        this.board.reset();
        this.currentPlayer = PlayerColor.BLACK;
        this.gameState = GameState.IN_PROGRESS;
        this.moveHistory = [];
    }

    /**
     * Checks if a move is valid
     */
    public isValidMove(position: Position): boolean {
        if (this.gameState !== GameState.IN_PROGRESS) {
            return false;
        }
        return this.board.isValidPosition(position) && this.board.isEmpty(position);
    }

    /**
     * Gets the winner (if any)
     */
    public getWinner(): PlayerColor | null {
        if (this.gameState === GameState.BLACK_WIN) {
            return PlayerColor.BLACK;
        }
        if (this.gameState === GameState.WHITE_WIN) {
            return PlayerColor.WHITE;
        }
        return null;
    }
}
