import { Position, PlayerColor } from './types';

/**
 * Represents the Gomoku game board
 */
export class Board {
    private readonly size: number = 15;
    private board: PlayerColor[][];

    constructor() {
        this.board = this.createEmptyBoard();
    }

    /**
     * Creates an empty board
     */
    private createEmptyBoard(): PlayerColor[][] {
        return Array(this.size)
            .fill(null)
            .map(() => Array(this.size).fill(PlayerColor.EMPTY));
    }

    /**
     * Gets the size of the board
     */
    public getSize(): number {
        return this.size;
    }

    /**
     * Gets the color at a specific position
     */
    public getCell(position: Position): PlayerColor {
        if (!this.isValidPosition(position)) {
            throw new Error('Invalid position');
        }
        return this.board[position.row][position.col];
    }

    /**
     * Places a stone at the specified position
     */
    public placeStone(position: Position, color: PlayerColor): boolean {
        if (!this.isValidPosition(position)) {
            return false;
        }
        if (this.board[position.row][position.col] !== PlayerColor.EMPTY) {
            return false;
        }
        this.board[position.row][position.col] = color;
        return true;
    }

    /**
     * Checks if a position is valid
     */
    public isValidPosition(position: Position): boolean {
        return (
            position.row >= 0 &&
            position.row < this.size &&
            position.col >= 0 &&
            position.col < this.size
        );
    }

    /**
     * Checks if a position is empty
     */
    public isEmpty(position: Position): boolean {
        return this.getCell(position) === PlayerColor.EMPTY;
    }

    /**
     * Checks if the board is full
     */
    public isFull(): boolean {
        for (let row = 0; row < this.size; row++) {
            for (let col = 0; col < this.size; col++) {
                if (this.board[row][col] === PlayerColor.EMPTY) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Resets the board to empty state
     */
    public reset(): void {
        this.board = this.createEmptyBoard();
    }

    /**
     * Gets a copy of the current board state
     */
    public getBoard(): PlayerColor[][] {
        return this.board.map(row => [...row]);
    }

    /**
     * Checks if there's a winning line at the given position
     */
    public checkWin(position: Position, color: PlayerColor): boolean {
        const directions = [
            { dr: 0, dc: 1 },  // Horizontal
            { dr: 1, dc: 0 },  // Vertical
            { dr: 1, dc: 1 },  // Diagonal \
            { dr: 1, dc: -1 }  // Diagonal /
        ];

        for (const { dr, dc } of directions) {
            if (this.countInLine(position, color, dr, dc) >= 5) {
                return true;
            }
        }

        return false;
    }

    /**
     * Counts consecutive stones in a line (both directions)
     */
    private countInLine(position: Position, color: PlayerColor, dr: number, dc: number): number {
        let count = 1; // Count the current stone

        // Count in positive direction
        count += this.countDirection(position, color, dr, dc);

        // Count in negative direction
        count += this.countDirection(position, color, -dr, -dc);

        return count;
    }

    /**
     * Counts consecutive stones in one direction
     */
    private countDirection(position: Position, color: PlayerColor, dr: number, dc: number): number {
        let count = 0;
        let row = position.row + dr;
        let col = position.col + dc;

        while (
            row >= 0 &&
            row < this.size &&
            col >= 0 &&
            col < this.size &&
            this.board[row][col] === color
        ) {
            count++;
            row += dr;
            col += dc;
        }

        return count;
    }
}
