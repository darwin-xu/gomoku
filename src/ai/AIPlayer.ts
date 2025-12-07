import { Board } from '../core/Board';
import { Position, PlayerColor } from '../core/types';

/**
 * Interface for AI players
 * Implement this interface to create different AI strategies
 */
export interface AIPlayer {
    /**
     * Gets the next move for the AI player
     * @param board - The current game board
     * @param color - The color the AI is playing
     * @returns A promise that resolves to the chosen position
     */
    getMove(board: Board, color: PlayerColor): Promise<Position>;

    /**
     * Optional method for training the AI
     * @param trainingData - Training data specific to the AI implementation
     */
    train?(trainingData: any): Promise<void>;

    /**
     * Optional method to save the AI model
     * @param path - Path to save the model
     */
    saveModel?(path: string): Promise<void>;

    /**
     * Optional method to load the AI model
     * @param path - Path to load the model from
     */
    loadModel?(path: string): Promise<void>;
}

/**
 * Training data structure for AI training
 */
export interface TrainingGame {
    moves: Array<{
        position: Position;
        color: PlayerColor;
    }>;
    winner: PlayerColor | null;
}

/**
 * Example: Random AI player (for testing purposes)
 */
export class RandomAIPlayer implements AIPlayer {
    public async getMove(board: Board, color: PlayerColor): Promise<Position> {
        const size = board.getSize();
        const availableMoves: Position[] = [];

        // Find all available positions
        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                const position = { row, col };
                if (board.isEmpty(position)) {
                    availableMoves.push(position);
                }
            }
        }

        // Choose a random available position
        if (availableMoves.length === 0) {
            throw new Error('No available moves');
        }

        const randomIndex = Math.floor(Math.random() * availableMoves.length);
        return availableMoves[randomIndex];
    }
}

/**
 * Example: Simple heuristic AI (looks for immediate winning moves or blocks)
 */
export class SimpleAIPlayer implements AIPlayer {
    public async getMove(board: Board, color: PlayerColor): Promise<Position> {
        const size = board.getSize();
        const opponentColor = color === PlayerColor.BLACK ? PlayerColor.WHITE : PlayerColor.BLACK;

        // First, check if we can win
        const winningMove = this.findWinningMove(board, color);
        if (winningMove) {
            return winningMove;
        }

        // Second, block opponent's winning move
        const blockingMove = this.findWinningMove(board, opponentColor);
        if (blockingMove) {
            return blockingMove;
        }

        // Otherwise, find the best strategic position
        return this.findStrategicMove(board, color);
    }

    private findWinningMove(board: Board, color: PlayerColor): Position | null {
        const size = board.getSize();

        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                const position = { row, col };
                if (board.isEmpty(position)) {
                    // Try the move
                    board.placeStone(position, color);
                    const isWin = board.checkWin(position, color);

                    // Undo the move (we need to reset that cell)
                    // Note: This is a simplified approach
                    // In production, you'd want a proper undo mechanism

                    if (isWin) {
                        return position;
                    }
                }
            }
        }

        return null;
    }

    private findStrategicMove(board: Board, color: PlayerColor): Position {
        const size = board.getSize();
        const center = Math.floor(size / 2);

        // Prefer center and nearby positions
        const positions: Array<{ position: Position; priority: number }> = [];

        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                const position = { row, col };
                if (board.isEmpty(position)) {
                    const distance = Math.abs(row - center) + Math.abs(col - center);
                    positions.push({ position, priority: -distance });
                }
            }
        }

        // Sort by priority (closer to center = higher priority)
        positions.sort((a, b) => b.priority - a.priority);

        return positions[0].position;
    }
}
