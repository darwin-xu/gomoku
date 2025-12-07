/**
 * Represents a position on the Gomoku board
 */
export interface Position {
    row: number;
    col: number;
}

/**
 * Represents the color of a player's stone
 */
export enum PlayerColor {
    BLACK = 'BLACK',
    WHITE = 'WHITE',
    EMPTY = 'EMPTY'
}

/**
 * Represents the state of the game
 */
export enum GameState {
    IN_PROGRESS = 'IN_PROGRESS',
    BLACK_WIN = 'BLACK_WIN',
    WHITE_WIN = 'WHITE_WIN',
    DRAW = 'DRAW'
}

/**
 * Represents a move in the game
 */
export interface Move {
    position: Position;
    color: PlayerColor;
}
