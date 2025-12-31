import { describe, it, expect, beforeEach } from 'vitest';
import { Board } from './Board';
import { PlayerColor } from './types';

describe('Board', () => {
    let board: Board;

    beforeEach(() => {
        board = new Board();
    });

    describe('constructor', () => {
        it('creates a 15x15 board', () => {
            expect(board.getSize()).toBe(15);
        });

        it('creates an empty board', () => {
            const state = board.getBoard();
            for (let row = 0; row < 15; row++) {
                for (let col = 0; col < 15; col++) {
                    expect(state[row][col]).toBe(PlayerColor.EMPTY);
                }
            }
        });
    });

    describe('getCell', () => {
        it('returns EMPTY for unoccupied cells', () => {
            expect(board.getCell({ row: 7, col: 7 })).toBe(PlayerColor.EMPTY);
        });

        it('returns the placed color after placeStone', () => {
            board.placeStone({ row: 5, col: 5 }, PlayerColor.BLACK);
            expect(board.getCell({ row: 5, col: 5 })).toBe(PlayerColor.BLACK);
        });

        it('throws for invalid position', () => {
            expect(() => board.getCell({ row: -1, col: 0 })).toThrow('Invalid position');
            expect(() => board.getCell({ row: 0, col: 15 })).toThrow('Invalid position');
            expect(() => board.getCell({ row: 20, col: 20 })).toThrow('Invalid position');
        });
    });

    describe('placeStone', () => {
        it('places a stone and returns true', () => {
            const result = board.placeStone({ row: 7, col: 7 }, PlayerColor.BLACK);
            expect(result).toBe(true);
            expect(board.getCell({ row: 7, col: 7 })).toBe(PlayerColor.BLACK);
        });

        it('returns false for invalid position', () => {
            expect(board.placeStone({ row: -1, col: 0 }, PlayerColor.BLACK)).toBe(false);
            expect(board.placeStone({ row: 0, col: 15 }, PlayerColor.WHITE)).toBe(false);
        });

        it('returns false for occupied position', () => {
            board.placeStone({ row: 5, col: 5 }, PlayerColor.BLACK);
            expect(board.placeStone({ row: 5, col: 5 }, PlayerColor.WHITE)).toBe(false);
            expect(board.placeStone({ row: 5, col: 5 }, PlayerColor.BLACK)).toBe(false);
        });

        it('can place both BLACK and WHITE stones', () => {
            board.placeStone({ row: 0, col: 0 }, PlayerColor.BLACK);
            board.placeStone({ row: 0, col: 1 }, PlayerColor.WHITE);
            expect(board.getCell({ row: 0, col: 0 })).toBe(PlayerColor.BLACK);
            expect(board.getCell({ row: 0, col: 1 })).toBe(PlayerColor.WHITE);
        });
    });

    describe('isValidPosition', () => {
        it('returns true for valid positions', () => {
            expect(board.isValidPosition({ row: 0, col: 0 })).toBe(true);
            expect(board.isValidPosition({ row: 14, col: 14 })).toBe(true);
            expect(board.isValidPosition({ row: 7, col: 7 })).toBe(true);
        });

        it('returns false for negative coordinates', () => {
            expect(board.isValidPosition({ row: -1, col: 0 })).toBe(false);
            expect(board.isValidPosition({ row: 0, col: -1 })).toBe(false);
        });

        it('returns false for out-of-bounds coordinates', () => {
            expect(board.isValidPosition({ row: 15, col: 0 })).toBe(false);
            expect(board.isValidPosition({ row: 0, col: 15 })).toBe(false);
        });
    });

    describe('isEmpty', () => {
        it('returns true for empty cells', () => {
            expect(board.isEmpty({ row: 7, col: 7 })).toBe(true);
        });

        it('returns false for occupied cells', () => {
            board.placeStone({ row: 7, col: 7 }, PlayerColor.BLACK);
            expect(board.isEmpty({ row: 7, col: 7 })).toBe(false);
        });
    });

    describe('isFull', () => {
        it('returns false for empty board', () => {
            expect(board.isFull()).toBe(false);
        });

        it('returns false for partially filled board', () => {
            board.placeStone({ row: 0, col: 0 }, PlayerColor.BLACK);
            expect(board.isFull()).toBe(false);
        });

        it('returns true when all cells are filled', () => {
            for (let row = 0; row < 15; row++) {
                for (let col = 0; col < 15; col++) {
                    const color = (row + col) % 2 === 0 ? PlayerColor.BLACK : PlayerColor.WHITE;
                    board.placeStone({ row, col }, color);
                }
            }
            expect(board.isFull()).toBe(true);
        });
    });

    describe('reset', () => {
        it('clears all stones from the board', () => {
            board.placeStone({ row: 0, col: 0 }, PlayerColor.BLACK);
            board.placeStone({ row: 7, col: 7 }, PlayerColor.WHITE);
            board.reset();

            expect(board.getCell({ row: 0, col: 0 })).toBe(PlayerColor.EMPTY);
            expect(board.getCell({ row: 7, col: 7 })).toBe(PlayerColor.EMPTY);
        });
    });

    describe('getBoard', () => {
        it('returns a copy of the board state', () => {
            board.placeStone({ row: 5, col: 5 }, PlayerColor.BLACK);
            const copy = board.getBoard();

            // Modify the copy
            copy[5][5] = PlayerColor.WHITE;

            // Original should be unchanged
            expect(board.getCell({ row: 5, col: 5 })).toBe(PlayerColor.BLACK);
        });
    });

    describe('checkWin', () => {
        describe('horizontal wins', () => {
            it('detects 5 in a row horizontally', () => {
                for (let col = 0; col < 5; col++) {
                    board.placeStone({ row: 7, col }, PlayerColor.BLACK);
                }
                expect(board.checkWin({ row: 7, col: 2 }, PlayerColor.BLACK)).toBe(true);
            });

            it('does not detect 4 in a row as win', () => {
                for (let col = 0; col < 4; col++) {
                    board.placeStone({ row: 7, col }, PlayerColor.BLACK);
                }
                expect(board.checkWin({ row: 7, col: 2 }, PlayerColor.BLACK)).toBe(false);
            });

            it('detects win from any position in the line', () => {
                for (let col = 3; col < 8; col++) {
                    board.placeStone({ row: 7, col }, PlayerColor.BLACK);
                }
                // Check from first, middle, and last position
                expect(board.checkWin({ row: 7, col: 3 }, PlayerColor.BLACK)).toBe(true);
                expect(board.checkWin({ row: 7, col: 5 }, PlayerColor.BLACK)).toBe(true);
                expect(board.checkWin({ row: 7, col: 7 }, PlayerColor.BLACK)).toBe(true);
            });
        });

        describe('vertical wins', () => {
            it('detects 5 in a row vertically', () => {
                for (let row = 0; row < 5; row++) {
                    board.placeStone({ row, col: 7 }, PlayerColor.WHITE);
                }
                expect(board.checkWin({ row: 2, col: 7 }, PlayerColor.WHITE)).toBe(true);
            });

            it('works at board edges', () => {
                for (let row = 10; row < 15; row++) {
                    board.placeStone({ row, col: 0 }, PlayerColor.BLACK);
                }
                expect(board.checkWin({ row: 12, col: 0 }, PlayerColor.BLACK)).toBe(true);
            });
        });

        describe('diagonal wins', () => {
            it('detects diagonal win (top-left to bottom-right)', () => {
                for (let i = 0; i < 5; i++) {
                    board.placeStone({ row: i, col: i }, PlayerColor.BLACK);
                }
                expect(board.checkWin({ row: 2, col: 2 }, PlayerColor.BLACK)).toBe(true);
            });

            it('detects diagonal win (top-right to bottom-left)', () => {
                for (let i = 0; i < 5; i++) {
                    board.placeStone({ row: i, col: 14 - i }, PlayerColor.WHITE);
                }
                expect(board.checkWin({ row: 2, col: 12 }, PlayerColor.WHITE)).toBe(true);
            });

            it('detects diagonal win in the middle of the board', () => {
                for (let i = 0; i < 5; i++) {
                    board.placeStone({ row: 5 + i, col: 5 + i }, PlayerColor.BLACK);
                }
                expect(board.checkWin({ row: 7, col: 7 }, PlayerColor.BLACK)).toBe(true);
            });
        });

        describe('no false positives', () => {
            it('does not detect win for wrong color', () => {
                for (let col = 0; col < 5; col++) {
                    board.placeStone({ row: 7, col }, PlayerColor.BLACK);
                }
                expect(board.checkWin({ row: 7, col: 2 }, PlayerColor.WHITE)).toBe(false);
            });

            it('does not detect win with gaps', () => {
                board.placeStone({ row: 7, col: 0 }, PlayerColor.BLACK);
                board.placeStone({ row: 7, col: 1 }, PlayerColor.BLACK);
                // Gap at col: 2
                board.placeStone({ row: 7, col: 3 }, PlayerColor.BLACK);
                board.placeStone({ row: 7, col: 4 }, PlayerColor.BLACK);
                board.placeStone({ row: 7, col: 5 }, PlayerColor.BLACK);
                expect(board.checkWin({ row: 7, col: 3 }, PlayerColor.BLACK)).toBe(false);
            });

            it('does not detect win with opponent stone in between', () => {
                board.placeStone({ row: 7, col: 0 }, PlayerColor.BLACK);
                board.placeStone({ row: 7, col: 1 }, PlayerColor.BLACK);
                board.placeStone({ row: 7, col: 2 }, PlayerColor.WHITE); // Opponent
                board.placeStone({ row: 7, col: 3 }, PlayerColor.BLACK);
                board.placeStone({ row: 7, col: 4 }, PlayerColor.BLACK);
                board.placeStone({ row: 7, col: 5 }, PlayerColor.BLACK);
                expect(board.checkWin({ row: 7, col: 4 }, PlayerColor.BLACK)).toBe(false);
            });
        });

        describe('more than 5 in a row', () => {
            it('detects 6 in a row as win', () => {
                for (let col = 0; col < 6; col++) {
                    board.placeStone({ row: 7, col }, PlayerColor.BLACK);
                }
                expect(board.checkWin({ row: 7, col: 3 }, PlayerColor.BLACK)).toBe(true);
            });
        });
    });
});
