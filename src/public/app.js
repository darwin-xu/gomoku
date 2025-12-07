// Game constants
const BOARD_SIZE = 15;
const CELL_SIZE = 40;
const STONE_RADIUS = 15;
const BOARD_PADDING = 20;

// Game state
let game = null;
let canvas = null;
let ctx = null;
let aiEnabled = false;

// Player colors
const PlayerColor = {
    BLACK: 'BLACK',
    WHITE: 'WHITE',
    EMPTY: 'EMPTY'
};

const GameState = {
    IN_PROGRESS: 'IN_PROGRESS',
    BLACK_WIN: 'BLACK_WIN',
    WHITE_WIN: 'WHITE_WIN',
    DRAW: 'DRAW'
};

// Initialize the game
function initGame() {
    canvas = document.getElementById('gameBoard');
    ctx = canvas.getContext('2d');

    // Initialize game state
    game = {
        board: createEmptyBoard(),
        currentPlayer: PlayerColor.BLACK,
        gameState: GameState.IN_PROGRESS,
        moveHistory: []
    };

    // Set up event listeners
    canvas.addEventListener('click', handleBoardClick);
    document.getElementById('resetBtn').addEventListener('click', resetGame);
    document.getElementById('undoBtn').addEventListener('click', undoMove);
    document.getElementById('aiMoveBtn').addEventListener('click', makeAIMove);
    document.getElementById('aiToggle').addEventListener('change', (e) => {
        aiEnabled = e.target.checked;
    });

    // Draw initial board
    drawBoard();
    updateUI();
}

// Create empty board
function createEmptyBoard() {
    return Array(BOARD_SIZE)
        .fill(null)
        .map(() => Array(BOARD_SIZE).fill(PlayerColor.EMPTY));
}

// Draw the game board
function drawBoard() {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid lines
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;

    for (let i = 0; i < BOARD_SIZE; i++) {
        // Vertical lines
        ctx.beginPath();
        ctx.moveTo(BOARD_PADDING + i * CELL_SIZE, BOARD_PADDING);
        ctx.lineTo(BOARD_PADDING + i * CELL_SIZE, BOARD_PADDING + (BOARD_SIZE - 1) * CELL_SIZE);
        ctx.stroke();

        // Horizontal lines
        ctx.beginPath();
        ctx.moveTo(BOARD_PADDING, BOARD_PADDING + i * CELL_SIZE);
        ctx.lineTo(BOARD_PADDING + (BOARD_SIZE - 1) * CELL_SIZE, BOARD_PADDING + i * CELL_SIZE);
        ctx.stroke();
    }

    // Draw star points (traditional Gomoku board markers)
    const starPoints = [
        [3, 3], [3, 11], [11, 3], [11, 11], [7, 7]
    ];
    ctx.fillStyle = '#333';
    starPoints.forEach(([row, col]) => {
        ctx.beginPath();
        ctx.arc(
            BOARD_PADDING + col * CELL_SIZE,
            BOARD_PADDING + row * CELL_SIZE,
            3,
            0,
            2 * Math.PI
        );
        ctx.fill();
    });

    // Draw stones
    for (let row = 0; row < BOARD_SIZE; row++) {
        for (let col = 0; col < BOARD_SIZE; col++) {
            const color = game.board[row][col];
            if (color !== PlayerColor.EMPTY) {
                drawStone(row, col, color);
            }
        }
    }
}

// Draw a stone at the specified position
function drawStone(row, col, color) {
    const x = BOARD_PADDING + col * CELL_SIZE;
    const y = BOARD_PADDING + row * CELL_SIZE;

    ctx.beginPath();
    ctx.arc(x, y, STONE_RADIUS, 0, 2 * Math.PI);

    if (color === PlayerColor.BLACK) {
        ctx.fillStyle = '#2c3e50';
        ctx.fill();
        ctx.strokeStyle = '#1a252f';
        ctx.lineWidth = 2;
        ctx.stroke();
    } else {
        ctx.fillStyle = '#ecf0f1';
        ctx.fill();
        ctx.strokeStyle = '#34495e';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}

// Handle board click
function handleBoardClick(event) {
    if (game.gameState !== GameState.IN_PROGRESS) {
        return;
    }

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Convert to board coordinates
    const col = Math.round((x - BOARD_PADDING) / CELL_SIZE);
    const row = Math.round((y - BOARD_PADDING) / CELL_SIZE);

    // Make the move
    if (makeMove(row, col)) {
        // If AI is enabled and it's now white's turn, make AI move
        if (aiEnabled && game.currentPlayer === PlayerColor.WHITE && game.gameState === GameState.IN_PROGRESS) {
            setTimeout(() => makeAIMove(), 500);
        }
    }
}

// Make a move
function makeMove(row, col) {
    // Validate position
    if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) {
        return false;
    }

    if (game.board[row][col] !== PlayerColor.EMPTY) {
        return false;
    }

    // Place stone
    game.board[row][col] = game.currentPlayer;
    game.moveHistory.push({ row, col, color: game.currentPlayer });

    // Check for win
    if (checkWin(row, col, game.currentPlayer)) {
        game.gameState = game.currentPlayer === PlayerColor.BLACK ? GameState.BLACK_WIN : GameState.WHITE_WIN;
    } else if (isBoardFull()) {
        game.gameState = GameState.DRAW;
    } else {
        // Switch player
        game.currentPlayer = game.currentPlayer === PlayerColor.BLACK ? PlayerColor.WHITE : PlayerColor.BLACK;
    }

    drawBoard();
    updateUI();
    return true;
}

// Check for win
function checkWin(row, col, color) {
    const directions = [
        { dr: 0, dc: 1 },  // Horizontal
        { dr: 1, dc: 0 },  // Vertical
        { dr: 1, dc: 1 },  // Diagonal \
        { dr: 1, dc: -1 }  // Diagonal /
    ];

    for (const { dr, dc } of directions) {
        if (countInLine(row, col, color, dr, dc) >= 5) {
            return true;
        }
    }

    return false;
}

// Count stones in line
function countInLine(row, col, color, dr, dc) {
    let count = 1;
    count += countDirection(row, col, color, dr, dc);
    count += countDirection(row, col, color, -dr, -dc);
    return count;
}

// Count in one direction
function countDirection(row, col, color, dr, dc) {
    let count = 0;
    let r = row + dr;
    let c = col + dc;

    while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && game.board[r][c] === color) {
        count++;
        r += dr;
        c += dc;
    }

    return count;
}

// Check if board is full
function isBoardFull() {
    for (let row = 0; row < BOARD_SIZE; row++) {
        for (let col = 0; col < BOARD_SIZE; col++) {
            if (game.board[row][col] === PlayerColor.EMPTY) {
                return false;
            }
        }
    }
    return true;
}

// Update UI
function updateUI() {
    const statusElement = document.getElementById('gameStatus');
    const playerStone = document.getElementById('currentPlayerStone');

    // Update player indicator
    playerStone.className = 'player-stone ' + (game.currentPlayer === PlayerColor.BLACK ? 'black' : 'white');

    // Update status
    if (game.gameState === GameState.BLACK_WIN) {
        statusElement.textContent = 'Black Wins!';
        statusElement.style.color = '#2c3e50';
    } else if (game.gameState === GameState.WHITE_WIN) {
        statusElement.textContent = 'White Wins!';
        statusElement.style.color = '#95a5a6';
    } else if (game.gameState === GameState.DRAW) {
        statusElement.textContent = 'Draw!';
        statusElement.style.color = '#7f8c8d';
    } else {
        statusElement.textContent = 'Game in progress';
        statusElement.style.color = '#667eea';
    }

    // Update button states
    document.getElementById('undoBtn').disabled = game.moveHistory.length === 0;
}

// Reset game
function resetGame() {
    game = {
        board: createEmptyBoard(),
        currentPlayer: PlayerColor.BLACK,
        gameState: GameState.IN_PROGRESS,
        moveHistory: []
    };
    drawBoard();
    updateUI();
}

// Undo move
function undoMove() {
    if (game.moveHistory.length === 0) {
        return;
    }

    const lastMove = game.moveHistory.pop();
    game.board[lastMove.row][lastMove.col] = PlayerColor.EMPTY;
    game.currentPlayer = lastMove.color;
    game.gameState = GameState.IN_PROGRESS;

    drawBoard();
    updateUI();
}

// Simple AI move (random available position near center)
function makeAIMove() {
    if (game.gameState !== GameState.IN_PROGRESS) {
        return;
    }

    const availableMoves = [];
    const center = Math.floor(BOARD_SIZE / 2);

    // Find available positions, prefer center
    for (let row = 0; row < BOARD_SIZE; row++) {
        for (let col = 0; col < BOARD_SIZE; col++) {
            if (game.board[row][col] === PlayerColor.EMPTY) {
                const distance = Math.abs(row - center) + Math.abs(col - center);
                availableMoves.push({ row, col, priority: -distance });
            }
        }
    }

    if (availableMoves.length === 0) {
        return;
    }

    // Sort by priority and add some randomness
    availableMoves.sort((a, b) => b.priority - a.priority);
    const topMoves = availableMoves.slice(0, Math.min(5, availableMoves.length));
    const move = topMoves[Math.floor(Math.random() * topMoves.length)];

    makeMove(move.row, move.col);
}

// Initialize when page loads
window.addEventListener('load', initGame);
