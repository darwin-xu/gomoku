# Gomoku Game

A traditional Gomoku (Five in a Row) game built with TypeScript and Node.js.

## Features

- Traditional Gomoku rules (15x15 board, 5-in-a-row to win)
- Simple and modern UI
- AI interface for future AI implementation and training

## Setup

```bash
# Install dependencies
npm install

# Build the project
npm run build

# Start the server
npm start

# Development mode (auto-rebuild)
npm run dev
```

Then open http://localhost:3000 in your browser.

## Project Structure

- `src/core/` - Core game logic
- `src/ai/` - AI player interface
- `src/public/` - Web UI
- `dist/` - Compiled JavaScript output

## Game Rules

- Two players take turns placing stones on a 15x15 board
- The first player to get 5 stones in a row (horizontal, vertical, or diagonal) wins
- Black plays first

## AI Interface

The game provides an interface for AI players. Implement the `AIPlayer` interface to create your own AI:

```typescript
interface AIPlayer {
    getMove(board: Board, color: PlayerColor): Promise<Position>;
}
```
