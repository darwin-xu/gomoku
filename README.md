# Gomoku Game

A traditional Gomoku (Five in a Row) game built with TypeScript and Node.js.

## Features

- Traditional Gomoku rules (15x15 board, 5-in-a-row to win)
- Simple and modern UI
- AI interface for future AI implementation and training

# Setup

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

## Python AI (self-play training + CoreML export)


```bash
# Create venv and install python deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r python_ai/requirements.txt

# Train with self-play (PyTorch, uses MPS if available)
# Uses AlphaZero-style self-play with MCTS visit targets.
python -m python_ai.train --episodes 200 --simulations 64 --augment --model-path python_ai/checkpoints/policy_value.pt

# Train with a live telemetry dashboard
python -m python_ai.train --episodes 200 --simulations 64 --augment --model-path python_ai/checkpoints/policy_value.pt --telemetry
# Then open http://127.0.0.1:8765 in a browser

# Optional: export to CoreML for Apple Neural Engine
python -m python_ai.train --episodes 5 --model-path python_ai/checkpoints/policy_value.pt --coreml-path python_ai/checkpoints/policy_value.mlpackage
```

Note: CoreML export depends on `coremltools` compatibility with your Python + PyTorch versions. On very new Python/Torch (e.g. Python 3.13+), export will be skipped/fail with a message while training still succeeds; use a separate env with Python 3.11/3.12 and a coremltools-supported PyTorch version (often <= 2.7) to export.

Key flags:
- `--model-path` choose which model file to save/load
- `--coreml-path` export a CoreML model for ANE-friendly inference
- `--resume` continue training from an existing checkpoint
 - `--replay-path` save/load replay buffer (defaults to `<model-path>.replay.npz`), so `--resume` can keep training data
 - `--simulations` MCTS simulations per move (quality vs speed)
 - `--augment` enable 8-way board symmetry augmentation
 - `--telemetry` start a local dashboard (loss/time trends)

Stopping/continuing with different parameters:
- You can stop training anytime (Ctrl+C) and resume later with `--resume` while changing parameters like `--simulations`, `--games-per-episode`, `--batch-size`, etc.
- Do not change network size (`--channels/--blocks`) unless you start a new checkpoint.

Training logs print per-episode policy/value loss and replay size.

## Play with the Python AI + UI

```bash
source .venv/bin/activate
python -m python_ai.player --model-path python_ai/checkpoints/policy_value.pt --mcts-sims 64 --open
# Or use CoreML backend (requires exported .mlpackage)
python -m python_ai.player --coreml --model-path python_ai/checkpoints/policy_value.mlpackage --open
```

This serves the existing UI (default http://127.0.0.1:8000). In the UI, enable "Play against AI"; keep "Use remote AI service" checked to use the Python backend. Uncheck to fall back to the browser heuristic.

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
