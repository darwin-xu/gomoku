# Gomoku AI Coding Agent Instructions

## Project Overview

Dual-language Gomoku (Five-in-a-Row) implementation:
- **TypeScript/Node.js**: Web UI and game engine (`src/`)
- **Python**: AlphaZero-style AI training pipeline (`python_ai/`)
- **Integration**: Python Flask service (`player.py`) serves AI moves to the TypeScript UI via REST API

## Architecture

### Two-Part System
1. **Game UI (TypeScript)**: Simple Express server serving static HTML/JS at `:3000`
2. **AI Service (Python)**: Flask server at `:8000` exposing `/api/ai-move` endpoint

### Data Flow
```
User → Browser UI → Express (:3000) → Fetch → Flask (:8000) → MCTS + Neural Net → AI Move → UI
```

### Core Components
- `src/core/Board.ts`: 15×15 board state, win detection (5-in-a-row)
- `src/core/Game.ts`: Game logic orchestration
- `src/ai/AIPlayer.ts`: TypeScript interface for AI implementations (defines contract)
- `python_ai/model.py`: ResNet policy-value network (~2.5M params)
- `python_ai/mcts.py`: PUCT-based tree search with Dirichlet noise
- `python_ai/train.py`: Self-play training loop with replay buffer

## Development Workflows

### Building & Testing
```bash
# TypeScript UI
npm install && npm run build     # Compile TS → dist/
npm start                         # Run Express server
npm run dev                       # Watch mode (auto-rebuild)

# Python AI
python3 -m venv .venv && source .venv/bin/activate
pip install -r python_ai/requirements.txt
python -m python_ai.train --episodes 2 --simulations 8  # Quick smoke test
python -m python_ai.player --model-path model.pt --open  # Launch UI + AI
```

### Training Workflow
**Critical**: Never change `--channels` or `--blocks` when using `--resume` (architecture mismatch error). Use different `--model-path` for new architectures.

```bash
# Initial training with telemetry
python -m python_ai.train --episodes 200 --simulations 64 --augment --telemetry
# Dashboard: http://127.0.0.1:8765

# Resume from checkpoint (preserves model + optimizer + replay buffer)
python -m python_ai.train --resume --episodes 500

# Compare strength objectively
python -m python_ai.eval --a new.pt --b old.pt --games 200 --sims 64
```

## Key Conventions

### Python AI Module Organization
- **Training**: `train.py` orchestrates self-play → `self_play.py` generates games → `mcts.py` selects moves
- **Inference**: `player.py` adds tactical guardrails (immediate win/block, 4-in-a-row) before MCTS fallback
- **State encoding**: 3-channel (current player stones, opponent stones, bias channel)
- **Symmetry**: 8-way dihedral augmentation in `self_play.py` (rotations + flips)

### TypeScript Structure
- **Strict mode enabled**: All type annotations required
- **Module pattern**: Each core component (Board, Game) is self-contained
- **Interface-driven**: `AIPlayer` interface allows multiple AI backends

### Checkpoint Format
Saved as `.pt` with dict structure:
```python
{
    "state_dict": model.state_dict(),
    "config": {"channels": 128, "blocks": 8},
    "optimizer": optimizer.state_dict(),  # For --resume
    "training": {"episode": N}
}
```
Replay buffer sidecar: `<model>.pt.replay.npz` (auto-loaded with `--resume`)

## Integration Points

### Cross-Language Communication
- Python Flask (`player.py`) expects POST to `/api/ai-move` with JSON:
  ```json
  {"board": [[...]], "currentPlayer": "BLACK|WHITE"}
  ```
- Returns: `{"row": int, "col": int}`
- UI JavaScript (`src/public/app.js`) handles API calls

### Device Support
- **MPS (Apple Silicon)**: Auto-detected via `torch.backends.mps.is_available()`
- **CoreML export**: Optional, requires Python 3.11/3.12 (not 3.13+)
- Falls back to CPU if MPS unavailable

## Common Patterns

### MCTS with Neural Network
```python
# At root, add Dirichlet noise for exploration (training only)
noise = np.random.dirichlet([alpha] * valid_moves)
priors = (1 - frac) * nn_priors + frac * noise

# PUCT selection: Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))
```

### Tactical Guardrails (player.py)
Priority order in inference:
1. Take immediate win (5-in-a-row)
2. Block opponent's immediate win
3. Create 4-in-a-row threat
4. Block opponent's 4-in-a-row
5. MCTS with neural network

### Temperature Decay
```python
# Episode N: temp = max(min_temp, initial_temp * decay^(N-1))
# Default: 1.0 → 0.995^N → 0.1 (more exploitation as training progresses)
```

## Project-Specific Quirks

- **No tests yet**: Manual verification via training smoke tests
- **Checkpoints excluded**: `.gitignore` blocks `*.pt`, `*.mlpackage`, `python_ai/checkpoints/`
- **Telemetry is optional**: Dashboard only runs if `--telemetry` flag set
- **Training losses ≠ strength**: Use arena evaluation (`eval.py`) for true skill measurement
- **Board encoding**: Player perspective normalized (current player always in channel 0)

## Essential Files for AI Agents

- `IMPLEMENTATION_SUMMARY.md`: Complete feature documentation + performance data
- `QUICK_START.md`: Step-by-step training guide with timing estimates
- `python_ai/train.py`: ~400 lines, main entry point for understanding training flow
- `python_ai/mcts.py`: ~150 lines, core MCTS logic with PUCT
- `src/ai/AIPlayer.ts`: TypeScript AI contract (shows expected interface)

## Debugging Tips

- **Training stuck**: Check `python_ai/checkpoints/telemetry.jsonl` for loss trends
- **MPS errors on Mac**: System falls back to CPU automatically (expected)
- **Import errors**: Ensure `python_ai/__init__.py` exists and Python path includes repo root
- **Architecture mismatch**: Used `--resume` with different `--channels/--blocks`
