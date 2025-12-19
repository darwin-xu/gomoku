# Gomoku AI Quick Start Guide

Get started with training and playing against your Gomoku AI in minutes!

## Prerequisites

- Python 3.9+ (3.11-3.12 recommended for CoreML export)
- 2GB+ free disk space
- 4GB+ RAM recommended

## Installation

```bash
# 1. Clone the repository (if not already done)
git clone https://github.com/darwin-xu/gomoku.git
cd gomoku

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r python_ai/requirements.txt
```

## Training Your First Model

### Option 1: Quick Training (7-10 minutes)
Train a basic model with minimal simulations for quick results:

```bash
python -m python_ai.train \
    --episodes 50 \
    --simulations 16 \
    --augment \
    --model-path python_ai/checkpoints/quick_model.pt
```

### Option 2: Quality Training with Telemetry (1-2 hours)
Train a stronger model with real-time monitoring:

```bash
python -m python_ai.train \
    --episodes 200 \
    --simulations 64 \
    --augment \
    --telemetry \
    --model-path python_ai/checkpoints/policy_value.pt
```

Then open http://127.0.0.1:8765 in your browser to watch training progress!

### Option 3: Overnight Training (8+ hours)
For maximum strength:

```bash
python -m python_ai.train \
    --episodes 1000 \
    --simulations 128 \
    --augment \
    --telemetry \
    --model-path python_ai/checkpoints/strong_model.pt
```

## Playing Against Your AI

Once you have a trained model:

```bash
# Start the player service with your model
python -m python_ai.player \
    --model-path python_ai/checkpoints/policy_value.pt \
    --mcts-sims 64 \
    --open
```

This will:
1. Start a web server at http://127.0.0.1:8000
2. Automatically open your browser
3. Let you play against your AI!

In the UI:
- Enable "Play against AI"
- Keep "Use remote AI service" checked
- Click on the board to make moves

## Resuming Training

Stop training anytime (Ctrl+C) and resume later:

```bash
python -m python_ai.train \
    --resume \
    --episodes 500 \
    --simulations 64 \
    --augment \
    --model-path python_ai/checkpoints/policy_value.pt
```

The system automatically:
- Loads the model weights
- Restores optimizer state
- Continues from the last episode
- Loads the replay buffer

## Checking Training Progress

### View Checkpoint Info
```bash
python -m python_ai.inspect \
    --model-path python_ai/checkpoints/policy_value.pt
```

Output shows:
- Model configuration (channels, blocks)
- Training episode number
- Parameter count
- Replay buffer status

### Compare Model Strength
Test if your new model is stronger than an old one:

```bash
python -m python_ai.eval \
    --a python_ai/checkpoints/policy_value.pt \
    --b python_ai/checkpoints/older_checkpoint.pt \
    --games 100 \
    --sims 64
```

If model A wins >60% of games, it's meaningfully stronger!

## Tips for Success

### Training
- **Start small**: Use 16-32 simulations for first 100 episodes
- **Enable augmentation**: `--augment` gives 8x more training data
- **Monitor with telemetry**: `--telemetry` helps spot problems early
- **Save frequently**: Default `--save-every 25` is good
- **Be patient**: Real strength emerges after 200-500 episodes

### Hyperparameters
- **Fast iteration**: `--simulations 8-16` (weak but fast)
- **Balanced**: `--simulations 64` (default, good quality/speed)
- **Strong play**: `--simulations 128-256` (slow but strong)
- **Batch size**: Default 128 is good, reduce if out of memory
- **Learning rate**: Default 1e-3 works well, try 5e-4 if unstable

### Troubleshooting
- **Slow training**: Reduce `--simulations` or `--games-per-episode`
- **High memory**: Reduce `--batch-size` or `--replay-size`
- **Weak AI**: Train longer (500+ episodes), increase simulations
- **No improvement**: Check telemetry for loss trends, try lower learning rate

## Advanced Features

### Export to CoreML (macOS only)
For optimal performance on Apple Silicon:

```bash
python -m python_ai.train \
    --episodes 200 \
    --model-path python_ai/checkpoints/policy_value.pt \
    --coreml-path python_ai/checkpoints/policy_value.mlpackage
```

Then use with:
```bash
python -m python_ai.player \
    --coreml \
    --model-path python_ai/checkpoints/policy_value.mlpackage \
    --open
```

### Custom Network Size
For more capacity (slower training):

```bash
python -m python_ai.train \
    --channels 256 \
    --blocks 12 \
    --episodes 200 \
    --model-path python_ai/checkpoints/large_model.pt
```

**⚠️ IMPORTANT**: Never change `--channels` or `--blocks` when resuming from a checkpoint!

Changing the network architecture causes an architecture mismatch error when loading the checkpoint. The saved weights won't fit the new model structure. If you need a different network size, start a new training run with a different `--model-path`.

## Next Steps

1. **Train your first model** with quick settings (50 episodes)
2. **Play against it** to see current strength
3. **Continue training** with `--resume` for 200+ episodes
4. **Compare versions** with arena evaluation
5. **Share your results** - what episode count gives good play?

## Getting Help

- See `IMPLEMENTATION_SUMMARY.md` for detailed documentation
- Check `README.md` for full feature list
- Run `python -m python_ai.train --help` for all options

## Example Training Session

```bash
# Day 1: Initial training with monitoring
python -m python_ai.train \
    --episodes 100 \
    --simulations 64 \
    --augment \
    --telemetry \
    --model-path python_ai/checkpoints/my_model.pt

# Check progress
python -m python_ai.inspect --model-path python_ai/checkpoints/my_model.pt

# Day 2: Continue training
python -m python_ai.train \
    --resume \
    --episodes 300 \
    --simulations 64 \
    --augment \
    --model-path python_ai/checkpoints/my_model.pt

# Day 3: Compare with initial checkpoint
python -m python_ai.eval \
    --a python_ai/checkpoints/my_model.pt \
    --b python_ai/checkpoints/my_model.pt.backup \
    --games 200 \
    --sims 64

# Play against the trained AI
python -m python_ai.player \
    --model-path python_ai/checkpoints/my_model.pt \
    --mcts-sims 128 \
    --open
```

Happy training! 🎮
