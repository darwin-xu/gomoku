"""Run a simple Flask service that serves the UI and provides AI moves."""
from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

# Allow running as a script: python python_ai/player.py
if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

from python_ai.gomoku_env import BOARD_SIZE
from python_ai.gomoku_env import GomokuEnv
from python_ai.mcts import MCTS
from python_ai.model import PolicyValueNet, get_device, load_checkpoint


class TorchAIAgent:
    def __init__(self, model_path: Path, mcts_simulations: int = 0, c_puct: float = 1.5, restrict_move_distance: int | None = 2):
        device_cfg = get_device()
        self.device = device_cfg.device
        self.model = load_checkpoint(str(model_path), map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        self.mcts_simulations = int(mcts_simulations)
        self.c_puct = float(c_puct)
        # Important: Dirichlet noise is for self-play exploration, not for inference.
        self.mcts = (
            MCTS(
                model=self.model,
                device=self.device,
                c_puct=self.c_puct,
                dirichlet_alpha=0.3,
                dirichlet_frac=0.0,
                restrict_move_distance=restrict_move_distance,
            )
            if self.mcts_simulations > 0
            else None
        )

    def predict(self, board: np.ndarray, current_player: int, include_policy: bool = False) -> Tuple[int, int, Optional[list[float]]]:
        policy_out: Optional[list[float]] = None

        if self.mcts is not None:
            env = GomokuEnv()
            env.board = board.astype(np.int8).copy()
            env.current_player = int(current_player)
            env.winner = None
            action, visits = self.mcts.run(env, simulations=self.mcts_simulations, temperature=1e-3)
            if include_policy:
                probs = visits / (visits.sum() + 1e-8)
                policy_out = probs.tolist()
            row, col = divmod(action, BOARD_SIZE)
            return row, col, policy_out

        state = self._encode(board, current_player).to(self.device)
        with torch.no_grad():
            policy_logits, _ = self.model(state)

        # Work on the model device to avoid CPU/MPS mismatches
        logits = policy_logits[0]
        mask = torch.from_numpy((board.flatten() == 0)).to(self.device)
        logits = logits.masked_fill(~mask, -1e9)

        if include_policy:
            probs = torch.softmax(logits, dim=-1)
            policy_out = probs.detach().cpu().tolist()

        action = int(torch.argmax(logits).item())
        row, col = divmod(action, BOARD_SIZE)
        return row, col, policy_out

    @staticmethod
    def _encode(board: np.ndarray, perspective: int) -> torch.Tensor:
        current = (board == perspective).astype(np.float32)
        opponent = (board == -perspective).astype(np.float32)
        bias = np.full_like(current, fill_value=1.0 if perspective == 1 else 0.0)
        stacked = np.stack([current, opponent, bias], axis=0)
        return torch.from_numpy(stacked).unsqueeze(0)


class CoreMLAIAgent:
    def __init__(self, coreml_path: Path):
        try:
            import coremltools as ct
        except Exception as e:
            raise RuntimeError("coremltools not available; install to use CoreML backend") from e
        self.model = ct.models.MLModel(str(coreml_path))

    def predict(self, board: np.ndarray, current_player: int) -> Tuple[int, int]:
        # CoreML expects NHWC by default; we exported NCHW so keep that
        current = (board == current_player).astype(np.float32)
        opponent = (board == -current_player).astype(np.float32)
        bias = np.full_like(current, fill_value=1.0 if current_player == 1 else 0.0)
        stacked = np.stack([current, opponent, bias], axis=0)
        input_arr = stacked[np.newaxis, ...]
        out = self.model.predict({"input": input_arr})
        policy = out[list(out.keys())[0]].reshape(-1)
        mask = (board.flatten() == 0)
        policy[~mask] = -1e9
        action = int(np.argmax(policy))
        row, col = divmod(action, BOARD_SIZE)
        return row, col


def create_app(static_dir: Path, agent) -> Flask:
    app = Flask(__name__, static_folder=str(static_dir), template_folder=str(static_dir))

    @app.route('/')
    def index():
        return send_from_directory(static_dir, 'index.html')

    @app.route('/<path:path>')
    def serve_static(path: str):
        return send_from_directory(static_dir, path)

    @app.route('/api/ai-move', methods=['POST'])
    def ai_move():
        data: Dict = request.get_json(force=True)
        board = np.array(data.get('board', []), dtype=np.int8)
        current_player = data.get('currentPlayer', 'BLACK')
        perspective = 1 if current_player.upper() == 'BLACK' else -1
        include_policy = bool(data.get('debugPolicy'))
        row, col, policy = agent.predict(board, perspective, include_policy=include_policy)
        resp = {"row": int(row), "col": int(col)}
        if include_policy and policy is not None:
            resp["policy"] = policy
        return jsonify(resp)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI service and UI")
    parser.add_argument("--model-path", type=str, default="./python_ai/checkpoints/policy_value.pt")
    parser.add_argument("--coreml", action="store_true", help="Use CoreML backend if available")
    parser.add_argument("--static-dir", type=str, default="./src/public", help="Path to UI assets")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Open browser automatically")
    parser.add_argument("--mcts-sims", type=int, default=64, help="MCTS simulations per move (PyTorch backend only)")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--restrict-move-distance", type=int, default=2, help="Limit candidate moves to this Manhattan distance from existing stones (match training). Use -1 to disable.")
    args = parser.parse_args()

    static_dir = Path(args.static_dir).resolve()
    model_path = Path(args.model_path).resolve()

    if args.coreml:
        agent = CoreMLAIAgent(model_path)
        print("Using CoreML backend (will attempt to utilize ANE)")
    else:
        restrict = None if int(args.restrict_move_distance) < 0 else int(args.restrict_move_distance)
        agent = TorchAIAgent(
            model_path,
            mcts_simulations=args.mcts_sims,
            c_puct=args.c_puct,
            restrict_move_distance=restrict,
        )
        msg = f"Using PyTorch backend (MPS if available), MCTS sims={args.mcts_sims}, restrict_moves={restrict if restrict is not None else 'none'}"
        print(msg)

    app = create_app(static_dir, agent)

    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(f"http://{args.host}:{args.port}"),).start()

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
