"""Simple telemetry for training: JSONL + tiny HTTP dashboard."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse


_DASHBOARD_HTML = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Gomoku Training Telemetry</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; }
    .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px 14px; }
    .small { color: #555; font-size: 12px; }
    canvas { border: 1px solid #ddd; border-radius: 10px; }
    label { font-size: 14px; }
  </style>
</head>
<body>
  <h2>Gomoku Training Telemetry</h2>
  <div class=\"row\">
    <div class=\"card\">
      <div><strong>Status</strong>: <span id=\"status\">-</span></div>
      <div><strong>Episode</strong>: <span id=\"episode\">-</span></div>
      <div><strong>Replay</strong>: <span id=\"replay\">-</span></div>
      <div class=\"small\" id=\"updated\">-</div>
    </div>
    <div class=\"card\">
      <label>Auto-refresh <input id=\"auto\" type=\"checkbox\" checked /></label>
      <label style=\"margin-left:10px\">Interval (ms) <input id=\"interval\" type=\"number\" value=\"1000\" min=\"200\" step=\"100\"/></label>
      <button id=\"refresh\" style=\"margin-left:10px\">Refresh</button>
    </div>
  </div>

  <h3>Loss (policy/value)</h3>
  <canvas id=\"loss\" width=\"1100\" height=\"300\"></canvas>

  <h3>Episode time (sec)</h3>
  <canvas id=\"time\" width=\"1100\" height=\"220\"></canvas>

<script>
const lossCanvas = document.getElementById('loss');
const timeCanvas = document.getElementById('time');

function drawSeries(canvas, series, opts) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0,0,w,h);

  const pad = 30;
  const xs = series.map(p => p.x);
  const ysAll = series.flatMap(p => p.y.filter(v => Number.isFinite(v)));

  if (xs.length < 2 || ysAll.length === 0) {
    ctx.fillStyle = '#666';
    ctx.fillText('No data yet', pad, pad);
    return;
  }

  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  let yMin = Math.min(...ysAll);
  let yMax = Math.max(...ysAll);
  if (yMin === yMax) { yMin -= 1; yMax += 1; }

  // axes
  ctx.strokeStyle = '#ddd';
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, h - pad);
  ctx.lineTo(w - pad, h - pad);
  ctx.stroke();

  function sx(x) { return pad + (x - xMin) * (w - 2*pad) / (xMax - xMin); }
  function sy(y) { return (h - pad) - (y - yMin) * (h - 2*pad) / (yMax - yMin); }

  // y labels
  ctx.fillStyle = '#666';
  ctx.font = '12px system-ui';
  ctx.fillText(yMax.toFixed(2), 4, pad + 4);
  ctx.fillText(yMin.toFixed(2), 4, h - pad);

  // draw each line
  const colors = opts.colors;
  const labels = opts.labels;
  for (let lineIdx = 0; lineIdx < labels.length; lineIdx++) {
    ctx.strokeStyle = colors[lineIdx];
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (const p of series) {
      const y = p.y[lineIdx];
      if (!Number.isFinite(y)) continue;
      const X = sx(p.x);
      const Y = sy(y);
      if (!started) {
        ctx.moveTo(X, Y);
        started = true;
      } else {
        ctx.lineTo(X, Y);
      }
    }
    ctx.stroke();
  }

  // legend
  ctx.fillStyle = '#111';
  ctx.fillText(opts.title, pad, 18);
  let lx = pad;
  let ly = 22;
  for (let i = 0; i < labels.length; i++) {
    ctx.fillStyle = colors[i];
    ctx.fillRect(lx, ly, 10, 10);
    ctx.fillStyle = '#111';
    ctx.fillText(labels[i], lx + 14, ly + 10);
    lx += 120;
  }
}

async function fetchMetrics() {
  const res = await fetch('/metrics?limit=400');
  return await res.json();
}

function parseFloatOrNaN(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

async function refresh() {
  const data = await fetchMetrics();
  const points = data.points || [];
  if (points.length) {
    const last = points[points.length - 1];
    document.getElementById('status').textContent = last.status ?? '-';
    document.getElementById('episode').textContent = String(last.episode ?? '-');
    document.getElementById('replay').textContent = `${last.replay_size ?? '-'} / ${last.min_replay ?? '-'}`;
    document.getElementById('updated').textContent = `Updated: ${new Date((last.ts||Date.now())*1000).toLocaleString()}`;
  }

  const lossSeries = points.map(p => ({
    x: p.episode,
    y: [parseFloatOrNaN(p.policy_loss), parseFloatOrNaN(p.value_loss)]
  }));
  drawSeries(lossCanvas, lossSeries, {
    title: 'Loss',
    labels: ['policy_loss', 'value_loss'],
    colors: ['#1f77b4', '#ff7f0e']
  });

  const timeSeries = points.map(p => ({
    x: p.episode,
    y: [parseFloatOrNaN(p.episode_sec), parseFloatOrNaN(p.avg_ep_sec)]
  }));
  drawSeries(timeCanvas, timeSeries, {
    title: 'Time',
    labels: ['episode_sec', 'avg_ep_sec'],
    colors: ['#2ca02c', '#d62728']
  });
}

document.getElementById('refresh').addEventListener('click', refresh);

let timer = null;
function schedule() {
  if (timer) clearInterval(timer);
  if (!document.getElementById('auto').checked) return;
  const interval = Math.max(200, Number(document.getElementById('interval').value) || 1000);
  timer = setInterval(refresh, interval);
}

document.getElementById('auto').addEventListener('change', schedule);
document.getElementById('interval').addEventListener('change', schedule);

refresh().then(schedule);
</script>
</body>
</html>
"""


@dataclass
class TelemetryConfig:
    enabled: bool
    host: str
    port: int
    path: Path
    keep_points: int = 5000


class Telemetry:
    def __init__(self, cfg: TelemetryConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._points: Deque[Dict[str, Any]] = deque(maxlen=int(cfg.keep_points))
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._file_lock = threading.Lock()

    def log(self, point: Dict[str, Any]) -> None:
        if not self.cfg.enabled:
            return
        point = dict(point)
        point.setdefault("ts", time.time())
        with self._lock:
            self._points.append(point)
        # Append JSONL for offline inspection
        try:
            self.cfg.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(point, ensure_ascii=False)
            with self._file_lock:
                with self.cfg.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            # Never crash training due to telemetry
            pass

    def get_points(self, limit: int = 500) -> Dict[str, Any]:
        with self._lock:
            pts = list(self._points)[-int(limit):]
        return {"points": pts}

    def start_server(self) -> None:
        if not self.cfg.enabled:
            return
        if self._server is not None:
            return

        telemetry = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    body = _DASHBOARD_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if parsed.path == "/metrics":
                    qs = parse_qs(parsed.query)
                    limit = 500
                    if "limit" in qs:
                        try:
                            limit = int(qs["limit"][0])
                        except Exception:
                            limit = 500
                    payload = telemetry.get_points(limit=limit)
                    body = json.dumps(payload).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                self.send_response(404)
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                # Keep training output clean
                return

        self._server = ThreadingHTTPServer((self.cfg.host, int(self.cfg.port)), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop_server(self) -> None:
        if self._server is None:
            return
        try:
            self._server.shutdown()
            self._server.server_close()
        finally:
            self._server = None
            self._thread = None


def make_telemetry(
    *,
    enabled: bool,
    host: str,
    port: int,
    path: Path,
    keep_points: int = 5000,
) -> Telemetry:
    return Telemetry(TelemetryConfig(enabled=enabled, host=host, port=port, path=path, keep_points=keep_points))
