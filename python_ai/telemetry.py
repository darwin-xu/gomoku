"""Telemetry reporting for training.

Old behavior (removed): training started its own telemetry web server.

New behavior:
- Only the dashboard hosts the telemetry web UI.
- Training can *optionally* POST telemetry points to a running dashboard.
- If no dashboard is available, training continues with no telemetry.

This module is intentionally best-effort: it must never crash training.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass
class TelemetryConfig:
    enabled: bool
    dashboard_url: str
    job_id: str
    timeout_sec: float = 0.2


class Telemetry:
    def __init__(self, cfg: TelemetryConfig) -> None:
        self.cfg = cfg

    def log(self, point: Dict[str, Any]) -> None:
        if not self.cfg.enabled:
            return

        try:
            payload = dict(point)
            payload.setdefault("ts", time.time())
            payload.setdefault("job_id", self.cfg.job_id)

            url = self.cfg.dashboard_url.rstrip("/") + f"/api/telemetry/{self.cfg.job_id}"
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = Request(url=url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            with urlopen(req, timeout=float(self.cfg.timeout_sec)):
                return
        except Exception:
            # Never crash training due to telemetry transport.
            return


def make_telemetry(*, enabled: bool, dashboard_url: Optional[str], job_id: Optional[str]) -> Telemetry:
    if not enabled or not dashboard_url:
        return Telemetry(TelemetryConfig(enabled=False, dashboard_url="", job_id=""))

    return Telemetry(
        TelemetryConfig(
            enabled=True,
            dashboard_url=str(dashboard_url),
            job_id=str(job_id or "train"),
        )
    )
