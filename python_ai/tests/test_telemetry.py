"""Tests for telemetry module."""
from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List

import pytest

from python_ai.telemetry import Telemetry, TelemetryConfig, make_telemetry


class TestTelemetryConfig:
    """TelemetryConfig tests."""

    def test_config_defaults(self) -> None:
        """Config has sensible defaults."""
        cfg = TelemetryConfig(
            enabled=True,
            dashboard_url="http://localhost:8080",
            job_id="test",
        )
        assert cfg.timeout_sec == 0.2

    def test_config_custom_timeout(self) -> None:
        """Can customize timeout."""
        cfg = TelemetryConfig(
            enabled=True,
            dashboard_url="http://localhost:8080",
            job_id="test",
            timeout_sec=1.0,
        )
        assert cfg.timeout_sec == 1.0


class TestTelemetryDisabled:
    """Tests for disabled telemetry."""

    def test_disabled_does_not_send(self) -> None:
        """Disabled telemetry does not attempt to send."""
        cfg = TelemetryConfig(enabled=False, dashboard_url="", job_id="")
        telemetry = Telemetry(cfg)

        # Should not raise even with no server
        telemetry.log({"episode": 1, "loss": 0.5})

    def test_make_telemetry_disabled(self) -> None:
        """make_telemetry with enabled=False returns disabled instance."""
        telemetry = make_telemetry(enabled=False, dashboard_url=None, job_id=None)
        assert telemetry.cfg.enabled is False


class TestTelemetryEnabled:
    """Tests for enabled telemetry."""

    def test_make_telemetry_enabled(self) -> None:
        """make_telemetry with enabled=True creates proper config."""
        telemetry = make_telemetry(
            enabled=True,
            dashboard_url="http://127.0.0.1:9999",
            job_id="job123",
        )
        assert telemetry.cfg.enabled is True
        assert telemetry.cfg.dashboard_url == "http://127.0.0.1:9999"
        assert telemetry.cfg.job_id == "job123"

    def test_make_telemetry_no_url_disables(self) -> None:
        """make_telemetry without URL disables telemetry."""
        telemetry = make_telemetry(enabled=True, dashboard_url=None, job_id="x")
        assert telemetry.cfg.enabled is False

    def test_make_telemetry_default_job_id(self) -> None:
        """make_telemetry uses 'train' as default job_id."""
        telemetry = make_telemetry(
            enabled=True,
            dashboard_url="http://localhost:8080",
            job_id=None,
        )
        assert telemetry.cfg.job_id == "train"


class TestTelemetryLogging:
    """Tests for telemetry log behavior."""

    def test_log_adds_timestamp(self) -> None:
        """Log adds timestamp if not present."""
        received: List[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                received.append(json.loads(body))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass  # suppress logging

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request)
        thread.start()

        try:
            cfg = TelemetryConfig(
                enabled=True,
                dashboard_url=f"http://127.0.0.1:{port}",
                job_id="test_job",
                timeout_sec=2.0,
            )
            telemetry = Telemetry(cfg)
            telemetry.log({"episode": 42})
        finally:
            thread.join(timeout=3)
            server.server_close()

        assert len(received) == 1
        assert "ts" in received[0]
        assert received[0]["episode"] == 42
        assert received[0]["job_id"] == "test_job"

    def test_log_preserves_existing_ts(self) -> None:
        """Log preserves user-provided timestamp."""
        received: List[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                received.append(json.loads(body))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request)
        thread.start()

        try:
            cfg = TelemetryConfig(
                enabled=True,
                dashboard_url=f"http://127.0.0.1:{port}",
                job_id="test",
                timeout_sec=2.0,
            )
            telemetry = Telemetry(cfg)
            telemetry.log({"episode": 1, "ts": 12345.0})
        finally:
            thread.join(timeout=3)
            server.server_close()

        assert len(received) == 1
        assert received[0]["ts"] == 12345.0

    def test_log_handles_connection_error(self) -> None:
        """Log does not raise on connection error."""
        cfg = TelemetryConfig(
            enabled=True,
            dashboard_url="http://127.0.0.1:1",  # unlikely to be listening
            job_id="test",
            timeout_sec=0.1,
        )
        telemetry = Telemetry(cfg)

        # Should not raise
        telemetry.log({"episode": 1})

    def test_log_url_format(self) -> None:
        """Log posts to correct URL path."""
        received_paths: List[str] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received_paths.append(self.path)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.handle_request)
        thread.start()

        try:
            cfg = TelemetryConfig(
                enabled=True,
                dashboard_url=f"http://127.0.0.1:{port}",
                job_id="my_job_123",
                timeout_sec=2.0,
            )
            telemetry = Telemetry(cfg)
            telemetry.log({"x": 1})
        finally:
            thread.join(timeout=3)
            server.server_close()

        assert len(received_paths) == 1
        assert received_paths[0] == "/api/telemetry/my_job_123"
