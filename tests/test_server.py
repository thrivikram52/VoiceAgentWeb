"""Smoke test — verify server starts and serves frontend."""

import pytest
from fastapi.testclient import TestClient

from server.main import app


class TestServerSmoke:
    def test_frontend_served(self):
        """Verify the frontend index.html is served at /."""
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Voice Agent" in resp.text

    def test_websocket_connects(self):
        """Verify WebSocket endpoint accepts connections."""
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert data["type"] == "state"
            assert data["state"] == "LISTENING"
