"""WebSocket publisher for privacy-safe aggregated federated metrics."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import threading
from typing import Any, Dict, Set

import websockets


def build_round_metric_message(
    server_round: int, loss: float, accuracy: float, latency_sec: float
) -> Dict[str, Any]:
    """Create the stable, browser-friendly message emitted after each round.

    Only aggregate metrics are included deliberately: raw MRI data, individual
    client metrics, gradients, and model weights must remain private.
    """
    return {
        "event": "round_metrics",
        "round": int(server_round),
        "loss": float(loss),
        "accuracy": float(accuracy),
        "latency_sec": float(latency_sec),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


class LiveMetricsPublisher:
    """Run a WebSocket endpoint in a background thread for Flower callbacks."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._connections: Set[Any] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._start_error: Exception | None = None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def start(self, timeout: float = 5.0) -> bool:
        """Start the endpoint without blocking the Flower aggregation server."""
        if self._thread is not None:
            return self._start_error is None
        self._thread = threading.Thread(target=self._run, name="live-metrics-ws", daemon=True)
        self._thread.start()
        self._ready.wait(timeout)
        return self._start_error is None and self._ready.is_set()

    def publish_round(self, server_round: int, loss: float, accuracy: float, latency_sec: float) -> None:
        """Schedule a non-blocking broadcast from Flower's synchronous callback."""
        if self._loop is None or self._start_error is not None:
            return
        message = json.dumps(build_round_metric_message(server_round, loss, accuracy, latency_sec))
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self._loop)

    def stop(self, timeout: float = 5.0) -> None:
        """Close clients and stop the background event loop."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._stopped.set)
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as exc:  # Server startup must not take down Flower.
            self._start_error = exc
            self._ready.set()
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        async with websockets.serve(self._handle_connection, self.host, self.port):
            self._ready.set()
            while not self._stopped.is_set():
                await asyncio.sleep(0.1)

    async def _handle_connection(self, websocket: Any) -> None:
        self._connections.add(websocket)
        try:
            await websocket.send(json.dumps({"event": "connected", "endpoint": self.url}))
            await websocket.wait_closed()
        finally:
            self._connections.discard(websocket)

    async def _broadcast(self, message: str) -> None:
        disconnected = []
        for connection in tuple(self._connections):
            try:
                await connection.send(message)
            except Exception:
                disconnected.append(connection)
        for connection in disconnected:
            self._connections.discard(connection)
