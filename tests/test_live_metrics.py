import pytest
import socket

websockets = pytest.importorskip("websockets")

from src.live_metrics import build_round_metric_message
from src.live_metrics import LiveMetricsPublisher


def test_round_metric_message_contains_only_aggregate_fields() -> None:
    message = build_round_metric_message(2, 0.123, 0.91, 1.5)
    assert message["event"] == "round_metrics"
    assert message["round"] == 2
    assert message["loss"] == pytest.approx(0.123)
    assert message["accuracy"] == pytest.approx(0.91)
    assert message["latency_sec"] == pytest.approx(1.5)
    assert {"weights", "parameters", "images", "mri", "client_metrics"}.isdisjoint(message)


def test_websocket_client_receives_round_metrics() -> None:
    with socket.socket() as port_probe:
        port_probe.bind(("127.0.0.1", 0))
        port = port_probe.getsockname()[1]

    publisher = LiveMetricsPublisher("127.0.0.1", port)
    assert publisher.start()
    try:
        from websockets.sync.client import connect

        with connect(publisher.url) as client:
            assert '"event": "connected"' in client.recv()
            publisher.publish_round(1, 0.2, 0.8, 1.0)
            payload = client.recv(timeout=2)
            assert '"event": "round_metrics"' in payload
            assert '"round": 1' in payload
    finally:
        publisher.stop()
