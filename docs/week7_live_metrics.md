# Week 7 — Live Aggregator Metrics

The central Flower server now starts a privacy-safe WebSocket endpoint at
`ws://127.0.0.1:8765` by default. It broadcasts one JSON message after each
completed federated evaluation round.

```json
{
  "event": "round_metrics",
  "round": 1,
  "loss": 0.000286,
  "accuracy": 0.8044,
  "latency_sec": 2.788,
  "timestamp": "2026-10-08T00:00:00+00:00"
}
```

The payload excludes raw patient data, local client metrics, model parameters,
and encryption keys. Dashboard clients should connect to the WebSocket and
update their loss/accuracy display whenever `event` is `round_metrics`.

## Run

The normal simulation starts the stream automatically. To change its address:

```powershell
python src/server.py --metrics-host 127.0.0.1 --metrics-port 8765
```

For tests or headless runs, disable it with `--disable-live-metrics`.
