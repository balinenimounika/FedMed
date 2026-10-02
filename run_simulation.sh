#!/usr/bin/env bash
# FedMed Federated Learning Simulation Runner for Linux/macOS
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "  FedMed Federated Learning Simulation (Bash/Linux/macOS)"
echo "=========================================================="

# Select Python environment
if [ -d "$SCRIPT_DIR/.venv" ]; then
    PYTHON_CMD="$SCRIPT_DIR/.venv/bin/python"
    echo "[Env] Using virtualenv Python: $PYTHON_CMD"
else
    PYTHON_CMD="python3"
    echo "[Env] Using system Python: $PYTHON_CMD"
fi

mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/results"

SERVER_LOG="$SCRIPT_DIR/logs/server.log"
CLIENT0_LOG="$SCRIPT_DIR/logs/client_0.log"
CLIENT1_LOG="$SCRIPT_DIR/logs/client_1.log"

rm -f "$SERVER_LOG" "$CLIENT0_LOG" "$CLIENT1_LOG"

# Trap SIGINT and SIGTERM to kill spawned background jobs
cleanup() {
    echo -e "\n[Interrupted] Terminating running simulation processes..."
    [ -n "$SERVER_PID" ] && kill -9 "$SERVER_PID" 2>/dev/null || true
    [ -n "$CLIENT0_PID" ] && kill -9 "$CLIENT0_PID" 2>/dev/null || true
    [ -n "$CLIENT1_PID" ] && kill -9 "$CLIENT1_PID" 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM

echo "[1/4] Starting FedMed Server on 127.0.0.1:8080..."
"$PYTHON_CMD" -u src/server.py > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "[2/4] Waiting for server port 8080 to become ready..."
TIMEOUT=30
ELAPSED=0
SERVER_READY=0

while [ $ELAPSED -lt $TIMEOUT ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[ERROR] Server process died prematurely."
        cat "$SERVER_LOG"
        exit 1
    fi

    if nc -z 127.0.0.1 8080 2>/dev/null || (echo > /dev/tcp/127.0.0.1/8080) 2>/dev/null; then
        SERVER_READY=1
        break
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

if [ $SERVER_READY -ne 1 ]; then
    echo "[ERROR] Server failed to start on 127.0.0.1:8080 within $TIMEOUT seconds."
    kill -9 "$SERVER_PID" 2>/dev/null || true
    cat "$SERVER_LOG"
    exit 1
fi

echo "[Server] FedMed Server is live and ready."

echo "[3/4] Launching Client 0 and Client 1..."
"$PYTHON_CMD" -u src/client.py --client-id 0 > "$CLIENT0_LOG" 2>&1 &
CLIENT0_PID=$!

"$PYTHON_CMD" -u src/client.py --client-id 1 > "$CLIENT1_LOG" 2>&1 &
CLIENT1_PID=$!

echo "[4/4] Simulation running. Waiting for 3 rounds to complete..."

wait "$SERVER_PID"
SERVER_STATUS=$?

wait "$CLIENT0_PID"
CLIENT0_STATUS=$?

wait "$CLIENT1_PID"
CLIENT1_STATUS=$?

echo ""
echo "=========================================================="
echo "  Simulation Completed. Process Exit Statuses:           "
echo "=========================================================="
echo "Server Exit Code  : $SERVER_STATUS"
echo "Client 0 Exit Code: $CLIENT0_STATUS"
echo "Client 1 Exit Code: $CLIENT1_STATUS"

if [ $SERVER_STATUS -eq 0 ] && [ $CLIENT0_STATUS -eq 0 ] && [ $CLIENT1_STATUS -eq 0 ]; then
    echo -e "\n[SUCCESS] Federated learning completed successfully!"
    if [ -f "$SCRIPT_DIR/results/training_history.csv" ]; then
        echo -e "\nTraining History (results/training_history.csv):"
        cat "$SCRIPT_DIR/results/training_history.csv"
    fi
    exit 0
else
    echo -e "\n[FAIL] One or more processes failed. Inspecting logs:"
    echo "--- Server Log ---"
    cat "$SERVER_LOG"
    echo "--- Client 0 Log ---"
    cat "$CLIENT0_LOG"
    echo "--- Client 1 Log ---"
    cat "$CLIENT1_LOG"
    exit 1
fi
