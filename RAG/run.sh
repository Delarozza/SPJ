#!/usr/bin/env bash
set -e

# Prechod do priečinka skriptu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="../.venv/bin/python3"
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

# Overenie / spustenie lokálneho Ollama servera, ak ešte nebeží
if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    if [ -f "$HOME/.local/bin/ollama" ]; then
        echo "[run.sh] Spúšťam lokálny Ollama server na pozadí..."
        CUDA_VISIBLE_DEVICES="" "$HOME/.local/bin/ollama" serve > /tmp/ollama.log 2>&1 &
        sleep 2
    fi
fi

"$PYTHON" task.py "$@"
