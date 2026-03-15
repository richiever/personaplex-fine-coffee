#!/bin/bash
# PersonaPlex Coffee Shop Server Startup Script
# Handles model download, greeting token generation, and server startup

set -e

echo "=========================================="
echo "PersonaPlex Coffee Shop Server"
echo "=========================================="

# Check for required environment variables
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set!"
    echo "Please set HF_TOKEN to your HuggingFace token with access to AnthrolyticB/personaplex-coffee-v1"
    exit 1
fi

# Define paths — use /app (not /workspace, which RunPod overwrites with a volume mount)
MODEL_DIR="/app/model"
GREETING_TOKENS="/app/barista_greeting_tokens.pt"
REPO_DIR="/app"

# Step 1: Download model weights if not present
if [ ! -f "$MODEL_DIR/model_3.safetensors" ]; then
    echo ""
    echo "[1/4] Downloading model weights from HuggingFace..."
    echo "      Repository: AnthrolyticB/personaplex-coffee-v1"
    echo "      Target: $MODEL_DIR"

    huggingface-cli download AnthrolyticB/personaplex-coffee-v1 barista_greeting_tokens.pt \
        --local-dir "$MODEL_DIR" \
        --token "$HF_TOKEN"

    echo ""
    echo "      Downloading model_3.safetensors from AnthrolyticB/infinisona-v1..."
    huggingface-cli download AnthrolyticB/infinisona-v1 model_3.safetensors \
        --local-dir "$MODEL_DIR" \
        --token "$HF_TOKEN"

    echo "      Model weights downloaded successfully!"
else
    echo ""
    echo "[1/4] Model weights already present, skipping download."
fi

# Step 2: Ensure barista greeting tokens are present
# The tokens are downloaded from HuggingFace alongside the model weights.
# Fallback: generate them on-the-fly if missing (adds ~15-20s).
if [ ! -f "$GREETING_TOKENS" ]; then
    # Check if downloaded alongside model weights
    if [ -f "$MODEL_DIR/barista_greeting_tokens.pt" ]; then
        echo ""
        echo "[2/4] Copying greeting tokens from model directory..."
        cp "$MODEL_DIR/barista_greeting_tokens.pt" "$GREETING_TOKENS"
    else
        echo ""
        echo "[2/4] Generating barista greeting tokens (not found in download)..."
        echo "      This requires GPU and will take a moment."

        cd "$REPO_DIR" || { echo "ERROR: Could not cd to $REPO_DIR"; exit 1; }
        GREETING_SAVE_DIR="/app" python generate_greeting_tokens.py

        if [ ! -f "$GREETING_TOKENS" ]; then
            echo "ERROR: Failed to generate greeting tokens!"
            exit 1
        fi
        echo "      Greeting tokens generated successfully!"
    fi
else
    echo ""
    echo "[2/4] Greeting tokens already present, skipping."
fi

# Step 3: Start Cloudflare quick tunnel for WebSocket passthrough
echo ""
echo "[3/4] Starting Cloudflare quick tunnel..."
TUNNEL_LOG="/tmp/cloudflared.log"
cloudflared tunnel --url http://localhost:8998 --no-autoupdate > "$TUNNEL_LOG" 2>&1 &

TUNNEL_URL=""
for i in $(seq 1 30); do
    TUNNEL_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -1)
    [ -n "$TUNNEL_URL" ] && break
    sleep 1
done

if [ -z "$TUNNEL_URL" ]; then
    echo "ERROR: Cloudflare tunnel failed to start"
    cat "$TUNNEL_LOG"
    exit 1
fi
export TUNNEL_URL
echo "      Tunnel URL: $TUNNEL_URL"

# Step 4: Launch the Moshi server
echo ""
echo "[4/4] Starting Moshi server..."
echo "      Host: 0.0.0.0"
echo "      Port: 8998"
echo "      Model: $MODEL_DIR/model_3.safetensors"
echo "      User voice prompt: $GREETING_TOKENS"
echo ""
echo "=========================================="
echo "Server starting on port 8998"
echo "=========================================="
echo ""

cd "$REPO_DIR" || { echo "ERROR: Could not cd to $REPO_DIR"; exit 1; }
exec python -m moshi.server \
    --moshi-weight "$MODEL_DIR/model_3.safetensors" \
    --mimi-weight /app/codec/tokenizer-e351c8d8-checkpoint125.safetensors \
    --tokenizer /app/codec/tokenizer_spm_32k_3.model \
    --voice-prompt-dir /app/codec/voices \
    --static none \
    --user-voice-prompt "$GREETING_TOKENS" \
    --host 0.0.0.0 \
    --port 8998
