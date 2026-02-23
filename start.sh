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

# Define paths
MODEL_DIR="/workspace/model"
GREETING_TOKENS="/workspace/barista_greeting_tokens.pt"
REPO_DIR="/workspace/personaplex-fine-coffee"

# Step 1: Download model weights if not present
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo ""
    echo "[1/3] Downloading model weights from HuggingFace..."
    echo "      Repository: AnthrolyticB/personaplex-coffee-v1"
    echo "      Target: $MODEL_DIR"

    huggingface-cli download AnthrolyticB/personaplex-coffee-v1 \
        --local-dir "$MODEL_DIR" \
        --token "$HF_TOKEN"

    echo "      Model weights downloaded successfully!"
else
    echo ""
    echo "[1/3] Model weights already present, skipping download."
fi

# Step 2: Ensure barista greeting tokens are present
# The tokens are downloaded from HuggingFace alongside the model weights.
# Fallback: generate them on-the-fly if missing (adds ~15-20s).
if [ ! -f "$GREETING_TOKENS" ]; then
    # Check if downloaded alongside model weights
    if [ -f "$MODEL_DIR/barista_greeting_tokens.pt" ]; then
        echo ""
        echo "[2/3] Copying greeting tokens from model directory..."
        cp "$MODEL_DIR/barista_greeting_tokens.pt" "$GREETING_TOKENS"
    else
        echo ""
        echo "[2/3] Generating barista greeting tokens (not found in download)..."
        echo "      This requires GPU and will take a moment."

        cd "$REPO_DIR" || { echo "ERROR: Could not cd to $REPO_DIR"; exit 1; }
        GREETING_SAVE_DIR="/workspace" python generate_greeting_tokens.py

        if [ ! -f "$GREETING_TOKENS" ]; then
            echo "ERROR: Failed to generate greeting tokens!"
            exit 1
        fi
        echo "      Greeting tokens generated successfully!"
    fi
else
    echo ""
    echo "[2/3] Greeting tokens already present, skipping."
fi

# Step 3: Launch the Moshi server
echo ""
echo "[3/3] Starting Moshi server..."
echo "      Host: 0.0.0.0"
echo "      Port: 8998"
echo "      Model: $MODEL_DIR/model.safetensors"
echo "      User voice prompt: $GREETING_TOKENS"
echo ""
echo "=========================================="
echo "Server ready! Connect via WebSocket at:"
echo "ws://<your-runpod-ip>:8998/api/moshi"
echo "=========================================="
echo ""

cd "$REPO_DIR" || { echo "ERROR: Could not cd to $REPO_DIR"; exit 1; }
exec python -m moshi.server \
    --moshi-weight "$MODEL_DIR/model.safetensors" \
    --user-voice-prompt "$GREETING_TOKENS" \
    --host 0.0.0.0 \
    --port 8998