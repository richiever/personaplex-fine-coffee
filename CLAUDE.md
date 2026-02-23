# PersonaPlex Coffee - Backend

## Role Mapping (CRITICAL)

- **AGENT = CUSTOMER** (the fine-tuned model)
- **USER = BARISTA** (the human operator)

System prompts describe the customer. The model plays the customer role. The human speaks as the barista.

## Architecture

Moshi real-time speech-to-speech server with PersonaPlex fine-tuning for coffee shop customer personas.

- **Server**: `moshi.server` on port 8998
- **WebSocket endpoint**: `ws://<host>:8998/api/chat` (binary protocol)
- **Model**: AnthrolyticB/personaplex-coffee-v1 (HuggingFace)
- **Audio codec**: Mimi (8 codebooks, 12.5fps, 24kHz)

## Key Files

| File | Purpose |
|------|---------|
| `moshi/moshi/server.py` | WebSocket server, handles binary audio frames |
| `moshi/moshi/models/lm.py` | Language model with persona conditioning |
| `moshi/moshi/models/loaders.py` | Model weight loading from HuggingFace |
| `generate_greeting_tokens.py` | Pre-generates barista greeting via Chatterbox TTS + Mimi |
| `start.sh` | Startup script: model download, greeting gen, server launch |
| `Dockerfile.runpod` | Pre-built Docker image for RunPod deployment |
| `lora_train_fixed.py` | LoRA training script for persona fine-tuning |

## Commands

```bash
# Run server locally (requires GPU + model weights)
python -m moshi.server --moshi-weight /path/to/model.safetensors \
  --user-voice-prompt /path/to/barista_greeting_tokens.pt \
  --host 0.0.0.0 --port 8998

# Generate greeting tokens (requires GPU)
python generate_greeting_tokens.py

# Build Docker image
docker build -f Dockerfile.runpod --build-arg HF_TOKEN=$HF_TOKEN -t anthrolytic/personaplex-coffee:latest .
```

## Docker Image

- Registry: `anthrolytic/personaplex-coffee:latest`
- Base: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Bakes in model weights at build time (via HF_TOKEN build arg)
- Exposes ports 8998 (moshi) and 8888 (jupyter)
- CI: `.github/workflows/docker-build.yml` builds on push to main

## Training

- LoRA fine-tuning on Moshi's LM head
- Training data: coffee shop conversation pairs in `data/`
- Tensor structure: Stream 0 = text, Streams 1-8 = customer audio, Streams 9-16 = barista audio

## Environment Variables

- `HF_TOKEN`: HuggingFace token for model access (required)
