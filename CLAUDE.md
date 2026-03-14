# PersonaPlex Coffee - Backend

## Role Mapping (CRITICAL)

- **AGENT = CUSTOMER** (the fine-tuned model)
- **USER = BARISTA** (the human operator)

System prompts describe the customer. The model plays the customer role. The human speaks as the barista.

## Architecture

Moshi real-time speech-to-speech server with PersonaPlex fine-tuning for coffee shop customer personas.

- **Server**: `moshi.server` on port 8998
- **WebSocket endpoint**: `wss://<host>:8998/api/chat` (binary protocol)
- **HTTP root**: `GET /` (used by frontend health check to detect readiness)
- **Model**: AnthrolyticB/personaplex-coffee-v1 (HuggingFace)
- **Audio codec**: Mimi (8 codebooks, 12.5fps, 24kHz)
- **Voices**: Downloaded from `nvidia/personaplex-7b-v1` as `voices.tgz`, extracted at startup. Voice selected via `voice_prompt` WebSocket query param (e.g. `NATM3.pt`).

## Key Files

| File | Purpose |
|------|---------|
| `moshi/moshi/server.py` | WebSocket server, handles binary audio frames |
| `moshi/moshi/models/lm.py` | Language model with persona conditioning |
| `moshi/moshi/models/loaders.py` | Model weight loading from HuggingFace (`DEFAULT_REPO = nvidia/personaplex-7b-v1`) |
| `generate_greeting_tokens.py` | Pre-generates barista greeting via Chatterbox TTS + Mimi |
| `start.sh` | Startup script: model download, greeting gen, server launch |
| `Dockerfile.runpod` | Docker image for RunPod deployment |
| `.github/workflows/docker-build.yml` | CI: builds and pushes Docker image on push to main |

## Docker Image

- **Registry**: `anthrolytic/personaplex-coffee:latest` (Docker Hub)
- **Base**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Size**: ~24GB (includes PyTorch 2.6, CUDA 12.4, Moshi, Chatterbox TTS, model weights)
- **Working directory**: `/app` (NOT `/workspace` — RunPod mounts a volume at `/workspace` that overwrites image files)
- **Entrypoint**: NVIDIA entrypoint at `/opt/nvidia/nvidia_entrypoint.sh` (from base image)
- **CMD**: `bash /app/start.sh` (must be absolute path — NVIDIA entrypoint overrides WORKDIR)
- **Ports**: 8998 (Moshi server), 8888 (Jupyter)
- **Build secrets**: HF_TOKEN via BuildKit secret mount (`--mount=type=secret,id=hf_token`)

### Build Locally

```bash
DOCKER_BUILDKIT=1 docker build \
  -f Dockerfile.runpod \
  --secret id=hf_token,env=HF_TOKEN \
  -t anthrolytic/personaplex-coffee:latest .
```

### CI Pipeline

GitHub Actions builds on every push to `main`. Requires these repository secrets:
- `DOCKERHUB_USERNAME`: Docker Hub username
- `DOCKERHUB_TOKEN`: Docker Hub access token
- `HF_TOKEN`: HuggingFace token for downloading model weights at build time

### Critical Docker Lessons

1. **`/workspace` is off-limits**: RunPod unconditionally mounts a volume at `/workspace` regardless of `volumeInGb` setting. All COPY'd files and WORKDIR must be outside `/workspace` (we use `/app`).
2. **CMD must use absolute path**: The NVIDIA base image sets `ENTRYPOINT ['/opt/nvidia/nvidia_entrypoint.sh']` which overrides WORKDIR at runtime. `CMD ["bash", "start.sh"]` fails; must be `CMD ["bash", "/app/start.sh"]`.
3. **HF_TOKEN via BuildKit secrets, not ARG**: Using `ARG HF_TOKEN` bakes the token into image layer history. Use `RUN --mount=type=secret,id=hf_token` instead.
4. **Shell expansion in RUN**: When using `--mount=type=secret`, inline `$(cat /run/secrets/hf_token)` works. Assigning to a variable with backslash continuation does NOT (the variable is empty at expansion time).

## RunPod Deployment

### REST API (used by frontend)

The frontend uses the RunPod REST API at `https://rest.runpod.io/v1/pods`:

| Action | Method | Endpoint |
|--------|--------|----------|
| Create pod | `POST` | `/v1/pods` |
| Get pod | `GET` | `/v1/pods/{id}` |
| Stop pod | `POST` | `/v1/pods/{id}/stop` |
| Resume pod | `POST` | `/v1/pods/{id}/start` |
| Delete pod | `DELETE` | `/v1/pods/{id}` |

### REST API vs GraphQL

The GraphQL API (`podFindAndDeployOnDemand`) had persistent `SUPPLY_CONSTRAINT` errors for A40 despite availability. The REST API works reliably. Key schema differences:

| Field | GraphQL | REST |
|-------|---------|------|
| GPU type | `gpuTypeId: "NVIDIA A40"` (string) | `gpuTypeIds: ["NVIDIA A40"]` (array) |
| Ports | `ports: "8998/http,8888/http"` (string) | `ports: ["8998/http", "8888/http"]` (array) |
| Env vars | `env: [{key, value}]` (array of objects) | `env: {KEY: "value"}` (object) |
| Readiness | `pod { runtime { uptimeInSeconds } }` | **Not available** — `runtime` field does not exist in REST API |

### REST API Readiness Detection

The REST API response is **static** — no field changes to indicate the container is running. `runtime`, `uptimeInSeconds`, and `publicIp` are never populated. The only way to detect readiness is to **probe the actual service endpoint** at `https://{podId}-8998.proxy.runpod.net/`.

### Stop/Resume Strategy

Pods are **stopped** (not deleted) after sessions to enable fast resume:
- **Cold start** (new pod): ~5-10 min (image pull + model load)
- **Warm start** (resume stopped pod): ~30s-2 min
- Stopped pods are tied to their original machine. If resume fails (machine GPU occupied), the stale pod is deleted and a new one is created.
- Stopped pods incur storage cost but no GPU cost.

## Docker Template System (Future)

When implementing a RunPod template for faster deployments:

### Template Configuration

```
Name:            personaplex-coffee
Type:            Pod
Compute:         Nvidia GPU
Container image: anthrolytic/personaplex-coffee:latest
Start command:   bash /app/start.sh
Container disk:  50 GB
Volume disk:     0 GB
HTTP ports:      8888, 8998
Env vars:        HF_TOKEN=<value>
```

### Template Benefits

- RunPod pre-caches template images across their fleet, eliminating the ~5-8 min image pull
- Cold start drops from ~10 min to ~2 min (just model loading into VRAM)
- Combined with stop/resume, most users get ~30s startup

### Template Integration in launch-pod.ts

When using a template, the `launch-pod.ts` API route should:
1. Use `templateId` in the create pod body
2. **Remove** `imageName` — the template controls the image (if both are set, behavior is unpredictable; in testing, `templateId` overrode `imageName` at runtime despite the API response showing the custom image)
3. Still pass `env` with `HF_TOKEN` (template env vars can be overridden per-pod)

```typescript
const body = {
  name: 'personaplex-coffee',
  gpuTypeIds: ['NVIDIA A40'],
  templateId: '<template-id-here>',  // replaces imageName
  containerDiskInGb: 50,
  volumeInGb: 0,
  ports: ['8998/http', '8888/http'],
  env: { HF_TOKEN: hfToken },
}
```

### Template ID Discovery

After creating the template in RunPod dashboard, the template ID can be found:
- In the URL when viewing the template
- Via `GET /v1/templates` (if available in REST API)
- In the pod response after creating with that template (`pod.templateId`)

## Startup Flow (start.sh)

```
1. Check HF_TOKEN is set
2. Download model weights from HuggingFace (if not baked into image)
3. Copy/generate barista greeting tokens
4. Launch moshi.server on 0.0.0.0:8998
```

All paths use `/app` prefix. Model stored at `/app/model/`, greeting tokens at `/app/barista_greeting_tokens.pt`.

## Environment Variables

- `HF_TOKEN`: HuggingFace token for model access (required)
- `GREETING_SAVE_DIR`: Override greeting token save path (default: `/app`)

## Training

- LoRA fine-tuning on Moshi's LM head
- Training data: coffee shop conversation pairs in `data/`
- Tensor structure: Stream 0 = text, Streams 1-8 = customer audio, Streams 9-16 = barista audio

### Training Data Sources (local paths)

| File | Location | Conv IDs | Count |
|------|----------|----------|-------|
| `training.json` | `C:\Users\User\Videos\stuff\trainingPersonaPlex\training.json` | conv_0001–conv_0200 | 200 |
| `retail_training.json` | `C:\Users\User\Videos\stuff\trainingPersonaPlex\personaplex-fine-coffee\retail_training.json` | conv_0201–conv_1200 | 1000 |

### Critical Training/Inference Alignment Issues

1. **Audio normalization**: Inference pipeline normalizes audio to -24 LUFS (`normalize_audio` in `lm.py:979,1031`). Training .pt files MUST use the same normalization on TTS output before Mimi encoding, otherwise Mimi produces different tokens for the same speech.

2. **Barista greeting in training data**: The server injects a `--user-voice-prompt` (barista greeting) at inference start. Training data must include an equivalent first `user` turn so the model learns to respond after a greeting. Greetings:
   - Coffee shop conversations (training.json): "Welcome to the coffee shop, how can I help you?"
   - Retail conversations (retail_training.json): "Welcome to the store, how can I help you?"

3. **Repetition penalty**: Server uses `repetition_penalty=1.3` with 50-token window on text logits to prevent degenerate loops.

### LoRA Config (fix/lora-qkv-training branch)

| Parameter | Value |
|-----------|-------|
| Rank | 32 |
| Alpha | 64 |
| LR | 1.5e-6 |
| Targets | `self_attn.in_proj`, `self_attn.out_proj`, `gating.linear_in`, `gating.linear_out` |
| Epochs | 4 |

The `in_proj` module was registered as `nn.Linear` (not raw weight) so PEFT can discover it. Depformer path still uses raw weight for `multi_linear()` compatibility. Checkpoint key remap `in_proj_weight` ↔ `in_proj.weight` handled in `loaders.py` (load) and `lora_train_fixed.py` (merge).

### RunPod API Key

Key: (set in user's global CLAUDE.md)

### HuggingFace

- Token: set via `HF_TOKEN` env var
- Training data repo: `AnthrolyticB/personaplex-training-data-test` (dataset)
- Model repo: `nvidia/personaplex-7b-v1` (gated)

### Pipeline Script

`runpod_pipeline.py` — full data pipeline:
1. Download training.json from HF
2. Download LibriSpeech voice samples (100 agent + 100 user)
3. Generate TTS audio with Chatterbox TTS
4. Encode through Mimi + assemble .pt files [17, T]
5. Upload .pt files to HF

Key config: `SILENCE_TOKENS`, `SINE_TOKENS`, `PAD_TOKEN=3`, `ZERO_TOKEN=-1`, `TARGET_SR=24000`
