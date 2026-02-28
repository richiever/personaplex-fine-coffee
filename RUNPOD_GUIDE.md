# RunPod Pipeline Guide

## Option A: Full Pipeline (4x A40) + Training (H100)

### Step 1: 4x A40 Pod — Setup

```bash
apt-get update && apt-get install -y ffmpeg

pip install chatterbox-tts huggingface_hub sentencepiece peft safetensors torchaudio soundfile moshi

git clone -b CoffeePlex-Extended-Context https://github.com/richiever/personaplex-fine-coffee /workspace/data
cd /workspace/data
```

### Step 2: 4x A40 Pod — Download model weights

```bash
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('/workspace/weights', exist_ok=True)
hf_hub_download('kyutai/moshika-pytorch-bf16', 'tokenizer-e351c8d8-checkpoint125.safetensors', local_dir='/workspace/weights')
hf_hub_download('kyutai/moshika-pytorch-bf16', 'tokenizer_spm_32k_3.model', local_dir='/workspace/weights')
"
```

### Step 3: 4x A40 Pod — Download existing .pt files (conv_0000-conv_0200)

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'AnthrolyticB/personaplex-training-data-test',
    repo_type='dataset',
    allow_patterns='conv_0*.pt',
    local_dir='/workspace/pt_files',
)
snapshot_download(
    'AnthrolyticB/personaplex-training-data-test',
    repo_type='dataset',
    allow_patterns='conv_01*.pt',
    local_dir='/workspace/pt_files',
)
snapshot_download(
    'AnthrolyticB/personaplex-training-data-test',
    repo_type='dataset',
    allow_patterns='conv_020*.pt',
    local_dir='/workspace/pt_files',
)
"
```

### Step 4: 4x A40 Pod — Run pipeline (conv_0201-conv_1200)

Auto-detects 4 GPUs, splits TTS across them. Downloads retail_training.json from HF, downloads LibriSpeech voices, runs TTS, Mimi encodes, uploads .pt files to HF.

```bash
python runpod_pipeline.py --retail-only
```

Expected time: ~6 hours (TTS ~4.3h on 4 GPUs, Mimi ~1.5h, upload ~15min).

### Step 5: 4x A40 Pod — Verify

```bash
python -c "
from pathlib import Path
pts = sorted(Path('/workspace/pt_files').glob('*.pt'))
print(f'Total .pt files: {len(pts)}')
print(f'First: {pts[0].name}, Last: {pts[-1].name}')
expected = set(f'conv_{i:04d}.pt' for i in range(0, 1201))
have = set(f.name for f in pts)
missing = sorted(expected - have)
print(f'Missing: {len(missing)}')
if missing: print(missing[:10])
"
```

All 1201 .pt files should be present (conv_0000.pt - conv_1200.pt). The pipeline auto-uploads them to HuggingFace, so you can now terminate the A40 pod.

---

## Option B: H100 Training Only

Use this after the A40 pipeline has finished and uploaded .pt files to HF, or if you want to re-train later.

### Step 1: H100 Pod — Setup

```bash
pip install huggingface_hub moshi peft safetensors torchaudio

git clone -b CoffeePlex-Extended-Context https://github.com/richiever/personaplex-fine-coffee /workspace/data
cd /workspace/data
```

### Step 2: H100 Pod — Download all .pt files from HF

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'AnthrolyticB/personaplex-training-data-test',
    repo_type='dataset',
    allow_patterns='*.pt',
    local_dir='/workspace/pt_files',
)
"
```

### Step 3: H100 Pod — Download model weights

```bash
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('/workspace/weights', exist_ok=True)
hf_hub_download('kyutai/moshika-pytorch-bf16', 'tokenizer-e351c8d8-checkpoint125.safetensors', local_dir='/workspace/weights')
hf_hub_download('kyutai/moshika-pytorch-bf16', 'tokenizer_spm_32k_3.model', local_dir='/workspace/weights')
"
```

### Step 4: H100 Pod — Train

```bash
python lora_train_fixed.py --epochs 4 --grad-accum-steps 8
```

Expected time: ~45-90 min on H100 (vs ~2-3h on A40).

---

## Config Summary

| Parameter | Value | Reason |
|-----------|-------|--------|
| Epochs | 4 | 6x data vs original 8-epoch run |
| Grad accum steps | 8 | Larger effective batch for bigger dataset |
| Early stop patience | 2 | More data = faster convergence signal |
| Semantic weight | 50x | Reduced from 100x for acoustic stability with 6x data |
| LoRA rank | 16 | Unchanged |
| Learning rate | 2e-6 | Unchanged |

## Data Summary

| Dataset | Conversations | Conv IDs | Source |
|---------|--------------|----------|--------|
| Original (coffee shop) | 201 | conv_0000-conv_0200 | training.json |
| Retail (coffee + general) | 1000 | conv_0201-conv_1200 | retail_training.json |
| **Total** | **1201** | conv_0000-conv_1200 | All .pt files on HF |
