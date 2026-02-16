# Quick Launch: PersonaPlex Server (with wait-for-user patch)

## Fresh RunPod Instance

```bash
# 1. Fix PyTorch
pip uninstall torch torchaudio torchvision -y
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0

# 2. Clone repo
cd /workspace
git clone https://github.com/richiever/personaplex-fine-coffee.git

# 3. Uninstall stock Moshi, install modified version (editable)
pip uninstall moshi -y
cd /workspace/personaplex-fine-coffee/moshi
pip install -e .

# 4. Install other deps
pip install huggingface_hub sentencepiece

# 5. Set HF token
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# 6. Download fine-tuned model weights
huggingface-cli download AnthrolyticB/personaplex-coffee-v1 \
  --local-dir /workspace/model

# 7. Launch server
python -m moshi.server \
  --moshi-weight /workspace/model/model.safetensors \
  --host 0.0.0.0 --port 8998
```

## Returning to an existing RunPod

```bash
# Pull latest changes (editable install picks them up automatically)
cd /workspace/personaplex-fine-coffee
git pull

# Launch server
python -m moshi.server \
  --moshi-weight /workspace/model/model.safetensors \
  --host 0.0.0.0 --port 8998
```

## How it works

1. Server starts and loads model
2. Model stays **silent** — no audio or text output
3. You speak first as the barista: "Hi, welcome in!"
4. VAD detects your voice, model begins responding as the customer
5. Natural back-and-forth conversation
