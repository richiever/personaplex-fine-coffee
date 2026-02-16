# RunPod Setup Guide - PersonaPlex Fine-Tuned Model

Complete instructions for running your fine-tuned PersonaPlex model on RunPod.

---

## 1. Create RunPod Instance

- **Template:** PyTorch 2.1 or newer
- **GPU:** RTX 4090 or A6000 (min 24GB VRAM)
- **Disk:** 50GB+

---

## 2. Clone and Install

```bash
# SSH into RunPod
cd /workspace

# Clone your updated repo with wait-for-user feature
git clone https://github.com/richiever/personaplex-fine-coffee.git
cd personaplex-fine-coffee

# Install dependencies
pip install -e moshi
pip install huggingface-hub
```

---

## 3. Login to HuggingFace

```bash
# Login to HuggingFace (you'll need your token)
huggingface-cli login
```

---

## 4. Find Mimi Codec Location

First, let's check where the Mimi codec is located:

```bash
python3 << 'EOF'
from huggingface_hub import HfApi
api = HfApi()

print("Checking kyutai/moshiko-pytorch-bf16 for Mimi:")
try:
    files = api.list_repo_files("kyutai/moshiko-pytorch-bf16")
    for f in files:
        if 'mimi' in f.lower():
            print(f"  - {f}")
except:
    print("  Repo not found")

print("\nChecking kyutai/moshi-mlx for Mimi:")
try:
    files = api.list_repo_files("kyutai/moshi-mlx")
    for f in files:
        if 'mimi' in f.lower():
            print(f"  - {f}")
except:
    print("  Repo not found")

print("\nChecking nvidia/personaplex-7b-v1 again:")
files = api.list_repo_files("nvidia/personaplex-7b-v1")
for f in files:
    print(f"  - {f}")
EOF
```

---

## 5. Download Required Files

### Download Base Model Files (Tokenizer, Voices)

```bash
python3 << 'EOF'
from huggingface_hub import hf_hub_download

print("Downloading base model files...")

# Download tokenizer from base model
tokenizer_path = hf_hub_download("nvidia/personaplex-7b-v1", "tokenizer_spm_32k_3.model")
print(f"✓ Tokenizer: {tokenizer_path}")

# Download tokenizer embeddings (backup, might be needed)
tokenizer_emb_path = hf_hub_download("nvidia/personaplex-7b-v1", "tokenizer-e351c8d8-checkpoint125.safetensors")
print(f"✓ Tokenizer embeddings: {tokenizer_emb_path}")

# Download voices for voice prompts
voices_path = hf_hub_download("nvidia/personaplex-7b-v1", "voices.tgz")
print(f"✓ Voices: {voices_path}")

print("\n✓ Base files downloaded!")
EOF
```

### Download Your Fine-Tuned Model

```bash
# Replace YOUR_USERNAME/YOUR_REPO with your actual HuggingFace repo
python3 << 'EOF'
from huggingface_hub import hf_hub_download

# ⚠️ UPDATE THIS LINE WITH YOUR REPO ⚠️
HF_REPO = "YOUR_USERNAME/YOUR_REPO"  # e.g., "AnthrolyticB/personaplex-lora-merged"

print(f"Downloading fine-tuned model from {HF_REPO}...")

# Download your merged LoRA model
# The file might be named: model.safetensors, merged_model.safetensors, or similar
# Check your repo first if unsure
model_path = hf_hub_download(HF_REPO, "model.safetensors")  # or "merged_model.safetensors"
print(f"✓ Fine-tuned model: {model_path}")

print("\n✓ Your model downloaded!")
EOF
```

### Download Mimi Codec

**⚠️ TO BE DETERMINED - Run step 4 first to find the correct Mimi location**

Once you know where Mimi is, download it:

```bash
# Example (update based on step 4 results):
python3 << 'EOF'
from huggingface_hub import hf_hub_download

# UPDATE THIS BASED ON STEP 4 FINDINGS
mimi_path = hf_hub_download("REPO_NAME_HERE", "mimi_weight.safetensors")
print(f"✓ Mimi codec: {mimi_path}")
EOF
```

---

## 6. Find Downloaded File Paths

```bash
# Find exact paths for all downloaded files
MOSHI_PATH=$(find /root/.cache/huggingface -name "model.safetensors" -o -name "merged_model.safetensors" | grep -v nvidia | head -1)
MIMI_PATH=$(find /root/.cache/huggingface -name "*mimi*.safetensors" | head -1)
TOKENIZER_PATH=$(find /root/.cache/huggingface -name "tokenizer_spm_32k_3.model" | head -1)

echo "=== Downloaded Files ==="
echo "Model:     $MOSHI_PATH"
echo "Mimi:      $MIMI_PATH"
echo "Tokenizer: $TOKENIZER_PATH"
echo "======================="
```

---

## 7. Run the Server

```bash
python -m moshi.server \
  --moshi-weight "$MOSHI_PATH" \
  --mimi-weight "$MIMI_PATH" \
  --tokenizer "$TOKENIZER_PATH" \
  --host 0.0.0.0 \
  --port 8998
```

**Alternative: If you get path errors, use absolute paths:**

```bash
python -m moshi.server \
  --moshi-weight /root/.cache/huggingface/hub/models--YOUR_USERNAME--YOUR_REPO/snapshots/HASH/model.safetensors \
  --mimi-weight /root/.cache/huggingface/hub/models--MIMI_REPO/snapshots/HASH/mimi_weight.safetensors \
  --tokenizer /root/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/HASH/tokenizer_spm_32k_3.model \
  --host 0.0.0.0 \
  --port 8998
```

---

## 8. Access the Web UI

The server will print:
```
Access the Web UI directly at http://YOUR_RUNPOD_IP:8998
```

1. Copy that URL
2. Open in your browser
3. You'll see the PersonaPlex chat interface

---

## 9. Test the Wait-for-User Feature

### How It Works Now:

1. **Connect** to the chat
2. **System prompts load** (voice + text prompt)
3. **Server logs**: `"System prompts loaded. Waiting for user to speak first..."`
4. **Model stays silent** (no more random generation!)
5. **You speak naturally**: "Hi! Welcome to Morning Brew, what can I get you today?"
6. **Server detects speech**: `"User started speaking - agent will now respond"`
7. **Model responds** as the customer

✨ **No more rushing to say "Welcome to Morning Brew"!** ✨

---

## 10. System Prompt Examples

Try different customer personas in the text prompt field:

### Angry Customer
```
You are an angry customer who has been waiting 15 minutes for a mobile order that still isn't ready
```

### Friendly Regular
```
You are a friendly regular customer who visits every morning and always orders a large oat milk latte
```

### Nervous First-Timer
```
You are a nervous first-time customer who doesn't know what to order and needs help
```

### Indecisive Customer
```
You are an indecisive customer who keeps changing their mind about what they want
```

### In-a-Rush Customer
```
You are a busy professional in a rush who needs to order quickly and leave
```

---

## 11. Quick Start Script

Save as `/workspace/run_personaplex.sh`:

```bash
#!/bin/bash
cd /workspace/personaplex-fine-coffee

# Find paths
MOSHI_PATH=$(find /root/.cache/huggingface -name "model.safetensors" -o -name "merged_model.safetensors" | grep -v nvidia | head -1)
MIMI_PATH=$(find /root/.cache/huggingface -name "*mimi*.safetensors" | head -1)
TOKENIZER_PATH=$(find /root/.cache/huggingface -name "tokenizer_spm_32k_3.model" | head -1)

echo "=== Starting PersonaPlex Server ==="
echo "Model:     $MOSHI_PATH"
echo "Mimi:      $MIMI_PATH"
echo "Tokenizer: $TOKENIZER_PATH"
echo "===================================="

python -m moshi.server \
  --moshi-weight "$MOSHI_PATH" \
  --mimi-weight "$MIMI_PATH" \
  --tokenizer "$TOKENIZER_PATH" \
  --host 0.0.0.0 \
  --port 8998
```

Make executable and run:
```bash
chmod +x /workspace/run_personaplex.sh
/workspace/run_personaplex.sh
```

---

## Troubleshooting

### Issue: "Model file not found"
```bash
# List all downloaded models
find /root/.cache/huggingface -name "*.safetensors" -type f
```

### Issue: "Mimi codec not found"
Run step 4 again to locate Mimi, or check if it's bundled in the main model.

### Issue: "CUDA out of memory"
Use a larger GPU (A6000 or A100) or enable CPU offload:
```bash
python -m moshi.server --cpu-offload ...
```

### Issue: "Model not responding after system prompts"
Check server logs for: `"Waiting for user to speak first..."`
Then check if VAD threshold is too high (adjust in `server.py` line 222: `chunk_rms > 0.01`)

---

## What Changed?

### New Feature: Wait-for-User Mode

**Before:**
- System prompts load
- Model immediately starts generating
- User had to quickly say "Welcome to Morning Brew" to snap it into place
- Model would hallucinate if user didn't speak fast enough

**After:**
- System prompts load
- Model waits silently (suppresses agent audio output)
- User speaks naturally when ready
- VAD detects user speech → model starts responding
- No more hallucinations or rushed greetings!

### Code Changes:
- `moshi/models/lm.py`: Added `waiting_for_first_user_input` flag
- `moshi/server.py`: Added simple VAD (Voice Activity Detection)
- Agent audio suppressed until user speaks

---

## Next Steps

1. ✅ Complete steps 1-7 to get server running
2. 🧪 Test with different personas
3. 🔧 Adjust VAD threshold if needed (`server.py` line 222)
4. 📊 Compare to base model behavior
5. 🚀 Iterate and improve!

---

**Questions?** Check the troubleshooting section or review server logs for errors.
