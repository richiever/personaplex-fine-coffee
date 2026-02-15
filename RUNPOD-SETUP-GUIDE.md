# Complete RunPod Setup Guide for PersonaPlex Fine-Tuning

**From GitHub to trained model in ~30 minutes**

---

## Prerequisites

- RunPod account with credits
- HuggingFace account with token
- GitHub repository: `https://github.com/richiever/personaplex-fine-coffee`

---

## Part 1: Launch RunPod Instance

### 1.1 Create Pod

1. Go to https://runpod.io
2. Click "Deploy" → "GPU Pods"
3. Select GPU:
   - **Recommended:** RTX A6000 (48GB VRAM) - $0.79/hr
   - **Minimum:** RTX 4090 (24GB VRAM) - $0.69/hr
   - **Budget:** RTX 3090 (24GB VRAM) - $0.44/hr
4. Template: **PyTorch 2.0+**
5. Disk: **50GB minimum** (100GB recommended)
6. Click "Deploy"

### 1.2 Connect to Pod

Once pod is running:

1. Click "Connect" → "Start Web Terminal"
2. Or use SSH: Click "Connect via SSH" and copy the command

---

## Part 2: Initial Setup (5 minutes)

### 2.1 Fix PyTorch Version

**Why:** PersonaPlex requires torch 2.6.0 for compatibility

```bash
pip uninstall torch torchaudio torchvision -y
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0
```

**Verify:**
```bash
python -c "import torch; print(torch.__version__)"
# Should output: 2.6.0
```

### 2.2 Set HuggingFace Token

```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Make it persistent across sessions:
echo 'export HF_TOKEN="hf_YOUR_TOKEN_HERE"' >> ~/.bashrc
```

**Get your token:**
1. Go to https://huggingface.co/settings/tokens
2. Create token with "Read" access
3. Accept PersonaPlex license: https://huggingface.co/nvidia/personaplex-7b-v1

---

## Part 3: Clone Repository & Install Dependencies (3 minutes)

### 3.1 Clone Repo

```bash
cd /workspace
git clone https://github.com/richiever/personaplex-fine-coffee.git
cd personaplex-fine-coffee
```

**Verify:**
```bash
ls -la
# Should see: runpod_pipeline.py, moshi/, client/, README.md, etc.
```

### 3.2 Install Moshi

```bash
cd /workspace/personaplex-fine-coffee
pip install ./moshi
```

**Expected output:**
```
Successfully installed moshi-0.0.1 ...
```

### 3.3 Install Additional Dependencies

```bash
pip install chatterbox-tts huggingface_hub sentencepiece peft
```

**For Blackwell GPUs (GB200, etc.):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

---

## Part 4: Download Training Data (2 minutes)

### 4.1 Download Conversations

```bash
cd /workspace

huggingface-cli download AnthrolyticB/personaplex-training-data-v2 training.json \
  --repo-type dataset \
  --local-dir /workspace \
  --force-download
```

**Verify:**
```bash
ls -lh /workspace/training.json
# Should see: training.json (XX KB)

# Check conversation count:
python -c "import json; print(len(json.load(open('/workspace/training.json'))))"
# Should output: 200 (or whatever your dataset size is)
```

### 4.2 Inspect Training Data (Optional)

```bash
python << 'EOF'
import json
with open('/workspace/training.json') as f:
    data = json.load(f)

print(f"Total conversations: {len(data)}")
print("\nSample conversation:")
conv = data[0]
print(f"  ID: {conv['conversation_id']}")
print(f"  System Prompt: {conv.get('system_prompt', 'N/A')[:100]}...")
print(f"  Turns: {len(conv['turns'])}")
print(f"  First turn: {conv['turns'][0]}")
EOF
```

---

## Part 5: Download Pre-Generated .pt Training Files (2 minutes)

### 5.1 Download from HuggingFace

**Pre-generated .pt files are ready to download** - saves 10-15 minutes!

```bash
cd /workspace
mkdir -p pt_files

huggingface-cli download AnthrolyticB/personaplex-training-data-v2 \
  --include "pt_files/*.pt" \
  --repo-type dataset \
  --local-dir /workspace \
  --force-download
```

**What this downloads:**
- 200 pre-encoded .pt files (conversation tensors)
- Already aligned with fixed text-audio clustering
- Ready for training immediately

### 5.2 Verify Downloaded Files

```bash
# Count files
ls /workspace/pt_files/*.pt | wc -l
# Should output: 200

# Check file sizes
ls -lh /workspace/pt_files/*.pt | head -5
# Should see: conv_0000.pt, conv_0001.pt, etc. (~500KB-2MB each)

# Verify tensor structure
python << 'EOF'
import torch
sample = torch.load('/workspace/pt_files/conv_0000.pt')
print(f"Shape: {sample.shape}")
print(f"Streams: {sample.shape[0]} (expected: 17)")
print(f"Frames: {sample.shape[1]}")
print(f"✓ .pt files are valid!")
EOF
```

**Expected output:**
```
Shape: torch.Size([17, 755])
Streams: 17 (expected: 17)
Frames: 755
✓ .pt files are valid!
```

---

## Part 6: Alternative - Generate .pt Files Yourself (Optional)

**Skip this section if you downloaded pre-generated files in Part 5!**

<details>
<summary><b>Click to expand: Generate .pt files from scratch (10-15 minutes)</b></summary>

### Why generate yourself?
- You modified the pipeline code (text-audio alignment)
- You want to use different training conversations
- You want to customize voice samples

### 6.1 Copy Pipeline Script

```bash
cp /workspace/personaplex-fine-coffee/runpod_pipeline.py /workspace/
```

### 6.2 Run Pipeline

```bash
cd /workspace
python runpod_pipeline.py
```

**Expected output:**
```
============================================================
STEP 1: Download training.json
============================================================
  Downloading from HuggingFace...
  ✓ Downloaded training.json (200 conversations)

============================================================
STEP 2: Download voice samples
============================================================
  Downloading 100 agent voices from LibriSpeech...
  Downloading 100 user voices from LibriSpeech...
  ✓ Downloaded 200 voice samples

============================================================
STEP 3: Generate TTS audio
============================================================
  Initializing Chatterbox TTS...
  [1/200] conv_0000: Generating 15 turns...
  ...
  ✓ Generated audio for 200 conversations

============================================================
STEP 4: Encode with Mimi + assemble .pt files
============================================================
  Loading Mimi encoder...
  [1/200] conv_0000... 755 frames (60.4s)
  ...
  ✓ Created 200 .pt files

Done! .pt files saved to /workspace/pt_files/
```

**Skip steps if you already have data:**
```bash
# Skip voice download (if /workspace/voices/ exists)
python runpod_pipeline.py --skip-voices

# Skip TTS generation (if /workspace/audio/ exists)
python runpod_pipeline.py --skip-voices --skip-tts

# Only regenerate .pt files (if you changed alignment code)
python runpod_pipeline.py --skip-conversations --skip-voices --skip-tts
```

</details>

---

## Part 7: Copy LoRA Training Script (1 minute)

### 7.1 Copy Fixed Training Script

**Use the new semantic-weighted loss training script:**

```bash
cp /workspace/personaplex-fine-coffee/lora_train_fixed.py /workspace/
```

**Verify:**
```bash
ls -lh /workspace/lora_train_fixed.py
# Should exist and be ~25KB

# Check for semantic weighting (key feature):
grep -n "SEMANTIC_WEIGHT" /workspace/lora_train_fixed.py
# Should show line with "SEMANTIC_WEIGHT = 100.0"
```

**What's different in lora_train_fixed.py:**
- ✅ Semantic-weighted loss (100× for semantic codebook, 1× for acoustic)
- ✅ Better loss tracking (separate text/semantic/acoustic metrics)
- ✅ Improved LoRA application
- ✅ Fixed safetensors handling

---

## Part 8: Apply Training Patches (2 minutes)

### 8.1 LOCAL-TRAINING-PATCH

**Why:** Allows using local .pt files instead of downloading from HuggingFace

```bash
python3 << 'EOF'
import os
from pathlib import Path

with open('/workspace/lora_train.py', 'r') as f:
    content = f.read()

if 'Check if repo_id is a local directory' not in content:
    old_code = '''def download_training_data(repo_id, local_dir):
    """Download all .pt files from a HuggingFace dataset repo."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing files in {repo_id}...")
    files = list_repo_files(repo_id, repo_type="dataset")'''

    new_code = '''def download_training_data(repo_id, local_dir):
    """Download all .pt files from a HuggingFace dataset repo, or use local directory."""
    local_dir = Path(local_dir)

    # Check if repo_id is a local directory
    if os.path.exists(repo_id) and os.path.isdir(repo_id):
        print(f"Using local .pt files from {repo_id}...")
        pt_files = sorted(Path(repo_id).glob("*.pt"))
        print(f"Found {len(pt_files)} local .pt files")
        return pt_files

    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing files in {repo_id}...")
    files = list_repo_files(repo_id, repo_type="dataset")'''

    content = content.replace(old_code, new_code)
    with open('/workspace/lora_train.py', 'w') as f:
        f.write(content)
    print("✓ LOCAL-TRAINING-PATCH applied!")
else:
    print("✓ Already patched")
EOF
```

### 8.2 SHARED-TENSORS-PATCH

**Why:** Fixes safetensors error when saving LoRA weights

```bash
python3 << 'EOF'
with open('/workspace/lora_train.py', 'r') as f:
    content = f.read()

if 'Clone shared tensors to avoid safetensors error' not in content:
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if 'save_file(state_dict, output)' in line and line.strip().startswith('save_file'):
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(f'{indent}# Clone shared tensors to avoid safetensors error')
            new_lines.append(f'{indent}state_dict = {{k: v.clone() for k, v in state_dict.items()}}')
            new_lines.append(line)
        else:
            new_lines.append(line)
    with open('/workspace/lora_train.py', 'w') as f:
        f.write('\n'.join(new_lines))
    print("✓ SHARED-TENSORS-PATCH applied!")
else:
    print("✓ Already patched")
EOF
```

**Verify both patches:**
```bash
grep -n "Check if repo_id is a local directory" /workspace/lora_train.py
grep -n "Clone shared tensors" /workspace/lora_train.py
# Both should show line numbers (patches applied)
```

---

## Part 8: Train PersonaPlex Model (20 minutes)

### 8.1 Start Training with Semantic-Weighted Loss

```bash
cd /workspace

python lora_train_fixed.py \
  --dataset-repo /workspace/pt_files \
  --epochs 2 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lr 2e-6
```

**Training hyperparameters explained:**
- `--epochs 2`: Two full passes through data (balance between underfit/overfit)
- `--lora-rank 16`: LoRA rank (16 is sweet spot, 32 overfits)
- `--lora-alpha 32`: LoRA scaling (2× rank is standard)
- `--lr 2e-6`: Learning rate (conservative for fine-tuning)

**What's different with semantic weighting:**
- Semantic codebook (meaning): 100× weight
- Acoustic codebooks (sound quality): 1× weight
- Model learns WHAT to say before perfecting HOW it sounds
- Reduces hallucinations by prioritizing content coherence

### 8.2 Monitor Training

**Expected output with semantic weighting:**
```
================================================================================
PersonaPlex LoRA Fine-Tuning (Semantic-Weighted Loss)
================================================================================

Config:
  Base model: nvidia/personaplex-7b-v1
  Dataset: /workspace/pt_files (200 files)
  Epochs: 2
  LoRA rank: 16
  LoRA alpha: 32
  Learning rate: 2e-6
  Semantic weight: 100.0
  Acoustic weight: 1.0

Downloading PersonaPlex weights...
Loading model...
Applying LoRA...
trainable params: 18,874,368 || all params: 7,118,874,368 || trainable%: 0.265

Starting training...

Epoch 1/2:
  [50/200] loss=120.45, text=10.2, sem=100.3, ac=30.1
  [100/200] loss=85.23, text=8.5, sem=65.8, ac=28.7
  [150/200] loss=60.12, text=7.1, sem=42.9, ac=27.3
  [200/200] loss=48.31, text=6.2, sem=32.1, ac=26.5
  Avg Epoch Loss: 78.53

Epoch 2/2:
  [50/200] loss=40.15, text=5.8, sem=24.7, ac=25.9
  [100/200] loss=28.90, text=4.9, sem=15.2, ac=25.1
  [150/200] loss=22.34, text=4.3, sem=10.8, ac=24.7
  [200/200] loss=18.12, text=4.0, sem=8.1, ac=24.5
  Avg Epoch Loss: 27.38

Training complete!
Saved LoRA weights to: /workspace/lora_weights.safetensors
```

**Key metrics explained:**
- `loss`: Total loss (text + semantic + acoustic)
- `text`: Text token prediction loss
- `sem`: Semantic codebook loss (100× weighted)
- `ac`: Acoustic codebooks loss (1× weighted, averaged)

### 8.3 Expected Loss Curves with Semantic Weighting

**Good training (NEW expectations):**
- Epoch 1: Total ~120 → ~48 (semantic drops fastest: 100 → 32)
- Epoch 2: Total ~48 → ~18 (semantic drops to ~8, acoustic stable ~25)

**What this means:**
- ✅ **High initial loss (~120)** is EXPECTED due to 100× semantic weighting
- ✅ **Semantic drops much faster** than acoustic (70-80% reduction vs 15-20%)
- ✅ **Acoustic stays relatively stable** (~30 → ~25) - quality preserved
- ✅ **Text loss drops moderately** (~10 → ~4) - better alignment helps

**Warning signs:**
- Semantic loss < 5 after epoch 2: **Possible overfitting** (but check validation)
- Semantic loss > 40 after epoch 2: **Undertrained** (increase epochs or LR)
- Acoustic loss increasing: **Problem** (should stay stable or decrease slightly)
- Total loss oscillating wildly: **Learning rate too high**

### 8.4 Training Time

- RTX A6000: ~15-20 minutes (same as before)
- RTX 4090: ~18-22 minutes (same as before)
- RTX 3090: ~22-25 minutes (same as before)

**Note:** Semantic weighting adds minimal overhead (~5%) - slightly more complex loss computation but same overall training time.

---

## Part 9: Merge LoRA Weights (2 minutes)

### 9.1 Run Merge

```bash
python /workspace/lora_train_fixed.py \
  --merge-only \
  --lora-weights /workspace/lora_weights.safetensors \
  --lora-rank 16 \
  --lora-alpha 32
```

**Expected output:**
```
================================================================================
Merging LoRA Weights
================================================================================

Loading base model: nvidia/personaplex-7b-v1...
Loading LoRA weights: /workspace/lora_weights.safetensors...
Merging...
Saving merged model to: /workspace/merged_model.safetensors...

✓ Merge complete!
Final model: /workspace/merged_model.safetensors (14.2 GB)
```

### 9.2 Verify Merged Model

```bash
ls -lh /workspace/merged_model.safetensors
# Should be ~14-15 GB

# Quick integrity check:
python << 'EOF'
from safetensors import safe_open
tensors = {}
with safe_open("/workspace/merged_model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)
print(f"✓ Loaded {len(tensors)} tensors")
print(f"Sample keys: {list(tensors.keys())[:5]}")
EOF
```

---

## Part 10: Serve Model (1 minute)

### 10.1 Kill Existing Server

```bash
pkill -f "moshi.server"
```

### 10.2 Start Server with Merged Model

```bash
python -m moshi.server \
  --hf-repo kyutai/moshika-pytorch-bf16 \
  --moshi-weight /workspace/merged_model.safetensors \
  --host 0.0.0.0 \
  --port 8998 &
```

**Important flags:**
- `--hf-repo kyutai/moshika-pytorch-bf16`: Base weights (Mimi codec, tokenizer)
- `--moshi-weight /workspace/merged_model.safetensors`: Your fine-tuned weights
- `--host 0.0.0.0`: Allow external connections
- `--port 8998`: Default PersonaPlex port

### 10.3 Check Server Logs

```bash
tail -f /tmp/moshi_server.log
```

**Expected output:**
```
Starting Moshi server...
Loading model from /workspace/merged_model.safetensors...
Server running on 0.0.0.0:8998
```

---

## Part 11: Access Web UI

### 11.1 Get RunPod Public URL

**In RunPod dashboard:**
1. Click on your pod
2. Under "Connection Details", find: `TCP Port Mappings`
3. Look for port 8998 → should show external URL

**Example URL:**
```
https://abc123def-8998.proxy.runpod.net
```

### 11.2 Test Without System Prompt (Baseline)

Open browser to:
```
https://abc123def-8998.proxy.runpod.net
```

**Expected behavior:**
- Should load web UI
- Mic access prompt appears
- After 440Hz tone, you can speak
- Model responds in default persona (conversational)

---

## Part 12: Test with System Prompts

**CRITICAL:** You MUST include `?text_prompt=` in URL to test fine-tuning!

**Expected improvements with semantic weighting:**
- ✅ Better instruction following (model prioritizes WHAT to say)
- ✅ Reduced hallucinations (semantic coherence weighted 100×)
- ✅ Stays on topic (coffee shop only)
- ✅ Maintains persona consistency
- ⚪ Slight quality tradeoff OK (acoustic weighted 1×, but still good)

### 12.1 Test Angry Customer

```
https://abc123def-8998.proxy.runpod.net/?text_prompt=You are an angry customer at Morning Grind cafe. Menu: Latte ($4.50), Americano ($3.75). You ordered a latte but got whole milk instead of oat milk and you're lactose intolerant. Demand an immediate remake. Start irritated and escalate to furious. Interrupt with phrases like "Just remake it!" Only discuss your order and complaint.
```

**Testing protocol:**
1. Open URL in browser
2. Grant mic access
3. Wait for 440Hz tone (model ready)
4. Speak as barista: "Hi, welcome to Morning Grind! What can I get for you?"
5. Listen to model response

**✓ Good response:**
- Sounds angry
- Mentions the wrong milk order
- Demands remake
- Stays on topic (coffee only)
- No hallucinations about unrelated topics

**✗ Bad response:**
- Friendly/neutral tone (didn't follow prompt)
- Talks about weather, politics, etc. (hallucinating)
- Doesn't mention the order issue
- Long rambling unrelated to coffee

### 12.2 Test Nervous Customer

```
https://abc123def-8998.proxy.runpod.net/?text_prompt=You are a nervous customer at Coffee Corner. Menu: Americano ($3.75), Latte ($4.50). You are intimidated by the fancy menu and don't understand terms like "macchiato". You speak hesitantly with frequent pauses and apologize often. Focus only on ordering and understanding the menu.
```

**Test dialogue:**
- You: "Hi! What can I help you with today?"
- Model: *[Should sound nervous, hesitant, apologetic]*

**✓ Good response:**
- Nervous tone with pauses
- Asks clarifying questions
- Apologizes
- Stays focused on menu

### 12.3 Test Friendly Regular

```
https://abc123def-8998.proxy.runpod.net/?text_prompt=You are a regular customer at Brew & Bean. Menu: Black Coffee ($3), Latte ($4.50). You are an upbeat regular ordering a large black coffee to go. Your speech is casual and warm with natural conversational flow. Stay focused on coffee and the menu.
```

**Test dialogue:**
- You: "Hey! Welcome back!"
- Model: *[Should sound friendly, casual, upbeat]*

**✓ Good response:**
- Warm, friendly tone
- Orders confidently (knows what they want)
- Natural conversational flow
- Stays on coffee topic

---

## Part 13: Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1:** Use smaller batch size
```bash
# Edit lora_train_fixed.py, find:
# BATCH_SIZE = 1
# Already at minimum, use CPU offload instead
```

**Solution 2:** Enable CPU offload
```bash
# Add to training command:
python lora_train_fixed.py \
  --dataset-repo /workspace/pt_files \
  --epochs 2 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lr 2e-6 \
  --cpu-offload  # Add this flag
```

**Solution 3:** Use smaller GPU and reduce rank
```bash
# Use rank 8 instead of 16:
python lora_train_fixed.py \
  --dataset-repo /workspace/pt_files \
  --epochs 2 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lr 2e-6
```

---

### Issue: Model still hallucinating

**Diagnosis:**
1. Did you include `?text_prompt=` in URL? (Most common mistake!)
2. Is the system prompt detailed enough?
3. Is training loss reasonable? (Total ~18-20, Semantic ~8-10)

**Solutions:**
1. Always test WITH text_prompt parameter
2. Make system prompts more specific:
   ```
   BAD:  "You are an angry customer"
   GOOD: "You are an angry customer at [CAFE NAME]. Menu: [ITEMS]. You ordered [X] but got [Y] instead. [SPECIFIC COMPLAINT]. Only discuss your order."
   ```
3. If loss is good but still hallucinating, try:
   - Increase semantic weight to 150× or 200× (edit lora_train_fixed.py)
   - Increase epochs to 3
   - Regenerate training data with better persona consistency

---

### Issue: Loss values seem very high

**Diagnosis:** This is EXPECTED with semantic weighting!

**Solution:** Don't panic! Initial loss ~120 is normal because semantic codebook is weighted 100×. What matters:
- ✅ Loss decreases over time (120 → 18)
- ✅ Semantic drops faster than acoustic
- ✅ Model performance improves (test with system prompts)

---

### Issue: Torch version error

**Diagnosis:** Wrong PyTorch version

**Solution:** Re-run Part 2.1

---

### Issue: Server won't start / crashes

**Check logs:**
```bash
tail -100 /tmp/moshi_server.log
```

**Common causes:**
1. Port 8998 already in use: `pkill -f moshi.server`
2. Missing weights: Check `/workspace/merged_model.safetensors` exists
3. Corrupted model: Re-run merge (Part 10)

---

## Part 14: Upload Model to HuggingFace (Optional)

### 14.1 Create HuggingFace Repo

1. Go to https://huggingface.co/new
2. Name: `personaplex-coffee-shop` (or your choice)
3. Type: Model
4. Public or Private

### 14.2 Upload Merged Model

```bash
huggingface-cli upload \
  YOUR_USERNAME/personaplex-coffee-shop \
  /workspace/merged_model.safetensors \
  merged_model.safetensors \
  --repo-type model
```

### 14.3 Upload Training Data (Optional)

```bash
huggingface-cli upload \
  YOUR_USERNAME/personaplex-coffee-shop \
  /workspace/training.json \
  training.json \
  --repo-type model
```

---

## Complete Timeline

| Step | Duration | Cumulative |
|------|----------|------------|
| Launch RunPod | 2 min | 2 min |
| Part 2: Initial setup | 5 min | 7 min |
| Part 3: Clone & install | 3 min | 10 min |
| Part 4: Download training.json | 2 min | 12 min |
| Part 5: Download .pt files | 2 min | 14 min |
| Part 6: (Optional - skip if using pre-generated) | 0 min | 14 min |
| Part 7: Copy training script | 1 min | 15 min |
| Part 8: Training (semantic-weighted) | 20 min | 35 min |
| Part 9: Merge | 2 min | 37 min |
| Part 10-11: Serve & test | 2 min | 39 min |

**Total: ~39 minutes from RunPod launch to testing** ⚡

*Note: Using pre-generated .pt files saves ~10 minutes vs generating from scratch!*

---

## Quick Reference Commands

**Full workflow (copy-paste):**
```bash
# Part 2
pip uninstall torch torchaudio torchvision -y
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0
export HF_TOKEN="hf_YOUR_TOKEN"

# Part 3
cd /workspace
git clone https://github.com/richiever/personaplex-fine-coffee.git
cd personaplex-fine-coffee
pip install ./moshi
pip install chatterbox-tts huggingface_hub sentencepiece peft

# Part 4
huggingface-cli download AnthrolyticB/personaplex-training-data-v2 training.json \
  --repo-type dataset --local-dir /workspace --force-download

# Part 5 (Download pre-generated .pt files - FAST!)
huggingface-cli download AnthrolyticB/personaplex-training-data-v2 \
  --include "pt_files/*.pt" \
  --repo-type dataset \
  --local-dir /workspace \
  --force-download

# Part 6 (Skip - already have .pt files from Part 5)

# Part 7
cp /workspace/personaplex-fine-coffee/lora_train_fixed.py /workspace/

# Part 8 (Training with semantic-weighted loss)
python /workspace/lora_train_fixed.py \
  --dataset-repo /workspace/pt_files \
  --epochs 2 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lr 2e-6

# Part 9 (Merge)
python /workspace/lora_train_fixed.py \
  --merge-only \
  --lora-weights /workspace/lora_weights.safetensors \
  --lora-rank 16 \
  --lora-alpha 32

# Part 10 (Serve)
pkill -f "moshi.server"
python -m moshi.server \
  --hf-repo kyutai/moshika-pytorch-bf16 \
  --moshi-weight /workspace/merged_model.safetensors \
  --host 0.0.0.0 \
  --port 8998 &
```

---

## Next Steps After Successful Training

1. **Test all 4 moods:** angry, nervous, indecisive, friendly
2. **Compare to base model:** Run base PersonaPlex on port 8999 for A/B testing
3. **Iterate on data:** If hallucinations persist, refine training conversations
4. **Adjust hyperparameters:** Try rank 24 or 32 if underfitting
5. **Deploy to production:** Upload to HuggingFace, integrate with your app

---

## Support

- **GitHub Issues:** https://github.com/richiever/personaplex-fine-coffee/issues
- **PersonaPlex Paper:** https://arxiv.org/abs/2602.06053
- **RunPod Docs:** https://docs.runpod.io

---

**Guide Version:** 1.0
**Last Updated:** 2026-02-15
**Tested On:** RTX A6000, PyTorch 2.6.0
