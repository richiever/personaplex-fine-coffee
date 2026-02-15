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

huggingface-cli download AnthrolyticB/personaplex-training-data-test training.json \
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

## Part 5: Copy Pipeline Script (1 minute)

### 5.1 Copy to Workspace Root

**Why:** Easier to run from /workspace without path issues

```bash
cp /workspace/personaplex-fine-coffee/runpod_pipeline.py /workspace/
```

**Verify:**
```bash
ls -lh /workspace/runpod_pipeline.py
# Should exist and be ~20KB
```

---

## Part 6: Generate .pt Training Files (10 minutes)

### 6.1 Run Pipeline

**Full pipeline (downloads voices + generates TTS + encodes):**
```bash
cd /workspace
python runpod_pipeline.py
```

**Skip steps you've already done:**
```bash
# If you already have voices and audio:
python runpod_pipeline.py --skip-voices --skip-tts

# If you just need to regenerate .pt files:
python runpod_pipeline.py --skip-conversations --skip-voices --skip-tts
```

### 6.2 Monitor Progress

**Expected output:**
```
============================================================
STEP 1: Download training.json
============================================================
  Already have training.json (200 conversations)

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
  [2/200] conv_0001: Generating 18 turns...
  ...
  ✓ Generated audio for 200 conversations

============================================================
STEP 4: Encode with Mimi + assemble .pt files
============================================================
  Loading Mimi encoder...
  [1/200] conv_0000... 755 frames (60.4s)
  [2/200] conv_0001... 730 frames (58.4s)
  ...
  ✓ Created 200 .pt files

Done! .pt files saved to /workspace/pt_files/
```

### 6.3 Verify .pt Files

```bash
ls -lh /workspace/pt_files/*.pt | head -5
# Should see: conv_0000.pt, conv_0001.pt, etc.

# Count files:
ls /workspace/pt_files/*.pt | wc -l
# Should output: 200

# Check tensor shape:
python << 'EOF'
import torch
sample = torch.load('/workspace/pt_files/conv_0000.pt')
print(f"Shape: {sample.shape}")
print(f"Streams: {sample.shape[0]} (expected: 17)")
print(f"Frames: {sample.shape[1]}")
EOF
```

**Expected output:**
```
Shape: torch.Size([17, 755])
Streams: 17 (expected: 17)
Frames: 755
```

---

## Part 7: Download LoRA Training Script (1 minute)

### 7.1 Check if Script Exists

```bash
ls -lh /workspace/lora_train.py
```

**If missing, copy from repo:**
```bash
cp /workspace/personaplex-fine-coffee/legacy_v1/lora_train.py /workspace/
# OR download directly:
wget -O /workspace/lora_train.py https://raw.githubusercontent.com/richiever/personaplex-fine-coffee/main/lora_train.py
```

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

## Part 9: Train PersonaPlex Model (20 minutes)

### 9.1 Start Training

```bash
cd /workspace

python lora_train.py \
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

### 9.2 Monitor Training

**Expected output:**
```
================================================================================
PersonaPlex LoRA Fine-Tuning
================================================================================

Config:
  Base model: nvidia/personaplex-7b-v1
  Dataset: /workspace/pt_files (200 files)
  Epochs: 2
  LoRA rank: 16
  LoRA alpha: 32
  Learning rate: 2e-6

Downloading PersonaPlex weights...
Loading model...
Applying LoRA...
trainable params: 18,874,368 || all params: 7,118,874,368 || trainable%: 0.265

Starting training...

Epoch 1/2:
  [1/200] Loss: 45.23
  [50/200] Loss: 38.17
  [100/200] Loss: 36.42
  [150/200] Loss: 35.01
  [200/200] Loss: 34.85
  Avg Epoch Loss: 37.21

Epoch 2/2:
  [1/200] Loss: 34.12
  [50/200] Loss: 31.45
  [100/200] Loss: 29.87
  [150/200] Loss: 28.93
  [200/200] Loss: 28.41
  Avg Epoch Loss: 30.56

Training complete!
Saved LoRA weights to: /workspace/lora_weights.safetensors
```

### 9.3 Expected Loss Curves

**Good training:**
- Epoch 1: ~45 → ~35
- Epoch 2: ~35 → ~28-30

**Warning signs:**
- Loss < 20: **Overfitting** (reduce epochs or increase dropout)
- Loss > 40 after epoch 2: **Undertrained** (increase epochs or learning rate)
- Loss oscillating wildly: **Learning rate too high**

### 9.4 Training Time

- RTX A6000: ~15-20 minutes
- RTX 4090: ~18-22 minutes
- RTX 3090: ~22-25 minutes

---

## Part 10: Merge LoRA Weights (2 minutes)

### 10.1 Run Merge

```bash
python /workspace/lora_train.py \
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

### 10.2 Verify Merged Model

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

## Part 11: Serve Model (1 minute)

### 11.1 Kill Existing Server

```bash
pkill -f "moshi.server"
```

### 11.2 Start Server with Merged Model

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

### 11.3 Check Server Logs

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

## Part 12: Access Web UI

### 12.1 Get RunPod Public URL

**In RunPod dashboard:**
1. Click on your pod
2. Under "Connection Details", find: `TCP Port Mappings`
3. Look for port 8998 → should show external URL

**Example URL:**
```
https://abc123def-8998.proxy.runpod.net
```

### 12.2 Test Without System Prompt (Baseline)

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

## Part 13: Test with System Prompts

**CRITICAL:** You MUST include `?text_prompt=` in URL to test fine-tuning!

### 13.1 Test Angry Customer

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

### 13.2 Test Nervous Customer

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

### 13.3 Test Friendly Regular

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

## Part 14: Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1:** Use smaller batch size
```bash
# Edit lora_train.py, find:
# BATCH_SIZE = 1
# Already at minimum, use CPU offload instead
```

**Solution 2:** Enable CPU offload
```bash
# Add to training command:
python lora_train.py \
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
python lora_train.py \
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
3. Is training loss reasonable (~28-30)?

**Solutions:**
1. Always test WITH text_prompt parameter
2. Make system prompts more specific:
   ```
   BAD:  "You are an angry customer"
   GOOD: "You are an angry customer at [CAFE NAME]. Menu: [ITEMS]. You ordered [X] but got [Y] instead. [SPECIFIC COMPLAINT]. Only discuss your order."
   ```
3. If loss is good but still hallucinating, increase epochs to 3

---

### Issue: "Repo id must be in the form 'org/repo'"

**Diagnosis:** LOCAL-TRAINING-PATCH not applied

**Solution:** Re-run Step 8.1

---

### Issue: "Some tensors share memory, this will lead to errors"

**Diagnosis:** SHARED-TENSORS-PATCH not applied

**Solution:** Re-run Step 8.2

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

## Part 15: Upload Model to HuggingFace (Optional)

### 15.1 Create HuggingFace Repo

1. Go to https://huggingface.co/new
2. Name: `personaplex-coffee-shop` (or your choice)
3. Type: Model
4. Public or Private

### 15.2 Upload Merged Model

```bash
huggingface-cli upload \
  YOUR_USERNAME/personaplex-coffee-shop \
  /workspace/merged_model.safetensors \
  merged_model.safetensors \
  --repo-type model
```

### 15.3 Upload Training Data (Optional)

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
| Part 4: Download data | 2 min | 12 min |
| Part 5: Copy pipeline | 1 min | 13 min |
| Part 6: Generate .pt files | 10 min | 23 min |
| Part 7-8: Patches | 3 min | 26 min |
| Part 9: Training | 20 min | 46 min |
| Part 10: Merge | 2 min | 48 min |
| Part 11-12: Serve & test | 2 min | 50 min |

**Total: ~50 minutes from RunPod launch to testing**

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
huggingface-cli download AnthrolyticB/personaplex-training-data-test training.json \
  --repo-type dataset --local-dir /workspace --force-download

# Part 5-6
cp /workspace/personaplex-fine-coffee/runpod_pipeline.py /workspace/
cd /workspace
python runpod_pipeline.py --skip-voices --skip-tts

# Part 7-8 (patches - run the Python scripts from guide)

# Part 9
python /workspace/lora_train.py \
  --dataset-repo /workspace/pt_files \
  --epochs 2 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lr 2e-6

# Part 10
python /workspace/lora_train.py \
  --merge-only \
  --lora-weights /workspace/lora_weights.safetensors \
  --lora-rank 16 \
  --lora-alpha 32

# Part 11
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
