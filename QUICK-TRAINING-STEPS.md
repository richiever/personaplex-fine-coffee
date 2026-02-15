# Quick Training Steps (RunPod)

**Complete workflow from fresh RunPod to trained model**

---

## Step 1: Fix Torch Version
```bash
pip uninstall torch torchaudio torchvision -y
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0
```

---

## Step 2: Get Latest Code
```bash
cd /workspace
git clone https://github.com/richiever/personaplex-fine-coffee.git
cd personaplex-fine-coffee
git pull

# Copy pipeline
cp runpod_pipeline.py /workspace/
```

---

## Step 3: Download Training Data
```bash
huggingface-cli download AnthrolyticB/personaplex-training-data-test training.json \
  --repo-type dataset \
  --local-dir /workspace \
  --force-download
```

---

## Step 4: Generate .pt Files with Hybrid System Prompts (440Hz sine wave)
```bash
rm /workspace/pt_files/*.pt
python /workspace/runpod_pipeline.py --skip-conversations --skip-voices --skip-tts
```

**Expected output:**
```
[1/200] conv_0000... 755 frames (60.4s)
[2/200] conv_0001... 730 frames (58.4s)
...
Done! 200 .pt files in /workspace/pt_files/
```

---

## Step 5: Apply LOCAL-TRAINING-PATCH
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

---

## Step 6: Apply SHARED-TENSORS-PATCH
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

---

## Step 7: Train (rank 16, 2 epochs, ~20 min)
```bash
python /workspace/lora_train.py \
  --dataset-repo /workspace/pt_files \
  --epochs 2 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lr 2e-6
```

**Expected loss:**
- Epoch 1: ~45 → ~35
- Epoch 2: ~35 → ~28-30

**Warning signs:**
- Loss < 20: Overfitting (reduce epochs)
- Loss > 40 after epoch 2: Undertrained (increase epochs)

---

## Step 8: Merge
```bash
python /workspace/lora_train.py \
  --merge-only \
  --lora-weights /workspace/lora_weights.safetensors \
  --lora-rank 16 \
  --lora-alpha 32
```

**Output:** Creates `/workspace/merged_model.safetensors`

---

## Step 9: Serve
```bash
pkill -f "moshi.server"
python -m moshi.server \
  --hf-repo kyutai/moshika-pytorch-bf16 \
  --moshi-weight /workspace/merged_model.safetensors \
  --host 0.0.0.0 \
  --port 8998 &
```

**Check logs:**
```bash
tail -f /tmp/moshi_server.log
```

---

## Step 10: Test with System Prompt

**CRITICAL:** You MUST include `?text_prompt=` in the URL!

### Example Test URLs:

**Angry Customer:**
```
https://<pod-id>-8998.proxy.runpod.net/?text_prompt=You are an angry customer at Morning Grind cafe. Menu: Latte ($4.50), Americano ($3.75). You ordered a latte but got whole milk instead of oat milk and you're lactose intolerant. Demand an immediate remake. Start irritated and escalate to furious. Interrupt with phrases like "Just remake it!" Only discuss your order and complaint.
```

**Nervous Customer:**
```
https://<pod-id>-8998.proxy.runpod.net/?text_prompt=You are a nervous customer at Coffee Corner. Menu: Americano ($3.75), Latte ($4.50). You are intimidated by the fancy menu and don't understand terms like "macchiato". You speak hesitantly with frequent pauses and apologize often. Focus only on ordering and understanding the menu.
```

**Friendly Customer:**
```
https://<pod-id>-8998.proxy.runpod.net/?text_prompt=You are a regular customer at Brew & Bean. Menu: Black Coffee ($3), Latte ($4.50). You are an upbeat regular ordering a large black coffee to go. Your speech is casual and warm with natural conversational flow. Stay focused on coffee and the menu.
```

---

## Testing Protocol

1. Open test URL with system prompt
2. Wait for 440Hz tone (model ready)
3. Speak as barista: "Hi, welcome! What can I get for you?"
4. Model responds as customer with specified mood
5. Verify:
   - ✓ Stays in character
   - ✓ Stays on topic (coffee only)
   - ✓ No hallucinations about unrelated topics
   - ✓ Appropriate speech patterns for mood

---

## Timeline

- Step 1-6: ~5 minutes (setup)
- Step 7: ~15-20 minutes (training)
- Step 8: ~2 minutes (merge)
- Step 9-10: ~1 minute (serve + test)

**Total: ~25 minutes from start to testing**

---

## Key Improvements vs Previous Attempts

✓ **Hybrid System Prompt** with 440Hz sine wave (not silence!)
✓ **PAD tokens** for agent text during system prompt
✓ **SILENCE_TOKENS** for agent audio during system prompt
✓ **Proper delimiters** (`<system>` tags)
✓ **Rank 16** (not 32) to prevent overfitting
✓ **Local training** (no HuggingFace rate limits)
✓ **Detailed system prompts** (customer service format, not Fisher corpus)

---

## Troubleshooting

**Problem:** Model still hallucinating
**Solution:** Make sure you're testing WITH `?text_prompt=` in URL

**Problem:** "Repo id must be in the form..."
**Solution:** LOCAL-TRAINING-PATCH not applied (run Step 5)

**Problem:** "Some tensors share memory"
**Solution:** SHARED-TENSORS-PATCH not applied (run Step 6)

**Problem:** Torch version error
**Solution:** Re-run Step 1

---

## Next Steps After Training

1. Test all 4 moods (angry, nervous, indecisive, friendly)
2. Compare to base model (optional: run base on port 8999)
3. If good: Upload model with `/workspace/upload_model.py`
4. If bad: Iterate on training data or hyperparameters
