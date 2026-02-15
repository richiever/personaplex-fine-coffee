#!/usr/bin/env python3
"""
Merge LoRA weights from raw .safetensors file (no re-training needed!)

Usage:
  python merge_from_safetensors.py
"""

import json
import shutil
import torch
from pathlib import Path

# Configuration (must match your training config)
LORA_RANK = 16
LORA_ALPHA = 32
TARGET_MODULES = [
    "self_attn.out_proj",
    "gating.linear_in",
    "gating.linear_out",
]

# Paths
BASE_MODEL_PATH = "/workspace/personaplex-model/model.safetensors"
LORA_WEIGHTS_PATH = "/workspace/lora_weights.safetensors"
LORA_ADAPTER_DIR = "/workspace/lora_adapter"
MERGED_OUTPUT = "/workspace/merged_model.safetensors"

print("=" * 80)
print("Merge LoRA from .safetensors File (No Re-Training!)")
print("=" * 80)

# Step 1: Create adapter directory
print("\n[1/5] Creating LoRA adapter directory...")
Path(LORA_ADAPTER_DIR).mkdir(parents=True, exist_ok=True)

# Step 2: Create adapter_config.json
print("[2/5] Creating adapter_config.json...")
adapter_config = {
    "alpha_pattern": {},
    "auto_mapping": None,
    "base_model_name_or_path": BASE_MODEL_PATH,
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": False,
    "init_lora_weights": True,
    "layers_pattern": None,
    "layers_to_transform": None,
    "loftq_config": {},
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": 0.1,
    "megatron_config": None,
    "megatron_core": "megatron.core",
    "modules_to_save": None,
    "peft_type": "LORA",
    "r": LORA_RANK,
    "rank_pattern": {},
    "revision": None,
    "target_modules": TARGET_MODULES,
    "task_type": "CAUSAL_LM",
    "use_dora": False,
    "use_rslora": False,
}

config_path = Path(LORA_ADAPTER_DIR) / "adapter_config.json"
with open(config_path, 'w') as f:
    json.dump(adapter_config, f, indent=2)
print(f"  ✓ Created: {config_path}")

# Step 3: Copy LoRA weights
print("[3/5] Copying LoRA weights...")
adapter_weights_path = Path(LORA_ADAPTER_DIR) / "adapter_model.safetensors"
shutil.copy(LORA_WEIGHTS_PATH, adapter_weights_path)
print(f"  ✓ Copied to: {adapter_weights_path}")

# Step 4: Load and merge
print("[4/5] Loading models and merging...")
from moshi.models import loaders
from peft import PeftModel

print("  Loading base model...")
lm = loaders.get_moshi_lm(BASE_MODEL_PATH, device='cuda', dtype=torch.bfloat16)

# Add missing method for PEFT compatibility
if not hasattr(lm, 'prepare_inputs_for_generation'):
    lm.prepare_inputs_for_generation = lambda *args, **kwargs: {}

print("  Loading LoRA adapter...")
lm = PeftModel.from_pretrained(lm, LORA_ADAPTER_DIR)

print("  Merging...")
merged_model = lm.merge_and_unload()

# Step 5: Save merged model
print("[5/5] Saving merged model...")
from safetensors.torch import save_file

state_dict = {k: v.clone() for k, v in merged_model.state_dict().items()}
save_file(state_dict, MERGED_OUTPUT)

print(f"\n✓ Merge complete!")
print(f"  Merged model: {MERGED_OUTPUT}")
print(f"\nServe with:")
print(f"  python -m moshi.server \\")
print(f"    --hf-repo kyutai/moshika-pytorch-bf16 \\")
print(f"    --moshi-weight {MERGED_OUTPUT}")
print("=" * 80)
