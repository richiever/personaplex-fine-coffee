# Debug PersonaPlex Layer Names

When LoRA fails with "Target modules not found", use this script to find the correct layer names.

## Problem

```
ValueError: Target modules {'self_attn.v_proj', 'mlp.gate_proj', ...} not found in the base model.
```

## Solution: Find Actual Layer Names

### On RunPod:

```bash
python << 'EOF'
from moshi.models import loaders

print("Loading PersonaPlex...")
lm = loaders.get_moshi_lm(
    '/workspace/personaplex-model/model.safetensors',
    device='cpu',
    dtype='bfloat16'
)

print("\n" + "=" * 80)
print("PersonaPlex Model Layer Names")
print("=" * 80)

# Print all layers containing attention or MLP
attn_layers = []
mlp_layers = []
other_layers = []

for name, module in lm.named_modules():
    if 'attn' in name.lower():
        attn_layers.append(name)
    elif 'mlp' in name.lower() or 'ffn' in name.lower():
        mlp_layers.append(name)
    elif any(x in name.lower() for x in ['proj', 'linear', 'dense']):
        other_layers.append(name)

print("\n### Attention Layers:")
for layer in attn_layers[:20]:  # First 20
    print(f"  - {layer}")

print("\n### MLP/FFN Layers:")
for layer in mlp_layers[:20]:  # First 20
    print(f"  - {layer}")

print("\n### Other Projection Layers:")
for layer in other_layers[:20]:  # First 20
    print(f"  - {layer}")

print("\n" + "=" * 80)
print("Copy the relevant layer names above")
print("=" * 80)
EOF
```

## Update lora_train_fixed.py

Once you have the layer names, update lines 348-356 in `lora_train_fixed.py`:

```python
target_modules=[
    # OLD (doesn't work):
    # 'self_attn.q_proj',
    # 'self_attn.k_proj',
    # ...

    # NEW (use actual PersonaPlex layer names):
    'transformer.layers.0.attention.q_proj',  # Example - replace with actual
    'transformer.layers.0.attention.k_proj',
    # etc.
]
```

## Common Moshi/PersonaPlex Layer Name Patterns

Depending on the architecture, layers might be named:

- `transformer.h.{i}.attn.{qkv}_proj`
- `model.layers.{i}.self_attn.{qkv}_proj`
- `lm_head.transformer.{i}.attention.{qkv}`
- `decoder.layers.{i}.{component}`

**The script above will show you the exact names!**

## Quick Fix Option

If layers have indices (like `layers.0`, `layers.1`, etc.), you can use wildcards:

```python
target_modules=[
    ".*attn.*q_proj",
    ".*attn.*k_proj",
    ".*attn.*v_proj",
    ".*attn.*o_proj",
    ".*mlp.*gate",
    ".*mlp.*up",
    ".*mlp.*down",
]
```

PEFT supports regex patterns, so `.*` matches any prefix.
