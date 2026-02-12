# Fine-Tuning PersonaPlex

This guide explains how to fine-tune PersonaPlex for custom conversational scenarios, including coffee shop ordering, technical support, or other domain-specific dialogues using the Fisher corpus approach.

## Overview

PersonaPlex is built on the Moshi architecture and was trained on:
1. **Fisher English Corpus** - Real casual conversations with LLM-labeled prompts
2. **Synthetic customer service dialogues** - Various service scenarios (restaurants, rental services, etc.)

The model can be fine-tuned for new domains while maintaining its full-duplex conversational capabilities.

## Is `lm.py` the Right File?

**Yes!** The `moshi/moshi/models/lm.py` file contains the `LMModel` class with the `forward_train()` method, which is specifically designed for fine-tuning.

### Key Components:

- **`LMModel`**: The main transformer-based language model that processes audio codes
- **`forward_train()`**: The method to use in your training loop for fine-tuning
- **`LMGen`**: Used for inference only (not needed for training)

## Fine-Tuning for Coffee Shop Ordering

### Step 1: Prepare Your Data

For a coffee shop ordering scenario (similar to the "Jerusalem Shakshuka" example in the README):

1. **Create conversational scripts** with coffee shop dialogues:
   ```
   Customer: "Hi, I'd like to order a coffee."
   Barista: "Sure! What can I get for you today?"
   Customer: "Do you have any seasonal drinks?"
   Barista: "Yes, we have a Pumpkin Spice Latte for $5.50..."
   ```

2. **Generate audio data**:
   - Use TTS (text-to-speech) to create audio files
   - Or record real conversations
   - Ensure you have both customer and barista voices

3. **Create role prompts** following the README format:
   ```python
   prompt = """You work for Fine Coffee Shop which is a coffee shop and your name is Alex. 
   Information: Menu includes Latte ($4.50), Cappuccino ($4.75), Espresso ($3.00), 
   Americano ($3.50). Sizes: Small, Medium, Large (+$0.50 each size up). 
   Pastries: Croissant ($3.50), Muffin ($3.00), Cookie ($2.50). 
   Available for mobile order pickup."""
   ```

### Step 2: Encode Audio to Codes

PersonaPlex works with audio codes from the MIMI codec. You'll need to:

1. Load the MIMI compression model
2. Convert your audio files to codes
3. Format as tensors with shape `[batch_size, num_codebooks, sequence_length]`

Example structure:
```python
from moshi.models.loaders import get_mimi
from moshi.models.lm import LMModel

# Load MIMI codec
mimi = get_mimi()

# Encode audio
with torch.no_grad():
    codes = mimi.encode(audio_tensor)  # Shape: [B, K, T]
```

### Step 3: Create Training Script

Here's a minimal training loop structure:

```python
import torch
import torch.nn.functional as F
from moshi.models.loaders import get_moshi_lm
from torch.utils.data import DataLoader

# Load pre-trained PersonaPlex model
model = get_moshi_lm()  # Loads personaplex-7b-v1
model.train()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        codes = batch['codes']  # Shape: [B, K, T]
        targets = batch['targets']  # Ground truth codes
        
        # Forward pass
        output = model.forward_train(codes)
        
        # Compute loss on audio codebooks
        audio_loss = F.cross_entropy(
            output.logits[output.mask].reshape(-1, model.card),
            targets[output.mask].reshape(-1)
        )
        
        # Compute loss on text
        text_loss = F.cross_entropy(
            output.text_logits[output.text_mask].reshape(-1, model.text_card),
            batch['text_targets'][output.text_mask].reshape(-1)
        )
        
        # Combined loss
        loss = audio_loss + text_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save fine-tuned model
torch.save(model.state_dict(), "coffee_shop_model.pt")
```

### Step 4: Evaluate Your Model

Use the `offline.py` script to test your fine-tuned model:

```bash
HF_TOKEN=<TOKEN> \
python -m moshi.offline \
  --voice-prompt "NATF2.pt" \
  --text-prompt "You work for Fine Coffee Shop which is a coffee shop and your name is Alex. Information: Menu includes Latte ($4.50), Cappuccino ($4.75)..." \
  --input-wav "test_coffee_order.wav" \
  --output-wav "output.wav" \
  --output-text "output.json"
```

## Understanding the Fisher Corpus Approach

PersonaPlex was trained on the Fisher English Corpus, which contains:
- Real telephone conversations
- Natural turn-taking and interruptions
- Casual, open-ended dialogue

When fine-tuning for coffee ordering:
- You leverage these conversational capabilities
- Add domain-specific knowledge (coffee menu, prices)
- The model learns both the **conversational flow** and **domain facts**

## Tips for Best Results

1. **Data Quality**: Use high-quality audio (16kHz recommended)
2. **Balanced Prompts**: Mix specific coffee shop scenarios with general conversational prompts
3. **Fisher-style Prompts**: Include casual conversation prompts like:
   ```
   "You enjoy having a good conversation. Have a casual discussion about coffee preferences and favorite drinks."
   ```
4. **Gradual Fine-tuning**: Use a low learning rate (1e-5 to 1e-6) to preserve pre-trained knowledge
5. **Validation**: Test on held-out coffee shop scenarios to avoid overfitting

## Architecture Details

The `LMModel` uses a dual-transformer architecture:
- **Main Transformer**: Processes all codebooks together (text + 8 audio streams)
- **Depformer**: Predicts each codebook independently, conditioned on main transformer output
- **Delays**: Each audio stream has a different delay pattern for causal modeling

This architecture enables:
- Full-duplex conversation (simultaneous speaking/listening)
- Low-latency responses (~200ms)
- Natural interruption handling

## References

- PersonaPlex Paper: https://arxiv.org/abs/2602.06053
- Moshi Architecture: https://arxiv.org/abs/2410.00037
- Fisher English Corpus: https://catalog.ldc.upenn.edu/LDC2004T19

## Questions?

For implementation details, see:
- `moshi/moshi/models/lm.py` - Model definition and training method
- `moshi/moshi/offline.py` - Offline evaluation script
- `moshi/moshi/server.py` - Real-time server implementation
- Main README.md - Usage examples and prompting guide
