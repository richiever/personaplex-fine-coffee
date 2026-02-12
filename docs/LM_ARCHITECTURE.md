# Language Model Architecture (lm.py)

## Overview

The `moshi/moshi/models/lm.py` file implements the **core language model backbone** for PersonaPlex and Moshi, a real-time full-duplex conversational speech model. This file contains the transformer-based architecture that jointly processes text and audio tokens for natural spoken interactions.

## Purpose

The language model serves as the central component that:

1. **Processes Multiple Token Streams**: Handles 1 text stream + 8 audio codebook streams in parallel
2. **Enables Voice and Role Control**: Supports conditioning on text prompts (role) and audio prompts (voice)
3. **Provides Low-Latency Streaming**: Optimized for real-time conversational interaction with <200ms latency
4. **Manages Multi-Modal Generation**: Jointly generates coherent text and audio tokens using a hierarchical architecture

## Key Components

### 1. Data Structures

#### `LMOutput` (dataclass)
Encapsulates model outputs during training:
- `logits`: Audio token predictions `[B, K, T, card]` - re-aligned with input
- `mask`: Valid position indicators `[B, K, T]`
- `text_logits`: Text token predictions `[B, 1, T, text_card]`
- `text_mask`: Valid text position indicators `[B, 1, T]`

#### `_LMGenState` (dataclass)
Manages state during streaming generation:
- `cache`: Circular buffer storing recent tokens
- `provided`: Flags indicating which tokens are teacher-forced
- `initial`: Initial tokens for sequence start
- `graphed_main`, `graphed_embeddings`, `graphed_depth`: CUDA graph wrappers for optimized inference
- `offset`: Current position in circular cache

### 2. Core Classes

#### `ScaledEmbedding` (extends `torch.nn.Embedding`)
Enhanced embedding layer with:
- **Layer normalization** (optional): Stabilizes training
- **Zero-token handling**: Special token ID (-1) outputs exactly zero vector
- **Learning rate boost**: Scaled embeddings for better convergence

**Key Parameters**:
- `norm`: Whether to apply layer normalization
- `zero_idx`: Special token that outputs zero vector (default: -1)

#### `LMModel` (extends `StreamingContainer`)
The main transformer-based language model with dual-stage architecture:

**Architecture Overview**:
```
Input Tokens (Text + Audio) 
    ↓
Embedding Layer (sum across all codebooks)
    ↓
Main Transformer (processes all streams jointly)
    ↓
├── Text Logits (direct projection)
└── Audio Logits (via Depformer)
        ↓
    Depformer Transformer (autoregressive over 8 audio codebooks)
        ↓
    8 Linear Projections → 8 Audio Token Predictions
```

**Key Parameters**:
- `n_q=8`: Number of audio codebook streams
- `dep_q=8`: Number of streams processed by depformer
- `card=1024`: Audio vocabulary size (per codebook)
- `text_card=32000`: Text vocabulary size
- `dim=128`: Main transformer dimension
- `depformer_dim=256`: Depformer transformer dimension
- `delays`: Per-codebook delays for causal modeling

**Special Token IDs**:
- `initial_token_id`: Start-of-sequence for audio (value: `card`)
- `text_initial_token_id`: Start-of-sequence for text (value: `text_card`)
- `text_padding_token_id`: Text padding token
- `zero_token_id=-1`: Skip this position (no input/output)
- `ungenerated_token_id=-2`: Predict this token (partial teacher forcing)

**Key Methods**:

##### Training
- `embed_codes(sequence)`: Converts token IDs to embeddings, summing across all codebook embeddings
- `forward_codes(sequence)`: Full forward pass through main transformer
- `forward_depformer_training(sequence, transformer_out)`: Batch processing through depformer for all 8 audio tokens
- `forward_train(codes)`: Complete training forward pass with delay handling and loss computation

##### Inference (Streaming)
- `forward_embeddings(input)`: Processes embeddings through main transformer, returns text logits
- `forward_depformer(depformer_cb_index, sequence, transformer_out)`: Generates single audio token via depformer (called 8 times per step)

#### `LMGen` (extends `StreamingModule`)
Generation and inference engine that orchestrates streaming token generation:

**Key Parameters**:
- `lm_model`: The underlying LMModel
- `use_sampling=True`: Whether to sample or use greedy decoding
- `temp=0.8`: Temperature for audio token sampling
- `temp_text=0.7`: Temperature for text token sampling
- `top_k=250`: Top-K sampling for audio
- `top_k_text=25`: Top-K sampling for text
- `audio_silence_frame_cnt=1`: Number of silence frames to insert
- `text_prompt_tokens`: Optional text prompt token IDs
- `save_voice_prompt_embeddings`: Whether to save voice embeddings

**Key Methods**:

##### Voice Prompting
- `load_voice_prompt(filepath, mimi_encoder)`: 
  - Loads audio file
  - Normalizes to -24 LUFS
  - Encodes using MIMI codec
  - Stores as conditioning signal

- `step_system_prompts(mimi_encoder)`:
  - Processes voice prompt audio
  - Inserts silence frames
  - Processes text prompt tokens
  - Prepares model for user interaction

##### Streaming Generation
- `step(in_tokens, force_text_token)`:
  - Main inference loop (called once per time step)
  - Manages circular cache buffer
  - Handles teacher forcing (partial or full)
  - Returns generated text and audio tokens

- `prepare_step_input(in_tokens, force_text_token)`:
  - Prepares inputs with proper delays
  - Manages circular cache rotation
  - Handles special tokens (zero, ungenerated)

- `process_transformer_output(transformer_out, text_logits, target)`:
  - Samples text token from main transformer
  - Computes optional loss metrics
  - Returns sampled token and statistics

- `depformer_step(transformer_out, state, target)`:
  - Iteratively generates 8 audio tokens
  - Each token depends on previous tokens in sequence
  - Uses cached depformer state for efficiency
  - Returns all 8 audio tokens

### 3. Utility Functions

#### Token Delay Management
- `_delay_sequence(delays, tensor, padding)`: 
  - Applies per-codebook delays to token sequences
  - Handles multi-stream causal dependencies
  - Used to create shifted input sequences

- `_undelay_sequence(delays, tensor, fill_value)`:
  - Reverses delay operation
  - Creates validity masks for loss computation
  - Fills invalid positions with NaN or specified value

#### Audio Processing
- `create_sinewave(duration, sample_rate)`:
  - Generates 440 Hz tone (for silence replacement during debugging)
  - Used to detect silence vs. actual audio

- `normalize_audio(wav, sr, target_lufs)`:
  - Normalizes mono audio to target loudness level
  - Default: -24 LUFS for consistent voice prompts
  - Uses pyloudnorm library

- `load_audio(filepath, sample_rate)`:
  - Loads and resamples audio files
  - Returns mono or stereo PCM data

- `_iterate_audio(sample_pcm, sample_interval_size)`:
  - Iterator that yields audio chunks
  - Handles padding for final incomplete chunk
  - Used for streaming audio processing

- `encode_from_sphn(mimi, samples, max_batch)`:
  - Encodes audio samples using MIMI codec
  - Batches samples for efficiency
  - Yields encoded tokens one sample at a time

#### Loss and Debugging
- `create_loss_report(...)`:
  - Computes detailed loss statistics during generation
  - Tracks forced vs. sampled tokens
  - Computes ranking of forced tokens in probability distribution
  - Returns per-channel losses and diagnostics

## Architecture Details

### Two-Stage Transformer Design

The model uses a **two-stage architecture** to balance quality and latency:

1. **Main Transformer**:
   - Processes all 9 token streams (1 text + 8 audio) jointly
   - Captures cross-modal dependencies
   - Outputs:
     - Text token prediction directly via linear layer
     - Shared representation for audio token prediction

2. **Depformer (Depth Transformer)**:
   - Takes main transformer output + previous audio token
   - Autoregressively predicts 8 audio tokens in sequence
   - Each token prediction conditions on previously predicted tokens
   - Enables hierarchical audio token modeling
   - Runs in streaming mode during inference (one token at a time)

### Delay Pattern

Each token stream has an associated **delay** value:
- **Text stream (delay=0)**: No delay
- **Audio streams (delays vary)**: Staggered delays enable causal modeling across codebooks

During training:
1. Input sequences are delayed using `_delay_sequence()`
2. Model predicts tokens at each position
3. Predictions are un-delayed using `_undelay_sequence()` to align with targets
4. Masks indicate which positions have valid predictions

During inference:
1. Circular cache manages delayed tokens automatically
2. Each `step()` call advances the cache by one position
3. Proper delays ensure causal dependencies are maintained

### Streaming and Caching

The `LMGen` class implements efficient streaming generation:

1. **Circular Cache Buffer**:
   - Stores recent tokens (size: context length + max delay)
   - Rotates on each step using modulo arithmetic
   - Avoids costly tensor concatenation

2. **CUDA Graphs**:
   - Pre-compiled computational graphs for fixed-size operations
   - Eliminates kernel launch overhead
   - Achieves ~30% speedup for inference

3. **State Management**:
   - Main transformer maintains autoregressive cache
   - Depformer maintains separate cache (reset after each step)
   - Voice prompt embeddings cached for reuse

## Usage in PersonaPlex

### Training Flow

```python
# Simplified training pseudocode
lm_model = LMModel(n_q=8, dep_q=8, card=1024, text_card=32000, ...)

# codes: [B, K=9, T] tensor of token IDs
# K=9: 1 text + 8 audio streams
output = lm_model.forward_train(codes)

# output.logits: [B, K=8, T, card] - audio predictions
# output.mask: [B, K=8, T] - valid positions
# output.text_logits: [B, 1, T, text_card] - text predictions
# output.text_mask: [B, 1, T] - valid positions

# Compute cross-entropy loss with masks
```

### Inference Flow

```python
# Simplified inference pseudocode
lm_gen = LMGen(lm_model, device='cuda', temp=0.8, temp_text=0.7, ...)

# Load voice prompt
lm_gen.load_voice_prompt("voice.wav", mimi_encoder)

# Process system prompts (voice + silence + text prompt + silence)
lm_gen.step_system_prompts(mimi_encoder)

# Streaming generation loop
while True:
    # in_tokens: [B, K=9] - user's text and audio tokens for this step
    # force_text_token: Optional teacher-forced text token
    
    out_text, out_audio = lm_gen.step(in_tokens, force_text_token)
    
    # out_text: [B] - generated text token
    # out_audio: [B, 8] - generated audio tokens (8 codebooks)
    
    # Decode audio tokens to waveform using MIMI decoder
    # Continue conversation...
```

## Key Constants

```python
AUDIO_TOKENS_PER_STREAM = 8  # Number of audio codebooks
FRAME_RATE_HZ = 12.5          # 80ms per frame
SILENCE_TOKENS = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]  # Pre-computed silence
SINE_TOKENS = [430, 1268, 381, 1611, 1095, 1495, 56, 472]      # Pre-computed 440Hz tone
```

## Integration with Other Components

### MIMI Audio Codec
- `lm.py` operates on discrete tokens from MIMI codec
- MIMI compresses 32kHz audio to 8 codebook streams @ 12.5 Hz
- Each codebook has vocabulary size of 2048 (PersonaPlex uses 1024)

### Transformer Modules
- Uses `StreamingTransformer` from `../modules/transformer.py`
- Supports various positional embeddings (sin, rope, etc.)
- Implements efficient streaming with KV-cache

### Sampling
- Uses `sample_token()` from `../utils/sampling.py`
- Supports temperature scaling and top-k filtering
- Can return logits for external sampling

## Performance Considerations

1. **Memory Efficiency**:
   - Circular cache avoids growing tensors
   - Depformer state reset after each main transformer step
   - Optional CPU offloading for large models

2. **Latency Optimization**:
   - CUDA graphs reduce kernel launch overhead
   - Streaming mode processes one time step at a time
   - Depformer runs in parallel with audio decoding

3. **Quality vs. Speed**:
   - Higher temperatures → more diverse but less coherent
   - Larger top-k → more exploration
   - Depformer depth affects audio quality vs. latency tradeoff

## References

- **PersonaPlex Paper**: [arXiv:2602.06053](https://arxiv.org/abs/2602.06053)
- **Moshi Architecture**: [arXiv:2410.00037](https://arxiv.org/abs/2410.00037)
- **Helium LLM**: Base model for text understanding
- **MIMI Codec**: Audio compression to discrete tokens

## Summary

`lm.py` is the **heart of PersonaPlex/Moshi's language modeling**, implementing:

- ✅ **Multi-stream transformer** for joint text and audio processing
- ✅ **Depformer architecture** for hierarchical audio token prediction  
- ✅ **Streaming generation** with circular caching and CUDA graphs
- ✅ **Voice and text prompting** for persona control
- ✅ **Low-latency inference** suitable for real-time conversation

The two-stage design (main transformer + depformer) balances modeling capacity with streaming efficiency, enabling natural full-duplex speech interactions.
