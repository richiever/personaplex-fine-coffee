# PersonaPlex Fine Coffee - Project Instructions

## Critical: This is a PersonaPlex project, NOT Moshi

- The base model is **nvidia/personaplex-7b-v1** (PersonaPlex 7B)
- The `moshi` pip package is only a dependency used for model loading (`from moshi.models import loaders`)
- NEVER reference `kyutai/moshika-pytorch-bf16` in guides or user-facing docs
- All tokenizer weights (Mimi codec, SentencePiece) exist in `nvidia/personaplex-7b-v1` directly
- When writing docs, guides, or comments: say "PersonaPlex", not "Moshi"
- The codec is called "Mimi" (part of PersonaPlex), not "Moshi codec"

## Architecture

- PersonaPlex-7B: speech-to-speech LM with persona conditioning
- Training format: .pt files of shape [17, T] at 12.5 Hz
  - Row 0: text tokens (inner monologue)
  - Rows 1-8: agent audio (8 codebooks)
  - Rows 9-16: user audio (8 codebooks)
- LoRA fine-tuning targets: self_attn.in_proj, self_attn.out_proj, gating.linear_in, gating.linear_out
- Semantic weight (codebook 0) >> acoustic weight (codebooks 1-7) to learn WHAT to say, not HOW to sound

## Dataset

- HF repo: `AnthrolyticB/personaplex-training-data-test`
- conv_0000-conv_0200: original coffee shop conversations (training.json)
- conv_0201-conv_1200: retail conversations - coffee shop + general retail (retail_training.json)
- Total: 1201 conversations, 70 personas

## Key Files

- `runpod_pipeline.py`: Full data pipeline (TTS + Mimi encode + upload). Supports multi-GPU TTS and `--retail-only` flag
- `lora_train_fixed.py`: LoRA training with semantic-weighted loss
- `generate_retail_conversations.py`: Coffee shop conversation generator (ollama subagents)
- `generate_retail_general.py`: General retail conversation generator (12 store types)
- `upload_training.py`: Validate + upload training JSON to HF
- `RUNPOD_GUIDE.md`: Step-by-step RunPod instructions
