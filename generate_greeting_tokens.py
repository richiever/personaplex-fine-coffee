"""
One-time script to generate preset barista greeting tokens.

Run on RunPod where Mimi and Chatterbox TTS are available:
    python generate_greeting_tokens.py

Generates TTS audio, applies LUFS normalization (matching training pipeline),
encodes through Mimi, and saves tokens for inference.
"""
import os
import torch
import numpy as np
from moshi.models import loaders

GREETING_TEXT = "Welcome to the coffee shop, how can I help you?"
SAMPLE_RATE = 24000
TARGET_LUFS = -24.0
FRAME_SIZE = 1920  # 24000 / 12.5


def normalize_audio_lufs(wav_np, sr, target_lufs=-24.0):
    """Normalize mono audio to target LUFS (matches training pipeline)."""
    import pyloudnorm as pyln
    if wav_np.ndim == 2 and wav_np.shape[0] == 1:
        wav_np = wav_np[0]
    if len(wav_np) < int(sr * 0.5):
        return wav_np
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav_np)
    if np.isinf(loudness):
        return wav_np
    return pyln.normalize.loudness(wav_np, loudness, target_lufs)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = os.environ.get("GREETING_SAVE_DIR", "/workspace")

    # Step 1: Generate TTS audio
    print("Generating TTS audio...")
    from chatterbox.tts import ChatterboxTTS
    tts = ChatterboxTTS.from_pretrained(device=device)
    wav = tts.generate(GREETING_TEXT)

    # wav is a tensor, ensure correct shape
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dim() == 3:
        wav = wav.squeeze(0)

    # Resample to 24kHz if needed
    if wav.shape[-1] > SAMPLE_RATE * 10:
        print(f"Warning: audio is {wav.shape[-1] / SAMPLE_RATE:.1f}s, trimming to 5s")
        wav = wav[:, :SAMPLE_RATE * 5]

    print(f"Audio shape: {wav.shape}, duration: {wav.shape[-1] / SAMPLE_RATE:.2f}s")

    # Step 2: LUFS normalization (matches training pipeline)
    print(f"Normalizing audio to {TARGET_LUFS} LUFS...")
    wav_np = wav.squeeze(0).cpu().numpy()
    wav_np = normalize_audio_lufs(wav_np, SAMPLE_RATE, TARGET_LUFS)
    wav = torch.from_numpy(wav_np).unsqueeze(0)
    print(f"Normalized audio shape: {wav.shape}")

    # Step 3: Encode through Mimi
    print("Loading Mimi encoder...")
    from huggingface_hub import hf_hub_download
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.eval()

    with torch.no_grad():
        # Mimi expects (batch, channels, time)
        audio_tensor = wav.to(device)
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Encode in frames
        all_tokens = []
        mimi.streaming_forever(1)
        for start in range(0, audio_tensor.shape[-1], FRAME_SIZE):
            chunk = audio_tensor[:, :, start:start + FRAME_SIZE]
            if chunk.shape[-1] < FRAME_SIZE:
                # Pad last chunk
                chunk = torch.nn.functional.pad(chunk, (0, FRAME_SIZE - chunk.shape[-1]))
            codes = mimi.encode(chunk)  # (1, 8, num_frames)
            for f in range(codes.shape[-1]):
                frame_tokens = codes[0, :, f].cpu().numpy().tolist()
                all_tokens.append(frame_tokens)

    print(f"\nGenerated {len(all_tokens)} frames of tokens")
    print(f"Duration: {len(all_tokens) / 12.5:.2f}s")

    # Step 4: Output as Python constant
    print("\n# Paste this into lm.py:")
    print(f"# Barista greeting: \"{GREETING_TEXT}\"")
    print(f"# {len(all_tokens)} frames @ 12.5fps = {len(all_tokens) / 12.5:.2f}s")
    print("BARISTA_GREETING_TOKENS = [")
    for frame in all_tokens:
        print(f"    {frame},")
    print("]")

    # Also save as .pt for convenience
    token_tensor = torch.tensor(all_tokens, dtype=torch.long)
    save_path = os.path.join(save_dir, "barista_greeting_tokens.pt")
    torch.save(token_tensor, save_path)
    print(f"\nSaved to {save_path} ({token_tensor.shape})")


if __name__ == "__main__":
    main()
