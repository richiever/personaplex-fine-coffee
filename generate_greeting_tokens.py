"""
One-time script to generate preset barista greeting tokens.

Run on RunPod where Mimi and Chatterbox TTS are available:
    python generate_greeting_tokens.py

Outputs the token array to paste into lm.py as BARISTA_GREETING_TOKENS.
"""
import torch
import numpy as np
from moshi.models import loaders

GREETING_TEXT = "Hi, welcome to the coffee shop, what can I get for you?"
SAMPLE_RATE = 24000
FRAME_SIZE = 1920  # 24000 / 12.5

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Step 2: Encode through Mimi
    print("Loading Mimi encoder...")
    mimi = loaders.get_mimi(loaders.DEFAULT_REPO, device=device)
    mimi.eval()

    with torch.no_grad():
        # Mimi expects (batch, channels, time)
        audio_tensor = wav.to(device)
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Encode in frames
        all_tokens = []
        mimi.reset_streaming()
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

    # Step 3: Output as Python constant
    print("\n# Paste this into lm.py:")
    print(f"# Barista greeting: \"{GREETING_TEXT}\"")
    print(f"# {len(all_tokens)} frames @ 12.5fps = {len(all_tokens) / 12.5:.2f}s")
    print("BARISTA_GREETING_TOKENS = [")
    for frame in all_tokens:
        print(f"    {frame},")
    print("]")

    # Also save as .pt for convenience
    token_tensor = torch.tensor(all_tokens, dtype=torch.long)
    torch.save(token_tensor, "/workspace/barista_greeting_tokens.pt")
    print(f"\nSaved to /workspace/barista_greeting_tokens.pt ({token_tensor.shape})")


if __name__ == "__main__":
    main()
