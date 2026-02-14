"""
Full data pipeline for RunPod:
  1. Download training.json from HuggingFace (single file, array of conversations)
  2. Download 100 agent + 100 user voice samples from LibriSpeech
  3. Generate audio with Chatterbox TTS (random voice pairs)
  4. Encode through Mimi + assemble .pt files (variable silence padding)
  5. Upload .pt files to HuggingFace

Usage:
    pip install chatterbox-tts
    python runpod_pipeline.py

    # Skip steps you've already done:
    python runpod_pipeline.py --skip-voices --skip-tts
"""

import argparse
import json
import random
import os
from pathlib import Path

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torchaudio
import soundfile as sf

# ============================================================
# Config
# ============================================================

HF_REPO = "AnthrolyticB/personaplex-training-data-test"
TRAINING_JSON = Path("/workspace/training.json")
AUDIO_DIR = Path("/workspace/audio")
PT_DIR = Path("/workspace/pt_files")
AGENT_VOICE_DIR = Path("/workspace/refs/agent_voices")
USER_VOICE_DIR = Path("/workspace/refs/user_voices")

TARGET_SR = 24000
AGENT_VOICE_COUNT = 100
USER_VOICE_COUNT = 100

# Silence padding (variable per paper section 3.2.2)
SILENCE_TOKENS = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]
ZERO_TOKEN = -1


def detect_mood(system_prompt):
    """Detect mood from system prompt keywords."""
    prompt_lower = system_prompt.lower()
    if "angry" in prompt_lower:
        return "angry"
    elif "nervous" in prompt_lower:
        return "nervous"
    elif "indecisive" in prompt_lower:
        return "indecisive"
    else:
        return "friendly"


def load_conversations():
    """Load all conversations from training.json."""
    with open(TRAINING_JSON) as f:
        data = json.load(f)

    for conv in data:
        for turn in conv["turns"]:
            turn["role"] = turn["role"].lower()

        if "conversation_id" not in conv:
            idx = data.index(conv)
            conv["conversation_id"] = f"conv_{idx:04d}"

        # Auto-detect mood from system_prompt if not explicitly set
        if "mood" not in conv:
            conv["mood"] = detect_mood(conv.get("system_prompt", ""))

    return data


# ============================================================
# Step 1: Download training.json
# ============================================================

def download_conversations():
    from huggingface_hub import HfApi

    print("=" * 60)
    print("STEP 1: Download training.json")
    print("=" * 60)

    if TRAINING_JSON.exists():
        with open(TRAINING_JSON) as f:
            data = json.load(f)
        print(f"  Already have training.json ({len(data)} conversations)")
        return len(data)

    api = HfApi()
    print("  Downloading training.json...")
    api.hf_hub_download(
        repo_id=HF_REPO, filename="training.json",
        repo_type="dataset", local_dir="/workspace",
    )

    with open(TRAINING_JSON) as f:
        data = json.load(f)
    print(f"  {len(data)} conversations loaded")
    return len(data)


# ============================================================
# Step 2: Download voice samples
# ============================================================

def download_voices():
    print("\n" + "=" * 60)
    print("STEP 2: Download voice samples from LibriSpeech")
    print("=" * 60)

    AGENT_VOICE_DIR.mkdir(parents=True, exist_ok=True)
    USER_VOICE_DIR.mkdir(parents=True, exist_ok=True)

    existing_agent = len(list(AGENT_VOICE_DIR.glob("*.wav")))
    existing_user = len(list(USER_VOICE_DIR.glob("*.wav")))
    if existing_agent >= AGENT_VOICE_COUNT and existing_user >= USER_VOICE_COUNT:
        print(f"  Already have {existing_agent} agent + {existing_user} user voices, skipping")
        return

    print("  Downloading LibriSpeech train-clean-100 (~6GB, cached after first time)...")
    dataset = torchaudio.datasets.LIBRISPEECH(
        "/workspace/refs/temp_libri", url="train-clean-100", download=True
    )

    print("  Scanning for diverse speakers...")
    speakers = {}
    for i in range(len(dataset)):
        waveform, sr, text, speaker_id, chapter_id, utterance_id = dataset[i]
        duration = waveform.shape[1] / sr

        if duration < 4.0 or duration > 15.0:
            continue

        if speaker_id not in speakers:
            speakers[speaker_id] = {"best_idx": i, "best_dur": duration}
        elif duration > speakers[speaker_id]["best_dur"]:
            speakers[speaker_id] = {"best_idx": i, "best_dur": duration}

    print(f"  Found {len(speakers)} unique speakers")

    speaker_ids = list(speakers.keys())
    random.seed(42)
    random.shuffle(speaker_ids)

    def save_voice(speaker_id, output_path):
        info = speakers[speaker_id]
        waveform, sr, text, sid, _, _ = dataset[info["best_idx"]]
        if waveform.shape[0] > 1:
            waveform = waveform[0:1]
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        torchaudio.save(str(output_path), waveform, TARGET_SR)

    agent_speakers = speaker_ids[:AGENT_VOICE_COUNT]
    user_speakers = speaker_ids[AGENT_VOICE_COUNT:AGENT_VOICE_COUNT + USER_VOICE_COUNT]

    while len(agent_speakers) < AGENT_VOICE_COUNT:
        agent_speakers.append(random.choice(speaker_ids))
    while len(user_speakers) < USER_VOICE_COUNT:
        user_speakers.append(random.choice(speaker_ids))

    print(f"  Saving {AGENT_VOICE_COUNT} agent voices...")
    for i, spk in enumerate(agent_speakers):
        path = AGENT_VOICE_DIR / f"agent_{i:03d}.wav"
        if not path.exists():
            save_voice(spk, path)

    print(f"  Saving {USER_VOICE_COUNT} user voices...")
    for i, spk in enumerate(user_speakers):
        path = USER_VOICE_DIR / f"user_{i:03d}.wav"
        if not path.exists():
            save_voice(spk, path)

    print(f"  Done! {AGENT_VOICE_COUNT} agent + {USER_VOICE_COUNT} user voices")


# ============================================================
# Step 3: Generate TTS audio
# ============================================================

def generate_tts():
    print("\n" + "=" * 60)
    print("STEP 3: Generate audio with Chatterbox TTS")
    print("=" * 60)

    from chatterbox.tts import ChatterboxTTS
    tts_model = ChatterboxTTS.from_pretrained(device="cuda")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    agent_voices = sorted(AGENT_VOICE_DIR.glob("*.wav"))
    user_voices = sorted(USER_VOICE_DIR.glob("*.wav"))
    conversations = load_conversations()

    print(f"  {len(conversations)} conversations, {len(agent_voices)} agent voices, {len(user_voices)} user voices")

    random.seed(123)

    for ci, conv in enumerate(conversations):
        conv_id = conv["conversation_id"]

        agent_voice = str(random.choice(agent_voices))
        user_voice = str(random.choice(user_voices))

        expected_turns = len(conv["turns"])
        existing = list(AUDIO_DIR.glob(f"{conv_id}_turn_*.wav"))
        if len(existing) >= expected_turns:
            print(f"  [{ci + 1}/{len(conversations)}] {conv_id} — already done, skipping")
            continue

        print(f"  [{ci + 1}/{len(conversations)}] {conv_id} (agent: {Path(agent_voice).name}, user: {Path(user_voice).name})")

        for ti, turn in enumerate(conv["turns"]):
            out_path = AUDIO_DIR / f"{conv_id}_turn_{ti:02d}_{turn['role']}.wav"
            if out_path.exists():
                continue

            ref = agent_voice if turn["role"] == "agent" else user_voice

            try:
                wav = tts_model.generate(turn["text"], audio_prompt_path=ref)
                sf.write(str(out_path), wav.squeeze(0).cpu().numpy().T, tts_model.sr)
            except Exception as e:
                print(f"    ERROR on turn {ti}: {e}")
                import numpy as np
                silence = np.zeros((TARGET_SR * 2, 1), dtype=np.float32)
                sf.write(str(out_path), silence, TARGET_SR)

    print(f"\n  TTS generation complete! Audio in {AUDIO_DIR}/")


# ============================================================
# Step 4: Mimi encode + assemble .pt files
# ============================================================

def assemble_pt_files():
    print("\n" + "=" * 60)
    print("STEP 4: Mimi encode + assemble .pt files")
    print("=" * 60)

    import sentencepiece as spm
    from moshi.models import loaders

    PT_DIR.mkdir(parents=True, exist_ok=True)

    print("  Loading Mimi codec...")
    mimi = loaders.get_mimi("/workspace/weights/tokenizer-e351c8d8-checkpoint125.safetensors", device="cuda")
    mimi.eval()

    print("  Loading text tokenizer...")
    sp = spm.SentencePieceProcessor(model_file="/workspace/weights/tokenizer_spm_32k_3.model")

    silence_frame = torch.tensor(SILENCE_TOKENS, dtype=torch.long).unsqueeze(1)

    conversations = load_conversations()
    random.seed(456)

    for ci, conv in enumerate(conversations):
        conv_id = conv["conversation_id"]
        pt_path = PT_DIR / f"{conv_id}.pt"

        if pt_path.exists():
            print(f"  [{ci + 1}/{len(conversations)}] {conv_id} — already assembled, skipping")
            continue

        all_audio_exists = all(
            (AUDIO_DIR / f"{conv_id}_turn_{ti:02d}_{t['role']}.wav").exists()
            for ti, t in enumerate(conv["turns"])
        )
        if not all_audio_exists:
            print(f"  [{ci + 1}/{len(conversations)}] {conv_id} — missing audio, skipping")
            continue

        print(f"  [{ci + 1}/{len(conversations)}] {conv_id}...", end=" ")

        # Encode system prompt if present
        system_prompt = conv.get("system_prompt", "")
        system_tokens = []
        system_frames = 0

        if system_prompt:
            # Remove <system> tags if present, then encode
            cleaned_prompt = system_prompt.replace("<system>", "").strip()
            system_tokens = sp.Encode(cleaned_prompt)
            # Allocate ~2 seconds (25 frames at 12.5 Hz) for system prompt
            system_frames = 25

        agent_chunks = []
        user_chunks = []
        text_info = []
        frame_offset = system_frames  # Start after system prompt frames

        for ti, turn in enumerate(conv["turns"]):
            wav_path = AUDIO_DIR / f"{conv_id}_turn_{ti:02d}_{turn['role']}.wav"
            audio, sr = sf.read(str(wav_path))
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).cuda()

            with torch.no_grad():
                codes = mimi.encode(audio_tensor)
            codes = codes.squeeze(0).cpu()
            T_turn = codes.shape[1]

            if turn["role"] == "agent":
                agent_chunks.append(codes)
                user_chunks.append(silence_frame.repeat(1, T_turn))
                text_info.append({
                    "text": turn["text"],
                    "start_frame": frame_offset,
                    "num_frames": T_turn,
                })
            else:
                user_chunks.append(codes)
                agent_chunks.append(silence_frame.repeat(1, T_turn))

            frame_offset += T_turn

            # Mood-aware silence padding (section 3.2.2)
            mood = conv.get("mood", "friendly")
            if ti < len(conv["turns"]) - 1:
                next_role = conv["turns"][ti + 1]["role"]
                if mood == "angry":
                    # Angry: minimal gaps, interrupts quickly
                    if turn["role"] == "user" and next_role == "agent":
                        gap = random.randint(1, 3)  # 80-240ms
                    else:
                        gap = random.randint(1, 4)  # 80-320ms
                elif mood == "nervous":
                    # Nervous: longer pauses, hesitation
                    if turn["role"] == "user" and next_role == "agent":
                        gap = random.randint(6, 15)  # 480-1200ms
                    else:
                        gap = random.randint(8, 20)  # 640-1600ms
                elif mood == "indecisive":
                    # Indecisive: medium-long pauses, thinking
                    if turn["role"] == "user" and next_role == "agent":
                        gap = random.randint(5, 12)  # 400-960ms
                    else:
                        gap = random.randint(6, 16)  # 480-1280ms
                else:
                    # Friendly: normal pacing
                    if turn["role"] == "user" and next_role == "agent":
                        gap = random.randint(2, 8)  # 160-640ms
                    else:
                        gap = random.randint(4, 12)  # 320-960ms

                agent_chunks.append(silence_frame.repeat(1, gap))
                user_chunks.append(silence_frame.repeat(1, gap))
                frame_offset += gap

        agent_codes = torch.cat(agent_chunks, dim=1)
        user_codes = torch.cat(user_chunks, dim=1)
        T_conv = agent_codes.shape[1]  # Conversation frames only
        T_total = system_frames + T_conv  # System prompt + conversation

        # Initialize text tokens for entire sequence
        text_tokens = torch.full((1, T_total), ZERO_TOKEN, dtype=torch.long)

        # Add system prompt tokens at the beginning
        if system_tokens:
            for j, tid in enumerate(system_tokens):
                frame_idx = int(j * system_frames / len(system_tokens))
                if frame_idx < system_frames:
                    text_tokens[0, frame_idx] = tid
        # Add conversation text tokens (offset by system_frames)
        for info in text_info:
            token_ids = sp.Encode(info["text"])
            start = info["start_frame"]  # Already includes system_frames offset
            num_frames = info["num_frames"]
            for j, tid in enumerate(token_ids):
                frame_idx = start + int(j * num_frames / len(token_ids))
                if frame_idx < T_total:
                    text_tokens[0, frame_idx] = tid

        # Add silence frames for system prompt duration in audio streams
        if system_frames > 0:
            system_silence_agent = silence_frame.repeat(1, system_frames)
            system_silence_user = silence_frame.repeat(1, system_frames)
            agent_codes = torch.cat([system_silence_agent, agent_codes], dim=1)
            user_codes = torch.cat([system_silence_user, user_codes], dim=1)

        final = torch.cat([text_tokens, agent_codes, user_codes], dim=0)

        assert final.shape[0] == 17
        assert final[1:9].min() >= 0
        assert final[1:9].max() <= 2047

        torch.save(final, str(pt_path))
        print(f"{T_total} frames ({T_total / 12.5:.1f}s)")

    count = len(list(PT_DIR.glob("*.pt")))
    print(f"\n  Done! {count} .pt files in {PT_DIR}/")


# ============================================================
# Step 5: Upload .pt files to HuggingFace
# ============================================================

def upload_pt_files():
    print("\n" + "=" * 60)
    print("STEP 5: Upload .pt files to HuggingFace")
    print("=" * 60)

    from huggingface_hub import HfApi

    api = HfApi()
    pt_files = sorted(PT_DIR.glob("*.pt"))

    print(f"  Uploading {len(pt_files)} .pt files to {HF_REPO}...")

    for f in pt_files:
        print(f"  Uploading {f.name}...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=HF_REPO,
            repo_type="dataset",
        )

    print(f"\n  Done! {len(pt_files)} files uploaded to {HF_REPO}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Full PersonaPlex data pipeline")
    parser.add_argument("--skip-conversations", action="store_true")
    parser.add_argument("--skip-voices", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-assemble", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    if not args.skip_conversations:
        download_conversations()
    if not args.skip_voices:
        download_voices()
    if not args.skip_tts:
        generate_tts()
    if not args.skip_assemble:
        assemble_pt_files()
    if not args.skip_upload:
        upload_pt_files()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("Next: python /workspace/data/lora_train.py --epochs 2 --lora-rank 64 --lora-alpha 128 --lr 5e-6")


if __name__ == "__main__":
    main()
