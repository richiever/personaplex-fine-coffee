"""
Full data pipeline for RunPod:
  1. Download training.json from HuggingFace (single file, array of conversations)
  2. Download 1000 agent + 1000 user voice samples from LibriSpeech
  3. Generate audio with Chatterbox TTS (random voice pairs)
  4. Encode through Mimi + assemble .pt files (variable silence padding)
  5. Upload .pt files to HuggingFace

Usage:
    pip install chatterbox-tts
    python runpod_pipeline.py

    # Retail-only mode (processes conv_0201-conv_1200):
    python runpod_pipeline.py --retail-only

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
RETAIL_TRAINING_JSON = Path("/workspace/retail_training.json")
AUDIO_DIR = Path("/workspace/audio")
PT_DIR = Path("/workspace/pt_files")
AGENT_VOICE_DIR = Path("/workspace/refs/agent_voices")
USER_VOICE_DIR = Path("/workspace/refs/user_voices")

TARGET_SR = 24000
AGENT_VOICE_COUNT = 100
USER_VOICE_COUNT = 100

# Silence padding (variable per paper section 3.2.2)
SILENCE_TOKENS = [948, 243, 1178, 546, 1736, 1030, 1978, 2008]
# 440 Hz sine wave tokens for user audio during system prompt (from PersonaPlex lm.py)
SINE_TOKENS = [430, 1268, 381, 1611, 1095, 1495, 56, 472]
# PAD token for agent text during voice/silence phases
PAD_TOKEN = 3
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


def load_conversations(retail_only=False):
    """Load conversations from training.json or retail_training.json."""
    if retail_only:
        source = RETAIL_TRAINING_JSON
    else:
        source = TRAINING_JSON

    with open(source) as f:
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

def download_conversations(retail_only=False):
    from huggingface_hub import HfApi

    print("=" * 60)
    print("STEP 1: Download conversation data")
    print("=" * 60)

    api = HfApi()

    if retail_only:
        # Download retail_training.json
        if RETAIL_TRAINING_JSON.exists():
            with open(RETAIL_TRAINING_JSON) as f:
                data = json.load(f)
            print(f"  Already have retail_training.json ({len(data)} conversations)")
            return len(data)

        print("  Downloading retail_training.json...")
        api.hf_hub_download(
            repo_id=HF_REPO, filename="retail_training.json",
            repo_type="dataset", local_dir="/workspace",
        )

        with open(RETAIL_TRAINING_JSON) as f:
            data = json.load(f)
        print(f"  {len(data)} retail conversations loaded")
        return len(data)
    else:
        # Download original training.json
        if TRAINING_JSON.exists():
            with open(TRAINING_JSON) as f:
                data = json.load(f)
            print(f"  Already have training.json ({len(data)} conversations)")
            return len(data)

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

    # Create LibriSpeech download directory
    libri_dir = Path("/workspace/refs/temp_libri")
    libri_dir.mkdir(parents=True, exist_ok=True)

    print("  Downloading LibriSpeech train-clean-100 (~6GB, cached after first time)...")
    torchaudio.datasets.LIBRISPEECH(
        str(libri_dir), url="train-clean-100", download=True
    )

    print("  Scanning for diverse speakers (fast file-based scan)...")
    speakers = {}  # speaker_id -> {"flac_path": path, "best_dur": dur}
    libri_root = libri_dir / "LibriSpeech" / "train-clean-100"
    flac_files = sorted(libri_root.glob("*/*/*.flac"))
    print(f"  Found {len(flac_files)} .flac files, checking durations...")
    for flac_path in flac_files:
        # Path format: speaker_id/chapter_id/speaker_id-chapter_id-utterance_id.flac
        speaker_id = int(flac_path.parent.parent.name)
        info = torchaudio.info(str(flac_path))
        duration = info.num_frames / info.sample_rate

        if duration < 4.0 or duration > 15.0:
            continue

        if speaker_id not in speakers:
            speakers[speaker_id] = {"flac_path": str(flac_path), "best_dur": duration}
        elif duration > speakers[speaker_id]["best_dur"]:
            speakers[speaker_id] = {"flac_path": str(flac_path), "best_dur": duration}

    print(f"  Found {len(speakers)} unique speakers")

    speaker_ids = list(speakers.keys())
    random.seed(42)
    random.shuffle(speaker_ids)

    def save_voice(speaker_id, output_path):
        info = speakers[speaker_id]
        waveform, sr = torchaudio.load(info["flac_path"])
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

def _tts_worker(gpu_id, conv_shard, agent_voices, user_voices, audio_dir, seed):
    """TTS worker for a single GPU. Runs in a subprocess."""
    import time as _time
    import numpy as np
    from chatterbox.tts import ChatterboxTTS

    device = f"cuda:{gpu_id}"
    print(f"  [GPU {gpu_id}] Loading ChatterBox TTS...")
    tts_model = ChatterboxTTS.from_pretrained(device=device)

    rng = random.Random(seed + gpu_id)
    completed = 0
    total_turns = sum(len(c["turns"]) for c in conv_shard)
    start = _time.time()

    for ci, conv in enumerate(conv_shard):
        conv_id = conv["conversation_id"]
        agent_voice = str(rng.choice(agent_voices))
        user_voice = str(rng.choice(user_voices))

        expected_turns = len(conv["turns"])
        existing = list(audio_dir.glob(f"{conv_id}_turn_*.wav"))
        if len(existing) >= expected_turns:
            completed += expected_turns
            continue

        elapsed = _time.time() - start
        if completed > 0:
            rate = elapsed / completed
            eta = (total_turns - completed) * rate
            eta_str = f"ETA {eta/3600:.1f}h" if eta > 3600 else f"ETA {eta/60:.0f}m"
        else:
            eta_str = "starting..."

        print(f"  [GPU {gpu_id}] [{ci + 1}/{len(conv_shard)}] {conv_id} [{eta_str}]")

        for ti, turn in enumerate(conv["turns"]):
            out_path = audio_dir / f"{conv_id}_turn_{ti:02d}_{turn['role']}.wav"
            if out_path.exists():
                completed += 1
                continue

            ref = agent_voice if turn["role"] == "agent" else user_voice
            try:
                wav = tts_model.generate(turn["text"], audio_prompt_path=ref)
                sf.write(str(out_path), wav.squeeze(0).cpu().numpy().T, tts_model.sr)
            except Exception as e:
                print(f"  [GPU {gpu_id}]   ERROR {conv_id} turn {ti}: {e}")
                silence = np.zeros((TARGET_SR * 2, 1), dtype=np.float32)
                sf.write(str(out_path), silence, TARGET_SR)
            completed += 1

    elapsed = _time.time() - start
    print(f"  [GPU {gpu_id}] Done! {completed} turns in {elapsed/3600:.1f}h")


def generate_tts(retail_only=False):
    import time as _time

    print("\n" + "=" * 60)
    print("STEP 3: Generate audio with Chatterbox TTS")
    print("=" * 60)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    agent_voices = sorted(AGENT_VOICE_DIR.glob("*.wav"))
    user_voices = sorted(USER_VOICE_DIR.glob("*.wav"))
    conversations = load_conversations(retail_only=retail_only)

    num_gpus = torch.cuda.device_count()
    print(f"  {len(conversations)} conversations, {len(agent_voices)} agent voices, {len(user_voices)} user voices")
    print(f"  GPUs available: {num_gpus}")

    if num_gpus <= 1:
        # Single GPU path
        print("  Running single-GPU TTS...")
        _tts_worker(0, conversations, agent_voices, user_voices, AUDIO_DIR, 123)
    else:
        # Multi-GPU: split conversations across GPUs
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        # Split conversations into shards
        shards = [[] for _ in range(num_gpus)]
        for i, conv in enumerate(conversations):
            shards[i % num_gpus].append(conv)

        print(f"  Splitting {len(conversations)} conversations across {num_gpus} GPUs:")
        for i, shard in enumerate(shards):
            print(f"    GPU {i}: {len(shard)} conversations")

        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=_tts_worker,
                args=(gpu_id, shards[gpu_id], agent_voices, user_voices, AUDIO_DIR, 123),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Check for failures
        failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
        if failed:
            print(f"  WARNING: GPU workers {failed} exited with errors")

    total_audio = len(list(AUDIO_DIR.glob("*.wav")))
    print(f"\n  TTS generation complete! {total_audio} audio files in {AUDIO_DIR}/")


# ============================================================
# Step 4: Mimi encode + assemble .pt files
# ============================================================

def assemble_pt_files(retail_only=False):
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

    conversations = load_conversations(retail_only=retail_only)
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

        # Build Hybrid System Prompt (PersonaPlex format)
        system_prompt = conv.get("system_prompt", "")
        hybrid_prompt_frames = 0
        hybrid_text_tokens = []
        hybrid_agent_audio = []
        hybrid_user_audio = []

        if system_prompt:
            # Wrap with <system> delimiters as PersonaPlex expects
            if not system_prompt.strip().startswith("<system>"):
                system_prompt = f"<system> {system_prompt.strip()} <system>"
            else:
                # Already has tags, ensure proper spacing
                system_prompt = system_prompt.replace("<system>", "<system> ").replace("  ", " ")

            # Encode system prompt text
            system_tokens = sp.Encode(system_prompt)

            # Phase 1: Silence padding before text prompt (~0.5s = 6 frames at 12.5 Hz)
            pre_silence_frames = 6
            for _ in range(pre_silence_frames):
                hybrid_text_tokens.append(PAD_TOKEN)
                hybrid_agent_audio.append(torch.tensor(SILENCE_TOKENS, dtype=torch.long).unsqueeze(1))
                hybrid_user_audio.append(torch.tensor(SINE_TOKENS, dtype=torch.long).unsqueeze(1))

            # Phase 2: Text prompt with system tokens
            # Distribute system tokens across frames (~2s = 25 frames)
            text_prompt_frames = max(25, len(system_tokens))  # At least 25 frames
            for i in range(text_prompt_frames):
                # Distribute tokens evenly across frames
                if i < len(system_tokens):
                    hybrid_text_tokens.append(system_tokens[i])
                else:
                    hybrid_text_tokens.append(PAD_TOKEN)  # Pad remaining frames

                hybrid_agent_audio.append(torch.tensor(SILENCE_TOKENS, dtype=torch.long).unsqueeze(1))
                hybrid_user_audio.append(torch.tensor(SINE_TOKENS, dtype=torch.long).unsqueeze(1))

            # Phase 3: Silence padding after text prompt (~0.5s = 6 frames)
            post_silence_frames = 6
            for _ in range(post_silence_frames):
                hybrid_text_tokens.append(PAD_TOKEN)
                hybrid_agent_audio.append(torch.tensor(SILENCE_TOKENS, dtype=torch.long).unsqueeze(1))
                hybrid_user_audio.append(torch.tensor(SINE_TOKENS, dtype=torch.long).unsqueeze(1))

            hybrid_prompt_frames = pre_silence_frames + text_prompt_frames + post_silence_frames

        agent_chunks = []
        user_chunks = []
        text_info = []
        frame_offset = hybrid_prompt_frames  # Start after hybrid system prompt

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

        # Concatenate conversation audio chunks
        agent_codes = torch.cat(agent_chunks, dim=1) if agent_chunks else torch.empty((8, 0), dtype=torch.long)
        user_codes = torch.cat(user_chunks, dim=1) if user_chunks else torch.empty((8, 0), dtype=torch.long)
        T_conv = agent_codes.shape[1]  # Conversation frames only
        T_total = hybrid_prompt_frames + T_conv  # Hybrid prompt + conversation

        # Build text token sequence
        text_tokens = torch.full((1, T_total), ZERO_TOKEN, dtype=torch.long)

        # Add hybrid prompt text tokens at the beginning
        for i, token in enumerate(hybrid_text_tokens):
            if i < T_total:
                text_tokens[0, i] = token

        # Add conversation text tokens (clustered at speech onset - CRITICAL FIX #2)
        # Based on Moshi research: text represents "inner monologue" before speaking
        # Tokens should cluster in first 30% of turn, not spread evenly
        for info in text_info:
            token_ids = sp.Encode(info["text"])
            start = info["start_frame"]  # Already includes hybrid_prompt_frames offset
            num_frames = info["num_frames"]

            # Cluster text at onset (first 30% of turn)
            # This mimics thinking → speaking pattern
            onset_frames = max(1, int(num_frames * 0.3))

            for j, tid in enumerate(token_ids):
                if j < onset_frames:  # Only place tokens in onset window
                    frame_idx = start + j
                    if frame_idx < T_total:
                        text_tokens[0, frame_idx] = tid
                # Tokens beyond onset window are left as padding (-1)
                # Model learns: "stop thinking, start speaking"

        # Concatenate hybrid prompt audio with conversation audio
        if hybrid_prompt_frames > 0:
            # Concatenate all hybrid prompt audio frames
            hybrid_agent_audio_cat = torch.cat(hybrid_agent_audio, dim=1)  # Shape: (8, hybrid_prompt_frames)
            hybrid_user_audio_cat = torch.cat(hybrid_user_audio, dim=1)    # Shape: (8, hybrid_prompt_frames)

            # Prepend hybrid prompt to conversation
            agent_codes = torch.cat([hybrid_agent_audio_cat, agent_codes], dim=1)
            user_codes = torch.cat([hybrid_user_audio_cat, user_codes], dim=1)

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

    print(f"  Uploading {len(pt_files)} .pt files to {HF_REPO} (using upload_folder)...")

    api.upload_folder(
        folder_path=str(PT_DIR),
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns="*.pt",
        commit_message=f"Upload {len(pt_files)} .pt files",
    )

    print(f"\n  Done! {len(pt_files)} files uploaded to {HF_REPO}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Full PersonaPlex data pipeline")
    parser.add_argument("--retail-only", action="store_true",
                        help="Process only retail conversations (conv_0201-conv_1200 from retail_training.json)")
    parser.add_argument("--skip-conversations", action="store_true")
    parser.add_argument("--skip-voices", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-assemble", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    if args.retail_only:
        print("=" * 60)
        print("RETAIL-ONLY MODE: Processing conv_0201-conv_1200")
        print("=" * 60)

    if not args.skip_conversations:
        download_conversations(retail_only=args.retail_only)
    if not args.skip_voices:
        download_voices()
    if not args.skip_tts:
        generate_tts(retail_only=args.retail_only)
    if not args.skip_assemble:
        assemble_pt_files(retail_only=args.retail_only)
    if not args.skip_upload:
        upload_pt_files()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("Next: python lora_train_fixed.py --epochs 4 --grad-accum-steps 8")


if __name__ == "__main__":
    main()
