#!/usr/bin/env python3
"""
Upload training.json to HuggingFace dataset repository.

Usage:
    python upload_training.py training.json
    python upload_training.py training.json --repo YOUR_USERNAME/personaplex-training-data-v2
    python upload_training.py training.json --validate-only
"""

import json
import sys
import argparse
from pathlib import Path
from collections import Counter
from huggingface_hub import HfApi, create_repo


def validate_training_data(filepath: Path) -> tuple[bool, list[str]]:
    """
    Validate training.json structure and persona consistency.

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except FileNotFoundError:
        return False, [f"File not found: {filepath}"]

    # Check: Must be a list
    if not isinstance(data, list):
        errors.append("Root element must be a JSON array")
        return False, errors

    # Check: Total conversations
    total = len(data)
    if total == 0:
        errors.append("No conversations found")
        return False, errors

    print(f"\nValidating {total} conversations...")

    # Check: Persona distribution
    personas = Counter(c.get('persona_id', 'MISSING') for c in data)

    if 'MISSING' in personas:
        errors.append(f"{personas['MISSING']} conversations missing 'persona_id'")

    num_personas = len([p for p in personas.keys() if p != 'MISSING'])
    print(f"  ✓ Found {num_personas} unique personas")

    # Warn if not 20 personas × 10 each (but don't fail)
    expected_personas = 20
    expected_per_persona = total // expected_personas if total % expected_personas == 0 else None

    if num_personas != expected_personas:
        errors.append(f"Expected {expected_personas} personas, found {num_personas}")

    if expected_per_persona:
        for persona_id, count in personas.items():
            if persona_id == 'MISSING':
                continue
            if count != expected_per_persona:
                errors.append(f"{persona_id}: {count} conversations (expected {expected_per_persona})")

    # Check: Conversation structure
    for i, conv in enumerate(data):
        conv_id = conv.get('conversation_id', f'conv_{i:04d}')

        # Required fields
        if 'turns' not in conv:
            errors.append(f"{conv_id}: Missing 'turns' field")
            continue

        turns = conv['turns']

        # Check: Turn count
        num_turns = len(turns)
        if num_turns < 15:
            errors.append(f"{conv_id}: Only {num_turns} turns (expected 15-25)")
        elif num_turns > 25:
            errors.append(f"{conv_id}: {num_turns} turns (expected 15-25, but OK)")

        # Check: First turn is agent
        if turns and turns[0].get('role') != 'agent':
            errors.append(f"{conv_id}: First turn must be 'agent' (customer), got '{turns[0].get('role')}'")

        # Check: Alternating roles
        for j in range(1, len(turns)):
            prev_role = turns[j-1].get('role')
            curr_role = turns[j].get('role')

            if prev_role == curr_role:
                errors.append(f"{conv_id}: Non-alternating roles at turn {j} ({prev_role} → {curr_role})")
                break

        # Check: Role values
        for j, turn in enumerate(turns):
            role = turn.get('role')
            if role not in ['agent', 'user']:
                errors.append(f"{conv_id}: Turn {j} has invalid role '{role}' (expected 'agent' or 'user')")

            if 'text' not in turn or not turn['text'].strip():
                errors.append(f"{conv_id}: Turn {j} missing or empty 'text'")

        # Check: System prompt exists
        if 'system_prompt' not in conv or not conv['system_prompt'].strip():
            errors.append(f"{conv_id}: Missing 'system_prompt'")

    # Summary
    print(f"\nValidation Summary:")
    print(f"  Total conversations: {total}")
    print(f"  Unique personas: {num_personas}")

    if num_personas > 0:
        avg_per_persona = total / num_personas
        print(f"  Avg conversations per persona: {avg_per_persona:.1f}")

    if errors:
        print(f"\n❌ Found {len(errors)} issues:")
        for err in errors[:20]:  # Show first 20
            print(f"     - {err}")
        if len(errors) > 20:
            print(f"     ... and {len(errors) - 20} more")
        return False, errors
    else:
        print(f"\n✓ All validation checks passed!")
        return True, []


def upload_to_huggingface(filepath: Path, repo_id: str, token: str = None):
    """
    Upload training.json to HuggingFace dataset repository.

    Args:
        filepath: Path to training.json
        repo_id: HuggingFace repo ID (e.g., "username/dataset-name")
        token: HuggingFace API token (or use HF_TOKEN env var)
    """
    api = HfApi(token=token)

    # Check if repo exists, create if not
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"\n✓ Repository exists: {repo_id}")
    except Exception:
        print(f"\nCreating new dataset repository: {repo_id}")
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                exist_ok=True
            )
            print(f"✓ Created repository: {repo_id}")
        except Exception as e:
            print(f"❌ Failed to create repository: {e}")
            print("\nMake sure:")
            print("  1. You have a valid HuggingFace token (HF_TOKEN env var or --token)")
            print("  2. The token has 'write' permissions")
            print("  3. The repository name is valid (username/dataset-name)")
            sys.exit(1)

    # Upload file
    print(f"\nUploading {filepath.name} to {repo_id}...")

    try:
        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo="training.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload training data ({filepath.stat().st_size // 1024}KB)"
        )
        print(f"\n✅ Successfully uploaded to https://huggingface.co/datasets/{repo_id}")
        print(f"\nDownload command:")
        print(f"  huggingface-cli download {repo_id} training.json \\")
        print(f"    --repo-type dataset --local-dir /workspace --force-download")
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Upload training.json to HuggingFace dataset repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate only (no upload)
  python upload_training.py training.json --validate-only

  # Upload to default repo
  python upload_training.py training.json

  # Upload to custom repo
  python upload_training.py training.json --repo username/my-dataset

  # Use specific HuggingFace token
  python upload_training.py training.json --token hf_xxxxx

Environment Variables:
  HF_TOKEN    HuggingFace API token (can be set instead of --token)
        """
    )

    parser.add_argument(
        'filepath',
        type=Path,
        help='Path to training.json file'
    )

    parser.add_argument(
        '--repo',
        type=str,
        default='AnthrolyticB/personaplex-training-data-v2',
        help='HuggingFace dataset repo ID (default: AnthrolyticB/personaplex-training-data-v2)'
    )

    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='HuggingFace API token (or use HF_TOKEN env var)'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate, do not upload'
    )

    args = parser.parse_args()

    # Validate
    print("="*80)
    print("Training Data Validation & Upload")
    print("="*80)

    is_valid, errors = validate_training_data(args.filepath)

    if not is_valid:
        print(f"\n❌ Validation failed with {len(errors)} errors")
        print("\nFix these issues before uploading.")
        sys.exit(1)

    if args.validate_only:
        print("\n✓ Validation complete (--validate-only, skipping upload)")
        sys.exit(0)

    # Upload
    upload_to_huggingface(
        filepath=args.filepath,
        repo_id=args.repo,
        token=args.token
    )


if __name__ == '__main__':
    main()
