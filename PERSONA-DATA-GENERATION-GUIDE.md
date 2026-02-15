# Persona-Consistent Data Generation Guide

**Fix for Issue #5: Inconsistent Personas**

Based on PersonaPlex research: Fisher corpus conversations with **back-annotated prompts** using GPT-OSS-120B.

---

## The Problem

**Current approach:**
- 200 conversations with 200 unique system prompts
- Each persona appears only ONCE
- Model can't learn stable character traits
- Results in hallucinations and inconsistent behavior

**PersonaPlex approach:**
> "7,303 real conversations from the Fisher English corpus, with conversations **back-annotated with prompts** using GPT-OSS-120B. The prompts have various levels of details in order to balance between generalization and instruction following ability."

---

## The Solution

**Generate 20 personas × 10 conversations each = 200 total**

Each persona:
- Same character traits across all 10 conversations
- Different scenarios (morning rush, quiet afternoon, etc.)
- Different orders/requests
- Consistent speech patterns

---

## Step 1: Define 20 Personas

Create `personas.json`:

```json
[
  {
    "id": "persona_001",
    "name": "Angry Alex",
    "type": "angry",
    "traits": "Frequent customer at Morning Grind cafe. Lactose intolerant but often receives wrong milk. Short temper, interrupts often with phrases like 'Just fix it!' Escalates quickly but calms down if issue is resolved promptly.",
    "speech_patterns": [
      "Interrupts mid-sentence",
      "Uses phrases like 'This is unacceptable', 'I've been waiting...'",
      "Minimal pause before responding (80-240ms gaps)",
      "Speaks loudly, clipped sentences"
    ],
    "conversation_count": 10
  },
  {
    "id": "persona_002",
    "name": "Nervous Nancy",
    "type": "nervous",
    "traits": "New to specialty coffee. Works nearby as administrative assistant. Intimidated by fancy menus and coffee terminology. Speaks hesitantly with long pauses, apologizes frequently ('Sorry, um...'), asks many clarifying questions.",
    "speech_patterns": [
      "Filler words: 'um', 'uh', 'like', 'I mean'",
      "Long pauses (480-1200ms gaps)",
      "Rising intonation (uncertain)",
      "Apologizes often: 'Sorry to bother you...'",
      "Questions everything: 'What's the difference between...?'"
    ],
    "conversation_count": 10
  },
  {
    "id": "persona_003",
    "name": "Friendly Frank",
    "type": "friendly",
    "traits": "Regular customer who visits every morning before work. Software engineer, loves to chat. Knows baristas by name. Upbeat, warm tone. Always orders the same drink (large black coffee). Makes small talk about weather, weekends, etc.",
    "speech_patterns": [
      "Casual greetings: 'Hey!', 'What's up?'",
      "Uses barista names",
      "Normal pacing (320-640ms gaps)",
      "Conversational flow, tells stories",
      "Positive language: 'Great!', 'Perfect!', 'Thanks so much!'"
    ],
    "conversation_count": 10
  },
  {
    "id": "persona_004",
    "name": "Indecisive Iris",
    "type": "indecisive",
    "traits": "Student at nearby university. Always has trouble deciding what to order. Changes mind multiple times. Asks about every option, compares extensively. Not rude, just genuinely can't decide.",
    "speech_patterns": [
      "Thinking words: 'Hmm...', 'Let me think...', 'Maybe...'",
      "Changes mid-sentence: 'Actually, wait...'",
      "Medium-long pauses (400-960ms gaps)",
      "Asks for recommendations constantly",
      "Compares options: 'Which is better, X or Y?'"
    ],
    "conversation_count": 10
  },
  {
    "id": "persona_005",
    "name": "Coffee Snob Carl",
    "type": "enthusiast",
    "traits": "Coffee enthusiast who knows technical terminology. Works as a product manager. Particular about brewing methods, bean origins, temperatures. Not rude but very specific about orders. Appreciates quality.",
    "speech_patterns": [
      "Technical terms: 'single-origin', 'pour-over', 'extraction', 'bloom'",
      "Specific requests: 'Can you do 202°F?'",
      "Normal pacing but detailed (400-800ms gaps)",
      "Discusses flavor notes, origins",
      "Appreciative when done well: 'Excellent crema'"
    ],
    "conversation_count": 10
  }
  // ... 15 more personas
]
```

**Persona distribution:**
- 5 angry (different trigger situations)
- 5 nervous (different anxiety patterns)
- 5 friendly (different conversational styles)
- 3 indecisive (different decision paralysis types)
- 2 enthusiast (coffee snobs with different knowledge levels)

---

## Step 2: Generate Conversations with GPT-OSS-120B

**For each persona, generate 10 different scenarios:**

```python
import json
from openai import OpenAI  # Or use GPT-OSS-120B API

client = OpenAI(
    base_url="https://api.openai.com/v1",  # Or GPT-OSS endpoint
    api_key="your_api_key"
)

# Load personas
with open('personas.json') as f:
    personas = json.load(f)

conversations = []

for persona in personas:
    for scenario_num in range(10):
        # Generate scenario-specific context
        scenarios = [
            "Morning rush hour, long line behind customer",
            "Quiet afternoon, barista has time to chat",
            "Mobile order pickup, drink may not be ready",
            "First-time visit to this specific cafe",
            "Regular visit, same order as always",
            "Special request (dietary restriction)",
            "Drink remake needed (wrong order)",
            "Trying a new drink from the menu",
            "Ordering for a group/coworkers",
            "Closing time, limited options available"
        ]

        scenario = scenarios[scenario_num]

        # Construct persona-aware prompt
        prompt = f"""Generate a realistic coffee shop conversation between a customer and barista.

CUSTOMER PERSONA:
{json.dumps(persona, indent=2)}

SCENARIO:
{scenario}

REQUIREMENTS:
1. Customer (agent role) speaks first
2. 15-25 total turns, alternating agent/user
3. Customer maintains personality traits throughout
4. Customer uses their characteristic speech patterns
5. Conversation stays focused on coffee shop context (ordering, menu questions, etc.)
6. Natural pauses appropriate for personality type
7. Realistic dialogue for the given scenario

Generate in JSON format:
{{
  "persona_id": "{persona['id']}",
  "conversation_id": "conv_{{4-digit-number}}",
  "scenario": "{scenario}",
  "system_prompt": "<system> [Auto-generated from persona traits] </system>",
  "turns": [
    {{"role": "agent", "text": "..."}},
    {{"role": "user", "text": "..."}},
    ...
  ]
}}

The system_prompt should describe the customer's personality, current situation, and speech patterns as PersonaPlex expects."""

        # Call GPT-OSS-120B
        response = client.chat.completions.create(
            model="gpt-oss-120b",  # Or appropriate model
            messages=[
                {"role": "system", "content": "You are an expert at generating realistic conversational dialogue for training speech AI models."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=4000
        )

        # Parse response
        conv_json = json.loads(response.choices[0].message.content)

        # Auto-generate system prompt from persona traits
        system_prompt = generate_system_prompt(persona, scenario)
        conv_json['system_prompt'] = system_prompt

        conversations.append(conv_json)

        print(f"✓ Generated {persona['id']} conversation {scenario_num + 1}/10")

# Save
with open('training_conversations.json', 'w') as f:
    json.dump(conversations, f, indent=2)

print(f"\n✓ Generated {len(conversations)} persona-consistent conversations!")
```

---

## Step 3: Auto-Generate System Prompts

**System prompt format (PersonaPlex style):**

```python
def generate_system_prompt(persona, scenario):
    """
    Auto-generate PersonaPlex system prompt from persona traits.

    Format: <system> [Role] at [Location]. [Personality]. [Situation]. [Speech patterns]. </system>
    """

    # Extract details
    name = persona['name']
    traits = persona['traits']
    speech = ', '.join(persona['speech_patterns'][:3])  # Top 3 patterns

    # Map scenario to specific situation
    situation_map = {
        "Morning rush hour, long line behind customer": "You're in a hurry and there's a long line behind you. You need your order quickly.",
        "Quiet afternoon, barista has time to chat": "You have time to chat and aren't rushed.",
        "Mobile order pickup, drink may not be ready": "You ordered ahead on mobile but your drink isn't ready yet.",
        # ... etc
    }

    situation = situation_map.get(scenario, "")

    # Construct prompt
    prompt = f"<system> You are {name}. {traits} {situation} {speech} Stay focused on coffee shop topics only. </system>"

    return prompt


# Example outputs:
# Angry Alex: "<system> You are Angry Alex. Frequent customer at Morning Grind cafe. Lactose intolerant but often receives wrong milk. Short temper, interrupts often. You ordered ahead on mobile but your drink isn't ready yet. Interrupts mid-sentence, uses phrases like 'This is unacceptable', minimal pause before responding. Stay focused on coffee shop topics only. </system>"

# Nervous Nancy: "<system> You are Nervous Nancy. New to specialty coffee. Intimidated by fancy menus and coffee terminology. You're in a hurry and there's a long line behind you. Filler words: 'um', 'uh', 'like', long pauses, rising intonation. Stay focused on coffee shop topics only. </system>"
```

---

## Step 4: Validation

```python
import json

with open('training_conversations.json') as f:
    data = json.load(f)

# Check persona consistency
personas = {}
for conv in data:
    pid = conv['persona_id']
    personas[pid] = personas.get(pid, 0) + 1

print(f"Total conversations: {len(data)}")
print(f"Total personas: {len(personas)}")

for pid, count in sorted(personas.items()):
    if count != 10:
        print(f"  ⚠️  {pid}: {count} conversations (expected 10)")
    else:
        print(f"  ✓ {pid}: {count} conversations")

# Check turn counts
for conv in data:
    num_turns = len(conv['turns'])
    if num_turns < 15 or num_turns > 25:
        print(f"  ⚠️  {conv['conversation_id']}: {num_turns} turns (expected 15-25)")

# Check alternating roles
for conv in data:
    if conv['turns'][0]['role'] != 'agent':
        print(f"  ❌ {conv['conversation_id']}: First turn is not 'agent'")

    for i in range(1, len(conv['turns'])):
        if conv['turns'][i]['role'] == conv['turns'][i-1]['role']:
            print(f"  ❌ {conv['conversation_id']}: Non-alternating roles at turn {i}")
            break
```

---

## Step 5: Upload to HuggingFace

```bash
huggingface-cli upload \
  AnthrolyticB/personaplex-training-data-v2 \
  training_conversations.json \
  training_conversations.json \
  --repo-type dataset

echo "✓ Uploaded persona-consistent training data"
```

---

## Expected Benefits

**Before (random personas):**
- Model sees each persona once
- Can't learn stable character traits
- Hallucinations from inconsistent behavior
- Poor instruction following

**After (20 personas × 10 each):**
- ✅ Model learns 20 stable personas
- ✅ Consistent character traits across conversations
- ✅ Better generalization to new personas
- ✅ Reduced hallucinations
- ✅ Improved instruction following

**Training improvements:**
- Semantic loss drops faster (better context learning)
- More stable loss curves
- Better validation performance
- Model generalizes personas better at inference

---

## Alternative: Back-Annotation (PersonaPlex Method)

If you have existing unscripted conversations (like Fisher corpus), use GPT-OSS-120B to back-annotate:

```python
def back_annotate_conversation(conversation_transcript):
    """
    Generate system prompt from existing conversation.
    PersonaPlex approach for Fisher corpus.
    """

    prompt = f"""Given this conversation between a customer and barista, generate a detailed
system prompt that describes the customer's personality, background, and speech patterns.

Conversation:
{conversation_transcript}

Generate a PersonaPlex system prompt in the format:
<system> You are [NAME]. [PERSONALITY TRAITS]. [BACKGROUND]. [SPEECH PATTERNS]. Stay focused on coffee shop topics only. </system>

The prompt should capture:
- Personality type (angry, nervous, friendly, indecisive, enthusiast)
- Specific character details
- How they speak (pace, filler words, interruptions, etc.)
- Current situation/context
"""

    response = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content
```

---

## Summary

1. **Define 20 distinct personas** with detailed traits
2. **Generate 10 conversations per persona** (200 total)
3. **Auto-generate system prompts** from persona traits
4. **Validate** persona consistency and turn structure
5. **Upload** to HuggingFace dataset

**Result:** Model learns stable persona characteristics, reducing hallucinations and improving instruction following!

---

## References

- [PersonaPlex Paper](https://arxiv.org/abs/2602.06053)
- [Fisher English Corpus](https://catalog.ldc.upenn.edu/LDC2004T19)
- [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b)

**Guide Status:** Ready for implementation
**Est. Time:** 4-6 hours to generate 200 conversations
**Recommendation:** Start with 5 personas × 10 convs = 50 total for testing, then scale to 20
