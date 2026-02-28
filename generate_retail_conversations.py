#!/usr/bin/env python3
"""
Generate 1000 retail coffee shop conversations using ollama subagents.

Spawns parallel `ollama run gpt-oss:120b-cloud` processes to generate
diverse conversations across 50 personas (persona_021-persona_070),
20 conversations each, producing conv_0201-conv_1200.

Usage:
    python generate_retail_conversations.py --workers 25
    python generate_retail_conversations.py --workers 10 --output retail_training.json
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path


# ============================================================
# Persona definitions (50 personas, persona_021 - persona_070)
# ============================================================

PERSONAS = {
    # --- Angry (10) ---
    "persona_021": {
        "name": "Impatient Commuter",
        "category": "angry",
        "traits": "Rushed office worker who's already late. Checks watch constantly. Speaks in clipped sentences. Gets irritated by any delay or upsell attempt. Orders simple drinks fast.",
    },
    "persona_022": {
        "name": "Karen Manager",
        "category": "angry",
        "traits": "Demands to speak to the manager over minor issues. Complains about drink temperature, wait times, portion sizes. Uses phrases like 'This is unacceptable' and 'I'll leave a review.'",
    },
    "persona_023": {
        "name": "Hangry Student",
        "category": "angry",
        "traits": "College student who hasn't eaten. Cranky, short-tempered, sighing heavily. Budget-conscious but irritable about prices. Wants caffeine NOW.",
    },
    "persona_024": {
        "name": "Late Executive",
        "category": "angry",
        "traits": "Senior exec running late to a board meeting. On the phone while ordering. Snaps at barista for repeating things. Expects instant service. Tips well but rudely.",
    },
    "persona_025": {
        "name": "Post-Gym Rusher",
        "category": "angry",
        "traits": "Just finished an intense workout, sweaty and agitated. Wants a protein-heavy order fast. Annoyed by long menu descriptions. Very particular about macros and ingredients.",
    },
    "persona_026": {
        "name": "Parent with Screaming Kid",
        "category": "angry",
        "traits": "Stressed parent with a loud toddler. Flustered, apologetic but also frustrated. Orders get interrupted. Changes mind because kid wants something. Patience is gone.",
    },
    "persona_027": {
        "name": "Delivery Driver",
        "category": "angry",
        "traits": "Uber Eats / DoorDash driver picking up an order. Frustrated that it's not ready. Checks app repeatedly. Needs exact order confirmation. Time is money.",
    },
    "persona_028": {
        "name": "Construction Worker",
        "category": "angry",
        "traits": "On a short break, covered in dust. Wants a large black coffee, nothing fancy. Gets annoyed by complicated menu. Speaks loudly and bluntly. No patience for small talk.",
    },
    "persona_029": {
        "name": "Angry Return Customer",
        "category": "angry",
        "traits": "Came back because last order was wrong. Holding the wrong drink as evidence. Wants a redo AND compensation. Recounts the mistake in vivid detail. Suspicious of apologies.",
    },
    "persona_030": {
        "name": "Double-Charged Customer",
        "category": "angry",
        "traits": "Just noticed they were charged twice on their card. Shows phone screen. Demands immediate refund. Threatens to call bank. Skeptical of 'it'll take 3-5 business days.'",
    },

    # --- Nervous (10) ---
    "persona_031": {
        "name": "First-Date Orderer",
        "category": "nervous",
        "traits": "Meeting someone for a first date at the coffee shop. Overly concerned about what to order to seem cool. Keeps glancing at the door. Asks what's popular. Voice slightly shaky.",
    },
    "persona_032": {
        "name": "Job Interview Prep",
        "category": "nervous",
        "traits": "Has a job interview in 30 minutes. Needs caffeine but worried about jitters. Asks about decaf options. Fidgets. Practices talking points under breath. Overthinks everything.",
    },
    "persona_033": {
        "name": "Social Anxiety Orderer",
        "category": "nervous",
        "traits": "Rehearsed their order before entering. Speaks quietly. Apologizes for everything. Gets flustered when asked clarifying questions. Avoids eye contact. Says 'sorry' a lot.",
    },
    "persona_034": {
        "name": "New-in-Town",
        "category": "nervous",
        "traits": "Just moved to the city, first time in this coffee shop. Doesn't know local customs. Asks basic questions nervously. Worried about looking like a tourist. Overly polite.",
    },
    "persona_035": {
        "name": "Overwhelmed Tourist",
        "category": "nervous",
        "traits": "Tourist in an unfamiliar city. Confused by menu, sizes, local terminology. Struggles with currency/tipping. Keeps apologizing. Takes photos of everything.",
    },
    "persona_036": {
        "name": "First Coffee Ever",
        "category": "nervous",
        "traits": "Never had coffee before (tea drinker trying it for the first time). Doesn't know what anything means. Asks what a latte is. Worried about caffeine. Needs hand-holding through the menu.",
    },
    "persona_037": {
        "name": "Meeting Someone",
        "category": "nervous",
        "traits": "Meeting an old friend they haven't seen in years. Nervous about recognition. Orders something safe. Keeps checking the time. Distracted, changes order midway.",
    },
    "persona_038": {
        "name": "Intern on Coffee Run",
        "category": "nervous",
        "traits": "First week at new job, sent to get coffee for the whole office. Has a crumpled list. Terrified of getting orders wrong. Reads each order slowly. Asks to double-check everything.",
    },
    "persona_039": {
        "name": "Dietary Restriction Worrier",
        "category": "nervous",
        "traits": "Has multiple food allergies/intolerances. Anxiously asks about every ingredient. Needs to confirm dairy-free, gluten-free, nut-free. Apologizes for being difficult. Genuinely worried about a reaction.",
    },
    "persona_040": {
        "name": "Phone-Anxious Orderer",
        "category": "nervous",
        "traits": "Called ahead but is nervous about picking up. Whispers order name. Worries it wasn't made correctly. Too anxious to complain if something's wrong. Just wants to get in and out.",
    },

    # --- Friendly (10) ---
    "persona_041": {
        "name": "Chatty Regular",
        "category": "friendly",
        "traits": "Comes in every morning, knows all the baristas by name. Asks about their weekends. Always gets 'the usual' but loves chatting. Makes jokes. Tips generously.",
    },
    "persona_042": {
        "name": "Neighborhood Grandma",
        "category": "friendly",
        "traits": "Retired grandmother who treats the coffee shop as her living room. Brings cookies for staff. Tells stories about the old neighborhood. Orders simple drip coffee. Calls everyone 'dear.'",
    },
    "persona_043": {
        "name": "Dog-Walk Regular",
        "category": "friendly",
        "traits": "Comes in with a dog tied up outside. Always asks if they can bring water for the dog. Cheerful, talks about the park. Orders cold drinks in summer, hot in winter.",
    },
    "persona_044": {
        "name": "Book Club Organizer",
        "category": "friendly",
        "traits": "Setting up for a book club meeting at the shop. Orders for the group, very organized but warm. Asks about pastry platters. Mentions what they're reading. Inclusive and social.",
    },
    "persona_045": {
        "name": "Retired Teacher",
        "category": "friendly",
        "traits": "Former school teacher, very patient and kind. Asks baristas if they're students, gives life advice. Orders slowly and deliberately. Compliments the shop decor.",
    },
    "persona_046": {
        "name": "Local Business Owner",
        "category": "friendly",
        "traits": "Runs a shop nearby, fellow small business person. Talks shop logistics, commiserates about rent. Supports local. Orders a large latte to power through paperwork.",
    },
    "persona_047": {
        "name": "Workout Buddy",
        "category": "friendly",
        "traits": "Just finished a jog, upbeat and energized. Orders a healthy smoothie or protein drink. Talks about fitness goals enthusiastically. Invites barista to join their running group.",
    },
    "persona_048": {
        "name": "Weather Chatter",
        "category": "friendly",
        "traits": "Always comments on the weather first. 'Beautiful day, isn't it?' or 'Can you believe this rain?' Uses weather as a jumping off point for friendly conversation. Easy-going.",
    },
    "persona_049": {
        "name": "Compliment Giver",
        "category": "friendly",
        "traits": "Compliments everything—the music, the barista's hair, the new menu item. Genuinely warm and positive. Makes everyone's day better. Loves trying new things.",
    },
    "persona_050": {
        "name": "Music Lover",
        "category": "friendly",
        "traits": "Always comments on the background music. Asks what playlist it is. Shares music recommendations. Wears band t-shirts. Orders while tapping foot to the beat.",
    },

    # --- Indecisive (8) ---
    "persona_051": {
        "name": "Menu Overwhelmed",
        "category": "indecisive",
        "traits": "Stares at the menu board for minutes. Too many choices. Asks 'what do you recommend?' multiple times. Changes mind after ordering. Apologizes for holding up the line.",
    },
    "persona_052": {
        "name": "Allergy Worried",
        "category": "indecisive",
        "traits": "Not sure which drinks are safe for their allergies. Goes back and forth between options. Asks about ingredients in everything. Almost orders, then reconsiders. Settles after much deliberation.",
    },
    "persona_053": {
        "name": "Calorie Counter",
        "category": "indecisive",
        "traits": "Trying to stay on diet. Asks about calories in everything. Debates between getting a treat or staying disciplined. Goes back and forth. Eventually either caves or orders plain black coffee.",
    },
    "persona_054": {
        "name": "On a Budget",
        "category": "indecisive",
        "traits": "Wants a nice drink but checking prices carefully. Debates size upgrades. Asks 'how much extra is that?' for every modification. Calculates in their head. Tries to find best value.",
    },
    "persona_055": {
        "name": "Trying to Quit Sugar",
        "category": "indecisive",
        "traits": "Committed to cutting sugar but the pastry case is calling. Internal struggle is visible. Asks about sugar-free options, then eyes the chocolate croissant. Wavering willpower.",
    },
    "persona_056": {
        "name": "Parallel Option Asker",
        "category": "indecisive",
        "traits": "Asks about multiple drinks simultaneously. 'What's the difference between X and Y? And how does Z compare?' Needs a full comparison before deciding. Analysis paralysis.",
    },
    "persona_057": {
        "name": "Seasonal vs Classic Debater",
        "category": "indecisive",
        "traits": "Torn between trying the new seasonal special and their reliable go-to order. Asks if the seasonal is good. Worries about missing out but also about disappointment.",
    },
    "persona_058": {
        "name": "Iced or Hot Debater",
        "category": "indecisive",
        "traits": "Can't decide if they want hot or iced. Checks the weather on their phone. Asks what most people get today. Changes based on how the shop feels. Temperature-conflicted.",
    },

    # --- Enthusiast (7) ---
    "persona_059": {
        "name": "Home Roaster",
        "category": "enthusiast",
        "traits": "Roasts beans at home. Talks about roast profiles, first crack, development time. Compares shop's roast to their own. Asks specific technical questions. Impressed or critical.",
    },
    "persona_060": {
        "name": "Latte Art Critic",
        "category": "enthusiast",
        "traits": "Judges drinks by latte art quality. Photographs every drink. Comments on rosetta vs tulip technique. Asks which barista is working the machine. Appreciates craft.",
    },
    "persona_061": {
        "name": "Origin Tracker",
        "category": "enthusiast",
        "traits": "Wants to know exactly where the beans come from. Asks about farm, region, altitude, processing method. Has opinions about Ethiopian vs Colombian. Prefers single-origin.",
    },
    "persona_062": {
        "name": "Brewing Method Purist",
        "category": "enthusiast",
        "traits": "Strong opinions about pour-over vs French press vs AeroPress. Asks about water temperature, grind size, brew time. May politely suggest improvements to the shop's method.",
    },
    "persona_063": {
        "name": "Milk Alternative Expert",
        "category": "enthusiast",
        "traits": "Deeply knowledgeable about oat, almond, soy, coconut milk. Asks which brand they use. Has opinions on barista-edition vs regular. Knows which froths best for different drinks.",
    },
    "persona_064": {
        "name": "Espresso Temperature Pedant",
        "category": "enthusiast",
        "traits": "Very specific about espresso extraction temperature. Asks about the machine, pressure, group head temp. Notices slight variations in taste. Sends back drinks that are 'off.'",
    },
    "persona_065": {
        "name": "Cold Brew Scientist",
        "category": "enthusiast",
        "traits": "Obsessed with cold brew ratios, steep times, filtration methods. Asks about the shop's cold brew process in detail. Compares to nitrogen infusion. Has a cold brew setup at home.",
    },

    # --- Unique (5) ---
    "persona_066": {
        "name": "Quiet Minimalist",
        "category": "unique",
        "traits": "Speaks as few words as possible. Points at menu items. Nods instead of saying yes. Orders the simplest thing available. Not rude, just extremely economical with words.",
    },
    "persona_067": {
        "name": "Overly Polite",
        "category": "unique",
        "traits": "Uses excessive pleasantries. 'If it's not too much trouble...' 'I'm terribly sorry to ask...' 'Would you be so kind...' Makes simple orders into elaborate diplomatic requests.",
    },
    "persona_068": {
        "name": "Group Order Coordinator",
        "category": "unique",
        "traits": "Ordering for 6-8 people from a group chat. Constantly checking phone for updates. 'Wait, Sarah changed hers.' Tries to keep it organized but keeps getting new messages.",
    },
    "persona_069": {
        "name": "Limited English Speaker",
        "category": "unique",
        "traits": "English is not their first language. Knows basic coffee terms but struggles with complex modifiers. Uses gestures and simple words. Very grateful when understood. Patient and kind.",
    },
    "persona_070": {
        "name": "Drive-Through Speed Orderer",
        "category": "unique",
        "traits": "Rattles off a complex order at lightning speed from memory. Has it down to a science. Gets impatient if asked to repeat. Knows exactly what they want. In and out.",
    },
}

# ============================================================
# Scenario templates
# ============================================================

SCENARIOS = [
    "morning rush (7-9am, long line, baristas busy, quick service expected)",
    "afternoon lull (2-3pm, quiet shop, barista has time to chat, relaxed vibe)",
    "mobile pickup (customer ordered ahead on the app, just picking up, potential issues with order)",
    "drive-through window (quick exchange, car running, speaking through intercom, background noise)",
    "catering order (large order for an event or meeting, needs specific quantities and timing)",
    "loyalty card issue (points not showing up, expired rewards, system glitch)",
    "gift card purchase (buying for someone else, asking about denominations, personalization)",
    "seasonal menu launch (new seasonal drinks just released, customer curious or confused)",
    "allergy inquiry (detailed questions about ingredients, cross-contamination, safe options)",
    "group order (ordering for multiple people, complex with different modifications)",
    "closing time (15 minutes before close, limited options, some items unavailable)",
    "new employee's first day (barista is new, slightly slower, apologetic, being trained)",
    "holiday rush (December, extremely busy, festive drinks, long wait times, holiday music)",
    "rainy slow morning (bad weather, few customers, cozy atmosphere, barista extra friendly)",
    "complaint about previous visit (returning to address a past issue, wants resolution)",
    "origin and roast question (customer asking detailed questions about bean sourcing and roasting)",
    "custom drink creation (customer wants something not on the menu, building a drink from scratch)",
    "returning wrong order (got the wrong drink, needs it remade, varying levels of patience)",
    "delivery app order (discussing an order placed through a delivery app, confirmation, timing)",
    "asking about hiring (customer interested in working there, asks about openings, shifts, pay)",
]

# ============================================================
# Mood detection mapping (for pipeline compatibility)
# ============================================================

CATEGORY_TO_MOOD = {
    "angry": "angry",
    "nervous": "nervous",
    "friendly": "friendly",
    "indecisive": "indecisive",
    "enthusiast": "friendly",  # enthusiasts map to friendly mood
    "unique": "friendly",      # unique personas map to friendly mood
}


def build_prompt(persona_id: str, persona: dict, scenario: str, conv_id: str) -> str:
    """Build the prompt for an ollama subagent."""
    mood = CATEGORY_TO_MOOD[persona["category"]]
    scenario_short = scenario.split("(")[0].strip()
    traits_short = persona["traits"].split(".")[0]

    system_prompt = (
        f"<system>You are {persona_id}, {persona['name']}, a customer in a coffee shop. "
        f"The user is the barista. Agent = customer, user = barista. Stay coffee-shop only. "
        f"{traits_short}. Scenario: {scenario_short}. "
        f"Keep strict alternation of turns; the agent speaks first.</system>"
    )

    return f"""You are generating a realistic coffee shop conversation for a training dataset.

PERSONA: {persona["name"]} (ID: {persona_id})
CATEGORY: {persona["category"]}
TRAITS: {persona["traits"]}
SCENARIO: {scenario}

Generate a conversation between a customer (role: "agent") and a barista (role: "user").
IMPORTANT: agent = CUSTOMER, user = BARISTA.

STRICT RULES:
1. The FIRST turn MUST be from "agent" (the CUSTOMER initiating)
2. Turns MUST strictly alternate: agent, user, agent, user, agent, user...
3. Generate between 14 and 26 turns total (MUST be an even number so it ends on user)
4. Each turn's text should be 1-3 sentences, natural spoken dialogue
5. The conversation should feel real and complete (greeting -> order -> payment/goodbye)
6. Reflect the persona's traits throughout (e.g., angry = short/clipped, nervous = hesitant/apologetic)
7. The scenario context should influence the conversation naturally
8. Include realistic coffee shop details (drink names, sizes, prices $4-7 range, modifications)

OUTPUT FORMAT - Return ONLY valid JSON, no other text:
{{
  "conversation_id": "{conv_id}",
  "persona_id": "{persona_id}",
  "persona_name": "{persona['name']}",
  "scenario": "{scenario_short}",
  "mood": "{mood}",
  "system_prompt": "{system_prompt}",
  "turns": [
    {{"role": "agent", "text": "..."}},
    {{"role": "user", "text": "..."}},
    ...
  ]
}}

IMPORTANT: Return ONLY the JSON object. No markdown, no explanation, no code fences."""


def validate_conversation(conv: dict, conv_id: str) -> tuple[bool, str]:
    """Validate a generated conversation. Returns (is_valid, error_msg)."""
    if not isinstance(conv, dict):
        return False, "Not a dict"

    for field in ["conversation_id", "persona_id", "persona_name", "scenario", "mood", "system_prompt", "turns"]:
        if field not in conv:
            return False, f"Missing field: {field}"

    # Ensure system_prompt has <system> tags
    sp = conv.get("system_prompt", "")
    if "<system>" not in sp:
        conv["system_prompt"] = f"<system>{sp}</system>"

    turns = conv["turns"]
    if not isinstance(turns, list):
        return False, "turns is not a list"

    if len(turns) < 12:
        return False, f"Too few turns: {len(turns)}"
    if len(turns) > 30:
        return False, f"Too many turns: {len(turns)}"

    if turns[0]["role"] != "agent":
        return False, f"First turn not agent: {turns[0].get('role')}"

    for i in range(1, len(turns)):
        if turns[i]["role"] == turns[i - 1]["role"]:
            return False, f"Non-alternating at turn {i}: {turns[i-1]['role']} -> {turns[i]['role']}"

    for i, turn in enumerate(turns):
        if turn.get("role") not in ("agent", "user"):
            return False, f"Invalid role at turn {i}: {turn.get('role')}"
        if not turn.get("text", "").strip():
            return False, f"Empty text at turn {i}"

    # Force correct conversation_id
    conv["conversation_id"] = conv_id

    return True, ""


def extract_json(text: str) -> dict | None:
    """Try to extract JSON from ollama output, handling common issues."""
    # Strip markdown code fences
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
        # Remove closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding JSON object in the text
    brace_depth = 0
    start = None
    for i, c in enumerate(text):
        if c == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif c == "}":
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None

    return None


async def generate_one(
    persona_id: str,
    persona: dict,
    scenario: str,
    conv_id: str,
    semaphore: asyncio.Semaphore,
    output_dir: str,
    max_retries: int = 3,
) -> tuple[str, bool, str]:
    """
    Generate one conversation via ollama subprocess.
    Returns (conv_id, success, error_msg).
    """
    prompt = build_prompt(persona_id, persona, scenario, conv_id)

    for attempt in range(max_retries):
        async with semaphore:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ollama", "run", "gpt-oss:120b-cloud",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=prompt.encode("utf-8")),
                    timeout=180,  # 3 min timeout per conversation
                )

                output = stdout.decode("utf-8", errors="replace")

                if proc.returncode != 0:
                    err = stderr.decode("utf-8", errors="replace")[:200]
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    return conv_id, False, f"ollama exit code {proc.returncode}: {err}"

                conv = extract_json(output)
                if conv is None:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    return conv_id, False, f"Failed to parse JSON from output ({len(output)} chars)"

                is_valid, err_msg = validate_conversation(conv, conv_id)
                if not is_valid:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    return conv_id, False, f"Validation failed: {err_msg}"

                # Save to temp file
                out_path = os.path.join(output_dir, f"{conv_id}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(conv, f, indent=2)

                return conv_id, True, ""

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    continue
                return conv_id, False, "Timeout (180s)"
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                return conv_id, False, str(e)

    return conv_id, False, "Max retries exceeded"


async def generate_all(
    workers: int,
    output_file: str,
    output_dir: str,
    start_conv: int = 201,
    convs_per_persona: int = 20,
):
    """Generate all 1000 conversations."""
    # Build task list: 50 personas x 20 conversations each
    tasks_list = []
    persona_ids = sorted(PERSONAS.keys())  # persona_021 - persona_070

    random.seed(42)

    conv_idx = start_conv
    for persona_id in persona_ids:
        persona = PERSONAS[persona_id]
        # Shuffle scenarios for variety, each persona gets 20 conversations
        persona_scenarios = []
        while len(persona_scenarios) < convs_per_persona:
            batch = list(SCENARIOS)
            random.shuffle(batch)
            persona_scenarios.extend(batch)
        persona_scenarios = persona_scenarios[:convs_per_persona]

        for scenario in persona_scenarios:
            conv_id = f"conv_{conv_idx:04d}"
            tasks_list.append((persona_id, persona, scenario, conv_id))
            conv_idx += 1

    total = len(tasks_list)
    print(f"\nGenerating {total} conversations with {workers} parallel workers")
    print(f"  Personas: {len(persona_ids)} ({persona_ids[0]} - {persona_ids[-1]})")
    print(f"  Conv IDs: conv_{start_conv:04d} - conv_{conv_idx - 1:04d}")
    print(f"  Output: {output_file}")
    print()

    # Skip already-generated conversations
    existing = set()
    for f in Path(output_dir).glob("conv_*.json"):
        existing.add(f.stem)
    if existing:
        print(f"  Found {len(existing)} existing conversations, skipping those")
        tasks_list = [(pid, p, s, cid) for pid, p, s, cid in tasks_list if cid not in existing]
        print(f"  {len(tasks_list)} remaining to generate")

    semaphore = asyncio.Semaphore(workers)

    completed = 0
    failed = 0
    errors = []
    start_time = time.time()

    async def run_and_track(pid, persona, scenario, conv_id):
        nonlocal completed, failed
        result = await generate_one(pid, persona, scenario, conv_id, semaphore, output_dir)
        cid, success, err = result
        if success:
            completed += 1
        else:
            failed += 1
            errors.append((cid, err))

        done = completed + failed
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (len(tasks_list) - done) / rate if rate > 0 else 0
        status = "OK" if success else "FAIL - " + err[:60]
        msg = (
            f"  [{done}/{len(tasks_list)}] {cid}: {status}  "
            f"({rate:.1f}/min, ETA {remaining:.0f}s)"
        )
        print(msg.encode("ascii", "replace").decode())
        return result

    # Launch all tasks
    coros = [
        run_and_track(pid, persona, scenario, conv_id)
        for pid, persona, scenario, conv_id in tasks_list
    ]
    await asyncio.gather(*coros)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Generation complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Success: {completed + len(existing)}")
    print(f"  Failed: {failed}")

    if errors:
        print(f"\nFailed conversations:")
        for cid, err in errors[:20]:
            print(f"  {cid}: {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    # Merge all temp files into output
    print(f"\nMerging into {output_file}...")
    all_convs = []
    for f in sorted(Path(output_dir).glob("conv_*.json")):
        with open(f) as fh:
            conv = json.load(fh)
            all_convs.append(conv)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_convs, f, indent=2)

    print(f"  Wrote {len(all_convs)} conversations to {output_file}")

    # Validation summary
    personas_found = set(c["persona_id"] for c in all_convs)
    print(f"\n  Personas: {len(personas_found)}")
    turn_counts = [len(c["turns"]) for c in all_convs]
    if turn_counts:
        print(f"  Turns: min={min(turn_counts)}, max={max(turn_counts)}, avg={sum(turn_counts)/len(turn_counts):.1f}")

    return len(all_convs)


def main():
    parser = argparse.ArgumentParser(description="Generate retail coffee conversations via ollama")
    parser.add_argument("--workers", type=int, default=25, help="Number of parallel ollama subagents (default: 25)")
    parser.add_argument("--output", type=str, default="retail_training.json", help="Output JSON file (default: retail_training.json)")
    parser.add_argument("--temp-dir", type=str, default=None, help="Temp directory for individual conversation files")
    parser.add_argument("--start-conv", type=int, default=201, help="Starting conversation index (default: 201)")
    parser.add_argument("--convs-per-persona", type=int, default=20, help="Conversations per persona (default: 20)")
    args = parser.parse_args()

    # Create temp directory
    if args.temp_dir:
        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix="retail_conv_")

    print("=" * 60)
    print("Retail Conversation Generator")
    print("=" * 60)
    print(f"  Workers: {args.workers}")
    print(f"  Output: {args.output}")
    print(f"  Temp dir: {temp_dir}")
    print(f"  Personas: {len(PERSONAS)}")
    print(f"  Convs/persona: {args.convs_per_persona}")
    print(f"  Total target: {len(PERSONAS) * args.convs_per_persona}")

    count = asyncio.run(generate_all(
        workers=args.workers,
        output_file=args.output,
        output_dir=temp_dir,
        start_conv=args.start_conv,
        convs_per_persona=args.convs_per_persona,
    ))

    expected = len(PERSONAS) * args.convs_per_persona
    if count >= expected:
        print(f"\nAll {count} conversations generated successfully!")
    else:
        print(f"\nGenerated {count}/{expected} conversations. {expected - count} missing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
