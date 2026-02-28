#!/usr/bin/env python3
"""
Generate general retail conversations using ollama subagents.

Fills remaining conv IDs (gaps in conv_0201-conv_1200) with diverse
retail scenarios: clothing, electronics, grocery, hardware, pharmacy,
bookstore, furniture, pet store, sporting goods, beauty, etc.

Usage:
    python generate_retail_general.py --workers 25 --temp-dir /tmp/retail_conv_7_oye6hv
"""

import argparse
import asyncio
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path


# ============================================================
# Retail store types + scenario pools
# ============================================================

STORE_TYPES = [
    {
        "type": "clothing store",
        "details": "Mid-range clothing retailer with men's, women's, and kids' sections. Carries jeans, tops, dresses, outerwear, and accessories. Has fitting rooms and a return counter.",
        "scenarios": [
            "returning a shirt that doesn't fit after washing",
            "looking for a specific dress for a wedding",
            "exchange request, wrong size ordered online",
            "browsing clearance rack, asking about additional discounts",
            "buying a gift for someone, unsure of size",
            "loyalty program signup at checkout",
            "complaint about a zipper that broke after one wear",
            "asking if a sold-out item can be ordered from another location",
            "trying on jeans, needs different sizes brought to fitting room",
            "price match request against an online competitor",
        ],
    },
    {
        "type": "electronics store",
        "details": "Consumer electronics retailer selling phones, laptops, TVs, headphones, cables, smart home devices. Has a repair/support counter and warranty plans.",
        "scenarios": [
            "buying a laptop for college, comparing models",
            "returning a defective phone charger",
            "asking about extended warranty on a TV",
            "trading in an old phone for credit",
            "confused about which HDMI cable to buy",
            "picking up an online order, item is missing from the box",
            "comparing wireless earbuds, budget under $100",
            "smart home setup questions, compatibility concerns",
            "screen protector installation request",
            "complaint about a repair that took too long",
        ],
    },
    {
        "type": "grocery store",
        "details": "Full-service grocery with produce, deli, bakery, dairy, frozen, and household aisles. Has self-checkout and a customer service desk.",
        "scenarios": [
            "can't find a specific product, asking for help locating it",
            "complaint about expired product on the shelf",
            "asking the deli counter for a custom sandwich order",
            "price discrepancy at checkout, item rang up wrong",
            "asking about store brand vs name brand differences",
            "returning spoiled produce purchased yesterday",
            "large catering order for a party, needs advance notice",
            "using coupons and the total seems off",
            "asking about gluten-free or allergen-free options",
            "self-checkout machine is malfunctioning",
        ],
    },
    {
        "type": "hardware store",
        "details": "Home improvement store with lumber, plumbing, electrical, paint, tools, and garden sections. Staff can cut wood and mix paint.",
        "scenarios": [
            "DIY project help, doesn't know which screws to buy",
            "paint color matching from a sample chip",
            "returning a power tool that stopped working",
            "asking for wood to be cut to specific dimensions",
            "plumbing emergency, needs parts to fix a leak tonight",
            "comparing drill models for home use",
            "asking about delivery for large lumber order",
            "garden section, picking plants for a shady yard",
            "electrical wiring question, safety concerns",
            "bulk discount request for a contractor",
        ],
    },
    {
        "type": "pharmacy/drugstore",
        "details": "Pharmacy with prescription counter, OTC medications, vitamins, personal care, cosmetics, and snacks. Has a photo printing kiosk.",
        "scenarios": [
            "picking up a prescription, insurance issue at the counter",
            "asking pharmacist about drug interactions",
            "looking for a specific OTC cold medicine",
            "photo printing kiosk isn't working, needs help",
            "returning a cosmetic product that caused a reaction",
            "asking about flu shot availability and walk-in timing",
            "price question on generic vs brand-name medication",
            "buying first aid supplies, unsure what to get for a burn",
            "loyalty card points dispute",
            "asking about COVID test kit availability",
        ],
    },
    {
        "type": "bookstore",
        "details": "Independent bookstore with fiction, nonfiction, children's, and used book sections. Has a reading nook and hosts author events.",
        "scenarios": [
            "looking for a book recommendation for a 10-year-old",
            "special order request for an out-of-print title",
            "asking about upcoming author signing events",
            "returning a book received as a duplicate gift",
            "browsing used section, asking about trade-in policy",
            "buying textbooks, asking about student discount",
            "looking for a specific genre, can't remember the author",
            "gift wrapping request for a birthday present",
            "asking if they carry audiobooks or e-readers",
            "complaint about an online order that never arrived",
        ],
    },
    {
        "type": "furniture store",
        "details": "Home furniture retailer with showroom floor displaying sofas, beds, dining sets, desks, and storage. Offers delivery and assembly services.",
        "scenarios": [
            "couch shopping, testing cushion firmness",
            "delivery scheduling for a large sectional",
            "returning a desk chair that's uncomfortable",
            "asking about financing options for a bedroom set",
            "mattress comparison, overwhelmed by choices",
            "damage claim on a table delivered with a scratch",
            "measuring a bookshelf to see if it fits a specific wall",
            "assembly service pricing and timeline questions",
            "custom fabric selection for a sofa",
            "clearance floor model, negotiating price",
        ],
    },
    {
        "type": "pet store",
        "details": "Pet supply store with food, toys, accessories, grooming services, and live animals (fish, small pets). Has a grooming salon.",
        "scenarios": [
            "first-time puppy owner, needs starter supplies",
            "asking about grain-free vs regular dog food",
            "booking a grooming appointment for a nervous dog",
            "returning a harness that doesn't fit",
            "looking at fish tanks, asking about setup and maintenance",
            "complaint about a toy that fell apart in one day",
            "asking about flea/tick treatment options",
            "buying a birthday cake for a dog",
            "cat owner looking for scratching post recommendations",
            "asking about pet adoption events",
        ],
    },
    {
        "type": "sporting goods store",
        "details": "Sporting goods retailer with equipment for running, cycling, camping, fitness, and team sports. Has a shoe fitting area and bike repair.",
        "scenarios": [
            "getting fitted for running shoes, pronation analysis",
            "returning hiking boots that gave blisters",
            "buying camping gear for a first-time trip",
            "bike tune-up drop-off and timeline",
            "comparing yoga mats, thickness and material",
            "asking about team jersey customization",
            "tennis racket restringing service",
            "looking for cold weather running gear",
            "kid needs cleats for soccer season, growing fast",
            "warranty claim on a treadmill motor",
        ],
    },
    {
        "type": "beauty supply store",
        "details": "Beauty and cosmetics store with skincare, makeup, haircare, fragrances, and nail products. Has testers and a consultation area.",
        "scenarios": [
            "shade matching for foundation, tried three already",
            "returning an opened skincare product that irritated skin",
            "looking for a fragrance gift, describing the recipient's taste",
            "asking about cruelty-free and vegan product lines",
            "hair dye consultation, going from dark to light",
            "loyalty program redemption, confusion about points",
            "product recommendation for acne-prone sensitive skin",
            "curling iron comparison, ceramic vs titanium",
            "buying professional-grade shampoo for salon quality at home",
            "complaint about a subscription box missing items",
        ],
    },
    {
        "type": "auto parts store",
        "details": "Automotive parts and accessories retailer. Sells oil, filters, batteries, brake pads, wiper blades, and car care products. Offers free battery testing and wiper installation.",
        "scenarios": [
            "needs a battery for a 2018 Honda Civic, not sure which one",
            "check engine light on, asking about OBD2 scanner",
            "returning wrong oil filter, needs exchange",
            "wiper blade installation help in the parking lot",
            "asking about brake pad brands, ceramic vs semi-metallic",
            "buying supplies for an oil change, first time doing it",
            "headlight bulb replacement, which one fits their car",
            "coolant flush supplies, needs advice on which coolant",
            "car won't start, brought battery in for testing",
            "looking for touch-up paint to match car color",
        ],
    },
    {
        "type": "toy store",
        "details": "Specialty toy store with board games, building sets, dolls, action figures, puzzles, outdoor toys, and educational toys. Gift wrapping available.",
        "scenarios": [
            "birthday gift for a 7-year-old, not sure what they like",
            "looking for a specific LEGO set that's sold out everywhere",
            "returning a toy with missing pieces",
            "educational toy recommendations for a toddler",
            "comparing board games for family game night",
            "asking about age-appropriate toys, safety concerns",
            "gift registry or wish list for a child's birthday party",
            "buying in bulk for a school classroom donation",
            "remote control car not working out of the box",
            "holiday shopping, needs help picking for multiple kids",
        ],
    },
]

# ============================================================
# Customer persona archetypes (general retail)
# ============================================================

CUSTOMER_PERSONAS = [
    {"name": "Frustrated Returner", "mood": "angry", "traits": "Had a bad experience with a product. Wants a refund, not a replacement. Impatient with process. Brings receipt but expects hassle."},
    {"name": "Clueless Gift Buyer", "mood": "nervous", "traits": "Buying a gift for someone they don't know well. Asks lots of questions. Worried about picking wrong thing. Needs reassurance."},
    {"name": "Bargain Hunter", "mood": "friendly", "traits": "Looking for deals. Asks about clearance, coupons, price matching. Friendly but persistent about getting best price."},
    {"name": "Overwhelmed Parent", "mood": "angry", "traits": "Shopping with kids who are acting up. Distracted, stressed. Needs to get in and out fast. Short answers."},
    {"name": "Chatty Regular", "mood": "friendly", "traits": "Comes in often. Knows staff by name. Tells stories. Asks about their day. Takes their time browsing."},
    {"name": "Online-Returns-In-Store", "mood": "nervous", "traits": "Ordered online, returning in store. Unsure about process. Has confirmation email ready. Worried about getting refund."},
    {"name": "Expert Customer", "mood": "friendly", "traits": "Knows more about the product than the staff. Asks very technical questions. Patient but expects knowledgeable answers."},
    {"name": "Complaint Escalator", "mood": "angry", "traits": "Previous visit went badly. Back to resolve it. Wants manager. Documents everything. Threatens reviews."},
    {"name": "Indecisive Browser", "mood": "indecisive", "traits": "Can't decide between options. Asks staff to choose for them. Goes back and forth. Eventually picks after much deliberation."},
    {"name": "Rush-Hour Shopper", "mood": "angry", "traits": "On lunch break, 15 minutes to shop. Speaks fast. Annoyed by lines. Knows what they want, just needs to find it."},
    {"name": "First-Time Buyer", "mood": "nervous", "traits": "Never bought this type of product before. Needs everything explained. Apologizes for asking basic questions. Grateful for help."},
    {"name": "Senior Citizen", "mood": "friendly", "traits": "Patient, polite, takes time. May need help reading labels or using technology. Appreciates personal attention. Shares life stories."},
    {"name": "Bulk Buyer", "mood": "friendly", "traits": "Buying large quantities for business or event. Asks about bulk discounts. Organized with a list. Efficient."},
    {"name": "Warranty Warrior", "mood": "angry", "traits": "Product broke just after warranty period. Argues it should still be covered. Has documentation. Persistent about getting repair or replacement."},
    {"name": "Comparison Shopper", "mood": "indecisive", "traits": "Has researched online, now wants to see products in person. Asks detailed comparison questions. Pulls up phone to check reviews."},
    {"name": "Non-English Speaker", "mood": "nervous", "traits": "Limited English. Uses simple words and gestures. Very grateful when understood. Patient. May use translation app."},
    {"name": "Loyalty Program Obsessed", "mood": "friendly", "traits": "Maximizes every point and reward. Asks about bonus point events. Knows the program better than staff. Friendly but strategic."},
    {"name": "Suspicious Customer", "mood": "angry", "traits": "Thinks they're being overcharged or scammed. Checks every price. Questions fees. Wants itemized receipts. Distrustful."},
    {"name": "Apologetic Complainer", "mood": "nervous", "traits": "Has a legitimate complaint but feels bad about raising it. Starts with 'I hate to bother you but...' Accepts whatever resolution is offered."},
    {"name": "Phone Distracted", "mood": "indecisive", "traits": "On their phone during the entire transaction. Half-listening. Asks staff to repeat things. Distracted ordering/checking out."},
]

# ============================================================
# Prompt builder + JSON extraction + validation
# (Same infra as generate_retail_conversations.py)
# ============================================================

CATEGORY_TO_MOOD = {
    "angry": "angry",
    "nervous": "nervous",
    "friendly": "friendly",
    "indecisive": "indecisive",
}


def build_prompt(persona: dict, store: dict, scenario: str, conv_id: str, persona_id: str) -> str:
    mood = persona["mood"]
    store_type = store["type"]
    traits_short = persona["traits"].split(".")[0]

    system_prompt = (
        f"<system>You are {persona_id}, {persona['name']}, a customer in a {store_type}. "
        f"The user is the store employee. Agent = customer, user = employee. "
        f"Stay in the {store_type} context only. "
        f"{traits_short}. Scenario: {scenario}. "
        f"Keep strict alternation of turns; the agent speaks first.</system>"
    )

    return f"""You are generating a realistic retail store conversation for a training dataset.

STORE TYPE: {store_type}
STORE DETAILS: {store["details"]}
CUSTOMER PERSONA: {persona["name"]}
CUSTOMER MOOD: {mood}
CUSTOMER TRAITS: {persona["traits"]}
SCENARIO: {scenario}

Generate a conversation between a customer (role: "agent") and a store employee (role: "user").
IMPORTANT: agent = CUSTOMER, user = STORE EMPLOYEE.

STRICT RULES:
1. The FIRST turn MUST be from "agent" (the CUSTOMER initiating)
2. Turns MUST strictly alternate: agent, user, agent, user, agent, user...
3. Generate between 14 and 26 turns total (MUST be an even number so it ends on user)
4. Each turn's text should be 1-3 sentences, natural spoken dialogue
5. The conversation should feel real and complete (greeting -> issue/browse -> resolution/checkout -> goodbye)
6. Reflect the persona's traits throughout (e.g., angry = short/clipped, nervous = hesitant/apologetic)
7. Include realistic retail details (product names, prices, policies, store sections)
8. Make it specific to a {store_type}, not generic

OUTPUT FORMAT - Return ONLY valid JSON, no other text:
{{
  "conversation_id": "{conv_id}",
  "persona_id": "{persona_id}",
  "persona_name": "{persona['name']}",
  "scenario": "{scenario}",
  "mood": "{mood}",
  "system_prompt": "{system_prompt}",
  "turns": [
    {{"role": "agent", "text": "..."}},
    {{"role": "user", "text": "..."}},
    ...
  ]
}}

IMPORTANT: Return ONLY the JSON object. No markdown, no explanation, no code fences."""


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
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


def validate_conversation(conv, conv_id):
    if not isinstance(conv, dict):
        return False, "Not a dict"
    for field in ["conversation_id", "persona_id", "persona_name", "scenario", "mood", "system_prompt", "turns"]:
        if field not in conv:
            return False, f"Missing field: {field}"
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
            return False, f"Non-alternating at turn {i}"
    for i, turn in enumerate(turns):
        if turn.get("role") not in ("agent", "user"):
            return False, f"Invalid role at turn {i}"
        if not turn.get("text", "").strip():
            return False, f"Empty text at turn {i}"
    conv["conversation_id"] = conv_id
    return True, ""


async def generate_one(conv_id, persona, persona_id, store, scenario, semaphore, output_dir, max_retries=3):
    prompt = build_prompt(persona, store, scenario, conv_id, persona_id)
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
                    timeout=180,
                )
                output = stdout.decode("utf-8", errors="replace")
                if proc.returncode != 0:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                        continue
                    return conv_id, False, f"exit code {proc.returncode}"
                conv = extract_json(output)
                if conv is None:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    return conv_id, False, "JSON parse failed"
                is_valid, err = validate_conversation(conv, conv_id)
                if not is_valid:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue
                    return conv_id, False, f"Validation: {err}"
                out_path = os.path.join(output_dir, f"{conv_id}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(conv, f, indent=2)
                return conv_id, True, ""
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    continue
                return conv_id, False, "Timeout"
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                return conv_id, False, str(e)
    return conv_id, False, "Max retries"


async def generate_all(workers, output_dir, output_file):
    # Find which conv IDs are missing
    existing = set()
    for f in os.listdir(output_dir):
        if f.startswith("conv_") and f.endswith(".json"):
            existing.add(f.replace(".json", ""))

    all_ids = [f"conv_{i:04d}" for i in range(201, 1201)]
    missing_ids = [cid for cid in all_ids if cid not in existing]

    if not missing_ids:
        print("All 1000 conversations already exist!")
        return 0

    print(f"Found {len(existing)} existing, {len(missing_ids)} to generate")

    # Build tasks: assign each missing conv_id a random store + scenario + persona
    random.seed(99)
    tasks_list = []
    for cid in missing_ids:
        store = random.choice(STORE_TYPES)
        scenario = random.choice(store["scenarios"])
        persona = random.choice(CUSTOMER_PERSONAS)
        # Generate a persona_id based on conv number
        conv_num = int(cid.replace("conv_", ""))
        pid = f"persona_{(conv_num % 50) + 21:03d}"
        tasks_list.append((cid, persona, pid, store, scenario))

    print(f"Generating {len(tasks_list)} general retail conversations with {workers} workers")
    print(f"Store types: {len(STORE_TYPES)} | Personas: {len(CUSTOMER_PERSONAS)}")
    print()

    semaphore = asyncio.Semaphore(workers)
    completed = 0
    failed = 0
    errors = []
    start_time = time.time()

    async def run_and_track(cid, persona, pid, store, scenario):
        nonlocal completed, failed
        result = await generate_one(cid, persona, pid, store, scenario, semaphore, output_dir)
        _, success, err = result
        if success:
            completed += 1
        else:
            failed += 1
            errors.append((cid, err))
        done = completed + failed
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (len(tasks_list) - done) / rate if rate > 0 else 0
        status = "OK" if success else f"FAIL - {err[:50]}"
        msg = f"  [{done}/{len(tasks_list)}] {cid} ({store['type']}): {status}  ({rate:.1f}/min, ETA {remaining:.0f}s)"
        print(msg.encode("ascii", "replace").decode())
        return result

    coros = [run_and_track(*t) for t in tasks_list]
    await asyncio.gather(*coros)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.0f}s. Success: {completed}, Failed: {failed}")

    if errors:
        print(f"\nFailed:")
        for cid, err in errors[:20]:
            print(f"  {cid}: {err}")

    # Merge ALL files (coffee + retail) into output
    print(f"\nMerging all conversations into {output_file}...")
    all_convs = []
    for f in sorted(os.listdir(output_dir)):
        if f.startswith("conv_") and f.endswith(".json"):
            with open(os.path.join(output_dir, f), encoding="utf-8") as fh:
                all_convs.append(json.load(fh))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_convs, f, indent=2)

    print(f"  Wrote {len(all_convs)} conversations")
    return len(all_convs)


def main():
    parser = argparse.ArgumentParser(description="Generate general retail conversations (fills gaps)")
    parser.add_argument("--workers", type=int, default=25)
    parser.add_argument("--output", type=str, default="retail_training.json")
    parser.add_argument("--temp-dir", type=str, required=True, help="Same temp dir as coffee shop convos")
    args = parser.parse_args()

    print("=" * 60)
    print("General Retail Conversation Generator")
    print("=" * 60)
    print(f"  Workers: {args.workers}")
    print(f"  Temp dir: {args.temp_dir}")
    print(f"  Store types: {len(STORE_TYPES)}")
    print(f"  Personas: {len(CUSTOMER_PERSONAS)}")

    count = asyncio.run(generate_all(
        workers=args.workers,
        output_dir=args.temp_dir,
        output_file=args.output,
    ))

    if count >= 1000:
        print(f"\nAll {count} conversations generated!")
    else:
        print(f"\n{count}/1000 conversations. Run again to fill remaining gaps.")
        sys.exit(1)


if __name__ == "__main__":
    main()
