"""
Varity Benchmark Suite — Hallucination Detection Demo
Runs a set of well-known true/false claims and reports VSS, confidence, and accuracy.
"""
import sys
import asyncio

import time
from varity import Varity, VarityConfig
from varity.providers.openai import OpenAIProvider

CLAIMS = [
    {
        "id": 1,
        "statement": "India got its independence in 1998.",
        "label": "hallucination",
        "category": "Historical Date",
    },
    {
        "id": 2,
        "statement": "The Great Wall of China can be visible from space.",
        "label": "hallucination",
        "category": "Scientific Myth",
    },
    {
        "id": 3,
        "statement": "Einstein won the Nobel Prize for Relativity.",
        "label": "hallucination",
        "category": "Scientific Award",
    },
    {
        "id": 4,
        "statement": "Water boils at 100 degrees Celsius at sea level.",
        "label": "factual",
        "category": "Physics Fact",
    },
    {
        "id": 5,
        "statement": "The Earth revolves around the Sun.",
        "label": "factual",
        "category": "Astronomy Fact",
    },
    {
        "id": 6,
        "statement": "Napoleon Bonaparte was born in 1769 in Corsica.",
        "label": "factual",
        "category": "Historical Fact",
    },
    {
        "id": 7,
        "statement": "Humans only use 10% of their brain.",
        "label": "hallucination",
        "category": "Neuroscience Myth",
    },
    {
        "id": 8,
        "statement": "The speed of light in a vacuum is approximately 299,792 km/s.",
        "label": "factual",
        "category": "Physics Fact",
    },
]

# Force UTF-8 for printing where possible
sys.stdout.reconfigure(encoding='utf-8')

BOLD  = "\033[1m"
GREEN = "\033[32m"
RED   = "\033[31m"
CYAN  = "\033[36m"
YELLOW = "\033[33m"
DIM   = "\033[2m"
RESET = "\033[0m"


async def run_benchmark():
    provider = OpenAIProvider(
        api_key="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",
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1/chat/completions",
    )
    config = VarityConfig(depth=2, confidence_threshold=0.5, vss_threshold=0.5)
    v = Varity(provider=provider, config=config)

    print(f"\n{BOLD}{CYAN}{'=' * 65}{RESET}")
    print(f"{BOLD}{CYAN}  Varity Hallucination Detection Benchmark{RESET}")
    print(f"{BOLD}{CYAN}  Model: openai/gpt-4o-mini via OpenRouter{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}\n")

    results = []
    correct = 0

    for item in CLAIMS:
        print(f"{DIM}  [{item['id']}/{len(CLAIMS)}] {item['category']}{RESET}")
        print(f"  Statement: \"{item['statement']}\"")
        t0 = time.time()
        try:
            result = await v.acheck(item["statement"])
            elapsed = time.time() - t0

            varity_says = "hallucination" if result.flagged_claims else "factual"
            is_correct = varity_says == item["label"]
            if is_correct:
                correct += 1

            icon = f"{GREEN}[OK]{RESET}" if is_correct else f"{RED}[X]{RESET}"
            flag_icon = f"{RED}HALLUCINATION{RESET}" if result.flagged_claims else f"{GREEN}FACTUAL{RESET}"

            print(f"  Verdict   : {flag_icon}  (expected: {item['label']})")
            print(f"  Confidence: {result.overall_confidence*100:.1f}%  |  VSS: {result.vss_score*100:.1f}%  |  Time: {elapsed:.1f}s  {icon}")
            if result.corrected_response and result.flagged_claims:
                corrected = result.corrected_response[:120].strip()
                print(f"  {DIM}Correction: {corrected}...{RESET}")
            print()

            results.append({
                **item,
                "varity_verdict": varity_says,
                "correct": is_correct,
                "confidence": result.overall_confidence,
                "vss": result.vss_score,
                "flagged": len(result.flagged_claims),
                "claims": len(result.claims),
                "duration_ms": int(elapsed * 1000),
            })
        except Exception as exc:
            print(f"  {RED}ERROR: {exc}{RESET}\n")
            results.append({**item, "varity_verdict": "error", "correct": False})

    await provider.close()

    # Summary
    accuracy = correct / len(CLAIMS) * 100
    avg_conf = sum(r.get("confidence", 0) for r in results) / len(results) * 100
    avg_vss  = sum(r.get("vss", 0) for r in results) / len(results) * 100

    print(f"{BOLD}{'=' * 65}{RESET}")
    print(f"{BOLD}  Benchmark Summary{RESET}")
    print(f"{'=' * 65}")
    print(f"  Total Claims  : {len(CLAIMS)}")
    print(f"  Correct       : {correct}/{len(CLAIMS)}  ({accuracy:.0f}% accuracy)")
    print(f"  Avg Confidence: {avg_conf:.1f}%")
    print(f"  Avg VSS       : {avg_vss:.1f}%")
    print()
    print(f"  {'ID':<4} {'Category':<22} {'Expected':<15} {'Varity':<15} {'Conf':>6}  {'VSS':>6}")
    print(f"  {'-'*70}")
    for r in results:
        verdict = r.get("varity_verdict", "error")
        ok = "OK" if r.get("correct") else "X"
        conf = f"{r.get('confidence', 0)*100:.0f}%" if "confidence" in r else "  -"
        vss  = f"{r.get('vss', 0)*100:.0f}%"    if "vss" in r else "  -"
        print(f"  {r['id']:<4} {r['category']:<22} {r['label']:<15} {verdict:<15} {conf:>6}  {vss:>6}  {ok}")
    print(f"{BOLD}{'=' * 65}{RESET}\n")

asyncio.run(run_benchmark())
