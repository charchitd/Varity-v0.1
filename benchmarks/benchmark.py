"""Varity benchmark runner.

Measures end-to-end latency, token usage, and hallucination detection rate
across a set of test responses with known ground truth.

Usage:
    python benchmarks/benchmark.py --provider gemini --key AIzaSy... [--depth 1]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Benchmark dataset — responses with known ground truth labels
# ---------------------------------------------------------------------------

BENCHMARK_CASES: list[dict[str, Any]] = [
    {
        "id": "b01",
        "response": "Python was created by Guido van Rossum and first released in 1991.",
        "expected_flagged": 0,
        "notes": "Fully correct — no claims should be flagged.",
    },
    {
        "id": "b02",
        "response": (
            "The Great Wall of China is clearly visible from space with the naked eye. "
            "It was built entirely during the Ming Dynasty."
        ),
        "expected_flagged": 1,
        "notes": "Space visibility myth should be flagged; Ming Dynasty partial truth.",
    },
    {
        "id": "b03",
        "response": (
            "Water boils at 100°C at standard atmospheric pressure. "
            "The speed of light is approximately 299,792 km/s. "
            "Einstein published his general theory of relativity in 1915."
        ),
        "expected_flagged": 0,
        "notes": "All three claims are correct.",
    },
    {
        "id": "b04",
        "response": (
            "Shakespeare was born in 1564 in Stratford-upon-Avon. "
            "He wrote exactly 40 plays and 200 sonnets."
        ),
        "expected_flagged": 1,
        "notes": "Birth date/place correct; play/sonnet counts are wrong (37 plays, 154 sonnets).",
    },
    {
        "id": "b05",
        "response": (
            "The Amazon River is the longest river in the world. "
            "The Nile is the second longest."
        ),
        "expected_flagged": 1,
        "notes": "The Nile is generally considered longest; Amazon is debated.",
    },
    {
        "id": "b06",
        "response": "Mount Everest is the tallest mountain on Earth at 8,849 metres.",
        "expected_flagged": 0,
        "notes": "Correct height (2020 survey).",
    },
    {
        "id": "b07",
        "response": (
            "Penicillin was discovered by Alexander Fleming in 1928. "
            "It was the first antibiotic ever used in humans in 1942."
        ),
        "expected_flagged": 0,
        "notes": "Both claims broadly correct.",
    },
    {
        "id": "b08",
        "response": (
            "The human body has 206 bones in adulthood. "
            "Babies are born with approximately 270 bones."
        ),
        "expected_flagged": 0,
        "notes": "Both anatomical facts are correct.",
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_benchmark(
    provider_name: str,
    api_key: str,
    model: str | None,
    depth: int,
    output_path: str | None,
) -> None:
    from varity import Varity, VarityConfig
    from varity.providers import get_provider

    kwargs: dict[str, object] = {}
    if model:
        kwargs["model"] = model

    provider = get_provider(provider_name, api_key=api_key, **kwargs)
    config = VarityConfig(depth=depth, strategy="full")
    varity = Varity(provider=provider, config=config)

    results = []
    total_flagged_correct = 0
    total_cases = len(BENCHMARK_CASES)
    total_latency_ms = 0
    total_tokens = 0

    print(f"\nVarity Benchmark  |  provider={provider_name}  depth={depth}")
    print("=" * 70)

    try:
        for case in BENCHMARK_CASES:
            t0 = time.monotonic()
            try:
                result = await varity.acheck(case["response"])
                elapsed = int((time.monotonic() - t0) * 1000)
                actual_flagged = len(result.flagged_claims)
                expected = case["expected_flagged"]
                detection_ok = (
                    (expected == 0 and actual_flagged == 0)
                    or (expected > 0 and actual_flagged > 0)
                )
                if detection_ok:
                    total_flagged_correct += 1

                tok = result.token_usage.get("total_tokens", 0)
                total_latency_ms += elapsed
                total_tokens += tok

                status = "PASS" if detection_ok else "FAIL"
                print(
                    f"  [{case['id']}] {status}  "
                    f"conf={result.overall_confidence:.2f}  "
                    f"vss={result.vss_score:.2f}  "
                    f"flagged={actual_flagged}(exp={expected})  "
                    f"{elapsed}ms  ~{tok}tok"
                )
                results.append({
                    "id": case["id"],
                    "status": status,
                    "overall_confidence": result.overall_confidence,
                    "vss_score": result.vss_score,
                    "flagged_count": actual_flagged,
                    "expected_flagged": expected,
                    "total_tokens": tok,
                    "duration_ms": elapsed,
                    "notes": case["notes"],
                })
            except Exception as exc:
                elapsed = int((time.monotonic() - t0) * 1000)
                print(f"  [{case['id']}] ERROR  {exc}  {elapsed}ms")
                results.append({"id": case["id"], "status": "ERROR", "error": str(exc)})
    finally:
        await provider.close()

    # Summary
    accuracy = total_flagged_correct / total_cases * 100
    avg_latency = total_latency_ms / total_cases if total_cases else 0
    avg_tokens = total_tokens / total_cases if total_cases else 0

    print("=" * 70)
    print(f"  Detection accuracy : {accuracy:.1f}%  ({total_flagged_correct}/{total_cases})")
    print(f"  Avg latency        : {avg_latency:.0f} ms")
    print(f"  Avg tokens (est.)  : {avg_tokens:.0f}")
    print()

    if output_path:
        summary = {
            "provider": provider_name,
            "depth": depth,
            "accuracy_pct": round(accuracy, 1),
            "avg_latency_ms": round(avg_latency),
            "avg_tokens": round(avg_tokens),
            "cases": results,
        }
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Results written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Varity benchmark runner")
    parser.add_argument("--provider", default="anthropic", help="Provider name")
    parser.add_argument("--key", required=True, help="API key")
    parser.add_argument("--model", default=None, help="Override default model")
    parser.add_argument("--depth", type=int, default=1, help="Verification depth (default: 1)")
    parser.add_argument("--output", default=None, help="Write JSON results to this file")
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            provider_name=args.provider,
            api_key=args.key,
            model=args.model,
            depth=args.depth,
            output_path=args.output,
        )
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
