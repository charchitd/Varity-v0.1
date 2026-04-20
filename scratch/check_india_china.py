import asyncio
from varity import Varity, VarityConfig
from varity.providers.openai import OpenAIProvider

CLAIM = "India got its independence in 1998. The Great Wall of China can be visible from space."

async def main():
    provider = OpenAIProvider(
        api_key="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",
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1/chat/completions",
    )
    config = VarityConfig(depth=2)
    v = Varity(provider=provider, config=config)
    result = await v.acheck(CLAIM)
    await provider.close()

    print()
    print("=" * 60)
    print("  Varity Hallucination Check")
    print("=" * 60)
    print(f"  Claims found    : {len(result.claims)}")
    print(f"  Flagged claims  : {len(result.flagged_claims)}")
    print(f"  Confidence      : {result.overall_confidence*100:.1f}%")
    print(f"  VSS             : {result.vss_score*100:.1f}%")
    print(f"  Duration        : {result.duration_ms} ms")
    total = result.token_usage["total_tokens"]
    print(f"  Tokens (est.)   : {total:,}")
    print()
    print("  Claims")
    for i, c in enumerate(result.claims, 1):
        flag = "!!" if c.flagged else "OK"
        print(f"   {i}. {flag} [{c.claim_type:10s}] conf={c.confidence:.2f} vss={c.vss_score:.2f}  {c.text}")
    if result.flagged_claims:
        print()
        print("  Flagged details")
        for c in result.flagged_claims:
            steps = [s for s in result.verification_chain if s.claim_text == c.text]
            flips = sum(1 for j in range(1, len(steps)) if steps[j].verdict != steps[j-1].verdict)
            cross = next((s.verdict for s in steps if s.depth == -1), "n/a")
            print(f"    - {c.text}")
            print(f"      flips={flips}, vss={c.vss_score:.2f}, cross={cross}")
    if result.corrected_response:
        print()
        print("  Corrected Response")
        print(f"  {result.corrected_response}")
    print("=" * 60)

asyncio.run(main())
