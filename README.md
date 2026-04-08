# Varity — Recursive Self-Checking for LLM Hallucination Reduction

[![PyPI - Version](https://img.shields.io/pypi/v/varity.svg)](https://pypi.org/project/varity/)
[![Python Versions](https://img.shields.io/pypi/pyversions/varity.svg)](https://pypi.org/project/varity/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

**Varity** is a lightweight Python library that decomposes LLM-generated text into
atomic claims, recursively verifies each claim using the LLM itself, and returns
per-claim confidence scores with optional auto-corrections.

Its core contribution is the **Verdict Stability Score (VSS)** — a metric derived from
how consistently an LLM upholds or contradicts its own sub-claims across repeated
verification passes. Claims that "flip" verdicts under recursive pressure are
quantitatively flagged as likely hallucinations.

## Why Varity?

| Problem | Varity's Approach |
|---|---|
| Single-pass fact-checking misses nuanced errors | Recursive depth-N verification exposes instability |
| External knowledge bases go stale | Uses the LLM's own parametric knowledge as the oracle |
| Heavy SDK dependencies increase attack surface | Zero vendor SDKs — raw `httpx` only |
| API keys leak through telemetry | Strict BYOK — keys are never logged, cached, or transmitted beyond the provider endpoint |

## Installation

```bash
pip install varity
```

Requires **Python 3.9+**. Core dependencies: `pydantic>=2.0`, `httpx>=0.25`, `tiktoken>=0.5`.

## Supported Providers

| Provider | Default Model | Free Tier |
|---|---|---|
| **Google Gemini** | `gemini-2.0-flash` | Yes |
| **Anthropic Claude** | `claude-sonnet-4-20250514` | No (credits required) |
| **OpenAI** | `gpt-4o-mini` | No (credits required) |

All providers are accessed via direct HTTP — no `google-generativeai`, `anthropic`, or
`openai` SDK packages are required.

## Quick Start

### 1. Set your API key

```bash
# Option A: Environment variable
export VARITY_PROVIDER="gemini"
export VARITY_API_KEY="your-api-key"

# Option B: Create a .env file in your project root
echo 'VARITY_PROVIDER=gemini' > .env
echo 'VARITY_API_KEY=your-api-key' >> .env
```

### 2. Verify a response programmatically

```python
import asyncio
from varity import Varity, VarityConfig
from varity.providers import get_provider

async def main():
    provider = get_provider("gemini", api_key="your-api-key")
    config = VarityConfig(depth=1, confidence_threshold=0.6)
    varity = Varity(provider=provider, config=config)

    result = await varity.acheck(
        "The Eiffel Tower is 10,000 feet tall and was completed in 1887."
    )

    print(f"Confidence : {result.overall_confidence:.2f}")
    print(f"VSS        : {result.vss_score:.2f}")
    print(f"Claims     : {len(result.claims)}")
    print(f"Flagged    : {len(result.flagged_claims)}")

    for claim in result.flagged_claims:
        print(f"  [FLAGGED] {claim.text}")
        print(f"            verdict={claim.verdict}, vss={claim.vss_score:.2f}")

    if result.corrected_response:
        print(f"\nCorrected  : {result.corrected_response}")

    await provider.close()

asyncio.run(main())
```

### 3. Use the CLI

```bash
# Single-text evaluation
varity check "Einstein won the Nobel Prize for Relativity." --provider gemini

# Batch processing from JSONL
varity batch input.jsonl output.jsonl --provider openai

# Interactive demo
varity demo
```

## How It Works

Varity executes a deterministic five-stage pipeline:

```
Input Text
    │
    ▼
┌──────────────────────┐
│  1. Claim Decomposer │  Splits text into atomic Claim objects
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐  ┌───────────┐
│2. Self │  │3. Cross   │  Independent verification paths
│Verifier│  │  Checker  │
└───┬────┘  └─────┬─────┘
    │             │
    └──────┬──────┘
           ▼
┌──────────────────────┐
│ 4. Confidence        │  Bayesian scoring + VSS calculation
│    Aggregator        │  (counts verdict flips across depths)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ 5. Correction        │  Rewrites text removing flagged claims
│    Generator         │  (only when flagged_claims > 0)
└──────────────────────┘
```

**Verdict Stability Score (VSS):** For each claim, Varity counts how many times the
verdict flipped between `supported` and `contradicted` across recursive depths.
A claim verified as `supported` at every depth receives VSS = 1.0. A claim that
flips on every pass approaches VSS = 0.0. Claims below the configured
`confidence_threshold` are flagged and eligible for automatic correction.

## Configuration Reference

`VarityConfig` accepts the following parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `depth` | `int` | `1` | Number of recursive self-verification passes (0 = single pass) |
| `confidence_threshold` | `float` | `0.5` | Claims scoring below this are flagged |
| `strategy` | `str` | `"standard"` | Verification strategy (`"quick"`, `"standard"`, `"thorough"`) |
| `max_claims` | `int` | `20` | Maximum number of claims to extract per response |
| `enable_correction` | `bool` | `True` | Whether to generate corrected text for flagged claims |

## Return Schema

`CheckResult` contains:

| Field | Type | Description |
|---|---|---|
| `original_response` | `str` | The input text that was evaluated |
| `claims` | `list[Claim]` | All extracted atomic claims with individual scores |
| `flagged_claims` | `list[Claim]` | Subset of claims below the confidence threshold |
| `corrected_response` | `str \| None` | Auto-corrected text (if corrections were generated) |
| `overall_confidence` | `float` | Weighted average confidence across all claims |
| `vss_score` | `float` | Average VSS across all claims |
| `verification_chain` | `list[VerificationStep]` | Full audit trail of every verification pass |
| `duration_ms` | `int` | Wall-clock execution time in milliseconds |
| `token_usage` | `dict` | Estimated token consumption breakdown |

## Stress Testing

The included `test101.py` script runs Varity against a known-hallucination payload
over a configurable number of cycles:

```bash
# Run 100 consecutive evaluation cycles
python test101.py --cycles 100

# Or configure via environment
export VARITY_CYCLES=50
python test101.py
```

## Development

```bash
# Clone and install in development mode
git clone https://github.com/yourusername/varity.git
cd varity
pip install -e ".[dev]"

# Run the test suite (76 unit tests + 10 integration tests)
pytest tests/ -v

# Lint and type-check
ruff check .
mypy --strict varity/
```

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
