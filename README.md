<div align="center">
  <h1>Varity v0.1</h1>
  <p><em>Recursive Self-Checking for LLM Hallucination Reduction</em></p>
  
  [![PyPI - Version](https://img.shields.io/pypi/v/varity.svg)](https://pypi.org/project/varity/)
  [![Python Versions](https://img.shields.io/pypi/pyversions/varity.svg)](https://pypi.org/project/varity/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  [![Type Checked](https://img.shields.io/badge/mypy-strict-success.svg)](#)
  [![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](#)
</div>

---
# Overview

**Varity** is a lightweight, zero-dependency Python library designed to natively mitigate Large Language Model (LLM) hallucinations. It operates by systematically decomposing generated responses into atomic claims, recursively verifying each claim across iterative context depths, and computing a **Verdict Stability Score (VSS)**.

Unlike traditional single-pass evaluation frameworks, Varity asserts that hallucinatory or uncertain generations are mathematically unstable. By challenging the LLM to verify its own sub-claims recursively, unstable claims will "flip" their verdicts under analytical pressure. Varity measures these algorithmic flips to calculate rigorous confidence bounds.

### Key Capabilities

- **Recursive Verification (Depth N):** Stresses the model to re-evaluate claims repeatedly to track verdict stability.
- **Verdict Stability Score (VSS):** A mathematical metric bounding the resilience of an LLM generation against self-contradiction.
- **Provider Agnostic (BYOK):** Supports Anthropic, OpenAI, and Google Gemini via raw HTTP integrations, ensuring zero telemetry and guaranteeing Bring-Your-Own-Key data sovereignty.
- **Graceful Degradation:** Safely handles upstream provider rate limits (`HTTP 429`) and degradation faults without interrupting the execution pipeline.
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

```text
 __      __        _ _         
 \ \    / /       (_) |        
  \ \  / /_ _ _ __ _| |_ _   _ 
   \ \/ / _` | '__| | __| | | |
    \  / (_| | |  | | |_| |_| |
     \/ \__,_|_|  |_|\__|\__, |  v0.1
                          __/ |
                         |___/ 
```

```bash
# Single-text evaluation
varity check "Einstein won the Nobel Prize for Relativity." --provider gemini

# Batch processing from JSONL
varity batch input.jsonl output.jsonl --provider openai

# Interactive demo
varity demo
```

### CI/CD Integration

Varity is designed to be easily integrated into CI/CD pipelines to enforce hallucination checks on generated outputs before deployment.

#### Example: GitHub Actions

Create a `.github/workflows/varity-check.yml` file:

```yaml
name: Varity Hallucination Check
on: [push, pull_request]

jobs:
  varity_check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"
      - name: Install dependencies
        run: pip install varity
      - name: Run dynamic cycle checks
        env:
          VARITY_PROVIDER: ${{ secrets.VARITY_PROVIDER }}
          VARITY_API_KEY: ${{ secrets.VARITY_API_KEY }}
        run: |
          # Example: Run 5 evaluation cycles on your test script
          python test101.py --cycles 5
```

## How It Works

## Core Architecture

Varity governs a strict 5-stage deterministic evaluation flow:

```mermaid
graph TD;
    A[Raw Response Payload] --> B[1. Claim Decomposer]
    B --> C[2. Recursive Self-Verifier]
    B --> D[3. Independent Cross-Check]
    C --> E[4. Confidence Aggregator]
    D --> E
    E --> F[5. Correction Generator]
    F --> G[Validated Output Structure]
```

1. **Claim Decomposition**: Segments cohesive text strings into isolated, atomic `Claim` schema nodes.
2. **Recursive Self-Verification**: Executes isolated iterative passes across isolated claims (Depth 0...N), dynamically tracking historical verdict variance.
3. **Cross-Checking**: Instantiates an identical external process verifying the claim devoid of the initial contextual bias.
4. **Confidence Aggregator**: Maps the volume of boolean "flips" and base metric alignments to construct the total `vss_score`.
5. **Correction Generation**: Automatically rebuilds text omitting nodes scored beneath the rigorous confidence threshold.
```

**Verdict Stability Score (VSS):** For each claim, Varity counts how many times the
verdict flipped between `supported` and `contradicted` across recursive depths.
A claim verified as `supported` at every depth receives VSS = 1.0. A claim that
flips on every pass approaches VSS = 0.0. Claims below the configured
`confidence_threshold` are flagged and eligible for automatic correction.

```
## Configuration Reference

`VarityConfig` accepts the following parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `depth` | `int` | `1` | Number of recursive self-verification passes (0 = single pass) |
| `confidence_threshold` | `float` | `0.5` | Claims scoring below this are flagged |
| `vss_threshold` | `float` | `0.5` | Claims with VSS below this are flagged (independently of confidence) |
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

## Commercial Use Cases

Because Varity mathematically filters out unstable generations, it serves as the perfect underlying engine for building high-value, hallucination-free applications:

### 1. "Zero-Hallucination" Legal or Medical Writers
General LLMs are dangerous in high-stakes fields because they can invent case studies or medical facts with complete semantic confidence. By piping raw LLM output through Varity (`depth=3`) and only rendering the `corrected_response` in your UI, you guarantee factuality for professionals who cannot afford hallucinations.

### 2. Academic & SEO Fact-Checking Automation
Content teams and researchers spend countless hours manually fact-checking AI outputs. Varity can be wrapped into a Chrome Extension or text-editor plugin where users highlight generated text and instantly receive a boolean breakdown of Verified vs. Hallucinated claims, drastically reducing manual audit times.

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
git clone https://github.com/charchitd/Varity-v0.1.git
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
