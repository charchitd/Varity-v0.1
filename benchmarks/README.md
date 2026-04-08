# Varity Benchmarks

## Overview

The benchmark measures three things:
1. **Detection accuracy** — does Varity flag responses with known errors?
2. **Latency** — end-to-end wall-clock time per check
3. **Token usage** — estimated tokens consumed per check

## Dataset

`benchmark.py` contains 8 hand-labeled test cases with known ground truth
(`expected_flagged: 0 or 1`).  A case passes if:

- `expected_flagged == 0` and Varity flags nothing
- `expected_flagged >= 1` and Varity flags at least one claim

This is a directional accuracy metric — it doesn't check *which* claim was
flagged, only that hallucination was detected.

## Running

```bash
# Gemini (free tier — recommended for benchmarking)
python benchmarks/benchmark.py \
  --provider gemini \
  --key AIzaSy... \
  --depth 1 \
  --output benchmarks/results_gemini.json

# Anthropic
python benchmarks/benchmark.py \
  --provider anthropic \
  --key sk-ant-... \
  --depth 2 \
  --output benchmarks/results_anthropic.json

# OpenAI
python benchmarks/benchmark.py \
  --provider openai \
  --key sk-... \
  --depth 1
```

## Interpreting Results

| Metric | What to look for |
|--------|-----------------|
| Detection accuracy | ≥ 75% is good for depth=1; ≥ 85% for depth=2 |
| Avg latency | < 10s for depth=1 (quick), < 30s for depth=2 (full) |
| Avg tokens | ~500–2000 per check depending on claim count and depth |

## Expected Baseline (depth=1, gemini-2.0-flash)

These are approximate targets — actual results vary by model and date:

```
Detection accuracy : ~75%
Avg latency        : ~6000 ms
Avg tokens (est.)  : ~800
```

## Adding Test Cases

Add entries to `BENCHMARK_CASES` in `benchmark.py`:

```python
{
    "id": "b09",
    "response": "Your response text here.",
    "expected_flagged": 1,   # 0 = clean, 1+ = has errors
    "notes": "What the error is and why it should be flagged.",
},
```

Keep `notes` precise — they help diagnose failures when Varity misses an error.
