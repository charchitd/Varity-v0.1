# Zero-Cost Setup — Gemini Free Tier

Varity works with Google Gemini's free tier, which allows a meaningful number
of API calls per day at no cost.  This is the recommended way to try Varity
without spending money.

---

## Step 1 — Get a Free Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with a Google account
3. Click **Get API key → Create API key**
4. Copy the key (starts with `AIza...`)

**Free tier limits (as of 2026):**
- `gemini-2.0-flash` (default): 15 requests/minute, 1,500 requests/day
- No credit card required

---

## Step 2 — Install Varity

```bash
pip install varity
```

Or from source:

```bash
git clone https://github.com/yourusername/varity
cd varity
pip install -e ".[dev]"
```

---

## Step 3 — Run Your First Check

### CLI

```bash
varity check \
  --provider gemini \
  --key AIzaSy... \
  --response "The Eiffel Tower was built in 1887 and stands 324 metres tall."
```

### Python

```python
from varity import Varity, VarityConfig
from varity.providers import get_provider

provider = get_provider("gemini", api_key="AIzaSy...")
config = VarityConfig(depth=1, strategy="quick")  # fewer API calls

varity = Varity(provider=provider, config=config)
result = varity.check(
    "The Eiffel Tower was built in 1887 and stands 324 metres tall."
)

print(f"Confidence : {result.overall_confidence:.1%}")
print(f"VSS        : {result.vss_score:.1%}")
print(f"Flagged    : {len(result.flagged_claims)} claim(s)")
```

### Async (FastAPI, Jupyter)

```python
import asyncio
from varity import Varity
from varity.providers import get_provider

async def main() -> None:
    provider = get_provider("gemini", api_key="AIzaSy...")
    varity = Varity(provider=provider)
    result = await varity.acheck("Python was released in 1991.")
    print(result.model_dump_json(indent=2))

asyncio.run(main())
```

---

## Step 4 — Live Demo

```bash
VARITY_PROVIDER=gemini VARITY_API_KEY=AIzaSy... varity demo
```

Or:

```bash
varity demo --provider gemini --key AIzaSy...
```

---

## API Call Budget

A single `varity.check()` call with default settings (`depth=2`, `strategy="full"`)
makes approximately:

```
1 (decompose) + N × 3 (verify at depth 0/1/2) + N (cross-check) + 0-1 (correct)
```

For a response with 5 claims: **1 + 15 + 5 = 21 calls**.

With `strategy="quick"` (depth=1): **1 + 10 + 5 = 16 calls**.

At 15 req/min free-tier limit, a quick check on a 5-claim response completes
in about 2–3 seconds when not rate-limited.

---

## Rate Limit Handling

Varity's base provider automatically retries on HTTP 429 (rate limited)
with exponential back-off (1s → 2s → 4s, 3 attempts).  For batch
processing large files, consider adding `--depth 1` to reduce call volume.

---

## Switching Providers

Changing provider requires only changing the `get_provider` call — the
pipeline, scoring, and output format are identical across providers.

```python
# Gemini (free tier)
provider = get_provider("gemini", api_key="AIzaSy...")

# Anthropic
provider = get_provider("anthropic", api_key="sk-ant-...")

# OpenAI
provider = get_provider("openai", api_key="sk-...")

# Override model
provider = get_provider("openai", api_key="sk-...", model="gpt-4o")
```
