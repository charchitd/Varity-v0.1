# Varity — Recursive Self-Checking for LLM Hallucination Reduction

## Project Overview
Varity is a Python library (pip install varity) that takes any LLM response, decomposes it into atomic claims, recursively verifies each claim using the LLM itself, and returns confidence scores with auto-corrections. The novel contribution is the Verdict Stability Score (VSS) — using verdict flips across recursive verification depths as a hallucination signal.

## Architecture
```
varity/
├── __init__.py          # Public API: Varity class, __version__
├── checker.py           # RecursiveChecker: main pipeline orchestrator
├── strategies/
│   ├── claim_decompose.py   # Response → list[Claim]
│   ├── self_verify.py       # Recursive depth-N verification loop
│   ├── cross_check.py       # Independent second-opinion check
│   └── confidence.py        # Bayesian scoring + VSS calculation
├── providers/
│   ├── base.py          # BaseLLMProvider ABC
│   ├── anthropic.py     # Claude via user's API key (httpx)
│   ├── openai.py        # GPT via user's API key (httpx)
│   └── gemini.py        # Gemini via user's API key (httpx)
├── models.py            # Pydantic v2: Claim, CheckResult, VerificationStep, VarityConfig
├── prompts.py           # ALL prompt templates as string constants
├── exceptions.py        # VarityError, ProviderError, DecompositionError, etc.
├── cli.py               # CLI entry point: varity check / varity demo / varity batch
└── utils.py             # Retry decorator, token counter, cost estimator, logging
```

## Tech Stack
- Python >=3.9, Pydantic >=2.0, httpx >=0.25, tiktoken >=0.5
- No heavy frameworks — httpx for HTTP, no SDK dependencies
- Tests: pytest, pytest-asyncio, pytest-mock
- Lint: ruff | Types: mypy (strict) | Build: setuptools via pyproject.toml

## BYOK Design Principle
IMPORTANT: Varity uses Bring Your Own Key. The user supplies their API key. We NEVER store, log, cache, or transmit keys anywhere except the direct API call. Every provider __init__ takes `api_key: str`. This is the core zero-cost model.

## Core Algorithm
Pipeline runs in order:
1. ClaimDecomposer → response text → list of typed atomic Claims
2. SelfVerifier → for each claim, verify at depth 0..N recursively
3. CrossChecker → independent second opinion (different prompt, optionally different provider)
4. ConfidenceAggregator → Bayesian scoring with VSS (verdict stability)
5. Correction generator → rewrite flagged claims

Key metric: `flip_count` — times verdict changes across depths. High flips = unstable = hallucinated.

## Commands
```bash
pip install -e ".[dev]"           # Dev install
pytest tests/ -v                  # All tests
pytest tests/test_checker.py -v   # Single file
ruff check varity/                # Lint
mypy varity/ --strict             # Type check
python -m build                   # Build package
python -m varity.cli demo         # CLI demo
```

## Code Style
- Type hints on ALL public functions and methods
- Docstrings: Google style, one-line summary + Args/Returns/Raises
- Async internally, sync wrappers for public API (asyncio.run)
- Use `asyncio.gather` for parallel claim verification — never sequential
- Pydantic v2 patterns — `model_validate`, not v1
- Raw httpx for providers — no SDK dependencies
- Error messages must include provider name and HTTP status
- All prompts in prompts.py as UPPER_SNAKE_CASE constants — never inline

## Important Rules
- NEVER log or print API keys, even partially
- NEVER add anthropic/openai SDKs as required deps — httpx only
- ALL prompt templates go in prompts.py as module-level constants
- Every strategy must degrade gracefully: if it fails, pipeline continues with partial data
- New providers subclass BaseLLMProvider, implement complete() and complete_json()
- Unit tests use MockProvider — never real API calls
- Integration tests in tests/integration/, gated on VARITY_API_KEY env var
- CLI uses ANSI codes for color — no rich/click deps

## Testing
- Mock providers for unit tests, real calls for integration
- Target: 40+ tests before v0.1
- Every change must pass: pytest + ruff + mypy --strict

## Publishing
- PyPI: `varity` | npm: `varity` (varity-js/) | GitHub Pages: docs/ | CLI: [project.scripts]

## Task Flow
1. Plan to tasks/current.md before multi-file changes
2. Mark done as you go
3. Run pytest after every significant change — fix before moving on
4. After corrections, update lessons.md

## Relevant Docs
Read these when working on specific areas:
- `docs/prompts-guide.md` — prompt engineering rationale for each template
- `docs/algorithm.md` — detailed recursive verification algorithm spec
- `docs/zero-cost-setup.md` — Gemini free tier guide
- `benchmarks/README.md` — how to run and interpret benchmarks

## Lessons
<!-- Add lessons here when mistakes are corrected -->
