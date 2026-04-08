# Varity Algorithm — Recursive Self-Verification

## Overview

Varity takes any LLM response text and runs it through a five-stage pipeline
to produce per-claim confidence scores and a **Verdict Stability Score (VSS)**
— a hallucination signal derived from how often the LLM changes its own mind.

---

## Stage 1 — Claim Decomposition

**Class:** `ClaimDecomposer`

The response text is sent to the LLM with a structured JSON prompt asking it
to break the text into atomic, independently verifiable claims.

Each claim has:
- `text` — the atomic claim string
- `claim_type` — `factual | temporal | numerical | causal | opinion`
- `source_span` — `(start, end)` character indices in the original text

**Failure mode:** If the LLM returns invalid JSON or the network fails,
the decomposer returns an empty list and the pipeline returns a
`CheckResult` with `overall_confidence=1.0` (no claims = nothing to flag).

---

## Stage 2 — Recursive Self-Verification

**Class:** `SelfVerifier`

For each claim, the verifier runs **`depth + 1` passes** (indexed 0 through
`depth`). At each pass the LLM sees:
- The claim text
- The verdicts from all prior passes

This forces the model to **challenge its own reasoning** at each depth,
rather than anchoring to the first answer.

Each pass produces a `VerificationStep`:
```
depth=0 → verdict_0, reasoning_0, confidence_delta_0
depth=1 → verdict_1, reasoning_1, confidence_delta_1  (sees verdict_0)
depth=2 → verdict_2, reasoning_2, confidence_delta_2  (sees verdict_0, verdict_1)
```

**Verdict types:** `supported | contradicted | uncertain`

**Parallelism:** All claims are verified concurrently via `asyncio.gather`.

---

## Stage 3 — Cross-Check

**Class:** `CrossChecker`

Each claim is checked independently using a **different prompt framing**
with no prior verdicts passed in. This eliminates anchoring bias from Stage 2.

Each cross-check produces a `VerificationStep` at `depth = -1` (distinguishes
it from self-verify steps in the chain).

**Parallelism:** All claims are cross-checked concurrently.

---

## Stage 4 — Confidence Aggregation & VSS

**Class:** `ConfidenceAggregator`

### Per-Claim Confidence

Starting from a prior based on the initial verdict:

| Initial verdict | Prior confidence |
|-----------------|-----------------|
| `supported`     | 0.75            |
| `uncertain`     | 0.45            |
| `contradicted`  | 0.15            |

Each subsequent depth updates confidence with a weighted delta:

```
confidence = clamp(confidence + delta × 0.25)
```

The cross-check contributes a lighter nudge:

```
confidence = clamp(confidence + cross_delta × 0.20)
```

### Verdict Stability Score (VSS)

The VSS measures how stable a claim's verdict was across all passes:

```
flip_count     = number of times verdict changed between consecutive passes
max_flips      = (depth) + (1 if cross-check exists else 0)
vss_per_claim  = 1.0 - (flip_count / max_flips)   [1.0 if max_flips == 0]
```

**Confidence is penalised by instability:**

```
confidence = clamp(confidence × (0.5 + 0.5 × vss))
```

This means a claim that always gets `supported` but flips to `contradicted`
at depth 2 will have its confidence significantly reduced even if the final
verdict is `supported`.

### Overall Metrics

```
overall_confidence = mean(claim.confidence  for all claims)
overall_vss        = mean(claim.vss_score   for all claims)
```

### Flagging

A claim is flagged when:

```
claim.confidence < config.confidence_threshold   (default 0.5)
```

---

## Stage 5 — Correction Generator

If any claims are flagged, the full original response is sent to the LLM
with a list of flagged claims and their scores. The LLM is asked to:

- **Qualify** uncertain claims with words like "reportedly" or "allegedly"
- **Correct** contradicted claims if the truth is known
- **Preserve** all unflagged content verbatim

The corrected text is stored in `CheckResult.corrected_response`.

---

## Token Usage Estimation

Token counts are estimated post-hoc using tiktoken by reconstructing the
prompts from known templates and measuring completion text from
`VerificationStep.reasoning` fields. This is an approximation — actual
billed tokens vary by provider.

---

## Configuration

| Parameter              | Default | Effect                                      |
|------------------------|---------|---------------------------------------------|
| `depth`                | 2       | Self-verify passes per claim (0 = 1 pass)   |
| `strategy`             | `full`  | `quick`=depth 1, `paranoid`=depth max(4,N)  |
| `confidence_threshold` | 0.5     | Claims below this are flagged               |
| `max_claims`           | 20      | Decomposition claim limit                   |
| `timeout`              | 30      | Provider timeout in seconds                 |
