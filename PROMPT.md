# PROMPT.md — Varity v0.1 Prompt Engineering & Research Context

## Purpose of This File

This file documents the **why** behind every prompt template in Varity. It serves three audiences:
1. **Claude Code** — read this before modifying any prompt in `prompts.py`
2. **Contributors** — understand the research rationale before changing verification logic
3. **Researchers** — trace the connection between Varity's prompts and the hallucination detection literature

---

## 1. RESEARCH CONTEXT

### 1.1 The Problem

LLMs hallucinate — they generate fluent, confident, factually wrong text. Hallucination rates range from 15-52% across models (2026 benchmarks), exceeding 60% in medical and legal domains. Current detection methods fall into five categories, each with limitations:

| Approach | Example | Limitation |
|---|---|---|
| Retrieval-Augmented | FActScore, REFCHECKER | Requires external knowledge base |
| Uncertainty Estimation | Semantic Entropy, INSIDE | Requires white-box model access |
| Self-Consistency | SelfCheckGPT, SAC3 | Expensive (5-20 re-generations) |
| NLI/Entailment | SummaC, TrueTeacher | Requires reference text |
| Multi-Agent Debate | FactCheck-GPT | Complex, expensive |

### 1.2 Varity's Novel Approach

Varity introduces **recursive depth-based verification** with the **Verdict Stability Score (VSS)**:

```
VSS(claim) = 1.0 - (flip_count / verification_depth)
```

**Key distinction from SelfCheckGPT:**
- SelfCheckGPT: horizontal sampling → generates N independent responses, measures agreement
- Varity: vertical depth → recursively verifies the SAME response at increasing meta-levels

**Why this works:** LLMs are better at verification than generation (asymmetric competence). When asked "Is X true?", they access different reasoning pathways than when generating X. Recursive meta-verification ("Is your verification reliable?") surfaces instability that single-pass checking misses.

### 1.3 The Self-Correction Critique

Huang et al. (2023) argue "LLMs Cannot Self-Correct Reasoning Yet" — self-correction without external feedback degrades performance. Varity addresses this:

1. **Verification ≠ Correction.** Varity asks "Is this accurate?" not "Fix this." Verification is a discriminative task; correction is generative. LLMs perform better on discriminative tasks.
2. **Stability as signal.** We don't trust any single verification. We measure whether the verdict is STABLE across depths. Instability itself is the signal.
3. **Cross-model checking.** Optional second provider breaks the echo chamber.
4. **No anchoring.** Meta-verify prompts deliberately hide the previous depth's confidence score.

### 1.4 Key References

- Manakul et al. (2023). SelfCheckGPT. EMNLP 2023. — Closest prior work (horizontal consistency)
- Min et al. (2023). FActScore. EMNLP 2023. — Pioneered atomic claim decomposition
- Wang et al. (2023). Self-Consistency. ICLR 2023. — Foundational self-consistency for CoT
- Zhang et al. (2023). SAC3. — Semantic-aware cross-check consistency
- Chen et al. (2026). ICQ Framework. NLPCC/Springer. — Iterative chain-query validation
- Jung et al. (2022). Maieutic Prompting. EMNLP. — Recursive explanations for consistency
- Huang et al. (2023). Self-Correction Limits. ICLR 2024. — The critique we must address

---

## 2. PROMPT DESIGN PRINCIPLES

Every prompt in Varity follows these principles. Violating them will degrade accuracy.

### P1: Structured JSON Output
All prompts request JSON output with an explicit schema. This enables deterministic parsing and eliminates regex/string-matching fragility. Use `complete_json()` with a schema parameter.

### P2: Role Separation
The LLM is assigned a specific role per prompt (claim extractor, fact verifier, independent auditor). Roles constrain the response space and reduce off-topic generation.

### P3: No Anchoring at Meta-Levels
Meta-verify prompts MUST NOT reveal the previous depth's confidence score or verdict. Including them creates anchoring bias — the model will rubber-stamp its previous judgment. Only the claim text and the previous depth's reasoning (not verdict) are passed forward.

### P4: Structural Differentiation for Cross-Check
The cross-check prompt MUST be structurally different from the self-verify prompt. Same prompt structure → same reasoning patterns → echo chamber. The cross-check uses a question-answer format instead of the self-verify's assertion-evaluation format.

### P5: Claim Type Awareness
The decomposition prompt distinguishes claim types (factual, temporal, numerical, causal, opinion) because verification strategies differ. Temporal claims need date-checking prompts. Numerical claims need magnitude-checking prompts. Opinions should not be flagged as hallucinations.

### P6: Minimal Context Leakage
Prompts reveal the minimum information needed. The cross-check prompt does NOT see the original self-verification result. The meta-verify prompt does NOT see the confidence score. Each prompt gets only what it needs to do its job independently.

---

## 3. PROMPT TEMPLATES

### 3.1 CLAIM_DECOMPOSE_PROMPT

**Purpose:** Break a free-text LLM response into atomic, independently verifiable claims.

**Design rationale:**
- Inspired by FActScore's atomic fact decomposition
- Each claim must be independently verifiable (not dependent on other claims for meaning)
- Claim types enable strategy-specific verification downstream
- Character spans enable surgical correction (replace flagged text without disturbing the rest)
- Verifiability score filters out opinions early (saves API calls)

```python
CLAIM_DECOMPOSE_SYSTEM = """You are a precise claim extraction engine. Your task is to decompose
text into atomic, independently verifiable claims.

Rules:
- Each claim must be a single factual assertion that can be verified in isolation
- Tag each claim with its type: factual, temporal, numerical, causal, or opinion
- Score verifiability from 0.0 to 1.0 (opinions=0.1, hard facts with dates/numbers=0.95)
- Record the character start and end positions from the original text
- Ignore filler phrases, hedging language ("might", "possibly"), and transitions
- Do NOT split a claim so aggressively that it loses meaning
- If the text contains no verifiable claims, return an empty array

Respond with ONLY a valid JSON array, no preamble, no markdown fences."""

CLAIM_DECOMPOSE_USER = """Decompose the following text into atomic claims.

Original prompt that generated this text:
{context}

Text to decompose:
{response}

Return JSON array where each element has:
{{
  "text": "the atomic claim text",
  "claim_type": "factual|temporal|numerical|causal|opinion",
  "span_start": <int>,
  "span_end": <int>,
  "verifiability": <float 0.0-1.0>
}}"""
```

**Anti-patterns to avoid:**
- Don't ask for "all possible claims" — leads to over-decomposition
- Don't include examples of hallucinated claims — biases extraction toward finding errors
- Don't use "find errors" language — decomposition is neutral, not adversarial

---

### 3.2 SELF_VERIFY_PROMPT (Depth 0)

**Purpose:** First-pass verification — ask the LLM if a specific claim is accurate.

**Design rationale:**
- Chain-of-thought reasoning improves verification accuracy
- Asking for "evidence" forces the model to ground its reasoning
- The verdict is a 3-way classification (supported/contradicted/uncertain) not binary — uncertainty is a valid signal
- Reasoning is captured for the meta-verification step

```python
SELF_VERIFY_SYSTEM = """You are a rigorous fact verification engine. Given a claim extracted
from an LLM response, determine whether the claim is factually accurate.

You MUST:
- Think step by step about the claim's accuracy
- Consider what evidence supports or contradicts the claim
- Provide a clear verdict: supported, contradicted, or uncertain
- Explain your reasoning in 2-3 sentences
- If uncertain, explain what information would resolve the uncertainty

Respond with ONLY valid JSON, no preamble."""

SELF_VERIFY_USER = """Verify the following claim:

Claim: "{claim_text}"
Claim type: {claim_type}
Original prompt context: "{original_prompt}"

Return JSON:
{{
  "verdict": "supported|contradicted|uncertain",
  "reasoning": "Your step-by-step reasoning",
  "confidence": <float 0.0-1.0>,
  "evidence_cited": "What evidence or knowledge supports your verdict",
  "suggested_correction": "If contradicted, what is the correct information? Otherwise null"
}}"""
```

**Why 3-way verdicts:** Binary (true/false) misses the critical "uncertain" case. Uncertain claims at depth 0 that become "contradicted" at depth 1 are the strongest hallucination signals.

---

### 3.3 META_VERIFY_PROMPT (Depth 1+)

**Purpose:** Verify the verification. Ask the LLM to evaluate whether its previous reasoning was sound.

**Design rationale:**
- CRITICAL: Does NOT include the previous verdict or confidence score (prevents anchoring)
- Only includes the previous reasoning text — the model must re-evaluate from reasoning, not from a label
- Asks specifically about reasoning quality, not just agreement
- This is where verdict flips happen — flips are the hallucination signal

```python
META_VERIFY_SYSTEM = """You are a meta-verification engine. You are reviewing a previous
fact-checking assessment to determine if the reasoning was sound and the conclusion reliable.

You MUST:
- Evaluate the REASONING, not just agree with the previous conclusion
- Consider whether the reasoning contains assumptions, circular logic, or gaps
- Provide your own independent assessment of the claim's accuracy
- Be willing to DISAGREE with the previous assessment if the reasoning is flawed

IMPORTANT: Do not defer to the previous assessment. You are an independent reviewer.

Respond with ONLY valid JSON, no preamble."""

META_VERIFY_USER = """Review this fact-checking assessment:

Original claim: "{claim_text}"
Previous reasoning: "{previous_reasoning}"

Questions to consider:
1. Is the reasoning logically sound?
2. Are there assumptions that weren't justified?
3. Could the previous assessment be wrong? Why or why not?
4. Based on your own knowledge, what is your independent verdict?

Return JSON:
{{
  "verdict": "supported|contradicted|uncertain",
  "reasoning": "Your evaluation of the previous reasoning AND your own assessment",
  "confidence": <float 0.0-1.0>,
  "previous_reasoning_quality": "sound|flawed|incomplete",
  "disagreement_reason": "If you disagree with previous, explain why. Otherwise null"
}}"""
```

**The anchoring prevention strategy:**
```
                    ┌─────────────────────────────────────┐
                    │  WHAT GETS PASSED TO META-VERIFY:   │
                    │                                     │
                    │  ✅ claim_text (always)              │
                    │  ✅ previous_reasoning (text only)   │
                    │                                     │
                    │  ❌ previous_verdict (NEVER)         │
                    │  ❌ previous_confidence (NEVER)      │
                    │  ❌ cumulative_score (NEVER)         │
                    └─────────────────────────────────────┘
```

If you pass the verdict, the model anchors to it. If you pass the confidence, the model calibrates to it. Neither produces independent meta-verification. Only the reasoning text provides enough context for the model to re-evaluate without anchoring.

---

### 3.4 CROSS_CHECK_PROMPT

**Purpose:** Independent second opinion using a structurally different prompt format.

**Design rationale:**
- Uses question-answer format (not assertion-evaluation like self-verify)
- Does NOT see any self-verification results (complete independence)
- Can optionally use a different provider (Claude for self-verify, Gemini for cross-check)
- Asks the model to answer the question itself, then compare with the claim

```python
CROSS_CHECK_SYSTEM = """You are an independent fact-checking assistant. You will be given a
factual claim and the question that originally prompted it. Your job is to independently
assess whether the claim is accurate.

IMPORTANT: You have NOT seen any previous verification of this claim. Your assessment
must be entirely independent. Think about what you know, then evaluate the claim.

Respond with ONLY valid JSON, no preamble."""

CROSS_CHECK_USER = """Question that was asked: "{original_prompt}"

A response included this claim: "{claim_text}"

Step 1: Based on your knowledge, what is the accurate answer to the question above?
Step 2: Does the claim align with your answer?
Step 3: If the claim is inaccurate, what specifically is wrong?

Return JSON:
{{
  "your_answer": "What you believe the accurate information is",
  "claim_is_accurate": true|false,
  "alignment_reasoning": "How does the claim compare to your answer?",
  "correction": "If inaccurate, the corrected claim text. Otherwise null"
}}"""
```

**Structural differentiation from SELF_VERIFY:**
```
SELF_VERIFY format:       "Here is a claim. Is it accurate?"
                          (assertion → evaluation)

CROSS_CHECK format:       "Here is a question. What's the answer?
                           Now, does this claim match your answer?"
                          (question → answer → comparison)
```

The two-step cross-check format forces the model to generate its own answer FIRST, then compare. This prevents it from rationalizing the claim as correct (which happens with direct evaluation).

---

### 3.5 CORRECTION_PROMPT

**Purpose:** Generate natural-language corrections for flagged claims.

**Design rationale:**
- Corrections must read naturally in context (not like a diff or patch)
- The model gets both the flagged claim AND the surrounding text
- Corrections should preserve the author's style and tone
- Must handle partial corrections (wrong date but right event)

```python
CORRECTION_SYSTEM = """You are a factual correction assistant. Given a claim that has been
flagged as likely inaccurate, generate a corrected version that:
1. Fixes the factual error
2. Reads naturally in the context of the surrounding text
3. Preserves the original writing style and tone
4. Changes the minimum amount of text needed to fix the error

Respond with ONLY valid JSON, no preamble."""

CORRECTION_USER = """The following claim has been flagged as likely inaccurate:

Flagged claim: "{claim_text}"
Reason flagged: "{verification_reasoning}"
Suggested correction from verification: "{suggested_correction}"

Surrounding context in original text:
"...{text_before}[FLAGGED CLAIM]{text_after}..."

Return JSON:
{{
  "corrected_claim": "The factually corrected version of the claim",
  "correction_explanation": "Brief explanation of what was wrong and what was fixed",
  "confidence_in_correction": <float 0.0-1.0>
}}"""
```

---

## 4. PROMPT TUNING GUIDELINES

### When to modify prompts:
- False positive rate > 30% → loosen self-verify criteria, add "only flag clear errors"
- False negative rate > 40% → tighten meta-verify, add "be skeptical of unsupported assertions"
- Decomposition produces too many claims → add "limit to substantive factual claims, skip trivial details"
- Decomposition produces too few claims → remove "ignore" instructions, lower verifiability threshold
- Cross-check always agrees with self-verify → make the structural difference more extreme
- Confidence scores cluster near 0.5 → adjust scoring weights in confidence.py, not prompts

### When NOT to modify prompts:
- To fix a bug in the Python code (prompts are not code)
- To handle a provider-specific API format (that's in providers/)
- To change confidence scoring math (that's in strategies/confidence.py)
- To add new features (add new prompts, don't overload existing ones)

### Testing prompt changes:
1. Run the 50-case benchmark dataset BEFORE your change
2. Make the change
3. Run the benchmark AFTER
4. Compare precision, recall, F1
5. If any metric drops > 5%, revert
6. Document the change and its impact in a commit message

---

## 5. PROVIDER-SPECIFIC PROMPT NOTES

### Anthropic (Claude)
- Best with chain-of-thought in system prompt
- Responds well to "think step by step"
- JSON output is reliable with explicit schema
- Recommended model for self-verify: claude-sonnet-4-20250514 (cost-effective)

### OpenAI (GPT)
- Use `response_format: { type: "json_object" }` for guaranteed JSON
- Less reliable with complex nested JSON schemas
- "Respond with ONLY valid JSON" must be in both system and user messages

### Gemini
- Free tier has rate limits (15 RPM) — build in delays for cross-check
- JSON mode available via `response_mime_type: "application/json"`
- Good for cross-checking because it uses different training data than Claude/GPT
- Recommended as the $0 cross-check provider in documentation

---

## 6. CONFIDENCE SCORING REFERENCE

Not a prompt, but critical context for understanding how prompt outputs are consumed:

```
Base score: 0.5 (neutral prior)

Per depth level:
  "supported"    → score += 0.2 × decay^depth
  "contradicted" → score -= 0.3 × decay^depth
  "uncertain"    → score += 0.0
  decay = 0.7

Stability:
  Consistent verdicts across all depths → +0.15
  Each verdict flip → -0.10

Cross-check (if available):
  Agreement with self-verify → +0.10
  Disagreement → -0.15

Final: clamp to [0.0, 1.0]

Bands:
  0.0-0.3 → LIKELY HALLUCINATED
  0.3-0.5 → UNCERTAIN
  0.5-0.7 → PROBABLY ACCURATE
  0.7-0.9 → HIGH CONFIDENCE
  0.9-1.0 → VERY HIGH CONFIDENCE
```

---

## 7. PROMPT EVOLUTION LOG

Track all prompt changes here with dates, rationale, and benchmark impact.

| Date | Prompt | Change | Precision Δ | Recall Δ | Rationale |
|---|---|---|---|---|---|
| v0.1.2 | All | Initial versions | baseline | baseline | Launch |

---

*This document is the single source of truth for all prompt-related decisions in Varity.
Before changing any prompt in `prompts.py`, read the relevant section here first.*
