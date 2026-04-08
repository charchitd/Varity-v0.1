# Varity Prompts Guide

All prompt templates live in `varity/prompts.py` as `UPPER_SNAKE_CASE` module-level
string constants. **Never inline prompts** — always reference these constants.

---

## DECOMPOSE_SYSTEM / DECOMPOSE_USER

**Purpose:** Break a response into a JSON list of atomic claims.

**Key design decisions:**

1. **JSON-only output enforced in the system prompt.** Asking for pure JSON
   prevents the model from wrapping output in markdown or adding explanations,
   which would break `complete_json()`.

2. **`source_span` is requested but treated as best-effort.** Models often
   return incorrect character offsets. The `ClaimDecomposer._parse_one()`
   method validates the span and falls back to `(0, len(original))` if invalid.

3. **`max_claims` is injected via `.format()`** to prevent the model from
   over-decomposing short responses.

4. **claim_type enum is explicit in the system prompt.** Listing the allowed
   values reduces hallucinated types (e.g., "assertion", "belief").

**Failure modes handled:**
- Model returns prose instead of JSON → `complete_json()` raises `DecompositionError`
- `ClaimDecomposer.decompose()` catches this and returns `[]`

---

## VERIFY_SYSTEM / VERIFY_USER

**Purpose:** Ask the LLM to verify a single claim at a given recursion depth.

**Key design decisions:**

1. **Depth is shown explicitly** (`"This is verification pass {depth}"`).
   This signals to the model that this is a re-examination, not a first look.

2. **Prior verdicts are injected as context.** At depth 0, this is "None".
   At depth 2, the model sees `"supported, contradicted"` and is asked to
   reconcile the flip. This is the core mechanism that generates the VSS signal.

3. **`confidence_delta` is requested as a float in [-1, 1].** This gives
   finer signal than a binary flag. Values outside the range are clamped in
   `SelfVerifier._verify_at_depth()`.

4. **Scepticism instruction for numerical/temporal claims.** The prompt adds
   `"Be especially sceptical of numerical, temporal, and causal claims"` —
   these are the most hallucination-prone claim types.

**What NOT to change:**
- Do not remove the prior verdicts injection. Without it, depth > 0 passes
  have no reason to disagree with depth 0, making VSS meaningless.
- Do not ask the model to "justify itself" — this anchors it further.

---

## CROSS_CHECK_SYSTEM / CROSS_CHECK_USER

**Purpose:** Get an independent second opinion with no prior verdict context.

**Key design decisions:**

1. **No prior verdicts passed.** This is intentional — cross-check is a
   fresh perspective. Passing priors would make it redundant with self-verify.

2. **`depth = -1` is used** to mark cross-check steps in the verification
   chain, separating them from the self-verify steps (depth >= 0).

3. **Different framing from VERIFY.** Where VERIFY asks "verify the claim",
   CROSS_CHECK asks "provide an independent verification". The phrasing
   difference alone reduces anchoring to the same answer pattern.

---

## CORRECT_SYSTEM / CORRECT_USER

**Purpose:** Rewrite the original response to correct or qualify flagged claims.

**Key design decisions:**

1. **Only modify flagged claims.** The system prompt explicitly says to
   preserve everything else verbatim. Without this, models tend to paraphrase
   the whole response, losing the user's original tone and structure.

2. **Qualifier strategy vs. removal strategy.** The prompt distinguishes:
   - `uncertain` → add qualifiers ("reportedly", "allegedly", "may")
   - `contradicted` → correct if known truth exists, otherwise remove

3. **Output only the corrected text.** No preamble, no explanation.
   The corrected text is stored as-is in `CheckResult.corrected_response`.

4. **Flagged claims are formatted with type + confidence + VSS.**
   This gives the model extra signal about *how* bad each claim is:
   ```
   - [numerical] It stretches exactly 13,170 miles. (confidence=0.31, vss=0.50)
   ```

---

## Modifying Prompts

When changing a prompt:

1. Update `prompts.py` — the constant, not inline.
2. Update `docs/prompts-guide.md` — explain the rationale.
3. Run `pytest tests/ -v` — strategy tests use mock providers and will catch
   format string errors immediately.
4. Run an integration test if possible to validate LLM behaviour.

Common mistakes:
- Adding a `{variable}` placeholder without updating the `.format()` call
  in the strategy → `KeyError` at runtime.
- Removing the JSON-only instruction → `complete_json()` will raise
  `DecompositionError` intermittently on real providers.
