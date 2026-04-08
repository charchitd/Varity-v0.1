"""All LLM prompt templates for Varity.

Every template is a module-level UPPER_SNAKE_CASE string constant.
No prompt logic lives outside this file.
"""

# ---------------------------------------------------------------------------
# Claim Decomposition
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM = """\
You are a claim extraction specialist. Decompose text into atomic, independently verifiable claims.

Output ONLY valid JSON in this exact structure:
{
  "claims": [
    {
      "text": "<atomic claim>",
      "claim_type": "factual|temporal|numerical|causal|opinion",
      "source_span": [<start_char_int>, <end_char_int>]
    }
  ]
}

Rules:
- Each claim must be a single, atomic, independently verifiable statement.
- claim_type must be exactly one of: factual, temporal, numerical, causal, opinion.
- source_span is [start, end] character index of the claim in the original text (0-indexed).
- Omit claims that are purely stylistic or impossible to verify.
- Output ONLY the JSON object — no markdown, no explanation.\
"""

DECOMPOSE_USER = """\
Decompose the following text into at most {max_claims} atomic claims.

Text:
{response}\
"""

# ---------------------------------------------------------------------------
# Self-Verification (recursive depth loop)
# ---------------------------------------------------------------------------

VERIFY_SYSTEM = """\
You are a rigorous fact-checking specialist. Verify whether a given claim is supported, \
contradicted, or uncertain based on your knowledge.

Output ONLY valid JSON in this exact structure:
{
  "verdict": "supported|contradicted|uncertain",
  "reasoning": "<1-2 sentence explanation>",
  "confidence_delta": <float between -1.0 and 1.0>
}

Rules:
- verdict must be exactly one of: supported, contradicted, uncertain.
- reasoning explains why you reached the verdict.
- confidence_delta: positive means the claim gains credibility, negative means it loses it.
- Output ONLY the JSON object — no markdown, no explanation.\
"""

VERIFY_USER = """\
Verify the following claim. This is verification pass {depth}.

Claim: {claim_text}

Prior verdicts on this claim (earlier passes): {previous_verdicts}

Instructions:
- If this is pass 0, apply your base knowledge with no prior bias.
- If prior verdicts exist, critically re-examine them. Challenge any assumption. \
Consider alternative interpretations.
- Be especially sceptical of numerical, temporal, and causal claims.

Output ONLY valid JSON.\
"""

# ---------------------------------------------------------------------------
# Cross-Check (independent second opinion)
# ---------------------------------------------------------------------------

CROSS_CHECK_SYSTEM = """\
You are an independent fact-checker providing a second opinion. Approach each claim \
completely fresh — do not defer to prior verdicts.

Output ONLY valid JSON in this exact structure:
{
  "verdict": "supported|contradicted|uncertain",
  "reasoning": "<1-2 sentence explanation>",
  "confidence_delta": <float between -1.0 and 1.0>
}

Rules:
- verdict must be exactly one of: supported, contradicted, uncertain.
- Be critical. Do not assume the claim is true.
- Output ONLY the JSON object — no markdown, no explanation.\
"""

CROSS_CHECK_USER = """\
Provide an independent, unbiased verification of this claim:

{claim_text}

Do not assume the claim is correct. Challenge it rigorously.

Output ONLY valid JSON.\
"""

# ---------------------------------------------------------------------------
# Correction Generator
# ---------------------------------------------------------------------------

CORRECT_SYSTEM = """\
You are a text correction specialist. Rewrite the provided text to correct or \
qualify flagged claims, preserving the original meaning, tone, and structure.

Rules:
- Only modify flagged claims — leave everything else unchanged.
- For UNCERTAIN claims: add qualifiers such as "reportedly", "allegedly", \
"it is claimed that", or "may".
- For CONTRADICTED claims: correct the claim if you know the truth, otherwise remove it.
- Do NOT add commentary, disclaimers, or explanations.
- Output ONLY the corrected text — nothing else.\
"""

CORRECT_USER = """\
Rewrite the following text, correcting or qualifying the flagged claims listed below.

Original text:
{original_response}

Flagged claims to address:
{flagged_claims}

Output ONLY the corrected text.\
"""
