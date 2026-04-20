"""
Varity Demo App — Streamlit
Recursive Self-Checking for LLM Hallucination Reduction
"""

import asyncio
import sys
import time
from pathlib import Path

import streamlit as st

# Allow running from repo root or demo/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Varity Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Example responses (with known ground truth)
# ---------------------------------------------------------------------------

EXAMPLES = {
    "-- Choose an example --": "",
    "🧱 Great Wall myth (has errors)": (
        "The Great Wall of China was built entirely during the Ming Dynasty. "
        "It is clearly visible from space with the naked eye and stretches "
        "exactly 13,170 miles. Construction began in 221 BC under Emperor "
        "Qin Shi Huang."
    ),
    "🐍 Python history (mostly correct)": (
        "Python was created by Guido van Rossum and first released in 1991. "
        "It is named after the British comedy group Monty Python. "
        "Python 3.0 was released in 2008 and was not backward-compatible with Python 2."
    ),
    "🌍 Mixed facts (some errors)": (
        "The Amazon River is the longest river in the world at over 6,400 km. "
        "Mount Everest is the tallest mountain on Earth at 8,849 metres. "
        "The Sahara Desert is the largest desert on Earth. "
        "Australia is both a country and a continent."
    ),
    "🔬 Science claims (all correct)": (
        "Water boils at 100 degrees Celsius at standard atmospheric pressure. "
        "The speed of light in a vacuum is approximately 299,792 kilometres per second. "
        "DNA is a double-helix structure discovered by Watson and Crick in 1953. "
        "Penicillin was discovered by Alexander Fleming in 1928."
    ),
    "📅 Historical dates (has errors)": (
        "World War II ended in 1945. "
        "The Berlin Wall fell in 1989. "
        "The first Moon landing occurred in 1967. "
        "The French Revolution began in 1789."
    ),
    "✍️ Paste your own text": "custom",
}

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    import varity
    st.image(f"https://img.shields.io/badge/Varity-v{varity.__version__}-blue", width=120)
    st.title("⚙️ Configuration")

    st.markdown("### 1. Provider")
    provider_name = st.selectbox(
        "LLM Provider",
        ["anthropic", "openai", "gemini"],
        format_func=lambda x: {
            "anthropic": "🟣 Anthropic (Claude)",
            "openai": "🟢 OpenAI (GPT)",
            "gemini": "🔵 Google Gemini (free tier)",
        }[x],
    )

    default_models = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.0-flash",
    }
    model_override = st.text_input(
        "Model (optional override)",
        placeholder=default_models[provider_name],
    )

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Paste your API key here",
        help="Your key is used only for the direct API call. Never stored or logged.",
    )

    st.markdown("---")
    st.markdown("### 2. Verification Settings")

    strategy = st.selectbox(
        "Strategy",
        ["quick", "full", "paranoid"],
        index=1,
        format_func=lambda x: {
            "quick": "⚡ Quick  (depth=1, fewer API calls)",
            "full": "🔍 Full   (depth=2, balanced)",
            "paranoid": "🛡️ Paranoid (depth=4, thorough)",
        }[x],
    )

    depth = st.slider(
        "Verification Depth",
        min_value=0,
        max_value=5,
        value=2,
        help="Number of recursive verification passes per claim (0 = 1 pass, 2 = 3 passes).",
    )

    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Claims with confidence below this value are flagged.",
    )

    vss_threshold = st.slider(
        "VSS Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Claims with VSS (Verdict Stability Score) below this value are flagged, independently of confidence. Low VSS means the LLM kept changing its verdict — a hallucination signal.",
    )

    max_claims = st.slider(
        "Max Claims",
        min_value=3,
        max_value=30,
        value=10,
        help="Maximum number of claims to extract from the response.",
    )

    st.markdown("---")
    st.markdown("### 3. About VSS")
    st.info(
        "**Verdict Stability Score (VSS)** measures how often the LLM "
        "changes its verdict about a claim across recursive verification passes.\n\n"
        "- VSS = 1.0 → verdict never changed → likely grounded\n"
        "- VSS = 0.0 → verdict flipped every pass → likely hallucinated"
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🔍 Varity — LLM Hallucination Detector")
st.markdown(
    "Paste any LLM response below. Varity will decompose it into atomic claims, "
    "recursively verify each one, compute a **Verdict Stability Score (VSS)**, "
    "flag suspicious claims, and suggest a corrected response."
)

# ---------------------------------------------------------------------------
# How it works — collapsible
# ---------------------------------------------------------------------------

with st.expander("📖 How Varity Works — Step by Step", expanded=False):
    st.markdown("""
    Varity runs a **5-stage pipeline** on any text:

    | Stage | What happens |
    |-------|-------------|
    | **1. Decompose** | The LLM breaks the response into atomic, verifiable claims |
    | **2. Self-Verify** | Each claim is verified recursively across `depth+1` passes — the model re-examines its own verdict at each depth |
    | **3. Cross-Check** | An independent second-opinion check using a different prompt framing |
    | **4. Score (VSS)** | Verdict flips across depths are counted. More flips = lower VSS = more likely hallucinated |
    | **5. Correct** | Flagged claims are rewritten with qualifiers or corrections |

    **The key insight:** If an LLM keeps changing its mind about a claim across multiple passes,
    that *instability itself* is a hallucination signal — even if the final verdict is "supported".

    ```
    VSS = 1 - (flip_count / max_possible_flips)

    Confidence = prior_by_verdict × Bayesian_updates × (0.5 + 0.5 × VSS)
    ```
    """)

# ---------------------------------------------------------------------------
# Example selector
# ---------------------------------------------------------------------------

st.markdown("### Try an Example")
selected_example = st.selectbox("Load a pre-built example:", list(EXAMPLES.keys()))

if selected_example == "✍️ Paste your own text":
    response_text = st.text_area(
        "Response to verify",
        height=140,
        placeholder="Paste any LLM-generated text here...",
    )
elif EXAMPLES[selected_example]:
    response_text = st.text_area(
        "Response to verify",
        value=EXAMPLES[selected_example],
        height=140,
    )
else:
    response_text = st.text_area(
        "Response to verify",
        height=140,
        placeholder="Select an example above or paste your own text...",
    )

# ---------------------------------------------------------------------------
# Validate inputs before run
# ---------------------------------------------------------------------------

ready = bool(response_text.strip()) and bool(api_key.strip())

if not api_key.strip():
    st.warning("⚠️ Enter your API key in the sidebar to run a check.")
if not response_text.strip():
    st.info("💡 Select an example or paste text above to get started.")

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_clicked = st.button(
        "🚀 Run Varity Check",
        disabled=not ready,
        type="primary",
        use_container_width=True,
    )
with col_info:
    if ready:
        call_estimate = 1 + max_claims * (depth + 2)
        st.caption(
            f"Estimated API calls: ~{call_estimate}  |  "
            f"Strategy: {strategy}  |  Depth: {depth}  |  Provider: {provider_name}"
        )

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

if run_clicked and ready:
    from varity import Varity, VarityConfig
    from varity.exceptions import VarityError
    from varity.providers import get_provider

    st.markdown("---")
    st.markdown("### 🔄 Running Pipeline...")

    progress = st.progress(0, text="Initialising...")
    status_box = st.empty()

    try:
        # Build provider + config
        kwargs = {}
        if model_override.strip():
            kwargs["model"] = model_override.strip()

        provider = get_provider(provider_name, api_key=api_key.strip(), **kwargs)
        config = VarityConfig(
            depth=depth,
            strategy=strategy,
            confidence_threshold=threshold,
            vss_threshold=vss_threshold,
            max_claims=max_claims,
        )
        varity = Varity(provider=provider, config=config)

        # Stage progress updates
        progress.progress(10, text="Stage 1: Decomposing claims...")
        status_box.info("Sending response to LLM for claim extraction...")

        start = time.monotonic()

        # Run async pipeline in sync Streamlit context
        result = asyncio.run(varity.acheck(response_text))

        elapsed = time.monotonic() - start
        progress.progress(100, text="Done!")
        status_box.empty()

        # ------------------------------------------------------------------
        # Results header
        # ------------------------------------------------------------------

        st.markdown("---")
        st.markdown("## ✅ Results")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Claims Found", len(result.claims))
        m2.metric("Flagged", len(result.flagged_claims))

        conf_pct = f"{result.overall_confidence:.1%}"
        conf_delta = "low ⚠️" if result.overall_confidence < 0.5 else "good ✓"
        m3.metric("Confidence", conf_pct, delta=conf_delta,
                  delta_color="inverse" if result.overall_confidence < 0.5 else "normal")

        vss_pct = f"{result.vss_score:.1%}"
        vss_delta = "unstable ⚠️" if result.vss_score < 0.6 else "stable ✓"
        m4.metric("VSS", vss_pct, delta=vss_delta,
                  delta_color="inverse" if result.vss_score < 0.6 else "normal")

        total_tok = result.token_usage.get("total_tokens", 0)
        m5.metric("Tokens (est.)", f"{total_tok:,}" if total_tok else "—")

        st.caption(f"Pipeline completed in {elapsed:.1f}s  |  {result.duration_ms} ms measured")

        # ------------------------------------------------------------------
        # Claims table
        # ------------------------------------------------------------------

        if result.claims:
            st.markdown("### 📋 Claims")
            st.markdown(
                "Each row is one atomic claim extracted from the response. "
                "**Flagged** claims have confidence below your threshold OR VSS below your VSS threshold."
            )

            for i, claim in enumerate(result.claims, 1):
                flagged = claim.flagged
                icon = "🔴" if flagged else "🟢"
                bg = "#fff0f0" if flagged else "#f0fff4"
                border = "#ff4444" if flagged else "#44bb44"

                with st.container():
                    st.markdown(
                        f"""
                        <div style="
                            background:{bg};
                            border-left: 4px solid {border};
                            padding: 10px 14px;
                            border-radius: 4px;
                            margin-bottom: 8px;
                        ">
                            <b>{icon} Claim {i}</b> &nbsp;
                            <span style="background:#e8e8e8;padding:2px 6px;border-radius:3px;font-size:0.8em">{claim.claim_type}</span>
                            &nbsp;
                            <span style="font-size:0.85em;color:#555">{claim.verification_notes}</span>
                            <br/>
                            <span style="font-size:1em">{claim.text}</span>
                            <br/>
                            <span style="font-size:0.85em">
                                Confidence: <b>{claim.confidence:.2f}</b> &nbsp;|&nbsp;
                                VSS: <b>{claim.vss_score:.2f}</b> &nbsp;|&nbsp;
                                Flips: <b>{claim.flip_count}</b>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ------------------------------------------------------------------
        # Verification chain
        # ------------------------------------------------------------------

        if result.verification_chain:
            with st.expander(
                f"🔗 Verification Chain ({len(result.verification_chain)} steps)", expanded=False
            ):
                st.markdown(
                    "Each row shows one LLM verification call. "
                    "`depth >= 0` = self-verify pass, `depth = -1` = cross-check."
                )
                chain_data = [
                    {
                        "Claim": step.claim_text[:55] + "..." if len(step.claim_text) > 55 else step.claim_text,
                        "Depth": "cross-check" if step.depth == -1 else str(step.depth),
                        "Verdict": step.verdict,
                        "Delta": f"{step.confidence_delta:+.2f}",
                        "Reasoning": step.reasoning[:80] + "..." if len(step.reasoning) > 80 else step.reasoning,
                    }
                    for step in result.verification_chain
                ]
                st.dataframe(chain_data, use_container_width=True)

        # ------------------------------------------------------------------
        # Corrected response
        # ------------------------------------------------------------------

        if result.corrected_response:
            st.markdown("### ✏️ Corrected Response")
            st.success(result.corrected_response)

            col_orig, col_corr = st.columns(2)
            with col_orig:
                st.markdown("**Original**")
                st.text_area("", value=result.original_response, height=120,
                             disabled=True, key="orig")
            with col_corr:
                st.markdown("**Corrected**")
                st.text_area("", value=result.corrected_response, height=120,
                             disabled=True, key="corr")
        else:
            if result.claims:
                st.success("✅ No claims flagged — no correction needed.")

        # ------------------------------------------------------------------
        # Raw JSON
        # ------------------------------------------------------------------

        with st.expander("📄 Raw JSON Output", expanded=False):
            st.json(result.model_dump())

    except VarityError as e:
        progress.empty()
        st.error(f"**Varity Error:** {e}")
    except Exception as e:
        progress.empty()
        err_str = str(e)
        if "Invalid API key" in err_str or "401" in err_str:
            st.error("❌ Invalid API key. Check your key in the sidebar.")
        elif "429" in err_str:
            st.warning("⏳ Rate limited by provider. Wait a moment and try again.")
        else:
            st.error(f"❌ Unexpected error: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#999;font-size:0.85em">
        Varity v0.1.9 &nbsp;|&nbsp;
        BYOK — your API key is used only for the direct LLM call, never stored &nbsp;|&nbsp;
        <a href="https://github.com/yourusername/varity">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
