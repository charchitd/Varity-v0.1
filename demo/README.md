# Varity Demo App

Interactive Streamlit demo for the Varity hallucination detection library.

---

## Quick Start (3 steps)

### Step 1 — Install dependencies

```bash
cd demo/
pip install -r requirements.txt
```

Or install from the repo root:

```bash
pip install -e ".[dev]"
pip install streamlit
```

### Step 2 — Run the app

```bash
# From the repo root:
streamlit run demo/app.py

# Or from inside demo/:
streamlit run app.py
```

The app opens at **http://localhost:8501**

### Step 3 — Try it

1. Select a **provider** in the sidebar (Gemini is free)
2. Paste your **API key**
3. Pick one of the **5 built-in examples** from the dropdown
4. Click **Run Varity Check**

---

## Built-in Examples

| Example | What to expect |
|---------|---------------|
| 🧱 Great Wall myth | 1-2 claims flagged (space visibility, exact length) |
| 🐍 Python history | All claims pass — mostly correct facts |
| 🌍 Mixed facts | Amazon longest river claim flagged |
| 🔬 Science claims | All pass — verified scientific facts |
| 📅 Historical dates | Moon landing date (1967) flagged — should be 1969 |

---

## Getting a Free API Key (Gemini)

1. Go to https://aistudio.google.com/
2. Sign in → **Get API key → Create API key**
3. Copy the key (starts with `AIza...`)
4. Paste it in the sidebar

No credit card needed. Free tier: 15 requests/minute.

---

## What the App Shows

```
┌─────────────────────────────────────────────────────┐
│  Sidebar                  │  Main Area               │
│  ─────────────────────── │  ─────────────────────── │
│  Provider selector        │  How it works (expander) │
│  API key (masked)         │  Example selector        │
│  Strategy: quick/full/    │  Text area               │
│    paranoid               │  [Run Varity Check]      │
│  Depth slider (0-5)       │                          │
│  Flag threshold (0.1-0.9) │  Results:                │
│  Max claims (3-30)        │  • Metrics row           │
│  About VSS explanation    │  • Per-claim cards       │
│                           │  • Verification chain    │
│                           │  • Corrected response    │
│                           │  • Raw JSON (expander)   │
└─────────────────────────────────────────────────────┘
```

---

## Deploy to Streamlit Cloud (Free)

1. Push the repo to GitHub
2. Go to https://share.streamlit.io/
3. Click **New app → select your repo → set main file to `demo/app.py`**
4. Click **Deploy**

Users supply their own API keys in the sidebar — no keys stored server-side.

---

## Configuration Reference

| Setting | Default | Effect |
|---------|---------|--------|
| Strategy: Quick | depth=1 | Fastest, ~15 API calls for 5 claims |
| Strategy: Full | depth=2 | Balanced, ~20 API calls for 5 claims |
| Strategy: Paranoid | depth=4 | Thorough, ~35 API calls for 5 claims |
| Flag Threshold | 0.5 | Claims below this confidence are flagged |
| Max Claims | 10 | Limits extraction to keep API costs low |
