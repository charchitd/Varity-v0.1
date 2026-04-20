# Varity v0.1.9 — Launch Roadmap

## [x] Phase 1 — Pre-publish Checklist (1–2 days)

### [x] 1.1 Fix placeholder URLs in pyproject.toml
```toml
Homepage = "https://github.com/charchitd/varity"
Documentation = "https://charchitd.github.io/varity"
Repository = "https://github.com/charchitd/varity"
"Bug Tracker" = "https://github.com/charchitd/varity/issues"
```

### [x] 1.2 Clean build
```bash
rm -rf dist/ build/ varity.egg-info/
python -m build
ls dist/   # varity-0.1.0-py3-none-any.whl + .tar.gz
```

### [x] 1.3 Verify install
```bash
pip install dist/varity-0.1.0-py3-none-any.whl
python -c "import varity; print(varity.__version__)"
varity demo
```

### [x] 1.4 Full gate
```bash
python -m pytest tests/ -v
python -m ruff check varity/
python -m mypy varity/ --strict
```

---

## [x] Phase 2 — CI + Badges (half day)

### [x] 2.1 GitHub Actions CI
Create `.github/workflows/ci.yml` and `.github/workflows/publish.yml` with strict OIDC support.

### [x] 2.2 README badges
```markdown
[![PyPI - Version](https://img.shields.io/pypi/v/varity.svg)](https://pypi.org/project/varity/)
[![Python Versions](https://img.shields.io/pypi/pyversions/varity.svg)](https://pypi.org/project/varity/)
```

### [x] 2.3 Repo topics
Added: `llm`, `hallucination-detection`, `ai-safety`, `fact-checking`, `nlp`, `python`, `vss`, `llm-reliability`, `prompt-engineering`, `anthropic`, `openai`, `gemini`

---

## [x] Phase 3 — PyPI Publish (1 hour)

```bash
# 1. Bump version
# 2. Re-build package
# 3. Publish to PyPI
pip install varity
```

---

## [x] Phase 4 — GitHub Pages (half day)

### [x] 4.1 Launch Varity UI at `/docs`
### [x] 4.2 Whitepaper `CONCEPTS.md` mathematical logic

---

## [ ] Phase 5 — Benchmarks (before any big post)

Run benchmarks, publish this table:

| Model | Claim Accuracy | VSS Correlation | Hallucination Detection Rate |
|-------|---------------|-----------------|------------------------------|
| GPT-4o | X% | X | X% |
| Claude 3.5 | X% | X | X% |
| Gemini 1.5 | X% | X | X% |

Numbers = credibility = shareability.

---

## [ ] Phase 6 — Launch & Awareness

### The Hook (lead every post with this)
> "What if your LLM verified its own answers — recursively — and told you exactly how stable that verdict was?"

### [ ] 6.1 Hacker News (highest ROI)
Post Tuesday–Thursday, 9–11am EST. One shot.
```
Show HN: Varity – recursive self-checking for LLM hallucinations with Verdict Stability Score

I built a Python library that decomposes LLM responses into atomic claims,
verifies each one recursively at depth N, and uses verdict *flip count*
across depths as a hallucination signal (VSS). BYOK — works with Anthropic,
OpenAI, Gemini. pip install varity

GitHub: [url]
```

### [ ] 6.2 Reddit
- **r/MachineLearning** — technical, include VSS algorithm + benchmark table
- **r/LocalLLaMA** — practical: "I added recursive self-checking to my LLM pipeline"
- **r/Python** — pip install angle, clean code snippet
- **r/artificial** — general AI audience

### [ ] 6.3 Dev.to / Medium article
**Title:** "How I reduced LLM hallucinations by 40% using recursive self-verification"

Structure: Problem → Insight (verdict instability = hallucination signal) → VSS explained → Code snippet → Benchmarks → Repo link

Cross-post to: Dev.to, Medium, Towards Data Science, Hashnode

### [ ] 6.4 Twitter/X Thread
```
1/ I built a Python library that catches LLM hallucinations
   using recursive self-verification.

   Key insight: if an LLM flips its verdict when re-checking
   the same claim at deeper recursion depths, that's a
   hallucination signal. I call it the Verdict Stability Score (VSS). 🧵

2/ Here's how it works: [diagram]
3/ 5 lines of code: [snippet]
4/ BYOK — bring your own OpenAI/Anthropic/Gemini key
5/ pip install varity | GitHub: [url]

#LLM #AIEngineering #Python #MachineLearning
```

### [ ] 6.5 GitHub virality
- [x] Add 12 topic tags
- [x] Add `CONTRIBUTING.md`
- [ ] Submit to `awesome-llm`, `awesome-hallucination`, `awesome-ai-safety` lists
- [ ] Open `good first issue` + `help wanted` issues
- [ ] Pin repo on profile

---

## Priority Execution Order

| When | Action | Status |
|------|--------|--------|
| **Week 1** | Fix URLs → clean build → PyPI publish → CI green → badges | ✅ DONE |
| **Week 1** | GitHub Pages live | ✅ DONE |
| **Week 1** | Architecture / Concepts Mathematical Whitepaper | ✅ DONE |
| **Week 2** | Run benchmarks → write Dev.to article | ⏳ PENDING |
| **Week 2** | **Show HN post** (single highest-leverage move) | ⏳ PENDING |
| **Week 3** | Reddit posts → Twitter thread → awesome-list PRs | ⏳ PENDING |
| **Ongoing** | Respond to every issue/comment in <24h | ⏳ PENDING |

---

## Single Most Important Action

**The Show HN post with benchmark numbers.** That's the one that can take a Python library from 0 to 500 stars overnight. Everything else supports it.
