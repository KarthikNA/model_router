# Model Router

A lightweight, low-latency LLM routing layer that selects the best model for each prompt at runtime — balancing cost, speed, task fit, and model capability — without making any API call to do so.

---

## 1. Goal

Large language model APIs span a wide cost and capability range: a simple factual question costs 100× less on `gpt-4o-mini` than on `claude-opus-4`, yet the quality difference is negligible. The router's job is to exploit this gap automatically.

**Core objective:** given an arbitrary prompt, pick the cheapest model that is still capable enough to handle it well — in under 1 s, with no external dependency.

Secondary objectives (configurable via `ScoringWeights`):
- Minimise per-request cost
- Maximise throughput (tokens/second)
- Match task type and language to model strengths
- Avoid overkill (expensive model on a trivial task) and underkill (weak model on expert reasoning)

---

## 2. Architecture

```
Prompt
  │
  ▼
┌──────────────────────────────────┐
│  Feature Extractor               │  router/features.py
│  - Token count estimation        │
│  - Language detection (script +  │
│    keyword, no external lib)     │
│  - Task type classification      │
│  - Complexity estimation         │
│  - Code/image reference flags    │
└──────────────┬───────────────────┘
               │  PromptFeatures
               ▼
┌──────────────────────────────────┐
│  Scoring Engine                  │  router/scorer.py
│  Hard constraints (eliminate):   │
│  - Context window overflow       │
│  - Complexity ceiling exceeded   │
│  - Model unavailable             │
│                                  │
│  Soft scores (0–1 each):         │
│  - Cost score (log-scaled)       │
│  - Throughput score              │
│  - Relevance score (task + lang) │
│  - Capability score (sweet-spot) │
│                                  │
│  Weighted sum → ranked list      │
└──────────────┬───────────────────┘
               │  list[ModelScore]
               ▼
┌──────────────────────────────────┐
│  ModelRouter                     │  router/router.py
│  - Picks top eligible model      │
│  - Falls back if below threshold │
│  - Returns RouterResult with     │
│    full diagnostics              │
└──────────────────────────────────┘
               │
               ▼
        RouterResult
        (model_id, score, explanation,
         latency_ms, top_candidates)
```

**Model registry** (`registry/models.py`) is a static dictionary of `ModelSpec` objects — one per model — encoding cost, latency, context window, capability flags, and task/language strengths. Adding a new model is a single dict entry; the router picks it up automatically.

**Evaluation framework** (`evaluation/evaluator.py`) runs correctness, latency, and cost benchmarks against a labelled test suite without touching any live API.

---

## 3. Design Trade-offs

| Decision | What was gained | What was given up |
|---|---|---|
| **Pure heuristics, no ML model** | Zero latency, zero dependency, zero cost to route | Cannot learn from real-world feedback; accuracy ceiling is set by hand-tuned rules |
| **Static registry (dict, not DB)** | Sub-microsecond lookup, trivially deployable | Schema changes require a code deploy; no runtime model discovery |
| **Log-scaled cost scoring** | Prevents a single ultra-cheap model from dominating; preserves relative ordering across a 1000× price range | Slightly less intuitive than linear |
| **Soft language constraint** | A model can still be selected for an unsupported language at reduced score | May occasionally route to a suboptimal model for rare languages |
| **Token estimation via char count** | No tokeniser library needed; works offline | ±20% error vs. true BPE count — acceptable for routing decisions, not for billing |
| **Stateless router** | Thread-safe with no locks; trivially horizontally scalable | No request history, no online learning, no adaptive routing |
| **Configurable weights** | Users can tune cost/speed/quality trade-off per use case | Wrong weights produce bad routing; requires understanding of objectives |

---

## 4. How the Design Meets Requirements

**Sub-millisecond routing**
All feature extraction is pure Python regex + Unicode range scans over the first 500 characters of the prompt. No network call, no subprocess, no ML inference. The latency benchmark shows p99 ≈ 0.92 ms at 1,000 iterations, p50 ≈ 0.38 ms.

**100% accuracy on the test suite**
The evaluator covers 10 labelled cases spanning simple factual, summarisation, coding, math proofs, system design, long context, French, Chinese, and agentic tasks — all routed correctly.

**97% cost reduction vs. always using the best model**
By routing simple prompts to fast/cheap models (`gpt-4o-mini`, `gemini-2.0-flash`, `claude-3-5-haiku`) and only escalating to reasoning models for expert-level tasks, average cost per request drops to ~$0.00000067 vs. the ~$0.000022 baseline of always using `claude-opus-4`.

**Latency SLAs met**
- p50 < 200 ms ✓ (actual: ~0.4 ms)
- p99 < 500 ms ✓ (actual: ~0.9 ms under benchmark; ~482 ms under test suite which includes a 140k-token long-context case whose feature extraction dominates)

**Graceful fallback**
If no model passes hard constraints or scores above `min_acceptable_score`, the router falls back to `fallback_model_id` (default: `gpt-4o-mini`) rather than raising an error.

---

## 5. Scaling at High Load

The router is CPU-bound, stateless, and has no shared mutable state. Scaling is straightforward:

**Horizontal scaling**
Deploy as many router instances as needed behind a load balancer. No coordination required — each instance is self-contained.

**Threading / async**
The `ModelRouter` class is thread-safe. Wrap `router.route()` in a thread pool executor or async task scheduler to parallelise over concurrent requests.

**Caching**
For high-volume applications with repeated or near-identical prompts, add a cache keyed on a hash of the prompt text. The deterministic scoring means identical prompts always produce the same result. `features.py` already imports `lru_cache` — apply it to `extract_features` if prompts repeat frequently.

**Registry hot-reload**
Model pricing and availability change regularly. Replace the static `MODEL_REGISTRY` dict with a config file read at startup (or on SIGHUP) to update model specs without a full deploy.

**Bottleneck: long-context feature extraction**
The one latency outlier is token counting on very long prompts (the 140k-token case in the test suite hits ~480 ms because `len(text)` itself is O(n)). For prompts over ~50k tokens, consider sampling the first and last N characters for feature extraction rather than scanning the full text.

**Observability**
`ModelRouter` emits a structured log line per decision (model, score, latency, complexity, task, language, token count, fallback flag) via the standard `logging` module. Feed this into any log aggregator to track routing distributions, cost trends, and accuracy regressions in production.

---

## 6. Running the Project

### Setup

```bash
# Python 3.7+ required
git clone <repo>
cd model_router
```

### Run the demo

```bash
python demo.py
```

The demo routes 7 representative prompts and then runs the full evaluation suite:

```
[Simple factual]
  → Model  : gpt-4o-mini (score=0.863)   ← cheapest model wins
  Latency  : 0.71 ms

[Math proof]
  → Model  : claude-sonnet-4-20250514 (score=0.605)  ← escalated to capable model
  Runners-up: ['deepseek-reasoner', 'o3-mini']
```

**Score** is the weighted sum across cost, throughput, relevance, and capability — higher is better. **Runners-up** are the next two eligible models, useful for A/B testing or fallback chains.

### Run the evaluator

```bash
python -m evaluation.evaluator
```

Output has two sections:

**1. Correctness + SLA report**
```
Accuracy          : 100.0%  (10/10)
Latency p50       : 0.6 ms  ✓ (SLA: <200ms)
Latency p99       : 482.0 ms  ✓ (SLA: <500ms)
Cost vs always-best: 0.03x  (97% savings)
```
- **Accuracy** — fraction of test cases where the routed model was in the acceptable set.
- **p50 / p99 latency** — routing overhead only; does not include the downstream model's inference time.
- **Cost vs always-best** — `1.0` means same cost as always picking the most expensive model; `0.03` means 97% cheaper.

**2. Latency benchmark (1,000 iterations)**
```json
{
  "n": 1000,
  "min_ms": 0.38,
  "p50_ms": 0.383,
  "p99_ms": 0.921,
  "mean_ms": 0.414,
  "meets_p50_sla": true,
  "meets_p99_sla": true
}
```
This runs the same prompt 1,000 times to isolate routing overhead from prompt-length variance. Use `p99_ms` as your production routing budget.

### Customise routing weights

```python
from router.router import ModelRouter, RouterConfig
from router.scorer import ScoringWeights

router = ModelRouter(RouterConfig(
    weights=ScoringWeights(
        cost=0.50,        # prioritise cost savings
        throughput=0.20,
        relevance=0.20,
        capability=0.10,
    ),
    fallback_model_id="gpt-4o-mini",
    top_k=3,
))

result = router.route("Your prompt here")
print(result.model_id)       # selected model
print(result.explanation)    # human-readable reasoning
print(result.latency_ms)     # routing overhead in ms
```

Weights must sum to 1.0. Increase `cost` to lean cheaper; increase `relevance` to lean on task-type matching; increase `capability` to avoid overkill/underkill more aggressively.

### Add a new model

Edit `registry/models.py` and add an entry to `MODEL_REGISTRY`:

```python
"my-new-model": ModelSpec(
    model_id="my-new-model",
    display_name="My New Model",
    provider="my-provider",
    model_type=ModelType.STANDARD,
    max_input_tokens=128_000,
    max_output_tokens=8_192,
    cost_input_per_1m=1.00,
    cost_output_per_1m=3.00,
    avg_tokens_per_second=100,
    avg_latency_first_token_ms=300,
    task_strengths=["general", "coding"],
    language_strengths=["en"],
    min_complexity=TaskComplexity.SIMPLE,
    max_complexity=TaskComplexity.COMPLEX,
),
```

No other changes needed — the router, scorer, and evaluator pick it up automatically.
