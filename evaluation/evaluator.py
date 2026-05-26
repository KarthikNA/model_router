"""
Evaluation Framework for the Model Router.

Three evaluation dimensions:
  1. Correctness  — does the router pick the right model?
  2. Latency      — does it meet p50 < 200ms, p99 < 500ms?
  3. Cost savings — how much cheaper is routed vs always-best?

Run:
    python -m evaluation.evaluator
"""

from __future__ import annotations

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) in ("evaluation", "router", "registry") else _HERE
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

from registry.models import TaskComplexity
from router.features import extract_features
from router.router import ModelRouter, RouterConfig
from router.scorer import ScoringWeights


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    prompt: str
    expected_model_ids: list[str]    # acceptable set (not always one answer)
    description: str
    min_acceptable_complexity: TaskComplexity | None = None


STANDARD_TEST_SUITE: list[TestCase] = [
    # Trivial / simple → cheap fast model
    TestCase(
        prompt="What is the capital of France?",
        expected_model_ids=["gpt-4o-mini", "claude-3-5-haiku-20241022", "gemini-2.0-flash"],
        description="Simple factual — should pick cheapest fast model",
        min_acceptable_complexity=TaskComplexity.TRIVIAL,
    ),
    TestCase(
        prompt="Summarise this paragraph in one sentence: The quick brown fox jumps over the lazy dog.",
        expected_model_ids=["gpt-4o-mini", "claude-3-5-haiku-20241022", "gemini-2.0-flash"],
        description="Summarisation — cheap model sufficient",
    ),

    # Coding → code-capable model (gemini-2.0-flash and deepseek are valid choices)
    TestCase(
        prompt="Write a Python function that implements binary search on a sorted list.",
        expected_model_ids=["gpt-4o", "claude-sonnet-4-20250514", "deepseek-chat", "gemini-2.0-flash", "gpt-4o-mini"],
        description="Coding — standard model or code specialist",
    ),
    TestCase(
        prompt="```python\ndef foo(x):\n    return x*2\n```\nDebug this function and add type hints.",
        expected_model_ids=["gpt-4o", "claude-sonnet-4-20250514", "deepseek-chat", "gemini-2.0-flash"],
        description="Coding with code block",
    ),

    # Complex reasoning → reasoning model
    TestCase(
        prompt="Prove that there are infinitely many prime numbers using Euclid's proof. Then extend the argument to show that the primes are not periodic.",
        expected_model_ids=["o3-mini", "deepseek-reasoner", "claude-opus-4-20250514", "gemini-2.5-pro", "claude-sonnet-4-20250514"],
        description="Math proof — reasoning model needed",
        min_acceptable_complexity=TaskComplexity.COMPLEX,
    ),
    TestCase(
        prompt="Design a distributed rate limiter for 10M RPS. Consider consistency trade-offs, failure modes, and operational complexity. Step by step.",
        expected_model_ids=["o3-mini", "claude-opus-4-20250514", "gemini-2.5-pro", "claude-sonnet-4-20250514", "deepseek-chat", "deepseek-reasoner"],
        description="Complex system design — reasoning or capable standard",
        min_acceptable_complexity=TaskComplexity.COMPLEX,
    ),

    # Long context
    TestCase(
        prompt="Analyse this document: " + "Lorem ipsum dolor sit amet. " * 5000,
        expected_model_ids=["gemini-2.5-pro", "gemini-2.0-flash", "claude-sonnet-4-20250514", "claude-opus-4-20250514"],
        description="Long context — model with large window needed",
    ),

    # Non-English — fast models that support the language are acceptable
    TestCase(
        prompt="Expliquez-moi le théorème de Bayes en termes simples.",
        expected_model_ids=["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.0-flash", "gpt-4o-mini", "claude-3-5-haiku-20241022"],
        description="French prompt — model with French support",
    ),
    TestCase(
        prompt="用中文解释深度学习的基本原理。",
        expected_model_ids=["gpt-4o", "deepseek-chat", "gemini-2.5-pro", "gemini-2.0-flash", "gpt-4o-mini"],
        description="Chinese prompt — model with Chinese support",
    ),

    # Agentic
    TestCase(
        prompt="You are an agent. Use the available tools to research the top 5 ML papers of 2024 and compile a summary report with citations.",
        expected_model_ids=["claude-sonnet-4-20250514", "claude-opus-4-20250514", "gpt-4o", "deepseek-reasoner", "o3-mini"],
        description="Agentic task — capable model with function calling",
        min_acceptable_complexity=TaskComplexity.COMPLEX,
    ),
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    test_case: TestCase
    routed_model_id: str
    correct: bool
    latency_ms: float
    score: float
    explanation: str
    features_extraction_ms: float


@dataclass
class EvaluationReport:
    accuracy: float              # fraction of test cases correctly routed
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    meets_p50_sla: bool         # < 200ms
    meets_p99_sla: bool         # < 500ms
    avg_cost_per_request: float
    cost_vs_always_best: float  # ratio: 1.0 = same cost, 0.5 = 50% cheaper
    results: list[TestResult] = field(default_factory=list)
    failures: list[TestResult] = field(default_factory=list)

    def print_summary(self) -> None:
        print("\n" + "="*60)
        print("MODEL ROUTER — EVALUATION REPORT")
        print("="*60)
        print(f"Accuracy          : {self.accuracy*100:.1f}%  ({sum(r.correct for r in self.results)}/{len(self.results)})")
        print(f"Latency p50       : {self.p50_latency_ms:.1f} ms  {'✓' if self.meets_p50_sla else '✗'} (SLA: <200ms)")
        print(f"Latency p95       : {self.p95_latency_ms:.1f} ms")
        print(f"Latency p99       : {self.p99_latency_ms:.1f} ms  {'✓' if self.meets_p99_sla else '✗'} (SLA: <500ms)")
        print(f"Avg cost/request  : ${self.avg_cost_per_request*1000:.4f}m")
        print(f"Cost vs always-best: {self.cost_vs_always_best:.2f}x  ({(1-self.cost_vs_always_best)*100:.0f}% savings)")
        print()
        if self.failures:
            print(f"FAILURES ({len(self.failures)}):")
            for f in self.failures:
                print(f"  ✗ [{f.test_case.description}]")
                print(f"    Got: {f.routed_model_id}")
                print(f"    Expected one of: {f.test_case.expected_model_ids}")
        print("="*60)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RouterEvaluator:

    def __init__(self, router: ModelRouter | None = None):
        self.router = router or ModelRouter()

    def run(
        self,
        test_suite: list[TestCase] | None = None,
        warmup_rounds: int = 3,
    ) -> EvaluationReport:
        cases = test_suite or STANDARD_TEST_SUITE

        # Warm up (JIT, caches)
        for _ in range(warmup_rounds):
            self.router.route("hello")

        results: list[TestResult] = []

        for case in cases:
            result_obj = self.router.route(case.prompt)
            correct = result_obj.model_id in case.expected_model_ids

            results.append(TestResult(
                test_case=case,
                routed_model_id=result_obj.model_id,
                correct=correct,
                latency_ms=result_obj.latency_ms,
                score=result_obj.score,
                explanation=result_obj.explanation,
                features_extraction_ms=result_obj.features.extraction_time_ms,
            ))

        return self._build_report(results)

    def run_latency_benchmark(
        self,
        prompt: str = "Explain the concept of recursion in programming.",
        n_iterations: int = 1000,
    ) -> dict:
        """Isolated latency benchmark — returns percentile breakdown."""
        latencies: list[float] = []

        # Warmup
        for _ in range(10):
            self.router.route(prompt)

        for _ in range(n_iterations):
            t0 = time.monotonic()
            self.router.route(prompt)
            latencies.append((time.monotonic() - t0) * 1000)

        latencies.sort()
        n = len(latencies)

        return {
            "n": n,
            "min_ms": latencies[0],
            "p50_ms": latencies[int(n * 0.50)],
            "p90_ms": latencies[int(n * 0.90)],
            "p95_ms": latencies[int(n * 0.95)],
            "p99_ms": latencies[int(n * 0.99)],
            "max_ms": latencies[-1],
            "mean_ms": statistics.mean(latencies),
            "meets_p50_sla": latencies[int(n * 0.50)] < 200,
            "meets_p99_sla": latencies[int(n * 0.99)] < 500,
        }

    def _build_report(self, results: list[TestResult]) -> EvaluationReport:
        latencies = sorted(r.latency_ms for r in results)
        n = len(latencies)

        def pct(p: float) -> float:
            return latencies[min(int(n * p), n - 1)]

        accuracy = sum(r.correct for r in results) / n if n else 0.0
        p50 = pct(0.50)
        p95 = pct(0.95)
        p99 = pct(0.99)

        # Cost analysis
        routed_costs = [r.test_case and _estimated_cost(r.routed_model_id) for r in results]
        # "always best" baseline = most expensive model
        best_cost = max(_estimated_cost(m) for m in [
            "claude-opus-4-20250514", "gpt-4o", "gemini-2.5-pro"
        ])

        valid_costs = [c for c in routed_costs if c is not None]
        avg_cost = statistics.mean(valid_costs) if valid_costs else 0.0
        cost_ratio = avg_cost / best_cost if best_cost else 1.0

        failures = [r for r in results if not r.correct]

        return EvaluationReport(
            accuracy=accuracy,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            meets_p50_sla=p50 < 200,
            meets_p99_sla=p99 < 500,
            avg_cost_per_request=avg_cost,
            cost_vs_always_best=cost_ratio,
            results=results,
            failures=failures,
        )


def _estimated_cost(model_id: str) -> float:
    """Rough cost for a 500-token input / 200-token output request."""
    from registry.models import get_model
    spec = get_model(model_id)
    if spec is None:
        return 0.0
    return (500 * spec.cost_input_per_1m + 200 * spec.cost_output_per_1m) / 1_000_000


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Running standard test suite...")
    evaluator = RouterEvaluator()
    report = evaluator.run()
    report.print_summary()

    print("\nRunning latency benchmark (1000 iterations)...")
    bench = evaluator.run_latency_benchmark(n_iterations=1000)
    print(json.dumps(bench, indent=2))

    sys.exit(0 if report.accuracy >= 0.75 and report.meets_p99_sla else 1)
