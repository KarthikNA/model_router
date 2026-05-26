#!/usr/bin/env python3
"""
Quick demo — run from the model_router/ directory:
    python demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from router.router import ModelRouter, RouterConfig
from router.scorer import ScoringWeights
from evaluation.evaluator import RouterEvaluator


DEMO_PROMPTS = [
    ("Simple factual", "What is the boiling point of water?"),
    ("Code generation", "Write a Python class implementing a min-heap."),
    ("Math proof",     "Prove by induction that the sum of first n integers equals n(n+1)/2."),
    ("French query",   "Résumez les principes de la thermodynamique."),
    ("Chinese query",  "用简单的语言解释机器学习。"),
    ("Long context",   "Summarise the following: " + "text " * 10_000),
    ("Agentic task",   "You are an agent with web search tools. Research and compare GPT-4o vs Claude Sonnet 4 capabilities for enterprise use."),
]


def main():
    print("=" * 65)
    print("MODEL ROUTER — DEMO")
    print("=" * 65)

    router = ModelRouter(RouterConfig(
        weights=ScoringWeights(cost=0.40, throughput=0.25, relevance=0.25, capability=0.10),
        top_k=3,
        enable_logging=False,
    ))

    for label, prompt in DEMO_PROMPTS:
        result = router.route(prompt)
        short_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
        print(f"\n[{label}]")
        print(f"  Prompt   : {short_prompt}")
        print(f"  → Model  : {result.model_id} (score={result.score:.3f})")
        print(f"  Latency  : {result.latency_ms:.2f} ms")
        print(f"  Complexity: {result.features.complexity.value} | Tasks: {result.features.task_types[:2]} | Lang: {result.features.language}")
        if result.top_candidates:
            runners_up = [c.model.model_id for c in result.top_candidates[1:3]]
            print(f"  Runners-up: {runners_up}")

    # Evaluation
    print("\n" + "=" * 65)
    print("RUNNING EVALUATION SUITE")
    print("=" * 65)
    evaluator = RouterEvaluator(router)
    report = evaluator.run()
    report.print_summary()

    # Latency benchmark
    print("\nLATENCY BENCHMARK (500 iterations)")
    bench = evaluator.run_latency_benchmark(n_iterations=500)
    print(f"  p50: {bench['p50_ms']:.2f} ms  {'✓' if bench['meets_p50_sla'] else '✗'}")
    print(f"  p95: {bench['p95_ms']:.2f} ms")
    print(f"  p99: {bench['p99_ms']:.2f} ms  {'✓' if bench['meets_p99_sla'] else '✗'}")
    print(f"  mean: {bench['mean_ms']:.2f} ms")


if __name__ == "__main__":
    main()
