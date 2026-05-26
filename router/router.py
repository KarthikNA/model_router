"""
ModelRouter — public API for the model routing system.

Usage:
    from router.router import ModelRouter, RouterConfig

    router = ModelRouter()
    result = router.route("Explain the proof of Fermat's Last Theorem")
    print(result.model_id)          # e.g. "o3-mini"
    print(result.explanation)       # human-readable reasoning
    print(result.latency_ms)        # routing latency
"""

from __future__ import annotations

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) in ("evaluation", "router", "registry") else _HERE
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import time
import logging
from dataclasses import dataclass, field

from registry.models import ModelSpec, get_model
from router.features import PromptFeatures, extract_features
from router.scorer import ModelScore, ScoringWeights, score_models

logger = logging.getLogger(__name__)


@dataclass
class RouterConfig:
    """Tune routing behaviour without touching code."""

    # Scoring weights — must sum to 1.0
    weights: ScoringWeights = field(default_factory=ScoringWeights)

    # Force specific model IDs regardless of scoring (e.g. for A/B test)
    force_model_id: str | None = None

    # Fallback model if no eligible candidate found
    fallback_model_id: str = "gpt-4o-mini"

    # Minimum score to accept a model (below this → use fallback)
    min_acceptable_score: float = 0.05

    # Return top-N candidates in result (for observability)
    top_k: int = 3

    # Log routing decisions
    enable_logging: bool = True


@dataclass
class RouterResult:
    model_id: str
    model_spec: ModelSpec
    score: float

    # Context
    features: PromptFeatures
    top_candidates: list[ModelScore]

    # Observability
    latency_ms: float
    explanation: str
    is_fallback: bool = False
    forced: bool = False


class ModelRouter:
    """
    Thread-safe, stateless model router.

    The router is intentionally stateless — all state lives in RouterConfig
    and the model registry. This makes it safe to share across threads/coroutines.
    """

    def __init__(self, config: RouterConfig | None = None):
        self.config = config or RouterConfig()

    def route(self, prompt: str) -> RouterResult:
        """
        Route a prompt to the best model.

        Args:
            prompt: Raw prompt text from the user / agent.

        Returns:
            RouterResult with model_id, explanation, and diagnostics.

        Raises:
            ValueError: If no eligible model found and no fallback configured.
        """
        t0 = time.monotonic()

        # ── Forced routing (for testing / overrides) ────────────────────────
        if self.config.force_model_id:
            spec = get_model(self.config.force_model_id)
            if spec is None:
                raise ValueError(f"Forced model '{self.config.force_model_id}' not in registry")
            features = extract_features(prompt)
            latency = (time.monotonic() - t0) * 1000
            return RouterResult(
                model_id=spec.model_id,
                model_spec=spec,
                score=1.0,
                features=features,
                top_candidates=[],
                latency_ms=latency,
                explanation=f"Forced routing to {spec.model_id}",
                forced=True,
            )

        # ── Feature extraction ───────────────────────────────────────────────
        features = extract_features(prompt)

        # ── Scoring ──────────────────────────────────────────────────────────
        scored = score_models(features, weights=self.config.weights)

        eligible = [s for s in scored if not s.eliminated]
        top_k = eligible[: self.config.top_k]

        # ── Selection ────────────────────────────────────────────────────────
        is_fallback = False
        selected: ModelScore | None = None

        if eligible and eligible[0].total_score >= self.config.min_acceptable_score:
            selected = eligible[0]
        else:
            # Fallback
            is_fallback = True
            fallback_spec = get_model(self.config.fallback_model_id)
            if fallback_spec is None:
                raise ValueError(
                    f"Fallback model '{self.config.fallback_model_id}' not in registry"
                )
            # Create a synthetic score entry
            selected = ModelScore(
                model=fallback_spec,
                total_score=0.0,
                cost_score=0.0,
                throughput_score=0.0,
                relevance_score=0.0,
                capability_score=0.0,
            )

        latency_ms = (time.monotonic() - t0) * 1000

        explanation = _build_explanation(selected, features, is_fallback)

        if self.config.enable_logging:
            logger.info(
                "route decision | model=%s score=%.3f latency_ms=%.1f "
                "complexity=%s task=%s lang=%s tokens=%d fallback=%s",
                selected.model.model_id,
                selected.total_score,
                latency_ms,
                features.complexity.value,
                features.task_types[0] if features.task_types else "unknown",
                features.language,
                features.token_count,
                is_fallback,
            )

        return RouterResult(
            model_id=selected.model.model_id,
            model_spec=selected.model,
            score=selected.total_score,
            features=features,
            top_candidates=top_k,
            latency_ms=latency_ms,
            explanation=explanation,
            is_fallback=is_fallback,
        )

    def route_batch(self, prompts: list[str]) -> list[RouterResult]:
        """Route multiple prompts. Each is independent."""
        return [self.route(p) for p in prompts]


# ---------------------------------------------------------------------------
# Explanation builder
# ---------------------------------------------------------------------------

def _build_explanation(
    selected: ModelScore,
    features: PromptFeatures,
    is_fallback: bool,
) -> str:
    model = selected.model
    parts = [f"Selected: {model.display_name} ({model.model_id})"]

    if is_fallback:
        parts.append("⚠ Using fallback — no model passed eligibility threshold")
        return " | ".join(parts)

    parts.append(f"Score: {selected.total_score:.3f}")
    parts.append(
        f"Signals: complexity={features.complexity.value}, "
        f"task={features.task_types[0] if features.task_types else 'general'}, "
        f"lang={features.language}, tokens≈{features.token_count}"
    )
    parts.append(
        f"Scores breakdown → cost={selected.cost_score:.2f} "
        f"throughput={selected.throughput_score:.2f} "
        f"relevance={selected.relevance_score:.2f} "
        f"capability={selected.capability_score:.2f}"
    )
    parts.append(
        f"Model: type={model.model_type.value}, "
        f"cost=${model.cost_input_per_1m}/1M in, "
        f"{model.avg_tokens_per_second} tok/s"
    )
    return "\n".join(parts)
