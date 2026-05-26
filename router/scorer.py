"""
Scoring Engine — multi-objective scoring for model selection.

Objectives (configurable weights):
  1. cost          — minimise cost per request
  2. throughput    — maximise tokens/second
  3. relevance     — task-type and language match
  4. capability    — can the model actually handle this prompt?

Hard constraints (eliminate before scoring):
  - context window must fit input tokens
  - complexity must be within model's range
  - language must be supported (soft — degrades score if not)
  - model must be available
"""

from __future__ import annotations

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) in ("evaluation", "router", "registry") else _HERE
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataclasses import dataclass
import math

from registry.models import ModelSpec, TaskComplexity, get_available_models
from router.features import PromptFeatures

_COMPLEXITY_ORDER = [
    TaskComplexity.TRIVIAL,
    TaskComplexity.SIMPLE,
    TaskComplexity.MODERATE,
    TaskComplexity.COMPLEX,
    TaskComplexity.EXPERT,
]


@dataclass
class ScoringWeights:
    cost: float = 0.30          # Optimise for cost
    throughput: float = 0.20    # Optimise for speed
    relevance: float = 0.35     # Task / language match (primary differentiator)
    capability: float = 0.15    # Headroom / appropriateness

    def validate(self) -> None:
        total = self.cost + self.throughput + self.relevance + self.capability
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")


@dataclass
class ModelScore:
    model: ModelSpec
    total_score: float
    cost_score: float
    throughput_score: float
    relevance_score: float
    capability_score: float
    eliminated: bool = False
    elimination_reason: str = ""

    def __repr__(self) -> str:
        if self.eliminated:
            return f"<{self.model.model_id} ELIMINATED: {self.elimination_reason}>"
        return (
            f"<{self.model.model_id} total={self.total_score:.3f} "
            f"cost={self.cost_score:.3f} tput={self.throughput_score:.3f} "
            f"rel={self.relevance_score:.3f} cap={self.capability_score:.3f}>"
        )


def _complexity_index(c: TaskComplexity) -> int:
    return _COMPLEXITY_ORDER.index(c)


# ---------------------------------------------------------------------------
# Individual scoring functions (each returns [0, 1])
# ---------------------------------------------------------------------------

def _score_cost(model: ModelSpec, all_models: list[ModelSpec]) -> float:
    """Lower cost → higher score. Log-scaled to avoid extreme outliers dominating."""
    avg_cost = model.cost_input_per_1m + model.cost_output_per_1m * 0.5
    # Log scale: map [0.01, 200] → roughly [0, 1]
    # score = 1 - normalised_log_cost
    log_cost = math.log1p(avg_cost)
    max_log = math.log1p(max(
        m.cost_input_per_1m + m.cost_output_per_1m * 0.5 for m in all_models
    ))
    return 1.0 - (log_cost / max_log)


def _score_throughput(model: ModelSpec, all_models: list[ModelSpec]) -> float:
    """Higher tokens/second → higher score."""
    max_tps = max(m.avg_tokens_per_second for m in all_models)
    return model.avg_tokens_per_second / max_tps


def _score_relevance(model: ModelSpec, features: PromptFeatures) -> float:
    """How well does the model's strengths match detected task types and language?"""
    task_score = 0.0
    if features.task_types:
        primary_task = features.task_types[0]
        if primary_task in model.task_strengths:
            task_score = 1.0
        elif any(t in model.task_strengths for t in features.task_types[:3]):
            task_score = 0.6
        elif "general" in model.task_strengths:
            task_score = 0.3
        else:
            task_score = 0.1
    else:
        task_score = 0.5  # no signal

    # Language bonus
    lang_score = 1.0 if features.language in model.language_strengths else 0.5
    if features.language == "en":
        lang_score = 1.0  # All models support English

    return 0.7 * task_score + 0.3 * lang_score


def _score_capability(model: ModelSpec, features: PromptFeatures) -> float:
    """
    Penalise overkill (expensive model for trivial task) and underkill
    (weak model for complex task). Sweet-spot scoring.
    """
    req_idx = _complexity_index(features.complexity)
    min_idx = _complexity_index(model.min_complexity)
    max_idx = _complexity_index(model.max_complexity)

    if req_idx < min_idx:
        # Overkill — penalise hard: expensive model on easy task
        gap = min_idx - req_idx
        return max(0.0, 1.0 - gap * 0.5)

    if req_idx > max_idx:
        # Underkill — penalise sharply; no stretch allowed
        gap = req_idx - max_idx
        return max(0.0, 0.4 - gap * 0.3)

    # Within range — full score
    return 1.0


# ---------------------------------------------------------------------------
# Hard constraint filtering
# ---------------------------------------------------------------------------

def _is_eligible(model: ModelSpec, features: PromptFeatures) -> tuple[bool, str]:
    if not model.is_available:
        return False, "model unavailable"

    if features.token_count > model.max_input_tokens:
        return False, (
            f"context overflow: {features.token_count} tokens > "
            f"{model.max_input_tokens} max"
        )

    req_idx = _complexity_index(features.complexity)
    max_idx = _complexity_index(model.max_complexity)
    if req_idx > max_idx:
        return False, (
            f"complexity mismatch: {features.complexity} > {model.max_complexity}"
        )

    if features.has_image_reference and not model.supports_vision:
        # Soft: don't hard-eliminate, but caller can downrank
        pass

    return True, ""


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_models(
    features: PromptFeatures,
    weights: ScoringWeights | None = None,
    candidate_models: list[ModelSpec] | None = None,
) -> list[ModelScore]:
    """
    Score all available models against extracted prompt features.
    Returns list sorted by total_score descending.
    """
    if weights is None:
        weights = ScoringWeights()
    weights.validate()

    models = candidate_models or get_available_models()

    results: list[ModelScore] = []

    for model in models:
        eligible, reason = _is_eligible(model, features)

        if not eligible:
            results.append(ModelScore(
                model=model,
                total_score=0.0,
                cost_score=0.0,
                throughput_score=0.0,
                relevance_score=0.0,
                capability_score=0.0,
                eliminated=True,
                elimination_reason=reason,
            ))
            continue

        cost_s = _score_cost(model, models)
        tput_s = _score_throughput(model, models)
        rel_s = _score_relevance(model, features)
        cap_s = _score_capability(model, features)

        total = (
            weights.cost * cost_s
            + weights.throughput * tput_s
            + weights.relevance * rel_s
            + weights.capability * cap_s
        )

        results.append(ModelScore(
            model=model,
            total_score=total,
            cost_score=cost_s,
            throughput_score=tput_s,
            relevance_score=rel_s,
            capability_score=cap_s,
        ))

    # Sort: eligible first, then by score desc
    results.sort(key=lambda r: (not r.eliminated, r.total_score), reverse=True)
    return results
