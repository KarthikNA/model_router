"""
Model Registry — single source of truth for all model metadata.
Add new models here; the router picks them up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelType(str, Enum):
    """Broad capability category used during hard-constraint filtering."""
    REASONING = "reasoning"       # o1, o3, DeepSeek-R1 style
    STANDARD = "standard"         # GPT-4o, Claude Sonnet, Gemini Pro
    FAST = "fast"                 # GPT-4o-mini, Claude Haiku, Gemini Flash
    CODE = "code"                 # Codex, DeepSeek-Coder
    MULTIMODAL = "multimodal"     # Vision-capable
    EMBEDDING = "embedding"       # Text embeddings only


class TaskComplexity(str, Enum):
    """Ordered complexity ladder — used to match prompts to model capability ranges."""
    TRIVIAL = "trivial"           # < 50 tokens, simple lookup
    SIMPLE = "simple"             # factual, single-step
    MODERATE = "moderate"         # multi-step, moderate reasoning
    COMPLEX = "complex"           # deep reasoning, long chains
    EXPERT = "expert"             # research-grade, math proofs


@dataclass
class ModelSpec:
    """Static descriptor for a single LLM — cost, latency, capability, and provider metadata."""
    model_id: str
    display_name: str
    model_type: ModelType

    # Context window
    max_input_tokens: int
    max_output_tokens: int

    # Cost (USD per 1M tokens)
    cost_input_per_1m: float
    cost_output_per_1m: float

    # Performance
    avg_tokens_per_second: float      # throughput
    avg_latency_first_token_ms: float # TTFT

    # Capabilities
    supports_vision: bool = False
    supports_function_calling: bool = True
    supports_json_mode: bool = True

    # Specialities — task types this model excels at
    task_strengths: list[str] = field(default_factory=list)

    # Language strengths beyond English (ISO 639-1 codes)
    language_strengths: list[str] = field(default_factory=list)

    # Minimum complexity this model should handle (avoids overkill)
    min_complexity: TaskComplexity = TaskComplexity.TRIVIAL
    # Maximum complexity this model handles well
    max_complexity: TaskComplexity = TaskComplexity.EXPERT

    # Availability / reliability
    availability_sla: float = 0.999   # 99.9%
    is_available: bool = True

    # Provider
    provider: str = "unknown"

    @property
    def cost_per_request_estimate(self) -> float:
        """Rough estimate for a 1k-token round trip."""
        return (1000 * self.cost_input_per_1m + 500 * self.cost_output_per_1m) / 1_000_000

    @property
    def cost_efficiency_score(self) -> float:
        """Higher = cheaper. Normalised to [0,1] across registry."""
        return 1.0 / (self.cost_input_per_1m + 1e-9)


# ---------------------------------------------------------------------------
# Registry — edit / extend freely
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelSpec] = {

    # ── OpenAI ──────────────────────────────────────────────────────────────
    "gpt-4o": ModelSpec(
        model_id="gpt-4o",
        display_name="GPT-4o",
        provider="openai",
        model_type=ModelType.STANDARD,
        max_input_tokens=128_000,
        max_output_tokens=16_384,
        cost_input_per_1m=2.50,
        cost_output_per_1m=10.00,
        avg_tokens_per_second=80,
        avg_latency_first_token_ms=400,
        supports_vision=True,
        task_strengths=["general", "writing", "analysis", "coding", "summarisation"],
        language_strengths=["en", "es", "fr", "de", "zh", "ja"],
        min_complexity=TaskComplexity.SIMPLE,
        max_complexity=TaskComplexity.COMPLEX,
    ),
    "gpt-4o-mini": ModelSpec(
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        provider="openai",
        model_type=ModelType.FAST,
        max_input_tokens=128_000,
        max_output_tokens=16_384,
        cost_input_per_1m=0.15,
        cost_output_per_1m=0.60,
        avg_tokens_per_second=150,
        avg_latency_first_token_ms=200,
        supports_vision=True,
        task_strengths=["general", "summarisation", "classification", "extraction"],
        language_strengths=["en", "es", "fr"],
        min_complexity=TaskComplexity.TRIVIAL,
        max_complexity=TaskComplexity.MODERATE,
    ),
    "o3-mini": ModelSpec(
        model_id="o3-mini",
        display_name="o3-mini",
        provider="openai",
        model_type=ModelType.REASONING,
        max_input_tokens=200_000,
        max_output_tokens=100_000,
        cost_input_per_1m=1.10,
        cost_output_per_1m=4.40,
        avg_tokens_per_second=40,
        avg_latency_first_token_ms=2000,
        supports_vision=False,
        task_strengths=["math", "reasoning", "coding", "logic", "proofs"],
        language_strengths=["en"],
        min_complexity=TaskComplexity.COMPLEX,
        max_complexity=TaskComplexity.EXPERT,
    ),

    # ── Anthropic ────────────────────────────────────────────────────────────
    "claude-3-5-haiku-20241022": ModelSpec(
        model_id="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        provider="anthropic",
        model_type=ModelType.FAST,
        max_input_tokens=200_000,
        max_output_tokens=8_096,
        cost_input_per_1m=0.80,
        cost_output_per_1m=4.00,
        avg_tokens_per_second=200,
        avg_latency_first_token_ms=150,
        task_strengths=["summarisation", "extraction", "classification", "general"],
        language_strengths=["en", "fr", "de", "es"],
        min_complexity=TaskComplexity.TRIVIAL,
        max_complexity=TaskComplexity.MODERATE,
    ),
    "claude-sonnet-4-20250514": ModelSpec(
        model_id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        provider="anthropic",
        model_type=ModelType.STANDARD,
        max_input_tokens=200_000,
        max_output_tokens=64_000,
        cost_input_per_1m=3.00,
        cost_output_per_1m=15.00,
        avg_tokens_per_second=100,
        avg_latency_first_token_ms=350,
        task_strengths=["general", "coding", "analysis", "writing", "agentic"],
        language_strengths=["en", "fr", "de", "es", "zh", "ja", "ko"],
        min_complexity=TaskComplexity.SIMPLE,
        max_complexity=TaskComplexity.EXPERT,
    ),
    "claude-opus-4-20250514": ModelSpec(
        model_id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        provider="anthropic",
        model_type=ModelType.REASONING,
        max_input_tokens=200_000,
        max_output_tokens=32_000,
        cost_input_per_1m=15.00,
        cost_output_per_1m=75.00,
        avg_tokens_per_second=50,
        avg_latency_first_token_ms=800,
        task_strengths=["reasoning", "research", "complex_analysis", "coding", "agentic"],
        language_strengths=["en", "fr", "de", "es", "zh", "ja"],
        min_complexity=TaskComplexity.COMPLEX,
        max_complexity=TaskComplexity.EXPERT,
    ),

    # ── Google ───────────────────────────────────────────────────────────────
    "gemini-2.0-flash": ModelSpec(
        model_id="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        provider="google",
        model_type=ModelType.FAST,
        max_input_tokens=1_000_000,
        max_output_tokens=8_192,
        cost_input_per_1m=0.10,
        cost_output_per_1m=0.40,
        avg_tokens_per_second=250,
        avg_latency_first_token_ms=100,
        supports_vision=True,
        task_strengths=["summarisation", "extraction", "classification", "long_context"],
        language_strengths=["en", "hi", "bn", "te", "ta", "mr", "es", "fr"],
        min_complexity=TaskComplexity.TRIVIAL,
        max_complexity=TaskComplexity.MODERATE,
    ),
    "gemini-2.5-pro": ModelSpec(
        model_id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        provider="google",
        model_type=ModelType.REASONING,
        max_input_tokens=1_000_000,
        max_output_tokens=65_536,
        cost_input_per_1m=1.25,
        cost_output_per_1m=10.00,
        avg_tokens_per_second=60,
        avg_latency_first_token_ms=600,
        supports_vision=True,
        task_strengths=["reasoning", "long_context", "coding", "math", "research"],
        language_strengths=["en", "hi", "bn", "te", "ta", "es", "fr", "de"],
        min_complexity=TaskComplexity.MODERATE,
        max_complexity=TaskComplexity.EXPERT,
    ),

    # ── DeepSeek ─────────────────────────────────────────────────────────────
    "deepseek-chat": ModelSpec(
        model_id="deepseek-chat",
        display_name="DeepSeek V3",
        provider="deepseek",
        model_type=ModelType.STANDARD,
        max_input_tokens=64_000,
        max_output_tokens=8_192,
        cost_input_per_1m=0.27,
        cost_output_per_1m=1.10,
        avg_tokens_per_second=120,
        avg_latency_first_token_ms=300,
        task_strengths=["coding", "general", "analysis"],
        language_strengths=["en", "zh"],
        min_complexity=TaskComplexity.SIMPLE,
        max_complexity=TaskComplexity.COMPLEX,
    ),
    "deepseek-reasoner": ModelSpec(
        model_id="deepseek-reasoner",
        display_name="DeepSeek R1",
        provider="deepseek",
        model_type=ModelType.REASONING,
        max_input_tokens=64_000,
        max_output_tokens=32_000,
        cost_input_per_1m=0.55,
        cost_output_per_1m=2.19,
        avg_tokens_per_second=30,
        avg_latency_first_token_ms=3000,
        task_strengths=["math", "reasoning", "coding", "logic"],
        language_strengths=["en", "zh"],
        min_complexity=TaskComplexity.COMPLEX,
        max_complexity=TaskComplexity.EXPERT,
    ),
}


def get_available_models() -> list[ModelSpec]:
    """Return all models where is_available is True."""
    return [m for m in MODEL_REGISTRY.values() if m.is_available]


def get_model(model_id: str) -> Optional[ModelSpec]:
    """Look up a model by ID; returns None if not in the registry."""
    return MODEL_REGISTRY.get(model_id)
