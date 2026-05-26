"""
Feature Extractor — derives routing signals from raw prompt text.
All operations are local (no LLM call) to stay within latency budget.
"""

from __future__ import annotations

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) in ("evaluation", "router", "registry") else _HERE
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import re
import time
from dataclasses import dataclass
from functools import lru_cache

from registry.models import TaskComplexity


# ---------------------------------------------------------------------------
# Language detection (lightweight — no external lib required)
# ---------------------------------------------------------------------------

# Unicode range → ISO 639-1
_SCRIPT_RANGES: list[tuple[tuple[int, int], str]] = [
    ((0x4E00, 0x9FFF), "zh"),    # CJK Unified Ideographs
    ((0x3040, 0x309F), "ja"),    # Hiragana
    ((0x30A0, 0x30FF), "ja"),    # Katakana
    ((0xAC00, 0xD7AF), "ko"),    # Hangul
    ((0x0900, 0x097F), "hi"),    # Devanagari
    ((0x0980, 0x09FF), "bn"),    # Bengali
    ((0x0C00, 0x0C7F), "te"),    # Telugu
    ((0x0B80, 0x0BFF), "ta"),    # Tamil
    ((0x0600, 0x06FF), "ar"),    # Arabic
    ((0x0400, 0x04FF), "ru"),    # Cyrillic
    ((0x0370, 0x03FF), "el"),    # Greek
    ((0x0E00, 0x0E7F), "th"),    # Thai
    ((0x0080, 0x00FF), "es"),    # Latin-1 Supplement (covers accented latin)
]

# Simple keyword sets per language (supplement script detection)
_LANG_KEYWORDS: dict[str, list[str]] = {
    "fr": ["le ", "la ", "les ", "de ", "du ", "je ", "vous ", "nous ", "est ", "une "],
    "es": ["el ", "la ", "los ", "las ", "que ", "de ", "en ", "una ", "es ", "por "],
    "de": ["der ", "die ", "das ", "und ", "ich ", "sie ", "ein ", "ist ", "mit ", "für "],
    "pt": ["o ", "a ", "os ", "as ", "que ", "de ", "em ", "um ", "uma ", "não "],
}


def detect_language(text: str) -> str:
    """Best-effort language detection. Returns ISO 639-1 code, default 'en'."""
    sample = text[:500]

    # Script-based detection (fast O(n))
    script_counts: dict[str, int] = {}
    for ch in sample:
        cp = ord(ch)
        for (lo, hi), lang in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                script_counts[lang] = script_counts.get(lang, 0) + 1
                break

    if script_counts:
        dominant = max(script_counts, key=lambda k: script_counts[k])
        if script_counts[dominant] > 5:
            return dominant

    # Keyword-based for Latin scripts
    lower = sample.lower()
    for lang, keywords in _LANG_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits >= 3:
            return lang

    return "en"


# ---------------------------------------------------------------------------
# Task-type signals
# ---------------------------------------------------------------------------

_TASK_PATTERNS: dict[str, list[str]] = {
    "coding": [
        r"\bcode\b", r"\bfunction\b", r"\bclass\b", r"\bdebugg?\b",
        r"\bimport\b", r"\bdef\b", r"\bsql\b", r"\bapi\b",
        r"```", r"\brefactor\b", r"\bunit test\b",
    ],
    "math": [
        r"\bsolve\b", r"\bequation\b", r"\bcalculate?\b", r"\bproof\b",
        r"\bderive\b", r"\bintegral\b", r"\bderivative\b", r"\bmatrix\b",
        r"[=\+\-\*/\^]{2,}", r"\bformula\b",
    ],
    "reasoning": [
        r"\breason\b", r"\banalyse?\b", r"\bcompare\b", r"\bexplain why\b",
        r"\bstep.by.step\b", r"\bthink through\b", r"\bpros and cons\b",
        r"\bdecide\b", r"\bstrateg\b", r"\btradeoff\b",
    ],
    "summarisation": [
        r"\bsummar\b", r"\btldr\b", r"\bbrief\b", r"\bcondense\b",
        r"\bkey points\b", r"\bmain idea\b",
    ],
    "extraction": [
        r"\bextract\b", r"\blist all\b", r"\bfind all\b", r"\bidentify\b",
        r"\bpull out\b", r"\bparse\b",
    ],
    "writing": [
        r"\bwrite\b", r"\bdraft\b", r"\bcompose\b", r"\bessay\b",
        r"\bblog\b", r"\bemail\b", r"\bletter\b", r"\bstory\b",
    ],
    "classification": [
        r"\bclassif\b", r"\bcategor\b", r"\blabel\b", r"\bsentiment\b",
        r"\bdetect\b",
    ],
    "long_context": [],  # set dynamically based on token count
    "agentic": [
        r"\bagent\b", r"\bautonomous\b", r"\bmulti.step\b", r"\bworkflow\b",
        r"\btool\b", r"\bfunction call\b",
    ],
    "general": [],  # fallback
}

# Compiled once at module load
_COMPILED_PATTERNS: dict[str, list[re.Pattern]] = {
    task: [re.compile(p, re.IGNORECASE) for p in patterns]
    for task, patterns in _TASK_PATTERNS.items()
    if patterns  # skip empty lists
}


def detect_task_types(text: str) -> list[str]:
    """Returns ranked list of task types detected in the prompt."""
    scores: dict[str, int] = {}
    for task, patterns in _COMPILED_PATTERNS.items():
        score = sum(1 for p in patterns if p.search(text))
        if score > 0:
            scores[task] = score

    # Always include "general" as fallback
    if not scores:
        return ["general"]

    return sorted(scores, key=lambda k: scores[k], reverse=True)


# ---------------------------------------------------------------------------
# Complexity estimation
# ---------------------------------------------------------------------------

_COMPLEXITY_SIGNALS = {
    TaskComplexity.EXPERT: [
        r"\bproof\b", r"\bprove\b", r"\bby induction\b", r"\blemma\b",
        r"\btheorem\b", r"\bresearch\b", r"\bnovel\b", r"\boriginal\b",
        r"\bhypothes\b", r"\bdissertation\b", r"\bacademic\b",
        r"\boptimize across\b", r"\btrade.?off analysis\b",
        r"\bformally\b", r"\bmathematical(ly)?\b",
    ],
    TaskComplexity.COMPLEX: [
        r"\bcomplex\b", r"\badvanced\b", r"\bmulti.step\b", r"\bchain\b",
        r"\barchitect\b", r"\bdesign\b", r"\banalyze\b",
        r"\bcomprehensive\b", r"\bin.depth\b", r"\btrade.?off\b",
        r"\bfailure mode\b", r"\bscalabilit\b", r"\bdistributed\b",
        r"\bstep.by.step\b", r"\bsystem design\b",
    ],
    TaskComplexity.MODERATE: [
        r"\bexplain\b", r"\bcompare\b", r"\bdifference\b", r"\bhow does\b",
        r"\bwhy\b", r"\bwhen should\b", r"\bpros\b", r"\bcons\b",
        r"\bimplement\b", r"\bwrite a\b", r"\bcreate a\b",
    ],
    TaskComplexity.SIMPLE: [
        r"\bwhat is\b", r"\bdefine\b", r"\blist\b", r"\bexample\b",
        r"\bsummariz\b",
    ],
    TaskComplexity.TRIVIAL: [
        r"^hi[\s!.]*$", r"^hello[\s!.]*$", r"^thanks?[\s!.]*$", r"\bping\b",
    ],
}

_COMPILED_COMPLEXITY: dict[TaskComplexity, list[re.Pattern]] = {
    level: [re.compile(p, re.IGNORECASE) for p in patterns]
    for level, patterns in _COMPLEXITY_SIGNALS.items()
}

_COMPLEXITY_ORDER = [
    TaskComplexity.TRIVIAL,
    TaskComplexity.SIMPLE,
    TaskComplexity.MODERATE,
    TaskComplexity.COMPLEX,
    TaskComplexity.EXPERT,
]


def estimate_complexity(text: str, token_count: int) -> TaskComplexity:
    """Heuristic complexity estimation."""
    # Length-based floor
    if token_count > 4000:
        length_floor = TaskComplexity.MODERATE
    elif token_count > 1000:
        length_floor = TaskComplexity.SIMPLE
    else:
        length_floor = TaskComplexity.TRIVIAL

    # Pattern-based
    pattern_complexity = TaskComplexity.TRIVIAL
    for level in reversed(_COMPLEXITY_ORDER):
        patterns = _COMPILED_COMPLEXITY[level]
        if any(p.search(text) for p in patterns):
            pattern_complexity = level
            break

    # Return the higher of the two
    idx_length = _COMPLEXITY_ORDER.index(length_floor)
    idx_pattern = _COMPLEXITY_ORDER.index(pattern_complexity)
    return _COMPLEXITY_ORDER[max(idx_length, idx_pattern)]


# ---------------------------------------------------------------------------
# Token estimation (no tokeniser needed — close enough for routing)
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """GPT-style rough estimate: ~4 chars per token for English."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class PromptFeatures:
    """All routing signals derived from a raw prompt, computed without any LLM call."""
    raw_text: str
    token_count: int
    language: str
    task_types: list[str]          # ordered, most likely first
    complexity: TaskComplexity
    has_code_block: bool
    has_image_reference: bool
    extraction_time_ms: float


def extract_features(prompt: str) -> PromptFeatures:
    """Main entry-point. Runs in < 5 ms for typical prompts."""
    t0 = time.monotonic()

    token_count = estimate_tokens(prompt)
    language = detect_language(prompt)
    task_types = detect_task_types(prompt)
    complexity = estimate_complexity(prompt, token_count)
    has_code_block = "```" in prompt or "<code>" in prompt.lower()
    has_image_reference = bool(
        re.search(r"\b(image|photo|picture|screenshot|diagram|figure)\b", prompt, re.I)
    )

    # Long-context signal
    if token_count > 50_000 and "long_context" not in task_types:
        task_types.insert(0, "long_context")

    elapsed_ms = (time.monotonic() - t0) * 1000

    return PromptFeatures(
        raw_text=prompt,
        token_count=token_count,
        language=language,
        task_types=task_types,
        complexity=complexity,
        has_code_block=has_code_block,
        has_image_reference=has_image_reference,
        extraction_time_ms=elapsed_ms,
    )
