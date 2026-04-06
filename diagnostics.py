"""
Diagnostic module for debugging and monitoring the optimization process.

Provides logging utilities: prompt hashing, LLM call counting, metric
formatting, and population/candidate summaries.
Activated by the ENABLE_DIAGNOSTIC_LOGS flag in config.
"""

from __future__ import annotations

import hashlib
from statistics import mean as _mean
from typing import List, Optional, TYPE_CHECKING

from data_structures import Metrics
from config import ENABLE_DIAGNOSTIC_LOGS

if TYPE_CHECKING:
    from data_structures import PromptNode


def is_enabled() -> bool:
    """Return True if diagnostic logging is enabled."""
    return bool(ENABLE_DIAGNOSTIC_LOGS)


def prompt_id(prompt: str) -> str:
    """Short SHA-1 hash of a prompt string for identification in logs."""
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]


def llm_calls(llm) -> int:
    """Return the total number of API calls made by the LLM client."""
    return getattr(llm, "total_api_calls", 0)


def format_stage_weights(weights: dict) -> str:
    """Format a stage-weights dictionary as a compact string."""
    return "{" + ", ".join(f"{k}:{v}" for k, v in sorted(weights.items())) + "}"


def preview_text(text: Optional[str], max_chars: int = None) -> str:
    """Truncate text to max_chars characters for log previews."""
    if text is None:
        return "<none>"
    value = text.strip()
    limit = 4000 if max_chars is None else max_chars
    if limit <= 0 or len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def print_prompt(label: str, prompt: str) -> None:
    """Log a prompt text."""
    print(f"[diag] {label} text: {prompt}")


def print_timing(label: str, seconds: float) -> None:
    """Log a standardized timing entry."""
    print(f"[diag] timing [{label}]: {seconds:.2f}s")


def print_population(label: str, nodes: List) -> None:
    """Log a brief population summary: size, min/max/mean score."""
    scores = [n.selection_score() for n in nodes]
    if scores:
        print(
            f"[diag] {label}: n={len(nodes)} "
            f"min={min(scores):.3f} max={max(scores):.3f} "
            f"mean={_mean(scores):.3f}"
        )
    else:
        print(f"[diag] {label}: n=0")


def scores_summary(values: List[float], precision: int = 3) -> str:
    """Format a list of score values as a compact string."""
    if not values:
        return "[]"
    fmt = f".{precision}f"
    return "[" + ", ".join(format(v, fmt) for v in values) + "]"


def print_candidates_summary(label: str, nodes: List) -> None:
    """Log a summary of candidate nodes sorted by descending score."""
    if not nodes:
        print(f"[diag] {label}: (empty)")
        return
    sorted_nodes = sorted(nodes, key=lambda n: n.selection_score(), reverse=True)
    scores = [n.selection_score() for n in sorted_nodes]
    print(
        f"[diag] {label}: n={len(nodes)} "
        f"best={scores[0]:.3f} worst={scores[-1]:.3f} "
        f"scores={scores_summary(scores)}"
    )


def print_eval_summary(
    label: str,
    metrics: "Metrics",
    stage: int,
    llm_calls_before: int,
    llm_calls_after: int,
    elapsed_s: float,
) -> None:
    """Log a final evaluation summary."""
    api_delta = llm_calls_after - llm_calls_before
    per_metric = ", ".join(f"{n}={v:.3f}" for n, v in sorted(metrics.metrics.items()))
    print(
        f"[diag] eval [{label}]: stage={stage} composite={metrics.composite_score():.4f} "
        f"metrics=[{per_metric}] llm_calls={api_delta} time={elapsed_s:.2f}s"
    )
