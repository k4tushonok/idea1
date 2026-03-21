from __future__ import annotations

import hashlib
from typing import Optional

from data_structures import Metrics
from config import ENABLE_DIAGNOSTIC_LOGS


def is_enabled() -> bool:
    return bool(ENABLE_DIAGNOSTIC_LOGS)


def prompt_id(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]


def preview_text(text: Optional[str], max_chars: int = None) -> str:
    if text is None:
        return "<none>"
    value = text.strip()
    limit = 4000 if max_chars is None else max_chars
    if limit <= 0 or len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def print_prompt(label: str, prompt: str) -> None:
    print(f"[diag] {label} text: {prompt}")


def metrics_line(metrics: Metrics) -> str:
    ordered = sorted(metrics.metrics.items(), key=lambda kv: kv[0])
    body = ", ".join(f"{name}={value:.3f}" for name, value in ordered)
    return f"{body}, composite={metrics.composite_score():.3f}"


def print_metrics(label: str, metrics: Metrics) -> None:
    print(f"[diag] {label} metrics: {metrics_line(metrics)}")
