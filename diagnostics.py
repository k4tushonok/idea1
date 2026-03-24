from __future__ import annotations

import hashlib
from collections import Counter
from statistics import mean as _mean
from typing import List, Optional, TYPE_CHECKING

from data_structures import Metrics
from config import ENABLE_DIAGNOSTIC_LOGS

if TYPE_CHECKING:
    from data_structures import PromptNode


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


def print_section(label: str) -> None:
    """Печатает раздел-разделитель для диагностических блоков."""
    print(f"[diag] {'─' * 8} {label} {'─' * 8}")


def print_timing(label: str, seconds: float) -> None:
    """Стандартизированная запись времени выполнения."""
    print(f"[diag] timing [{label}]: {seconds:.2f}s")


def print_population(label: str, nodes: List) -> None:
    """Краткая сводка по популяции: размер, мин/макс/среднее score."""
    scores = [n.selection_score() for n in nodes]
    if scores:
        print(
            f"[diag] {label}: n={len(nodes)} "
            f"min={min(scores):.3f} max={max(scores):.3f} "
            f"mean={_mean(scores):.3f}"
        )
    else:
        print(f"[diag] {label}: n=0")


def sources_summary(nodes: List) -> str:
    """Строка с количеством узлов по источникам."""
    cnt = Counter(n.source.value for n in nodes)
    return ", ".join(f"{src}={c}" for src, c in sorted(cnt.items()))


def scores_summary(values: List[float], precision: int = 3) -> str:
    """Форматирует список score-значений в сжатую строку."""
    if not values:
        return "[]"
    fmt = f".{precision}f"
    return "[" + ", ".join(format(v, fmt) for v in values) + "]"


def print_candidates_summary(label: str, nodes: List) -> None:
    """Сводка по набору кандидатов с сортировкой по убыванию score."""
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
