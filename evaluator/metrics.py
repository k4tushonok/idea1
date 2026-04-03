from typing import List, Dict
from abc import ABC, abstractmethod
from collections import Counter
import re
import string

from nltk.translate.meteor_score import meteor_score as _nltk_meteor

from data_structures import Example


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> List[str]:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


_NO_ANSWER_VARIANTS: List[str] = ["", "no answer", "no answer."]


def _get_gold_answers(ex: Example) -> List[str]:
    all_ans = ex.metadata.get("all_answers") if ex.metadata else None
    if all_ans is not None:
        # Filter out empty-after-normalization, keep unique
        filtered = [a for a in all_ans if normalize_answer(a)]
        if not filtered:
            return _NO_ANSWER_VARIANTS
        return filtered
    # Fallback: single expected_output
    if ex.expected_output is None:
        return _NO_ANSWER_VARIANTS
    if ex.expected_output.strip().lower() in ("no answer", "no answer.", ""):
        return _NO_ANSWER_VARIANTS
    return [ex.expected_output]


class MetricEvaluator(ABC):
    """Базовый класс метрики"""

    name: str

    def __init__(self):
        pass

    def supports_examples(self, examples: List[Example]) -> bool:
        return True

    @abstractmethod
    def evaluate(self, prompt: str, examples: List[Example]) -> float:
        """Возвращает оценку промпта от 0.0 до 1.0"""
        pass


class CheapMetric(MetricEvaluator):
    """Метрика, работающая без вызовов LLM — чистое текстовое сравнение."""

    pass


class ExactMatchMetric(CheapMetric):
    name = "exact_match"

    def evaluate(self, prompt: str, examples: List[Example]) -> float:
        if not examples:
            return 0.0
        scores = []
        for ex in examples:
            if ex.actual_output is None:
                scores.append(0.0)
                continue
            gold_answers = _get_gold_answers(ex)
            em = max(compute_exact(gold, ex.actual_output) for gold in gold_answers)
            scores.append(float(em))
        return sum(scores) / len(scores) if scores else 0.0


class TokenF1Metric(CheapMetric):
    name = "token_f1"

    def evaluate(self, prompt: str, examples: List[Example]) -> float:
        if not examples:
            return 0.0
        scores = []
        for ex in examples:
            if ex.actual_output is None:
                scores.append(0.0)
                continue
            gold_answers = _get_gold_answers(ex)
            f1 = max(compute_f1(gold, ex.actual_output) for gold in gold_answers)
            scores.append(f1)
        return sum(scores) / len(scores) if scores else 0.0


class RougeLMetric(CheapMetric):
    name = "rouge_l"

    def supports_examples(self, examples: List[Example]) -> bool:
        return any(
            (ex.metadata and ex.metadata.get("references")) or ex.expected_output
            for ex in examples
        )

    def evaluate(self, prompt: str, examples: List[Example]) -> float:
        if not examples:
            return 0.0

        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        scores = []
        for ex in examples:
            if ex.actual_output is None:
                scores.append(0.0)
                continue
            references = [ex.expected_output]
            if ex.metadata and "references" in ex.metadata:
                references = ex.metadata["references"]
            refs = [r for r in references if r]
            if not refs:
                scores.append(0.0)
                continue
            best = max(
                scorer.score(ref, ex.actual_output)["rougeL"].fmeasure for ref in refs
            )
            scores.append(best)
        return sum(scores) / len(scores) if scores else 0.0


class MeteorMetric(CheapMetric):
    name = "meteor"

    def supports_examples(self, examples: List[Example]) -> bool:
        return any(
            (ex.metadata and ex.metadata.get("references")) or ex.expected_output
            for ex in examples
        )

    def evaluate(self, prompt: str, examples: List[Example]) -> float:
        if not examples:
            return 0.0
        scores = []
        for ex in examples:
            if ex.actual_output is None:
                scores.append(0.0)
                continue
            references = [ex.expected_output]
            if ex.metadata and "references" in ex.metadata:
                references = ex.metadata["references"]
            refs = [r for r in references if r]
            if not refs:
                scores.append(0.0)
                continue
            hyp_tok = ex.actual_output.strip().lower().split()
            if not hyp_tok:
                scores.append(0.0)
                continue
            best = max(_nltk_meteor([r.lower().split()], hyp_tok) for r in refs)
            scores.append(best)
        return sum(scores) / len(scores) if scores else 0.0


def _extract_final_number(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    # 1. #### marker
    m = re.search(r"####\s*(.+)", text)
    if m:
        return m.group(1).strip().replace(",", "")
    # 2. "The answer is <number>" / "answer: <number>" patterns
    m = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*\$?\s*(-?[\d,]+\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).replace(",", "")
    # 3. "= <number>" at the end of a line
    m = re.search(r"=\s*\$?\s*(-?[\d,]+\.?\d*)\s*$", text, re.MULTILINE)
    if m:
        return m.group(1).replace(",", "")
    # 4. Last number in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return normalize_answer(text)


class NumericExactMatchMetric(CheapMetric):
    name = "numeric_exact_match"

    def evaluate(self, prompt: str, examples: List[Example]) -> float:
        if not examples:
            return 0.0
        scores = []
        for ex in examples:
            if ex.actual_output is None:
                scores.append(0.0)
                continue
            gold_answers = _get_gold_answers(ex)
            pred = _extract_final_number(ex.actual_output)
            em = max(
                int(
                    normalize_answer(_extract_final_number(g)) == normalize_answer(pred)
                )
                for g in gold_answers
            )
            scores.append(float(em))
        return sum(scores) / len(scores) if scores else 0.0


class BertScoreMetric(CheapMetric):
    """BERTScore F1 — model-based semantic similarity metric.
    Uses bert_score library with a lightweight default model.
    """

    name = "bertscore"

    _model_type: str = "roberta-large"

    def supports_examples(self, examples: List[Example]) -> bool:
        return any(
            (ex.metadata and ex.metadata.get("references")) or ex.expected_output
            for ex in examples
        )

    def evaluate(self, prompt: str, examples: List[Example]) -> float:
        if not examples:
            return 0.0

        from bert_score import score as bert_score_fn  # lazy import

        preds: List[str] = []
        refs: List[str] = []
        group_ids: List[int] = []

        for i, ex in enumerate(examples):
            if ex.actual_output is None:
                continue
            references = [ex.expected_output] if ex.expected_output else []
            if ex.metadata and "references" in ex.metadata:
                references = ex.metadata["references"]
            references = [r for r in references if r]
            if not references:
                continue
            for ref in references:
                preds.append(ex.actual_output.strip())
                refs.append(ref.strip())
                group_ids.append(i)

        if not preds:
            return 0.0

        _P, _R, F1 = bert_score_fn(
            preds,
            refs,
            model_type=self._model_type,
            verbose=False,
        )
        f1_list = F1.tolist()

        # Take max F1 per example across all references
        all_scores = [0.0] * len(examples)
        for gid, f1_val in zip(group_ids, f1_list):
            clamped = max(0.0, min(1.0, f1_val))
            if clamped > all_scores[gid]:
                all_scores[gid] = clamped

        return sum(all_scores) / len(examples)


METRIC_REGISTRY: Dict[str, type] = {
    "exact_match": ExactMatchMetric,
    "token_f1": TokenF1Metric,
    "numeric_exact_match": NumericExactMatchMetric,
    "rouge_l": RougeLMetric,
    "meteor": MeteorMetric,
    "bertscore": BertScoreMetric,
}
