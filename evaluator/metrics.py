"""
Prompt quality evaluation metrics.

Implementations for various task types:
- ExactMatch, TokenF1        — for QA tasks (SQuAD)
- NumericExactMatch          — for numeric tasks (GSM8K)
- ROUGE-L, METEOR, BERTScore — for generation tasks (XSum, CommonGen)

All metrics inherit CheapMetric and are registered in METRIC_REGISTRY.
"""

from typing import List, Dict
from abc import ABC, abstractmethod
from collections import Counter

from nltk.translate.meteor_score import meteor_score as _nltk_meteor

from data_structures import Example, normalize_answer, extract_final_number, NO_ANSWER_VARIANTS


def get_tokens(s: str) -> List[str]:
    """Tokenize a normalized string."""
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> int:
    """Exact match of two normalized answers (0 or 1)."""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    """Token-F1 between gold and predicted answers."""
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


_NO_ANSWER_VARIANTS = NO_ANSWER_VARIANTS


def _get_gold_answers(ex: Example) -> List[str]:
    """Return the list of valid gold answers from metadata or expected_output."""
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
    """Abstract base class for prompt evaluation metrics."""

    name: str

    def __init__(self):
        pass

    def supports_examples(self, examples: List[Example]) -> bool:
        """Return True if this metric supports the given example set."""
        return True

    @abstractmethod
    def evaluate(self, prompt: str, examples: List[Example]) -> float:
        """Return a prompt score from 0.0 to 1.0."""
        pass


class CheapMetric(MetricEvaluator):
    """A metric that requires no LLM calls — pure text comparison."""

    pass


class ExactMatchMetric(CheapMetric):
    """Exact-match metric for QA tasks."""

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
    """Token-F1 metric for QA tasks."""

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
    """ROUGE-L F1 metric for text generation tasks."""

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
    """METEOR metric for text generation tasks."""

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



class NumericExactMatchMetric(CheapMetric):
    """Exact match on extracted numeric answers."""

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
            pred = extract_final_number(ex.actual_output)
            em = max(
                int(
                    normalize_answer(extract_final_number(g)) == normalize_answer(pred)
                )
                for g in gold_answers
            )
            scores.append(float(em))
        return sum(scores) / len(scores) if scores else 0.0


class BertScoreMetric(CheapMetric):
    """BERTScore F1 metric — semantic similarity based on BERT embeddings."""

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
