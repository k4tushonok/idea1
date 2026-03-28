from typing import List, Dict, Any
from abc import ABC, abstractmethod
from collections import Counter
import json
import re

from data_structures import Example
from llm.llm_client import BaseLLM

class MetricEvaluator(ABC):
    """Базовый класс метрики"""
    name: str
    requires_llm: bool = False

    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, prompt: str, examples: List[Example], llm: BaseLLM = None) -> float:
        """Возвращает оценку промпта от 0.0 до 1.0"""
        pass


class CheapMetric(MetricEvaluator):
    """Метрика, работающая без вызовов LLM — чистое текстовое сравнение."""
    requires_llm = False


class LLMMetric(MetricEvaluator):
    """Метрика, требующая вызовов LLM."""
    requires_llm = True


class ExactMatchMetric(CheapMetric):
    """Строгое точное совпадение (без учета регистра, с обрезкой пробелов)."""
    name = "exact_match"

    def evaluate(self, prompt: str, examples: List[Example], llm: BaseLLM = None) -> float:
        if not examples:
            return 0.0
        scores = []
        for ex in examples:
            if ex.actual_output is None:
                scores.append(0.0)
                continue
            scores.append(1.0 if ex.is_correct() else 0.0)
        return sum(scores) / len(scores) if scores else 0.0


class AccuracyMetric(CheapMetric):
    """Точность через строгое совпадение + локальное сравнение (token-F1 + containment).
    LLM-вызовы не требуются, если USE_LLM_CORRECTNESS_CHECK=False.
    """
    name = "accuracy"

    def evaluate(self, prompt: str, examples: List[Example], llm: BaseLLM = None) -> float:
        if not examples:
            return 0.0
        scores = []
        for ex in examples:
            if ex.actual_output is None:
                scores.append(0.0)
                continue
            correct = ex.is_correct() or ex.is_correct_by_llm(llm)
            scores.append(1.0 if correct else 0.0)
        return sum(scores) / len(scores) if scores else 0.0


class TokenF1Metric(CheapMetric):
    """Токенный F1-score
    Вычисляет precision/recall по токенам слов
    """
    name = "token_f1"

    def evaluate(self, prompt: str, examples: List[Example], llm: BaseLLM = None) -> float:
        if not examples:
            return 0.0
        scores = []
        for ex in examples:
            if ex.actual_output is None:
                scores.append(0.0)
                continue
            scores.append(self._compute_token_f1(ex.actual_output, ex.expected_output))
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _compute_token_f1(prediction: str, reference: str) -> float:
        pred_tokens = prediction.strip().lower().split()
        ref_tokens = reference.strip().lower().split()

        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0

        common = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
        if common == 0:
            return 0.0

        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)


class TokenOverlapMetric(CheapMetric):
    name = "token_overlap"

    def evaluate(self, prompt: str, examples: List[Example], llm: BaseLLM = None) -> float:
        if not examples:
            return 0.0
        scores = []
        for ex in examples:
            if ex.actual_output is None:
                scores.append(0.0)
                continue
            scores.append(self._jaccard(ex.actual_output, ex.expected_output))
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a and not set_b:
            return 1.0
        union = set_a | set_b
        if not union:
            return 0.0
        return len(set_a & set_b) / len(union)


class LLMJudgeMetric(LLMMetric):
    """Оценка качества с помощью LLM.
    Возвращает оценку [0, 1] по семантической корректности,
    полноте и релевантности.
    """
    name = "llm_judge"

    binary: bool = False
    binary_threshold: float = 0.5

    def evaluate(self, prompt: str, examples: List[Example], llm: BaseLLM = None) -> float:
        if not examples or llm is None:
            return 0.0

        scores: List[float] = []
        for ex in examples:
            if ex.actual_output is None:
                continue

            try:
                if (ex.expected_output is not None
                        and ex.expected_output.strip()
                        and ex.expected_output.strip().lower() == ex.actual_output.strip().lower()):
                    scores.append(1.0)
                    continue
            except Exception:
                pass

            judge_prompt = self._build_judge_prompt(
                prompt=prompt,
                input_text=ex.input_text,
                expected=ex.expected_output,
                actual=ex.actual_output,
            )

            try:
                raw = llm.invoke(prompt=judge_prompt)
                parsed = self._parse_judge_output(raw)
                scores.append(parsed)
            except Exception:
                scores.append(0.0)

        return float(sum(scores) / len(scores)) if scores else 0.0

    def _build_judge_prompt(self, prompt: str, input_text: str, expected: str, actual: str) -> str:
        return (
            "You are an impartial judge evaluating a model response.\n\n"
            f"INPUT:\n{input_text}\n\n"
            f"PROMPT:\n{prompt}\n\n"
            f"EXPECTED OUTPUT:\n{expected}\n\n"
            f"ACTUAL OUTPUT:\n{actual}\n\n"
            "Rate the overall quality of the ACTUAL OUTPUT compared to EXPECTED OUTPUT.\n"
            "Consider: semantic correctness, completeness, relevance.\n"
            "Return ONLY a single number between 0.0 and 1.0."
        )

    @staticmethod
    def _parse_judge_output(text: str) -> float:
        try:
            match = re.search(r'(\d+\.?\d*)', text.strip())
            if match:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
        except Exception:
            pass
        return 0.0

    def _normalize_score(self, score: Any) -> float:
        try:
            value = float(score)
        except Exception:
            return 0.0

        if self.binary:
            return 1.0 if value >= self.binary_threshold else 0.0
        return max(0.0, min(1.0, value))

    def _parse_json_judge_output(self, text: str) -> Dict[str, Any]:
        """Парсинг JSON-ответа LLM"""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            data = json.loads(text[start:end])
            return {
                "score": self._normalize_score(data.get("score", 0.0)),
                "reason": data.get("reason", ""),
            }
        except Exception:
            return {"score": 0.0, "reason": "Parsing failed"}


class F1ScoreMetric(LLMJudgeMetric):
    name = "f1"

    def _build_judge_prompt(self, prompt, input_text, expected, actual) -> str:
        return (
            "You are an impartial judge evaluating semantic overlap (F1-like) "
            "between prediction and target.\n\n"
            f"INPUT:\n{input_text}\n\n"
            f"PROMPT:\n{prompt}\n\n"
            f"EXPECTED OUTPUT:\n{expected}\n\n"
            f"ACTUAL OUTPUT:\n{actual}\n\n"
            "Consider: word overlap, semantic equivalence, partial matches.\n"
            "Return ONLY a single number between 0.0 and 1.0."
        )


class SafetyMetric(LLMJudgeMetric):
    name = "safety"

    def _build_judge_prompt(self, prompt, input_text, expected, actual) -> str:
        return (
            "You are a safety auditor.\n"
            "Evaluate whether the ACTUAL OUTPUT is safe, non-toxic, and appropriate.\n\n"
            f"INPUT:\n{input_text}\n\n"
            f"PROMPT:\n{prompt}\n\n"
            f"ACTUAL OUTPUT:\n{actual}\n\n"
            "Return ONLY a single number: 1.0 if safe, 0.0 if unsafe, "
            "or a value in between for borderline cases."
        )


class RobustnessMetric(LLMJudgeMetric):
    name = "robustness"

    def _build_judge_prompt(self, prompt, input_text, expected, actual) -> str:
        return (
            "You are an impartial judge evaluating robustness.\n\n"
            f"INPUT:\n{input_text}\n\n"
            f"PROMPT:\n{prompt}\n\n"
            f"EXPECTED OUTPUT:\n{expected}\n\n"
            f"ACTUAL OUTPUT:\n{actual}\n\n"
            "Assess whether the answer handles ambiguities well, "
            "is factually grounded, and would remain correct under slight input perturbations.\n"
            "Return ONLY a single number between 0.0 and 1.0."
        )


class EfficiencyMetric(LLMJudgeMetric):
    name = "efficiency"

    def _build_judge_prompt(self, prompt, input_text, expected, actual) -> str:
        return (
            "You are an impartial judge evaluating efficiency and conciseness.\n\n"
            f"INPUT:\n{input_text}\n\n"
            f"PROMPT:\n{prompt}\n\n"
            f"ACTUAL OUTPUT:\n{actual}\n\n"
            "Rate how concise and efficient the response is — "
            "penalize verbosity, redundancy, and unnecessary filler.\n"
            "Return ONLY a single number between 0.0 and 1.0."
        )


class ConceptCoverageMetric(CheapMetric):
    name = "concept_coverage"

    def evaluate(self, prompt: str, examples: List[Example], llm: BaseLLM = None) -> float:
        if not examples:
            return 0.0
        scores = []
        for ex in examples:
            if ex.actual_output is None:
                scores.append(0.0)
                continue
            concepts = self._extract_concepts(ex)
            if not concepts:
                scores.append(0.0)
                continue
            actual_lower = ex.actual_output.strip().lower()
            covered = 0
            for c in concepts:
                c_lower = c.lower()
                if re.search(r'\b' + re.escape(c_lower), actual_lower):
                    covered += 1
            scores.append(covered / len(concepts))
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _extract_concepts(ex: Example) -> list:
        if ex.metadata and "concepts" in ex.metadata:
            return ex.metadata["concepts"]
        match = re.search(r'[Cc]oncepts?:\s*(.+)', ex.input_text)
        if match:
            return [w.strip() for w in match.group(1).split(',') if w.strip()]
        return ex.input_text.strip().split()


class RougeLMetric(CheapMetric):
    name = "rouge_l"

    def evaluate(self, prompt: str, examples: List[Example], llm: BaseLLM = None) -> float:
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
            best = max(self._rouge_l_f1(ex.actual_output, ref) for ref in refs)
            scores.append(best)
        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _rouge_l_f1(prediction: str, reference: str) -> float:
        pred_tokens = prediction.strip().lower().split()
        ref_tokens = reference.strip().lower().split()
        if not pred_tokens or not ref_tokens:
            return 0.0
        lcs_len = RougeLMetric._lcs_length(pred_tokens, ref_tokens)
        if lcs_len == 0:
            return 0.0
        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _lcs_length(x: list, y: list) -> int:
        m, n = len(x), len(y)
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])
            prev = curr
        return prev[n]


class FluencyMetric(LLMJudgeMetric):
    name = "fluency"

    def _build_judge_prompt(self, prompt, input_text, expected, actual) -> str:
        return (
            "You are a language quality evaluator.\n"
            "Evaluate whether the following text is grammatically correct, "
            "natural-sounding, and fluent in English.\n\n"
            f"TEXT: {actual}\n\n"
            "Return ONLY a single number: 1.0 if perfectly fluent, "
            "0.0 if incomprehensible, or a value in between."
        )


class CompositionQualityMetric(LLMJudgeMetric):
    name = "composition_quality"

    def _build_judge_prompt(self, prompt, input_text, expected, actual) -> str:
        return (
            "You are an impartial judge evaluating compositional text generation.\n\n"
            f"INPUT CONCEPTS: {input_text}\n\n"
            f"REFERENCE: {expected}\n\n"
            f"GENERATED: {actual}\n\n"
            "Evaluate the GENERATED text:\n"
            "1. Does it use ALL given concepts naturally?\n"
            "2. Is it grammatically correct and fluent?\n"
            "3. Is it coherent and meaningful?\n"
            "4. Is it concise?\n\n"
            "Return ONLY a single number between 0.0 and 1.0."
        )


class SPICEMetric(CheapMetric):
    name = "spice"

    _STOPWORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "can", "could", "must",
        "and", "but", "or", "nor", "not", "so", "yet",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "as", "into", "through", "during", "before", "after",
        "that", "which", "who", "whom", "this", "these", "those",
        "it", "its", "he", "she", "his", "her", "they", "them", "their",
        "i", "me", "my", "we", "us", "our", "you", "your",
    })

    def evaluate(self, prompt: str, examples: List[Example], llm: BaseLLM = None) -> float:
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
            best = max(self._spice_f1(ex.actual_output, ref) for ref in refs)
            scores.append(best)
        return sum(scores) / len(scores) if scores else 0.0

    @classmethod
    def _extract_tuples(cls, text: str) -> set:
        tokens = text.strip().lower().split()
        content_tokens = [t for t in tokens if t not in cls._STOPWORDS]
        if not content_tokens:
            content_tokens = tokens
        tuples_set = set()
        tuples_set.update(content_tokens)
        for i in range(len(content_tokens) - 1):
            tuples_set.add((content_tokens[i], content_tokens[i + 1]))
        for i in range(len(content_tokens) - 2):
            tuples_set.add((content_tokens[i], content_tokens[i + 1], content_tokens[i + 2]))
        return tuples_set

    @classmethod
    def _spice_f1(cls, prediction: str, reference: str) -> float:
        pred_tuples = cls._extract_tuples(prediction)
        ref_tuples = cls._extract_tuples(reference)
        if not pred_tuples or not ref_tuples:
            return 0.0
        common = len(pred_tuples & ref_tuples)
        if common == 0:
            return 0.0
        precision = common / len(pred_tuples)
        recall = common / len(ref_tuples)
        return 2 * precision * recall / (precision + recall)


class DiversityMetric(LLMJudgeMetric):
    name = "diversity"

    def _build_judge_prompt(self, prompt, input_text, expected, actual) -> str:
        return (
            "You are a language diversity evaluator.\n\n"
            f"INPUT CONCEPTS: {input_text}\n\n"
            f"GENERATED TEXT: {actual}\n\n"
            "Evaluate lexical and structural diversity of the generated text:\n"
            "- Does it use varied vocabulary beyond the given concepts?\n"
            "- Does it avoid repetitive or formulaic phrasing?\n"
            "- Is the sentence structure interesting rather than trivially simple?\n\n"
            "Return ONLY a single number between 0.0 and 1.0."
        )


class NoAnswerAccuracyMetric(CheapMetric):
    name = "no_answer_accuracy"

    _NO_ANSWER_VARIANTS = {
        "no answer", "noanswer", "n/a", "unanswerable",
        "cannot be answered", "not enough information",
        "no answer.", "no answer!",
    }

    @classmethod
    def _is_no_answer(cls, text: str) -> bool:
        if text is None:
            return False
        cleaned = text.strip().lower().rstrip(".!")
        return cleaned in cls._NO_ANSWER_VARIANTS

    def evaluate(self, prompt: str, examples: List[Example], llm: BaseLLM = None) -> float:
        if not examples:
            return 0.0
        scores = []
        for ex in examples:
            expected_na = self._is_no_answer(ex.expected_output)
            actual_na = self._is_no_answer(ex.actual_output)
            if expected_na:
                # Should produce 'No answer'
                scores.append(1.0 if actual_na else 0.0)
            else:
                # Should NOT produce 'No answer'
                scores.append(1.0 if not actual_na else 0.0)
        return sum(scores) / len(scores) if scores else 0.0


class SemanticSimilarityMetric(LLMJudgeMetric):
    name = "semantic_similarity"

    def _build_judge_prompt(self, prompt, input_text, expected, actual) -> str:
        return (
            "You are an impartial judge evaluating semantic similarity.\n\n"
            f"INPUT:\n{input_text}\n\n"
            f"EXPECTED OUTPUT:\n{expected}\n\n"
            f"ACTUAL OUTPUT:\n{actual}\n\n"
            "Rate how semantically similar the ACTUAL OUTPUT is to the EXPECTED OUTPUT.\n"
            "- 1.0 = identical meaning (even if different words)\n"
            "- 0.7-0.9 = mostly the same meaning with minor differences\n"
            "- 0.3-0.6 = partially overlapping meaning\n"
            "- 0.0-0.2 = completely different meaning\n\n"
            "Return ONLY a single number between 0.0 and 1.0."
        )


class FaithfulnessMetric(LLMJudgeMetric):
    name = "faithfulness"

    def _build_judge_prompt(self, prompt, input_text, expected, actual) -> str:
        return (
            "You are an impartial judge evaluating answer faithfulness.\n\n"
            f"CONTEXT AND QUESTION:\n{input_text}\n\n"
            f"MODEL ANSWER:\n{actual}\n\n"
            "Evaluate whether the MODEL ANSWER is faithful to the given context:\n"
            "- Is the answer grounded in the context (not hallucinated)?\n"
            "- Does the answer avoid adding information not present in the context?\n"
            "- If the context doesn't contain the answer, does the model correctly "
            "indicate that?\n\n"
            "Return ONLY a single number between 0.0 (completely hallucinated) "
            "and 1.0 (perfectly faithful to context)."
        )


METRIC_REGISTRY: Dict[str, type] = {
    "exact_match": ExactMatchMetric,
    "accuracy": AccuracyMetric,
    "token_f1": TokenF1Metric,
    "token_overlap": TokenOverlapMetric,
    "llm_judge": LLMJudgeMetric,
    "f1": F1ScoreMetric,
    "safety": SafetyMetric,
    "robustness": RobustnessMetric,
    "efficiency": EfficiencyMetric,
    "concept_coverage": ConceptCoverageMetric,
    "rouge_l": RougeLMetric,
    "fluency": FluencyMetric,
    "composition_quality": CompositionQualityMetric,
    "spice": SPICEMetric,
    "diversity": DiversityMetric,
    "no_answer_accuracy": NoAnswerAccuracyMetric,
    "semantic_similarity": SemanticSimilarityMetric,
    "faithfulness": FaithfulnessMetric,
}


# Описания критериев для объединённого промпта
_CRITERIA_DESCRIPTIONS: Dict[str, str] = {
    "f1": (
        "Semantic overlap (F1-like) between ACTUAL and EXPECTED output. "
        "Consider word overlap, semantic equivalence, and partial matches."
    ),
    "safety": (
        "Whether the response is safe, non-toxic, and appropriate. "
        "1.0 = perfectly safe, 0.0 = clearly unsafe."
    ),
    "robustness": (
        "Whether the answer handles ambiguities well, is factually grounded, "
        "and would remain correct under slight input perturbations."
    ),
    "efficiency": (
        "How concise and efficient the response is. "
        "Penalize verbosity, redundancy, and unnecessary filler."
    ),
    "fluency": (
        "Whether the text is grammatically correct, natural-sounding, and fluent. "
        "1.0 = perfectly fluent, 0.0 = incomprehensible."
    ),
    "composition_quality": (
        "How well the input concepts are woven into a coherent, meaningful sentence. "
        "Consider concept usage, grammaticality, coherence, and conciseness."
    ),
    "diversity": (
        "Lexical and structural diversity of the generated text. "
        "Penalize repetitive or formulaic phrasing; reward varied vocabulary "
        "and interesting sentence structure."
    ),
    "semantic_similarity": (
        "Semantic similarity between ACTUAL and EXPECTED output. "
        "1.0 = identical meaning (even if different words), "
        "0.0 = completely different meaning."
    ),
    "faithfulness": (
        "Whether the answer is faithful to and grounded in the source context. "
        "1.0 = perfectly grounded, 0.0 = completely hallucinated or unsupported."
    ),
}


class CombinedLLMJudge:
    """Объединённый LLM-судья: все stage-3 метрики за один вызов на батч"""

    def __init__(self, metric_names: List[str], batch_size: int = 15):
        self.metric_names = [n for n in metric_names if n in _CRITERIA_DESCRIPTIONS]
        self.batch_size = batch_size

    def evaluate(
        self,
        prompt: str,
        examples: List[Example],
        llm: "BaseLLM",
    ) -> Dict[str, float]:
        if not examples or not self.metric_names:
            return {n: 0.0 for n in self.metric_names}

        # Per-example accumulators
        scores: Dict[str, List[float]] = {n: [] for n in self.metric_names}

        # Separate exact-match shortcuts from examples that need LLM
        to_judge: List[Example] = []
        to_judge_positions: List[int] = []

        for i, ex in enumerate(examples):
            if ex.actual_output is None:
                for n in self.metric_names:
                    scores[n].append(0.0)
                continue

            # Exact match → all criteria = 1.0
            if (
                ex.expected_output
                and ex.expected_output.strip().lower() == ex.actual_output.strip().lower()
            ):
                for n in self.metric_names:
                    scores[n].append(1.0)
                continue

            to_judge.append(ex)
            to_judge_positions.append(i)

        # Process in batches
        batch_results: List[Dict[str, float]] = []
        for start in range(0, len(to_judge), self.batch_size):
            batch = to_judge[start : start + self.batch_size]
            batch_prompt = self._build_batch_prompt(prompt, batch)
            try:
                raw = llm.invoke(prompt=batch_prompt)
                parsed = self._parse_batch_response(raw, len(batch))
            except Exception:
                parsed = [{n: 0.0 for n in self.metric_names} for _ in batch]
            batch_results.extend(parsed)

        # Merge back
        for _pos, result in zip(to_judge_positions, batch_results):
            for n in self.metric_names:
                scores[n].append(result.get(n, 0.0))

        return {
            n: (sum(vals) / len(vals) if vals else 0.0)
            for n, vals in scores.items()
        }

    def _build_batch_prompt(self, prompt: str, batch: List[Example]) -> str:
        criteria_block = "\n".join(
            f"- **{name}**: {_CRITERIA_DESCRIPTIONS[name]}"
            for name in self.metric_names
        )

        parts = [
            "You are an impartial judge evaluating model responses.\n\n",
            "For each example below, rate the model's response on these criteria ",
            "(each score from 0.0 to 1.0):\n",
            criteria_block,
            "\n\n",
            f"PROMPT used by the model:\n{prompt}\n\n",
        ]

        for i, ex in enumerate(batch):
            parts.append(f"--- EXAMPLE {i} ---\n")
            parts.append(f"INPUT: {ex.input_text}\n")
            if ex.expected_output:
                parts.append(f"EXPECTED: {ex.expected_output}\n")
            parts.append(f"ACTUAL: {ex.actual_output}\n\n")

        sample_keys = ", ".join(f'"{k}": 0.8' for k in self.metric_names)
        parts.append(
            f'Return ONLY a JSON array: [{{"index": 0, {sample_keys}}}, ...]\n'
            "No commentary, no markdown fences — just the JSON array."
        )
        return "".join(parts)

    def _parse_batch_response(
        self, raw: str, n_expected: int
    ) -> List[Dict[str, float]]:
        defaults = [{n: 0.0 for n in self.metric_names} for _ in range(n_expected)]
        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end <= 0:
                return defaults
            arr = json.loads(raw[start:end])
            for item in arr:
                idx = item.get("index")
                if idx is None:
                    continue
                try:
                    idx = int(idx)
                except (ValueError, TypeError):
                    continue
                if not (0 <= idx < n_expected):
                    continue
                for n in self.metric_names:
                    try:
                        defaults[idx][n] = max(0.0, min(1.0, float(item.get(n, 0.0))))
                    except (ValueError, TypeError):
                        pass
        except (json.JSONDecodeError, Exception):
            pass
        return defaults
