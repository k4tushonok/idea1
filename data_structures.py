"""
Core data structures for the prompt optimization system.

Defines the main classes:
- Example       — a training/test example with answer evaluation methods
- TextGradient  — a textual gradient (error analysis + improvement directions)
- EditOperation — a recorded prompt edit operation
- Metrics       — evaluation metrics with composite scoring
- PromptNode    — a node in the prompt evolution tree
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, ClassVar
from datetime import datetime
from enum import Enum
import uuid
import re as _re
import string as _string

from config import (
    METRICS_CONFIG,
    CORRECTNESS_TOKEN_F1_THRESHOLD,
)

_DEFAULT_METRIC_WEIGHTS: Dict[str, float] = {
    m["name"]: m["weight"] for m in METRICS_CONFIG
}
NO_ANSWER_VARIANTS: List[str] = ["", "no answer", "no answer."]
_NO_ANSWER_VARIANTS = NO_ANSWER_VARIANTS  # backward-compat alias



def normalize_answer(s: str) -> str:
    """Normalize an answer: lowercase, remove articles, punctuation, and extra whitespace."""
    def remove_articles(text):
        regex = _re.compile(r"\b(a|an|the)\b", _re.UNICODE)
        return _re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(_string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_final_number(text: str) -> str:
    """Extract the final numeric answer from a model reasoning chain."""
    if text is None:
        return ""
    text = text.strip()
    m = _re.search(r"####\s*(.+)", text)
    if m:
        return m.group(1).strip().replace(",", "")
    m = _re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*\$?\s*(-?[\d,]+\.?\d*)",
        text,
        _re.IGNORECASE,
    )
    if m:
        return m.group(1).replace(",", "")
    m = _re.search(r"=\s*\$?\s*(-?[\d,]+\.?\d*)\s*$", text, _re.MULTILINE)
    if m:
        return m.group(1).replace(",", "")
    numbers = _re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return normalize_answer(text)


class OptimizationSource(Enum):
    """Source that produced a given prompt node."""

    INITIAL = "initial"  # Initial seed prompt
    LOCAL = "local"      # Produced by local optimization
    GLOBAL = "global"    # Produced by global optimization
    MANUAL = "manual"    # Manually provided


@dataclass
class Example:
    """A single training/evaluation example for prompting."""

    input_text: str                                           # Input fed to the prompt
    expected_output: str                                      # Ground-truth answer
    actual_output: Optional[str] = None                       # Model-generated answer
    metadata: Dict[str, Any] = field(default_factory=dict)    # Optional task metadata

    def is_numeric_qa_task(self) -> bool:
        """Return True if this example belongs to a numeric QA task (e.g. GSM8K)."""
        if not self.metadata:
            return False
        if self.metadata.get("task_type") == "numeric_qa":
            return True
        return bool(self.metadata.get("numeric_qa"))

    def _gold_answers(self) -> List[str]:
        """Return all valid reference answers from metadata or expected_output."""
        all_ans = self.metadata.get("all_answers") if self.metadata else None
        if all_ans is not None:
            gold_list = [a for a in all_ans if normalize_answer(a)]
            return gold_list if gold_list else _NO_ANSWER_VARIANTS
        if self.expected_output is None:
            return _NO_ANSWER_VARIANTS
        if self.expected_output.strip().lower() in ("no answer", "no answer.", ""):
            return _NO_ANSWER_VARIANTS
        return [self.expected_output]

    @staticmethod
    def _compute_token_f1(actual: str, expected: str) -> float:
        """Compute token-level F1 between the actual and expected answers."""
        from collections import Counter as _Counter

        actual_norm = normalize_answer(actual)
        expected_norm = normalize_answer(expected)
        a_tokens = actual_norm.split()
        e_tokens = expected_norm.split()

        if not a_tokens or not e_tokens:
            return float(a_tokens == e_tokens)

        common = sum((_Counter(a_tokens) & _Counter(e_tokens)).values())
        if common == 0:
            return 0.0

        precision = common / len(a_tokens)
        recall = common / len(e_tokens)
        return 2 * precision * recall / (precision + recall)

    def is_correct(self) -> bool:
        """Return True if actual_output exactly matches any gold answer."""
        if self.actual_output is None:
            return False
        gold_list = self._gold_answers()
        return any(
            normalize_answer(gold) == normalize_answer(self.actual_output)
            for gold in gold_list
        )

    def qa_exact_match_score(self) -> float:
        """Exact-match score against any reference answer (0.0 or 1.0)."""
        if self.actual_output is None:
            return 0.0
        actual_norm = normalize_answer(self.actual_output)
        return max(
            float(normalize_answer(gold) == actual_norm)
            for gold in self._gold_answers()
        )

    def qa_token_f1_score(self) -> float:
        """Maximum token-F1 score across all reference answers."""
        if self.actual_output is None:
            return 0.0
        return max(
            self._compute_token_f1(self.actual_output, gold)
            for gold in self._gold_answers()
        )

    def numeric_exact_match_score(self) -> float:
        """Exact match on extracted numeric answer vs. reference."""
        if self.actual_output is None:
            return 0.0
        actual_num = normalize_answer(extract_final_number(self.actual_output))
        return max(
            float(normalize_answer(extract_final_number(gold)) == actual_num)
            for gold in self._gold_answers()
        )

    def numeric_token_f1_score(self) -> float:
        """Token-F1 for numeric answers after extracting the final number."""
        if self.actual_output is None:
            return 0.0
        actual_num = extract_final_number(self.actual_output)
        return max(
            self._compute_token_f1(actual_num, extract_final_number(gold))
            for gold in self._gold_answers()
        )

    def generation_concept_coverage_score(self) -> float:
        """Fraction of required concepts from metadata covered in the generated output."""
        if (
            self.actual_output is None
            or not self.metadata
            or "concepts" not in self.metadata
        ):
            return 0.0
        concepts = self.metadata.get("concepts") or []
        if not concepts:
            return 0.0
        actual_norm = normalize_answer(self.actual_output)
        covered = sum(
            1
            for c in concepts
            if _re.search(r"\b" + _re.escape(c.lower()) + r"\w*\b", actual_norm)
        )
        return covered / len(concepts)

    @staticmethod
    def _lcs_length(x: List[str], y: List[str]) -> int:
        """Compute the length of the Longest Common Subsequence (LCS) of two token lists."""
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

    @classmethod
    def _rouge_l_f1(cls, prediction: str, reference: str) -> float:
        """Compute ROUGE-L F1 between prediction and reference using LCS."""
        pred_tokens = normalize_answer(prediction).split()
        ref_tokens = normalize_answer(reference).split()
        if not pred_tokens or not ref_tokens:
            return 0.0
        lcs_len = cls._lcs_length(pred_tokens, ref_tokens)
        if lcs_len == 0:
            return 0.0
        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    def generation_reference_token_f1_score(self) -> float:
        """Maximum token-F1 against all references for generation tasks."""
        if self.actual_output is None:
            return 0.0
        references = []
        if self.metadata and "references" in self.metadata:
            references = [r for r in self.metadata["references"] if r]
        elif self.expected_output:
            references = [self.expected_output]
        if not references:
            return 0.0
        return max(
            self._compute_token_f1(self.actual_output, ref) for ref in references
        )

    def generation_reference_rouge_l_score(self) -> float:
        """Maximum ROUGE-L F1 against all references for generation tasks."""
        if self.actual_output is None:
            return 0.0
        references = []
        if self.metadata and "references" in self.metadata:
            references = [r for r in self.metadata["references"] if r]
        elif self.expected_output:
            references = [self.expected_output]
        if not references:
            return 0.0
        return max(self._rouge_l_f1(self.actual_output, ref) for ref in references)

    def generation_optimization_score(self) -> float:
        """Composite score for generation tasks: 55% concept coverage + 25% token-F1 + 20% ROUGE-L."""
        coverage = self.generation_concept_coverage_score()
        ref_f1 = self.generation_reference_token_f1_score()
        rouge_l = self.generation_reference_rouge_l_score()
        return 0.55 * coverage + 0.25 * ref_f1 + 0.20 * rouge_l

    def is_generation_success(self) -> bool:
        """Return True if full concept coverage and optimization_score >= 0.68."""
        if self.actual_output is None:
            return False
        coverage = self.generation_concept_coverage_score()
        if coverage < 1.0:
            return False
        return self.generation_optimization_score() >= 0.68

    def is_strict_qa_success(self) -> bool:
        """Strict QA success: exact match score equals 1.0."""
        if self.actual_output is None:
            return False
        return self.qa_exact_match_score() >= 1.0

    def is_numeric_qa_success(self) -> bool:
        """Return True if the extracted numeric answer exactly matches the reference."""
        if self.actual_output is None:
            return False
        return self.numeric_exact_match_score() >= 1.0

    def is_success_for_optimization(self) -> bool:
        """Success signal used for success/failure split and failure mining."""
        if self.actual_output is None:
            return False
        if self.is_numeric_qa_task():
            return self.is_numeric_qa_success()
        if self.metadata and "all_answers" in self.metadata:
            return self.is_strict_qa_success()
        if (
            self.metadata
            and "references" in self.metadata
            and "concepts" in self.metadata
        ):
            return self.is_generation_success()
        return self.is_correct() or self.is_correct_heuristic()

    _correctness_cache: ClassVar[Dict[tuple, bool]] = {}

    def is_correct_heuristic(self) -> bool:
        """Heuristic correctness check using token-F1 and substring containment."""
        if self.actual_output is None:
            return False

        if self.is_correct():
            return True

        return self._is_similar_locally(
            self.actual_output, self.expected_output, self.metadata
        )

    @staticmethod
    def _is_similar_locally(
        actual: str, expected: str, metadata: Dict[str, Any] = None
    ) -> bool:
        """Token-F1 + containment check + concept coverage for generation tasks"""
        if actual is None or expected is None:
            return False
        actual_norm = normalize_answer(actual)
        expected_norm = normalize_answer(expected)
        if not actual_norm or not expected_norm:
            return actual_norm == expected_norm  # both empty → True
        # 0. Concept coverage for generation tasks
        if metadata and isinstance(metadata, dict) and "concepts" in metadata:
            concepts = metadata["concepts"]
            if concepts:
                covered = sum(
                    1
                    for c in concepts
                    if _re.search(r"\b" + _re.escape(c.lower()), actual_norm)
                )
                if covered >= len(concepts):
                    return True
        # 1. Containment: if expected is a substring of actual, it's correct
        if expected_norm in actual_norm:
            return True
        # 2. Token F1 check
        from collections import Counter as _Counter

        a_tokens = actual_norm.split()
        e_tokens = expected_norm.split()
        if not a_tokens or not e_tokens:
            return False
        common = sum((_Counter(a_tokens) & _Counter(e_tokens)).values())
        if common == 0:
            return False
        precision = common / len(a_tokens)
        recall = common / len(e_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1 >= CORRECTNESS_TOKEN_F1_THRESHOLD

    def to_dict(self) -> Dict:
        """Serialize the example to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Example":
        """Create an Example from a dictionary."""
        return cls(**data)


@dataclass
class TextGradient:
    """Textual gradient: natural-language description of what is wrong and how to improve the prompt."""

    failure_examples: List[Example] = field(
        default_factory=list
    )  # Examples where the prompt failed
    success_examples: List[Example] = field(
        default_factory=list
    )  # Examples where the prompt succeeded
    error_analysis: str = ""              # Analysis of observed errors
    suggested_direction: str = ""         # Suggested improvement direction
    specific_suggestions: List[str] = field(
        default_factory=list
    )  # Concrete edit recommendations
    priority: float = 1.0                 # Gradient priority in [0.0, 1.0]
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata for analysis

    def to_dict(self) -> Dict:
        return {
            "failure_examples": [ex.to_dict() for ex in self.failure_examples],
            "success_examples": [ex.to_dict() for ex in self.success_examples],
            "error_analysis": self.error_analysis,
            "suggested_direction": self.suggested_direction,
            "specific_suggestions": self.specific_suggestions,
            "priority": self.priority,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TextGradient":
        data = data.copy()
        data["failure_examples"] = [
            Example.from_dict(ex) for ex in data["failure_examples"]
        ]
        data["success_examples"] = [
            Example.from_dict(ex) for ex in data["success_examples"]
        ]
        return cls(**data)


@dataclass
class EditOperation:
    """A recorded prompt edit. Tracks what was changed and why."""

    description: str                                 # Human-readable description of the change
    gradient_source: Optional[TextGradient] = None   # Gradient that triggered this edit
    before_snippet: str = ""                         # Prompt excerpt before the edit
    after_snippet: str = ""                          # Prompt excerpt after the edit

    def to_dict(self) -> Dict:
        return {
            "description": self.description,
            "gradient_source": (
                self.gradient_source.to_dict() if self.gradient_source else None
            ),
            "before_snippet": self.before_snippet,
            "after_snippet": self.after_snippet,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EditOperation":
        data = data.copy()
        data.pop("operation_type", None)
        if data.get("gradient_source"):
            data["gradient_source"] = TextGradient.from_dict(data["gradient_source"])
        return cls(**data)


@dataclass
class Metrics:
    """Prompt evaluation metrics with a composite score for ranking candidates.

    Weights are set by the scorer at creation time and define the contribution of each
    metric to the composite score.  If weights is empty, the global
    _DEFAULT_METRIC_WEIGHTS derived from METRICS_CONFIG is used.
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)

    def composite_score(self) -> float:
        """Weighted sum of metrics: Σ(weight_i * metric_i)."""
        w = self.weights if self.weights else _DEFAULT_METRIC_WEIGHTS
        keys = set(list(self.metrics.keys()) + list(w.keys()))
        return sum(self.metrics.get(k, 0.0) * w.get(k, 0.0) for k in keys)

    def to_dict(self) -> Dict:
        d = self.metrics.copy()
        d["composite_score"] = self.composite_score()
        d["_weights"] = self.weights.copy()
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "Metrics":
        data = data.copy()
        data.pop("composite_score", None)
        weights = data.pop("_weights", {})
        m = cls()
        m.metrics = {k: float(v) for k, v in data.items()}
        m.weights = weights if isinstance(weights, dict) else {}
        return m


@dataclass
class PromptNode:
    """A node in the prompt evolution tree. Stores the full prompt state and its lineage."""

    id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )  # Unique prompt identifier
    prompt_text: str = ""                           # Prompt text
    parent_id: Optional[str] = None                 # Parent prompt identifier
    children_ids: List[str] = field(
        default_factory=list
    )  # Child prompt identifiers
    generation: int = 0                             # Generation number (depth in the tree)
    source: OptimizationSource = OptimizationSource.INITIAL  # Optimization source
    operations: List[EditOperation] = field(default_factory=list)  # Edit history
    metrics: Metrics = field(default_factory=Metrics)             # Evaluation metrics
    timestamp: datetime = field(default_factory=datetime.now)     # Creation timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)        # Additional metadata
    is_evaluated: bool = False                      # Whether the prompt has been evaluated
    evaluation_examples: Dict[str, List[Example]] = field(
        default_factory=lambda: {  # Examples used during evaluation
            "success": [],
            "failures": [],
        }
    )
    evaluation_examples_by_split: Dict[str, Dict[str, List[Example]]] = field(
        default_factory=dict
    )

    def add_child(self, child_id: str):
        """Register a child node ID."""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def selection_score(self) -> float:
        """Score used for candidate selection: prefers full_validation_score, falls back to composite_score."""
        try:
            return float(self.metadata["full_validation_score"])
        except Exception:
            return self.metrics.composite_score()

    def selection_accuracy(self) -> float:
        """Accuracy used for selection: prefers full_validation_accuracy, falls back to accuracy metric."""
        try:
            return float(self.metadata["full_validation_accuracy"])
        except Exception:
            return float(self.metrics.metrics.get("accuracy", 0.0))

    def to_dict(self) -> Dict:
        """Serialize the node to a dictionary for persistence."""
        return {
            "id": self.id,
            "prompt_text": self.prompt_text,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "generation": self.generation,
            "source": self.source.value,
            "operations": [op.to_dict() for op in self.operations],
            "metrics": self.metrics.to_dict(),
            "evaluation_examples": {
                "success": [ex.to_dict() for ex in self.evaluation_examples["success"]],
                "failures": [
                    ex.to_dict() for ex in self.evaluation_examples["failures"]
                ],
            },
            "evaluation_examples_by_split": {
                split: {
                    "success": [ex.to_dict() for ex in by_split.get("success", [])],
                    "failures": [ex.to_dict() for ex in by_split.get("failures", [])],
                }
                for split, by_split in self.evaluation_examples_by_split.items()
            },
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "is_evaluated": self.is_evaluated,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PromptNode":
        """Deserialize a PromptNode from a saved dictionary."""
        data = data.copy()
        data["source"] = OptimizationSource(data["source"])
        data["operations"] = [EditOperation.from_dict(op) for op in data["operations"]]
        data["metrics"] = Metrics.from_dict(data["metrics"])
        data["evaluation_examples"] = {
            "success": [
                Example.from_dict(ex) for ex in data["evaluation_examples"]["success"]
            ],
            "failures": [
                Example.from_dict(ex) for ex in data["evaluation_examples"]["failures"]
            ],
        }
        raw_by_split = data.get("evaluation_examples_by_split", {})
        data["evaluation_examples_by_split"] = {
            split: {
                "success": [
                    Example.from_dict(ex) for ex in by_split.get("success", [])
                ],
                "failures": [
                    Example.from_dict(ex) for ex in by_split.get("failures", [])
                ],
            }
            for split, by_split in raw_by_split.items()
        }
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
