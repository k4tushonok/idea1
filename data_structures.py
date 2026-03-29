from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, ClassVar
from datetime import datetime
from enum import Enum
import uuid
import re as _re
import string as _string

from config import (
    METRIC_WEIGHTS,
    CORRECTNESS_TOKEN_F1_THRESHOLD,
)

_NO_ANSWER_VARIANTS = ["", "no answer", "no answer."]

def _normalize_answer(s: str) -> str:
    def remove_articles(text):
        regex = _re.compile(r'\b(a|an|the)\b', _re.UNICODE)
        return _re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(_string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

class OptimizationSource(Enum):
    """Источник оптимизации промпта"""
    INITIAL = "initial"          # Начальный промпт
    LOCAL = "local"              # Локальная оптимизация 
    GLOBAL = "global"            # Глобальная оптимизация 
    MANUAL = "manual"            # Ручное редактирование

@dataclass
class Example:
    """Пример для обучения/тестирования промпта"""
    input_text: str                                         # Входные данные
    expected_output: str                                    # Ожидаемый результат
    actual_output: Optional[str] = None                     # Фактический результат 
    metadata: Dict[str, Any] = field(default_factory=dict)  # Метаданные
    
    def _gold_answers(self) -> List[str]:
        all_ans = self.metadata.get("all_answers") if self.metadata else None
        if all_ans is not None:
            gold_list = [a for a in all_ans if _normalize_answer(a)]
            return gold_list if gold_list else _NO_ANSWER_VARIANTS
        if self.expected_output is None:
            return _NO_ANSWER_VARIANTS
        if self.expected_output.strip().lower() in ("no answer", "no answer.", ""):
            return _NO_ANSWER_VARIANTS
        return [self.expected_output]

    @staticmethod
    def _compute_token_f1(actual: str, expected: str) -> float:
        from collections import Counter as _Counter

        actual_norm = _normalize_answer(actual)
        expected_norm = _normalize_answer(expected)
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
        """Проверка корректности ответа"""
        if self.actual_output is None:
            return False
        gold_list = self._gold_answers()
        return any(
            _normalize_answer(gold) == _normalize_answer(self.actual_output)
            for gold in gold_list
        )

    def qa_exact_match_score(self) -> float:
        if self.actual_output is None:
            return 0.0
        actual_norm = _normalize_answer(self.actual_output)
        return max(
            float(_normalize_answer(gold) == actual_norm)
            for gold in self._gold_answers()
        )

    def qa_token_f1_score(self) -> float:
        if self.actual_output is None:
            return 0.0
        return max(
            self._compute_token_f1(self.actual_output, gold)
            for gold in self._gold_answers()
        )

    def generation_concept_coverage_score(self) -> float:
        if self.actual_output is None or not self.metadata or "concepts" not in self.metadata:
            return 0.0
        concepts = self.metadata.get("concepts") or []
        if not concepts:
            return 0.0
        actual_norm = _normalize_answer(self.actual_output)
        covered = sum(
            1 for c in concepts
            if _re.search(r'\b' + _re.escape(c.lower()) + r'\w*\b', actual_norm)
        )
        return covered / len(concepts)

    @staticmethod
    def _lcs_length(x: List[str], y: List[str]) -> int:
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
        pred_tokens = _normalize_answer(prediction).split()
        ref_tokens = _normalize_answer(reference).split()
        if not pred_tokens or not ref_tokens:
            return 0.0
        lcs_len = cls._lcs_length(pred_tokens, ref_tokens)
        if lcs_len == 0:
            return 0.0
        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    def generation_reference_token_f1_score(self) -> float:
        if self.actual_output is None:
            return 0.0
        references = []
        if self.metadata and "references" in self.metadata:
            references = [r for r in self.metadata["references"] if r]
        elif self.expected_output:
            references = [self.expected_output]
        if not references:
            return 0.0
        return max(self._compute_token_f1(self.actual_output, ref) for ref in references)

    def generation_reference_rouge_l_score(self) -> float:
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
        coverage = self.generation_concept_coverage_score()
        ref_f1 = self.generation_reference_token_f1_score()
        rouge_l = self.generation_reference_rouge_l_score()
        return 0.55 * coverage + 0.25 * ref_f1 + 0.20 * rouge_l

    def is_generation_success(self) -> bool:
        if self.actual_output is None:
            return False
        coverage = self.generation_concept_coverage_score()
        if coverage < 1.0:
            return False
        return self.generation_optimization_score() >= 0.68

    def is_strict_qa_success(self) -> bool:
        if self.actual_output is None:
            return False
        return self.qa_exact_match_score() >= 1.0

    def is_success_for_optimization(self) -> bool:
        """Сигнал успеха для split на success/failure и mining ошибок"""
        if self.actual_output is None:
            return False
        if self.metadata and "all_answers" in self.metadata:
            return self.is_strict_qa_success()
        if self.metadata and "references" in self.metadata and "concepts" in self.metadata:
            return self.is_generation_success()
        return self.is_correct() or self.is_correct_heuristic()

    _correctness_cache: ClassVar[Dict[tuple, bool]] = {}

    def is_correct_heuristic(self) -> bool:
        """Проверка корректности ответа эвристически (token-F1 + containment)"""
        if self.actual_output is None:
            return False

        if self.is_correct():
            return True

        return self._is_similar_locally(self.actual_output, self.expected_output, self.metadata)

    @staticmethod
    def _is_similar_locally(actual: str, expected: str, metadata: Dict[str, Any] = None) -> bool:
        """Token-F1 + containment check + concept coverage for generation tasks"""
        if actual is None or expected is None:
            return False
        actual_norm = _normalize_answer(actual)
        expected_norm = _normalize_answer(expected)
        if not actual_norm or not expected_norm:
            return actual_norm == expected_norm  # both empty → True
        # 0. Concept coverage for generation tasks
        if metadata and isinstance(metadata, dict) and "concepts" in metadata:
            concepts = metadata["concepts"]
            if concepts:
                covered = sum(1 for c in concepts if _re.search(r'\b' + _re.escape(c.lower()), actual_norm))
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
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Example':
        return cls(**data)

@dataclass
class TextGradient:
    """Текстовый градиент. Описывает на естественном языке, что не так и как исправить"""
    failure_examples: List[Example] = field(default_factory=list) # Примеры, на которых промпт ошибся
    success_examples: List[Example] = field(default_factory=list) # Примеры, на которых промпт работает корректно
    error_analysis: str = ""                                      # Анализ ошибок
    suggested_direction: str = ""                                 # Предложенное направление улучшения
    specific_suggestions: List[str] = field(default_factory=list) # Конкретные рекомендации по изменению
    priority: float = 1.0                                         # Приоритет этого градиента (0.0 - 1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)        # Метаданные для анализа
    
    def to_dict(self) -> Dict:
        return {
            "failure_examples": [ex.to_dict() for ex in self.failure_examples],
            "success_examples": [ex.to_dict() for ex in self.success_examples],
            "error_analysis": self.error_analysis,
            "suggested_direction": self.suggested_direction,
            "specific_suggestions": self.specific_suggestions,
            "priority": self.priority,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TextGradient':
        data = data.copy()
        data["failure_examples"] = [Example.from_dict(ex) for ex in data["failure_examples"]]
        data["success_examples"] = [Example.from_dict(ex) for ex in data["success_examples"]]
        return cls(**data)

@dataclass
class EditOperation:
    """Операция редактирования промпта. Отслеживает, что и почему было изменено"""
    description: str                                # Описание изменения
    gradient_source: Optional[TextGradient] = None  # Градиент, вызвавший изменение
    before_snippet: str = ""                        # Фрагмент промпта до изменения
    after_snippet: str = ""                         # Фрагмент промпта после изменения
    
    def to_dict(self) -> Dict:
        return {
            "description": self.description,
            "gradient_source": self.gradient_source.to_dict() if self.gradient_source else None,
            "before_snippet": self.before_snippet,
            "after_snippet": self.after_snippet
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EditOperation':
        data = data.copy()
        data.pop("operation_type", None)
        if data.get("gradient_source"):
            data["gradient_source"] = TextGradient.from_dict(data["gradient_source"])
        return cls(**data)

@dataclass
class Metrics:
    """Метрики оценки промпта. Композитная оценка для ранжирования кандидатов.
    weights задаются scorer-ом при создании и определяют вклад каждой метрики
    в композитный score.  Если weights пуст — используется глобальный
    METRIC_WEIGHTS из config.
    """
    metrics: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)

    def composite_score(self) -> float:
        """Взвешенная сумма метрик: Σ(weight_i * metric_i)."""
        w = self.weights if self.weights else METRIC_WEIGHTS
        keys = set(list(self.metrics.keys()) + list(w.keys()))
        return sum(self.metrics.get(k, 0.0) * w.get(k, 0.0) for k in keys)

    def to_dict(self) -> Dict:
        d = self.metrics.copy()
        d["composite_score"] = self.composite_score()
        d["_weights"] = self.weights.copy()
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'Metrics':
        data = data.copy()
        data.pop("composite_score", None)
        weights = data.pop("_weights", {})
        m = cls()
        m.metrics = {k: float(v) for k, v in data.items()}
        m.weights = weights if isinstance(weights, dict) else {}
        return m

@dataclass
class PromptNode:
    """Узел в дереве эволюции промптов. Хранит полную информацию о промпте и его происхождении"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))                      # Уникальный идентификатор промпта
    prompt_text: str = ""                                                           # Текст промпта
    parent_id: Optional[str] = None                                                 # Идентификатор родительского промпта
    children_ids: List[str] = field(default_factory=list)                           # Идентификаторы дочерних промптов
    generation: int = 0                                                             # Номер поколения (глубина в дереве)
    source: OptimizationSource = OptimizationSource.INITIAL                         # Источник оптимизации
    operations: List[EditOperation] = field(default_factory=list)                   # История изменений
    metrics: Metrics = field(default_factory=Metrics)                               # Оценка промпта
    timestamp: datetime = field(default_factory=datetime.now)                       # Временная метка
    metadata: Dict[str, Any] = field(default_factory=dict)                          # Дополнительные метаданные
    is_evaluated: bool = False                                                      # Оценен ли промпт
    evaluation_examples: Dict[str, List[Example]] = field(default_factory=lambda: { # Примеры, на которых оценивался промпт
        "success": [],
        "failures": []
    })
    evaluation_examples_by_split: Dict[str, Dict[str, List[Example]]] = field(default_factory=dict)
    
    def add_child(self, child_id: str):
        """Добавление дочернего узла"""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def selection_score(self) -> float:
        try:
            return float(self.metadata["full_validation_score"])
        except Exception:
            return self.metrics.composite_score()

    def selection_accuracy(self) -> float:
        try:
            return float(self.metadata["full_validation_accuracy"])
        except Exception:
            return float(self.metrics.metrics.get("accuracy", 0.0))
    
    def to_dict(self) -> Dict:
        """Сериализация для сохранения"""
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
                "failures": [ex.to_dict() for ex in self.evaluation_examples["failures"]]
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
            "is_evaluated": self.is_evaluated
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PromptNode':
        """Десериализация из сохраненного формата"""
        data = data.copy()
        data["source"] = OptimizationSource(data["source"])
        data["operations"] = [EditOperation.from_dict(op) for op in data["operations"]]
        data["metrics"] = Metrics.from_dict(data["metrics"])
        data["evaluation_examples"] = {
            "success": [Example.from_dict(ex) for ex in data["evaluation_examples"]["success"]],
            "failures": [Example.from_dict(ex) for ex in data["evaluation_examples"]["failures"]]
        }
        raw_by_split = data.get("evaluation_examples_by_split", {})
        data["evaluation_examples_by_split"] = {
            split: {
                "success": [Example.from_dict(ex) for ex in by_split.get("success", [])],
                "failures": [Example.from_dict(ex) for ex in by_split.get("failures", [])],
            }
            for split, by_split in raw_by_split.items()
        }
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
