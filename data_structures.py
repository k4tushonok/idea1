from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
import uuid

from config import (
    METRIC_WEIGHTS,
    LINEAGE_RECENT_OPS_LIMIT,
    USE_LLM_CORRECTNESS_CHECK,
    LOCAL_SIMILARITY_THRESHOLD,
)

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
    
    def is_correct(self) -> bool:
        """Проверка корректности ответа"""
        if self.actual_output is None:
            return False
        return self.expected_output.strip().lower() == self.actual_output.strip().lower()

    def is_correct_by_llm(self, llm)-> bool:
        """Проверка корректности ответа"""
        if self.actual_output is None:
            return False
        
        try:
            if self.expected_output is not None and self.expected_output.strip() and \
               self.expected_output.strip().lower() == self.actual_output.strip().lower():
                return True
        except Exception:
            print("LLM correctness evaluation failed during direct comparison")
            pass

        if not USE_LLM_CORRECTNESS_CHECK:
            return self._is_similar_locally(self.actual_output, self.expected_output)

        prompt = (f"There are two answers on the same question. "
                  f"'Expected output' is a true answer used as label during dataset training. "
                  f"'Actual answer' is an answer of LLM model. "
                  f"You need to estimate semantic similarity of these answers. "
                  f"Return 'Yes' if these answers are semantically the same (for example: '1984' and 'It was in 1984'), "
                  f"and return 'No' otherwise.\n"
                  f"# Output format: return 'Yes' or 'No'\n"
                  f"# Answers to compare:\n"
                  f"- Actual answer: {self.actual_output}\n"
                  f"- Expected output: {self.expected_output}\n")
        try:
            response = llm.invoke(prompt)
            return response.strip().lower() == 'yes'
        except Exception:
            print("LLM correctness evaluation failed")
            return False

    @staticmethod
    def _is_similar_locally(actual: str, expected: str) -> bool:
        if actual is None or expected is None:
            return False
        a = [t for t in actual.lower().split() if t]
        e = [t for t in expected.lower().split() if t]
        if not a or not e:
            return False
        set_a = set(a)
        set_e = set(e)
        overlap = len(set_a & set_e)
        union = len(set_a | set_e)
        if union == 0:
            return False
        return (overlap / union) >= LOCAL_SIMILARITY_THRESHOLD

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
    """Метрики оценки промпта. Композитная оценка для ранжирования кандидатов"""
    metrics: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.0,
        "safety": 0.0,
        "robustness": 0.0,
        "efficiency": 0.0,
        "f1": 0.0
    })

    def composite_score(self) -> float:
        """Вычисление композитной оценки как взвешенной суммы метрик"""
        keys = set(list(self.metrics.keys()) + list(METRIC_WEIGHTS.keys()))
        return sum(self.metrics.get(k, 0.0) * METRIC_WEIGHTS.get(k, 0.0) for k in keys)

    def to_dict(self) -> Dict:
        d = self.metrics.copy()
        d["composite_score"] = self.composite_score()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Metrics':
        data = data.copy()
        data.pop("composite_score", None)
        m = cls()
        m.metrics = {k: float(v) for k, v in data.items()}
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
    is_front: bool = False                                                          # На фронте Парето (по нескольким метрикам)
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
    
    def get_lineage_summary(self) -> str:
        """Краткая сводка о происхождении промпта. Полезно для передачи LLM в контексте оптимизации"""
        ops_summary = ", ".join([op.description for op in self.operations[-LINEAGE_RECENT_OPS_LIMIT:]])
        return f"Gen {self.generation}, Source: {self.source.value}, Recent ops: [{ops_summary}], Score: {self.selection_score():.3f}"
    
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
            "is_evaluated": self.is_evaluated,
            "is_front": self.is_front
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
