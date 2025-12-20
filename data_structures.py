from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
import uuid

class OptimizationSource(Enum):
    """Источник оптимизации промпта"""
    INITIAL = "initial"          # Начальный промпт
    LOCAL = "local"              # Локальная оптимизация 
    GLOBAL = "global"            # Глобальная оптимизация 
    MANUAL = "manual"            # Ручное редактирование

class OperationType(Enum):
    """Типы операций редактирования промпта"""
    ADD_INSTRUCTION = "add_instruction"        # Добавление инструкции
    MODIFY_INSTRUCTION = "modify_instruction"  # Изменение существующей инструкции
    REMOVE_INSTRUCTION = "remove_instruction"  # Удаление инструкции
    ADD_EXAMPLE = "add_example"                # Добавление примера
    REPHRASE = "rephrase"                      # Переформулировка
    RESTRUCTURE = "restructure"                # Структурная реорганизация
    ADD_CONSTRAINT = "add_constraint"          # Добавление ограничения
    CLARIFY = "clarify"                        # Уточнение формулировки

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
    operation_type: OperationType
    description: str                                # Описание изменения
    gradient_source: Optional[TextGradient] = None  # Градиент, вызвавший изменение
    before_snippet: str = ""                        # Фрагмент промпта до изменения
    after_snippet: str = ""                         # Фрагмент промпта после изменения
    
    def to_dict(self) -> Dict:
        return {
            "operation_type": self.operation_type.value,
            "description": self.description,
            "gradient_source": self.gradient_source.to_dict() if self.gradient_source else None,
            "before_snippet": self.before_snippet,
            "after_snippet": self.after_snippet
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EditOperation':
        data = data.copy()
        data["operation_type"] = OperationType(data["operation_type"])
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
    weights: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.5,
        "safety": 0.2,
        "robustness": 0.1,
        "efficiency": 0.1,
        "f1": 0.1
    })

    def composite_score(self) -> float:
        """Вычисление композитной оценки как взвешенной суммы метрик"""
        keys = set(list(self.metrics.keys()) + list(self.weights.keys()))
        return sum(self.metrics.get(k, 0.0) * self.weights.get(k, 0.0) for k in keys)
    
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
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt_text: str = ""
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    generation: int = 0                                                             # Номер поколения (глубина в дереве)
    source: OptimizationSource = OptimizationSource.INITIAL
    operations: List[EditOperation] = field(default_factory=list)                   # История изменений
    metrics: Metrics = field(default_factory=Metrics)                               # Оценка промпта
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_evaluated: bool = False                                                      # Оценен ли промпт
    is_front: bool = False                                                          # На фронте Парето (по нескольким метрикам)
    evaluation_examples: Dict[str, List[Example]] = field(default_factory=lambda: { # Примеры, на которых оценивался промпт
        "success": [],
        "failures": []
    })
    
    def add_child(self, child_id: str):
        """Добавление дочернего узла"""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
    
    def get_lineage_summary(self) -> str:
        """Краткая сводка о происхождении промпта. Полезно для передачи LLM в контексте оптимизации"""
        ops_summary = ", ".join([op.operation_type.value for op in self.operations[-3:]])
        return f"Gen {self.generation}, Source: {self.source.value}, Recent ops: [{ops_summary}], Score: {self.metrics.composite_score():.3f}"
    
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
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

@dataclass
class OptimizationConfig:
    """Конфигурация процесса оптимизации"""
    # Параметры локальной оптимизации
    local_iterations_per_generation: int = 5 # Количество локальных итераций 
    local_candidates_per_iteration: int = 3  # Количество редакций на итерацию (ширина локального поиска)
    local_batch_size: int = 10               # Размер мини-батча для градиентов (число редакций)

    # Параметры глобальной оптимизации
    global_trigger_interval: int = 3         # Каждые N поколений
    global_candidates: int = 3               # Ширина глобального поиска
    global_history_window: int = 20          # Сколько узлов истории анализировать
    
    # Общие параметры
    max_generations: int = 10                # Максимальное число поколений
    population_size: int = 5                 # Сколько промптов держим в активном пуле
    
    # Early stopping
    patience: int = 3                        # Поколений без улучшения до остановки
    min_improvement: float = 0.01            # Минимальное улучшение для продолжения
    
    # Diversity control
    diversity_bonus: float = 0.05            # Бонус за разнообразие
    similarity_threshold: float = 0.9        # Порог схожести
    
    # API параметры
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # Метрики и оценка
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.5,
        "safety": 0.2,
        "robustness": 0.2,
        "efficiency": 0.1,
        "f1": 0.0
    })
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OptimizationConfig':
        return cls(**data)