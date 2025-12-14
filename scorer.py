from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import re

from data_structures import (
    Example, 
    Metrics, 
    PromptNode,
    OptimizationConfig
)
from llm_client import create_llm_client

class MetricEvaluator(ABC):
    """Базовый класс для всех метрик"""
    
    def __init__(self, name: str, default_weight: float = 1.0):
        self.name = name
        self.default_weight = default_weight
    
    @abstractmethod
    def evaluate(self, prompt: str, examples: List[Example], **kwargs) -> float:
        """Возвращает score в диапазоне [0.0, 1.0]"""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, default_weight={self.default_weight})"

class AccuracyMetric(MetricEvaluator):
    """Метрика точности. Процент примеров, где actual_output соответствует expected_output"""
    
    def __init__(self, exact_match: bool = False, case_sensitive: bool = False):
        """
        Args:
            exact_match: Строгое совпадение или содержание
            case_sensitive: Учитывать регистр
        """
        super().__init__(name="accuracy", default_weight=0.5)
        self.exact_match = exact_match
        self.case_sensitive = case_sensitive
    
    def evaluate(self, prompt: str, examples: List[Example], scorer: Optional[Any] = None, **kwargs) -> float:
        if not examples:
            return 0.0
        
        correct_count = 0
        total = 0
        
        for example in examples:
            if example.actual_output is None:
                continue
            
            total += 1
            expected = example.expected_output or ""
            actual = example.actual_output or ""
            
            # Нормализация для сравнения
            if not self.case_sensitive:
                expected = expected.lower()
                actual = actual.lower()
            
            expected = expected.strip()
            actual = actual.strip()
            
            # Гарантированное присвоение is_correct в любом случае
            if self.exact_match:
                is_correct = (expected == actual)
            else:
                # Мягкая проверка: ожидаемое содержится в фактическом или наоборот
                is_correct = (expected in actual) or (actual in expected)
            
            if is_correct:
                correct_count += 1
        
        return correct_count / total if total > 0 else 0.0

class F1ScoreMetric(MetricEvaluator):
    """F1-метрика для задач классификации"""
    
    def __init__(self, average: str = "macro"):
        """
        Args:
            average: Тип усреднения ('macro', 'micro')
        """
        super().__init__(name="f1_score", default_weight=0.5)
        self.average = average
    
    def evaluate(self, prompt: str, examples: List[Example], **kwargs) -> float:
        if not examples:
            return 0.0
        
        # Собираем предсказания и истинные метки
        y_true = []
        y_pred = []
        
        for example in examples:
            if example.actual_output is None:
                continue
            
            y_true.append(example.expected_output.strip().lower())
            y_pred.append(example.actual_output.strip().lower())
        
        if not y_true:
            return 0.0
        
        # Вычисляем F1 для каждого класса
        labels = list(set(y_true + y_pred))
        f1_scores = []
        
        for label in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
        
        # Усреднение
        if self.average == "macro":
            return float(np.mean(f1_scores))
        elif self.average == "micro":
            labels = set(y_true + y_pred)
            tp = fp = fn = 0
            for label in labels:
                for t, p in zip(y_true, y_pred):
                    if p == label and t == label:
                        tp += 1
                    elif p == label and t != label:
                        fp += 1
                    elif p != label and t == label:
                        fn += 1
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

#TODO: стоит пересмотреть
class SafetyMetric(MetricEvaluator):
    """Метрика безопасности. Использует простую эвристику и паттерны"""
    
    def __init__(self):
        super().__init__(name="safety", default_weight=0.2)
        
        # Список токсичных паттернов
        self.toxic_patterns = [
            r'\b(hate|racist|sexist|offensive)\b',
            r'\b(kill|murder|violence)\b',
            r'\b(stupid|idiot|dumb)\b',
        ]
        
        # Список безопасных индикаторов
        self.safe_patterns = [
            r'\b(helpful|respectful|kind|polite)\b',
            r'\b(safe|appropriate|professional)\b',
        ]
    
    def evaluate(self, prompt: str, examples: List[Example], **kwargs) -> float:
        """
        Оценка безопасности
        1.0 = полностью безопасно, 0.0 = очень токсично
        """
        if not examples:
            # Оцениваем только сам промпт
            return self._evaluate_text(prompt)
        
        # Оцениваем выходы модели
        safety_scores = []
        
        for example in examples:
            if example.actual_output is None:
                continue
            
            score = self._evaluate_text(example.actual_output)
            safety_scores.append(score)
        
        # Также учитываем безопасность самого промпта
        prompt_safety = self._evaluate_text(prompt)
        
        if safety_scores:
            output_safety = np.mean(safety_scores)
            # Берем минимум - если хоть что-то небезопасно
            return min(prompt_safety, output_safety)
        else:
            return prompt_safety
    
    def _evaluate_text(self, text: str) -> float:
        """Оценка безопасности одного текста"""
        text_lower = text.lower()
        
        # Считаем токсичные паттерны
        toxic_count = 0
        for pattern in self.toxic_patterns:
            toxic_count += len(re.findall(pattern, text_lower))
        
        # Считаем безопасные паттерны
        safe_count = 0
        for pattern in self.safe_patterns:
            safe_count += len(re.findall(pattern, text_lower))
        
        #TODO: можно улучшить модель оценки
        # Простая формула: штраф за токсичность, бонус за безопасность
        if toxic_count > 0:
            score = max(0.0, 1.0 - 0.2 * toxic_count)
        else:
            score = min(1.0, 0.9 + 0.1 * safe_count)
        
        return score

#TODO: стоит пересмотреть
class RobustnessMetric(MetricEvaluator):
    """Метрика устойчивости - насколько стабильно работает промпт на разных вариациях входов (adversarial examples, perturbations)"""
    
    def __init__(self, perturbation_types: List[str] = None):
        """
        Args:
            perturbation_types: Типы возмущений для тестирования
        """
        super().__init__(name="robustness", default_weight=0.2)
        
        if perturbation_types is None:
            perturbation_types = ["typos", "case", "punctuation"]
        
        self.perturbation_types = perturbation_types
    
    def evaluate(self, prompt: str, examples: List[Example], **kwargs) -> float:
        """
        Оценка устойчивости
        Проверяем консистентность ответов на слегка измененных входах
        """
        if not examples:
            return 1.0  # Нет примеров - считаем устойчивым
        
        consistency_scores = []
        
        for example in examples:
            if example.actual_output is None:
                continue
            
            # Для упрощения - проверяем длину и структуру ответа
            #TODO: нужно генерировать возмущенные версии и сравнивать
            output_length = len(example.actual_output.split())
            expected_length = len(example.expected_output.split())
            
            # Проверка на адекватность длины
            length_ratio = output_length / max(expected_length, 1)
            length_score = 1.0 if 0.5 <= length_ratio <= 2.0 else 0.5
            
            # Проверка на наличие структуры
            has_substance = output_length > 3
            substance_score = 1.0 if has_substance else 0.3
            
            consistency_scores.append((length_score + substance_score) / 2)
        
        return float(np.mean(consistency_scores)) if consistency_scores else 1.0

#TODO: стоит пересмотреть
class EfficiencyMetric(MetricEvaluator):
    """Метрика эффективности - оценка краткости и лаконичности промпта"""
    
    def __init__(self, max_prompt_length: int = 1000, max_output_length: int = 500):
        """
        Args:
            max_prompt_length: Желаемая максимальная длина промпта
            max_output_length: Желаемая максимальная длина выхода
        """
        super().__init__(name="efficiency", default_weight=0.1)
        self.max_prompt_length = max_prompt_length
        self.max_output_length = max_output_length
    
    def evaluate(self, prompt: str, examples: List[Example], **kwargs) -> float:
        """
        Оценка эффективности
        1.0 = оптимальная длина, ниже = слишком длинный
        """
        # Оценка длины промпта (в токенах ~ слова * 1.3)
        prompt_tokens = len(prompt.split()) * 1.3
        prompt_score = min(1.0, self.max_prompt_length / max(prompt_tokens, 1))
        
        # Оценка длины выходов
        if examples:
            output_lengths = []
            for example in examples:
                if example.actual_output:
                    output_tokens = len(example.actual_output.split()) * 1.3
                    output_lengths.append(output_tokens)
            
            if output_lengths:
                avg_output = float(np.mean(output_lengths))
                output_score = min(1.0, self.max_output_length / max(avg_output, 1))
            else:
                output_score = 1.0
        else:
            output_score = 1.0
        
        # Комбинируем оценки
        return 0.6 * prompt_score + 0.4 * output_score

class PromptScorer:
    """Главный класс для оценки промптов. Управляет набором метрик и выполняет оценку"""
    
    def __init__(self, config: OptimizationConfig, api_config: Optional[Dict[str, str]] = None, custom_metrics: List[MetricEvaluator] = None):
        """
        Args:
            config: Конфигурация оптимизации
            api_config: Конфигурация API {"provider": "...", "{provider}_api_key": "..."}
            custom_metrics: Дополнительные пользовательские метрики
        """
        self.config = config
        self.api_config = api_config or {}
        
        # Инициализируем LLM клиент (через фабрику для согласованной инициализации)
        self.llm = create_llm_client(self.config, self.api_config)
        
        # Регистрируем стандартные метрики
        self.metrics: Dict[str, MetricEvaluator] = {}
        self.register_metric(AccuracyMetric())
        self.register_metric(F1ScoreMetric())
        self.register_metric(SafetyMetric())
        self.register_metric(RobustnessMetric())
        self.register_metric(EfficiencyMetric())
        
        # Добавляем пользовательские метрики
        if custom_metrics:
            for metric in custom_metrics:
                self.register_metric(metric)
         
        # Синхронизируем веса метрик с конфигом
        self._sync_metric_weights_with_config()
                
        # Статистика
        self.total_evaluations = 0
        # Последние выполненные примеры для последующей аналитики (предотвращает повторные вызовы API)
        self._last_eval_examples: Optional[List[Example]] = None
    
    def _sync_metric_weights_with_config(self):
        """Синхронизация весов метрик с конфигурацией"""
        for name, evaluator in self.metrics.items():
            if name not in self.config.metric_weights:
                self.config.metric_weights[name] = evaluator.default_weight
                
        # Нормализуем веса
        s = sum(self.config.metric_weights.values())
        if s <= 0:
            n = len(self.config.metric_weights) or 1
            for k in self.config.metric_weights:
                self.config.metric_weights[k] = 1.0 / n
        else:
            for k in self.config.metric_weights:
                self.config.metric_weights[k] = self.config.metric_weights[k] / s

    def register_metric(self, metric: MetricEvaluator):
        """
        Регистрация новой метрики
        
        Args:
            metric: Экземпляр MetricEvaluator
        """
        self.metrics[metric.name] = metric
        if metric.name not in self.config.metric_weights:
            self.config.metric_weights[metric.name] = metric.default_weight
        print(f"Registered metric: {metric} with weight {self.config.metric_weights[metric.name]:.3f}")
    
    def execute_prompt(self, prompt: str, input_text: str, max_tokens: int = 1000) -> str:
        """
        Выполнение промпта на одном примере
        
        Args:
            prompt: Текст промпта (система)
            input_text: Входные данные (пользователь)
            max_tokens: Максимум токенов в ответе
            
        Returns:
            Ответ модели
        """
        if self.llm.provider is None:
            raise ValueError("No provider configured for model execution")
        
        # Формируем полный промпт для LLM
        full_prompt = f"{prompt}\n\nInput: {input_text}"
        
        # Используем LLMClient для вызова
        text = self.llm.call(
            full_prompt,
            temperature=self.config.temperature,
            max_tokens=max_tokens
        )
        
        return text
    
    def execute_prompt_batch(self, prompt: str, examples: List[Example]) -> List[Example]:
        """
        Выполнение промпта на батче примеров
        
        Args:
            prompt: Текст промпта
            examples: Список примеров (без actual_output)
            
        Returns:
            Примеры с заполненным actual_output
        """
        if self.llm.provider is None:
            raise ValueError('No provider configured. Cannot execute prompts.')
        
        results: List[Example] = []
        errors: List[Tuple[int, Exception]] = []
        
        for idx, ex in enumerate(examples):
            try:
                out = self.execute_prompt(prompt, ex.input_text)
                ex.actual_output = out
            except Exception as e:
                ex.actual_output = None  # Оставляем None вместо маскирования ошибки
                errors.append((idx, e))
            
            results.append(ex)
        
        # Если были ошибки, логируем их (но не блокируем выполнение)
        if errors:
            print(f"⚠ {len(errors)} execution errors occurred:")
            for idx, err in errors[:3]:  # Показываем первые 3 ошибки
                print(f"  - Example {idx}: {str(err)[:100]}")
            if len(errors) > 3:
                print(f"  ... and {len(errors) - 3} more errors")
            
        return results

    
    def evaluate_prompt(self, prompt: str, examples: List[Example], execute: bool = True) -> Metrics:
        """
        Полная оценка промпта по всем метрикам
        
        Args:
            prompt: Текст промпта для оценки
            examples: Тестовые примеры
            execute: Выполнить ли промпт или использовать existing actual_output
            
        Returns:
            Объект Metrics со всеми оценками
        """
        # Выполняем промпт, если нужно - используем копию примеров чтобы не мутировать исходные
        eval_examples = examples
        if execute:
            # Копируем примеры перед модификацией и сохраняем их для последующего анализа
            from copy import deepcopy
            eval_examples = deepcopy(examples)
            eval_examples = self.execute_prompt_batch(prompt, eval_examples)
            # Сохраняем выполненные примеры для последующей аналитики (в evaluate_node и др.)
            self._last_eval_examples = eval_examples
            self._last_eval_prompt = prompt  # Помечаем, какой промпт был выполнен
        
        # Вычисляем каждую метрику
        metrics = Metrics(weights=self.config.metric_weights)
        
        for metric_name, evaluator in self.metrics.items():
            score = evaluator.evaluate(prompt, eval_examples)
            
            # Присваиваем значение соответствующему полю
            if hasattr(metrics, metric_name):
                setattr(metrics, metric_name, score)
            else:
                # Дополнительные метрики идут в extra_metrics
                metrics.extra_metrics[metric_name] = score
        
        self.total_evaluations += 1
        return metrics
    
    def evaluate_node(self, node: PromptNode, test_examples: List[Example], execute: bool = True) -> PromptNode:
        """
        Оценка узла промпта
        Обновляет metrics и evaluation_examples в узле
        
        Args:
            node: Узел для оценки
            test_examples: Тестовые примеры
            execute: Выполнить ли промпт
            
        Returns:
            Обновленный узел
        """
        # Оцениваем промпт
        metrics = self.evaluate_prompt(
            node.prompt_text,
            test_examples,
            execute=execute
        )
        
        # Обновляем узел
        node.metrics = metrics
        node.is_evaluated = True
        
        # Разделяем примеры на успехи и провалы
        successes = []
        failures = []

        # Используем сохранённые примеры из evaluate_prompt() чтобы не делать дублирующие API вызовы
        eval_examples = None
        if execute:
            # Если evaluate_prompt только что выполнял вызов API, используем кэшированные результаты
            # Проверяем: (1) кэш существует, (2) промпт совпадает, (3) количество примеров совпадает
            if (getattr(self, '_last_eval_examples', None) is not None and 
                getattr(self, '_last_eval_prompt', None) == node.prompt_text and
                len(self._last_eval_examples) == len(test_examples)):
                eval_examples = self._last_eval_examples
                print(f"  ✓ Using cached examples ({len(eval_examples)} evaluated)")
            else:
                # Fallback: выполняем снова
                from copy import deepcopy
                print(f"  ⚠ Cache miss - re-executing batch")
                if getattr(self, '_last_eval_prompt', None) != node.prompt_text:
                    print(f"    Reason: prompt changed")
                elif len(getattr(self, '_last_eval_examples', [])) != len(test_examples):
                    print(f"    Reason: size mismatch ({len(self._last_eval_examples)} vs {len(test_examples)})")
                eval_examples = deepcopy(test_examples)
                eval_examples = self.execute_prompt_batch(node.prompt_text, eval_examples)
        else:
            # Если execute=False, используем existing actual_output из test_examples
            eval_examples = test_examples

        # Собираем successes и failures
        for example in eval_examples:
            if example.actual_output is None:
                continue

            if example.is_correct():
                successes.append(example)
            else:
                failures.append(example)
        
        # Сохраняем в узел
        node.evaluation_examples = {
            "success": successes,
            "failures": failures
        }
        
        print(f"  Successes: {len(successes)}, Failures: {len(failures)}")
        
        return node
    
    def compare_prompts(self, nodes: List[PromptNode], test_examples: List[Example]) -> List[Tuple[PromptNode, Metrics]]:
        """
        Сравнение нескольких промптов
        
        Args:
            nodes: Список узлов для сравнения
            test_examples: Тестовые примеры
            
        Returns:
            Список (узел, метрики), отсортированный по composite_score
        """
        results = []
        
        for node in nodes:
            if not node.is_evaluated:
                self.evaluate_node(node, test_examples)
            results.append((node, node.metrics))
        
        # Сортируем по композитной метрике
        results.sort(key=lambda x: x[1].composite_score(), reverse=True)
        return results
    
    def get_failure_examples(self, node: PromptNode, test_examples: List[Example], top_k: int = 10) -> List[Example]:
        """
        Получение примеров, на которых промпт ошибся
        Используется для генерации градиентов
        
        Args:
            node: Узел промпта
            test_examples: Тестовые примеры
            top_k: Количество примеров для возврата
            
        Returns:
            Список примеров с ошибками
        """
        if not node.is_evaluated:
            self.evaluate_node(node, test_examples)
        
        failures = node.evaluation_examples.get('failures', [])
        
        # Берем top_k наиболее "интересных" провалов
        #TODO: можно добавить ранжирование по типу ошибки
        return failures[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Статистика работы scorer'а"""
        return {
            "total_evaluations": self.total_evaluations,
            "total_api_calls": getattr(self.llm, 'total_api_calls', 0),
            "registered_metrics": list(self.metrics.keys()),
            "metric_weights": self.config.metric_weights,
            "provider": self.llm.provider,
            "model": self.llm.model
        }
    
    def __repr__(self):
        return f"PromptScorer(metrics={len(self.metrics)}, evaluations={self.total_evaluations})"