from typing import List, Dict, Optional
from copy import deepcopy
from evaluator.metrics import (
    MetricEvaluator,
    AccuracyMetric,
    F1ScoreMetric,
    SafetyMetric,
    RobustnessMetric,
    EfficiencyMetric
)
from data_structures import Example, Metrics, PromptNode, OptimizationConfig
from llm_client import create_llm_client

class PromptScorer:
    def __init__(self, config: OptimizationConfig, api_config: Optional[Dict[str, str]] = None):
        self.config = config
        self.api_config = api_config or {}

        # Основной LLM для генерации ответов
        self.llm = create_llm_client(self.config, self.api_config)
        # LLM для оценки
        self.judge_llm = create_llm_client(self.config, self.api_config)

        # Регистрируем метрики
        self.metrics: Dict[str, MetricEvaluator] = {}
        for metric_cls in [AccuracyMetric, F1ScoreMetric, SafetyMetric, RobustnessMetric, EfficiencyMetric]:
            metric = metric_cls(config=self.config)
            self.metrics[metric.name] = metric

        self._sync_metric_weights()
        self._last_eval_examples: Optional[List[Example]] = None

    def _sync_metric_weights(self):
        """
            Нормализует веса метрик на основе config.metric_weights
            Если веса не заданы или сумма ≤ 0, задаёт равные веса
            Иначе нормализует их так, чтобы сумма всех весов была 1
        """
        n = len(self.metrics) or 1
        weights: Dict[str, float] = {}
        
        for name in self.metrics:
            weights[name] = self.config.metric_weights.get(name, 1.0 / n)
        
        total = sum(weights.values())
        if total <= 0:
            for name in weights:
                weights[name] = 1.0 / n
        else:
            for name in weights:
                weights[name] /= total

        self.config.metric_weights = weights

    def execute_prompt(self, prompt: str, input_text: str) -> str:
        """Выполнение промпта на одном примере"""
        full_prompt = f"{prompt}\n\nInput:\n{input_text}"
        return self.llm.call(full_prompt, temperature=self.config.temperature, max_tokens=self.config.max_tokens)

    def execute_prompt_batch(self, prompt: str, examples: List[Example]) -> List[Example]:
        """Применяет промпт ко всему списку примеров. Для каждого Example сохраняет actual_output"""
        results = []

        for ex in examples:
            ex.actual_output = self.execute_prompt(prompt, ex.input_text)
            results.append(ex)

        return results

    def evaluate_prompt(self, prompt: str, examples: List[Example], execute: bool = True) -> Metrics:
        eval_examples = examples 
        
        if execute: 
            eval_examples = deepcopy(examples) 
            eval_examples = self.execute_prompt_batch(prompt, eval_examples) 
            self._last_eval_examples = eval_examples

        # Создаём объект Metrics и задаём веса из конфига
        metrics = Metrics()
        metrics.weights = self.config.metric_weights.copy()

        for name, metric in self.metrics.items():
            weight = self.config.metric_weights.get(name, 0.0)
            if weight <= 0:
                metrics.metrics[name] = 0.0
                continue
            score = metric.evaluate(prompt=prompt, examples=eval_examples, judge_llm=self.judge_llm)
            metrics.metrics[name] = float(score)

        return metrics

    def evaluate_node(self, node: PromptNode, test_examples: List[Example], execute: bool = True) -> PromptNode:
        """Оценка узла промпта, сохраняет метрики и примеры успеха/неудачи"""
        metrics = self.evaluate_prompt(node.prompt_text, test_examples, execute=execute)
        node.metrics = metrics
        node.is_evaluated = True

        eval_examples = self._last_eval_examples or []

        # Разделяем успешные и неуспешные примеры
        successes, failures = [], []
        for ex in eval_examples:
            if ex.actual_output and ex.is_correct():
                successes.append(ex)
            else:
                failures.append(ex)

        node.evaluation_examples = { "success": successes, "failures": failures }
        return node