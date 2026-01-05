from typing import List, Dict, Optional
from copy import deepcopy
from evaluator.metrics import MetricEvaluator, AccuracyMetric, F1ScoreMetric, SafetyMetric, RobustnessMetric, EfficiencyMetric
from prompts.templates import Templates
from data_structures import Example, Metrics, PromptNode
from llm.llm_client import BaseLLM
from config import METRIC_WEIGHTS

class PromptScorer:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        
        # Регистрируем метрики
        self.metrics: Dict[str, MetricEvaluator] = {}
        for metric_cls in [AccuracyMetric, F1ScoreMetric, SafetyMetric, RobustnessMetric, EfficiencyMetric]:
            metric = metric_cls()
            self.metrics[metric.name] = metric

        self._last_eval_examples: Optional[List[Example]] = None

    def execute_prompt(self, prompt: str, input_text: str) -> str:
        """Выполнение промпта на одном примере"""
        full_prompt = f"{prompt}\n\nInput:\n{input_text}"
        return self.llm.invoke(prompt=full_prompt)

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
        metrics.weights = METRIC_WEIGHTS.copy()

        for name, metric in self.metrics.items():
            weight = METRIC_WEIGHTS.get(name, 0.0)
            if weight <= 0:
                metrics.metrics[name] = 0.0
                continue
            score = metric.evaluate(prompt=prompt, examples=eval_examples, llm=self.llm)
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
    
    def calculate_edit_distance(self, prompt1: str, prompt2: str) -> float:
        """
        Семантическое расстояние между промптами, вычисляемое LLM.

        0.0 — промпты эквивалентны по смыслу и структуре
        1.0 — промпты принципиально разные
        """

        template = Templates.load_template("evaluation")
        prompt = template.format(prompt1=prompt1, prompt2=prompt2)
        
        try:
            result = self.llm.invoke(prompt=prompt)
            return max(0.0, min(1.0, float(result)))
        except Exception as e:
            print(f"LLM distance evaluation failed, falling back: {e}")

            words1 = set(prompt1.lower().split())
            words2 = set(prompt2.lower().split())
            if not words1 and not words2:
                return 0.0
            similarity = len(words1 & words2) / max(len(words1 | words2), 1)
            return 1.0 - similarity

    def calculate_pairwise_metric(self, nodes: List[PromptNode], max_distance_pairs: int) -> List[float]:
        values = []
        for i in range(min(len(nodes), max_distance_pairs)):
            for j in range(i + 1, min(len(nodes), max_distance_pairs)):
                values.append(self.calculate_edit_distance(nodes[i].prompt_text, nodes[j].prompt_text))
        return values