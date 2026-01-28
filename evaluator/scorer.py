from typing import List, Dict, Optional
from evaluator.metrics import MetricEvaluator, AccuracyMetric, F1ScoreMetric, SafetyMetric, RobustnessMetric, EfficiencyMetric
from prompts.templates import Templates
from data_structures import Example, Metrics, PromptNode
from llm.llm_client import BaseLLM
from functools import lru_cache
import random
from tqdm import tqdm
from config import METRIC_WEIGHTS, MAX_EXAMPLES_PER_NODE

class PromptScorer:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.max_examples_per_node = MAX_EXAMPLES_PER_NODE
        self.seed = 42
        
        # Регистрируем метрики
        self.metrics: Dict[str, MetricEvaluator] = {}
        for metric_cls in [AccuracyMetric, F1ScoreMetric, SafetyMetric, RobustnessMetric, EfficiencyMetric]:
            metric = metric_cls()
            if METRIC_WEIGHTS.get(metric.name, 0.0) > 0:
                self.metrics[metric.name] = metric

        self._last_eval_examples: Optional[List[Example]] = None

    @lru_cache(maxsize=10_000)
    def _cached_llm_call(self, prompt: str) -> str:
        return self.llm.invoke(prompt=prompt)
        
    def _sample_examples(self, examples: List[Example]) -> List[Example]:
        if len(examples) <= self.max_examples_per_node:
            return examples

        rnd = random.Random(self.seed)
        return rnd.sample(examples, self.max_examples_per_node)
    
    def execute_prompt_batch(self, prompt: str, examples: List[Example], progress_bar: bool = False) -> List[Example]:
        if progress_bar:
            for i, ex in enumerate(tqdm(examples)):
                ex.actual_output = self.execute_prompt(prompt, ex.input_text)
        else:
            for i, ex in enumerate(examples):
                ex.actual_output = self.execute_prompt(prompt, ex.input_text)

        return examples
    
    def execute_prompt(self, prompt: str, input_text: str) -> str:
        """Выполнение промпта на одном примере"""
        full_prompt = f"{prompt}\n\nInput:\n{input_text}"
        return self.llm.invoke(prompt=full_prompt)

    def evaluate_prompt(self, prompt: str,
                        examples: List[Example],
                        execute: bool = True,
                        sample: bool = True) -> Metrics:
        if sample:
            eval_examples = self._sample_examples(examples)
        else:
            eval_examples = examples
        
        if execute:
            eval_examples = [
                Example(input_text=ex.input_text, expected_output=ex.expected_output)
                for ex in eval_examples
            ]

            eval_examples = self.execute_prompt_batch(prompt, eval_examples, not sample)
            self._last_eval_examples = eval_examples

        metrics = Metrics()
        metrics.weights = METRIC_WEIGHTS.copy()

        for name, metric in self.metrics.items():
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
            if ex.actual_output and (ex.is_correct() or ex.is_correct_by_llm(self.llm)):
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