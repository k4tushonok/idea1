from typing import List, Dict, Optional
from evaluator.metrics import MetricEvaluator, AccuracyMetric, F1ScoreMetric, SafetyMetric, RobustnessMetric, EfficiencyMetric
from prompts.templates import Templates
from data_structures import Example, Metrics, PromptNode
from llm.llm_client import BaseLLM
from functools import lru_cache
import random
from tqdm import tqdm
from config import METRIC_WEIGHTS, MAX_EXAMPLES_PER_NODE
import json
from config import BATCH_EVAL_SIZE

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
        
        try:
            self.llm.invoke = lru_cache(maxsize=10000)(self.llm.invoke)
        except Exception:
            print("Failed to apply LLM invoke caching")
            pass
        
    def _sample_examples(self, examples: List[Example]) -> List[Example]:
        if len(examples) <= self.max_examples_per_node:
            return examples

        rnd = random.Random(self.seed)
        return rnd.sample(examples, self.max_examples_per_node)
    
    def execute_prompt_batch(self, prompt: str, examples: List[Example], progress_bar: bool = False) -> List[Example]:
        if len(examples) <= 1:
            for i, ex in enumerate(examples):
                ex.actual_output = self.execute_prompt(prompt, ex.input_text)
            return examples

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        for batch in chunks(examples, BATCH_EVAL_SIZE):
            batch_prompt = self._build_batch_prompt(prompt, batch)
            try:
                raw = self.llm.invoke(prompt=batch_prompt)
                parsed = self._parse_batch_response(raw, len(batch))
                if parsed and len(parsed) == len(batch):
                    for ex, out in zip(batch, parsed):
                        ex.actual_output = out
                    continue
            except Exception:
                print("Batch execution failed, falling back to single execution")
                pass

            if progress_bar:
                for i, ex in enumerate(tqdm(batch)):
                    print(f"Falling back to single execution for example {i+1}/{len(batch)}")
                    ex.actual_output = self.execute_prompt(prompt, ex.input_text)
            else:
                for ex in batch:
                    print("Falling back to single execution for one example")
                    ex.actual_output = self.execute_prompt(prompt, ex.input_text)

        return examples

    def _build_batch_prompt(self, prompt: str, examples: List[Example]) -> str:
        header = (
            "For each input below, produce the model OUTPUT for that input when using the given PROMPT. "
            "Return a JSON array of objects with keys 'index' (int) and 'output' (string).\n\n"
        )
        body = [header, "PROMPT:\n", prompt, "\n\n"]
        for i, ex in enumerate(examples):
            body.append(f"INPUT {i}:\n{ex.input_text}\n\n")
        body.append("Return JSON now:")
        return "".join(body)

    def _parse_batch_response(self, response_text: str, n_expected: int) -> Optional[List[str]]:
        try:
            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            if start == -1 or end == -1:
                return None
            arr_text = response_text[start:end]
            arr = json.loads(arr_text)
            outputs = [None] * len(arr)
            for item in arr:
                idx = int(item.get("index")) if item.get("index") is not None else None
                out = item.get("output") if item.get("output") is not None else ""
                if idx is None:
                    return None
                outputs[idx] = out
            if any(o is None for o in outputs):
                return None
            return outputs
        except Exception:
            print("Failed to parse batch response")
            return None
    
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