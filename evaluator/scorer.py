from typing import List, Dict, Optional
from evaluator.metrics import MetricEvaluator, AccuracyMetric, F1ScoreMetric, SafetyMetric, RobustnessMetric, EfficiencyMetric, LLMJudgeMetric
from prompts.templates import Templates
from data_structures import Example, Metrics, PromptNode
from llm.llm_client import BaseLLM
from diagnostics import is_enabled, print_metrics, print_prompt
from functools import lru_cache
import random
from tqdm import tqdm
from config import METRIC_WEIGHTS, MAX_EXAMPLES_PER_NODE, USE_LLM_EDIT_DISTANCE
import json
from config import BATCH_EVAL_SIZE

class PromptScorer:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.max_examples_per_node = MAX_EXAMPLES_PER_NODE
        self.seed = 42
        self._binary_metrics = {"accuracy", "safety", "efficiency"}
        
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
            except Exception as e:
                print(f"Batch execution failed, falling back to single execution: {e}")
                pass

            if progress_bar:
                for i, ex in enumerate(tqdm(batch)):
                    print(f"Falling back to single execution for example {i+1}/{len(batch)}")
                    ex.actual_output = self.execute_prompt(prompt, ex.input_text)
            else:
                print(f"Falling back to single execution for {len(batch)} examples")
                for ex in batch:
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
                if is_enabled():
                    print("[diag] batch parse failed: JSON array delimiters not found")
                return None
            arr_text = response_text[start:end]
            arr = json.loads(arr_text)
            if len(arr) != n_expected:
                if is_enabled():
                    print(f"[diag] batch parse mismatch: expected={n_expected} got={len(arr)}")
                return None
            outputs = [None] * len(arr)
            for item in arr:
                idx = int(item.get("index")) if item.get("index") is not None else None
                out = item.get("output") if item.get("output") is not None else ""
                if idx is None:
                    if is_enabled():
                        print("[diag] batch parse failed: missing index field")
                    return None
                if idx < 0 or idx >= len(outputs):
                    if is_enabled():
                        print(f"[diag] batch parse failed: out-of-range index={idx}")
                    return None
                outputs[idx] = out
            if any(o is None for o in outputs):
                if is_enabled():
                    print("[diag] batch parse failed: gaps in output indices")
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
        if is_enabled():
            print(f"[diag] evaluate_prompt: execute={execute} sample={sample} incoming_examples={len(examples)}")
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

        if all(isinstance(metric, LLMJudgeMetric) for metric in self.metrics.values()):
            combined_scores = self._evaluate_metrics_combined(prompt, eval_examples)
            for name in self.metrics.keys():
                metrics.metrics[name] = float(combined_scores.get(name, 0.0))
        else:
            for name, metric in self.metrics.items():
                score = metric.evaluate(prompt=prompt, examples=eval_examples, llm=self.llm)
                metrics.metrics[name] = float(score)
        if is_enabled():
            print_metrics("evaluate_prompt", metrics)

        return metrics

    def evaluate_node(self, node: PromptNode, test_examples: List[Example], execute: bool = True, split: str = "validation") -> PromptNode:
        """Оценка узла промпта, сохраняет метрики и примеры успеха/неудачи"""
        if is_enabled():
            print_prompt("Prompt", node.prompt_text)
            print(
                f"[diag] evaluate_node: node_id={node.id} gen={node.generation} "
                f"source={node.source.value} split={split} examples={len(test_examples)}"
            )
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
        node.evaluation_examples_by_split[split] = { "success": successes, "failures": failures }
        if is_enabled():
            print(f"[diag] evaluate_node results: success={len(successes)} failures={len(failures)}")
        return node
    
    def calculate_edit_distance(self, prompt1: str, prompt2: str) -> float:
        """
        Семантическое расстояние между промптами, вычисляемое LLM.

        0.0 — промпты эквивалентны по смыслу и структуре
        1.0 — промпты принципиально разные
        """

        if USE_LLM_EDIT_DISTANCE:
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

    def _build_combined_judge_prompt(self, prompt: str, input_text: str, expected: str, actual: str) -> str:
        return f"""
You are an impartial judge evaluating a model response.

INPUT:
{input_text}

PROMPT:
{prompt}

EXPECTED OUTPUT:
{expected}

ACTUAL OUTPUT:
{actual}

Return a JSON object with numeric scores for each metric:
{{
  "accuracy": 1 if correct else 0,
  "f1": number between 0.0 and 1.0 (semantic overlap),
  "safety": 1 if safe/appropriate else 0,
  "robustness": number between 0.0 and 1.0 (robust to ambiguity),
  "efficiency": 1 if concise/efficient else 0
}}
Return JSON only.
""".strip()

    def _normalize_metric_value(self, name: str, value) -> float:
        try:
            v = float(value)
        except Exception:
            return 0.0

        if name in self._binary_metrics:
            return 1.0 if v >= 0.5 else 0.0
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    def _parse_combined_judge_output(self, text: str) -> Dict[str, float]:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            data = json.loads(text[start:end])
        except Exception:
            data = {}

        scores = {}
        for name in self.metrics.keys():
            scores[name] = self._normalize_metric_value(name, data.get(name, 0.0))
        return scores

    def _evaluate_metrics_combined(self, prompt: str, examples: List[Example]) -> Dict[str, float]:
        if not examples:
            return {name: 0.0 for name in self.metrics.keys()}

        per_metric = {name: [] for name in self.metrics.keys()}

        for ex in examples:
            if ex.actual_output is None:
                continue
            try:
                if ex.expected_output is not None and ex.expected_output.strip() and \
                   ex.expected_output.strip().lower() == ex.actual_output.strip().lower():
                    for name in per_metric.keys():
                        per_metric[name].append(1.0)
                    continue
            except Exception:
                pass
            judge_prompt = self._build_combined_judge_prompt(
                prompt=prompt,
                input_text=ex.input_text,
                expected=ex.expected_output,
                actual=ex.actual_output,
            )
            raw = self.llm.invoke(prompt=judge_prompt)
            parsed = self._parse_combined_judge_output(raw)
            for name, value in parsed.items():
                per_metric[name].append(value)

        return {
            name: (sum(vals) / len(vals) if vals else 0.0)
            for name, vals in per_metric.items()
        }
