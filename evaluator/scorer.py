"""
Prompt scorer.

Executes prompts on examples (individually and in batches),
computes metrics, caches results, and splits examples into
successes and failures for further analysis.
"""

from typing import List, Dict, Optional, Tuple
from copy import deepcopy
import statistics
import random
import json
import hashlib
import time
from tqdm import tqdm

from evaluator.metrics import MetricEvaluator, METRIC_REGISTRY
from prompts.templates import Templates
from data_structures import Example, Metrics, PromptNode
from llm.llm_client import BaseLLM
from diagnostics import (
    is_enabled,
    print_prompt,
    llm_calls,
    format_stage_weights,
    print_eval_summary,
)
from config import (
    METRICS_CONFIG,
    MAX_EXAMPLES_PER_NODE,
    BATCH_EVAL_SIZE,
    EVAL_TEMPERATURE,
    EVAL_SEED,
)

def _iter_chunks(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]





class PromptScorer:
    """Prompt scorer: execution, metrics, and caching.

    Supports batched execution, multi-stage metrics,
    an evaluation cache, and pairwise distances for deduplication.
    """

    def __init__(self, llm: BaseLLM, metrics_config: Optional[List[Dict]] = None):
        self.llm = llm
        self.max_examples_per_node = MAX_EXAMPLES_PER_NODE
        self.seed = EVAL_SEED

        self._metrics_config: List[Dict] = metrics_config or METRICS_CONFIG

        self._metric_instances: Dict[str, MetricEvaluator] = {}

        for cfg in self._metrics_config:
            name = cfg["name"]
            cls = METRIC_REGISTRY.get(name)
            if cls is None:
                print(f"[scorer] WARNING: unknown metric '{name}' — skipped")
                continue
            self._metric_instances[name] = cls()

        self._weights_by_max_stage: Dict[int, Dict[str, float]] = {}
        for max_stage in (1, 2, 3):
            active = [m for m in self._metrics_config if m["stage"] <= max_stage]
            self._weights_by_max_stage[max_stage] = {
                m["name"]: m["weight"] for m in active
            }

        self._last_eval_examples: Optional[List[Example]] = None

        self._eval_cache: Dict[
            Tuple[str, str, int], Tuple["Metrics", Optional[List[Example]]]
        ] = {}
        self._edit_distance_cache: Dict[frozenset, float] = {}

    @staticmethod
    def _hash_prompt(prompt: str) -> str:
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_examples(examples: List[Example]) -> str:
        """Fast hash of an example set based on input_text, used to identify a batch."""
        h = hashlib.md5()
        for ex in examples:
            h.update(ex.input_text[:200].encode("utf-8"))
        return h.hexdigest()

    def clear_eval_cache(self):
        """Clear evaluation and edit-distance caches (called between generations)."""
        self._eval_cache.clear()
        self._edit_distance_cache.clear()

    def _sample_examples(
        self, examples: List[Example], seed_offset: int = 0
    ) -> List[Example]:
        """Sample a subset of examples for evaluation."""
        if len(examples) <= self.max_examples_per_node:
            return examples
        rnd = random.Random(self.seed + seed_offset)
        return rnd.sample(examples, self.max_examples_per_node)

    def execute_prompt_batch(
        self, prompt: str, examples: List[Example], progress_bar: bool = False
    ) -> List[Example]:
        """Execute a prompt on an array of examples using batch LLM calls."""
        if len(examples) <= 1:
            if is_enabled():
                print(f"[diag] execute_prompt_batch: single example mode")
            for ex in examples:
                ex.actual_output = self.execute_prompt(
                    prompt,
                    ex.input_text,
                    temperature=EVAL_TEMPERATURE,
                )
            return examples

        total = len(examples)
        batch_list = list(_iter_chunks(examples, BATCH_EVAL_SIZE))
        n_batches = len(batch_list)
        calls_before = llm_calls(self.llm)

        if is_enabled():
            print(
                f"[diag] execute_prompt_batch: total={total} "
                f"batch_size={BATCH_EVAL_SIZE} n_batches={n_batches} "
                f"llm_calls_before={calls_before}"
            )

        done = 0
        for b_idx, batch in enumerate(batch_list, 1):
            batch_prompt = self._build_batch_prompt(prompt, batch)
            parsed = None
            try:
                raw = self.llm.invoke(
                    prompt=batch_prompt,
                    temperature=EVAL_TEMPERATURE,
                )
                parsed = self._parse_batch_response(raw, len(batch))
            except Exception as e:
                print(f"Batch execution failed, falling back to single execution: {e}")

            if parsed is not None:
                filled = 0
                missing_indices = []
                for i, (ex, out) in enumerate(zip(batch, parsed)):
                    if out is not None:
                        ex.actual_output = out
                        filled += 1
                    else:
                        missing_indices.append(i)

                if not missing_indices:
                    done += len(batch)
                    if is_enabled():
                        print(
                            f"[diag]   batch {b_idx}/{n_batches}: OK "
                            f"({done}/{total} done, llm_calls={llm_calls(self.llm)})"
                        )
                    continue

                if is_enabled():
                    print(
                        f"[diag]   batch {b_idx}/{n_batches}: PARTIAL "
                        f"({filled}/{len(batch)} from batch, "
                        f"falling back for {len(missing_indices)} missing)"
                    )
                for idx in missing_indices:
                    batch[idx].actual_output = self.execute_prompt(
                        prompt,
                        batch[idx].input_text,
                        temperature=EVAL_TEMPERATURE,
                    )
                done += len(batch)
                continue

            if is_enabled():
                print(
                    f"[diag]   batch {b_idx}/{n_batches}: FALLBACK to single execution ({len(batch)} examples)"
                )

            if progress_bar:
                for i, ex in enumerate(tqdm(batch)):
                    print(
                        f"Falling back to single execution for example {i+1}/{len(batch)}"
                    )
                    ex.actual_output = self.execute_prompt(
                        prompt,
                        ex.input_text,
                        temperature=EVAL_TEMPERATURE,
                    )
            else:
                print(f"Falling back to single execution for {len(batch)} examples")
                for ex in batch:
                    ex.actual_output = self.execute_prompt(
                        prompt,
                        ex.input_text,
                        temperature=EVAL_TEMPERATURE,
                    )
            done += len(batch)

        if is_enabled():
            calls_after = llm_calls(self.llm)
            print(
                f"[diag] execute_prompt_batch done: "
                f"llm_calls_delta={calls_after - calls_before} total_calls={calls_after}"
            )

        return examples

    def _build_batch_prompt(self, prompt: str, examples: List[Example]) -> str:
        """Build a batched prompt for simultaneous processing of multiple examples."""
        header = (
            "You must emulate multiple completely independent assistant calls.\n"
            "Each item below is a separate fresh interaction with no shared memory.\n"
            "For item i, pretend the assistant receives exactly one user message and produces exactly one assistant reply.\n"
            "That user message is shown verbatim between ITEM i USER MESSAGE START and END.\n"
            "For every item, generate the same reply you would produce if that item were sent alone in a separate call.\n"
            "Do not compare items, do not combine contexts, do not reuse information across items, and do not add explanations.\n"
            "Return ONLY a valid JSON array. Each item must be an object with exactly two keys:\n"
            '- "index": the integer input index\n'
            '- "output": the exact assistant reply string for that independent item\n\n'
            "The array must contain exactly one object for every input index from 0 to N-1.\n"
        )
        body = [header, "\n"]
        for i, ex in enumerate(examples):
            body.append(
                f"ITEM {i} USER MESSAGE START\n"
                f"{prompt}\n\nInput:\n{ex.input_text}\n"
                f"ITEM {i} USER MESSAGE END\n\n"
            )
        body.append("Return the JSON array now:")
        return "".join(body)

    def _parse_batch_response(
        self, response_text: str, n_expected: int
    ) -> Optional[List[str]]:
        """Parse the JSON response from a batch execution into a list of answers."""
        try:
            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            if start == -1 or end == -1:
                if is_enabled():
                    print("[diag] batch parse failed: JSON array delimiters not found")
                return None
            arr = json.loads(response_text[start:end])

            outputs: List[Optional[str]] = [None] * n_expected
            good = 0
            for item in arr:
                idx = int(item.get("index")) if item.get("index") is not None else None
                out = item.get("output") if item.get("output") is not None else ""
                if idx is None or idx < 0 or idx >= n_expected:
                    if is_enabled():
                        print(f"[diag] batch parse: skipping bad index={idx}")
                    continue
                outputs[idx] = out
                good += 1

            if good == 0:
                if is_enabled():
                    print("[diag] batch parse failed: no valid entries")
                return None

            if good < n_expected and is_enabled():
                missing = [i for i, o in enumerate(outputs) if o is None]
                print(
                    f"[diag] batch parse partial: {good}/{n_expected} valid, "
                    f"missing indices={missing}"
                )

            return outputs
        except Exception as e:
            if is_enabled():
                print(f"[diag] batch parse exception: {e}")
            print("Failed to parse batch response")
            return None

    def execute_prompt(
        self, prompt: str, input_text: str, temperature: Optional[float] = None
    ) -> str:
        """Execute a prompt on a single example input."""
        full_prompt = f"{prompt}\n\nInput:\n{input_text}"
        effective_temperature = (
            EVAL_TEMPERATURE if temperature is None else temperature
        )
        return self.llm.invoke(prompt=full_prompt, temperature=effective_temperature)

    def evaluate_prompt(
        self,
        prompt: str,
        examples: List[Example],
        execute: bool = True,
        sample: bool = True,
        seed_offset: int = 0,
        stage: int = 2,
    ) -> Metrics:
        """Evaluate a prompt: execute it on examples and compute all active metrics.

        Supports caching, example sampling, and multi-stage metrics
        with weight normalization.
        """
        _calls_start = llm_calls(self.llm)
        _t0 = time.time()

        if is_enabled():
            will_use = (
                min(len(examples), self.max_examples_per_node)
                if sample
                else len(examples)
            )
            print(
                f"[diag] evaluate_prompt: execute={execute} sample={sample} "
                f"stage={stage} incoming={len(examples)} will_use={will_use} "
                f"weights={format_stage_weights(self._weights_by_max_stage.get(stage, {}))} "
                f"llm_calls_at_start={_calls_start}"
            )
        if sample:
            eval_examples = self._sample_examples(examples, seed_offset)
        else:
            eval_examples = examples

        p_hash = self._hash_prompt(prompt)
        e_hash = self._hash_examples(eval_examples)
        cache_key = (p_hash, e_hash, stage)
        cached = self._eval_cache.get(cache_key)
        if cached is not None:
            cached_metrics, cached_examples = cached
            if is_enabled():
                print(f"[diag] evaluate_prompt: cache HIT (stage={stage})")
            self._last_eval_examples = (
                deepcopy(cached_examples) if cached_examples is not None else None
            )
            return cached_metrics

        if execute:
            eval_examples = [
                Example(
                    input_text=ex.input_text,
                    expected_output=ex.expected_output,
                    metadata=dict(ex.metadata) if ex.metadata else {},
                )
                for ex in eval_examples
            ]

            eval_examples = self.execute_prompt_batch(prompt, eval_examples, not sample)
            self._last_eval_examples = eval_examples

        if is_enabled():
            _calls_after_exec = llm_calls(self.llm)
            print(
                f"[diag] evaluate_prompt execution done: "
                f"llm_calls_for_execution={_calls_after_exec - _calls_start}"
            )
            for idx, ex in enumerate(eval_examples[:3]):
                out_preview = (ex.actual_output or "<None>")[:80]
                exp_preview = (ex.expected_output or "<None>")[:80]
                print(
                    f"[diag]   example[{idx}] expected='{exp_preview}' actual='{out_preview}'"
                )
            if len(eval_examples) > 3:
                print(f"[diag]   ... and {len(eval_examples) - 3} more examples")

        metrics = Metrics()
        active_metric_names = set()

        metrics_to_eval = []

        for name, instance in self._metric_instances.items():
            cfg = next((c for c in self._metrics_config if c["name"] == name), None)
            if cfg is None or cfg["stage"] > stage:
                if is_enabled():
                    print(
                        f"[diag]   metric '{name}' skipped (stage {cfg['stage'] if cfg else '?'} > {stage})"
                    )
                continue
            if not instance.supports_examples(eval_examples):
                if is_enabled():
                    print(
                        f"[diag]   metric '{name}' skipped (unsupported for current examples)"
                    )
                continue
            metrics_to_eval.append((name, instance, cfg))
            active_metric_names.add(name)

        for name, instance, cfg in metrics_to_eval:
            _m_t0 = time.time()
            score = instance.evaluate(prompt=prompt, examples=eval_examples)
            _m_elapsed = time.time() - _m_t0
            metrics.metrics[name] = float(score)
            if is_enabled():
                _m_calls = llm_calls(self.llm)
                print(
                    f"[diag]   metric '{name}': {score:.4f} "
                    f"(stage={cfg['stage']} weight={cfg['weight']} "
                    f"time={_m_elapsed:.2f}s llm_calls={_m_calls})"
                )

        # Normalise weights so composite ∈ [0, 1] for any stage
        active_weights = {
            m["name"]: m["weight"]
            for m in self._metrics_config
            if m["stage"] <= stage and m["name"] in active_metric_names
        }

        total_w = sum(active_weights.values())
        if total_w > 0 and abs(total_w - 1.0) > 1e-6:
            metrics.weights = {k: v / total_w for k, v in active_weights.items()}
        else:
            metrics.weights = active_weights.copy()

        _elapsed = time.time() - _t0
        _calls_end = llm_calls(self.llm)

        if is_enabled():
            print_eval_summary(
                f"evaluate_prompt",
                metrics,
                stage,
                _calls_start,
                _calls_end,
                _elapsed,
            )

        self._eval_cache[cache_key] = (
            metrics,
            deepcopy(eval_examples) if execute else None,
        )

        return metrics

    def evaluate_node(
        self,
        node: PromptNode,
        test_examples: List[Example],
        execute: bool = True,
        split: str = "validation",
        seed_offset: int = 0,
        stage: int = 1,
    ) -> PromptNode:
        """Evaluate a node fully: execution + metrics + success/failure split."""
        _node_t0 = time.time()
        _node_calls_start = llm_calls(self.llm)

        if is_enabled():
            print_prompt("Prompt", node.prompt_text)
            print(
                f"[diag] evaluate_node: node_id={node.id} gen={node.generation} "
                f"source={node.source.value} split={split} stage={stage} "
                f"examples={len(test_examples)} seed_offset={seed_offset}"
            )

        metrics = self.evaluate_prompt(
            node.prompt_text,
            test_examples,
            execute=execute,
            seed_offset=seed_offset,
            stage=stage,
        )
        node.metrics = metrics
        node.is_evaluated = True

        eval_examples = self._last_eval_examples or []

        # Split into successes and failures
        successes, failures = [], []
        for ex in eval_examples:
            if ex.is_success_for_optimization():
                successes.append(ex)
            else:
                failures.append(ex)

        node.evaluation_examples = {"success": successes, "failures": failures}
        node.evaluation_examples_by_split[split] = {
            "success": successes,
            "failures": failures,
        }

        _node_elapsed = time.time() - _node_t0
        _node_calls_end = llm_calls(self.llm)

        if is_enabled():
            print(
                f"[diag] evaluate_node results: success={len(successes)} "
                f"failures={len(failures)} total={len(successes) + len(failures)} "
                f"composite={metrics.composite_score():.4f} selection={node.selection_score():.4f} "
                f"stage={stage} llm_calls={_node_calls_end - _node_calls_start} "
                f"time={_node_elapsed:.2f}s"
            )
            for name, value in sorted(metrics.metrics.items()):
                w = metrics.weights.get(name, 0.0)
                print(
                    f"[diag]   {name}={value:.4f}  weight={w}  contribution={value * w:.4f}"
                )
        return node

    def calculate_edit_distance(self, prompt1: str, prompt2: str) -> float:
        """Jaccard distance between two prompts based on word-bag overlap."""
        h1 = self._hash_prompt(prompt1)
        h2 = self._hash_prompt(prompt2)
        pair_key = frozenset((h1, h2))
        if pair_key in self._edit_distance_cache:
            return self._edit_distance_cache[pair_key]

        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())
        if not words1 and not words2:
            dist = 0.0
        else:
            similarity = len(words1 & words2) / max(len(words1 | words2), 1)
            dist = 1.0 - similarity
        self._edit_distance_cache[pair_key] = dist
        return dist

    def calculate_pairwise_metric(
        self, nodes: List[PromptNode], max_distance_pairs: int
    ) -> List[float]:
        """Pairwise Jaccard distances between prompt nodes."""
        values = []
        for i in range(min(len(nodes), max_distance_pairs)):
            for j in range(i + 1, min(len(nodes), max_distance_pairs)):
                values.append(
                    self.calculate_edit_distance(
                        nodes[i].prompt_text, nodes[j].prompt_text
                    )
                )
        return values
