from typing import List, Dict, Optional
import hashlib
import numpy as np
from collections import Counter
import time
import statistics
from prompts.templates import Templates
from llm.llm_client import BaseLLM
from data_structures import Example, PromptNode, OptimizationSource, EditOperation
from history_manager import HistoryManager
from evaluator.scorer import PromptScorer
from prompt_editor import PromptEditor
from llm.llm_response_parser import MarkdownParser
from diagnostics import is_enabled, prompt_id, scores_summary, print_candidates_summary
from config import (
    TOP_BEST_NODES,
    MAX_DISTANCE_PAIRS,
    STAGNATION_SIMILARITY_THRESHOLD,
    GLOBAL_CANDIDATES,
    SIMILARITY_THRESHOLD,
    GLOBAL_TRIGGER_INTERVAL,
    MIN_IMPROVEMENT,
    GLOBAL_MIN_IMPROVEMENT,
    GLOBAL_HISTORY_WINDOW,
    EXEMPLAR_COUNT,
    HISTORY_SCORE_THRESHOLD,
    EXEMPLAR_SELECTION_STRATEGY,
    GLOBAL_TEMPERATURE,
    MAX_INSTRUCTION_LENGTH,
    RECENT_GENERATIONS_FOR_DIVERSITY,
    DIVERSITY_DISTANCE_THRESHOLD,
    COMMON_WORDS_TOP_K,
    COMMON_WORD_MIN_FREQ,
)


class GlobalOptimizer:
    def __init__(
        self,
        history_manager: HistoryManager,
        scorer: PromptScorer,
        prompt_editor: PromptEditor,
        llm: BaseLLM,
        task_description: str = "",
    ):
        self.history = history_manager
        self.scorer = scorer
        self.editor = prompt_editor
        self.llm = llm
        self.task_description: str = task_description

        # Статистика глобальной оптимизации
        self.total_global_steps = 0
        self.total_candidates_generated = 0
        self.successful_global_changes = 0

        # История глобальных стратегий
        self.applied_strategies: List[Dict] = []

        # Wrong-exemplars: единый накопленный счётчик провалов по input_text (все шаги)
        self._failure_counter: Counter = Counter()
        self._failure_examples_cache: Dict[str, Example] = {}
        self._processed_node_ids: set = set()
        self._seen_prompt_hashes: set = set()
        self._baseline_score_override: Optional[float] = None

    def optimize(
        self,
        current_generation: int,
        train_examples: List[Example],
        validation_examples: List[Example],
        baseline_score: Optional[float] = None,
    ) -> List[PromptNode]:
        print("\n" + "=" * 60)
        print(f"GLOBAL OPTIMIZATION STEP | Generation {current_generation}")
        print("=" * 60)
        if is_enabled():
            print(
                f"[diag] global optimize input: generation={current_generation} "
                f"train_examples={len(train_examples)} validation_examples={len(validation_examples)}"
            )

        start_time = time.time()
        self.total_global_steps += 1

        self._seen_prompt_hashes.clear()
        self._baseline_score_override = baseline_score

        # Шаг 1: Анализ истории оптимизации
        print("Step 1: Analyzing optimization history...")
        history_analysis = self._analyze_history()
        if is_enabled():
            stag = history_analysis["stagnation"]
            div = history_analysis["diversity"]
            print(
                f"[diag] history analysis: stagnant={stag['is_stagnant']} "
                f"avg_similarity={stag['avg_similarity']:.3f} "
                f"diversity={div['diversity_score']:.3f} "
                f"needs_diversification={div['needs_diversification']}"
            )
            best_els = history_analysis["best_elements"]["top_scores"]
            print(f"[diag] top_{len(best_els)}_scores: {scores_summary(best_els)}")

        # Шаг 2: Генерация кандидатов через мета-оптимизатор (N отдельных LLM вызовов)
        print(
            "\nStep 2: Generating candidates via meta-optimizer (N separate calls)..."
        )
        exemplars = self._select_exemplars(train_examples, current_generation)
        if is_enabled():
            print(f"[diag] QA-exemplars selected={len(exemplars)}")
        candidates = self._generate_candidates_from_history(
            history_analysis, current_generation, exemplars
        )

        print(
            f"Created {len(candidates)} candidates from {GLOBAL_CANDIDATES} separate LLM calls"
        )
        self.total_candidates_generated += len(candidates)

        if not candidates:
            print("No candidates generated — skipping evaluation")
            return []

        # Шаг 3: Full evaluation of all candidates on validation set
        print(
            f"\nStep 3: Full evaluation of {len(candidates)} candidates on validation set..."
        )
        evaluated_candidates = self._evaluate_global_candidates(
            candidates, validation_examples
        )
        previous_best_score = (
            history_analysis["summary"]["best_nodes"][0]["score"]
            if history_analysis["summary"]["best_nodes"]
            else 0.0
        )
        accepted_candidates = [
            candidate
            for candidate in evaluated_candidates
            if candidate.selection_score()
            >= previous_best_score + GLOBAL_MIN_IMPROVEMENT
        ]

        # Шаг 4: Анализируем результаты
        print("\nStep 4: Analyzing results...")
        self._analyze_global_results(evaluated_candidates, history_analysis)
        if is_enabled():
            print(
                f"[diag] global acceptance: accepted={len(accepted_candidates)}/{len(evaluated_candidates)} "
                f"threshold={previous_best_score + GLOBAL_MIN_IMPROVEMENT:.3f}"
            )

        print(f"\nCompleted in {time.time() - start_time:.2f}s")

        if is_enabled():
            print_candidates_summary(
                f"global evaluated candidates gen={current_generation}",
                evaluated_candidates,
            )

        return accepted_candidates

    def _analyze_history(self) -> Dict:
        """Анализ всей истории оптимизации. Определяет паттерны, проблемы и возможности"""
        best_nodes = self.history.get_best_nodes(TOP_BEST_NODES)

        return {
            "summary": self.history.get_optimization_summary(),
            "best_nodes": best_nodes,
            "best_node": best_nodes[0] if best_nodes else None,
            "worst_nodes": self._get_worst_nodes(),
            "stagnation": self._analyze_stagnation(best_nodes),
            "diversity": self._analyze_diversity(),
            "best_elements": self._extract_best_elements(),
        }

    def _get_worst_nodes(self, bottom_k: int = 3) -> List[PromptNode]:
        """Получение худших узлов по композитной метрике"""
        nodes = self.history.get_evaluated_nodes()
        if not nodes:
            return []
        nodes.sort(key=lambda n: n.selection_score())
        return nodes[:bottom_k]

    def _analyze_stagnation(self, best_nodes: List[PromptNode]) -> Dict:
        """Анализ застоя в оптимизации"""
        if len(best_nodes) < 2:
            return {
                "is_stagnant": False,
                "avg_similarity": 0.0,
                "needs_exploration": False,
            }

        similarities = [
            1.0 - d
            for d in self.scorer.calculate_pairwise_metric(
                best_nodes, MAX_DISTANCE_PAIRS
            )
        ]
        avg_similarity = statistics.mean(similarities)

        return {
            "is_stagnant": avg_similarity > STAGNATION_SIMILARITY_THRESHOLD,
            "avg_similarity": avg_similarity,
            "needs_exploration": avg_similarity > STAGNATION_SIMILARITY_THRESHOLD,
            "best_score": best_nodes[0].selection_score() if best_nodes else 0.0,
        }

    def _analyze_diversity(self) -> Dict:
        """Анализ разнообразия в популяции"""
        gens = sorted(self.history.nodes_by_generation.keys())[
            -RECENT_GENERATIONS_FOR_DIVERSITY:
        ]
        nodes = [n for g in gens for n in self.history.get_nodes_by_generation(g)]

        if len(nodes) < 2:
            return {"diversity_score": 0.0, "needs_diversification": True}

        distances = self.scorer.calculate_pairwise_metric(nodes, MAX_DISTANCE_PAIRS)
        avg = statistics.mean(distances)

        return {
            "diversity_score": avg,
            "needs_diversification": avg < DIVERSITY_DISTANCE_THRESHOLD,
        }

    def _extract_best_elements(self) -> Dict:
        best_nodes = self.history.get_best_nodes(top_k=TOP_BEST_NODES)

        if not best_nodes:
            return {"prompts": [], "common_phrases": []}

        # Извлекаем общие фразы/паттерны
        all_words = []
        for node in best_nodes:
            words = node.prompt_text.lower().split()
            all_words.extend(words)

        # Частотный анализ
        word_freq = Counter(all_words)
        common_words = [
            word
            for word, count in word_freq.most_common(COMMON_WORDS_TOP_K)
            if count >= COMMON_WORD_MIN_FREQ
        ]

        # Возвращаем сами объекты PromptNode, а не только текст
        return {
            "prompts": best_nodes,
            "common_phrases": common_words,
            "top_scores": [node.selection_score() for node in best_nodes],
        }

    def _get_meta_prompt_nodes(self) -> List[PromptNode]:
        """Узлы, которые попадут в мета-промпт: фильтрация по порогу + окно."""
        all_evaluated = sorted(
            self.history.get_evaluated_nodes(), key=lambda n: n.selection_score()
        )
        above_threshold = [
            n for n in all_evaluated if n.selection_score() >= HISTORY_SCORE_THRESHOLD
        ]
        if above_threshold:
            all_evaluated = above_threshold
        elif is_enabled():
            print(
                f"[diag] _get_meta_prompt_nodes: no nodes above threshold {HISTORY_SCORE_THRESHOLD:.3f}, using all"
            )
        return all_evaluated[-GLOBAL_HISTORY_WINDOW:]

    def _update_failure_counter(self) -> None:
        """Обновляет накопленный счётчик провалов из ещё не обработанных узлов истории."""
        for node in self.history.get_evaluated_nodes():
            if node.id in self._processed_node_ids:
                continue
            for ex in node.evaluation_examples.get("failures", []):
                self._failure_counter[ex.input_text] += 1
                if ex.input_text not in self._failure_examples_cache:
                    self._failure_examples_cache[ex.input_text] = ex
            self._processed_node_ids.add(node.id)

    def _top_exemplars_from_counter(
        self, train_examples: List[Example], counter: Counter
    ) -> List[Example]:
        """Возвращает топ-EXEMPLAR_COUNT примеров по частоте провалов в counter.

        Приоритет поиска:
        1. train_examples (lookup по input_text) — совпадает, если провал произошёл на обучающем примере.
        2. _failure_examples_cache — всегда содержит Example-объекты провалившихся примеров;
           нужен, потому что оценка узлов ведётся на validation_examples, тогда как сюда
           передаются train_examples. Эти сплиты не пересекаются → без fallback exemplars=0.
        """
        train_lookup = {ex.input_text: ex for ex in train_examples}
        result = []
        from_train = 0
        from_cache = 0
        for input_text, _ in counter.most_common():
            ex = train_lookup.get(input_text)
            if ex is not None:
                from_train += 1
            else:
                ex = self._failure_examples_cache.get(input_text)
                if ex is not None:
                    from_cache += 1
            if ex is not None:
                result.append(
                    Example(
                        input_text=ex.input_text, expected_output=ex.expected_output
                    )
                )
                if len(result) >= EXEMPLAR_COUNT:
                    break
        if is_enabled():
            print(
                f"[diag] _top_exemplars_from_counter: total_counter={len(counter)} "
                f"train_lookup_size={len(train_lookup)} cache_size={len(self._failure_examples_cache)} "
                f"result={len(result)} (from_train={from_train} from_cache={from_cache})"
            )
        return result

    def _exemplars_current_most_frequent(
        self, train_examples: List[Example]
    ) -> List[Example]:
        """Стратегия current_most_frequent: топ-K по счётчику провалов только
        среди инструкций, показанных в текущем мета-промпте."""
        counter: Counter = Counter()
        for node in self._get_meta_prompt_nodes():
            for ex in node.evaluation_examples.get("failures", []):
                counter[ex.input_text] += 1
                if ex.input_text not in self._failure_examples_cache:
                    self._failure_examples_cache[ex.input_text] = ex
        return self._top_exemplars_from_counter(train_examples, counter)

    def _exemplars_random(
        self, train_examples: List[Example], seed: int
    ) -> List[Example]:
        """Случайная выборка EXEMPLAR_COUNT примеров. seed=current_generation — меняется каждый шаг."""
        k = min(EXEMPLAR_COUNT, len(train_examples))
        if k == 0:
            return []
        rng = np.random.default_rng(seed)
        indices = sorted(
            rng.choice(len(train_examples), size=k, replace=False).tolist()
        )
        return [
            Example(
                input_text=train_examples[i].input_text,
                expected_output=train_examples[i].expected_output,
            )
            for i in indices
        ]

    def _exemplars_constant(self, train_examples: List[Example]) -> List[Example]:
        """Фиксированная выборка EXEMPLAR_COUNT примеров (seed=0, не меняется между шагами)."""
        return self._exemplars_random(train_examples, seed=0)

    def _select_exemplars(
        self, train_examples: List[Example], current_generation: int
    ) -> List[Example]:
        """Выбирает wrong-exemplars согласно EXEMPLAR_SELECTION_STRATEGY.

        Стратегии:
          accumulative_most_frequent — топ-K по накопленному счётчику за всю историю (default)
          current_most_frequent      — топ-K по счётчику провалов инструкций текущего мета-промпта
          random                     — случайная выборка, seed=current_generation
          constant                   — фиксированная случайная выборка, seed=0
        """
        if is_enabled():
            print(
                f"[diag] _select_exemplars: strategy={EXEMPLAR_SELECTION_STRATEGY!r} generation={current_generation}"
            )

        if EXEMPLAR_SELECTION_STRATEGY == "accumulative_most_frequent":
            self._update_failure_counter()
            return self._top_exemplars_from_counter(
                train_examples, self._failure_counter
            )
        elif EXEMPLAR_SELECTION_STRATEGY == "current_most_frequent":
            return self._exemplars_current_most_frequent(train_examples)
        elif EXEMPLAR_SELECTION_STRATEGY == "random":
            return self._exemplars_random(train_examples, seed=current_generation)
        elif EXEMPLAR_SELECTION_STRATEGY == "constant":
            return self._exemplars_constant(train_examples)
        else:
            raise ValueError(
                f"Unknown EXEMPLAR_SELECTION_STRATEGY: {EXEMPLAR_SELECTION_STRATEGY!r}. "
                f"Valid values: accumulative_most_frequent, current_most_frequent, random, constant"
            )

    def _generate_candidates_from_history(
        self,
        history_analysis: Dict,
        current_generation: int,
        exemplars: Optional[List[Example]] = None,
    ) -> List[PromptNode]:
        best_nodes = history_analysis["best_elements"]["prompts"]
        if not best_nodes:
            return []

        best_node = best_nodes[0]

        # Узлы для мета-промпта: фильтрация по порогу + окно
        history_nodes = self._get_meta_prompt_nodes()

        meta_prompt = Templates.build_meta_optimizer_prompt(
            history_nodes,
            best_node,
            exemplars,
            task_description=self.task_description,
        )

        if is_enabled():
            print(
                f"[diag] meta-optimizer: history_nodes={len(history_nodes)} "
                f"best_score={best_node.selection_score():.3f} "
                f"exemplars={len(exemplars) if exemplars else 0} "
                f"separate_calls={GLOBAL_CANDIDATES}"
            )

        import re as _re

        candidates = []

        for call_idx in range(GLOBAL_CANDIDATES):
            try:
                raw = self.llm.invoke(
                    prompt=meta_prompt, temperature=GLOBAL_TEMPERATURE
                )
            except Exception as e:
                print(f"    Error in meta-optimizer call {call_idx+1}: {e}")
                continue

            if "<INS>" not in raw:
                if is_enabled():
                    print(
                        f"[diag] call {call_idx+1}: no <INS> tag in response, skipping"
                    )
                continue
            else:
                start_index = raw.index("<INS>") + len("<INS>")
            if "</INS>" not in raw:
                end_index = len(raw)
            else:
                end_index = raw.index("</INS>")
            new_text = raw[start_index:end_index].strip()

            if not new_text:
                if is_enabled():
                    print(f"[diag] call {call_idx+1}: empty extraction, skipping")
                continue

            # Skip if the extracted text still contains raw <INS> tags (artifact)
            if "<INS>" in new_text or "</INS>" in new_text:
                if is_enabled():
                    print(
                        f"[diag] call {call_idx+1}: contains INS tag artifact, skipping"
                    )
                continue
            if len(new_text) > MAX_INSTRUCTION_LENGTH:
                if is_enabled():
                    print(
                        f"[diag] call {call_idx+1}: too long ({len(new_text)} > {MAX_INSTRUCTION_LENGTH}), skipping"
                    )
                continue

            # Дедупликация: MD5 (точное совпадение) → edit-distance (похожие в текущем батче)
            text_hash = hashlib.md5(new_text.encode()).hexdigest()
            if text_hash in self._seen_prompt_hashes:
                if is_enabled():
                    print(f"[diag] call {call_idx+1}: exact-duplicate (md5), skipping")
                continue
            is_duplicate = False
            for c in candidates:
                similarity = 1.0 - self.scorer.calculate_edit_distance(
                    new_text, c.prompt_text
                )
                if similarity > SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    if is_enabled():
                        print(
                            f"[diag] call {call_idx+1}: near-duplicate (sim={similarity:.3f}), skipping"
                        )
                    break
            if is_duplicate:
                continue
            self._seen_prompt_hashes.add(text_hash)

            operation = EditOperation(
                description=f"Meta-optimizer call {call_idx+1} (gen {current_generation})",
                before_snippet=best_node.prompt_text[:100] + "...",
                after_snippet=new_text[:100] + "...",
            )
            strategy_meta = {
                "description": "meta-optimizer-single",
                "action": "single generation",
            }
            node = PromptNode(
                prompt_text=new_text,
                parent_id=best_node.id,
                generation=current_generation,
                source=OptimizationSource.GLOBAL,
                operations=[operation],
                metadata={"global_strategy": strategy_meta},
            )
            candidates.append(node)
            self.applied_strategies.append(
                {
                    "generation": current_generation,
                    "strategy": strategy_meta,
                    "candidate_id": node.id,
                }
            )
            if is_enabled():
                print(
                    f"[diag] call {call_idx+1}: accepted node_id={node.id} "
                    f"prompt_id={prompt_id(new_text)} len={len(new_text)}"
                )

        return candidates

    def _evaluate_global_candidates(
        self, candidates: List[PromptNode], examples: List[Example]
    ) -> List[PromptNode]:
        """Оценка глобальных кандидатов на validation set. Точное совпадение текста промпта → переиспользуем метрики из истории."""
        evaluated = []
        for i, candidate in enumerate(candidates, 1):
            print(f"  Evaluating global candidate {i}/{len(candidates)}...", end=" ")
            try:
                # Точное строковое совпадение: ищем уже оценённый узел с тем же текстом
                cached_node = next(
                    (
                        self.history.get_node(nid)
                        for nid in self.history.nodes_by_prompt_text.get(
                            candidate.prompt_text, []
                        )
                        if self.history.get_node(nid)
                        and self.history.get_node(nid).is_evaluated
                    ),
                    None,
                )
                if cached_node is not None:
                    candidate.metrics = cached_node.metrics
                    candidate.is_evaluated = True
                    candidate.evaluation_examples = cached_node.evaluation_examples
                    score = candidate.metrics.composite_score()
                    print(f"Score: {score:.3f} (cached)")
                else:
                    candidate = self.scorer.evaluate_node(
                        candidate, examples, execute=True, split="validation"
                    )
                    score = candidate.metrics.composite_score()
                    print(f"Score: {score:.3f}")
                self.history.add_node(candidate)
                evaluated.append(candidate)
            except Exception as e:
                print(f"Error: {e}")
                continue

        return evaluated

    def _analyze_global_results(
        self, evaluated_candidates: List[PromptNode], history_analysis: Dict
    ):
        """Анализ результатов глобального шага. Определяет, какие стратегии сработали"""
        if not evaluated_candidates:
            print("No candidates to analyze")
            return

        if getattr(self, "_baseline_score_override", None) is not None:
            previous_best_score = self._baseline_score_override
            if is_enabled():
                print(
                    f"[diag] _analyze_global_results: using baseline_score_override={previous_best_score:.3f}"
                )
        else:
            previous_best_score = (
                history_analysis["summary"]["best_nodes"][0]["score"]
                if history_analysis["summary"]["best_nodes"]
                else 0.0
            )

        print("\n--- Global Step Results ---")
        print(f"Previous best score: {previous_best_score:.3f}")

        # Анализируем каждого кандидата
        improvements = []
        for candidate in evaluated_candidates:
            score = candidate.metrics.composite_score()
            improvement = score - previous_best_score

            strategy = candidate.metadata.get("global_strategy", {})
            strategy_desc = strategy.get("description", "Unknown")[:70]

            print(f"\n  Strategy: {strategy_desc}")
            print(f"  Score: {score:.3f} (Δ {improvement:+.3f})")

            if improvement >= MIN_IMPROVEMENT:
                improvements.append(
                    {
                        "candidate": candidate,
                        "strategy": strategy,
                        "improvement": improvement,
                    }
                )
                self.successful_global_changes += 1

        if improvements:
            best_improvement = max(improvements, key=lambda x: x["improvement"])
            print(f"\n✓ Best global improvement: {best_improvement['improvement']:.3f}")
            print(
                f"  From strategy: {best_improvement['strategy'].get('description', 'Unknown')[:70]}"
            )
        else:
            print("\n✗ No improvements from global step")

    def should_trigger_global_step(
        self, current_generation: int, stagnation_gens: int = 0
    ) -> bool:
        """Определение, нужно ли запускать глобальный шаг"""
        from config import FORCE_GLOBAL_AFTER_STAGNATION

        # Триггер 1: Регулярный интервал
        if current_generation % GLOBAL_TRIGGER_INTERVAL == 0:
            if is_enabled():
                print(
                    f"[diag] global trigger: interval (gen={current_generation} % {GLOBAL_TRIGGER_INTERVAL} == 0)"
                )
            return True

        # Триггер 2: Форсированный глобальный шаг при стагнации
        if stagnation_gens >= FORCE_GLOBAL_AFTER_STAGNATION:
            print(
                f"Global step FORCED by stagnation ({stagnation_gens} gens without improvement)"
            )
            return True

        # Триггер 3: Обнаружен застой в истории
        stagnation_info = self.history.get_stagnation_info()
        if stagnation_info["is_stagnant"]:
            print("Global step triggered by history stagnation")
            return True

        return False

    def get_statistics(self) -> Dict:
        """Статистика глобальной оптимизации"""
        return {
            "total_global_steps": self.total_global_steps,
            "total_candidates_generated": self.total_candidates_generated,
            "successful_global_changes": self.successful_global_changes,
            "success_rate": self.successful_global_changes
            / max(self.total_candidates_generated, 1),
            "strategies_applied": len(self.applied_strategies),
        }

    def get_best_strategies(self, top_k: int) -> List[Dict]:
        """Получение самых успешных стратегий"""
        strategy_results = []

        for item in self.applied_strategies:
            candidate_id = item["candidate_id"]
            candidate = self.history.get_node(candidate_id)

            if candidate and candidate.is_evaluated:
                strategy_results.append(
                    {
                        "strategy": item["strategy"],
                        "generation": item["generation"],
                        "score": candidate.metrics.composite_score(),
                        "candidate_id": candidate_id,
                    }
                )

        strategy_results.sort(key=lambda x: x["score"], reverse=True)
        return strategy_results[:top_k]

    def __repr__(self):
        stats = self.get_statistics()
        return f"GlobalOptimizer(steps={stats['total_global_steps']}, success_rate={stats['success_rate']:.2f})"
