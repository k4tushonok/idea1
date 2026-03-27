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
from diagnostics import is_enabled, prompt_id, scores_summary, print_candidates_summary, llm_calls, print_gate_comparison
from config import (TOP_BEST_NODES,
                    MAX_DISTANCE_PAIRS,
                    COMMON_WORDS_TOP_K,
                    COMMON_WORD_MIN_FREQ,
                    FAILED_PERCENTILE,
                    FAILED_OP_MIN_COUNT,
                    MIN_GLOBAL_SOURCE_USAGE,
                    STAGNATION_SIMILARITY_THRESHOLD,
                    DIVERSITY_DISTANCE_THRESHOLD,
                    LOW_DIVERSITY_THRESHOLD,
                    MAX_DIVERSITY_SAMPLES,
                    MIN_NODES_FOR_DIVERSITY,
                    GLOBAL_CANDIDATES,
                    SIMILARITY_THRESHOLD,
                    GLOBAL_TRIGGER_INTERVAL,
                    RECENT_GENERATIONS_FOR_DIVERSITY,
                    GLOBAL_OPT_AVG_PATH_LENGTH,
                    MIN_IMPROVEMENT,
                    GLOBAL_HISTORY_WINDOW,
                    EXEMPLAR_COUNT,
                    FEW_SHOT_COUNT,
                    HISTORY_SCORE_THRESHOLD,
                    EXEMPLAR_SELECTION_STRATEGY,
                    CROSSOVER_CANDIDATES,
                    GLOBAL_QUALITY_GATE_TOLERANCE,
                    MINI_BATCH_RATIO,
                    GLOBAL_PRESCREEN_GATE,
                    GLOBAL_OPTIMIZER_TEMPERATURE)
    
class GlobalOptimizer:
    def __init__(self, history_manager: HistoryManager, scorer: PromptScorer, prompt_editor: PromptEditor, llm: BaseLLM):
        self.history = history_manager
        self.scorer = scorer
        self.editor = prompt_editor
        self.llm = llm
        self._cache: Dict[str, str] = {}
        
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
        self.reflection_context: str = ""
        
    def optimize(self, current_generation: int, train_examples: List[Example], validation_examples: List[Example]) -> List[PromptNode]:
        print("\n" + "=" * 60)
        print(f"GLOBAL OPTIMIZATION STEP | Generation {current_generation}")
        print("=" * 60)
        if is_enabled():
            print(
                f"[diag] global optimize input: generation={current_generation} "
                f"train_examples={len(train_examples)} validation_examples={len(validation_examples)}"
            )
        
        start_time = time.time()
        
        # Шаг 1: Анализ истории оптимизации
        print("Step 1: Analyzing optimization history...")
        history_analysis = self._analyze_history()
        if is_enabled():
            stag = history_analysis["stagnation"]
            div = history_analysis["diversity"]
            failed = history_analysis["failed_directions"]
            unexplored = history_analysis["unexplored_space"]
            print(
                f"[diag] history analysis: stagnant={stag['is_stagnant']} "
                f"avg_similarity={stag['avg_similarity']:.3f} "
                f"diversity={div['diversity_score']:.3f} "
                f"needs_diversification={div['needs_diversification']}"
            )
            if failed:
                print(f"[diag] failed_directions ({len(failed)}): {'; '.join(failed[:3])}")
            if unexplored:
                print(f"[diag] unexplored_space: {'; '.join(unexplored[:3])}")
            best_els = history_analysis["best_elements"]["top_scores"]
            print(f"[diag] top_{len(best_els)}_scores: {scores_summary(best_els)}")
        
        # Шаг 2: Генерация кандидатов через мета-оптимизатор (история → LLM → новая инструкция)
        print("\nStep 2: Generating candidates via meta-optimizer...")
        exemplars = self._select_exemplars(train_examples, current_generation)
        few_shot = self._select_few_shot_examples(train_examples, current_generation)
        if is_enabled():
            print(f"[diag] wrong-exemplars selected={len(exemplars)}, few-shot={len(few_shot)}")
        candidates = self._generate_candidates_from_history(history_analysis, current_generation, exemplars, few_shot)
        
        # Генерация кандидатов-кроссоверов из лучших промптов
        print("\nStep 2b: Generating crossover candidates...")
        crossover_candidates = self._generate_crossover_candidates(history_analysis, current_generation)
        if crossover_candidates:
            candidates.extend(crossover_candidates)
            print(f"Added {len(crossover_candidates)} crossover candidates")
        
        print(f"Created {len(candidates)} total candidates (meta={len(candidates) - len(crossover_candidates)}, crossover={len(crossover_candidates)})")
        self.total_candidates_generated += len(candidates)

        # Шаг 3: Pre-screen on mini-batch, then full evaluate only promising candidates
        print("\nStep 3: Pre-screening global candidates on mini-batch...")
        mini_batch = self._create_mini_batch(validation_examples, seed=current_generation)
        prescreened = self._prescreen_global_candidates(candidates, mini_batch, history_analysis)
        
        if not prescreened:
            print("All global candidates filtered by pre-screen — skipping full evaluation")
            self.total_global_steps += 1
            return []
        
        print(f"\nStep 3b: Full evaluation of {len(prescreened)}/{len(candidates)} pre-screened candidates...")
        evaluated_candidates = self._evaluate_global_candidates(prescreened, validation_examples)

        best_composite = 0.0
        best_node = history_analysis.get("best_node")
        if best_node and best_node.is_evaluated:
            
            best_composite = best_node.selection_score() * GLOBAL_QUALITY_GATE_TOLERANCE

        valid_for_refinement = [
            c for c in evaluated_candidates
            if c.selection_score() >= best_composite
        ]
        if len(valid_for_refinement) != len(evaluated_candidates):
            print(
                f"Filtered out {len(evaluated_candidates) - len(valid_for_refinement)} global candidates "
                f"below composite gate {best_composite:.3f}"
            )

        # Анализируем только допустимых кандидатов
        print("\nStep 4: Analyzing results...")
        candidates_to_analyze = valid_for_refinement if valid_for_refinement else evaluated_candidates
        self._analyze_global_results(candidates_to_analyze, history_analysis)

        print(f"\nCompleted in {time.time() - start_time:.2f}s")

        self.total_global_steps += 1

        if is_enabled():
            print_candidates_summary(f"global evaluated candidates gen={current_generation}", evaluated_candidates)
            if valid_for_refinement:
                print_candidates_summary("global valid_for_refinement", valid_for_refinement)

        if not valid_for_refinement:
            print(f"  No global candidates passed quality gate {best_composite:.3f} — returning {len(evaluated_candidates)} for local refinement")
            return evaluated_candidates

        return valid_for_refinement
    
    def _analyze_history(self) -> Dict:
        """Анализ всей истории оптимизации. Определяет паттерны, проблемы и возможности"""
        best_nodes = self.history.get_best_nodes(TOP_BEST_NODES)        
        
        return {
            "summary": self.history.get_optimization_summary(),
            "best_nodes": best_nodes,
            "best_node": best_nodes[0] if best_nodes else None,
            "worst_nodes": self._get_worst_nodes(),
            "patterns": self._identify_patterns(best_nodes),
            "stagnation": self._analyze_stagnation(best_nodes),
            "diversity": self._analyze_diversity(),
            "best_elements": self._extract_best_elements(),
            "failed_directions": self._identify_failed_directions(),
            "unexplored_space": self._identify_unexplored_space()
        }

    def _get_worst_nodes(self, bottom_k: int = 3) -> List[PromptNode]:
        """Получение худших узлов по композитной метрике"""
        nodes = self.history.get_evaluated_nodes()
        if not nodes:
            return []
        nodes.sort(key=lambda n: n.selection_score())
        return nodes[:bottom_k]
    
    def _identify_patterns(self, best_nodes: List[PromptNode]) -> Dict:
        """Определение паттернов в истории оптимизации"""
        return {
            "successful_operations": self.history.analyze_successful_operations(),
            "avg_path_length": GLOBAL_OPT_AVG_PATH_LENGTH
        }
        
    def _analyze_stagnation(self, best_nodes: List[PromptNode]) -> Dict:
        """Анализ застоя в оптимизации"""
        if len(best_nodes) < 2:
            return {"is_stagnant": False, "avg_similarity": 0.0, "needs_exploration": False}

        similarities = [1.0 - d for d in self.scorer.calculate_pairwise_metric(best_nodes, MAX_DISTANCE_PAIRS)]
        avg_similarity = statistics.mean(similarities)

        return {
            "is_stagnant": avg_similarity > STAGNATION_SIMILARITY_THRESHOLD,
            "avg_similarity": avg_similarity,
            "needs_exploration": avg_similarity > STAGNATION_SIMILARITY_THRESHOLD,
            "best_score": best_nodes[0].selection_score() if best_nodes else 0.0
        }
            
    def _analyze_diversity(self) -> Dict:
        """Анализ разнообразия в популяции"""
        gens = sorted(self.history.nodes_by_generation.keys())[-RECENT_GENERATIONS_FOR_DIVERSITY:]
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
        """Извлечение лучших элементов из успешных промптов"""
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
        common_words = [word for word, count in word_freq.most_common(COMMON_WORDS_TOP_K) if count >= COMMON_WORD_MIN_FREQ]
        
        # Возвращаем сами объекты PromptNode, а не только текст
        return {
            "prompts": best_nodes,
            "common_phrases": common_words,
            "top_scores": [node.selection_score() for node in best_nodes]
        }

    def _identify_failed_directions(self) -> List[str]:
        """Определение неудачных направлений, которые стоит избегать"""
        # Узлы с низкими скорами
        all_evaluated = self.history.get_evaluated_nodes()
        if not all_evaluated:
            return []
        
        # Берем нижние
        threshold = np.percentile([n.selection_score() for n in all_evaluated], FAILED_PERCENTILE)
        ops = Counter(
            op.description
            for n in all_evaluated
            if n.selection_score() <= threshold
            for op in n.operations
        )
        
        return [
            f"Avoid excessive use of {op}"
            for op, c in ops.items()
            if c >= FAILED_OP_MIN_COUNT
        ]

    def _identify_unexplored_space(self) -> List[str]:
        """Определение неисследованных областей пространства промптов"""
        unexplored = []
        sources = Counter(n.source.value for n in self.history.nodes.values())
        if sources[OptimizationSource.GLOBAL.value] < MIN_GLOBAL_SOURCE_USAGE:
            unexplored.append("Need more global structural changes")
        return unexplored

    def _get_meta_prompt_nodes(self) -> List[PromptNode]:
        """Узлы, которые попадут в мета-промпт: фильтрация по порогу + окно."""
        all_evaluated = sorted(self.history.get_evaluated_nodes(), key=lambda n: n.selection_score())
        above_threshold = [n for n in all_evaluated if n.selection_score() >= HISTORY_SCORE_THRESHOLD]
        if above_threshold:
            all_evaluated = above_threshold
        elif is_enabled():
            print(f"[diag] _get_meta_prompt_nodes: no nodes above threshold {HISTORY_SCORE_THRESHOLD:.3f}, using all")
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

    def _top_exemplars_from_counter(self, train_examples: List[Example], counter: Counter) -> List[Example]:
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
                result.append(Example(input_text=ex.input_text, expected_output=ex.expected_output))
                if len(result) >= EXEMPLAR_COUNT:
                    break
        if is_enabled():
            print(
                f"[diag] _top_exemplars_from_counter: total_counter={len(counter)} "
                f"train_lookup_size={len(train_lookup)} cache_size={len(self._failure_examples_cache)} "
                f"result={len(result)} (from_train={from_train} from_cache={from_cache})"
            )
        return result

    def _exemplars_current_most_frequent(self, train_examples: List[Example]) -> List[Example]:
        """Стратегия current_most_frequent: топ-K по счётчику провалов только
        среди инструкций, показанных в текущем мета-промпте."""
        counter: Counter = Counter()
        for node in self._get_meta_prompt_nodes():
            for ex in node.evaluation_examples.get("failures", []):
                counter[ex.input_text] += 1
                if ex.input_text not in self._failure_examples_cache:
                    self._failure_examples_cache[ex.input_text] = ex
        return self._top_exemplars_from_counter(train_examples, counter)

    def _exemplars_random(self, train_examples: List[Example], seed: int) -> List[Example]:
        """Случайная выборка EXEMPLAR_COUNT примеров. seed=current_generation — меняется каждый шаг."""
        k = min(EXEMPLAR_COUNT, len(train_examples))
        if k == 0:
            return []
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(len(train_examples), size=k, replace=False).tolist())
        return [Example(input_text=train_examples[i].input_text,
                        expected_output=train_examples[i].expected_output)
                for i in indices]

    def _exemplars_constant(self, train_examples: List[Example]) -> List[Example]:
        """Фиксированная выборка EXEMPLAR_COUNT примеров (seed=0, не меняется между шагами)."""
        return self._exemplars_random(train_examples, seed=0)

    def _select_exemplars(self, train_examples: List[Example], current_generation: int) -> List[Example]:
        """Выбирает wrong-exemplars согласно EXEMPLAR_SELECTION_STRATEGY.

        Стратегии:
          accumulative_most_frequent — топ-K по накопленному счётчику за всю историю (default)
          current_most_frequent      — топ-K по счётчику провалов инструкций текущего мета-промпта
          random                     — случайная выборка, seed=current_generation
          constant                   — фиксированная случайная выборка, seed=0
        """
        if is_enabled():
            print(f"[diag] _select_exemplars: strategy={EXEMPLAR_SELECTION_STRATEGY!r} generation={current_generation}")

        if EXEMPLAR_SELECTION_STRATEGY == "accumulative_most_frequent":
            self._update_failure_counter()
            return self._top_exemplars_from_counter(train_examples, self._failure_counter)
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

    def _select_few_shot_examples(self, train_examples: List[Example], current_generation: int) -> List[Example]:
        best_nodes = self.history.get_best_nodes(1)
        if best_nodes:
            successes = best_nodes[0].evaluation_examples.get("success", [])
            if successes:
                rng = np.random.default_rng(current_generation)
                k = min(FEW_SHOT_COUNT, len(successes))
                indices = sorted(rng.choice(len(successes), size=k, replace=False).tolist())
                selected = [successes[i] for i in indices]
                if is_enabled():
                    print(f"[diag] few-shot: {len(selected)} success examples from best node")
                return selected

        # Fallback: random from train
        rng = np.random.default_rng(current_generation + 1000)
        k = min(FEW_SHOT_COUNT, len(train_examples))
        indices = sorted(rng.choice(len(train_examples), size=k, replace=False).tolist())
        return [train_examples[i] for i in indices]

    def _generate_candidates_from_history(self, history_analysis: Dict, current_generation: int, exemplars: Optional[List[Example]] = None, few_shot_examples: Optional[List[Example]] = None) -> List[PromptNode]:
        """Мета-оптимизатор: вся история (промпт + скор) + wrong-exemplars вставляются в мета-промпт,
        LLM напрямую генерирует новую инструкцию."""
        best_nodes = history_analysis["best_elements"]["prompts"]
        if not best_nodes:
            return []
        best_node = best_nodes[0]

        # Узлы для мета-промпта: фильтрация по порогу + окно
        history_nodes = self._get_meta_prompt_nodes()

        meta_prompt = Templates.build_meta_optimizer_prompt(
            history_nodes, best_node, exemplars, few_shot_examples,
            reflection_context=self.reflection_context,
        )
        if is_enabled():
            print(
                f"[diag] meta-optimizer: history_nodes={len(history_nodes)} "
                f"best_score={best_node.selection_score():.3f} "
                f"exemplars={len(exemplars) if exemplars else 0} "
                f"few_shot={len(few_shot_examples) if few_shot_examples else 0}"
            )

        candidates = []
        for i in range(GLOBAL_CANDIDATES):
            prompt = meta_prompt + f"\n\n(Generate variation {i+1} of {GLOBAL_CANDIDATES} — be creative and diverse)"
            try:
                raw = self.llm.invoke(prompt=prompt, temperature=GLOBAL_OPTIMIZER_TEMPERATURE)
                # Извлекаем текст из тегов <INS>...</INS>; fallback — нормализация Markdown
                if "<INS>" in raw and "</INS>" in raw:
                    new_text = raw[raw.index("<INS>") + len("<INS>"):raw.index("</INS>")].strip()
                else:
                    new_text = MarkdownParser.normalize_prompt_text(raw)
            except Exception as e:
                print(f"    Error generating meta-optimizer candidate {i + 1}: {e}")
                continue
            # Пропуск артефактов: тег <INS> просочился в извлечённый текст
            if "INS" in new_text:
                if is_enabled():
                    print(f"[diag] candidate skipped: contains 'INS' artifact")
                continue

            # Дедупликация: MD5 (точное совпадение с историей) → edit-distance (похожие в текущем батче)
            text_hash = hashlib.md5(new_text.encode()).hexdigest()
            if text_hash in self._seen_prompt_hashes:
                if is_enabled():
                    print(f"[diag] exact-duplicate skipped (md5): prompt_id={prompt_id(new_text)}")
                continue
            is_duplicate = False
            for c in candidates:
                similarity = 1.0 - self.scorer.calculate_edit_distance(new_text, c.prompt_text)
                if similarity > SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    if is_enabled():
                        print(
                            f"[diag] near-duplicate skipped: "
                            f"new_prompt_id={prompt_id(new_text)} similarity={similarity:.3f}"
                        )
                    break
            if is_duplicate:
                continue
            self._seen_prompt_hashes.add(text_hash)

            operation = EditOperation(
                description=f"Meta-optimizer (gen {current_generation})",
                before_snippet=best_node.prompt_text[:100] + "...",
                after_snippet=new_text[:100] + "..."
            )
            strategy_meta = {"description": "meta-optimizer", "action": "full-history meta-prompt"}
            node = PromptNode(
                prompt_text=new_text,
                parent_id=best_node.id,
                generation=current_generation,
                source=OptimizationSource.GLOBAL,
                operations=[operation],
                metadata={"global_strategy": strategy_meta}
            )
            candidates.append(node)
            self.applied_strategies.append(
                {"generation": current_generation, "strategy": strategy_meta, "candidate_id": node.id}
            )
            if is_enabled():
                print(
                    f"[diag] meta-optimizer candidate accepted: node_id={node.id} "
                    f"prompt_id={prompt_id(new_text)} len={len(new_text)}"
                )

        return candidates
    
    def _generate_crossover_candidates(self, history_analysis: Dict, current_generation: int) -> List[PromptNode]:
        """Комбинирование лучших элементов из топовых промптов.
        
        Берёт пары высокооценочных промптов и просит LLM создать новый промпт,
        наследующий сильные стороны обоих родителей и разрешающий конфликты.
        """
        best_nodes = history_analysis["best_elements"]["prompts"]
        if len(best_nodes) < 2:
            if is_enabled():
                print("[diag] crossover skipped: need at least 2 best nodes")
            return []
        
        candidates = []
        num_pairs = min(CROSSOVER_CANDIDATES, len(best_nodes) - 1)
        
        for i in range(num_pairs):
            parent_a = best_nodes[i]
            parent_b = best_nodes[i + 1]
            
            # Пропускаем, если родители слишком похожи (кроссовер был бы избыточным)
            similarity = 1.0 - self.scorer.calculate_edit_distance(parent_a.prompt_text, parent_b.prompt_text)
            if similarity > SIMILARITY_THRESHOLD:
                if is_enabled():
                    print(f"[diag] crossover pair {i+1} skipped: similarity={similarity:.3f} > threshold")
                continue
            
            crossover_prompt = Templates.build_crossover_prompt(parent_a, parent_b)
            
            try:
                raw = self.llm.invoke(prompt=crossover_prompt, temperature=GLOBAL_OPTIMIZER_TEMPERATURE)
                
                if "<INS>" in raw and "</INS>" in raw:
                    new_text = raw[raw.index("<INS>") + len("<INS>"):raw.index("</INS>")].strip()
                else:
                    new_text = MarkdownParser.normalize_prompt_text(raw)
                
                # Пропускаем артефакты извлечения
                if "INS" in new_text:
                    if is_enabled():
                        print(f"[diag] crossover candidate skipped: contains 'INS' artifact")
                    continue
                
                # Проверка дедупликации
                text_hash = hashlib.md5(new_text.encode()).hexdigest()
                if text_hash in self._seen_prompt_hashes:
                    if is_enabled():
                        print(f"[diag] crossover exact-duplicate skipped")
                    continue
                
                # Проверка похожести с существующими кандидатами
                is_duplicate = False
                for c in candidates:
                    sim = 1.0 - self.scorer.calculate_edit_distance(new_text, c.prompt_text)
                    if sim > SIMILARITY_THRESHOLD:
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue
                
                self._seen_prompt_hashes.add(text_hash)
                
                operation = EditOperation(
                    description=f"Crossover (gen {current_generation}): top-{i+1} x top-{i+2}",
                    before_snippet=f"A({parent_a.selection_score():.3f}): {parent_a.prompt_text[:60]}... + B({parent_b.selection_score():.3f}): {parent_b.prompt_text[:60]}...",
                    after_snippet=new_text[:100] + "..."
                )
                strategy_meta = {
                    "description": "crossover",
                    "action": f"crossover of top-{i+1} (score={parent_a.selection_score():.3f}) and top-{i+2} (score={parent_b.selection_score():.3f})"
                }
                node = PromptNode(
                    prompt_text=new_text,
                    parent_id=parent_a.id,
                    generation=current_generation,
                    source=OptimizationSource.GLOBAL,
                    operations=[operation],
                    metadata={"global_strategy": strategy_meta}
                )
                candidates.append(node)
                self.applied_strategies.append({
                    "generation": current_generation,
                    "strategy": strategy_meta,
                    "candidate_id": node.id
                })
                
                if is_enabled():
                    print(
                        f"[diag] crossover candidate accepted: node_id={node.id} "
                        f"prompt_id={prompt_id(new_text)} len={len(new_text)}"
                    )
                    
            except Exception as e:
                print(f"    Error generating crossover candidate {i+1}: {e}")
        
        return candidates
    
    def _create_mini_batch(self, validation_examples: List[Example], seed: int = 0) -> List[Example]:
        import random as rng_module
        mini_size = max(5, int(len(validation_examples) * MINI_BATCH_RATIO))
        if mini_size >= len(validation_examples):
            return validation_examples
        rng = rng_module.Random(42 + seed)
        indices = sorted(rng.sample(range(len(validation_examples)), mini_size))
        return [validation_examples[i] for i in indices]
    
    def _prescreen_global_candidates(self, candidates: List[PromptNode], mini_batch: List[Example], history_analysis: Dict) -> List[PromptNode]:
        import time as _t
        _gps_t0 = _t.time()
        _gps_calls0 = llm_calls(self.scorer.llm)

        best_node = history_analysis.get("best_node")
        gate_score = 0.0
        if best_node and best_node.is_evaluated:
            best_stage1 = self.scorer.evaluate_prompt(
                best_node.prompt_text, mini_batch,
                execute=True, sample=False, stage=1,
            )
            gate_score = best_stage1.composite_score() * GLOBAL_PRESCREEN_GATE
            if is_enabled():
                print(f"[diag] prescreen gate: best_stage1={best_stage1.composite_score():.3f} "
                      f"× {GLOBAL_PRESCREEN_GATE} = {gate_score:.3f}")
        
        passed = []
        for i, candidate in enumerate(candidates, 1):
            try:
                metrics = self.scorer.evaluate_prompt(
                    candidate.prompt_text, mini_batch,
                    execute=True, sample=False, stage=1
                )
                score = metrics.composite_score()
                did_pass = score >= gate_score
                print(f"  Pre-screen global {i}/{len(candidates)}: {score:.3f} (gate={gate_score:.3f}, stage 1)", end="")
                if did_pass:
                    print(" ✓ passed")
                    passed.append(candidate)
                else:
                    print(" ✗ filtered")
                if is_enabled():
                    print_gate_comparison(
                        f"global_prescreen_{i}", score, gate_score, stage=1, passed=did_pass,
                    )
            except Exception as e:
                print(f"  Pre-screen error {i}: {e}")
        
        if is_enabled():
            _gps_elapsed = _t.time() - _gps_t0
            _gps_calls1 = llm_calls(self.scorer.llm)
            print(
                f"[diag] global pre-screen summary: {len(passed)}/{len(candidates)} passed, "
                f"gate={gate_score:.3f} llm_calls={_gps_calls1 - _gps_calls0} "
                f"time={_gps_elapsed:.2f}s"
            )

        print(f"Pre-screen: {len(passed)}/{len(candidates)} candidates passed gate")
        return passed
    
    def _evaluate_global_candidates(self, candidates: List[PromptNode], validation_examples: List[Example]) -> List[PromptNode]:
        """Оценка глобальных кандидатов. Точное совпадение текста промпта → переиспользуем метрики из истории."""
        evaluated = []
        for i, candidate in enumerate(candidates, 1):
            print(f"  Evaluating global candidate {i}/{len(candidates)}...", end=" ")
            try:
                # Точное строковое совпадение: ищем уже оценённый узел с тем же текстом
                cached_node = next(
                    (self.history.get_node(nid)
                     for nid in self.history.nodes_by_prompt_text.get(candidate.prompt_text, [])
                     if self.history.get_node(nid) and self.history.get_node(nid).is_evaluated),
                    None
                )
                if cached_node is not None:
                    candidate.metrics = cached_node.metrics
                    candidate.is_evaluated = True
                    candidate.evaluation_examples = cached_node.evaluation_examples
                    score = candidate.metrics.composite_score()
                    print(f"Score: {score:.3f} (cached)")
                else:
                    candidate = self.scorer.evaluate_node(candidate, validation_examples, execute=True, split="validation")
                    score = candidate.metrics.composite_score()
                    print(f"Score: {score:.3f}")
                self.history.add_node(candidate)
                evaluated.append(candidate)
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        return evaluated    
    
    def _analyze_global_results(self, evaluated_candidates: List[PromptNode], history_analysis: Dict):
        """Анализ результатов глобального шага. Определяет, какие стратегии сработали"""
        if not evaluated_candidates:
            print("No candidates to analyze")
            return
        
        # Сравниваем с лучшим до глобального шага
        previous_best_score = history_analysis["summary"]["best_nodes"][0]["score"] if history_analysis["summary"]["best_nodes"] else 0.0
        
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
                improvements.append({
                    "candidate": candidate,
                    "strategy": strategy,
                    "improvement": improvement
                })
                self.successful_global_changes += 1
        
        if improvements:
            best_improvement = max(improvements, key=lambda x: x["improvement"])
            print(f"\n✓ Best global improvement: {best_improvement['improvement']:.3f}")
            print(f"  From strategy: {best_improvement['strategy'].get('description', 'Unknown')[:70]}")
        else:
            print("\n✗ No improvements from global step")     
    
    def should_trigger_global_step(self, current_generation: int) -> bool:
        """Определение, нужно ли запускать глобальный шаг"""
        # Триггер 1: Регулярный интервал
        if current_generation % GLOBAL_TRIGGER_INTERVAL == 0:
            if is_enabled():
                print(f"[diag] global trigger: interval (gen={current_generation} % {GLOBAL_TRIGGER_INTERVAL} == 0)")
            return True
        
        # Триггер 2: Обнаружен застой
        stagnation_info = self.history.get_stagnation_info()
        if stagnation_info["is_stagnant"]:
            print("Global step triggered by stagnation")
            return True
            
        return False
    
    def get_statistics(self) -> Dict:
        """Статистика глобальной оптимизации"""
        return {
            "total_global_steps": self.total_global_steps,
            "total_candidates_generated": self.total_candidates_generated,
            "successful_global_changes": self.successful_global_changes,
            "success_rate": self.successful_global_changes / max(self.total_candidates_generated, 1),
            "strategies_applied": len(self.applied_strategies)
        }

    def get_best_strategies(self, top_k: int) -> List[Dict]:
        """Получение самых успешных стратегий"""
        strategy_results = []
        
        for item in self.applied_strategies:
            candidate_id = item["candidate_id"]
            candidate = self.history.get_node(candidate_id)
            
            if candidate and candidate.is_evaluated:
                strategy_results.append({
                    "strategy": item["strategy"],
                    "generation": item["generation"],
                    "score": candidate.metrics.composite_score(),
                    "candidate_id": candidate_id
                })
        
        strategy_results.sort(key=lambda x: x["score"], reverse=True)
        return strategy_results[:top_k]
    
    def __repr__(self):
        stats = self.get_statistics()
        return f"GlobalOptimizer(steps={stats['total_global_steps']}, success_rate={stats['success_rate']:.2f})"
