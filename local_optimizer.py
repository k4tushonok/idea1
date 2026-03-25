from typing import List, Dict, Optional, Set, Tuple
from copy import deepcopy
import time
import random
from data_structures import Example, PromptNode, TextGradient
from history_manager import HistoryManager
from evaluator.scorer import PromptScorer
from text_gradient_generator import TextGradientGenerator
from prompt_editor import PromptEditor
from diagnostics import is_enabled, prompt_id, preview_text, print_population, print_timing, scores_summary
from config import (
    LOCAL_ITERATIONS_PER_GENERATION,
    MIN_IMPROVEMENT,
    LOCAL_BATCH_SIZE,
    CLUSTERING_FAILURE_MULTIPLIER,
    MAX_CONTEXT_OPERATIONS,
    MIN_EXAMPLES_FOR_CONTRASTIVE,
    PATIENCE,
    SIMILARITY_THRESHOLD,
    LOCAL_PARENTS_PER_ITERATION,
    MINI_BATCH_RATIO,
    PRE_SCREEN_TOP_K,
    GRADIENT_MOMENTUM,
    DIVERSITY_WEIGHT,
    MAX_GRADIENT_PAIRS,
)


class LocalOptimizer:
    def __init__(self, history_manager: HistoryManager, scorer: PromptScorer, gradient_generator: TextGradientGenerator, prompt_editor: PromptEditor, llm):
        self.history = history_manager
        self.scorer = scorer
        self.gradient_gen = gradient_generator
        self.editor = prompt_editor
        
        # Статистика локальной оптимизации
        self.total_iterations = 0
        self.improvements_count = 0
        self.iteration_stats: List[Dict] = []
        
        # Кэш для предотвращения повторной оценки
        self._train_outcomes_cache: Dict[str, Tuple] = {}
        self._evaluated_prompts: Set[str] = set()
        self.llm = llm
        
        # Отслеживание импульса градиентов
        # Какие направления градиентов исторически приводили к улучшениям
        self._momentum_buffer: Dict[str, float] = {}
        # Адаптивный масштаб шага: увеличивается при стагнации, уменьшается при улучшении
        self._step_scale: float = 1.0
    
    def optimize(self, starting_node: PromptNode, train_examples: List[Example], validation_examples: List[Example]) -> PromptNode:
        print(f"\n{'='*60}")
        print(f"Starting Local Optimization")
        print(f"{'='*60}\n")
        
        # Добавляем начальный узел в историю, если его там нет
        if not self.history.get_node(starting_node.id):
            self.history.add_node(starting_node)
        
        # Оцениваем начальный узел, если не оценен
        if not starting_node.is_evaluated:
            print(f"Evaluating starting node...")
            starting_node = self.scorer.evaluate_node(starting_node, validation_examples, execute=True, split="validation")
            self.history.update_node(starting_node.id, starting_node)
            print(f"Starting score: {starting_node.metrics.composite_score():.3f}")
        
        # Текущий лучший узел
        current_beam: List[PromptNode] = [starting_node]
        best_score = starting_node.selection_score()
        best_accuracy = starting_node.selection_accuracy()
        no_improve_iters = 0
        
        # Создаём мини-батч для быстрой предварительной фильтрации кандидатов
        mini_batch = self._create_mini_batch(validation_examples)
        print(f"Pre-screening mini-batch: {len(mini_batch)}/{len(validation_examples)} examples")
        
        for iteration in range(LOCAL_ITERATIONS_PER_GENERATION):
            iteration_start_time = time.time()
            calls_before = getattr(self.scorer.llm, 'total_api_calls', 0)
            
            print(f"\n--- Iteration {iteration + 1}/{LOCAL_ITERATIONS_PER_GENERATION} (no_improve={no_improve_iters}/{PATIENCE}) ---")
            if is_enabled():
                print_population(f"beam state iter={iteration + 1}", current_beam)
                
            # Берём топ-кандидатов из beam как родителей
            parents = sorted(current_beam, key=lambda n: n.selection_score(), reverse=True)
            if len(parents) > LOCAL_PARENTS_PER_ITERATION:
                parents = parents[:LOCAL_PARENTS_PER_ITERATION]
            
            print(f"Generating from {len(parents)} parents...")
            
            # ================================================================
            # ФАЗА 1: НАКОПЛЕНИЕ ГРАДИЕНТОВ
            # Собираем градиенты со ВСЕХ родителей, применяем импульсный бонус,
            # затем сортируем и выбираем наиболее перспективные направления.
            # ================================================================
            gradient_parent_pairs: List[Tuple[PromptNode, TextGradient]] = []
            
            for p_idx, parent in enumerate(parents, 1):
                print(f"  Parent {p_idx}/{len(parents)} (score={parent.selection_score():.3f})")
            
                # Шаг 1: Получаем провалы текущего лучшего промпта
                failure_examples, success_examples, real_rate = self._get_train_examples_outcomes(parent, train_examples)
                if is_enabled():
                    print(f"[diag] train outcomes: failures={len(failure_examples)} successes={len(success_examples)}")
                if not failure_examples:
                    print("  No failures found - prompt is perfect on training set!")
                    continue
                
                gradients = self._generate_gradients(parent, failure_examples, success_examples)
                print(f"    Generated {len(gradients)} gradients")
                
                # Применяем импульсный бонус на основе исторической успешности градиентов
                for g in gradients:
                    self._apply_momentum_boost(g)
                
                gradient_parent_pairs.extend((parent, g) for g in gradients)
            
            if not gradient_parent_pairs:
                print("✗ No gradients generated from any parent")
                no_improve_iters += 1
                self.total_iterations += 1
                self._record_iteration_stats(iteration, iteration_start_time, calls_before)
                continue
            
            # Сортируем накопленные градиенты по приоритету (наивысший первым)
            gradient_parent_pairs.sort(key=lambda pair: pair[1].priority, reverse=True)
            top_pairs = gradient_parent_pairs[:MAX_GRADIENT_PAIRS]
            
            if is_enabled():
                print(f"[diag] gradient accumulation: total={len(gradient_parent_pairs)} selected={len(top_pairs)}")
                for i, (p, g) in enumerate(top_pairs):
                    print(f"[diag]   gradient {i+1}: priority={g.priority:.3f} cluster={g.metadata.get('cluster', 'n/a')}")
            
            # ================================================================
            # ФАЗА 2: ГЕНЕРАЦИЯ КАНДИДАТОВ
            # Генерируем варианты из каждой топовой пары градиент-родитель
            # ================================================================
            all_candidates: List[PromptNode] = []
            for parent, gradient in top_pairs:
                try:
                    variants = self.editor.generate_variants(parent.prompt_text, gradient, parent_node=parent)
                    if is_enabled():
                        print(f"[diag] variants from gradient: {len(variants)}")
                    all_candidates.extend(variants)
                except Exception as e:
                    print(f"    Error generating variants: {e}")
                    continue
            
            # Дедупликация между собой и относительно beam
            unique_candidates = self._deduplicate_candidates(all_candidates, current_beam)
            print(f"Total unique candidates after dedup: {len(unique_candidates)}")
            
            if not unique_candidates:
                print("✗ No unique candidates generated")
                no_improve_iters += 1
                self.total_iterations += 1
                self._record_iteration_stats(iteration, iteration_start_time, calls_before)
                continue
            
            # ================================================================
            # ФАЗА 3: ПРЕДВАРИТЕЛЬНЫЙ ОТБОР НА МИНИ-БАТЧЕ
            # Оцениваем кандидатов сначала на маленьком подмножестве, затем
            # полностью оцениваем только top-K.
            # ================================================================
            if len(unique_candidates) > PRE_SCREEN_TOP_K:
                print(f"Pre-screening {len(unique_candidates)} candidates on mini-batch ({len(mini_batch)} examples)...")
                pre_scores = self._pre_screen_candidates(unique_candidates, mini_batch)
                
                ranked = sorted(zip(unique_candidates, pre_scores), key=lambda x: x[1], reverse=True)
                top_candidates = [c for c, s in ranked[:PRE_SCREEN_TOP_K]]
                
                if is_enabled():
                    all_pre = [s for _, s in ranked]
                    print(f"[diag] pre-screen scores: {scores_summary(all_pre)}")
                
                print(f"Pre-screened to top {len(top_candidates)} candidates")
            else:
                top_candidates = unique_candidates
            
            # ================================================================
            # ФАЗА 4: ПОЛНАЯ ОЦЕНКА
            # ================================================================
            evaluated_candidates = self._evaluate_candidates(top_candidates, validation_examples)
            print(f"Evaluated {len(evaluated_candidates)} candidates")
            
            # Порог качества: средняя точность beam как базовый уровень для фильтрации слабых кандидатов
            beam_accuracies = [n.metrics.metrics.get("accuracy", 0.0) for n in current_beam]
            baseline_accuracy = sum(beam_accuracies) / len(beam_accuracies)
            eligible_candidates = [
                c for c in evaluated_candidates
                if c.metrics.metrics.get("accuracy", 0.0) >= baseline_accuracy
            ]
            if len(eligible_candidates) != len(evaluated_candidates):
                print(
                    f"Filtered out {len(evaluated_candidates) - len(eligible_candidates)} candidates "
                    f"by quality gate (baseline accuracy={baseline_accuracy:.3f})"
                )
            
            if eligible_candidates:
                # ================================================================
                # ФАЗА 5: ОТБОР BEAM С УЧЁТОМ РАЗНООБРАЗИЯ
                # Вместо жадного отбора добавляем небольшой бонус за разнообразие,
                # чтобы предотвратить коллапс beam (сходимость всех промптов к одному тексту).
                # ================================================================
                all_for_beam = eligible_candidates + current_beam
                current_beam = self._diversity_aware_select(all_for_beam, LOCAL_PARENTS_PER_ITERATION)
                
                best_candidate = current_beam[0]
                candidate_score = best_candidate.selection_score()
                improvement = candidate_score - best_score
                print(f"Best candidate score: {candidate_score:.3f} (Δ {improvement:+.3f})")
                if is_enabled():
                    beam_scores = [n.selection_score() for n in current_beam]
                    print(f"[diag] updated beam scores: {scores_summary(beam_scores)}")
                    eval_scores = sorted([n.selection_score() for n in eligible_candidates], reverse=True)
                    print(f"[diag] eligible candidates scores: {scores_summary(eval_scores)}")

                if candidate_score + 1e-8 >= best_score + MIN_IMPROVEMENT:
                    print(f"✓ Improvement found! Updating beam")
                    # Положительное обновление импульса для успешных направлений градиентов
                    self._update_momentum(gradient_parent_pairs, improved=True)
                    best_score = candidate_score
                    no_improve_iters = 0
                    self.improvements_count += 1
                    self._step_scale = max(0.5, self._step_scale * 0.9)
                else:
                    print(f"Beam updated but no significant improvement over best")
                    self._update_momentum(gradient_parent_pairs, improved=False)
                    no_improve_iters += 1
                    self._step_scale = min(2.0, self._step_scale * 1.15)
            else:
                print("✗ No valid candidates generated")
                self._update_momentum(gradient_parent_pairs, improved=False)
                no_improve_iters += 1
                self._step_scale = min(2.0, self._step_scale * 1.15)
            
            self._record_iteration_stats(iteration, iteration_start_time, calls_before)
            self.total_iterations += 1
            
            # Early stopping
            if no_improve_iters >= PATIENCE:
                print(f"\nEarly stopping after {no_improve_iters} iterations without improvement")
                if is_enabled():
                    print_population("beam at early stop", current_beam)
                break
        
        print(f"\n{'='*60}")
        print(f"Local Optimization Complete")
        print(f"Final score: {best_score:.3f}")
        print(f"Improvements: {self.improvements_count}")
        print(f"{'='*60}\n")
        
        return max(current_beam, key=lambda n: n.selection_score())
    
    def _get_train_examples_outcomes(self, node: PromptNode, examples: List[Example]) -> Tuple[List[Example], List[Example], float]:
        cache_key = node.prompt_text
        if cache_key in self._train_outcomes_cache:
            failures, successes, rate = self._train_outcomes_cache[cache_key]
            print(f"  Train outcomes: cached ({len(failures)} failures, {rate:.1%})")
            return failures, successes, rate
        
        print(f"Executing prompt on {len(examples)} examples to find failures...")
        eval_examples = [
            Example(input_text=ex.input_text, expected_output=ex.expected_output, metadata=dict(ex.metadata))
            for ex in examples
        ]
        executed_examples = self.scorer.execute_prompt_batch(node.prompt_text, eval_examples)

        failures: List[Example] = []
        successes: List[Example] = []
        for ex in executed_examples:
            if ex.actual_output and (ex.is_correct() or ex.is_correct_by_llm(self.llm)):
                successes.append(ex)
            else:
                failures.append(ex)

        real_rate = len(failures) / max(len(executed_examples), 1)
        print(f"Train failures: {len(failures)}/{len(executed_examples)} ({real_rate:.1%})")

        if len(failures) > LOCAL_BATCH_SIZE:
            step = len(failures) / LOCAL_BATCH_SIZE
            failures_for_gradient = [failures[int(i * step)] for i in range(LOCAL_BATCH_SIZE)]
        else:
            failures_for_gradient = failures

        if is_enabled():
            print(
                f"[diag] train outcomes after cap: failures={len(failures_for_gradient)} "
                f"successes={len(successes)} failure_rate={real_rate:.3f}"
            )

        self._train_outcomes_cache[cache_key] = (failures_for_gradient, successes, real_rate)
        return failures_for_gradient, successes, real_rate
    
    def _generate_gradients(self, node: PromptNode, failure_examples: List[Example], success_examples: List[Example]) -> List[TextGradient]:
        """Генерация текстовых градиентов на основе провалов и успехов"""
        print("Generating text gradients...")
        if is_enabled():
            print(
                f"[diag] gradient input: prompt_id={prompt_id(node.prompt_text)} "
                f"failures={len(failure_examples)} successes={len(success_examples)}"
            )
        context = { "generation": node.generation, "previous_attempts": len(self.history.get_lineage(node.id)) }

        if node.generation > 0:
            successful_ops = self.history.analyze_successful_operations()
            if successful_ops:
                context["successful_operations"] = list(successful_ops.keys())[:MAX_CONTEXT_OPERATIONS]
        
        # Передаём масштаб шага в контекст для адаптивной интенсивности градиента
        context["step_scale"] = self._step_scale
        
        if len(failure_examples) > LOCAL_BATCH_SIZE * CLUSTERING_FAILURE_MULTIPLIER:
            gradients = self.gradient_gen.generate_clustered_gradients(node, failure_examples, success_examples, context)
        else:
            gradients = self.gradient_gen.generate_gradients_batch(node.prompt_text, failure_examples, success_examples)

        if len(success_examples) >= MIN_EXAMPLES_FOR_CONTRASTIVE and len(failure_examples) >= MIN_EXAMPLES_FOR_CONTRASTIVE:
            print("Generating contrastive gradient...")
            try:
                contrastive_gradient = self.gradient_gen.generate_contrastive_gradient(node.prompt_text, failure_examples[:MIN_EXAMPLES_FOR_CONTRASTIVE], success_examples[:MIN_EXAMPLES_FOR_CONTRASTIVE])
                gradients.insert(0, contrastive_gradient)
            except Exception as e:
                print(f"Failed to generate contrastive gradient: {e}")
        if is_enabled():
            for idx, gradient in enumerate(gradients, start=1):
                cluster = gradient.metadata.get("cluster", "n/a")
                print(
                    f"[diag] gradient {idx}: priority={gradient.priority:.3f} "
                    f"suggestions={len(gradient.specific_suggestions)} cluster={cluster}"
                )
                print(f"[diag]   direction='{preview_text(gradient.suggested_direction, 220)}'")
        return gradients
    
    def _generate_candidates(self, parent_node: PromptNode, gradients: List[TextGradient]) -> List[PromptNode]:
        """Генерация кандидатов на основе градиентов"""
        all_candidates: List[PromptNode] = []
        
        for i, gradient in enumerate(gradients):
            print(f"  Generating variants from gradient {i+1}/{len(gradients)}")
            try:
                variants = self.editor.generate_variants(parent_node.prompt_text, gradient, parent_node=parent_node)
                if is_enabled():
                    print(f"[diag] variants generated from gradient {i+1}: {len(variants)}")
                    for v_idx, variant in enumerate(variants, start=1):
                        print(
                            f"[diag]   variant {v_idx}: node_id={variant.id} "
                            f"prompt_id={prompt_id(variant.prompt_text)} len={len(variant.prompt_text)}"
                        )
                all_candidates.extend(variants)
            except Exception as e:
                print(f"    Error generating variants: {e}")
                continue
        
        # Фильтрация дубликатов
        unique: List[PromptNode] = []
        for candidate in all_candidates:
            is_duplicate = any(
                1.0 - self.scorer.calculate_edit_distance(candidate.prompt_text, u.prompt_text) > SIMILARITY_THRESHOLD
                for u in unique
            )
            if not is_duplicate:
                unique.append(candidate)
        print(f"  Generated {len(all_candidates)} variants, {len(unique)} unique")
        return unique
    
 
    def _create_mini_batch(self, validation_examples: List[Example]) -> List[Example]:
        """Создание детерминированного мини-батча для предварительного отбора.
        Использует стратифицированную выборку для репрезентативного покрытия."""
        mini_size = max(5, int(len(validation_examples) * MINI_BATCH_RATIO))
        if mini_size >= len(validation_examples):
            return validation_examples
        rng = random.Random(42)
        indices = sorted(rng.sample(range(len(validation_examples)), mini_size))
        return [validation_examples[i] for i in indices]
    
    def _pre_screen_candidates(self, candidates: List[PromptNode], mini_batch: List[Example]) -> List[float]:
        """Быстрая оценка на мини-батче для предварительного отбора.
        Возвращает список композитных оценок, по одной на кандидата."""
        scores = []
        for i, candidate in enumerate(candidates):
            try:
                metrics = self.scorer.evaluate_prompt(
                    candidate.prompt_text, mini_batch,
                    execute=True, sample=False
                )
                score = metrics.composite_score()
                scores.append(score)
                print(f"  Pre-screen {i+1}/{len(candidates)}: {score:.3f}")
            except Exception as e:
                print(f"  Pre-screen error {i+1}: {e}")
                scores.append(0.0)
        return scores
    
    def _apply_momentum_boost(self, gradient: TextGradient):
        """Применение импульсного бонуса к приоритету градиента на основе исторической успешности.
        Градиенты в направлениях, которые ранее приводили к улучшениям, получают повышенный приоритет."""
        direction_key = self._gradient_direction_key(gradient)
        if direction_key in self._momentum_buffer:
            momentum = self._momentum_buffer[direction_key]
            if momentum > 0:
                old_priority = gradient.priority
                gradient.priority = min(1.0, gradient.priority + momentum * GRADIENT_MOMENTUM)
                if is_enabled():
                    print(f"[diag] momentum boost: {old_priority:.3f} -> {gradient.priority:.3f} (key={direction_key[:30]})")
            elif momentum < -0.3:
                # Штрафуем направления, которые стабильно проваливаются
                gradient.priority = max(0.0, gradient.priority + momentum * 0.1)
    
    def _gradient_direction_key(self, gradient: TextGradient) -> str:
        """Создание хэшируемого ключа для отслеживания направления градиента."""
        cluster = gradient.metadata.get("cluster", "")
        if cluster and cluster not in ("n/a", "all"):
            return f"cluster:{cluster}"
        gtype = gradient.metadata.get("type", "")
        if gtype:
            return f"type:{gtype}"
        if gradient.specific_suggestions:
            words = gradient.specific_suggestions[0].lower().split()[:5]
            return f"suggestion:{' '.join(words)}"
        return f"direction:{gradient.suggested_direction[:50]}"
    
    def _update_momentum(self, gradient_parent_pairs: List[Tuple], improved: bool):
        """Обновление буфера импульса по итогам итерации.
        Успешные направления получают положительный импульс, неуспешные — небольшой штраф."""
        delta = 0.15 if improved else -0.05
        for _, gradient in gradient_parent_pairs:
            key = self._gradient_direction_key(gradient)
            current = self._momentum_buffer.get(key, 0.0)
            self._momentum_buffer[key] = max(-0.5, min(1.0, current + delta))
    
    def _deduplicate_candidates(self, candidates: List[PromptNode], beam: List[PromptNode]) -> List[PromptNode]:
        """Дедупликация кандидатов между собой и относительно существующих членов beam."""
        unique: List[PromptNode] = []
        seen = set()
        beam_pids = {prompt_id(n.prompt_text) for n in beam}
        
        for c in candidates:
            pid = prompt_id(c.prompt_text)
            if pid in seen or pid in beam_pids:
                continue
            is_dup = any(
                1.0 - self.scorer.calculate_edit_distance(c.prompt_text, u.prompt_text) > SIMILARITY_THRESHOLD
                for u in unique
            )
            if not is_dup:
                seen.add(pid)
                unique.append(c)
        return unique
    
    def _diversity_aware_select(self, candidates: List[PromptNode], beam_size: int) -> List[PromptNode]:
        """Отбор членов beam с бонусом за разнообразие для предотвращения коллапса.
        
        Жадный алгоритм: сначала берём абсолютно лучшего,
        затем для остальных позиций добавляем бонус за разнообразие,
        пропорциональный минимальному расстоянию до уже выбранных.
        Score = selection_score + DIVERSITY_WEIGHT * min_distance_to_selected
        """
        if len(candidates) <= beam_size:
            return sorted(candidates, key=lambda n: n.selection_score(), reverse=True)
        
        # Сначала дедупликация по тексту промпта
        seen_pids = set()
        deduped = []
        for c in candidates:
            pid = prompt_id(c.prompt_text)
            if pid not in seen_pids:
                seen_pids.add(pid)
                deduped.append(c)
        
        candidates_sorted = sorted(deduped, key=lambda n: n.selection_score(), reverse=True)
        selected = [candidates_sorted[0]]
        remaining = list(candidates_sorted[1:])
        
        while len(selected) < beam_size and remaining:
            best_idx = -1
            best_combined = -float('inf')
            
            for i, candidate in enumerate(remaining):
                score = candidate.selection_score()
                min_dist = min(
                    self.scorer.calculate_edit_distance(candidate.prompt_text, s.prompt_text)
                    for s in selected
                )
                combined = score + DIVERSITY_WEIGHT * min_dist
                if combined > best_combined:
                    best_combined = combined
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
            else:
                break
        
        return selected
    
    def _record_iteration_stats(self, iteration: int, start_time: float, calls_before: int):
        """Запись статистики для данной итерации."""
        iteration_time = time.time() - start_time
        calls_after = getattr(self.scorer.llm, 'total_api_calls', 0)
        calls_delta = calls_after - calls_before
        self.iteration_stats.append({
            "iteration": iteration + 1,
            "time": iteration_time,
            "llm_calls": calls_delta
        })
        print(f"Iteration time: {iteration_time:.2f}s — LLM calls: {calls_delta} (total: {calls_after})")
        if is_enabled():
            print_timing(f"local iteration {iteration + 1}", iteration_time)
    
    def _evaluate_candidates(self, candidates: List[PromptNode], validation_examples: List[Example]) -> List[PromptNode]:
        """Оценка кандидатов на валидационном наборе"""
        evaluated = []
        
        for i, candidate in enumerate(candidates):
            # Проверяем, не оценивали ли мы уже этот промпт
            key = candidate.prompt_text
            if key in self._evaluated_prompts:
                print(f"  Candidate {i+1}/{len(candidates)}: Skipped (already evaluated)")
                continue

            existing = self.history.find_node_by_prompt_text(candidate.prompt_text, evaluated_only=True)
            if existing is not None:
                print(f"  Candidate {i+1}/{len(candidates)}: Reused cached evaluation")
                candidate.metrics = deepcopy(existing.metrics)
                candidate.metadata = deepcopy(existing.metadata)
                candidate.is_evaluated = True
                candidate.is_front = existing.is_front
                candidate.evaluation_examples = deepcopy(existing.evaluation_examples)
                candidate.evaluation_examples_by_split = deepcopy(existing.evaluation_examples_by_split)
                self.history.add_node(candidate)
                self._evaluated_prompts.add(key)
                evaluated.append(candidate)
                continue
            
            print(f"  Evaluating candidate {i+1}/{len(candidates)}...", end=" ")
            if is_enabled():
                print(
                    f"\n[diag] candidate details: node_id={candidate.id} "
                    f"prompt_id={prompt_id(candidate.prompt_text)} len={len(candidate.prompt_text)}"
                )

            candidate = self.scorer.evaluate_node(candidate, validation_examples, execute=True, split="validation")
            score = candidate.metrics.composite_score()
            print(f"Score: {score:.3f}")
                
            self.history.add_node(candidate)
            self._evaluated_prompts.add(key)
            evaluated.append(candidate)
        
        return evaluated
    
    def _select_best_candidate(self, candidates: List[PromptNode]) -> Optional[PromptNode]:
        """Выбор лучшего кандидата по композитной оценке"""
        if not candidates:
            return None
        
        candidates_sorted = sorted(candidates, key=lambda c: c.metrics.composite_score(), reverse=True)
        return candidates_sorted[0]
    
    def get_statistics(self) -> Dict:
        """Статистика оптимизации"""
        return {
            "total_iterations": self.total_iterations,
            "improvements_count": self.improvements_count,
            "improvement_rate": self.improvements_count / max(self.total_iterations, 1),
            "iteration_stats": self.iteration_stats,
            "avg_iteration_time": (sum(s["time"] for s in self.iteration_stats) / len(self.iteration_stats)) if self.iteration_stats else None,
            "total_llm_calls_by_local": sum(s["llm_calls"] for s in self.iteration_stats),
            "momentum_buffer_size": len(self._momentum_buffer),
            "step_scale": self._step_scale,
        }
