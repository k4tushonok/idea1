from typing import List, Dict, Optional, Set, Tuple
import time
import random
from data_structures import Example, PromptNode, TextGradient
from history_manager import HistoryManager
from evaluator.scorer import PromptScorer
from text_gradient_generator import TextGradientGenerator
from prompt_editor import PromptEditor
from diagnostics import is_enabled, prompt_id, preview_text
from config import LOCAL_ITERATIONS_PER_GENERATION, MIN_IMPROVEMENT, LOCAL_BATCH_SIZE, CLUSTERING_FAILURE_MULTIPLIER, MAX_CONTEXT_OPERATIONS, MIN_EXAMPLES_FOR_CONTRASTIVE, PATIENCE, SIMILARITY_THRESHOLD

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
        self._evaluated_prompts: Set[str] = set()
        self.llm = llm
    
    def optimize(self, starting_node: PromptNode, train_examples: List[Example], validation_examples: List[Example]) -> PromptNode:
        print(f"\n{'='*60}")
        print(f"Starting Local Optimization")
        print(f"{'='*60}\n")

        # Делим validation: 70% для оценки кандидатов, 30% holdout для подтверждения
        split = int(len(validation_examples) * 0.7)
        eval_examples = validation_examples[:split]
        holdout_examples = validation_examples[split:]

        # Добавляем начальный узел в историю, если его там нет
        if not self.history.get_node(starting_node.id):
            self.history.add_node(starting_node)
        
        # Оцениваем начальный узел, если не оценен
        if not starting_node.is_evaluated:
            print(f"Evaluating starting node...")
            starting_node = self.scorer.evaluate_node(starting_node, eval_examples, execute=True, split="validation")
            self.history.update_node(starting_node.id, starting_node)
            print(f"Starting score: {starting_node.metrics.composite_score():.3f}")
        
        # Текущий лучший узел
        current_best = starting_node
        best_score = current_best.metrics.composite_score()
        no_improve_iters = 0
        prev_real_rate = 1.0
        
        # Основной цикл оптимизации
        for iteration in range(LOCAL_ITERATIONS_PER_GENERATION):
            iteration_start_time = time.time()
            calls_before = getattr(self.scorer.llm, 'total_api_calls', 0)
            
            print(f"\n--- Iteration {iteration + 1} ---")
            if is_enabled():
                print(
                    f"[diag] local iteration state: current_best_id={current_best.id} "
                    f"prompt_id={prompt_id(current_best.prompt_text)} score={best_score:.3f}"
                )

            # Шаг 1: Получаем провалы
            failure_examples, success_examples, real_rate = self._get_train_examples_outcomes(current_best, train_examples)
            if is_enabled():
                print(f"[diag] train outcomes: failures={len(failure_examples)} successes={len(success_examples)}")

            if not failure_examples:
                print("No failures found - prompt is perfect on training set!")
                if holdout_examples:
                    print(f"\nValidation Set Evaluation:")
                    test_metrics = self.scorer.evaluate_prompt(
                        current_best.prompt_text, holdout_examples, execute=True, sample=False)
                    print(f"  Validation score: {test_metrics.composite_score():.3f}")
                    print(f"  Validation accuracy: {test_metrics.metrics['accuracy']:.3f}")
                    print(f"  Validation f1: {test_metrics.metrics['f1']:.3f}")
                break

            # Шаг 2: Генерируем градиенты
            gradients = self._generate_gradients(current_best, failure_examples, success_examples)
            print(f"Generated {len(gradients)} gradients")

            # Шаг 3: Создаём варианты
            candidates = self._generate_candidates(current_best, gradients)
            print(f"Generated {len(candidates)} candidate prompts")

            # Шаг 4: Оцениваем кандидатов на eval_examples (70%)
            evaluated_candidates = self._evaluate_candidates(candidates, eval_examples)
            print(f"Evaluated {len(evaluated_candidates)} candidates")

            # Шаг 5: Выбираем лучшего
            best_candidate = self._select_best_candidate(evaluated_candidates)
            if best_candidate:
                candidate_score = best_candidate.metrics.composite_score()
                improvement = candidate_score - best_score
                print(f"Best candidate score: {candidate_score:.3f} (Δ {improvement:+.3f})")

                if candidate_score + 1e-8 >= best_score + MIN_IMPROVEMENT:
                    # Подтверждаем улучшение на holdout (30%)
                    holdout_metrics = self.scorer.evaluate_prompt(best_candidate.prompt_text, holdout_examples, execute=True, sample=False)
                    holdout_score = holdout_metrics.composite_score()

                    if holdout_score + 1e-8 >= best_score + MIN_IMPROVEMENT:
                        print(f"✓ Improvement found! New best: {candidate_score:.3f} (holdout: {holdout_score:.3f})")
                        current_best = best_candidate
                        best_score = holdout_score  # обновляем по holdout
                        no_improve_iters = 0
                        self.improvements_count += 1

                        print(f"\nValidation Set Evaluation:")
                        print(f"  Validation score: {holdout_score:.3f}")
                        print(f"  Validation accuracy: {holdout_metrics.metrics['accuracy']:.3f}")
                        print(f"  Validation f1: {holdout_metrics.metrics['f1']:.3f}")
                        print(f"  Validation robustness: {holdout_metrics.metrics['robustness']:.3f}")
                        print(f"  Validation efficiency: {holdout_metrics.metrics['efficiency']:.3f}")
                        print(f"  Validation safety: {holdout_metrics.metrics['safety']:.3f}")
                    else:
                        print(f"✗ Improvement not confirmed on holdout: {holdout_score:.3f}")
                        no_improve_iters += 1
                else:
                    print(f"✗ No significant improvement")
                    no_improve_iters += 1
            else:
                print("✗ No valid candidates generated")
                no_improve_iters += 1

            # Early stopping по failure rate
            if real_rate >= prev_real_rate - 0.01:
                no_improve_iters += 1
            prev_real_rate = real_rate

            iteration_time = time.time() - iteration_start_time
            calls_after = getattr(self.scorer.llm, 'total_api_calls', 0)
            calls_delta = calls_after - calls_before
            # Record iteration stats
            self.iteration_stats.append({
                "iteration": iteration + 1,
                "time": iteration_time,
                "llm_calls": calls_delta
            })
            print(f"Iteration time: {iteration_time:.2f}s — LLM calls: {calls_delta} (total: {calls_after})")
            
            self.total_iterations += 1
            
            # Early stopping
            if no_improve_iters >= PATIENCE:
                print(f"\nEarly stopping after {no_improve_iters} iterations without improvement")
                break
        
        print(f"\n{'='*60}")
        print(f"Local Optimization Complete")
        print(f"Final score: {best_score:.3f}")
        print(f"Improvements: {self.improvements_count}")
        print(f"{'='*60}\n")
        
        return current_best
    
    def _get_train_examples_outcomes(self, node: PromptNode, examples: List[Example]) -> Tuple[List[Example], List[Example], float]:
        """Вычисляет train-failures/successes, возвращает реальный failure rate до обрезки."""
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

        failures_for_gradient = (
            random.sample(failures, LOCAL_BATCH_SIZE)
            if len(failures) > LOCAL_BATCH_SIZE
            else failures
        )

        if is_enabled():
            print(
                f"[diag] train outcomes after cap: failures={len(failures_for_gradient)} "
                f"successes={len(successes)} failure_rate={real_rate:.3f}"
            )

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
        
        # Опция 1: Кластеризация провалов и генерация градиентов для каждого кластера
        if len(failure_examples) > LOCAL_BATCH_SIZE * CLUSTERING_FAILURE_MULTIPLIER:
            gradients = self.gradient_gen.generate_clustered_gradients(node, failure_examples, success_examples, context)
        # Опция 2: Батчевая генерация градиентов
        else:
            gradients = self.gradient_gen.generate_gradients_batch(node.prompt_text, failure_examples, success_examples)

        # Добавляем контрастный градиент, если есть примеры для контраста
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
        
        # Фильтруем дубликаты (по содержимому промпта)
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
    
    def _evaluate_candidates(self, candidates: List[PromptNode], validation_examples: List[Example]) -> List[PromptNode]:
        """Оценка кандидатов"""
        evaluated = []
        
        for i, candidate in enumerate(candidates):
            # Проверяем, не оценивали ли мы уже этот промпт
            key = str(hash(candidate.prompt_text))
            if key in self._evaluated_prompts:
                print(f"  Candidate {i+1}/{len(candidates)}: Skipped (already evaluated)")
                continue
            
            print(f"  Evaluating candidate {i+1}/{len(candidates)}...", end=" ")
            if is_enabled():
                print(
                    f"\n[diag] candidate details: node_id={candidate.id} "
                    f"prompt_id={prompt_id(candidate.prompt_text)} len={len(candidate.prompt_text)}"
                )

            candidate = self.scorer.evaluate_node(candidate, validation_examples, execute=True, split="validation", seed_offset=i)
            score = candidate.metrics.composite_score()
            print(f"Score: {score:.3f}")
                
            self.history.add_node(candidate)
            self._evaluated_prompts.add(key)
            evaluated.append(candidate)
        
        return evaluated
    
    def _select_best_candidate(self, candidates: List[PromptNode]) -> Optional[PromptNode]:
        """Выбор лучшего кандидата"""
        if not candidates:
            return None
        
        candidates_sorted = sorted(candidates, key=lambda c: c.metrics.composite_score(), reverse=True)
        return candidates_sorted[0]
    
    def get_statistics(self) -> Dict:
        """Статистика локальной оптимизации"""
        return {
            "total_iterations": self.total_iterations,
            "improvements_count": self.improvements_count,
            "improvement_rate": self.improvements_count / max(self.total_iterations, 1),
            "iteration_stats": self.iteration_stats,
            "avg_iteration_time": (sum(s["time"] for s in self.iteration_stats) / len(self.iteration_stats)) if self.iteration_stats else None,
            "total_llm_calls_by_local": sum(s["llm_calls"] for s in self.iteration_stats)
        }
