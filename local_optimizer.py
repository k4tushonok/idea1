from typing import List, Dict, Optional, Set
import time
import random
from data_structures import Example, PromptNode, TextGradient
from history_manager import HistoryManager
from evaluator.scorer import PromptScorer
from text_gradient_generator import TextGradientGenerator
from prompt_editor import PromptEditor
from config import LOCAL_ITERATIONS_PER_GENERATION, MIN_IMPROVEMENT, LOCAL_BATCH_SIZE, CLUSTERING_FAILURE_MULTIPLIER, MAX_CONTEXT_OPERATIONS, MIN_EXAMPLES_FOR_CONTRASTIVE, PATIENCE, SIMILARITY_THRESHOLD

class LocalOptimizer:
    def __init__(self, history_manager: HistoryManager, scorer: PromptScorer, gradient_generator: TextGradientGenerator, prompt_editor: PromptEditor):
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
            starting_node = self.scorer.evaluate_node(starting_node, validation_examples, execute=True)
            self.history.update_node(starting_node.id, starting_node)
            print(f"Starting score: {starting_node.metrics.composite_score():.3f}")
        
        # Текущий лучший узел
        current_best = starting_node
        best_score = current_best.metrics.composite_score()
        no_improve_iters = 0
        
        # Основной цикл оптимизации
        for iteration in range(LOCAL_ITERATIONS_PER_GENERATION):
            iteration_start_time = time.time()
            calls_before = getattr(self.scorer.llm, 'total_api_calls', 0)
            
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Шаг 1: Получаем провалы текущего лучшего промпта
            failure_examples = self._get_failure_examples(current_best, train_examples)
            if not failure_examples:
                print("No failures found - prompt is perfect on training set!")
                break
            success_examples = current_best.evaluation_examples.get("success", [])
            print(f"Failures: {len(failure_examples)}, Successes: {len(success_examples)}")
            
            # Шаг 2: Генерируем текстовые градиенты
            gradients = self._generate_gradients(current_best, failure_examples, success_examples)
            print(f"Generated {len(gradients)} gradients")
            
            # Шаг 3: Создаем варианты на основе градиентов
            candidates = self._generate_candidates(current_best, gradients)
            print(f"Generated {len(candidates)} candidate prompts")
            
            # Шаг 4: Оцениваем кандидатов
            evaluated_candidates = self._evaluate_candidates(candidates, validation_examples)
            print(f"Evaluated {len(evaluated_candidates)} candidates")
            
            # Шаг 5: Выбираем лучшего кандидата
            best_candidate = self._select_best_candidate(evaluated_candidates)
            if best_candidate:
                candidate_score = best_candidate.metrics.composite_score()
                improvement = candidate_score - best_score
                print(f"Best candidate score: {candidate_score:.3f} (Δ {improvement:+.3f})")
                
                # Проверяем, есть ли улучшение
                if candidate_score + 1e-8 >= best_score + MIN_IMPROVEMENT:
                    print(f"✓ Improvement found! New best: {candidate_score:.3f}")
                    current_best = best_candidate
                    best_score = candidate_score
                    no_improve_iters = 0
                    self.improvements_count += 1
                else:
                    print(f"✗ No significant improvement")
                    no_improve_iters += 1
            else:
                print("✗ No valid candidates generated")
                no_improve_iters += 1
            
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
    
    def _get_failure_examples(self, node: PromptNode, examples: List[Example]) -> List[Example]:
        """Получение примеров, на которых промпт ошибается"""
        # Если узел уже оценен, берем из него
        if node.is_evaluated and node.evaluation_examples.get("failures"):
            failures = node.evaluation_examples["failures"]
            # Если провалов много, берем случайное подмножество для разнообразия
            if len(failures) > LOCAL_BATCH_SIZE:
                failures = random.sample(failures, LOCAL_BATCH_SIZE)
            return failures
        
        # Иначе выполняем промпт на примерах
        print(f"Executing prompt on {len(examples)} examples to find failures...")
        executed_examples = self.scorer.execute_prompt_batch(node.prompt_text, examples)
        
        # Фильтруем провалы
        failures = [ex for ex in executed_examples if not ex.is_correct()]
        return failures
    
    def _generate_gradients(self, node: PromptNode, failure_examples: List[Example], success_examples: List[Example]) -> List[TextGradient]:
        """Генерация текстовых градиентов на основе провалов и успехов"""
        print("Generating text gradients...")
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
        
        return gradients
    
    def _generate_candidates(self, parent_node: PromptNode, gradients: List[TextGradient]) -> List[PromptNode]:
        """Генерация кандидатов на основе градиентов"""
        all_candidates: List[PromptNode] = []
        
        for i, gradient in enumerate(gradients):
            print(f"  Generating variants from gradient {i+1}/{len(gradients)}")
            try:
                variants = self.editor.generate_variants(parent_node.prompt_text, gradient, parent_node=parent_node)
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
            candidate = self.scorer.evaluate_node(candidate, validation_examples, execute=True)
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