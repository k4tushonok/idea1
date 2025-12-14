from typing import List, Dict, Optional, Set
import numpy as np
from collections import defaultdict
import time

from data_structures import (
    Example,
    PromptNode,
    TextGradient,
    OptimizationConfig
)
from history_manager import HistoryManager
from scorer import PromptScorer
from text_gradient_generator import TextGradientGenerator
from prompt_editor import PromptEditor

class LocalOptimizer:
    """Локальный оптимизатор - выполняет итеративное улучшение промпта на основе текстовых градиентов"""
    
    def __init__(self, config: OptimizationConfig, history_manager: HistoryManager, scorer: PromptScorer, gradient_generator: TextGradientGenerator, prompt_editor: PromptEditor):
        """
        Args:
            config: Конфигурация оптимизации
            history_manager: Менеджер истории
            scorer: Система оценки промптов
            gradient_generator: Генератор текстовых градиентов
            prompt_editor: Редактор промптов
        """
        self.config = config
        self.history = history_manager
        self.scorer = scorer
        self.gradient_gen = gradient_generator
        self.editor = prompt_editor
        
        # Статистика локальной оптимизации
        self.total_iterations = 0
        self.total_candidates_generated = 0
        self.total_candidates_evaluated = 0
        self.improvements_count = 0
        
        # Кэш для предотвращения повторной оценки
        self._evaluated_prompts: Set[str] = set()
    
    # ОСНОВНОЙ ЦИКЛ ЛОКАЛЬНОЙ ОПТИМИЗАЦИИ
    
    def optimize(self, starting_node: PromptNode, train_examples: List[Example], validation_examples: List[Example], num_iterations: Optional[int] = None) -> PromptNode:
        """
        Запуск полного цикла локальной оптимизации
        
        Args:
            starting_node: Начальный узел (промпт)
            train_examples: Обучающие примеры
            validation_examples: Валидационные примеры (для выбора лучшего)
            num_iterations: Количество итераций (по умолчанию из конфига)
            
        Returns:
            Лучший найденный узел
        """
        if num_iterations is None:
            num_iterations = self.config.local_iterations_per_generation
        
        print(f"\n{'='*60}")
        print(f"Starting Local Optimization")
        print(f"Starting node: {starting_node.id[:8]}")
        print(f"Generation: {starting_node.generation}")
        print(f"Iterations: {num_iterations}")
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
        
        # Отслеживание для early stopping
        iterations_without_improvement = 0
        
        # Основной цикл оптимизации
        for iteration in range(num_iterations):
            print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
            
            iteration_start_time = time.time()
            
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
            self.total_candidates_generated += len(candidates)
            
            # Шаг 4: Оцениваем кандидатов
            evaluated_candidates = self._evaluate_candidates(candidates, validation_examples)
            
            self.total_candidates_evaluated += len(evaluated_candidates)
            
            # Шаг 5: Выбираем лучшего кандидата
            best_candidate = self._select_best_candidate(evaluated_candidates, current_best)
            
            if best_candidate:
                candidate_score = best_candidate.metrics.composite_score()
                improvement = candidate_score - best_score
                
                print(f"Best candidate score: {candidate_score:.3f} (Δ {improvement:+.3f})")
                
                # Проверяем, есть ли улучшение
                if candidate_score + 1e-8 >= best_score + self.config.min_improvement:
                    print(f"✓ Improvement found! New best: {candidate_score:.3f}")
                    current_best = best_candidate
                    best_score = candidate_score
                    iterations_without_improvement = 0
                    self.improvements_count += 1
                else:
                    print(f"✗ No significant improvement")
                    iterations_without_improvement += 1
            else:
                print("✗ No valid candidates generated")
                iterations_without_improvement += 1
            
            iteration_time = time.time() - iteration_start_time
            print(f"Iteration time: {iteration_time:.2f}s")
            
            # Early stopping
            if iterations_without_improvement >= self.config.patience:
                print(f"\nEarly stopping: {iterations_without_improvement} iterations without improvement")
                break
            
            self.total_iterations += 1
        
        print(f"\n{'='*60}")
        print(f"Local Optimization Complete")
        print(f"Final score: {best_score:.3f}")
        print(f"Improvements: {self.improvements_count}")
        print(f"{'='*60}\n")
        
        return current_best

    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    
    def _get_failure_examples(self, node: PromptNode, examples: List[Example]) -> List[Example]:
        """
        Получение примеров, на которых промпт ошибается
        
        Args:
            node: Узел для проверки
            examples: Примеры для тестирования
            
        Returns:
            Список примеров с ошибками
        """
        # Если узел уже оценен, берем из него
        if node.is_evaluated and node.evaluation_examples.get("failures"):
            failures = node.evaluation_examples["failures"]
            
            # Если провалов много, берем случайное подмножество для разнообразия
            if len(failures) > self.config.local_batch_size * 3:
                import random
                failures = random.sample(failures, self.config.local_batch_size * 3)
            
            return failures
        
        # Иначе выполняем промпт на примерах
        print(f"Executing prompt on {len(examples)} examples to find failures...")
        executed_examples = self.scorer.execute_prompt_batch(node.prompt_text, examples)
        
        # Фильтруем провалы
        failures = [ex for ex in executed_examples if not ex.is_correct()]
        
        return failures
    
    def _generate_gradients(self, node: PromptNode, failure_examples: List[Example], success_examples: List[Example]) -> List[TextGradient]:
        """
        Генерация текстовых градиентов
        
        Args:
            node: Текущий узел
            failure_examples: Примеры провалов
            success_examples: Примеры успехов
            
        Returns:
            Список текстовых градиентов
        """
        # Контекст для генератора градиентов
        context = {
            "generation": node.generation,
            "previous_attempts": len(self.history.get_lineage(node.id)),
        }
        
        # Если есть история, добавляем информацию об успешных операциях
        if node.generation > 0:
            successful_ops = self.history.analyze_successful_operations()
            if successful_ops:
                context["successful_operations"] = list(successful_ops.keys())[:5]
        
        # Опция 1: Кластеризация провалов и генерация градиентов для каждого кластера
        if len(failure_examples) > self.config.local_batch_size * 2:
            print("Clustering failures by error type...")
            clusters = self.gradient_gen.cluster_failure_types(failure_examples)
            
            gradients = []
            for cluster_name, cluster_failures in clusters.items():
                if not cluster_failures:
                    continue
                
                print(f"  Cluster '{cluster_name}': {len(cluster_failures)} failures")
                
                gradient = self.gradient_gen.generate_gradient(
                    node.prompt_text,
                    cluster_failures[:self.config.local_batch_size],
                    success_examples[:5],
                    context
                )
                gradient.metadata["cluster"] = cluster_name
                gradients.append(gradient)
            
            # Ограничиваем количество градиентов
            gradients.sort(key=lambda g: g.priority, reverse=True)
            gradients = gradients[:self.config.local_candidates_per_iteration]
        
        # Опция 2: Батчевая генерация градиентов
        else:
            gradients = self.gradient_gen.generate_gradients_batch(
                node.prompt_text,
                failure_examples,
                success_examples,
                batch_size=self.config.local_batch_size,
                num_gradients=self.config.local_candidates_per_iteration
            )
        
        # Добавляем контрастный градиент, если есть хорошие примеры для контраста
        if len(success_examples) >= 3 and len(failure_examples) >= 3:
            print("Generating contrastive gradient...")
            try:
                contrastive_gradient = self.gradient_gen.generate_contrastive_gradient(
                    node.prompt_text,
                    failure_examples[:5],
                    success_examples[:5]
                )
                gradients.insert(0, contrastive_gradient)  # Добавляем в начало
            except Exception as e:
                print(f"Failed to generate contrastive gradient: {e}")
        
        return gradients
    
    def _generate_candidates(self, parent_node: PromptNode, gradients: List[TextGradient]) -> List[PromptNode]:
        """
        Генерация кандидатов на основе градиентов
        
        Args:
            parent_node: Родительский узел
            gradients: Список градиентов
            
        Returns:
            Список кандидатов (PromptNode)
        """
        all_candidates = []
        
        for i, gradient in enumerate(gradients):
            print(f"  Generating variants from gradient {i+1}/{len(gradients)} (priority: {gradient.priority:.2f})...")
            
            try:
                # Генерируем варианты для этого градиента
                # Количество вариантов зависит от приоритета градиента
                num_variants = max(1, int(gradient.priority * self.config.local_candidates_per_iteration))
                
                variants = self.editor.generate_variants(
                    parent_node.prompt_text,
                    gradient,
                    num_variants=num_variants,
                    parent_node=parent_node
                )
                
                all_candidates.extend(variants)
                
            except Exception as e:
                print(f"    Error generating variants: {e}")
                continue
        
        # Фильтруем дубликаты (по содержимому промпта)
        unique_candidates = self._filter_duplicates(all_candidates)
        
        print(f"  Generated {len(all_candidates)} variants, {len(unique_candidates)} unique")
        
        return unique_candidates
    
    def _evaluate_candidates(self, candidates: List[PromptNode], validation_examples: List[Example]) -> List[PromptNode]:
        """
        Оценка кандидатов
        
        Args:
            candidates: Список кандидатов
            validation_examples: Валидационные примеры
            
        Returns:
            Оцененные кандидаты
        """
        evaluated = []
        
        for i, candidate in enumerate(candidates):
            # Проверяем, не оценивали ли мы уже этот промпт
            prompt_hash = hash(candidate.prompt_text)
            if str(prompt_hash) in self._evaluated_prompts:
                print(f"  Candidate {i+1}/{len(candidates)}: Skipped (already evaluated)")
                continue
            
            print(f"  Evaluating candidate {i+1}/{len(candidates)}...", end=" ")
            
            try:
                # Оцениваем кандидата
                candidate = self.scorer.evaluate_node(
                    candidate,
                    validation_examples,
                    execute=True
                )
                
                score = candidate.metrics.composite_score()
                print(f"Score: {score:.3f}")
                
                # Добавляем в историю
                self.history.add_node(candidate)
                
                # Помечаем как оцененный
                self._evaluated_prompts.add(str(prompt_hash))
                
                evaluated.append(candidate)
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        return evaluated
    
    def _select_best_candidate(self, candidates: List[PromptNode], current_best: PromptNode) -> Optional[PromptNode]:
        """
        Выбор лучшего кандидата
        
        Args:
            candidates: Оцененные кандидаты
            current_best: Текущий лучший узел
            
        Returns:
            Лучший кандидат или None
        """
        if not candidates:
            return None
        
        # Сортируем по базовому скору (composite_score)
        # Diversity используется только на уровне популяции, а не на уровне выбора лучшего
        candidates_sorted = sorted(
            candidates,
            key=lambda c: c.metrics.composite_score(),
            reverse=True
        )
        
        # Выводим топ-3 для информации
        print("\n  Top candidates (by base score):")
        for i, cand in enumerate(candidates_sorted[:3], 1):
            score = cand.metrics.composite_score()
            diversity = self.editor.calculate_edit_distance(
                cand.prompt_text,
                current_best.prompt_text
            )
            print(f"    {i}. Score: {score:.3f} (diversity: {diversity:.3f})")
        
        best_candidate = candidates_sorted[0]
        
        return best_candidate
    
    def _filter_duplicates(self, candidates: List[PromptNode]) -> List[PromptNode]:
        """
        Фильтрация дубликатов кандидатов
        Два промпта считаются дубликатами, если их similarity выше порога
        
        Args:
            candidates: Список кандидатов
            
        Returns:
            Уникальные кандидаты
        """
        if not candidates:
            return []
        
        unique = [candidates[0]]
        
        for candidate in candidates[1:]:
            is_duplicate = False
            
            for existing in unique:
                similarity = 1.0 - self.editor.calculate_edit_distance(
                    candidate.prompt_text,
                    existing.prompt_text
                )
                
                if similarity > self.config.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(candidate)
        
        return unique
    
    # ДОПОЛНИТЕЛЬНЫЕ МЕТОДЫ ДЛЯ АНАЛИЗА
 
    def get_statistics(self) -> Dict:
        """Статистика локальной оптимизации"""
        return {
            "total_iterations": self.total_iterations,
            "total_candidates_generated": self.total_candidates_generated,
            "total_candidates_evaluated": self.total_candidates_evaluated,
            "improvements_count": self.improvements_count,
            "improvement_rate": self.improvements_count / max(self.total_iterations, 1),
            "avg_candidates_per_iteration": self.total_candidates_generated / max(self.total_iterations, 1),
        }
    
    def __repr__(self):
        stats = self.get_statistics()
        return f"LocalOptimizer(iterations={stats['total_iterations']}, improvements={stats['improvements_count']})"