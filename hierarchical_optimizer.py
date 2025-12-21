from typing import List, Dict, Optional
import time
from datetime import datetime
import json
import os
from data_structures import (
    Example,
    PromptNode,
    OptimizationConfig,
    OptimizationSource
)
from history_manager import HistoryManager
from evaluator.scorer import PromptScorer
from text_gradient_generator import TextGradientGenerator
from prompt_editor import PromptEditor
from local_optimizer import LocalOptimizer
from global_optimizer import GlobalOptimizer

class HierarchicalOptimizer:
    """
    Главный оркестратор иерархической оптимизации промптов
    
    Архитектура:
    1. Локальная оптимизация - детальные улучшения
    2. Глобальная оптимизация - структурные изменения
    3. Чередование: N локальных итераций -> 1 глобальный шаг
    4. Управление популяцией лучших промптов
    5. Early stopping и конвергенция
    """
    
    def __init__(self, config: OptimizationConfig, api_config: Optional[Dict[str, str]] = None):
        """
        Args:
            config: Конфигурация оптимизации
            api_config: Конфигурация API {"provider": "...", "model": "...", "{provider}_api_key": "..."}
        """
        self.config = config
        self.api_config = api_config or {}
        
        # Инициализация компонентов
        print("Initializing Hierarchical Optimizer...")
        
        # History manager
        self.history = HistoryManager(config)
        
        # Scorer
        self.scorer = PromptScorer(config=config, api_config=api_config)
        
        # Text gradient generator
        self.gradient_gen = TextGradientGenerator(config=config, api_config=api_config)
        
        # Prompt editor
        self.editor = PromptEditor(config=config, api_config=api_config)
        
        # Local optimizer
        self.local_optimizer = LocalOptimizer(config=config, history_manager=self.history, scorer=self.scorer, gradient_generator=self.gradient_gen, prompt_editor=self.editor)
        
        # Global optimizer
        self.global_optimizer = GlobalOptimizer(config=config, history_manager=self.history, scorer=self.scorer, prompt_editor=self.editor, api_config=api_config)
        
        # Метаданные оптимизации
        self.start_time = None
        self.end_time = None
        self.best_node = None
        self.optimization_log = []
        
        print("✓ Initialization complete\n")
    
    # ОСНОВНОЙ МЕТОД ОПТИМИЗАЦИИ
    
    def optimize(self, initial_prompt: str, train_examples: List[Example], validation_examples: List[Example], test_examples: Optional[List[Example]] = None, save_dir: Optional[str] = None) -> PromptNode:
        """
        Запуск полного цикла иерархической оптимизации
        
        Args:
            initial_prompt: Начальный промпт
            train_examples: Обучающие примеры (для градиентов)
            validation_examples: Валидационные примеры (для выбора)
            test_examples: Тестовые примеры (для финальной оценки)
            save_dir: Директория для сохранения результатов
            
        Returns:
            Лучший найденный PromptNode
        """
        self.start_time = time.time()
        
        print("="*80)
        print("HIERARCHICAL PROMPT OPTIMIZATION")
        print("="*80)
        print(f"Configuration:")
        print(f"  Max generations: {self.config.max_generations}")
        print(f"  Local iterations per generation: {self.config.local_iterations_per_generation}")
        print(f"  Global trigger interval: {self.config.global_trigger_interval}")
        print(f"  Population size: {self.config.population_size}")
        print(f"  Patience: {self.config.patience}")
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_examples)}")
        print(f"  Validation: {len(validation_examples)}")
        if test_examples:
            print(f"  Test: {len(test_examples)}")
        print("="*80 + "\n")
        
        # Создаем начальный узел
        initial_node = PromptNode(
            prompt_text=initial_prompt,
            generation=0,
            source=OptimizationSource.INITIAL
        )
        
        # Оцениваем начальный промпт
        print("Evaluating initial prompt...")
        initial_node = self.scorer.evaluate_node(
            initial_node,
            validation_examples,
            execute=True
        )
        
        initial_score = initial_node.metrics.composite_score()
        print(f"Initial score: {initial_score:.3f}")
        print(f"  Accuracy: {initial_node.metrics.metrics['accuracy']:.3f}")
        print(f"  Safety: {initial_node.metrics.metrics['safety']:.3f}")
        print(f"  Robustness: {initial_node.metrics.metrics['robustness']:.3f}")
        print(f"  Efficiency: {initial_node.metrics.metrics['efficiency']:.3f}")
        print(f"  F1: {initial_node.metrics.metrics['f1']:.3f}\n")
        
        # Добавляем в историю
        self.history.add_node(initial_node)
        
        # Инициализация популяции
        population = [initial_node]
        self.best_node = initial_node
        best_score = initial_score
        
        # Отслеживание для early stopping
        generations_without_improvement = 0
        generation = 0  # Инициализируем переменную перед циклом
        
        # Основной цикл оптимизации
        for generation in range(1, self.config.max_generations + 1):
            print("\n" + "="*80)
            print(f"GENERATION {generation}/{self.config.max_generations}")
            print("="*80 + "\n")
            
            generation_start_time = time.time()
            
            # ЛОКАЛЬНАЯ ОПТИМИЗАЦИЯ
            
            print(f"Phase 1: Local Optimization")
            print(f"  Population size: {len(population)}")
            
            new_candidates = []
            
            # Локальная оптимизация для каждого узла в популяции
            for i, node in enumerate(population, 1):
                print(f"\n  Optimizing node {i}/{len(population)} (score: {node.metrics.composite_score():.3f})")
                
                try:
                    improved_node = self.local_optimizer.optimize(
                        starting_node=node,
                        train_examples=train_examples,
                        validation_examples=validation_examples,
                        num_iterations=self.config.local_iterations_per_generation
                    )
                    
                    new_candidates.append(improved_node)
                    
                except Exception as e:
                    print(f"  Error in local optimization: {e}")
                    new_candidates.append(node)  # Сохраняем оригинальный узел
            
            # ГЛОБАЛЬНАЯ ОПТИМИЗАЦИЯ (если триггер сработал)
            
            if self.global_optimizer.should_trigger_global_step(generation):
                print(f"\nPhase 2: Global Optimization (Triggered)")
                
                try:
                    global_candidates = self.global_optimizer.optimize(
                        current_generation=generation,
                        train_examples=train_examples,
                        validation_examples=validation_examples
                    )
                    
                    # Локальная оптимизация для каждого глобального кандидата
                    print(f"\nRefining {len(global_candidates)} global candidates with local optimization...")
                    
                    for i, global_candidate in enumerate(global_candidates, 1):
                        print(f"\n  Refining global candidate {i}/{len(global_candidates)}")
                        
                        try:
                            refined = self.local_optimizer.optimize(
                                starting_node=global_candidate,
                                train_examples=train_examples,
                                validation_examples=validation_examples,
                                num_iterations=self.config.local_iterations_per_generation // 2  # Меньше итераций
                            )
                            new_candidates.append(refined)
                            
                        except Exception as e:
                            print(f"    Error refining: {e}")
                            new_candidates.append(global_candidate)
                    
                except Exception as e:
                    print(f"Error in global optimization: {e}")
            
            else:
                print(f"\nPhase 2: Global Optimization (Skipped)")
            
            # ОБНОВЛЕНИЕ ПОПУЛЯЦИИ
            
            print(f"\nPhase 3: Population Update")
            
            # Выбираем лучших для следующего поколения
            population = self._select_population(
                new_candidates,
                population_size=self.config.population_size
            )
            
            # Обновляем лучший узел
            generation_best = max(population, key=lambda n: n.metrics.composite_score())
            generation_best_score = generation_best.metrics.composite_score()
            
            print(f"\n  Generation best: {generation_best_score:.3f}")
            print(f"  Overall best: {best_score:.3f}")
            
            # Проверяем улучшение
            improvement = generation_best_score - best_score
            
            if improvement + 1e-8 >= self.config.min_improvement:
                print(f"  ✓ Improvement: +{improvement:.3f}")
                self.best_node = generation_best
                best_score = generation_best_score
                generations_without_improvement = 0
            else:
                print(f"  ✗ No significant improvement")
                generations_without_improvement += 1
            
            # Логирование
            generation_time = time.time() - generation_start_time
            log_entry = {
                "generation": generation,
                "best_score": generation_best_score,
                "overall_best": best_score,
                "improvement": improvement,
                "population_size": len(population),
                "time": generation_time
            }
            self.optimization_log.append(log_entry)
            
            print(f"\n  Generation time: {generation_time:.2f}s")
            
            # Промежуточное сохранение
            if save_dir and generation % 5 == 0:
                self._save_checkpoint(save_dir, generation)
            
            # EARLY STOPPING
            
            if generations_without_improvement >= self.config.patience:
                print(f"\n{'='*80}")
                print(f"EARLY STOPPING")
                print(f"No improvement for {generations_without_improvement} generations")
                print(f"{'='*80}\n")
                break
        
        # ФИНАЛИЗАЦИЯ
        
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"\nResults:")
        print(f"  Initial score: {initial_score:.3f}")
        print(f"  Final score: {best_score:.3f}")
        print(f"  Improvement: +{best_score - initial_score:.3f} ({((best_score - initial_score) / max(initial_score, 0.01) * 100):.1f}%)")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Generations: {generation}")
        
        # Финальная оценка на тестовом наборе
        if test_examples:
            print(f"\nTest Set Evaluation:")
            test_metrics = self.scorer.evaluate_prompt(
                self.best_node.prompt_text,
                test_examples,
                execute=True
            )
            print(f"  Test score: {test_metrics.composite_score():.3f}")
            print(f"  Test accuracy: {test_metrics.metrics['accuracy']:.3f}")
        
        # Сохранение результатов
        if save_dir:
            self._save_final_results(save_dir, test_examples)
        
        print("="*80 + "\n")
        
        return self.best_node
    
    # УПРАВЛЕНИЕ ПОПУЛЯЦИЕЙ
    
    def _select_population(self, candidates: List[PromptNode], population_size: int) -> List[PromptNode]:
        """
        Выбор популяции для следующего поколения
        Балансирует между качеством (exploitation) и разнообразием (exploration)
        
        Args:
            candidates: Кандидаты для выбора
            population_size: Размер популяции
            
        Returns:
            Выбранная популяция
        """
        if len(candidates) <= population_size:
            return candidates
        
        # Находим и сохраняем абсолютно лучшего кандидата
        best_candidate = max(candidates, key=lambda n: n.metrics.composite_score())
        
        # Стратегия 1: Паретто-фронт (если есть)
        front = self.history.get_pareto_front()
        front_ids = {node.id for node in front}
        
        front_candidates = [c for c in candidates if c.id in front_ids]
        
        # Стратегия 2: Топ по композитной метрике
        candidates_sorted = sorted(
            candidates,
            key=lambda n: n.metrics.composite_score(),
            reverse=True
        )
        
        # Комбинируем: берем фронт + лучших по метрике
        selected = []
        selected_ids = set()
        
        # Первым добавляем абсолютно лучшего кандидата
        selected.append(best_candidate)
        selected_ids.add(best_candidate.id)
        
        # Добавляем узлы с фронта
        for node in front_candidates:
            if node.id not in selected_ids and len(selected) < population_size:
                selected.append(node)
                selected_ids.add(node.id)
        
        # Дополняем лучшими по метрике
        for node in candidates_sorted:
            if node.id not in selected_ids and len(selected) < population_size:
                selected.append(node)
                selected_ids.add(node.id)
        
        # Стратегия 3: Добавляем diversity если нужно
        if len(selected) < population_size:
            # Вычисляем разнообразие оставшихся кандидатов
            remaining = [c for c in candidates if c.id not in selected_ids]
            
            for candidate in remaining:
                # Вычисляем минимальное расстояние до уже выбранных
                min_distance = min(
                    self.scorer.calculate_edit_distance(
                        candidate.prompt_text,
                        selected_node.prompt_text
                    )
                    for selected_node in selected
                )
                
                candidate.metadata["diversity_score"] = min_distance
            
            # Сортируем по diversity
            remaining.sort(key=lambda n: n.metadata.get("diversity_score", 0), reverse=True)
            
            # Добавляем самых разнообразных
            for node in remaining:
                if len(selected) >= population_size:
                    break
                selected.append(node)
                selected_ids.add(node.id)
        
        print(f"  Selected population ({len(selected)}):")
        print(f"    Best: {best_candidate.metrics.composite_score():.3f} (generation {best_candidate.generation})")
        print(f"    Pareto: {len([n for n in selected if n.id in front_ids])}")
        
        return selected[:population_size]
    
    # АНАЛИЗ И ОТЧЕТЫ
    
    def get_optimization_report(self) -> Dict:
        """
        Генерация детального отчета об оптимизации
        
        Returns:
            Словарь с полным отчетом
        """
        if not self.best_node:
            return {"error": "Optimization not run yet"}
        
        # Базовая информация
        report = {
            "optimization_info": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_time_seconds": self.end_time - self.start_time if self.end_time else None,
                "generations": len(self.optimization_log),
                "config": self.config.to_dict()
            },
            
            # Результаты
            "results": {
                "best_score": self.best_node.metrics.composite_score(),
                "best_metrics": self.best_node.metrics.to_dict(),
                "best_prompt": self.best_node.prompt_text,
                "best_node_id": self.best_node.id
            },
            
            # История оптимизации
            "optimization_log": self.optimization_log,
            
            # Статистика компонентов
            "component_statistics": {
                "history": self.history.get_statistics(),
                "local_optimizer": self.local_optimizer.get_statistics(),
                "global_optimizer": self.global_optimizer.get_statistics()
            },
            
            # Траектория лучшего узла
            "best_node_lineage": [
                {
                    "id": node.id,
                    "generation": node.generation,
                    "source": node.source.value,
                    "score": node.metrics.composite_score() if node.is_evaluated else None
                }
                for node in self.history.get_lineage(self.best_node.id)
            ],
            
            # Успешные стратегии
            "best_global_strategies": self.global_optimizer.get_best_strategies(top_k=5)
        }
        
        return report
    
    def visualize_optimization_trajectory(self) -> str:
        """
        Создание текстовой визуализации траектории оптимизации
        
        Returns:
            Строка с визуализацией
        """
        if not self.optimization_log:
            return "No optimization data available"
        
        viz = "\n" + "="*80 + "\n"
        viz += "OPTIMIZATION TRAJECTORY\n"
        viz += "="*80 + "\n\n"
        
        # График прогресса
        viz += "Generation | Best Score | Overall Best | Improvement\n"
        viz += "-"*60 + "\n"
        
        for entry in self.optimization_log:
            gen = entry["generation"]
            gen_best = entry["best_score"]
            overall_best = entry["overall_best"]
            improvement = entry.get("improvement", 0)
            
            # Простой ASCII bar для визуализации
            bar_length = int(gen_best * 50)
            bar = "█" * bar_length
            
            viz += f"{gen:4d}       | {gen_best:.3f}      | {overall_best:.3f}       | "
            viz += f"{improvement:+.3f} {bar}\n"
        
        viz += "\n" + "="*80 + "\n"
        
        return viz
    
    # СОХРАНЕНИЕ И ЗАГРУЗКА
    
    def _save_checkpoint(self, save_dir: str, generation: int):
        """Сохранение checkpoint'а"""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(save_dir, f"checkpoint_gen_{generation}.json")
        self.history.save(checkpoint_path)
        
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    def _save_final_results(self, save_dir: str, test_examples: Optional[List[Example]] = None):
        """Сохранение финальных результатов"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Сохраняем историю
        history_path = os.path.join(save_dir, "optimization_history.json")
        self.history.save(history_path)
        
        # 2. Сохраняем отчет
        report = self.get_optimization_report()
        report_path = os.path.join(save_dir, "optimization_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 3. Сохраняем лучший промпт
        best_prompt_path = os.path.join(save_dir, "best_prompt.txt")
        with open(best_prompt_path, 'w', encoding='utf-8') as f:
            f.write(self.best_node.prompt_text)
        
        # 4. Сохраняем траекторию
        trajectory_path = os.path.join(save_dir, "trajectory.txt")
        with open(trajectory_path, 'w', encoding='utf-8') as f:
            f.write(self.visualize_optimization_trajectory())
        
        print(f"\nResults saved to: {save_dir}")
        print(f"  - optimization_history.json")
        print(f"  - optimization_report.json")
        print(f"  - best_prompt.txt")
        print(f"  - trajectory.txt")

    # СРАВНЕНИЕ С BASELINE
    
    def compare_with_baseline(self, baseline_prompt: str, test_examples: List[Example]) -> Dict:
        """
        Сравнение оптимизированного промпта с baseline
        
        Args:
            baseline_prompt: Baseline промпт
            test_examples: Тестовые примеры
            
        Returns:
            Словарь с результатами сравнения
        """
        if not self.best_node:
            raise ValueError("Optimization not run yet")
        
        print("\nComparing with baseline...")
        
        # Оцениваем baseline
        baseline_metrics = self.scorer.evaluate_prompt(
            baseline_prompt,
            test_examples,
            execute=True
        )
        
        # Оцениваем оптимизированный
        optimized_metrics = self.scorer.evaluate_prompt(
            self.best_node.prompt_text,
            test_examples,
            execute=True
        )
        
        comparison = {
            "baseline": {
                "composite_score": baseline_metrics.composite_score(),
                "accuracy": baseline_metrics.metrics["accuracy"],
                "safety": baseline_metrics.metrics["safety"],
                "robustness": baseline_metrics.metrics["robustness"],
                "efficiency": baseline_metrics.metrics["efficiency"],
                "f1": baseline_metrics.metrics["f1"]
            },
            "optimized": {
                "composite_score": optimized_metrics.composite_score(),
                "accuracy": optimized_metrics.metrics["accuracy"],
                "safety": optimized_metrics.metrics["safety"],
                "robustness": optimized_metrics.metrics["robustness"],
                "efficiency": optimized_metrics.metrics["efficiency"],
                "f1": optimized_metrics.metrics["f1"]
            },
            "improvements": {
                "composite_score": optimized_metrics.composite_score() - baseline_metrics.composite_score(),
                "accuracy": optimized_metrics.metrics["accuracy"] - baseline_metrics.metrics["accuracy"],
                "safety": optimized_metrics.metrics["safety"] - baseline_metrics.metrics["safety"],
                "robustness": optimized_metrics.metrics["robustness"] - baseline_metrics.metrics["robustness"],
                "efficiency": optimized_metrics.metrics["efficiency"] - baseline_metrics.metrics["efficiency"],
                "f1": optimized_metrics.metrics["f1"] - baseline_metrics.metrics["f1"]
            }
        }
        
        print("\nComparison Results:")
        print(f"  Baseline score: {baseline_metrics.composite_score():.3f}")
        print(f"  Optimized score: {optimized_metrics.composite_score():.3f}")
        print(f"  Improvement: {comparison['improvements']['composite_score']:+.3f}")
        
        return comparison
    
    def __repr__(self):
        if self.best_node:
            return f"HierarchicalOptimizer(best_score={self.best_node.metrics.composite_score():.3f}, generations={len(self.optimization_log)})"
        else:
            return "HierarchicalOptimizer(not optimized yet)"