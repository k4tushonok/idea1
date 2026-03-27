from typing import List, Dict, Optional
import time
import json
import os
from data_structures import Example, PromptNode, OptimizationSource
from history_manager import HistoryManager
from evaluator.scorer import PromptScorer
from text_gradient_generator import TextGradientGenerator
from prompt_editor import PromptEditor
from local_optimizer import LocalOptimizer
from global_optimizer import GlobalOptimizer
from llm.llm_client import create_llm
from diagnostics import (
    is_enabled, prompt_id,
    print_population, print_timing, sources_summary, scores_summary, print_candidates_summary
)
from config import MAX_GENERATIONS, POPULATION_SIZE, MIN_IMPROVEMENT, PATIENCE, TOP_BEST_NODES, DIVERSITY_WEIGHT, GLOBAL_REFINE_WITH_LOCAL, STAGE3_TOP_K, REFLECTION_ENABLED

class HierarchicalOptimizer:
    def __init__(self):
        self.llm = create_llm()
        self.history = HistoryManager()
        self.scorer = PromptScorer(llm=self.llm)
        self.gradient_gen = TextGradientGenerator(llm=self.llm)
        self.editor = PromptEditor(llm=self.llm)
        self.local_optimizer = LocalOptimizer(history_manager=self.history, scorer=self.scorer, gradient_generator=self.gradient_gen, prompt_editor=self.editor, llm=self.llm)
        self.global_optimizer = GlobalOptimizer(history_manager=self.history, scorer=self.scorer, prompt_editor=self.editor, llm=self.llm)
        
        # Метаданные оптимизации
        self.start_time = None
        self.end_time = None
        self.best_node = None
        self.optimization_log = []
        # Межпоколенческая рефлексия
        self._reflection_context = ""
    
    def optimize(self, initial_prompt: str, train_examples: List[Example], validation_examples: List[Example], test_examples: Optional[List[Example]] = None, save_dir: Optional[str] = None) -> PromptNode:
        self.start_time = time.time()
        
        # Создаем начальный узел
        initial_node = PromptNode(
            prompt_text=initial_prompt,
            generation=0,
            source=OptimizationSource.INITIAL
        )
        
        # Оцениваем начальный промпт
        print("Evaluating initial prompt...")
        if is_enabled():
            print(
                f"[diag] initial node: prompt_id={prompt_id(initial_prompt)} "
                f"len={len(initial_prompt)} chars"
            )
        initial_node = self.scorer.evaluate_node(
            initial_node,
            validation_examples,
            execute=True,
            split="validation",
        )
        
        initial_score = initial_node.metrics.composite_score()
        print(f"Initial score: {initial_score:.3f}")
        for name, value in sorted(initial_node.metrics.metrics.items()):
            print(f"  {name}: {value:.3f}")
        print()
        
        # Добавляем в историю
        self.history.add_node(initial_node)
        
        # Инициализация популяции
        population = [initial_node]
        self.best_node = initial_node
        best_score = initial_score
        
        # Отслеживание для early stopping
        generations_without_improvement = 0
        generation = 0
        
        # Основной цикл оптимизации
        for generation in range(1, MAX_GENERATIONS + 1):
            print("\n" + "="*80)
            print(f"GENERATION {generation}/{MAX_GENERATIONS}")
            print("="*80 + "\n")
            
            generation_start_time = time.time()
            llm_calls_gen_start = getattr(self.llm, 'total_api_calls', 0)
            
            # ЛОКАЛЬНАЯ ОПТИМИЗАЦИЯ
            
            print(f"Phase 1: Local Optimization")
            print(f"  Population size: {len(population)}")
            if is_enabled():
                print_population("population before local opt", population)
                print(f"[diag] population sources: {sources_summary(population)}")
            
            new_candidates = []
            
            # Оптимизируем только лучший узел популяции (остальные проходят без изменений)
            best_pop_node = max(population, key=lambda n: n.selection_score())
            
            for i, node in enumerate(population, 1):
                if node.id != best_pop_node.id:
                    # Сохраняем неоптимизированные узлы для разнообразия
                    new_candidates.append(node)
                    continue
                    
                print(f"\n  Optimizing best node (score: {node.selection_score():.3f})")
                if is_enabled():
                    print(
                        f"[diag] population node: node_id={node.id} "
                        f"prompt_id={prompt_id(node.prompt_text)} gen={node.generation}"
                    )
                
                try:
                    self.local_optimizer._evaluated_prompts.clear()
                    self.editor._cache.clear()
                    self.gradient_gen._cache.clear()
                    
                    # Передаём контекст рефлексии в генератор градиентов
                    if self._reflection_context:
                        self.gradient_gen.reflection_context = self._reflection_context
                    
                    improved_node = self.local_optimizer.optimize(
                        starting_node=node,
                        train_examples=train_examples,
                        validation_examples=validation_examples
                    )
                    
                    delta = improved_node.selection_score() - node.selection_score()
                    print(f"  Best node score: {node.selection_score():.3f} → {improved_node.selection_score():.3f} (Δ {delta:+.3f})")
                    new_candidates.append(improved_node)
                    
                except Exception as e:
                    print(f"  Error in local optimization: {e}")
                    new_candidates.append(node)
            
            # ГЛОБАЛЬНАЯ ОПТИМИЗАЦИЯ (если триггер сработал)
            
            if self.global_optimizer.should_trigger_global_step(generation):
                print(f"\nPhase 2: Global Optimization (Triggered)")
                
                try:
                    # Передаём контекст рефлексии в глобальный оптимизатор
                    self.global_optimizer.reflection_context = self._reflection_context
                    
                    global_candidates = self.global_optimizer.optimize(
                        current_generation=generation,
                        train_examples=train_examples,
                        validation_examples=validation_examples
                    )
                    
                    if global_candidates:
                        # Сортируем и берём только топ
                        global_candidates_sorted = sorted(global_candidates, key=lambda n: n.selection_score(), reverse=True)
                        top_global = global_candidates_sorted[:2]
                        
                        if GLOBAL_REFINE_WITH_LOCAL:
                            print(f"\nRefining {len(top_global)} global candidates with local optimization...")
                            
                            for i, global_candidate in enumerate(top_global, 1):
                                print(f"\n  Refining global candidate {i}/{len(top_global)}")
                                try:
                                    self.local_optimizer._evaluated_prompts.clear()
                                    self.local_optimizer._train_outcomes_cache.clear()
                                    self.editor._cache.clear()
                                    self.gradient_gen._cache.clear()
                                    
                                    refined = self.local_optimizer.optimize(
                                        starting_node=global_candidate,
                                        train_examples=train_examples,
                                        validation_examples=validation_examples
                                    )
                                    new_candidates.append(refined)
                                    
                                except Exception as e:
                                    print(f"    Error refining: {e}")
                                    new_candidates.append(global_candidate)
                        else:
                            print(f"\nInjecting {len(top_global)} global candidates into population (no local refinement)")
                            new_candidates.extend(top_global)
                    
                except Exception as e:
                    print(f"Error in global optimization: {e}")
            
            else:
                print(f"\nPhase 2: Global Optimization (Skipped)")
            
            # ОБНОВЛЕНИЕ ПОПУЛЯЦИИ
            
            print(f"\nPhase 3: Population Update")
            
            # Выбираем лучших для следующего поколения
            population = self._select_population(
                new_candidates,
                population_size=POPULATION_SIZE
            )
            
            # Обновляем лучший узел
            generation_best = max(population, key=lambda n: n.selection_score())
            generation_best_score = generation_best.selection_score()
            
            llm_calls_gen_end = getattr(self.llm, 'total_api_calls', 0)
            llm_calls_gen_delta = llm_calls_gen_end - llm_calls_gen_start
            print(f"\n  Generation best: {generation_best_score:.3f}")
            print(f"  Overall best: {best_score:.3f}")
            print(f"  LLM calls this generation: {llm_calls_gen_delta} (total: {llm_calls_gen_end})")
            if is_enabled():
                print(
                    f"[diag] generation best node: node_id={generation_best.id} "
                    f"prompt_id={prompt_id(generation_best.prompt_text)}"
                )
                print_population(f"population after selection gen={generation}", population)
                print(f"[diag] population sources after selection: {sources_summary(population)}")
                print_candidates_summary(f"all new_candidates gen={generation}", new_candidates)
            
            # Проверяем улучшение
            improvement = generation_best_score - best_score
            
            if improvement + 1e-8 >= MIN_IMPROVEMENT:
                print(f"  ✓ Improvement: +{improvement:.3f}")
                prev_best = self.best_node
                self.best_node = generation_best
                best_score = generation_best_score
                generations_without_improvement = 0
                
                # Генерируем рефлексию для следующего поколения
                if REFLECTION_ENABLED:
                    try:
                        self._reflection_context = self._generate_reflection(
                            prev_best, generation_best, train_examples
                        )
                    except Exception as e:
                        print(f"  Reflection generation failed: {e}")
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
            if is_enabled():
                print_timing(f"generation {generation}", generation_time)
            
            # EARLY STOPPING
            if generations_without_improvement >= PATIENCE:
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
        print(f"  Final score (sampled): {best_score:.3f}")

        # Переоценка лучшего промпта на полном валидационном наборе
        print(f"\n  Re-evaluating best prompt on full validation ({len(validation_examples)} examples)...")
        full_val_metrics = self.scorer.evaluate_prompt(
            self.best_node.prompt_text,
            validation_examples,
            execute=True,
            sample=False,
            stage=1,
        )
        full_val_score = full_val_metrics.composite_score()
        print(f"  Full validation score: {full_val_score:.3f}")
        self.best_node.metadata["full_validation_score"] = full_val_score
        self.best_node.metadata["full_validation_metrics"] = full_val_metrics.to_dict()

        print(f"  Improvement: +{best_score - initial_score:.3f} ({((best_score - initial_score) / max(initial_score, 0.01) * 100):.1f}%)")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Generations: {generation}")
        
        # Финальная оценка на тестовом наборе
        if test_examples:
            print(f"\nTest Set Evaluation (Stage 3):")
            test_metrics = self.scorer.evaluate_prompt(
                self.best_node.prompt_text,
                test_examples,
                execute=True,
                sample=False,
                stage=3,
            )
            print(f"  Test composite score: {test_metrics.composite_score():.3f}")
            for name, value in sorted(test_metrics.metrics.items()):
                print(f"  Test {name}: {value:.3f}")
            self.best_node.metadata["test_metrics"] = test_metrics.to_dict()
        
        # Сохранение результатов
        if save_dir:
            self._save_final_results(save_dir, test_examples)
        
        print("="*80 + "\n")
        
        return self.best_node
    
    def _generate_reflection(self, prev_best: PromptNode, curr_best: PromptNode, train_examples: List[Example]) -> str:
        """Межпоколенческая рефлексия

        LLM анализирует, что улучшилось и какое направление выбрать.
        """
        # Получение текущих провалов для контекста
        failures_block = ""
        if curr_best.evaluation_examples.get("failures"):
            failures = curr_best.evaluation_examples["failures"][:3]
            failure_lines = []
            for i, ex in enumerate(failures, 1):
                failure_lines.append(
                    f"{i}. Input: {ex.input_text[:200]}\n"
                    f"   Expected: {ex.expected_output[:100]}\n"
                    f"   Got: {(ex.actual_output or 'None')[:100]}"
                )
            failures_block = "\n".join(failure_lines)
        
        reflection_prompt = (
            "You are reviewing the optimization progress of a prompt.\n\n"
            f"PREVIOUS BEST PROMPT (Score: {prev_best.selection_score():.3f}):\n"
            f"```\n{prev_best.prompt_text}\n```\n\n"
            f"CURRENT BEST PROMPT (Score: {curr_best.selection_score():.3f}):\n"
            f"```\n{curr_best.prompt_text}\n```\n\n"
        )
        if failures_block:
            reflection_prompt += (
                "KEY REMAINING FAILURES:\n"
                f"{failures_block}\n\n"
            )
        reflection_prompt += (
            "Provide a brief, actionable reflection (3-5 sentences):\n"
            "1. What specific improvements worked?\n"
            "2. What are the remaining weaknesses?\n"
            "3. What direction should the next optimization step take?"
        )
        
        try:
            reflection = self.llm.invoke(prompt=reflection_prompt)
            print(f"  Reflection: {reflection[:200]}...")
            return reflection
        except Exception as e:
            print(f"  Reflection failed: {e}")
            return ""
    
    def _select_population(self, candidates: List[PromptNode], population_size: int) -> List[PromptNode]:
        """Отбор популяции: дедупликация по тексту, затем top-K по score"""
        if len(candidates) <= population_size:
            return candidates
        
        # Дедупликация по prompt_id
        seen_pids = set()
        unique = []
        for c in sorted(candidates, key=lambda n: n.selection_score(), reverse=True):
            pid = prompt_id(c.prompt_text)
            if pid not in seen_pids:
                seen_pids.add(pid)
                unique.append(c)
        
        selected = unique[:population_size]
        
        print(f"  Selected population ({len(selected)}):")
        if selected:
            print(f"    Best: {selected[0].selection_score():.3f} (generation {selected[0].generation})")
        
        return selected
    
    def get_optimization_report(self) -> Dict:
        if not self.best_node:
            return {"error": "Optimization not run yet"}
        
        # Базовая информация
        report = {
            "optimization_info": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_time_seconds": self.end_time - self.start_time if self.end_time else None,
                "generations": len(self.optimization_log)
            },
            
            # Результаты
            "results": {
                "best_score": self.best_node.metadata.get(
                    "full_validation_score",
                    self.best_node.metrics.composite_score()
                ),
                "best_score_sampled": self.best_node.metrics.composite_score(),
                "best_score_full_validation": self.best_node.metadata.get("full_validation_score"),
                "best_accuracy_full_validation": self.best_node.metadata.get("full_validation_accuracy"),
                "best_metrics_sampled": self.best_node.metrics.to_dict(),
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
            "best_global_strategies": self.global_optimizer.get_best_strategies(top_k=TOP_BEST_NODES)
        }
        
        return report
    
    def visualize_optimization_trajectory(self) -> str:
        """Создание текстовой визуализации траектории оптимизации"""
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

    def compare_with_baseline(self, baseline_prompt: str, test_examples: List[Example]) -> Dict:
        """Сравнение оптимизированного промпта с baseline"""
        if not self.best_node:
            raise ValueError("Optimization not run yet")
        
        print("\nComparing with baseline...")
        
        # Оцениваем baseline
        baseline_metrics = self.scorer.evaluate_prompt(
            baseline_prompt,
            test_examples,
            execute=True,
            stage=3,
        )
        
        # Оцениваем оптимизированный
        optimized_metrics = self.scorer.evaluate_prompt(
            self.best_node.prompt_text,
            test_examples,
            execute=True,
            stage=3,
        )
        
        all_keys = sorted(set(list(baseline_metrics.metrics.keys()) + list(optimized_metrics.metrics.keys())))
        
        comparison = {
            "baseline": {
                "composite_score": baseline_metrics.composite_score(),
                **{k: baseline_metrics.metrics.get(k, 0.0) for k in all_keys}
            },
            "optimized": {
                "composite_score": optimized_metrics.composite_score(),
                **{k: optimized_metrics.metrics.get(k, 0.0) for k in all_keys}
            },
            "improvements": {
                "composite_score": optimized_metrics.composite_score() - baseline_metrics.composite_score(),
                **{k: optimized_metrics.metrics.get(k, 0.0) - baseline_metrics.metrics.get(k, 0.0) for k in all_keys}
            }
        }
        
        print("\nComparison Results:")
        print(f"  Baseline score: {baseline_metrics.composite_score():.3f}")
        print(f"  Optimized score: {optimized_metrics.composite_score():.3f}")
        print(f"  Improvement: {comparison['improvements']['composite_score']:+.3f}")
        for k in all_keys:
            delta = comparison['improvements'][k]
            print(f"  {k}: {delta:+.3f}")
        
        return comparison
    
    def __repr__(self):
        if self.best_node:
            return f"HierarchicalOptimizer(best_score={self.best_node.metrics.composite_score():.3f}, generations={len(self.optimization_log)})"
        else:
            return "HierarchicalOptimizer(not optimized yet)"
