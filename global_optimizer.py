from typing import List, Dict, Tuple, Callable
import numpy as np
from collections import Counter
import time
import statistics
from prompts.templates import Templates
from llm.llm_client import BaseLLM
from data_structures import Example, PromptNode, OptimizationSource, OperationType
from history_manager import HistoryManager
from evaluator.scorer import PromptScorer
from prompt_editor import PromptEditor
from llm.llm_response_parser import StrategyParser
from config import (TOP_BEST_NODES,
                    MAX_DISTANCE_PAIRS,
                    COMMON_WORDS_TOP_K,
                    COMMON_WORD_MIN_FREQ,
                    FAILED_PERCENTILE,
                    FAILED_OP_MIN_COUNT,
                    MIN_OPERATION_USAGE,
                    MIN_GLOBAL_SOURCE_USAGE,
                    STAGNATION_SIMILARITY_THRESHOLD,
                    DIVERSITY_DISTANCE_THRESHOLD,
                    LOW_DIVERSITY_THRESHOLD,
                    MAX_DIVERSITY_SAMPLES,
                    MIN_NODES_FOR_DIVERSITY,
                    GLOBAL_TRIGGER_INTERVAL,
                    COMMON_SUBSEQ_LENGTHS,
                    TOP_COMMON_SUBSEQ,
                    RECENT_GENERATIONS_FOR_DIVERSITY,
                    GLOBAL_OPT_AVG_PATH_LENGTH,)
    
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
        
        self._strategy_dispatch: Dict[str, Callable] = {
            "COMBINE": self.editor.apply_combine_strategy,
            "RESTRUCTURE": self.editor.apply_restructure_strategy,
            "DIVERSIFY": self.editor.apply_diversify_strategy,
            "SPECIALIZE": self.editor.apply_specialize_strategy,
            "SIMPLIFY": self.editor.apply_simplify_strategy,
            "EXPAND": self.editor.apply_expand_strategy,
        }        
        
    def optimize(self, current_generation: int, train_examples: List[Example], validation_examples: List[Example]) -> List[PromptNode]:
        print("\n" + "=" * 60)
        print(f"GLOBAL OPTIMIZATION STEP | Generation {current_generation}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Шаг 1: Анализ истории оптимизации
        print("Step 1: Analyzing optimization history...")
        history_analysis = self._analyze_history()
        
        # Шаг 2: Генерация глобальных стратегий
        print("\nStep 2: Generating global strategies...")
        strategies = self._generate_global_strategies(history_analysis)
        
        print(f"Generated {len(strategies)} strategies")
        for i, strategy in enumerate(strategies, 1):
            print(f"  {i}. {strategy['type']}: {strategy['description'][:80]}...")
        
        # Шаг 3: Применение стратегий и создание кандидатов
        print("\nStep 3: Creating global candidates...")
        candidates = self._apply_strategies(strategies, history_analysis, current_generation)
        
        print(f"Created {len(candidates)} global candidates")
        self.total_candidates_generated += len(candidates)
        
        # Шаг 4: Оценка кандидатов
        print("\nStep 4: Evaluating global candidates...")
        evaluated_candidates = self._evaluate_global_candidates(candidates, train_examples)
        
        # Шаг 5: Анализ результатов
        print("\nStep 5: Analyzing results...")
        self._analyze_global_results(evaluated_candidates, history_analysis)
        
        print(f"\nCompleted in {time.time() - start_time:.2f}s")
        
        self.total_global_steps += 1
        return evaluated_candidates
    
    def _analyze_history(self) -> Dict:
        """Анализ всей истории оптимизации. Определяет паттерны, проблемы и возможности"""
        best_nodes = self.history.get_best_nodes(TOP_BEST_NODES)        
        
        return {
            "summary": self.history.get_optimization_summary(),
            "best_nodes": best_nodes,
            "best_node": best_nodes[0] if best_nodes else None,
            "patterns": self._identify_patterns(best_nodes),
            "stagnation": self._analyze_stagnation(best_nodes),
            "diversity": self._analyze_diversity(),
            "best_elements": self._extract_best_elements(),
            "failed_directions": self._identify_failed_directions(),
            "unexplored_space": self._identify_unexplored_space()
        }
    
    def _identify_patterns(self, best_nodes: List[PromptNode]) -> Dict:
        """Определение паттернов в истории оптимизации"""
        trajectories = [
            [op.operation_type.value for n in self.history.get_lineage(node.id) for op in n.operations]
            for node in best_nodes
        ]
        return {
            "successful_operations": self.history.analyze_successful_operations(),
            "common_sequences": self._find_common_subsequences(trajectories),
            "avg_path_length": GLOBAL_OPT_AVG_PATH_LENGTH
        }
        
    def _find_common_subsequences(self, sequences: List[List[str]], min_length: int = 2) -> List[Tuple]:
        """Поиск общих подпоследовательностей в траекториях оптимизации"""
        counts = Counter(
            tuple(seq[i:i+n])
            for seq in sequences
            for n in COMMON_SUBSEQ_LENGTHS
            for i in range(len(seq) - n + 1)
        )
        return [s for s, c in counts.items() if c >= 2 and len(s) >= min_length][:TOP_COMMON_SUBSEQ]    # Топ общих подпоследовательностей  
    
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
            "best_score": best_nodes[0].metrics.composite_score() if best_nodes else 0.0
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
        
        # Извлекаем предложения из лучших промптов
        best_prompts = [node.prompt_text for node in best_nodes]
        
        return {
            "prompts": best_prompts,
            "common_phrases": common_words,
            "top_scores": [node.metrics.composite_score() for node in best_nodes]
        }

    def _identify_failed_directions(self) -> List[str]:
        """Определение неудачных направлений, которые стоит избегать"""
        # Узлы с низкими скорами
        all_evaluated = self.history.get_evaluated_nodes()
        if not all_evaluated:
            return []
        
        # Берем нижние
        threshold = np.percentile([n.metrics.composite_score() for n in all_evaluated], FAILED_PERCENTILE)
        ops = Counter(
            op.operation_type.value
            for n in all_evaluated
            if n.metrics.composite_score() <= threshold
            for op in n.operations
        )
        
        return [
            f"Avoid excessive use of {op}"
            for op, c in ops.items()
            if c >= FAILED_OP_MIN_COUNT
        ]

    def _identify_unexplored_space(self) -> List[str]:
        """Определение неисследованных областей пространства промптов"""
        ops = Counter(
            op.operation_type.value
            for n in self.history.nodes.values()
            for op in n.operations
        )
        unexplored = [
            f"Consider trying {op.value} operations"
            for op in OperationType
            if ops[op.value] < MIN_OPERATION_USAGE
        ]

        sources = Counter(n.source.value for n in self.history.nodes.values())
        if sources[OptimizationSource.GLOBAL.value] < MIN_GLOBAL_SOURCE_USAGE:
            unexplored.append("Need more global structural changes")

        return unexplored

    def _generate_global_strategies(self, history_analysis: Dict) -> List[Dict]:
        """Генерация глобальных стратегий на основе анализа истории"""
        strategy_prompt = Templates.build_strategy_prompt(history_analysis)
        try:
            if strategy_prompt in self._cache:
                response_text = self._cache[strategy_prompt]
            else:
                response_text = self.llm.invoke(prompt=strategy_prompt)
                self._cache[strategy_prompt] = response_text

            strategies = StrategyParser.parse_strategies(response_text)
            return strategies
        except Exception as e:
            print(f"Error generating strategies: {e}")
    
    def _apply_strategies(self, strategies: List[Dict], history_analysis: Dict, current_generation: int) -> List[PromptNode]:
        """Применение стратегий и создание глобальных кандидатов"""
        candidates = []
        for strategy in strategies:
            handler = self._strategy_dispatch.get(strategy["type"], self.editor.apply_generic_strategy)
            node = handler(strategy, history_analysis, current_generation)
            if node:
                candidates.append(node)
                self.applied_strategies.append(
                    {"generation": current_generation, "strategy": strategy, "candidate_id": node.id}
                )
        return candidates
    
    def _evaluate_global_candidates(self, candidates: List[PromptNode], validation_examples: List[Example]) -> List[PromptNode]:
        """Оценка глобальных кандидатов"""
        evaluated = []
        for i, candidate in enumerate(candidates, 1):
            print(f"  Evaluating global candidate {i}/{len(candidates)}...", end=" ")
            try:
                candidate = self.scorer.evaluate_node(candidate, validation_examples, execute=True)
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
            strategy_type = strategy.get("type", "Unknown")
            
            print(f"\n  Strategy: {strategy_type}")
            print(f"  Score: {score:.3f} (Δ {improvement:+.3f})")
            
            if improvement > 0:
                improvements.append({
                    "candidate": candidate,
                    "strategy": strategy,
                    "improvement": improvement
                })
                self.successful_global_changes += 1
        
        if improvements:
            best_improvement = max(improvements, key=lambda x: x["improvement"])
            print(f"\n✓ Best global improvement: {best_improvement['improvement']:.3f}")
            print(f"  From strategy: {best_improvement['strategy'].get('type', 'Unknown')}")
        else:
            print("\n✗ No improvements from global step")     
    
    def should_trigger_global_step(self, current_generation: int) -> bool:
        """Определение, нужно ли запускать глобальный шаг"""
        # Триггер 1: Регулярный интервал
        if current_generation % GLOBAL_TRIGGER_INTERVAL == 0:
            return True
        
        # Триггер 2: Обнаружен застой
        stagnation_info = self.history.get_stagnation_info()
        if stagnation_info["is_stagnant"]:
            print("Global step triggered by stagnation")
            return True
        
        # Триггер 3: Низкое разнообразие
        current_gen_nodes = self.history.get_nodes_by_generation(current_generation)
        if len(current_gen_nodes) >= MIN_NODES_FOR_DIVERSITY:
            distances = []
            for i in range(min(MAX_DIVERSITY_SAMPLES, len(current_gen_nodes))):
                for j in range(i+1, min(MAX_DIVERSITY_SAMPLES, len(current_gen_nodes))):
                    distances.append(self.scorer.calculate_edit_distance(current_gen_nodes[i].prompt_text, current_gen_nodes[j].prompt_text))
            if distances and np.mean(distances) < LOW_DIVERSITY_THRESHOLD:
                print("Global step triggered by low diversity")
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