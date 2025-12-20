from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict, Counter
import time

from llm.llm_client import create_llm

from data_structures import (
    Example,
    PromptNode,
    OptimizationConfig,
    OptimizationSource,
    OperationType,
    EditOperation
)
from history_manager import HistoryManager
from evaluator.scorer import PromptScorer
from prompt_editor import PromptEditor

class GlobalOptimizer:
    """
    Глобальный оптимизатор - выполняет крупные структурные изменения
    на основе анализа всей истории оптимизации
    
    Отличия от локального:
    - Анализирует всю историю, а не только текущий узел
    - Делает структурные изменения (реорганизация, комбинирование)
    - Запускается периодически, а не на каждой итерации
    - Может создавать совершенно новые направления
    """
    
    def __init__(self, config: OptimizationConfig, history_manager: HistoryManager, scorer: PromptScorer, prompt_editor: PromptEditor, api_config: Optional[Dict[str, str]] = None):
        """
        Args:
            config: Конфигурация оптимизации
            history_manager: Менеджер истории
            scorer: Система оценки промптов
            prompt_editor: Редактор промптов
            api_config: Конфигурация API
        """
        self.config = config
        self.history = history_manager
        self.scorer = scorer
        self.editor = prompt_editor
        self.api_config = api_config or {}
        self.provider = self.api_config.get("provider")
        
        # Инициализация LLM клиента
        self.llm = create_llm(self.config, self.api_config)
        
        # Статистика глобальной оптимизации
        self.total_global_steps = 0
        self.total_candidates_generated = 0
        self.successful_global_changes = 0
        
        # История глобальных стратегий
        self.applied_strategies: List[Dict] = []
        
    # ОСНОВНОЙ МЕТОД ГЛОБАЛЬНОЙ ОПТИМИЗАЦИИ
    
    def optimize(self, current_generation: int, train_examples: List[Example], validation_examples: List[Example]) -> List[PromptNode]:
        """
        Запуск глобального шага оптимизации
        
        Args:
            current_generation: Текущее поколение
            train_examples: Обучающие примеры
            validation_examples: Валидационные примеры
            
        Returns:
            Список новых глобальных кандидатов
        """
        print(f"\n{'='*60}")
        print(f"GLOBAL OPTIMIZATION STEP")
        print(f"Generation: {current_generation}")
        print(f"{'='*60}\n")
        
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
        evaluated_candidates = self._evaluate_global_candidates(candidates, validation_examples)
        
        # Шаг 5: Анализ результатов
        print("\nStep 5: Analyzing results...")
        self._analyze_global_results(evaluated_candidates, history_analysis)
        
        elapsed_time = time.time() - start_time
        print(f"\nGlobal optimization completed in {elapsed_time:.2f}s")
        print(f"{'='*60}\n")
        
        self.total_global_steps += 1
        return evaluated_candidates
    
    # АНАЛИЗ ИСТОРИИ ОПТИМИЗАЦИИ
    
    def _analyze_history(self) -> Dict:
        """
        Глубокий анализ всей истории оптимизации
        Определяет паттерны, проблемы и возможности
        
        Returns:
            Словарь с детальным анализом
        """
        # Получаем сводку из history manager
        summary = self.history.get_optimization_summary(recent_window=self.config.global_history_window)
        
        # Дополнительный анализ
        analysis = {
            "summary": summary,
            "patterns": self._identify_patterns(),
            "stagnation": self._analyze_stagnation(),
            "diversity": self._analyze_diversity(),
            "best_elements": self._extract_best_elements(),
            "failed_directions": self._identify_failed_directions(),
            "unexplored_space": self._identify_unexplored_space()
        }
        
        return analysis
    
    def _identify_patterns(self) -> Dict:
        """Определение паттернов в истории оптимизации"""
        # Анализ успешных операций
        successful_ops = self.history.analyze_successful_operations(min_improvement=0.05)
        
        # Анализ траекторий лучших узлов
        best_nodes = self.history.get_best_nodes(top_k=5)
        
        trajectories = []
        for node in best_nodes:
            lineage = self.history.get_lineage(node.id)
            ops_sequence = []
            for n in lineage:
                for op in n.operations:
                    ops_sequence.append(op.operation_type.value)
            trajectories.append(ops_sequence)
        
        # Поиск общих последовательностей операций
        common_sequences = self._find_common_subsequences(trajectories)
        
        return {
            "successful_operations": successful_ops,
            "common_sequences": common_sequences,
            "best_nodes_count": len(best_nodes),
            "avg_path_length": np.mean([len(t) for t in trajectories]) if trajectories else 0
        }
    
    def _analyze_stagnation(self) -> Dict:
        """Анализ застоя в оптимизации"""
        stagnation_info = self.history.get_stagnation_info(window=5)
        
        # Дополнительный анализ - где застряли
        best_nodes = self.history.get_best_nodes(top_k=10)
        
        if best_nodes:
            # Анализ схожести лучших промптов
            similarities = []
            for i in range(len(best_nodes)):
                for j in range(i+1, len(best_nodes)):
                    dist = self.editor.calculate_edit_distance(best_nodes[i].prompt_text, best_nodes[j].prompt_text)
                    similarities.append(1.0 - dist)
            avg_similarity = np.mean(similarities) if similarities else 0.0
        else:
            avg_similarity = 0.0
        
        return {
            "is_stagnant": stagnation_info["is_stagnant"],
            "best_score": stagnation_info.get("best_score", 0.0),
            "avg_similarity": avg_similarity,
            "needs_exploration": avg_similarity > 0.7  # Слишком похожи
        }
    
    def _analyze_diversity(self) -> Dict:
        """Анализ разнообразия в популяции"""
        # Берем последние узлы из разных генераций
        current_gen = max(self.history.nodes_by_generation.keys()) if self.history.nodes_by_generation else 0
        
        recent_nodes = []
        for gen in range(max(0, current_gen - 3), current_gen + 1):
            recent_nodes.extend(self.history.get_nodes_by_generation(gen))
        
        if len(recent_nodes) < 2:
            return {"diversity_score": 0.0, "needs_diversification": True}
        
        # Вычисляем попарные расстояния
        distances = []
        for i in range(min(20, len(recent_nodes))):  # Ограничиваем для производительности
            for j in range(i+1, min(20, len(recent_nodes))):
                dist = self.editor.calculate_edit_distance(recent_nodes[i].prompt_text, recent_nodes[j].prompt_text)
                distances.append(dist)
        avg_distance = np.mean(distances) if distances else 0.0
        
        return {
            "diversity_score": avg_distance,
            "needs_diversification": avg_distance < 0.3  # Слишком похожи
        }
    
    def _extract_best_elements(self) -> Dict:
        """Извлечение лучших элементов из успешных промптов"""
        best_nodes = self.history.get_best_nodes(top_k=5)
        
        if not best_nodes:
            return {"prompts": [], "common_phrases": []}
        
        # Извлекаем общие фразы/паттерны
        all_words = []
        for node in best_nodes:
            words = node.prompt_text.lower().split()
            all_words.extend(words)
        
        # Частотный анализ
        word_freq = Counter(all_words)
        common_words = [word for word, count in word_freq.most_common(20) if count >= 3]
        
        # Извлекаем предложения из лучших промптов
        best_prompts = [node.prompt_text for node in best_nodes]
        
        return {
            "prompts": best_prompts,
            "common_phrases": common_words,
            "top_scores": [node.metrics.composite_score() for node in best_nodes]
        }
    
    def _identify_failed_directions(self) -> List[str]:
        """Определение неудачных направлений, которые стоит избегать"""
        failed_directions = []
        
        # Узлы с низкими скорами
        all_evaluated = self.history.get_evaluated_nodes()
        if not all_evaluated:
            return []
        
        # Берем нижние 20%
        threshold = np.percentile([n.metrics.composite_score() for n in all_evaluated], 20)
        
        poor_nodes = [n for n in all_evaluated if n.metrics.composite_score() <= threshold]
        
        # Анализируем их операции
        failed_ops = defaultdict(int)
        for node in poor_nodes:
            for op in node.operations:
                failed_ops[op.operation_type.value] += 1
        
        # Направления, которые часто приводят к плохим результатам
        for op_type, count in failed_ops.items():
            if count >= 3:
                failed_directions.append(f"Avoid excessive use of {op_type}")
        
        return failed_directions
    
    def _identify_unexplored_space(self) -> List[str]:
        """Определение неисследованных областей пространства промптов"""
        unexplored = []
        
        # Проверяем, какие типы операций редко использовались
        all_nodes = list(self.history.nodes.values())
        all_ops = []
        for node in all_nodes:
            all_ops.extend([op.operation_type.value for op in node.operations])
        
        op_counts = Counter(all_ops)
        
        # Все возможные типы операций
        all_op_types = [op.value for op in OperationType]
        
        for op_type in all_op_types:
            if op_counts[op_type] < 2:
                unexplored.append(f"Consider trying {op_type} operations")
        
        # Проверяем разнообразие источников
        source_counts = defaultdict(int)
        for node in all_nodes:
            source_counts[node.source.value] += 1
        
        if source_counts[OptimizationSource.GLOBAL.value] < 3:
            unexplored.append("Need more global structural changes")
        
        return unexplored
    
    # ГЕНЕРАЦИЯ ГЛОБАЛЬНЫХ СТРАТЕГИЙ
    
    def _generate_global_strategies(self, history_analysis: Dict) -> List[Dict]:
        """
        Генерация глобальных стратегий на основе анализа истории
        Использует LLM для создания крупных структурных изменений
        
        Args:
            history_analysis: Результат анализа истории
            
        Returns:
            Список стратегий (каждая - словарь с описанием)
        """
        if not getattr(self, 'llm', None) or self.llm.provider is None:
            # Fallback на эвристические стратегии
            return self._generate_heuristic_strategies(history_analysis)
        
        # Формируем промпт для LLM
        strategy_prompt = self._build_strategy_prompt(history_analysis)
        
        try:
            response_text = self.llm.invoke(prompt=strategy_prompt, temperature=0.8)
            
            # Парсим стратегии из ответа
            strategies = self._parse_strategies(response_text)
            
            # Если парсинг не сработал, используем эвристики
            if not strategies:
                strategies = self._generate_heuristic_strategies(history_analysis)
            
            return strategies
            
        except Exception as e:
            print(f"Error generating strategies: {e}")
            return self._generate_heuristic_strategies(history_analysis)
    
    def _build_strategy_prompt(self, history_analysis: Dict) -> str:
        """Построение промпта для генерации стратегий. Загружает шаблон из prompts/strategy.txt"""
        from prompts.loader import load_template
        
        summary = history_analysis["summary"]
        patterns = history_analysis["patterns"]
        stagnation = history_analysis["stagnation"]
        diversity = history_analysis["diversity"]
        
        # Подготавливаем блоки для заполнения шаблона
        best_prompts_block = ""
        for i, node_info in enumerate(summary['best_nodes'][:3], 1):
            best_prompts_block += f"{i}. Score {node_info['score']:.3f} (Gen {node_info['generation']}):\n"
            best_prompts_block += f"   {node_info['prompt_preview']}\n\n"
        
        failed_directions_block = ""
        if history_analysis.get("failed_directions"):
            for direction in history_analysis["failed_directions"]:
                failed_directions_block += f"- {direction}\n"
        else:
            failed_directions_block = "None identified"
        
        unexplored_space_block = ""
        if history_analysis.get("unexplored_space"):
            for area in history_analysis["unexplored_space"]:
                unexplored_space_block += f"- {area}\n"
        else:
            unexplored_space_block = "None identified"
        
        # Загружаем шаблон и заполняем его
        template = load_template("strategy")
        prompt = template.format(
            total_nodes=summary['total_nodes'],
            current_generation=summary['current_generation'],
            best_score=summary['best_nodes'][0]['score'] if summary['best_nodes'] else 0.0,
            pareto_front_size=summary['pareto_front_size'],
            successful_operations=patterns['successful_operations'],
            avg_path_length=patterns['avg_path_length'],
            is_stagnant=stagnation['is_stagnant'],
            stagnation_best_score=stagnation['best_score'],
            avg_similarity=stagnation['avg_similarity'],
            needs_exploration=stagnation['needs_exploration'],
            diversity_score=diversity['diversity_score'],
            needs_diversification=diversity['needs_diversification'],
            best_prompts_block=best_prompts_block,
            failed_directions_block=failed_directions_block,
            unexplored_space_block=unexplored_space_block
        )
        
        return prompt
    
    def _parse_strategies(self, response_text: str) -> List[Dict]:
        """Парсинг стратегий из ответа LLM"""
        strategies = []
        
        import re
        strategy_blocks = re.split(r'STRATEGY\s+\d+:', response_text)
        
        for block in strategy_blocks[1:]:  # Пропускаем первый пустой
            try:
                # Извлекаем компоненты стратегии
                type_match = re.search(r'TYPE:\s*(\w+)', block)
                desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?=RATIONALE:|SPECIFIC_ACTION:|$)', block, re.DOTALL)
                rationale_match = re.search(r'RATIONALE:\s*(.+?)(?=SPECIFIC_ACTION:|$)', block, re.DOTALL)
                action_match = re.search(r'SPECIFIC_ACTION:\s*(.+?)(?=STRATEGY|$)', block, re.DOTALL)
                
                if type_match and desc_match:
                    strategy = {
                        "type": type_match.group(1).strip().upper(),
                        "description": desc_match.group(1).strip(),
                        "rationale": rationale_match.group(1).strip() if rationale_match else "",
                        "action": action_match.group(1).strip() if action_match else ""
                    }
                    strategies.append(strategy)
            except Exception as e:
                print(f"Error parsing strategy block: {e}")
                continue
        
        return strategies
    
    def _generate_heuristic_strategies(self, history_analysis: Dict) -> List[Dict]:
        """Генерация стратегий на основе эвристик (fallback). Используется если LLM недоступен"""
        strategies = []
        
        stagnation = history_analysis["stagnation"]
        diversity = history_analysis["diversity"]
        patterns = history_analysis["patterns"]
        
        # Стратегия 1: Комбинирование, если есть хорошие кандидаты
        if len(history_analysis["best_elements"]["prompts"]) >= 2:
            strategies.append({
                "type": "COMBINE",
                "description": "Combine best elements from top-performing prompts",
                "rationale": f"Top prompts have scores {history_analysis['best_elements']['top_scores'][:3]}",
                "action": "Merge successful components from multiple high-scoring prompts"
            })
        
        # Стратегия 2: Диверсификация, если застой или низкое разнообразие
        if stagnation["is_stagnant"] or diversity["needs_diversification"]:
            strategies.append({
                "type": "DIVERSIFY",
                "description": "Explore completely different prompt structure",
                "rationale": f"Stagnation detected (similarity: {stagnation['avg_similarity']:.2f})",
                "action": "Generate prompts with significantly different structure and approach"
            })
        
        # Стратегия 3: Реструктуризация
        strategies.append({
            "type": "RESTRUCTURE",
            "description": "Reorganize prompt into clear sections",
            "rationale": "Structural organization may improve clarity",
            "action": "Add sections: Instructions, Examples, Constraints, Output Format"
        })
        
        # Стратегия 4: Специализация на основе успешных операций
        if patterns["successful_operations"]:
            top_op = max(patterns["successful_operations"].items(), key=lambda x: x[1])[0]
            strategies.append({
                "type": "SPECIALIZE",
                "description": f"Focus on {top_op} operations",
                "rationale": f"{top_op} has been most successful",
                "action": f"Apply multiple {top_op} operations in sequence"
            })
        
        return strategies[:self.config.global_candidates]
    
    # ПРИМЕНЕНИЕ СТРАТЕГИЙ
    
    def _apply_strategies(self, strategies: List[Dict], history_analysis: Dict, current_generation: int) -> List[PromptNode]:
        """
        Применение стратегий и создание глобальных кандидатов
        
        Args:
            strategies: Список стратегий
            history_analysis: Анализ истории
            current_generation: Текущее поколение
            
        Returns:
            Список новых PromptNode
        """
        candidates = []
        
        for i, strategy in enumerate(strategies, 1):
            print(f"\n  Applying strategy {i}: {strategy['type']}")
            
            try:
                if strategy["type"] == "COMBINE":
                    candidate = self._apply_combine_strategy(strategy, history_analysis, current_generation)
                elif strategy["type"] == "RESTRUCTURE":
                    candidate = self._apply_restructure_strategy(strategy, history_analysis, current_generation)
                elif strategy["type"] == "DIVERSIFY":
                    candidate = self._apply_diversify_strategy(strategy, history_analysis, current_generation)
                elif strategy["type"] == "SPECIALIZE":
                    candidate = self._apply_specialize_strategy(strategy, history_analysis, current_generation)
                elif strategy["type"] == "SIMPLIFY":
                    candidate = self._apply_simplify_strategy(strategy, history_analysis, current_generation)
                elif strategy["type"] == "EXPAND":
                    candidate = self._apply_expand_strategy(strategy, history_analysis, current_generation)
                else:
                    # Неизвестный тип - используем общий подход
                    candidate = self._apply_generic_strategy(strategy, history_analysis, current_generation)
                
                if candidate:
                    candidates.append(candidate)
                    
                    # Сохраняем информацию о стратегии
                    self.applied_strategies.append({
                        "generation": current_generation,
                        "strategy": strategy,
                        "candidate_id": candidate.id
                    })
                
            except Exception as e:
                print(f"    Error applying strategy: {e}")
                continue
        
        return candidates
    
    def _apply_combine_strategy(self, strategy: Dict, analysis: Dict, generation: int) -> Optional[PromptNode]:
        """Комбинирование лучших промптов"""
        best_prompts = analysis["best_elements"]["prompts"]
        
        if len(best_prompts) < 2:
            return None
        
        # Используем editor для комбинирования
        combined_node = self.editor.combine_prompts(
            best_prompts[:3],  # Берем топ-3
            combination_strategy="best_elements"
        )
        
        # Обновляем метаданные
        combined_node.generation = generation
        combined_node.source = OptimizationSource.GLOBAL
        combined_node.metadata["global_strategy"] = strategy
        
        return combined_node
    
    def _apply_restructure_strategy(self, strategy: Dict, analysis: Dict, generation: int) -> Optional[PromptNode]:
        """Реструктуризация промпта"""
        # Берем лучший промпт как основу
        best_node = self.history.get_best_nodes(top_k=1)[0]
        
        # Применяем операцию реструктуризации
        restructured = self.editor.apply_specific_operation(
            best_node.prompt_text,
            OperationType.RESTRUCTURE,
            strategy["action"],
            parent_node=best_node
        )
        
        restructured.generation = generation
        restructured.metadata["global_strategy"] = strategy
        
        return restructured
    
    def _apply_diversify_strategy(self, strategy: Dict, analysis: Dict, generation: int) -> Optional[PromptNode]:
        """Создание разнообразного промпта"""
        if not getattr(self, 'llm', None) or self.llm.provider is None:
            return None
        
        from prompts.loader import load_template
        
        # Получаем лучшие промпты для референса
        best_prompts = analysis["best_elements"]["prompts"]
        
        # Подготавливаем блок с лучшими промптами
        best_prompts_block = ""
        for i, prompt in enumerate(best_prompts[:2], 1):
            best_prompts_block += f"\nApproach {i}:\n{prompt[:300]}...\n"
        
        # Загружаем шаблон
        template = load_template("diversify")
        diversify_prompt = template.format(best_prompts_block=best_prompts_block, specific_guidance=strategy['action'])
        
        try:
            new_prompt_text = self.llm.invoke(prompt=diversify_prompt, temperature=0.9)
            
            operation = EditOperation(
                operation_type=OperationType.RESTRUCTURE,
                description=f"DIVERSIFY: {strategy['description']}",
            )
            
            node = PromptNode(
                prompt_text=new_prompt_text,
                generation=generation,
                source=OptimizationSource.GLOBAL,
                operations=[operation],
                metadata={"global_strategy": strategy}
            )
            
            return node
            
        except Exception as e:
            print(f"    Error in diversify: {e}")
            return None
    
    def _apply_specialize_strategy(self, strategy: Dict, analysis: Dict, generation: int) -> Optional[PromptNode]:
        """Специализация промпта"""
        best_node = self.history.get_best_nodes(top_k=1)[0]
        
        specialized = self.editor.apply_specific_operation(
            best_node.prompt_text,
            OperationType.ADD_CONSTRAINT,
            strategy["action"],
            parent_node=best_node
        )
        
        specialized.generation = generation
        specialized.metadata["global_strategy"] = strategy
        
        return specialized
    
    def _apply_simplify_strategy(self, strategy: Dict, analysis: Dict, generation: int) -> Optional[PromptNode]:
        """Упрощение промпта"""
        best_node = self.history.get_best_nodes(top_k=1)[0]
        
        # Используем LLM для упрощения
        if not getattr(self, 'llm', None) or self.llm.provider is None:
            return None
        
        from prompts.loader import load_template
        
        # Загружаем шаблон
        template = load_template("simplify")
        simplify_prompt = template.format(current_prompt=best_node.prompt_text, guidance=strategy['action'])
        
        try:
            simplified_text = self.llm.invoke(prompt=simplify_prompt, temperature=0.5)
            
            operation = EditOperation(
                operation_type=OperationType.REPHRASE,
                description=f"SIMPLIFY: {strategy['description']}",
            )
            
            node = PromptNode(
                prompt_text=simplified_text,
                generation=generation,
                source=OptimizationSource.GLOBAL,
                operations=[operation],
                metadata={"global_strategy": strategy}
            )
            
            return node
            
        except Exception as e:
            print(f"    Error in simplify: {e}")
            return None
    
    def _apply_expand_strategy(self, strategy: Dict, analysis: Dict, generation: int) -> Optional[PromptNode]:
        """Расширение промпта"""
        best_node = self.history.get_best_nodes(top_k=1)[0]
        
        expanded = self.editor.apply_specific_operation(
            best_node.prompt_text,
            OperationType.ADD_INSTRUCTION,
            strategy["action"],
            parent_node=best_node
        )
        
        expanded.generation = generation
        expanded.metadata["global_strategy"] = strategy
        
        return expanded
    
    def _apply_generic_strategy(self, strategy: Dict, analysis: Dict, generation: int) -> Optional[PromptNode]:
        """Общий подход для неизвестных типов стратегий"""
        best_node = self.history.get_best_nodes(top_k=1)[0]
        
        generic = self.editor.apply_specific_operation(
            best_node.prompt_text,
            OperationType.MODIFY_INSTRUCTION,
            strategy["action"],
            parent_node=best_node
        )
        
        generic.generation = generation
        generic.metadata["global_strategy"] = strategy
        
        return generic
    
    # ОЦЕНКА ГЛОБАЛЬНЫХ КАНДИДАТОВ
    
    def _evaluate_global_candidates(self, candidates: List[PromptNode], validation_examples: List[Example]) -> List[PromptNode]:
        """
        Оценка глобальных кандидатов
        
        Args:
            candidates: Список кандидатов
            validation_examples: Валидационные примеры
            
        Returns:
            Оцененные кандидаты
        """
        evaluated = []
        
        for i, candidate in enumerate(candidates, 1):
            print(f"  Evaluating global candidate {i}/{len(candidates)}...", end=" ")
            
            try:
                candidate = self.scorer.evaluate_node(
                    candidate,
                    validation_examples,
                    execute=True
                )
                
                score = candidate.metrics.composite_score()
                print(f"Score: {score:.3f}")
                
                # Добавляем в историю
                self.history.add_node(candidate)
                
                evaluated.append(candidate)
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        return evaluated
    
    # АНАЛИЗ РЕЗУЛЬТАТОВ ГЛОБАЛЬНОГО ШАГА
    
    def _analyze_global_results(self, evaluated_candidates: List[PromptNode], history_analysis: Dict):
        """
        Анализ результатов глобального шага
        Определяет, какие стратегии сработали
        
        Args:
            evaluated_candidates: Оцененные кандидаты
            history_analysis: Анализ истории до глобального шага
        """
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
    
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    
    def _find_common_subsequences(self, sequences: List[List[str]], min_length: int = 2) -> List[Tuple]:
        """
        Поиск общих подпоследовательностей в траекториях оптимизации
        
        Args:
            sequences: Список последовательностей операций
            min_length: Минимальная длина подпоследовательности
            
        Returns:
            Список общих подпоследовательностей с их частотой
        """
        if not sequences or len(sequences) < 2:
            return []
        
        # Простой подход: ищем биграммы и триграммы
        subsequence_counts = defaultdict(int)
        
        for seq in sequences:
            # Биграммы
            for i in range(len(seq) - 1):
                bigram = tuple(seq[i:i+2])
                subsequence_counts[bigram] += 1
            
            # Триграммы
            for i in range(len(seq) - 2):
                trigram = tuple(seq[i:i+3])
                subsequence_counts[trigram] += 1
        
        # Фильтруем - оставляем только те, что встречаются в нескольких последовательностях
        common = [(subseq, count) for subseq, count in subsequence_counts.items() 
                  if count >= 2 and len(subseq) >= min_length]
        
        # Сортируем по частоте
        common.sort(key=lambda x: x[1], reverse=True)
        return common[:10]  # Топ-10
    
    def should_trigger_global_step(self, current_generation: int) -> bool:
        """
        Определение, нужно ли запускать глобальный шаг
        
        Args:
            current_generation: Текущее поколение
            
        Returns:
            True если нужно запустить глобальный шаг
        """
        # Триггер 1: Регулярный интервал
        if current_generation % self.config.global_trigger_interval == 0:
            return True
        
        # Триггер 2: Обнаружен застой
        stagnation_info = self.history.get_stagnation_info()
        if stagnation_info["is_stagnant"]:
            print("Global step triggered by stagnation")
            return True
        
        # Триггер 3: Низкое разнообразие
        current_gen_nodes = self.history.get_nodes_by_generation(current_generation)
        if len(current_gen_nodes) >= 3:
            distances = []
            for i in range(min(5, len(current_gen_nodes))):
                for j in range(i+1, min(5, len(current_gen_nodes))):
                    dist = self.editor.calculate_edit_distance(
                        current_gen_nodes[i].prompt_text,
                        current_gen_nodes[j].prompt_text
                    )
                    distances.append(dist)
            
            if distances and np.mean(distances) < 0.2:
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
            "strategies_applied": len(self.applied_strategies),
            "strategy_distribution": self._get_strategy_distribution()
        }
    
    def _get_strategy_distribution(self) -> Dict[str, int]:
        """Распределение использованных стратегий"""
        distribution = defaultdict(int)
        for item in self.applied_strategies:
            strategy_type = item["strategy"].get("type", "Unknown")
            distribution[strategy_type] += 1
        return dict(distribution)
    
    def get_best_strategies(self, top_k: int = 5) -> List[Dict]:
        """
        Получение самых успешных стратегий
        
        Args:
            top_k: Количество стратегий для возврата
            
        Returns:
            Список лучших стратегий с их результатами
        """
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
        
        # Сортируем по score
        strategy_results.sort(key=lambda x: x["score"], reverse=True)
        
        return strategy_results[:top_k]
    
    def __repr__(self):
        stats = self.get_statistics()
        return f"GlobalOptimizer(steps={stats['total_global_steps']}, success_rate={stats['success_rate']:.2f})"