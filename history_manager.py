from typing import List, Dict, Optional
from collections import defaultdict
import json
import os
from datetime import datetime
import numpy as np
from data_structures import PromptNode, OptimizationSource
from config import DEFAULT_PARETO_METRICS, DEFAULT_STAGNATION_WINDOW, MIN_IMPROVEMENT, GLOBAL_HISTORY_WINDOW, TOP_BEST_NODES

class HistoryManager:
    def __init__(self):
        self.nodes: Dict[str, PromptNode] = {}                                        # Основное хранилище: id -> PromptNode
        self.nodes_by_generation: Dict[int, List[str]] = defaultdict(list)            # Индекс по поколениям
        self.nodes_by_source: Dict[OptimizationSource, List[str]] = defaultdict(list) # Индекс по источникам
        self.root_ids: List[str] = []                                                 # Корневые узлы (начальные промпты)
        self.total_evaluations = 0                                                    # Общее число оцененных узлов
        self.creation_time = datetime.now()                                           # Время создания истории
    
    def add_node(self, node: PromptNode) -> str:
        # Добавляем в основное хранилище
        self.nodes[node.id] = node
        
        # Обновляем индексы
        self.nodes_by_generation[node.generation].append(node.id)
        self.nodes_by_source[node.source].append(node.id)
        
        # Если нет родителя - это корень
        if node.parent_id is None:
            if node.id not in self.root_ids:
                self.root_ids.append(node.id)
        else:
            # Добавляем связь к родителю
            if node.parent_id in self.nodes:
                self.nodes[node.parent_id].add_child(node.id)
        
        if node.is_evaluated:
            self.total_evaluations += 1
        
        return node.id
    
    def get_node(self, node_id: str) -> Optional[PromptNode]:
        """Получение узла по ID"""
        return self.nodes.get(node_id)
    
    def update_node(self, node_id: str, node: PromptNode):
        """Обновление существующего узла. Используется после оценки промпта"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        old_node = self.nodes[node_id]
        
        # Обновляем узел
        self.nodes[node_id] = node
        
        # Если поменялись важные поля — обновляем индексы
        if (old_node.generation != node.generation or old_node.source != node.source):
            if node_id in self.nodes_by_generation[old_node.generation]:
                self.nodes_by_generation[old_node.generation].remove(node_id)
            self.nodes_by_generation[node.generation].append(node_id)
            if node_id in self.nodes_by_source[old_node.source]:
                self.nodes_by_source[old_node.source].remove(node_id)
            self.nodes_by_source[node.source].append(node_id)
        # Скорректировать total_evaluations
        if not old_node.is_evaluated and node.is_evaluated:
            self.total_evaluations += 1
    
    def get_lineage(self, node_id: str) -> List[PromptNode]:
        """Получение полной родословной узла (от корня до узла)"""
        lineage = []
        current_id = node_id
        
        while current_id is not None:
            node = self.get_node(current_id)
            if node is None:
                break
            lineage.append(node)
            current_id = node.parent_id
        
        # Разворачиваем, чтобы корень был первым
        return list(reversed(lineage))
    
    def get_nodes_by_generation(self, generation: int) -> List[PromptNode]:
        """Получение всех узлов определенного поколения"""
        node_ids = self.nodes_by_generation.get(generation, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def get_nodes_by_source(self, source: OptimizationSource) -> List[PromptNode]:
        """Получение всех узлов от определенного источника"""
        node_ids = self.nodes_by_source.get(source, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def get_evaluated_nodes(self) -> List[PromptNode]:
        """Получение всех оцененных узлов"""
        return [node for node in self.nodes.values() if node.is_evaluated]
    
    def get_best_nodes(self, top_k: int) -> List[PromptNode]:
        """Получение лучших узлов по метрике"""
        # Фильтруем оцененные узлы
        candidates = self.get_evaluated_nodes()
        
        if not candidates:
            return []
        
        # Сортируем
        candidates.sort(key=lambda n: n.metrics.composite_score(), reverse=True)
        
        return candidates[:top_k]
    
    def get_pareto_front(self) -> List[PromptNode]:
        """Определение Pareto-front"""
        nodes = self.get_evaluated_nodes()
        if len(nodes) <= 1:
            return nodes
        
        # Извлекаем значения метрик для каждого узла
        values = []
        for node in nodes:
            node_values = []
            for metric in DEFAULT_PARETO_METRICS:
                if hasattr(node.metrics, metric):
                    node_values.append(getattr(node.metrics, metric))
                else:
                    node_values.append(0.0)
            values.append(node_values)
        
        values = np.array(values)
        
        # Находим фронт
        # Узел на фронте, если нет другого узла, который его доминирует
        is_pareto = np.ones(len(nodes), dtype=bool)
        
        for i in range(len(nodes)):
            if not is_pareto[i]:
                continue
            # Проверяем, доминирует ли кто-то узел i
            for j in range(len(nodes)):
                if i == j:
                    continue
                # j доминирует i, если j >= i по всем метрикам и строго > хотя бы по одной
                if np.all(values[j] >= values[i]) and np.any(values[j] > values[i]):
                    is_pareto[i] = False
                    break
        
        front = [nodes[i] for i in range(len(nodes)) if is_pareto[i]]
        
        # Помечаем узлы на фронте
        for node in front:
            node.is_front = True
        
        return front
    
    def analyze_successful_operations(self) -> Dict[str, int]:
        """Анализ типов операций, которые привели к улучшению"""
        operation_counts = defaultdict(int)
        
        for node in self.get_evaluated_nodes():
            if node.parent_id is None:
                continue
            
            parent = self.get_node(node.parent_id)
            if parent is None or not parent.is_evaluated:
                continue
            
            # Проверяем, есть ли улучшение
            improvement = (node.metrics.composite_score() - parent.metrics.composite_score())
            
            if improvement >= MIN_IMPROVEMENT:
                # Считаем операции, приведшие к улучшению
                for op in node.operations:
                    operation_counts[op.operation_type.value] += 1
        
        return dict(operation_counts)
    
    def get_stagnation_info(self) -> Dict[str, any]:
        """Определение застоя в оптимизации"""
        current_gen = max(self.nodes_by_generation.keys()) if self.nodes_by_generation else 0
        
        if current_gen < DEFAULT_STAGNATION_WINDOW:
            return {"is_stagnant": False, "best_score": 0.0}
        
        # Берем лучшие узлы из последних window поколений
        recent_best_scores = []
        for gen in range(max(0, current_gen - DEFAULT_STAGNATION_WINDOW + 1), current_gen + 1):
            gen_nodes = self.get_nodes_by_generation(gen)
            evaluated = [n for n in gen_nodes if n.is_evaluated]
            if evaluated:
                best_score = max(n.metrics.composite_score() for n in evaluated)
                recent_best_scores.append(best_score)
        
        if len(recent_best_scores) < 2:
            return {"is_stagnant": False, "best_score": recent_best_scores[0] if recent_best_scores else 0.0}
        
        # Проверяем, есть ли прогресс
        max_score = max(recent_best_scores)
        min_score = min(recent_best_scores)
        improvement = max_score - min_score
        
        is_stagnant = improvement < MIN_IMPROVEMENT
        
        return {
            "is_stagnant": is_stagnant,
            "best_score": max_score,
            "improvement": improvement,
            "recent_scores": recent_best_scores
        }
    
    def get_optimization_summary(self) -> Dict[str, any]:
        """Сводка оптимизации для глобального оптимизатора"""
        # Получаем лучшие узлы
        best_nodes = self.get_best_nodes(top_k=TOP_BEST_NODES)
        front = self.get_pareto_front()
        
        # Анализ успешных операций
        successful_ops = self.analyze_successful_operations()
        
        # Последние узлы
        all_nodes = list(self.nodes.values())
        all_nodes.sort(key=lambda n: n.timestamp, reverse=True)
        recent_nodes = all_nodes[:GLOBAL_HISTORY_WINDOW]
        
        # Статистика по источникам
        source_stats = {
            source.value: len(self.get_nodes_by_source(source))
            for source in OptimizationSource
        }
        
        # Проверка застоя
        stagnation = self.get_stagnation_info()
        
        # Распределение по поколениям
        generation_stats = {
            gen: len(nodes) for gen, nodes in self.nodes_by_generation.items()
        }
        
        return {
            "total_nodes": len(self.nodes),
            "total_evaluations": self.total_evaluations,
            "current_generation": max(self.nodes_by_generation.keys()) if self.nodes_by_generation else 0,
            "best_nodes": [
                {
                    "id": node.id,
                    "score": node.metrics.composite_score(),
                    "generation": node.generation,
                    "source": node.source.value,
                    "prompt_preview": node.prompt_text[:100] + "..."
                } for node in best_nodes
            ],
            "pareto_front_size": len(front),
            "successful_operations": successful_ops,
            "recent_nodes_count": len(recent_nodes),
            "source_distribution": source_stats,
            "generation_distribution": generation_stats,
            "stagnation_info": stagnation
        }
    
    def save(self, filepath: str):
        """Сохранение всей истории в файл"""
        data = {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "root_ids": self.root_ids,
            "total_evaluations": self.total_evaluations,
            "creation_time": self.creation_time.isoformat(),
            "save_time": datetime.now().isoformat()
        }
        
        # Создаем директорию, если не существует
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"History saved to {filepath} ({len(self.nodes)} nodes)")
    
    def get_statistics(self) -> Dict[str, any]:
        """Общая статистика по истории"""
        evaluated = self.get_evaluated_nodes()
        
        return {
            "total_nodes": len(self.nodes),
            "evaluated_nodes": len(evaluated),
            "root_nodes": len(self.root_ids),
            "generations": len(self.nodes_by_generation),
            "max_generation": max(self.nodes_by_generation.keys()) if self.nodes_by_generation else 0,
            "best_score": max(n.metrics.composite_score() for n in evaluated) if evaluated else 0.0,
            "avg_score": np.mean([n.metrics.composite_score() for n in evaluated]) if evaluated else 0.0,
            "total_evaluations": self.total_evaluations
        }
    
    def __len__(self) -> int:
        """Количество узлов в истории"""
        return len(self.nodes)
    
    def __repr__(self) -> str:
        """Строковое представление"""
        stats = self.get_statistics()
        return f"HistoryManager(nodes={stats['total_nodes']}, gen={stats['max_generation']}, best={stats['best_score']:.3f})"