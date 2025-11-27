from typing import List, Dict, Optional, Tuple, Callable
from collections import defaultdict, deque
import json
import os
from datetime import datetime
import numpy as np

from data_structures import (
    PromptNode, 
    OptimizationSource, 
    OptimizationConfig
)

class HistoryManager:
    """
    Центральное хранилище дерева эволюции промптов
    Поддерживает:
    - Добавление и поиск узлов
    - Анализ истории оптимизации
    - Поиск лучших промптов
    - Анализ паттернов успешных изменений
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Инициализация менеджера истории
        
        Args:
            config: Конфигурация оптимизации
        """
        self.config = config                                                          # Конфигурация оптимизации
        self.nodes: Dict[str, PromptNode] = {}                                        # Основное хранилище: id -> PromptNode
        self.nodes_by_generation: Dict[int, List[str]] = defaultdict(list)            # Индекс по поколениям
        self.nodes_by_source: Dict[OptimizationSource, List[str]] = defaultdict(list) # Индекс по источникам
        self.root_ids: List[str] = []                                                 # Корневые узлы (начальные промпты)
        self.total_evaluations = 0                                                    # Общее число оцененных узлов
        self.creation_time = datetime.now()                                           # Время создания истории
    
    # БАЗОВЫЕ ОПЕРАЦИИ С УЗЛАМИ
    
    def add_node(self, node: PromptNode) -> str:
        """
        Добавление узла в историю
        Автоматически обновляет индексы и связи
        
        Args:
            node: Узел для добавления
            
        Returns:
            ID добавленного узла
        """
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
    
    def delete_node(self, node_id: str, mode: str = "error"):
        """
        Удаление узла из истории.

        Args:
            node_id: ID удаляемого узла.
            mode: Поведение при наличии дочерних узлов:
                - "error" (default): бросить ошибку, если есть дети
                - "reparent": перевесить детей на родителя (или сделать корневыми, если родителя нет)
                - "recursive": рекурсивно удалить всех потомков (включая node_id)
        """
        if node_id not in self.nodes:
            return  # ничего делать не нужно

        node = self.nodes[node_id]

        # Обработка детей в зависимости от режима
        if node.children_ids:
            if mode == "error":
                raise RuntimeError(f"Cannot delete node {node_id}: it has children. Use mode='reparent' or mode='recursive'")
            elif mode == "reparent":
                parent_id = node.parent_id
                for child_id in list(node.children_ids):
                    child = self.nodes.get(child_id)
                    if child is None:
                        continue
                    # Перепривязать child к родителю удаляемого узла
                    child.parent_id = parent_id
                    # Убрать child из списка детей удаляемого узла (будет удалено ниже) и добавить к новому родителю (если есть)
                    if parent_id and parent_id in self.nodes:
                        parent = self.nodes[parent_id]
                        if child_id not in parent.children_ids:
                            parent.children_ids.append(child_id)
                    else:
                        # Если родителя нет, то child становится корневым
                        if child_id not in self.root_ids:
                            self.root_ids.append(child_id)
            elif mode == "recursive":
                # Рекурсивно удаляем всех детей 
                for child_id in list(node.children_ids):
                    self.delete_node(child_id, mode="recursive")
            else:
                raise ValueError(f"Unknown mode '{mode}' for delete_node")

        # Удаление из индексов
        gen_list = self.nodes_by_generation.get(node.generation)
        if gen_list and node_id in gen_list:
            try:
                gen_list.remove(node_id)
            except ValueError:
                pass

        src_list = self.nodes_by_source.get(node.source)
        if src_list and node_id in src_list:
            try:
                src_list.remove(node_id)
            except ValueError:
                pass

        # root ids
        if node_id in self.root_ids:
            try:
                self.root_ids.remove(node_id)
            except ValueError:
                pass

        # Связь с родителем
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node_id in parent.children_ids:
                try:
                    parent.children_ids.remove(node_id)
                except ValueError:
                    pass

        # Уменьшаем счётчик оценок, если нужно
        if node.is_evaluated and self.total_evaluations > 0:
            self.total_evaluations -= 1

        # Удалить сам узел
        if node_id in self.nodes:
            del self.nodes[node_id]
    
    # НАВИГАЦИЯ ПО ДЕРЕВУ
    
    def get_lineage(self, node_id: str) -> List[PromptNode]:
        """
        Получение полной родословной узла (от корня до узла)
        Полезно для понимания эволюции промпта
        
        Args:
            node_id: ID узла
            
        Returns:
            Список узлов от корня до заданного узла
        """
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
    
    def get_children(self, node_id: str) -> List[PromptNode]:
        """Получение всех дочерних узлов"""
        node = self.get_node(node_id)
        if node is None:
            return []
        
        return [self.nodes[child_id] for child_id in node.children_ids 
                if child_id in self.nodes]
    
    def get_descendants(self, node_id: str) -> List[PromptNode]:
        """Получение всех потомков узла (рекурсивно). Используется для анализа всей ветки оптимизации"""
        descendants = []
        queue = deque([node_id])
        
        while queue:
            current_id = queue.popleft()
            node = self.get_node(current_id)
            if node is None:
                continue
            
            for child_id in node.children_ids:
                if child_id in self.nodes:
                    descendants.append(self.nodes[child_id])
                    queue.append(child_id)
        
        return descendants
    
    def get_siblings(self, node_id: str) -> List[PromptNode]:
        """Получение узлов-сиблингов (с тем же родителем). Полезно для сравнения альтернативных редакций"""
        node = self.get_node(node_id)
        if node is None or node.parent_id is None:
            return []
        
        parent = self.get_node(node.parent_id)
        if parent is None:
            return []
        
        return [self.nodes[child_id] for child_id in parent.children_ids 
                if child_id != node_id and child_id in self.nodes]
    
    # ПОИСК И ФИЛЬТРАЦИЯ
    
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
    
    def filter_nodes(self, predicate: Callable[[PromptNode], bool]) -> List[PromptNode]:
        """
        Фильтрация узлов по произвольному предикату
        
        Args:
            predicate: Функция, возвращающая True для нужных узлов
            
        Returns:
            Список отфильтрованных узлов
        """
        return [node for node in self.nodes.values() if predicate(node)]
    
    # АНАЛИЗ И РАНЖИРОВАНИЕ
    
    #TODO: не только composite
    def get_best_nodes(self, top_k: int = 10, metric: str = "composite", generation: Optional[int] = None) -> List[PromptNode]:
        """
        Получение лучших узлов по метрике
        
        Args:
            top_k: Количество узлов для возврата
            metric: Метрика для сортировки ("composite", "accuracy", "safety", etc.)
            generation: Опционально - только из конкретного поколения
            
        Returns:
            Отсортированный список лучших узлов
        """
        # Фильтруем оцененные узлы
        candidates = self.get_evaluated_nodes()
        
        if generation is not None:
            candidates = [n for n in candidates if n.generation == generation]
        
        if not candidates:
            return []
        
        # Сортируем по выбранной метрике
        if metric == "composite":
            candidates.sort(key=lambda n: n.metrics.composite_score(), reverse=True)
        elif hasattr(candidates[0].metrics, metric):
            candidates.sort(key=lambda n: getattr(n.metrics, metric), reverse=True)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return candidates[:top_k]
    
    #TODO: улучшить
    def get_pareto_front(self, metrics: List[str] = None) -> List[PromptNode]:
        """
        Нахождение фронта Паретто по нескольким метрикам
        Узлы на фронте - это те, которые не доминируются другими
        
        Args:
            metrics: Список метрик для рассмотрения (по умолчанию: accuracy, safety, robustness)
            
        Returns:
            Список узлов на фронте Паретто
        """
        if metrics is None:
            metrics = ["accuracy", "safety", "robustness"]
        
        nodes = self.get_evaluated_nodes()
        if not nodes:
            return []
        
        # Извлекаем значения метрик для каждого узла
        values = []
        for node in nodes:
            node_values = []
            for metric in metrics:
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
    
    # АНАЛИЗ ПАТТЕРНОВ ОПТИМИЗАЦИИ
    
    def analyze_successful_operations(self, min_improvement: float = 0.05) -> Dict[str, int]:
        """
        Анализ типов операций, которые привели к улучшению
        Помогает глобальному оптимизатору понять, что работает
        
        Args:
            min_improvement: Минимальное улучшение для считания успешным
            
        Returns:
            Словарь {тип_операции: количество_успехов}
        """
        operation_counts = defaultdict(int)
        
        for node in self.get_evaluated_nodes():
            if node.parent_id is None:
                continue
            
            parent = self.get_node(node.parent_id)
            if parent is None or not parent.is_evaluated:
                continue
            
            # Проверяем, есть ли улучшение
            improvement = (node.metrics.composite_score() - 
                          parent.metrics.composite_score())
            
            if improvement >= min_improvement:
                # Считаем операции, приведшие к улучшению
                for op in node.operations:
                    operation_counts[op.operation_type.value] += 1
        
        return dict(operation_counts)
    
    def get_optimization_trajectory(self, node_id: str) -> List[Tuple[int, float]]:
        """
        Получение траектории оптимизации (поколение, метрика) от корня до узла
        Полезно для визуализации прогресса
        
        Args:
            node_id: ID конечного узла
            
        Returns:
            Список кортежей (поколение, композитная_метрика)
        """
        lineage = self.get_lineage(node_id)
        trajectory = []
        
        for node in lineage:
            if node.is_evaluated:
                trajectory.append((node.generation, node.metrics.composite_score()))
        
        return trajectory
    
    def get_stagnation_info(self, window: int = 5) -> Dict[str, any]:
        """
        Определение застоя в оптимизации
        Используется для early stopping
        
        Args:
            window: Окно последних поколений для анализа
            
        Returns:
            Словарь с информацией о застое
        """
        current_gen = max(self.nodes_by_generation.keys()) if self.nodes_by_generation else 0
        
        if current_gen < window:
            return {"is_stagnant": False, "best_score": 0.0}
        
        # Берем лучшие узлы из последних window поколений
        recent_best_scores = []
        for gen in range(max(0, current_gen - window + 1), current_gen + 1):
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
        
        is_stagnant = improvement < self.config.min_improvement
        
        return {
            "is_stagnant": is_stagnant,
            "best_score": max_score,
            "improvement": improvement,
            "recent_scores": recent_best_scores
        }
    
    # ГЛОБАЛЬНАЯ АНАЛИТИКА ДЛЯ GLOBAL OPTIMIZER
    
    def get_optimization_summary(self, recent_window: int = 20) -> Dict[str, any]:
        """
        Сводка оптимизации для глобального оптимизатора
        Содержит всю важную информацию для принятия решения о структурных изменениях
        
        Args:
            recent_window: Размер окна последних узлов для анализа
            
        Returns:
            Словарь с полной аналитикой
        """
        # Получаем лучшие узлы
        best_nodes = self.get_best_nodes(top_k=5)
        front = self.get_pareto_front()
        
        # Анализ успешных операций
        successful_ops = self.analyze_successful_operations()
        
        # Последние узлы
        all_nodes = list(self.nodes.values())
        all_nodes.sort(key=lambda n: n.timestamp, reverse=True)
        recent_nodes = all_nodes[:recent_window]
        
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
    
    # СЕРИАЛИЗАЦИЯ И СОХРАНЕНИЕ
    
    def save(self, filepath: str):
        """
        Сохранение всей истории в файл
        
        Args:
            filepath: Путь к файлу для сохранения
        """
        data = {
            "config": self.config.to_dict(),
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
    
    @classmethod
    def load(cls, filepath: str) -> 'HistoryManager':
        """
        Загрузка истории из файла
        
        Args:
            filepath: Путь к файлу
            
        Returns:
            Загруженный HistoryManager
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Восстанавливаем конфигурацию
        config = OptimizationConfig.from_dict(data["config"])
        manager = cls(config)
        
        # Восстанавливаем узлы
        for node_id, node_data in data["nodes"].items():
            node = PromptNode.from_dict(node_data)
            manager.nodes[node_id] = node
            
            # Восстанавливаем индексы
            manager.nodes_by_generation[node.generation].append(node_id)
            manager.nodes_by_source[node.source].append(node_id)
        
        manager.root_ids = data["root_ids"]
        manager.total_evaluations = data["total_evaluations"]
        manager.creation_time = datetime.fromisoformat(data["creation_time"])
        
        print(f"History loaded from {filepath} ({len(manager.nodes)} nodes)")
        return manager
    
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    
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