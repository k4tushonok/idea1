from typing import List, Dict, Optional
import random
from collections import defaultdict
from llm_client import create_llm_client
from llm_response_parser import LLMResponseParser

from data_structures import (
    Example,
    TextGradient,
    OptimizationConfig
)

class TextGradientGenerator:
    """Генератор текстовых градиентов"""
    
    def __init__(self, config: OptimizationConfig, api_config: Optional[Dict[str, str]] = None):
        """
        Args:
            config: Конфигурация оптимизации
            api_config: Конфигурация API
        """
        self.config = config
        self.api_config = api_config or {}
        
        # Инициализация LLM клиента
        self.llm = create_llm_client(self.config, self.api_config)
        
        # Статистика
        self.total_gradients_generated = 0

    # ГЕНЕРАЦИЯ ТЕКСТОВЫХ ГРАДИЕНТОВ
    
    def generate_gradient(self, current_prompt: str, failure_examples: List[Example], success_examples: List[Example] = None, context: Optional[Dict] = None) -> TextGradient:
        """
        Генерация одного текстового градиента
        Анализирует провалы и успехи, предлагает направление улучшения
        
        Args:
            current_prompt: Текущий промпт
            failure_examples: Примеры, где промпт ошибся
            success_examples: Примеры, где промпт работает (для контраста)
            context: Дополнительный контекст (история оптимизации, метаданные)
            
        Returns:
            TextGradient с анализом и рекомендациями
        """
        if self.llm.provider is None:
            raise ValueError("No LLM client configured. Cannot generate gradients.")
        
        if not failure_examples:
            raise ValueError("Need at least one failure example to generate gradient")
        
        if success_examples is None:
            success_examples = []
        
        # Формируем промпт для анализа
        analysis_prompt = self._build_analysis_prompt(
            current_prompt,
            failure_examples,
            success_examples,
            context
        )
        
        # Вызываем LLM для анализа
        try:
            analysis_text = self.llm.call(analysis_prompt)
            
            # Парсим ответ LLM и извлекаем компоненты градиента
            gradient = self._parse_gradient_response(
                analysis_text,
                failure_examples,
                success_examples
            )
            
            self.total_gradients_generated += 1
            return gradient
            
        except Exception as e:
            print(f"Error generating gradient: {e}")
            # Возвращаем пустой градиент в случае ошибки
            return TextGradient(
                failure_examples=failure_examples,
                success_examples=success_examples,
                error_analysis="Failed to generate analysis",
                suggested_direction="Unable to provide suggestions",
                priority=0.0
            )
    
    def generate_gradients_batch(self, current_prompt: str, failure_examples: List[Example], success_examples: List[Example] = None, batch_size: int = None, num_gradients: int = 3) -> List[TextGradient]:
        """
        Генерация нескольких градиентов из разных подмножеств провалов
        Позволяет захватить разные типы ошибок
        
        Args:
            current_prompt: Текущий промпт
            failure_examples: Все примеры провалов
            success_examples: Примеры успехов
            batch_size: Размер батча провалов для каждого градиента
            num_gradients: Количество градиентов для генерации
            
        Returns:
            Список текстовых градиентов
        """
        if batch_size is None:
            batch_size = self.config.local_batch_size
        
        if success_examples is None:
            success_examples = []
        
        n = len(failure_examples)
        if n == 0:
            return []      
          
        gradients = []
        
        if n >= batch_size * num_gradients:
            indices = list(range(n))
            random.shuffle(indices)
            for i in range(num_gradients):
                batch_idx = indices[i * batch_size:(i + 1) * batch_size]
                batch_failures = [failure_examples[j] for j in batch_idx]
                grad = self.generate_gradient(current_prompt, batch_failures, success_examples[:5] if success_examples else [])
                gradients.append(grad)
        else:
            for _ in range(num_gradients):
                if n >= batch_size:
                    sampled = random.sample(range(n), k=batch_size)
                else:
                    sampled = [random.randrange(n) for _ in range(batch_size)]
                batch_failures = [failure_examples[j] for j in sampled]
                grad = self.generate_gradient(current_prompt, batch_failures, success_examples[:5] if success_examples else [])
                gradients.append(grad)
        
        # Ранжируем градиенты по приоритету
        gradients.sort(key=lambda g: getattr(g, "priority", 0.5), reverse=True)
        return gradients
    
    # ПОСТРОЕНИЕ ПРОМПТА ДЛЯ АНАЛИЗА
    
    def _build_analysis_prompt(self, current_prompt: str, failure_examples: List[Example], success_examples: List[Example], context: Optional[Dict]) -> str:
        """Построение промпта для LLM, который будет анализировать провалы. Шаблон загружается из prompts/analysis.txt"""
        from prompts.loader import load_template

        # Собираем блоки примеров и контекста
        failure_block = ""
        for i, example in enumerate(failure_examples[:20], 1):
            failure_block += f"Example {i}:\n  Input: {example.input_text}\n  Expected: {example.expected_output}\n  Actual: {example.actual_output}\n\n"

        success_section = ""
        if success_examples:
            success_section = "SUCCESS EXAMPLES (where the prompt worked correctly):\n\n"
            for i, example in enumerate(success_examples[:20], 1):
                success_section += f"Example {i}:\n  Input: {example.input_text}\n  Expected: {example.expected_output}\n  Actual: {example.actual_output}\n\n"

        context_block = ""
        if context:
            parts = []
            if "previous_attempts" in context:
                parts.append(f"Previous attempts: {context['previous_attempts']}")
            if "successful_operations" in context:
                parts.append(f"Successful operations in the past: {context['successful_operations']}")
            if "generation" in context:
                parts.append(f"Current generation: {context['generation']}")
            context_block = "\n".join(parts) or "None"

        template = load_template("analysis")
        return template.format(
            current_prompt=current_prompt,
            failure_examples_block=failure_block,
            success_examples_section=success_section,
            context_block=context_block,
        )
    
    # ПАРСИНГ ОТВЕТА LLM
    
    def _parse_gradient_response(self, response_text: str, failure_examples: List[Example], success_examples: List[Example]) -> TextGradient:
        """
        Парсинг ответа LLM и извлечение компонентов текстового градиента
        
        Args:
            response_text: Ответ от LLM
            failure_examples: Исходные примеры провалов
            success_examples: Исходные примеры успехов
            
        Returns:
            Структурированный TextGradient
        """
        # Используем консолидированный парсер для разбиения на секции
        markers = ['## ERROR ANALYSIS', '## SUGGESTED DIRECTION', '## SPECIFIC SUGGESTIONS', '## PRIORITY']
        sections = LLMResponseParser.split_by_markers(response_text, markers)
        
        # Извлекаем компоненты из секций
        error_analysis = sections.get('## ERROR ANALYSIS', '').strip()
        suggested_direction = sections.get('## SUGGESTED DIRECTION', '').strip()
        
        # Извлекаем specific suggestions как нумерованный список
        suggestions_section = sections.get('## SPECIFIC SUGGESTIONS', '')
        specific_suggestions = LLMResponseParser.extract_numbered_list(suggestions_section)
        
        # Извлекаем приоритет
        priority_section = sections.get('## PRIORITY', response_text)
        priority = LLMResponseParser.extract_priority(priority_section)
        
        # Если не удалось распарсить основные компоненты, используем полный текст
        if not error_analysis and not suggested_direction:
            error_analysis = response_text[:500]
            suggested_direction = "See error analysis for details"
        
        return TextGradient(
            failure_examples=failure_examples,
            success_examples=success_examples,
            error_analysis=error_analysis,
            suggested_direction=suggested_direction,
            specific_suggestions=specific_suggestions,
            priority=priority
        )
    
    # КОНТРАСТНАЯ ГЕНЕРАЦИЯ 
    
    def generate_contrastive_gradient(self, current_prompt: str, hard_negatives: List[Example], hard_positives: List[Example]) -> TextGradient:
        """
        Генерация градиента с использованием контрастных примеров
        
        Args:
            current_prompt: Текущий промпт
            hard_negatives: Сложные примеры, которые должны быть отклонены
            hard_positives: Сложные примеры, которые должны быть приняты
            
        Returns:
            TextGradient с фокусом на различение
        """
        if not getattr(self, 'llm', None) or self.llm.provider is None:
            raise ValueError("No LLM client configured.")
        
        from prompts.loader import load_template
        
        # Подготавливаем блоки примеров
        hard_negatives_block = ""
        for i, example in enumerate(hard_negatives[:5], 1):
            hard_negatives_block += f"\n{i}. Input: {example.input_text}\n"
            hard_negatives_block += f"   Actual output: {example.actual_output}\n"
        
        hard_positives_block = ""
        for i, example in enumerate(hard_positives[:5], 1):
            hard_positives_block += f"\n{i}. Input: {example.input_text}\n"
            hard_positives_block += f"   Actual output: {example.actual_output}\n"
        
        # Загружаем шаблон
        template = load_template("contrastive")
        analysis_prompt = template.format(
            current_prompt=current_prompt,
            hard_negatives_block=hard_negatives_block,
            hard_positives_block=hard_positives_block
        )
        
        try:
            analysis_text = self.llm.call(analysis_prompt)
            
            gradient = self._parse_gradient_response(
                analysis_text,
                hard_negatives,
                hard_positives
            )
            
            # Повышаем приоритет контрастных градиентов
            gradient.priority = min(1.0, gradient.priority + 0.1)
            gradient.metadata["type"] = "contrastive"
            
            return gradient
            
        except Exception as e:
            print(f"Error generating contrastive gradient: {e}")
            return TextGradient(
                failure_examples=hard_negatives,
                success_examples=hard_positives,
                error_analysis="Failed to generate contrastive analysis",
                suggested_direction="Unable to provide suggestions",
                priority=0.0
            )
    
    # КЛАСТЕРИЗАЦИЯ ОШИБОК
    
    def cluster_failure_types(self, failure_examples: List[Example]) -> Dict[str, List[Example]]:
        """
        Кластеризация провалов по типам ошибок
        Помогает генерировать более специфичные градиенты для разных типов проблем
        
        Args:
            failure_examples: Список провалов
            
        Returns:
            Словарь {тип_ошибки: список_примеров}
        """
        if not getattr(self, 'llm', None) or self.llm.provider is None:
            # Простая эвристическая кластеризация без LLM
            return self._heuristic_clustering(failure_examples)
        
        # Если примеров мало, не кластеризуем
        if len(failure_examples) < 5:
            return {"all": failure_examples}
        
        from prompts.loader import load_template
        
        # Подготавливаем блок примеров
        examples_block = ""
        for i, example in enumerate(failure_examples[:20], 1):  # Ограничиваем 20
            examples_block += f"{i}. Input: {example.input_text}\n"
            examples_block += f"   Expected: {example.expected_output}\n"
            examples_block += f"   Actual: {example.actual_output}\n\n"
        
        # Загружаем шаблон и заполняем его
        template = load_template("clustering")
        clustering_prompt = "Analyze these failure examples and group them by error type.\n\n"
        clustering_prompt += examples_block
        clustering_prompt += template
        
        try:
            response_text = self.llm.call(clustering_prompt, temperature=0.5)
            clusters = self._parse_clusters(response_text, failure_examples)
            return clusters
            
        except Exception as e:
            print(f"Error clustering failures: {e}")
            return self._heuristic_clustering(failure_examples)
    
    def _heuristic_clustering(self, failure_examples: List[Example]) -> Dict[str, List[Example]]:
        """Простая эвристическая кластеризация без LLM. Группирует по длине ошибки и типу несовпадения"""
        clusters = defaultdict(list)
        
        for example in failure_examples:
            # Определяем тип ошибки эвристически
            expected = example.expected_output.strip().lower()
            actual = example.actual_output.strip().lower() if example.actual_output else ""
            
            if not actual:
                cluster_name = "no_output"
            elif len(actual) < 5:
                cluster_name = "too_short"
            elif len(actual) > len(expected) * 3:
                cluster_name = "too_verbose"
            elif expected[:20] not in actual and actual[:20] not in expected:
                cluster_name = "completely_wrong"
            else:
                cluster_name = "partial_match"
            
            clusters[cluster_name].append(example)
        
        return dict(clusters)
    
    def _parse_clusters(self, response_text: str, failure_examples: List[Example]) -> Dict[str, List[Example]]:
        """Парсинг ответа LLM о кластерах"""
        clusters = defaultdict(list)
        
        lines = response_text.split('\n')
        current_category = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("CATEGORY:"):
                current_category = line.replace("CATEGORY:", "").strip()
            elif line.startswith("EXAMPLES:") and current_category:
                # Извлекаем номера примеров
                examples_str = line.replace("EXAMPLES:", "").strip()
                example_nums = []
                
                import re
                matches = re.findall(r'\d+', examples_str)
                example_nums = [int(m) - 1 for m in matches]  # -1 для 0-based индекса
                
                # Добавляем примеры в кластер
                for idx in example_nums:
                    if 0 <= idx < len(failure_examples):
                        clusters[current_category].append(failure_examples[idx])
        
        # Если не удалось распарсить, возвращаем все в один кластер
        if not clusters:
            clusters["all"] = failure_examples
        
        return dict(clusters)
    
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    
    def get_statistics(self) -> Dict:
        """Статистика генерации градиентов"""
        return {
            "total_gradients_generated": self.total_gradients_generated,
            "total_api_calls": self.llm.total_api_calls,
            "model": self.llm.model
        }
    
    def __repr__(self):
        return f"TextGradientGenerator(provider={self.llm.provider}, gradients={self.total_gradients_generated})"