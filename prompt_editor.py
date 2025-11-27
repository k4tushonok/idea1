from typing import List, Dict, Optional
from llm_client import LLMClient, create_llm_client
from llm_response_parser import LLMResponseParser
import re

from data_structures import (
    TextGradient,
    EditOperation,
    OperationType,
    PromptNode,
    OptimizationSource,
    OptimizationConfig
)

class PromptEditor:
    """Редактор промптов - применяет текстовые градиенты для создания улучшенных вариантов. Использует LLM для интеллектуального редактирования на основе градиентов"""
    
    def __init__(self, config: OptimizationConfig, api_config: Optional[Dict[str, str]] = None, model: str = None):
        """
        Args:
            config: Конфигурация оптимизации
            api_config: Конфигурация API {"provider": "...", "{provider}_api_key": "..."}
            model: Модель для генерации вариантов
        """
        self.config = config
        self.model = model
        self.api_config = api_config or {}
        self.provider = self.api_config.get("provider")
        
        # Инициализация LLM клиента
        self.llm = create_llm_client(self.config, self.api_config, self.model)
        
        # Статистика
        self.total_edits = 0
    
    # ОСНОВНАЯ ГЕНЕРАЦИЯ ВАРИАНТОВ
    
    def generate_variants(self, current_prompt: str, gradient: TextGradient, num_variants: int = None, parent_node: Optional[PromptNode] = None) -> List[PromptNode]:
        """
        Генерация вариантов промпта на основе текстового градиента
        
        Args:
            current_prompt: Текущий промпт
            gradient: Текстовый градиент с рекомендациями
            num_variants: Количество вариантов (по умолчанию из конфига)
            parent_node: Родительский узел (для связи в дереве)
            
        Returns:
            Список PromptNode с новыми вариантами промптов
        """
        if not getattr(self, 'llm', None) or self.llm.provider is None:
            raise ValueError("No LLM client configured. Cannot generate variants.")
        
        if num_variants is None:
            num_variants = self.config.local_candidates_per_iteration
        
        # Формируем промпт для генерации вариантов
        editing_prompt = self._build_editing_prompt(
            current_prompt,
            gradient,
            num_variants
        )
        
        try:
            response_text = self.llm.call(editing_prompt, max_tokens=4000)
            
            # Парсим варианты из ответа
            variants = self._parse_variants(
                response_text,
                current_prompt,
                gradient,
                parent_node
            )
            
            self.total_edits += len(variants)
            return variants
            
        except Exception as e:
            print(f"Error generating variants: {e}")
            # Возвращаем хотя бы один вариант с базовыми изменениями
            return self._generate_fallback_variant(current_prompt, gradient, parent_node)
    
    # ПОСТРОЕНИЕ ПРОМПТА ДЛЯ РЕДАКТИРОВАНИЯ
    
    def _build_editing_prompt(self, current_prompt: str, gradient: TextGradient, num_variants: int) -> str:
        """Построение промпта для LLM, который будет генерировать улучшенные варианты. Загружает шаблон из prompts/editing.txt"""
        from prompts.loader import load_template
        
        # Подготавливаем блоки для заполнения шаблона
        specific_suggestions_block = ""
        for i, suggestion in enumerate(gradient.specific_suggestions, 1):
            specific_suggestions_block += f"{i}. {suggestion}\n"
        
        failure_examples_block = ""
        if gradient.failure_examples:
            for i, example in enumerate(gradient.failure_examples[:3], 1):
                failure_examples_block += f"{i}. Input: {example.input_text}\n"
                failure_examples_block += f"   Expected: {example.expected_output}\n"
                failure_examples_block += f"   Got: {example.actual_output}\n\n"
        
        # Загружаем и заполняем шаблон
        template = load_template("editing")
        prompt = template.format(
            current_prompt=current_prompt,
            error_analysis=gradient.error_analysis,
            suggested_direction=gradient.suggested_direction,
            specific_suggestions_block=specific_suggestions_block,
            failure_examples_block=failure_examples_block,
            num_variants=num_variants
        )
        
        return prompt
    
    # ПАРСИНГ ВАРИАНТОВ
    
    def _parse_variants(self, response_text: str, original_prompt: str, gradient: TextGradient, parent_node: Optional[PromptNode]) -> List[PromptNode]:
        """
        Парсинг вариантов промптов из ответа LLM
        
        Args:
            response_text: Ответ от LLM
            original_prompt: Оригинальный промпт
            gradient: Использованный градиент
            parent_node: Родительский узел
            
        Returns:
            Список новых PromptNode
        """
        variants = []
        
        # Используем парсер для извлечения блоков кода
        code_blocks = LLMResponseParser.extract_code_blocks(response_text)
        
        for block in code_blocks:
            try:
                variant_node = self._parse_single_variant_from_block(
                    block,
                    original_prompt,
                    gradient,
                    parent_node
                )
                if variant_node:
                    variants.append(variant_node)
            except Exception as e:
                print(f"Error parsing variant block: {e}")
                continue
        
        # Если не удалось распарсить стандартным способом, пробуем альтернативный
        if not variants:
            variants = self._parse_variants_alternative(
                response_text,
                original_prompt,
                gradient,
                parent_node
            )
        
        return variants
    
    def _parse_single_variant_from_block(self, block: str, original_prompt: str, gradient: TextGradient, parent_node: Optional[PromptNode]) -> Optional[PromptNode]:
        """Парсинг одного варианта из блока текста - консолидированная версия"""
        
        # Извлекаем описание изменений
        changes_match = re.search(r'CHANGES MADE:\s*(.+?)(?=OPERATION TYPE:|PROMPT:|$)', block, re.DOTALL)
        changes_description = changes_match.group(1).strip() if changes_match else "Modified based on gradient"
        
        # Извлекаем тип операции
        op_type_match = re.search(r'OPERATION TYPE:\s*(\w+)', block)
        op_type_str = op_type_match.group(1).strip().lower() if op_type_match else "modify_instruction"
        
        # Преобразуем строку в OperationType
        operation_type = self._string_to_operation_type(op_type_str)
        
        # Извлекаем новый промпт
        prompt_match = re.search(r'PROMPT:\s*```(.*?)```', block, re.DOTALL)
        if not prompt_match:
            # Пробуем без backticks
            prompt_match = re.search(r'PROMPT:\s*(.+?)(?=VARIANT|$)', block, re.DOTALL)
        
        if not prompt_match:
            return None
        
        new_prompt = prompt_match.group(1).strip()
        
        # Проверяем, что промпт не пустой и отличается от оригинала
        if not new_prompt or len(new_prompt) < 20:
            return None
        
        # Создаем EditOperation
        operation = EditOperation(
            operation_type=operation_type,
            description=changes_description,
            gradient_source=gradient,
            before_snippet=original_prompt[:200] + "...",
            after_snippet=new_prompt[:200] + "..."
        )
        
        # Создаем новый PromptNode
        new_generation = (parent_node.generation + 1) if parent_node else 1
        
        node = PromptNode(
            prompt_text=new_prompt,
            parent_id=parent_node.id if parent_node else None,
            generation=new_generation,
            source=OptimizationSource.LOCAL,
            operations=[operation]
        )
        
        return node
    
    def _parse_variants_alternative(self, response_text: str, original_prompt: str, gradient: TextGradient, parent_node: Optional[PromptNode]) -> List[PromptNode]:
        """Альтернативный метод парсинга - если основной не сработал. Просто извлекаем все блоки кода как варианты"""
        variants = []
        
        # Используем парсер для извлечения всех блоков кода
        code_blocks = LLMResponseParser.extract_code_blocks(response_text)
        
        for i, block in enumerate(code_blocks):
            block = block.strip()
            
            # Пропускаем слишком короткие блоки
            if len(block) < 50:
                continue
            
            # Создаем базовую операцию
            operation = EditOperation(
                operation_type=OperationType.MODIFY_INSTRUCTION,
                description=f"Variant {i+1} based on gradient analysis",
                gradient_source=gradient
            )
            
            new_generation = (parent_node.generation + 1) if parent_node else 1
            
            node = PromptNode(
                prompt_text=block,
                parent_id=parent_node.id if parent_node else None,
                generation=new_generation,
                source=OptimizationSource.LOCAL,
                operations=[operation]
            )
            
            variants.append(node)
        
        return variants
    
    # СПЕЦИФИЧЕСКИЕ ТИПЫ РЕДАКТИРОВАНИЯ
    
    def apply_specific_operation(self, current_prompt: str, operation_type: OperationType, content: str, parent_node: Optional[PromptNode] = None) -> PromptNode:
        """
        Применение конкретной операции редактирования
        Полезно для глобального оптимизатора
        
        Args:
            current_prompt: Текущий промпт
            operation_type: Тип операции
            content: Содержимое для добавления/изменения
            parent_node: Родительский узел
            
        Returns:
            Новый PromptNode
        """
        if not getattr(self, 'llm', None) or self.llm.provider is None:
            raise ValueError("No LLM client configured.")
        
        from prompts.loader import load_template
        
        # Формируем промпт в зависимости от типа операции
        if operation_type == OperationType.ADD_INSTRUCTION:
            template = load_template("add_instruction")
            editing_prompt = template.format(
                current_prompt=current_prompt,
                content=content
            )
        elif operation_type == OperationType.ADD_EXAMPLE:
            template = load_template("add_example")
            editing_prompt = template.format(
                current_prompt=current_prompt,
                content=content
            )
        elif operation_type == OperationType.RESTRUCTURE:
            template = load_template("restructure")
            editing_prompt = template.format(
                current_prompt=current_prompt,
                content=content
            )
        elif operation_type == OperationType.CLARIFY:
            editing_prompt = f"""Clarify the following aspect of the prompt:

            CURRENT PROMPT:
            ```
            {current_prompt}
            ```

            CLARIFICATION NEEDED:
            {content}

            Make the prompt clearer regarding this point. Output the complete modified prompt.
        """
        else:
            # Общий случай
            editing_prompt = f"""Modify the prompt as follows:

                CURRENT PROMPT:
                ```
                {current_prompt}
                ```

                MODIFICATION:
                {content}

                Apply this modification. Output the complete modified prompt.
            """
        
        try:
            new_prompt = self.llm.call(editing_prompt, temperature=0.7, max_tokens=2000)
            
            # Убираем markdown форматирование если есть
            new_prompt = re.sub(r'^```.*?\n', '', new_prompt)
            new_prompt = re.sub(r'\n```$', '', new_prompt)
            new_prompt = new_prompt.strip()
            
            # Создаем операцию
            operation = EditOperation(
                operation_type=operation_type,
                description=content[:200],
                before_snippet=current_prompt[:200] + "...",
                after_snippet=new_prompt[:200] + "..."
            )
            
            new_generation = (parent_node.generation + 1) if parent_node else 1
            
            node = PromptNode(
                prompt_text=new_prompt,
                parent_id=parent_node.id if parent_node else None,
                generation=new_generation,
                source=OptimizationSource.GLOBAL,  # Предполагаем глобальный источник
                operations=[operation]
            )
            
            self.total_edits += 1
            return node
            
        except Exception as e:
            print(f"Error applying operation: {e}")
            # Возвращаем исходный промпт в случае ошибки
            return parent_node if parent_node else PromptNode(prompt_text=current_prompt)
    
    # КОМБИНИРОВАНИЕ ПРОМПТОВ
    
    def combine_prompts(self, prompts: List[str], combination_strategy: str = "best_elements") -> PromptNode:
        """
        Комбинирование нескольких успешных промптов
        Используется глобальным оптимизатором
        
        Args:
            prompts: Список промптов для комбинирования
            combination_strategy: Стратегия комбинирования
                - "best_elements": Берет лучшие элементы из каждого
                - "sequential": Последовательное добавление инструкций
                - "synthesize": Синтезирует новый промпт
                
        Returns:
            Новый PromptNode с комбинированным промптом
        """
        if not getattr(self, 'llm', None) or self.llm.provider is None:
            raise ValueError("No LLM client configured.")
        
        if len(prompts) < 2:
            raise ValueError("Need at least 2 prompts to combine")
        
        from prompts.loader import load_template
        
        # Подготавливаем блок с промптами
        prompts_block = ""
        for i, p in enumerate(prompts, 1):
            prompts_block += f"\nPROMPT {i}:\n```\n{p}\n```\n"
        
        # Загружаем базовый шаблон
        template = load_template("combine")
        combining_prompt = template.format(prompts_block=prompts_block)
        
        # Добавляем специфичный для стратегии контент
        if combination_strategy == "best_elements":
            adding_prompt = """
                - Identify the strongest instructions from each prompt
                - Combine them coherently without redundancy
                - Maintain clarity and logical flow
                - Output a single, unified prompt that is better than any individual prompt
            """
        elif combination_strategy == "sequential":
            adding_prompt = """
                - Organize instructions in a logical order
                - Remove redundancies
                - Ensure smooth transitions between sections
                - Output a single, well-structured prompt
            """
        else:  # synthesize
            adding_prompt = """
                - Understand the common goals and patterns
                - Create a fresh prompt that achieves the same objectives more effectively
                - Don't just merge - innovate based on what works in each
                - Output a single, optimized prompt
            """
        
        combining_prompt += "\nGUIDELINES:\n" + adding_prompt
        
        try:
            combined_prompt = self.llm.call(combining_prompt, temperature=0.7, max_tokens=3000)
            
            combined_prompt = re.sub(r'^```.*?\n', '', combined_prompt)
            combined_prompt = re.sub(r'\n```$', '', combined_prompt)
            combined_prompt = combined_prompt.strip()
            
            # Создаем операцию
            operation = EditOperation(
                operation_type=OperationType.RESTRUCTURE,
                description=f"Combined {len(prompts)} prompts using {combination_strategy} strategy",
                before_snippet=f"Combined from {len(prompts)} sources",
                after_snippet=combined_prompt[:200] + "..."
            )
            
            node = PromptNode(
                prompt_text=combined_prompt,
                source=OptimizationSource.GLOBAL,
                operations=[operation]
            )
            
            self.total_edits += 1
            return node
            
        except Exception as e:
            print(f"Error combining prompts: {e}")
            # Возвращаем первый промпт в случае ошибки
            return PromptNode(prompt_text=prompts[0])
    
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    
    def _string_to_operation_type(self, op_str: str) -> OperationType:
        """Преобразование строки в OperationType"""
        op_str = op_str.lower().replace(" ", "_")
        
        mapping = {
            "add_instruction": OperationType.ADD_INSTRUCTION,
            "modify_instruction": OperationType.MODIFY_INSTRUCTION,
            "remove_instruction": OperationType.REMOVE_INSTRUCTION,
            "add_example": OperationType.ADD_EXAMPLE,
            "rephrase": OperationType.REPHRASE,
            "restructure": OperationType.RESTRUCTURE,
            "add_constraint": OperationType.ADD_CONSTRAINT,
            "clarify": OperationType.CLARIFY,
        }
        
        return mapping.get(op_str, OperationType.MODIFY_INSTRUCTION)
    
    def _generate_fallback_variant(self, current_prompt: str, gradient: TextGradient, parent_node: Optional[PromptNode]) -> List[PromptNode]:
        """Генерация fallback варианта, если основной метод не сработал. Делает простые изменения на основе градиента"""
        # Простое добавление рекомендаций к промпту
        suggestions_text = "\n".join([f"- {s}" for s in gradient.specific_suggestions[:3]])
        
        modified_prompt = f"""{current_prompt}

            Additional guidance based on error analysis:
            {suggestions_text}
        """
        
        operation = EditOperation(
            operation_type=OperationType.ADD_INSTRUCTION,
            description="Fallback: Added suggestions from gradient",
            gradient_source=gradient
        )
        
        new_generation = (parent_node.generation + 1) if parent_node else 1
        
        node = PromptNode(
            prompt_text=modified_prompt,
            parent_id=parent_node.id if parent_node else None,
            generation=new_generation,
            source=OptimizationSource.LOCAL,
            operations=[operation]
        )
        
        return [node]
    
    def calculate_edit_distance(self, prompt1: str, prompt2: str) -> float:
        """
        Вычисление расстояния редактирования между промптами
        Используется для оценки diversity
        
        Returns:
            Нормализованное расстояние от 0.0 (идентичны) до 1.0 (полностью разные)
        """
        # Простая метрика на основе слов
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        # Jaccard distance
        similarity = len(intersection) / len(union)
        distance = 1.0 - similarity
        
        return distance
    
    def get_statistics(self) -> Dict:
        """Статистика редактирования"""
        return {
            "total_edits": self.total_edits,
            "total_api_calls": getattr(self.llm, 'total_api_calls', 0),
            "model": getattr(self.llm, 'model', self.model)
        }
    
    def __repr__(self):
        return f"PromptEditor(model={self.model}, edits={self.total_edits})"