import re
from typing import List, Dict
from collections import defaultdict
from data_structures import Example
from data_structures import TextGradient
from data_structures import PromptNode
from typing import Optional
from data_structures import EditOperation
from data_structures import OperationType
from data_structures import OptimizationSource

class LLMResponseParser:
    @staticmethod
    def extract_code_blocks(text: str) -> List[str]:
        """Извлечение всех блоков кода из текста"""
        pattern = r'```(?:\w+)?\s*\n?(.*?)\n?```' 
        matches = re.findall(pattern, text, re.DOTALL)
        
        return [
            block.strip()
            for block in matches
            if block and len(block.strip()) >= 20
        ]
    
    @staticmethod
    def extract_prompt(block: str) -> Optional[str]:
        match = re.search(r'PROMPT:\s*```(.*?)```', block, re.DOTALL)
        if not match:
            match = re.search(r'PROMPT:\s*(.+)', block, re.DOTALL)
        return match.group(1).strip() if match else None
    
    @staticmethod
    def extract_description(block: str) -> str:
        match = re.search(r'CHANGES MADE:\s*(.+?)(?=OPERATION TYPE|PROMPT|$)', block, re.DOTALL)
        return match.group(1).strip() if match else "Edited based on gradient"

    @staticmethod
    def extract_operation_type(block: str) -> OperationType:
        match = re.search(r'OPERATION TYPE:\s*(\w+)', block)
        return LLMResponseParser.string_to_operation_type(match.group(1)) if match else OperationType.MODIFY_INSTRUCTION
    
    @staticmethod
    def extract_numbered_list(text: str) -> List[str]:
        """
        Извлечение нумерованного списка из текста
        Работает с форматами: 
            - 1. item
            - 1) item
            - - item
            - * item
            - • item
        
        Args:
            text: Текст содержащий список
            
        Returns:
            Список пунктов
        """
        pattern = r'^(?:\d+[.)]\s*|[-*•]\s+)(.+?)$'
        items: List[str] = []

        for line in text.splitlines():
            match = re.match(pattern, line.strip())
            if match:
                item = match.group(1).strip()
                if len(item) > 5:
                    items.append(item)

        return items

    @staticmethod
    def extract_priority(text: str) -> float:
        """
        Извлечение значения приоритета из текста
        Ищет числа между 0.0 и 1.0
        
        Args:
            text: Текст содержащий приоритет
            
        Returns:
            Значение приоритета от 0.0 до 1.0
        """
        pattern = r'(?:priority|score)?\s*:?\s*([0-1]?\.?\d{1,2})'
        for value in re.findall(pattern, text, re.IGNORECASE):
            try:
                score = float(value)
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                pass

        return 0.5
    
    @staticmethod
    def split_by_markers(text: str, markers: List[str], case_sensitive: bool = False) -> Dict[str, str]:
        """
        Разбиение текста на секции по маркерам
        
        Args:
            text: Текст для разбиения
            markers: Список маркеров (например, ['## ERROR ANALYSIS', '## SUGGESTIONS'])
            case_sensitive: Учитывать ли регистр
            
        Returns:
            Словарь {маркер: содержимое}
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        sections: Dict[str, str] = {}

        sorted_markers = sorted(markers, key=len, reverse=True)
        found = []

        for marker in sorted_markers:
            for match in re.finditer(re.escape(marker), text, flags):
                found.append((match.start(), match.end(), marker))

        found.sort(key=lambda x: x[0])

        for i, (start, end, marker) in enumerate(found):
            next_start = found[i + 1][0] if i + 1 < len(found) else len(text)
            sections[marker.upper()] = text[end:next_start].strip()

        return sections

    @staticmethod
    def parse_clusters(response_text: str, failure_examples: List[Example]) -> Dict[str, List[Example]]:
        """Парсинг ответа LLM о кластерах"""
        clusters = defaultdict(list)
        lines = response_text.split('\n')
        current_category = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("CATEGORY:"):
                current_category = line.replace("CATEGORY:", "").strip()
            elif line.startswith("EXAMPLES:") and current_category:
                matches = [int(m)-1 for m in re.findall(r'\d+', line)]
                for idx in matches:
                    if 0 <= idx < len(failure_examples):
                        clusters[current_category].append(failure_examples[idx])
        
        if not clusters:
            clusters["all"] = failure_examples
        
        return dict(clusters)
    
    @staticmethod
    def parse_gradient_response(response_text: str, failure_examples: List[Example], success_examples: List[Example], batch_index: int = None, cluster_name: str = None) -> TextGradient:
        """Парсинг ответа LLM и извлечение компонентов текстового градиента"""
        # Используем консолидированный парсер для разбиения на секции
        markers = ['## ERROR ANALYSIS', '## SUGGESTED DIRECTION', '## SPECIFIC SUGGESTIONS', '## PRIORITY']
        sections = LLMResponseParser.split_by_markers(response_text, markers)
        
        # Извлекаем компоненты из секций
        error_analysis = sections.get('## ERROR ANALYSIS', '').strip() or response_text[:500]
        suggested_direction = sections.get('## SUGGESTED DIRECTION', '').strip() or "See error analysis for details"

        # Извлекаем specific suggestions как нумерованный список
        specific_suggestions = LLMResponseParser.extract_numbered_list(sections.get('## SPECIFIC SUGGESTIONS', ''))
        
        # Извлекаем приоритет
        priority = min(max(LLMResponseParser.extract_priority(sections.get('## PRIORITY', '')), 0.0), 1.0)

        gradient = TextGradient(
            failure_examples=failure_examples,
            success_examples=success_examples,
            error_analysis=error_analysis,
            suggested_direction=suggested_direction,
            specific_suggestions=specific_suggestions,
            priority=priority
        )
        gradient.metadata["batch_index"] = batch_index
        gradient.metadata["cluster"] = cluster_name
        return gradient
    
    @staticmethod
    def parse_variants(response_text: str, original_prompt: str, gradient: TextGradient, parent_node: Optional[PromptNode]) -> List[PromptNode]:
        """Парсинг вариантов промптов из ответа LLM"""
        nodes: List[PromptNode] = []

        for block in LLMResponseParser.extract_code_blocks(response_text):
            node = LLMResponseParser.parse_single_variant(block, original_prompt, gradient, parent_node)
            if node:
                nodes.append(node)

        return nodes    
    
    @staticmethod
    def parse_single_variant(block: str, original_prompt: str, gradient: TextGradient, parent_node: Optional[PromptNode]) -> Optional[PromptNode]:
        new_prompt = LLMResponseParser.extract_prompt(block)
        if not new_prompt or len(new_prompt) < 20:
            return None

        operation_type = LLMResponseParser.extract_operation_type(block)
        description = LLMResponseParser.extract_description(block)

        operation = EditOperation(
            operation_type=operation_type,
            description=description,
            gradient_source=gradient,
            before_snippet=original_prompt[:200] + "...",
            after_snippet=new_prompt[:200] + "..."
        )

        generation = parent_node.generation + 1 if parent_node else 1

        return PromptNode(
            prompt_text=new_prompt,
            parent_id=parent_node.id if parent_node else None,
            generation=generation,
            source=OptimizationSource.LOCAL,
            operations=[operation]
        )

    @staticmethod
    def string_to_operation_type(value: str) -> OperationType:
        normalized = value.lower().replace(" ", "_")
        return OperationType.__members__.get(normalized.upper(), OperationType.MODIFY_INSTRUCTION)
    
    @staticmethod    
    def strip_code_fences(text: str) -> str:
        return re.sub(r'^```.*?\n|\n```$', '', text, flags=re.DOTALL).strip()
