import re
from typing import List, Dict

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
