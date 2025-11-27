import re
from typing import List, Dict, Any


class LLMResponseParser:
    """Базовый класс для парсинга LLM ответов"""
    
    @staticmethod
    def extract_code_blocks(text: str, language: str = "") -> List[str]:
        """
        Извлечение всех блоков кода (между backticks) из текста
        
        Args:
            text: Текст для парсинга
            language: Опциональный фильтр по языку (например, 'python')
            
        Returns:
            Список извлеченных блоков кода
        """
        # Регулярное выражение для блоков кода с тройными backticks
        pattern = r'```(?:' + language + r')?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        blocks = []
        for match in matches:
            block = match.strip()
            if block and len(block) >= 20:  # Фильтруем слишком короткие блоки
                blocks.append(block)
        
        return blocks
    
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
        sections = {}
        flags = 0 if case_sensitive else re.IGNORECASE
        
        # Сортируем маркеры по длине (от длинных к коротким) для правильного поиска
        sorted_markers = sorted(markers, key=len, reverse=True)
        
        current_pos = 0
        sections_found = []
        
        for marker in sorted_markers:
            matches = list(re.finditer(re.escape(marker), text, flags))
            for match in matches:
                sections_found.append((match.start(), match.end(), marker))
        
        # Сортируем по позиции
        sections_found.sort(key=lambda x: x[0])
        
        # Извлекаем содержимое для каждого маркера
        for i, (start, end, marker) in enumerate(sections_found):
            next_start = sections_found[i + 1][0] if i + 1 < len(sections_found) else len(text)
            content = text[end:next_start].strip()
            sections[marker.upper()] = content
        
        return sections
    
    @staticmethod
    def extract_numbered_list(text: str) -> List[str]:
        """
        Извлечение нумерованного списка из текста
        Работает с форматами: 1. item, 1) item, - item, * item, • item
        
        Args:
            text: Текст содержащий список
            
        Returns:
            Список пунктов
        """
        # Регулярное выражение для нумерованных/маркированных пунктов
        pattern = r'^(?:\d+[.)]\s*|[-*•]\s+)(.+?)$'
        
        items = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            match = re.match(pattern, line)
            if match:
                item = match.group(1).strip()
                if item and len(item) > 5:  # Фильтруем коротких пунктов
                    items.append(item)
        
        return items
    
    @staticmethod
    def extract_key_value_pairs(text: str, separator: str = ":") -> Dict[str, str]:
        """
        Извлечение пар ключ-значение из текста
        
        Args:
            text: Текст содержащий пары
            separator: Разделитель между ключом и значением
            
        Returns:
            Словарь {ключ: значение}
        """
        pairs = {}
        
        for line in text.split('\n'):
            line = line.strip()
            if separator not in line:
                continue
            
            key, value = line.split(separator, 1)
            key = key.strip().lower()
            value = value.strip()
            
            if key and value:
                pairs[key] = value
        
        return pairs
    
    @staticmethod
    def clean_markdown(text: str) -> str:
        """
        Удаление Markdown форматирования из текста
        
        Args:
            text: Текст с Markdown
            
        Returns:
            Очищенный текст
        """
        # Удаляем блоки кода
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # Удаляем инлайн код
        text = re.sub(r'`[^`]+`', '', text)
        
        # Удаляем жирный текст
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        
        # Удаляем курсив
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        
        # Удаляем ссылки
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        
        # Удаляем заголовки
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    @staticmethod
    def extract_sections_with_content(text: str, min_length: int = 50) -> Dict[str, str]:
        """
        Извлечение всех секций с заголовками из структурированного текста
        
        Args:
            text: Структурированный текст
            min_length: Минимальная длина содержимого секции
            
        Returns:
            Словарь {заголовок: содержимое}
        """
        sections = {}
        
        # Ищем заголовки (##, ###, etc) или разделители (---, ===)
        pattern = r'^(?:#+\s+|.+?\n(?:---+|-{3,}|={3,}))\s*(.+?)$'
        
        current_section = None
        current_content = []
        
        for line in text.split('\n'):
            # Проверяем, является ли это заголовком
            header_match = re.match(r'^#+\s+(.+)', line)
            if header_match:
                # Сохраняем предыдущую секцию
                if current_section and len('\n'.join(current_content)) >= min_length:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                current_section = header_match.group(1).strip()
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Сохраняем последнюю секцию
        if current_section and len('\n'.join(current_content)) >= min_length:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
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
        # Ищем паттерн типа "priority: 0.75" или просто "0.75"
        pattern = r'(?:priority|score)?\s*:?\s*([0-1]?\.?\d{1,2})'
        
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for match in matches:
            try:
                value = float(match)
                if 0.0 <= value <= 1.0:
                    return value
            except ValueError:
                continue
        
        # Если не найдено, возвращаем средний приоритет
        return 0.5
    
    @staticmethod
    def split_by_delimiter(text: str, delimiter: str = "\n\n", min_length: int = 50) -> List[str]:
        """
        Разбиение текста по разделителю на блоки
        
        Args:
            text: Текст для разбиения
            delimiter: Разделитель
            min_length: Минимальная длина блока
            
        Returns:
            Список блоков
        """
        blocks = text.split(delimiter)
        return [block.strip() for block in blocks if len(block.strip()) >= min_length]
