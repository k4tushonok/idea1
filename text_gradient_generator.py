from typing import List, Dict, Optional
from prompts.loader import load_template
import random
import re           
from collections import defaultdict
from llm.llm_client import create_llm
from llm.llm_response_parser import LLMResponseParser
from data_structures import (
    Example,
    TextGradient,
    OptimizationConfig
)

class TextGradientGenerator:
    def __init__(self, config: OptimizationConfig, api_config: Optional[Dict[str, str]] = None):
        self.config = config
        self.api_config = api_config or {}
        self.llm = create_llm(self.config, self.api_config)

    def generate_gradient(self, current_prompt: str, failure_examples: List[Example], success_examples: List[Example] = None, context: Optional[Dict] = None) -> TextGradient:
        """Генерация одного текстового градиента"""
        if not failure_examples:
            raise ValueError("Need at least one failure example to generate gradient")
        
        analysis_prompt = self._build_analysis_prompt(current_prompt, failure_examples, success_examples, context)
        
        try:
            analysis_text = self.llm.invoke(prompt=analysis_prompt, temperature=self.config.temperature, max_tokens=self.config.max_tokens)
            gradient = self._parse_gradient_response(analysis_text, failure_examples, success_examples)
            return gradient
        except Exception as e:
            print(f"Error generating gradient: {e}")
            return TextGradient(
                failure_examples=failure_examples,
                success_examples=success_examples,
                error_analysis="Failed to generate analysis",
                suggested_direction="Unable to provide suggestions",
                priority=0.0
            )
    
    def generate_gradients_batch(self, current_prompt: str, failure_examples: List[Example], success_examples: List[Example] = None) -> List[TextGradient]:
        """Генерация нескольких градиентов"""
        batch_size = self.config.local_batch_size
        num_gradients=self.config.local_candidates_per_iteration
        
        n = len(failure_examples)
        if n == 0:
            return []      
          
        # Сначала кластеризуем провалы и делаем по одному градиенту на кластер
        clusters = self.cluster_failure_types(failure_examples)

        # Сортируем кластеры по размеру (большие сначала) и ограничиваем количеством градиентов
        cluster_items = sorted(list(clusters.items()), key=lambda kv: len(kv[1]), reverse=True)
        selected = cluster_items[:min(num_gradients, len(cluster_items))]

        # Создаём батчи — максимум по batch_size примеров из каждого кластера
        batches = []
        cluster_names = []
        for name, examples in selected:
            cluster_names.append(name)
            batches.append(examples[:batch_size])

        if not batches:  # fallback к случайным батчам
            for i in range(num_gradients):
                sampled_indices = random.sample(range(n), k=min(batch_size, n))
                batches.append([failure_examples[j] for j in sampled_indices])
                cluster_names.append(f"cluster_{i}")

        # Инструкция: вернуть N градиентов, каждый с наборами секций
        header = (
            f"You are an assistant for prompt optimization. Generate {len(batches)} separate gradient analyses, one per cluster. "
            "For each cluster listed below, produce a block starting with a header in the format: '### GRADIENT <i> - <cluster_name>' (i starting at 1). "                
            "Each block must contain the following sections exactly: '## ERROR ANALYSIS', '## SUGGESTED DIRECTION', "
            "'## SPECIFIC SUGGESTIONS' (a numbered list), and '## PRIORITY' (a number from 0.0 to 1.0). "
            "Return the blocks in the same order as the clusters and avoid extra commentary."
        )

        # Собираем блоки с провалами/успехами
        body_parts = []
        for i, batch_failures in enumerate(batches, start=1):
            cluster_name = cluster_names[i - 1]
            failure_block = self._format_examples(batch_failures, max_count=self.config.local_max_examples)
            success_section = self._format_examples(success_examples[:5], max_count=5) if success_examples else ""
            body_parts.append(f"--- CLUSTER: {cluster_name} (SET {i}) ---\n{failure_block}{success_section}")

        combined_prompt = header + "\n\n" + "\n\n".join(body_parts)

        gradients = []
        try:
            response_text = self.llm.invoke(prompt=combined_prompt, temperature=self.config.temperature, max_tokens=self.config.max_tokens)
            
            splits = response_text.split('### GRADIENT')
            for idx, block in enumerate(splits[1:], start=1):
                block_text = "### GRADIENT" + block
                cluster_name = None
                for name in cluster_names:
                    if name in block_text:
                        cluster_name = name
                        break
                batch_index = cluster_names.index(cluster_name) if cluster_name else idx - 1
                grad = self._parse_gradient_response(block_text, batches[batch_index], success_examples[:5] if success_examples else [], batch_index=batch_index, cluster_name=cluster_name)
                gradients.append(grad)
        except Exception as e:
            print(f"Batch LLM call failed, falling back to per-gradient calls: {e}")
            for batch_index, batch_failures in enumerate(batches):
                grad = self.generate_gradient(current_prompt, batch_failures, success_examples[:5] if success_examples else [])
                grad.metadata["batch_index"] = batch_index
                grad.metadata["cluster"] = cluster_names[batch_index]
                gradients.append(grad)

        gradients.sort(key=lambda g: getattr(g, "priority", 0.5), reverse=True)
        return gradients
    
    def _build_analysis_prompt(self, current_prompt: str, failure_examples: List[Example], success_examples: List[Example], context: Optional[Dict]) -> str:
        """Построение промпта для LLM, который будет анализировать провалы. Шаблон загружается из prompts/analysis.txt"""
        blocks = {
            "current_prompt": current_prompt,
            "failure_examples_block": self._format_examples(failure_examples, max_count=self.config.local_max_examples),
            "success_examples_section": self._format_examples(success_examples, max_count=self.config.local_max_examples) if success_examples else "",
            "context_block": ""
        }

        if context:
            parts = []
            if "previous_attempts" in context:
                parts.append(f"Previous attempts: {context['previous_attempts']}")
            if "successful_operations" in context:
                parts.append(f"Successful operations in the past: {context['successful_operations']}")
            if "generation" in context:
                parts.append(f"Current generation: {context['generation']}")
            blocks["context_block"] = "\n".join(parts) or "None"

        template = load_template("analysis")
        return template.format(**blocks)

    def _parse_gradient_response(self, response_text: str, failure_examples: List[Example], success_examples: List[Example], batch_index: int = None, cluster_name: str = None) -> TextGradient:
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
    
    def generate_contrastive_gradient(self, current_prompt: str, hard_negatives: List[Example], hard_positives: List[Example]) -> TextGradient:
        """Генерация градиента с использованием контрастных примеров"""
        hard_negatives_block = self._format_examples(hard_negatives, max_count=5)
        hard_positives_block = self._format_examples(hard_positives, max_count=5)
        
        # Загружаем шаблон
        template = load_template("contrastive")
        analysis_prompt = template.format(current_prompt=current_prompt, hard_negatives_block=hard_negatives_block, hard_positives_block=hard_positives_block)
        
        try:
            analysis_text = self.llm.invoke(prompt=analysis_prompt, temperature=self.config.temperature, max_tokens=self.config.max_tokens)
            gradient = self._parse_gradient_response(analysis_text, hard_negatives, hard_positives, batch_index=0, cluster_name="contrastive")
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
    
    def cluster_failure_types(self, failure_examples: List[Example]) -> Dict[str, List[Example]]:
        """Кластеризация провалов по типам ошибок"""
        if len(failure_examples) < 5:
            return {"all": failure_examples}
        
        examples_block = self._format_examples(failure_examples, max_count=self.config.local_max_examples)
        template = load_template("clustering")
        clustering_prompt = "Analyze these failure examples and group them by error type.\n\n" + examples_block + template
        
        try:
            response_text = self.llm.invoke(prompt=clustering_prompt, temperature=self.config.temperature, max_tokens=self.config.max_tokens)
            return self._parse_clusters(response_text, failure_examples)
        except Exception as e:
            print(f"Error clustering failures: {e}")
            return {"all": failure_examples}        
    
    def _format_examples(self, examples: List[Example], max_count: int = None, include_expected: bool = True) -> str:
        block = ""
        for i, example in enumerate(examples[:max_count], 1):
            block += f"Example {i}:\n  Input: {example.input_text}\n"
            if include_expected:
                block += f"  Expected: {example.expected_output}\n"
            block += f"  Actual: {example.actual_output}\n\n"
        return block
        
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
                matches = [int(m)-1 for m in re.findall(r'\d+', line)]
                for idx in matches:
                    if 0 <= idx < len(failure_examples):
                        clusters[current_category].append(failure_examples[idx])
        
        if not clusters:
            clusters["all"] = failure_examples
        
        return dict(clusters)