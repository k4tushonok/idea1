from typing import List, Dict, Optional
from prompts.templates import Templates
import random
from llm.llm_client import BaseLLM
from llm.llm_response_parser import GradientParser, ClusterParser
from data_structures import Example, TextGradient, PromptNode
from config import LOCAL_MAX_EXAMPLES, LOCAL_BATCH_SIZE, LOCAL_CANDIDATES_PER_ITERATION, MAX_SUCCESS_EXAMPLES, SUCCESS_EXAMPLE_LIMIT, DEFAULT_PRIORITY, CONTRASTIVAE_PRIORITY_BOOST, FAILURE_EXAMPLE_LIMIT, BATCH_SUCCESS_EXAMPLE_LIMIT

class TextGradientGenerator:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate_gradient(self, current_prompt: str, failure_examples: List[Example], success_examples: List[Example] = None, context: Optional[Dict] = None) -> TextGradient:
        """Генерация одного текстового градиента"""
        if not failure_examples:
            raise ValueError("Need at least one failure example to generate gradient")

        analysis_prompt = Templates.build_analysis_prompt(current_prompt, failure_examples, success_examples, context, LOCAL_MAX_EXAMPLES)

        try:
            analysis_text = self.llm.invoke(prompt=analysis_prompt)
            gradient = GradientParser.parse_gradient_response(analysis_text, failure_examples, success_examples)
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
        if not failure_examples:
            return []   
          
        # Сначала кластеризуем провалы и делаем по одному градиенту на кластер
        clusters = self.cluster_failure_types(failure_examples)

        # Сортируем кластеры по размеру (большие сначала) и ограничиваем количеством градиентов
        cluster_items = sorted(list(clusters.items()), key=lambda kv: len(kv[1]), reverse=True)
        selected = cluster_items[:min(LOCAL_CANDIDATES_PER_ITERATION, len(cluster_items))]

        # Создаём батчи — максимум по batch_size примеров из каждого кластера
        batches: List[List[Example]] = []
        cluster_names: List[str] = []
        for name, examples in selected:
            cluster_names.append(name)
            batches.append(examples[:LOCAL_BATCH_SIZE])

        if not batches:  # fallback к случайным батчам
            for i in range(LOCAL_CANDIDATES_PER_ITERATION):
                sampled_indices = random.sample(range(len(failure_examples)), k=min(LOCAL_BATCH_SIZE, len(failure_examples)))
                batches.append([failure_examples[j] for j in sampled_indices])
                cluster_names.append(f"cluster_{i}")

        gradients = []
        combined_prompt = Templates.build_gradients_batch_prompt(batches, cluster_names, success_examples, max_count=LOCAL_MAX_EXAMPLES)
        
        try:
            response_text = self.llm.invoke(prompt=combined_prompt)
            gradients = GradientParser.parse_batch_response(response_text=response_text, batches=batches, cluster_names=cluster_names, success_examples=success_examples[:BATCH_SUCCESS_EXAMPLE_LIMIT] if success_examples else [])
        except Exception as e:
            print(f"Batch LLM call failed, falling back to per-gradient calls: {e}")
            for batch_index, batch_failures in enumerate(batches):
                grad = self.generate_gradient(current_prompt, batch_failures, success_examples[:SUCCESS_EXAMPLE_LIMIT] if success_examples else [])
                grad.metadata["batch_index"] = batch_index
                grad.metadata["cluster"] = cluster_names[batch_index]
                gradients.append(grad)

        gradients.sort(key=lambda g: getattr(g, "priority", DEFAULT_PRIORITY), reverse=True)
        return gradients
    
    def generate_clustered_gradients(self, node: PromptNode, failure_examples: List[Example], success_examples: List[Example], context: Dict) -> List[TextGradient]:
        print("Clustering failures by error type...")
        clusters = self.cluster_failure_types(failure_examples)
        gradients: List[TextGradient] = []
        
        for name, cluster_failures in clusters.items():
            if not cluster_failures:
                continue
            print(f"  Cluster '{name}': {len(cluster_failures)} failures")
            gradient = self.generate_gradient(
                node.prompt_text,
                cluster_failures[:LOCAL_BATCH_SIZE],
                success_examples[:MAX_SUCCESS_EXAMPLES],
                context
            )
            gradient.metadata["cluster"] = name
            gradients.append(gradient)
            
        gradients.sort(key=lambda g: g.priority, reverse=True)
        return gradients[:LOCAL_CANDIDATES_PER_ITERATION]
    
    def generate_contrastive_gradient(self, current_prompt: str, hard_negatives: List[Example], hard_positives: List[Example]) -> TextGradient:
        """Генерация градиента с использованием контрастных примеров"""
        analysis_prompt = Templates.build_contrastive_prompt(current_prompt, hard_negatives, hard_positives)
        
        try:
            analysis_text = self.llm.invoke(prompt=analysis_prompt)
            gradient = GradientParser.parse_gradient_response(analysis_text, hard_negatives, hard_positives, batch_index=0, cluster_name="contrastive")
            gradient.priority = min(1.0, gradient.priority + CONTRASTIVAE_PRIORITY_BOOST)
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
        if len(failure_examples) < FAILURE_EXAMPLE_LIMIT:
            return {"all": failure_examples}

        clustering_prompt = Templates.build_clustering_prompt(failure_examples, max_count=LOCAL_MAX_EXAMPLES)
        
        try:
            response_text = self.llm.invoke(prompt=clustering_prompt)
            return ClusterParser.parse_clusters(response_text, failure_examples)
        except Exception as e:
            print(f"Error clustering failures: {e}")
            return {"all": failure_examples}        