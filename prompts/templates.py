from pathlib import Path
from typing import List
from data_structures import Example, OperationType, TextGradient
from typing import Dict, Optional

TEMPLATES_DIR = Path(__file__).parent

class Templates:
    @staticmethod
    def load_template(name: str) -> str:
        path = TEMPLATES_DIR / f"{name}.txt"
        if not path.exists():
            return f"{{{{missing_template:{name}}}}}"
        return path.read_text(encoding="utf-8")
    
    @staticmethod
    def format_examples(examples: List[Example], max_count: int = None, include_expected: bool = True) -> str:
        block = ""
        for i, example in enumerate(examples[:max_count], 1):
            block += f"Example {i}:\n  Input: {example.input_text}\n"
            if include_expected:
                block += f"  Expected: {example.expected_output}\n"
            block += f"  Actual: {example.actual_output}\n\n"
        return block
    
    @staticmethod    
    def combination_guidelines(strategy: str) -> str:
        return {
            "best_elements": (
                "- Extract strongest instructions\n"
                "- Remove redundancy\n"
                "- Produce a unified prompt"
            ),
            "sequential": (
                "- Order instructions logically\n"
                "- Merge overlapping parts"
            ),
            "synthesize": (
                "- Infer common intent\n"
                "- Create a new optimized prompt"
            ),
        }.get(strategy, "")   
        
    @staticmethod
    def build_analysis_prompt(current_prompt: str, failure_examples: List[Example], success_examples: List[Example], context: Optional[Dict], max_count: int) -> str:
        """Построение промпта для LLM, который будет анализировать провалы. Шаблон загружается из prompts/analysis.txt"""
        blocks = {
            "current_prompt": current_prompt,
            "failure_examples_block": Templates.format_examples(failure_examples, max_count),
            "success_examples_section": Templates.format_examples(success_examples, max_count) if success_examples else "",
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

        template = Templates.load_template("analysis")
        return template.format(**blocks)
    
    @staticmethod
    def build_clustering_prompt(failure_examples: List[Example], max_count: int) -> str:
        """Построение промпта для кластеризации провалов по типам ошибок"""
        examples_block = Templates.format_examples(failure_examples, max_count=max_count)
        template = Templates.load_template("clustering")
        return "Analyze these failure examples and group them by error type.\n\n" + examples_block + template

    @staticmethod
    def build_contrastive_prompt(current_prompt: str, hard_negatives: List[Example], hard_positives: List[Example]) -> str:
        """Построение промпта для контрастного анализа"""
        hard_negatives_block = Templates.format_examples(hard_negatives, max_count=5)
        hard_positives_block = Templates.format_examples(hard_positives, max_count=5)

        template = Templates.load_template("contrastive")
        return template.format(current_prompt=current_prompt, hard_negatives_block=hard_negatives_block, hard_positives_block=hard_positives_block)
        
    @staticmethod
    def build_gradients_batch_prompt(batches: list, cluster_names: list, success_examples: List[Example], max_count: int) -> str:
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
            failure_block = Templates.format_examples(batch_failures, max_count=max_count)
            success_section = Templates.format_examples(success_examples[:5], max_count=5) if success_examples else ""
            body_parts.append(f"--- CLUSTER: {cluster_name} (SET {i}) ---\n{failure_block}{success_section}")
            
        return header + "\n\n" + "\n\n".join(body_parts)
    
    @staticmethod
    def build_specific_prompt(operation_type: OperationType, current_prompt: str, content: str) -> str:
        TEMPLATE_BY_OPERATION = {
            OperationType.ADD_INSTRUCTION: "add_instruction",
            OperationType.ADD_EXAMPLE: "add_example",
            OperationType.RESTRUCTURE: "restructure",
            OperationType.CLARIFY: "clarify",
        }
        
        template_name = TEMPLATE_BY_OPERATION.get(operation_type, "overall")
        template = Templates.load_template(template_name)
        return template.format(current_prompt=current_prompt, content=content)

    @staticmethod
    def build_combine_prompt(prompts: List[str], combination_strategy: str) -> str:
        prompts_block = "\n".join(f"PROMPT {i}:\n```\n{p}\n```" for i, p in enumerate(prompts, 1))
        template = Templates.load_template("combine")
        combining_prompt = template.format(prompts_block=prompts_block)
        combining_prompt += "\nGUIDELINES:\n" + Templates.combination_guidelines(combination_strategy)
        return combining_prompt
    
    @staticmethod
    def build_editing_prompt(current_prompt: str, gradient: TextGradient, num_variants: int) -> str:
        suggestions = "\n".join(f"{i}. {s}" for i, s in enumerate(gradient.specific_suggestions, 1))

        failures = "\n".join(
            f"{i}. Input: {e.input_text}\n   Expected: {e.expected_output}\n   Got: {e.actual_output}"
            for i, e in enumerate(gradient.failure_examples[:3], 1)
        )
        
        template = Templates.load_template("editing")
        return template.format(
            current_prompt=current_prompt,
            error_analysis=gradient.error_analysis,
            suggested_direction=gradient.suggested_direction,
            specific_suggestions_block=suggestions,
            failure_examples_block=failures,
            num_variants=num_variants
        )