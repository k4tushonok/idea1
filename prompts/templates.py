from pathlib import Path
from typing import List
from data_structures import Example, TextGradient, PromptNode
from typing import Dict, Optional

TEMPLATES_DIR = Path(__file__).parent

class Templates:
    @staticmethod
    def load_template(name: str) -> str:
        """Загрузка шаблона из файла prompts/<name>.txt"""
        path = TEMPLATES_DIR / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        return path.read_text(encoding="utf-8")
    
    @staticmethod
    def format_examples(examples: List[Example], max_count: int = None, include_expected: bool = True) -> str:
        """Форматирование списка примеров в текстовый блок"""
        block = ""
        for i, example in enumerate(examples[:max_count], 1):
            block += f"Example {i}:\n  Input: {example.input_text}\n"
            if include_expected:
                block += f"  Expected: {example.expected_output}\n"
            block += f"  Actual: {example.actual_output}\n\n"
        return block
    
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
    def build_gradients_batch_prompt(current_prompt: str, batches: list, cluster_names: list, success_examples: List[Example], max_count: int, reflection_context: str = "") -> str:
        """Пакетный промпт для генерации N градиентов (по одному на кластер провалов)"""
        # Инструкция: вернуть N градиентов, каждый с наборами секций
        header = (
            f"You are an expert prompt engineer analyzing why a prompt fails on certain examples. "
            f"Generate {len(batches)} separate gradient analyses, one per cluster of failures. "
            "For each cluster listed below, produce a block starting with a header in the format: '### GRADIENT <i> - <cluster_name>' (i starting at 1). "                
            "Each block must contain the following sections exactly: '## ERROR ANALYSIS', '## SUGGESTED DIRECTION', "
            "'## SPECIFIC SUGGESTIONS' (a numbered list), and '## PRIORITY' (a number from 0.0 to 1.0). "
            "Return the blocks in the same order as the clusters and avoid extra commentary.\n\n"
            "CRITICAL: Prefer high-impact structural changes (restructure order, add constraints, change formatting) over tiny wording tweaks.\n"
            "Do NOT suggest copying specific values from examples into the prompt."
        )
        
        reflection_block = ""
        if reflection_context:
            reflection_block = (
                f"\n\nREFLECTION FROM PREVIOUS OPTIMIZATION STEP "
                f"(use this to guide your analysis):\n{reflection_context}\n"
            )
        
        prompt_block = f"\nCURRENT PROMPT BEING ANALYZED:\n```\n{current_prompt}\n```\n"
        
        # Собираем блоки с провалами/успехами
        body_parts = []
        for i, batch_failures in enumerate(batches, start=1):
            cluster_name = cluster_names[i - 1]
            failure_block = Templates.format_examples(batch_failures, max_count=max_count)
            success_section = ""
            if success_examples:
                success_section = f"\nSUCCESS EXAMPLES (where the prompt worked correctly):\n{Templates.format_examples(success_examples[:5], max_count=5)}"
            body_parts.append(f"--- CLUSTER: {cluster_name} (SET {i}) ---\nFAILURE EXAMPLES:\n{failure_block}{success_section}")
            
        return header + reflection_block + prompt_block + "\n\n".join(body_parts)
    
    @staticmethod
    def build_meta_optimizer_prompt(
        history_nodes: List,
        best_node,
        exemplars: Optional[List[Example]] = None,
        few_shot_examples: Optional[List[Example]] = None,
        reflection_context: str = "",
    ) -> str:
        """Мета-промпт: история + wrong-exemplars + few-shot Q&A примеры + reflection"""
        history_lines = []
        for node in history_nodes:
            history_lines.append(
                f"Instruction: {node.prompt_text}\nScore: {node.selection_score() * 100:.1f}"
            )
        history_block = "\n\n".join(history_lines)

        if exemplars:
            ex_lines = []
            for i, ex in enumerate(exemplars, 1):
                line = f"{i}. Q: {ex.input_text[:300]}\n   Expected: {ex.expected_output}"
                if ex.actual_output:
                    line += f"\n   Model output: {ex.actual_output[:200]}"
                ex_lines.append(line)
            exemplars_block = "\n".join(ex_lines)
        else:
            exemplars_block = "None"

        if few_shot_examples:
            fs_lines = [
                f"{i}. Q: {ex.input_text[:300]}\n   A: {ex.expected_output}"
                for i, ex in enumerate(few_shot_examples, 1)
            ]
            few_shot_block = "\n".join(fs_lines)
        else:
            few_shot_block = "None"

        reflection_block = ""
        if reflection_context:
            reflection_block = (
                f"\nOptimization insight (what worked and what still fails):\n"
                f"{reflection_context}\n"
            )

        template = Templates.load_template("meta_optimizer")
        return template.format(
            history_block=history_block,
            best_score=best_node.selection_score() * 100,
            exemplars_block=exemplars_block,
            few_shot_block=few_shot_block,
            reflection_block=reflection_block,
        )

    @staticmethod
    def build_editing_prompt(current_prompt: str, gradient: TextGradient, num_variants: int) -> str:
        """Промпт для генерации вариантов промпта на основе градиента."""
        suggestions = "\n".join(f"{i}. {s}" for i, s in enumerate(gradient.specific_suggestions, 1))

        failure_examples = gradient.failure_examples
        if failure_examples:
            n = len(failure_examples)
            if n <= 3:
                sampled = failure_examples
            else:
                indices = [0, n // 2, n - 1]
                sampled = [failure_examples[i] for i in indices]
        else:
            sampled = []

        failures = "\n".join(
            f"{i}. Input: {e.input_text}\n   Expected: {e.expected_output}\n   Got: {e.actual_output}"
            for i, e in enumerate(sampled, 1)
        )

        template = Templates.load_template("editing")
        return template.format(
            current_prompt=current_prompt,
            failures_block=failures,
            error_analysis=gradient.error_analysis,
            suggested_direction=gradient.suggested_direction,
            specific_suggestions_block=suggestions,
            num_variants=num_variants
        )

    @staticmethod
    def build_crossover_prompt(node_a: 'PromptNode', node_b: 'PromptNode') -> str:
        """Построение промпта кроссовера, комбинирующего лучшие элементы двух топовых промптов."""
        template = Templates.load_template("crossover")
        return template.format(
            prompt_a=node_a.prompt_text,
            score_a=f"{node_a.selection_score():.3f}",
            prompt_b=node_b.prompt_text,
            score_b=f"{node_b.selection_score():.3f}",
        )
