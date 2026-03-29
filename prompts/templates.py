from pathlib import Path
from typing import List
from data_structures import Example, TextGradient, PromptNode
from typing import Dict, Optional

TEMPLATES_DIR = Path(__file__).parent

DEFAULT_TASK_DESCRIPTION = "a general language task"

class Templates:
    @staticmethod
    def load_template(name: str) -> str:
        """Загрузка шаблона из файла prompts/<name>.txt"""
        path = TEMPLATES_DIR / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        return path.read_text(encoding="utf-8")
    
    @staticmethod
    def format_examples(examples: List[Example], max_count: int = None, include_expected: bool = True, truncate_input: int = 300) -> str:
        """Форматирование списка примеров в текстовый блок"""
        block = ""
        for i, example in enumerate(examples[:max_count], 1):
            input_text = example.input_text
            if truncate_input and len(input_text) > truncate_input:
                input_text = input_text[:truncate_input] + "..."
            block += f"Example {i}:\n  Input: {input_text}\n"
            if include_expected:
                block += f"  Expected: {example.expected_output}\n"
            actual = example.actual_output or "None"
            if truncate_input and len(actual) > truncate_input:
                actual = actual[:truncate_input] + "..."
            block += f"  Actual: {actual}\n\n"
        return block

    @staticmethod
    def format_error_string(failure_examples: List[Example]) -> str:
        """Форматирование ошибок"""
        error_string = ""
        for idx, ex in enumerate(failure_examples):
            error_string += f"## Example {idx + 1}\n"
            error_string += (
                f'Text: "{ex.input_text.strip()}"\n'
                f"Expected: {ex.expected_output}\n"
                f"Got: {ex.actual_output or 'None'}\n\n"
            )
        return error_string.strip()

    @staticmethod
    def build_analysis_prompt(current_prompt: str, error_string: str, num_feedbacks: int = 5, task_description: str = "") -> str:
        """Построение промпта для получения текстовых градиентов.
        
        Использует простой формат: причины ошибок обёрнуты в <START>/<END> теги.
        """
        template = Templates.load_template("analysis")
        return template.format(
            current_prompt=current_prompt,
            error_string=error_string,
            num_feedbacks=num_feedbacks,
            task_description=task_description or DEFAULT_TASK_DESCRIPTION,
        )
    
    @staticmethod
    def build_editing_prompt(current_prompt: str, gradient: TextGradient, num_variants: int, task_description: str = "") -> str:
        """Построение промпта для применения градиента.
        
        Использует формат: новые промпты обёрнуты в <START>/<END> теги.
        """
        template = Templates.load_template("editing")
        return template.format(
            current_prompt=current_prompt,
            error_str=gradient.metadata.get("error_string", ""),
            feedback_str=gradient.error_analysis,
            num_variants=num_variants,
            task_description=task_description or DEFAULT_TASK_DESCRIPTION,
        )

    @staticmethod
    def build_synonym_prompt(prompt_text: str) -> str:
        """Построение промпта для генерации synonym/paraphrase."""
        template = Templates.load_template("synonym")
        return template.format(prompt_text=prompt_text)
    
    @staticmethod
    def _bucketize_float(num: float, n_buckets: int = 100) -> int:
        num = max(0.0, min(1.0, num))
        return round(num * n_buckets)

    @staticmethod
    def build_meta_optimizer_prompt(
        history_nodes: List,
        best_node,
        exemplars: Optional[List[Example]] = None,
        task_description: str = "",
        num_score_buckets: int = 100,
    ) -> str:
        sorted_nodes = sorted(history_nodes, key=lambda n: n.selection_score())

        history_lines = []
        for node in sorted_nodes:
            score_int = Templates._bucketize_float(node.selection_score(), num_score_buckets)
            history_lines.append(
                f"text:\n{node.prompt_text}\nscore:\n{score_int}"
            )
        history_block = "\n\n".join(history_lines)

        if exemplars:
            ex_lines = ["Below are some example problems the instruction must solve."]
            for ex in exemplars:
                ex_lines.append(
                    f"\nInput:\n{ex.input_text[:300]}\n\n"
                    f"Ground truth answer:\n{ex.expected_output}"
                )
            exemplars_block = "\n".join(ex_lines) + "\n"
        else:
            exemplars_block = ""

        template = Templates.load_template("meta_optimizer")
        return template.format(
            history_block=history_block,
            exemplars_block=exemplars_block,
            task_description=task_description or DEFAULT_TASK_DESCRIPTION,
        )

