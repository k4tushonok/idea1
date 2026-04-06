"""
Prompt templates for the optimization system.

Provides methods for building analysis prompts (gradients),
editing prompts, synonym generation, and meta-optimization.
Templates are loaded from .txt files in the prompts/ directory.
"""

from pathlib import Path
from typing import List
from data_structures import Example, TextGradient, PromptNode
from typing import Dict, Optional

TEMPLATES_DIR = Path(__file__).parent

DEFAULT_TASK_DESCRIPTION = "a general language task"


class Templates:
    """Static helper class for building prompts from templates."""

    @staticmethod
    def load_template(name: str) -> str:
        """Load a template from prompts/<name>.txt."""
        path = TEMPLATES_DIR / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        return path.read_text(encoding="utf-8")

    @staticmethod
    def format_examples(
        examples: List[Example],
        max_count: int = None,
        include_expected: bool = True,
        truncate_input: int = 300,
    ) -> str:
        """Format a list of examples as a text block with optional truncation."""
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
        """Format failure examples as a numbered text block."""
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
    def build_analysis_prompt(
        current_prompt: str,
        error_string: str,
        num_feedbacks: int = 5,
        task_description: str = "",
    ) -> str:
        """Build an error-analysis prompt to obtain textual gradients."""
        template = Templates.load_template("analysis")
        return template.format(
            current_prompt=current_prompt,
            error_string=error_string,
            num_feedbacks=num_feedbacks,
            task_description=task_description or DEFAULT_TASK_DESCRIPTION,
        )

    @staticmethod
    def build_editing_prompt(
        current_prompt: str,
        gradient: TextGradient,
        num_variants: int,
        task_description: str = "",
    ) -> str:
        """Build a gradient-application prompt (new prompts returned in <START>/<END> tags)."""
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
        """Build a prompt for generating a paraphrase of the given prompt."""
        template = Templates.load_template("synonym")
        return template.format(prompt_text=prompt_text)

    @staticmethod
    def _bucketize_float(num: float, n_buckets: int = 100) -> int:
        """Quantise a float value in [0, 1] to an integer bucket."""
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
        """Build the meta-optimizer prompt for global optimization.

        Includes node history with quantised scores
        and optional QA exemplars.
        """
        sorted_nodes = sorted(history_nodes, key=lambda n: n.selection_score())

        history_lines = []
        for node in sorted_nodes:
            score_int = Templates._bucketize_float(
                node.selection_score(), num_score_buckets
            )
            history_lines.append(f"text:\n{node.prompt_text}\nscore:\n{score_int}")
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
