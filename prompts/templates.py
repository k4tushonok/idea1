from pathlib import Path
from typing import List
from data_structures import Example

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
