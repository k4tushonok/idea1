from typing import List, Dict, Optional
from llm.llm_client import BaseLLM
from llm.llm_response_parser import VariantParser
from prompts.templates import Templates
from data_structures import TextGradient, EditOperation, PromptNode, OptimizationSource
from diagnostics import is_enabled, prompt_id, preview_text
from config import LOCAL_CANDIDATES_PER_ITERATION

class PromptEditor:
    """Редактор промптов: генерирует варианты на основе текстовых градиентов"""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self._cache: Dict[str, str] = {}

    def generate_variants(self, current_prompt: str, gradient: TextGradient, parent_node: Optional[PromptNode] = None) -> List[PromptNode]:
        """Генерация вариантов промпта на основе текстового градиента"""
        editing_prompt = Templates.build_editing_prompt(current_prompt, gradient, LOCAL_CANDIDATES_PER_ITERATION)
        if is_enabled():
            print(
                f"[diag] generate_variants: base_prompt_id={prompt_id(current_prompt)} "
                f"gradient_priority={gradient.priority:.3f} suggestions={len(gradient.specific_suggestions)}"
            )
        try:
            if editing_prompt in self._cache:
                response_text = self._cache[editing_prompt]
            else:
                response_text = self.llm.invoke(prompt=editing_prompt)
                self._cache[editing_prompt] = response_text
            variants = VariantParser.parse_variants(response_text, current_prompt, gradient, parent_node)
            if is_enabled():
                print(f"[diag] parsed variants count: {len(variants)}")
            return variants
        except Exception as e:
            print(f"Error generating variants: {e}")
            # Возвращаем хотя бы один fallback-вариант
            additions = "\n".join(f"- {s}" for s in gradient.specific_suggestions)
            new_prompt = f"{current_prompt}\n\nAdditional guidance:\n{additions}"

            operation = EditOperation(
                description="Fallback variant from gradient",
                gradient_source=gradient
            )

            return [
                PromptNode(
                    prompt_text=new_prompt,
                    parent_id=parent_node.id if parent_node else None,
                    generation=parent_node.generation + 1 if parent_node else 1,
                    source=OptimizationSource.LOCAL,
                    operations=[operation]
                )
            ]