from typing import List, Dict, Optional
from llm.llm_client import BaseLLM
from llm.llm_response_parser import MarkdownParser
from llm.llm_response_parser import VariantParser
from prompts.templates import Templates
from data_structures import TextGradient, EditOperation, PromptNode, OptimizationSource
from diagnostics import is_enabled, prompt_id, preview_text
from config import LOCAL_CANDIDATES_PER_ITERATION

class PromptEditor:
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
            # Возвращаем хотя бы один вариант с базовыми изменениями
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
    
    def apply_strategy(
        self,
        strategy: Dict,
        analysis: Dict,
        generation: int,
        variation_id: Optional[int] = None,
    ) -> Optional[PromptNode]:
        """Единственный обработчик глобальных стратегий. LLM получает лучший промпт и свободное описание изменения"""
        best_nodes = analysis["best_elements"]["prompts"]
        if not best_nodes:
            return None
        best_node = best_nodes[0]

        prompt = Templates.build_specific_prompt("overall", best_node.prompt_text, strategy["action"])
        if variation_id is not None:
            prompt = f"{prompt}\n\nVARIANT_ID: {variation_id} (do not include in output)"
        if is_enabled():
            print(
                f"[diag] apply_strategy: base_prompt_id={prompt_id(best_node.prompt_text)} "
                f"action='{preview_text(strategy.get('action', ''), 200)}'"
            )
        try:
            if prompt in self._cache:
                new_text = MarkdownParser.normalize_prompt_text(self._cache[prompt])
            else:
                resp = self.llm.invoke(prompt=prompt)
                self._cache[prompt] = resp
                new_text = MarkdownParser.normalize_prompt_text(resp)
        except Exception as e:
            print(f"    Error in apply_strategy: {e}")
            return None

        operation = EditOperation(
            description=strategy["description"],
            before_snippet=best_node.prompt_text + "...",
            after_snippet=new_text + "..."
        )
        return PromptNode(
            prompt_text=new_text,
            parent_id=best_node.id,
            generation=generation,
            source=OptimizationSource.GLOBAL,
            operations=[operation],
            metadata={"global_strategy": strategy}
        )