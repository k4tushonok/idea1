from typing import List, Dict, Optional
from llm.llm_client import create_llm
from llm.llm_response_parser import MarkdownParser
from llm.llm_response_parser import VariantParser
from prompts.templates import Templates
from data_structures import TextGradient, EditOperation, OperationType, PromptNode, OptimizationSource, OptimizationConfig

class PromptEditor:
    def __init__(self, config: OptimizationConfig, api_config: Optional[Dict[str, str]] = None):
        self.config = config
        self.api_config = api_config or {}
        self.llm = create_llm(self.config, self.api_config)
    
    def generate_variants(self, current_prompt: str, gradient: TextGradient, parent_node: Optional[PromptNode] = None) -> List[PromptNode]:
        """Генерация вариантов промпта на основе текстового градиента"""
        editing_prompt = Templates.build_editing_prompt(current_prompt, gradient, self.config.local_candidates_per_iteration)
        try:
            response_text = self.llm.invoke(prompt=editing_prompt)
            variants = VariantParser.parse_variants(response_text, current_prompt, gradient, parent_node)
            return variants
        except Exception as e:
            print(f"Error generating variants: {e}")
            # Возвращаем хотя бы один вариант с базовыми изменениями
            additions = "\n".join(f"- {s}" for s in gradient.specific_suggestions)
            new_prompt = f"{current_prompt}\n\nAdditional guidance:\n{additions}"

            operation = EditOperation(
                operation_type=OperationType.ADD_INSTRUCTION,
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
    
    def apply_specific_operation(self, current_prompt: str, operation_type: OperationType, content: str, parent_node: Optional[PromptNode] = None) -> PromptNode:
        """Применение конкретной операции редактирования к промпту"""
        prompt = Templates.build_specific_prompt(operation_type, current_prompt, content)

        try:
            new_prompt = MarkdownParser.strip_code_fences(self.llm.invoke(prompt=prompt))
        except Exception:
            new_prompt = current_prompt

        operation = EditOperation(
            operation_type=operation_type,
            description=content,
            before_snippet=current_prompt + "...",
            after_snippet=new_prompt + "..."
        )
        
        generation = parent_node.generation + 1 if parent_node else 1

        return PromptNode(
            prompt_text=new_prompt,
            parent_id=parent_node.id if parent_node else None,
            generation=generation,
            source=OptimizationSource.GLOBAL,
            operations=[operation]
        )
        
    def combine_prompts(self, prompts: List[str], combination_strategy: str = "best_elements") -> PromptNode:
        """Комбинирование нескольких успешных промптов"""
        if len(prompts) < 2:
            raise ValueError("Need at least 2 prompts to combine")
        
        combining_prompt = Templates.build_combine_prompt(prompts, combination_strategy)
        
        try:
            combined_prompt = MarkdownParser.strip_code_fences(self.llm.invoke(prompt=combining_prompt))
        except Exception:
            combined_prompt = prompts[0]

        operation = EditOperation(
            operation_type=OperationType.RESTRUCTURE,
            description=f"Combined {len(prompts)} prompts ({combination_strategy})",
            before_snippet=f"{len(prompts)} source prompts",
            after_snippet=combined_prompt + "..."
        )

        return PromptNode(
            prompt_text=combined_prompt,
            source=OptimizationSource.GLOBAL,
            operations=[operation]
        )