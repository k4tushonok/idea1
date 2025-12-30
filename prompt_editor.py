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
        
    def apply_editor_op(self, best_node: PromptNode, op_type: OperationType, action: str, generation: int, strategy: Dict) -> PromptNode:
        node = self.apply_specific_operation(best_node.prompt_text, op_type, action, parent_node=best_node)
        node.generation = generation
        node.metadata["global_strategy"] = strategy
        return node 
    
    def apply_specialize_strategy(self, strategy: Dict, analysis: Dict, generation: int) -> Optional[PromptNode]:
        """Специализация промпта"""
        best_node: PromptNode = analysis["best_elements"]["prompts"][0]
        return self.apply_editor_op(best_node, OperationType.ADD_CONSTRAINT, strategy["action"], generation, strategy)
    
    def apply_expand_strategy(self, strategy: Dict, analysis: Dict, generation: int) -> Optional[PromptNode]:
        """Расширение промпта"""
        best_node: PromptNode = analysis["best_elements"]["prompts"][0]
        return self.apply_editor_op(best_node, OperationType.ADD_INSTRUCTION, strategy["action"], generation, strategy)

    def apply_restructure_strategy(self, strategy: Dict, analysis: Dict, generation: int) -> Optional[PromptNode]:
        """Реструктуризация промпта"""
        best_node: PromptNode = analysis["best_elements"]["prompts"][0]
        return self.apply_editor_op(best_node, OperationType.RESTRUCTURE, strategy["action"], generation, strategy)

    def apply_generic_strategy(self, strategy: Dict, analysis: Dict, generation: int) -> Optional[PromptNode]:
        """Общий подход для неизвестных типов стратегий"""
        best_node: PromptNode = analysis["best_elements"]["prompts"][0]
        return self.apply_editor_op(best_node, OperationType.MODIFY_INSTRUCTION, strategy["action"], generation, strategy)
    
    def apply_simplify_strategy(self, strategy: Dict, analysis: Dict, generation: int) -> Optional[PromptNode]:
        """Упрощение промпта"""
        best_node: PromptNode = analysis["best_elements"]["prompts"][0]
        simplify_prompt = Templates.build_simplify_prompt(best_node.prompt_text, strategy['action'])
        try:
            simplified_text = self.llm.invoke(prompt=simplify_prompt)
            operation = EditOperation(operation_type=OperationType.REPHRASE, description=f"SIMPLIFY: {strategy['description']}")
            node = PromptNode(
                prompt_text=simplified_text,
                generation=generation,
                source=OptimizationSource.GLOBAL,
                operations=[operation],
                metadata={"global_strategy": strategy}
            )
            return node
        except Exception as e:
            print(f"    Error in simplify: {e}")
            return None
        
    def apply_diversify_strategy(self, strategy: Dict, analysis: Dict, generation: int) -> Optional[PromptNode]:
        """Создание разнообразного промпта"""
        diversify_prompt = Templates.build_diversify_prompt(analysis["best_elements"]["prompts"], strategy['action'])
        try:
            new_prompt_text = self.llm.invoke(prompt=diversify_prompt)
            operation = EditOperation(operation_type=OperationType.RESTRUCTURE, description=f"DIVERSIFY: {strategy['description']}")
            node = PromptNode(
                prompt_text=new_prompt_text,
                generation=generation,
                source=OptimizationSource.GLOBAL,
                operations=[operation],
                metadata={"global_strategy": strategy}
            )
            return node
        except Exception as e:
            print(f"    Error in diversify: {e}")
            return None
        
    def apply_combine_strategy(self, strategy: Dict, analysis: Dict, generation: int) -> Optional[PromptNode]:
        """Комбинирование лучших промптов"""
        combined_node = self.combine_prompts(analysis["best_elements"]["prompts"][:3], combination_strategy="best_elements")
        combined_node.generation = generation
        combined_node.source = OptimizationSource.GLOBAL
        combined_node.metadata["global_strategy"] = strategy
        return combined_node

