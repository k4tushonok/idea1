from typing import List, Dict, Optional
from llm.llm_client import create_llm
from llm.llm_response_parser import LLMResponseParser
from prompts.templates import Templates
from data_structures import (
    TextGradient,
    EditOperation,
    OperationType,
    PromptNode,
    OptimizationSource,
    OptimizationConfig
)

TEMPLATE_BY_OPERATION = {
    OperationType.ADD_INSTRUCTION: "add_instruction",
    OperationType.ADD_EXAMPLE: "add_example",
    OperationType.RESTRUCTURE: "restructure",
    OperationType.CLARIFY: "clarify",
}

class PromptEditor:
    def __init__(self, config: OptimizationConfig, api_config: Optional[Dict[str, str]] = None):
        self.config = config
        self.api_config = api_config or {}
        self.llm = create_llm(self.config, self.api_config)
    
    def generate_variants(self, current_prompt: str, gradient: TextGradient, parent_node: Optional[PromptNode] = None) -> List[PromptNode]:
        """Генерация вариантов промпта на основе текстового градиента"""
        editing_prompt = self._build_editing_prompt(current_prompt, gradient)
        try:
            response_text = self.llm.invoke(prompt=editing_prompt, max_tokens=self.config.max_tokens, temperature=self.config.temperature)
            variants = LLMResponseParser.parse_variants(response_text, current_prompt, gradient, parent_node)
            return variants
        except Exception as e:
            print(f"Error generating variants: {e}")
            # Возвращаем хотя бы один вариант с базовыми изменениями
            additions = "\n".join(f"- {s}" for s in gradient.specific_suggestions[:3])
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
        """Apply a concrete edit operation to the prompt"""
        template_name = TEMPLATE_BY_OPERATION.get(operation_type, "overall")
        template = Templates.load_template(template_name)
        prompt = template.format(current_prompt=current_prompt, content=content)

        try:
            new_prompt = LLMResponseParser.strip_code_fences(self.llm.invoke(prompt=prompt))
        except Exception:
            new_prompt = current_prompt

        operation = EditOperation(
            operation_type=operation_type,
            description=content[:200],
            before_snippet=current_prompt[:200] + "...",
            after_snippet=new_prompt[:200] + "..."
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
        
        # Подготавливаем блок с промптами
        prompts_block = "\n".join(f"PROMPT {i}:\n```\n{p}\n```" for i, p in enumerate(prompts, 1))

        # Загружаем базовый шаблон
        template = Templates.load_template("combine")
        combining_prompt = template.format(prompts_block=prompts_block)
        combining_prompt += "\nGUIDELINES:\n" + Templates.combination_guidelines(combination_strategy)
        
        try:
            combined_prompt = LLMResponseParser.strip_code_fences(self.llm.invoke(prompt=combining_prompt))
        except Exception:
            combined_prompt = prompts[0]

        operation = EditOperation(
            operation_type=OperationType.RESTRUCTURE,
            description=f"Combined {len(prompts)} prompts ({combination_strategy})",
            before_snippet=f"{len(prompts)} source prompts",
            after_snippet=combined_prompt[:200] + "..."
        )

        return PromptNode(
            prompt_text=combined_prompt,
            source=OptimizationSource.GLOBAL,
            operations=[operation]
        )
    
    def _build_editing_prompt(self, current_prompt: str, gradient: TextGradient) -> str:
        """Построение промпта для LLM, который будет генерировать улучшенные варианты. Загружает шаблон из prompts/editing.txt"""
        # Подготавливаем блоки для заполнения шаблона
        suggestions = "\n".join(f"{i}. {s}" for i, s in enumerate(gradient.specific_suggestions, 1))

        failures = "\n".join(
            f"{i}. Input: {e.input_text}\n   Expected: {e.expected_output}\n   Got: {e.actual_output}"
            for i, e in enumerate(gradient.failure_examples[:3], 1)
        )
        
        # Загружаем и заполняем шаблон
        template = Templates.load_template("editing")
        return template.format(
            current_prompt=current_prompt,
            error_analysis=gradient.error_analysis,
            suggested_direction=gradient.suggested_direction,
            specific_suggestions_block=suggestions,
            failure_examples_block=failures,
            num_variants=self.config.local_candidates_per_iteration
        )