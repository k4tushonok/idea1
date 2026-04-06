"""
Prompt editor.

Applies textual gradients to generate improved prompt variants
and creates Monte Carlo paraphrases (synonym samples)
to broaden the search space.
"""

from typing import List, Dict, Optional
from llm.llm_client import BaseLLM
from llm.llm_response_parser import TaggedTextParser, MarkdownParser
from prompts.templates import Templates
from data_structures import TextGradient, EditOperation, PromptNode, OptimizationSource
from diagnostics import is_enabled, prompt_id, preview_text
from config import STEPS_PER_GRADIENT, MC_SAMPLES_PER_STEP, MIN_PROMPT_LENGTH


class PromptEditor:
    """Prompt editor: applies gradients and generates paraphrases."""

    def __init__(self, llm: BaseLLM, task_description: str = ""):
        self.llm = llm
        self._cache: Dict[str, str] = {}
        self.task_description: str = task_description

    def generate_variants(
        self,
        current_prompt: str,
        gradient: TextGradient,
        parent_node: Optional[PromptNode] = None,
    ) -> List[PromptNode]:
        """Apply a gradient to generate improved prompt variants.

        The LLM receives the current prompt + failures + feedback and
        produces STEPS_PER_GRADIENT improved prompts wrapped in <START>/<END> tags.
        """
        editing_prompt = Templates.build_editing_prompt(
            current_prompt,
            gradient,
            STEPS_PER_GRADIENT,
            task_description=self.task_description,
        )

        if is_enabled():
            print(
                f"[diag] generate_variants (apply_gradient): "
                f"base_prompt_id={prompt_id(current_prompt)} "
                f"feedback='{preview_text(gradient.error_analysis, 120)}'"
            )

        try:
            response_text = self.llm.invoke(prompt=editing_prompt)

            # Parse <START>/<END> tags → list of new prompt texts
            new_prompts = TaggedTextParser.parse_tagged_text(
                response_text, "<START>", "<END>"
            )
            if is_enabled():
                print(f"[diag] apply_gradient parsed {len(new_prompts)} prompts")

            nodes: List[PromptNode] = []
            for new_prompt_text in new_prompts:
                # Normalize: strip code-fence artifacts and meta-prefixes
                try:
                    new_prompt_text = MarkdownParser.normalize_prompt_text(
                        new_prompt_text
                    )
                except ValueError:
                    continue

                if len(new_prompt_text) < MIN_PROMPT_LENGTH:
                    continue

                operation = EditOperation(
                    description=f"apply_gradient: {gradient.error_analysis[:80]}",
                    gradient_source=gradient,
                    before_snippet=current_prompt[:200] + "...",
                    after_snippet=new_prompt_text[:200] + "...",
                )

                generation = parent_node.generation + 1 if parent_node else 1
                node = PromptNode(
                    prompt_text=new_prompt_text,
                    parent_id=parent_node.id if parent_node else None,
                    generation=generation,
                    source=OptimizationSource.LOCAL,
                    operations=[operation],
                )
                nodes.append(node)

            if is_enabled():
                print(f"[diag] generate_variants -> {len(nodes)} valid nodes")
            return nodes

        except Exception as e:
            print(f"Error in apply_gradient: {e}")
            # Fallback: append feedback as an additional instruction
            new_prompt = (
                f"{current_prompt}\n\nAdditional guidance:\n- {gradient.error_analysis}"
            )
            operation = EditOperation(
                description="Fallback variant from gradient",
                gradient_source=gradient,
            )
            return [
                PromptNode(
                    prompt_text=new_prompt,
                    parent_id=parent_node.id if parent_node else None,
                    generation=parent_node.generation + 1 if parent_node else 1,
                    source=OptimizationSource.LOCAL,
                    operations=[operation],
                )
            ]

    def generate_synonyms(
        self,
        prompt_text: str,
        n: int = None,
        parent_node: Optional[PromptNode] = None,
    ) -> List[PromptNode]:
        """Generate MC paraphrases of a prompt to broaden the search space."""
        n = n if n is not None else MC_SAMPLES_PER_STEP
        if n <= 0:
            return []

        synonym_prompt = Templates.build_synonym_prompt(prompt_text)

        if is_enabled():
            print(
                f"[diag] generate_synonyms: n={n} "
                f"prompt_id={prompt_id(prompt_text)}"
            )

        nodes: List[PromptNode] = []
        for _ in range(n):
            try:
                response = self.llm.invoke(prompt=synonym_prompt)

                new_text = response.strip()
                try:
                    new_text = MarkdownParser.normalize_prompt_text(new_text)
                except ValueError:
                    continue

                if len(new_text) < MIN_PROMPT_LENGTH:
                    continue
                # Do not return the original unchanged prompt
                if new_text.strip().lower() == prompt_text.strip().lower():
                    continue
                if any(
                    n.prompt_text.strip().lower() == new_text.strip().lower()
                    for n in nodes
                ):
                    continue

                operation = EditOperation(
                    description="MC synonym/paraphrase",
                )
                generation = parent_node.generation + 1 if parent_node else 1
                node = PromptNode(
                    prompt_text=new_text,
                    parent_id=parent_node.id if parent_node else None,
                    generation=generation,
                    source=OptimizationSource.LOCAL,
                    operations=[operation],
                )
                nodes.append(node)
            except Exception as e:
                print(f"Error generating synonym: {e}")
                continue

        if is_enabled():
            print(f"[diag] generate_synonyms -> {len(nodes)} synonyms")
        return nodes
