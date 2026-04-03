from typing import List, Dict, Optional, Tuple
from prompts.templates import Templates
import random
from llm.llm_client import BaseLLM
from llm.llm_response_parser import TaggedTextParser
from data_structures import Example, TextGradient
from diagnostics import is_enabled, prompt_id
from config import (
    N_GRADIENTS,
    ERRORS_PER_GRADIENT,
    GRADIENTS_PER_ERROR,
    DEFAULT_PRIORITY,
)


class TextGradientGenerator:

    def __init__(self, llm: BaseLLM, task_description: str = ""):
        self.llm = llm
        self._cache: Dict[str, str] = {}
        # Рефлексия из предыдущего поколения
        self.reflection_context: str = ""
        self.task_description: str = task_description

    @staticmethod
    def _sample_error_str(
        failure_examples: List[Example],
        n: int = 4,
    ) -> Tuple[str, List[Example]]:
        """Семплирование n случайных ошибок и форматирование в строку"""
        sample_idxs = random.sample(
            range(len(failure_examples)),
            min(len(failure_examples), n),
        )
        sampled = [failure_examples[i] for i in sample_idxs]
        error_string = Templates.format_error_string(sampled)
        return error_string, sampled

    def _get_gradients(
        self,
        current_prompt: str,
        error_string: str,
        num_feedbacks: int = 5,
    ) -> List[str]:
        """Получение текстовых градиентов (причин ошибок) от LLM.

        Формат ответа: причины обёрнуты в ``<START>`` / ``<END>`` теги
        """
        gradient_prompt = Templates.build_analysis_prompt(
            current_prompt=current_prompt,
            error_string=error_string,
            num_feedbacks=num_feedbacks,
            task_description=self.task_description,
        )

        if is_enabled():
            print(
                f"[diag] _get_gradients: prompt_id={prompt_id(current_prompt)} "
                f"num_feedbacks={num_feedbacks}"
            )

        try:
            if gradient_prompt in self._cache:
                response = self._cache[gradient_prompt]
            else:
                response = self.llm.invoke(prompt=gradient_prompt)
                self._cache[gradient_prompt] = response

            feedbacks = TaggedTextParser.parse_tagged_text(response, "<START>", "<END>")
            if is_enabled():
                print(f"[diag] _get_gradients parsed {len(feedbacks)} feedbacks")
            return feedbacks
        except Exception as e:
            print(f"Error getting gradients: {e}")
            return []

    def generate_gradients_batch(
        self,
        current_prompt: str,
        failure_examples: List[Example],
        success_examples: List[Example] = None,
    ) -> List[TextGradient]:
        """Генерация градиентов ``get_gradients``.

        Для каждой из ``N_GRADIENTS`` итераций:
          1. Случайно семплируем ``ERRORS_PER_GRADIENT`` ошибок
          2. Получаем ``GRADIENTS_PER_ERROR`` feedback-причин от LLM
          3. Оборачиваем каждый feedback в ``TextGradient``
        """
        if not failure_examples:
            return []

        gradients: List[TextGradient] = []

        for i in range(N_GRADIENTS):
            error_string, sampled_failures = self._sample_error_str(
                failure_examples, n=ERRORS_PER_GRADIENT
            )
            feedbacks = self._get_gradients(
                current_prompt,
                error_string,
                num_feedbacks=GRADIENTS_PER_ERROR,
            )

            for feedback in feedbacks:
                gradient = TextGradient(
                    failure_examples=sampled_failures,
                    success_examples=success_examples or [],
                    error_analysis=feedback,
                    suggested_direction=feedback,
                    specific_suggestions=[feedback],
                    priority=DEFAULT_PRIORITY,
                    metadata={
                        "error_string": error_string,
                        "gradient_iteration": i,
                    },
                )
                gradients.append(gradient)

            if is_enabled():
                print(
                    f"[diag] gradient iteration {i + 1}/{N_GRADIENTS}: "
                    f"{len(feedbacks)} feedbacks from {len(sampled_failures)} errors"
                )

        if is_enabled():
            print(f"[diag] total gradients generated: {len(gradients)}")

        return gradients
