from __future__ import annotations
from abc import ABC, abstractmethod
from openai import OpenAI
from google import genai
from google.genai import types as genai_types
from config import PROVIDER, API_KEY, MODEL, TEMPERATURE, MAX_TOKENS

class BaseLLM(ABC):
    def __init__(self):
        self.total_api_calls = 0

    def invoke(self, prompt: str) -> str:
        self.total_api_calls += 1
        return self._generate(prompt)

    @abstractmethod
    def _generate(self, prompt: str) -> str:
        pass
    
class OpenAILLM(BaseLLM):
    def __init__(self):
        super().__init__()
        self.client = OpenAI(api_key=API_KEY)
        self.model = MODEL

    def _generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        return response.choices[0].message.content

class GeminiLLM(BaseLLM):
    def __init__(self):
        super().__init__()
        self.client = genai.Client(api_key=API_KEY)
        self.model = MODEL

    def _generate(self, prompt: str) -> str:
        config = genai_types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        return response.text

def create_llm() -> BaseLLM:
    provider = PROVIDER

    if provider == "openai":
        return OpenAILLM()
    elif provider == "gemini":
        return GeminiLLM()
    else:
        raise ValueError(f"Unsupported provider: {provider}")