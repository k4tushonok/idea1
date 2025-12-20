from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from openai import OpenAI
from google import genai
from google.genai import types as genai_types
from data_structures import OptimizationConfig

class BaseLLM(ABC):
    def __init__(self, provider: str):
        self.total_api_calls = 0
        self.provider = provider

    def invoke(self, prompt: str, *, temperature: Optional[float] = None, max_tokens: Optional[int] = None, **kwargs: Any) -> str:
        self.total_api_calls += 1
        return self._generate(prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)

    @abstractmethod
    def _generate(self, prompt: str, temperature: Optional[float], max_tokens: Optional[int], **kwargs: Any) -> str:
        pass
    
class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, model: str, default_temperature: float, default_max_tokens: int):
        super().__init__(provider="openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    def _generate(self, prompt: str, temperature: Optional[float], max_tokens: Optional[int], **_) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens or self.default_max_tokens,
            temperature=temperature or self.default_temperature
        )
        return response.choices[0].message.content

class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str, model: str, default_temperature: float, default_max_tokens: int):
        super().__init__(provider="gemini") 

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    def _generate(self, prompt: str, temperature: Optional[float], max_tokens: Optional[int], **_) -> str:
        config = genai_types.GenerateContentConfig(
            temperature=temperature or self.default_temperature,
            max_output_tokens=max_tokens or self.default_max_tokens,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config
        )
        return response.text

def create_llm(config: OptimizationConfig, api_config: Optional[Dict[str, str]] = None) -> BaseLLM:
    api_config = api_config or {}
    provider = api_config.get("provider")

    temperature = getattr(config, "temperature", 0.7)
    max_tokens = getattr(config, "max_tokens", 2000)

    if provider == "openai":
        return OpenAILLM(
            api_key=api_config["api_key"],
            model=api_config.get("model", "gpt-4o"),
            default_temperature=temperature,
            default_max_tokens=max_tokens
        )
    elif provider == "gemini":
        return GeminiLLM(
            api_key=api_config["api_key"],
            model=api_config.get("model", "gemini-1.5-flash"),
            default_temperature=temperature,
            default_max_tokens=max_tokens
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")