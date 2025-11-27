try:
    import anthropic
except Exception:
    anthropic = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

from typing import Optional, Dict


def create_llm_client(config, api_config: Optional[Dict[str, str]] = None, model: Optional[str] = None) -> 'LLMClient':
    api_config = api_config or {}
    temperature = getattr(config, 'temperature', 0.7)
    max_tokens = getattr(config, 'max_tokens', 2000)
    
    return LLMClient(
        api_config=api_config,
        model=model,
        default_temperature=temperature,
        default_max_tokens=max_tokens
    )


class LLMClient:
    def __init__(self, api_config: Optional[Dict[str, str]] = None, model: Optional[str] = None, default_temperature: float = 0.7, default_max_tokens: int = 2000):
        self.api_config = api_config or {}
        self.provider = self.api_config.get("provider")
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

        self.client = None
        self.gemini_model = None
        self.total_api_calls = 0

        self._init_client()

    def _init_client(self):
        if self.provider == "anthropic":
            key = self.api_config.get("anthropic_api_key")
            if anthropic is None:
                raise RuntimeError("anthropic SDK not available. Install with: pip install anthropic")
            self.client = anthropic.Anthropic(api_key=key)
            self.model = self.model or self.api_config.get("anthropic_model", "claude-sonnet-4-20250514")
        elif self.provider == "openai":
            key = self.api_config.get("openai_api_key")
            if OpenAI is None:
                raise RuntimeError("openai SDK not available. Install with: pip install openai")
            self.client = OpenAI(api_key=key)
            self.model = self.model or self.api_config.get("openai_model", "gpt-4o")
        elif self.provider == "gemini":
            key = self.api_config.get("gemini_api_key")
            if genai is None:
                raise RuntimeError("google-generativeai SDK not available. Install with: pip install google-generativeai")
            genai.configure(api_key=key)
            model_name = self.model or self.api_config.get("gemini_model", "gemini-pro")
            self.gemini_model = genai.GenerativeModel(model_name)
            self.model = model_name
        elif self.provider is not None:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def call(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None, model: Optional[str] = None) -> str:
        if self.provider is None:
            raise ValueError("No provider configured")

        temp = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        model = model or self.model

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temp,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.content[0].text
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temp
                )
                text = response.choices[0].message.content
            elif self.provider == "gemini":
                generation_config = genai.types.GenerationConfig(
                    temperature=temp,
                    max_output_tokens=max_tokens,
                )
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                text = response.text
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            self.total_api_calls += 1
            return text
        except Exception:
            raise

    def get_statistics(self) -> Dict[str, Optional[str]]:
        return {
            "provider": self.provider,
            "model": self.model,
            "total_api_calls": self.total_api_calls
        }

    def __repr__(self):
        return f"LLMClient(provider={self.provider}, model={self.model}, calls={self.total_api_calls})"
