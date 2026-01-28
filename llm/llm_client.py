from __future__ import annotations
from abc import ABC, abstractmethod
from openai import OpenAI
from google import genai
from google.genai import types as genai_types
from config import PROVIDER, API_KEY, MODEL, TEMPERATURE, MAX_TOKENS
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseLLM(ABC):
    def __init__(self):
        self.total_api_calls = 0

    def invoke(self, prompt: str) -> str:
        self.total_api_calls += 1
        return self._generate(prompt)

    @abstractmethod
    def _generate(self, prompt: str) -> str:
        pass


class LocalQwenModel(BaseLLM):
    def __init__(self):
        super().__init__()
        self.model_name = MODEL
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="cuda"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
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
    elif provider == "local_qwen":
        return LocalQwenModel()
    else:
        raise ValueError(f"Unsupported provider: {provider}")