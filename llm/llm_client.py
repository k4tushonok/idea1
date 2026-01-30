from __future__ import annotations
from abc import ABC, abstractmethod
from openai import OpenAI
from google import genai
from google.genai import types as genai_types
from config import (
    PROVIDER,
    API_KEY,
    MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    LLM_CACHE_ENABLED,
    CACHE_MAX_SIZE,
    LLM_PERSISTENT_CACHE,
    CACHE_DB_PATH,
    CACHE_TTL_SECONDS,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import OrderedDict
from typing import Any, Tuple, Optional
import sqlite3
import os
import time
import hashlib

class BaseLLM(ABC):
    def __init__(self):
        self.total_invocations = 0
        self.total_api_calls = 0

        self._cache_enabled = bool(LLM_CACHE_ENABLED)
        self._cache_max = int(CACHE_MAX_SIZE) if CACHE_MAX_SIZE else 10000
        self._cache: "OrderedDict[Tuple[Any,...], str]" = OrderedDict()
        
        self._persistent_enabled = bool(LLM_PERSISTENT_CACHE)
        self._persistent = None
        if self._persistent_enabled:
            try:
                db_path = CACHE_DB_PATH
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                self._persistent = SQLiteCache(db_path)
            except Exception:
                self._persistent = None

    def invoke(self, prompt: str) -> str:
        self.total_invocations += 1

        key = (prompt, getattr(self, "model", None), TEMPERATURE, MAX_TOKENS)

        if self._cache_enabled:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        cache_key = None
        if self._persistent_enabled and self._persistent:
            try:
                h = hashlib.sha256()
                h.update(prompt.encode('utf-8'))
                h.update(str(getattr(self, 'model', '')).encode('utf-8'))
                h.update(str(TEMPERATURE).encode('utf-8'))
                h.update(str(MAX_TOKENS).encode('utf-8'))
                cache_key = h.hexdigest()
                cached = self._persistent.get(cache_key, ttl=int(CACHE_TTL_SECONDS))
                if cached is not None:
                    if self._cache_enabled:
                        try:
                            self._cache[key] = cached
                        except Exception:
                            pass
                    print("LLM persistent cache hit")
                    return cached
            except Exception:
                print("Persistent LLM cache retrieval failed")
                pass

        result = self._generate(prompt)
        self.total_api_calls += 1

        if self._persistent_enabled and self._persistent and cache_key is not None:
            try:
                self._persistent.set(cache_key, result)
            except Exception:
                pass

        if self._cache_enabled:
            try:
                self._cache[key] = result
                if len(self._cache) > self._cache_max:
                    self._cache.popitem(last=False)
            except Exception:
                print("LLM caching failed")
                pass

        return result
        
    @abstractmethod
    def _generate(self, prompt: str) -> str:
        pass

class SQLiteCache:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._ensure_table()

    def _ensure_table(self):
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                ts INTEGER
            )
            """
        )
        self._conn.commit()

    def get(self, key: str, ttl: Optional[int] = None) -> Optional[str]:
        try:
            cur = self._conn.cursor()
            cur.execute("SELECT value, ts FROM llm_cache WHERE key = ?", (key,))
            row = cur.fetchone()
            if not row:
                return None
            value, ts = row
            if ttl and (int(time.time()) - int(ts) > int(ttl)):
                try:
                    cur.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
                    self._conn.commit()
                except Exception:
                    print("Failed to delete expired cache entry")
                    pass
                return None
            return value
        except Exception:
            return None

    def set(self, key: str, value: str):
        try:
            ts = int(time.time())
            cur = self._conn.cursor()
            cur.execute("INSERT OR REPLACE INTO llm_cache (key, value, ts) VALUES (?, ?, ?)", (key, value, ts))
            self._conn.commit()
        except Exception:
            print("Failed to set cache entry")
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