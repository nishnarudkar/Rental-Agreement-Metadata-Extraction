"""Minimal client for a local language model served by Ollama.

Everything runs on this machine: requests go to the Ollama server on
localhost, and no document text leaves the computer.

Generation is pinned for reproducibility (temperature 0, fixed seed) and the
context window is set explicitly on every request: Ollama's default is only
4096 tokens on GPUs with less than 24 GB, which would silently cut off the
longer agreements. A guard checks the server-reported prompt size so
truncation is reported instead of happening silently.

Settings can be overridden with environment variables (or a ``.env`` file):
``LLM_MODEL``, ``LLM_NUM_CTX``, ``OLLAMA_HOST``.
"""

import os
import time
from dataclasses import dataclass, field

import ollama
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

DEFAULT_MODEL = "qwen2.5:7b"  # Qwen2.5-7B-Instruct, Q4_K_M quantization (4.7 GB)
DEFAULT_NUM_CTX = 16384


class LLMError(RuntimeError):
    """The local model could not produce a usable answer."""


class LLMOutputError(LLMError):
    """The answer was malformed or cut off; asking again may succeed."""


class LLMInputTooLongError(LLMError):
    """The prompt does not fit the context window; retrying cannot help."""


@dataclass
class LLMConfig:
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", DEFAULT_MODEL))
    num_ctx: int = field(default_factory=lambda: int(os.getenv("LLM_NUM_CTX", DEFAULT_NUM_CTX)))
    host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    temperature: float = 0.0
    seed: int = 42
    num_predict: int = 1024  # cap on generated tokens
    timeout: float = 600.0  # seconds; first call includes loading the model into VRAM
    keep_alive: str = "10m"  # keep the model loaded between calls

    def options(self) -> dict:
        return {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }


@dataclass
class LLMResult:
    parsed: BaseModel
    raw: str
    prompt_tokens: int
    output_tokens: int
    seconds: float
    load_seconds: float
    tokens_per_second: float


class LocalLLM:
    def __init__(self, config: LLMConfig | None = None, client=None):
        self.config = config or LLMConfig()
        self.client = client or ollama.Client(host=self.config.host, timeout=self.config.timeout)

    def check(self) -> None:
        """Raise LLMError with a fix-it hint if the server or model is missing."""
        try:
            installed = {m.model for m in self.client.list().models}
        except Exception as exc:
            raise LLMError(
                f"Ollama server not reachable at {self.config.host} ({type(exc).__name__}). "
                "Start the Ollama app or run `ollama serve`."
            ) from exc
        wanted = self.config.model if ":" in self.config.model else f"{self.config.model}:latest"
        if wanted not in installed:
            raise LLMError(f"Model '{self.config.model}' is not installed. Run `ollama pull {self.config.model}`.")

    def generate_json(self, system: str, user: str, schema: type[BaseModel]) -> LLMResult:
        """Ask for an answer constrained to ``schema``'s JSON schema and validate it."""
        start = time.perf_counter()
        try:
            response = self.client.chat(
                model=self.config.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                format=schema.model_json_schema(),
                options=self.config.options(),
                keep_alive=self.config.keep_alive,
            )
        except Exception as exc:
            raise LLMError(f"Ollama request failed ({type(exc).__name__}: {exc})") from exc
        seconds = time.perf_counter() - start

        prompt_tokens = response.prompt_eval_count or 0
        output_tokens = response.eval_count or 0
        if prompt_tokens + self.config.num_predict > self.config.num_ctx:
            raise LLMInputTooLongError(
                f"Prompt uses {prompt_tokens} tokens; with {self.config.num_predict} reserved for the answer "
                f"it does not fit num_ctx={self.config.num_ctx}, so the input may have been truncated."
            )
        if response.done_reason == "length":
            raise LLMOutputError(f"Answer was cut off at num_predict={self.config.num_predict} tokens.")

        raw = response.message.content
        try:
            parsed = schema.model_validate_json(raw)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'answer'}: {e['msg']}" for e in exc.errors()[:5]
            )
            raise LLMOutputError(f"Model output does not match {schema.__name__}: {problems}") from exc

        eval_seconds = (response.eval_duration or 0) / 1e9
        return LLMResult(
            parsed=parsed,
            raw=raw,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            seconds=round(seconds, 2),
            load_seconds=round((response.load_duration or 0) / 1e9, 2),
            tokens_per_second=round(output_tokens / eval_seconds, 1) if eval_seconds else 0.0,
        )
