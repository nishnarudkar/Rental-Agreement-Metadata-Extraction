"""LLM client tests with a fake Ollama client (no server or model needed)."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from src.llm import LLMConfig, LLMError, LocalLLM


class Answer(BaseModel):
    name: str
    amount: int | None


def _response(content, prompt_tokens=100, output_tokens=20, done_reason="stop"):
    return SimpleNamespace(
        message=SimpleNamespace(content=content),
        prompt_eval_count=prompt_tokens,
        eval_count=output_tokens,
        eval_duration=2_000_000_000,
        load_duration=500_000_000,
        done_reason=done_reason,
    )


class FakeClient:
    def __init__(self, response=None, error=None, models=("qwen2.5:7b",)):
        self.response, self.error, self.models, self.calls = response, error, models, []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response

    def list(self):
        if self.error:
            raise self.error
        return SimpleNamespace(models=[SimpleNamespace(model=m) for m in self.models])


def _llm(client, **config):
    return LocalLLM(LLMConfig(model="qwen2.5:7b", num_ctx=4096, num_predict=256, **config), client=client)


def test_valid_json_is_parsed_and_timed():
    client = FakeClient(_response('{"name": "A. Tenant", "amount": 5000}'))
    result = _llm(client).generate_json("system", "user", Answer)
    assert result.parsed == Answer(name="A. Tenant", amount=5000)
    assert (result.prompt_tokens, result.output_tokens, result.tokens_per_second) == (100, 20, 10.0)
    assert result.load_seconds == 0.5


def test_request_is_deterministic_schema_constrained_and_sets_context():
    client = FakeClient(_response('{"name": "x", "amount": null}'))
    _llm(client).generate_json("sys", "doc", Answer)
    call = client.calls[0]
    assert call["format"] == Answer.model_json_schema()
    assert call["options"] == {"temperature": 0.0, "seed": 42, "num_ctx": 4096, "num_predict": 256}
    assert [m["role"] for m in call["messages"]] == ["system", "user"]


def test_invalid_json_raises():
    client = FakeClient(_response('{"name": "x"'))
    with pytest.raises(LLMError, match="does not match Answer"):
        _llm(client).generate_json("s", "u", Answer)


def test_wrong_types_raise():
    client = FakeClient(_response('{"name": "x", "amount": "five thousand"}'))
    with pytest.raises(LLMError, match="does not match Answer"):
        _llm(client).generate_json("s", "u", Answer)


def test_possible_truncation_is_reported():
    client = FakeClient(_response('{"name": "x", "amount": 1}', prompt_tokens=3900))
    with pytest.raises(LLMError, match="may have been truncated"):
        _llm(client).generate_json("s", "u", Answer)


def test_cut_off_answer_is_reported():
    client = FakeClient(_response('{"name": "x", "amount": 1}', done_reason="length"))
    with pytest.raises(LLMError, match="cut off"):
        _llm(client).generate_json("s", "u", Answer)


def test_server_errors_are_wrapped():
    client = FakeClient(error=ConnectionError("refused"))
    with pytest.raises(LLMError, match="Ollama request failed"):
        _llm(client).generate_json("s", "u", Answer)


def test_check_reports_unreachable_server():
    with pytest.raises(LLMError, match="not reachable"):
        _llm(FakeClient(error=ConnectionError("refused"))).check()


def test_check_reports_missing_model():
    with pytest.raises(LLMError, match="ollama pull qwen2.5:7b"):
        _llm(FakeClient(models=("llama3:8b",))).check()


def test_check_passes_when_model_installed():
    _llm(FakeClient()).check()


def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:3b")
    monkeypatch.setenv("LLM_NUM_CTX", "8192")
    config = LLMConfig()
    assert (config.model, config.num_ctx) == ("qwen2.5:3b", 8192)
