"""
OpenAI SDK surface tests — catch import/attribute errors before they reach production.

Verifies that every attribute and method we call on the openai SDK exists in the
installed version. Runs offline (no API key required).
"""

import inspect
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent import PROVIDERS, _build_client, _call_llm

# ── PROVIDERS config ──────────────────────────────────────────────────────────


class TestProviders:
    def test_groq_present(self):
        assert "Groq" in PROVIDERS

    def test_cerebras_present(self):
        assert "Cerebras" in PROVIDERS

    def test_each_provider_has_required_keys(self):
        for name, cfg in PROVIDERS.items():
            assert "base_url" in cfg, f"{name} missing base_url"
            assert "api_key_env" in cfg, f"{name} missing api_key_env"
            assert "model" in cfg, f"{name} missing model"

    def test_groq_model(self):
        assert PROVIDERS["Groq"]["model"] == "llama-3.3-70b-versatile"

    def test_cerebras_model(self):
        assert PROVIDERS["Cerebras"]["model"] == "llama3.3-70b"


# ── openai SDK attributes we use in agent.py ─────────────────────────────────


class TestOpenAIAttributes:
    def test_openai_client_importable(self):
        assert hasattr(openai, "OpenAI")

    def test_rate_limit_error_importable(self):
        assert hasattr(openai, "RateLimitError")

    def test_api_status_error_importable(self):
        assert hasattr(openai, "APIStatusError")

    def test_openai_accepts_base_url_and_api_key(self):
        sig = inspect.signature(openai.OpenAI.__init__)
        assert "base_url" in sig.parameters
        assert "api_key" in sig.parameters

    def test_chat_completions_create_exists(self):
        client = openai.OpenAI(base_url="https://example.com", api_key="dummy")
        assert callable(client.chat.completions.create)

    def test_create_signature_has_required_params(self):
        client = openai.OpenAI(base_url="https://example.com", api_key="dummy")
        sig = inspect.signature(client.chat.completions.create)
        for param in ("model", "messages", "temperature"):
            assert param in sig.parameters, f"create() missing param: {param}"


# ── _build_client ─────────────────────────────────────────────────────────────


class TestBuildClient:
    def test_groq_returns_client_and_model(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        client, model = _build_client("Groq")
        assert isinstance(client, openai.OpenAI)
        assert model == "llama-3.3-70b-versatile"

    def test_cerebras_returns_client_and_model(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
        client, model = _build_client("Cerebras")
        assert isinstance(client, openai.OpenAI)
        assert model == "llama3.3-70b"

    def test_unknown_provider_raises(self):
        with pytest.raises(KeyError):
            _build_client("UnknownProvider")


# ── error classes ─────────────────────────────────────────────────────────────


class TestErrorClasses:
    def test_rate_limit_error_is_exception(self):
        assert issubclass(openai.RateLimitError, Exception)

    def test_api_status_error_has_status_code(self):
        assert issubclass(openai.APIStatusError, Exception)

    def test_httpx_timeout_importable(self):
        assert hasattr(httpx, "TimeoutException")
        assert hasattr(httpx, "ReadTimeout")
        assert hasattr(httpx, "ConnectTimeout")

    def test_read_timeout_is_subclass_of_timeout_exception(self):
        assert issubclass(httpx.ReadTimeout, httpx.TimeoutException)


# ── _call_llm ─────────────────────────────────────────────────────────────────


class TestCallLlm:
    def _mock_client(self, side_effect):
        client = MagicMock()
        client.chat.completions.create.side_effect = side_effect
        return client

    def test_rate_limit_retries_and_raises(self):
        client = self._mock_client(openai.RateLimitError("rate limited", response=MagicMock(status_code=429), body={}))
        with patch("agent.time.sleep"), pytest.raises(openai.RateLimitError):
            _call_llm(client, "model", [], max_retries=3)
        assert client.chat.completions.create.call_count == 3

    def test_httpx_timeout_retries_and_raises(self):
        client = self._mock_client(httpx.ReadTimeout("timed out"))
        on_retry_calls = []
        with patch("agent.time.sleep"), pytest.raises(httpx.ReadTimeout):
            _call_llm(
                client,
                "model",
                [],
                max_retries=3,
                on_retry=lambda: on_retry_calls.append(1),
            )
        assert client.chat.completions.create.call_count == 3
        assert len(on_retry_calls) == 2

    def test_success_on_second_attempt(self):
        fake_response = MagicMock()
        fake_response.choices[0].message.content = "```python\nresult = 'hello'\n```"
        fake_response.usage.prompt_tokens = 10
        fake_response.usage.completion_tokens = 5
        client = self._mock_client(
            [
                httpx.ReadTimeout("timed out"),
                fake_response,
            ]
        )
        with patch("agent.time.sleep"):
            result = _call_llm(client, "model", [], max_retries=3)
        assert result == "```python\nresult = 'hello'\n```"

    def test_503_retries(self):
        err = openai.APIStatusError(
            "service unavailable",
            response=MagicMock(status_code=503),
            body={},
        )
        client = self._mock_client(err)
        with patch("agent.time.sleep"), pytest.raises(openai.APIStatusError):
            _call_llm(client, "model", [], max_retries=2)
        assert client.chat.completions.create.call_count == 2

    def test_400_does_not_retry(self):
        err = openai.APIStatusError(
            "bad request",
            response=MagicMock(status_code=400),
            body={},
        )
        client = self._mock_client(err)
        with pytest.raises(openai.APIStatusError):
            _call_llm(client, "model", [], max_retries=3)
        assert client.chat.completions.create.call_count == 1
