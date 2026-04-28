"""
SDK API surface tests — catch AttributeErrors before they reach production.

These tests verify that every attribute, class, and method we call on the
google-genai SDK actually exists in the installed version. They run offline
(no API key required) and should be the first tests to run in CI.
"""
import inspect

import pytest
from google import genai
from google.genai import errors as genai_errors
from google.genai import types


# ── types.* attributes we reference in agent.py ───────────────────────────────

class TestTypesAttributes:
    def test_tool_exists(self):
        assert hasattr(types, "Tool"), "types.Tool missing"

    def test_function_declaration_exists(self):
        assert hasattr(types, "FunctionDeclaration"), "types.FunctionDeclaration missing"

    def test_generate_content_config_exists(self):
        assert hasattr(types, "GenerateContentConfig"), "types.GenerateContentConfig missing"

    def test_content_exists(self):
        assert hasattr(types, "Content"), "types.Content missing"

    def test_part_exists(self):
        assert hasattr(types, "Part"), "types.Part missing"

    def test_http_options_exists(self):
        assert hasattr(types, "HttpOptions"), "types.HttpOptions missing — timeout cannot be set"

    def test_no_request_options(self):
        """Guard against re-introducing the removed types.RequestOptions."""
        assert not hasattr(types, "RequestOptions"), (
            "types.RequestOptions exists in this SDK version — update agent.py to use it"
        )


class TestHttpOptions:
    def test_timeout_field(self):
        fields = types.HttpOptions.model_fields
        assert "timeout" in fields, "HttpOptions.timeout field missing"

    def test_instantiation(self):
        opts = types.HttpOptions(timeout=120)
        assert opts.timeout == 120

    def test_timeout_accepted_by_generate_content_config(self):
        fields = types.GenerateContentConfig.model_fields
        assert "http_options" in fields, (
            "GenerateContentConfig.http_options missing — cannot pass HttpOptions for timeout"
        )

    def test_config_with_http_options(self):
        config = types.GenerateContentConfig(
            temperature=0.2,
            http_options=types.HttpOptions(timeout=120),
        )
        assert config.http_options.timeout == 120


class TestGenerateContentConfig:
    def test_system_instruction_field(self):
        assert "system_instruction" in types.GenerateContentConfig.model_fields

    def test_tools_field(self):
        assert "tools" in types.GenerateContentConfig.model_fields

    def test_temperature_field(self):
        assert "temperature" in types.GenerateContentConfig.model_fields


class TestFunctionDeclaration:
    def test_parameters_json_schema_accepted(self):
        """Verify FunctionDeclaration accepts parameters_json_schema (not input_schema)."""
        fd = types.FunctionDeclaration(
            name="test_fn",
            description="A test function",
            parameters_json_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
        )
        assert fd.name == "test_fn"


class TestPart:
    def test_from_text_requires_keyword(self):
        """from_text must be called with text= keyword (positional arg removed in SDK)."""
        part = types.Part.from_text(text="hello")
        assert part.text == "hello"

    def test_from_text_no_positional(self):
        sig = inspect.signature(types.Part.from_text)
        params = list(sig.parameters.values())
        # All params should be keyword-only or have no positional-only marker that
        # would let a bare string pass — calling with positional raises TypeError.
        with pytest.raises(TypeError):
            types.Part.from_text("hello")  # type: ignore[call-arg]

    def test_from_function_response_exists(self):
        assert callable(types.Part.from_function_response)

    def test_from_function_response_works(self):
        part = types.Part.from_function_response(
            name="my_tool",
            response={"result": "ok"},
        )
        assert part.function_response.name == "my_tool"


class TestContent:
    def test_instantiation_with_role_and_parts(self):
        content = types.Content(
            role="user",
            parts=[types.Part.from_text(text="hi")],
        )
        assert content.role == "user"
        assert len(content.parts) == 1


# ── genai.Client ──────────────────────────────────────────────────────────────

class TestGenaiClient:
    def test_client_importable(self):
        assert hasattr(genai, "Client")

    def test_client_accepts_api_key(self):
        sig = inspect.signature(genai.Client.__init__)
        assert "api_key" in sig.parameters

    def test_models_generate_content_exists(self):
        # Instantiate with a dummy key — no network call made here
        client = genai.Client(api_key="dummy")
        assert callable(client.models.generate_content)

    def test_generate_content_signature(self):
        client = genai.Client(api_key="dummy")
        sig = inspect.signature(client.models.generate_content)
        assert "model" in sig.parameters
        assert "contents" in sig.parameters
        assert "config" in sig.parameters


# ── error classes ─────────────────────────────────────────────────────────────

class TestErrorClasses:
    def test_client_error_exists(self):
        assert hasattr(genai_errors, "ClientError")

    def test_server_error_exists(self):
        assert hasattr(genai_errors, "ServerError")

    def test_client_error_has_code(self):
        e = genai_errors.ClientError("test", {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "test"})
        assert hasattr(e, "code")

    def test_server_error_has_code(self):
        e = genai_errors.ServerError("test", {"code": 503, "status": "UNAVAILABLE", "message": "test"})
        assert hasattr(e, "code")
