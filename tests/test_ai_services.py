from __future__ import annotations

import json

import httpx
import openai
import pytest

import ai_services as services


@pytest.mark.parametrize(
    ("reasoning", "expected"),
    [
        (None, {}),
        ({"mandatory": False, "supported_efforts": ["high", "low"]}, {"enabled": False, "exclude": True}),
        ({"mandatory": True, "supported_efforts": ["high", "low", "minimal"]}, {"effort": "minimal", "exclude": True}),
        ({"mandatory": True, "supported_efforts": ["none", "low"]}, {"effort": "low", "exclude": True}),
        ({"mandatory": True, "supported_efforts": None}, {"effort": "minimal", "exclude": True}),
        ({"mandatory": True}, {"exclude": True}),
    ],
)
def test_minimum_reasoning(reasoning, expected) -> None:
    model = services.ModelInfo("test", metadata={"reasoning": reasoning})
    assert services.minimum_reasoning(model) == expected


def mock_client(monkeypatch, handler) -> None:
    monkeypatch.setattr(
        services,
        "create_client",
        lambda *args: openai.OpenAI(
            api_key="test-key", base_url="https://example.test/v1", max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
    )


def test_fetch_models_preserves_metadata(monkeypatch) -> None:
    metadata = {"id": "vendor/text", "name": "Text", "reasoning": {"mandatory": True}}

    def handler(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [metadata, {
            "id": "vendor/image", "architecture": {"output_modalities": ["image"]},
        }]})

    mock_client(monkeypatch, handler)
    models = services.fetch_models("OpenRouter", "test-key")
    assert len(models) == 1
    assert models[0].metadata["reasoning"] == {"mandatory": True}


@pytest.mark.parametrize("service", ["OpenAI", "OpenRouter"])
def test_corrector_request_and_response(monkeypatch, service) -> None:
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["messages"][1]["role"] == "user"
        assert "temperature" not in body
        assert "max_tokens" not in body
        if service == "OpenRouter":
            assert body["reasoning"] == {"enabled": False, "exclude": True}
            assert "response_format" not in body
        else:
            assert body["response_format"]["json_schema"]["strict"] is True
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "index": 0,
            "message": {"role": "assistant", "content": '{"items":[{"id":0,"corrected_lines":["text"]}]}'},
        }]})

    mock_client(monkeypatch, handler)
    corrector = services.ServiceCorrector(service, "key", services.ModelInfo(
        "model", metadata={"reasoning": {"mandatory": False}},
    ))
    try:
        result = corrector.invoke([("system", "Correct JSON"), ("human", "text")])
        assert result["parsed"].items[0].corrected_lines == ["text"]
    finally:
        corrector.close()


@pytest.mark.parametrize(
    ("parameters", "format_type"),
    [(["structured_outputs"], "json_schema"), (["response_format"], "json_object"), ([], None)],
)
def test_openrouter_output_format_follows_metadata(monkeypatch, parameters, format_type) -> None:
    def handler(request):
        body = json.loads(request.content)
        assert body.get("response_format", {}).get("type") == format_type
        assert body["reasoning"] == {"effort": "low", "exclude": True}
        assert "reasoning_effort" not in body
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "index": 0,
            "message": {"role": "assistant", "content": '{"items":[]}'},
        }]})

    mock_client(monkeypatch, handler)
    model = services.ModelInfo("vendor/model", metadata={
        "supported_parameters": parameters,
        "reasoning": {"mandatory": True, "supported_efforts": ["high", "low"]},
    })
    corrector = services.ServiceCorrector("OpenRouter", "key", model)
    try:
        assert corrector.invoke([("system", "Correct JSON"), ("human", "text")])["parsed"].items == []
    finally:
        corrector.close()