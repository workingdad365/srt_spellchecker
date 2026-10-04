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
    monkeypatch.setattr(
        services,
        "create_async_client",
        lambda *args: openai.AsyncOpenAI(
            api_key="test-key", base_url="https://example.test/v1", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
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


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("openai/gpt-oss-120b:nitro", "openai/gpt-oss-120b"),
        ("openai/gpt-oss-120b:floor", "openai/gpt-oss-120b"),
        ("vendor/model:free", "vendor/model:free"),
        ("vendor/model:thinking", "vendor/model:thinking"),
        ("vendor/model:free:nitro", "vendor/model:free"),
        ("vendor/model:nitro-preview", "vendor/model:nitro-preview"),
        (" vendor/model ", "vendor/model"),
        ("vendor/model", "vendor/model"),
    ],
)
def test_openrouter_base_model_id_preserves_non_routing_variants(model_id, expected) -> None:
    assert services.openrouter_base_model_id(model_id) == expected


@pytest.mark.parametrize("suffix", ["", ":nitro", ":floor"])
def test_fetch_providers_preserves_endpoint_tags_and_metadata(monkeypatch, suffix) -> None:
    endpoints = [
        {"tag": "z-provider", "provider_name": "Z Provider"},
        {
            "tag": "deepinfra/turbo", "provider_name": "DeepInfra",
            "supported_parameters": ["response_format"], "quantization": "bf16",
        },
        {"tag": "deepinfra/bf16", "provider_name": "DeepInfra"},
        {"tag": "deepinfra/turbo", "provider_name": "DeepInfra"},
        {"tag": "fallback-name"},
        {"tag": "", "provider_name": "Missing tag"},
        {"provider_name": "Missing tag"},
        None,
    ]

    def handler(request):
        assert request.url.path == "/v1/models/openai/gpt-oss-120b/endpoints"
        return httpx.Response(200, json={"data": {"endpoints": endpoints}})

    mock_client(monkeypatch, handler)
    providers = services.fetch_providers(
        "OpenRouter", "test-key", services.ModelInfo("openai/gpt-oss-120b" + suffix),
    )
    assert [provider.id for provider in providers] == [
        "deepinfra/bf16", "deepinfra/turbo", "fallback-name", "z-provider",
    ]
    assert providers[1].metadata == endpoints[1]
    assert providers[2].name == "fallback-name"


def test_fetch_providers_encodes_model_id_in_path(monkeypatch) -> None:
    def handler(request):
        assert request.url.raw_path == b"/v1/models/vendor/model%3Afree%3Fquery%3Dvalue%23fragment/endpoints"
        assert request.url.query == b""
        return httpx.Response(200, json={"data": {"endpoints": []}})

    mock_client(monkeypatch, handler)
    assert services.fetch_providers(
        "OpenRouter", "test-key", services.ModelInfo("vendor/model:free?query=value#fragment"),
    ) == []


def test_fetch_providers_skips_openai_without_request(monkeypatch) -> None:
    def unexpected_client(*_args):
        pytest.fail("OpenAI는 프로바이더를 조회하지 않아야 합니다.")

    monkeypatch.setattr(services, "create_client", unexpected_client)
    assert services.fetch_providers("OpenAI", "", services.ModelInfo("gpt-test")) == []


@pytest.mark.parametrize("payload", [{}, {"data": None}, {"data": {"endpoints": None}}])
def test_fetch_providers_rejects_invalid_catalog(monkeypatch, payload) -> None:
    mock_client(monkeypatch, lambda _request: httpx.Response(200, json=payload))
    with pytest.raises(ValueError, match="프로바이더 목록"):
        services.fetch_providers("OpenRouter", "test-key", services.ModelInfo("vendor/model"))


@pytest.mark.parametrize("model_id", ["", "model", "/model", "vendor/", "../model", "vendor/.."])
def test_fetch_providers_rejects_invalid_model_id_without_request(monkeypatch, model_id) -> None:
    def unexpected_client(*_args):
        pytest.fail("잘못된 모델 ID로 요청을 보내면 안 됩니다.")

    monkeypatch.setattr(services, "create_client", unexpected_client)
    with pytest.raises(ValueError, match="작성자/모델"):
        services.fetch_providers("OpenRouter", "test-key", services.ModelInfo(model_id))


def test_fetch_providers_propagates_api_failure(monkeypatch) -> None:
    mock_client(monkeypatch, lambda _request: httpx.Response(
        503, json={"error": {"message": "Unavailable", "code": 503}},
    ))
    with pytest.raises(openai.APIStatusError):
        services.fetch_providers("OpenRouter", "test-key", services.ModelInfo("vendor/model"))


@pytest.mark.parametrize("reasoning", [None, {"mandatory": False}])
def test_openrouter_corrector_pins_provider_and_preserves_model_suffix(monkeypatch, reasoning) -> None:
    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-oss-120b:nitro"
        assert body["provider"] == {"only": ["deepinfra/turbo"], "allow_fallbacks": False}
        if reasoning is None:
            assert "reasoning" not in body
        else:
            assert body["reasoning"] == {"enabled": False, "exclude": True}
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "index": 0,
            "message": {"role": "assistant", "content": '{"items":[]}'},
        }]})

    mock_client(monkeypatch, handler)
    model = services.ModelInfo("openai/gpt-oss-120b:nitro", metadata={"reasoning": reasoning})
    corrector = services.ServiceCorrector("OpenRouter", "key", model, provider=" deepinfra/turbo ")
    try:
        assert corrector.invoke([("system", "Correct JSON"), ("human", "text")])["parsed"].items == []
    finally:
        corrector.close()


@pytest.mark.parametrize(
    ("service", "provider"), [("OpenAI", "deepinfra/turbo"), ("OpenRouter", "")],
)
def test_corrector_omits_provider_for_openai_or_automatic_routing(monkeypatch, service, provider) -> None:
    def handler(request):
        assert "provider" not in json.loads(request.content)
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "index": 0,
            "message": {"role": "assistant", "content": '{"items":[]}'},
        }]})

    mock_client(monkeypatch, handler)
    corrector = services.ServiceCorrector(service, "key", services.ModelInfo("model"), provider=provider)
    try:
        assert corrector.invoke([("system", "Correct JSON"), ("human", "text")])["parsed"].items == []
    finally:
        corrector.close()
