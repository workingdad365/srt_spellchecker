from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import anthropic
import openai

from srt_spellchecker import CorrectionBatch, check_cancelled

BASE_URLS = {
    "OpenAI": "https://api.openai.com/v1",
    "OpenRouter": "https://openrouter.ai/api/v1",
    "Anthropic": "https://api.anthropic.com",
}
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderInfo:
    id: str
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def openrouter_base_model_id(model_id: str) -> str:
    model_id = model_id.strip()
    base_id, separator, variant = model_id.rpartition(":")
    if separator and variant in {"nitro", "floor"}:
        return base_id
    return model_id


def _client_options(service: str, api_key: str) -> dict[str, Any]:
    if service not in BASE_URLS:
        raise ValueError(f"지원하지 않는 서비스: {service}")
    if not api_key.strip():
        raise ValueError("API 키를 입력하세요.")
    return dict(
        api_key=api_key.strip(),
        base_url=BASE_URLS[service],
        timeout=180,
        max_retries=0,
        default_headers={"X-Title": "SRT Spellchecker"} if service == "OpenRouter" else {},
    )


def create_client(service: str, api_key: str) -> openai.OpenAI | anthropic.Anthropic:
    if service == "Anthropic":
        return anthropic.Anthropic(**_client_options(service, api_key))
    return openai.OpenAI(**_client_options(service, api_key))


def create_async_client(service: str, api_key: str) -> openai.AsyncOpenAI | anthropic.AsyncAnthropic:
    if service == "Anthropic":
        return anthropic.AsyncAnthropic(**_client_options(service, api_key))
    return openai.AsyncOpenAI(**_client_options(service, api_key))


def fetch_models(service: str, api_key: str) -> list[ModelInfo]:
    with create_client(service, api_key) as client:
        models = client.with_options(timeout=30).models.list()
        result = {}
        for model in models:
            metadata = model.model_dump()
            architecture = metadata.get("architecture") or {}
            if service == "OpenRouter" and "text" not in architecture.get("output_modalities", ["text"]):
                continue
            result[model.id] = ModelInfo(
                model.id, metadata.get("name") or metadata.get("display_name") or model.id, metadata,
            )
    return sorted(result.values(), key=lambda model: model.id.casefold())


def fetch_providers(service: str, api_key: str, model: ModelInfo) -> list[ProviderInfo]:
    if service in {"OpenAI", "Anthropic"}:
        return []
    if service != "OpenRouter":
        raise ValueError(f"지원하지 않는 서비스: {service}")
    model_id = openrouter_base_model_id(model.id)
    author, separator, slug = model_id.partition("/")
    if not separator or not author or not slug or author in {".", ".."} or slug in {".", ".."}:
        raise ValueError("OpenRouter 모델 ID는 작성자/모델 형식이어야 합니다.")
    path = f"/models/{quote(author, safe='')}/{quote(slug, safe='')}/endpoints"
    with create_client(service, api_key) as client:
        payload = client.with_options(timeout=30).get(path, cast_to=dict)
    data = payload.get("data") if isinstance(payload, dict) else None
    endpoints = data.get("endpoints") if isinstance(data, dict) else None
    if not isinstance(endpoints, list):
        raise ValueError("OpenRouter 응답에 프로바이더 목록이 없습니다.")
    result = {}
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        provider_id = endpoint.get("tag")
        if not isinstance(provider_id, str) or not provider_id.strip():
            continue
        provider_id = provider_id.strip()
        name = endpoint.get("provider_name")
        if not isinstance(name, str) or not name.strip():
            name = provider_id
        result.setdefault(provider_id, ProviderInfo(provider_id, name.strip(), endpoint))
    return sorted(result.values(), key=lambda provider: (provider.name.casefold(), provider.id.casefold()))


def minimum_reasoning(model: ModelInfo) -> dict[str, Any]:
    reasoning = model.metadata.get("reasoning")
    if not isinstance(reasoning, dict):
        return {}
    if reasoning.get("mandatory") is False:
        return {"enabled": False, "exclude": True}
    supported = reasoning.get("supported_efforts", [])
    if supported is None:
        supported = EFFORTS
    allowed = [effort for effort in EFFORTS[1:] if effort in supported]
    if allowed:
        return {"effort": allowed[0], "exclude": True}
    return {"exclude": True}


def reasoning_label(service: str, model: ModelInfo) -> str:
    if service in {"OpenAI", "Anthropic"}:
        return "모델 기본값"
    policy = minimum_reasoning(model)
    if policy.get("enabled") is False:
        return "추론 안 함"
    if "effort" in policy:
        return f"최소 추론: {policy['effort']}"
    return "추론 정보 없음 / 모델 기본값"


def correction_schema() -> dict[str, Any]:
    schema = CorrectionBatch.model_json_schema()
    schema["additionalProperties"] = False
    for definition in schema.get("$defs", {}).values():
        if definition.get("type") == "object":
            definition["additionalProperties"] = False
    return schema


class ServiceCorrector:
    def __init__(
        self, service: str, api_key: str, model: ModelInfo,
        *, is_cancelled: Callable[[], bool] | None = None, provider: str = "",
    ) -> None:
        self.client = create_async_client(service, api_key)
        self.service = service
        self.model = model
        self.provider = provider.strip()
        self.is_cancelled = is_cancelled
        self._runner = asyncio.Runner()
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._runner.run(self.client.close())
        finally:
            self._runner.close()
            self._closed = True

    async def _request(self, request: dict[str, Any]) -> Any:
        check_cancelled(self.is_cancelled)
        create = self.client.messages.create if self.service == "Anthropic" else self.client.chat.completions.create
        task = asyncio.create_task(create(**request))
        try:
            while not task.done():
                check_cancelled(self.is_cancelled)
                await asyncio.wait({task}, timeout=0.1)
            check_cancelled(self.is_cancelled)
            return task.result()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def invoke(self, messages: list[tuple[str, str]]) -> dict[str, Any]:
        schema = correction_schema()
        if self.service == "Anthropic":
            return self._invoke_anthropic(messages, schema)
        request: dict[str, Any] = {
            "model": self.model.id,
            "messages": [
                {"role": "user" if role == "human" else role, "content": text}
                for role, text in messages
            ],
        }
        parameters = self.model.metadata.get("supported_parameters") or []
        if self.service == "OpenAI" or "structured_outputs" in parameters:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "CorrectionBatch", "strict": True, "schema": schema},
            }
        elif "response_format" in parameters:
            request["response_format"] = {"type": "json_object"}
        request["messages"][0]["content"] += (
            "\nReturn only a JSON object matching this schema: " + json.dumps(schema)
        )
        if self.service == "OpenRouter":
            policy = minimum_reasoning(self.model)
            extra_body = {}
            if policy:
                extra_body["reasoning"] = policy
            if self.provider:
                extra_body["provider"] = {"only": [self.provider], "allow_fallbacks": False}
            if extra_body:
                request["extra_body"] = extra_body
        response = self._runner.run(self._request(request))
        if not response.choices:
            raise ValueError("모델 응답에 교정 결과가 없습니다.")
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise ValueError(f"모델 응답이 완료되지 않았습니다: {choice.finish_reason}")
        content = choice.message.content
        if not content:
            raise ValueError("모델이 빈 응답을 반환했습니다.")
        parsed = CorrectionBatch.model_validate_json(content)
        return {"parsed": parsed, "parsing_error": None}

    def _invoke_anthropic(self, messages: list[tuple[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        system = "\n\n".join(text for role, text in messages if role == "system")
        system += "\nReturn only a JSON object matching this schema, without Markdown fences: " + json.dumps(schema)
        request = {
            "model": self.model.id,
            "max_tokens": 8192,
            "system": system,
            "messages": [
                {"role": "user" if role == "human" else role, "content": text}
                for role, text in messages if role != "system"
            ],
        }
        response = self._runner.run(self._request(request))
        if response.stop_reason != "end_turn":
            raise ValueError(f"모델 응답이 완료되지 않았습니다: {response.stop_reason}")
        content = "".join(block.text for block in response.content if block.type == "text")
        if not content.strip():
            raise ValueError("모델이 빈 응답을 반환했습니다.")
        parsed = CorrectionBatch.model_validate_json(content)
        return {"parsed": parsed, "parsing_error": None}
