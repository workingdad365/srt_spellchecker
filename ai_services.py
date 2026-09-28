from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import openai

from srt_spellchecker import CorrectionBatch

BASE_URLS = {
    "OpenAI": "https://api.openai.com/v1",
    "OpenRouter": "https://openrouter.ai/api/v1",
}
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def create_client(service: str, api_key: str) -> openai.OpenAI:
    if service not in BASE_URLS:
        raise ValueError(f"지원하지 않는 서비스: {service}")
    if not api_key.strip():
        raise ValueError("API 키를 입력하세요.")
    return openai.OpenAI(
        api_key=api_key.strip(),
        base_url=BASE_URLS[service],
        timeout=180,
        max_retries=0,
        default_headers={"X-Title": "SRT Spellchecker"} if service == "OpenRouter" else {},
    )


def fetch_models(service: str, api_key: str) -> list[ModelInfo]:
    with create_client(service, api_key) as client:
        models = client.with_options(timeout=30).models.list()
        result = {}
        for model in models:
            metadata = model.model_dump()
            architecture = metadata.get("architecture") or {}
            if service == "OpenRouter" and "text" not in architecture.get("output_modalities", ["text"]):
                continue
            result[model.id] = ModelInfo(model.id, metadata.get("name") or model.id, metadata)
    return sorted(result.values(), key=lambda model: model.id.casefold())


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
    if service == "OpenAI":
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
    def __init__(self, service: str, api_key: str, model: ModelInfo) -> None:
        self.client = create_client(service, api_key)
        self.service = service
        self.model = model

    def close(self) -> None:
        self.client.close()

    def invoke(self, messages: list[tuple[str, str]]) -> dict[str, Any]:
        schema = correction_schema()
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
            if policy:
                request["extra_body"] = {"reasoning": policy}
        response = self.client.chat.completions.create(**request)
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