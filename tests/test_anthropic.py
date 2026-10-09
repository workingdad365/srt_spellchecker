from __future__ import annotations

import asyncio
import json
from threading import Event

import httpx2 as httpx
import pytest

import ai_services as services
import srt_spellchecker as sc
import srt_spellchecker_gui as gui
from test_gui import app, isolated_settings, window


def install_api(monkeypatch, handler):
    original = services._client_options

    def options(service, key):
        result = original(service, key)
        result["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return result

    monkeypatch.setattr(services, "_client_options", options)
    corrector = services.ServiceCorrector("Anthropic", " test-key ", services.ModelInfo("claude-test"))
    return corrector


def response(content='{"items":[{"id":0,"corrected_lines":["교정"]}]}', stop="end_turn"):
    return httpx.Response(200, json={
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-test",
        "content": [{"type": "text", "text": content}], "stop_reason": stop,
        "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 10},
    })


def test_models_paginate_and_preserve_display_names(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.path == "/v1/models"
        assert request.headers["x-api-key"] == "test-key"
        second = request.url.params.get("after_id") == "claude-b"
        model_id = "claude-a" if second else "claude-b"
        return httpx.Response(200, json={
            "data": [{"id": model_id, "type": "model", "display_name": model_id.upper(),
                      "created_at": "2026-01-01T00:00:00Z"}],
            "has_more": not second, "first_id": model_id, "last_id": model_id,
        })

    original = services._client_options

    def options(service, key):
        return {**original(service, key), "http_client": httpx.Client(transport=httpx.MockTransport(handler))}

    monkeypatch.setattr(services, "_client_options", options)
    models = services.fetch_models("Anthropic", " test-key ")
    assert [model.id for model in models] == ["claude-a", "claude-b"]
    assert models[0].name == "CLAUDE-A"
    assert len(requests) == 2
    assert services.fetch_providers("Anthropic", "", models[0]) == []


def test_messages_request_and_validation(monkeypatch):
    def handler(request):
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "test-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(request.content)
        assert body["system"].startswith("교정 지시")
        assert body["messages"] == [{"role": "user", "content": "원문"}]
        assert body["max_tokens"] == 8192
        assert not {"response_format", "provider", "thinking", "temperature"} & body.keys()
        return response()

    corrector = install_api(monkeypatch, handler)
    try:
        result = corrector.invoke([("system", "교정 지시"), ("human", "원문")])
        assert result["parsed"].items[0].corrected_lines == ["교정"]
    finally:
        corrector.close()
    assert corrector.client.is_closed()


@pytest.mark.parametrize(("content", "stop"), [
    ('{"items":[]}', "max_tokens"), ('{"items":[]}', "refusal"),
    ("", "end_turn"), ("not json", "end_turn"), ('{"wrong":[]}', "end_turn"),
])
def test_rejects_incomplete_or_invalid_results(monkeypatch, content, stop):
    corrector = install_api(monkeypatch, lambda _: response(content, stop))
    try:
        with pytest.raises(ValueError):
            corrector.invoke([("human", "원문")])
    finally:
        corrector.close()


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_fatal_errors_are_not_retried(monkeypatch, status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"type": "error", "error": {"type": "api_error", "message": "failure"}})

    corrector = install_api(monkeypatch, handler)
    try:
        with pytest.raises(sc.FATAL_API_ERRORS):
            sc.correct_batch_with_retry(corrector, [{"id": 0, "lines": ["원문"]}], None)
        assert len(requests) == 1
    finally:
        corrector.close()


def test_rate_limit_retries_with_server_delay(monkeypatch):
    requests = []
    waits = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={"retry-after": "2"}, json={
                "type": "error", "error": {"type": "rate_limit_error", "message": "wait"},
            })
        return response()

    monkeypatch.setattr(sc, "wait_for_retry", lambda seconds, _: waits.append(seconds))
    corrector = install_api(monkeypatch, handler)
    try:
        assert sc.correct_batch_with_retry(corrector, [{"id": 0, "lines": ["원문"]}], None)
        assert waits == [2]
        assert len(requests) == 2
    finally:
        corrector.close()


def test_inflight_cancellation_closes_request(monkeypatch):
    cancelled = Event()
    finished = Event()

    async def handler(request):
        cancelled.set()
        try:
            await asyncio.Future()
        finally:
            finished.set()

    corrector = install_api(monkeypatch, handler)
    corrector.is_cancelled = cancelled.is_set
    try:
        with pytest.raises(sc.CorrectionCancelled):
            corrector.invoke([("human", "원문")])
        assert finished.is_set()
    finally:
        corrector.close()
    assert corrector.client.is_closed()


def test_gui_service_key_and_model_restore(window, app):
    window.service_combo.setCurrentText("Anthropic")
    window.key_edit.setText("anthropic-secret")
    window._models_loaded([services.ModelInfo("claude-test", "Claude Test")])
    window.model_combo.setCurrentIndex(0)
    window.service_combo.setCurrentText("OpenAI")
    assert window.key_edit.text() == ""
    window.service_combo.setCurrentText("Anthropic")
    assert window.key_edit.text() == "anthropic-secret"
    assert window.selected_model().id == "claude-test"
    assert window.provider_combo.isHidden()
    restored = gui.MainWindow()
    try:
        assert restored.current_service == "Anthropic"
        assert restored.key_edit.text() == "anthropic-secret"
        assert restored.selected_model().id == "claude-test"
    finally:
        restored.close()
        restored.deleteLater()
