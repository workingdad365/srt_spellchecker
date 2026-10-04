from __future__ import annotations

from threading import Event

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QDialog, QPlainTextEdit

import srt_spellchecker_gui as gui
from ai_services import ModelInfo, ProviderInfo
from test_gui import (
    EchoService, app, finish_work, isolated_settings, prepare_model, wait_until, window,
)


def prepare_router(window, model: ModelInfo | None = None) -> ModelInfo:
    model = model or ModelInfo("openai/gpt-oss-120b", "GPT OSS", {
        "supported_parameters": ["structured_outputs", "reasoning"],
    })
    window.service_combo.setCurrentText("OpenRouter")
    window.key_edit.setText("router-secret")
    window._models_loaded([model])
    window.model_combo.setCurrentIndex(0)
    window._provider_timer.stop()
    return model


def select_provider(window, provider: ProviderInfo) -> None:
    index = next(
        index for index in range(window.provider_combo.count())
        if window.provider_combo.itemData(index) == provider
    )
    window.provider_combo.setCurrentIndex(index)


@pytest.mark.parametrize("suffix", ["nitro", "floor"])
def test_router_suffix_keeps_model_metadata_and_allows_correction(window, tmp_path, suffix):
    model = prepare_router(window)
    window._files_loaded([tmp_path / "subtitle.srt"])
    window.model_combo.setEditText(f"{model.id}:{suffix}")
    window._provider_timer.stop()

    selected = window.selected_model()
    assert selected.id == f"{model.id}:{suffix}"
    assert selected.name == model.name
    assert selected.metadata == model.metadata
    assert window.start_button.isEnabled()
    assert window.settings.load_preferences().models["OpenRouter"].id == selected.id


@pytest.mark.parametrize("text", ["missing/model:nitro", "openai/gpt-oss-120b:unknown"])
def test_router_unrecognized_model_does_not_enable_correction(window, tmp_path, text):
    prepare_router(window)
    window._files_loaded([tmp_path / "subtitle.srt"])
    window.model_combo.setEditText(text)

    assert window.selected_model() is None
    assert not window.start_button.isEnabled()


def test_openai_does_not_accept_openrouter_routing_suffix(window, tmp_path):
    prepare_model(window)
    window._files_loaded([tmp_path / "subtitle.srt"])
    window.model_combo.setEditText("test-model:nitro")

    assert window.selected_model() is None
    assert not window.start_button.isEnabled()
    assert not window.provider_combo.isVisibleTo(window.settings_panel)


def test_suffix_selection_survives_model_refresh_and_restart(window, app, monkeypatch):
    model = prepare_router(window)
    window.model_combo.setEditText(f"{model.id}:nitro")
    window._provider_timer.stop()
    updated = ModelInfo(model.id, "Updated GPT OSS", {"supported_parameters": ["response_format"]})
    window._models_loaded([updated])
    window._provider_timer.stop()

    assert window.selected_model().id == f"{model.id}:nitro"
    assert window.selected_model().metadata == updated.metadata
    window.close()
    calls = []
    monkeypatch.setattr(gui, "fetch_providers", lambda *_args: calls.append(True) or [])
    restored = gui.MainWindow()
    try:
        assert restored.selected_model().id == f"{model.id}:nitro"
        assert restored.selected_model().metadata == updated.metadata
        assert not restored._provider_timer.isActive()
        assert calls == []
    finally:
        restored.close()
        restored.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_provider_lookup_populates_searchable_choices_and_persists_selection(window, app, monkeypatch):
    model = prepare_router(window)
    providers = [
        ProviderInfo("cerebras", "Cerebras", {"supported_parameters": ["response_format"]}),
        ProviderInfo("groq", "Groq", {"supported_parameters": []}),
    ]
    calls = []

    def fetch(service, key, selected):
        calls.append((service, key, selected.id))
        return providers

    monkeypatch.setattr(gui, "fetch_providers", fetch)
    window._load_providers()
    finish_work(window, app)

    assert calls == [("OpenRouter", "router-secret", model.id)]
    assert window.provider_combo.isEditable()
    assert window.provider_combo.completer() is not None
    assert window.provider_combo.count() == 3
    assert window.provider_combo.itemData(0) is None
    assert window.selected_provider() is None
    select_provider(window, providers[0])
    assert window.selected_provider() == providers[0]
    saved = window.settings.load_preferences().openrouter_providers[model.id]
    assert saved.id == "cerebras"
    assert saved.name == "Cerebras"
    assert saved.supported_parameters == ["response_format"]
    window.provider_combo.setCurrentIndex(0)
    assert window.selected_provider() is None
    assert model.id not in window.settings.load_preferences().openrouter_providers


def test_provider_selection_is_per_base_model_and_restores_with_suffix(window, app):
    model = prepare_router(window)
    provider = ProviderInfo("groq", "Groq", {"supported_parameters": []})
    second = ModelInfo("qwen/second-model")
    window._provider_cache[model.id] = [provider]
    window._refresh_provider_options()
    select_provider(window, provider)
    window._models_loaded([model, second])
    window.model_combo.setCurrentIndex(1)
    window._provider_timer.stop()
    assert window.selected_provider() is None

    window.model_combo.setEditText(f"{model.id}:nitro")
    window._provider_timer.stop()
    assert window.selected_provider() == provider
    window.close()
    restored = gui.MainWindow()
    try:
        assert restored.selected_model().id == f"{model.id}:nitro"
        assert restored.selected_provider() == provider
        assert restored.settings.load_preferences().openrouter_providers[model.id].id == provider.id
        assert not restored._provider_timer.isActive()
    finally:
        restored.close()
        restored.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_unrecognized_provider_text_does_not_silently_use_automatic_routing(window, tmp_path):
    model = prepare_router(window)
    provider = ProviderInfo("groq", "Groq")
    window._provider_cache[model.id] = [provider]
    window._refresh_provider_options()
    window._files_loaded([tmp_path / "subtitle.srt"])
    select_provider(window, provider)
    window.provider_combo.setEditText("unknown-provider")

    assert window.selected_provider() is None
    assert not window.start_button.isEnabled()
    window.provider_combo.setCurrentIndex(0)
    assert window.start_button.isEnabled()


def test_model_fetch_also_refreshes_selected_models_providers(window, app, monkeypatch):
    model = prepare_router(window)
    provider = ProviderInfo("cerebras", "Cerebras")
    calls = []
    monkeypatch.setattr(gui, "fetch_models", lambda *_args: [model])

    def fetch(service, key, selected):
        calls.append((service, key, selected.id))
        return [provider]

    monkeypatch.setattr(gui, "fetch_providers", fetch)
    window.fetch_button.click()
    wait_until(lambda: bool(calls))
    finish_work(window, app)

    assert calls == [("OpenRouter", "router-secret", model.id)]
    assert any(window.provider_combo.itemData(index) == provider for index in range(window.provider_combo.count()))


def test_returning_from_evaluation_resumes_pending_provider_lookup(window, app, monkeypatch):
    model = prepare_router(window)
    provider = ProviderInfo("groq", "Groq")
    calls = []

    def fetch(service, key, selected):
        calls.append((service, key, selected.id))
        return [provider]

    monkeypatch.setattr(gui, "fetch_providers", fetch)
    window.model_combo.setCurrentIndex(-1)
    window.model_combo.setCurrentIndex(0)
    assert window._provider_timer.isActive()
    window.evaluation_radio.click()
    assert not window._provider_timer.isActive()
    assert calls == []

    window.correction_radio.click()
    wait_until(lambda: bool(calls))
    finish_work(window, app)

    assert calls == [("OpenRouter", "router-secret", model.id)]
    assert any(window.provider_combo.itemData(index) == provider for index in range(window.provider_combo.count()))


def test_provider_missing_from_refreshed_list_requires_explicit_choice(window, app, tmp_path, monkeypatch):
    model = prepare_router(window)
    provider = ProviderInfo("old-provider", "이전 프로바이더")
    window._provider_cache[model.id] = [provider]
    window._refresh_provider_options()
    select_provider(window, provider)
    window._files_loaded([tmp_path / "subtitle.srt"])
    assert window.start_button.isEnabled()

    window._provider_cache.pop(model.id)
    monkeypatch.setattr(gui, "fetch_providers", lambda *_args: [ProviderInfo("new-provider", "새 프로바이더")])
    window._load_providers()
    finish_work(window, app)

    assert window.selected_provider().id == provider.id
    assert window.settings.load_preferences().openrouter_providers[model.id].id == provider.id
    assert not window.start_button.isEnabled()
    window.provider_combo.setCurrentIndex(0)
    assert window.start_button.isEnabled()


def test_provider_lookup_failure_keeps_selection_and_redacts_key(window, app, monkeypatch):
    model = prepare_router(window)
    provider = ProviderInfo("groq", "Groq")
    window._provider_cache[model.id] = [provider]
    window._refresh_provider_options()
    select_provider(window, provider)
    window._provider_cache.pop(model.id)

    def fail(_service, key, _model):
        raise RuntimeError(f"프로바이더 조회 실패: {key}")

    monkeypatch.setattr(gui, "fetch_providers", fail)
    window._load_providers()
    finish_work(window, app)

    assert window.selected_provider().id == provider.id
    assert window.settings.load_preferences().openrouter_providers[model.id].id == provider.id
    assert "프로바이더 조회 실패" in window.log_view.toPlainText()
    assert "router-secret" not in window.log_view.toPlainText()


def test_provider_response_from_previous_key_does_not_replace_current_choices(window, app, monkeypatch):
    prepare_router(window)
    started, release = Event(), Event()

    def fetch(*_args):
        started.set()
        assert release.wait(10)
        return [ProviderInfo("stale-provider", "이전 키 프로바이더")]

    monkeypatch.setattr(gui, "fetch_providers", fetch)
    window._load_providers()
    try:
        wait_until(started.is_set)
        window.key_edit.setText("replacement-key")
    finally:
        release.set()
        finish_work(window, app)

    assert window.selected_model() is None
    assert window.selected_provider() is None
    assert all(
        window.provider_combo.itemData(index) is None
        for index in range(window.provider_combo.count())
    )


def test_provider_routes_correction_records_overview_and_allows_recorrection(window, app, tmp_path, monkeypatch):
    model = prepare_router(window)
    first = ProviderInfo("cerebras", "Cerebras", {"supported_parameters": ["response_format"]})
    second = ProviderInfo("groq", "Groq", {"supported_parameters": []})
    window._provider_cache[model.id] = [first, second]
    window._refresh_provider_options()
    select_provider(window, first)
    window.model_combo.setEditText(f"{model.id}:nitro")
    window._provider_timer.stop()
    calls = []

    class Service(EchoService):
        def __init__(self, service, key, selected, **kwargs):
            super().__init__(service, key, selected, **kwargs)
            calls.append((
                service, selected.id, kwargs.get("provider"),
                list(selected.metadata["supported_parameters"]),
            ))

    monkeypatch.setattr(gui, "ServiceCorrector", Service)
    monkeypatch.setattr(gui, "correct_file", lambda path, *_args, **_kwargs: (path.with_stem("revised"), []))
    source = tmp_path / "subtitle.srt"
    window._files_loaded([source])
    window.start_button.click()
    finish_work(window, app)

    assert calls == [("OpenRouter", f"{model.id}:nitro", first.id, ["response_format"])]
    assert model.metadata == {"supported_parameters": ["structured_outputs", "reasoning"]}
    assert window.correction_overviews[source].provider_id == first.id
    assert window.correction_overviews[source].provider_name == first.name
    assert window.settings.load_worklist().files[0].overview.provider_id == first.id
    assert not window.start_button.isEnabled()
    window.table.cellWidget(0, 4).click()
    dialog = window.findChild(QDialog)
    assert first.name in dialog.findChild(QPlainTextEdit).toPlainText()
    dialog.close()

    select_provider(window, second)
    assert window.start_button.isEnabled()
    select_provider(window, first)
    assert not window.start_button.isEnabled()
    select_provider(window, second)
    window.start_button.click()
    finish_work(window, app)
    assert calls[-1] == ("OpenRouter", f"{model.id}:nitro", second.id, [])
    assert model.metadata == {"supported_parameters": ["structured_outputs", "reasoning"]}
    assert window.correction_overviews[source].provider_id == second.id
    assert not window.start_button.isEnabled()


def test_switch_to_openai_hides_provider_and_does_not_send_it(window, app, tmp_path, monkeypatch):
    model = prepare_router(window)
    provider = ProviderInfo("groq", "Groq")
    window._provider_cache[model.id] = [provider]
    window._refresh_provider_options()
    select_provider(window, provider)
    window.service_combo.setCurrentText("OpenAI")
    prepare_model(window)
    assert not window.provider_combo.isVisibleTo(window.settings_panel)
    assert window.selected_provider() is None
    calls = []

    class Service(EchoService):
        def __init__(self, service, key, selected, **kwargs):
            super().__init__(service, key, selected, **kwargs)
            calls.append((service, kwargs.get("provider") or None))

    monkeypatch.setattr(gui, "ServiceCorrector", Service)
    monkeypatch.setattr(gui, "correct_file", lambda path, *_args, **_kwargs: (path.with_stem("revised"), []))
    window._files_loaded([tmp_path / "subtitle.srt"])
    window.start_button.click()
    finish_work(window, app)

    assert calls == [("OpenAI", None)]
    window.service_combo.setCurrentText("OpenRouter")
    window._provider_timer.stop()
    assert window.selected_provider() == provider
