from __future__ import annotations

import pytest
from keyring.backend import KeyringBackend
from PySide6.QtCore import QSettings

import app_settings as settings
from ai_services import ModelInfo


class MemoryCredentials(KeyringBackend):
    priority = 1

    def __init__(self):
        self.values = {}

    def get_password(self, service, username):
        return self.values.get((service, username))

    def set_password(self, service, username, password):
        self.values[service, username] = password

    def delete_password(self, service, username):
        del self.values[service, username]


def test_preferences_round_trip_preserves_model_metadata(tmp_path) -> None:
    path = str(tmp_path / "settings.ini")
    store = settings.AppSettings(QSettings(path, QSettings.Format.IniFormat))
    model = ModelInfo("vendor/model", "Model", {
        "reasoning": {"mandatory": True, "supported_efforts": ["high", "low"]},
        "supported_parameters": ["structured_outputs"],
        "unused_field": "not saved",
    })
    preferences = settings.Preferences(service="OpenRouter", wrap=True, max_line_length=30)
    preferences.models["OpenRouter"] = settings.SavedModel.from_model(model)
    store.save_preferences(preferences)
    restored = settings.AppSettings(QSettings(path, QSettings.Format.IniFormat)).load_preferences()
    assert restored == preferences
    assert restored.models["OpenRouter"].to_model().metadata["reasoning"]["mandatory"] is True
    assert "unused_field" not in restored.models["OpenRouter"].metadata


def test_keys_use_separate_vault_entries_and_never_preferences(tmp_path, monkeypatch) -> None:
    credentials = MemoryCredentials()
    monkeypatch.setattr(settings, "native_credentials", lambda: credentials)
    path = tmp_path / "settings.ini"
    store = settings.AppSettings(QSettings(str(path), QSettings.Format.IniFormat))
    store.save_key("OpenAI", "test-openai-secret")
    store.save_key("OpenRouter", "test-router-secret")
    store.save_preferences(settings.Preferences())
    assert store.load_key("OpenAI") == "test-openai-secret"
    assert store.load_key("OpenRouter") == "test-router-secret"
    assert "secret" not in path.read_text(encoding="utf-8")
    store.save_key("OpenAI", "")
    store.save_key("OpenAI", "")
    assert store.load_key("OpenAI") is None
    assert store.load_key("OpenRouter") == "test-router-secret"


def test_vault_failure_does_not_leak_secret_or_fallback(tmp_path, monkeypatch) -> None:
    def fail():
        raise RuntimeError("test-secret")

    monkeypatch.setattr(settings, "native_credentials", fail)
    path = tmp_path / "settings.ini"
    store = settings.AppSettings(QSettings(str(path), QSettings.Format.IniFormat))
    with pytest.raises(settings.SettingsError) as error:
        store.save_key("OpenAI", "test-secret")
    assert "test-secret" not in str(error.value)
    assert not path.exists()
    with pytest.raises(settings.SettingsError):
        store.load_key("OpenAI")


@pytest.mark.parametrize("raw", ["not-json", '{"service":"unknown"}', '{"max_line_length":0}'])
def test_invalid_preferences_report_error(tmp_path, raw) -> None:
    store = settings.AppSettings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    store.store.setValue("preferences", raw)
    with pytest.raises(settings.SettingsError):
        store.load_preferences()


def test_preferences_write_error_is_reported(tmp_path, monkeypatch) -> None:
    store = settings.AppSettings(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr(store.store, "status", lambda: QSettings.Status.AccessError)
    with pytest.raises(settings.SettingsError, match="저장하지 못했습니다"):
        store.save_preferences(settings.Preferences())