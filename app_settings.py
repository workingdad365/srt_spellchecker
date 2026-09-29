from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

from keyring.backend import KeyringBackend
from pydantic import BaseModel, Field, ValidationError
from PySide6.QtCore import QIODevice, QSaveFile, QSettings, QStandardPaths

from ai_services import ModelInfo
from srt_spellchecker import DEFAULT_MAX_LINE_LENGTH


class SettingsError(Exception):
    pass


class SavedModel(BaseModel):
    id: str = Field(min_length=1)
    name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_model(cls, model: ModelInfo) -> SavedModel:
        return cls(
            id=model.id, name=model.name,
            metadata={
                key: model.metadata[key]
                for key in ("reasoning", "supported_parameters")
                if key in model.metadata
            },
        )

    def to_model(self) -> ModelInfo:
        return ModelInfo(self.id, self.name, self.metadata)


class Preferences(BaseModel):
    service: Literal["OpenAI", "OpenRouter"] = "OpenAI"
    models: dict[str, SavedModel] = Field(default_factory=dict)
    wrap: bool = False
    max_line_length: int = Field(default=DEFAULT_MAX_LINE_LENGTH, ge=1, le=200)
    concurrent_files: int = Field(default=1, ge=1, le=32)


class SavedWorkFile(BaseModel):
    path: str = Field(min_length=1)
    state: Literal["대기", "교정 중", "완료", "검토 필요", "실패", "중단"] = "대기"
    output: str = ""
    review_logs: list[str] = Field(default_factory=list)


class SavedWorklist(BaseModel):
    version: Literal[1] = 1
    files: list[SavedWorkFile] = Field(default_factory=list)


def native_credentials() -> KeyringBackend:
    if sys.platform == "win32":
        from keyring.backends.Windows import WinVaultKeyring

        return WinVaultKeyring()
    if sys.platform == "darwin":
        from keyring.backends.macOS import Keyring

        return Keyring()
    from keyring.backends.SecretService import Keyring

    return Keyring()


class AppSettings:
    def __init__(self, store: QSettings | None = None) -> None:
        self.store = store if store is not None else QSettings("drasys", "srt-spellchecker")
        self.worklist_path = (
            Path(store.fileName() + ".worklist.json") if store is not None
            else Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericDataLocation))
            / "srt-spellchecker" / "worklist.json"
        )
        self._credentials: KeyringBackend | None = None

    def load_worklist(self) -> SavedWorklist:
        try:
            return SavedWorklist.model_validate_json(self.worklist_path.read_bytes())
        except FileNotFoundError:
            return SavedWorklist()
        except (OSError, ValidationError, ValueError):
            raise SettingsError("저장된 작업 목록을 읽지 못했습니다. 작업 목록 파일을 확인하세요.") from None

    def save_worklist(self, worklist: SavedWorklist) -> None:
        try:
            self.worklist_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise SettingsError("작업 목록 저장 폴더를 만들지 못했습니다.") from None
        data = worklist.model_dump_json().encode("utf-8")
        target = QSaveFile(str(self.worklist_path))
        if not target.open(QIODevice.OpenModeFlag.WriteOnly):
            raise SettingsError("작업 목록을 저장하지 못했습니다. 저장소 접근 권한을 확인하세요.")
        if target.write(data) != len(data):
            target.cancelWriting()
            raise SettingsError("작업 목록을 모두 기록하지 못했습니다. 저장 공간을 확인하세요.")
        if not target.commit():
            raise SettingsError("작업 목록 저장을 완료하지 못했습니다. 이전 저장본을 유지합니다.")

    def load_preferences(self) -> Preferences:
        try:
            preferences = Preferences.model_validate_json(self.store.value("preferences", "{}"))
        except (ValidationError, TypeError, ValueError):
            raise SettingsError("저장된 설정을 읽을 수 없어 기본값을 사용합니다.") from None
        if self.store.status() != QSettings.Status.NoError:
            raise SettingsError("설정 저장소를 읽을 수 없어 기본값을 사용합니다.")
        return preferences

    def save_preferences(self, preferences: Preferences) -> None:
        self.store.setValue("preferences", preferences.model_dump_json())
        self.store.sync()
        if self.store.status() != QSettings.Status.NoError:
            raise SettingsError("설정을 저장하지 못했습니다. 저장소 접근 권한을 확인하세요.")

    def _keyring(self) -> KeyringBackend:
        if self._credentials is None:
            self._credentials = native_credentials()
        return self._credentials

    def load_key(self, service: str) -> str | None:
        try:
            return self._keyring().get_password(f"srt-spellchecker/{service}", "api-key")
        except Exception:
            raise SettingsError(f"{service} API 키를 자격 증명 저장소에서 읽지 못했습니다.") from None

    def save_key(self, service: str, api_key: str) -> None:
        target = f"srt-spellchecker/{service}"
        try:
            backend = self._keyring()
            if api_key:
                backend.set_password(target, "api-key", api_key)
            elif backend.get_password(target, "api-key") is not None:
                backend.delete_password(target, "api-key")
        except Exception:
            raise SettingsError(
                f"{service} API 키를 저장하지 못했습니다. 이번 실행에서만 사용합니다."
            ) from None
