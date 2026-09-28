from __future__ import annotations

import tomllib
from pathlib import Path
from threading import Event

import httpx
import openai
import pytest
import shiboken6
from PySide6.QtCore import QEvent, QEventLoop, QMimeData, QPoint, QPointF, QSettings, QTimer, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QMessageBox

import srt_spellchecker_gui as gui
import app_settings
from ai_services import ModelInfo
from test_srt_spellchecker import FakeCorrector, SAMPLE, echo
from test_settings import MemoryCredentials


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application
    application.processEvents()
    shiboken6.delete(application)


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    credentials = MemoryCredentials()
    path = tmp_path / "settings.ini"
    monkeypatch.setattr(app_settings, "native_credentials", lambda: credentials)
    monkeypatch.setattr(gui, "AppSettings", lambda: app_settings.AppSettings(
        QSettings(str(path), QSettings.Format.IniFormat),
    ))
    return path, credentials


@pytest.fixture
def window(app, isolated_settings):
    widget = gui.MainWindow()
    yield widget
    widget.cancel_work()
    for worker in (widget.worker, widget.file_loader):
        if worker is not None:
            worker.wait(5000)
    app.processEvents()
    widget.close()
    widget.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def wait_until(predicate) -> None:
    if not predicate():
        loop = QEventLoop()
        timer = QTimer()
        check = QTimer()
        check.timeout.connect(lambda: loop.quit() if predicate() else None)
        check.start(10)
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(5000)
        loop.exec()
        check.stop()
        timer.stop()
    assert predicate()


def finish_work(window, app) -> None:
    wait_until(lambda: window.worker is None and window.file_loader is None)
    app.processEvents()
    assert window.worker is None
    assert window.file_loader is None


def prepare_model(window) -> None:
    window.key_edit.setText("test-secret")
    window._models_loaded([ModelInfo("test-model")])
    window.model_combo.setCurrentIndex(0)


def test_version_matches_package_and_titles(window) -> None:
    project_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with project_path.open("rb") as project_file:
        project = tomllib.load(project_file)
    assert project["project"]["version"] == gui.__version__ == "1.1.2"
    expected_title = "SRT Spellchecker v1.1.2"
    assert window.windowTitle() == expected_title
    assert any(label.text() == expected_title for label in window.findChildren(QLabel))


def test_default_state_and_model_selection(window) -> None:
    assert not window.start_button.isEnabled()
    assert not window.fetch_button.isEnabled()
    assert window.key_edit.echoMode() == QLineEdit.EchoMode.Password
    prepare_model(window)
    assert window.selected_model().id == "test-model"
    window.model_combo.setEditText("unknown-model")
    assert window.selected_model() is None
    window.model_combo.setCurrentIndex(0)
    window.key_edit.setText("different-secret")
    assert window.selected_model() is None


def test_service_switch_clears_models_and_keeps_separate_keys(window) -> None:
    prepare_model(window)
    window.service_combo.setCurrentText("OpenRouter")
    assert window.model_combo.count() == 0
    assert window.key_edit.text() == ""
    window.key_edit.setText("router-secret")
    window.service_combo.setCurrentText("OpenAI")
    assert window.key_edit.text() == "test-secret"
    window.service_combo.setCurrentText("OpenRouter")
    assert window.key_edit.text() == "router-secret"


def test_models_load_in_background(window, app, monkeypatch) -> None:
    window.service_combo.setCurrentText("OpenRouter")
    window.key_edit.setText("router-secret")
    model = ModelInfo("vendor/model", metadata={
        "reasoning": {"mandatory": True, "supported_efforts": ["high", "minimal"]},
    })
    monkeypatch.setattr(gui, "fetch_models", lambda service, key: [model])
    window.fetch_button.click()
    assert not window.settings_panel.isEnabled()
    finish_work(window, app)
    assert window.model_combo.count() == 1
    assert window.selected_model() is None
    window.model_combo.setCurrentIndex(0)
    assert "minimal" in window.reasoning_note.text()


def test_models_error_recovers_controls_without_leaking_key(window, app, monkeypatch) -> None:
    def fail(service, key):
        raise ValueError(f"bad request: {key}")

    monkeypatch.setattr(gui, "fetch_models", fail)
    window.key_edit.setText("test-secret")
    window.load_models()
    finish_work(window, app)
    assert window.settings_panel.isEnabled()
    assert window.model_combo.count() == 0
    assert "test-secret" not in window.log_view.toPlainText()
    assert "[API KEY]" in window.log_view.toPlainText()


def test_drop_folder_and_files_recursively(window, app, tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    source = nested / "한글 자막.SRT"
    source.write_text(SAMPLE, encoding="utf-8")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(tmp_path)), QUrl.fromLocalFile(str(source))])
    enter = QDragEnterEvent(
        QPoint(10, 10), Qt.DropAction.CopyAction, mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    drop = QDropEvent(
        QPointF(10, 10), Qt.DropAction.CopyAction, mime,
        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(window.table.viewport(), enter)
    assert enter.isAccepted()
    QApplication.sendEvent(window.table.viewport(), drop)
    assert drop.isAccepted()
    finish_work(window, app)
    assert window.paths == [source]
    assert window.table.rowCount() == 1
    window.table.selectRow(0)
    window.remove_button.click()
    assert window.paths == []


class EchoService(FakeCorrector):
    def __init__(self, *args):
        super().__init__(echo)

    def close(self):
        pass


def test_gui_correction_flow(window, app, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
    source = tmp_path / "subtitle.srt"
    source.write_text(SAMPLE, encoding="utf-8")
    window._files_loaded([source])
    prepare_model(window)
    assert window.start_button.isEnabled()
    window.start_button.click()
    assert not window.start_button.isEnabled()
    finish_work(window, app)
    assert window.table.item(0, 1).text() == "완료"
    assert window.table.item(0, 2).text() == "subtitle_revised.srt"
    assert (tmp_path / "subtitle_revised.srt").read_text(encoding="utf-8") == SAMPLE
    assert window.progress_bar.value() == 1000
    assert "저장 1개" in window.status_label.text()
    assert not window.start_button.isEnabled()
    window.start_correction()
    assert window.worker is None


@pytest.mark.parametrize("logs", [[], ["[확인필요] 검토 대상"]])
def test_new_files_do_not_repeat_completed_files(window, app, tmp_path, monkeypatch, logs) -> None:
    processed = []

    def correct_file(path, *_args, **_kwargs):
        processed.append(path)
        return path.with_stem(path.stem + "_revised"), logs

    monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
    monkeypatch.setattr(gui, "correct_file", correct_file)
    first = tmp_path / "first.srt"
    second = tmp_path / "second.srt"
    window._files_loaded([first])
    prepare_model(window)
    window.start_correction()
    finish_work(window, app)
    window._files_loaded([first, second])
    assert window.start_button.isEnabled()
    window.start_correction()
    finish_work(window, app)
    assert processed == [first, second]
    expected_state = "검토 필요" if logs else "완료"
    assert window.table.item(0, 1).text() == expected_state
    assert window.table.item(0, 2).text() == "first_revised.srt"
    assert window.table.item(1, 1).text() == expected_state
    assert not window.start_button.isEnabled()
    window.clear_files()
    window._files_loaded([first, second])
    assert window.paths == []
    assert not window.start_button.isEnabled()


@pytest.mark.parametrize("stop", [None, "cancel", "auth"])
def test_drop_during_correction_appends_to_queue(window, app, tmp_path, monkeypatch, stop) -> None:
    started = Event()
    release = Event()
    processed = []
    paths = [tmp_path / name for name in ("first.srt", "second.srt", "third.srt")]
    for path in paths:
        path.write_text(SAMPLE, encoding="utf-8")

    def correct_file(path, *_args, **kwargs):
        processed.append(path)
        if path == paths[0]:
            started.set()
            assert release.wait(5)
            if stop == "auth":
                response = httpx.Response(401, request=httpx.Request("POST", "https://example.test"))
                raise openai.AuthenticationError("bad key", response=response, body=None)
        gui.check_cancelled(kwargs["is_cancelled"])
        return path.with_stem(path.stem + "_revised"), []

    monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
    monkeypatch.setattr(gui, "correct_file", correct_file)
    window._files_loaded(paths[:2])
    prepare_model(window)
    window.start_correction()
    try:
        assert started.wait(5)
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(paths[2])), QUrl.fromLocalFile(str(paths[0]))])
        enter = QDragEnterEvent(
            QPoint(10, 10), Qt.DropAction.CopyAction, mime,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        )
        drop = QDropEvent(
            QPointF(10, 10), Qt.DropAction.CopyAction, mime,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(window.table.viewport(), enter)
        assert enter.isAccepted()
        QApplication.sendEvent(window.table.viewport(), drop)
        assert drop.isAccepted()
        assert not window.start_button.isEnabled()
        assert not window.remove_button.isEnabled()
        assert not window.clear_button.isEnabled()
        wait_until(lambda: window.file_loader is None)
        assert window.paths == paths
        assert window.table.item(2, 1).text() == "대기"
        if stop == "cancel":
            window.cancel_work()
    finally:
        release.set()
    finish_work(window, app)
    assert window.paths == paths
    if stop is None:
        assert processed == paths
        assert all(window.table.item(row, 1).text() == "완료" for row in range(3))
        assert window.progress_bar.value() == 1000
        assert "저장 3개" in window.status_label.text()
        assert not window.start_button.isEnabled()
    else:
        assert processed == paths[:1]
        assert window.table.item(0, 1).text() == ("중단" if stop == "cancel" else "실패")
        assert all(window.table.item(row, 1).text() == "대기" for row in (1, 2))
        assert window.start_button.isEnabled()


def test_queue_waits_for_slow_file_discovery(window, app, tmp_path, monkeypatch) -> None:
    started = Event()
    release = Event()
    scanning = Event()
    release_scan = Event()
    processed = []
    paths = [tmp_path / name for name in ("first.srt", "second.srt", "third.srt")]
    for path in paths:
        path.write_text(SAMPLE, encoding="utf-8")
    collect = gui.collect_srt_files

    def correct_file(path, *_args, **_kwargs):
        processed.append(path)
        if path == paths[0]:
            started.set()
            assert release.wait(5)
        return path.with_stem(path.stem + "_revised"), []

    def slow_collect(sources, is_cancelled):
        scanning.set()
        assert release_scan.wait(5)
        return collect(sources, is_cancelled)

    monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
    monkeypatch.setattr(gui, "correct_file", correct_file)
    monkeypatch.setattr(gui, "collect_srt_files", slow_collect)
    window._files_loaded(paths[:1])
    prepare_model(window)
    window.start_correction()
    try:
        assert started.wait(5)
        window.add_paths(paths[1:2])
        assert scanning.wait(5)
        release.set()
        wait_until(lambda: window.worker is None)
        assert window.file_loader is not None
        assert not window.start_button.isEnabled()
        assert not window.settings_panel.isEnabled()
        assert not window.clear_button.isEnabled()
        window.add_paths(paths[2:])
    finally:
        release.set()
        release_scan.set()
    finish_work(window, app)
    assert processed == paths
    assert window.paths == paths
    assert "저장 3개" in window.status_label.text()
    assert not window.start_button.isEnabled()


def test_file_error_does_not_stop_next_file(window, app, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
    source = tmp_path / "valid.srt"
    source.write_text(SAMPLE, encoding="utf-8")
    window._files_loaded([tmp_path / "missing.srt", source])
    prepare_model(window)
    window.start_correction()
    finish_work(window, app)
    assert window.table.item(0, 1).text() == "실패"
    assert window.table.item(1, 1).text() == "완료"


def test_auth_error_stops_queue(window, app, tmp_path, monkeypatch) -> None:
    response = httpx.Response(401, request=httpx.Request("POST", "https://example.test"))
    error = openai.AuthenticationError("bad key", response=response, body=None)

    class FailedService(EchoService):
        def invoke(self, messages):
            raise error

    monkeypatch.setattr(gui, "ServiceCorrector", FailedService)
    paths = [tmp_path / "first.srt", tmp_path / "second.srt"]
    for source in paths:
        source.write_text(SAMPLE, encoding="utf-8")
    window._files_loaded(paths)
    prepare_model(window)
    window.start_correction()
    finish_work(window, app)
    assert window.table.item(0, 1).text() == "실패"
    assert window.table.item(1, 1).text() == "대기"
    assert not list(tmp_path.glob("*_revised.srt"))


def test_close_waits_for_background_task(window, app, monkeypatch) -> None:
    monkeypatch.setattr(gui, "fetch_models", lambda *args: [])
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    window.key_edit.setText("test-secret")
    window.load_models()
    window.close()
    assert window.close_pending
    finish_work(window, app)
    assert window.key_edit.text() == ""


def test_cancel_inflight_request_keeps_source_and_next_file(window, app, tmp_path, monkeypatch) -> None:
    started = Event()
    release = Event()

    class WaitingService(EchoService):
        def invoke(self, messages):
            started.set()
            assert release.wait(5)
            return super().invoke(messages)

    monkeypatch.setattr(gui, "ServiceCorrector", WaitingService)
    paths = [tmp_path / "first.srt", tmp_path / "second.srt"]
    for source in paths:
        source.write_text(SAMPLE, encoding="utf-8")
    window._files_loaded(paths)
    prepare_model(window)
    window.start_correction()
    try:
        assert started.wait(5)
        window.cancel_button.click()
        assert not window.cancel_button.isEnabled()
    finally:
        release.set()
    finish_work(window, app)
    assert window.table.item(0, 1).text() == "중단"
    assert window.table.item(1, 1).text() == "대기"
    assert not list(tmp_path.glob("*_revised.srt"))
    assert window.start_button.isEnabled()


@pytest.mark.parametrize(("width", "height"), [(720, 620), (980, 800)])
def test_layout_fits_window(window, app, width, height) -> None:
    window.resize(width, height)
    window.show()
    app.processEvents()
    widgets = [
        window.settings_panel, window.files_button, window.table, window.log_view,
        window.wrap_check, window.start_button, window.cancel_button,
        window.progress_bar, window.status_label,
    ]
    bounds = []
    for widget in widgets:
        rect = widget.rect()
        rect.moveTopLeft(widget.mapTo(window, QPoint(0, 0)))
        assert window.rect().contains(rect)
        assert rect.width() > 0 and rect.height() > 0
        bounds.append(rect)
    for index, first in enumerate(bounds):
        assert all(not first.intersects(second) for second in bounds[index + 1:])


def test_settings_restore_after_close(window, app, isolated_settings) -> None:
    prepare_model(window)
    window.service_combo.setCurrentText("OpenRouter")
    window.key_edit.setText("router-secret")
    model = ModelInfo("vendor/model", metadata={
        "reasoning": {"mandatory": True, "supported_efforts": ["high", "minimal"]},
        "supported_parameters": ["structured_outputs"],
    })
    window._models_loaded([model])
    window.model_combo.setCurrentIndex(0)
    window.wrap_check.setChecked(True)
    window.length_spin.setValue(32)
    window.show_key.setChecked(True)
    window.close()
    restored = gui.MainWindow()
    try:
        assert restored.current_service == "OpenRouter"
        assert restored.key_edit.text() == "router-secret"
        assert restored.selected_model() == model
        assert "minimal" in restored.reasoning_note.text()
        assert restored.wrap_check.isChecked()
        assert restored.length_spin.value() == 32
        assert restored.key_edit.echoMode() == QLineEdit.EchoMode.Password
        restored.service_combo.setCurrentText("OpenAI")
        assert restored.selected_model().id == "test-model"
        assert restored.key_edit.text() == "test-secret"
        assert restored.paths == []
        assert "secret" not in isolated_settings[0].read_text(encoding="utf-8")
    finally:
        restored.close()
        restored.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_preferences_save_before_close(window) -> None:
    prepare_model(window)
    window.wrap_check.setChecked(True)
    window.length_spin.setValue(40)
    saved = window.settings.load_preferences()
    assert saved.models["OpenAI"].id == "test-model"
    assert saved.wrap is True
    assert saved.max_line_length == 40


def test_key_edit_saves_and_clears_credentials(window) -> None:
    window.key_edit.setText("changed-secret")
    window.key_edit.editingFinished.emit()
    assert window.settings.load_key("OpenAI") == "changed-secret"
    window.key_edit.clear()
    window.key_edit.editingFinished.emit()
    assert window.settings.load_key("OpenAI") is None


def test_model_refresh_reselects_previous_and_updates_metadata(window) -> None:
    prepare_model(window)
    updated = ModelInfo("test-model", metadata={"supported_parameters": ["response_format"]})
    window._models_loaded([ModelInfo("another-model"), updated])
    assert window.selected_model() == updated
    assert window.settings.load_preferences().models["OpenAI"].metadata == updated.metadata
    window._models_loaded([ModelInfo("another-model")])
    assert window.selected_model() is None
    assert "OpenAI" not in window.settings.load_preferences().models


def test_key_save_failure_keeps_current_key_without_leaking(window, monkeypatch) -> None:
    def fail(*args):
        raise app_settings.SettingsError("자격 증명 저장소 오류")

    monkeypatch.setattr(window.settings, "save_key", fail)
    window.key_edit.setText("unsaved-secret")
    window.key_edit.editingFinished.emit()
    assert window.key_edit.text() == "unsaved-secret"
    assert "자격 증명 저장소 오류" in window.log_view.toPlainText()
    assert "unsaved-secret" not in window.log_view.toPlainText()


def test_key_debounce_saves_without_focus_change(window, app) -> None:
    window.key_save_timer.setInterval(0)
    window.key_edit.setText("auto-saved-secret")
    app.processEvents()
    assert window.settings.load_key("OpenAI") == "auto-saved-secret"


def test_saved_key_takes_priority_over_environment(window, app, monkeypatch) -> None:
    window.settings.save_key("OpenAI", "vault-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-environment-secret")
    restored = gui.MainWindow()
    try:
        assert restored.key_edit.text() == "vault-secret"
        restored.service_combo.setCurrentText("OpenRouter")
        assert restored.key_edit.text() == "router-environment-secret"
    finally:
        restored.close()
        restored.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_corrupt_settings_start_with_defaults_and_warning(window, app) -> None:
    window.settings.store.setValue("preferences", "invalid-json")
    window.settings.store.sync()
    restored = gui.MainWindow()
    try:
        assert restored.current_service == "OpenAI"
        assert restored.selected_model() is None
        assert not restored.wrap_check.isChecked()
        assert "기본값" in restored.log_view.toPlainText()
    finally:
        restored.close()
        restored.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)