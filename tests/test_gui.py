from __future__ import annotations

import codecs
import csv
import tomllib
import subprocess
import sys
from pathlib import Path
from threading import Event, Lock

import httpx
import openai
import pytest
import shiboken6
from PySide6.QtCore import QEvent, QEventLoop, QMimeData, QPoint, QPointF, QSettings, QTimer, Qt, QUrl
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QPalette
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QLineEdit, QMessageBox, QPlainTextEdit

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
    for worker in (widget.worker, widget.file_loader, *widget.correction_workers.values()):
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
    wait_until(lambda: window.worker is None and window.file_loader is None and not window.correction_workers and not window._correction_active)
    app.processEvents()
    assert window.worker is None
    assert window.file_loader is None
    assert not window.correction_workers


def prepare_model(window) -> None:
    window.key_edit.setText("test-secret")
    window._models_loaded([ModelInfo("test-model")])
    window.model_combo.setCurrentIndex(0)


def test_mode_radio_buttons_are_exclusive_and_switch_panels(window):
    assert window.correction_radio.isChecked()
    assert not window.evaluation_radio.isChecked()
    window.evaluation_radio.click()
    assert window.evaluation_radio.isChecked()
    assert not window.correction_radio.isChecked()
    assert window.settings_panel.isHidden()
    assert window.start_button.text() == "평가 시작"
    window.evaluation_radio.click()
    assert window.evaluation_radio.isChecked()
    window.correction_radio.click()
    assert window.correction_radio.isChecked()
    assert not window.evaluation_radio.isChecked()
    assert not window.settings_panel.isHidden()
    assert window.evaluation_table.isHidden()
    assert window.start_button.text() == "교정 시작"


def test_evaluation_mode_without_api_and_numeric_sort(window, app, tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("two.srt", "failed.srt", "ten.srt", "zero.srt")]
    window._files_loaded(paths)
    window._file_state(0, "완료", str(tmp_path / "two_revised.srt"))
    correction_states = [window.table.item(row, 1).text() for row in range(4)]
    window.evaluation_radio.click()
    assert window.start_button.isEnabled()
    assert window.settings_panel.isHidden()
    assert window.wrap_check.isHidden()
    assert not window.export_button.isEnabled()
    created = []
    monkeypatch.setattr(gui, "create_spacer", lambda: created.append(1) or object())

    def evaluate(path, spacer, **kwargs):
        if path.name == "failed.srt":
            raise ValueError("잘못된 자막")
        return {
            "two.srt": gui.EvaluationResult(2, 100),
            "ten.srt": gui.EvaluationResult(10, 10000),
            "zero.srt": gui.EvaluationResult(0, 500),
        }[path.name]

    monkeypatch.setattr(gui, "evaluate_file", evaluate)
    window.start_correction()
    assert not window.correction_radio.isEnabled()
    assert not window.evaluation_radio.isEnabled()
    assert not window.export_button.isEnabled()
    finish_work(window, app)
    assert created == [1]
    table = window.evaluation_table
    assert [table.item(row, 0).text() for row in range(4)] == [
        str(Path(tmp_path.name) / name) for name in ("ten.srt", "two.srt", "zero.srt", "failed.srt")
    ]
    assert table.item(0, 0).toolTip() == str(paths[2])
    assert [table.item(row, 2).text() for row in range(4)] == ["10", "2", "0", "—"]
    assert [table.item(row, 3).text() for row in range(4)] == ["10000", "100", "500", "—"]
    assert [table.item(row, 4).text() for row in range(4)] == ["1.00", "20.00", "0.00", "—"]
    assert table.isSortingEnabled()
    assert window.export_button.isEnabled()
    table.sortItems(4, Qt.SortOrder.DescendingOrder)
    assert [table.item(row, 0).text() for row in range(4)] == [
        str(Path(tmp_path.name) / name) for name in ("two.srt", "ten.srt", "zero.srt", "failed.srt")
    ]
    assert "오류 12건" in window.status_label.text()
    assert [window.table.item(row, 1).text() for row in range(4)] == correction_states
    assert window.paths == paths
    window.start_correction()
    finish_work(window, app)
    assert table.item(0, 0).text() == str(Path(tmp_path.name) / "ten.srt")
    window.correction_radio.click()
    assert window.evaluation_table.isHidden()
    assert window.export_button.isHidden()
    assert not window.settings_panel.isHidden()
    assert window.table.item(0, 2).text() == str(Path(tmp_path.name) / "two_revised.srt")


def test_evaluation_sorts_only_when_finished(window, app, tmp_path, monkeypatch):
    paths = [tmp_path / "first.srt", tmp_path / "second.srt"]
    window._files_loaded(paths)
    window.evaluation_radio.click()
    entered, release = Event(), Event()
    monkeypatch.setattr(gui, "create_spacer", object)

    def evaluate(path, spacer, **kwargs):
        if path == paths[1]:
            entered.set()
            release.wait(5)
        return gui.EvaluationResult(10 if path == paths[1] else 2, 100)

    monkeypatch.setattr(gui, "evaluate_file", evaluate)
    window.start_correction()
    try:
        wait_until(entered.is_set)
        assert window.evaluation_table.item(0, 0).text() == str(Path(tmp_path.name) / "first.srt")
        assert not window.files_button.isEnabled()
        assert not window.evaluation_table.isSortingEnabled()
    finally:
        release.set()
        finish_work(window, app)
    assert window.evaluation_table.item(0, 0).text() == str(Path(tmp_path.name) / "second.srt")


def test_evaluation_frequency_rounding_and_empty_text(window):
    table = window.evaluation_table
    table.setRowCount(3)
    window._evaluation_result(0, "완료", gui.EvaluationResult(1, 3000))
    window._evaluation_result(1, "완료", gui.EvaluationResult(1, 3001))
    window._evaluation_result(2, "완료", gui.EvaluationResult(0, 0))
    assert [table.item(row, 4).text() for row in range(3)] == ["0.33", "0.33", "—"]
    assert table.item(2, 3).text() == "0"
    table.sortItems(4, Qt.SortOrder.DescendingOrder)
    assert [table.item(row, 3).text() for row in range(3)] == ["3000", "3001", "0"]


def prepare_csv_table(window):
    window.evaluation_radio.click()
    table = window.evaluation_table
    table.setRowCount(4)
    for row, (name, state, result) in enumerate([
        ('기린의 날개 (2012)\\한글,"자막".srt', "완료", gui.EvaluationResult(10, 10000)),
        ("빈도 높은 자막.srt", "완료", gui.EvaluationResult(2, 100)),
        ("실패.srt", "실패", None),
        ("중단.srt", "중단", None),
    ]):
        table.setItem(row, 0, gui.QTableWidgetItem(name))
        window._evaluation_result(row, state, result)
    table.sortItems(4, Qt.SortOrder.DescendingOrder)
    window._update_controls()


def test_export_csv_preserves_table_order_headers_and_values(window, tmp_path, monkeypatch):
    prepare_csv_table(window)
    path = tmp_path / "평가.csv"
    monkeypatch.setattr(gui.QFileDialog, "getSaveFileName", lambda *args: (str(path), "CSV (*.csv)"))
    window.export_button.click()
    assert path.read_bytes().startswith(codecs.BOM_UTF8)
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    assert rows == [
        ["평가 자막", "평가 상태", "띄어쓰기 오류 수", "글자 수", "1,000자당 오류 수"],
        ["빈도 높은 자막.srt", "완료", "2", "100", "20.00"],
        ['기린의 날개 (2012)\\한글,"자막".srt', "완료", "10", "10000", "1.00"],
        ["실패.srt", "실패", "—", "—", "—"],
        ["중단.srt", "중단", "—", "—", "—"],
    ]
    assert "CSV 저장 완료" in window.status_label.text()
    window.clear_files()
    assert not window.export_button.isEnabled()


def test_export_csv_cancel_keeps_status(window, monkeypatch):
    prepare_csv_table(window)
    previous = window.status_label.text()
    monkeypatch.setattr(gui.QFileDialog, "getSaveFileName", lambda *args: ("", ""))
    window.export_evaluation_csv()
    assert window.status_label.text() == previous


@pytest.mark.parametrize("failure", ["open", "write", "commit"])
def test_export_csv_failure_preserves_existing_file(window, tmp_path, monkeypatch, failure):
    prepare_csv_table(window)
    path = tmp_path / "existing.csv"
    path.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(gui.QFileDialog, "getSaveFileName", lambda *args: (str(path), "CSV (*.csv)"))
    original_save_file = gui.QSaveFile

    class FailingSaveFile(original_save_file):
        def open(self, mode):
            return False if failure == "open" else super().open(mode)

        def write(self, data):
            if failure == "write":
                super().write(data[:10])
                return 10
            return super().write(data)

        def commit(self):
            return False if failure == "commit" else super().commit()

    monkeypatch.setattr(gui, "QSaveFile", FailingSaveFile)
    window.export_evaluation_csv()
    assert path.read_text(encoding="utf-8") == "existing"
    assert "CSV 저장 실패" in window.status_label.text()
    assert window.export_button.isEnabled()


def test_evaluation_worker_cancel_and_initialization_failure(app, tmp_path, monkeypatch):
    worker = gui.EvaluationWorker([tmp_path / "one.srt", tmp_path / "two.srt"])
    results, summaries = [], []
    worker.file_result.connect(lambda *args: results.append(args))
    worker.summary.connect(summaries.append)
    monkeypatch.setattr(gui, "create_spacer", object)
    monkeypatch.setattr(gui, "evaluate_file", lambda *args, **kwargs: (_ for _ in ()).throw(gui.CorrectionCancelled()))
    worker.run()
    assert results == [(0, "평가 중", None), (0, "중단", None)]
    assert "미처리 2개" in summaries[-1]
    results.clear()
    monkeypatch.setattr(gui, "create_spacer", lambda: (_ for _ in ()).throw(RuntimeError("초기화 실패")))
    worker.run()
    assert results == [(0, "실패", None), (1, "실패", None)]
    assert "실패 2개" in summaries[-1]


def test_version_matches_package_and_titles(window) -> None:
    project_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with project_path.open("rb") as project_file:
        project = tomllib.load(project_file)
    assert project["project"]["version"] == gui.__version__ == "1.1.5"
    expected_title = "SRT Spellchecker v1.1.5"
    assert window.windowTitle() == expected_title
    assert any(label.text() == expected_title for label in window.findChildren(QLabel))


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("end_state", ["완료", "검토 필요", "실패", "중단", "대기"])
def test_active_row_highlight_follows_file_state(window, tmp_path, dark, end_state) -> None:
    palette = window.table.palette()
    palette.setColor(QPalette.ColorRole.Base, QColor("#202020" if dark else "#ffffff"))
    window.table.setPalette(palette)
    window._files_loaded([tmp_path / "first.srt", tmp_path / "second.srt"])
    window._file_state(0, "교정 중", "")
    expected = "#214b3a" if dark else "#d9f2e7"
    for column in range(window.table.columnCount()):
        item = window.table.item(0, column)
        assert item.background().color().name() == expected
        assert item.font().bold()
    assert not window.table.item(1, 0).font().bold()
    for column in (3, 4):
        assert expected in window.table.cellWidget(0, column).styleSheet()
    window._file_state(0, end_state, "")
    for column in range(window.table.columnCount()):
        item = window.table.item(0, column)
        assert item.data(Qt.ItemDataRole.BackgroundRole) is None
        assert item.data(Qt.ItemDataRole.ForegroundRole) is None
        assert not item.font().bold()
    for column in (3, 4):
        assert window.table.cellWidget(0, column).styleSheet() == ""
    window._file_state(1, "교정 중", "")
    assert window.table.item(1, 0).background().color().name() == expected
    assert window.table.alternatingRowColors()


@pytest.mark.parametrize("active_row", [0, 1])
def test_active_row_background_is_rendered(window, app, tmp_path, active_row) -> None:
    window._files_loaded([tmp_path / "first.srt", tmp_path / "second.srt"])
    window.show()
    window._file_state(active_row, "교정 중", "")
    app.processEvents()
    viewport = window.table.viewport()
    image = viewport.grab().toImage()
    scale = image.devicePixelRatio()
    item = window.table.item(active_row, 0)
    rect = window.table.visualItemRect(item)
    rendered = image.pixelColor(int((rect.right() - 10) * scale), int((rect.bottom() - 6) * scale))
    assert rendered == item.background().color()


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


def test_worklist_restores_without_closing_original_window(window, app, tmp_path, monkeypatch) -> None:
    paths = [tmp_path / name for name in ("review.srt", "running.srt", "waiting.srt")]
    window._files_loaded(paths)
    output = str(tmp_path / "review_revised.srt")
    logs = [
        '[되돌림] 자막 #13: 원본 유지\n'
        '  원본(교정 입력): ["안녕"]\n'
        '  모델 응답: ["안녕?"]\n'
        '  추가 문장부호: "?" (U+003F) +1개'
    ]
    window._store_review(paths[0], output, logs)
    window._file_state(0, "검토 필요", output)
    window._file_state(1, "교정 중", "")
    restored = gui.MainWindow()
    try:
        assert restored.paths == paths
        assert restored.table.item(0, 1).text() == "검토 필요"
        assert restored.table.item(0, 2).toolTip() == output
        assert restored.review_results[paths[0]] == (output, logs)
        assert restored.table.cellWidget(0, 3).isEnabled()
        assert paths[0] in restored.completed_paths
        assert restored.table.item(1, 1).text() == "대기"
        assert restored.table.item(2, 1).text() == "대기"
        assert restored.worker is None
        processed = []

        def correct_file(path, *_args, **_kwargs):
            processed.append(path)
            return path.with_stem(path.stem + "_revised"), []

        monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
        monkeypatch.setattr(gui, "correct_file", correct_file)
        prepare_model(restored)
        restored.start_correction()
        finish_work(restored, app)
        assert processed == paths[1:]
        assert restored.review_results[paths[0]] == (output, logs)
        assert not restored.start_button.isEnabled()
    finally:
        restored.close()
        restored.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("remove", ["clear", "waiting", "selected"])
def test_cleared_worklist_stays_empty_after_restart(window, app, tmp_path, remove) -> None:
    window._files_loaded([tmp_path / "subtitle.srt"])
    if remove == "clear":
        window.clear_files()
    elif remove == "waiting":
        window.remove_waiting_file(window.paths[0])
    else:
        window.table.selectRow(0)
        window.remove_selected()
    restored = gui.MainWindow()
    try:
        assert restored.paths == []
        assert restored.table.rowCount() == 0
    finally:
        restored.close()
        restored.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_worklist_survives_abrupt_process_exit(app, isolated_settings, tmp_path) -> None:
    settings_path, _credentials = isolated_settings
    source = tmp_path / "interrupted.srt"
    script = """
import os
import sys
from pathlib import Path
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from app_settings import AppSettings
from srt_spellchecker_gui import MainWindow

application = QApplication([])
settings = AppSettings(QSettings(sys.argv[1], QSettings.Format.IniFormat))
settings.load_key = lambda service: None
window = MainWindow(settings)
window._files_loaded([Path(sys.argv[2])])
window._file_state(0, "교정 중", "")
os._exit(23)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(settings_path), str(source)],
        cwd=Path(gui.__file__).parent, capture_output=True, timeout=20, check=False,
    )
    assert result.returncode == 23, result.stderr
    restored = gui.MainWindow()
    try:
        assert restored.paths == [source]
        assert restored.table.item(0, 1).text() == "대기"
        assert restored.worker is None
    finally:
        restored.close()
        restored.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_worklist_save_error_is_visible(window, tmp_path, monkeypatch) -> None:
    def fail(_worklist):
        raise app_settings.SettingsError("작업 목록 저장 실패")

    monkeypatch.setattr(window.settings, "save_worklist", fail)
    source = tmp_path / "subtitle.srt"
    window._files_loaded([source])
    assert window.paths == [source]
    assert "작업 목록 저장 실패" in window.log_view.toPlainText()


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
    assert window.table.item(0, 0).text() == str(Path(tmp_path.name) / "subtitle.srt")
    assert window.table.item(0, 2).text() == str(Path(tmp_path.name) / "subtitle_revised.srt")
    assert (tmp_path / "subtitle_revised.srt").read_text(encoding="utf-8") == SAMPLE
    assert window.progress_bar.value() == 1000
    assert "저장 1개" in window.status_label.text()
    assert not window.start_button.isEnabled()
    window.start_correction()
    assert window.worker is None


@pytest.mark.parametrize("limit", [2, 3])
def test_concurrent_correction_limit_refill_and_progress(window, app, tmp_path, monkeypatch, limit):
    paths = [tmp_path / f"{index}.srt" for index in range(limit + 2)]
    entered = {path: Event() for path in paths}
    release = {path: Event() for path in paths}
    clients, closed, active = [], [], set()
    lock = Lock()
    peak = 0

    class Service(EchoService):
        def __init__(self, *args):
            super().__init__(*args)
            clients.append(self)

        def close(self):
            closed.append(self)

    def correct_file(path, corrector, *_args, **kwargs):
        nonlocal peak
        with lock:
            active.add(path)
            peak = max(peak, len(active))
        kwargs["on_log"]("테스트 배치")
        kwargs["on_progress"](1, 2)
        entered[path].set()
        try:
            assert release[path].wait(10)
            return path.with_stem(path.stem + "_revised"), [f"[검토] {path.name}"]
        finally:
            with lock:
                active.remove(path)

    monkeypatch.setattr(gui, "ServiceCorrector", Service)
    monkeypatch.setattr(gui, "correct_file", correct_file)
    window._files_loaded(paths)
    prepare_model(window)
    window.concurrency_spin.setValue(limit)
    window.start_correction()
    try:
        wait_until(lambda: all(entered[path].is_set() for path in paths[:limit]))
        wait_until(lambda: window.progress_bar.value() == int(limit * 500 / len(paths)))
        assert len(window.correction_workers) == limit
        assert not entered[paths[limit]].is_set()
        assert not window.concurrency_spin.isEnabled()
        assert not window.correction_radio.isEnabled()
        release[paths[0]].set()
        wait_until(entered[paths[limit]].is_set)
        assert any(not release[path].is_set() for path in paths[1:limit])
        assert window.table.item(0, 1).text() == "검토 필요"
        assert f"[{paths[0]}] 테스트 배치" in window.log_view.toPlainText()
    finally:
        for event in release.values():
            event.set()
        finish_work(window, app)
    assert peak == limit
    assert len(clients) == len(paths) == len(closed)
    assert len({id(client) for client in clients}) == len(paths)
    assert window.progress_bar.value() == 1000
    for row, path in enumerate(paths):
        assert window.table.item(row, 2).text() == str(Path(path.parent.name) / (path.stem + "_revised.srt"))
        assert window.review_results[path][1] == [f"[검토] {path.name}"]
    assert f"저장 {len(paths)}개" in window.status_label.text()


@pytest.mark.parametrize("stop", ["cancel", "close", "auth"])
def test_concurrent_stop_waits_for_all_running_files(window, app, tmp_path, monkeypatch, stop):
    paths = [tmp_path / f"{index}.srt" for index in range(3)]
    entered = {path: Event() for path in paths}
    release = {path: Event() for path in paths}

    def correct_file(path, *_args, **kwargs):
        entered[path].set()
        assert release[path].wait(10)
        if stop == "auth" and path == paths[0]:
            response = httpx.Response(401, request=httpx.Request("POST", "https://example.test"))
            raise openai.AuthenticationError("bad key", response=response, body=None)
        gui.check_cancelled(kwargs["is_cancelled"])
        return path.with_stem(path.stem + "_revised"), []

    monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
    monkeypatch.setattr(gui, "correct_file", correct_file)
    window._files_loaded(paths)
    prepare_model(window)
    window.concurrency_spin.setValue(2)
    window.start_correction()
    try:
        wait_until(lambda: entered[paths[0]].is_set() and entered[paths[1]].is_set())
        if stop == "auth":
            release[paths[0]].set()
            wait_until(lambda: not window._correction_active)
        elif stop == "close":
            monkeypatch.setattr(gui.QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
            window.close()
            assert window.close_pending
        else:
            window.cancel_work()
        assert window.correction_workers
        assert all(worker.isInterruptionRequested() for worker in window.correction_workers.values())
        assert not entered[paths[2]].is_set()
        assert not window.start_button.isEnabled()
    finally:
        for event in release.values():
            event.set()
        finish_work(window, app)
    assert not entered[paths[2]].is_set()
    assert window.table.item(0, 1).text() == ("실패" if stop == "auth" else "중단")
    assert window.table.item(1, 1).text() == "중단"
    assert window.table.item(2, 1).text() == "대기"


def test_concurrent_queue_add_and_remove(window, app, tmp_path, monkeypatch):
    paths = [tmp_path / f"{index}.srt" for index in range(4)]
    for path in paths:
        path.write_text(SAMPLE, encoding="utf-8")
    entered = {path: Event() for path in paths}
    release = Event()

    def correct_file(path, *_args, **kwargs):
        entered[path].set()
        assert release.wait(10)
        return path.with_stem(path.stem + "_revised"), []

    monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
    monkeypatch.setattr(gui, "correct_file", correct_file)
    window._files_loaded(paths[:2])
    prepare_model(window)
    window.concurrency_spin.setValue(2)
    window.start_correction()
    try:
        wait_until(lambda: all(entered[path].is_set() for path in paths[:2]))
        window.add_paths(paths[2:])
        wait_until(lambda: window.file_loader is None)
        window.remove_waiting_file(paths[1])
        assert paths[1] in window.paths
        window.remove_waiting_file(paths[2])
        assert paths[2] not in window.paths
    finally:
        release.set()
        finish_work(window, app)
    assert not entered[paths[2]].is_set()
    assert entered[paths[3]].is_set()
    assert window.table.item(2, 2).text() == str(Path(tmp_path.name) / "3_revised.srt")
    assert window.progress_bar.value() == 1000


def test_correction_initialization_failure_marks_file_and_stops_queue(window, app, tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("연결 초기화 실패")

    monkeypatch.setattr(gui, "ServiceCorrector", fail)
    window._files_loaded([tmp_path / "first.srt", tmp_path / "second.srt"])
    prepare_model(window)
    window.start_correction()
    finish_work(window, app)
    assert window.table.item(0, 1).text() == "실패"
    assert window.table.item(1, 1).text() == "대기"
    assert "실패 1개, 미처리 1개" in window.status_label.text()


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
    assert window.table.item(0, 2).text() == str(Path(tmp_path.name) / "first_revised.srt")
    assert window.table.item(1, 1).text() == expected_state
    assert not window.start_button.isEnabled()
    window.clear_files()
    window._files_loaded([first, second])
    assert window.paths == []
    assert not window.start_button.isEnabled()


@pytest.mark.parametrize("during_correction", [False, True])
def test_review_button_shows_only_selected_file_logs(window, app, tmp_path, monkeypatch, during_correction) -> None:
    paths = [tmp_path / "first.srt", tmp_path / "second.srt"]
    started = Event()
    release = Event()
    notes = {
        paths[0]: [
            '[되돌림] 자막 #13: 문장부호 추가\n'
            '  원본(교정 입력): ["안녕"]\n'
            '  모델 응답: ["안녕?"]\n'
            '  추가 문장부호: "?" (U+003F) +1개',
            "[확인필요] test-secret",
        ],
        paths[1]: ["[누락] 자막 #22: 원본 유지"],
    }

    def correct_file(path, *_args, **_kwargs):
        if during_correction and path == paths[1]:
            started.set()
            assert release.wait(5)
        return path.with_stem(path.stem + "_revised"), notes[path]

    monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
    monkeypatch.setattr(gui, "correct_file", correct_file)
    window._files_loaded(paths)
    prepare_model(window)
    window.start_correction()
    try:
        if during_correction:
            wait_until(started.is_set)
        else:
            finish_work(window, app)
        window.log_view.clear()
        window.table.cellWidget(0, 3).click()
        dialog = window.findChild(QDialog)
        assert dialog is not None and dialog.isVisible()
        assert not dialog.isModal()
        details = dialog.findChild(QPlainTextEdit)
        assert details.isReadOnly()
        text = details.toPlainText()
        assert str(paths[0]) in text
        assert "first_revised.srt" in text
        assert notes[paths[0]][0] in text
        assert notes[paths[1]][0] not in text
        assert "test-secret" not in text
        assert "[API KEY]" in text
        assert "test-secret" not in window.settings.worklist_path.read_text(encoding="utf-8")
        assert not window.table.cellWidget(0, 4).isEnabled()
        window.remove_waiting_file(paths[0])
        assert paths[0] in window.paths
        dialog.close()
    finally:
        release.set()
    finish_work(window, app)


@pytest.mark.parametrize("remove_all", [False, True])
def test_remove_waiting_file_during_correction(window, app, tmp_path, monkeypatch, remove_all) -> None:
    started = Event()
    release = Event()
    processed = []
    paths = [tmp_path / name for name in ("first.srt", "second.srt", "third.srt")]
    for path in paths:
        path.write_text(SAMPLE, encoding="utf-8")

    def correct_file(path, *_args, **_kwargs):
        processed.append(path)
        if path == paths[0]:
            started.set()
            assert release.wait(5)
        return path.with_stem(path.stem + "_revised"), []

    monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
    monkeypatch.setattr(gui, "correct_file", correct_file)
    window._files_loaded(paths)
    prepare_model(window)
    window.start_correction()
    try:
        assert started.wait(5)
        app.processEvents()
        assert not window.table.cellWidget(0, 4).isEnabled()
        window.remove_waiting_file(paths[0])
        assert window.paths == paths
        assert window.table.cellWidget(1, 4).isEnabled()
        window.table.selectRow(1)
        window.table.cellWidget(1, 4).click()
        assert window.paths == [paths[0], paths[2]]
        assert window.table.rowCount() == 2
        if remove_all:
            window.table.cellWidget(1, 4).click()
            assert window.paths == paths[:1]
    finally:
        release.set()
    finish_work(window, app)
    expected = paths[:1] if remove_all else [paths[0], paths[2]]
    assert processed == expected
    assert all(path.read_text(encoding="utf-8") == SAMPLE for path in paths)
    assert window.table.item(len(expected) - 1, 2).text() == str(
        Path(tmp_path.name) / (expected[-1].stem + "_revised.srt")
    )
    assert f"저장 {len(expected)}개" in window.status_label.text()
    assert window.progress_bar.value() == 1000


def test_fatal_error_summary_includes_previous_completed_files(window, app, tmp_path, monkeypatch) -> None:
    paths = [tmp_path / name for name in ("first.srt", "second.srt", "third.srt")]

    def correct_file(path, *_args, **_kwargs):
        if path == paths[1]:
            response = httpx.Response(401, request=httpx.Request("POST", "https://example.test"))
            raise openai.AuthenticationError("bad key", response=response, body=None)
        return path.with_stem(path.stem + "_revised"), ["[되돌림] 자막 #13"]

    monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
    monkeypatch.setattr(gui, "correct_file", correct_file)
    window._files_loaded(paths)
    prepare_model(window)
    window.start_correction()
    finish_work(window, app)
    assert "저장 1개 (검토 1개), 실패 1개, 미처리 1개" in window.status_label.text()
    assert window.table.cellWidget(0, 3).isEnabled()
    assert not window.table.cellWidget(1, 3).isEnabled()
    assert not window.table.cellWidget(1, 4).isEnabled()
    assert window.table.cellWidget(2, 4).isEnabled()


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

    def slow_collect(sources, is_cancelled, on_progress):
        scanning.set()
        assert release_scan.wait(5)
        return collect(sources, is_cancelled, on_progress)

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
        assert not window.loading_panel.isHidden()
        assert window.progress_bar.maximum() == 1000
        release.set()
        wait_until(lambda: not window.correction_workers)
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
    assert window.loading_panel.isHidden()
    assert window.paths == paths
    assert "저장 3개" in window.status_label.text()
    assert not window.start_button.isEnabled()


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
def test_file_loading_indicator_lifecycle(window, app, tmp_path, monkeypatch, mode, outcome):
    entered, release = Event(), Event()
    path = tmp_path / "found.srt"
    (window.evaluation_radio if mode else window.correction_radio).click()
    window.progress_bar.setValue(450)

    def collect(sources, is_cancelled, on_progress):
        on_progress(1234, str(tmp_path))
        entered.set()
        assert release.wait(5)
        if outcome == "error":
            raise OSError("폴더 접근 실패")
        gui.check_cancelled(is_cancelled)
        return [path]

    monkeypatch.setattr(gui, "collect_srt_files", collect)
    window.add_paths([tmp_path])
    try:
        wait_until(lambda: entered.is_set() and "1,234" in window.loading_label.text())
        assert not window.loading_panel.isHidden()
        assert window.loading_progress.minimum() == window.loading_progress.maximum() == 0
        assert window.loading_label.toolTip() == str(tmp_path)
        assert window.progress_bar.value() == 450
        assert window.cancel_button.isEnabled()
        assert not window.start_button.isEnabled()
        if outcome == "cancel":
            window.cancel_work()
            assert "파일 탐색 중단 대기" in window.loading_label.text()
            assert "API" not in window.status_label.text()
    finally:
        release.set()
        finish_work(window, app)
    assert window.loading_panel.isHidden()
    assert not window.cancel_button.isEnabled()
    assert window.paths == ([path] if outcome == "success" else [])
    if outcome == "error":
        assert "폴더 접근 실패" in window.status_label.text()
    elif outcome == "cancel":
        assert "파일 불러오기 중단됨" in window.status_label.text()


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
    window._files_loaded([Path("layout.srt")])
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
    for column in (3, 4):
        button = window.table.cellWidget(0, column)
        assert window.table.viewport().rect().contains(button.geometry())
        assert button.width() >= 24 and button.height() >= 24


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
    window.concurrency_spin.setValue(3)
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
        assert restored.concurrency_spin.value() == 3
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
    window.concurrency_spin.setValue(2)
    saved = window.settings.load_preferences()
    assert saved.models["OpenAI"].id == "test-model"
    assert saved.wrap is True
    assert saved.max_line_length == 40
    assert saved.concurrent_files == 2


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
