from __future__ import annotations

import codecs
import csv
import os
import tomllib
import subprocess
import sys
from pathlib import Path
from threading import Event, Lock

import httpx
import openai
import pytest
import shiboken6
from PySide6.QtCore import QEvent, QEventLoop, QMimeData, QPoint, QPointF, QRect, QSettings, QTimer, Qt, QUrl
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QImage, QPainter, QPalette
from PySide6.QtWidgets import (
    QApplication, QDialog, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QStyleOptionViewItem, QTableWidgetItem,
)

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


def test_partial_evaluation_status_logs_csv_and_filter(window, app, tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("partial.srt", "unscorable.srt", "normal.srt")]
    for path, body in zip(paths, ["오류\n반갑 습니다", "오류", "안녕하세요"]):
        path.write_text("1\r\n00:00:01,000 --> 00:00:02,000\n" + body + "\n", encoding="utf-8", newline="")
    originals = [path.read_bytes() for path in paths]

    class Spacer:
        def space(self, text, **_kwargs):
            return "오류!" if text == "오류" else text.replace(" ", "")

    monkeypatch.setattr(gui, "create_spacer", Spacer)
    window._files_loaded(paths)
    window.evaluation_radio.click()
    window.start_evaluation()
    finish_work(window, app)
    table = window.evaluation_table
    table.sortItems(4, Qt.SortOrder.DescendingOrder)
    assert [table.item(row, 0).toolTip() for row in range(3)] == [str(paths[index]) for index in (0, 2, 1)]
    assert table.item(0, 1).text() == "부분 평가 (1줄 제외)"
    assert "자막 #1, 본문 1줄" in table.item(0, 1).toolTip()
    assert [table.item(0, column).text() for column in (2, 3, 4)] == ["1", "5", "200.00"]
    assert table.item(1, 1).text() == "완료"
    assert table.item(2, 1).text() == "평가 불가 (1줄 제외)"
    assert [table.item(2, column).text() for column in (2, 3, 4)] == ["—", "—", "—"]
    assert "완료 2개 (부분 평가 1개), 오류 1건, 실패 1개, 미처리 0개" in window.status_label.text()
    assert window.progress_bar.value() == 1000
    assert f"[제외] {paths[0]}: 자막 #1" in window.log_view.toPlainText()
    assert 'Kiwi 결과: "오류!"' in window.log_view.toPlainText()
    csv_path = tmp_path / "partial.csv"
    monkeypatch.setattr(gui.QFileDialog, "getSaveFileName", lambda *args: (str(csv_path), "CSV (*.csv)"))
    window.export_button.click()
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[1][1:] == ["부분 평가 (1줄 제외)", "1", "5", "200.00", ""]
    assert rows[3][1:] == ["평가 불가 (1줄 제외)", "—", "—", "—", ""]
    window.evaluation_threshold_spin.setValue(100)
    window.filter_evaluation_button.click()
    assert window.paths == [paths[0]]
    assert table.rowCount() == 1
    window.correction_radio.click()
    assert window.paths == [paths[0]]
    assert [path.read_bytes() for path in paths] == originals


@pytest.mark.parametrize(("marker", "timecode", "review_location", "reason"), [
    ("31", "00:00:01,000 --> 00:00:02,000", "자막 #31", '파일 시작이 "1\\r\\n" 또는 "1\\n"이 아님'),
    ("krCc", "00:00:01,000 --> 00:00:02,000", "자막 #31, 본문 1줄", "문자열 발견"),
    ("eGcC", "00:00:01,000 --> 00:00:02,000", "자막 #31, 본문 1줄", "문자열 발견"),
    ("&nbsp;", "00:00:01,000 --> 00:00:02,000", "자막 #31, 본문 1줄", "문자열 발견"),
    ("01:20:09,818", "01:20:09,818 --> 01:20:09,818", "자막 #31", "종료시간이 시작시간보다 같거나 빠름"),
    ("01:20:09,817", "01:20:09,818 --> 01:20:09,817", "자막 #31", "종료시간이 시작시간보다 같거나 빠름"),
])
def test_evaluation_review_reasons_gui(window, app, tmp_path, monkeypatch, marker, timecode, review_location, reason):
    path = tmp_path / "markers.srt"
    content = f"31\n{timecode}\n{marker}\n"
    path.write_bytes(content.encode("utf-8"))

    class Spacer:
        def space(self, text, **_kwargs):
            return text

    monkeypatch.setattr(gui, "create_spacer", Spacer)
    window._files_loaded([path])
    window.evaluation_radio.click()
    window.start_evaluation()
    finish_work(window, app)
    table = window.evaluation_table
    assert table.item(0, 1).text() == "검토 필요"
    assert table.item(0, 2).text() == "0"
    assert review_location in table.item(0, 1).toolTip()
    assert marker in table.item(0, 5).text()
    assert reason in window.log_view.toPlainText()
    assert table.cellWidget(0, 5).isEnabled()
    table.cellWidget(0, 5).click()
    dialog = next(dialog for dialog in window.findChildren(QDialog) if dialog.isVisible())
    details = dialog.findChild(QPlainTextEdit).toPlainText()
    assert review_location in details and marker in details and reason in details
    dialog.close()
    csv_path = tmp_path / "markers.csv"
    monkeypatch.setattr(gui.QFileDialog, "getSaveFileName", lambda *args: (str(csv_path), "CSV (*.csv)"))
    window.export_evaluation_csv()
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[1][1] == "검토 필요"
    assert review_location in rows[1][-1] and marker in rows[1][-1] and reason in rows[1][-1]
    assert path.read_bytes() == content.encode("utf-8")


@pytest.mark.parametrize("skipped", [0, 1, 2])
def test_evaluation_encoding_timeline_review_dialog_csv_and_filter(window, app, tmp_path, monkeypatch, skipped):
    paths = [tmp_path / "normal.srt", tmp_path / "timeline.srt"]
    paths[0].write_text("1\r\n00:00:01,000 --> 00:00:02,000\n정상\n", encoding="utf-8", newline="")
    first = "변경" if skipped == 2 else "안녕 하세요"
    second = "변경" if skipped else "반갑 습니다"
    source = f"1\r\n00:00:03,000 --> 00:00:04,000\n{first}\n\n12\n00:00:01,000 --> 00:00:02,000\n{second}\n"
    paths[1].write_bytes(source.encode("utf-16"))

    class Spacer:
        def space(self, text, **_kwargs):
            if text != "정상":
                assert paths[1].read_bytes() == source.encode("utf-8-sig")
            return text + "!" if text == "변경" else text.replace(" ", "")

    monkeypatch.setattr(gui, "create_spacer", Spacer)
    window._files_loaded(paths)
    window._store_review(paths[0], "old_revised.srt", ["기존 LLM 검토 내역"])
    old_reviews = dict(window.review_results)
    window.evaluation_radio.click()
    window.start_evaluation()
    finish_work(window, app)
    table = window.evaluation_table
    table.sortItems(0, Qt.SortOrder.DescendingOrder)
    assert table.item(0, 0).toolTip() == str(paths[1])
    state = table.item(0, 1).text()
    assert "검토 필요" in state
    if skipped:
        assert f"{skipped}줄 제외" in state
        assert ("평가 불가" if skipped == 2 else "부분 평가") in state
    assert table.item(0, 4).text() == ("—" if skipped == 2 else "200.00")
    assert "UTF-8 BOM (원본 교체 완료)" in table.item(0, 1).toolTip()
    assert "자막 #12: 시작시간 역행" in table.item(0, 5).text()
    assert "직전 자막 #1: 00:00:03,000" in table.item(0, 5).toolTip()
    assert "검토 필요 1개" in window.status_label.text()
    assert "[인코딩 변환]" in window.log_view.toPlainText()
    assert "[확인필요] 자막 #12" in window.log_view.toPlainText()
    table.cellWidget(0, 5).click()
    dialog = next(dialog for dialog in window.findChildren(QDialog) if dialog.isVisible())
    assert paths[1].name in dialog.windowTitle()
    details = dialog.findChild(QPlainTextEdit).toPlainText()
    assert str(paths[1]) in details and "자막 #12: 시작시간 역행" in details
    dialog.close()
    assert not table.cellWidget(1, 5).isEnabled()
    assert window.review_results == old_reviews
    assert not window.completed_paths
    assert [window.table.item(row, 1).text() for row in range(2)] == ["대기", "대기"]
    csv_path = tmp_path / "timeline.csv"
    monkeypatch.setattr(gui.QFileDialog, "getSaveFileName", lambda *args: (str(csv_path), "CSV (*.csv)"))
    window.export_evaluation_csv()
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0][-1] == "검토"
    assert "자막 #12: 시작시간 역행" in rows[1][-1]
    window.evaluation_threshold_spin.setValue(100)
    window.filter_evaluation_results()
    assert window.paths == ([] if skipped == 2 else [paths[1]])
    assert paths[1].read_bytes() == source.encode("utf-8-sig")
    if skipped != 2:
        assert table.cellWidget(0, 5).isEnabled()
        source = source.replace("00:00:03,000", "00:00:00,000")
        paths[1].write_bytes(source.encode("utf-8-sig"))
        window.start_evaluation()
        finish_work(window, app)
        assert "검토 필요" not in table.item(0, 1).text()
        assert not table.cellWidget(0, 5).isEnabled()
        assert table.item(0, 5).text() == ""


@pytest.mark.parametrize(("threshold", "kept_indices"), [
    (0.0, [0, 1, 2, 3, 6]), (0.33, [1, 2, 3]), (101.0, []),
])
def test_evaluation_threshold_filters_correction_targets(window, app, tmp_path, monkeypatch, threshold, kept_indices):
    paths = [tmp_path / name for name in (
        "below.srt", "equal.srt", "above.srt", "high.srt", "empty.srt", "failed.srt", "zero.srt",
    )]
    content = "1\n00:00:01,000 --> 00:00:02,000\n본문\n"
    for path in paths:
        path.write_text(content, encoding="utf-8")
    results = [
        gui.EvaluationResult(1, 3031), gui.EvaluationResult(33, 100000),
        gui.EvaluationResult(1, 3000), gui.EvaluationResult(10, 100),
        gui.EvaluationResult(0, 0), None, gui.EvaluationResult(0, 100),
    ]
    monkeypatch.setattr(gui, "create_spacer", object)

    def evaluate(path, _spacer, **_kwargs):
        result = results[paths.index(path)]
        if result is None:
            raise ValueError("평가 실패")
        return result

    monkeypatch.setattr(gui, "evaluate_file", evaluate)
    window._files_loaded(paths)
    for path in paths:
        window._store_review(path, "", ["기존 검토 기록"])
    window.evaluation_radio.click()
    assert not window.filter_evaluation_button.isEnabled()
    window.start_evaluation()
    assert not window.filter_evaluation_button.isEnabled()
    finish_work(window, app)
    window.evaluation_table.sortItems(4, Qt.SortOrder.DescendingOrder)
    window.evaluation_threshold_spin.setValue(threshold)
    assert window.filter_evaluation_button.isEnabled()
    window.resize(720, 620)
    window.show()
    app.processEvents()
    panel = window.evaluation_filter_panel
    spin = window.evaluation_threshold_spin
    button = window.filter_evaluation_button
    assert panel.rect().contains(spin.geometry())
    assert panel.rect().contains(button.geometry())
    assert not spin.geometry().intersects(button.geometry())
    window.filter_evaluation_button.click()
    expected = [paths[index] for index in kept_indices]
    assert window.paths == expected
    assert window.table.rowCount() == len(expected)
    assert window.evaluation_table.rowCount() == len(expected)
    assert {
        window.evaluation_table.item(row, 0).toolTip() for row in range(len(expected))
    } == {str(path) for path in expected}
    assert window.count_label.text() == f"자막 {len(expected)}개"
    assert set(window.review_results) == set(expected)
    assert [item.path for item in window.settings.load_worklist().files] == [str(path) for path in expected]
    assert window.export_button.isEnabled() == bool(expected)
    assert window.filter_evaluation_button.isEnabled() == bool(expected)
    window.filter_evaluation_results()
    assert window.paths == expected
    assert all(path.read_text(encoding="utf-8") == content for path in paths)
    corrected = []

    def correct(path, _corrector, _wrap_length, **_kwargs):
        corrected.append(path)
        return path.with_stem(path.stem + "_revised"), []

    monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
    monkeypatch.setattr(gui, "correct_file", correct)
    window.correction_radio.click()
    assert window.evaluation_filter_panel.isHidden()
    assert not window.filter_evaluation_button.isEnabled()
    prepare_model(window)
    assert window.start_button.isEnabled() == bool(expected)
    window.start_correction()
    finish_work(window, app)
    assert set(corrected) == set(expected)
    assert len(corrected) == len(expected)


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
        assert not window.filter_evaluation_button.isEnabled()
        window.filter_evaluation_results()
        assert window.paths == paths
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


def test_evaluation_double_click_opens_sorted_rows_file_and_folder(window, tmp_path, monkeypatch):
    paths = [tmp_path / folder / "subtitle.srt" for folder in ("first", "second")]
    table = window.evaluation_table
    table.setRowCount(2)
    for row, path in enumerate(paths):
        item = gui.QTableWidgetItem(path.name)
        item.setToolTip(str(path))
        table.setItem(row, 0, item)
        window._evaluation_result(row, "완료", gui.EvaluationResult(row + 1, 100))
    table.setSortingEnabled(True)
    table.sortItems(2, Qt.SortOrder.DescendingOrder)
    assert table.item(0, 0).toolTip() == str(paths[1])
    opened = []
    monkeypatch.setattr(gui.QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    for column in range(5):
        table.cellDoubleClicked.emit(0, column)
    assert opened == [QUrl.fromLocalFile(str(paths[1])), QUrl.fromLocalFile(str(paths[1].parent))]
    table.item(0, 0).setToolTip("")
    table.cellDoubleClicked.emit(0, 0)
    table.cellDoubleClicked.emit(0, 1)
    assert len(opened) == 2


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
        ["평가 자막", "평가 상태", "띄어쓰기 오류 수", "글자 수", "1,000자당 오류 수", "검토"],
        ["빈도 높은 자막.srt", "완료", "2", "100", "20.00", ""],
        ['기린의 날개 (2012)\\한글,"자막".srt', "완료", "10", "10000", "1.00", ""],
        ["실패.srt", "실패", "—", "—", "—", ""],
        ["중단.srt", "중단", "—", "—", "—", ""],
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


def test_real_kiwi_evaluation_with_cpu_autodetection(tmp_path):
    source = tmp_path / "real.srt"
    source.write_text("1\n00:00:01,000 --> 00:00:02,000\n안녕 하세요\n", encoding="utf-8")
    script = """
import sys
import time
from pathlib import Path
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from app_settings import AppSettings
from srt_spellchecker_gui import MainWindow

application = QApplication([])
settings = AppSettings(QSettings(sys.argv[1], QSettings.Format.IniFormat))
settings.load_key = lambda service: None
settings.save_key = lambda service, key: None
window = MainWindow(settings)
window._files_loaded([Path(sys.argv[2])])
window.evaluation_radio.click()
window.start_button.click()
assert window.worker is not None
deadline = time.monotonic() + 30
while window.worker is not None and time.monotonic() < deadline:
    application.processEvents()
    time.sleep(0.01)
assert window.worker is None, "평가 완료 시간 초과"
table = window.evaluation_table
assert [table.item(0, column).text() for column in (1, 2, 3)] == ["완료", "1", "5"]
assert window.export_button.isEnabled()
assert "실패 0개" in window.status_label.text()
window.close()
"""
    environment = os.environ.copy()
    environment.pop("KIWI_ARCH_TYPE", None)
    environment.update(QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "settings.ini"), str(source)],
        cwd=Path(gui.__file__).parent, env=environment, capture_output=True,
        encoding="utf-8", timeout=45, check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, f"종료 코드 {result.returncode:#x}\n{result.stderr}"
    assert source.read_text(encoding="utf-8").endswith("안녕 하세요\n")


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
    assert project["project"]["version"] == gui.__version__ == "1.1.7"
    expected_title = "SRT Spellchecker v1.1.7"
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


@pytest.mark.parametrize("table_name", ["table", "evaluation_table"])
@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("row", [0, 1])
def test_table_hover_differs_from_selection_and_preserves_active_background(window, table_name, dark, row):
    table = getattr(window, table_name)
    palette = table.palette()
    palette.setColor(QPalette.ColorRole.Base, QColor("#202020" if dark else "#ffffff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#d76098"))
    table.setPalette(palette)
    table.setRowCount(2)
    item = QTableWidgetItem("자막")
    table.setItem(row, 0, item)
    index = table.model().index(row, 0)
    flags = gui.QStyle.StateFlag

    def rendered_background(state):
        option = QStyleOptionViewItem()
        option.initFrom(table)
        option.widget = table
        option.rect = QRect(0, 0, 200, 32)
        option.state = flags.State_Enabled | flags.State_Active | state
        if row:
            option.features |= QStyleOptionViewItem.ViewItemFeature.Alternate
        image = QImage(200, 32, QImage.Format.Format_ARGB32)
        image.fill(palette.base().color())
        painter = QPainter(image)
        try:
            table.itemDelegate().paint(painter, option, index)
        finally:
            painter.end()
        return image.pixelColor(185, 25)

    hovered = rendered_background(flags.State_MouseOver)
    selected = rendered_background(flags.State_Selected)
    assert hovered == QColor("#353535" if dark else "#eeeeee")
    assert hovered != selected
    assert rendered_background(flags.State_Selected | flags.State_MouseOver) == selected
    active_background = QColor("#214b3a" if dark else "#d9f2e7")
    item.setBackground(active_background)
    assert rendered_background(flags.State_MouseOver) == active_background
    assert rendered_background(flags.State_Selected | flags.State_MouseOver) == rendered_background(flags.State_Selected)


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


@pytest.mark.parametrize("llm_review", [False, True])
def test_correction_merges_evaluation_reviews_and_restores_them(window, app, tmp_path, monkeypatch, llm_review):
    paths = [tmp_path / folder / "subtitle.srt" for folder in ("review", "normal")]
    source = (
        "1\n00:00:01,000 --> 00:00:02,000\n안녕하세요\n\n"
        "41\n39:03:20,785 --> 00:05:43,587\n안녕하세요\n\n"
        "42\n00:05:46,250 --> 00:05:47,000\n안녕하세요\n\n"
        "186\n18:56:09,663 --> 00:17:18,831\n안녕하세요\n\n"
        "187\n00:17:29,529 --> 00:17:30,000\n안녕하세요\n"
    )
    for path, content in zip(paths, (source, SAMPLE)):
        path.parent.mkdir()
        path.write_bytes(content.encode("utf-8"))

    class Spacer:
        def space(self, text, **_kwargs):
            return text

    def respond(payload):
        response = echo(payload)
        if llm_review and len(payload) == 5:
            response["parsed"].items[0].corrected_lines = []
        return response

    class Service(EchoService):
        def __init__(self, *args):
            FakeCorrector.__init__(self, respond)

    monkeypatch.setattr(gui, "create_spacer", Spacer)
    monkeypatch.setattr(gui, "ServiceCorrector", Service)
    window._files_loaded(paths)
    window.evaluation_radio.click()
    window.start_evaluation()
    finish_work(window, app)
    evaluation_logs = list(window.evaluation_reviews[paths[0]])
    assert len(evaluation_logs) == 4
    assert not window.review_results
    window.evaluation_table.sortItems(0, Qt.SortOrder.AscendingOrder)
    window.evaluation_threshold_spin.setValue(0)
    window.filter_evaluation_results()
    window.correction_radio.click()
    prepare_model(window)
    window.concurrency_spin.setValue(2)
    window.start_correction()
    finish_work(window, app)

    output, logs = window.review_results[paths[0]]
    assert logs[:4] == evaluation_logs
    assert len(logs) == 4 + int(llm_review)
    if llm_review:
        assert "빈 교정 결과" in logs[-1]
    assert Path(output) == paths[0].with_stem("subtitle_revised")
    assert Path(output).read_text(encoding="utf-8") == source
    assert window.table.item(0, 1).text() == "검토 필요"
    assert window.table.cellWidget(0, 3).isEnabled()
    assert window.table.item(1, 1).text() == "완료"
    assert not window.table.cellWidget(1, 3).isEnabled()
    assert paths[1] not in window.review_results
    assert "저장 2개 (검토 1개)" in window.status_label.text()
    window.table.cellWidget(0, 3).click()
    dialog = next(dialog for dialog in window.findChildren(QDialog) if dialog.isVisible())
    details = dialog.findChild(QPlainTextEdit).toPlainText()
    assert f"결과: {output}" in details
    assert f"검토 {len(logs)}건" in details
    assert all(details.count(entry) == 1 for entry in logs)
    dialog.close()

    restored = gui.MainWindow()
    try:
        assert restored.review_results[paths[0]] == (output, logs)
        assert restored.table.item(0, 1).text() == "검토 필요"
        assert restored.table.cellWidget(0, 3).isEnabled()
        assert set(paths) == restored.completed_paths
    finally:
        restored.close()
        restored.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("first_result", ["failure", "cancel", "reevaluate"])
def test_evaluation_reviews_survive_retry_and_use_latest_evaluation(window, app, tmp_path, monkeypatch, first_result):
    path = tmp_path / "retry.srt"
    logs = ("[확인필요] 자막 #41: 시작시간 역행",)
    result = gui.EvaluationResult(0, 0, warnings=("평가에서 제외",), review_logs=logs)
    monkeypatch.setattr(gui, "create_spacer", object)
    monkeypatch.setattr(gui, "evaluate_file", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
    window._files_loaded([path])
    window.evaluation_radio.click()
    window.start_evaluation()
    finish_work(window, app)
    assert "평가 불가" in window.evaluation_table.item(0, 1).text()

    def fail(*_args, **_kwargs):
        if first_result == "cancel":
            raise gui.CorrectionCancelled()
        raise OSError("저장 실패")

    if first_result == "reevaluate":
        result = gui.EvaluationResult(0, 5)
        window.start_evaluation()
        finish_work(window, app)
    else:
        monkeypatch.setattr(gui, "correct_file", fail)
        window.correction_radio.click()
        prepare_model(window)
        window.start_correction()
        finish_work(window, app)
        assert window.table.item(0, 1).text() == ("실패" if first_result == "failure" else "중단")
        assert not window.table.cellWidget(0, 3).isEnabled()
        assert window.evaluation_reviews[path] == logs

    monkeypatch.setattr(gui, "correct_file", lambda *_args, **_kwargs: (
        path.with_stem("retry_revised"), [] if first_result == "reevaluate" else list(logs),
    ))
    window.correction_radio.click()
    prepare_model(window)
    window.start_correction()
    finish_work(window, app)
    if first_result == "reevaluate":
        assert window.table.item(0, 1).text() == "완료"
        assert path not in window.review_results
    else:
        assert window.table.item(0, 1).text() == "검토 필요"
        assert window.review_results[path][1] == list(logs)


@pytest.mark.parametrize("remove", ["clear", "waiting", "selected", "filter"])
def test_removing_files_discards_evaluation_reviews(window, app, tmp_path, monkeypatch, remove):
    path = tmp_path / "removed.srt"
    monkeypatch.setattr(gui, "create_spacer", object)
    monkeypatch.setattr(gui, "evaluate_file", lambda *_args, **_kwargs: gui.EvaluationResult(
        0, 5, review_logs=("[확인필요] 자막 #41: 시작시간 역행",),
    ))
    window._files_loaded([path])
    window.evaluation_radio.click()
    window.start_evaluation()
    finish_work(window, app)
    assert path in window.evaluation_reviews
    if remove == "clear":
        window.clear_files()
    elif remove == "waiting":
        window.remove_waiting_file(path)
    elif remove == "selected":
        window.table.selectRow(0)
        window.remove_selected()
    else:
        window.evaluation_threshold_spin.setValue(1)
        window.filter_evaluation_results()
    assert not window.paths
    assert not window.evaluation_reviews


@pytest.mark.parametrize(("batch_size", "expected_sizes"), [
    (1, [1] * 27), (10, [10, 10, 7]), (25, [25, 2]), (200, [27]),
])
def test_gui_batch_size_controls_actual_requests(window, app, tmp_path, monkeypatch, batch_size, expected_sizes):
    batches = []

    def respond(payload):
        batches.append(payload)
        return echo(payload)

    class Service(EchoService):
        def __init__(self, *args):
            FakeCorrector.__init__(self, respond)

    monkeypatch.setattr(gui, "ServiceCorrector", Service)
    source = tmp_path / "batches.srt"
    content = "\n\n".join(
        f"{index + 1}\n00:00:01,000 --> 00:00:02,000\n자막 {index + 1}"
        for index in range(27)
    ) + "\n"
    source.write_text(content, encoding="utf-8")
    window._files_loaded([source])
    prepare_model(window)
    assert window.batch_size_spin.value() == 25
    window.batch_size_spin.setValue(batch_size)
    window.start_correction()
    finish_work(window, app)
    assert [len(batch) for batch in batches] == expected_sizes
    assert [item["id"] for batch in batches for item in batch] == list(range(27))
    assert source.with_stem("batches_revised").read_text(encoding="utf-8") == content
    assert window.table.item(0, 1).text() == "완료"
    assert window.progress_bar.value() == 1000
    assert f"[배치 크기] 요청당 자막 {batch_size}개" in window.log_view.toPlainText()


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
        assert kwargs["batch_size"] == 10
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
    window.batch_size_spin.setValue(10)
    window.start_correction()
    try:
        wait_until(lambda: all(entered[path].is_set() for path in paths[:limit]))
        wait_until(lambda: window.progress_bar.value() == int(limit * 500 / len(paths)))
        assert len(window.correction_workers) == limit
        assert not entered[paths[limit]].is_set()
        assert not window.concurrency_spin.isEnabled()
        assert not window.batch_size_spin.isEnabled()
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
    window.batch_size_spin.setValue(50)
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
        assert restored.batch_size_spin.value() == 50
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
    window.batch_size_spin.setValue(40)
    saved = window.settings.load_preferences()
    assert saved.models["OpenAI"].id == "test-model"
    assert saved.wrap is True
    assert saved.max_line_length == 40
    assert saved.concurrent_files == 2
    assert saved.batch_size == 40


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


@pytest.mark.parametrize("saved_key", ["vault-secret", ""])
def test_gui_uses_only_saved_keys_and_ignores_environment(window, app, tmp_path, monkeypatch, saved_key) -> None:
    window.settings.save_key("OpenAI", saved_key)
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-environment-secret")
    monkeypatch.setenv("OPENAI_MODEL", "environment-model")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=dotenv-secret\n", encoding="utf-8")
    restored = gui.MainWindow()
    try:
        assert restored.key_edit.text() == saved_key
        assert restored.selected_model() is None
        restored.service_combo.setCurrentText("OpenRouter")
        assert restored.key_edit.text() == ""
        assert not restored.fetch_button.isEnabled()
        assert not restored.start_button.isEnabled()
    finally:
        restored.close()
        restored.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_cleared_gui_key_stays_empty_after_restart(window, app, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    window.key_edit.setText("gui-secret")
    window.key_edit.editingFinished.emit()
    assert window.settings.load_key("OpenAI") == "gui-secret"
    window.key_edit.clear()
    window.key_edit.editingFinished.emit()
    restored = gui.MainWindow()
    try:
        assert restored.key_edit.text() == ""
        assert restored.settings.load_key("OpenAI") is None
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
