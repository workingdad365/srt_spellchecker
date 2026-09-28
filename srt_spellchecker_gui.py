from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PySide6.QtCore import QMimeData, QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QCompleter,
    QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSizePolicy, QSpinBox, QSplitter, QStyle, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ai_services import BASE_URLS, ModelInfo, ServiceCorrector, fetch_models, reasoning_label
from app_settings import AppSettings, Preferences, SavedModel, SettingsError
from srt_spellchecker import (
    DEFAULT_MAX_LINE_LENGTH, FATAL_API_ERRORS, CorrectionCancelled,
    check_cancelled, collect_srt_files, correct_file,
)


class FileTable(QTableWidget):
    paths_dropped = Signal(list)

    def __init__(self) -> None:
        super().__init__(0, 3)
        self.setHorizontalHeaderLabels(["원본 자막", "상태", "결과 파일"])
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.verticalHeader().hide()
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.setToolTip("SRT 파일 또는 폴더를 드래그하여 추가")

    @staticmethod
    def local_paths(mime: QMimeData) -> list[Path]:
        return [Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile()]

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self.isEnabled() and self.local_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        self.dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        paths = self.local_paths(event.mimeData())
        if self.isEnabled() and paths:
            event.acceptProposedAction()
            self.paths_dropped.emit(paths)
        else:
            event.ignore()


class BackgroundTask(QThread):
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, action: Callable[[], Any], parent: QWidget) -> None:
        super().__init__(parent)
        self.action = action

    def run(self) -> None:
        try:
            result = self.action()
            if not self.isInterruptionRequested():
                self.result.emit(result)
        except CorrectionCancelled:
            pass
        except Exception as error:
            self.failed.emit(str(error))


class CorrectionWorker(QThread):
    log = Signal(str)
    file_state = Signal(int, str, str)
    progress = Signal(int)
    summary = Signal(str)

    def __init__(
        self, paths: list[Path], service: str, api_key: str, model: ModelInfo,
        wrap_length: int | None, parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.paths = list(paths)
        self.service = service
        self.api_key = api_key
        self.model = model
        self.wrap_length = wrap_length

    def write_log(self, message: str) -> None:
        self.log.emit(message.replace(self.api_key, "[API KEY]"))

    def run(self) -> None:
        completed = 0
        failed = 0
        review = 0
        corrector = None
        try:
            check_cancelled(self.isInterruptionRequested)
            corrector = ServiceCorrector(self.service, self.api_key, self.model)
            for index, path in enumerate(self.paths):
                check_cancelled(self.isInterruptionRequested)
                self.file_state.emit(index, "교정 중", "")
                self.write_log(f"[파일] {path}")
                try:
                    output, logs = correct_file(
                        path, corrector, self.wrap_length,
                        on_log=self.write_log,
                        on_progress=lambda done, total, file_index=index: self.progress.emit(
                            int((file_index + done / total) / len(self.paths) * 1000)
                        ),
                        is_cancelled=self.isInterruptionRequested,
                    )
                    for message in logs:
                        self.write_log(message)
                    completed += 1
                    review += bool(logs)
                    self.file_state.emit(index, "검토 필요" if logs else "완료", str(output))
                    self.write_log(f"[저장] {output}")
                except CorrectionCancelled:
                    self.file_state.emit(index, "중단", "")
                    raise
                except Exception as error:
                    failed += 1
                    self.file_state.emit(index, "실패", "")
                    self.write_log(f"[오류] {path}: {error}")
                    if isinstance(error, FATAL_API_ERRORS):
                        break
                self.progress.emit(int((index + 1) / len(self.paths) * 1000))
        except CorrectionCancelled:
            self.write_log("[중단] 미완료 파일은 저장하지 않았습니다.")
        except Exception as error:
            self.write_log(f"[오류] {error}")
        finally:
            if corrector is not None:
                corrector.close()
            self.api_key = ""
            pending = len(self.paths) - completed - failed
            state = "중단" if self.isInterruptionRequested() else "작업 종료"
            self.summary.emit(
                f"{state}: 저장 {completed}개 (검토 {review}개), 실패 {failed}개, 미처리 {pending}개"
            )


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings | None = None) -> None:
        super().__init__()
        self.settings = settings if settings is not None else AppSettings()
        self._restoring = True
        startup_errors: list[str] = []
        try:
            self.preferences = self.settings.load_preferences()
        except SettingsError as error:
            self.preferences = Preferences()
            startup_errors.append(str(error))
        self.paths: list[Path] = []
        self.worker: QThread | None = None
        self.close_pending = False
        self.keys: dict[str, str] = {}
        for service in BASE_URLS:
            try:
                saved_key = self.settings.load_key(service)
            except SettingsError as error:
                saved_key = None
                startup_errors.append(str(error))
            self.keys[service] = (
                saved_key if saved_key is not None
                else os.getenv(f"{service.upper()}_API_KEY", "").strip()
            )
        self._persisted_keys = dict(self.keys)
        self.current_service = self.preferences.service
        self.key_save_timer = QTimer(self)
        self.key_save_timer.setSingleShot(True)
        self.key_save_timer.setInterval(600)
        self.key_save_timer.timeout.connect(self._save_current_key)
        self.setWindowTitle("SRT Spellchecker")
        self.setWindowIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.resize(980, 800)
        self.setMinimumSize(720, 620)
        self._build_ui()
        self.service_combo.setCurrentText(self.current_service)
        self.key_edit.setText(self.keys[self.current_service])
        self.wrap_check.setChecked(self.preferences.wrap)
        self.length_spin.setValue(self.preferences.max_line_length)
        self._restore_model()
        self._restoring = False
        self._model_changed()
        for message in startup_errors:
            self.show_error(message)

    def _button(self, text: str, icon: QStyle.StandardPixmap, slot) -> QPushButton:
        button = QPushButton(text)
        button.setIcon(self.style().standardIcon(icon))
        button.clicked.connect(slot)
        return button

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)
        heading = QLabel("SRT Spellchecker")
        font = heading.font()
        font.setPointSize(18)
        font.setBold(True)
        heading.setFont(font)
        root.addWidget(heading)

        self.settings_panel = QWidget()
        form = QFormLayout(self.settings_panel)
        form.setContentsMargins(0, 0, 0, 0)
        self.service_combo = QComboBox()
        self.service_combo.addItems(list(BASE_URLS))
        self.service_combo.currentTextChanged.connect(self._service_changed)
        form.addRow("AI 서비스", self.service_combo)
        key_row = QHBoxLayout()
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setToolTip("API 키는 운영체제 자격 증명 저장소에 저장")
        self.key_edit.textChanged.connect(self._key_changed)
        self.key_edit.editingFinished.connect(self._save_current_key)
        self.show_key = QCheckBox("키 표시")
        self.show_key.toggled.connect(lambda checked: self.key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        ))
        key_row.addWidget(self.key_edit, 1)
        key_row.addWidget(self.show_key)
        form.addRow("API 키", key_row)
        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.model_combo.setMinimumContentsLength(18)
        self.model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model_combo.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        self.model_combo.completer().setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.model_combo.currentTextChanged.connect(self._model_changed)
        self.fetch_button = self._button(
            "모델 가져오기", QStyle.StandardPixmap.SP_BrowserReload, self.load_models,
        )
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.fetch_button)
        form.addRow("모델", model_row)
        self.reasoning_note = QLabel("미선택")
        self.reasoning_note.setWordWrap(True)
        form.addRow("추론", self.reasoning_note)
        root.addWidget(self.settings_panel)

        tools = QHBoxLayout()
        self.files_button = self._button("파일 추가", QStyle.StandardPixmap.SP_FileIcon, self.choose_files)
        self.folder_button = self._button("폴더 추가", QStyle.StandardPixmap.SP_DirOpenIcon, self.choose_folder)
        self.remove_button = self._button("", QStyle.StandardPixmap.SP_DialogDiscardButton, self.remove_selected)
        self.remove_button.setToolTip("선택한 파일 제거")
        self.clear_button = self._button("목록 비우기", QStyle.StandardPixmap.SP_TrashIcon, self.clear_files)
        tools.addWidget(self.files_button)
        tools.addWidget(self.folder_button)
        tools.addWidget(self.remove_button)
        tools.addWidget(self.clear_button)
        tools.addStretch()
        self.count_label = QLabel("자막 0개")
        tools.addWidget(self.count_label)
        root.addLayout(tools)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.table = FileTable()
        self.table.paths_dropped.connect(self.add_paths)
        self.table.itemSelectionChanged.connect(self._update_controls)
        self.table.cellDoubleClicked.connect(self.open_file_location)
        splitter.addWidget(self.table)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(10000)
        self.log_view.setMinimumHeight(80)
        splitter.addWidget(self.log_view)
        splitter.setSizes([300, 140])
        root.addWidget(splitter, 1)

        options = QHBoxLayout()
        self.wrap_check = QCheckBox("긴 줄 나누기")
        self.length_spin = QSpinBox()
        self.length_spin.setRange(1, 200)
        self.length_spin.setValue(DEFAULT_MAX_LINE_LENGTH)
        self.length_spin.setSuffix(" 자")
        self.length_spin.setEnabled(False)
        self.wrap_check.toggled.connect(self._update_controls)
        self.wrap_check.toggled.connect(self._save_preferences)
        self.length_spin.valueChanged.connect(self._save_preferences)
        options.addWidget(self.wrap_check)
        options.addWidget(self.length_spin)
        options.addStretch()
        self.start_button = self._button("교정 시작", QStyle.StandardPixmap.SP_MediaPlay, self.start_correction)
        self.cancel_button = self._button("중단", QStyle.StandardPixmap.SP_MediaStop, self.cancel_work)
        options.addWidget(self.start_button)
        options.addWidget(self.cancel_button)
        root.addLayout(options)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        root.addWidget(self.progress_bar)
        self.status_label = QLabel("대기")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        self.setCentralWidget(central)

    def selected_model(self) -> ModelInfo | None:
        index = self.model_combo.findText(self.model_combo.currentText())
        return self.model_combo.itemData(index) if index >= 0 else None

    def _update_controls(self) -> None:
        busy = self.worker is not None
        self.settings_panel.setEnabled(not busy)
        self.fetch_button.setEnabled(not busy and bool(self.key_edit.text().strip()))
        self.model_combo.setEnabled(not busy and self.model_combo.count() > 0)
        for widget in (self.files_button, self.folder_button, self.table, self.wrap_check):
            widget.setEnabled(not busy)
        self.length_spin.setEnabled(not busy and self.wrap_check.isChecked())
        self.remove_button.setEnabled(not busy and bool(self.table.selectedItems()))
        self.clear_button.setEnabled(not busy and bool(self.paths))
        self.start_button.setEnabled(
            not busy and bool(self.paths) and bool(self.key_edit.text().strip())
            and self.selected_model() is not None
        )
        self.cancel_button.setEnabled(busy and not self.worker.isInterruptionRequested())
        self.count_label.setText(f"자막 {len(self.paths)}개")

    def _service_changed(self, service: str) -> None:
        if self._restoring:
            return
        self._save_current_key()
        self._restoring = True
        self.current_service = service
        self.preferences.service = "OpenRouter" if service == "OpenRouter" else "OpenAI"
        self.show_key.setChecked(False)
        self.key_edit.setText(self.keys[service])
        self._restore_model()
        self._restoring = False
        self._model_changed()
        self._save_preferences()

    def _key_changed(self) -> None:
        if self._restoring:
            return
        self.keys[self.current_service] = self.key_edit.text().strip()
        self.model_combo.clear()
        self._model_changed()
        self.key_save_timer.start()

    def _model_changed(self) -> None:
        if self._restoring:
            return
        model = self.selected_model()
        self.reasoning_note.setText(reasoning_label(self.current_service, model) if model else "미선택")
        if model is not None:
            self.preferences.models[self.current_service] = SavedModel.from_model(model)
            self._save_preferences()
        self._update_controls()

    def _restore_model(self) -> None:
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        saved = self.preferences.models.get(self.current_service)
        if saved is not None:
            self.model_combo.addItem(saved.id, saved.to_model())
            self.model_combo.setItemData(0, saved.name, Qt.ItemDataRole.ToolTipRole)
        self.model_combo.blockSignals(False)

    def _save_preferences(self) -> None:
        if self._restoring:
            return
        self.preferences.wrap = self.wrap_check.isChecked()
        self.preferences.max_line_length = self.length_spin.value()
        try:
            self.settings.save_preferences(self.preferences)
        except SettingsError as error:
            self.show_error(str(error))

    def _save_current_key(self) -> None:
        if self._restoring:
            return
        self.key_save_timer.stop()
        api_key = self.key_edit.text().strip()
        self.keys[self.current_service] = api_key
        if api_key == self._persisted_keys[self.current_service]:
            return
        try:
            self.settings.save_key(self.current_service, api_key)
        except SettingsError as error:
            self.show_error(str(error))
        else:
            self._persisted_keys[self.current_service] = api_key

    def _start_worker(self, worker: QThread, message: str) -> None:
        self.worker = worker
        self.status_label.setText(message)
        worker.finished.connect(self._worker_finished)
        self._update_controls()
        worker.start()

    def _worker_finished(self) -> None:
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.wait()
            if worker.isInterruptionRequested():
                self.status_label.setText("작업 중단됨")
            worker.deleteLater()
        self._update_controls()
        if self.close_pending:
            self.close()

    def show_error(self, message: str) -> None:
        for key in [self.key_edit.text().strip(), *self.keys.values()]:
            if key:
                message = message.replace(key, "[API KEY]")
        self.status_label.setText("오류: " + message)
        self.log_view.appendPlainText("[오류] " + message)

    def load_models(self) -> None:
        if self.worker is not None or not self.key_edit.text().strip():
            return
        self._save_current_key()
        service, api_key = self.current_service, self.key_edit.text().strip()
        self.model_combo.clear()
        worker = BackgroundTask(lambda: fetch_models(service, api_key), self)
        worker.result.connect(self._models_loaded)
        worker.failed.connect(self.show_error)
        self._start_worker(worker, "모델 목록 조회 중")

    def _models_loaded(self, models: list[ModelInfo]) -> None:
        previous = self.preferences.models.get(self.current_service)
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for model in models:
            self.model_combo.addItem(model.id, model)
            self.model_combo.setItemData(
                self.model_combo.count() - 1, model.name, Qt.ItemDataRole.ToolTipRole,
            )
        selected = self.model_combo.findText(previous.id) if previous is not None else -1
        self.model_combo.setCurrentIndex(selected)
        self.model_combo.blockSignals(False)
        if previous is not None and selected < 0:
            self.preferences.models.pop(self.current_service, None)
            self._save_preferences()
        self._model_changed()
        self.status_label.setText(f"모델 {len(models)}개" if models else "사용 가능한 모델이 없습니다.")

    def choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "자막 선택", "", "SRT (*.srt *.SRT)")
        self.add_paths([Path(path) for path in paths])

    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "자막 폴더 선택")
        if path:
            self.add_paths([Path(path)])

    def add_paths(self, paths: list[Path]) -> None:
        if not paths or self.worker is not None:
            return
        worker = BackgroundTask(lambda: collect_srt_files(paths, worker.isInterruptionRequested), self)
        worker.result.connect(self._files_loaded)
        worker.failed.connect(self.show_error)
        self._start_worker(worker, "자막 파일 탐색 중")

    def _files_loaded(self, paths: list[Path]) -> None:
        known = set(self.paths)
        added = 0
        for path in paths:
            if path in known:
                continue
            known.add(path)
            self.paths.append(path)
            row = self.table.rowCount()
            self.table.insertRow(row)
            item = QTableWidgetItem(path.name)
            item.setToolTip(str(path))
            self.table.setItem(row, 0, item)
            self.table.setItem(row, 1, QTableWidgetItem("대기"))
            self.table.setItem(row, 2, QTableWidgetItem(""))
            added += 1
        self.status_label.setText(f"자막 {added}개 추가" if paths else "SRT 파일이 없습니다.")
        self._update_controls()

    def remove_selected(self) -> None:
        if self.worker is not None:
            return
        rows = sorted({item.row() for item in self.table.selectedItems()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
            del self.paths[row]
        self._update_controls()

    def clear_files(self) -> None:
        if self.worker is None:
            self.table.setRowCount(0)
            self.paths.clear()
            self.progress_bar.setValue(0)
            self._update_controls()

    def open_file_location(self, row: int, column: int) -> None:
        item = self.table.item(row, column)
        path = item.toolTip() if item is not None else ""
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))

    def start_correction(self) -> None:
        model = self.selected_model()
        if self.worker is not None or not self.paths or model is None or not self.key_edit.text().strip():
            return
        self._save_current_key()
        for row in range(len(self.paths)):
            self._file_state(row, "대기", "")
        self.progress_bar.setValue(0)
        self.log_view.clear()
        self.log_view.appendPlainText(f"[모델] {self.current_service} / {model.id}")
        self.log_view.appendPlainText(f"[추론] {reasoning_label(self.current_service, model)}")
        worker = CorrectionWorker(
            self.paths, self.current_service, self.key_edit.text().strip(), model,
            self.length_spin.value() if self.wrap_check.isChecked() else None, self,
        )
        worker.log.connect(self.log_view.appendPlainText)
        worker.file_state.connect(self._file_state)
        worker.progress.connect(self.progress_bar.setValue)
        worker.summary.connect(self._summary)
        self._start_worker(worker, "교정 중")

    def _file_state(self, row: int, state: str, output: str) -> None:
        self.table.item(row, 1).setText(state)
        self.table.item(row, 2).setText(Path(output).name if output else "")
        self.table.item(row, 2).setToolTip(output)

    def _summary(self, message: str) -> None:
        self.status_label.setText(message)
        self.log_view.appendPlainText(message)

    def cancel_work(self) -> None:
        if self.worker is not None:
            self.worker.requestInterruption()
            self.status_label.setText("중단 요청됨: 진행 중인 API 요청 종료 대기")
            self._update_controls()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker is not None:
            answer = QMessageBox.question(self, "작업 진행 중", "작업을 중단하고 창을 닫으시겠습니까?")
            if answer == QMessageBox.StandardButton.Yes:
                self.close_pending = True
                self.cancel_work()
            event.ignore()
            return
        self._save_current_key()
        self._save_preferences()
        self.key_save_timer.stop()
        self._restoring = True
        self.keys.clear()
        self._persisted_keys.clear()
        self.key_edit.clear()
        event.accept()


def main() -> None:
    load_dotenv()
    app = QApplication(sys.argv)
    app.setApplicationName("SRT Spellchecker")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()