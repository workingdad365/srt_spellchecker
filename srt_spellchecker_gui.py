from __future__ import annotations

import csv
import io
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QIODevice, QMimeData, QModelIndex, QSaveFile, QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QCloseEvent, QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QCompleter, QDialog, QDoubleSpinBox,
    QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton,
    QSizePolicy, QSpinBox, QSplitter, QStyle, QStyledItemDelegate, QStyleOptionViewItem,
    QTableWidget, QTableWidgetItem, QToolButton,
    QVBoxLayout, QWidget,
)

from ai_services import BASE_URLS, ModelInfo, ServiceCorrector, fetch_models, reasoning_label
from app_settings import AppSettings, Preferences, SavedModel, SavedWorkFile, SavedWorklist, SettingsError
from spacing_evaluation import EvaluationResult, create_spacer, evaluate_file
from srt_spellchecker import (
    BATCH_SIZE, DEFAULT_MAX_LINE_LENGTH, FATAL_API_ERRORS, MAX_BATCH_SIZE, CorrectionCancelled,
    check_cancelled, collect_srt_files, correct_file,
)


__version__ = "1.1.7"


class TableItemDelegate(QStyledItemDelegate):
    def initStyleOption(self, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        super().initStyleOption(option, index)
        if option.state & QStyle.StateFlag.State_MouseOver:
            option.state &= ~QStyle.StateFlag.State_MouseOver
            if (
                not option.state & QStyle.StateFlag.State_Selected
                and option.backgroundBrush.style() == Qt.BrushStyle.NoBrush
            ):
                dark = option.palette.base().color().lightness() < 128
                option.backgroundBrush = QBrush(QColor("#353535" if dark else "#eeeeee"))


class FileTable(QTableWidget):
    paths_dropped = Signal(list)

    def __init__(self) -> None:
        super().__init__(0, 5)
        self.setItemDelegate(TableItemDelegate(self))
        self.setMouseTracking(True)
        self.setHorizontalHeaderLabels(["원본 자막", "상태", "결과 파일", "검토", "제거"])
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
        for column in (3, 4):
            self.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self.setColumnWidth(column, 44)
        self.verticalHeader().setDefaultSectionSize(32)
        self.setToolTip("SRT 파일 또는 폴더를 드래그하여 추가")

    def set_row_active(self, row: int, active: bool) -> None:
        dark = self.palette().base().color().lightness() < 128
        background = QColor("#214b3a" if dark else "#d9f2e7")
        foreground = QColor("#e3f5ec" if dark else "#123d2c")
        for column in range(self.columnCount()):
            item = self.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                self.setItem(row, column, item)
            item.setData(Qt.ItemDataRole.BackgroundRole, background if active else None)
            item.setData(Qt.ItemDataRole.ForegroundRole, foreground if active else None)
            font = item.font()
            font.setBold(active)
            item.setFont(font)
            button = self.cellWidget(row, column)
            if button is not None:
                button.setStyleSheet(
                    f"QToolButton {{ background-color: {background.name()}; }}" if active else ""
                )

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


class FileDiscoveryTask(BackgroundTask):
    progress = Signal(int, str)

    def __init__(self, paths: list[Path], parent: QWidget) -> None:
        super().__init__(
            lambda: collect_srt_files(paths, self.isInterruptionRequested, self.progress.emit), parent,
        )
        self.cancelled = False

    def run(self) -> None:
        try:
            super().run()
        finally:
            self.cancelled = self.isInterruptionRequested()


class CorrectionWorker(QThread):
    log = Signal(str)
    file_state = Signal(int, str, str)
    review_ready = Signal(int, str, list)
    progress = Signal(int)
    summary = Signal(str)

    def __init__(
        self, paths: list[Path], service: str, api_key: str, model: ModelInfo,
        wrap_length: int | None, parent: QWidget | None = None,
        *, batch_size: int = BATCH_SIZE,
    ) -> None:
        super().__init__(parent)
        self.paths = list(paths)
        self.service = service
        self.api_key = api_key
        self.model = model
        self.wrap_length = wrap_length
        self.batch_size = batch_size
        self.aborted = False

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
                        batch_size=self.batch_size,
                        on_log=self.write_log,
                        on_progress=lambda done, total, file_index=index: self.progress.emit(
                            int((file_index + done / total) / len(self.paths) * 1000)
                        ),
                        is_cancelled=self.isInterruptionRequested,
                    )
                    for message in logs:
                        self.write_log(message)
                    if logs:
                        self.review_ready.emit(index, str(output), [
                            message.replace(self.api_key, "[API KEY]") for message in logs
                        ])
                    completed += 1
                    review += bool(logs)
                    self.file_state.emit(index, "검토 필요" if logs else "완료", str(output))
                    self.write_log(f"[저장] {output}")
                except CorrectionCancelled:
                    self.file_state.emit(index, "중단", "")
                    raise
                except Exception as error:
                    failed += 1
                    fatal = isinstance(error, FATAL_API_ERRORS)
                    if fatal:
                        self.aborted = True
                    self.file_state.emit(index, "실패", "")
                    self.write_log(f"[오류] {path}: {error}")
                    if fatal:
                        break
                self.progress.emit(int((index + 1) / len(self.paths) * 1000))
        except CorrectionCancelled:
            self.aborted = True
            self.write_log("[중단] 미완료 파일은 저장하지 않았습니다.")
        except Exception as error:
            self.aborted = True
            self.write_log(f"[오류] {error}")
            for index in range(completed + failed, len(self.paths)):
                self.file_state.emit(index, "실패", "")
                failed += 1
        finally:
            if corrector is not None:
                corrector.close()
            self.api_key = ""
            pending = len(self.paths) - completed - failed
            state = "중단" if self.isInterruptionRequested() else "작업 종료"
            self.summary.emit(
                f"{state}: 저장 {completed}개 (검토 {review}개), 실패 {failed}개, 미처리 {pending}개"
            )


class EvaluationNumberItem(QTableWidgetItem):
    def __init__(self, value: int | float | None, *, decimals: int | None = None) -> None:
        text = "—" if value is None else str(value) if decimals is None else f"{value:.{decimals}f}"
        super().__init__(text)
        self.setData(Qt.ItemDataRole.UserRole, value if value is not None else -1)

    def __lt__(self, other: QTableWidgetItem) -> bool:
        return self.data(Qt.ItemDataRole.UserRole) < other.data(Qt.ItemDataRole.UserRole)


class EvaluationWorker(QThread):
    file_result = Signal(int, str, object)
    progress = Signal(int)
    log = Signal(str)
    summary = Signal(str)

    def __init__(self, paths: list[Path], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.paths = list(paths)

    def run(self) -> None:
        completed = partial = reviewed = failed = errors = 0
        try:
            check_cancelled(self.isInterruptionRequested)
            spacer = create_spacer()
            for index, path in enumerate(self.paths):
                check_cancelled(self.isInterruptionRequested)
                self.file_result.emit(index, "평가 중", None)
                try:
                    result = evaluate_file(
                        path, spacer, is_cancelled=self.isInterruptionRequested,
                        on_progress=lambda done, total, row=index: self.progress.emit(
                            int((row + done / total) / len(self.paths) * 1000)
                        ),
                        on_log=self.log.emit,
                    )
                    for warning in result.warnings:
                        self.log.emit(f"[제외] {path}: {warning}")
                    if result.review_logs:
                        reviewed += 1
                        for entry in result.review_logs:
                            self.log.emit(f"[검토] {path}\n{entry}")
                    if result.skipped_line_count and not result.character_count:
                        failed += 1
                        self.file_result.emit(index, "평가 불가", result)
                    else:
                        completed += 1
                        errors += result.error_count
                        if result.skipped_line_count:
                            partial += 1
                        state = "부분 평가" if result.skipped_line_count else "완료"
                        self.file_result.emit(index, "검토 필요" if result.review_logs else state, result)
                except CorrectionCancelled:
                    self.file_result.emit(index, "중단", None)
                    raise
                except Exception as error:
                    failed += 1
                    self.file_result.emit(index, "실패", None)
                    self.log.emit(f"[오류] {path}: {error}")
                self.progress.emit(int((index + 1) / len(self.paths) * 1000))
        except CorrectionCancelled:
            pass
        except Exception as error:
            self.log.emit(f"[오류] Kiwi 초기화: {error}")
            for index in range(len(self.paths)):
                self.file_result.emit(index, "실패", None)
            failed = len(self.paths)
        finally:
            state = "평가 중단" if self.isInterruptionRequested() else "평가 종료"
            partial_summary = f" (부분 평가 {partial}개)" if partial else ""
            review_summary = f", 검토 필요 {reviewed}개" if reviewed else ""
            self.summary.emit(
                f"{state}: 완료 {completed}개{partial_summary}, 오류 {errors}건, 실패 {failed}개, "
                f"미처리 {len(self.paths) - completed - failed}개{review_summary}"
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
        self.completed_paths: set[Path] = set()
        self.review_results: dict[Path, tuple[str, list[str]]] = {}
        self.evaluation_reviews: dict[Path, tuple[str, ...]] = {}
        self.worker: QThread | None = None
        self.correction_workers: dict[Path, CorrectionWorker] = {}
        self._file_progress: dict[Path, int] = {}
        self._correction_stop_reason = "작업 종료"
        self.file_loader: FileDiscoveryTask | None = None
        self._file_requests: list[list[Path]] = []
        self._correction_active = False
        self._session_paths: list[Path] = []
        self._attempted_paths: set[Path] = set()
        self.close_pending = False
        self.keys: dict[str, str] = {}
        for service in BASE_URLS:
            try:
                saved_key = self.settings.load_key(service)
            except SettingsError as error:
                saved_key = None
                startup_errors.append(str(error))
            self.keys[service] = saved_key or ""
        self._persisted_keys = dict(self.keys)
        self.current_service = self.preferences.service
        self.key_save_timer = QTimer(self)
        self.key_save_timer.setSingleShot(True)
        self.key_save_timer.setInterval(600)
        self.key_save_timer.timeout.connect(self._save_current_key)
        self.setWindowTitle(f"SRT Spellchecker v{__version__}")
        self.setWindowIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.resize(980, 800)
        self.setMinimumSize(720, 620)
        self._build_ui()
        self.service_combo.setCurrentText(self.current_service)
        self.key_edit.setText(self.keys[self.current_service])
        self.wrap_check.setChecked(self.preferences.wrap)
        self.length_spin.setValue(self.preferences.max_line_length)
        self.concurrency_spin.setValue(self.preferences.concurrent_files)
        self.batch_size_spin.setValue(self.preferences.batch_size)
        self._restore_model()
        try:
            self._restore_worklist()
        except SettingsError as error:
            startup_errors.append(str(error))
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
        heading = QLabel(self.windowTitle())
        font = heading.font()
        font.setPointSize(18)
        font.setBold(True)
        heading.setFont(font)
        root.addWidget(heading)

        mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.correction_radio = QRadioButton("LLM 교정")
        self.evaluation_radio = QRadioButton("간이평가")
        for button in (self.correction_radio, self.evaluation_radio):
            self.mode_group.addButton(button)
            mode_layout.addWidget(button)
        self.correction_radio.setChecked(True)
        self.evaluation_radio.toggled.connect(self._mode_changed)
        mode_layout.addStretch()
        root.addLayout(mode_layout)

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
        self.concurrency_spin = QSpinBox()
        self.concurrency_spin.setRange(1, 32)
        self.concurrency_spin.setSuffix(" 개")
        self.concurrency_spin.setToolTip("동시에 교정할 파일 수. 1개는 순차 처리, 변경값은 다음 교정부터 적용")
        self.concurrency_spin.valueChanged.connect(self._save_preferences)
        form.addRow("동시 교정 파일 수", self.concurrency_spin)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, MAX_BATCH_SIZE)
        self.batch_size_spin.setValue(BATCH_SIZE)
        self.batch_size_spin.setSuffix(" 개")
        self.batch_size_spin.setToolTip("API 요청 한 번에 보낼 자막 블록 수. 기본 25개, 변경값은 다음 교정부터 적용")
        self.batch_size_spin.valueChanged.connect(self._save_preferences)
        form.addRow("요청당 자막 수", self.batch_size_spin)
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

        self.loading_panel = QWidget()
        loading_layout = QHBoxLayout(self.loading_panel)
        loading_layout.setContentsMargins(0, 0, 0, 0)
        self.loading_label = QLabel("파일 불러오는 중 · SRT 0개 발견")
        loading_layout.addWidget(self.loading_label, 1)
        self.loading_progress = QProgressBar()
        self.loading_progress.setRange(0, 0)
        self.loading_progress.setTextVisible(False)
        self.loading_progress.setFixedWidth(180)
        self.loading_progress.setAccessibleName("자막 파일 탐색 진행 중")
        loading_layout.addWidget(self.loading_progress)
        self.loading_panel.hide()
        root.addWidget(self.loading_panel)

        self.evaluation_filter_panel = QWidget()
        filter_layout = QHBoxLayout(self.evaluation_filter_panel)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        threshold_label = QLabel("1,000자당 오류 수")
        self.evaluation_threshold_spin = QDoubleSpinBox()
        self.evaluation_threshold_spin.setRange(0, 1_000_000)
        self.evaluation_threshold_spin.setDecimals(2)
        self.evaluation_threshold_spin.setSuffix(" 이상")
        self.evaluation_threshold_spin.setAccessibleName("남길 최소 1,000자당 오류 수")
        threshold_label.setBuddy(self.evaluation_threshold_spin)
        self.filter_evaluation_button = self._button(
            "기준 이상만 남기기", QStyle.StandardPixmap.SP_DialogDiscardButton, self.filter_evaluation_results,
        )
        self.filter_evaluation_button.setToolTip(
            "반올림 전 오류율로 비교. 기준 미만과 평가 실패·미평가·오류율 없는 항목을 두 목록에서 제거. 원본 파일 유지"
        )
        filter_layout.addWidget(threshold_label)
        filter_layout.addWidget(self.evaluation_threshold_spin)
        filter_layout.addWidget(self.filter_evaluation_button)
        filter_layout.addStretch()
        self.evaluation_filter_panel.hide()
        root.addWidget(self.evaluation_filter_panel)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.table = FileTable()
        self.table.paths_dropped.connect(self.add_paths)
        self.table.itemSelectionChanged.connect(self._update_controls)
        self.table.cellDoubleClicked.connect(self.open_subtitle_target)
        splitter.addWidget(self.table)
        self.evaluation_table = QTableWidget(0, 6)
        self.evaluation_table.setItemDelegate(TableItemDelegate(self.evaluation_table))
        self.evaluation_table.setMouseTracking(True)
        self.evaluation_table.cellDoubleClicked.connect(self.open_evaluation_target)
        self.evaluation_table.setHorizontalHeaderLabels([
            "평가 자막", "평가 상태", "띄어쓰기 오류 수", "글자 수", "1,000자당 오류 수", "검토",
        ])
        self.evaluation_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.evaluation_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.evaluation_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.evaluation_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        for column in (3, 4):
            self.evaluation_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.evaluation_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.evaluation_table.setColumnWidth(5, 44)
        self.evaluation_table.horizontalHeaderItem(3).setToolTip(
            "평가 대상 본문의 문자 수: 공백·줄바꿈·서식 태그·줄 시작 대사 표식 제외, 문장부호 포함"
        )
        self.evaluation_table.horizontalHeaderItem(4).setToolTip(
            "오류 수 ÷ 글자 수 × 1,000 (글자 수가 0이면 —). 평가 종료 후 제목을 눌러 정렬"
        )
        self.evaluation_table.setToolTip(
            "Kiwi 추정치: 공백 삽입·삭제 위치당 1건. 전체 평가 종료 후 오류 수 내림차순 정렬. 열 제목으로 정렬 변경"
        )
        self.evaluation_table.hide()
        splitter.addWidget(self.evaluation_table)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(10000)
        self.log_view.setMinimumHeight(80)
        splitter.addWidget(self.log_view)
        splitter.setSizes([300, 200, 140])
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
        self.export_button = self._button(
            "CSV 내보내기", QStyle.StandardPixmap.SP_DialogSaveButton, self.export_evaluation_csv,
        )
        self.export_button.setToolTip("평가 테이블의 열과 현재 정렬 순서를 그대로 CSV로 저장")
        self.export_button.hide()
        options.addWidget(self.export_button)
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

    def _mode_changed(self) -> None:
        evaluation = self.evaluation_radio.isChecked()
        self.settings_panel.setVisible(not evaluation)
        self.wrap_check.setVisible(not evaluation)
        self.length_spin.setVisible(not evaluation)
        self.evaluation_table.setVisible(evaluation)
        self.export_button.setVisible(evaluation)
        self.evaluation_filter_panel.setVisible(evaluation)
        for column in (1, 2, 3):
            self.table.setColumnHidden(column, evaluation)
        self.start_button.setText("평가 시작" if evaluation else "교정 시작")
        self.progress_bar.setValue(0)
        self.status_label.setText("간이평가 대기" if evaluation else "교정 대기")
        self._update_controls()

    def _update_controls(self) -> None:
        busy = bool(self.worker or self.file_loader or self._correction_active or self.correction_workers)
        can_add = not self.close_pending and self.worker is None
        self.settings_panel.setEnabled(not busy)
        self.correction_radio.setEnabled(not busy)
        self.evaluation_radio.setEnabled(not busy)
        self.fetch_button.setEnabled(not busy and bool(self.key_edit.text().strip()))
        self.model_combo.setEnabled(not busy and self.model_combo.count() > 0)
        for widget in (self.files_button, self.folder_button, self.table):
            widget.setEnabled(can_add)
        self.wrap_check.setEnabled(not busy)
        self.length_spin.setEnabled(not busy and self.wrap_check.isChecked())
        self.remove_button.setEnabled(not busy and bool(self.table.selectedItems()))
        self.clear_button.setEnabled(not busy and bool(self.paths))
        self.export_button.setEnabled(
            not busy and self.evaluation_radio.isChecked() and self.evaluation_table.rowCount() > 0
        )
        self.evaluation_filter_panel.setEnabled(
            not busy and not self.close_pending and self.evaluation_radio.isChecked()
            and self.evaluation_table.isSortingEnabled() and self.evaluation_table.rowCount() > 0
        )
        self.start_button.setEnabled(
            not busy and (
                bool(self.paths) if self.evaluation_radio.isChecked() else
                any(path not in self.completed_paths for path in self.paths)
                and bool(self.key_edit.text().strip()) and self.selected_model() is not None
            )
        )
        self.cancel_button.setEnabled(any(
            worker is not None and not worker.isInterruptionRequested()
            for worker in (self.worker, self.file_loader, *self.correction_workers.values())
        ))
        self.count_label.setText(f"자막 {len(self.paths)}개")
        for row, path in enumerate(self.paths):
            self._update_row_actions(row, path)

    def _update_row_actions(self, row: int, path: Path) -> None:
        state = self.table.item(row, 1).text()
        self.table.cellWidget(row, 3).setEnabled(state == "검토 필요" and path in self.review_results)
        self.table.cellWidget(row, 4).setEnabled(
            state == "대기" and not self.close_pending
            and self.worker is None
        )

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

    def _restore_worklist(self) -> None:
        worklist = self.settings.load_worklist()
        self._files_loaded([Path(item.path) for item in worklist.files])
        for item in worklist.files:
            path = Path(item.path).resolve()
            if item.review_logs:
                self._store_review(path, item.output, item.review_logs)
            state = "대기" if item.state == "교정 중" else item.state
            self._file_state(self.paths.index(path), state, item.output)
        if self.paths:
            self.status_label.setText(f"작업 목록 {len(self.paths)}개 복원")

    def _save_worklist(self) -> None:
        if self._restoring:
            return
        worklist = SavedWorklist(files=[
            SavedWorkFile(
                path=str(path),
                state=self.table.item(row, 1).text(),
                output=self.table.item(row, 2).toolTip(),
                review_logs=self.review_results.get(path, ("", []))[1],
            )
            for row, path in enumerate(self.paths)
        ])
        try:
            self.settings.save_worklist(worklist)
        except SettingsError as error:
            self.show_error(str(error))

    def _save_preferences(self) -> None:
        if self._restoring:
            return
        self.preferences.wrap = self.wrap_check.isChecked()
        self.preferences.max_line_length = self.length_spin.value()
        self.preferences.concurrent_files = self.concurrency_spin.value()
        self.preferences.batch_size = self.batch_size_spin.value()
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
            if isinstance(worker, EvaluationWorker):
                self.evaluation_table.horizontalHeader().setSortIndicator(2, Qt.SortOrder.DescendingOrder)
                self.evaluation_table.setSortingEnabled(True)
            if worker.isInterruptionRequested():
                self.status_label.setText("작업 중단됨")
            worker.deleteLater()
        self._advance_correction()
        self._update_controls()
        if self.close_pending and self.file_loader is None and not self.correction_workers:
            self.close()

    def show_error(self, message: str) -> None:
        for key in [self.key_edit.text().strip(), *self.keys.values()]:
            if key:
                message = message.replace(key, "[API KEY]")
        self.status_label.setText("오류: " + message)
        self.log_view.appendPlainText("[오류] " + message)

    def load_models(self) -> None:
        if self.worker is not None or self.file_loader is not None or self._correction_active or self.correction_workers or not self.key_edit.text().strip():
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
        if not paths or self.close_pending:
            return
        if self.worker is not None:
            return
        self._file_requests.append(list(paths))
        self._load_next_files()

    def _load_next_files(self) -> None:
        if self.file_loader is not None or not self._file_requests:
            return
        paths = self._file_requests.pop(0)
        worker = FileDiscoveryTask(paths, self)
        worker.progress.connect(self._file_discovery_progress)
        worker.result.connect(self._files_loaded)
        worker.failed.connect(self.show_error)
        worker.finished.connect(self._file_loader_finished)
        self.file_loader = worker
        self.loading_label.setText("파일 불러오는 중 · SRT 0개 발견")
        self.loading_label.setToolTip(str(paths[0]))
        self.loading_panel.show()
        if not self._correction_active:
            self.status_label.setText("자막 파일 탐색 중")
        self._update_controls()
        worker.start()

    def _file_discovery_progress(self, count: int, location: str) -> None:
        stopping = self.file_loader is not None and self.file_loader.isInterruptionRequested()
        state = "파일 탐색 중단 대기" if stopping else "파일 불러오는 중"
        self.loading_label.setText(f"{state} · SRT {count:,}개 발견")
        self.loading_label.setToolTip(location)

    def _file_loader_finished(self) -> None:
        worker = self.file_loader
        self.file_loader = None
        if worker is not None:
            worker.wait()
            if worker.cancelled and self.worker is None and not self.correction_workers:
                self.status_label.setText("파일 불러오기 중단됨")
            worker.deleteLater()
        self._load_next_files()
        self.loading_panel.setVisible(self.file_loader is not None)
        self._advance_correction()
        self._update_controls()
        if self.close_pending and self.worker is None and self.file_loader is None and not self.correction_workers:
            self.close()

    def _files_loaded(self, paths: list[Path]) -> None:
        known = set(self.paths)
        added = 0
        for path in paths:
            path = path.resolve()
            if path in known or (path in self.completed_paths and self.correction_radio.isChecked()):
                continue
            known.add(path)
            self.paths.append(path)
            row = self.table.rowCount()
            self.table.insertRow(row)
            item = QTableWidgetItem(str(Path(path.parent.name) / path.name))
            item.setToolTip(str(path))
            self.table.setItem(row, 0, item)
            self.table.setItem(row, 1, QTableWidgetItem("대기"))
            self.table.setItem(row, 2, QTableWidgetItem(""))
            for column, icon, tooltip, action in (
                (3, QStyle.StandardPixmap.SP_FileDialogContentsView, "파일별 검토 내역", self.show_review),
                (4, QStyle.StandardPixmap.SP_TrashIcon, "대기열에서 제거 (원본 파일 유지)", self.remove_waiting_file),
            ):
                button = QToolButton()
                button.setIcon(self.style().standardIcon(icon))
                button.setToolTip(tooltip)
                button.setAccessibleName(tooltip)
                button.clicked.connect(lambda _checked=False, target=path, callback=action: callback(target))
                self.table.setCellWidget(row, column, button)
            if self._correction_active:
                self._session_paths.append(path)
            added += 1
        if self._correction_active:
            self.log_view.appendPlainText(f"[대기열] 자막 {added}개 추가")
        else:
            self.status_label.setText(f"자막 {added}개 추가" if paths else "SRT 파일이 없습니다.")
        self._update_controls()
        self._save_worklist()

    def _store_review(self, path: Path, output: str, logs: list[str]) -> None:
        self.review_results[path] = (output, list(logs))

    def show_review(self, path: Path) -> None:
        result = self.review_results.get(path)
        if result is None:
            return
        output, logs = result
        self._show_review_dialog(path, output, logs)

    def _show_review_dialog(self, path: Path, output: str, logs: list[str] | tuple[str, ...]) -> None:
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle(f"검토 내역 - {path.name}")
        dialog.resize(760, 480)
        layout = QVBoxLayout(dialog)
        details = QPlainTextEdit()
        details.setReadOnly(True)
        details.setPlainText(f"원본: {path}\n결과: {output}\n검토 {len(logs)}건\n\n" + "\n".join(logs))
        layout.addWidget(details)
        dialog.show()

    def remove_waiting_file(self, path: Path) -> None:
        if path not in self.paths or self.close_pending:
            return
        if self.worker is not None:
            return
        row = self.paths.index(path)
        if self.table.item(row, 1).text() != "대기":
            return
        if path in self.correction_workers:
            return
        del self.paths[row]
        self.table.removeRow(row)
        if path in self._session_paths:
            self._session_paths.remove(path)
        self._attempted_paths.discard(path)
        self.review_results.pop(path, None)
        self.evaluation_reviews.pop(path, None)
        if self._correction_active:
            self._update_progress()
        self.log_view.appendPlainText(f"[대기열] 제거: {path}")
        self._update_controls()
        self._save_worklist()

    def remove_selected(self) -> None:
        if self.worker is not None or self.file_loader is not None or self._correction_active or self.correction_workers:
            return
        rows = sorted({item.row() for item in self.table.selectedItems()}, reverse=True)
        for row in rows:
            self.review_results.pop(self.paths[row], None)
            self.evaluation_reviews.pop(self.paths[row], None)
            del self.paths[row]
            self.table.removeRow(row)
        self._update_controls()
        self._save_worklist()

    def clear_files(self) -> None:
        if self.worker is None and self.file_loader is None and not self._correction_active and not self.correction_workers:
            self.paths.clear()
            self.table.setRowCount(0)
            self.review_results.clear()
            self.evaluation_reviews.clear()
            self.evaluation_table.setRowCount(0)
            self.progress_bar.setValue(0)
            self._update_controls()
            self._save_worklist()

    def open_subtitle_target(self, row: int, column: int) -> None:
        if column not in (0, 1, 2):
            return
        item = self.table.item(row, 0 if column == 1 else column)
        path = item.toolTip() if item is not None else ""
        if path:
            if column == 1:
                path = str(Path(path).parent)
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def open_evaluation_target(self, row: int, column: int) -> None:
        if column not in (0, 1):
            return
        item = self.evaluation_table.item(row, 0)
        path = item.toolTip() if item is not None else ""
        if path:
            if column == 1:
                path = str(Path(path).parent)
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def start_correction(self) -> None:
        if self.evaluation_radio.isChecked():
            self.start_evaluation()
            return
        model = self.selected_model()
        paths = [path for path in self.paths if path not in self.completed_paths]
        if (
            self.worker is not None or self.file_loader is not None or self._correction_active
            or self.correction_workers
            or not paths or model is None or not self.key_edit.text().strip()
        ):
            return
        self._save_current_key()
        self._correction_active = True
        self._session_paths = paths
        self._attempted_paths.clear()
        self._file_progress.clear()
        self._correction_stop_reason = "작업 종료"
        for path in paths:
            self._file_state(self.paths.index(path), "대기", "")
        self.progress_bar.setValue(0)
        self.log_view.clear()
        self.log_view.appendPlainText(f"[모델] {self.current_service} / {model.id}")
        self.log_view.appendPlainText(f"[추론] {reasoning_label(self.current_service, model)}")
        self.log_view.appendPlainText(f"[동시 교정] 최대 {self.concurrency_spin.value()}개 파일")
        self.log_view.appendPlainText(f"[배치 크기] 요청당 자막 {self.batch_size_spin.value()}개")
        self._advance_correction()

    def start_evaluation(self) -> None:
        if self.worker is not None or self.file_loader is not None or self._correction_active or self.correction_workers or not self.paths:
            return
        self.evaluation_reviews.clear()
        self.evaluation_table.setSortingEnabled(False)
        self.evaluation_table.setRowCount(0)
        for row, path in enumerate(self.paths):
            self.evaluation_table.insertRow(row)
            item = QTableWidgetItem(str(Path(path.parent.name) / path.name))
            item.setToolTip(str(path))
            self.evaluation_table.setItem(row, 0, item)
            self._evaluation_result(row, "대기", None)
        self.progress_bar.setValue(0)
        self.log_view.clear()
        worker = EvaluationWorker(self.paths, self)
        worker.file_result.connect(self._evaluation_result)
        worker.progress.connect(self.progress_bar.setValue)
        worker.log.connect(self.log_view.appendPlainText)
        worker.summary.connect(self._summary)
        self._start_worker(worker, "간이평가 중: Kiwi 준비 및 띄어쓰기 분석")

    def _evaluation_result(self, row: int, state: str, result: EvaluationResult | None) -> None:
        state_item = QTableWidgetItem(state)
        state_item.setData(Qt.ItemDataRole.UserRole, state)
        review_logs = result.review_logs if result is not None else ()
        label = state
        if review_logs and state != "검토 필요":
            label += " / 검토 필요"
        if result is not None and result.skipped_line_count:
            partial_label = "부분 평가, " if state == "검토 필요" else ""
            label += f" ({partial_label}{result.skipped_line_count}줄 제외)"
        state_item.setText(label)
        if result is not None:
            conversion = (
                (f"{result.converted_from} -> UTF-8 BOM (원본 교체 완료)",)
                if result.converted_from else ()
            )
            state_item.setToolTip("\n\n".join(conversion + result.warnings + review_logs))
        self.evaluation_table.setItem(row, 1, state_item)
        review_item = QTableWidgetItem("\n\n".join(review_logs))
        review_item.setToolTip(review_item.text())
        self.evaluation_table.setItem(row, 5, review_item)
        review_button = QToolButton()
        review_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView))
        review_button.setToolTip(f"간이평가 검토 내역 ({len(review_logs)}건)")
        review_button.setAccessibleName("간이평가 검토 내역")
        path_item = self.evaluation_table.item(row, 0)
        path_text = path_item.toolTip() if path_item is not None else ""
        if review_logs and path_text:
            self.evaluation_reviews[Path(path_text)] = review_logs
        review_button.setEnabled(bool(review_logs and path_text))
        review_button.clicked.connect(
            lambda _checked=False, path=path_text, logs=review_logs: self._show_review_dialog(Path(path), path, logs)
        )
        self.evaluation_table.setCellWidget(row, 5, review_button)
        if state == "평가 불가":
            result = None
        self.evaluation_table.setItem(row, 2, EvaluationNumberItem(result.error_count if result else None))
        self.evaluation_table.setItem(row, 3, EvaluationNumberItem(result.character_count if result else None))
        self.evaluation_table.setItem(row, 4, EvaluationNumberItem(
            result.errors_per_1000 if result else None, decimals=2,
        ))

    def filter_evaluation_results(self) -> None:
        if not self.filter_evaluation_button.isEnabled():
            return
        threshold = self.evaluation_threshold_spin.value()
        kept_paths: set[Path] = set()
        table = self.evaluation_table
        for row in range(table.rowCount()):
            path_item = table.item(row, 0)
            state_item = table.item(row, 1)
            rate_item = table.item(row, 4)
            if path_item is None or state_item is None or rate_item is None:
                continue
            rate = rate_item.data(Qt.ItemDataRole.UserRole)
            if (
                state_item.data(Qt.ItemDataRole.UserRole) in {"완료", "부분 평가", "검토 필요"}
                and rate is not None and rate >= threshold and path_item.toolTip()
            ):
                kept_paths.add(Path(path_item.toolTip()))
        kept_paths.intersection_update(self.paths)
        removed_count = len(self.paths) - len(kept_paths)
        for row in range(len(self.paths) - 1, -1, -1):
            path = self.paths[row]
            if path not in kept_paths:
                self.review_results.pop(path, None)
                self.evaluation_reviews.pop(path, None)
                self._attempted_paths.discard(path)
                self._file_progress.pop(path, None)
                del self.paths[row]
                self.table.removeRow(row)
        self._session_paths = [path for path in self._session_paths if path in kept_paths]
        for row in range(table.rowCount() - 1, -1, -1):
            item = table.item(row, 0)
            if item is None or not item.toolTip() or Path(item.toolTip()) not in kept_paths:
                table.removeRow(row)
        message = (
            f"평가 필터: 1,000자당 오류 수 {threshold:.2f} 이상, "
            f"유지 {len(self.paths)}개 / 목록 제거 {removed_count}개"
        )
        self.status_label.setText(message)
        self.log_view.appendPlainText(message)
        self._update_controls()
        self._save_worklist()

    def export_evaluation_csv(self) -> None:
        if not self.export_button.isEnabled():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "간이평가 CSV 저장", "간이평가.csv", "CSV (*.csv)",
        )
        if not path:
            return
        table = self.evaluation_table
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow([
            table.horizontalHeaderItem(column).text() for column in range(table.columnCount())
        ])
        for row in range(table.rowCount()):
            writer.writerow([
                table.item(row, column).text() if table.item(row, column) is not None else ""
                for column in range(table.columnCount())
            ])
        data = buffer.getvalue().encode("utf-8-sig")
        output = QSaveFile(path)
        try:
            if not output.open(QIODevice.OpenModeFlag.WriteOnly):
                raise OSError(output.errorString())
            if output.write(data) != len(data):
                raise OSError(output.errorString())
            if not output.commit():
                raise OSError(output.errorString())
        except OSError as error:
            output.cancelWriting()
            self.show_error(f"CSV 저장 실패: {error}")
            return
        self._summary(f"CSV 저장 완료: {path} (자막 {table.rowCount()}개)")

    def _advance_correction(self) -> None:
        if not self._correction_active or self.worker is not None:
            return
        if any(worker.aborted for worker in self.correction_workers.values()):
            self._stop_corrections("작업 종료")
            return
        paths = [path for path in self._session_paths if path not in self._attempted_paths]
        if not paths:
            if not self.correction_workers and self.file_loader is None and not self._file_requests:
                self._correction_active = False
                self._summarize_correction("작업 종료")
            elif self.correction_workers:
                self.status_label.setText(f"교정 중: 동시 {len(self.correction_workers)}개 파일")
            return
        model = self.selected_model()
        if model is None:
            self._correction_active = False
            return
        available = self.concurrency_spin.value() - len(self.correction_workers)
        for path in paths[:max(0, available)]:
            self._start_correction_file(path, model)
        self._update_progress()
        self.status_label.setText(f"교정 중: 동시 {len(self.correction_workers)}개 파일")
        self._update_controls()

    def _start_correction_file(self, path: Path, model: ModelInfo) -> None:
        self._attempted_paths.add(path)
        self._file_progress[path] = 0
        self.review_results.pop(path, None)
        self._file_state(self.paths.index(path), "교정 중", "")
        worker = CorrectionWorker(
            [path], self.current_service, self.key_edit.text().strip(), model,
            self.length_spin.value() if self.wrap_check.isChecked() else None, self,
            batch_size=self.batch_size_spin.value(),
        )
        self.correction_workers[path] = worker
        worker.log.connect(lambda message: self.log_view.appendPlainText(f"[{path}] {message}"))
        worker.file_state.connect(lambda _row, state, output: self._file_state(self.paths.index(path), state, output))
        worker.review_ready.connect(lambda _row, output, logs: self._store_review(path, output, logs))
        worker.progress.connect(lambda value: self._update_progress(path, value))
        worker.finished.connect(self._correction_finished)
        worker.start()

    def _stop_corrections(self, reason: str) -> None:
        self._correction_active = False
        self._correction_stop_reason = reason
        for worker in self.correction_workers.values():
            worker.requestInterruption()

    def _correction_finished(self) -> None:
        worker = self.sender()
        path = worker.paths[0]
        worker.wait()
        del self.correction_workers[path]
        if worker.aborted and self._correction_active:
            self._stop_corrections("작업 종료")
        worker.deleteLater()
        if self._correction_active:
            self._advance_correction()
        elif not self.correction_workers:
            self._summarize_correction(self._correction_stop_reason)
        self._update_controls()
        if self.close_pending and self.worker is None and self.file_loader is None and not self.correction_workers:
            self.close()

    def _summarize_correction(self, state: str) -> None:
        states = [self.table.item(self.paths.index(path), 1).text() for path in self._session_paths]
        review = states.count("검토 필요")
        completed = states.count("완료") + review
        failed = states.count("실패")
        self._summary(
            f"{state}: 저장 {completed}개 (검토 {review}개), 실패 {failed}개, "
            f"미처리 {len(states) - completed - failed}개"
        )

    def _update_progress(self, path: Path | None = None, value: int = 0) -> None:
        if path is not None:
            self._file_progress[path] = value
        total = len(self._session_paths)
        self.progress_bar.setValue(
            int(sum(self._file_progress.get(path, 0) for path in self._session_paths) / total) if total else 0
        )

    def _file_state(self, row: int, state: str, output: str) -> None:
        path = self.paths[row]
        if state in {"완료", "검토 필요"}:
            logs = list(dict.fromkeys([
                *self.evaluation_reviews.get(path, ()),
                *self.review_results.get(path, ("", []))[1],
            ]))
            if logs:
                self._store_review(path, output, logs)
                state = "검토 필요"
        self.table.item(row, 1).setText(state)
        output_path = Path(output)
        self.table.item(row, 2).setText(str(Path(output_path.parent.name) / output_path.name) if output else "")
        self.table.item(row, 2).setToolTip(output)
        if state in {"완료", "검토 필요"}:
            self.completed_paths.add(self.paths[row])
        self.table.set_row_active(row, state == "교정 중")
        self._update_row_actions(row, self.paths[row])
        self._save_worklist()

    def _summary(self, message: str) -> None:
        self.status_label.setText(message)
        self.log_view.appendPlainText(message)

    def cancel_work(self) -> None:
        self._stop_corrections("중단")
        self._file_requests.clear()
        for worker in (self.worker, self.file_loader):
            if worker is not None:
                worker.requestInterruption()
        if self.file_loader is not None:
            self.loading_label.setText(self.loading_label.text().replace("파일 불러오는 중", "파일 탐색 중단 대기"))
        if self.worker is not None or self.file_loader is not None or self.correction_workers:
            self.status_label.setText(
                "중단 요청됨: 진행 중인 분석 종료 대기" if isinstance(self.worker, EvaluationWorker)
                else "중단 요청됨: 파일 탐색 종료 대기" if self.worker is None and not self.correction_workers
                else "중단 요청됨: 진행 중인 API 요청 종료 대기"
            )
            self._update_controls()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker is not None or self.file_loader is not None or self.correction_workers:
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
    app = QApplication(sys.argv)
    app.setApplicationName("SRT Spellchecker")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
