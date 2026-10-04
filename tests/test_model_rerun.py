from __future__ import annotations

from threading import Event

import pytest
from PySide6.QtCore import QEvent

import app_settings
import srt_spellchecker_gui as gui
from ai_services import ModelInfo
from test_gui import EchoService, app, finish_work, isolated_settings, prepare_model, wait_until, window


def select_model(window, model_id, name=""):
    window._models_loaded([ModelInfo(model_id, name)])
    window.model_combo.setCurrentIndex(0)


def mark_completed(window, path, model_id, *, service="OpenAI", state="완료"):
    window._store_overview(path, app_settings.CorrectionOverview(
        service=service, model_id=model_id, elapsed_seconds=1.0,
    ))
    window._file_state(window.paths.index(path), state, str(path.with_stem(path.stem + "_revised")))
    window._update_controls()


@pytest.mark.parametrize("state", ["완료", "검토 필요"])
def test_completed_file_runs_again_only_after_model_changes(window, app, tmp_path, monkeypatch, state):
    path = tmp_path / "subtitle.srt"
    prepare_model(window)
    window._files_loaded([path])
    mark_completed(window, path, "test-model", state=state)
    overview = window.correction_overviews[path]
    processed = []

    def correct_file(source, *_args, **_kwargs):
        processed.append(source)
        return source.with_stem(source.stem + "_revised"), []

    monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
    monkeypatch.setattr(gui, "correct_file", correct_file)
    assert not window.start_button.isEnabled()
    window.start_correction()
    assert not window.correction_workers
    select_model(window, "test-model", "새 표시 이름")
    assert not window.start_button.isEnabled()
    select_model(window, "replacement-model")
    assert window.start_button.isEnabled()
    assert window.correction_overviews[path] == overview
    select_model(window, "test-model")
    assert not window.start_button.isEnabled()
    select_model(window, "replacement-model")
    window.start_correction()
    finish_work(window, app)
    assert processed == [path]
    assert window.correction_overviews[path].model_id == "replacement-model"
    assert not window.start_button.isEnabled()
    select_model(window, "test-model")
    assert window.start_button.isEnabled()


def test_changing_provider_allows_same_model_id_again(window, app, tmp_path, monkeypatch):
    path = tmp_path / "subtitle.srt"
    prepare_model(window)
    window._files_loaded([path])
    mark_completed(window, path, "test-model")
    window.service_combo.setCurrentText("OpenRouter")
    prepare_model(window)
    assert window.start_button.isEnabled()
    monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
    monkeypatch.setattr(gui, "correct_file", lambda source, *_args, **_kwargs: (
        source.with_stem(source.stem + "_revised"), [],
    ))
    window.start_correction()
    finish_work(window, app)
    assert window.correction_overviews[path].service == "OpenRouter"
    assert not window.start_button.isEnabled()


def test_mixed_models_select_only_files_needing_selected_model(window, app, tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("same.srt", "different.srt", "waiting.srt")]
    prepare_model(window)
    window._files_loaded(paths)
    mark_completed(window, paths[0], "test-model")
    mark_completed(window, paths[1], "previous-model")
    original_overview = window.correction_overviews[paths[0]]
    processed = []

    def correct_file(source, *_args, **_kwargs):
        processed.append(source)
        return source.with_stem(source.stem + "_revised"), []

    monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
    monkeypatch.setattr(gui, "correct_file", correct_file)
    window.start_correction()
    finish_work(window, app)
    assert processed == paths[1:]
    assert window.correction_overviews[paths[0]] == original_overview
    assert not window.start_button.isEnabled()


@pytest.mark.parametrize("outcome", ["failure", "cancel"])
def test_failed_rerun_can_retry_same_model(window, app, tmp_path, monkeypatch, outcome):
    path = tmp_path / "subtitle.srt"
    prepare_model(window)
    window._files_loaded([path])
    mark_completed(window, path, "previous-model")

    def fail(*_args, **_kwargs):
        raise RuntimeError("교정 실패") if outcome == "failure" else gui.CorrectionCancelled()

    monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
    monkeypatch.setattr(gui, "correct_file", fail)
    window.start_correction()
    finish_work(window, app)
    assert window.table.item(0, 1).text() == ("실패" if outcome == "failure" else "중단")
    assert path not in window.completed_paths
    assert window.start_button.isEnabled()
    monkeypatch.setattr(gui, "correct_file", lambda source, *_args, **_kwargs: (
        source.with_stem(source.stem + "_revised"), [],
    ))
    window.start_correction()
    finish_work(window, app)
    assert window.table.item(0, 1).text() == "완료"
    assert not window.start_button.isEnabled()


def test_restored_completed_file_compares_saved_overview_model(window, app, tmp_path):
    path = tmp_path / "subtitle.srt"
    prepare_model(window)
    window._files_loaded([path])
    mark_completed(window, path, "test-model")
    select_model(window, "replacement-model")
    window._save_current_key()
    restored = gui.MainWindow()
    try:
        assert restored.start_button.isEnabled()
        select_model(restored, "test-model")
        assert not restored.start_button.isEnabled()
    finally:
        restored.close()
        restored.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("saved_model", [False, True])
def test_legacy_completion_uses_restored_selection_without_fabricating_overview(
    window, app, tmp_path, saved_model,
):
    path = tmp_path / "legacy.srt"
    if saved_model:
        prepare_model(window)
        window._save_current_key()
    window._files_loaded([path])
    window._file_state(0, "완료", str(path.with_stem("legacy_revised")))
    restored = gui.MainWindow()
    try:
        if saved_model:
            assert not restored.start_button.isEnabled()
            select_model(restored, "replacement-model")
        else:
            prepare_model(restored)
        assert restored.start_button.isEnabled()
        assert path not in restored.correction_overviews
        assert restored.settings.load_worklist().files[0].overview is None
    finally:
        restored.close()
        restored.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("remove", ["clear", "selected"])
def test_removed_completed_file_is_available_again_for_changed_model(window, tmp_path, remove):
    path = tmp_path / "subtitle.srt"
    prepare_model(window)
    window._files_loaded([path])
    mark_completed(window, path, "test-model")
    if remove == "clear":
        window.clear_files()
    else:
        window.table.selectRow(0)
        window.remove_selected()
    window._files_loaded([path])
    assert not window.paths
    select_model(window, "replacement-model")
    window._files_loaded([path])
    assert window.paths == [path]
    assert window.start_button.isEnabled()


def test_completed_file_can_join_active_queue_for_different_model(window, app, tmp_path, monkeypatch):
    completed = tmp_path / "completed.srt"
    added = tmp_path / "new.srt"
    prepare_model(window)
    window._files_loaded([completed])
    mark_completed(window, completed, "test-model")
    window.clear_files()
    select_model(window, "replacement-model")
    window._files_loaded([added])
    started = Event()
    release = Event()
    processed = []

    def correct_file(source, *_args, **_kwargs):
        processed.append(source)
        if source == added:
            started.set()
            assert release.wait(10)
        return source.with_stem(source.stem + "_revised"), []

    monkeypatch.setattr(gui, "ServiceCorrector", EchoService)
    monkeypatch.setattr(gui, "correct_file", correct_file)
    window.start_correction()
    try:
        wait_until(started.is_set)
        window._files_loaded([completed])
        window._advance_correction()
        assert window._session_paths == [added, completed]
    finally:
        release.set()
        finish_work(window, app)
    assert processed == [added, completed]
    assert window.correction_overviews[completed].model_id == "replacement-model"
    assert not window.start_button.isEnabled()
