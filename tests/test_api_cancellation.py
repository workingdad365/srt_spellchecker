from __future__ import annotations

import asyncio
import json
from threading import Event

import httpx
import openai
import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QDialog, QMessageBox, QPlainTextEdit

import ai_services as services
import srt_spellchecker as sc
import srt_spellchecker_gui as gui
from test_gui import (
    SAMPLE, app, finish_work, isolated_settings, prepare_model, wait_until, window,
)


def install_async_api(monkeypatch, handler):
    clients = []

    def create_client(*_args):
        client = openai.AsyncOpenAI(
            api_key="test-key", base_url="https://example.test/v1", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(services, "create_async_client", create_client)
    return clients


def request_payload(request):
    messages = json.loads(request.content)["messages"]
    return json.loads(messages[1]["content"].split("JSON:\n", 1)[1])


def echo_response(request):
    items = [
        {"id": item["id"], "corrected_lines": item["lines"]}
        for item in request_payload(request)
    ]
    return httpx.Response(200, json={"choices": [{
        "finish_reason": "stop", "index": 0,
        "message": {"role": "assistant", "content": json.dumps({"items": items})},
    }]})


def test_cancelled_before_invoke_does_not_send_request(monkeypatch):
    requests = []

    async def handler(request):
        requests.append(request)
        return echo_response(request)

    clients = install_async_api(monkeypatch, handler)
    corrector = services.ServiceCorrector(
        "OpenAI", "test-key", services.ModelInfo("test-model"),
        is_cancelled=lambda: True,
    )
    try:
        with pytest.raises(sc.CorrectionCancelled):
            corrector.invoke(sc.build_messages([{"id": 0, "lines": ["원문"]}], None))
        assert not requests
    finally:
        corrector.close()
    assert clients[0].is_closed()


def test_cancel_at_response_completion_is_not_retried_or_logged_as_failure(monkeypatch):
    cancelled = Event()
    requests = []
    logs = []

    async def handler(request):
        requests.append(request)
        cancelled.set()
        return echo_response(request)

    clients = install_async_api(monkeypatch, handler)
    corrector = services.ServiceCorrector(
        "OpenAI", "test-key", services.ModelInfo("test-model"),
        is_cancelled=cancelled.is_set,
    )
    try:
        with pytest.raises(sc.CorrectionCancelled):
            sc.correct_batch_with_retry(
                corrector, [{"id": 0, "lines": ["원문"]}], None,
                on_log=logs.append, is_cancelled=cancelled.is_set,
            )
        assert len(requests) == 1
        assert not logs
    finally:
        corrector.close()
    assert clients[0].is_closed()


def test_multiple_batches_reuse_event_loop_and_close_client(monkeypatch, tmp_path):
    loops = []

    async def handler(request):
        loops.append(asyncio.get_running_loop())
        await asyncio.sleep(0)
        return echo_response(request)

    clients = install_async_api(monkeypatch, handler)
    source = tmp_path / "multiple.srt"
    source.write_text(SAMPLE, encoding="utf-8")
    corrector = services.ServiceCorrector("OpenAI", "test-key", services.ModelInfo("test-model"))
    try:
        output, logs = sc.correct_file(source, corrector, batch_size=1)
        assert output.is_file()
        assert not logs
        assert len(loops) == 2
        assert loops[0] is loops[1]
        assert not loops[0].is_closed()
    finally:
        corrector.close()
    assert clients[0].is_closed()
    assert loops[0].is_closed()
    corrector.close()


@pytest.mark.parametrize("phase", ["before_run", "during_initialization"])
def test_gui_cancel_before_first_file_starts_clears_running_state(
    window, app, tmp_path, monkeypatch, phase,
):
    requests = []

    async def handler(request):
        requests.append(request)
        return echo_response(request)

    clients = install_async_api(monkeypatch, handler)
    if phase == "before_run":
        class InterruptedWorker(gui.CorrectionWorker):
            def run(self):
                self.requestInterruption()
                super().run()

        monkeypatch.setattr(gui, "CorrectionWorker", InterruptedWorker)
    else:
        create_client = services.create_async_client

        def interrupt_during_initialization(*args):
            client = create_client(*args)
            QThread.currentThread().requestInterruption()
            return client

        monkeypatch.setattr(services, "create_async_client", interrupt_during_initialization)

    paths = [tmp_path / name for name in ("first.srt", "pending.srt")]
    for path in paths:
        path.write_text(SAMPLE, encoding="utf-8")
    window._files_loaded(paths)
    prepare_model(window)
    window.start_correction()
    finish_work(window, app)
    assert [window.table.item(row, 1).text() for row in range(2)] == ["중단", "대기"]
    assert not requests
    assert len(clients) == (phase == "during_initialization")
    assert all(client.is_closed() for client in clients)
    assert not window.correction_overviews
    assert not list(tmp_path.glob("*_revised.srt"))
    assert window.start_button.isEnabled()


@pytest.mark.parametrize("stop", ["cancel", "close"])
def test_gui_cancels_all_inflight_requests_without_waiting_for_responses(
    window, app, tmp_path, monkeypatch, stop,
):
    emergency_release = Event()
    delay_requests = Event()
    delay_requests.set()
    requests = []
    cancelled_requests = []
    finished_requests = []
    loops = []

    async def handler(request):
        marker = request_payload(request)[0]["lines"][0]
        requests.append(marker)
        loops.append(asyncio.get_running_loop())
        try:
            if delay_requests.is_set():
                while not emergency_release.is_set():
                    await asyncio.sleep(0.01)
            return echo_response(request)
        except asyncio.CancelledError:
            cancelled_requests.append(marker)
            raise
        finally:
            finished_requests.append(marker)

    clients = install_async_api(monkeypatch, handler)
    paths = [tmp_path / f"{index}.srt" for index in range(3)]
    markers = [f"파일 {index}" for index in range(3)]
    for path, marker in zip(paths, markers):
        path.write_text(SAMPLE.replace("안녕 하세요", marker), encoding="utf-8")
        sc.output_path_for(path).write_text("기존 교정 결과", encoding="utf-8")
    originals = {path: path.read_bytes() for path in paths}
    existing_results = {sc.output_path_for(path): sc.output_path_for(path).read_bytes() for path in paths}
    window._files_loaded(paths)
    prepare_model(window)
    window.concurrency_spin.setValue(2)
    window.show()
    app.processEvents()
    try:
        window.start_correction()
        wait_until(lambda: len(requests) == 2)
        if stop == "close":
            monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
            window.close()
            assert window.close_pending
        else:
            window.cancel_button.click()
            assert not window.cancel_button.isEnabled()

        finish_work(window, app)
        assert not emergency_release.is_set()
        assert sorted(requests) == markers[:2]
        assert sorted(cancelled_requests) == markers[:2]
        assert sorted(finished_requests) == markers[:2]
        assert all(client.is_closed() for client in clients)
        assert all(loop.is_closed() for loop in loops)
        assert [window.table.item(row, 1).text() for row in range(3)] == ["중단", "중단", "대기"]
        assert set(window.correction_overviews) == set(paths[:2])
        assert all(
            overview.model_id == "test-model" and overview.elapsed_seconds >= 0
            for overview in window.correction_overviews.values()
        )
        assert all(window.table.cellWidget(row, 4).isEnabled() for row in range(2))
        assert not window.table.cellWidget(2, 4).isEnabled()
        assert not window.completed_paths
        assert "[경고]" not in window.log_view.toPlainText()
        assert {path: path.read_bytes() for path in paths} == originals
        assert {path: path.read_bytes() for path in existing_results} == existing_results
        assert not list(tmp_path.glob("*_revised_*.srt"))

        if stop == "close":
            assert not window.isVisible()
            assert window.key_edit.text() == ""
        else:
            assert window.start_button.isEnabled()
            window.table.cellWidget(0, 4).click()
            dialog = window.findChild(QDialog)
            assert dialog is not None
            assert "상태: 중단" in dialog.findChild(QPlainTextEdit).toPlainText()
            dialog.close()
            delay_requests.clear()
            window.start_button.click()
            finish_work(window, app)
            assert len(requests) == 5
            assert window.completed_paths == set(paths)
            assert all(window.table.item(row, 1).text() == "완료" for row in range(3))
            assert all(path.with_stem(path.stem + "_revised_2").is_file() for path in paths)
            assert {path: path.read_bytes() for path in paths} == originals
            assert {path: path.read_bytes() for path in existing_results} == existing_results
            assert all(client.is_closed() for client in clients)
            assert all(loop.is_closed() for loop in loops)
    finally:
        emergency_release.set()
        window.cancel_work()
        finish_work(window, app)
