from __future__ import annotations

import codecs
import json
from pathlib import Path
from typing import Any, Callable

import httpx
import openai
import pytest

import srt_spellchecker as sc

SAMPLE = (
    "1\n00:00:01,000 --> 00:00:02,000\n안녕 하세요\n\n"
    "메모 블록\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\n- 첫줄\n- 둘째줄\n"
)


class FakeCorrector:
    """요청 payload를 받아 응답을 만드는 함수로 동작하는 가짜 모델."""

    def __init__(self, responder: Callable[[list[dict[str, Any]]], Any]) -> None:
        self.responder = responder
        self.calls = 0

    def invoke(self, messages: list[tuple[str, str]]) -> dict[str, Any]:
        self.calls += 1
        payload = json.loads(messages[1][1].split("JSON:\n", 1)[1])
        response = self.responder(payload)
        if isinstance(response, Exception):
            raise response
        return response


def ok(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"raw": None, "parsed": sc.CorrectionBatch(items=items), "parsing_error": None}


def echo(payload: list[dict[str, Any]]) -> dict[str, Any]:
    return ok([{"id": p["id"], "corrected_lines": p["lines"]} for p in payload])


# --- SRT 입출력 ---


def test_parse_and_render_round_trip_keeps_non_subtitle_block() -> None:
    blocks = sc.parse_srt_blocks(SAMPLE)
    assert [b.is_subtitle for b in blocks] == [True, False, True]
    assert blocks[2].text_lines == ["- 첫줄", "- 둘째줄"]
    assert sc.render_srt(blocks, "\n") == SAMPLE


def test_crlf_round_trip() -> None:
    content = SAMPLE.replace("\n", "\r\n")
    assert sc.detect_newline(content) == "\r\n"
    assert sc.render_srt(sc.parse_srt_blocks(content), "\r\n") == content


def test_decode_srt_handles_bom_utf8_and_cp949() -> None:
    assert sc.decode_srt(codecs.BOM_UTF8 + "1".encode()) == ("1", "utf-8-sig")
    assert sc.decode_srt("가".encode("utf-8")) == ("가", "utf-8")
    assert sc.decode_srt("가".encode("cp949")) == ("가", "utf-8")


def test_bom_does_not_hide_first_block() -> None:
    content, _ = sc.decode_srt(codecs.BOM_UTF8 + SAMPLE.encode("utf-8"))
    assert sc.parse_srt_blocks(content)[0].is_subtitle


def test_output_path_for() -> None:
    assert sc.output_path_for(Path("a/b.srt")) == Path("a/b_revised.srt")


# --- 검증/되돌림 ---


def test_has_added_punctuation() -> None:
    assert sc.has_added_punctuation("안녕", "안녕.")
    assert not sc.has_added_punctuation("안녕?", "안녕?")
    assert not sc.has_added_punctuation("안녕!!", "안녕!")


def test_sanitize_reverts_on_line_count_mismatch() -> None:
    lines, notes = sc.sanitize_lines(["가", "나"], ["가 나"], None)
    assert lines == ["가", "나"]
    assert len(notes) == 1 and notes[0].startswith("[되돌림]")


def test_sanitize_reverts_only_line_with_added_punctuation() -> None:
    lines, notes = sc.sanitize_lines(["안녕 하세요", "반갑 습니다"], ["안녕하세요.", "반갑습니다"], None)
    assert lines == ["안녕 하세요", "반갑습니다"]
    assert len(notes) == 1 and "1번째 줄" in notes[0]


def test_rewrap_lines_balances_and_respects_limit() -> None:
    assert sc.rewrap_lines(["짧은 줄"], 23) == ["짧은 줄"]
    assert sc.rewrap_lines(["가나다 라마바 사아자 차카타"], 8) == ["가나다 라마바", "사아자 차카타"]
    assert sc.rewrap_lines(["가나다라마바사아자차카타파하"], 5) is None


def test_sanitize_wrap_accepts_valid_model_split() -> None:
    lines, notes = sc.sanitize_lines(["가나다 라마바 사아자"], ["가나다 라마바", "사아자"], 8)
    assert lines == ["가나다 라마바", "사아자"]
    assert notes == []


def test_sanitize_wrap_rewraps_when_model_ignores_limit() -> None:
    lines, notes = sc.sanitize_lines(["가나다 라마바 사아자 차카타"], ["가나다 라마바 사아자 차카타"], 8)
    assert lines == ["가나다 라마바", "사아자 차카타"]
    assert notes[0].startswith("[재분할]")


def test_sanitize_wrap_flags_when_rule_cannot_be_met() -> None:
    lines, notes = sc.sanitize_lines(["가나다라마바사아자차"], ["가나다라마바사아자차"], 5)
    assert lines == ["가나다라마바사아자차"]
    assert notes[0].startswith("[확인필요]")


def test_sanitize_wrap_reverts_on_added_punctuation() -> None:
    lines, notes = sc.sanitize_lines(["안녕 하세요"], ["안녕하세요."], 23)
    assert lines == ["안녕 하세요"]
    assert notes[0].startswith("[되돌림]")


# --- 교정 흐름 ---


def test_revise_applies_corrections_and_skips_non_subtitle() -> None:
    def responder(payload: list[dict[str, Any]]) -> dict[str, Any]:
        return ok(
            [
                {"id": p["id"], "corrected_lines": [line.replace("안녕 하세요", "안녕하세요") for line in p["lines"]]}
                for p in payload
            ]
        )

    blocks = sc.parse_srt_blocks(SAMPLE)
    revised, logs = sc.revise_subtitles(blocks, FakeCorrector(responder))
    assert revised[0].text_lines == ["안녕하세요"]
    assert revised[1] == blocks[1]
    assert logs == []


def test_revise_retries_after_parsing_error() -> None:
    state = {"n": 0}

    def responder(payload: list[dict[str, Any]]) -> dict[str, Any]:
        state["n"] += 1
        if state["n"] == 1:
            return {"raw": None, "parsed": None, "parsing_error": ValueError("bad json")}
        return echo(payload)

    corrector = FakeCorrector(responder)
    _, logs = sc.revise_subtitles(sc.parse_srt_blocks(SAMPLE), corrector)
    assert corrector.calls == 2
    assert logs == []


def test_revise_keeps_original_when_batch_keeps_failing() -> None:
    corrector = FakeCorrector(lambda payload: RuntimeError("boom"))
    blocks = sc.parse_srt_blocks(SAMPLE)
    revised, logs = sc.revise_subtitles(blocks, corrector, batch_size=1)
    assert corrector.calls == sc.ATTEMPT_LIMIT * 2
    assert revised == blocks
    assert len(logs) == 2 and all(log.startswith("[실패]") for log in logs)


def test_revise_logs_missing_and_ignores_unrequested_ids() -> None:
    def responder(payload: list[dict[str, Any]]) -> dict[str, Any]:
        # 첫 항목만 응답하고, 요청하지 않은 id(비자막 블록 1)를 끼워 넣는다.
        return ok(
            [
                {"id": payload[0]["id"], "corrected_lines": ["안녕하세요"]},
                {"id": 1, "corrected_lines": ["오염"]},
            ]
        )

    blocks = sc.parse_srt_blocks(SAMPLE)
    revised, logs = sc.revise_subtitles(blocks, FakeCorrector(responder))
    assert revised[0].text_lines == ["안녕하세요"]
    assert revised[1] == blocks[1]
    assert revised[2] == blocks[2]
    assert logs == ["[누락] 자막 #2: 응답에 없어 원본 유지"]


def test_revise_logs_revert_with_sequence_label() -> None:
    def responder(payload: list[dict[str, Any]]) -> dict[str, Any]:
        return ok([{"id": p["id"], "corrected_lines": [line + "." for line in p["lines"]]} for p in payload])

    revised, logs = sc.revise_subtitles(sc.parse_srt_blocks(SAMPLE), FakeCorrector(responder))
    assert revised[0].text_lines == ["안녕 하세요"]
    assert logs[0] == "[되돌림] 자막 #1: 1번째 줄: 문장부호 추가 감지로 원본 유지"
    assert len(logs) == 3


def test_fatal_api_error_is_not_retried() -> None:
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    error = openai.AuthenticationError("invalid key", response=response, body=None)
    corrector = FakeCorrector(lambda payload: error)
    with pytest.raises(openai.AuthenticationError):
        sc.revise_subtitles(sc.parse_srt_blocks(SAMPLE), corrector)
    assert corrector.calls == 1


# --- 설정/CLI ---


def test_load_environment_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sc, "load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
    assert sc.load_environment() == ("sk-test", "gpt-5.6-luna", "low")

    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "none")
    assert sc.load_environment() == ("sk-test", "env-model", "none")
    assert sc.load_environment("cli-model")[1] == "cli-model"


def test_load_environment_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sc, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        sc.load_environment()


def test_build_llm_avoids_unsupported_gpt5_params() -> None:
    llm = sc.build_llm("sk-test", sc.DEFAULT_MODEL, "low")
    assert llm.model_name == "gpt-5.6-luna"
    assert llm.use_responses_api is True
    assert llm.reasoning == {"effort": "low"}
    assert llm.temperature is None
    assert llm.max_tokens is None


def test_run_preserves_bom_and_crlf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "in.srt"
    source.write_bytes(codecs.BOM_UTF8 + SAMPLE.replace("\n", "\r\n").encode("utf-8"))
    monkeypatch.setattr(sc, "load_dotenv", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(sc, "build_corrector", lambda *a: FakeCorrector(echo))

    output = sc.run([str(source)])
    assert output == tmp_path / "in_revised.srt"
    assert output.read_bytes() == source.read_bytes()
