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


@pytest.mark.parametrize("mark", ["..", "...", "....", "......", "…", "……", "⋯", "‥", "︙", "︰", "．．", ".…."])
def test_normalize_ellipsis(mark) -> None:
    assert sc.normalize_ellipsis(f"잠깐{mark} 기다려") == "잠깐... 기다려"


def test_normalize_ellipsis_preserves_single_periods_and_other_punctuation() -> None:
    text = "3.14입니다. example.com · 항목 • 목록"
    assert sc.normalize_ellipsis(text) == text


@pytest.mark.parametrize(("text", "expected"), [
    ("안녕하세요.", "안녕하세요"),
    ("안녕하세요.  ", "안녕하세요  "),
    ('"안녕하세요."', '"안녕하세요"'),
    ("‘안녕하세요.’", "‘안녕하세요’"),
    ("(안녕하세요.)", "(안녕하세요)"),
    ("<i><b>안녕하세요.</b></i>", "<i><b>안녕하세요</b></i>"),
    ("잠깐..", "잠깐..."),
    ("잠깐...", "잠깐..."),
    ("잠깐……", "잠깐..."),
    ('<i>"잠깐..."</i>', '<i>"잠깐..."</i>'),
    ("값은 3.14.", "값은 3.14"),
    ("값은 3.14", "값은 3.14"),
    ("example.com", "example.com"),
    ("U.S.A 방송", "U.S.A 방송"),
    ("정말?!", "정말?!"),
])
def test_normalize_subtitle_sentence_period(text, expected) -> None:
    assert sc.normalize_subtitle_punctuation(text) == expected


@pytest.mark.parametrize("wrap_length", [None, 23])
def test_sentence_period_removed_from_original_and_response(wrap_length) -> None:
    lines, notes = sc.sanitize_lines(["안녕 하세요."], ["안녕하세요."], wrap_length)
    assert lines == ["안녕하세요"]
    assert notes == []
    lines, notes = sc.sanitize_lines(["안녕 하세요"], ["안녕하세요."], wrap_length)
    assert lines == ["안녕하세요"]
    assert notes == []


def test_sentence_period_removed_before_wrap_length_validation() -> None:
    assert sc.sanitize_lines(["안녕하세요."], ["안녕하세요."], 5) == (["안녕하세요"], [])


def test_sentence_period_dialogue_and_rejected_response() -> None:
    lines, notes = sc.sanitize_lines(["안녕하세요.", "/잠깐…"], ["안녕하세요!", "/잠깐..."], None)
    assert lines == ["- 안녕하세요", "- 잠깐..."]
    assert notes[0].startswith("[되돌림]")


@pytest.mark.parametrize("response_kind", ["echo", "missing", "failed"])
def test_revise_removes_sentence_period_even_on_failure(response_kind) -> None:
    source = "1\n00:00:01,000 --> 00:00:02,000\n안녕하세요.\n잠깐…\n\n메모.\n"
    original = sc.parse_srt_blocks(source)

    def responder(payload):
        assert payload[0]["lines"] == ["안녕하세요", "잠깐..."]
        if response_kind == "failed":
            return RuntimeError("failed")
        return ok([]) if response_kind == "missing" else echo(payload)

    revised, _ = sc.revise_subtitles(original, FakeCorrector(responder))
    assert revised[0].text_lines == ["안녕하세요", "잠깐..."]
    assert revised[0].timecode == original[0].timecode
    assert revised[1] == original[1]
    assert original[0].text_lines == ["안녕하세요.", "잠깐…"]


@pytest.mark.parametrize("wrap_length", [None, 23])
def test_ellipsis_normalization_passes_punctuation_validation(wrap_length) -> None:
    lines, notes = sc.sanitize_lines(["잠깐.. 기다려"], ["잠깐… 기다려"], wrap_length)
    assert lines == ["잠깐... 기다려"]
    assert notes == []


@pytest.mark.parametrize("wrap_length", [None, 23])
def test_new_ellipsis_is_still_rejected(wrap_length) -> None:
    lines, notes = sc.sanitize_lines(["잠깐 기다려"], ["잠깐… 기다려"], wrap_length)
    assert lines == ["잠깐 기다려"]
    assert notes[0].startswith("[되돌림]")


def test_ellipsis_fallback_and_dialogue_formatting() -> None:
    lines, notes = sc.sanitize_lines(["잠깐..", "/왜…"], ["잠깐... 왜..."], None)
    assert lines == ["- 잠깐...", "- 왜..."]
    assert notes[0].startswith("[되돌림]")


@pytest.mark.parametrize("response_kind", ["echo", "missing", "failed"])
def test_revise_normalizes_ellipsis_even_on_missing_or_failed_response(response_kind) -> None:
    source = "1\n00:00:01,000 --> 00:00:02,000\n잠깐.. 기다려…\n\n메모..\n"
    original = sc.parse_srt_blocks(source)

    def responder(payload):
        assert payload[0]["lines"] == ["잠깐... 기다려..."]
        if response_kind == "failed":
            return RuntimeError("failed")
        if response_kind == "missing":
            return ok([])
        return echo(payload)

    revised, _ = sc.revise_subtitles(original, FakeCorrector(responder))
    assert revised[0].text_lines == ["잠깐... 기다려..."]
    assert revised[0].timecode == original[0].timecode
    assert revised[1] == original[1]
    assert original[0].text_lines == ["잠깐.. 기다려…"]


def test_sanitize_reverts_on_line_count_mismatch() -> None:
    lines, notes = sc.sanitize_lines(["가", "나"], ["가 나"], None)
    assert lines == ["가", "나"]
    assert len(notes) == 1 and notes[0].startswith("[되돌림]")


def test_sanitize_reverts_only_line_with_added_punctuation() -> None:
    lines, notes = sc.sanitize_lines(["안녕 하세요", "반갑 습니다"], ["안녕하세요!", "반갑습니다"], None)
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
    lines, notes = sc.sanitize_lines(["안녕 하세요"], ["안녕하세요!"], 23)
    assert lines == ["안녕 하세요"]
    assert notes[0].startswith("[되돌림]")


# --- 교정 흐름 ---


@pytest.mark.parametrize("wrap_length", [None, 23])
@pytest.mark.parametrize("second_prefix", ["/", "/ ", "-", "-   "])
def test_dialogue_markers_are_normalized(second_prefix, wrap_length) -> None:
    original = ["아메리카노 한 잔 주세요", second_prefix + "예 손님, 잠시만 기다려 주세요"]
    expected = ["- 아메리카노 한 잔 주세요", "- 예 손님, 잠시만 기다려 주세요"]
    lines, notes = sc.sanitize_lines(original, original, wrap_length)
    assert lines == expected
    assert notes == []


def test_dialogue_allows_markers_but_rejects_new_punctuation() -> None:
    original = ["안녕 하세요", "/ 반갑 습니다"]
    lines, notes = sc.sanitize_lines(original, ["- 안녕하세요!", "- 반갑습니다"], None)
    assert lines == ["- 안녕 하세요", "- 반갑습니다"]
    assert len(notes) == 1 and "문장부호 추가" in notes[0]


@pytest.mark.parametrize("wrap_length", [None, 23])
@pytest.mark.parametrize("corrected", [["한 잔 주세요 잠시만요"], ["한 잔 주세요", ""]])
def test_dialogue_keeps_two_speakers_on_invalid_response(wrap_length, corrected) -> None:
    lines, notes = sc.sanitize_lines(["한 잔 주세요", "/잠시만요"], corrected, wrap_length)
    assert lines == ["- 한 잔 주세요", "- 잠시만요"]
    assert notes[0].startswith("[되돌림]")


def test_dialogue_wrap_does_not_move_words_between_speakers() -> None:
    original = ["아메리카노 한 잔 주세요", "/네"]
    lines, notes = sc.sanitize_lines(original, original, 12)
    assert lines == ["- 아메리카노 한 잔 주세요", "- 네"]
    assert notes == ["[확인필요] 12자 초과: 두 사람의 대사 구분 유지"]


@pytest.mark.parametrize("original", [
    ["아메리카노 한 잔", "주세요"],
    ["커피/차", "메뉴입니다"],
    ["/한 사람의 대사"],
    ["/", "/"],
])
def test_non_dialogue_lines_keep_original_format(original) -> None:
    assert sc.sanitize_lines(original, original, None) == (original, [])


def test_revise_normalizes_dialogue_payload_and_preserves_srt_structure() -> None:
    source = "1\n00:00:01,000 --> 00:00:02,000\n아메리카노 한 잔 주세요\n/예 손님, 잠시만 기다려 주세요\n"
    expected = ["- 아메리카노 한 잔 주세요", "- 예 손님, 잠시만 기다려 주세요"]

    def responder(payload):
        assert payload[0]["lines"] == expected
        return echo(payload)

    original = sc.parse_srt_blocks(source)
    revised, logs = sc.revise_subtitles(original, FakeCorrector(responder))
    assert revised[0].text_lines == expected
    assert revised[0].sequence == original[0].sequence
    assert revised[0].timecode == original[0].timecode
    assert original[0].text_lines[1].startswith("/")
    assert logs == []


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
        return ok([{"id": p["id"], "corrected_lines": [line + "!" for line in p["lines"]]} for p in payload])

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


def test_collect_srt_recursively_and_deduplicate(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "first.srt"
    second = nested / "second.SRT"
    revised = nested / "second_revised.srt"
    for source in (first, second, revised, nested / "ignore.txt"):
        source.write_text(SAMPLE, encoding="utf-8")
    assert set(sc.collect_srt_files([tmp_path, nested, first])) == {first, second, revised}


def test_correct_file_keeps_existing_outputs_and_encoding(tmp_path: Path) -> None:
    source = tmp_path / "in.srt"
    source.write_bytes(codecs.BOM_UTF8 + SAMPLE.replace("\n", "\r\n").encode("utf-8"))
    existing = tmp_path / "in_revised.srt"
    existing.write_text("keep", encoding="utf-8")
    progress = []
    output, logs = sc.correct_file(
        source, FakeCorrector(echo), on_progress=lambda done, total: progress.append((done, total)),
    )
    assert output.name == "in_revised_2.srt"
    assert output.read_bytes() == source.read_bytes()
    assert existing.read_text() == "keep"
    assert progress == [(1, 1)]
    assert logs == []


def test_cancel_after_request_does_not_write_file(tmp_path: Path) -> None:
    source = tmp_path / "in.srt"
    source.write_text(SAMPLE, encoding="utf-8")
    corrector = FakeCorrector(echo)
    with pytest.raises(sc.CorrectionCancelled):
        sc.correct_file(source, corrector, is_cancelled=lambda: corrector.calls > 0)
    assert not sc.output_path_for(source).exists()


def test_cancel_before_request_does_not_call_model() -> None:
    corrector = FakeCorrector(echo)
    with pytest.raises(sc.CorrectionCancelled):
        sc.revise_subtitles(sc.parse_srt_blocks(SAMPLE), corrector, is_cancelled=lambda: True)
    assert corrector.calls == 0
