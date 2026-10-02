from pathlib import Path

import pytest

from spacing_evaluation import EvaluationResult, create_spacer, evaluate_file, spacing_error_count
from srt_spellchecker import CorrectionCancelled


@pytest.mark.parametrize(("source", "target", "count"), [
    ("안녕 하세요", "안녕하세요", 1),
    ("나는학교에간다", "나는 학교에 간다", 2),
    ("가 나다", "가나 다", 2),
    ("  가   나\t다  ", "가 나 다", 0),
    ("", "", 0),
])
def test_spacing_boundaries(source, target, count):
    assert spacing_error_count(source, target) == count


def test_non_whitespace_changes_are_not_counted():
    with pytest.raises(ValueError):
        spacing_error_count("안녕", "안녕!")


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp949"])
def test_evaluate_only_subtitle_text_without_writing(tmp_path, encoding):
    path = tmp_path / "sample.srt"
    source = '1\r\n00:00:01,000 --> 00:00:02,000\r\n<i>안녕 하세요</i>\r\n/ 반갑 습니다\r\n\r\n메모 블록\r\n'
    path.write_bytes(source.encode(encoding))
    before = path.read_bytes()
    calls = []

    class Spacer:
        def space(self, text, *, reset_whitespace):
            assert reset_whitespace
            calls.append(text)
            return text.replace(" ", "")

    progress = []
    result = evaluate_file(path, Spacer(), on_progress=lambda *args: progress.append(args))
    assert result == EvaluationResult(error_count=2, character_count=10)
    assert result.errors_per_1000 == 200
    assert calls == ["안녕 하세요", "반갑 습니다"]
    assert progress == [(1, 2), (2, 2)]
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_non_whitespace_change_skips_only_affected_line(tmp_path):
    path = tmp_path / "partial.srt"
    source = (
        "387\n00:00:01,000 --> 00:00:02,000\n안녕 하세요\n\n"
        "388\n00:00:03,000 --> 00:00:04,000\n"
        "<i>이게 교회의 권위를 았아가려는</i>\n/ 반갑 습니다\n"
    )
    path.write_bytes(source.encode("utf-8"))
    calls = []

    class Spacer:
        def space(self, text, *, reset_whitespace):
            assert reset_whitespace
            calls.append(text)
            if "았아가려는" in text:
                return "이게 교회의 권위 를 ᆯ았아 가려는"
            return text.replace(" ", "")

    progress = []
    result = evaluate_file(path, Spacer(), on_progress=lambda *args: progress.append(args))
    assert result.error_count == 2
    assert result.character_count == 10
    assert result.errors_per_1000 == 200
    assert result.skipped_line_count == 1
    assert "자막 #388, 본문 1줄" in result.warnings[0]
    assert "공백 외 문자 변경" in result.warnings[0]
    assert "이게 교회의 권위를 았아가려는" in result.warnings[0]
    assert "이게 교회의 권위 를 ᆯ았아 가려는" in result.warnings[0]
    assert calls[-1] == "반갑 습니다"
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert path.read_bytes() == source.encode("utf-8")
    assert list(tmp_path.iterdir()) == [path]


def test_all_lines_changed_have_no_valid_error_rate(tmp_path):
    path = tmp_path / "unscorable.srt"
    path.write_text("1\n00:00:01,000 --> 00:00:02,000\n첫 줄\n둘째 줄\n", encoding="utf-8")

    class Spacer:
        def space(self, text, **_kwargs):
            return text + "!"

    result = evaluate_file(path, Spacer())
    assert result.error_count == result.character_count == 0
    assert result.errors_per_1000 is None
    assert result.skipped_line_count == 2
    assert "본문 1줄" in result.warnings[0]
    assert "본문 2줄" in result.warnings[1]


def test_spacer_execution_errors_are_not_skipped(tmp_path):
    path = tmp_path / "error.srt"
    path.write_text("1\n00:00:01,000 --> 00:00:02,000\n안녕\n", encoding="utf-8")

    class Spacer:
        def space(self, text, **_kwargs):
            raise ValueError("분석기 실행 오류")

    with pytest.raises(ValueError, match="분석기 실행 오류"):
        evaluate_file(path, Spacer())


def test_cancel_after_skipped_line_stops_evaluation(tmp_path):
    path = tmp_path / "cancel.srt"
    path.write_text("1\n00:00:01,000 --> 00:00:02,000\n첫 줄\n둘째 줄\n", encoding="utf-8")
    calls = []
    progress = []

    class Spacer:
        def space(self, text, **_kwargs):
            calls.append(text)
            return text + "!"

    with pytest.raises(CorrectionCancelled):
        evaluate_file(
            path, Spacer(), is_cancelled=lambda: bool(progress),
            on_progress=lambda *args: progress.append(args),
        )
    assert calls == ["첫 줄"]
    assert progress == [(1, 2)]


def test_cancel_before_reading():
    with pytest.raises(CorrectionCancelled):
        evaluate_file(Path("missing.srt"), None, is_cancelled=lambda: True)


def test_invalid_subtitle_is_failure(tmp_path):
    path = tmp_path / "invalid.srt"
    path.write_text("not an srt", encoding="utf-8")
    with pytest.raises(ValueError, match="SRT"):
        evaluate_file(path, None)


def test_real_kiwi_spacing(tmp_path):
    path = tmp_path / "real.srt"
    path.write_text("1\n00:00:01,000 --> 00:00:02,000\n안녕하세요\n", encoding="utf-8")
    spacer = create_spacer()
    assert evaluate_file(path, spacer) == EvaluationResult(error_count=0, character_count=5)
    assert spacing_error_count("안녕 하세요", spacer.space("안녕 하세요", reset_whitespace=True)) == 1
    path.write_text("1\n00:00:01,000 --> 00:00:02,000\n\ufeff안녕하세요\n", encoding="utf-8-sig")
    assert evaluate_file(path, spacer) == EvaluationResult(error_count=0, character_count=5)


@pytest.mark.parametrize("body", [
    "\ufeff안녕 하세요", "<i>\ufeff안녕 하세요</i>",
    "\ufeff- 안녕 하세요", "안녕\ufeff 하세요\ufeff", "\ufeff/안녕 하세요",
])
def test_embedded_bom_is_excluded_without_modifying_file(tmp_path, body):
    path = tmp_path / "embedded_bom.srt"
    path.write_text("1\n00:00:01,000 --> 00:00:02,000\n" + body + "\n", encoding="utf-8-sig")
    before = path.read_bytes()

    class Spacer:
        def space(self, text, *, reset_whitespace):
            assert text == "안녕 하세요"
            return "안녕하세요"

    assert evaluate_file(path, Spacer()) == EvaluationResult(1, 5)
    assert path.read_bytes() == before


@pytest.mark.parametrize(("errors", "characters", "expected"), [
    (20, 10000, 2.0), (10, 1000, 10.0), (0, 100, 0.0), (0, 0, None),
])
def test_error_frequency(errors, characters, expected):
    assert EvaluationResult(errors, characters).errors_per_1000 == expected


@pytest.mark.parametrize(("body", "characters"), [
    ("<i>가\t나!</i>\n- 다 라?\n{\\an8}마", 7),
    ("<i></i>\n- ", 0),
    ("\ufeff", 0),
])
def test_character_count_uses_analyzed_text(tmp_path, body, characters):
    path = tmp_path / "characters.srt"
    path.write_text("1\n00:00:01,000 --> 00:00:02,000\n" + body + "\n", encoding="utf-8")

    class Spacer:
        def space(self, text, **kwargs):
            return text

    result = evaluate_file(path, Spacer())
    assert result == EvaluationResult(0, characters)
    assert result.errors_per_1000 == (0 if characters else None)
