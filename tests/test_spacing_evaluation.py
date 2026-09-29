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
