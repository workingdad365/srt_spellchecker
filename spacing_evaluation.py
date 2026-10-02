from __future__ import annotations

import codecs
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from srt_spellchecker import check_cancelled, parse_srt_blocks


class Spacer(Protocol):
    def space(self, text: str, *, reset_whitespace: bool = False) -> str: ...


@dataclass(frozen=True)
class EvaluationResult:
    error_count: int
    character_count: int
    warnings: tuple[str, ...] = ()
    converted_from: str | None = None
    review_logs: tuple[str, ...] = ()

    @property
    def skipped_line_count(self) -> int:
        return len(self.warnings)

    @property
    def errors_per_1000(self) -> float | None:
        return self.error_count / self.character_count * 1000 if self.character_count else None


def create_spacer() -> Spacer:
    from kiwipiepy import Kiwi

    return Kiwi(num_workers=1)


def spacing_error_count(original: str, corrected: str) -> int:
    """공백 삽입·삭제가 필요한 문자 사이 위치 수를 반환한다."""
    def boundaries(text: str) -> tuple[str, set[int]]:
        characters: list[str] = []
        spaces: set[int] = set()
        for character in text.strip():
            if character.isspace():
                spaces.add(len(characters))
            else:
                characters.append(character)
        return "".join(characters), spaces

    source, before = boundaries(original)
    target, after = boundaries(corrected)
    if source != target:
        raise ValueError("띄어쓰기 분석 결과에서 공백 외 문자가 변경되었습니다.")
    return len(before ^ after)


def _decode_source(raw: bytes) -> tuple[str, str]:
    for marker, encoding in (
        (codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16"),
        (codecs.BOM_UTF8, "utf-8-sig"),
    ):
        if raw.startswith(marker):
            return raw.decode(encoding), encoding
    if b"\x00" in raw:
        for encoding in ("utf-32-le", "utf-32-be", "utf-16-le", "utf-16-be"):
            try:
                content = raw.decode(encoding)
            except UnicodeDecodeError:
                continue
            if "\x00" not in content and any(block.is_subtitle for block in parse_srt_blocks(content)):
                return content, encoding
        raise ValueError("자막 인코딩을 판별할 수 없습니다. 원본을 유지합니다.")
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp949"), "cp949"


def _replace_with_utf8(
    path: Path, raw: bytes, content: str, is_cancelled: Callable[[], bool] | None,
) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(content.encode("utf-8-sig"))
            stream.flush()
            os.fsync(stream.fileno())
        shutil.copymode(path, temporary_path)
        check_cancelled(is_cancelled)
        if path.read_bytes() != raw:
            raise OSError("인코딩 변환 중 원본 파일이 변경되어 교체하지 않았습니다.")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except PermissionError:
                temporary_path.chmod(0o600)
                temporary_path.unlink()


def evaluate_file(
    path: Path, spacer: Spacer, *,
    is_cancelled: Callable[[], bool] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> EvaluationResult:
    """비 UTF-8 원본을 변환한 뒤 띄어쓰기와 시작시간 순서를 평가한다."""
    check_cancelled(is_cancelled)
    raw = path.read_bytes()
    content, encoding = _decode_source(raw)
    blocks = [block for block in parse_srt_blocks(content) if block.is_subtitle]
    if not blocks:
        raise ValueError("평가할 SRT 자막 블록이 없습니다.")
    converted_from = None
    if encoding not in {"utf-8", "utf-8-sig"}:
        check_cancelled(is_cancelled)
        _replace_with_utf8(path, raw, content, is_cancelled)
        converted_from = encoding
        if on_log is not None:
            on_log(f"[인코딩 변환] {path}: {encoding} -> UTF-8 BOM (원본 교체 완료)")
        content = path.read_bytes().decode("utf-8-sig")
        blocks = [block for block in parse_srt_blocks(content) if block.is_subtitle]
    lines = [
        ((block.sequence or "").strip(), line_number, line)
        for block in blocks
        for line_number, line in enumerate(block.text_lines, start=1)
    ]
    count = 0
    character_count = 0
    warnings: list[str] = []
    for index, (sequence, line_number, line) in enumerate(lines):
        check_cancelled(is_cancelled)
        # 본문에 섞인 BOM 문자는 분석과 글자 수 집계에서 제외한다.
        text = re.sub(r"<[^>]+>|\{\\[^}]*\}", "", line.replace("\ufeff", "")).strip()
        text = re.sub(r"^[-/]\s*", "", text)
        if text:
            corrected = spacer.space(text, reset_whitespace=True)
            check_cancelled(is_cancelled)
            try:
                line_errors = spacing_error_count(text, corrected)
            except ValueError:
                warnings.append(
                    f"자막 #{sequence}, 본문 {line_number}줄: 공백 외 문자 변경으로 평가에서 제외\n"
                    f"  분석 입력: {json.dumps(text, ensure_ascii=False)}\n"
                    f"  Kiwi 결과: {json.dumps(corrected, ensure_ascii=False)}"
                )
            else:
                count += line_errors
                character_count += sum(not character.isspace() for character in text)
        if on_progress is not None:
            on_progress(index + 1, len(lines))
    check_cancelled(is_cancelled)
    review_logs: list[str] = []
    first_sequence = (blocks[0].sequence or "").strip()
    if not content.startswith("1\r\n"):
        review_logs.append(
            f'[확인필요] 자막 #{first_sequence}: 파일 시작이 "1\\r\\n"이 아님\n'
            f"  파일 시작: {json.dumps(content[:40], ensure_ascii=False)}"
        )
    previous_start: tuple[str, int, str] | None = None
    for block in blocks:
        check_cancelled(is_cancelled)
        sequence = (block.sequence or "").strip()
        for line_number, line in enumerate(block.text_lines, start=1):
            check_cancelled(is_cancelled)
            folded = line.casefold()
            markers = [marker for marker in ("KRCC", "EGCC", "&nbsp") if marker.casefold() in folded]
            if markers:
                review_logs.append(
                    f"[확인필요] 자막 #{sequence}, 본문 {line_number}줄: 검토 대상 문자열 발견 ({', '.join(markers)})\n"
                    f"  원문: {json.dumps(line, ensure_ascii=False)}"
                )
        match = re.match(r"\s*(\d+):([0-5]\d):([0-5]\d)[,.](\d{3})\s*-->", block.timecode or "")
        if match is None:
            review_logs.append(f"[확인필요] 자막 #{sequence}: 시작시간 형식 확인 필요\n  {block.timecode}")
            previous_start = None
            continue
        hours, minutes, seconds, milliseconds = map(int, match.groups())
        start = ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds
        timestamp = f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
        if previous_start is not None and start < previous_start[1]:
            review_logs.append(
                f"[확인필요] 자막 #{sequence}: 시작시간 역행\n"
                f"  직전 자막 #{previous_start[0]}: {previous_start[2]}\n"
                f"  현재 자막 #{sequence}: {timestamp}"
            )
        previous_start = (sequence, start, timestamp)
        end_match = re.match(
            r"\s*(\d+):([0-5]\d):([0-5]\d)[,.](\d{3})(?=\s|$)",
            (block.timecode or "")[match.end():],
        )
        if end_match is None:
            review_logs.append(f"[확인필요] 자막 #{sequence}: 종료시간 형식 확인 필요\n  {block.timecode}")
            continue
        end_hours, end_minutes, end_seconds, end_milliseconds = map(int, end_match.groups())
        end = ((end_hours * 60 + end_minutes) * 60 + end_seconds) * 1000 + end_milliseconds
        if end <= start:
            end_timestamp = f"{end_hours:02}:{end_minutes:02}:{end_seconds:02},{end_milliseconds:03}"
            review_logs.append(
                f"[확인필요] 자막 #{sequence}: 종료시간이 시작시간보다 같거나 빠름\n"
                f"  시작시간: {timestamp}\n"
                f"  종료시간: {end_timestamp}"
            )
    return EvaluationResult(count, character_count, tuple(warnings), converted_from, tuple(review_logs))
