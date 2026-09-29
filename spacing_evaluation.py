from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from srt_spellchecker import check_cancelled, decode_srt, parse_srt_blocks


class Spacer(Protocol):
    def space(self, text: str, *, reset_whitespace: bool = False) -> str: ...


@dataclass(frozen=True)
class EvaluationResult:
    error_count: int
    character_count: int

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


def evaluate_file(
    path: Path, spacer: Spacer, *,
    is_cancelled: Callable[[], bool] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> EvaluationResult:
    """자막 본문의 띄어쓰기 오류 수와 공백 제외 문자 수를 반환한다."""
    check_cancelled(is_cancelled)
    content, _ = decode_srt(path.read_bytes())
    blocks = [block for block in parse_srt_blocks(content) if block.is_subtitle]
    if not blocks:
        raise ValueError("평가할 SRT 자막 블록이 없습니다.")
    lines = [line for block in blocks for line in block.text_lines]
    count = 0
    character_count = 0
    for index, line in enumerate(lines):
        check_cancelled(is_cancelled)
        # 본문에 섞인 BOM 문자는 분석과 글자 수 집계에서 제외한다.
        text = re.sub(r"<[^>]+>|\{\\[^}]*\}", "", line.replace("\ufeff", "")).strip()
        text = re.sub(r"^[-/]\s*", "", text)
        if text:
            character_count += sum(not character.isspace() for character in text)
            corrected = spacer.space(text, reset_whitespace=True)
            check_cancelled(is_cancelled)
            count += spacing_error_count(text, corrected)
        if on_progress is not None:
            on_progress(index + 1, len(lines))
    check_cancelled(is_cancelled)
    return EvaluationResult(count, character_count)
