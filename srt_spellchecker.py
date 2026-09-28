from __future__ import annotations

import argparse
import codecs
from collections import Counter
from collections.abc import Callable, Iterable
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import openai
from pydantic import BaseModel, Field

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "low"
BATCH_SIZE = 25
ATTEMPT_LIMIT = 3
DEFAULT_MAX_LINE_LENGTH = 23
MAX_WRAPPED_LINES = 2

# 재시도해도 결과가 달라지지 않는 오류. 배치 재시도 없이 즉시 중단한다.
FATAL_API_ERRORS = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.BadRequestError,
    openai.NotFoundError,
)


class SubtitleBlock(BaseModel):
    """SRT 블록 한 개를 표현한다."""

    raw_lines: list[str]
    is_subtitle: bool
    sequence: str | None = None
    timecode: str | None = None
    text_lines: list[str] = Field(default_factory=list)


class CorrectionItem(BaseModel):
    """교정 대상 캡션과 교정 결과를 매핑한다."""

    id: int
    corrected_lines: list[str]


class CorrectionBatch(BaseModel):
    """한 번의 모델 호출에서 반환되는 교정 결과 묶음."""

    items: list[CorrectionItem]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SRT 자막 파일의 오타/띄어쓰기를 교정한다."
    )
    parser.add_argument("srt_file", type=Path, help="입력 SRT 파일 경로")
    parser.add_argument(
        "--wrap",
        action="store_true",
        help="긴 줄을 최대 2줄로 나눈다 (기본: 원본 줄 구성 유지)",
    )
    parser.add_argument(
        "--max-line-length",
        type=int,
        default=DEFAULT_MAX_LINE_LENGTH,
        help=f"--wrap 사용 시 한 줄 최대 글자 수 (기본: {DEFAULT_MAX_LINE_LENGTH})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"사용할 OpenAI 모델 (기본: OPENAI_MODEL 환경 변수 또는 {DEFAULT_MODEL})",
    )
    args = parser.parse_args(argv)
    if args.max_line_length < 1:
        parser.error("--max-line-length는 1 이상이어야 합니다.")
    return args


def load_environment(model_override: str | None = None) -> tuple[str, str, str]:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "환경 변수가 비어 있습니다: OPENAI_API_KEY. .env.example를 복사한 .env를 채워주세요."
        )

    model = (model_override or os.getenv("OPENAI_MODEL", "")).strip() or DEFAULT_MODEL
    effort = os.getenv("OPENAI_REASONING_EFFORT", "").strip() or DEFAULT_REASONING_EFFORT
    return api_key, model, effort


def decode_srt(raw: bytes) -> tuple[str, str]:
    """바이트를 디코딩하고 (본문, 저장에 사용할 인코딩)을 반환한다."""
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        # 국내 자막에 흔한 CP949 입력은 UTF-8로 변환해 저장한다.
        return raw.decode("cp949"), "utf-8"


def detect_newline(content: str) -> str:
    if "\r\n" in content:
        return "\r\n"
    return "\n"


def parse_srt_blocks(content: str) -> list[SubtitleBlock]:
    normalized = content.replace("\r\n", "\n")
    parts = re.split(r"\n\s*\n", normalized.strip())

    blocks: list[SubtitleBlock] = []
    for part in parts:
        lines = part.split("\n")
        if len(lines) >= 2 and re.fullmatch(r"\d+", lines[0].strip()):
            if "-->" in lines[1]:
                blocks.append(
                    SubtitleBlock(
                        raw_lines=lines,
                        is_subtitle=True,
                        sequence=lines[0],
                        timecode=lines[1],
                        text_lines=lines[2:],
                    )
                )
                continue

        blocks.append(SubtitleBlock(raw_lines=lines, is_subtitle=False))

    return blocks


def render_srt(blocks: list[SubtitleBlock], newline: str) -> str:
    rendered_blocks: list[str] = []
    for block in blocks:
        if block.is_subtitle:
            lines = [block.sequence or "", block.timecode or "", *block.text_lines]
            rendered_blocks.append(newline.join(lines))
        else:
            rendered_blocks.append(newline.join(block.raw_lines))

    return f"{newline}{newline}".join(rendered_blocks) + newline


def chunked[T](items: list[T], size: int) -> list[list[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def normalize_ellipsis(text: str) -> str:
    return re.sub(
        r"[.\uff0e\u2025\u2026\u22ef\ufe19\ufe30]{2,}|[\u2025\u2026\u22ef\ufe19\ufe30]",
        "...",
        text,
    )


def normalize_subtitle_punctuation(text: str) -> str:
    text = normalize_ellipsis(text)
    return re.sub(
        r"(?<!\.)\.(?=(?:\s|[\"'\u2019\u201d\u3009\u300b\u300d\u300f)\]}]|</[A-Za-z][^>]*>)*$)",
        "",
        text,
    )


def punctuation_counter(text: str) -> Counter[str]:
    return Counter(ch for ch in text if unicodedata.category(ch).startswith("P"))


def has_added_punctuation(original: str, revised: str) -> bool:
    original_marks = punctuation_counter(original)
    revised_marks = punctuation_counter(revised)
    for mark, count in revised_marks.items():
        if count > original_marks.get(mark, 0):
            return True
    return False


def violates_wrap_rules(lines: list[str], max_length: int) -> bool:
    return len(lines) > MAX_WRAPPED_LINES or any(len(line) > max_length for line in lines)


def rewrap_lines(lines: list[str], max_length: int) -> list[str] | None:
    """공백 기준으로 최대 2줄에 균형 있게 다시 나눈다. 불가능하면 None을 반환한다."""
    words = " ".join(lines).split()
    text = " ".join(words)
    if len(text) <= max_length:
        return [text]

    best: tuple[int, list[str]] | None = None
    for i in range(1, len(words)):
        left = " ".join(words[:i])
        right = " ".join(words[i:])
        if len(left) > max_length or len(right) > max_length:
            continue
        imbalance = abs(len(left) - len(right))
        if best is None or imbalance < best[0]:
            best = (imbalance, [left, right])

    return best[1] if best else None


def format_dialogue_lines(lines: list[str]) -> list[str]:
    return ["- " + re.sub(r"^[-/]\s*", "", line.strip(), count=1) for line in lines]


def is_two_speaker_dialogue(lines: list[str]) -> bool:
    return (
        len(lines) == 2
        and re.match(r"^\s*[-/]\s*\S", lines[1]) is not None
        and all(line[2:].strip() for line in format_dialogue_lines(lines))
    )


def sanitize_lines(
    original_lines: list[str],
    corrected_lines: list[str],
    wrap_length: int | None,
) -> tuple[list[str], list[str]]:
    """모델 교정 결과를 검증해 (확정 줄 목록, 검토 로그)를 반환한다."""
    notes: list[str] = []
    original_lines = [normalize_subtitle_punctuation(line) for line in original_lines]
    corrected_lines = [normalize_subtitle_punctuation(line) for line in corrected_lines]

    dialogue = is_two_speaker_dialogue(original_lines)
    if dialogue:
        original_lines = format_dialogue_lines(original_lines)
        if len(corrected_lines) != 2 or not all(
            line[2:].strip() for line in format_dialogue_lines(corrected_lines)
        ):
            notes.append("[되돌림] 대사 구성 불일치로 대사 표기만 교정")
            corrected_lines = original_lines
        else:
            corrected_lines = format_dialogue_lines(corrected_lines)

    if wrap_length is None or dialogue:
        if len(corrected_lines) != len(original_lines):
            notes.append("[되돌림] 줄 수 불일치로 원본 유지")
            return original_lines, notes

        result: list[str] = []
        for number, (original, revised) in enumerate(
            zip(original_lines, corrected_lines, strict=True), start=1
        ):
            if has_added_punctuation(original, revised):
                notes.append(f"[되돌림] {number}번째 줄: 문장부호 추가 감지로 원본 유지")
                result.append(original)
                continue
            result.append(revised)
        if dialogue and wrap_length is not None and violates_wrap_rules(result, wrap_length):
            notes.append(f"[확인필요] {wrap_length}자 초과: 두 사람의 대사 구분 유지")
        return result, notes

    # 줄 나눔 모드는 줄 수가 달라질 수 있으므로 블록 단위로 검증한다.
    lines = [line.strip() for line in corrected_lines if line.strip()]
    if not lines:
        notes.append("[되돌림] 빈 교정 결과로 원본 유지")
        lines = list(original_lines)
    elif has_added_punctuation("".join(original_lines), "".join(lines)):
        notes.append("[되돌림] 문장부호 추가 감지로 원본 유지")
        lines = list(original_lines)

    if violates_wrap_rules(lines, wrap_length):
        rewrapped = rewrap_lines(lines, wrap_length)
        if rewrapped is None:
            notes.append(
                f"[확인필요] {wrap_length}자/{MAX_WRAPPED_LINES}줄 규칙을 맞출 수 없음"
            )
        else:
            notes.append("[재분할] 줄 길이 규칙 위반으로 코드에서 다시 나눔")
            lines = rewrapped

    return lines, notes


def build_llm(api_key: str, model: str, effort: str) -> ChatOpenAI:
    # GPT-5.x 계열 주의점
    # - temperature, top_p 등 샘플링 파라미터 미지원 (400 오류) -> 전달하지 않음
    # - max_tokens 미지원, 추론 토큰도 출력 한도에 포함됨 -> 출력 한도를 지정하지 않음
    # - 추론 강도는 reasoning.effort로 제어, Responses API 사용
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        use_responses_api=True,
        reasoning={"effort": effort},
        timeout=180,
        max_retries=3,
    )


def build_corrector(api_key: str, model: str, effort: str) -> Any:
    return build_llm(api_key, model, effort).with_structured_output(
        CorrectionBatch,
        method="json_schema",
        strict=True,
        include_raw=True,
    )


def build_messages(
    payload: list[dict[str, object]],
    wrap_length: int | None,
) -> list[tuple[str, str]]:
    if wrap_length is None:
        line_rule = "절대 줄을 합치거나 나누지 말고 입력의 줄 개수를 그대로 유지한다. "
    else:
        line_rule = (
            f"각 줄은 공백 포함 {wrap_length}자를 넘지 않게 하고 "
            f"항목당 최대 {MAX_WRAPPED_LINES}줄만 사용한다. "
            f"{wrap_length}자를 넘는 줄은 조사, 접속사 앞 등 문맥상 자연스러운 위치에서 "
            "균형 있게 나누고, 규칙을 이미 만족하는 항목은 원래 줄 구성을 유지한다. "
        )

    system_prompt = (
        "당신은 한국어 자막 교정 전문가다. "
        "OCR로 생성된 SRT 자막 문장을 문맥에 맞게 교정한다. "
        "오타, 띄어쓰기, 잘못 인식된 글자를 자연스럽게 수정하되 원래 의미는 유지한다. "
        "비속어와 욕설은 문맥에 맞는 순화어로 바꾸고, 사투리의 어휘와 어미는 자연스러운 표준어로 바꾼다. "
        "순화와 표준어 변환 시 원래 뜻과 감정, 존댓말과 반말의 구분을 유지하고 "
        "대사를 삭제하거나 새로운 내용을 덧붙이지 않는다. "
        "단어의 일부가 비속어와 같다는 이유만으로 정상적인 단어나 고유명사를 바꾸지 않는다. "
        "문맥상 뜻이나 대응 표현이 불확실하면 억지로 치환하지 않고 원문을 유지한다. "
        "본문에 한자로만 적힌 단어나 한자어는 한국식 한자 독음의 한글로 바꾼다. "
        "예: '主君'은 '주군', '主君을'은 '주군을'로 바꾸며 주변 조사와 문장은 유지한다. "
        "뜻풀이로 번역하거나 중국어·일본어 발음으로 음역하지 않는다. "
        "단, '주군(主君)'이나 '주군（主君）'처럼 한글 표현을 부연하는 괄호 안 한자는 "
        "한글로 바꾸거나 삭제하지 않고 괄호와 내용을 그대로 유지한다. "
        "같은 줄에 부연 표기와 독립된 한자가 함께 있으면 부연 표기만 보존하고 나머지는 독음으로 바꾼다. "
        "고유명사도 한국식 한자 독음을 적용하되 문맥상 독음을 확신할 수 없으면 원문을 유지한다. "
        f"{line_rule}"
        "두 줄 중 둘째 줄이 / 또는 -로 시작하는 자막은 두 사람의 대사다. "
        "각 대사 앞의 / 또는 - 표기를 '- ' (하이픈과 공백 한 칸)으로 통일하고, "
        "첫째 줄에도 '- '를 붙인다. 이 대사 표기에 한해 하이픈 추가를 허용한다. "
        "두 사람의 대사는 줄 길이 제한보다 화자 구분을 우선하여 반드시 두 줄로 유지하고 "
        "서로 합치거나 다른 화자의 줄로 옮기지 않는다. "
        "대사 구분 표식이 없는 일반 두 줄 자막을 임의로 두 사람의 대사로 바꾸지 않는다. "
        "기존 말줄임표는 점 두 개(..), 연속된 점, 특수문자 표기 모두 점 세 개(...)로 통일한다. "
        "한국어 자막의 문장 끝 마침표(.)는 생략한다. 원문에 있어도 제거하며 "
        "따옴표나 닫는 서식 태그 앞의 문장 끝 마침표도 제거한다. "
        "말줄임표(...)와 소수점, URL 및 약어 내부의 점은 마침표와 혼동하지 말고 유지한다. "
        "대사 구분용 하이픈과 기존 말줄임표의 표기 통일 이외의 문장부호를 임의로 추가하지 말아라. "
        "특히 원문에 없는 마침표나 말줄임표를 추가하지 마라. "
        "확신할 수 없는 줄은 원문을 그대로 반환한다. "
        "주어진 id를 빠짐없이 정확히 한 번씩 포함해야 한다."
    )
    human_prompt = (
        "아래 JSON 배열의 각 항목을 교정하라. "
        "각 항목은 id와 lines를 가지며, 교정 결과는 같은 id의 corrected_lines에 담는다.\n"
        "JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    return [("system", system_prompt), ("human", human_prompt)]


def request_corrections(
    corrector: Any,
    payload: list[dict[str, object]],
    wrap_length: int | None,
) -> list[CorrectionItem]:
    result = corrector.invoke(build_messages(payload, wrap_length))

    parsing_error = result.get("parsing_error")
    if parsing_error:
        raise ValueError(f"구조화 출력 파싱 실패: {parsing_error}")

    parsed = result.get("parsed")
    if parsed is None:
        raise ValueError("모델 응답에서 교정 결과를 찾을 수 없습니다.")

    return CorrectionBatch.model_validate(parsed).items


def correct_batch_with_retry(
    corrector: Any,
    payload: list[dict[str, object]],
    wrap_length: int | None,
    attempt_limit: int = ATTEMPT_LIMIT,
    *,
    on_log: Callable[[str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[CorrectionItem] | None:
    """배치를 교정한다. 재시도 후에도 실패하면 None을 반환한다."""
    for attempt in range(1, attempt_limit + 1):
        check_cancelled(is_cancelled)
        try:
            return request_corrections(corrector, payload, wrap_length)
        except FATAL_API_ERRORS:
            raise
        except Exception as error:
            message = f"[경고] 배치 교정 실패 (시도 {attempt}/{attempt_limit}): {error}"
            if on_log:
                on_log(message)
            else:
                print(message, file=sys.stderr)
    return None


def revise_subtitles(
    blocks: list[SubtitleBlock],
    corrector: Any,
    wrap_length: int | None = None,
    batch_size: int = BATCH_SIZE,
    *,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> tuple[list[SubtitleBlock], list[str]]:
    """자막을 교정하고 (교정된 블록, 검토 로그)를 반환한다."""
    logs: list[str] = []
    blocks = [
        block.model_copy(update={
            "text_lines": [normalize_subtitle_punctuation(line) for line in block.text_lines],
        }) if block.is_subtitle else block
        for block in blocks
    ]

    target_ids = [
        i
        for i, block in enumerate(blocks)
        if block.is_subtitle and any(line.strip() for line in block.text_lines)
    ]
    if not target_ids:
        return blocks, logs

    def label(index: int) -> str:
        return f"#{(blocks[index].sequence or '').strip()}"

    corrections: dict[int, list[str]] = {}
    batches = chunked(target_ids, batch_size)
    for batch_index, requested_ids in enumerate(batches, start=1):
        check_cancelled(is_cancelled)
        message = f"교정 진행 중: 배치 {batch_index}/{len(batches)}"
        if on_log:
            on_log(message)
        else:
            print(message)
        payload: list[dict[str, object]] = [
            {
                "id": i,
                "lines": format_dialogue_lines(blocks[i].text_lines)
                if is_two_speaker_dialogue(blocks[i].text_lines)
                else blocks[i].text_lines,
            }
            for i in requested_ids
        ]

        corrected = correct_batch_with_retry(
            corrector, payload, wrap_length, on_log=on_log, is_cancelled=is_cancelled,
        )
        check_cancelled(is_cancelled)
        if on_progress:
            on_progress(batch_index, len(batches))
        if corrected is None:
            logs.append(
                f"[실패] 배치 {batch_index}: 자막 {label(requested_ids[0])}~"
                f"{label(requested_ids[-1])} 교정 실패로 원본 유지"
            )
            continue

        # 요청하지 않은 id는 다른 자막을 덮어쓸 수 있으므로 무시한다.
        for item in corrected:
            if item.id in requested_ids:
                corrections[item.id] = item.corrected_lines

        for missing_id in requested_ids:
            if missing_id not in corrections:
                logs.append(f"[누락] 자막 {label(missing_id)}: 응답에 없어 원본 유지")

    revised: list[SubtitleBlock] = []
    for i, block in enumerate(blocks):
        corrected_lines = corrections.get(i)
        if corrected_lines is None:
            revised.append(block)
            continue

        final_lines, notes = sanitize_lines(block.text_lines, corrected_lines, wrap_length)
        for note in notes:
            tag, _, message = note.partition(" ")
            logs.append(f"{tag} 자막 {label(i)}: {message}")

        revised.append(block.model_copy(update={"text_lines": final_lines}))

    return revised, logs


def output_path_for(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_revised{input_path.suffix}")


class CorrectionCancelled(Exception):
    pass


def check_cancelled(is_cancelled: Callable[[], bool] | None) -> None:
    if is_cancelled and is_cancelled():
        raise CorrectionCancelled("교정이 중단되었습니다.")


def collect_srt_files(
    paths: Iterable[Path],
    is_cancelled: Callable[[], bool] | None = None,
) -> list[Path]:
    found: set[Path] = set()

    def raise_walk_error(error: OSError) -> None:
        raise error

    for path in paths:
        check_cancelled(is_cancelled)
        path = path.resolve()
        if path.is_dir():
            for directory, _, names in os.walk(path, onerror=raise_walk_error):
                check_cancelled(is_cancelled)
                for name in names:
                    candidate = Path(directory) / name
                    if candidate.suffix.lower() == ".srt" and candidate.is_file():
                        found.add(candidate.resolve())
        elif path.is_file() and path.suffix.lower() == ".srt":
            found.add(path)
        elif not path.exists():
            raise FileNotFoundError(f"입력 경로를 찾을 수 없습니다: {path}")
    return sorted(found, key=lambda path: str(path).casefold())


def correct_file(
    input_path: Path,
    corrector: Any,
    wrap_length: int | None = None,
    *,
    overwrite: bool = False,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> tuple[Path, list[str]]:
    check_cancelled(is_cancelled)
    content, encoding = decode_srt(input_path.read_bytes())
    blocks = parse_srt_blocks(content)
    revised, logs = revise_subtitles(
        blocks, corrector, wrap_length,
        on_log=on_log, on_progress=on_progress, is_cancelled=is_cancelled,
    )
    check_cancelled(is_cancelled)
    data = render_srt(revised, detect_newline(content)).encode(encoding)
    output = output_path_for(input_path)
    number = 1
    while True:
        try:
            with output.open("wb" if overwrite else "xb") as stream:
                stream.write(data)
            break
        except FileExistsError:
            number += 1
            output = input_path.with_name(f"{input_path.stem}_revised_{number}{input_path.suffix}")
    return output, logs


def run(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    input_path: Path = args.srt_file

    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    content, encoding = decode_srt(input_path.read_bytes())
    newline = detect_newline(content)

    api_key, model, effort = load_environment(args.model)
    corrector = build_corrector(api_key, model, effort)
    print(f"모델: {model} (추론 강도: {effort})")

    blocks = parse_srt_blocks(content)
    wrap_length = args.max_line_length if args.wrap else None
    revised_blocks, logs = revise_subtitles(blocks, corrector, wrap_length)

    # 텍스트 모드의 줄바꿈 자동 변환을 피하기 위해 바이트로 저장한다.
    output_path = output_path_for(input_path)
    output_path.write_bytes(render_srt(revised_blocks, newline).encode(encoding))

    for log in logs:
        print(log, file=sys.stderr)
    if logs:
        print(f"검토 로그 {len(logs)}건: 위 항목을 결과 파일에서 확인하세요.", file=sys.stderr)
    print(f"완료: {output_path}")
    return output_path


def main() -> None:
    if len(sys.argv) == 1:
        from srt_spellchecker_gui import main as gui_main

        gui_main()
        return
    try:
        run()
    except (FileNotFoundError, ValueError, UnicodeDecodeError, openai.APIError) as error:
        print(f"[오류] {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
