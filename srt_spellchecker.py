from __future__ import annotations

import argparse
import codecs
from collections.abc import Callable, Iterable
import json
import math
import os
import random
import re
import sys
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from time import monotonic, sleep
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import openai
from pydantic import BaseModel, Field

DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "low"
BATCH_SIZE = 25
MAX_BATCH_SIZE = 200
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


def expand_subtitle_lines(lines: list[str]) -> list[str]:
    """실제 줄바꿈과 한국어 문장 뒤의 공백 슬래시 대사 표식을 분리한다."""
    result: list[str] = []
    for line in lines:
        # 경로, 분수, 선택지 표기를 대사로 오인하지 않도록 문장 종결 뒤로 제한한다.
        expanded = re.sub(
            r"([가-힣]*(?:요|다|까|죠|지|네|어|아|야|해|돼|자)[.!?…]*)[ \t]+/(?=[ \t]*[가-힣])",
            r"\1\n/", line,
        )
        result.extend(expanded.splitlines() or [""])
    return result


def subtitle_segments(lines: list[str]) -> list[list[str]]:
    """대사 표식과 서식 줄의 경계를 보존하여 이어지는 일반 줄을 묶는다."""
    segments: list[list[str]] = []
    for line in lines:
        marked = re.match(r"^\s*[-/]\s*\S", line) is not None
        formatted = re.search(r"<[^>]+>|\{\\[^}]+\}", line) is not None
        previous_formatted = bool(segments and re.search(r"<[^>]+>|\{\\[^}]+\}", segments[-1][-1]))
        if not segments or marked or formatted or previous_formatted:
            segments.append([line])
        else:
            segments[-1].append(line)
    return segments


def format_subtitle_segments(segments: list[list[str]]) -> list[str]:
    dialogue = any(re.match(r"^\s*[-/]\s*\S", line) for group in segments for line in group)
    result = []
    for group in segments:
        if dialogue and not any(re.search(r"<[^>]+>|\{\\[^}]+\}", line) for line in group):
            result.append("- " + " ".join(re.sub(r"^[-/]\s*", "", line.strip(), count=1) for line in group))
        else:
            result.append(" ".join(line.strip() for line in group))
    return result


def prepare_subtitle_lines(lines: list[str]) -> list[str]:
    expanded = expand_subtitle_lines(lines)
    segments = subtitle_segments(expanded)
    return format_subtitle_segments(segments) if len(segments) > 1 else expanded


def sanitize_lines(
    original_lines: list[str],
    corrected_lines: list[str],
    wrap_length: int | None,
) -> tuple[list[str], list[str]]:
    """모델 교정 결과를 검증해 (확정 줄 목록, 검토 로그)를 반환한다."""
    notes: list[str] = []
    source_lines = list(original_lines)
    response_lines = list(corrected_lines)

    def revert_note(reason: str) -> str:
        details = [
            f"[되돌림] {reason}",
            f"  원본(검증 전, {len(source_lines)}줄): {json.dumps(source_lines, ensure_ascii=False)}",
            f"  모델 응답({len(response_lines)}줄): {json.dumps(response_lines, ensure_ascii=False)}",
        ]
        if original_lines != source_lines:
            details.append(
                f"  검사 원본(정리 후, {len(original_lines)}줄): {json.dumps(original_lines, ensure_ascii=False)}"
            )
        if corrected_lines != response_lines:
            details.append(
                f"  검사 응답(정리 후, {len(corrected_lines)}줄): {json.dumps(corrected_lines, ensure_ascii=False)}"
            )
        return "\n".join(details)

    original_lines = [normalize_subtitle_punctuation(line) for line in original_lines]
    corrected_lines = [normalize_subtitle_punctuation(line) for line in corrected_lines]

    original_lines = expand_subtitle_lines(original_lines)
    corrected_lines = expand_subtitle_lines(corrected_lines)
    segments = subtitle_segments(original_lines)
    if len(segments) > 1:
        source_line_count = len(original_lines)
        original_lines = format_subtitle_segments(segments)
        if len(corrected_lines) == source_line_count:
            response_segments = []
            offset = 0
            for group in segments:
                response_segments.append(corrected_lines[offset:offset + len(group)])
                offset += len(group)
        elif len(corrected_lines) == len(segments):
            response_segments = [[line] for line in corrected_lines]
        else:
            response_segments = subtitle_segments(corrected_lines)
        candidate = format_subtitle_segments(response_segments)
        # 서식은 설명 자막의 단서일 수 있으므로 같은 위치에 유지한다.
        tags = lambda line: re.findall(r"<[^>]+>|\{\\[^}]+\}", line)
        valid = len(candidate) == len(original_lines) and all(
            re.sub(r"^[-/]\s*", "", new.strip(), count=1).strip()
            and tags(old) == tags(new)
            and (not tags(old) or bool(re.match(r"^\s*[-/]", old)) == bool(re.match(r"^\s*[-/]", new)))
            for old, new in zip(original_lines, candidate)
        )
        if not valid:
            notes.append(revert_note("대사·서식 경계 불일치로 원본 구성 유지"))
            result = original_lines
        else:
            # 응답에서 생략된 대사 표식도 원본 구분에 맞춰 복구한다.
            result = [
                "- " + re.sub(r"^[-/]\s*", "", new.strip(), count=1)
                if old.startswith("- ") else new
                for old, new in zip(original_lines, candidate)
            ]
        if wrap_length is not None and any(len(line) > wrap_length for line in result):
            notes.append(f"[확인필요] {wrap_length}자/{MAX_WRAPPED_LINES}줄 초과: 대사·서식 구분 유지")
        return result, notes

    lines = [line.strip() for line in corrected_lines if line.strip()]
    if not lines:
        notes.append(revert_note("빈 교정 결과로 원본 유지"))
        lines = list(original_lines)

    if wrap_length is not None and violates_wrap_rules(lines, wrap_length):
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
        line_rule = "아래 대사 표식 정리에 필요한 경우 외에는 입력의 줄을 합치거나 나누지 않는다. "
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
        "입력의 줄바꿈을 유지한다. 두 줄이 짧거나 합쳐도 한 줄에 들어간다는 이유, "
        "같은 문장이나 같은 사람의 발언이라는 이유만으로 줄을 합치지 않는다. "
        "줄 구성 변경은 아래 대사 표식 정리와 명시된 긴 줄 나누기 규칙에 필요한 경우에만 허용한다. "
        f"{line_rule}"
        "줄 시작의 / 또는 -는 대사 경계이며, '마시지요 /네'처럼 문장 뒤의 /도 문맥상 대답이면 대사를 나눈다. "
        "경로, URL, 분수, '커피/차' 같은 선택지의 /는 대사 표식이 아니다. "
        "대사 표식으로 여러 발언이 구분된 블록을 정리할 때에만, "
        "한 발언에 속한 여러 줄을 공백으로 연결해 발언당 한 줄로 만들 수 있다. "
        "표식으로 구분된 발언은 각각 별도 줄로 유지한다. "
        "각 대사 앞의 / 또는 - 표기를 '- ' (하이픈과 공백 한 칸)으로 통일하고, "
        "첫 대사에도 '- '를 붙인다. 서식 태그가 있는 설명 자막에는 대사 표식을 임의로 붙이지 않는다. "
        "대사와 서식 줄의 구분은 길이·최대 줄 수 제한보다 우선하며 세 발언을 두 줄로 줄이지 않는다. "
        "다른 발언끼리 합치거나 내용을 옮기지 않고, 서식 태그와 설명 자막·대사 사이의 경계를 유지한다. "
        "대사 구분 표식이 없는 자막을 임의로 두 사람의 대사로 바꾸지 않는다. "
        "기존 말줄임표는 점 두 개(..), 연속된 점, 특수문자 표기 모두 점 세 개(...)로 통일한다. "
        "한국어 자막의 문장 끝 마침표(.)는 생략한다. 원문에 있어도 제거하며 "
        "따옴표나 닫는 서식 태그 앞의 문장 끝 마침표도 제거한다. "
        "말줄임표(...)와 소수점, URL 및 약어 내부의 점은 마침표와 혼동하지 말고 유지한다. "
        "문맥과 문장 구조에 필요한 문장부호는 추가하거나 수정할 수 있다. 불필요하게 추가하거나 반복하지 않는다. "
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


def rate_limit_delay(error: openai.RateLimitError, attempt: int) -> float:
    """서버의 재시도 시각을 우선하고 없으면 지수 대기와 무작위 지연을 적용한다."""
    headers = error.response.headers
    for name, divisor in (("retry-after-ms", 1000), ("retry-after", 1)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            seconds = float(value) / divisor
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                continue
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    return min(2 ** attempt, 60) + random.uniform(0, 1)


def wait_for_retry(seconds: float, is_cancelled: Callable[[], bool] | None) -> None:
    deadline = monotonic() + seconds
    while True:
        check_cancelled(is_cancelled)
        remaining = deadline - monotonic()
        if remaining <= 0:
            return
        sleep(min(remaining, 0.1))


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
            if isinstance(error, openai.RateLimitError) and attempt < attempt_limit:
                delay = rate_limit_delay(error, attempt)
                message = f"[요청 제한] {delay:.1f}초 대기 후 재시도"
                if on_log:
                    on_log(message)
                else:
                    print(message, file=sys.stderr)
                wait_for_retry(delay, is_cancelled)
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
                "lines": prepare_subtitle_lines(blocks[i].text_lines),
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
            final_lines, notes = block.text_lines, []
        else:
            final_lines, notes = sanitize_lines(block.text_lines, corrected_lines, wrap_length)

        # 누락·실패로 원본을 유지한 경우에도 최종 자막의 줄 수를 검사한다.
        if block.is_subtitle and len(final_lines) > MAX_WRAPPED_LINES:
            notes.append(
                f"[확인필요] 최종 자막 {len(final_lines)}줄: 최대 {MAX_WRAPPED_LINES}줄 초과, "
                "타임스탬프 분리 등 수동 편집 필요"
            )
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
    on_progress: Callable[[int, str], None] | None = None,
) -> list[Path]:
    found: set[Path] = set()
    last_report: float | None = None
    current_location = ""

    def report_progress(*, force: bool = False) -> None:
        nonlocal last_report
        now = monotonic()
        if on_progress is not None and (force or last_report is None or now - last_report >= 0.1):
            on_progress(len(found), current_location)
            last_report = now

    def raise_walk_error(error: OSError) -> None:
        raise error

    for path in paths:
        check_cancelled(is_cancelled)
        current_location = str(path)
        report_progress()
        path = path.resolve()
        if path.is_dir():
            for directory, _, names in os.walk(path, onerror=raise_walk_error):
                check_cancelled(is_cancelled)
                current_location = str(directory)
                report_progress()
                for name in names:
                    check_cancelled(is_cancelled)
                    candidate = Path(directory) / name
                    if candidate.suffix.lower() == ".srt" and candidate.is_file():
                        found.add(candidate.resolve())
                    report_progress()
        elif path.is_file() and path.suffix.lower() == ".srt":
            found.add(path)
        elif not path.exists():
            raise FileNotFoundError(f"입력 경로를 찾을 수 없습니다: {path}")
    check_cancelled(is_cancelled)
    report_progress(force=True)
    return sorted(found, key=lambda path: str(path).casefold())


def correct_file(
    input_path: Path,
    corrector: Any,
    wrap_length: int | None = None,
    *,
    batch_size: int = BATCH_SIZE,
    overwrite: bool = False,
    on_log: Callable[[str], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> tuple[Path, list[str]]:
    check_cancelled(is_cancelled)
    content, encoding = decode_srt(input_path.read_bytes())
    blocks = parse_srt_blocks(content)
    revised, logs = revise_subtitles(
        blocks, corrector, wrap_length, batch_size=batch_size,
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
