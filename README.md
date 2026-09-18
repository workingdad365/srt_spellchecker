# srt_spellchecker

SRT 자막 파일의 오타와 띄어쓰기를 OpenAI LLM으로 교정하는 도구.
HARDSUB 추출 후 OCR로 생성한 파일이나, 맞춤법을 무시하고 제작된 자막의 교정 용도로 사용한다.

## 특징

- 자막 번호, 타임코드, 비자막 블록, 줄바꿈(CRLF/LF), BOM 유무를 원본 그대로 유지
- UTF-8(BOM 포함) 입력 지원, CP949 입력은 UTF-8로 변환해 저장
- 모델이 규칙을 어긴 줄은 코드에서 원본으로 되돌림
  - 줄 수가 달라진 자막
  - 원본에 없던 문장부호가 추가된 줄
- 배치 단위 재시도 (최대 3회), 끝내 실패한 배치는 원본을 유지하고 나머지를 계속 처리
- 되돌림/누락/실패 내역을 검토 로그로 출력 (stderr)
- `--wrap` 옵션으로 긴 줄을 최대 2줄로 나눔 (기본 23자)

## 설정

1. `.env.example`을 복사해 `.env`를 만든다.
2. `.env`에 OpenAI 정보를 입력한다.

| 환경 변수 | 필수 | 기본값 | 설명 |
|---|---|---|---|
| `OPENAI_API_KEY` | O | - | OpenAI API 키 |
| `OPENAI_MODEL` | X | `gpt-5.6-luna` | 사용할 모델 |
| `OPENAI_REASONING_EFFORT` | X | `low` | 추론 강도 (`none`, `low`, `medium`, `high` 등 모델이 지원하는 값) |

## 실행

프로젝트 디렉터리에서 editable 방식으로 설치한다. 설치 후 소스 코드 변경 사항이 바로 반영된다.

```powershell
uv tool install --editable .
```

설치 후 어느 디렉터리에서든 명령으로 실행한다.

```powershell
srt-spellchecker <srt파일>
```

설치 없이 프로젝트 디렉터리에서 직접 실행할 수도 있다.

```powershell
uv run srt_spellchecker.py <srt파일>
```

| 옵션 | 설명 |
|---|---|
| `--wrap` | 긴 줄을 최대 2줄로 나눔. 모델이 규칙을 어기면 코드에서 공백 기준으로 다시 나눔 |
| `--max-line-length N` | `--wrap` 사용 시 한 줄 최대 글자 수 (기본 23) |
| `--model NAME` | 모델 지정 (`OPENAI_MODEL`보다 우선) |

실행 결과는 `<원본파일명>_revised.srt` 파일로 생성되며, 같은 이름이 있으면 덮어쓴다.

## 검토 로그

| 태그 | 의미 |
|---|---|
| `[되돌림]` | 줄 수 불일치 또는 문장부호 추가로 원본 유지 |
| `[재분할]` | `--wrap` 규칙 위반으로 코드에서 줄을 다시 나눔 |
| `[확인필요]` | `--wrap` 규칙을 맞출 수 없어 수동 확인 필요 |
| `[누락]` | 모델 응답에 해당 자막이 없어 원본 유지 |
| `[실패]` | 배치가 재시도 후에도 실패해 원본 유지 |

## 모델 관련 참고

GPT-5.x 계열 기준으로 작성됨.

- `temperature`, `top_p`, `max_tokens` 등 미지원 파라미터를 전달하지 않음
- Responses API와 `reasoning.effort`로 추론 강도를 제어함
- 구조화 출력(JSON 스키마, strict)으로 응답 형식을 강제함
- 인증 오류, 잘못된 요청, 없는 모델 등 재시도해도 의미 없는 오류는 즉시 중단함

## 테스트

```powershell
uv run pytest
```
