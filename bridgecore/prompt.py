"""프롬프트·요청 빌더.

- 시스템 프롬프트: prompts/<version>/system.md (가이드라인 요약 + 연습 예시). 모든 문항에서
  똑같으므로 cache_control을 붙여 프롬프트 캐싱으로 반복 비용을 줄인다.
- 사용자 메시지: 문항 하나의 문서정보·brief·문맥·평가대상 + 자동 점검 힌트 (+ 그림).
- 출력: 구조화 출력(output_config.format, json_schema)으로 JSON 형식을 보장한다.

Claude 5.5 세대 API 제약 (공식 문서 2026-10 확인):
  - Sonnet 5.5는 tool_choice type "tool"/"any"(도구 호출 강제)를 400으로 거부 → 구조화 출력 사용
  - temperature·top_p·top_k는 넣지 않는다 (Sonnet 5.5는 400, Haiku 5.5도 생략 권장)
  - 적응형 사고(adaptive thinking)가 기본으로 켜져 있고 effort로 깊이를 조절한다.
    사고 토큰은 max_tokens 안에 포함된다.
  - 사고를 끄는 값이 모델마다 다르다: Haiku 5.5 {"type":"disabled"}, Sonnet 5.5 {"type":"between_tools"}
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from pathlib import Path

from .rules import hints_text, run_rules
from .schema import JSON_SCHEMA

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"

EFFORTS = ("low", "medium", "high")
_THINKING_OFF = {"claude-haiku": {"type": "disabled"}, "claude-sonnet": {"type": "between_tools"}}

USER_TEMPLATE = """다음 문항을 평가 지침에 따라 판정하고, 지정된 JSON 형식으로만 답하세요.

[문항 ID] {item_id}

[문서정보]
{doc_info}

[평가조건(brief)]
{brief}

[원문 문단] (문맥)
{src_paragraph}

[문단 전체 번역] (문맥 — 누락·정렬 확인용)
{tgt_paragraph}

[평가대상 원문]
{src}

[평가대상 번역문]
{tgt}
{figure_note}{hints}"""

PARAGRAPH_ONLY_NOTE = ("(이 문항은 번역 문장이 따로 짝지어져 있지 않습니다. [문단 전체 번역]에서 평가대상 원문에 "
                       "대응하는 구간을 직접 찾아 그 구간을 평가하세요. 대응 구간을 특정할 수 없으면 "
                       "지침 8절에 따라 판단보류와 \"정렬 검토 필요\"로 두세요.)")


def system_text(version: str = "v0", screen: bool = False) -> str:
    """screen=True면 1차 선별용 추가문(screen_addendum.md)을 뒤에 붙인다 (넉넉하게 표시)."""
    text = (PROMPTS / version / "system.md").read_text(encoding="utf-8")
    if screen:
        add = PROMPTS / version / "screen_addendum.md"
        if not add.exists():
            raise FileNotFoundError(f"선별용 추가문이 없습니다: {add}")
        text += add.read_text(encoding="utf-8")
    return text


def prompt_fingerprint(version: str = "v0", screen: bool = False) -> str:
    """개발일지용: 시스템 프롬프트 + 출력 스키마 + 문항 틀 + 규칙 코드 해시 (엔진 버전 고정 기록).
    모델·effort·thinking은 meta.json에 따로 기록된다."""
    h = hashlib.sha256(system_text(version, screen).encode("utf-8"))
    h.update(json.dumps(JSON_SCHEMA, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    h.update(USER_TEMPLATE.encode("utf-8"))
    h.update(PARAGRAPH_ONLY_NOTE.encode("utf-8"))
    # 힌트 문장은 rules.py가 만들므로 규칙 코드가 바뀌어도 해시가 바뀌게 한다
    h.update((Path(__file__).parent / "rules.py").read_bytes())
    return h.hexdigest()[:12]


def paragraph_only(item: dict) -> bool:
    return not (item.get("tgt") or "").strip() and bool((item.get("tgt_paragraph") or "").strip())


def user_text(item: dict) -> str:
    para = paragraph_only(item)
    if para:
        # 문장 짝이 없으면 문단 단위로 숫자를 비교한다 (다른 문장의 숫자가 섞이지 않게)
        rules = run_rules({**item, "src": item.get("src_paragraph") or item.get("src"),
                           "tgt": item.get("tgt_paragraph")})
        rules["target_language_missing"] = False
    else:
        rules = run_rules(item)
    hint = hints_text(rules)
    return USER_TEMPLATE.format(
        item_id=item["item_id"],
        doc_info=item.get("doc_info") or "(없음)",
        brief=item.get("brief") or "(없음)",
        src_paragraph=item.get("src_paragraph") or "(없음)",
        tgt_paragraph=item.get("tgt_paragraph") or "(없음)",
        src=item.get("src") or "",
        tgt=PARAGRAPH_ONLY_NOTE if para else (item.get("tgt") or "(빈 출력)"),
        figure_note="\n[그림 자료] 이 문항의 원문 그림을 함께 첨부했습니다. 문맥으로만 사용하세요.\n"
        if item.get("figure") else "",
        hints=f"\n[자동 점검 힌트]\n{hint}\n" if hint else "",
    )


def build_messages(item: dict, version: str = "v0",
                   screen: bool = False) -> tuple[list[dict], list[dict]]:
    system = [{"type": "text", "text": system_text(version, screen),
               "cache_control": {"type": "ephemeral"}}]
    content = []
    fig = item.get("figure")
    if fig:
        path = ROOT / fig
        media = mimetypes.guess_type(str(path))[0] or "image/png"
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": media,
            "data": base64.b64encode(path.read_bytes()).decode("ascii")}})
    content.append({"type": "text", "text": user_text(item)})
    return system, [{"role": "user", "content": content}]


def thinking_off(model: str) -> dict:
    for prefix, value in _THINKING_OFF.items():
        if model.startswith(prefix):
            return value
    raise ValueError(f"이 모델의 사고 끄기 값을 모릅니다: {model} (prompt.py _THINKING_OFF에 추가)")


def build_request(item: dict, model: str, version: str = "v0", max_tokens: int = 16000,
                  screen: bool = False, effort: str | None = "medium",
                  thinking: bool = True) -> dict:
    """messages.create / Message Batches에 그대로 넣을 수 있는 요청 본문.
    effort=None이면 모델 기본값(Haiku 5.5 medium, Sonnet 5.5 high). max_tokens는 사고 토큰을 포함한다."""
    system, messages = build_messages(item, version, screen)
    output_config: dict = {"format": {"type": "json_schema", "schema": JSON_SCHEMA}}
    if effort:
        if effort not in EFFORTS:
            raise ValueError(f"effort는 {EFFORTS} 중 하나: {effort}")
        output_config["effort"] = effort
    req = {"model": model, "max_tokens": max_tokens, "system": system,
           "messages": messages, "output_config": output_config}
    if not thinking:
        req["thinking"] = thinking_off(model)
    return req
