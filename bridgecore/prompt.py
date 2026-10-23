"""프롬프트 빌더.

- 시스템 프롬프트: prompts/<version>/system.md (가이드라인 요약 + 연습 예시). 모든 문항에서
  똑같으므로 cache_control을 붙여 프롬프트 캐싱으로 반복 비용을 줄인다.
- 사용자 메시지: 문항 하나의 문서정보·brief·문맥·평가대상 + 자동 점검 힌트 (+ 그림).
- 출력: submit_judgment 도구 호출을 강제해 JSON 형식을 보장한다.
"""
from __future__ import annotations

import base64
import hashlib
import mimetypes
from pathlib import Path

from .rules import hints_text, run_rules
from .schema import JSON_SCHEMA

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"

TOOL_NAME = "submit_judgment"
TOOL = {
    "name": TOOL_NAME,
    "description": "한 문항의 평가 가능 여부, 오류 목록(먼저), 6축 판정(다음)을 제출합니다.",
    "input_schema": JSON_SCHEMA,
}

USER_TEMPLATE = """다음 문항을 평가 지침에 따라 판정하고 submit_judgment로 제출하세요.

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
    """개발일지용: 시스템 프롬프트 + 도구 스키마 + 문항 틀 + 규칙 코드 해시 (엔진 버전 고정 기록)."""
    h = hashlib.sha256(system_text(version, screen).encode("utf-8"))
    h.update(repr(TOOL).encode("utf-8"))
    h.update(USER_TEMPLATE.encode("utf-8"))
    # 힌트 문장은 rules.py가 만들므로 규칙 코드가 바뀌어도 해시가 바뀌게 한다
    h.update((Path(__file__).parent / "rules.py").read_bytes())
    return h.hexdigest()[:12]


def user_text(item: dict) -> str:
    rules = run_rules(item)
    hint = hints_text(rules)
    return USER_TEMPLATE.format(
        item_id=item["item_id"],
        doc_info=item.get("doc_info") or "(없음)",
        brief=item.get("brief") or "(없음)",
        src_paragraph=item.get("src_paragraph") or "(없음)",
        tgt_paragraph=item.get("tgt_paragraph") or "(없음)",
        src=item.get("src") or "",
        tgt=item.get("tgt") or "(빈 출력)",
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


def build_request(item: dict, model: str, version: str = "v0", max_tokens: int = 4000,
                  screen: bool = False) -> dict:
    """messages.create / Message Batches에 그대로 넣을 수 있는 요청 본문."""
    system, messages = build_messages(item, version, screen)
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": system,
        "messages": messages,
        "tools": [TOOL],
        "tool_choice": {"type": "tool", "name": TOOL_NAME},
    }
