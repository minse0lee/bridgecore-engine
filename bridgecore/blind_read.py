"""번역문만 읽은 독자의 해석 (원문을 보지 않는 별도 호출).

판정 모델은 원문을 함께 보기 때문에 어색한 번역도 원래 뜻으로 되돌려 읽기 쉽다
(예: 사물을 받는 '그들은'을 그 사물로 읽음, '9시간'을 학점으로 읽음).
그래서 원문 없이 번역문(과 한국어 문단 문맥, 평가조건)만 준 호출로 '한국어 독자가 이해하는 내용'을
먼저 적게 하고, 판정 프롬프트(v1.3b~)에 참고 정보로 넣는다.

결과는 문항 jsonl 사본에 `blind_reading` 필드로 저장한다 (원본 문항 파일은 건드리지 않음).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .prompt import paragraph_only
from .runner import get_client

SYSTEM = """당신은 영어 원문을 전혀 모르는 한국어 독자입니다. 주어진 한국어 문장만 읽고, 이 문장이 무엇을 말하는지 독자로서 이해한 대로 적습니다.
- 번역문이라는 점을 고려해 원래 뜻을 추측하거나 고쳐 읽지 않습니다. 한국어로 쓰인 그대로, 보통의 한국어 독자가 받아들일 뜻을 적습니다.
- 어색하거나 뜻이 둘 이상으로 읽히는 부분은 그렇다고 적습니다.
- 평가나 번역 품질 의견은 쓰지 않습니다. 해석만 씁니다.

아래 형식으로 짧게 답합니다(전체 8줄 이내).
뜻: (이 문장이 말하는 내용을 쉬운 한국어 한두 문장으로 풀어 씀)
가리키는 대상: (그·그들·이것 같은 대명사·지시어가 무엇을 가리키는 것으로 읽히는지. 없으면 "없음")
행위: (누가 무엇을 어디로/어디에서/어떻게 하는지)
수량·단위·용어: (수량과 단위, 전문 용어·제도 용어를 독자가 무엇으로 이해하는지. 없으면 "없음")
확실성·태도: (단정·추측·기대·의향·가능성·명령·권유 중 무엇으로 읽히는지)
애매한 부분: (없으면 "없음")"""

USER = """[평가조건]
{brief}

[한국어 문단 (문맥)]
{tgt_paragraph}

[읽을 문장]
{tgt}"""


def build_request(item: dict, model: str, max_tokens: int = 4000) -> dict:
    para = paragraph_only(item)
    tgt = item.get("tgt_paragraph") if para else item.get("tgt")
    return {"model": model, "max_tokens": max_tokens,
            "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": USER.format(
                brief=item.get("brief") or "(없음)", tgt_paragraph=item.get("tgt_paragraph") or "(없음)",
                tgt=tgt or "(빈 출력)")}],
            "output_config": {"effort": "low"}}


def _text(resp) -> str:
    return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text").strip()


def blind_read_items(items: list[dict], model: str, out_file: Path, workers: int = 6) -> Path:
    """items 각각에 blind_reading을 붙여 out_file(jsonl)에 쓴다. 이미 있는 문항은 다시 부르지 않는다."""
    done = {}
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("blind_reading"):
                    done[r["item_id"]] = r
    client = get_client()
    usage = {"input_tokens": 0, "output_tokens": 0}

    def one(it):
        if it["item_id"] in done:
            return done[it["item_id"]]
        resp = client.messages.create(**build_request(it, model))
        for k in usage:
            usage[k] += getattr(resp.usage, k, 0) or 0
        return {**it, "blind_reading": _text(resp), "blind_reading_model": model}

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, items))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(res)}문항 ({len(res) - len(done)}개 새로 호출) → {out_file} · 토큰 {usage}")
    return out_file
