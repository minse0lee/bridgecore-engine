"""Claude로 책 문장 번역 (MADLAD 번역과 비교용). 문단 단위로 요청하되 문장 id별 번역을 받아 6축 검증과 정렬을 맞춘다.

    python translation/llm_translate.py data/book/gatsby/sentences.jsonl OUT.jsonl --chapters I III --live

- 입력 문장 단위는 MADLAD와 같다(epub_tools extract). 번역 단위는 문단(앞 문단 2개 원문을 문맥으로 함께 줌)
- 출력 형식은 04 노트북과 같음: {id, en, mt, mt_system, mt_meta}
- 고유명사 표기·말투 지침은 SYSTEM에 고정 (프롬프트 해시를 mt_meta에 기록)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridgecore.runner import get_client  # noqa: E402

SYSTEM = """당신은 영미 문학 전문 번역가입니다. F. Scott Fitzgerald의 소설 『The Great Gatsby』(1925)를 한국어 독자를 위한 출판용 번역으로 옮깁니다.

원칙
1. 의미 보존: 원문의 내용·뉘앙스·인물의 의도를 빠짐없이 옮기고, 원문에 없는 내용을 덧붙이지 않습니다.
2. 문체: 자연스러운 한국어 문학 문장으로 씁니다. 영어 어순·구문을 그대로 따르는 직역투(예: "~하는 것은 ~이다", 과도한 대명사 "그것", 무생물 주어)를 피합니다.
3. 서술: 화자 닉 캐러웨이의 1인칭 회고 서술은 '-다'체로 씁니다.
4. 대사: 인물의 관계·성격·상황에 맞는 구어체로 씁니다(친한 사이는 반말, 처음 만난 사이·격식 있는 자리는 존댓말 등). 대사에서 "~는가?", "~하라" 같은 문어체를 쓰지 않습니다.
5. 시대 어휘: 1920년대 미국 영어의 당시 의미로 옮깁니다(예: gay = 유쾌한/즐거운, queer = 이상한).
6. 호칭: Miss ○○은 "○○ 양"으로, Mr. ○○은 "○○ 씨"로 옮깁니다. 직업 호칭(선생님 등)으로 바꾸지 않습니다.
7. 고유명사 표기: Nick Carraway 닉 캐러웨이, Jay Gatsby 제이 개츠비, Tom Buchanan 톰 뷰캐넌, Daisy 데이지, Jordan Baker 조던 베이커, Myrtle Wilson 머틀 윌슨, George Wilson 조지 윌슨, West Egg 웨스트에그, East Egg 이스트에그, Long Island 롱아일랜드, the Sound 해협.
8. 정렬: 입력의 각 문장(id)마다 번역을 정확히 하나씩 돌려줍니다. 문장을 합치거나 나누거나 순서를 바꾸지 않습니다. 한국어 어순에 맞게 한 문장 안에서 재배열하는 것은 괜찮습니다. 따옴표("…")와 대사 구분은 원문을 따릅니다.
"""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["translations"],
          "properties": {"translations": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["id", "ko"],
              "properties": {"id": {"type": "string"}, "ko": {"type": "string"}}}}}}
PROMPT_HASH = hashlib.sha256((SYSTEM + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()[:12]


def blocks(rows, chapters):
    out, cur = [], None
    for r in rows:
        if r["kind"] != "body" or r["skip"] == "pg_boilerplate":
            continue
        key = (r["file"], r["xpath"])
        if cur is None or cur["key"] != key:
            cur = {"key": key, "chapter": r["chapter"], "para": r["paragraph"], "sents": []}
            out.append(cur)
        cur["sents"].append(r)
    idx = [k for k, b in enumerate(out) if (not chapters or b["chapter"] in chapters) and any(s["translate"] for s in b["sents"])]
    return out, idx


def request(allb, k, model):
    b = allb[k]
    ctx = [x["para"] for x in allb[max(0, k - 2):k]]
    sents = [s for s in b["sents"] if s["translate"]]
    user = ("[앞 문단 원문 — 문맥 참고용, 번역하지 마세요]\n" + ("\n\n".join(ctx) if ctx else "(없음)") +
            "\n\n[이번 문단 원문 전체]\n" + b["para"] +
            "\n\n[번역할 문장 — id마다 하나씩]\n" + "\n".join(f'{s["id"]}\t{s["en"]}' for s in sents))
    return sents, {"model": model, "max_tokens": 8000,
                   "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                   "messages": [{"role": "user", "content": user}],
                   "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "medium"}}


def call(client, sents, req, tries=2):
    want = [s["id"] for s in sents]
    last = None
    for _ in range(tries):
        try:
            resp = client.messages.create(**req)
            text = "".join(getattr(c, "text", "") or "" for c in resp.content if getattr(c, "type", "") == "text")
            got = {t["id"]: t["ko"].strip() for t in json.loads(text)["translations"]}
            u = resp.usage
            usage = {k: getattr(u, k, 0) or 0 for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
            if [i for i in want if not got.get(i)]:
                last = f"id 누락 {[i for i in want if not got.get(i)]}"
                continue
            return got, usage, None
        except Exception as e:  # noqa: BLE001
            last = f"{e.__class__.__name__}: {e}"
    return {}, None, last


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("sentences", type=Path); p.add_argument("out", type=Path)
    p.add_argument("--chapters", nargs="*"); p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument("--workers", type=int, default=6); p.add_argument("--live", action="store_true")
    a = p.parse_args(argv)
    rows = [json.loads(l) for l in open(a.sentences, encoding="utf-8") if l.strip()]
    allb, idx = blocks(rows, a.chapters)
    done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8")} if a.out.exists() else set()
    todo = [k for k in idx if any(s["translate"] and s["id"] not in done for s in allb[k]["sents"])]
    n_s = sum(sum(s["translate"] for s in allb[k]["sents"]) for k in todo)
    print(f"문단 {len(todo)}개 · 문장 {n_s}개 · 모델 {a.model} · 프롬프트 {PROMPT_HASH}")
    if not a.live:
        print("실제 실행: --live"); return
    client = get_client()
    meta = {"model": a.model, "prompt_hash": PROMPT_HASH, "unit": "문단(앞 문단 2개 문맥)", "effort": "medium"}
    tot = {}
    fails = []

    def work(k):
        sents, req = request(allb, k, a.model)
        return k, sents, *call(client, sents, req)

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a", encoding="utf-8") as f:
        for n, (k, sents, got, usage, err) in enumerate(ex.map(work, todo), 1):
            if err:
                fails.append((k, err)); continue
            for kk, v in usage.items():
                tot[kk] = tot.get(kk, 0) + v
            for s in sents:
                f.write(json.dumps({"id": s["id"], "en": s["en"], "mt": got[s["id"]], "mt_system": f"claude_{a.model}",
                                    "mt_meta": meta}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"  {n}/{len(todo)} · {(time.time() - t0) / 60:.1f}분")
    print("완료", len(todo) - len(fails), "/", len(todo), "실패", fails[:5])
    print("토큰 합계:", tot)


if __name__ == "__main__":
    main()
