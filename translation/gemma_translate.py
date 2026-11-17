"""Gemma 4 31B(API)로 개츠비 문장 번역 — Claude·MADLAD 번역과 같은 조건으로 비교하기 위한 스크립트.

이 파일 하나와 sentences.jsonl만 있으면 돌아간다(표준 라이브러리만 사용, Colab·노트북 PC 어디서나).

    # 키는 환경변수로: OPENROUTER_API_KEY (또는 --key-env 로 다른 이름)
    python gemma_translate.py sentences.jsonl gemma_ch1_3.jsonl --chapters I III            # 건수·요청 미리보기
    python gemma_translate.py sentences.jsonl gemma_ch1_3.jsonl --chapters I III --live     # 실제 실행

- 번역 지침(SYSTEM)·번역 단위(문단, 앞 문단 2개 원문을 문맥으로)·출력 형식은 llm_translate.py(Claude)와 같다.
  다른 점은 JSON 출력 지시 한 문단(FORMAT)뿐 — Claude는 API의 JSON 스키마 기능을 썼기 때문.
- 출력: {id, en, mt, mt_system, mt_meta} 한 줄씩. 중간에 끊겨도 다시 실행하면 이어서 한다.
- 기본 접속처는 OpenRouter(OpenAI 호환 API). --base-url 로 다른 OpenAI 호환 주소를 쓸 수 있다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# llm_translate.py의 SYSTEM과 글자 하나까지 같아야 한다(tests/test_gemma_translate.py가 확인).
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
FORMAT = """
출력 형식
설명이나 코드 블록 표시 없이 아래 JSON 하나만 출력합니다.
{"translations": [{"id": "문장 id", "ko": "번역문"}, ...]}
"""
PROMPT_HASH = hashlib.sha256((SYSTEM + FORMAT).encode()).hexdigest()[:12]
DEFAULT_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemma-4-31b-it"


def blocks(rows, chapters):
    """llm_translate.blocks와 같은 규칙: (파일, xpath)가 같은 문장을 한 문단으로 묶는다."""
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


def user_text(allb, k):
    b = allb[k]
    ctx = [x["para"] for x in allb[max(0, k - 2):k]]
    sents = [s for s in b["sents"] if s["translate"]]
    text = ("[앞 문단 원문 — 문맥 참고용, 번역하지 마세요]\n" + ("\n\n".join(ctx) if ctx else "(없음)") +
            "\n\n[이번 문단 원문 전체]\n" + b["para"] +
            "\n\n[번역할 문장 — id마다 하나씩]\n" + "\n".join(f'{s["id"]}\t{s["en"]}' for s in sents))
    return sents, text


def parse(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    if not t.startswith("{"):
        m = re.search(r"\{.*\}", t, re.S)
        t = m.group(0) if m else t
    try:
        return {d["id"]: (d.get("ko") or "").strip() for d in json.loads(t)["translations"]}
    except json.JSONDecodeError:
        # 번역문 안의 따옴표(")를 이스케이프하지 않아 JSON이 깨진 경우: id·ko 쌍을 직접 뽑는다
        pairs = re.findall(r'"id"\s*:\s*"([^"]+)"\s*,\s*"ko"\s*:\s*"(.*?)"\s*}\s*(?=,\s*\{|\]|$)', t, re.S)
        if not pairs:
            raise
        return {i: k.replace('\\"', '"').strip() for i, k in pairs}


def post(url, key, body, timeout=180):
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          "X-Title": "bridgecore-gatsby-gemma"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def call(a, key, sents, text, tries=4):
    want = [s["id"] for s in sents]
    body = {"model": a.model, "temperature": a.temperature, "max_tokens": 8000,
            "messages": [{"role": "system", "content": SYSTEM + FORMAT}, {"role": "user", "content": text}]}
    if a.json_mode:
        body["response_format"] = {"type": "json_object"}
    last = None
    for n in range(tries):
        try:
            resp = post(a.base_url, key, body)
            got = parse(resp["choices"][0]["message"]["content"] or "")
            miss = [i for i in want if not got.get(i)]
            if miss:
                last = f"id 누락 {miss}"
                continue
            return got, resp.get("usage") or {}, resp.get("provider"), None
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
            if e.code in (400, 401, 403, 404):
                break  # 요청 자체가 잘못됨 — 다시 해도 같다
            time.sleep(2 ** n * 3)
        except Exception as e:  # noqa: BLE001
            last = f"{e.__class__.__name__}: {e}"
            time.sleep(2 ** n)
    return {}, None, None, last


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("sentences", type=Path); p.add_argument("out", type=Path)
    p.add_argument("--chapters", nargs="*")
    p.add_argument("--model", default=DEFAULT_MODEL); p.add_argument("--base-url", default=DEFAULT_URL)
    p.add_argument("--key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--no-json-mode", dest="json_mode", action="store_false")
    p.add_argument("--workers", type=int, default=4); p.add_argument("--live", action="store_true")
    a = p.parse_args(argv)
    rows = [json.loads(l) for l in open(a.sentences, encoding="utf-8") if l.strip()]
    allb, idx = blocks(rows, a.chapters)
    done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8")} if a.out.exists() else set()
    todo = [k for k in idx if any(s["translate"] and s["id"] not in done for s in allb[k]["sents"])]
    n_s = sum(sum(s["translate"] for s in allb[k]["sents"]) for k in todo)
    print(f"문단 {len(todo)}개 · 문장 {n_s}개 · 모델 {a.model} · 프롬프트 {PROMPT_HASH} · 이미 끝난 문장 {len(done)}개")
    if not a.live:
        if todo:
            print("\n--- 첫 요청 미리보기 ---\n" + user_text(allb, todo[0])[1][:1200])
        print("\n실제 실행: --live"); return
    key = os.environ.get(a.key_env)
    if not key:
        sys.exit(f"환경변수 {a.key_env}가 비어 있습니다.")
    meta = {"model": a.model, "base_url": a.base_url, "prompt_hash": PROMPT_HASH, "unit": "문단(앞 문단 2개 문맥)",
            "temperature": a.temperature, "json_mode": a.json_mode}
    tot, fails, providers = {}, [], {}

    def work(k):
        sents, text = user_text(allb, k)
        return k, sents, *call(a, key, sents, text)

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a", encoding="utf-8") as f:
        for n, (k, sents, got, usage, prov, err) in enumerate(ex.map(work, todo), 1):
            if err:
                fails.append((k, err)); print("  실패", k, err[:200]); continue
            for kk in ("prompt_tokens", "completion_tokens"):
                tot[kk] = tot.get(kk, 0) + (usage.get(kk) or 0)
            if usage.get("cost") is not None:
                tot["cost_usd"] = round(tot.get("cost_usd", 0) + usage["cost"], 6)
            providers[prov] = providers.get(prov, 0) + 1
            for s in sents:
                f.write(json.dumps({"id": s["id"], "en": s["en"], "mt": got[s["id"]], "mt_system": a.model.split("/")[-1],
                                    "mt_meta": {**meta, "provider": prov}}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"  {n}/{len(todo)} · {(time.time() - t0) / 60:.1f}분")
    print(f"완료 {len(todo) - len(fails)}/{len(todo)} 문단 · 실패 {len(fails)} (다시 실행하면 실패한 문단만 이어서 함)")
    print("토큰 합계:", tot, "· 처리한 곳:", providers)


if __name__ == "__main__":
    main()
