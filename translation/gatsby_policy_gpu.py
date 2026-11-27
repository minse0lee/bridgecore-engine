"""개츠비 번역 쪽 시험 — 번역 지침에 '용어집·호칭·말투 방침'을 더하면 번역이 얼마나 일관되게 바뀌는지 본다.

같은 폴더에 gatsby_full_gpu.py, gemma_translate.py, sentences.jsonl 을 둔다(책 전체 번역 때 쓴 것 그대로).

    python gatsby_policy_gpu.py --cond policy --show 1     # 프롬프트만 확인(모델을 올리지 않음)
    python gatsby_policy_gpu.py --cond policy              # ① 방침만 더함
    python gatsby_policy_gpu.py --cond policy_next5        # ② 방침 + 뒤 문단 5개 원문도 문맥으로 줌
    python gatsby_policy_gpu.py --cond base                # (선택) 방침 없이 처음 번역과 같은 조건으로 다시 — 같은 기계에서의 비교 기준

GPU 4장이면 두 조건을 동시에(터미널 두 개 또는 nohup):
    CUDA_VISIBLE_DEVICES=0,1 python gatsby_policy_gpu.py --cond policy
    CUDA_VISIBLE_DEVICES=2,3 python gatsby_policy_gpu.py --cond policy_next5

- 대상: 1·3장 전체 + 7장 가운데 S001940~S002040, S002100~S002130, S002269~S002441 문장이 든 문단(11/28 엔진 시험과 같은 문장).
- 번역 단위(문단)·앞 문단 5개 문맥·생각 모드 끔·greedy·기본 지침은 처음 번역 때와 같다. 달라지는 것은 지침 끝에 붙는 방침(POLICY)과,
  policy_next5 조건에서 뒤 문단 5개 원문이 문맥으로 더해진다는 점뿐이다.
- 결과: out/policy_{cond}.jsonl — {id, en, mt, cond, mt_system, mt_meta}. 끊기면 같은 명령으로 이어서 한다.
- 방침은 시험용 임시안이다(reports/gatsby_translation_policy_draft_2026-11-28.md). 출판용 원고의 최종 방침이 아니다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

try:
    import gatsby_full_gpu as G
except ImportError:      # 저장소 안에서 테스트할 때
    from translation import gatsby_full_gpu as G

POLICY = """
이 작품에 한한 추가 방침 (원칙 4·6·7을 보충합니다)
9. 고유명사 표기(추가): Michaelis 미카엘리스, Meyer Wolfshiem 마이어 울프심, Klipspringer 클립스프링거, Owl Eyes·the owl-eyed man 부엉이 눈, Camp Taylor 캠프 테일러, Catherine 캐서린, McKee 맥키, Dan Cody 댄 코디, Gatz 개츠, Sloane 슬론, Doctor T. J. Eckleburg T. J. 에클버그 박사, valley of ashes 재의 계곡, Sigourney Howard 시고니 하워드. 닉의 가정부(my Finn)는 "핀란드인 가정부"로 옮깁니다.
10. 호칭: 원문에 Miss·Mr.·Mrs.가 없으면 "양"·"씨"·"부인"을 붙이지 않습니다. 이름만 있으면 존댓말로 말할 때도 이름만 씁니다(예: "톰, 위스키 좀 따줘요"). 데이지는 결혼한 사람이므로 "데이지 양"이라고 하지 않습니다.
11. 관계어: 닉과 데이지의 관계에서 second cousin once removed는 "먼 친척", cousin은 "친척"으로 옮깁니다. 캐서린은 머틀의 "동생"입니다.
12. 말버릇: 개츠비의 old sport는 언제나 "친구"로 옮깁니다. 다른 인물이 그 말을 흉내 내거나 따질 때도 "친구"를 씁니다.
13. 말투: 인물 사이의 말투를 아래대로 지키고 한 장면 안에서 바꾸지 않습니다. "자네", "이보게", "~하게", "~하나?", "~하네" 같은 하게체는 쓰지 않습니다.
 - 개츠비: 누구에게나 격식 있는 존댓말(-습니다, -요). 데이지에게는 -요.
 - 닉: 톰에게는 반말, 그 밖의 사람에게는 존댓말(-요).
 - 톰: 닉·데이지·윌슨·머틀·하인에게 반말. 개츠비와 처음 보는 사람에게는 퉁명스러운 존댓말(-요).
 - 데이지: 닉·개츠비·톰에게 -요, 조던에게 반말.
 - 조던: 닉·톰에게 -요, 데이지에게 반말.
 - 윌슨: 톰에게 존댓말. 미카엘리스: 윌슨에게 존댓말(-요).
 - 머틀: 톰·캐서린·윌슨에게 반말, 그 밖의 사람에게는 -요.
 - 집사·하인: 주인과 손님에게 -습니다.
 - 여기에 없는 관계는 원칙 4를 따릅니다.
"""

CONDS = {                      # 조건 → (방침을 붙이는가, 뒤 문단 수)
    "base": (False, 0),
    "policy": (True, 0),
    "policy_next5": (True, 5),
}
CH7_RANGES = [(1940, 2040), (2100, 2130), (2269, 2441)]


def system_text(gt, cond):
    return gt.SYSTEM + (POLICY if CONDS[cond][0] else "") + gt.FORMAT


def prompt_hash(gt, cond):
    return hashlib.sha256(system_text(gt, cond).encode()).hexdigest()[:12]


def user_text(gt, allb, k, n_ctx, n_next):
    """n_next=0이면 gemma_translate.user_text와 글자까지 같다. n_next>0이면 뒤 문단 원문을 문맥으로 더한다."""
    sents, text = gt.user_text(allb, k, n_ctx=n_ctx)
    if n_next:
        nxt = [x["para"] for x in allb[k + 1:k + 1 + n_next]]
        head, tail = text.split("\n\n[번역할 문장 — id마다 하나씩]\n")
        text = (head + "\n\n[뒤 문단 원문 — 문맥 참고용, 번역하지 마세요]\n" + ("\n\n".join(nxt) if nxt else "(없음)") +
                "\n\n[번역할 문장 — id마다 하나씩]\n" + tail)
    return sents, text


def targets(allb, idx):
    """1·3장 전체 + 7장에서 정해 둔 문장이 든 문단."""
    def in_ch7(s):
        n = int(s["id"][1:])
        return any(a <= n <= b for a, b in CH7_RANGES)
    out = []
    for k in idx:
        b = allb[k]
        if b["chapter"] in ("I", "III") or (b["chapter"] == "VII" and any(s["translate"] and in_ch7(s) for s in b["sents"])):
            out.append(k)
    return out


def gen(m, gt, packs, cond, sample=False):
    """packs: [(sents, user_text)] → gatsby_full_gpu.Gemma.generate 와 같은 방식으로 생성·파싱."""
    torch = m.torch
    with torch.inference_mode():
        texts = [m.processor.apply_chat_template(
            [{"role": "system", "content": system_text(gt, cond)}, {"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True, enable_thinking=G.CFG["thinking"]) for _, u in packs]
        enc = m.processor(text=texts, return_tensors="pt", padding=True).to(m.model.device)
        n_in = enc["input_ids"].shape[1]
        need = max(len(p[0]) for p in packs) * 200 + 300
        out = m.model.generate(**enc, max_new_tokens=need, **(G.SAMPLE if sample else dict(do_sample=False)))
        res = []
        for (sents, _), ids in zip(packs, out[:, n_in:]):
            raw, content, think = m.split_answer(ids)
            res.append(G._check(gt, sents, content, raw, len(think), int((ids != m.processor.tokenizer.pad_token_id).sum())))
        return res


def gen_fake(m, gt, packs, cond, sample=False):
    res = []
    for sents, _ in packs:
        content = json.dumps({"translations": [{"id": s["id"], "ko": f"번역({cond}):" + s["en"]} for s in sents]}, ensure_ascii=False)
        res.append(G._check(gt, sents, content, content, 0, 10))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(Path(__file__).parent))
    ap.add_argument("--cond", required=True, choices=sorted(CONDS))
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="이 수만큼의 문단만(시험)")
    ap.add_argument("--show", type=int, default=0, help="프롬프트를 이 수만큼 출력하고 끝냄(모델을 올리지 않음)")
    ap.add_argument("--fake", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    base = Path(a.dir)
    sys.path.insert(0, str(base))
    import gemma_translate as gt
    assert gt.PROMPT_HASH == G.PROMPT_HASH, "gemma_translate.py가 처음 번역 때와 다릅니다"
    assert prompt_hash(gt, "base") == gt.PROMPT_HASH
    with_policy, n_next = CONDS[a.cond]
    n_ctx = G.CFG["n_ctx"]
    out_dir = base / "out"; out_dir.mkdir(exist_ok=True)
    rows = G.read_jsonl(base / "sentences.jsonl")
    allb, idx = gt.blocks(rows, None)
    tk = targets(allb, idx)
    want = [s["id"] for k in tk for s in allb[k]["sents"] if s["translate"]]

    tag = f"policy_{a.cond}"
    out_path = out_dir / f"{tag}.jsonl"
    done = {r["id"] for r in G.read_jsonl(out_path)} if out_path.exists() else set()
    ks = [k for k in tk if any(s["translate"] and s["id"] not in done for s in allb[k]["sents"])]
    packs = {k: user_text(gt, allb, k, n_ctx, n_next) for k in ks}
    order = sorted(ks, key=lambda k: len(packs[k][1]))
    if a.limit:
        order = order[::max(1, len(order) // a.limit)][:a.limit]
    print(f"[{tag}] 대상 문단 {len(tk)}(문장 {len(want)}) · 이미 한 문장 {len(done & set(want))} · 이번에 할 문단 {len(order)} · "
          f"방침 {'있음' if with_policy else '없음'} · 뒤 문단 {n_next}개 · 지침 지문 {prompt_hash(gt, a.cond)}")
    if a.show:
        print("=" * 30, "system에 더해지는 방침"); print(POLICY if with_policy else "(없음)")
        for k in order[len(order) // 2:len(order) // 2 + a.show]:
            print("=" * 30, "user"); print(packs[k][1])
        return 0
    if not order:
        print("더 번역할 것이 없습니다 →", out_path)
        return 0

    m = (G.Fake if a.fake else G.Gemma)(gt, allb, a.bs)
    run_gen = gen_fake if a.fake else gen
    meta = {**m.meta, "config": tag, "prompt_hash": prompt_hash(gt, a.cond), "base_prompt_hash": gt.PROMPT_HASH,
            "policy": with_policy, "n_ctx": n_ctx, "n_next": n_next}
    raw_log = open(out_dir / f"raw_fail_{tag}.jsonl", "a", encoding="utf-8")
    n_sent = 0

    def save(r, sampled=False):
        nonlocal n_sent
        with open(out_path, "a", encoding="utf-8") as f:
            for s in r["sents"]:
                if s["id"] in done:
                    continue
                f.write(json.dumps({"id": s["id"], "en": s["en"], "mt": r["got"][s["id"]], "cond": a.cond,
                                    "mt_system": f"gemma-4-31b-it-bf16-{tag}",
                                    "mt_meta": {**meta, "decoding": "sampling_retry" if sampled else "greedy"}}, ensure_ascii=False) + "\n")
                done.add(s["id"]); n_sent += 1

    t0, bs, i, fails, nb = time.time(), m.bs, 0, [], 0
    while i < len(order):
        chunk = order[i:i + bs]
        try:
            res = run_gen(m, gt, [packs[k] for k in chunk], a.cond)
        except Exception as e:
            if not m.is_oom(e) or bs == 1:
                raise
            m.free(); bs = max(1, bs // 2); print("  메모리 부족 → 묶음", bs); continue
        for k, r in zip(chunk, res):
            if r["err"]:
                fails.append(k); raw_log.write(json.dumps({"k": k, "err": r["err"], "raw": r["raw"]}, ensure_ascii=False) + "\n")
            else:
                save(r)
        i += len(chunk); nb += 1
        if nb % 5 == 0 or i >= len(order):
            el = time.time() - t0
            print(f"  {i}/{len(order)} 문단 · {el / 60:.1f}분 · 남은 예상 {el / i * (len(order) - i) / 60:.0f}분 · 실패 {len(fails)}", flush=True)
    left = []
    for k in fails:                      # greedy로 형식이 깨진 문단: 권장 샘플링으로 한 번 더
        r = run_gen(m, gt, [packs[k]], a.cond, sample=True)[0]
        if r["err"]:
            left.append(k); raw_log.write(json.dumps({"k": k, "err": r["err"], "raw": r["raw"], "sampling": True}, ensure_ascii=False) + "\n")
        else:
            save(r, sampled=True)
    raw_log.close()
    missing = [i for i in want if i not in done]
    info = {**meta, "tag": tag, "gen_sec": round(time.time() - t0), "n_sentences_this_run": n_sent, "n_want": len(want),
            "n_missing": len(missing), "failed_paragraphs": left, "load_sec": round(m.load_sec), "python": platform.python_version()}
    with open(out_dir / f"run_info_{tag}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False) + "\n")
    print(f"[{tag}] 끝 {info['gen_sec'] / 60:.1f}분 · 이번에 {n_sent}문장 · 아직 없는 문장 {len(missing)} · 끝내 실패 {len(left)}문단 → {out_path}")
    if missing:
        print("  빠진 문장이 있습니다 — 같은 명령으로 다시 실행하면 이어서 합니다. 예:", missing[:5])
    return 0


if __name__ == "__main__":
    sys.exit(main())
