"""개츠비 재번역 시험 — 검증엔진이 짚은 문장만 Gemma 4 31B(학습 없음, bf16)로 다시 번역한다.

같은 폴더에 gatsby_full_gpu.py, gemma_translate.py, sentences.jsonl, 현재 번역 jsonl, 짚인 문장 목록(flags jsonl)을 둔다.

    python gatsby_retranslate_gpu.py --flags flags_ch1_3_ctx5.jsonl --cond reason   # 엔진이 적은 문제를 알려 주고 재번역
    python gatsby_retranslate_gpu.py --flags flags_ch1_3_ctx5.jsonl --cond plain    # 문제는 알려 주지 않고 '다시 확인하라'고만 함(비교 기준)

GPU 4장이면 두 조건을 동시에(터미널 두 개 또는 nohup):
    CUDA_VISIBLE_DEVICES=0,1 python gatsby_retranslate_gpu.py --flags ... --cond reason
    CUDA_VISIBLE_DEVICES=2,3 python gatsby_retranslate_gpu.py --flags ... --cond plain

- 번역 단위·문맥(앞 문단 5개)·번역 지침·생각 모드 끔·greedy는 처음 번역 때와 같다. 달라지는 것은 '현재 번역'과 (reason 조건에서) 검토 의견이 함께 주어진다는 점뿐.
- 짚이지 않은 문장은 건드리지 않는다. 모델은 짚인 문장의 번역만 다시 낸다(그대로 두는 것도 허용).
- 결과: out/retrans_{cond}.jsonl — {id, en, mt_prev, mt, changed, cond, mt_system, mt_meta}. 끊기면 같은 명령으로 이어서 한다.
- 엔진은 고친 번역을 내지 않으므로 검토 의견에는 '무엇이 왜 문제인지'만 들어 있다.
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

AXIS_KO = {"Semantic": "의미(사실·수치·대상)", "Textual": "문장 구조·연결", "Context": "상황·목적", "Deontic": "의무·허가·금지의 강도",
           "Register": "말투·격식", "Cultural": "문화·제도·고유 표현"}

REVISE = {
    "reason": """
재번역 작업
아래에는 이미 번역된 문단과, 그중 다시 번역할 문장이 주어집니다. 다시 번역할 문장에는 검토자가 적은 검토 의견이 붙어 있습니다.
- 검토 의견이 지적한 문제를 고쳐 그 문장의 번역을 다시 씁니다.
- 지적되지 않은 부분은 가능한 한 그대로 둡니다. 문제없는 표현을 바꾸지 않습니다.
- 검토 의견이 원문에 비추어 틀렸다고 판단되면 현재 번역을 그대로 냅니다.
- 다시 번역할 문장만 출력합니다. 다른 문장은 출력하지 않습니다.
""",
    "plain": """
재번역 작업
아래에는 이미 번역된 문단과, 그중 다시 번역할 문장이 주어집니다.
- 다시 번역할 문장의 현재 번역을 원문과 다시 대조해, 문제가 있으면 고쳐 다시 씁니다.
- 문제가 없는 부분은 그대로 둡니다. 문제가 없으면 현재 번역을 그대로 냅니다.
- 다시 번역할 문장만 출력합니다. 다른 문장은 출력하지 않습니다.
""",
}


def prompt_hash(gt, cond):
    return hashlib.sha256((gt.SYSTEM + REVISE[cond] + gt.FORMAT).encode()).hexdigest()[:12]


def opinion(fl):
    """검토 의견 글: 오류 위치와 설명, 축별 판정 이유."""
    lines = []
    for e in fl.get("errors") or []:
        lines.append(f'  - 원문 "{e.get("src_span")}" → 번역 "{e.get("tgt_span")}": {e.get("description")}')
    for f in fl["flags"]:
        lines.append(f'  - [{AXIS_KO.get(f["axis"], f["axis"])} · {f.get("severity")}] {f.get("reason")}')
    return "\n".join(lines)


def user_text(gt, allb, k, cur, flags, cond, n_ctx):
    """cur: 문장 id → 현재 번역, flags: 문장 id → 짚인 내용. 돌려주는 sents는 다시 번역할 문장만."""
    b = allb[k]
    ctx = [x["para"] for x in allb[max(0, k - n_ctx):k]]
    para = [s for s in b["sents"] if s["translate"]]
    sents = [s for s in para if s["id"] in flags]
    text = ("[앞 문단 원문 — 문맥 참고용, 번역하지 마세요]\n" + ("\n\n".join(ctx) if ctx else "(없음)") +
            "\n\n[이번 문단 원문 전체]\n" + b["para"] +
            "\n\n[이번 문단의 현재 번역 — 참고용]\n" + "\n".join(f'{s["id"]}\t{cur.get(s["id"], "")}' for s in para) +
            "\n\n[다시 번역할 문장 — id마다 하나씩]")
    for s in sents:
        text += f'\n{s["id"]}\n  원문: {s["en"]}\n  현재 번역: {cur[s["id"]]}'
        if cond == "reason":
            text += "\n  검토 의견:\n" + opinion(flags[s["id"]])
    return sents, text


def gen(m, gt, packs, cond, sample=False):
    """packs: [(sents, user_text)] → gatsby_full_gpu.Gemma.generate 와 같은 방식으로 생성·파싱."""
    torch = m.torch
    with torch.inference_mode():
        texts = [m.processor.apply_chat_template(
            [{"role": "system", "content": gt.SYSTEM + REVISE[cond] + gt.FORMAT}, {"role": "user", "content": u}],
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
        content = json.dumps({"translations": [{"id": s["id"], "ko": f"재번역({cond}):" + s["en"]} for s in sents]}, ensure_ascii=False)
        res.append(G._check(gt, sents, content, content, 0, 10))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(Path(__file__).parent))
    ap.add_argument("--flags", required=True, help="짚인 문장 목록(jsonl)")
    ap.add_argument("--cond", required=True, choices=sorted(REVISE))
    ap.add_argument("--cur", default=None, help="현재 번역 jsonl(기본: out/gemma_full_bf16_ctx5.jsonl 이 있으면 그것, 없으면 gemma_ch1_3_bf16_ctx5.jsonl)")
    ap.add_argument("--tag", default="", help="결과 파일 이름 뒤에 붙일 말(예: full)")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="이 수만큼의 문단만(시험)")
    ap.add_argument("--show", type=int, default=0, help="프롬프트를 이 수만큼 출력하고 끝냄(모델을 올리지 않음)")
    ap.add_argument("--fake", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    base = Path(a.dir)
    sys.path.insert(0, str(base))
    import gemma_translate as gt
    assert gt.PROMPT_HASH == G.PROMPT_HASH, "gemma_translate.py가 처음 번역 때와 다릅니다"
    out_dir = base / "out"; out_dir.mkdir(exist_ok=True)
    rows = G.read_jsonl(base / "sentences.jsonl")
    allb, idx = gt.blocks(rows, None)
    cur_path = Path(a.cur) if a.cur else next(p for p in (out_dir / G.FINAL, base / "gemma_ch1_3_bf16_ctx5.jsonl") if p.exists())
    cur = {r["id"]: r["mt"] for r in G.read_jsonl(cur_path)}
    flags = {r["id"]: r for r in G.read_jsonl(base / a.flags if not Path(a.flags).is_absolute() else a.flags)}
    stale = [i for i, f in flags.items() if cur.get(i) != f["mt"]]
    assert not stale, f"짚인 문장 목록의 번역이 현재 번역 파일({cur_path.name})과 다릅니다 — 다른 번역을 검증한 결과일 수 있습니다: {stale[:3]}"

    tag = f"retrans_{a.cond}" + (f"_{a.tag}" if a.tag else "")
    out_path = out_dir / f"{tag}.jsonl"
    done = {r["id"] for r in G.read_jsonl(out_path)} if out_path.exists() else set()
    ks = [k for k in idx if any(s["translate"] and s["id"] in flags and s["id"] not in done for s in allb[k]["sents"])]
    todo_flags = {i: f for i, f in flags.items() if i not in done}
    packs = {k: user_text(gt, allb, k, cur, todo_flags, a.cond, G.CFG["n_ctx"]) for k in ks}
    order = sorted(ks, key=lambda k: len(packs[k][1]))
    if a.limit:
        order = order[::max(1, len(order) // a.limit)][:a.limit]
    print(f"[{tag}] 현재 번역 {cur_path.name} · 짚인 문장 {len(flags)} · 이미 한 문장 {len(done)} · 이번에 할 문단 {len(order)}(문장 {sum(len(packs[k][0]) for k in order)})")
    if a.show:
        for k in order[len(order) // 2:len(order) // 2 + a.show]:
            print("=" * 30, "system(재번역 부분)"); print(REVISE[a.cond]); print("=" * 30, "user"); print(packs[k][1])
        return 0
    if not order:
        return 0

    m = (G.Fake if a.fake else G.Gemma)(gt, allb, a.bs)
    run_gen = gen_fake if a.fake else gen
    meta = {**m.meta, "config": "retrans_ctx5", "retrans_cond": a.cond, "retrans_prompt_hash": prompt_hash(gt, a.cond),
            "flags_file": Path(a.flags).name, "cur_file": cur_path.name}
    raw_log = open(out_dir / f"raw_fail_{tag}.jsonl", "a", encoding="utf-8")
    n_same = n_sent = 0

    def save(r, sampled=False):
        nonlocal n_same, n_sent
        with open(out_path, "a", encoding="utf-8") as f:
            for s in r["sents"]:
                new = r["got"][s["id"]]
                f.write(json.dumps({"id": s["id"], "en": s["en"], "mt_prev": cur[s["id"]], "mt": new, "changed": new != cur[s["id"]],
                                    "cond": a.cond, "top_severity": flags[s["id"]].get("top_severity"),
                                    "mt_system": f"gemma-4-31b-it-bf16-{tag}",
                                    "mt_meta": {**meta, "decoding": "sampling_retry" if sampled else "greedy"}}, ensure_ascii=False) + "\n")
                n_sent += 1; n_same += new == cur[s["id"]]

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
    info = {**meta, "tag": tag, "gen_sec": round(time.time() - t0), "n_sentences_this_run": n_sent, "n_unchanged": n_same,
            "failed_paragraphs": left, "load_sec": round(m.load_sec), "python": platform.python_version()}
    with open(out_dir / f"run_info_{tag}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False) + "\n")
    print(f"[{tag}] 끝 {info['gen_sec'] / 60:.1f}분 · 이번에 {n_sent}문장 · 그대로 둔 문장 {n_same} · 끝내 실패 {len(left)}문단 → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
