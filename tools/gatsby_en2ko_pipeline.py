"""영→한 파이프라인(시험판)의 이음매: 새 번역 → 6축 검증 입력 → 짚인 문장 목록 → 재번역 결과 합치기.

    ① (GPU) 방침을 넣어 번역            translation/gatsby_policy_gpu.py      → policy_{cond}.jsonl
    ② (여기) 검증 입력 만들기            python tools/gatsby_en2ko_pipeline.py items --mt policy_policy.jsonl --name policy
       (여기) 검증                       python -m bridgecore.cli run --items-file <items> --version v1_1 ... --name <run>
    ③ (여기) 짚인 문장 목록              python tools/gatsby_en2ko_pipeline.py flags --name policy --run runs/<run>
    ④ (GPU) 짚인 문장만 재번역           translation/gatsby_retranslate_gpu.py --flags flags_pipe_policy.jsonl --cond reason --cur policy_policy.jsonl --policy --tag pipe_policy
    ⑤ (여기) 합치고 다시 검증할 입력     python tools/gatsby_en2ko_pipeline.py merge --mt policy_policy.jsonl --retrans retrans_reason_pipe_policy.jsonl --name policy

- 검증 입력은 지금까지와 같은 형식(문장 + 그 문단 + 앞 5·뒤 5문장)이다. 새 번역에 없는 문장의 문맥은 기존 번역(책 전체 번역)을 쓴다.
- 대상 문장은 11/28 엔진 시험의 603문장(1·3장 300 + 7장 303)이 기본이다. --all945 를 주면 새 번역에 있는 문장 전부.
"""
import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

D = Path("data/book/gatsby")
E = D / "eval"
FULL = D / "mt/translate_input__gemma_bf16_ctx5_full.jsonl"


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write(p, rs):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def overlay(new_mt, system):
    """책 전체 검증 입력에서 번역문과 문단 번역을 새 번역으로 바꾼다. 새 번역에 없는 문장은 그대로."""
    sents = rows(D / "sentences.jsonl")
    blk, sb = collections.defaultdict(list), {}
    for s in sents:
        if s["kind"] == "body" and str(s["translate"]) == "True":
            blk[s["block"]].append(s["id"]); sb[s["id"]] = s["block"]
    mt = {r["id"]: r["mt"] for r in rows(FULL)}
    mt.update(new_mt)
    base = rows(E / "ch1_3__gemma_bf16_ctx5.jsonl") + rows(E / "ch2_4to9__gemma_bf16_ctx5.jsonl")
    out = []
    for r in base:
        i = r["book_sentence_id"]
        if any(x in new_mt for x in blk[sb[i]]):          # 이 문단에 새 번역이 하나라도 있으면 문단 번역을 다시 만든다
            r = dict(r, tgt=mt[i], tgt_paragraph=" ".join(mt[x] for x in blk[sb[i]] if x in mt))
            if i in new_mt:
                r["mt_system"] = system
        out.append(r)
    return out


def target_ids(new_mt, all945):
    if all945:
        return set(new_mt)
    ids = {r["book_sentence_id"] for f in ("ch1_3__gemma_bf16_ctx5_pilot300.jsonl", "ch7__gemma_bf16_ctx5_sel303.jsonl") for r in rows(E / f)}
    miss = ids - set(new_mt)
    assert not miss, f"새 번역에 없는 대상 문장 {len(miss)}개: {sorted(miss)[:5]}"
    return ids


def cmd_items(a):
    new = rows(a.mt)
    new_mt = {r["id"]: r["mt"] for r in new}
    allr = overlay(new_mt, new[0].get("mt_system", a.name))
    ids = target_ids(new_mt, a.all945)
    sub = [r for r in allr if r["book_sentence_id"] in ids]
    p_all, p_sub = E / f"pipe_{a.name}__all.jsonl", E / f"pipe_{a.name}__target{len(sub)}.jsonl"
    out = E / f"pipe_{a.name}__target{len(sub)}_prev5next5.jsonl"
    write(p_all, allr); write(p_sub, sub)
    subprocess.run([sys.executable, "tools/gatsby_prev_context_items.py", "--items", str(p_sub), "--all", str(p_all),
                    "--n", "5", "--next", "5", "--out", str(out)], check=True)
    changed = sum(1 for r in sub if r["tgt"] != {x["id"]: x["mt"] for x in rows(FULL)}.get(r["book_sentence_id"]))
    print(f"검증 입력 {len(sub)}문장 → {out} (기존 번역과 달라진 문장 {changed})")


def cmd_flags(a):
    items = sorted(E.glob(f"pipe_{a.name}__target*_prev5next5.jsonl"))[-1]
    out = D / f"retrans/flags_pipe_{a.name}.jsonl"
    subprocess.run([sys.executable, "tools/gatsby_make_retrans_input.py", "--eval", str(items), "--run", *a.run, "--out", str(out)], check=True)
    fl = rows(out)
    print(f"짚인 문장 {len(fl)} → {out}", dict(collections.Counter(r.get("top_severity") for r in fl)))


def cmd_merge(a):
    new_mt = {r["id"]: r["mt"] for r in rows(a.mt)}
    re = {r["id"]: r for r in rows(a.retrans)}
    stale = [i for i, r in re.items() if r["mt_prev"] != new_mt.get(i)]
    assert not stale, f"재번역이 다른 번역을 바탕으로 한 것 같습니다: {stale[:3]}"
    final = dict(new_mt); final.update({i: r["mt"] for i, r in re.items()})
    write(D / f"mt/pipe_{a.name}__final.jsonl", [{"id": i, "mt": m, "retranslated": i in re} for i, m in final.items()])
    allr = overlay(final, f"pipe_{a.name}_final")
    sub = [r for r in allr if r["book_sentence_id"] in re]
    p_all, p_sub = E / f"pipe_{a.name}_final__all.jsonl", E / f"pipe_{a.name}_final__retrans{len(sub)}.jsonl"
    out = E / f"pipe_{a.name}_final__retrans{len(sub)}_prev5next5.jsonl"
    write(p_all, allr); write(p_sub, sub)
    subprocess.run([sys.executable, "tools/gatsby_prev_context_items.py", "--items", str(p_sub), "--all", str(p_all),
                    "--n", "5", "--next", "5", "--out", str(out)], check=True)
    print(f"재번역 {len(re)}문장(그대로 둔 것 {sum(1 for r in re.values() if not r['changed'])}) → 다시 검증할 입력 {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("items"); s.add_argument("--mt", required=True); s.add_argument("--name", required=True); s.add_argument("--all945", action="store_true")
    s = sub.add_parser("flags"); s.add_argument("--name", required=True); s.add_argument("--run", required=True, nargs="+")
    s = sub.add_parser("merge"); s.add_argument("--mt", required=True); s.add_argument("--retrans", required=True); s.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    {"items": cmd_items, "flags": cmd_flags, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
