"""재번역 결과를 6축 검증 입력(eval items)으로 만든다.

    python tools/gatsby_retrans_eval_items.py --cond reason

- 처음 번역의 검증 입력(ch1_3__gemma_bf16_ctx5.jsonl)에서 짚인 문장만 골라, 번역문(tgt)과 문단 번역(tgt_paragraph)을 재번역 결과로 바꾼다.
  같은 문단의 다른 짚인 문장도 같은 조건의 재번역으로 바뀐 상태가 문단 번역에 들어간다(실제로 고쳐진 문서와 같은 모습).
- 모델 답의 형식이 깨져(닫는 따옴표 누락) 저장되지 못한 문장은 raw_fail 기록의 greedy 출력에서 복구한다(복구 표시를 남김).
"""
import argparse
import json
import re
import sys
from pathlib import Path

D = Path("data/book/gatsby")


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def repair(raw):
    """{"id": "S…", "ko": "…"} 묶음을 닫는 따옴표가 빠져도 읽는다."""
    out = {}
    for m in re.finditer(r'\{"id":\s*"(S\d+)",\s*"ko":\s*"(.*?)"?\}(?=\s*(?:,\s*\{"id"|\]))', raw, re.S):
        out[m.group(1)] = m.group(2).replace('\\"', '"').strip()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--cond", required=True, choices=["reason", "plain"]); a = p.parse_args(argv)
    flags = {r["id"]: r for r in rows(D / "retrans/flags_ch1_3_ctx5.jsonl")}
    new = {r["id"]: r["mt"] for r in rows(D / f"retrans/retrans_{a.cond}_ch1_3.jsonl")}
    repaired = {}
    for r in rows(D / f"retrans/raw_fail_retrans_{a.cond}_ch1_3.jsonl"):
        if r.get("sampling"):
            continue                      # 복구는 greedy 출력만 쓴다(다른 문장과 같은 조건)
        for i, t in repair(r["raw"]).items():
            if i in flags and i not in new:
                repaired[i] = t
    new.update(repaired)
    missing = sorted(set(flags) - set(new))
    ev = rows(D / "eval/ch1_3__gemma_bf16_ctx5.jsonl")
    cur = {r["id"]: r["mt"] for r in rows(D / "mt/translate_input__gemma_bf16_ctx5_ch1_3.jsonl")} if (D / "mt/translate_input__gemma_bf16_ctx5_ch1_3.jsonl").exists() else {}
    by_para = {}
    for r in ev:
        by_para.setdefault(r["src_paragraph"], []).append(r)
    out = []
    for r in ev:
        sid = r["book_sentence_id"]
        if sid not in new:
            continue
        para = r["tgt_paragraph"]
        for q in by_para[r["src_paragraph"]]:          # 같은 문단의 짚인 문장을 모두 새 번역으로
            qs = q["book_sentence_id"]
            if qs in new and new[qs] != q["tgt"]:
                assert q["tgt"] in para, (sid, qs)
                para = para.replace(q["tgt"], new[qs], 1)
        out.append({**r, "tgt": new[sid], "tgt_paragraph": para, "mt_system": f"gemma-4-31b-it-bf16-retrans-{a.cond}",
                    "retrans": {"cond": a.cond, "prev_tgt": r["tgt"], "changed": new[sid] != r["tgt"], "format_repaired": sid in repaired}})
    dst = D / f"eval/ch1_3__gemma_bf16_ctx5_retrans_{a.cond}.jsonl"
    with open(dst, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{a.cond}: 문항 {len(out)} · 형식 복구 {sorted(repaired)} · 여전히 없음 {missing} · 바뀐 문장 {sum(r['retrans']['changed'] for r in out)} → {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
