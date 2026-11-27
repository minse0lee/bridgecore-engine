"""검증 입력의 문맥을 '문장이 속한 문단'에서 '앞 문장 N개 + 문단'으로 넓힌 문항을 만든다(프롬프트는 그대로).

    python tools/gatsby_prev_context_items.py --items data/book/gatsby/eval/ch1_3__gemma_bf16_ctx5_flagged150.jsonl \
        --all data/book/gatsby/eval/ch1_3__gemma_bf16_ctx5.jsonl --n 5 --out data/book/gatsby/eval/ch1_3__gemma_bf16_ctx5_flagged150_prev5.jsonl

- 엔진 프롬프트의 [원문 문단]·[문단 전체 번역] 칸에 앞 문장을 덧붙인다. 같은 장(章) 안에서만, 이미 그 문단에 들어 있는 문장은 빼고,
  평가 대상 문장 바로 앞 N개(책 문장 순서)를 넣는다. 번역 쪽에는 그 문장들의 번역을 같은 순서로 넣는다.
- 번역 모델이 번역할 때 본 문맥(앞 5문장)과 같은 범위를 엔진도 보게 하려는 것.
"""
import argparse
import json
import sys


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def chapter(r):
    for l in r["doc_info"].split("\n"):
        if l.startswith("장·절"):
            return l.split(":", 1)[1].strip()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--items", required=True); p.add_argument("--all", required=True, nargs="+")
    p.add_argument("--n", type=int, default=5); p.add_argument("--next", type=int, default=0); p.add_argument("--out", required=True)
    p.add_argument("--profile", help="작품 프로필 파일. 주면 [문서정보] 칸 끝에 덧붙인다(프롬프트는 그대로).")
    a = p.parse_args(argv)
    allr = [r for f in a.all for r in rows(f)]
    allr.sort(key=lambda r: r["book_sentence_id"])
    pos = {r["book_sentence_id"]: i for i, r in enumerate(allr)}
    out, nprev = [], []
    for r in rows(a.items):
        i = pos[r["book_sentence_id"]]
        prev = []
        j = i - 1
        while j >= 0 and len(prev) < a.n:
            q = allr[j]
            if chapter(q) != chapter(r):
                break
            if q["src_paragraph"] != r["src_paragraph"]:      # 같은 문단 문장은 이미 문단 칸에 있다
                prev.append(q)
            j -= 1
        prev.reverse()
        nprev.append(len(prev))
        nxt = []
        j = i + 1
        while j < len(allr) and len(nxt) < a.next:
            q = allr[j]
            if chapter(q) != chapter(r):
                break
            if q["src_paragraph"] != r["src_paragraph"]:
                nxt.append(q)
            j += 1
        sp, tp = r["src_paragraph"], r["tgt_paragraph"]
        if prev or nxt:
            sp = "(평가 대상 문장이 속한 문단)\n" + sp
            tp = "(평가 대상 문장이 속한 문단의 번역)\n" + tp
        if prev:
            sp = "(앞 문장)\n" + "\n".join(q["src"] for q in prev) + "\n\n" + sp
            tp = "(앞 문장의 번역)\n" + "\n".join(q["tgt"] for q in prev) + "\n\n" + tp
        if nxt:
            sp = sp + "\n\n(뒤 문장)\n" + "\n".join(q["src"] for q in nxt)
            tp = tp + "\n\n(뒤 문장의 번역)\n" + "\n".join(q["tgt"] for q in nxt)
        out.append({**r, "src_paragraph": sp, "tgt_paragraph": tp,
                    "context_variant": {"prev_n": a.n, "next_n": a.next, "prev_ids": [q["book_sentence_id"] for q in prev],
                                        "next_ids": [q["book_sentence_id"] for q in nxt]}})
    if a.profile:
        prof = open(a.profile, encoding="utf-8").read().strip()
        for r in out:
            r["doc_info"] = r["doc_info"].rstrip() + "\n\n" + prof
            r["context_variant"]["profile"] = a.profile
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"문항 {len(out)} · 앞 문장 수 분포 {sorted(set(nprev))} (평균 {sum(nprev)/len(nprev):.1f}) → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
