"""엔진이 훼손으로 짚은 문장을 재번역 입력(flags jsonl)으로 뽑는다.

    python tools/gatsby_make_retrans_input.py --eval data/book/gatsby/eval/ch1_3__gemma_bf16_ctx5.jsonl \
        --run runs/gatsby_ch13_gemma_bf16_ctx5 --out data/book/gatsby/retrans/flags_ch1_3_ctx5.jsonl

한 줄 = 문장 하나: {id(책 문장 id), item_id, en, mt, top_severity, flags:[{axis, severity, reason}], errors:[{src_span, tgt_span, description}]}
- 엔진이 낸 것 중 '무엇이 왜 문제인지'만 담는다. 엔진은 고친 번역을 내지 않으므로 수정안은 들어가지 않는다.
- 엔진 오류로 판정이 없는 문장은 넣지 않는다(건수만 알림).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from bridgecore.constants import AXES, VIOLATED  # noqa: E402
from bridgecore.vote import load_group  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True); p.add_argument("--run", required=True, nargs="+"); p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rows = [json.loads(l) for l in open(a.eval, encoding="utf-8") if l.strip()]
    G = load_group([Path(r) for r in a.run])
    out, noj = [], 0
    for r in rows:
        g = G.get(r["item_id"])
        if not g or not g.get("output") or g.get("errors"):
            noj += 1
            continue
        o = g["output"]
        bad = [x for x in AXES if o["axes"].get(x, {}).get("judgment") == VIOLATED]
        if not bad:
            continue
        sev = [o["axes"][x].get("severity") for x in bad]
        used = {e for x in bad for e in o["axes"][x].get("error_ids") or []}
        out.append({"id": r["book_sentence_id"], "item_id": r["item_id"], "en": r["src"], "mt": r["tgt"],
                    "top_severity": "critical" if "critical" in sev else "major" if "major" in sev else "minor",
                    "flags": [{"axis": x, "severity": o["axes"][x].get("severity"), "reason": o["axes"][x].get("reason")} for x in bad],
                    "errors": [{k: e.get(k) for k in ("src_span", "tgt_span", "description")} for e in o.get("errors") or [] if e.get("id") in used]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = {s: sum(r["top_severity"] == s for r in out) for s in ("minor", "major", "critical")}
    print(f"문장 {len(rows)} · 판정 없음 {noj} · 훼손으로 짚인 문장 {len(out)} {n} → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
