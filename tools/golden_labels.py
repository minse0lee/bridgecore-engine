"""골든셋 평가폼 v3(평가자 파일) 읽기·빈칸 점검·엔진 실행과 비교.

    python tools/golden_labels.py check  R003.xlsx                      # 빈칸 점검(300문항 전체, 개수만)
    python tools/golden_labels.py score  R003.xlsx runs/golden_dev100_v1_1_r1   # 개발 100문항만 엔진과 비교

- v3 평가 시트 열: 문항ID, 평가축, ①평가 대상?, ②훼손 기준?, 판정(자동), 심각도, 오류ID, 원문 근거구간, 번역문 근거구간,
  판정이유/보류사유, 최종판정, 최종 심각도, 변경사유, 지침·문맥 의견 (bridgecore.loader는 v1·v2 열 배치라 쓰지 않음)
- 판정은 엑셀 캐시값을 믿지 않고 ①②에서 다시 계산한다(평가폼 수식과 같은 규칙)
- 시험 200문항 라벨은 엔진 v2 고정 전까지 내용 분석 금지(사전 등록). check는 빈칸 개수만 세고, score는 개발 문항만 연다
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data/golden/split_final_newids_2026-11-13.csv"
AXES = ["Semantic", "Textual", "Context", "Deontic", "Register", "Cultural"]
COLS = ["item_id", "axis", "q1", "q2", "judgment_cached", "severity", "error_ids", "src_span", "tgt_span", "reason",
        "final_judgment", "final_severity", "change_reason", "opinion"]
FIRST = 9


def _c(v):
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def judge(q1, q2):
    """평가폼 E열 수식과 같은 규칙."""
    if not q1:
        return None
    if q1 == "아니오":
        return "N/A"
    if not q2:
        return None
    return {"예": "훼손", "아니오": "보존"}.get(q2, "판단보류")


def load_v3(xlsx) -> tuple[str, list[dict]]:
    ws = openpyxl.load_workbook(xlsx, data_only=True)["평가"]
    rater = _c(ws["B3"].value)
    rows = []
    for i, row in enumerate(ws.iter_rows(min_row=FIRST, max_col=len(COLS), values_only=True), FIRST):
        rec = {k: _c(v) for k, v in zip(COLS, row)}
        if not (rec["item_id"] or "").startswith("P"):
            continue
        rec["row"] = i
        rec["judgment"] = judge(rec["q1"], rec["q2"])
        rows.append(rec)
    return rater, rows


def missing(r) -> list[str]:
    """진행률 규칙(평가폼 7행): 보존·N/A는 판정만, 훼손은 심각도·오류ID·근거구간·판정이유, 판단보류는 보류사유."""
    j = r["judgment"]
    if j is None:
        return ["판정"]
    if j == "훼손":
        need = {"심각도": r["severity"], "오류ID": r["error_ids"], "근거구간": r["src_span"] or r["tgt_span"],
                "판정이유": r["reason"]}
        return [k for k, v in need.items() if not v]
    if j == "판단보류":
        return [] if r["reason"] else ["보류사유"]
    return []


def split_map():
    return {r["문항ID"]: r["분할"] for r in csv.DictReader(open(SPLIT, encoding="utf-8-sig"))}


def cmd_check(a):
    rater, rows = load_v3(a.xlsx)
    sp = split_map()
    out = {"rater": rater, "rows": len(rows), "items": len({r["item_id"] for r in rows})}
    for part in ("개발", "시험"):
        rs = [r for r in rows if sp.get(r["item_id"]) == part]
        miss = [(r["item_id"], r["axis"], m) for r in rs for m in [missing(r)] if m]
        out[part] = {"rows": len(rs), "incomplete_rows": len(miss),
                     "incomplete_items": len({m[0] for m in miss}),
                     "cached_vs_recomputed_diff": sum(1 for r in rs if (r["judgment_cached"] or None) != r["judgment"])}
        if part == "개발":   # 개발 문항만 어느 칸이 비었는지 보여 줌
            out[part]["incomplete_detail"] = [f"{i} {ax}: {', '.join(m)}" for i, ax, m in miss]
            out[part]["judgment_counts"] = dict(Counter(r["judgment"] for r in rs))
    print(json.dumps(out, ensure_ascii=False, indent=1))


def _engine(run_dir):
    eng = {}
    for l in open(Path(run_dir) / "outputs.jsonl", encoding="utf-8"):
        r = json.loads(l)
        o = r.get("output")
        if not o or r.get("errors"):
            continue
        if not o.get("evaluable", True):
            eng[r["item_id"]] = {ax: {"judgment": "평가불가"} for ax in AXES}
            continue
        eng[r["item_id"]] = o["axes"]
    return eng


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return p, r, f


def cmd_score(a):
    rater, rows = load_v3(a.xlsx)
    sp = split_map()
    eng = _engine(a.run_dir)
    dev = [r for r in rows if sp.get(r["item_id"]) == "개발"]
    skipped_items = sorted({r["item_id"] for r in dev if r["item_id"] not in eng})
    per = defaultdict(lambda: Counter())
    pairs = []
    for r in dev:
        e = eng.get(r["item_id"], {}).get(r["axis"])
        if e is None or r["judgment"] in (None, "판단보류"):
            continue
        ej, hj = e.get("judgment"), r["judgment"]
        c = per[r["axis"]]
        c["n"] += 1
        c["same3"] += ej == hj
        c["tp"] += ej == "훼손" and hj == "훼손"
        c["fp"] += ej == "훼손" and hj != "훼손"
        c["fn"] += ej != "훼손" and hj == "훼손"
        c["na_disagree"] += (ej == "N/A") != (hj == "N/A")
        if ej == "훼손" and hj == "훼손":
            c["sev_same"] += (e.get("severity") or "").lower() == (r["severity"] or "").lower()
        pairs.append({"item_id": r["item_id"], "axis": r["axis"], "human": hj, "human_sev": r["severity"],
                      "engine": ej, "engine_sev": e.get("severity"), "human_reason": r["reason"],
                      "engine_reason": e.get("reason")})
    tot = Counter()
    lines = [f"평가자 {rater} · 개발 {len({r['item_id'] for r in dev})}문항 · 엔진 결과 없음 {skipped_items}",
             "축 | n | 3분류 일치 | 훼손 정밀도 | 재현율 | F1 | 사람 훼손 | 엔진 훼손 | N/A 불일치 | 둘 다 훼손 중 심각도 일치"]
    for ax in AXES + ["전체"]:
        c = per[ax] if ax != "전체" else tot
        if ax != "전체":
            tot.update(per[ax])
        p, rcl, f = prf(c["tp"], c["fp"], c["fn"])
        fmt = lambda x: "-" if x is None else f"{x:.2f}"
        lines.append(f"{ax} | {c['n']} | {c['same3'] / c['n']:.2f} | {fmt(p)} | {fmt(rcl)} | {fmt(f)} | "
                     f"{c['tp'] + c['fn']} | {c['tp'] + c['fp']} | {c['na_disagree']} | {c['sev_same']}/{c['tp']}")
    print("\n".join(lines))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
        print("→", a.out)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check"); s.add_argument("xlsx"); s.set_defaults(fn=cmd_check)
    s = sub.add_parser("score"); s.add_argument("xlsx"); s.add_argument("run_dir"); s.add_argument("--out")
    s.set_defaults(fn=cmd_score)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
