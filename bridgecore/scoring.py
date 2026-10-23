"""채점: 엔진 출력 vs 임시 정답(다수결), 그리고 사람 간 기준선.

주 지표 (채점 규칙 v1):
  - 축별 '훼손 대 나머지' 정밀도·재현율·F1
  - 엔진 판단보류는 두 방식으로 함께 보고
      strict : 보류 = 비훼손 (보류로 도망가면 재현율이 깎임) ← 주 지표
      lenient: 보류 = 훼손 (보류를 '사람 검토 필요' 표시로 볼 때)
  - 축별 보류 건수·비율
  - 재현율 95% 윌슨 신뢰구간 + "n건 중 k건" (드문 축 보고용)
보조 지표:
  - 4범주 일치율 (정답 불일치 칸 제외)
  - 심각도 일치율 (정답 훼손 & 엔진 훼손인 칸)
형식 오류 문항: 6축 모두 비훼손 예측으로 처리하고 건수를 따로 보고한다.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations

from .constants import AXES, HOLD, SEVERITY_RANK, VIOLATED
from .schema import normalize, validate


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    # 정답 훼손이나 예측 훼손이 있는데 TP가 0이면 F1=0, 둘 다 없으면 정의 안 됨(None)
    f = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def predictions_from_outputs(outputs: dict[str, dict]) -> tuple[dict, dict]:
    """{문항ID: 엔진출력} → ({(문항ID, 축): {judgment, severity}}, {문항ID: 검증결과})."""
    preds, checks = {}, {}
    for item_id, out in outputs.items():
        out = normalize(out) if isinstance(out, dict) else out
        errs, warns = validate(out, item_id)
        checks[item_id] = {"errors": errs, "warnings": warns}
        for ax in AXES:
            a = ((out or {}).get("axes") or {}).get(ax) or {} if not errs else {}
            preds[(item_id, ax)] = {"judgment": a.get("judgment"), "severity": a.get("severity"),
                                    "format_error": bool(errs)}
    return preds, checks


def score(gold: dict, preds: dict, item_ids: list[str] | None = None) -> dict:
    """gold: build_gold() 결과, preds: predictions_from_outputs()[0]."""
    keys = [k for k in gold if item_ids is None or k[0] in set(item_ids)]
    per_axis = {}
    cases = []
    totals = defaultdict(Counter)
    for ax in AXES + ["ALL"]:
        ks = [k for k in keys if ax == "ALL" or k[1] == ax]
        c = Counter()
        sev_n = sev_exact = sev_within1 = 0
        l4_n = l4_hit = 0
        for k in ks:
            g = gold[k]
            p = preds.get(k) or {"judgment": None, "severity": None, "format_error": True}
            pj = p["judgment"]
            gv = g["violated"]
            is_hold = pj == HOLD
            strict = pj == VIOLATED
            lenient = pj in (VIOLATED, HOLD)
            for mode, pv in (("strict", strict), ("lenient", lenient)):
                c[f"{mode}_{'tp' if gv and pv else 'fp' if pv else 'fn' if gv else 'tn'}"] += 1
            c["hold"] += is_hold
            c["missing"] += pj is None
            if gv and pj == VIOLATED and g["severity"] and p.get("severity") in SEVERITY_RANK:
                sev_n += 1
                d = abs(SEVERITY_RANK[g["severity"]] - SEVERITY_RANK[p["severity"]])
                sev_exact += d == 0
                sev_within1 += d <= 1
            if g["label4"] is not None and pj is not None:
                l4_n += 1
                l4_hit += g["label4"] == pj
            if ax != "ALL" and (gv != strict or is_hold):
                cases.append({
                    "item_id": k[0], "axis": k[1],
                    "type": "보류" if is_hold else ("FN(놓침)" if gv else "FP(오탐)"),
                    "gold_votes": f"{g['n_violated']}/{g['n_raters']}",
                    "gold_label4": g["label4"] or "불일치", "pred": pj,
                    "gold_severity": g["severity"], "pred_severity": p.get("severity"),
                })
        n_pos = c["strict_tp"] + c["strict_fn"]
        per_axis[ax] = {
            "n": len(ks), "gold_violated": n_pos,
            "strict": prf(c["strict_tp"], c["strict_fp"], c["strict_fn"]),
            "lenient": prf(c["lenient_tp"], c["lenient_fp"], c["lenient_fn"]),
            "recall_ci95": wilson(c["strict_tp"], n_pos),
            "recall_count": f"{n_pos}건 중 {c['strict_tp']}건",
            "hold": c["hold"], "hold_rate": c["hold"] / len(ks) if ks else None,
            "missing_or_format_error": c["missing"],
            "label4_acc": l4_hit / l4_n if l4_n else None, "label4_n": l4_n,
            "severity_exact": sev_exact / sev_n if sev_n else None,
            "severity_within1": sev_within1 / sev_n if sev_n else None, "severity_n": sev_n,
        }
    return {"per_axis": per_axis, "cases": cases}


def human_baseline(ratings: list[dict], item_ids: list[str] | None = None) -> dict:
    """사람 간 기준선: 평가자 쌍마다 한쪽을 정답으로 본 '훼손' F1 (쌍에 대해 대칭)
    + 이진 Fleiss' kappa. 엔진 F1을 해석할 때 상한 참고치로 쓴다."""
    lab = defaultdict(dict)
    for r in ratings:
        if item_ids is None or r["item_id"] in set(item_ids):
            lab[(r["item_id"], r["axis"])][r["rater_id"]] = r["judgment"] == VIOLATED
    raters = sorted({r["rater_id"] for r in ratings})
    out = {}
    for ax in AXES + ["ALL"]:
        ks = [k for k in lab if ax == "ALL" or k[1] == ax]
        pair = {}
        for a, b in combinations(raters, 2):
            tp = sum(lab[k].get(a) and lab[k].get(b) for k in ks)
            fp = sum((not lab[k].get(a)) and lab[k].get(b) for k in ks)
            fn = sum(lab[k].get(a) and not lab[k].get(b) for k in ks)
            pair[f"{a}-{b}"] = prf(tp, fp, fn)["f1"]
        vals = [v for v in pair.values() if v is not None]
        out[ax] = {"pairwise_f1": pair,
                   "mean_pairwise_f1": sum(vals) / len(vals) if vals else None,
                   "fleiss_kappa": _fleiss_binary([lab[k] for k in ks], raters)}
    return out


def _fleiss_binary(cells: list[dict], raters: list[str]) -> float | None:
    n = len(raters)
    rows = [sum(c.get(r, False) for r in raters) for c in cells if len(c) == n]
    N = len(rows)
    if N == 0:
        return None
    p1 = sum(rows) / (N * n)
    pe = p1 ** 2 + (1 - p1) ** 2
    pbar = sum((k * (k - 1) + (n - k) * (n - k - 1)) / (n * (n - 1)) for k in rows) / N
    return (pbar - pe) / (1 - pe) if pe < 1 else None
