"""사람 판정 집계 → 임시 정답(다수결).

채점 규칙 v1 (2026-10-23 확정 제안, PROJECT_NOTES 3-5):
  1. 이진 정답: 3명 중 2명 이상이 '훼손'이면 훼손, 아니면 비훼손.
     평가자가 3명이면 이진 다수결은 항상 결정된다 (동률 없음).
     사람의 '판단보류'는 비훼손 쪽으로 센다.
  2. 심각도 정답: 이진 정답이 훼손인 칸만. 훼손이라고 본 평가자들의 심각도 중
     2명 이상 같으면 그 값, 아니면 낮은 쪽 중앙값(정렬 후 (n-1)//2 번째).
     예) minor·major → minor, minor·major·critical → major.
     심각도를 비운 훼손 행은 심각도 집계에서만 뺀다.
  3. 4범주 정답(보조 지표): 2명 이상 같은 판정이면 그 값, 3명 모두 다르면 None(불일치)
     → 4범주 일치율의 분모에서 제외.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .constants import AXES, SEVERITY_RANK, SEVERITIES, VIOLATED


def _severity_vote(sevs: list[str]) -> str | None:
    sevs = [s for s in sevs if s in SEVERITY_RANK]
    if not sevs:
        return None
    top, n = Counter(sevs).most_common(1)[0]
    if n >= 2:
        return top
    ranked = sorted(sevs, key=SEVERITY_RANK.get)
    return ranked[(len(ranked) - 1) // 2]


def build_gold(ratings: list[dict], min_votes: int = 2) -> dict[tuple[str, str], dict]:
    """{(문항ID, 축): 정답 레코드}."""
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in ratings:
        cells[(r["item_id"], r["axis"])].append(r)

    gold = {}
    for key, rs in cells.items():
        labels = [r["judgment"] for r in rs]
        n_v = sum(1 for j in labels if j == VIOLATED)
        is_v = n_v >= min_votes
        top, top_n = Counter(labels).most_common(1)[0]
        gold[key] = {
            "item_id": key[0], "axis": key[1],
            "n_raters": len(rs),
            "labels": {r["rater_id"]: r["judgment"] for r in rs},
            "n_violated": n_v,
            "violated": is_v,
            "severity": _severity_vote([r["severity"] for r in rs if r["judgment"] == VIOLATED]) if is_v else None,
            "label4": top if top_n >= 2 else None,
            "unanimous": top_n == len(rs),
        }
    return gold


def gold_summary(gold: dict) -> list[dict]:
    """축별 요약 (PROJECT_NOTES 4절 표와 같은 열)."""
    out = []
    for ax in AXES:
        g = [v for (_, a), v in gold.items() if a == ax]
        out.append({
            "axis": ax,
            "n_items": len(g),
            "unanimous": sum(v["unanimous"] for v in g),
            "all_violated": sum(v["n_violated"] == v["n_raters"] for v in g),
            "majority_violated": sum(v["violated"] for v in g),
            "any_violated": sum(v["n_violated"] >= 1 for v in g),
            "label4_undecided": sum(v["label4"] is None for v in g),
            "severity_dist": dict(Counter(v["severity"] for v in g if v["violated"])),
        })
    return out


def gold_rows(gold: dict) -> list[dict]:
    rows = []
    for (it, ax), v in sorted(gold.items(), key=lambda kv: (kv[0][0], AXES.index(kv[0][1]))):
        rows.append({
            "item_id": it, "axis": ax,
            **{f"label_{k}": lab for k, lab in sorted(v["labels"].items())},
            "n_violated": v["n_violated"], "gold_violated": int(v["violated"]),
            "gold_severity": v["severity"] or "", "gold_label4": v["label4"] or "불일치",
            "unanimous": int(v["unanimous"]),
        })
    return rows


__all__ = ["build_gold", "gold_summary", "gold_rows", "SEVERITIES"]


# ---------------------------------------------------------------- 합의 정답
def load_consensus(xlsx_path) -> dict[tuple[str, str], dict]:
    """합의본 엑셀(평가 시트 I·J·K·L열)에서 {(문항ID, 축): {final, severity, change_reason, opinion}}."""
    import re
    from openpyxl import load_workbook
    ws = load_workbook(xlsx_path, data_only=True)["평가"]
    out = {}
    for r in ws.iter_rows(min_row=9, values_only=True):
        if not r or not r[0] or not re.fullmatch(r"P\d{3}", str(r[0])):
            continue
        out[(r[0], r[1])] = {"final": r[8], "severity": r[9], "change_reason": r[10], "opinion": r[11]}
    return out


def build_gold_consensus(ratings: list[dict], consensus: dict) -> dict[tuple[str, str], dict]:
    """다수결 정답 구조를 그대로 두고 이진·심각도·4범주 정답만 합의 최종판정으로 바꾼다.
    합의 최종판정이 판단보류(합의 불가 포함)인 칸은 채점 규칙 v1과 같이 비훼손으로 세고 consensus_hold=True로 표시."""
    gold = build_gold(ratings)
    missing = [k for k in gold if k not in consensus or not consensus[k]["final"]]
    if missing:
        raise ValueError(f"합의 최종판정이 없는 칸 {len(missing)}개: {missing[:5]}")
    for k, g in gold.items():
        c = consensus[k]
        g["majority_violated"] = g["violated"]
        g["violated"] = c["final"] == VIOLATED
        g["severity"] = c["severity"] if g["violated"] else None
        g["label4"] = c["final"]
        g["consensus_hold"] = c["final"] == "판단보류"
    return gold
