"""선별 누락 비교: "Haiku 선별이 Sonnet 대비 얼마나 놓치는지" (PROJECT_NOTES 3-3).

선별 실행(screen)과 기준 실행(reference, 보통 같은 문항을 Sonnet 전체로 돌린 것)을 비교한다.
2단계에서 선별이 넘기지 않은 문항은 다시 보지 않으므로, 그 문항의 훼손은 영원히 놓친다.

  - 재판정 비율: 선별이 넘긴 문항 비율 → 2단계 비용을 좌우
  - 기준 대비 누락: 기준 실행이 훼손으로 본 칸 중, 선별이 넘기지 않은 문항에 있는 칸
  - 정답 대비 누락 (파일럿처럼 사람 정답이 있을 때): 다수결 훼손 칸 중 선별이 넘기지 않은 칸
    → 2단계 전체의 재현율 상한
"""
from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path

from .constants import AXES, VIOLATED
from .io_utils import fmt, read_json, write_csv
from .runner import ESCALATE_DEFAULT, escalation_list, latest_records


def compare_runs(screen_dir: Path, ref_dir: Path, gold: dict | None, reports_dir: Path,
                 on: tuple[str, ...] = ESCALATE_DEFAULT) -> Path:
    screen_dir, ref_dir = Path(screen_dir), Path(ref_dir)
    m1, m2 = read_json(screen_dir / "meta.json"), read_json(ref_dir / "meta.json")
    items = [i for i in m1["items"] if i in set(m2["items"])]
    esc = {r["item_id"]: r for r in escalation_list(screen_dir, on)}
    ref = latest_records(ref_dir)
    n_esc = sum(esc[i]["escalate"] for i in items)

    ref_cnt, ref_miss, gold_cnt, gold_miss = Counter(), Counter(), Counter(), Counter()
    missed = []
    for iid in items:
        out = (ref.get(iid) or {}).get("output") or {}
        for ax in AXES:
            ref_v = ((out.get("axes") or {}).get(ax) or {}).get("judgment") == VIOLATED
            gold_v = bool(gold and gold.get((iid, ax), {}).get("violated"))
            if ref_v:
                ref_cnt[ax] += 1
            if gold_v:
                gold_cnt[ax] += 1
            if not esc[iid]["escalate"] and (ref_v or gold_v):
                ref_miss[ax] += ref_v
                gold_miss[ax] += gold_v
                missed.append({"item_id": iid, "axis": ax, "ref_violated": int(ref_v),
                               "gold_violated": int(gold_v) if gold else "",
                               "ref_reason": ((out.get("axes") or {}).get(ax) or {}).get("reason")})

    tag = f"compare_{screen_dir.name}__vs__{ref_dir.name}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_csv(reports_dir / f"{tag}_missed.csv", missed,
              ["item_id", "axis", "ref_violated", "gold_violated", "ref_reason"])

    def rate(miss, cnt):
        return f"{cnt - miss}/{cnt} ({fmt((cnt - miss) / cnt) if cnt else '—'})"

    md = [f"# 선별 누락 비교 — {screen_dir.name} vs {ref_dir.name}", "",
          f"- 작성: {dt.date.today().isoformat()} · 선별 `{m1['model']}` (해시 `{m1['prompt_fingerprint']}`) · "
          f"기준 `{m2['model']}` (해시 `{m2['prompt_fingerprint']}`) · 공통 문항 {len(items)}개",
          f"- 재판정 기준: {', '.join(on)} · 재판정 대상 {n_esc}/{len(items)} "
          f"({n_esc / len(items):.0%}) ← 2단계 비용을 좌우", "",
          "## 축별: 선별이 넘긴 훼손 칸 (넘김/전체)", "",
          "| 축 | 기준 실행의 훼손 | 사람 다수결 훼손 |", "|---|---|---|"]
    for ax in AXES + ["ALL"]:
        rc = sum(ref_cnt.values()) if ax == "ALL" else ref_cnt[ax]
        rm = sum(ref_miss.values()) if ax == "ALL" else ref_miss[ax]
        gc = sum(gold_cnt.values()) if ax == "ALL" else gold_cnt[ax]
        gm = sum(gold_miss.values()) if ax == "ALL" else gold_miss[ax]
        md.append(f"| {ax} | {rate(rm, rc)} | {rate(gm, gc) if gold else '—'} |")
    md += ["", "- '사람 다수결 훼손' 열은 2단계 전체가 낼 수 있는 재현율의 상한 (선별이 넘기지 않으면 2차에서도 못 잡음)",
           f"- 놓친 칸 목록: `{tag}_missed.csv` ({len(missed)}칸)"]
    path = reports_dir / f"{tag}.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return path
