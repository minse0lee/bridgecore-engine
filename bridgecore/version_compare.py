"""프롬프트 버전 비교 (예: v0 → v1): 축별 P/R/F1(반복 실행 평균)과 판정이 바뀐 칸 목록.

정답은 평가자 다수결 임시 정답. 칸 목록에는 사람 표수(훼손 표 수/3)를 함께 적어,
고쳐진 칸·새로 틀린 칸이 사람끼리 갈린 칸인지 바로 볼 수 있게 한다.
"""
from __future__ import annotations

from pathlib import Path

from .constants import AXES, VIOLATED
from .io_utils import write_csv
from .runner import latest_records
from .scoring import predictions_from_outputs, score


def _preds(run: Path, ids: list[str]) -> dict:
    outs = {i: r.get("output") for i, r in latest_records(run).items() if i in set(ids)}
    return predictions_from_outputs(outs)[0]


def _axis_table(runs: list[Path], gold: dict, ids: list[str]) -> dict:
    acc: dict = {}
    for r in runs:
        sc = score(gold, _preds(r, ids), ids)["per_axis"]
        for ax in AXES + ["ALL"]:
            s = sc[ax]["strict"]
            a = acc.setdefault(ax, {"p": [], "r": [], "f1": [], "fp": [], "fn": []})
            a["p"].append(s["precision"]); a["r"].append(s["recall"]); a["f1"].append(s["f1"])
            a["fp"].append(s["fp"]); a["fn"].append(s["fn"])
    avg = lambda xs: None if any(x is None for x in xs) else sum(xs) / len(xs)
    return {ax: {k: avg(v) for k, v in a.items()} for ax, a in acc.items()}


def compare_versions(pairs: dict[str, tuple[list[Path], list[Path]]], gold: dict, ids: list[str],
                     reports_dir: Path, name: str = "v1_vs_v0_dev") -> Path:
    """pairs: {구성 이름: ([이전 버전 실행들], [새 버전 실행들])}"""
    f = lambda x: "-" if x is None else f"{x:.2f}"
    lines = [f"# 프롬프트 버전 비교 ({name})", "",
             "- 정답: 평가자 3명 다수결 임시 정답 (합의 전). 개발용 40문항 × 6축 = 240칸",
             "- 반복 실행이 있으면 평균. 오탐(FP)·놓침(FN)은 칸 수", ""]
    case_rows = []
    for label, (old_runs, new_runs) in pairs.items():
        a, b = _axis_table(old_runs, gold, ids), _axis_table(new_runs, gold, ids)
        lines += [f"## {label}", "", "| 축 | 정답 훼손 | 이전 P/R/F1 | 새 P/R/F1 | F1 변화 | 오탐 이전→새 | 놓침 이전→새 |",
                  "|---|---|---|---|---|---|---|"]
        for ax in AXES + ["ALL"]:
            n_pos = sum(1 for i in ids for x in ([ax] if ax != "ALL" else AXES) if gold[(i, x)]["violated"])
            d = (b[ax]["f1"] - a[ax]["f1"]) if a[ax]["f1"] is not None and b[ax]["f1"] is not None else None
            lines.append(f"| {'**전체**' if ax == 'ALL' else ax} | {n_pos} | {f(a[ax]['p'])}/{f(a[ax]['r'])}/{f(a[ax]['f1'])} | "
                         f"{f(b[ax]['p'])}/{f(b[ax]['r'])}/{f(b[ax]['f1'])} | {'-' if d is None else f'{d:+.2f}'} | "
                         f"{a[ax]['fp']:.1f}→{b[ax]['fp']:.1f} | {a[ax]['fn']:.1f}→{b[ax]['fn']:.1f} |")
        lines.append("")
        # 판정이 바뀐 칸 (각 버전 첫 실행 기준)
        po, pn = _preds(old_runs[0], ids), _preds(new_runs[0], ids)
        new_recs = latest_records(new_runs[0])
        fixed = broke = 0
        for i in ids:
            for ax in AXES:
                g = gold[(i, ax)]
                vo = po[(i, ax)]["judgment"] == VIOLATED
                vn = pn[(i, ax)]["judgment"] == VIOLATED
                if vo == vn:
                    continue
                right_old, right_new = vo == g["violated"], vn == g["violated"]
                fixed += right_new and not right_old
                broke += right_old and not right_new
                reason = ((((new_recs.get(i) or {}).get("output") or {}).get("axes") or {}).get(ax) or {}).get("reason", "")
                case_rows.append({"구성": label, "item_id": i, "axis": ax, "사람 훼손 표": f"{g['n_violated']}/3",
                                  "정답": "훼손" if g["violated"] else "아님",
                                  "이전": po[(i, ax)]["judgment"], "새": pn[(i, ax)]["judgment"],
                                  "결과": "고쳐짐" if right_new else "새로 틀림", "새 근거": reason})
        lines += [f"판정이 바뀐 칸(1회차끼리): 고쳐짐 {fixed} · 새로 틀림 {broke}", ""]
    lines += ["## 한계", "",
              "- 개발용 40문항은 v1을 만들 때 오류 유형을 참고한 문항이라 점수가 낙관적일 수 있다. 처음 보는 문항에서의 성능은 확인용 20문항으로 본다",
              "- 정답이 합의 전 다수결이다. 합의 결과가 오면 다시 채점한다",
              "- 같은 설정 반복 간 전체 F1이 ±0.05쯤 흔들린다. 그보다 작은 차이는 우연과 구분할 수 없다"]
    path = reports_dir / f"{name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(reports_dir / f"{name}_changed_cells.csv", case_rows)
    print("\n".join(lines))
    return path
