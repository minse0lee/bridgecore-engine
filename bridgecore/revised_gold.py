"""합의 정답에 수정 가이드라인(2026-11-01) 기준을 반영한 '수정 정답'으로 다시 채점.

합의본은 옛 가이드라인으로 만들어져서, 수정 기준과 다른 칸이 18칸 있다
(docs/guideline/합의_수정기준_차이_18칸.xlsx). 이 칸만 수정 기준 판정으로 바꾼다.

- 훼손/아님이 바뀌는 칸 15개: 아래 REVISED에 반영
- N/A↔보존처럼 '훼손 아님' 안에서만 바뀌는 칸 2개(P016·P021 Cultural): 채점에 영향 없음
- 수정 기준이 판정을 정하지 않은 칸 1개(P032 Deontic, '평가 대상이고 보존·훼손은 검수자 판정'):
  합의(N/A=훼손 아님)를 그대로 둔다
- P052 Semantic은 수정 기준 규칙상 '아님'이지만 확인 필요로 남은 경계 사례라, 민감도로 반대 경우도 본다
"""
from __future__ import annotations

from pathlib import Path

from .constants import AXES, VIOLATED
from .guideline_effect import _prf, sentence_gold, sentence_preds
from .io_utils import write_csv
from .runner import latest_records
from .version_compare import _axis_table, _preds

T, F = True, False
REVISED = {
    ("P013", "Semantic"): T, ("P013", "Textual"): T, ("P039", "Textual"): T,
    ("P045", "Semantic"): T, ("P045", "Deontic"): T, ("P049", "Semantic"): T,
    ("P041", "Semantic"): T, ("P033", "Semantic"): T, ("P046", "Context"): T,
    ("P002", "Register"): T, ("P014", "Cultural"): T, ("P043", "Cultural"): T,
    ("P007", "Cultural"): F, ("P053", "Semantic"): F, ("P052", "Semantic"): F,
}
NO_BINARY_CHANGE = [("P016", "Cultural"), ("P021", "Cultural"), ("P032", "Deontic")]
BORDERLINE = ("P052", "Semantic")


def apply(gold: dict, overrides: dict) -> dict:
    out = {k: dict(v) for k, v in gold.items()}
    for k, val in overrides.items():
        out[k]["violated"] = val
    return out


def _rater_axis_f1(gold: dict, ids: list[str]) -> dict:
    """평가자 한 명의 처음 판정을 '예측'으로 보고 정답과 비교 (사람 수준 참고값)."""
    raters = sorted({r for i in ids for a in AXES for r in gold[(i, a)]["labels"]})
    res = {}
    for r in raters:
        tp = fp = fn = 0
        for i in ids:
            for a in AXES:
                lab = gold[(i, a)]["labels"].get(r)
                p, g = lab == "훼손", gold[(i, a)]["violated"]
                tp += p and g; fp += p and not g; fn += g and not p
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        res[r] = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
    return res


def rescore(configs: dict[str, dict], gold: dict, splits: dict[str, list[str]], reports_dir: Path,
            name: str = "rescore_revised_gold") -> Path:
    """configs: {이름: {"kind": engine|generic, "dev": [runs], "holdout": [runs]}}"""
    g_new = apply(gold, REVISED)
    g_alt = apply(gold, {**REVISED, BORDERLINE: True})
    f = lambda x: "-" if x is None else f"{x:.2f}"
    lines = [f"# 수정 기준 반영 재채점 ({name})", "",
             "- 기존 정답: 평가자 합의본(11/1, 옛 가이드라인 기준). 판단보류는 훼손 아님",
             f"- 수정 정답: 합의본에서 수정 가이드라인 기준과 다른 18칸 중 훼손/아님이 바뀌는 {len(REVISED)}칸만 교체 "
             f"(훼손으로 {sum(REVISED.values())}칸, 아님으로 {sum(not v for v in REVISED.values())}칸)",
             "- P032 Deontic은 수정 기준이 판정을 정하지 않아 합의 그대로. P052 Semantic은 경계 사례라 '훼손'으로 본 경우도 함께 표시",
             "- 반복 실행이 있으면 평균. 엔진 프롬프트·실행 결과는 그대로이고 정답만 바뀜", ""]
    for sp, ids in splits.items():
        moved = [k for k in REVISED if k[0] in set(ids)]
        n_old = sum(gold[(i, a)]["violated"] for i in ids for a in AXES)
        n_new = sum(g_new[(i, a)]["violated"] for i in ids for a in AXES)
        lines += [f"## {'개발용' if sp == 'dev' else '확인용'} {len(ids)}문항", "",
                  f"바뀐 칸 {len(moved)}개: " + ", ".join(f"{i} {a}" for i, a in moved),
                  f"정답 훼손 칸 {n_old} → {n_new}", "",
                  "| 구성 | 축 단위 F1 기존 → 수정 | (P052 훼손으로 보면) | 문장 단위 F1 기존 → 수정 |", "|---|---|---|---|"]
        for label, c in configs.items():
            runs = [r for r in c.get(sp, []) if (r / "outputs.jsonl").exists()]
            if not runs:
                continue
            sg_old, sg_new = sentence_gold(gold, ids), sentence_gold(g_new, ids)
            s_old = sum(_prf(sentence_preds(r, c["kind"]), sg_old, ids)["f1"] for r in runs) / len(runs)
            s_new = sum(_prf(sentence_preds(r, c["kind"]), sg_new, ids)["f1"] for r in runs) / len(runs)
            if c["kind"] == "engine":
                a_old = _axis_table(runs, gold, ids)["ALL"]["f1"]
                a_new = _axis_table(runs, g_new, ids)["ALL"]["f1"]
                a_alt = _axis_table(runs, g_alt, ids)["ALL"]["f1"]
                ax = f"{f(a_old)} → **{f(a_new)}** ({a_new - a_old:+.2f})"
                alt = f(a_alt)
            else:
                ax, alt = "(축 판정 없음)", "-"
            lines.append(f"| {label} | {ax} | {alt} | {f(s_old)} → **{f(s_new)}** ({s_new - s_old:+.2f}) |")
        hr_old, hr_new = _rater_axis_f1(gold, ids), _rater_axis_f1(g_new, ids)
        lines += ["", "참고, 평가자 한 명의 처음 판정을 정답과 비교한 축 단위 F1: " +
                  ", ".join(f"{r} {hr_old[r]:.2f}→{hr_new[r]:.2f}" for r in hr_old) +
                  " (정답이 평가자 판정에서 나왔으므로 낙관적인 값. 처음 판정은 옛 가이드라인 기준)", ""]

    # 엔진 v1.1 Sonnet 기준 축별 표와 남은 오류
    main = configs.get("엔진 v1.1 (Sonnet)")
    rows = []
    if main:
        for sp, ids in splits.items():
            runs = [r for r in main.get(sp, []) if (r / "outputs.jsonl").exists()]
            if not runs:
                continue
            a, b = _axis_table(runs, gold, ids), _axis_table(runs, g_new, ids)
            lines += [f"## 엔진 v1.1 (Sonnet) 축별 · {'개발용' if sp == 'dev' else '확인용'}", "",
                      "| 축 | 정답 훼손(수정) | 기존 P/R/F1 | 수정 P/R/F1 | 오탐 | 놓침 |", "|---|---|---|---|---|---|"]
            for axn in AXES + ["ALL"]:
                npos = sum(g_new[(i, x)]["violated"] for i in ids for x in ([axn] if axn != "ALL" else AXES))
                lines.append(f"| {'**전체**' if axn == 'ALL' else axn} | {npos} | {f(a[axn]['p'])}/{f(a[axn]['r'])}/{f(a[axn]['f1'])} | "
                             f"{f(b[axn]['p'])}/{f(b[axn]['r'])}/{f(b[axn]['f1'])} | {b[axn]['fp']:.1f} | {b[axn]['fn']:.1f} |")
            lines.append("")
            # 남은 오류: 두 번 실행에서 모두 틀린 칸(일관된 오류)과 한 번만 틀린 칸
            preds = [_preds(r, ids) for r in runs]
            recs = [latest_records(r) for r in runs]
            for i in ids:
                for axn in AXES:
                    g = g_new[(i, axn)]["violated"]
                    js = [p[(i, axn)]["judgment"] or "출력 없음" for p in preds]
                    wrong = [(j == VIOLATED) != g for j in js]
                    if not any(wrong):
                        continue
                    reason = ((((recs[0].get(i) or {}).get("output") or {}).get("axes") or {}).get(axn) or {}).get("reason", "")
                    rows.append({"구분": "개발용" if sp == "dev" else "확인용", "item_id": i, "axis": axn,
                                 "오류": "놓침" if g else "오탐", "틀린 횟수": f"{sum(wrong)}/{len(js)}",
                                 "정답(수정)": "훼손" if g else "아님", "합의 원판정": gold[(i, axn)]["label4"],
                                 "처음 판정": " / ".join(g_new[(i, axn)]["labels"].get(r, "-") for r in sorted(g_new[(i, axn)]["labels"])),
                                 "수정 칸": "예" if (i, axn) in REVISED else "",
                                 "엔진 판정": " / ".join(js), "엔진 근거(1회차)": reason})
        # 18칸에서 엔진 판정
        lines += ["## 18칸에서 엔진 v1.1 (Sonnet) 판정", "", "| 문항 | 축 | 합의 | 수정 기준 | 엔진 (실행별) | 수정 기준과 |", "|---|---|---|---|---|---|"]
        for k in list(REVISED) + NO_BINARY_CHANGE:
            i, axn = k
            sp = next(s for s, ids in splits.items() if i in ids)
            runs = [r for r in main.get(sp, []) if (r / "outputs.jsonl").exists()]
            js = [_preds(r, [i])[(i, axn)]["judgment"] or "출력 없음" for r in runs]
            if k in REVISED:
                tgt = "훼손" if REVISED[k] else "아님"
                agree = sum((j == VIOLATED) == REVISED[k] for j in js)
                verdict = f"{agree}/{len(js)} 일치"
            else:
                tgt, verdict = ("N/A" if axn == "Cultural" else "평가 대상(판정 미정)"), "-"
            lines.append(f"| {i} ({'개발' if sp == 'dev' else '확인'}) | {axn} | {gold[k]['label4']} | {tgt} | {' / '.join(js)} | {verdict} |")
        lines.append("")
    n_cons = sum(1 for r in rows if r["틀린 횟수"].split("/")[0] == r["틀린 횟수"].split("/")[1])
    lines += [f"남은 오류 목록: `{name}_errors.csv` (v1.1 Sonnet, 한 번이라도 틀린 칸 {len(rows)}개, 모든 실행에서 틀린 칸 {n_cons}개)", "",
              "## 한계", "",
              "- 수정 정답은 합의본에 수안님 한 사람의 확정 판정(11/1)을 덮은 것이다. 18칸 외의 칸은 옛 가이드라인 기준 그대로라, 새 기준으로 다시 보면 더 바뀔 칸이 있을 수 있다",
              "- 개발용 40문항은 v1·v1.1을 만들 때 참고한 문항이고, 수정 기준도 엔진 v1.1에 이미 반영되어 있어 개발용 점수는 낙관적이다",
              "- 확인용 20문항은 이번에 두 번째로 채점했다(엔진 실행은 다시 하지 않음, 정답만 교체). 엔진을 이 결과에 맞춰 고치면 확인용으로서의 의미가 사라진다"]
    path = reports_dir / f"{name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(reports_dir / f"{name}_errors.csv", rows)
    print("\n".join(lines))
    return path
