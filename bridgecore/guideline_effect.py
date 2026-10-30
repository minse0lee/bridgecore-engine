"""가이드라인 효과 비교: 엔진(v0) vs 같은 모델의 기준선 프롬프트.

  - generic : 가이드라인·6축 없이 "번역 오류를 찾아라" (baseline.py) → 문장 단위로만 비교
  - b_axes  : 6축 이름과 판정값만 주고 상세 규칙·경계 규칙·예시는 뺀 프롬프트 → 문장·축 단위 비교
  - v0      : 현재 엔진

정답은 평가자 3명 다수결(임시 정답). 문장 단위 정답 = 여섯 축 중 하나라도 다수결 훼손.
같은 설정을 두 번 돌린 실행은 평균을 낸다. 차이의 신뢰구간은 문항을 다시 뽑는 쌍 부트스트랩.
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from .constants import AXES, VIOLATED
from .io_utils import read_json, read_jsonl
from .report import _calls_for_run, _cost
from .runner import latest_records
from .scoring import predictions_from_outputs, score

MAJOR_UP = {"major", "critical"}


def _prf(pred: dict, gold: dict, ids: list[str]) -> dict:
    tp = sum(1 for i in ids if pred.get(i) and gold[i])
    fp = sum(1 for i in ids if pred.get(i) and not gold[i])
    fn = sum(1 for i in ids if not pred.get(i) and gold[i])
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f1": f, "flagged": tp + fp}


def sentence_preds(run_dir: Path, kind: str, major_only: bool = False) -> dict[str, bool]:
    recs = latest_records(run_dir)
    out = {}
    for iid, rec in recs.items():
        o = rec.get("output") or {}
        if kind == "generic":
            errs = o.get("errors") or []
            out[iid] = any((not major_only) or e.get("severity") in MAJOR_UP for e in errs)
        else:
            axes = o.get("axes") or {}
            out[iid] = any(a.get("judgment") == VIOLATED and ((not major_only) or a.get("severity") in MAJOR_UP)
                           for a in axes.values())
    return out


def sentence_gold(gold: dict, ids: list[str]) -> dict[str, bool]:
    return {i: any(gold[(i, a)]["violated"] for a in AXES if (i, a) in gold) for i in ids}


def human_sentence_baseline(gold: dict, ids: list[str]) -> dict:
    raters = sorted({r for g in gold.values() for r in g["labels"]})
    flags = {r: {i: any(gold[(i, a)]["labels"].get(r) == VIOLATED for a in AXES) for i in ids} for r in raters}
    sg = sentence_gold(gold, ids)
    vs_majority = {r: _prf(flags[r], sg, ids)["f1"] for r in raters}
    pairs = {}
    for k, a in enumerate(raters):
        for b in raters[k + 1:]:
            pairs[f"{a}-{b}"] = _prf(flags[a], flags[b], ids)["f1"]
    return {"vs_majority": vs_majority, "pairs": pairs}


def bootstrap_diff(fam_a: list[dict], fam_b: list[dict], gold: dict, ids: list[str],
                   n: int = 5000, seed: int = 20261030) -> tuple[float, float, float, float]:
    """(관측 차이, 2.5%, 97.5%, A가 B보다 낮거나 같은 비율). 각 집단은 반복 실행 평균."""
    rng = random.Random(seed)

    def fam_f1(fam, sample):
        return sum(_prf(p, gold, sample)["f1"] for p in fam) / len(fam)

    obs = fam_f1(fam_a, ids) - fam_f1(fam_b, ids)
    diffs = []
    for _ in range(n):
        s = [rng.choice(ids) for _ in ids]
        diffs.append(fam_f1(fam_a, s) - fam_f1(fam_b, s))
    diffs.sort()
    return obs, diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1], sum(d <= 0 for d in diffs) / n


def run_comparison(families: dict[str, dict], gold: dict, ids: list[str], reports_dir: Path) -> Path:
    """families: {이름: {"kind": "engine"|"generic", "runs": [Path, ...], "note": str}}"""
    sg = sentence_gold(gold, ids)
    n_pos = sum(sg.values())
    lines = ["# 가이드라인 효과 비교 (개발용 40문항, 프롬프트 v0)", "",
             f"- 정답: 평가자 3명 다수결 임시 정답 (합의 전). 문장 단위 정답 = 여섯 축 중 하나라도 다수결 훼손 → "
             f"{len(ids)}문장 중 {n_pos}문장이 훼손, {len(ids) - n_pos}문장이 훼손 없음",
             "- 모든 방식이 같은 모델·같은 문항 정보(문서정보·평가조건·문단 문맥·그림)를 받는다. 바뀌는 것은 시스템 프롬프트뿐",
             "- 같은 설정을 두 번 돌려 평균을 냈다 (괄호 안은 1회·2회 값)", ""]

    lines += ["## 1. 문장 단위: 훼손이 있는 문장을 찾았나", "",
              "| 방식 | 모델 | 표시한 문장 | P | R | F1 | F1 (major 이상만 표시로 셀 때) | 문항당 비용 |",
              "|---|---|---|---|---|---|---|---|"]
    fam_preds: dict[str, list[dict]] = {}
    summary = {}
    for name, fam in families.items():
        preds = [sentence_preds(r, fam["kind"]) for r in fam["runs"]]
        preds_major = [sentence_preds(r, fam["kind"], major_only=True) for r in fam["runs"]]
        fam_preds[name] = preds
        ms = [_prf(p, sg, ids) for p in preds]
        mj = [_prf(p, sg, ids)["f1"] for p in preds_major]
        avg = {k: sum(m[k] for m in ms) / len(ms) for k in ("p", "r", "f1", "flagged")}
        costs = []
        for r in fam["runs"]:
            meta = read_json(r / "meta.json")
            c = _cost(_calls_for_run(r, meta, latest_records(r), set(ids)), len(ids))
            costs.append(c["usd_per_item"] if c else None)
        cost = (sum(costs) / len(costs)) if all(c is not None for c in costs) else None
        f1s = " · ".join(f"{m['f1']:.2f}" for m in ms)
        summary[name] = {**avg, "f1_runs": [m["f1"] for m in ms], "major_f1": sum(mj) / len(mj), "cost": cost}
        lines.append(f"| {name} | {fam['model']} | {avg['flagged']:.1f}/{len(ids)} | {avg['p']:.2f} | {avg['r']:.2f} | "
                     f"**{avg['f1']:.2f}** ({f1s}) | {sum(mj) / len(mj):.2f} | "
                     f"{'$%.4f' % cost if cost is not None else '-'} |")
    hb = human_sentence_baseline(gold, ids)
    lines += ["", f"참고 — 사람: 평가자 한 명을 다수결과 비교한 문장 단위 F1 "
              f"{', '.join(f'{r} {v:.2f}' for r, v in hb['vs_majority'].items())} "
              f"(자기 판정이 다수결에 들어가 사람에게 유리한 값). 평가자 둘끼리 비교 "
              f"{', '.join(f'{k} {v:.2f}' for k, v in hb['pairs'].items())}.", ""]

    lines += ["## 2. 차이의 신뢰구간 (문장 단위 F1, 쌍 부트스트랩 5,000회)", "",
              "| 비교 | 차이 | 95% 구간 | 차이가 0 이하인 비율 |", "|---|---|---|---|"]
    pairs = [(a, b) for a, b in [
        ("엔진 v0 (Sonnet)", "일반 프롬프트 (Sonnet)"), ("엔진 v0 (Haiku)", "일반 프롬프트 (Haiku)"),
        ("엔진 v0 2단계", "일반 프롬프트 (Sonnet)"),
        ("엔진 v0 (Sonnet)", "축 이름만 (Sonnet)"), ("엔진 v0 (Haiku)", "축 이름만 (Haiku)"),
        ("축 이름만 (Sonnet)", "일반 프롬프트 (Sonnet)")] if a in fam_preds and b in fam_preds]
    boot = {}
    for a, b in pairs:
        obs, lo, hi, p0 = bootstrap_diff(fam_preds[a], fam_preds[b], sg, ids)
        boot[(a, b)] = (obs, lo, hi, p0)
        lines.append(f"| {a} − {b} | {obs:+.2f} | {lo:+.2f} ~ {hi:+.2f} | {p0:.3f} |")

    lines += ["", "## 3. 축 단위 (6축 출력이 있는 방식만)", "",
              "| 방식 | 모델 | 전체 P | 전체 R | 전체 F1 | Context 오탐 | Textual 오탐 | Semantic F1 |",
              "|---|---|---|---|---|---|---|---|"]
    axis_summary = {}
    for name, fam in families.items():
        if fam["kind"] != "engine":
            continue
        rows = []
        for r in fam["runs"]:
            outs = {i: rec.get("output") for i, rec in latest_records(r).items() if i in set(ids)}
            preds, _ = predictions_from_outputs(outs)
            sc = score(gold, preds, ids)
            fp_ctx = sum(1 for i in ids if preds.get((i, "Context"), {}).get("judgment") == VIOLATED
                         and not gold[(i, "Context")]["violated"])
            fp_txt = sum(1 for i in ids if preds.get((i, "Textual"), {}).get("judgment") == VIOLATED
                         and not gold[(i, "Textual")]["violated"])
            al = sc["per_axis"]["ALL"]["strict"]
            rows.append((al["precision"] or 0, al["recall"] or 0, al["f1"] or 0, fp_ctx, fp_txt,
                         sc["per_axis"]["Semantic"]["strict"]["f1"] or 0))
        avg = [sum(x[k] for x in rows) / len(rows) for k in range(6)]
        axis_summary[name] = avg
        lines.append(f"| {name} | {fam['model']} | {avg[0]:.2f} | {avg[1]:.2f} | **{avg[2]:.2f}** | {avg[3]:.1f} | "
                     f"{avg[4]:.1f} | {avg[5]:.2f} |")

    # 축 단위 차이의 신뢰구간: 문항을 다시 뽑고 그 문항의 6칸을 함께 넣는다
    cell_preds = {}
    for name, fam in families.items():
        if fam["kind"] == "engine":
            cell_preds[name] = [predictions_from_outputs(
                {i: rec.get("output") for i, rec in latest_records(r).items() if i in set(ids)})[0]
                for r in fam["runs"]]

    def cell_f1(preds_list, sample):
        f = []
        for pr in preds_list:
            tp = fp = fn = 0
            for i in sample:
                for a in AXES:
                    pv = (pr.get((i, a)) or {}).get("judgment") == VIOLATED
                    gv = gold[(i, a)]["violated"]
                    tp += pv and gv; fp += pv and not gv; fn += gv and not pv
            f.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0)
        return sum(f) / len(f)

    ax_pairs = [(a, b) for a, b in [("엔진 v0 (Sonnet)", "축 이름만 (Sonnet)"), ("엔진 v0 (Haiku)", "축 이름만 (Haiku)")]
                if a in cell_preds and b in cell_preds]
    if ax_pairs:
        lines += ["", "축 단위 전체 F1 차이 (쌍 부트스트랩 5,000회, 문항 단위로 다시 뽑음)", "",
                  "| 비교 | 차이 | 95% 구간 | 차이가 0 이하인 비율 |", "|---|---|---|---|"]
        rng = random.Random(20261030)
        for a, b in ax_pairs:
            obs = cell_f1(cell_preds[a], ids) - cell_f1(cell_preds[b], ids)
            ds = []
            for _ in range(5000):
                smp = [rng.choice(ids) for _ in ids]
                ds.append(cell_f1(cell_preds[a], smp) - cell_f1(cell_preds[b], smp))
            ds.sort()
            lines.append(f"| {a} − {b} | {obs:+.2f} | {ds[125]:+.2f} ~ {ds[4874]:+.2f} | "
                         f"{sum(d <= 0 for d in ds) / 5000:.3f} |")

    lines += ["", "## 4. 일반 프롬프트가 훼손 없는 문장에 단 오류 (1회차, 앞부분)", ""]
    for name, fam in families.items():
        if fam["kind"] != "generic":
            continue
        recs = latest_records(fam["runs"][0])
        sev = Counter()
        ex = []
        for i in ids:
            if sg[i]:
                continue
            for e in (recs.get(i, {}).get("output") or {}).get("errors") or []:
                sev[e["severity"]] += 1
                if len(ex) < 8:
                    ex.append(f"- {i} [{e['severity']}] “{e['src_span'][:40]}” → “{e['tgt_span'][:40]}”: "
                              f"{e['description'][:110]}")
        lines += [f"**{name}** — 심각도 분포 {dict(sev)}", ""] + ex + [""]

    lines += ["## 읽는 법·한계", "",
              "- 정답이 가이드라인 기준으로 판정한 사람 다수결이다. 일반 프롬프트는 가이드라인이 오류로 치지 않는 것"
              "(어색함·번역투 등)도 오류로 적을 수 있어 정밀도에서 불리하다. 이 비교는 '가이드라인 기준의 훼손을 "
              "얼마나 맞게 찾는가'를 재는 것이고, 그게 이 엔진의 목표다",
              "- 40문항, 임시 정답이라 차이가 작으면 우연과 구분할 수 없다. 같은 설정 반복 간 흔들림도 함께 볼 것",
              "- 'major 이상만 표시로 셀 때'는 결과를 본 뒤 정한 기준이라 사후 비교다. 미리 정한 비교는 '오류를 하나라도 표시'다",
              "- 다수결 정답이 놓친 실제 오류가 있을 수 있다. 예: P013·P039는 다수결로는 훼손이 아니지만 수안님 확인에서 훼손으로 정리됐다 → 이런 칸의 표시는 오탐으로 집계된다",
              "- 형식 오류로 출력이 없는 문항은 '표시 없음'으로 셌다 (축 이름만 Sonnet 한 실행의 P007)",
              "- 확인용 20문항은 쓰지 않았다"]
    path = reports_dir / "guideline_effect_dev.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return path
