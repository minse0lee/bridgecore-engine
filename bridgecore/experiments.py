"""실험 도구.

agree: 같은 설정으로 두 번 돌린 실행의 판정 일치도 (temperature를 고정할 수 없어 실행마다 판정이 흔들림).
  - 칸별 4범주 일치율, 훼손 여부 일치율, 훼손 여부 Cohen's κ, 한쪽만 훼손인 칸 수
  - 각 실행의 정답 대비 F1과 그 차이 → "프롬프트를 바꿔 F1이 이만큼 변한 게 우연인지" 판단 기준
sweep: 선별 설정 조합별 2단계 결과·비용. 2차는 새로 부르지 않고, 같은 모델·프롬프트로 전체 문항을
  돌린 기준 실행(예: dev_sonnet)의 결과를 재판정 결과로 쓴다.
"""
from __future__ import annotations

import datetime as dt
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from .constants import AXES, VIOLATED
from .io_utils import fmt, read_json, read_jsonl, write_csv
from .runner import escalation_list, latest_records, merge_cascade
from .scoring import predictions_from_outputs, score


def _kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else None


def _judgments(run_dir: Path) -> dict:
    out = {}
    for iid, rec in latest_records(run_dir).items():
        axes = ((rec.get("output") or {}).get("axes") or {})
        for ax in AXES:
            a = axes.get(ax) or {}
            out[(iid, ax)] = (a.get("judgment"), a.get("severity"))
    return out


def run_agreement(a_dir: Path, b_dir: Path, gold: dict | None, reports_dir: Path) -> Path:
    a_dir, b_dir = Path(a_dir), Path(b_dir)
    ma, mb = read_json(a_dir / "meta.json"), read_json(b_dir / "meta.json")
    same = {k: (ma.get(k), mb.get(k)) for k in ("model", "prompt_fingerprint", "effort", "thinking", "screen")}
    diff = {k: v for k, v in same.items() if v[0] != v[1]}
    ja, jb = _judgments(a_dir), _judgments(b_dir)
    items = sorted(set(ma["items"]) & set(mb["items"]))
    flips = []
    rows = []
    for ax in AXES + ["ALL"]:
        ks = [(i, x) for i in items for x in (AXES if ax == "ALL" else [ax])]
        l4 = sum(ja.get(k, (None,))[0] == jb.get(k, (None,))[0] for k in ks)
        bin_pairs = [(ja.get(k, (None,))[0] == VIOLATED, jb.get(k, (None,))[0] == VIOLATED) for k in ks]
        both_v = [k for k in ks if ja.get(k, (None,))[0] == VIOLATED and jb.get(k, (None,))[0] == VIOLATED]
        sev = sum(ja[k][1] == jb[k][1] for k in both_v)
        rows.append({"axis": ax, "n": len(ks), "label4_agree": l4 / len(ks),
                     "violated_agree": sum(a == b for a, b in bin_pairs) / len(ks),
                     "kappa": _kappa(bin_pairs),
                     "only_one_violated": sum(a != b for a, b in bin_pairs),
                     "violated_a": sum(a for a, _ in bin_pairs), "violated_b": sum(b for _, b in bin_pairs),
                     "severity_agree": sev / len(both_v) if both_v else None, "severity_n": len(both_v)})
        if ax != "ALL":
            for k in ks:
                if ja.get(k, (None,))[0] != jb.get(k, (None,))[0]:
                    g = gold.get(k) if gold else None
                    flips.append({"item_id": k[0], "axis": k[1], "run_a": ja.get(k, (None,))[0],
                                  "run_b": jb.get(k, (None,))[0],
                                  "gold_votes": f"{g['n_violated']}/{g['n_raters']}" if g else ""})
    f1 = {}
    if gold:
        for name, d in (("a", a_dir), ("b", b_dir)):
            outs = {i: r.get("output") for i, r in latest_records(d).items() if i in items}
            res = score(gold, predictions_from_outputs(outs)[0], items)["per_axis"]
            f1[name] = {ax: res[ax]["strict"]["f1"] for ax in AXES + ["ALL"]}

    tag = f"agree_{a_dir.name}__{b_dir.name}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_csv(reports_dir / f"{tag}_flips.csv", flips, ["item_id", "axis", "run_a", "run_b", "gold_votes"])
    md = [f"# 반복 실행 일치도 — {a_dir.name} vs {b_dir.name}", "",
          f"- 작성: {dt.date.today().isoformat()} · 공통 문항 {len(items)}개 · 설정 "
          + ("동일" if not diff else f"**다름** {diff}") + f" (모델 `{ma['model']}`, 해시 `{ma['prompt_fingerprint']}`)", "",
          "| 축 | 4범주 일치 | 훼손 여부 일치 | κ(훼손 여부) | 한쪽만 훼손 | 훼손 수 A/B | 심각도 일치 |"
          + (" F1 A | F1 B | F1 차이 |" if f1 else ""),
          "|---|---|---|---|---|---|---|" + ("---|---|---|" if f1 else "")]
    for r in rows:
        line = (f"| {r['axis']} | {fmt(r['label4_agree'])} | {fmt(r['violated_agree'])} | {fmt(r['kappa'])} | "
                f"{r['only_one_violated']} | {r['violated_a']}/{r['violated_b']} | "
                f"{fmt(r['severity_agree'])} (n={r['severity_n']}) |")
        if f1:
            fa, fb = f1["a"][r["axis"]], f1["b"][r["axis"]]
            line += f" {fmt(fa)} | {fmt(fb)} | {fmt(abs(fa - fb)) if fa is not None and fb is not None else '—'} |"
        md.append(line)
    vote_mix = Counter(f["gold_votes"] for f in flips)
    md += ["", f"- 판정이 바뀐 칸 {len(flips)}개 (`{tag}_flips.csv`). 사람 훼손 표 수별: {dict(sorted(vote_mix.items()))}",
           "- 사람도 갈린 칸(1/3, 2/3)에서 흔들리면 문항 자체가 애매한 것, 0/3·3/3에서 흔들리면 엔진이 불안정한 것"]
    path = reports_dir / f"{tag}.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return path


def _calls_cost(recs: list[dict], prices: dict) -> float | None:
    usd = 0.0
    for r in recs:
        u = r.get("usage")
        if not u:
            continue
        p = prices.get(r.get("model"))
        if not p:
            p = next((v for k, v in prices.items() if isinstance(v, dict) and str(r.get("model")).startswith(k)), None)
        if not p:
            return None
        mult = prices.get("batch_multiplier", 1.0) if r.get("via_batch") else 1.0
        usd += mult * (u["input_tokens"] * p["input"] + u["output_tokens"] * p["output"]
                       + u["cache_creation_input_tokens"] * p["cache_write"]
                       + u["cache_read_input_tokens"] * p["cache_read"]) / 1e6
    return usd


def screen_sweep(screens: list[Path], stage2_ref: Path, gold: dict, prices: dict, reports_dir: Path,
                 ons=(("violated", "hold"), ("violated",)), baselines: list[Path] = ()) -> Path:
    """선별 실행 × 넘김 기준 조합마다 2단계 결과를 만들어 채점한다.
    재판정 결과는 stage2_ref(전체 문항을 2차 모델로 돌린 실행)에서 가져온다."""
    stage2_ref = Path(stage2_ref)
    ref_recs = latest_records(stage2_ref)
    ref_all = read_jsonl(stage2_ref / "outputs.jsonl")
    ref_cost_item = {}
    for r in ref_all:  # 문항별 2차 비용 (재시도 포함 마지막 성공 호출 기준)
        ref_cost_item[r["item_id"]] = r
    rows = []
    for b in baselines:
        b = Path(b)
        meta = read_json(b / "meta.json")
        recs = latest_records(b)
        items = meta["items"]
        res = score(gold, predictions_from_outputs({i: recs[i].get("output") for i in items})[0], items)["per_axis"]
        cost = _calls_cost([recs[i] for i in items], prices)
        rows.append({"name": f"{b.name} (단일 모델)", "escalated": "—", "gold_passed": "—", "res": res,
                     "cost_item": cost / len(items) if cost is not None else None})
    tmp = Path(tempfile.mkdtemp())
    try:
        for s in screens:
            s = Path(s)
            meta = read_json(s / "meta.json")
            items = meta["items"]
            srecs = latest_records(s)
            for on in ons:
                esc = {r["item_id"]: r["escalate"] for r in escalation_list(s, on)}
                merged = merge_cascade(s, stage2_ref, tmp / f"{s.name}_{'+'.join(on)}", on)
                mrecs = latest_records(merged)
                res = score(gold, predictions_from_outputs({i: mrecs[i].get("output") for i in items})[0],
                            items)["per_axis"]
                gpos = [k for k, v in gold.items() if k[0] in set(items) and v["violated"]]
                passed = sum(esc[k[0]] for k in gpos)
                c1 = _calls_cost([srecs[i] for i in items], prices)
                c2 = _calls_cost([ref_cost_item[i] for i in items if esc[i]], prices)
                cost = c1 + c2 if c1 is not None and c2 is not None else None
                rows.append({"name": f"{s.name} · 넘김={'+'.join(on)}",
                             "escalated": f"{sum(esc.values())}/{len(items)}",
                             "gold_passed": f"{passed}/{len(gpos)}", "res": res,
                             "cost_item": cost / len(items) if cost is not None else None})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    md = [f"# 선별 설정 비교 (2차 결과는 `{stage2_ref.name}` 재사용)", "",
          f"- 작성: {dt.date.today().isoformat()}",
          "", "| 방식 | 재판정 대상 | 사람 훼손 중 2차로 넘김 | 전체 P | 전체 R | 전체 F1 | Semantic F1 | 문항당 비용 | 1만 문장 환산 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        a, sem = r["res"]["ALL"]["strict"], r["res"]["Semantic"]["strict"]
        c = r["cost_item"]
        md.append(f"| {r['name']} | {r['escalated']} | {r['gold_passed']} | {fmt(a['precision'])} | "
                  f"{fmt(a['recall'])} | {fmt(a['f1'])} | {fmt(sem['f1'])} | "
                  f"{('$%.5f' % c) if c is not None else '—'} | {('$%.0f' % (c * 10000)) if c is not None else '—'} |")
    md += ["", "- 넘김=violated+hold: 훼손·판단보류 문항을 2차로 / 넘김=violated: 훼손 문항만",
           "- '사람 훼손 중 2차로 넘김'은 2단계가 낼 수 있는 재현율의 상한",
           "- 비용은 일반 호출 기준 (Batch면 절반)"]
    path = reports_dir / "screen_sweep_dev.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return path
