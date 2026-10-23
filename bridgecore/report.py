"""채점 리포트 (마크다운 + CSV)."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from . import gold as G
from .constants import AXES
from .io_utils import fmt, read_json, read_jsonl, write_csv, write_json
from .scoring import human_baseline, predictions_from_outputs, score

ROOT = Path(__file__).resolve().parent.parent


def _split_ids(split: str):
    if split == "all":
        return None
    return read_json(ROOT / "data" / "processed" / "split_pilot.json")[split]


def _cost(recs: list[dict], model: str) -> dict | None:
    prices_path = ROOT / "prices.json"
    if not prices_path.exists():
        return None
    p = read_json(prices_path).get(model) or {}
    if any(p.get(k) is None for k in ("input", "output", "cache_write", "cache_read")):
        return None
    tot = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0,
           "cache_read_input_tokens": 0}
    n = 0
    for r in recs:
        if r.get("usage"):
            n += 1
            for k in tot:
                tot[k] += r["usage"][k]
    usd = (tot["input_tokens"] * p["input"] + tot["output_tokens"] * p["output"]
           + tot["cache_creation_input_tokens"] * p["cache_write"]
           + tot["cache_read_input_tokens"] * p["cache_read"]) / 1e6
    return {"tokens": tot, "n_calls": n, "usd": usd, "usd_per_item": usd / n if n else None}


def _axis_table(per_axis: dict) -> list[str]:
    lines = ["| 축 | 정답 훼손 | P | R | F1 | 재현율 95% CI | 보류 | F1(보류=훼손) | 4범주 일치 | 심각도 일치 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for ax in AXES + ["ALL"]:
        a = per_axis[ax]
        s, l = a["strict"], a["lenient"]
        ci = a["recall_ci95"]
        lines.append(
            f"| {ax} | {a['gold_violated']} | {fmt(s['precision'])} | {fmt(s['recall'])} | "
            f"{fmt(s['f1'])} | {('%.2f–%.2f' % ci) if ci else '—'} ({a['recall_count']}) | "
            f"{a['hold']} | {fmt(l['f1'])} | {fmt(a['label4_acc'])} (n={a['label4_n']}) | "
            f"{fmt(a['severity_exact'])} (n={a['severity_n']}) |")
    return lines


def _baseline_table(bl: dict) -> list[str]:
    pairs = list(bl["ALL"]["pairwise_f1"])
    lines = ["| 축 | 평가자 쌍 평균 F1 | " + " | ".join(pairs) + " | Fleiss κ(훼손 여부) |",
             "|---|---|" + "---|" * len(pairs) + "---|"]
    for ax in AXES + ["ALL"]:
        b = bl[ax]
        lines.append(f"| {ax} | {fmt(b['mean_pairwise_f1'])} | "
                     + " | ".join(fmt(b['pairwise_f1'][p]) for p in pairs)
                     + f" | {fmt(b['fleiss_kappa'])} |")
    return lines


def score_run(run_dir: Path, items, ratings, reports_dir: Path, split: str = "all") -> Path:
    meta = read_json(run_dir / "meta.json")
    recs = read_jsonl(run_dir / "outputs.jsonl")
    ids = _split_ids(split)
    run_items = set(meta["items"]) if ids is None else set(meta["items"]) & set(ids)
    outputs = {r["item_id"]: r["output"] for r in recs if r["item_id"] in run_items}
    gold = G.build_gold(ratings)
    preds, checks = predictions_from_outputs(outputs)
    res = score(gold, preds, sorted(run_items))
    bl = human_baseline(ratings, sorted(run_items))
    cost = _cost([r for r in recs if r["item_id"] in run_items], meta["model"])

    by_id = {it["item_id"]: it for it in items}
    for c in res["cases"]:
        it = by_id[c["item_id"]]
        out = outputs.get(c["item_id"]) or {}
        ax = (out.get("axes") or {}).get(c["axis"]) or {}
        c["engine_reason"] = ax.get("reason")
        c["src"], c["tgt"] = it["src"], it["tgt"]
        c["human_reasons"] = " || ".join(
            f"{r['rater_id']}({r['judgment']}): {r['reason']}" for r in ratings
            if r["item_id"] == c["item_id"] and r["axis"] == c["axis"] and r["reason"])

    reports_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{run_dir.name}_{split}"
    write_csv(reports_dir / f"{tag}_cases.csv", res["cases"])
    write_json(reports_dir / f"{tag}_scores.json", {"meta": meta, "split": split, "scores": res["per_axis"],
                                                    "baseline": bl, "cost": cost, "format_checks": checks})
    fmt_err = [i for i, c in checks.items() if c["errors"]]
    n_warn = sum(len(c["warnings"]) for c in checks.values())
    md = [
        f"# 채점 리포트 — {run_dir.name} ({split})",
        "",
        f"- 채점일: {dt.date.today().isoformat()} · 모델 `{meta['model']}` · 프롬프트 `{meta['prompt_version']}` "
        f"(해시 `{meta['prompt_fingerprint']}`) · 문항 {len(run_items)}개",
        f"- 정답: 3명 다수결 임시 정답 (채점 규칙 v1, `bridgecore/gold.py`). 파일럿은 합의 전이라 골든셋이 아님",
        f"- 형식 오류 문항: {len(fmt_err)}개 {fmt_err if fmt_err else ''} (6축 비훼손으로 처리) · 경고 {n_warn}건",
        "",
        "## 축별 성능 (훼손 대 나머지, 보류=비훼손이 주 지표)",
        "",
        *_axis_table(res["per_axis"]),
        "",
        "## 같은 문항에서 사람 간 기준선",
        "",
        "한 평가자를 정답으로 놓고 다른 평가자를 채점한 F1. 엔진 F1이 이 수준이면 '사람만큼'으로 해석할 수 있음.",
        "",
        *_baseline_table(bl),
        "",
    ]
    if cost:
        md += ["## 비용 (prices.json 단가 기준)", "",
               f"- 호출 {cost['n_calls']}회 · 총 ${cost['usd']:.4f} · 문항당 ${cost['usd_per_item']:.5f}",
               f"- 토큰: {cost['tokens']}", ""]
    else:
        md += ["## 비용", "", "- `prices.json`에 이 모델 단가가 비어 있어 계산하지 않음. 토큰 합계는 scores.json 참고.", ""]
    md += [f"## 틀린 칸 목록: `{tag}_cases.csv` ({len(res['cases'])}건 — FP·FN·보류, 엔진 이유와 사람 이유 나란히)"]
    path = reports_dir / f"{tag}.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:8 + len(AXES) + 4]))
    print(f"\n리포트: {path}")
    return path


def baseline_report(items, ratings, reports_dir: Path) -> Path:
    gold = G.build_gold(ratings)
    summ = G.gold_summary(gold)
    sp_path = ROOT / "data" / "processed" / "split_pilot.json"
    md = ["# 파일럿 사람 판정 기준선", "",
          f"- 생성: {dt.date.today().isoformat()} · 평가자 {sorted({r['rater_id'] for r in ratings})} · 문항 {len(items)}개", "",
          "## 임시 정답(다수결) 요약", "",
          "| 축 | 3명 전원 일치 | 3명 모두 훼손 | 다수결 훼손 | 1명 이상 훼손 | 4범주 불일치 | 정답 심각도 |",
          "|---|---|---|---|---|---|---|"]
    for s in summ:
        md.append(f"| {s['axis']} | {s['unanimous']} | {s['all_violated']} | {s['majority_violated']} | "
                  f"{s['any_violated']} | {s['label4_undecided']} | {s['severity_dist']} |")
    for name, ids in [("전체 60문항", None)] + (
            [(f"{k} ({len(v)}문항)", v) for k, v in
             ((k, read_json(sp_path)[k]) for k in ("dev", "holdout"))] if sp_path.exists() else []):
        md += ["", f"## 사람 간 기준선 — {name}", "", *_baseline_table(human_baseline(ratings, ids))]
    path = reports_dir / "baseline_pilot.md"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return path
