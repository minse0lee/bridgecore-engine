"""채점 리포트 (마크다운 + CSV)."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from . import gold as G
from .constants import AXES
from .io_utils import fmt, read_json, read_jsonl, write_csv, write_json
from .runner import latest_records
from .scoring import human_baseline, predictions_from_outputs, score

ROOT = Path(__file__).resolve().parent.parent


def _split_ids(split: str):
    if split == "all":
        return None
    return read_json(ROOT / "data" / "processed" / "split_pilot.json")[split]


def _cost(calls: list[dict], n_items: int) -> dict | None:
    """calls: [{model, usage, via_batch}] — 2단계면 1차·2차 호출을 모두 넣는다.
    prices.json에 해당 모델 단가가 하나라도 비어 있으면 None."""
    prices_path = ROOT / "prices.json"
    if not prices_path.exists():
        return None
    prices = read_json(prices_path)
    batch_mult = prices.get("batch_multiplier")
    keys = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    by_model: dict[str, dict] = {}
    usd = 0.0
    for c in calls:
        u = c.get("usage")
        if not u:
            continue
        # 응답의 모델명에 날짜 접미사가 붙을 수 있어 접두사로도 찾는다
        p = prices.get(c["model"]) or next(
            (v for k, v in prices.items() if isinstance(v, dict) and c["model"].startswith(k)), {})
        if any(p.get(k) is None for k in ("input", "output", "cache_write", "cache_read")):
            return None
        if c.get("via_batch") and batch_mult is None:
            return None
        mult = batch_mult if c.get("via_batch") else 1.0
        usd += mult * (u["input_tokens"] * p["input"] + u["output_tokens"] * p["output"]
                       + u["cache_creation_input_tokens"] * p["cache_write"]
                       + u["cache_read_input_tokens"] * p["cache_read"]) / 1e6
        m = by_model.setdefault(c["model"], {"calls": 0, **{k: 0 for k in keys}})
        m["calls"] += 1
        for k in keys:
            m[k] += u[k]
    return {"by_model": by_model, "usd": usd, "n_items": n_items,
            "usd_per_item": usd / n_items if n_items else None}


def _calls_for_run(run_dir: Path, meta: dict, recs: dict, item_ids: set) -> list[dict]:
    if meta.get("type") == "cascade":
        rows = [r for iid, r in recs.items() if iid in item_ids]
    else:
        # 재시도 호출도 비용이 들었으므로 마지막 결과만이 아니라 모든 호출 기록을 센다
        rows = [r for r in read_jsonl(run_dir / "outputs.jsonl") if r["item_id"] in item_ids]
    calls = [{"model": r.get("model") or meta["model"], "usage": r.get("usage"),
              "via_batch": r.get("via_batch", False)} for r in rows]
    s1 = run_dir / "stage1_usage.json"
    if meta.get("type") == "cascade" and s1.exists():
        # 2차로 넘어간 문항은 1차 호출 비용도 들었다 (1차 결과를 그대로 쓴 문항은 위에 이미 포함)
        for iid, u in read_json(s1).items():
            if iid in item_ids and recs.get(iid, {}).get("stage") == 2:
                calls.append({"model": u["model"], "usage": u["usage"], "via_batch": u.get("via_batch", False)})
    return calls


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
    recs = latest_records(run_dir)
    ids = _split_ids(split)
    gold = G.build_gold(ratings)
    gold_items = {k[0] for k in gold}
    run_items = set(meta["items"]) if ids is None else set(meta["items"]) & set(ids)
    no_gold = sorted(run_items - gold_items)
    run_items &= gold_items  # 사람 정답이 있는 문항만 채점 (본검수·책 문항은 정답이 생긴 뒤)
    if not run_items:
        raise SystemExit(f"채점할 문항이 없습니다 (정답 없는 문항 {len(no_gold)}개).")
    outputs = {iid: r.get("output") for iid, r in recs.items() if iid in run_items}
    preds, checks = predictions_from_outputs(outputs)
    res = score(gold, preds, sorted(run_items))
    bl = human_baseline(ratings, sorted(run_items))
    cost = _cost(_calls_for_run(run_dir, meta, recs, run_items), len(run_items))

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
        *([f"- 2단계: 재판정 {sum(1 for i in run_items if recs.get(i, {}).get('escalate_reason'))}/{len(run_items)}문항"
           f" · 2차 결과 사용 {sum(1 for i in run_items if recs.get(i, {}).get('stage') == 2)}"] if meta.get("type") == "cascade" else []),
        *([f"- 정답 없어 채점에서 뺀 문항: {len(no_gold)}개"] if no_gold else []),
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
               f"- 총 ${cost['usd']:.4f} · 문항당 ${cost['usd_per_item']:.5f} (책 1만 문장이면 약 ${cost['usd_per_item'] * 10000:.0f})",
               *[f"- {m}: 호출 {v['calls']}회 · 입력 {v['input_tokens']:,} · 캐시 쓰기 "
                 f"{v['cache_creation_input_tokens']:,} · 캐시 읽기 {v['cache_read_input_tokens']:,} · "
                 f"출력 {v['output_tokens']:,}" for m, v in cost["by_model"].items()], ""]
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
