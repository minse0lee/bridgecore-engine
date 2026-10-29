"""명령줄 진입점.  python -m bridgecore.cli <명령>

  build     엑셀 3개 → data/processed/ (문항·평가·임시정답·데이터점검·그림)
  split     파일럿 개발용/확인용 분할 → data/processed/split_pilot.json (이미 있으면 거부)
  rules     규칙(숫자 검사·한글 비율) 결과 → data/processed/rules_pilot.csv
  prompt    문항 하나의 프롬프트를 화면에 출력 (API 호출 없음)
  run       엔진 실행 (기본 드라이런. --live 일 때만 API 호출, --batch면 Batch 제출, --screen이면 선별 모드)
  collect   Batch 결과 수집 (2단계 2차 Batch면 수집 후 자동 병합)
  cascade   선별 실행에서 훼손·보류 문항만 2차 모델로 재판정 → 병합
  merge     선별·2차 실행 수동 병합
  compare   선별이 기준 실행(예: Sonnet 전체) 대비 놓치는 훼손 비교
  score     엔진 출력 채점 → reports/
  baseline  사람 간 기준선 → reports/
  poc       정답 없이 읽는 PoC 보고서(HTML)와 검토용 엑셀 → reports/
  poc-tally 검토가 끝난 PoC 엑셀 집계 (엔진 표시 정확도·놓친 오류)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from . import gold as G
from . import loader as L
from .constants import AXES
from .io_utils import fmt, read_json, read_jsonl, write_csv, write_json, write_jsonl

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SPLIT_SEED = 20261023
N_HOLDOUT = 20


def _rater_files():
    files = sorted(glob.glob(str(RAW / "*_R00[0-9]*.xlsx")))
    if not files:
        sys.exit(f"평가자 엑셀이 없습니다: {RAW}")
    return files


def load_all():
    items = read_jsonl(PROC / "items.jsonl")
    ratings = read_jsonl(PROC / "ratings.jsonl")
    return items, ratings


def cmd_build(args):
    files = _rater_files()
    items = L.load_items(files[0])
    for f in files[1:]:
        other = L.load_items(f)
        diff = [a["item_id"] for a, b in zip(items, other)
                if {k: a[k] for k in ("src", "tgt", "src_paragraph", "tgt_paragraph", "brief")}
                != {k: b[k] for k in ("src", "tgt", "src_paragraph", "tgt_paragraph", "brief")}]
        if diff or len(other) != len(items):
            print(f"⚠ 문항 시트가 파일마다 다릅니다: {Path(f).name} {diff[:5]}")
    figs = L.extract_figures(files[0], PROC / "figures")
    for it in items:
        if it["item_id"] in figs:
            it["figure"] = str(Path(figs[it["item_id"]]).relative_to(ROOT))
    write_jsonl(PROC / "items.jsonl", items)

    ratings, ops = [], []
    for f in files:
        rs = L.load_ratings(f)
        ratings += rs
        ops += L.load_ops(f)
        print(f"  {Path(f).name}: 평가자 {rs[0]['rater_id'] if rs else '?'} · {len(rs)}행")
    corr = L.apply_corrections(ratings, ROOT / "data" / "corrections.csv")
    for c in corr:
        print(f"  수정 {'반영' if c['applied'] else '미반영'}: {c['rater_id']} {c['item_id']} {c['axis']} "
              f"{c['field']} {c['old']!r}→{c['new']!r} ({c['source']}) {c['note']}")
    write_jsonl(PROC / "ratings.jsonl", ratings)
    write_csv(PROC / "ratings.csv", ratings,
              ["rater_id", "item_id", "axis", "judgment", "severity", "error_ids",
               "src_span", "tgt_span", "reason", "opinion", "sheet_row"])
    write_csv(PROC / "ops.csv", ops)

    issues = L.check_ratings(ratings, [it["item_id"] for it in items])
    write_csv(PROC / "data_issues.csv", issues,
              ["rater_id", "item_id", "axis", "sheet_row", "kind", "detail"])

    gold = G.build_gold(ratings)
    write_csv(PROC / "gold_pilot.csv", G.gold_rows(gold))
    summ = G.gold_summary(gold)
    write_json(PROC / "gold_summary.json", summ)

    print(f"\n문항 {len(items)}개 · 평가 {len(ratings)}행 · 그림 {len(figs)}개 ({', '.join(sorted(figs))})")
    print(f"데이터 점검: {len(issues)}건" + ("" if not issues else " → data/processed/data_issues.csv"))
    for i in issues[:10]:
        print(f"  - {i['rater_id']} {i['item_id']} {i['axis']} (시트 {i['sheet_row']}행): {i['kind']} {i['detail']}")
    print("\n축별 임시 정답 (다수결)")
    print(f"{'축':<9}{'전원일치':>7}{'3명훼손':>7}{'다수결훼손':>9}{'1명이상':>7}{'4범주불일치':>10}  심각도")
    for s in summ:
        print(f"{s['axis']:<9}{s['unanimous']:>7}{s['all_violated']:>7}{s['majority_violated']:>9}"
              f"{s['any_violated']:>7}{s['label4_undecided']:>10}  {s['severity_dist']}")


def cmd_split(args):
    from .split import stratified_split
    path = PROC / "split_pilot.json"
    if path.exists() and not args.force:
        sys.exit(f"이미 분할이 있습니다: {path}\n확인용 오염을 막기 위해 덮어쓰지 않습니다 (--force로만 가능).")
    items, ratings = load_all()
    gold = G.build_gold(ratings)
    rare = {it for (it, ax), v in gold.items() if ax != "Semantic" and v["violated"]}
    sp = stratified_split(items, N_HOLDOUT, SPLIT_SEED, rare)
    for name in ("dev", "holdout"):
        ids = set(sp[name])
        sp[f"{name}_gold_violated"] = {ax: sum(v["violated"] for (it, a), v in gold.items()
                                              if a == ax and it in ids) for ax in AXES}
    write_json(path, sp)
    print(f"개발용 {len(sp['dev'])} / 확인용 {len(sp['holdout'])} (시드 {sp['seed']}, 장르 층화 {sp['holdout_by_group']})")
    print("다수결 훼손 수  개발용:", sp["dev_gold_violated"])
    print("               확인용:", sp["holdout_gold_violated"])


def cmd_rules(args):
    from .rules import run_rules
    items, ratings = load_all()
    gold = G.build_gold(ratings)
    rows = []
    for it in items:
        r = run_rules(it)
        rows.append({"item_id": it["item_id"], "genre": it["genre"],
                     "number_mismatch": int(bool(r["number_check"]["mismatch"])),
                     "src_numbers": " ".join(map(str, r["number_check"]["src"])),
                     "tgt_numbers": " ".join(map(str, r["number_check"]["tgt"])),
                     "hangul_ratio": round(r["hangul_ratio"], 3),
                     "target_language_missing": int(r["target_language_missing"]),
                     "gold_semantic_violated": int(gold[(it["item_id"], "Semantic")]["violated"])})
    write_csv(PROC / "rules_pilot.csv", rows)
    mm = [r for r in rows if r["number_mismatch"]]
    print(f"숫자 불일치 {len(mm)}문항: " + ", ".join(
        f"{r['item_id']}(원문 {r['src_numbers'] or '-'} / 번역 {r['tgt_numbers'] or '-'})" for r in mm))
    print(f"목표언어 미생성 의심: {[r['item_id'] for r in rows if r['target_language_missing']] or '없음'}")
    print(f"한글 비율 최저 5개: {sorted(((r['hangul_ratio'], r['item_id']) for r in rows))[:5]}")


def cmd_prompt(args):
    from .prompt import build_messages
    items, _ = load_all()
    it = next((i for i in items if i["item_id"] == args.item), None)
    if not it:
        sys.exit(f"문항 없음: {args.item}")
    system, messages = build_messages(it, version=args.version)
    print("=== SYSTEM ===")
    for blk in system:
        print(blk["text"])
    print("=== USER ===")
    for c in messages[0]["content"]:
        print(c["text"] if c["type"] == "text" else f"[이미지 {c['source']['media_type']}]")


def load_items_file(path: str) -> list[dict]:
    """파일럿 외 문항(본검수·책)을 jsonl로 받는다. 한 줄 = 문항 하나.
    필수: item_id, src, 그리고 tgt(번역 문장) 또는 tgt_paragraph(문단 전체 번역)
    선택: doc_info, brief, src_paragraph, figure(engine/ 기준 경로)"""
    rows = read_jsonl(path)
    # tgt(번역 문장)가 없으면 tgt_paragraph(문단 전체 번역)는 있어야 한다 → 엔진이 대응 구간을 직접 찾음
    bad = [i for i, r in enumerate(rows, 1) if not all(r.get(k) for k in ("item_id", "src"))
           or not ((r.get("tgt") or "").strip() or (r.get("tgt_paragraph") or "").strip())]
    if bad:
        sys.exit(f"{path}: item_id·src, 그리고 tgt 또는 tgt_paragraph가 없는 줄 {bad[:10]}")
    ids = [r["item_id"] for r in rows]
    if len(ids) != len(set(ids)):
        sys.exit(f"{path}: 중복된 item_id가 있습니다.")
    return rows


def _items_source(args) -> str:
    return str(Path(args.items_file).resolve()) if getattr(args, "items_file", None) else "pilot"


def _items_from_source(source: str) -> list[dict]:
    return load_items_file(source) if source != "pilot" else load_all()[0]


def _select_items(items, args):
    if args.items:
        want = set(args.items.split(","))
        sel = [i for i in items if i["item_id"] in want]
        if missing := want - {i["item_id"] for i in sel}:
            sys.exit(f"없는 문항: {sorted(missing)}")
        return sel
    if args.split:
        if args.items_file:
            sys.exit("--split은 파일럿 전용입니다. --items-file과 함께 쓰지 마세요.")
        sp = read_json(PROC / "split_pilot.json")
        if args.split == "holdout" and not args.i_know_holdout:
            sys.exit("확인용 20문항은 마지막에 한 번만 씁니다 (PROJECT_NOTES 9절). "
                     "정말 실행하려면 --i-know-holdout 을 붙이세요.")
        want = set(sp[args.split])
        return [i for i in items if i["item_id"] in want]
    if args.items_file:
        return items
    sys.exit("--split dev|holdout, --items P001,P002, 또는 --items-file 을 지정하세요.")


def _runner_kwargs(args, extra_meta=None):
    return dict(model=args.model, version=args.version, run_name=args.name, live=args.live,
                out_root=ROOT / "runs", max_tokens=args.max_tokens, screen=args.screen,
                extra_meta={"items_source": _items_source(args), **(extra_meta or {})},
                **_gen_kwargs(args))


def _gen_kwargs(args):
    return {"effort": None if args.effort == "default" else args.effort, "thinking": not args.no_thinking}


def _sync_kwargs(args):
    return {"workers": args.workers}


def cmd_run(args):
    from .runner import run, submit_batch
    items = _items_from_source(_items_source(args))
    sel = _select_items(items, args)
    if args.batch:
        submit_batch(sel, **_runner_kwargs(args))
    else:
        run(sel, **_runner_kwargs(args), **_sync_kwargs(args))


def cmd_collect(args):
    from .runner import collect_batch, merge_cascade
    out = Path(args.run_dir)
    if collect_batch(out):
        meta = read_json(out / "meta.json")
        if meta.get("cascade_of"):
            merge_cascade(Path(meta["cascade_of"]), out, Path(meta["cascade_out"]),
                          tuple(meta["escalate_on"]))


def cmd_cascade(args):
    """선별 실행 결과에서 훼손·보류 문항만 골라 2차 모델로 재판정하고 병합한다."""
    from .io_utils import write_csv
    from .runner import escalation_list, merge_cascade, run, submit_batch
    screen = Path(args.screen_run)
    m1 = read_json(screen / "meta.json")
    if not m1.get("screen"):
        print("⚠ 선별 모드(--screen)로 돌린 실행이 아닙니다. 그대로 진행합니다.")
    on = tuple(args.escalate_on.split(","))
    esc = escalation_list(screen, on)
    write_csv(screen / "escalations.csv", esc)
    ids = {r["item_id"] for r in esc if r["escalate"]}
    items = [i for i in _items_from_source(m1.get("items_source", "pilot")) if i["item_id"] in ids]
    print(f"재판정 대상 {len(items)}/{len(esc)}문항 (기준: {', '.join(on)}) → {screen / 'escalations.csv'}")
    args.name = args.name or f"{screen.name}__stage2_{args.model}"
    cascade_out = ROOT / "runs" / f"{screen.name}__cascade"
    extra = {"items_source": m1.get("items_source", "pilot"), "cascade_of": str(screen),
             "cascade_out": str(cascade_out), "escalate_on": list(on)}
    kw = dict(model=args.model, version=args.version, run_name=args.name, live=args.live,
              out_root=ROOT / "runs", max_tokens=args.max_tokens, screen=False, extra_meta=extra,
              **_gen_kwargs(args))
    if args.batch:
        submit_batch(items, **kw)
        if args.live:
            print(f"2차 Batch가 끝나면 collect 하세요. 수집 후 자동으로 병합합니다 → {cascade_out}")
        return
    stage2 = run(items, **kw, **_sync_kwargs(args))
    if args.live:
        merge_cascade(screen, stage2, cascade_out, on)


def cmd_merge(args):
    from .runner import merge_cascade
    screen, stage2 = Path(args.screen_run), Path(args.stage2_run)
    out = Path(args.out) if args.out else ROOT / "runs" / f"{screen.name}__cascade"
    merge_cascade(screen, stage2, out, tuple(args.escalate_on.split(",")))


def cmd_compare(args):
    from .compare import compare_runs
    gold = G.build_gold(load_all()[1]) if (PROC / "ratings.jsonl").exists() else None
    compare_runs(Path(args.screen_run), Path(args.ref_run), gold, REPORTS,
                 tuple(args.escalate_on.split(",")))


def cmd_agree(args):
    from .experiments import run_agreement
    gold = G.build_gold(load_all()[1])
    run_agreement(Path(args.run_a), Path(args.run_b), gold, REPORTS)


def cmd_sweep(args):
    from .experiments import screen_sweep
    gold = G.build_gold(load_all()[1])
    screen_sweep([Path(x) for x in args.screens], Path(args.stage2_ref), gold,
                 read_json(ROOT / "prices.json"), REPORTS, baselines=[Path(x) for x in args.baseline or []])


def cmd_score(args):
    from .report import score_run
    items, ratings = load_all()
    score_run(Path(args.run_dir), items, ratings, REPORTS, split=args.split)


def cmd_poc(args):
    from .poc_report import make_poc_report
    run_dir = Path(args.run_dir)
    meta = read_json(run_dir / "meta.json")
    source = str(Path(args.items_file).resolve()) if args.items_file else meta.get("items_source", "pilot")
    items = _items_from_source(source)
    if source == "pilot" and not args.allow_holdout:
        held = set(read_json(PROC / "split_pilot.json")["holdout"]) & set(meta.get("items") or [])
        if held:
            sys.exit(f"이 실행에 확인용 문항 {len(held)}개가 들어 있습니다. 시연·배포용 보고서에 확인용 문항을 "
                     "넣지 않으려면 개발용 실행을 쓰세요. 그래도 만들려면 --allow-holdout.")
    make_poc_report(run_dir, items, REPORTS, title=args.title, note=args.note, name=args.name)


def cmd_poc_tally(args):
    from .poc_report import tally_review
    res = tally_review(Path(args.xlsx))
    out = Path(args.xlsx).with_suffix(".tally.json")
    write_json(out, res)
    t = res["flags_total"]
    fmt_p = lambda x: "-" if x is None else f"{x:.2f}"
    print(f"훼손 표시 {t['맞음'] + t['틀림'] + t['애매'] + t['미입력']}칸: 맞음 {t['맞음']} · 틀림 {t['틀림']} · "
          f"애매 {t['애매']} · 미입력 {t['미입력']}  → 정밀도 {fmt_p(t['precision_lenient'])} "
          f"(애매를 틀림으로 치면 {fmt_p(t['precision_strict'])})")
    print(f"놓친 오류가 있는 문장 {res['n_items_with_missed']}/{res['n_items']} · 축별 {res['missed_by_axis']}")
    print(f"→ {out}")


def cmd_baseline(args):
    from .report import baseline_report
    items, ratings = load_all()
    baseline_report(items, ratings, REPORTS)


def _model_args(s, default_model):
    s.add_argument("--model", default=default_model)
    s.add_argument("--version", default="v0"); s.add_argument("--name")
    s.add_argument("--max-tokens", type=int, default=16000, help="사고 토큰 포함 상한")
    s.add_argument("--effort", choices=["low", "medium", "high", "default"], default="medium",
                   help="사고 깊이 (default = 모델 기본값)")
    s.add_argument("--no-thinking", action="store_true", help="적응형 사고 끄기")
    s.add_argument("--workers", type=int, default=1, help="동기 실행 시 동시 호출 수")
    s.add_argument("--live", action="store_true", help="실제 API 호출 (없으면 드라이런)")
    s.add_argument("--batch", action="store_true", help="Message Batches로 제출 (결과는 collect)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="bridgecore")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    s = sub.add_parser("split"); s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_split)
    sub.add_parser("rules").set_defaults(fn=cmd_rules)
    s = sub.add_parser("prompt"); s.add_argument("item"); s.add_argument("--version", default="v0")
    s.set_defaults(fn=cmd_prompt)

    s = sub.add_parser("run")
    s.add_argument("--split", choices=["dev", "holdout"]); s.add_argument("--items")
    s.add_argument("--items-file", help="파일럿 외 문항 jsonl (본검수·책)")
    s.add_argument("--screen", action="store_true", help="1차 선별 모드 (넉넉하게 표시)")
    s.add_argument("--i-know-holdout", action="store_true")
    _model_args(s, "claude-haiku-5-5")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("collect", help="Batch 결과 수집"); s.add_argument("run_dir")
    s.set_defaults(fn=cmd_collect)

    s = sub.add_parser("cascade", help="선별 실행 → 2차 재판정 → 병합")
    s.add_argument("screen_run")
    s.add_argument("--escalate-on", default="violated,hold", help="violated,hold 중 선택")
    _model_args(s, "claude-sonnet-5-5")
    s.set_defaults(fn=cmd_cascade)

    s = sub.add_parser("merge", help="선별 실행과 2차 실행을 수동 병합")
    s.add_argument("screen_run"); s.add_argument("stage2_run"); s.add_argument("--out")
    s.add_argument("--escalate-on", default="violated,hold")
    s.set_defaults(fn=cmd_merge)

    s = sub.add_parser("compare", help="선별이 기준 실행 대비 얼마나 놓치는지")
    s.add_argument("screen_run"); s.add_argument("ref_run")
    s.add_argument("--escalate-on", default="violated,hold")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("agree", help="같은 설정 두 실행의 판정 일치도")
    s.add_argument("run_a"); s.add_argument("run_b"); s.set_defaults(fn=cmd_agree)

    s = sub.add_parser("sweep", help="선별 설정별 2단계 결과·비용 (2차는 기준 실행 재사용)")
    s.add_argument("stage2_ref"); s.add_argument("screens", nargs="+")
    s.add_argument("--baseline", nargs="*")
    s.set_defaults(fn=cmd_sweep)

    s = sub.add_parser("score"); s.add_argument("run_dir")
    s.add_argument("--split", choices=["dev", "holdout", "all"], default="all")
    s.set_defaults(fn=cmd_score)
    sub.add_parser("baseline").set_defaults(fn=cmd_baseline)
    s = sub.add_parser("poc"); s.add_argument("run_dir")
    s.add_argument("--items-file", help="실행 때 쓴 문항 jsonl (기본: meta.json의 items_source)")
    s.add_argument("--title"); s.add_argument("--note", help="보고서 맨 위 안내에 넣을 한 줄")
    s.add_argument("--name", help="출력 파일 이름 앞부분 (기본: 실행 폴더 이름)")
    s.add_argument("--allow-holdout", action="store_true")
    s.set_defaults(fn=cmd_poc)
    s = sub.add_parser("poc-tally"); s.add_argument("xlsx"); s.set_defaults(fn=cmd_poc_tally)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
