"""명령줄 진입점.  python -m bridgecore.cli <명령>

  build     엑셀 3개 → data/processed/ (문항·평가·임시정답·데이터점검·그림)
  split     파일럿 개발용/확인용 분할 → data/processed/split_pilot.json (이미 있으면 거부)
  rules     규칙(숫자 검사·한글 비율) 결과 → data/processed/rules_pilot.csv
  prompt    문항 하나의 프롬프트를 화면에 출력 (API 호출 없음)
  run       엔진 실행 (기본 드라이런: 요청만 저장. --live 일 때만 API 호출)
  score     엔진 출력 채점 → reports/
  baseline  사람 간 기준선 → reports/
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


def _select_items(items, args):
    if args.items:
        want = set(args.items.split(","))
        return [i for i in items if i["item_id"] in want]
    if args.split:
        sp = read_json(PROC / "split_pilot.json")
        if args.split == "holdout" and not args.i_know_holdout:
            sys.exit("확인용 20문항은 마지막에 한 번만 씁니다 (PROJECT_NOTES 9절). "
                     "정말 실행하려면 --i-know-holdout 을 붙이세요.")
        want = set(sp[args.split])
        return [i for i in items if i["item_id"] in want]
    sys.exit("--split dev|holdout 또는 --items P001,P002 를 지정하세요.")


def cmd_run(args):
    from .runner import run
    items, _ = load_all()
    sel = _select_items(items, args)
    run(sel, model=args.model, version=args.version, run_name=args.name, live=args.live,
        out_root=ROOT / "runs", max_tokens=args.max_tokens)


def cmd_score(args):
    from .report import score_run
    items, ratings = load_all()
    score_run(Path(args.run_dir), items, ratings, REPORTS, split=args.split)


def cmd_baseline(args):
    from .report import baseline_report
    items, ratings = load_all()
    baseline_report(items, ratings, REPORTS)


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
    s.add_argument("--model", default="claude-haiku-5-5")
    s.add_argument("--version", default="v0"); s.add_argument("--name")
    s.add_argument("--max-tokens", type=int, default=4000)
    s.add_argument("--live", action="store_true", help="실제 API 호출 (없으면 드라이런)")
    s.add_argument("--i-know-holdout", action="store_true")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("score"); s.add_argument("run_dir")
    s.add_argument("--split", choices=["dev", "holdout", "all"], default="all")
    s.set_defaults(fn=cmd_score)
    sub.add_parser("baseline").set_defaults(fn=cmd_baseline)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
