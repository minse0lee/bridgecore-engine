"""골든셋 300문항 개발 100 / 시험 200 분할 (라벨 보기 전에 고정).
- 같은 문서(문서 ID)의 문항은 같은 쪽으로 (문서 단위 묶음)
- 선정 경로 × 장르 8칸의 개발 비율이 1/3에 가깝도록: 무작위 배정 200,000회 중 개발이 정확히 100문항이고
  칸별 |개발 수 − 목표| 합이 가장 작은 배정 (같으면 먼저 나온 것). seed 20261113
python tools/golden_split.py MAPPING.xlsx FORM.xlsx OUT_PREFIX"""
import collections, csv, hashlib, json, random, sys
from openpyxl import load_workbook

mp, form, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [r for r in load_workbook(mp)["문항별 경로"].iter_rows(min_row=5, values_only=True) if r[0]]
cells = collections.Counter((r[2], r[3]) for r in rows)
target = {c: n / 3 for c, n in cells.items()}
docs = collections.defaultdict(list)
for r in rows:
    docs[r[4]].append(r)
names = sorted(docs)
rng = random.Random(20261113)
best = None
for it in range(200_000):
    dev = {d for d in names if rng.random() < 1 / 3}
    n = sum(len(docs[d]) for d in dev)
    if n != 100:
        continue
    cnt = collections.Counter((r[2], r[3]) for d in dev for r in docs[d])
    score = sum(abs(cnt[c] - target[c]) for c in cells)
    if best is None or score < best[0]:
        best = (score, sorted(dev), it)
score, dev, it = best
dev = set(dev)
with open(out + ".csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["문항ID", "묶음", "선정 경로", "장르", "문서 ID", "분할"])
    for r in rows:
        w.writerow(list(r[:5]) + ["개발" if r[4] in dev else "시험"])
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
tab = {f"{c[0]}|{c[1]}": {"전체": cells[c], "개발": sum(1 for r in rows if (r[2], r[3]) == c and r[4] in dev)} for c in sorted(cells)}
meta = {"date": "2026-11-13", "seed": 20261113, "iterations": 200_000, "chosen_iteration": it, "score": round(score, 3),
        "n_dev": sum(1 for r in rows if r[4] in dev), "n_test": sum(1 for r in rows if r[4] not in dev),
        "docs_dev": len(dev), "docs_test": len(names) - len(dev), "cells": tab,
        "mapping_sha256": sha(mp), "form_sha256": sha(form),
        "rule": "라벨 확인 전 고정. 개발 100으로만 엔진 수정·선택, 시험 200은 엔진 고정 후 최종 평가에만 사용. 무작위·LLM 분석 부분집합 별도 보고"}
json.dump(meta, open(out + ".json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: meta[k] for k in ("n_dev", "n_test", "docs_dev", "docs_test", "score", "chosen_iteration")}, ensure_ascii=False))
for k, v in tab.items():
    print(f"  {k}: 전체 {v['전체']} → 개발 {v['개발']} / 시험 {v['전체'] - v['개발']}")
