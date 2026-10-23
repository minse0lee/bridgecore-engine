"""개발용/확인용 분할 (PROJECT_NOTES 3-6).

층화 무작위 분할. 층 = 장르 × '드문 축 훼손 문항 여부'
(Semantic 외 축에서 다수결 훼손이 하나라도 있는 문항).
장르만으로 나누면 시드 20261023에서 Register·Cultural 다수결 훼손이 모두 확인용에 몰려
개발용에서 그 축을 전혀 볼 수 없었다 (2026-10-23, 엔진 실행 전 1회 재분할).
라벨 층화는 '어느 문항이 어느 쪽인지'만 정하고 개별 정답을 프롬프트 개발에 노출하지 않는다.
시드·층 기준은 한 번 정하면 바꾸지 않는다.
"""
from __future__ import annotations

import random
from collections import defaultdict


def stratified_split(items: list[dict], n_holdout: int, seed: int,
                     rare_items: set[str] | None = None) -> dict[str, list[str]]:
    rare_items = rare_items or set()
    groups = defaultdict(list)
    for it in items:
        g = f"{it.get('genre') or '기타'}|{'드문축훼손' if it['item_id'] in rare_items else '그외'}"
        groups[g].append(it["item_id"])
    total = len(items)
    # 장르별 확인용 개수: 비율대로 배정, 남는 자리는 소수점이 큰 장르부터 (최대잉여법)
    quota = {g: n_holdout * len(ids) / total for g, ids in groups.items()}
    alloc = {g: int(q) for g, q in quota.items()}
    rest = n_holdout - sum(alloc.values())
    for g in sorted(quota, key=lambda g: (quota[g] - alloc[g], g), reverse=True)[:rest]:
        alloc[g] += 1

    rng = random.Random(seed)
    holdout = []
    for g in sorted(groups):
        ids = sorted(groups[g])
        holdout += rng.sample(ids, alloc[g])
    holdout = sorted(holdout)
    dev = sorted(it["item_id"] for it in items if it["item_id"] not in set(holdout))
    return {"seed": seed, "stratify_by": "genre x rare-axis majority violation",
            "dev": dev, "holdout": holdout,
            "holdout_by_group": {g: alloc[g] for g in sorted(alloc)}}
