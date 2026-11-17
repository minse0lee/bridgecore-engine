"""개츠비 1·3장 번역 4종 비교(학습 전 MADLAD · 학습 후 v1 · Claude · Gemma 4 31B) — 6축 v1.1 결과 기준.

    python tools/gatsby_compare4.py [--gemma gemma_bf16]

- 같은 문장(item_id)끼리 비교. Gemma가 번역을 못 낸 문장(형식 오류로 빠짐)은 '생성 실패'로 따로 세고,
  ① 공통 문장만 비교 ② 빠진 문장을 critical로 친 비교 두 가지를 같이 보인다
- 판정은 모두 같은 Claude(Sonnet 5.5) 엔진 → Claude 번역은 자기 평가 편향 가능성(사람 확인 필요)
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, ".")
from bridgecore.constants import AXES, VIOLATED  # noqa: E402
from bridgecore.vote import load_group  # noqa: E402

D = Path("data/book/gatsby/eval")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--gemma", nargs="+", default=["gemma_bf16"]); a = p.parse_args()
    SYS = [("base", "MADLAD 학습 전"), ("lora", "MADLAD 학습 후 v1"), ("claude", "Claude Sonnet 5.5"),
           ] + [(t, "Gemma 31B " + t.removeprefix("gemma_")) for t in a.gemma]
    ids = [json.loads(l)["item_id"] for l in open(D / "ch1_3__claude.jsonl")]
    have = {t: {json.loads(l)["item_id"] for l in open(D / f"ch1_3__{t}.jsonl")} for t, _ in SYS}
    G = {t: load_group([Path(f"runs/gatsby_ch13_{t}")]) for t, _ in SYS}

    def state(t, i):
        """None=판정 없음(엔진 오류), 'fail'=번역 없음, 'collapse'=생성 붕괴, 아니면 최고 심각도('' 이면 훼손 없음)"""
        if i not in have[t]:
            return "fail"
        r = G[t].get(i)
        if not r or not r.get("output") or r.get("errors"):
            return None
        o = r["output"]
        if o.get("evaluable") is False:
            return "collapse"
        s = [o["axes"][x].get("severity") for x in AXES if o["axes"].get(x, {}).get("judgment") == VIOLATED]
        return "critical" if "critical" in s else "major" if "major" in s else "minor" if s else ""

    st = {t: {i: state(t, i) for i in ids} for t, _ in SYS}
    common = [i for i in ids if all(st[t][i] not in (None, "fail") for t, _ in SYS)]
    anyv = lambda s: s not in ("", None)
    major = lambda s: s in ("major", "critical", "collapse", "fail")
    print(f"전체 {len(ids)}문장 · 모두 판정 있는 공통 문장 {len(common)}")
    print("번역 | 번역 없음 | 엔진 오류 | 생성 붕괴 | [공통] 훼손 1개 이상 | [공통] major 이상 | [전체, 번역 없음=critical] major 이상")
    for t, lab in SYS:
        vals = st[t]
        full = [i for i in ids if vals[i] is not None]
        print(f"{lab} | {sum(v == 'fail' for v in vals.values())} | {sum(v is None for v in vals.values())} | "
              f"{sum(v == 'collapse' for v in vals.values())} | {sum(anyv(vals[i]) for i in common) / len(common):.1%} | "
              f"{sum(major(vals[i]) for i in common) / len(common):.1%} | {sum(major(vals[i]) for i in full) / len(full):.1%}")

    def boot(f, x, y, n=5000):
        d = [int(f(st[y][i])) - int(f(st[x][i])) for i in common]
        rng = random.Random(20261117)
        bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
        return sum(d) / len(d), bs[int(n * .025)], bs[int(n * .975) - 1]

    pct = lambda v: f"{v * 100:+.1f}%p"
    pairs = [(x, g) for g in a.gemma for x in ("claude", "lora")]
    pairs += [(a.gemma[0], g) for g in a.gemma[1:]]
    for x, y in pairs:
        m, v = boot(major, x, y), boot(anyv, x, y)
        lx, ly = dict(SYS)[x], dict(SYS)[y]
        print(f"· {ly} − {lx} (공통 문장): major 이상 {pct(m[0])} (95% {pct(m[1])}~{pct(m[2])}), "
              f"훼손 1개 이상 {pct(v[0])} (95% {pct(v[1])}~{pct(v[2])})")
    # 축별 훼손 문장 수 (공통 문장)
    print("축별 훼손 문장 수(공통):")
    for t, lab in SYS:
        c = {x: 0 for x in AXES}
        for i in common:
            o = G[t][i]["output"]
            if o.get("evaluable") is False:
                continue
            for x in AXES:
                c[x] += o["axes"].get(x, {}).get("judgment") == VIOLATED
        print(f"  {lab}: " + ", ".join(f"{x} {c[x]}" for x in AXES))


if __name__ == "__main__":
    main()
