"""여러 번 실행한 판정을 칸(문항×축)마다 다수결로 합치기, 그리고 실행 간 흔들림 측정.

같은 프롬프트·모델이라도 temperature를 고정할 수 없어(Sonnet 5.5) 실행마다 판정이 조금씩 다르다.
3번 판정해 칸마다 다수결로 정하면 흔들림이 줄어드는지, 비용은 얼마나 드는지를 본다.

다수결 규칙 (칸마다)
  - 훼손 표가 과반이면 훼손. 심각도는 훼손 표들의 중앙값(짝수면 높은 쪽)
  - 아니면 훼손이 아닌 판정(보존·N/A·판단보류) 중 가장 많은 것. 동률이면 보존 > N/A > 판단보류 순
  - 근거·오류 구간은 다수 판정과 같은 판정을 낸 실행에서 가져온다. 문항마다 다수결과 가장 많이 일치하는
    실행을 대표로 삼고, 대표와 다른 칸만 다른 실행에서 가져와 오류 ID를 새로 붙인다
적응형 3회: 두 번 돌려 6칸이 모두 같으면 그대로 쓰고, 하나라도 다르면 세 번째를 돌린다.
  다수결 결과는 3회 다수결과 같고(두 표가 같으면 그게 과반), 비용만 줄어든다.
"""
from __future__ import annotations

import itertools
import statistics
from pathlib import Path

from .constants import AXES, HOLD, NA, PRESERVED, VIOLATED
from .io_utils import write_json, write_jsonl
from .runner import _ok, latest_records

SEV_ORDER = ["minor", "major", "critical"]
_TIE = [PRESERVED, NA, HOLD]


def load_group(dirs: list[Path]) -> dict[str, dict]:
    """한 번의 실행이 여러 폴더(예: 개발용·확인용)로 나뉘어 있으면 합친다."""
    out = {}
    for d in dirs:
        out.update(latest_records(Path(d)))
    return out


def _judg(out: dict | None, ax: str) -> str | None:
    if not out:
        return None
    return ((out.get("axes") or {}).get(ax) or {}).get("judgment")


def majority_cell(js: list[str | None], sevs: list[str | None]) -> tuple[str, str | None]:
    vs = [j for j in js if j is not None]
    n_v = sum(j == VIOLATED for j in vs)
    if vs and n_v * 2 > len(vs):
        ss = sorted((SEV_ORDER.index(s) for j, s in zip(js, sevs) if j == VIOLATED and s in SEV_ORDER))
        sev = SEV_ORDER[ss[len(ss) // 2]] if ss else None  # 짝수면 높은 쪽 중앙값
        return VIOLATED, sev
    rest = [j for j in vs if j != VIOLATED]
    if not rest:
        return PRESERVED, None
    best = max(_TIE, key=lambda c: (rest.count(c), -_TIE.index(c)))
    return best, None


def merge_outputs(outs: list[dict | None]) -> dict | None:
    outs_ok = [o for o in outs if o]
    if not outs_ok:
        return None
    if not all(o.get("evaluable", True) for o in outs_ok):
        # 평가 불가 판정이 과반이면 그 실행을 그대로 쓴다
        bad = [o for o in outs_ok if not o.get("evaluable", True)]
        if len(bad) * 2 > len(outs_ok):
            return bad[0]
        outs_ok = [o for o in outs_ok if o.get("evaluable", True)]
    maj = {}
    for ax in AXES:
        js = [_judg(o, ax) for o in outs_ok]
        sevs = [((o.get("axes") or {}).get(ax) or {}).get("severity") for o in outs_ok]
        maj[ax] = majority_cell(js, sevs)
    # 대표 실행: 다수결과 가장 많이 일치
    rep = max(outs_ok, key=lambda o: sum(_judg(o, ax) == maj[ax][0] for ax in AXES))
    errors = [dict(e) for e in (rep.get("errors") or [])]
    next_id = len(errors) + 1
    axes = {}
    for ax in AXES:
        j, sev = maj[ax]
        src = rep if _judg(rep, ax) == j else next(o for o in outs_ok if _judg(o, ax) == j) \
            if any(_judg(o, ax) == j for o in outs_ok) else rep
        entry = dict((src.get("axes") or {}).get(ax) or {})
        entry["judgment"], entry["severity"] = j, sev if j == VIOLATED else None
        if j != VIOLATED:
            entry["error_ids"] = []
        elif src is not rep:
            # 다른 실행의 오류를 가져와 새 ID를 붙인다
            src_err = {e["id"]: e for e in (src.get("errors") or [])}
            new_ids = []
            for eid in entry.get("error_ids") or []:
                if eid in src_err:
                    e = dict(src_err[eid]); e["id"] = f"E{next_id:03d}"; next_id += 1
                    errors.append(e); new_ids.append(e["id"])
            entry["error_ids"] = new_ids
        axes[ax] = entry
    return {**rep, "errors": errors, "axes": axes}


def vote_records(groups: list[dict[str, dict]], adaptive: bool = True) -> tuple[dict[str, dict], dict]:
    """groups: 실행별 {item_id: record}. 앞의 두 실행이 6칸 모두 같으면 그 둘로 끝낸다(adaptive).
    반환: ({item_id: 병합 record}, 비용 통계)"""
    ids = sorted(set().union(*[g.keys() for g in groups]))
    merged, calls = {}, 0
    for i in ids:
        recs = [g.get(i) for g in groups]
        outs = [r.get("output") if r and _ok(r) else None for r in recs]
        used = len(outs)
        if adaptive and len(outs) >= 3 and outs[0] and outs[1] and \
                all(_judg(outs[0], ax) == _judg(outs[1], ax) for ax in AXES):
            used = 2
        calls += used
        m = merge_outputs(outs[:used])
        merged[i] = {"item_id": i, "model": (recs[0] or {}).get("model"), "usage": None,
                     "output": m, "votes_used": used, "errors": [] if m else ["다수결할 출력 없음"]}
    return merged, {"items": len(ids), "calls": calls, "calls_per_item": calls / len(ids) if ids else 0}


def write_vote_run(groups_dirs: list[list[Path]], out_dir: Path, adaptive: bool = True) -> Path:
    groups = [load_group(ds) for ds in groups_dirs]
    merged, cost = vote_records(groups, adaptive)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "outputs.jsonl", list(merged.values()))
    from .io_utils import read_json
    m0 = read_json(Path(groups_dirs[0][0]) / "meta.json")
    write_json(out_dir / "meta.json", {**m0, "items": sorted(merged), "vote_of": [[str(d) for d in ds] for ds in groups_dirs],
                                       "vote_adaptive": adaptive, "vote_cost": cost})
    return out_dir


# ---------- 흔들림 측정 ----------
def _vio(out: dict | None, ax: str) -> bool:
    return _judg(out, ax) == VIOLATED


def cell_vector(recs: dict[str, dict], ids: list[str]) -> list[bool]:
    return [_vio((recs.get(i) or {}).get("output"), ax) for i in ids for ax in AXES]


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def stability(groups: list[dict[str, dict]], ids: list[str], k: int = 3) -> dict:
    """단일 실행끼리, 그리고 서로 겹치지 않는 k회 다수결 묶음끼리의 일치도."""
    single = [cell_vector(g, ids) for g in groups]
    pairs = list(itertools.combinations(range(len(groups)), 2))
    s_k = [kappa(single[a], single[b]) for a, b in pairs]
    s_diff = [sum(x != y for x, y in zip(single[a], single[b])) for a, b in pairs]
    res = {"single": {"kappa_mean": statistics.mean(s_k), "kappa_min": min(s_k), "kappa_max": max(s_k),
                      "cells_diff_mean": statistics.mean(s_diff), "n_pairs": len(pairs)}}
    # 겹치지 않는 두 묶음 (6회면 3+3 나누기 10가지)
    idx = range(len(groups))
    vk, vd, seen = [], [], set()
    for A in itertools.combinations(idx, k):
        B = tuple(x for x in idx if x not in A)[:k]
        if len(B) < k or (B, A) in seen:
            continue
        seen.add((A, B))
        va = cell_vector(vote_records([groups[x] for x in A], adaptive=False)[0], ids)
        vb = cell_vector(vote_records([groups[x] for x in B], adaptive=False)[0], ids)
        vk.append(kappa(va, vb)); vd.append(sum(x != y for x, y in zip(va, vb)))
    res["vote"] = {"kappa_mean": statistics.mean(vk), "kappa_min": min(vk), "kappa_max": max(vk),
                   "cells_diff_mean": statistics.mean(vd), "n_pairs": len(vk)}
    return res
