import json, random, statistics, collections, sys
from pathlib import Path
sys.path.insert(0, '.')
from bridgecore.vote import load_group
from bridgecore.constants import AXES, VIOLATED

R = Path('runs')
G = {'전': [load_group([R / f'mt_pilot60_base_r{k}']) for k in (1, 2)],
     'v1': [load_group([R / f'mt_pilot60_lora_r{k}']) for k in (1, 2)],
     'v2': [load_group([R / f'mt_pilot60_v2D_lora_r{k}']) for k in (1, 2)]}
ids = sorted(G['전'][0])
for k, gs in G.items():
    for g in gs:
        miss = [i for i in ids if not ((g.get(i) or {}).get('output') or {}).get('axes')]
        assert not miss, (k, miss)
tg = {t: {json.loads(l)['item_id']: json.loads(l)['tgt'] for l in open(f'data/mt_eval/{f}.jsonl')}
      for t, f in (('전', 'pilot60_lora_v1__base'), ('v1', 'pilot60_lora_v1__lora'), ('v2', 'pilot60_lora_v2_D__lora'))}
J = lambda g, i, a: g[i]['output']['axes'].get(a, {}).get('judgment')
S = lambda g, i, a: g[i]['output']['axes'].get(a, {}).get('severity')
cnt = lambda gs, axs: statistics.mean(sum(J(g, i, a) == VIOLATED for i in ids for a in axs) for g in gs)

print('| 축 | 학습 전 | v1 | v2 | v2−v1 |\n|---|---|---|---|---|')
for ax in AXES + ['합계']:
    axs = AXES if ax == '합계' else [ax]
    b, v1, v2 = (cnt(G[k], axs) for k in ('전', 'v1', 'v2'))
    print(f'| {ax} | {b:.1f} | {v1:.1f} | {v2:.1f} | {v2 - v1:+.1f} |')
for k, gs in G.items():
    print(k, '훼손 문장', statistics.mean(sum(any(J(g, i, a) == VIOLATED for a in AXES) for i in ids) for g in gs),
          '| major 이상 칸', statistics.mean(sum(J(g, i, a) == VIOLATED and S(g, i, a) in ('major', 'critical') for i in ids for a in AXES) for g in gs))
per = lambda gs, i: statistics.mean(sum(J(g, i, a) == VIOLATED for a in AXES) for g in gs)
def boot(x, y, seed=7):
    d = [per(G[y], i) - per(G[x], i) for i in ids]
    rng = random.Random(seed); bs = sorted(sum(rng.choice(d) for _ in ids) for _ in range(5000))
    return round(sum(d), 1), round(bs[125], 1), round(bs[4874], 1)
print('v2−전', boot('전', 'v2'), '| v1−전', boot('전', 'v1'), '| v2−v1', boot('v1', 'v2'))
# 실행 간 일치(판정 안정성)
for k, gs in G.items():
    agree = sum((J(gs[0], i, a) == VIOLATED) == (J(gs[1], i, a) == VIOLATED) for i in ids for a in AXES)
    print(k, '2회 실행 칸 일치', agree, '/', len(ids) * 6)
both = lambda k, i, a, v: all((J(g, i, a) == VIOLATED) == v for g in G[k])
fixed = [(i, a) for i in ids for a in AXES if both('v1', i, a, True) and both('v2', i, a, False)]
broke = [(i, a) for i in ids for a in AXES if both('v1', i, a, False) and both('v2', i, a, True)]
print('v1 훼손→v2 해소(2회 일치)', len(fixed), collections.Counter(a for _, a in fixed))
print('v1 정상→v2 새 훼손(2회 일치)', len(broke), collections.Counter(a for _, a in broke))
for lab, lst in (('해소', fixed), ('새 훼손', broke)):
    for i, a in lst:
        r = G['v2' if lab == '새 훼손' else 'v1'][0][i]['output']['axes'][a]['reason']
        print(f'  [{lab}] {i} {a} | v1: {tg["v1"][i][:70]} | v2: {tg["v2"][i][:70]} | {r[:140]}')
# 문장 끝 어미
def end(t):
    t = t.strip().rstrip('"”’\'_ ').rstrip('.!?…').strip()
    if t.endswith(('습니다', '니까', '십시오', '세요', '시오')): return '합쇼/하오'
    if t.endswith('요'): return '해요'
    if t.endswith(('다', '라', '가', '까', '냐', '자', '는가', '오')): return '하라/-다'
    return '기타'
for k in tg:
    print(k, '어미', collections.Counter(end(t) for t in tg[k].values()))
