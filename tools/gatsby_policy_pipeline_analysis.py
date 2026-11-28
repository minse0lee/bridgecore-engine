"""번역 쪽 방침 시험: 같은 603문장을 같은 엔진 설정(앞 5 + 뒤 5문장, 방침 없이)으로 검증한 결과 비교 (2026-11-28).

    기존 번역(오늘 A 실행) vs policy vs policy_next5 [vs base]
"""
import collections
import json
import sys
from pathlib import Path

AX = ['Semantic', 'Textual', 'Context', 'Deontic', 'Register', 'Cultural']
E = Path('data/book/gatsby/eval')
rows = lambda p: [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def out(dirs, item_files):
    sid = {r['item_id']: r['book_sentence_id'] for f in item_files for r in rows(E / f)}
    tgt = {r['book_sentence_id']: r['tgt'] for f in item_files for r in rows(E / f)}
    last = {}
    for d in dirs:
        for l in open(f'runs/{d}/outputs.jsonl', encoding='utf-8'):
            x = json.loads(l); last[x['item_id']] = x
    res = {}
    for i, x in last.items():
        o = x.get('output') or {}
        if not o.get('axes') or x.get('errors'):
            res[sid[i]] = None; continue
        bad = [a for a in AX if o['axes'][a]['judgment'] == '훼손']
        sev = [o['axes'][a].get('severity') for a in bad]
        res[sid[i]] = ('none' if not bad else 'major' if ('major' in sev or 'critical' in sev) else 'minor', bad,
                       {a: o['axes'][a].get('reason') for a in bad})
    return res, tgt


def cost(dirs):
    u = collections.Counter()
    for d in dirs:
        for l in open(f'runs/{d}/outputs.jsonl', encoding='utf-8'):
            for k, v in (json.loads(l).get('usage') or {}).items():
                if isinstance(v, (int, float)): u[k] += v
    return u['input_tokens'] * 2 / 1e6 + u['output_tokens'] * 10 / 1e6 + u['cache_read_input_tokens'] * .1 / 1e6 + u['cache_creation_input_tokens'] * 2.5 / 1e6


RUNS = {
    '기존': (['gatsby_ch13_pilot300_A_prev5next5', 'gatsby_ch7_sel303_A_prev5next5'],
           ['ch1_3__gemma_bf16_ctx5_pilot300_prev5next5.jsonl', 'ch7__gemma_bf16_ctx5_sel303_prev5next5.jsonl']),
    'policy': (['gatsby_pipe_policy_603_prev5next5'], ['pipe_policy__target603_prev5next5.jsonl']),
    'policy_next5': (['gatsby_pipe_policy_next5_603_prev5next5'], ['pipe_policy_next5__target603_prev5next5.jsonl']),
    'base': (['gatsby_pipe_base_603_prev5next5'], ['pipe_base__target603_prev5next5.jsonl']),
}
lab = lambda r: r[0] if r else 'err'

if __name__ == '__main__':
    R, T = {}, {}
    for k, (d, f) in RUNS.items():
        if all(Path(f'runs/{x}/outputs.jsonl').exists() for x in d):
            R[k], T[k] = out(d, f)
    ids = sorted(R['기존'])
    ch13 = {r['book_sentence_id'] for r in rows(E / 'ch1_3__gemma_bf16_ctx5_pilot300.jsonl')}
    for name, sub in (('전체 603', ids), ('1·3장 300', [i for i in ids if i in ch13]), ('7장 303', [i for i in ids if i not in ch13])):
        print(f'■ {name}')
        for k in R:
            c = collections.Counter(lab(R[k].get(i)) for i in sub)
            fl = c['minor'] + c['major']
            print(f"  {k:13s} 문제없음 {c['none']:3d} · minor {c['minor']:3d} · major {c['major']:2d} · 오류 {c['err']} → 짚인 문장 {fl} ({fl / (len(sub) - c['err']) * 100:.1f}%)")
    print('■ 축별 지적 수')
    for k in R:
        print(f'  {k:13s}', dict(collections.Counter(a for i in ids if R[k].get(i) for a in R[k][i][1])))
    print('■ 기존 → 새 번역 (둘 다 판정 있는 문장)')
    for k in R:
        if k == '기존': continue
        both = [i for i in ids if R['기존'].get(i) and R[k].get(i)]
        same = [i for i in both if T['기존'][i] == T[k][i]]
        f = lambda s: (sum(R['기존'][i][0] != 'none' and R[k][i][0] == 'none' for i in s), sum(R['기존'][i][0] == 'none' and R[k][i][0] != 'none' for i in s))
        ch = [i for i in both if i not in set(same)]
        print(f'  {k}: 번역이 바뀐 문장 {len(ch)} — 지적 사라짐 {f(ch)[0]} · 새로 생김 {f(ch)[1]} | 번역이 그대로인 문장 {len(same)} — 사라짐 {f(same)[0]} · 생김 {f(same)[1]} (엔진 흔들림 참고)')
    print('■ 비용')
    for k in R:
        print(f'  {k}: ${cost(RUNS[k][0]):.2f}')
