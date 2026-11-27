"""프로필 A·B 시험(1·3장 300문장) 집계. 2026-11-28 (4)에 적은 기준대로 센다."""
import collections
import json
import sys

AX = ['Semantic', 'Textual', 'Context', 'Deontic', 'Register', 'Cultural']
E = 'data/book/gatsby/eval/'


def out(d):
    last = {}
    for l in open(f'runs/{d}/outputs.jsonl', encoding='utf-8'):
        x = json.loads(l); last[x['item_id']] = x
    res = {}
    for i, x in last.items():
        o = x.get('output') or {}
        if not o.get('axes') or x.get('errors'):      # 형식 오류로 남은 문항은 판정에서 뺀다
            res[i] = None; continue
        try:
            bad = [a for a in AX if o['axes'][a]['judgment'] == '훼손']
        except Exception:
            res[i] = None; continue
        sev = [o['axes'][a].get('severity') for a in bad]
        res[i] = ('none' if not bad else 'major' if ('major' in sev or 'critical' in sev) else 'minor', bad,
                  {a: o['axes'][a].get('reason') for a in bad}, x)
    return res


def lab(r):
    return r[0] if r else 'err'


def main():
    A = out('gatsby_ch13_pilot300_A_prev5next5'); B = out('gatsby_ch13_pilot300_B_prev5next5_profile')
    fl = [json.loads(l) for l in open(E + 'ch1_3__gemma_bf16_ctx5_flagged150.jsonl')]
    un = [json.loads(l) for l in open(E + 'ch1_3__gemma_bf16_ctx5_unflagged150.jsonl')]
    sid = {r['item_id']: r['book_sentence_id'] for r in fl + un}
    V = {}
    for l in open('data/book/gatsby/retrans/review_claude_ch1_3_reason.tsv', encoding='utf-8'):
        p = l.rstrip('\n').split('\t')
        if p[0] != 'id': V[p[0]] = p
    F, U = [r['item_id'] for r in fl], [r['item_id'] for r in un]
    print('형식 오류(판정 없음): A', [sid[i] for i in F + U if not A.get(i)], '| B', [sid[i] for i in F + U if not B.get(i)])

    def dist(res, ids): return dict(collections.Counter(lab(res.get(i)) for i in ids))
    print('\n짚였던 150   A', dist(A, F), '| B', dist(B, F))
    for k, name in (('T', '맞는 지적 55'), ('P', '사소함 74'), ('F', '오판 21')):
        sub = [i for i in F if V[sid[i]][1] == k]
        print(f'  {name}: A', dist(A, sub), '| B', dist(B, sub))
    keep11 = ['S000139', 'S000121', 'S000140', 'S000141', 'S000634', 'S000742', 'S000743', 'S000041', 'S000821']
    print('\n안 짚였던 150 A', dist(A, U), '| B', dist(B, U))
    both = [i for i in F + U if A.get(i) and B.get(i)]
    tr = collections.Counter((lab(A[i]), lab(B[i])) for i in both)
    print('\nA→B 전이(둘 다 판정된', len(both), '문항):', dict(tr))
    flip = [i for i in both if (A[i][0] == 'none') != (B[i][0] == 'none')]
    print('짚음↔안 짚음이 갈린 문항', len(flip), '| A만 짚음', sum(A[i][0] != 'none' for i in flip), '| B만 짚음', sum(B[i][0] != 'none' for i in flip))
    for d in ('gatsby_ch13_pilot300_A_prev5next5', 'gatsby_ch13_pilot300_B_prev5next5_profile'):
        u = collections.Counter()
        for l in open(f'runs/{d}/outputs.jsonl', encoding='utf-8'):
            for k, v in (json.loads(l).get('usage') or {}).items():
                if isinstance(v, (int, float)): u[k] += v
        c = u['input_tokens'] * 2 / 1e6 + u['output_tokens'] * 10 / 1e6 + u['cache_read_input_tokens'] * .1 / 1e6 + u['cache_creation_input_tokens'] * 2.5 / 1e6
        print(d, dict(u), f'${c:.2f}')
    return A, B, sid, V, F, U


if __name__ == '__main__':
    main()
