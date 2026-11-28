"""번역 쪽 방침 시험: 같은 입력을 다시 돌려 엔진 판정의 흔들림 폭을 잰다 (1·3장 300문장, base와 policy_next5). 2026-11-28."""
import collections
import json
import sys
sys.path.insert(0, 'tools')
import gatsby_policy_pipeline_analysis as A

E = A.E
ch13 = sorted({r['book_sentence_id'] for r in A.rows(E / 'ch1_3__gemma_bf16_ctx5_pilot300.jsonl')})
fl = lambda r: r[0] != 'none'


def main():
    R = {}
    for c in ('base', 'policy_next5'):
        R[c, 1], _ = A.out(*A.RUNS[c])
        R[c, 2], _ = A.out([f'gatsby_pipe_{c}_ch13_300_rerun'], [f'pipe_{c}__ch13_300_prev5next5.jsonl'])
    for (c, k), r in R.items():
        cnt = collections.Counter(A.lab(r.get(i)) for i in ch13)
        print(f"{c} {k}차: 문제없음 {cnt['none']} · minor {cnt['minor']} · major {cnt['major']} · 판정 없음 {cnt['err']} → 짚인 문장 {cnt['minor'] + cnt['major']}")
    print('■ 같은 입력 두 번(흔들림)')
    for c in ('base', 'policy_next5'):
        a, b = R[c, 1], R[c, 2]
        both = [i for i in ch13 if a.get(i) and b.get(i)]
        d1 = sum(fl(a[i]) and not fl(b[i]) for i in both); d2 = sum(not fl(a[i]) and fl(b[i]) for i in both)
        f1 = sum(fl(a[i]) for i in both)
        print(f"  {c}: 둘 다 판정 {len(both)} · 1차만 짚임 {d1} · 2차만 짚임 {d2} · 뒤바뀐 문장 {d1 + d2} ({(d1 + d2) / len(both) * 100:.1f}%) · 짚인 수 {f1} → {f1 - d1 + d2}"
              f" · 1차에 짚인 문장이 2차에도 짚인 비율 {(f1 - d1) / f1 * 100:.0f}%")
    print('■ 조건 간 차이(base − policy_next5), 실행 조합별')
    for k1 in (1, 2):
        for k2 in (1, 2):
            a, b = R['base', k1], R['policy_next5', k2]
            both = [i for i in ch13 if a.get(i) and b.get(i)]
            print(f"  base {k1}차 vs next5 {k2}차: {sum(fl(a[i]) for i in both)} − {sum(fl(b[i]) for i in both)} = {sum(fl(a[i]) for i in both) - sum(fl(b[i]) for i in both)}"
                  f" (base만 {sum(fl(a[i]) and not fl(b[i]) for i in both)} · next5만 {sum(not fl(a[i]) and fl(b[i]) for i in both)})")
    allj = [i for i in ch13 if all(R[x].get(i) for x in R)]
    two = lambda c: [sum(fl(R[c, k][i]) for k in (1, 2)) for i in allj]
    b2, n2 = two('base'), two('policy_next5')
    print(f"■ 네 실행 모두 판정이 있는 {len(allj)}문장: 두 번 다 짚임 base {sum(x == 2 for x in b2)} · next5 {sum(x == 2 for x in n2)} / 한 번이라도 짚임 base {sum(x >= 1 for x in b2)} · next5 {sum(x >= 1 for x in n2)}"
          f" / 평균 짚인 수 base {sum(b2) / 2:.1f} · next5 {sum(n2) / 2:.1f}")
    print('■ 비용', ' · '.join(f"{c} ${A.cost([f'gatsby_pipe_{c}_ch13_300_rerun']):.2f}" for c in ('base', 'policy_next5')))


if __name__ == '__main__':
    main()
