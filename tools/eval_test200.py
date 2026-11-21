"""시험 200 평가 (사전등록 docs/사전등록_모창_v2_2026-11-13.md '시험 200 평가').
python tools/eval_test200.py --gold data/golden/consensus_test_2026-11-21.jsonl \
    --v1 runs/golden_test200_v1_1_r1 runs/golden_test200_v1_1_r2 --v2 runs/golden_test200_v2_r1 runs/golden_test200_v2_r2 \
    --raters data/golden/labels/R001_2026-11-20.xlsx ... --out reports/test200_v2_vs_v1_1
정답: 평가자 최초판정 다수결(판단보류=훼손 아님, 동수는 제외). 엔진 판단보류=훼손 아님. 엔진 미판정 문항은 그 실행에서 제외."""
import argparse, csv, json, random, sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'tools'))
from golden_labels import load_v3, split_map, fleiss
AX=['Semantic','Textual','Context','Deontic','Register','Cultural']
def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else None; r=tp/(tp+fn) if tp+fn else None
    f=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else None
    return p,r,f
def load_run(d):
    last={}
    for l in open(Path(d)/'outputs.jsonl',encoding='utf-8'):
        x=json.loads(l); last[x['item_id']]=x
    eng={}; bad=[]
    for iid,x in last.items():
        o=x.get('output') or {}
        if o.get('axes'): eng[iid]={ax:o['axes'][ax]['judgment']=='훼손' for ax in AX}
        else: bad.append(iid)
    return eng,sorted(bad)
def counts(gold,eng,items,axes=AX):
    tp=fp=fn=0
    for i in items:
        if i not in eng: continue
        for ax in axes:
            g=gold.get((i,ax))
            if g is None: continue
            e=eng[i][ax]; tp+=e and g; fp+=e and not g; fn+=(not e) and g
    return tp,fp,fn
def sent_counts(gold,eng,items):
    tp=fp=fn=0
    for i in items:
        if i not in eng: continue
        gs=[gold.get((i,ax)) for ax in AX]
        if any(g is True for g in gs): g=True
        elif all(g is False for g in gs): g=False
        else: continue
        e=any(eng[i].values()); tp+=e and g; fp+=e and not g; fn+=(not e) and g
    return tp,fp,fn
def mean(xs): xs=[x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else None
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gold'); ap.add_argument('--v1',nargs='+'); ap.add_argument('--v2',nargs='+')
    ap.add_argument('--raters',nargs='+'); ap.add_argument('--out'); ap.add_argument('--B',type=int,default=2000); a=ap.parse_args()
    cons=[json.loads(l) for l in open(a.gold,encoding='utf-8')]
    gold={(c['item_id'],c['axis']):(None if c['violated'] is None else c['violated']=='훼손') for c in cons}
    items=sorted({c['item_id'] for c in cons})
    meta={r['문항ID']:r for r in csv.DictReader(open(ROOT/'data/golden/split_final_newids_2026-11-13.csv',encoding='utf-8-sig'))}
    runs={'v1.1':[load_run(d) for d in a.v1],'v2':[load_run(d) for d in a.v2]}
    R=[]; P=lambda *x: R.append(' '.join(str(y) for y in x))
    P('# 시험 200 평가: 엔진 v2 vs v1.1\n')
    P(f'- 정답: 평가자 3명 최초판정 다수결 (`{a.gold}`). 문항-축 {len(gold)}칸 중 훼손 {sum(v is True for v in gold.values())}, 결론 없음(동수) {sum(v is None for v in gold.values())}칸은 제외')
    for v,rs in runs.items():
        P(f'- {v} 실행: '+', '.join(f'{Path(d).name} (미판정 {len(b)}{" "+str(b) if b else ""})' for d,(e,b) in zip(a.v1 if v=="v1.1" else a.v2,rs)))
    P('\n## 주 지표: 축 단위 F1 (실행 평균)\n')
    P('| 버전 | 축 단위 P | R | F1 | 실행별 F1 | 문장 단위 F1 |'); P('|---|---|---|---|---|---|')
    summ={}
    for v,rs in runs.items():
        per=[prf(*counts(gold,e,items)) for e,_ in rs]; sen=[prf(*sent_counts(gold,e,items))[2] for e,_ in rs]
        summ[v]=dict(p=mean([x[0] for x in per]),r=mean([x[1] for x in per]),f=mean([x[2] for x in per]),sent=mean(sen),runs=[x[2] for x in per])
        s=summ[v]; P(f"| {v} | {s['p']:.3f} | {s['r']:.3f} | **{s['f']:.3f}** | {' / '.join(f'{x:.3f}' for x in s['runs'])} | {s['sent']:.3f} |")
    # 짝 부트스트랩
    common=[i for i in items if all(i in e for v in runs for e,_ in runs[v])]
    rnd=random.Random(20261121); diffs=[]; sdiffs=[]
    pre={v:[{i:(counts(gold,e,[i]),sent_counts(gold,e,[i])) for i in common} for e,_ in rs] for v,rs in runs.items()}
    def agg(v,smp,k):
        out=[]
        for d in pre[v]:
            t=[0,0,0]
            for i in smp:
                for j in range(3): t[j]+=d[i][k][j]
            out.append(prf(*t)[2])
        return mean(out)
    for _ in range(a.B):
        smp=[rnd.choice(common) for _ in common]
        diffs.append(agg('v2',smp,0)-agg('v1.1',smp,0)); sdiffs.append(agg('v2',smp,1)-agg('v1.1',smp,1))
    q=lambda xs:(sorted(xs)[int(.025*len(xs))],sorted(xs)[int(.975*len(xs))-1])
    d0=agg('v2',common,0)-agg('v1.1',common,0); s0=agg('v2',common,1)-agg('v1.1',common,1)
    lo,hi=q(diffs); slo,shi=q(sdiffs)
    P(f'\nv2 − v1.1 (모든 실행에서 판정된 {len(common)}문항, 문항 단위 짝 부트스트랩 {a.B}회, 95% 구간)')
    P(f'- 축 단위 F1: **{d0:+.3f}** [{lo:+.3f}, {hi:+.3f}]'); P(f'- 문장 단위 F1: {s0:+.3f} [{slo:+.3f}, {shi:+.3f}]')
    P('\n## 축별 P / R / F1 (실행 평균)\n'); P('| 축 | 정답 훼손 | v1.1 P | R | F1 | v2 P | R | F1 |'); P('|---|---|---|---|---|---|---|---|')
    f=lambda x:'-' if x is None else f'{x:.2f}'
    for ax in AX:
        row=[ax,str(sum(gold.get((i,ax)) is True for i in items))]
        for v,rs in runs.items():
            per=[prf(*counts(gold,e,items,[ax])) for e,_ in rs]
            row+= [f(mean([x[k] for x in per])) for k in range(3)]
        P('| '+' | '.join(row)+' |')
    P('\n## 부분집합 (축 단위 F1, 실행 평균)\n'); P('| 부분집합 | 문항 | 정답 훼손 | v1.1 | v2 |'); P('|---|---|---|---|---|')
    subs={'직접분석':lambda m:m['선정 경로']=='직접분석','LLM 분석':lambda m:m['선정 경로']=='LLM 분석','무작위':lambda m:m['선정 경로']=='무작위',
          'LLM 분석 제외':lambda m:m['선정 경로']!='LLM 분석','WMT 제외':lambda m:m['비고']!='WMT24++','WMT만':lambda m:m['비고']=='WMT24++'}
    for name,fn in subs.items():
        its=[i for i in items if fn(meta[i])]
        vals=[mean([prf(*counts(gold,e,its))[2] for e,_ in runs[v]]) for v in runs]
        P(f"| {name} | {len(its)} | {sum(gold.get((i,ax)) is True for i in its for ax in AX)} | {f(vals[0])} | {f(vals[1])} |")
    if a.raters:
        sp=split_map(); rat={}
        for x in a.raters:
            rid,rows=load_v3(x); rat[rid]={(r['item_id'],r['axis']):r['judgment'] for r in rows if sp.get(r['item_id'])=='시험'}
        ids=sorted(rat); keys=sorted(gold)
        P('\n## 사람 수준 참고 (평가자 최초판정)\n')
        full=[[rat[r][k] for r in ids] for k in keys if all(rat[r].get(k) not in (None,'판단보류') for r in ids)]
        P(f"- Fleiss kappa(훼손 여부, 세 명 모두 판정한 {len(full)}칸): {fleiss([['훼손' if j=='훼손' else '아님' for j in row] for row in full],['훼손','아님'])}")
        P('- 평가자 한 명을 나머지 두 명이 같은 판정을 한 칸과 비교한 F1(두 명이 갈린 칸 제외). 엔진 점수와 기준이 달라 직접 비교용은 아님')
        for r in ids:
            o=[x for x in ids if x!=r]; tp=fp=fn=0
            for k in keys:
                js=[rat[x].get(k) for x in o]
                if any(j is None for j in js): continue
                g=[j=='훼손' for j in js]
                if g[0]!=g[1] or rat[r].get(k) is None: continue
                e=rat[r][k]=='훼손'; tp+=e and g[0]; fp+=e and not g[0]; fn+=(not e) and g[0]
            P(f'  - {r}: F1 {prf(tp,fp,fn)[2]:.3f} (P {f(prf(tp,fp,fn)[0])}, R {f(prf(tp,fp,fn)[1])})')
    txt='\n'.join(R)+'\n'; print(txt)
    if a.out:
        Path(a.out+'.md').write_text(txt,encoding='utf-8')
        Path(a.out+'.json').write_text(json.dumps({'summary':summ,'diff_axis':[d0,lo,hi],'diff_sent':[s0,slo,shi],'common_items':len(common)},ensure_ascii=False,indent=1),encoding='utf-8')
if __name__=='__main__': main()
