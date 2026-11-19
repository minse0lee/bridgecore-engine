"""개발 100 엔진 실행을 평가자 다수결(consensus_dev) 기준으로 채점. 미판정 문항 제외.
python tools/score_dev_majority.py runs/A runs/B ...  → 전체·축별 F1, Textual 오탐·놓침 문항"""
import json, sys
cons=[json.loads(l) for l in open('data/golden/consensus_dev_2026-11-18.jsonl')]
AX=['Semantic','Textual','Context','Deontic','Register','Cultural']
def f1(t): tp,fp,fn=t; p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; return (2*p*r/(p+r) if p+r else 0), p, r
for run in sys.argv[1:]:
    eng={}; bad=[]
    for l in open(f'{run}/outputs.jsonl'):
        x=json.loads(l); o=x.get('output') or {}
        if o.get('axes'):
            for ax,v in o['axes'].items(): eng[(x['item_id'],ax)]=v['judgment']=='훼손'
        else: bad.append(x['item_id'])
    res={a:[0,0,0] for a in AX+['ALL']}; tfp=[]; tfn=[]
    for c in cons:
        k=(c['item_id'],c['axis'])
        if k not in eng: continue
        g=c['violated']=='훼손'; e=eng[k]
        for a in ('ALL',c['axis']):
            t=res[a]
            if e and g: t[0]+=1
            elif e and not g: t[1]+=1
            elif g and not e: t[2]+=1
        if c['axis']=='Textual':
            if e and not g: tfp.append(c['item_id'])
            if g and not e: tfn.append(c['item_id'])
    F,P,R=f1(res['ALL'])
    print(f"{run.split('/')[-1]:28s} 전체 F1 {F:.3f} P {P:.3f} R {R:.3f} | "+' '.join(f"{a[:4]} {f1(res[a])[0]:.2f}" for a in AX)+f" | 미판정 {sorted(set(bad))}")
    print(f"   Textual 엔진훼손 {res['Textual'][0]+res['Textual'][1]} 오탐 {tfp} 놓침 {tfn}")
