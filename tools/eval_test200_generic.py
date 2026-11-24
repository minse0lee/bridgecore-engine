"""시험 200: 일반 프롬프트 기준선 vs 엔진, 문장 단위 (합의본 정답).
python tools/eval_test200_generic.py --gold data/golden/consensus_test_2026-11-24_합의.jsonl
문장 정답: 6축 중 하나라도 훼손이면 훼손. 일반 프롬프트: 오류를 하나라도 적으면 훼손(주), major 이상만 세는 기준은 참고."""
import argparse, json, sys
from pathlib import Path
AX=['Semantic','Textual','Context','Deontic','Register','Cultural']
def last(d):
    o={}
    for l in open(Path(d)/'outputs.jsonl',encoding='utf-8'):
        x=json.loads(l); o[x['item_id']]=x
    return o
def prf(pred,gold):
    tp=sum(pred[i] and gold[i] for i in gold); fp=sum(pred[i] and not gold[i] for i in gold); fn=sum((not pred[i]) and gold[i] for i in gold)
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return p,r,(2*tp/(2*tp+fp+fn) if tp+fp+fn else 0),fp
ap=argparse.ArgumentParser(); ap.add_argument('--gold'); a=ap.parse_args()
g={}
for l in open(a.gold,encoding='utf-8'):
    c=json.loads(l); g.setdefault(c['item_id'],[]).append(c['violated']=='훼손')
gold={i:any(v) for i,v in g.items()}
print(f'정답: {a.gold} · 문장 {len(gold)} 중 훼손 {sum(gold.values())}, 문제없음 {len(gold)-sum(gold.values())}')
tok={}
print('| 방식 | 실행 | 정밀도 | 재현율 | F1 | 오탐 문장 |'); print('|---|---|---|---|---|---|')
for name,runs,kind in [('일반 프롬프트(오류 하나라도)',['golden_test200_generic_r1','golden_test200_generic_r2'],'any'),
                       ('일반 프롬프트(major 이상만, 참고)',['golden_test200_generic_r1','golden_test200_generic_r2'],'major'),
                       ('엔진 v1.1',['golden_test200_v1_1_r1','golden_test200_v1_1_r2'],'eng'),('엔진 v2',['golden_test200_v2_r1','golden_test200_v2_r2'],'eng')]:
    for r in runs:
        o=last('runs/'+r); assert set(o)==set(gold),(r,len(o))
        if kind=='eng': pred={i:any(x['output']['axes'][ax]['judgment']=='훼손' for ax in AX) for i,x in o.items()}
        else:
            pred={}
            for i,x in o.items():
                es=(x.get('output') or {}).get('errors') or []
                pred[i]=bool(es) if kind=='any' else any(str(e.get('severity','')).lower() in('major','critical') for e in es)
            tok[r]={k:sum((x.get('usage') or {}).get(k,0) or 0 for x in o.values()) for k in('input_tokens','output_tokens','cache_creation_input_tokens','cache_read_input_tokens')}
        p,rc,f,fp=prf(pred,gold); print(f'| {name} | {r} | {p:.2f} | {rc:.2f} | {f:.3f} | {fp} |')
print('토큰',json.dumps(tok,ensure_ascii=False))
