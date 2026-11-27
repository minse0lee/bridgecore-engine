import json,collections,sys
AX=['Semantic','Textual','Context','Deontic','Register','Cultural']
E='data/book/gatsby/eval/'
def out(d,key=None):
    last={}
    for l in open(f'runs/{d}/outputs.jsonl',encoding='utf-8'):
        x=json.loads(l); last[x['item_id']]=x
    res={}
    for i,x in last.items():
        o=x.get('output') or {}
        if not o.get('axes') or x.get('errors'): res[i]=None; continue
        bad=[a for a in AX if o['axes'][a]['judgment']=='훼손']
        sev=[o['axes'][a].get('severity') for a in bad]
        res[i]=('none' if not bad else 'major' if ('major' in sev or 'critical' in sev) else 'minor',bad,{a:o['axes'][a].get('reason') for a in bad},x)
    return res
def lab(r): return r[0] if r else 'err'
A=out('gatsby_ch7_sel303_A_prev5next5'); B=out('gatsby_ch7_sel303_B_prev5next5_profile_v1sys'); O=out('gatsby_ch2_4to9_gemma_bf16_ctx5')
items={}
for l in open(E+'ch7__gemma_bf16_ctx5_sel303.jsonl'):
    r=json.loads(l); items[r['item_id']]=r
ids=list(items); sid={i:items[i]['book_sentence_id'] for i in ids}
def cost(d):
    u=collections.Counter()
    for l in open(f'runs/{d}/outputs.jsonl',encoding='utf-8'):
        for k,v in (json.loads(l).get('usage') or {}).items():
            if isinstance(v,(int,float)): u[k]+=v
    return dict(u), u['input_tokens']*2/1e6+u['output_tokens']*10/1e6+u['cache_read_input_tokens']*.1/1e6+u['cache_creation_input_tokens']*2.5/1e6
if __name__=='__main__':
    dist=lambda R: dict(collections.Counter(lab(R.get(i)) for i in ids))
    print('처음(문단만)',dist(O)); print('A',dist(A)); print('B',dist(B))
    print('형식 오류 A',[sid[i] for i in ids if not A.get(i)],'B',[sid[i] for i in ids if not B.get(i)])
    both=[i for i in ids if A.get(i) and B.get(i)]
    print('A→B',dict(collections.Counter((A[i][0],B[i][0]) for i in both)))
    fl=[i for i in both if (A[i][0]=='none')!=(B[i][0]=='none')]
    print('갈림',len(fl),'A만',sum(A[i][0]!='none' for i in fl),'B만',sum(B[i][0]!='none' for i in fl),'/',len(both))
    for d in ('gatsby_ch7_sel303_A_prev5next5','gatsby_ch7_sel303_B_prev5next5_profile_v1sys'):
        u,c=cost(d); print(d,u,'$%.2f'%c)
    hits=[i for i in ids if B.get(i) and '프로필' in json.dumps(B[i][3]['output'],ensure_ascii=False)]
    print('B 출력에 "프로필" 명시',len(hits),[sid[i] for i in hits])
    for nme,R in (('A',A),('B',B)): print(nme,'축별',dict(collections.Counter(a for i in ids if R.get(i) for a in R[i][1])))
