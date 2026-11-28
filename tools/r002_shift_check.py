"""R002 평가 파일의 행 밀림 점검과 시험 200 지표(Fleiss kappa, 갈린 칸, 평가자 한 명 대 나머지 F1). 2026-11-27·28."""
import sys,json,collections
sys.path.insert(0,'tools'); sys.path.insert(0,'.')
import golden_labels as gl
AX=gl.AXES
NEW="data/golden/labels/R002_2026-11-28수정.xlsx"
OLD='data/golden/labels/R002_2026-11-22보충.xlsx'
sp=gl.split_map()
def load(f): return {(r['item_id'],r['axis']):r for r in gl.load_v3(f)[1] if sp.get(r['item_id'])=='시험'}
R1=load('data/golden/labels/R001_2026-11-20.xlsx'); R3=load('data/golden/labels/R003_2026-11-20.xlsx')
R2o=load(OLD); R2n=load(NEW)
keys=sorted(R1)
P=lambda n:f'P{n:03d}'
def shifted(R2,lo,hi):
    """lo..hi 행의 내용을 한 문항 뒤로 옮긴 추정본(11/27 방식): 문항 n의 판정 = 원래 n-1행의 것"""
    out=dict(R2)
    for n in range(lo+1,hi+1):
        for a in AX: out[(P(n),a)]=R2[(P(n-1),a)]
    return out
v=lambda r: r['judgment']=='훼손'
def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return 2*p*r/(p+r) if p+r else 0
def stats(R2,name):
    R={'R001':R1,'R002':R2,'R003':R3}
    full=[k for k in keys if all(R[x][k]['judgment'] not in (None,'판단보류') for x in R)]
    kb=gl.fleiss([["훼손" if v(R[x][k]) else "아님" for x in R] for k in full],["훼손","아님"])
    split=sum(1 for k in keys if len({v(R[x][k]) for x in R})>1)
    res={}
    for x in R:      # 한 명 대 나머지: 나머지 둘의 훼손 여부가 일치한 칸만 놓고 본 F1 (11/27 보고와 같은 정의)
        o=[y for y in R if y!=x]; tp=fp=fn=0
        for k in keys:
            g1,g2=v(R[o[0]][k]),v(R[o[1]][k])
            if g1!=g2: continue
            m=v(R[x][k]); tp+=m and g1; fp+=m and not g1; fn+=(not m) and g1
        res[x]=round(prf(tp,fp,fn),3)
    print(f'{name}: kappa {kb} · 갈린 칸 {split} · 한 명 대 나머지 F1 {res}')
    return R
if __name__=='__main__':
    stats(R2o,'보충본(11/22)')
    stats(shifted(R2o,183,231),'11/27 추정 보정(P183~231 한 칸씩)')
    stats(R2n,'새 파일(11/28)')
    stats(shifted(R2n,183,199),'새 파일 + P183~199도 한 칸씩 옮긴 추정')
