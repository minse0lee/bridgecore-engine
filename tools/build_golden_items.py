"""평가폼 v3 '문항' 시트 → 엔진 입력 jsonl. 개발 100(items_dev100.jsonl)과 같은 형식.
python tools/build_golden_items.py --split 시험 --out data/golden/items_test200.jsonl
(--check: 개발 100을 다시 만들어 기존 파일과 같은지 확인)"""
import argparse, csv, json
from pathlib import Path
import openpyxl
ROOT=Path(__file__).resolve().parents[1]
FORM=ROOT/'data/golden/본평가_평가폼_300문항_v3_배포본_2026-11-13.xlsx'
SPLIT=ROOT/'data/golden/split_final_newids_2026-11-13.csv'
def items():
    ws=openpyxl.load_workbook(FORM,read_only=True,data_only=True)['문항']
    out=[]
    for r in ws.iter_rows(min_row=9,values_only=True):
        if not r[0] or not str(r[0]).startswith('P'): continue
        iid,doc,brief,sp,tp,s,t=[(str(c) if c is not None else '') for c in r[:7]]
        out.append({'item_id':iid,'doc_info':doc,'brief':brief,'src_paragraph':sp,'tgt_paragraph':tp,'src':s,'tgt':t})
    return out
def split():
    with open(SPLIT,encoding='utf-8-sig') as f:
        rd=csv.DictReader(f); return {r['new_id']:r['split'] for r in rd} if 'new_id' in rd.fieldnames else None
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--split'); ap.add_argument('--out'); ap.add_argument('--check',action='store_true'); a=ap.parse_args()
    its=items()
    import sys; sys.path.insert(0,str(ROOT/'tools'))
    from golden_labels import split_map
    sp=split_map()
    if a.check:
        old=[json.loads(l) for l in open(ROOT/'data/golden/items_dev100.jsonl',encoding='utf-8')]
        new=[i for i in its if sp.get(i['item_id'])=='개발']
        print('개발 재생성 일치:', old==new, len(old), len(new))
    if a.split:
        sel=[i for i in its if sp.get(i['item_id'])==a.split]
        Path(a.out).write_text(''.join(json.dumps(i,ensure_ascii=False)+'\n' for i in sel),encoding='utf-8'); print(a.split,len(sel),'→',a.out)
