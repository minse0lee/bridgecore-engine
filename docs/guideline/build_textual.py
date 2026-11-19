"""11/6 반영본 → 11/19 Textual 보완 (변경 추적). 사용: python build_textual.py <unpacked_dir>
평가자 판정 관행(단어 오역으로 생긴 지시·수식 변화는 Semantic에만)에 맞춰 Semantic/Textual 경계를 명시. 개발 100 평가자 다수결에서 확인한 관행: 한 명사구 안의 오역(P055·P090·P099 등)은 Semantic만, 문장 전체 구조 붕괴(P017·P052·P078)는 두 축 모두."""
import copy, sys
from lxml import etree
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'; ns={'w':W}
q=lambda t:'{%s}%s'%(W,t)
AUTH, DATE = '이민서', '2026-11-19T12:00:00Z'
_id=[19500]
def mark(tag):
    _id[0]+=1; e=etree.Element(q(tag)); e.set(q('id'),str(_id[0])); e.set(q('author'),AUTH); e.set(q('date'),DATE); return e
XS='{http://www.w3.org/XML/1998/namespace}space'
def runs_from(text, base_rpr):
    out=[]
    for k,part in enumerate(text.split('**')):
        if not part: continue
        r=etree.Element(q('r')); rp=copy.deepcopy(base_rpr) if base_rpr is not None else etree.Element(q('rPr'))
        for b in rp.findall('w:b',ns)+rp.findall('w:bCs',ns): rp.remove(b)
        if k%2==1:
            rp.insert(0, etree.Element(q('bCs'))); rp.insert(0, etree.Element(q('b')))
        r.append(rp); t=etree.SubElement(r,q('t')); t.text=part; t.set(XS,'preserve'); out.append(r)
    return out
def append_ins(p, text):
    rpr0=p.find('w:r/w:rPr',ns); ins=mark('ins')
    for r in runs_from(text, rpr0): ins.append(r)
    p.append(ins)
def ins_para(template, text):
    p=copy.deepcopy(template); rpr0=p.find('w:r/w:rPr',ns)
    for r in p.findall('w:r',ns): p.remove(r)
    for e in p.findall('w:ins',ns)+p.findall('w:del',ns): p.remove(e)
    ppr=p.find('w:pPr',ns)
    if ppr is None: ppr=etree.SubElement(p,q('pPr'))
    prpr=ppr.find('w:rPr',ns)
    if prpr is None: prpr=etree.SubElement(ppr,q('rPr'))
    prpr.insert(0, mark('ins'))
    ins=mark('ins')
    for r in runs_from(text, rpr0): ins.append(r)
    p.append(ins)
    for a in list(p.attrib):
        if a.endswith('paraId') or a.endswith('textId'): del p.attrib[a]
    return p
d=sys.argv[1]
tree=etree.parse(f'{d}/word/document.xml'); body=tree.getroot().find('w:body',ns)
txt=lambda e:''.join(e.itertext()).strip()
def find_p(prefix):
    hits=[p for p in body.iter(q('p')) if txt(p).startswith(prefix)]
    assert len(hits)==1,(prefix,len(hits)); return hits[0]
# 1) 축별 판단 질문 표의 Textual ①
append_ins(find_p('① 대명사나 소유 관계가 가리키는 대상이'),
  " 단, 단어나 구를 잘못 옮겨서 다른 대상이 된 것은 Semantic에서 판정합니다('여러 축에 함께 기록하는 경우' 참고).")
# 2) 여러 축에 함께 기록하는 경우: Semantic과 Textual
append_ins(find_p('내용 변화와 지시·논리·초점 관계의 훼손을 각각 설명할 수 있을 때'),
  " 단어나 구를 잘못 옮겨서 가리키는 대상이나 수식·소유 관계가 함께 달라진 것은 **Semantic에만** 기록합니다. 판단 질문: \"내용 오역을 바로잡으면 지시·연결·범위 문제도 함께 풀립니까?\" 그렇다면 Semantic에만 기록합니다. 내용 단어는 맞게 옮겼는데 대명사·연결어·초점 표현(only·even·more than 등)을 다르게 처리해서 관계가 달라진 경우에 Textual에 기록합니다. 단, 문장 전체의 구조가 무너져 절과 절이 어떻게 이어지는지 알 수 없게 된 경우는 단어를 바로잡아도 풀리지 않으므로 Semantic과 Textual에 함께 기록합니다.")
# 3) Ex3 뒤에 예시 추가
ex3=find_p('Ex3) Textual 단독 훼손')
new=ins_para(ex3, "**Ex3-1)** Semantic 단독 훼손(Textual 보존): \"The bank lent money to the startup, and the lender later demanded early repayment.\"이 \"은행이 스타트업에 돈을 빌려주었고, 이후 스타트업은 조기 상환을 요구했다.\"로 번역된 경우입니다. 'the lender'(은행)를 '스타트업'으로 잘못 옮겨 상환을 요구한 주체가 바뀌었으므로 **Semantic 훼손**입니다. 이 오역을 바로잡으면 누가 요구했는지도 함께 바로잡히므로 **Textual에는 기록하지 않습니다**. 반대로 \"Only the backup file is deleted when the disk is full.\"이 \"디스크가 가득 찼을 때만 백업 파일이 삭제된다.\"로 번역되면, 단어는 모두 맞게 옮겼지만 only가 다른 대상에 붙어 범위가 달라졌으므로 **Textual 훼손**입니다.")
# 빈 줄 문단이 Ex 사이에 있으면 같은 간격 유지
nxt=ex3.getnext()
if nxt is not None and nxt.tag==q('p') and txt(nxt)=='':
    sp=copy.deepcopy(nxt)
    for a in list(sp.attrib):
        if a.endswith('paraId') or a.endswith('textId'): del sp.attrib[a]
    ppr=sp.find('w:pPr',ns)
    if ppr is None: ppr=etree.SubElement(sp,q('pPr'))
    rp=ppr.find('w:rPr',ns)
    if rp is None: rp=etree.SubElement(ppr,q('rPr'))
    rp.insert(0,mark('ins'))
    nxt.addnext(new); new.addnext(sp)
else:
    ex3.addnext(new)
tree.write(f'{d}/word/document.xml', xml_declaration=True, encoding='UTF-8', standalone=True)
print('ok')
