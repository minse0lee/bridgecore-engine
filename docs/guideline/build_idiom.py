"""11/1 반영본 → 11/6 관용 표현 보강 (변경 추적). 사용: python build_idiom.py <unpacked_dir>"""
import copy, sys, re
from lxml import etree
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'; ns={'w':W}
q=lambda t:'{%s}%s'%(W,t)
AUTH, DATE = '이민서', '2026-11-06T12:00:00Z'
_id=[19000]
def mark(tag):
    _id[0]+=1; e=etree.Element(q(tag)); e.set(q('id'),str(_id[0])); e.set(q('author'),AUTH); e.set(q('date'),DATE); return e
XS='{http://www.w3.org/XML/1998/namespace}space'

def runs_from(text, base_rpr):
    """'**굵게**' 표기를 run 목록으로. '\n'은 줄바꿈(br)."""
    out=[]
    for k,part in enumerate(text.split('**')):
        if not part: continue
        r=etree.Element(q('r')); rp=copy.deepcopy(base_rpr) if base_rpr is not None else etree.Element(q('rPr'))
        for b in rp.findall('w:b',ns)+rp.findall('w:bCs',ns): rp.remove(b)
        if k%2==1:
            rp.insert(0, etree.Element(q('bCs'))); rp.insert(0, etree.Element(q('b')))
        r.append(rp)
        lines=part.split('\n')
        for j,ln in enumerate(lines):
            if j: etree.SubElement(r,q('br'))
            t=etree.SubElement(r,q('t')); t.text=ln; t.set(XS,'preserve')
        out.append(r)
    return out

def ins_para(template, text):
    p=copy.deepcopy(template)
    rpr0=p.find('w:r/w:rPr',ns)
    for r in p.findall('w:r',ns): p.remove(r)
    ppr=p.find('w:pPr',ns)
    if ppr is None: ppr=etree.SubElement(p,q('pPr'))
    prpr=ppr.find('w:rPr',ns)
    if prpr is None: prpr=etree.SubElement(ppr,q('rPr'))
    prpr.insert(0, mark('ins'))          # 문단 기호도 삽입으로 표시
    ins=mark('ins')
    for r in runs_from(text, rpr0): ins.append(r)
    p.append(ins)
    for a in list(p.attrib):
        if a.endswith('paraId') or a.endswith('textId'): del p.attrib[a]
    return p

def ins_row(template_row, cells):
    tr=copy.deepcopy(template_row)
    for a in list(tr.attrib):
        if a.endswith('paraId') or a.endswith('textId'): del tr.attrib[a]
    trpr=tr.find('w:trPr',ns)
    if trpr is None: trpr=etree.Element(q('trPr')); tr.insert(0,trpr)
    trpr.append(mark('ins'))
    for tc,text in zip(tr.findall('w:tc',ns), cells):
        p=tc.find('w:p',ns)
        new=ins_para(p, text)
        tc.replace(p,new)
    return tr

d=sys.argv[1]
tree=etree.parse(f'{d}/word/document.xml'); body=tree.getroot().find('w:body',ns)
kids=list(body)
txt=lambda e:''.join(e.itertext()).strip()
i_h=next(i for i,e in enumerate(kids) if txt(e)=='Semantic과 Context')
head_t, body_t, table_t, spacer_t = kids[i_h], kids[i_h+1], kids[i_h+2], kids[i_h+3]
assert txt(kids[i_h+4])=='Context와 Deontic'

new=[]
new.append(ins_para(head_t,'관용 표현'))
P=[
 "관용 표현은 개별 단어의 표면적 대응이 아니라 문맥에서 나타내는 관습적 의미를 기준으로 평가합니다. 직역 등으로 행위·대상·상태·내용이 달라지거나 전달되지 않으면 **Semantic 훼손**으로 판정합니다. 일반적인 표현으로 풀어 옮겼더라도 의미와 필요한 기능이 유지되면, 관용적 형식이나 비유 이미지가 달라졌다는 이유만으로 훼손을 부여하지 않습니다.",
 "관용 표현이라는 사실이나 문화적 유래만으로 Cultural을 적용하지 않습니다. 해당 문맥과 평가조건에서 보존해야 할 구체적인 문화적 대상·제도·관습이 있고, 그 지시 대상이나 문화적 의미가 왜곡·소실된 경우에 **Cultural 훼손**을 판정합니다. 그런 대상이 없으면 Cultural은 N/A입니다. Semantic과 Cultural에 함께 기록할 때는 내용의 변화와 문화적 의미의 변화를 각각 설명합니다.",
 "심각도는 3절의 두 질문을 그대로 적용합니다. 직역 때문에 번역문만 읽은 독자가 요지를 다르게 이해하면 major, 요지는 전달되고 부수적인 의미만 흐려지면 minor입니다. 관용 표현을 풀어 옮기면서 말투·격식이 달라져 화자와 상대의 관계가 다르게 읽히면 Register에서 판단하고, 말투가 조금 달라진 정도는 훼손으로 보지 않습니다.",
 "관용 표현이 있는 문장은 다음 순서로 확인합니다. **① Semantic**: 관용 표현이 문맥에서 뜻하는 행위·대상·상태가 번역에서도 전달됩니까? **② Context**: 질책하는 반문·요청·경고 같은 원문의 발화 기능이 번역에서도 유지됩니까? **③ Cultural**: 보존해야 할 구체적인 문화적 대상·제도·관습이 있습니까? 없으면 관용구라는 이유만으로 적용하지 않습니다.",
]
for t in P: new.append(ins_para(body_t,t))
# 예시 표: 기존 표 복사 → 머리글 행 유지, 데이터 행 교체
tb=copy.deepcopy(table_t)
rows=tb.findall('w:tr',ns); header, tmpl=rows[0], rows[1]
for r in rows: tb.remove(r)
trpr=header.find('w:trPr',ns)
if trpr is None: trpr=etree.Element(q('trPr')); header.insert(0,trpr)
trpr.append(mark('ins'))
for tc in header.findall('w:tc',ns):
    p=tc.find('w:p',ns); tc.replace(p, ins_para(p, txt(p) and ('**'+txt(p)+'**')))
tb.append(header)
ROWS=[
 ("It's raining cats and dogs.\n→ 고양이와 개가 비처럼 내리고 있다.",
  "비가 매우 많이 온다는 관습적 의미가 전달되지 않고 다른 사태로 읽히므로 **Semantic 훼손**(major)입니다. 관용 표현이라는 이유만으로 Cultural을 적용하지 않으므로 **Cultural은 N/A**입니다."),
 ("It's raining cats and dogs.\n→ 비가 억수같이 쏟아진다.\n→ 비가 아주 많이 온다.",
  "비유 이미지는 달라졌지만 비가 매우 많이 온다는 의미가 유지되므로 두 번역 모두 **Semantic 보존**입니다."),
 ("(상대를 꾸짖는 장면)\nDo you really think you can pull one over on me?\n→ 나한테 하나를 끌어올릴 수 있다고 생각해 봐.",
  "'속이다'라는 행위가 전달되지 않으므로 **Semantic 훼손**입니다. 대답을 바라지 않고 꾸짖는 반문이 생각해 보라는 지시로 바뀌었으므로 **Context 훼손**입니다. 보존해야 할 특정 문화적 대상·관습이 없으므로 **Cultural은 N/A**입니다."),
]
for a,b in ROWS: tb.append(ins_row(tmpl,[a,b]))
new.append(tb)
sp=copy.deepcopy(spacer_t)
for a_ in list(sp.attrib):
    if a_.endswith('paraId') or a_.endswith('textId'): del sp.attrib[a_]
sp.find('w:pPr',ns).find('w:rPr',ns).insert(0,mark('ins'))
new.append(sp)
anchor=kids[i_h+3]
for e in reversed(new): anchor.addnext(e)

# '여러 축에 함께 기록하는 경우' 표의 Semantic과 Cultural 칸에 한 문장 추가
cell=None
for tc in body.iter(q('tc')):
    if txt(tc).startswith('의미와 문화·제도적 기능이 모두 달라졌다는'):
        cell=tc; break
p=cell.find('w:p',ns); rpr0=p.find('w:r/w:rPr',ns)
ins=mark('ins')
for r in runs_from(" 관용 표현이라는 이유만으로도 Cultural을 적용하지 않습니다('관용 표현' 참고).", rpr0): ins.append(r)
p.append(ins)
tree.write(f'{d}/word/document.xml', xml_declaration=True, encoding='UTF-8', standalone=True)
print('ok')
