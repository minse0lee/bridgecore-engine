"""변경 내용 추적을 모두 반영하고 메모 표시를 지운 깨끗한 본을 만든다. 사용: python accept_clean.py clean/word/document.xml"""
import sys
from lxml import etree
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
q=lambda t:'{%s}%s'%(W,t)
p=sys.argv[1]; t=etree.parse(p); root=t.getroot()
for d in list(root.iter(q('del'))):
    if d.getparent().tag==q('rPr'): raise SystemExit('문단 표시 삭제가 있음')
    d.getparent().remove(d)
for i in list(root.iter(q('ins'))):
    par=i.getparent()
    if par.tag in (q('rPr'),q('trPr')): par.remove(i); continue
    idx=par.index(i)
    for k,ch in enumerate(list(i)): par.insert(idx+k,ch)
    par.remove(i)
for tag in ('commentRangeStart','commentRangeEnd'):
    for e in list(root.iter(q(tag))): e.getparent().remove(e)
for r in list(root.iter(q('commentReference'))):
    run=r.getparent(); run.getparent().remove(run)
t.write(p,xml_declaration=True,encoding='UTF-8',standalone=True)
print('ok')
