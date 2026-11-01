"""확인이 끝난 [확인 대기] 메모(1~7)를 지우고, 0·10번 메모 문구를 고친다. 사용: python prep_comments.py build/word"""
import sys
from pathlib import Path
from lxml import etree

D = Path(sys.argv[1])
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
W14 = 'http://schemas.microsoft.com/office/word/2010/wordml'
DROP = {str(i) for i in range(1, 8)}
EDIT = {
    '0': '이 수정 초안은 변경 내용 추적으로 표시했습니다. 수안님께 확인받은 기준(D1~D8, 11/1 답변)을 모두 반영했습니다. '
         '[운영 확인] 메모는 본검수 운영 방식이 정해지면 숫자를 확정할 부분입니다.',
    '10': "[합의 후 추가] 세 평가자 합의 결과가 오면 '평가자 판정과 확정 판정'을 나란히 보여주는 사례를 추가합니다. "
          '새 예시는 엔진 점검용으로 떼어 둔 파일럿 20문항과 겹치지 않도록 모두 합성 예문으로 작성했습니다.',
}

ct = etree.parse(str(D / 'comments.xml'))
para_ids = set()
for c in list(ct.getroot()):
    cid = c.get(f'{{{W}}}id')
    if cid in DROP:
        for p in c.iter(f'{{{W}}}p'):
            para_ids.add(p.get(f'{{{W14}}}paraId'))
        ct.getroot().remove(c)
    elif cid in EDIT:
        ts = list(c.iter(f'{{{W}}}t'))
        ts[0].text = EDIT[cid]
        for t in ts[1:]:
            t.text = ''
ct.write(str(D / 'comments.xml'), xml_declaration=True, encoding='UTF-8', standalone=True)

# commentsExtended(paraId) → commentsIds(paraId→durableId) → commentsExtensible(durableId)
durable = set()
ids = etree.parse(str(D / 'commentsIds.xml'))
for e in list(ids.getroot()):
    pid = next((v for k, v in e.attrib.items() if k.endswith('}paraId')), None)
    if pid in para_ids:
        durable.add(next(v for k, v in e.attrib.items() if k.endswith('}durableId')))
        ids.getroot().remove(e)
ids.write(str(D / 'commentsIds.xml'), xml_declaration=True, encoding='UTF-8', standalone=True)
for name, key, pool in (('commentsExtended.xml', 'paraId', para_ids), ('commentsExtensible.xml', 'durableId', durable)):
    t = etree.parse(str(D / name))
    for e in list(t.getroot()):
        v = next((v for k, v in e.attrib.items() if k.endswith('}' + key)), None)
        if v in pool:
            t.getroot().remove(e)
    t.write(str(D / name), xml_declaration=True, encoding='UTF-8', standalone=True)
print('지운 메모', sorted(DROP), '· 문단', len(para_ids), '· durable', len(durable))
