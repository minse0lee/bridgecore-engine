"""가이드라인 10/7판 → 수정 초안 (변경 내용 추적). 사용: python build.py work/word/document.xml"""
import copy, sys
from lxml import etree

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
ns = {'w': W}
q = lambda t: '{%s}%s' % (W, t)
AUTH, DATE = '이민서', '2026-10-27T12:00:00Z'
_id = [9000]


def nid():
    _id[0] += 1
    return str(_id[0])


def mark(tag):
    e = etree.Element(q(tag))
    e.set(q('id'), nid()); e.set(q('author'), AUTH); e.set(q('date'), DATE)
    return e


def rpr(bold=False, sz=None, color=None):
    r = etree.Element(q('rPr'))
    if bold:
        etree.SubElement(r, q('b')); etree.SubElement(r, q('bCs'))
    if color:
        etree.SubElement(r, q('color')).set(q('val'), color)
    if sz:
        etree.SubElement(r, q('sz')).set(q('val'), str(sz))
    etree.SubElement(r, q('lang')).set(q('eastAsia'), 'ko-KR')
    return r


def run(text, bold=False, sz=None, color=None, deleted=False):
    r = etree.Element(q('r'))
    r.append(rpr(bold, sz, color))
    t = etree.SubElement(r, q('delText' if deleted else 't'))
    t.text = text
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    return r


def segs(x):
    """'**굵게**보통' 형식을 [(text,bold)]로."""
    out, bold = [], False
    for i, part in enumerate(x.split('**')):
        if part:
            out.append((part, i % 2 == 1))
    return out


def ins_runs(text, sz=None, color=None, force_bold=False, comment=None):
    """삽입 run 묶음(w:ins) 리스트. comment가 있으면 범위 표시 포함."""
    out = []
    if comment is not None:
        cs = etree.Element(q('commentRangeStart')); cs.set(q('id'), str(comment)); out.append(cs)
    ins = mark('ins')
    for t, b in segs(text):
        ins.append(run(t, b or force_bold, sz, color))
    out.append(ins)
    if comment is not None:
        ce = etree.Element(q('commentRangeEnd')); ce.set(q('id'), str(comment)); out.append(ce)
        ins2 = mark('ins')
        r = etree.SubElement(ins2, q('r'))
        etree.SubElement(r, q('commentReference')).set(q('id'), str(comment))
        out.append(ins2)
    return out


def new_p(text, style=None, cell=False, sz=None, color=None, bold=False, comment=None, after=None):
    p = etree.Element(q('p'))
    pp = etree.SubElement(p, q('pPr'))
    if style:
        etree.SubElement(pp, q('pStyle')).set(q('val'), style)
    if cell:
        sp = etree.SubElement(pp, q('spacing')); sp.set(q('after'), '20'); sp.set(q('line'), '259'); sp.set(q('lineRule'), 'auto')
    elif after is not None:
        etree.SubElement(pp, q('spacing')).set(q('after'), str(after))
    pr = etree.SubElement(pp, q('rPr'))
    pr.append(mark('ins'))
    etree.SubElement(pr, q('lang')).set(q('eastAsia'), 'ko-KR')
    for e in ins_runs(text, sz, color, bold, comment):
        p.append(e)
    return p


def cell(width, fill, paras, header=False, comment=None):
    tc = etree.Element(q('tc'))
    pr = etree.SubElement(tc, q('tcPr'))
    w = etree.SubElement(pr, q('tcW')); w.set(q('w'), str(width)); w.set(q('type'), 'dxa')
    sh = etree.SubElement(pr, q('shd')); sh.set(q('val'), 'clear'); sh.set(q('color'), 'auto'); sh.set(q('fill'), fill)
    m = etree.SubElement(pr, q('tcMar'))
    for side, v in (('top', 75), ('left', 95), ('bottom', 75), ('right', 95)):
        e = etree.SubElement(m, q(side)); e.set(q('w'), str(v)); e.set(q('type'), 'dxa')
    etree.SubElement(pr, q('vAlign')).set(q('val'), 'center')
    if isinstance(paras, str):
        paras = [paras]
    for i, t in enumerate(paras):
        tc.append(new_p(t, cell=True, sz=19, color='FFFFFF' if header else None, bold=header,
                        comment=comment if i == 0 else None))
    return tc


def row(widths, cells, header=False, fill='FFFFFF', comments=None):
    tr = etree.Element(q('tr'))
    tp = etree.SubElement(tr, q('trPr'))
    etree.SubElement(tp, q('cantSplit'))
    if header:
        etree.SubElement(tp, q('tblHeader'))
    etree.SubElement(tp, q('jc')).set(q('val'), 'center')
    tp.append(mark('ins'))
    for i, (wd, c) in enumerate(zip(widths, cells)):
        cm = comments.get(i) if comments else None
        tr.append(cell(wd, '243B53' if header else fill, c, header, cm))
    return tr


def table(tmpl, widths, header, rows, comments=None):
    tb = etree.Element(q('tbl'))
    tb.append(copy.deepcopy(tmpl.find('w:tblPr', ns)))
    g = etree.SubElement(tb, q('tblGrid'))
    for wd in widths:
        etree.SubElement(g, q('gridCol')).set(q('w'), str(wd))
    tb.append(row(widths, header, header=True))
    for i, r in enumerate(rows):
        tb.append(row(widths, r, fill='FFFFFF' if i % 2 == 0 else 'F3F6F8', comments=(comments or {}).get(i)))
    return tb


def spacer():
    p = etree.Element(q('p'))
    pp = etree.SubElement(p, q('pPr'))
    etree.SubElement(pp, q('spacing')).set(q('after'), '0')
    pr = etree.SubElement(pp, q('rPr')); pr.append(mark('ins'))
    return p


def append_p(p, text, comment=None):
    for e in ins_runs(text, comment=comment):
        p.append(e)


def text_of(r):
    return ''.join(t.text or '' for t in r.findall('w:t', ns))


def replace_in_p(p, old, new, comment=None):
    """p 안의 한 run에서 old를 삭제 표시하고 new를 삽입 표시."""
    for r in p.findall('w:r', ns):
        tx = text_of(r)
        if old in tx:
            i = tx.index(old)
            before, after = tx[:i], tx[i + len(old):]
            base = r.find('w:rPr', ns)
            def mk(t, deleted=False):
                nr = etree.Element(q('r'))
                if base is not None:
                    nr.append(copy.deepcopy(base))
                e = etree.SubElement(nr, q('delText' if deleted else 't')); e.text = t
                e.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
                return nr
            seq = []
            if before:
                seq.append(mk(before))
            d = mark('del'); d.append(mk(old, True)); seq.append(d)
            if new:
                if comment is not None:
                    cs = etree.Element(q('commentRangeStart')); cs.set(q('id'), str(comment)); seq.append(cs)
                ins = mark('ins'); ins.append(mk(new)); seq.append(ins)
                if comment is not None:
                    ce = etree.Element(q('commentRangeEnd')); ce.set(q('id'), str(comment)); seq.append(ce)
                    i2 = mark('ins'); rr = etree.SubElement(i2, q('r'))
                    etree.SubElement(rr, q('commentReference')).set(q('id'), str(comment)); seq.append(i2)
            if after:
                seq.append(mk(after))
            parent = r.getparent(); idx = parent.index(r)
            parent.remove(r)
            for k, e in enumerate(seq):
                parent.insert(idx + k, e)
            return
    raise SystemExit(f'못 찾음: {old!r} in {"".join(p.itertext())[:40]}')


def add_rows(tbl, widths, rows, comments=None):
    n = len(tbl.findall('w:tr', ns)) - 1  # 헤더 제외 기존 행 수
    for i, r in enumerate(rows):
        fill = 'FFFFFF' if (n + i) % 2 == 0 else 'F3F6F8'
        tbl.append(row(widths, r, fill=fill, comments=(comments or {}).get(i)))


def append_cell(tbl, ri, ci, text, comment=None):
    tc = tbl.findall('w:tr', ns)[ri].findall('w:tc', ns)[ci]
    p = tc.findall('w:p', ns)[-1]
    for e in ins_runs(' ' + text, sz=19, comment=comment):
        p.append(e)


def after(anchor, *els):
    for e in reversed(els):
        anchor.addnext(e)


# ---------------------------------------------------------------- 본문
path = sys.argv[1]
tree = etree.parse(path)
body = tree.getroot().find('w:body', ns)
B = list(body)  # 원본 순서 스냅숏
TMPL = B[54]

# 머리
replace_in_p(B[1], '파일럿 60문항 |  2026년 10월 ', '본검수 300문항 | 2026년 10월 수정 초안(파일럿판: 10월 ', comment=0)
replace_in_p(B[1], '일', '일)')
replace_in_p(B[2], '원문의 의미와 의도가 번역문에서 보존되는지를 여섯 축으로 확인합니다.',
             '평가조건(brief)에 명시된 번역 요구 아래에서 원문의 의미와 기능이 번역문에 보존되는지를 여섯 축으로 확인합니다.')
append_p(B[2], ' 판정은 개인의 감이나 번역 선호가 아니라 이 가이드라인의 판단 질문과 사례를 기준으로 합니다.')

# 1절
append_p(B[8], '대화문·편지 문항은 평가조건에 화자와 청자의 관계, 기대되는 격식이 함께 적혀 있습니다.', comment=8)
append_p(B[12], ' 각 축은 2절의 **축별 판단 질문**에 순서대로 답하여 판정합니다. 첫 질문은 언제나 "이 축에서 평가할 대상이 있는가"이며, 없으면 N/A입니다.')
append_p(B[14], ' 보존·N/A는 판단 질문에 따라 근거를 확인했다면 따로 기록하지 않습니다.')
replace_in_p(B[16], '묶음은 B01=P001~P010부터 B06=P051~P060까지입니다.', '묶음 번호와 문항 범위는 운영 시트를 따릅니다.')
after(B[17],
      new_p('작업은 나누어 진행합니다', style='21'),
      new_p('하루 1~2묶음까지만 평가하고, 묶음 사이에는 쉬는 시간을 둡니다. 속도보다 근거 확인을 우선하며, 집중이 떨어지면 쉬었다가 다음 묶음을 시작합니다. 평가자마다 묶음 순서가 다르게 배정되므로 운영 시트에 안내된 순서를 따릅니다. 본평가 전에는 연습 문제를 먼저 풀고 확정 판정과 이유를 확인합니다. 파일럿 작업 시간을 기준으로 본평가는 1인당 약 13~20시간이 걸릴 것으로 예상합니다.', comment=9))

# 2절: 축별 판단 질문
W3 = [1500, 2900, 5824]
judq = table(TMPL, W3, ['평가축', '평가 대상(없으면 N/A)', '판단 질문(순서대로)'], [
    ['Semantic',
     '평가대상 문장이 표현한 내용 전체',
     ['① 인물·대상·행위·수량·시간·부정·조건·인과·확실성 중 원문과 달라진 항목을 특정할 수 있습니까? 예 → 훼손',
      '② 달라진 것이 평가어의 정도나 어감뿐입니까? 예 → 보존 (예: overbearing → "압도적인"). 단, 추측·가능성이 단정으로 바뀌는 등 확실성이 달라졌으면 ①에 해당하므로 훼손입니다.',
      '표현 방식이나 어순이 달라도 같은 사태를 기술하면 보존입니다.']],
    ['Textual',
     '지시 표현(대명사·소유 관계), 문장 간 연결, 초점·범위 표현',
     ['① 대명사나 소유 관계가 가리키는 대상이 바뀌었거나, 무엇을 가리키는지 알 수 없게 되었습니까?',
      '② 대조·양보·조건·인과·시간 순서 같은 연결 관계가 바뀌었습니까? 연결어를 그대로 옮기지 않았어도 관계가 남아 있고 한국어로 자연스럽게 재구성되었으면 보존입니다.',
      '③ only·even·as well·let alone처럼 초점이나 범위를 정하는 표현이 다른 대상에 붙거나 사라졌습니까?',
      '하나라도 예 → 훼손. 내용이 단순히 빠진 것은 Semantic에서만 판정합니다.']],
    ['Context',
     '그 문장이 문맥에서 하는 일(설명·요청·지시·질문·경고·제안·반어 등)',
     ['① 원문의 발화 기능이 번역문에서 다른 종류의 기능으로 바뀌었습니까? (지시 → 가능성 설명, 요청 → 능력 질문, 질책하는 수사의문 → 실제 질문 등) 예 → 훼손',
      '② 기능의 종류는 같고 강도나 말투만 달라졌습니까? 예 → 보존. 의무·권고의 강도는 Deontic, 높임·격식은 Register에서 판정합니다.',
      '번역이 독자에게 주는 실제 효과(설득력, 감정적 인상 등)는 판정하지 않습니다.']],
    ['Deontic',
     '누군가에게 의무·허가·금지·규범적 권고·자격 조건을 정하는 표현 (5절 "Deontic을 적용하는 문장" 표)',
     ['① 의무·허가·금지 사이의 범주가 바뀌었거나, 요구가 선택 사항으로 읽힙니까?',
      '② 규범을 받는 주체, 조건, 예외, 적용 범위가 바뀌었습니까?',
      '하나라도 예 → 훼손. should를 "-해야 합니다"로 옮기는 것 같은 통상적 대응은 강도 훼손이 아닙니다.']],
    ['Register',
     '화자·청자·독자의 관계가 드러나는 높임·호칭·격식·태도',
     ['① 번역문의 높임·호칭·격식이 평가조건과 문맥에 나타난 관계에 맞습니까? 영어에는 한국어식 높임 체계가 없으므로, 원문에 높임 표현이 있었는지가 아니라 그 관계에 맞는 높임인지로 판정합니다.',
      '② 호칭이 관계를 바꾸거나 원문에 없는 역할을 덧붙였습니까? 친족 호칭이 실제 친족이 아닌 사람에게 친근하게 쓰인 경우도 확인합니다.',
      '③ 비하·경멸·친밀함 같은 화자의 태도가 사라지거나 약해졌습니까?',
      '관계를 알 수 없으면 판단보류로 두고 필요한 관계 정보를 적습니다. 어색함만으로는 훼손이 아닙니다.']],
    ['Cultural',
     '그 문화·제도에서 고유하게 쓰이는 대상: 제도·기관명과 분류, 법령 편제, 도량형·기간 단위, 날짜·통화 표기, 관습·예법, 비칭, 역사적 사물. 인명·지명만 있으면 N/A입니다.',
     ['① 번역이 원문과 같은 대상·제도를 가리킵니까? 다른 제도·단위로 바꾸었거나 다른 것으로 오역했으면 훼손입니다.',
      '② 한국어 독자가 문맥에서 그 의미를 이해할 수 있습니까? 통용 표기나 정확한 음역은 보존이고, 음역 때문에 무엇을 가리키는지 알 수 없게 되었으면 훼손입니다.',
      '③ 날짜·통화·단위 표기가 평가조건이 요구한 로케일에 맞습니까?']],
], comments={0: {2: 1}, 5: {1: 3, 2: 4}})
after(B[25],
      new_p('축별 판단 질문', style='21'),
      new_p('각 축은 아래 질문에 순서대로 답하여 판정합니다. 먼저 "평가 대상" 칸에 해당하는 표현이 평가대상 문장에 있는지 확인하고, 없으면 N/A를 선택합니다. 질문에 답하는 과정이 보존 판정의 근거 확인입니다.'),
      judq)
append_p(B[24], ' 축별 N/A 조건은 아래 "축별 판단 질문"의 평가 대상 칸을 따릅니다.')

# 3절 심각도
after(B[31],
      new_p('minor와 major는 두 질문으로 가립니다. ① 번역문만 읽은 독자가 문장의 요지를 다르게 이해하게 됩니까? ② 독자가 해야 할 행동이나 그 조건·대상을 다르게 이해하게 됩니까? **하나라도 예이면 major, 둘 다 아니면 minor**입니다.'),
      new_p('critical은 평가조건에 명시된 사용 맥락에서 안전·권리·중대한 결정에 피해를 줄 근거가 있을 때만 선택합니다. 예를 들어 학습자료에 실린 응급 처치 설명은 평가조건이 실제 처치 지침으로 쓰인다고 밝히지 않았다면 major까지 검토합니다. 법령·문헌의 식별 번호나 수치가 그대로 남아 독자가 원 자료를 찾을 수 있다면, 그 앞의 용어 오역은 대개 minor입니다.'))

# 4절
append_p(B[44], ' 판단 질문에 순서대로 답하는 과정이 보존 판정의 근거 확인이므로, 보존에 별도의 이유를 적지 않아도 됩니다.')

# 5절 Semantic과 Context 표에 행 추가
add_rows(B[48], [4320, 5904], [
    [['Who do you think you are?', '→ 당신은 자신이 누구라고 생각하나요?', '(상대를 꾸짖는 장면)'],
     '원문은 대답을 바라지 않고 상대를 꾸짖는 수사의문입니다. 번역문은 상대의 생각을 실제로 묻는 질문으로 읽히므로 Context는 훼손입니다. 묻는 내용 자체는 같으므로 Semantic은 보존입니다.'],
], comments={0: {1: 7}})

# 5절 Context와 Deontic
deon = table(TMPL, [3300, 4124, 2800], ['문장 유형', '설명용 원문', 'Deontic'], [
    ['자격·요건을 정하는 문장', 'Only full-time students qualify for the grant.', '적용'],
    ['받을 권리·지급을 정하는 문장', 'Each recipient receives $500 per semester.', '적용'],
    ['독자에게 하는 조언·권고 (규범적 근거가 있는 행동 지침)', 'It is important that parents review the school rules.', '적용'],
    ['문학·대화 속이라도 화자가 스스로나 상대에게 의무·금지를 거는 문장', '"I must not tell anyone," she whispered.', '적용'],
    ['규정을 설명하는 계산 예시나 사례 서술', 'In this example, the student took 12 credits in the fall.', 'N/A'],
    ['대화 속 제안·선택 질문, 개인적 부탁이나 바람', '"Shall we leave now?"', 'N/A'],
    ['이용 방법 안내 (기능은 Context에서 판정)', 'Use the search bar to find a book.', 'N/A'],
    ['학습 활동 지시 (기능은 Context에서 판정)', 'Read the passage and answer the questions.', 'N/A'],
    ['앞 문장의 규범을 이어받기만 하고 그 문장에는 규범 표현이 없는 문장', 'This includes meals and lodging.', 'N/A'],
], comments={0: {0: 2}})
modal = table(TMPL, [2400, 3600, 4224], ['원문 표현', '통상적인 한국어 대응', '주의할 점'], [
    ['must (의무)', '-해야 한다', '추론의 must("must be tired")는 규범 의미가 아닙니다.'],
    ['must not', '-해서는 안 된다', '"-할 필요 없다", "-않아도 된다"로 옮기면 금지가 비의무로 바뀌어 훼손입니다.'],
    ['need not / do not have to', '-할 필요 없다, -않아도 된다', '"-해서는 안 된다"로 옮기면 비의무가 금지로 바뀝니다.'],
    ['may (허가)', '-해도 된다, -할 수 있다', '가능성의 may("may rain")와 구분합니다.'],
    ['should (권고)', '-해야 한다, -하는 것이 좋다', '예상의 should("should arrive soon")와 구분합니다.'],
])
after(B[52],
      new_p('의무가 권고로, 금지가 허용으로 바뀌는 것처럼 규범의 범주나 강도가 달라진 것은 Deontic에서 판정합니다. Context는 발화 기능의 종류(지시·요청·질문·설명·경고 등)가 바뀌었을 때만 훼손입니다.', comment=6),
      new_p('**예)** 실험실 안전 수칙의 "Students must wear eye protection."이 "학생들은 보안경을 착용하는 것이 좋습니다."로 번역되면, 의무가 권고로 약해졌으므로 **Deontic 훼손**입니다. 행동을 지시하는 기능은 남아 있으므로 **Context는 보존**입니다. 반면 학습 활동 지시 "Label each part of the diagram."이 "그림의 각 부분에 이름을 붙일 수 있습니다."로 번역되면 지시가 가능성 설명으로 바뀌므로 **Context 훼손**이고, 학습 활동 지시이므로 **Deontic은 N/A**입니다.'),
      new_p('Deontic을 적용하는 문장', style='21'),
      new_p('평가대상 문장이 누군가에게 의무·허가·금지·규범적 권고·자격 조건을 정하는지로 판단합니다. 권고는 규범적 근거가 있는 행동 지침만 해당하며, 화자의 개인적 취향이나 일어날 일에 대한 예상은 포함하지 않습니다.'),
      deon, spacer(),
      new_p('아래는 자주 나오는 조동사의 통상적인 대응입니다. 고정된 번역표가 아니라, 문맥을 확인한 뒤 참고하는 예시입니다.'),
      modal, spacer())

# 5절 여러 축 표
tb54 = B[54]
append_cell(tb54, 1, 1, 'please·kindly 같은 공손 표지를 단어로 옮기지 않았더라도 높임이나 종결어미로 공손함이 전달되면 보존입니다.')
append_cell(tb54, 2, 1, '내용이 단순히 빠진 것은 Semantic에서만 판정하고, 누락 때문에 지시·연결·초점 관계까지 바뀐 경우에만 Textual에도 기록합니다.')
append_cell(tb54, 3, 1, '단위를 환산하다가 숫자만 틀린 것은 Semantic에만 기록합니다. 단위 체계나 제도를 다른 것으로 바꾸어 한국어 독자가 다른 제도·단위로 이해하게 되면 Cultural에도 기록합니다. 판단 질문: "번역만 읽은 독자가 그 제도·단위를 다른 것으로 이해하게 됩니까?"')
add_rows(tb54, [2592, 7632], [
    ['Register와 Cultural', '비하·경멸 호칭이 중립적인 말로 바뀌면, 화자의 경멸하는 태도가 약해진 것은 Register에, 그 표현이 지닌 민족·집단 비하라는 문화적 의미가 사라진 것은 Cultural에 기록합니다.'],
], comments={0: {1: 5}})

# 5절 예시 추가 (Ex2 뒤)
after(B[58],
      new_p('**Ex3)** Textual 단독 훼손: "Prices rose, while wages fell."이 "물가가 오르는 동안 임금이 떨어졌다."로 번역된 경우입니다. 물가 상승과 임금 하락이라는 내용은 유지되므로 **Semantic은 보존**이지만, 두 사실을 대조하는 관계가 동시에 일어난 일의 시간 관계로 바뀌었으므로 **Textual 훼손**입니다. 반대로 "Because the sample was contaminated, the results were discarded."가 "샘플이 오염되어 결과를 폐기했다."로 번역되면 because를 그대로 옮기지 않았지만 인과가 "-어서"로 남아 있으므로 Textual은 보존입니다.', comment=10),
      new_p('**Ex4)** Context는 보존, Register는 훼손: 학교 안내문의 "Please contact the school office if you have any questions."가 "궁금한 거 있으면 학교 사무실에 연락해."로 번역된 경우입니다. 연락을 요청하는 기능은 유지되므로 **Context는 보존**입니다. 학교가 보호자에게 보내는 공식 안내를 반말로 바꾸었으므로 **Register 훼손**입니다.'),
      new_p('**Ex5)** Cultural 판정: 평가조건이 "한국 제도로 대체하지 않음"으로 정한 미국 학교 안내 자료에서 "Grade 9 students"를 "중학교 3학년 학생"으로 옮기면, 미국에서 고등학교 첫 학년인 9학년을 한국 학제로 바꾸어 독자가 중학생으로 이해하게 되므로 **Cultural 훼손**입니다. 미국식 날짜 "04/05/2026"(4월 5일)을 "2026년 5월 4일"로 옮기면 날짜가 달라졌으므로 **Semantic 훼손**이고, 로케일 표기를 잘못 해석했으므로 **Cultural도 훼손**입니다.'))

# 6절
append_p(B[68], ' 매우 어색하더라도 행위자·행위·대상을 원문과 맞춰 볼 수 있으면 일반 오류로 평가하고, 핵심 관계를 복원할 수 없을 때만 생성붕괴 검토를 요청합니다.')
append_p(B[70], ' 논의 후에도 합의가 되지 않은 칸은 최종판정을 판단보류로 두고, 변경사유에 "합의 불가"와 각자의 입장을 적습니다.')

tree.write(path, xml_declaration=True, encoding='UTF-8', standalone=True)
print('ok')
