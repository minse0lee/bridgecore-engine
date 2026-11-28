"""재번역 전후 비교 엑셀: ① 사람 확인용 50문장(전후 순서를 섞어 A/B로 가림) ② 정답 키 ③ Claude 판정 146문장 ④ 요약. 2026-11-28.

    python tools/gatsby_retrans_review_xlsx.py <out.xlsx>
"""
import csv
import json
import random
import sys

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, 'tools')
import gatsby_policy_pipeline_analysis as A  # noqa: E402

D = 'data/book/gatsby/'
SEED, N_HUMAN = 20261128, 50
FONT, BOLD = Font(name='Arial', size=10), Font(name='Arial', size=10, bold=True)
HEAD = PatternFill('solid', start_color='D9E1F2'); FILL = PatternFill('solid', start_color='FFFF00')
WRAP = Alignment(wrap_text=True, vertical='top')
KO = {'T': '맞는 지적', 'P': '맞지만 사소함', 'F': '엔진 오판', '+': '나아짐', '=': '비슷함', '-': '나빠짐',
      'none': '문제없음', 'minor': 'minor', 'major': 'major', 'err': '판정 없음'}


def sheet(wb, title, header, rows, widths):
    ws = wb.create_sheet(title)
    ws.append(header)
    for r in rows:
        ws.append(r)
    for row in ws.iter_rows():
        for c in row:
            c.font, c.alignment = FONT, WRAP
    for c in ws[1]:
        c.font, c.fill = BOLD, HEAD
    for i, w in enumerate(widths):
        ws.column_dimensions[chr(65 + i)].width = w
    ws.freeze_panes = 'A2'
    return ws


def main(out):
    rev = {r['id']: r for r in csv.DictReader(open(D + 'retrans/review_claude_pipe_policy_next5.tsv', encoding='utf-8'), delimiter='\t')}
    re_ = {json.loads(l)['id']: json.loads(l) for l in open(D + 'retrans/retrans_reason_pipe_policy_next5.jsonl', encoding='utf-8')}
    fl = {json.loads(l)['id']: json.loads(l) for l in open(D + 'retrans/flags_pipe_policy_next5.jsonl', encoding='utf-8')}
    body = [json.loads(l) for l in open(D + 'sentences.jsonl', encoding='utf-8')]
    body = [s for s in body if s['kind'] == 'body']; pos = {s['id']: n for n, s in enumerate(body)}
    F, _ = A.out(['gatsby_pipe_policy_next5_final_retrans146'], ['pipe_policy_next5_final__retrans146_prev5next5.jsonl'])
    ids = sorted(rev)
    ctx = lambda i: ' '.join(body[k]['en'] for k in range(max(0, pos[i] - 2), pos[i]))
    opinion = lambda i: '\n'.join(f"· {e.get('src_span')} → {e.get('tgt_span')}: {e.get('description')}" for e in fl[i]['errors']) or \
        '\n'.join(f"· [{f['axis']}] {f['reason']}" for f in fl[i]['flags'])

    wb = Workbook(); wb.remove(wb.active)
    rnd = random.Random(SEED)
    changed = [i for i in ids if re_[i]['changed']]
    pick = sorted(rnd.sample(changed, N_HUMAN))
    human, key = [], []
    for n, i in enumerate(pick, 1):
        swap = rnd.random() < 0.5                      # True면 A가 재번역 후
        a, b = (re_[i]['mt'], re_[i]['mt_prev']) if swap else (re_[i]['mt_prev'], re_[i]['mt'])
        human.append([n, i, ctx(i), re_[i]['en'], a, b, None, None])
        key.append([n, i, '후' if swap else '전', '전' if swap else '후', KO[rev[i]['retrans']]])
    ws = sheet(wb, '사람 확인용(50)', ['번호', '문장 id', '앞 문장(원문, 참고)', '원문', '번역 A', '번역 B', '더 나은 쪽 (A / B / 비슷함)', '메모'],
               human, [6, 10, 40, 45, 45, 45, 16, 30])
    dv = DataValidation(type='list', formula1='"A,B,비슷함"', allow_blank=True); ws.add_data_validation(dv); dv.add(f'G2:G{N_HUMAN + 1}')
    for r in range(2, N_HUMAN + 2):
        ws.cell(r, 7).fill = FILL
    sheet(wb, '정답 키(확인 후 보기)', ['번호', '문장 id', 'A는', 'B는', 'Claude 판정(재번역이)'], key, [6, 10, 8, 8, 20])
    full = [[i, re_[i]['en'], re_[i]['mt_prev'], re_[i]['mt'], fl[i]['top_severity'], opinion(i), KO[A.lab(F.get(i))],
             KO[rev[i]['flag']], KO[rev[i]['retrans']], rev[i]['note']] for i in ids]
    sheet(wb, 'Claude 판정(146)', ['문장 id', '원문', '재번역 전', '재번역 후', '처음 심각도', '엔진 검토 의견', '엔진 재검증', '엔진 지적은', '재번역은', '메모'],
          full, [10, 40, 40, 40, 9, 50, 10, 13, 9, 45])
    ws = wb.create_sheet('요약', 0)
    S = "'Claude 판정(146)'!"
    ws.append(['엔진 지적 \\ 재번역', '나아짐', '비슷함', '나빠짐', '합계'])
    for lab in ('맞는 지적', '맞지만 사소함', '엔진 오판'):
        r = ws.max_row + 1
        ws.append([lab] + [f'=COUNTIFS({S}$H$2:$H$147,$A{r},{S}$I$2:$I$147,{c}$1)' for c in 'BCD'] + [f'=SUM(B{r}:D{r})'])
    r = ws.max_row + 1
    ws.append(['합계'] + [f'=SUM({c}2:{c}{r - 1})' for c in 'BCDE'])
    ws.append([])
    ws.append(['엔진 재검증 \\ 재번역', '나아짐', '비슷함', '나빠짐', '합계'])
    h = ws.max_row
    for lab in ('문제없음', 'minor', 'major'):
        r = ws.max_row + 1
        ws.append([lab] + [f'=COUNTIFS({S}$G$2:$G$147,$A{r},{S}$I$2:$I$147,{c}${h})' for c in 'BCD'] + [f'=SUM(B{r}:D{r})'])
    ws.append([])
    for t in ('· 판정은 Claude가 원문·앞뒤 문장과 대조해 내린 것이며 사람 검수가 아님. 엔진과 같은 계열 모델이라 같은 종류의 실수를 할 수 있음.',
              "· '사람 확인용(50)' 시트: 실제로 바뀐 142문장에서 무작위 50문장(seed 20261128). 재번역 전·후를 A/B로 섞어 가렸음. 노란 칸에 더 나은 쪽을 고른 뒤 '정답 키' 시트와 대조.",
              '· 대상: 방침 + 뒤 5문단으로 번역한 603문장 가운데 검증엔진(6축 v1.1)이 짚은 146문장. 재번역은 Gemma 4 31B에 엔진 검토 의견을 주고 수행.'):
        ws.append([t])
    for row in ws.iter_rows():
        for c in row:
            c.font = FONT
    for rr in (1, h):
        for c in ws[rr]:
            c.font, c.fill = BOLD, HEAD
    ws.column_dimensions['A'].width = 24
    for c in 'BCDE':
        ws.column_dimensions[c].width = 10
    wb.save(out)
    print(out, '· 사람 확인용', len(pick), '· 전체', len(ids))


if __name__ == '__main__':
    main(sys.argv[1])
