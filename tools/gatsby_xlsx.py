"""개츠비 PoC 1·3장 번역·검증 결과 엑셀 (요약은 수식으로 상세 시트를 셈).
python tools/gatsby_xlsx.py OUT.xlsx"""
import json
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, ".")
sys.path.insert(0, "translation")
from bridgecore.constants import AXES, VIOLATED  # noqa: E402
from bridgecore.vote import load_group  # noqa: E402
import epub_tools as et  # noqa: E402

D = Path("data/book/gatsby")
OUT = Path(sys.argv[1])
F = "Arial"
HEAD = PatternFill("solid", fgColor="DDE4EE")
NOTE = Font(name=F, size=9, color="555555")
thin = Side(style="thin", color="BBBBBB")
BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
WRAP = Alignment(wrap_text=True, vertical="top")

items = {t: {json.loads(l)["item_id"]: json.loads(l) for l in open(D / f"eval/ch1_3__{t}.jsonl")} for t in ("lora", "base")}
G = {t: load_group([Path(f"runs/gatsby_ch13_{t}")]) for t in ("lora", "base")}
ok = lambda t, i: bool((G[t].get(i) or {}).get("output")) and not G[t][i].get("errors")


def collapsed(t, i):
    return G[t][i]["output"].get("evaluable") is False   # 생성 붕괴(반복·끊김·무관한 출력) → 평가 불가, critical로 셈


def axes_txt(t, i):
    if collapsed(t, i):
        return "생성붕괴(critical)"
    ax = G[t][i]["output"]["axes"]
    return ", ".join(f"{a}({ax[a].get('severity') or '-'})" for a in AXES if ax.get(a, {}).get("judgment") == VIOLATED)


def top_sev(t, i):
    if collapsed(t, i):
        return "critical"
    s = [G[t][i]["output"]["axes"][a].get("severity") for a in AXES if G[t][i]["output"]["axes"].get(a, {}).get("judgment") == VIOLATED]
    return "critical" if "critical" in s else "major" if "major" in s else "minor" if s else ""


def reasons(t, i):
    if collapsed(t, i):
        return "[평가 불가] " + (G[t][i]["output"].get("evaluable_note") or "")
    ax = G[t][i]["output"]["axes"]
    return "\n".join(f"[{a}] {ax[a].get('reason', '')}" for a in AXES if ax.get(a, {}).get("judgment") == VIOLATED)


def style_header(ws, row, n):
    for c in range(1, n + 1):
        cell = ws.cell(row, c)
        cell.font = Font(name=F, bold=True)
        cell.fill = HEAD
        cell.border = BOX
        cell.alignment = Alignment(wrap_text=True, vertical="center")


wb = Workbook()
summ = wb.active
summ.title = "요약"

# ── 장별 상세 시트
COLS = ["ID", "원문", "학습 전 번역", "학습 후 번역 (v1)", "학습 후 훼손 축", "학습 후 최고 심각도", "학습 후 엔진 근거",
        "학습 전 훼손 축", "학습 전 최고 심각도", "판정 상태"]
WID = [8, 48, 40, 40, 22, 11, 60, 22, 11, 12]
for ch, name in (("I", "1장"), ("III", "3장")):
    ws = wb.create_sheet(name)
    ws.append(COLS)
    style_header(ws, 1, len(COLS))
    for i, it in items["lora"].items():
        if it["doc_info"].split("장·절: ")[1] != ch:
            continue
        both = ok("lora", i) and ok("base", i)
        state = "완료" if both else "판정 실패(" + ", ".join(n for t, n in (("base", "학습 전"), ("lora", "학습 후")) if not ok(t, i)) + ")"
        ws.append([i, it["src"], items["base"][i]["tgt"], it["tgt"],
                   axes_txt("lora", i) if ok("lora", i) else "", top_sev("lora", i) if ok("lora", i) else "",
                   reasons("lora", i) if ok("lora", i) else "",
                   axes_txt("base", i) if ok("base", i) else "", top_sev("base", i) if ok("base", i) else "", state])
    for c, w in enumerate(WID, 1):
        ws.column_dimensions[get_column_letter(c)].width = w
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name=F, size=10)
            cell.alignment = WRAP
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(COLS))}{ws.max_row}"

# ── 요약 (수식)
summ["A1"] = "위대한 개츠비 PoC — 1·3장 번역 및 6축 검증 결과"
summ["A1"].font = Font(name=F, bold=True, size=14)
info = [
    "원문: Project Gutenberg #64317 (1925, 미국 퍼블릭 도메인) · 번역: MADLAD-400 10B, 학습 전 vs 학습 후(lora_v1, AI Hub 사회과학·교육 영→한 7.8만 문장) · 2026-11-11",
    "검증: BridgeCore 6축 검증엔진 v1.1 (Claude Sonnet 5.5, 1회 판정). 아래 수치는 사람이 확인하지 않은 엔진 판정입니다",
    "세는 기준: 학습 전·후 모두 판정이 끝난 문장만. 판정 실패(엔진 출력 형식 오류, 재시도 후에도 실패)는 각 장 시트 '판정 상태'에 표시",
]
for k, s in enumerate(info, 2):
    summ.cell(k, 1, s).font = NOTE
r0 = 6
hdr = ["항목", "1장 학습 전", "1장 학습 후", "3장 학습 전", "3장 학습 후", "합계 학습 전", "합계 학습 후"]
for c, h in enumerate(hdr, 1):
    summ.cell(r0, c, h)
style_header(summ, r0, len(hdr))
# 열 매핑: (시트, 훼손축 열, 심각도 열)
src = {"1장 학습 전": ("1장", "H", "I"), "1장 학습 후": ("1장", "E", "F"),
       "3장 학습 전": ("3장", "H", "I"), "3장 학습 후": ("3장", "E", "F")}
rows = [("판정 완료 문장 수", "n")] + [(f"{a} 훼손 문장", a) for a in AXES] + \
       [("생성 붕괴 문장 (반복·끊김 등, 평가 불가)", "생성붕괴"), ("훼손 1개 이상 문장 (생성 붕괴 포함)", "any"),
        ("major 이상 훼손 문장 (생성 붕괴 포함)", "major")]
for k, (label, kind) in enumerate(rows, r0 + 1):
    summ.cell(k, 1, label)
    for c, h in enumerate(hdr[1:5], 2):
        sh, col, sev = src[h]
        done = f"'{sh}'!$J:$J,\"완료\""
        if kind == "n":
            f = f"=COUNTIF('{sh}'!$J:$J,\"완료\")"
        elif kind == "any":
            f = f"=COUNTIFS('{sh}'!${col}:${col},\"?*\",{done})"
        elif kind == "major":
            f = f"=COUNTIFS('{sh}'!${sev}:${sev},\"major\",{done})+COUNTIFS('{sh}'!${sev}:${sev},\"critical\",{done})"
        else:
            f = f"=COUNTIFS('{sh}'!${col}:${col},\"*{kind}(*\",{done})"
        summ.cell(k, c, f)
    summ.cell(k, 6, f"=B{k}+D{k}")
    summ.cell(k, 7, f"=C{k}+E{k}")
last = r0 + len(rows)
# 비율 표
r1 = last + 2
summ.cell(r1, 1, "비율 (판정 완료 문장 대비)")
for c, h in enumerate(hdr[1:], 2):
    summ.cell(r1, c, h)
style_header(summ, r1, len(hdr))
for k, (label, kind) in enumerate(rows[1:], r1 + 1):
    src_row = r0 + 2 + (k - r1 - 1)
    summ.cell(k, 1, label)
    for c in range(2, 8):
        L = get_column_letter(c)
        summ.cell(k, c, f"=IF({L}${r0 + 1}=0,0,{L}{src_row}/{L}${r0 + 1})").number_format = "0%"
last2 = r1 + len(rows) - 1
for row in summ.iter_rows(min_row=r0, max_row=last2, max_col=7):
    for cell in row:
        cell.border = BOX
        if cell.font.bold is not True:
            cell.font = Font(name=F, size=10)

# 해석·패턴 (책 전체 번역에서 센 값, 하드코딩 → 출처 표시)
r2 = last2 + 2
notes = [
    ("해석", True),
    ("· 학습 후 번역은 major 이상 훼손 문장(생성 붕괴 포함) 비율이 학습 전보다 19.5%p 낮음 (같은 문장끼리 비교한 부트스트랩 95% 구간 −23.5~−15.5%p)", False),
    ("· 훼손이 1개 이상인 문장 비율도 16.5%p 낮음 (95% −20.4~−12.5%p). 생성 붕괴(반복·끊김·엉뚱한 출력)는 학습 전 21문장 → 학습 후 2문장", False),
    ("· 그래도 학습 후 문장의 약 절반에 훼손이 1개 이상, 약 5분의 1에 major 이상 훼손이 있음 → 출판하려면 사람 후편집이 필요한 수준", False),
    ("· 오류 대부분은 Semantic(뜻 변경·누락). Register(말투)는 엔진이 주로 minor로 판정", False),
    ("· 6축은 '뜻이 보존됐는지'를 보는 검증이라 문장이 어색한지(유창성)는 거의 반영되지 않음. 예: 학습 전 G0001은 매우 어색하지만 뜻이 판독 가능해 '보존'. 학습 전후의 자연스러움 차이는 표보다 큼(각 장 시트에서 직접 비교 권장)", False),
    ("", False),
    ("학습 데이터(사회과학·교육 문장)에서 온 패턴 — 책 전체 2,959문장 기준, 문자열 규칙으로 센 값", True),
    ("· 'Miss ○○'를 '○○ 선생님'으로 번역: 학습 후 18/33문장 (학습 전 3/33)", False),
    ("· 대사 속 의문문을 '~는가?'체로 번역: 학습 후 137/299문장 (학습 전 12/299) → 소설 대화가 딱딱해짐", False),
    ("· 번역이 원문 길이의 30% 미만(누락 의심, 원문 60자 초과 문장): 학습 후 16, 학습 전 18", False),
    ("", False),
    ("주의", True),
    ("· 엔진 정확도: 파일럿 60문항(사람 합의 정답)에서 축 단위 F1 약 0.7~0.8. 오탐·놓침이 있으므로 수치는 경향으로 봐 주세요", False),
    ("· 1·3장 외 2·4~9장도 번역은 되어 있으나(EPUB에 포함) 검증은 하지 않았습니다", False),
    ("· 번역 결과의 드문 글자 깨짐(<0xEB>… 형태의 바이트 토큰)은 디코딩 문제라 원래 글자로 복원했습니다 (학습 후 19곳, 학습 전 60곳)", False),
    ("· 검증 API 비용: 약 $13.3 (학습 전·후 각 638문장, 일반 호출)", False),
]
for k, (s, bold) in enumerate(notes, r2):
    cell = summ.cell(k, 1, s)
    cell.font = Font(name=F, size=10, bold=bold)
summ.column_dimensions["A"].width = 30
for c in range(2, 8):
    summ.column_dimensions[get_column_letter(c)].width = 13

# ── 주요 오류 사례
ex = wb.create_sheet("주요 오류 사례")
ex.append(["유형", "ID", "원문", "학습 후 번역 (v1)", "엔진 근거 (요약)"])
style_header(ex, 1, 5)
CASES = [
    ("인물 호칭 (학습 데이터 영향)", "G0217", "Semantic"), ("뜻 변경", "G0095", "Semantic"),
    ("대사 일부 누락", "G0381", "Context"), ("대사 누락", "G0175", "Context"),
    ("되묻는 말의 기능 변경", "G0401", "Context"), ("인과 관계 왜곡", "G0320", "Textual"),
    ("문장 미완결", "G0215", "Textual"), ("시대 어휘 (gay=유쾌한)", "G0437", "Cultural"),
    ("제도 용어 (대리 복무자)", "G0017", "Cultural"), ("군 편제 (First Division)", "G0428", "Cultural"),
    ("책 제목 음역", "G0168", "Cultural"),
]
for label, i, ax in CASES:
    if not ok("lora", i):
        continue
    ex.append([label, i, items["lora"][i]["src"], items["lora"][i]["tgt"], G["lora"][i]["output"]["axes"][ax]["reason"]])
for c, w in enumerate([22, 8, 50, 50, 60], 1):
    ex.column_dimensions[get_column_letter(c)].width = w
for row in ex.iter_rows(min_row=2):
    for cell in row:
        cell.font = Font(name=F, size=10)
        cell.alignment = WRAP
ex.freeze_panes = "C2"

# ── 전체 문단 대조 (검증 안 한 장 포함)
rows_s = et._load(D / "sentences.jsonl")
lb, _ = et._blocks_from_rows(rows_s, et._mt_map(D / "mt/translate_input__lora.jsonl"))
bb = {(b["file"], b["xpath"]): b["ko"] for b in et._blocks_from_rows(rows_s, et._mt_map(D / "mt/translate_input__base.jsonl"))[0]}
full = wb.create_sheet("전체 문단 대조")
full.append(["순번", "장", "원문", "학습 전 번역", "학습 후 번역 (v1)", "6축 검증"])
style_header(full, 1, 6)
k = 0
for b in lb:
    if b["skip"] == "pg_boilerplate" or b["kind"] != "body":
        continue
    k += 1
    full.append([k, b["chapter"], b["en"], bb[(b["file"], b["xpath"])], b["ko"], "함" if b["chapter"] in ("I", "III") else "안 함"])
for c, w in enumerate([7, 6, 55, 45, 45, 9], 1):
    full.column_dimensions[get_column_letter(c)].width = w
for row in full.iter_rows(min_row=2):
    for cell in row:
        cell.font = Font(name=F, size=10)
        cell.alignment = WRAP
full.freeze_panes = "C2"
full.auto_filter.ref = f"A1:F{full.max_row}"

wb.save(OUT)
print(OUT, {ws.title: ws.max_row for ws in wb})
