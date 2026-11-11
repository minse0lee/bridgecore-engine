"""개츠비 PoC 1·3장 번역·검증 결과 엑셀: 학습 전 MADLAD · 학습 후 v1 · Claude 번역 비교 (요약은 수식으로 상세 시트를 셈).
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

SYS = [("base", "학습 전"), ("lora", "학습 후 v1"), ("claude", "Claude")]   # Claude: Sonnet 5.5, 문단 단위 프롬프트 번역 (translation/llm_translate.py)
items = {t: {json.loads(l)["item_id"]: json.loads(l) for l in open(D / f"eval/ch1_3__{t}.jsonl")} for t, _ in SYS}
G = {t: load_group([Path(f"runs/gatsby_ch13_{t}")]) for t, _ in SYS}
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


import random
import re

ids_all = list(items["lora"])
done_ids = [i for i in ids_all if all(ok(t, i) for t, _ in SYS)]
chap = {i: items["lora"][i]["doc_info"].split("장·절: ")[1] for i in ids_all}


def viol_any(t, i):
    return collapsed(t, i) or any(G[t][i]["output"]["axes"].get(a, {}).get("judgment") == VIOLATED for a in AXES)


def viol_major(t, i):
    return top_sev(t, i) in ("major", "critical")


def boot(f, a, b, n=5000):
    d = [int(f(b, i)) - int(f(a, i)) for i in done_ids]
    rng = random.Random(20261111)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return sum(d) / len(d), bs[int(n * .025)], bs[int(n * .975) - 1]


pct = lambda x: f"{x * 100:+.1f}%p"
cmp_lines = []
for a, b, la, lb in (("base", "lora", "학습 전", "학습 후 v1"), ("lora", "claude", "학습 후 v1", "Claude"), ("base", "claude", "학습 전", "Claude")):
    m = boot(viol_major, a, b); v = boot(viol_any, a, b)
    cmp_lines.append(f"· {lb} − {la}: major 이상 문장 비율 {pct(m[0])} (95% {pct(m[1])}~{pct(m[2])}), 훼손 1개 이상 {pct(v[0])} (95% {pct(v[1])}~{pct(v[2])})")

# 문자열 규칙 패턴 (1·3장 638문장, 세 번역 모두)
pat = {}
for t, _ in SYS:
    its = items[t].values()
    miss = [x for x in its if "Miss " in x["src"]]
    q = [x for x in its if re.search(r"[“\"].*\?", x["src"])]
    pat[t] = (sum("선생님" in x["tgt"] for x in miss), len(miss),
              sum(bool(re.search(r"(는가|인가|었는가|겠는가)\?", x["tgt"])) for x in q), len(q))

wb = Workbook()
summ = wb.active
summ.title = "요약"

# ── 장별 상세 시트
COLS = ["ID", "원문", "학습 전 번역 (MADLAD)", "학습 후 번역 (MADLAD v1)", "Claude 번역",
        "학습 전 훼손 축", "학습 전 최고 심각도", "학습 후 훼손 축", "학습 후 최고 심각도", "학습 후 엔진 근거",
        "Claude 훼손 축", "Claude 최고 심각도", "Claude 엔진 근거", "판정 상태"]
WID = [8, 46, 36, 36, 36, 18, 10, 18, 10, 46, 18, 10, 46, 12]
AXCOL = {"base": ("F", "G"), "lora": ("H", "I"), "claude": ("K", "L")}
for ch, name in (("I", "1장"), ("III", "3장")):
    ws = wb.create_sheet(name)
    ws.append(COLS)
    style_header(ws, 1, len(COLS))
    for i, it in items["lora"].items():
        if chap[i] != ch:
            continue
        bad = [lab for t, lab in SYS if not ok(t, i)]
        state = "완료" if not bad else "판정 실패(" + ", ".join(bad) + ")"
        g = lambda f, t: f(t, i) if ok(t, i) else ""
        ws.append([i, it["src"], items["base"][i]["tgt"], it["tgt"], items["claude"][i]["tgt"],
                   g(axes_txt, "base"), g(top_sev, "base"), g(axes_txt, "lora"), g(top_sev, "lora"), g(reasons, "lora"),
                   g(axes_txt, "claude"), g(top_sev, "claude"), g(reasons, "claude"), state])
    for c, w in enumerate(WID, 1):
        ws.column_dimensions[get_column_letter(c)].width = w
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name=F, size=10)
            cell.alignment = WRAP
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(COLS))}{ws.max_row}"

# ── 요약 (수식)
summ["A1"] = "위대한 개츠비 PoC — 1·3장 번역 3종 비교 (6축 검증)"
summ["A1"].font = Font(name=F, bold=True, size=14)
info = [
    "원문: Project Gutenberg #64317 (1925, 미국 퍼블릭 도메인) · 1·3장 638문장 · 2026-11-11",
    "번역: ① MADLAD-400 10B 학습 전 ② 학습 후 v1 (AI Hub 사회과학·교육 영→한 7.8만 문장 LoRA, 문장 단위) ③ Claude Sonnet 5.5 (문학 번역 지침 프롬프트, 문단 단위·앞 문단 문맥)",
    "검증: BridgeCore 6축 검증엔진 v1.1 (Claude Sonnet 5.5, 1회 판정). 사람이 확인하지 않은 엔진 판정입니다",
    "세는 기준: 세 번역 모두 판정이 끝난 문장만. 판정 실패(엔진 출력 형식 오류)는 각 장 시트 '판정 상태'에 표시. 생성 붕괴(반복·끊김 등 평가 불가)는 critical로 셈",
]
for k, s in enumerate(info, 2):
    summ.cell(k, 1, s).font = NOTE
r0 = 7
labs = [lab for _, lab in SYS]
hdr = ["항목"] + [f"1장 {l}" for l in labs] + [f"3장 {l}" for l in labs] + [f"합계 {l}" for l in labs]
for c, h in enumerate(hdr, 1):
    summ.cell(r0, c, h)
style_header(summ, r0, len(hdr))
rows = [("판정 완료 문장 수", "n")] + [(f"{a} 훼손 문장", a) for a in AXES] + \
       [("생성 붕괴 문장 (반복·끊김 등)", "생성붕괴"), ("훼손 1개 이상 문장", "any"), ("major 이상 훼손 문장", "major")]
for k, (label, kind) in enumerate(rows, r0 + 1):
    summ.cell(k, 1, label)
    for si, (sh) in enumerate(("1장", "3장")):
        for ti, (t, _) in enumerate(SYS):
            col, sev = AXCOL[t]
            c = 2 + si * 3 + ti
            done = f"'{sh}'!$N:$N,\"완료\""
            if kind == "n":
                f = f"=COUNTIF('{sh}'!$N:$N,\"완료\")"
            elif kind == "any":
                f = f"=COUNTIFS('{sh}'!${col}:${col},\"?*\",{done})"
            elif kind == "major":
                f = f"=COUNTIFS('{sh}'!${sev}:${sev},\"major\",{done})+COUNTIFS('{sh}'!${sev}:${sev},\"critical\",{done})"
            else:
                f = f"=COUNTIFS('{sh}'!${col}:${col},\"*{kind}(*\",{done})"
            summ.cell(k, c, f)
    for ti in range(3):
        a, b = get_column_letter(2 + ti), get_column_letter(5 + ti)
        summ.cell(k, 8 + ti, f"={a}{k}+{b}{k}")
last = r0 + len(rows)
r1 = last + 2
summ.cell(r1, 1, "비율 (판정 완료 문장 대비)")
for c, h in enumerate(hdr[1:], 2):
    summ.cell(r1, c, h)
style_header(summ, r1, len(hdr))
for k, (label, kind) in enumerate(rows[1:], r1 + 1):
    src_row = r0 + 2 + (k - r1 - 1)
    summ.cell(k, 1, label)
    for c in range(2, len(hdr) + 1):
        L = get_column_letter(c)
        summ.cell(k, c, f"=IF({L}${r0 + 1}=0,0,{L}{src_row}/{L}${r0 + 1})").number_format = "0%"
last2 = r1 + len(rows) - 1
for row in summ.iter_rows(min_row=r0, max_row=last2, max_col=len(hdr)):
    for cell in row:
        cell.border = BOX
        if cell.font.bold is not True:
            cell.font = Font(name=F, size=10)

r2 = last2 + 2
notes = [("같은 문장끼리 비교 (부트스트랩 5,000회, 판정 완료 " + str(len(done_ids)) + "문장, 이 파일을 만들 때 계산)", True)] + \
        [(s, False) for s in cmp_lines] + [("", False),
        ("문자열 규칙으로 센 패턴 (1·3장)", True)] + \
        [(f"· {lab}: 'Miss ○○' → '선생님' {pat[t][0]}/{pat[t][1]}문장, 대사 의문문 '~는가?' {pat[t][2]}/{pat[t][3]}문장", False) for t, lab in SYS] + [
        ("", False),
        ("주의", True),
        ("· 자기 평가 편향 가능성: Claude 번역을 같은 Claude Sonnet 5.5가 검증했습니다. LLM 판정자가 자기(같은 계열) 출력을 후하게 보는 경향이 보고되어 있어, Claude 번역 수치는 유리하게 나왔을 수 있습니다. 사람 확인으로 교차 검증이 필요합니다", False),
        ("· 조건 차이: MADLAD는 문장 하나씩, Claude는 문단 단위(앞 문단 문맥 포함)·고유명사 표기·말투 지침을 받아 번역했습니다. 모델 차이와 번역 방식 차이가 섞인 비교입니다", False),
        ("· 6축은 '뜻이 보존됐는지'를 보는 검증이라 문장의 자연스러움(유창성)은 거의 반영되지 않습니다. 각 장 시트에서 번역을 직접 비교해 보세요", False),
        ("· 엔진 정확도: 파일럿 60문항(사람 합의 정답)에서 축 단위 F1 약 0.7~0.8. 수치는 경향으로 봐 주세요", False),
        ("· MADLAD 번역의 드문 글자 깨짐(바이트 토큰)은 디코딩 문제라 원래 글자로 복원했습니다 (학습 후 19곳, 학습 전 60곳, 책 전체 기준)", False),
        ("· 비용(API): Claude 번역 약 $2.1, 6축 검증 3종 합계 약 $19.1 (학습 전 $6.8, 학습 후 $6.5, Claude $5.9). MADLAD 번역은 Colab, 추가 비용 없음", False),
]
for k, (s, bold) in enumerate(notes, r2):
    summ.cell(k, 1, s).font = Font(name=F, size=10, bold=bold)
summ.column_dimensions["A"].width = 30
for c in range(2, len(hdr) + 1):
    summ.column_dimensions[get_column_letter(c)].width = 11

# ── 주요 오류 사례 (MADLAD v1) + 같은 문장의 Claude 번역
ex = wb.create_sheet("주요 오류 사례")
ex.append(["유형", "ID", "원문", "학습 후 번역 (v1)", "v1 엔진 근거", "Claude 번역", "Claude 판정"])
style_header(ex, 1, 7)
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
    ex.append([label, i, items["lora"][i]["src"], items["lora"][i]["tgt"], G["lora"][i]["output"]["axes"][ax]["reason"],
               items["claude"][i]["tgt"], (axes_txt("claude", i) or "훼손 없음") if ok("claude", i) else "판정 실패"])
for c, w in enumerate([20, 8, 46, 40, 50, 40, 18], 1):
    ex.column_dimensions[get_column_letter(c)].width = w
for row in ex.iter_rows(min_row=2):
    for cell in row:
        cell.font = Font(name=F, size=10)
        cell.alignment = WRAP
ex.freeze_panes = "C2"

# ── 전체 문단 대조 (MADLAD, 검증 안 한 장 포함)
rows_s = et._load(D / "sentences.jsonl")
lb, _ = et._blocks_from_rows(rows_s, et._mt_map(D / "mt/translate_input__lora.jsonl"))
bb = {(b["file"], b["xpath"]): b["ko"] for b in et._blocks_from_rows(rows_s, et._mt_map(D / "mt/translate_input__base.jsonl"))[0]}
full = wb.create_sheet("전체 문단 대조 (MADLAD)")
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
print(OUT, {ws.title: ws.max_row for ws in wb}, "판정 완료", len(done_ids))
print("\n".join(cmp_lines)); print(pat)
