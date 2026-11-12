"""개츠비 1·3장 블라인드 비교표: 무작위 30문장, 세 번역(학습 전·학습 후 v1·Claude)을 문장마다 A/B/C로 섞음.
평가표(평가자용)와 정답표(분석용)를 따로 만든다.  python tools/gatsby_blind.py OUT_DIR"""
import json, random, sys
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
sys.path.insert(0, ".")
from bridgecore.vote import load_group

D = Path("data/book/gatsby"); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
SYS = ["base", "lora", "claude"]
items = {t: {json.loads(l)["item_id"]: json.loads(l) for l in open(D / f"eval/ch1_3__{t}.jsonl")} for t in SYS}
G = {t: load_group([Path(f"runs/gatsby_ch13_{t}")]) for t in SYS}
ok = lambda t, i: bool((G[t].get(i) or {}).get("output")) and not G[t][i].get("errors")
pool = [i for i in items["lora"] if all(ok(t, i) for t in SYS) and len(items["lora"][i]["src"].split()) >= 5]
rng = random.Random(20261113)
pick = sorted(rng.sample(pool, 30))
F = "Arial"; YEL = PatternFill("solid", fgColor="FFF8E6"); HEAD = PatternFill("solid", fgColor="DDE4EE")
thin = Side(style="thin", color="BBBBBB"); BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
wb = Workbook(); ws = wb.active; ws.title = "비교"
ws["A1"] = "개츠비 번역 블라인드 비교 (30문장)"; ws["A1"].font = Font(name=F, bold=True, size=13)
notes = ["각 문장마다 번역 A·B·C의 순서는 무작위입니다(어느 것이 어떤 번역인지 문장마다 다름). 앞 문맥은 참고용입니다.",
         "① 뜻 오류: 원문과 뜻이 달라졌거나 빠지거나 덧붙은 곳이 있으면 '있음'  ② 자연스러움: 1(매우 어색)~5(출판물처럼 자연스러움)  ③ 가장 좋은 번역 하나",
         "노란 칸만 입력합니다. 메모는 자유롭게 적어 주세요. 다 하면 파일을 그대로 돌려주세요."]
for k, s in enumerate(notes, 2):
    ws.cell(k, 1, s).font = Font(name=F, size=9, color="555555")
hdr = ["번호", "앞 문맥 (원문 문단)", "원문", "번역 A", "번역 B", "번역 C",
       "A 뜻 오류", "B 뜻 오류", "C 뜻 오류", "A 자연스러움", "B 자연스러움", "C 자연스러움", "가장 좋은 번역", "메모"]
ws.append([]); ws.append(hdr); hr = ws.max_row
for c in range(1, len(hdr) + 1):
    x = ws.cell(hr, c); x.font = Font(name=F, bold=True); x.fill = HEAD; x.border = BOX; x.alignment = Alignment(wrap_text=True, vertical="center")
key = Workbook(); kws = key.active; kws.title = "정답표"; kws.append(["번호", "문항ID", "A", "B", "C"])
for n, i in enumerate(pick, 1):
    order = SYS[:]; rng.shuffle(order)
    it = items["lora"][i]
    ws.append([n, it["src_paragraph"], it["src"]] + [items[t][i]["tgt"] for t in order] + [""] * 7 + [""])
    kws.append([n, i] + order)
r1, r2 = hr + 1, ws.max_row
for row in ws.iter_rows(min_row=r1, max_row=r2):
    for c in row:
        c.font = Font(name=F, size=10); c.alignment = Alignment(wrap_text=True, vertical="top"); c.border = BOX
        if c.column >= 7: c.fill = YEL
dv1 = DataValidation(type="list", formula1='"있음,없음"', allow_blank=True); dv1.add(f"G{r1}:I{r2}")
dv2 = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True); dv2.add(f"J{r1}:L{r2}")
dv3 = DataValidation(type="list", formula1='"A,B,C,비슷함"', allow_blank=True); dv3.add(f"M{r1}:M{r2}")
for dv in (dv1, dv2, dv3): ws.add_data_validation(dv)
for col, w in zip("ABCDEFGHIJKLMN", [5, 40, 36, 32, 32, 32, 9, 9, 9, 9, 9, 9, 10, 24]):
    ws.column_dimensions[col].width = w
ws.freeze_panes = f"D{r1}"
wb.save(OUT / "개츠비_블라인드비교_30문장.xlsx"); key.save(OUT / "개츠비_블라인드비교_정답표_평가자에게보내지말것.xlsx")
print(len(pool), pick[:5])
