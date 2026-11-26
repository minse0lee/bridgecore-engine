"""개츠비 책 전체 번역 노트북(10_개츠비전체번역_gemma_학습전.ipynb) 생성기.  python build_notebook_10.py

06 노트북의 ctx5 설정(학습 없음, 앞 문단 5개, 생각 끔, greedy, bf16) 그대로 책 전체 2,954문장을 번역한다(끝의 표제·목차 중복 5줄 제외).
- 1·3장은 11/17 결과(gemma_ch1_3_bf16_ctx5.jsonl)를 올리면 다시 번역하지 않고 이어 붙임(이미 6축 검증한 번역과 같게 유지)
- 결과는 구글 드라이브에 문단 묶음마다 저장 → 세션이 끊겨도 다시 실행하면 이어서 함
번역 함수는 06 노트북 셀을 그대로 가져온다(설정이 같아야 1·3장 결과와 이어진다).
"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent
md = nbf.v4.new_markdown_cell; code = nbf.v4.new_code_cell
src06 = nbf.read(HERE / "06_개츠비번역_gemma_설정실험.ipynb", as_version=4).cells
install, load, funcs = src06[1].source, src06[5].source, src06[7].source
funcs = funcs.split("# 시험 1문단")[0]
load = load.replace("allb, idx = gt.blocks(rows, CHAPTERS)", "allb, idx = gt.blocks(rows, None)   # 책 전체")

cells = [
md("""# 위대한 개츠비 책 전체 번역 — Gemma 4 31B (학습 없음, A100 80GB)

설정은 11/17 `ctx5` 실험과 같음: 앞 문단 5개 문맥 · 생각 모드 끔 · greedy · bf16 원본 · 같은 지침.
책 전체 2,954문장(끝의 표제·목차 중복 5줄 제외), 예상 2.5~3시간(1·3장 638문장에 34.6분 걸렸음. 1·3장 결과를 올리면 그만큼 줄어듦).

**올릴 파일** (1번 셀 실행 시 파일 선택 창)
- 필수: `gemma_translate.py`, `sentences.jsonl`
- 권장: `gemma_ch1_3_bf16_ctx5.jsonl` (11/17 1·3장 결과) → 올리면 1·3장은 건너뜀. 이미 검증한 번역과 같은 문장을 유지하려면 꼭 올릴 것

**실행**: ① 설치 셀 → '런타임 → 세션 다시 시작' → 1번 셀부터 끝까지.
결과는 구글 드라이브 `MyDrive/bridgecore_gatsby/`에 계속 저장됨. 세션이 끊기면 1번부터 다시 실행(파일은 다시 안 올려도 됨) → 남은 문단만 이어서 번역.

드라이브를 못 쓰는 계정이면 자동으로 세션 안(`/content/bridgecore_gatsby/`)에 저장함. 이 경우 세션이 끊기면 결과가 사라지므로, 왼쪽 파일 탭에서 `out/gemma_full_bf16_ctx5.jsonl`을 중간에 한두 번 내려받아 둘 것. 끊긴 뒤에는 그 파일을 `gemma_ch1_3_bf16_ctx5.jsonl` 대신 올리면 이어서 번역함.

**결과 파일**: `gemma_full_bf16_ctx5.jsonl`, `run_info_full_ctx5.json` (끝나면 자동 다운로드)
"""),
code(install),
md("## 1. 설정·파일 올리기"),
code("""import json, sys, time, platform, inspect, re, shutil
from pathlib import Path
import torch
from google.colab import files, drive

NAME = 'full_ctx5'
CFG = dict(n_ctx=5, thinking=False)
CONFIGS = {NAME: CFG}
MODEL_ID = 'google/gemma-4-31B-it'
BS = 4
THINK_BUDGET = 4096

try:
    drive.mount('/content/drive')
    BASE = Path('/content/drive/MyDrive/bridgecore_gatsby')
except Exception as e:      # 드라이브를 못 쓰는 계정(공용 계정 등) → 세션 안에만 저장
    BASE = Path('/content/bridgecore_gatsby')
    print('드라이브 연결 안 됨 → 세션 안에 저장합니다. 세션이 끊기면 결과가 사라지니 4번 셀의 중간 저장 파일을 가끔 내려받으세요:', e)
OUT_DIR = BASE / 'out'; OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / 'gemma_full_bf16_ctx5.jsonl'
need = [n for n in ('gemma_translate.py', 'sentences.jsonl') if not (BASE / n).exists()]
if need or not OUT.exists():
    print('파일 선택 창에서 고르기:', need or '(필수 파일은 이미 있음)', '+ 권장 gemma_ch1_3_bf16_ctx5.jsonl — 없으면 취소')
    for name, data in files.upload().items():
        clean = re.sub(r' \\(\\d+\\)(?=\\.[^.]+$)', '', Path(name).name)
        if clean.endswith('.jsonl') and clean != 'sentences.jsonl':
            (OUT_DIR / 'prev_ch1_3.jsonl').write_bytes(data)
        else:
            (BASE / clean).write_bytes(data)

GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(torch.cuda.get_device_name(0), f'{GPU_GB:.0f}GB')
assert GPU_GB > 70, '80GB GPU가 아닙니다 — 1·3장 결과와 같은 원본(bf16)으로 돌려야 합니다'
assert (BASE / 'sentences.jsonl').exists() and (BASE / 'gemma_translate.py').exists(), '필수 파일 두 개를 올리세요'
sys.path.insert(0, str(BASE))
import gemma_translate as gt
assert 'n_ctx' in inspect.signature(gt.user_text).parameters, 'gemma_translate.py가 옛 버전입니다'
assert gt.PROMPT_HASH == '4f5cbcadc8d7', f'지침이 11/17과 다릅니다({gt.PROMPT_HASH}) — 같은 gemma_translate.py를 올리세요'

PREV = OUT_DIR / 'prev_ch1_3.jsonl'
if PREV.exists() and not OUT.exists():
    rows_prev = [json.loads(l) for l in open(PREV, encoding='utf-8') if l.strip()]
    assert all(r['mt_meta'].get('config') in ('ctx5', NAME) and r['mt_meta'].get('prompt_hash') == gt.PROMPT_HASH for r in rows_prev), \\
        '올린 1·3장 파일이 ctx5 결과가 아닙니다'
    shutil.copy(PREV, OUT); print('기존 결과', len(rows_prev), '문장을 이어 씀')
print('프롬프트', gt.PROMPT_HASH, '· 이미 번역된 문장', sum(1 for _ in open(OUT, encoding='utf-8')) if OUT.exists() else 0)
"""),
md("## 2. 모델 불러오기 (약 5분)"),
code(load.replace("CHAPTERS", "None")),
md("## 3. 번역 함수 (06 노트북과 동일)"),
code(funcs + """
# 시험 1문단 — 형식 확인 (저장하지 않음)
r = generate(idx[:1], CFG)[0]
print('오류', r['err'], '· 출력 토큰', r['out_tokens'])
for s in r['sents']:
    print('EN', s['en']); print('KO', r['got'].get(s['id']))
"""),
md("## 4. 책 전체 번역 (드라이브에 계속 저장, 끊기면 다시 실행해 이어서)"),
code("""info = run_config(NAME, CFG, OUT)
rows_out = [json.loads(l) for l in open(OUT, encoding='utf-8')]
ids = [r['id'] for r in rows_out]
assert len(ids) == len(set(ids)), '중복 문장이 있습니다'
print('번역', len(ids), '/', len(WANT), '· 빠짐', len(WANT - set(ids)), '· 빈 번역', sum(1 for r in rows_out if not (r['mt'] or '').strip()))
files.download(str(OUT)); files.download(str(OUT_DIR / f'run_info_{NAME}.json'))
fail = OUT_DIR / f'raw_fail_{NAME}.jsonl'
if fail.exists() and fail.stat().st_size:
    files.download(str(fail))
"""),
]
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
               "kernelspec": {"name": "python3", "display_name": "Python 3"}}
out = HERE / "10_개츠비전체번역_gemma_학습전.ipynb"
nbf.write(nb, out); print("→", out)
