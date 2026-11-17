"""학습 후 Gemma 개츠비 번역 노트북(08_개츠비번역_gemma_학습후.ipynb) 생성기.  python build_notebook_08.py

06 노트북(설정 실험)의 번역 코드를 그대로 쓰고, 원본(bf16) 모델에 07에서 학습한 어댑터를 얹어(합쳐서) 번역한다.
설정은 학습 전 비교 대상인 ctx5(앞 문단 5개, 생각 끔, greedy, 개츠비 지침)와 같다.
"""
import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
src = json.loads((HERE / "06_개츠비번역_gemma_설정실험.ipynb").read_text(encoding="utf-8"))
cells06 = ["".join(c["source"]) for c in src["cells"]]
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell


def sub(text, pairs):
    for a, b in pairs:
        assert text.count(a) == 1, a[:60]
        text = text.replace(a, b)
    return text


setup = sub(cells06[3], [
    ("RUN = ['ctx5', 'ctx5_think', 'ctx2_think']     # 돌릴 설정 (순서대로)\n", ""),
    ("CONFIGS = {'ctx5': dict(n_ctx=5, thinking=False), 'ctx5_think': dict(n_ctx=5, thinking=True),\n"
     "           'ctx2_think': dict(n_ctx=2, thinking=True), 'base': dict(n_ctx=2, thinking=False)}",
     "CFG = dict(n_ctx=5, thinking=False)          # 학습 전 비교 대상(ctx5)과 같은 설정\nNAME = 'lora_v1_ctx5'"),
    ("print('파일 선택 창에서 고르기: gemma_translate.py, sentences.jsonl (+ 선택: gemma_ch1_3_bf16.jsonl)')",
     "print('파일 선택 창에서 고르기: gemma_translate.py, sentences.jsonl, adapter_best.zip')"),
    ("assert GPU_GB > 70, '80GB GPU가 아닙니다 — 이 실험은 05와 같은 원본(bf16)이어야 비교가 됩니다'",
     "assert GPU_GB > 70, '80GB GPU가 아닙니다 — 학습 전 번역과 같은 원본(bf16)이어야 비교가 됩니다'"),
    ("PREV = OUT_DIR / 'gemma_ch1_3_bf16.jsonl'\n"
     "print('프롬프트', gt.PROMPT_HASH, '· 05 결과 올림' if PREV.exists() else '· 05 결과 없음(빈칸 채우기 건너뜀)')",
     "import zipfile\nADAPTER = BASE / 'adapter_best'\nassert (BASE / 'adapter_best.zip').exists(), 'adapter_best.zip을 올리세요 (드라이브 runs/gemma_lora_v1/)'\n"
     "zipfile.ZipFile(BASE / 'adapter_best.zip').extractall(ADAPTER)\n"
     "print('어댑터 파일', sorted(p.name for p in ADAPTER.iterdir()))\n"
     "assert (ADAPTER / 'adapter_config.json').exists(), '어댑터 zip 안에 adapter_config.json이 없습니다'\n"
     "print('프롬프트', gt.PROMPT_HASH)"),
])
load = sub(cells06[5], [
    ("model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map='cuda', token=HF_TOKEN).eval()",
     "model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map='cuda', token=HF_TOKEN)\n"
     "from peft import PeftModel\n"
     "model = PeftModel.from_pretrained(model, str(ADAPTER)).merge_and_unload().eval()   # 어댑터를 원본에 합침(번역 속도 유지)\n"
     "ADAPTER_CFG = json.loads((ADAPTER / 'adapter_config.json').read_text())"),
])
funcs = sub(cells06[7], [
    ("def run_config(name, cfg, out_path, only=None):\n    meta = {'model': MODEL_ID, 'runtime': 'colab_transformers_bf16',",
     "def run_config(name, cfg, out_path, only=None):\n    meta = {'model': MODEL_ID, 'adapter': 'gemma_lora_v1 (07, QLoRA nf4 학습 → bf16에 합침)',\n"
     "            'adapter_r': ADAPTER_CFG.get('r'), 'runtime': 'colab_transformers_bf16',"),
    ("'mt_system': f'gemma-4-31b-it-bf16-{name}',", "'mt_system': f'gemma-4-31b-it-bf16-{name}',"),
    ("# 시험 1문단 (생각 켬) — 형식 확인\nr = generate(idx[:1], CONFIGS['ctx5_think'])[0]",
     "# 시험 1문단 — 학습 후에도 형식(JSON)이 유지되는지 확인\nr = generate(idx[:1], CFG)[0]"),
])

cells = [
    md("""# 개츠비 1·3장 — 학습 후 Gemma 4 31B 번역 (A100 80GB)

07 노트북에서 학습한 어댑터(`adapter_best.zip`)를 **원본(bf16) 모델에 합쳐서** 번역한다.
설정은 학습 전 비교 대상인 **ctx5와 같음**(앞 문단 5개, 생각 끔, greedy, 개츠비 번역 지침) → 결과를 Claude에게 올리면 학습 전후 6축 비교

**올릴 파일** (1번 셀 실행 시 파일 선택 창에서 세 개 함께)
- `gemma_translate.py` (새 버전), `sentences.jsonl`
- `adapter_best.zip` — 개인 드라이브 `bridgecore_mt/runs/gemma_lora_v1/`에서 내려받은 것

**실행**: ① 설치 셀 → '런타임 → 세션 다시 시작' → 1번 셀부터 끝까지
**시간(추정)**: 모델 불러오기·합치기 5~10분 + 번역 35~45분
**결과**: 마지막 셀이 끝나면 왼쪽 파일 창 `/content/gemma/out/`에서 `gemma_ch1_3_bf16_lora_v1_ctx5.jsonl`, `run_info_lora_v1_ctx5.json`(있으면 `raw_fail_*`도) 내려받기
(자동 다운로드는 셀이 실행 중이면 밀릴 수 있어 파일 창에서 직접 받는 게 확실함)
"""),
    code(sub(cells06[1], [('accelerate\n', 'accelerate peft\n!pip -q uninstall -y torchao   # 옛 torchao가 있으면 peft가 어댑터를 못 불러옴\n')])),
    md("## 1. 설정·파일 올리기"),
    code(setup),
    md("## 2. 모델 불러오기 + 어댑터 합치기 (5~10분)"),
    code(load),
    md("## 3. 번역 함수 + 시험 1문단"),
    code(funcs),
    md("## 4. 전체 번역"),
    code("""out_path = OUT_DIR / f'gemma_ch1_3_bf16_{NAME}.jsonl'
info = run_config(NAME, CFG, out_path)
print('결과:', out_path)
for f in (out_path, OUT_DIR / f'run_info_{NAME}.json', OUT_DIR / f'raw_fail_{NAME}.jsonl'):
    if f.exists() and f.stat().st_size:
        files.download(str(f))
"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=src["metadata"])
out = HERE / "08_개츠비번역_gemma_학습후.ipynb"
nbf.write(nb, out)
print("→", out)
