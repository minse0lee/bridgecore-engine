"""Gemma 4 31B 개츠비 번역 노트북(05_개츠비번역_gemma4_31b.ipynb) 생성기.  python build_notebook_05.py

Colab A100·H100 80GB면 bf16 원본 그대로, A100 40GB면 4비트(NF4)로 줄여 띄워 번역한다(GPU 크기로 자동 선택).
번역 지침·문단 묶기·응답 파싱은 gemma_translate.py(API판)를 그대로 불러 써서 두 판이 어긋나지 않게 한다.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
md("""# 개츠비 1·3장 번역 — Gemma 4 31B (zero-shot, Colab A100)

**준비 (드라이브 `MyDrive/bridgecore_mt/gemma/`에 두 파일 올리기)**
- `gemma_translate.py` — 번역 지침·문단 묶기 (Claude 번역과 같은 지침)
- `sentences.jsonl` — 개츠비 문장 파일

**결과** `bridgecore_mt/gemma/out/gemma_ch1_3_bf16.jsonl` 또는 `gemma_ch1_3_nf4.jsonl` (+ `run_info_*.json`) → 이 파일을 Claude에게 올리면 6축 검증·비교

**GPU**: 80GB(A100·H100)면 원본 bf16(약 62GB), **A100 40GB면 자동으로 4비트(NF4, 약 18GB)**로 불러옴. 4비트는 원본과 계산이 조금 달라 번역이 약간 다를 수 있음(결과 파일 이름·mt_system에 nf4 표시). L4·T4 불가
**런타임 유형**: '런타임 → 런타임 유형 변경 → A100 GPU' + '높은 RAM'

**실행**: ① 설치 셀만 실행 → '런타임 → 세션 다시 시작' → 1번 셀부터 끝까지.
끊겨도 다시 실행하면 이미 번역한 문단은 건너뜀 (묶음마다 저장). 모델 다운로드는 세션마다 다시 함(약 62GB)
"""),
code("""# ① 맨 먼저 이 셀만 실행 → 끝나면 '런타임 → 세션 다시 시작' → 다음 셀부터
!pip list --format=freeze | grep -E '^(torch|torchvision|torchaudio)==' > /content/torch_pin.txt
!pip -q install -c /content/torch_pin.txt -U "transformers>=5.10.1" accelerate
!pip -q install -U "bitsandbytes>=0.46.1"
!pip -q uninstall -y torchao
!pip list --format=freeze | grep -E '^(transformers|torch|accelerate|bitsandbytes)=='
"""),
md("## 1. 설정"),
code("""import json, sys, time, platform
from pathlib import Path
import torch
from google.colab import drive
drive.mount('/content/drive')

BASE = Path('/content/drive/MyDrive/bridgecore_mt/gemma')
SENTS = BASE / 'sentences.jsonl'
OUT_DIR = BASE / 'out'
CHAPTERS = ['I', 'III']          # Claude·MADLAD 비교와 같은 범위 (1장·3장, 638문장)

MODEL_ID = 'google/gemma-4-31B-it'
THINKING = False                 # 생각 모드 끔 (번역만)
BS = 4                           # 한 번에 번역할 문단 수. 메모리 부족이 나면 자동으로 절반씩 줄임
MAX_NEW = 4096

GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(torch.cuda.get_device_name(0), f'{GPU_GB:.0f}GB')
assert GPU_GB > 35, 'A100(40GB 이상)이 아닙니다. 런타임 유형을 다시 고르세요'
LOAD_4BIT = GPU_GB < 70          # 40GB: 원본(약 62GB)이 안 들어가 4비트로
TAG = 'nf4' if LOAD_4BIT else 'bf16'
OUT = OUT_DIR / f'gemma_ch1_3_{TAG}.jsonl'
print('불러오는 방식:', TAG, '→', OUT.name)
if LOAD_4BIT:
    import bitsandbytes as bnb   # 여기서 오류가 나면 설치 셀을 다시 실행 → 세션 다시 시작
    print('bitsandbytes', bnb.__version__)
assert SENTS.exists(), f'없음: {SENTS}'
assert (BASE / 'gemma_translate.py').exists(), f'없음: {BASE / "gemma_translate.py"}'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE))
import gemma_translate as gt     # SYSTEM·FORMAT·blocks·user_text·parse 를 API판과 공유
print('프롬프트', gt.PROMPT_HASH)
"""),
md("## 2. 번역할 문단 준비"),
code("""rows = [json.loads(l) for l in open(SENTS, encoding='utf-8') if l.strip()]
allb, idx = gt.blocks(rows, CHAPTERS)
done = {json.loads(l)['id'] for l in open(OUT, encoding='utf-8')} if OUT.exists() else set()
todo = [k for k in idx if any(s['translate'] and s['id'] not in done for s in allb[k]['sents'])]
n_s = sum(sum(s['translate'] for s in allb[k]['sents']) for k in todo)
print(f'문단 {len(todo)}개 · 문장 {n_s}개 남음 (이미 끝난 문장 {len(done)}개)')
"""),
md("## 3. 모델 불러오기 (다운로드 포함 10~25분, 4비트는 변환 시간이 더 듦)"),
code("""from google.colab import userdata
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
import transformers

try:
    HF_TOKEN = userdata.get('HF_TOKEN')   # 없어도 됨. 다운로드가 막히면 열쇠 아이콘에 HF_TOKEN 추가
except Exception:
    HF_TOKEN = None

t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
tok = processor.tokenizer
tok.padding_side = 'left'                # 여러 문단을 한 번에 생성할 때 필요
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True) if LOAD_4BIT else None
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map='cuda', token=HF_TOKEN,
                                             quantization_config=quant)
model.eval()
LOAD_SEC = time.time() - t0
print(f'불러오기 {LOAD_SEC/60:.1f}분 · GPU 사용 {torch.cuda.memory_allocated()/1e9:.1f}GB')
"""),
md("## 4. 한 문단 시험 (출력 형식 확인)"),
code("""def prompt_text(k):
    sents, user = gt.user_text(allb, k)
    msgs = [{'role': 'system', 'content': gt.SYSTEM + gt.FORMAT}, {'role': 'user', 'content': user}]
    return sents, processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=THINKING)

def answer_of(ids):
    raw = processor.decode(ids, skip_special_tokens=False)
    try:
        content = processor.parse_response(raw)['content']
    except Exception:
        content = processor.decode(ids, skip_special_tokens=True)
    return raw, content

@torch.inference_mode()
def generate(ks):
    packs = [prompt_text(k) for k in ks]
    enc = processor(text=[p[1] for p in packs], return_tensors='pt', padding=True).to(model.device)
    n_in = enc['input_ids'].shape[1]
    need = max(len(p[0]) for p in packs) * 200 + 300      # 문장당 넉넉히 200토큰
    out = model.generate(**enc, max_new_tokens=min(MAX_NEW, need), do_sample=False)
    res = []
    for (sents, _), ids in zip(packs, out[:, n_in:]):
        raw, content = answer_of(ids)
        want = [s['id'] for s in sents]
        try:
            got = gt.parse(content)
            miss = [i for i in want if not got.get(i)]
            err = f'id 누락 {miss}' if miss else None
        except Exception as e:
            got, err = {}, f'파싱 실패 {e.__class__.__name__}'
        res.append((sents, got, err, raw))
    return res

t = time.time()
sents, got, err, raw = generate(todo[:1])[0]
print(f'{time.time()-t:.1f}초 · 오류 {err}')
for s in sents:
    print('EN', s['en']); print('KO', got.get(s['id'])); print()
print('--- 원출력 앞부분 ---'); print(raw[:600])
"""),
md("## 5. 전체 번역 (추정: 80GB 원본 20~60분, 40GB 4비트 40~90분)"),
code("""meta = {'model': MODEL_ID, 'runtime': f'colab_transformers_{TAG}', 'prompt_hash': gt.PROMPT_HASH,
        'unit': '문단(앞 문단 2개 문맥)', 'decoding': 'greedy', 'thinking': THINKING,
        'transformers': transformers.__version__, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
order = sorted(todo, key=lambda k: len(gt.user_text(allb, k)[1]))   # 길이 비슷한 문단끼리 묶어 빈칸 낭비를 줄임
fails, t0, bs, i, n_done = {}, time.time(), BS, 0, 0

def save(sents, got):
    with open(OUT, 'a', encoding='utf-8') as f:
        for s in sents:
            f.write(json.dumps({'id': s['id'], 'en': s['en'], 'mt': got[s['id']], 'mt_system': f'gemma-4-31b-it-{TAG}',
                                'mt_meta': meta}, ensure_ascii=False) + '\\n')

while i < len(order):
    ks = order[i:i + bs]
    try:
        res = generate(ks)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if bs == 1:
            raise
        bs = max(1, bs // 2); print('메모리 부족 → 묶음 크기', bs); continue
    for k, (sents, got, err, raw) in zip(ks, res):
        if err:
            fails[k] = (err, raw[-400:])
        else:
            save(sents, got)
    i += len(ks); n_done += len(ks)
    if n_done % 20 < len(ks):
        el = time.time() - t0
        print(f'  {n_done}/{len(order)} 문단 · {el/60:.1f}분 · 남은 예상 {el/n_done*(len(order)-n_done)/60:.0f}분 · 실패 {len(fails)}')

# 실패한 문단은 하나씩 다시 (그래도 실패하면 기록만)
for k in list(fails):
    sents, got, err, raw = generate([k])[0]
    if not err:
        save(sents, got); fails.pop(k)
    else:
        fails[k] = (err, raw[-400:])
GEN_SEC = time.time() - t0
print(f'완료 · {GEN_SEC/60:.1f}분 · 실패 문단 {len(fails)}개')
"""),
md("## 6. 확인·기록"),
code("""out_rows = [json.loads(l) for l in open(OUT, encoding='utf-8')]
ids = [r['id'] for r in out_rows]
want = {s['id'] for k in idx for s in allb[k]['sents'] if s['translate']}
print(f'번역 {len(set(ids))}/{len(want)}문장 · 중복 {len(ids) - len(set(ids))} · 빠짐 {len(want - set(ids))}')
info = {**meta, 'n_sentences': len(set(ids)), 'missing': sorted(want - set(ids)), 'load_sec': round(LOAD_SEC),
        'gen_sec': round(GEN_SEC), 'batch_final': bs, 'python': platform.python_version(),
        'fails': {str(k): v[0] for k, v in fails.items()}}
(OUT_DIR / f'run_info_{TAG}.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
for r in out_rows[:5]:
    print(r['en']); print('→', r['mt']); print()
print('결과 파일:', OUT)
"""),
]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
               "kernelspec": {"name": "python3", "display_name": "Python 3"}}
out = HERE / "05_개츠비번역_gemma4_31b.ipynb"
nbf.write(nb, out)
print("→", out)
