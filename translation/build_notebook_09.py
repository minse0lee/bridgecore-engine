"""Gemma 4 31B 검증엔진 시험 노트북(09_검증엔진_gemma4_31b_dev100.ipynb) 생성기.  python build_notebook_09.py
골든셋 개발용 100문항을 Claude 엔진(v2, 지문 fcfdd47986d2)과 같은 지침으로 Gemma가 판정한다.
입력 파일은 tools/gemma_judge_io.py export 로 만들고, 결과는 같은 도구의 import 로 엔진 형식으로 바꿔 채점한다.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
md("""# 검증엔진을 Gemma 4 31B로 — 골든셋 개발용 100문항 (학습 없음)
지금 엔진(Claude Sonnet 5.5, 프롬프트 v2)과 **같은 지침·같은 문항 틀·같은 자동 점검 힌트**로 Gemma가 6축 판정을 한다.
결과를 Claude에게 올리면 평가자 3인 다수결과 비교해 정밀도·재현율·F1을 Claude 엔진(F1 0.80~0.85)과 나란히 계산한다.

차이점 하나: Claude는 API가 JSON 형식을 강제하지만 Gemma는 그런 장치가 없어서, 같은 형식을 지침 끝에 글로 붙였다.
형식이 깨진 문항은 한 번 더(샘플링) 시도한다.

**GPU**: A100 80GB 권장(원본 bf16). 40GB면 자동으로 4비트(nf4) — 원본과 계산이 달라 결과도 조금 다를 수 있음(파일 이름에 표시)
**시간(추정)**: 모델 불러오기 10~25분 + 판정 1~2시간. 창을 열어 둘 것
**올릴 파일**: `gemma_judge_inputs.json` 하나 (1번 셀 실행 시 파일 선택 창)
**결과**: `gemma_judge_raw_<bf16|nf4>.jsonl`, `gemma_judge_info_<bf16|nf4>.json` → 이 두 파일을 Claude에게

**실행**: ① 설치 셀 → '런타임 → 세션 다시 시작' → 1번 셀부터 끝까지
중간에 끊기면 같은 순서로 다시 실행하되, 1번 셀에서 지난번 결과 파일(raw)도 같이 올리면 끝난 문항은 건너뛴다.
"""),
code("""# ① 맨 먼저 이 셀만 실행 → 끝나면 '런타임 → 세션 다시 시작' → 다음 셀부터
!pip list --format=freeze | grep -E '^(torch|torchvision|torchaudio)==' > /content/torch_pin.txt
!pip -q install -c /content/torch_pin.txt -U "transformers>=5.10.1" accelerate
!pip -q install -U "bitsandbytes>=0.46.1"
!pip -q uninstall -y torchao
!pip list --format=freeze | grep -E '^(transformers|torch|accelerate|bitsandbytes)=='
"""),
md("## 1. 설정·파일 올리기"),
code("""import json, time, platform, re
from pathlib import Path
import torch
from google.colab import files

MODEL_ID = 'google/gemma-4-31B-it'
THINKING = False      # 생각 모드. 켜면 몇 배 느려짐 — 먼저 끈 상태로 잰다
BS = 2                # 한 번에 판정할 문항 수 (지침이 길어서 작게). 메모리 부족이면 자동으로 줄임
MAX_NEW = 3000        # 문항당 출력 상한 (Claude 엔진 출력은 보통 600~1,500토큰)

BASE = Path('/content/gemma_judge'); BASE.mkdir(parents=True, exist_ok=True)
print('파일 선택 창에서 고르기: gemma_judge_inputs.json (+ 이어 하기면 지난 gemma_judge_raw_*.jsonl)')
for name, data in files.upload().items():
    clean = re.sub(r' \\(\\d+\\)(?=\\.[^.]+$)', '', Path(name).name)   # 같은 이름을 다시 올리면 붙는 ' (1)' 제거
    (BASE / clean).write_bytes(data)
INP = json.loads((BASE / 'gemma_judge_inputs.json').read_text(encoding='utf-8'))
print('문항', len(INP['items']), '· 프롬프트', INP['prompt_version'], '· 엔진 지문', INP['engine_fingerprint'],
      '· Gemma 지문', INP['gemma_fingerprint'])

GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(torch.cuda.get_device_name(0), f'{GPU_GB:.0f}GB')
assert GPU_GB > 35, 'A100(40GB 이상)이 아닙니다. 런타임 유형을 다시 고르세요'
LOAD_4BIT = GPU_GB < 70
TAG = 'nf4' if LOAD_4BIT else 'bf16'
RAW = BASE / f'gemma_judge_raw_{TAG}.jsonl'
INFO = BASE / f'gemma_judge_info_{TAG}.json'
if LOAD_4BIT:
    import bitsandbytes as bnb   # 여기서 오류가 나면 설치 셀을 다시 실행 → 세션 다시 시작
print('불러오는 방식:', TAG, '→', RAW.name)
"""),
md("## 2. 모델 불러오기 (10~25분)"),
code("""from google.colab import userdata
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
try:
    HF_TOKEN = userdata.get('HF_TOKEN')
except Exception:
    HF_TOKEN = None   # 공용 계정이면 열쇠 아이콘에 저장하지 말 것. 필요하면 아래 두 줄 주석을 풀어 입력
# from getpass import getpass
# HF_TOKEN = getpass('HF 토큰: ')
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
processor.tokenizer.padding_side = 'left'
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True) if LOAD_4BIT else None
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map='cuda', token=HF_TOKEN,
                                             quantization_config=quant).eval()
LOAD_SEC = time.time() - t0
print(f'불러오기 {LOAD_SEC/60:.1f}분 · GPU {torch.cuda.memory_allocated()/1e9:.1f}GB')
"""),
md("## 3. 판정 함수 + 한 문항 시험"),
code("""AXES = ['Semantic', 'Textual', 'Context', 'Deontic', 'Register', 'Cultural']
JUDG = {'보존', '훼손', 'N/A', '판단보류'}
SAMPLE = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=64)   # 공식 권장값 — 형식이 깨진 문항 재시도에만
ITEMS = {it['item_id']: it for it in INP['items']}

def prompt_text(iid):
    msgs = [{'role': 'system', 'content': INP['system']}, {'role': 'user', 'content': ITEMS[iid]['user']}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=THINKING)

def check(content, iid):
    \"\"\"형식만 가볍게 확인 (자세한 검증·채점은 Claude 쪽 도구가 엔진과 같은 기준으로 다시 함)\"\"\"
    m = re.search(r'```(?:json)?\\s*(\\{[\\s\\S]*\\})\\s*```', content)
    raw = m.group(1) if m else content[content.find('{'): content.rfind('}') + 1]
    try:
        o = json.loads(raw)
    except Exception as e:
        return f'JSON 파싱 실패 {e.__class__.__name__}'
    if o.get('item_id') != iid:
        return f'item_id 불일치 {o.get("item_id")}'
    ax = o.get('axes') or {}
    bad = [a for a in AXES if not isinstance(ax.get(a), dict) or ax[a].get('judgment') not in JUDG]
    return f'축 형식 오류 {bad}' if bad else None

@torch.inference_mode()
def generate(iids, sample=False):
    enc = processor(text=[prompt_text(i) for i in iids], return_tensors='pt', padding=True).to(model.device)
    n_in = enc['input_ids'].shape[1]
    t = time.time()
    out = model.generate(**enc, max_new_tokens=MAX_NEW, **(SAMPLE if sample else dict(do_sample=False)))
    sec = round((time.time() - t) / len(iids), 1)
    res = []
    for iid, ids in zip(iids, out[:, n_in:]):
        raw = processor.decode(ids, skip_special_tokens=False)
        try:
            content = processor.parse_response(raw).get('content') or ''
        except Exception:
            content = processor.decode(ids, skip_special_tokens=True)
        n_tok = int((ids != processor.tokenizer.pad_token_id).sum())
        res.append({'item_id': iid, 'model': f'gemma-4-31b-it-{TAG}', 'content': content, 'out_tokens': n_tok,
                    'truncated': n_tok >= MAX_NEW, 'sec': sec, 'decoding': 'sampling_retry' if sample else 'greedy',
                    'thinking': THINKING, 'err': check(content, iid)})
    return res

r = generate([INP['items'][0]['item_id']])[0]
print('형식 오류:', r['err'], '· 출력 토큰', r['out_tokens'], '· 걸린 시간', r['sec'], '초')
print(r['content'][:1500])
"""),
md("## 4. 100문항 판정 (진행 상황이 주기적으로 표시됨)"),
code("""def save(r):
    with open(RAW, 'a', encoding='utf-8') as f:
        f.write(json.dumps(r, ensure_ascii=False) + '\\n')

done = {}
if RAW.exists():
    for l in open(RAW, encoding='utf-8'):
        x = json.loads(l); done[x['item_id']] = x
todo = [i for i in ITEMS if i not in done or done[i]['err']]
todo.sort(key=lambda i: len(ITEMS[i]['user']))       # 길이가 비슷한 것끼리 묶어 패딩 낭비를 줄임
print('남은 문항', len(todo), '/', len(ITEMS))
t0, bs, k, fails = time.time(), BS, 0, []
while k < len(todo):
    ids = todo[k:k + bs]
    try:
        res = generate(ids)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if bs == 1:
            raise
        bs = max(1, bs // 2); print('  메모리 부족 → 묶음', bs); continue
    for r in res:
        save(r)
        if r['err']:
            fails.append(r['item_id'])
    k += len(ids)
    if k % 10 < len(ids) or k >= len(todo):
        el = time.time() - t0
        print(f'  {k}/{len(todo)} · {el/60:.1f}분 · 남은 예상 {el/k*(len(todo)-k)/60:.0f}분 · 형식 오류 {len(fails)}', flush=True)

retry_ok = []
for iid in fails:                     # 형식이 깨진 문항만 하나씩 샘플링으로 한 번 더
    r = generate([iid], sample=True)[0]; save(r)
    if not r['err']:
        retry_ok.append(iid)
final = {}
for l in open(RAW, encoding='utf-8'):
    x = json.loads(l); final[x['item_id']] = x
info = {'model': MODEL_ID, 'tag': TAG, 'thinking': THINKING, 'max_new_tokens': MAX_NEW, 'batch_final': bs,
        'engine_fingerprint': INP['engine_fingerprint'], 'gemma_fingerprint': INP['gemma_fingerprint'],
        'gen_min': round((time.time() - t0) / 60, 1), 'load_min': round(LOAD_SEC / 60, 1),
        'n_items': len(final), 'n_format_ok': sum(not x['err'] for x in final.values()),
        'retry_ok': retry_ok, 'still_bad': sorted(i for i, x in final.items() if x['err']),
        'mean_out_tokens': round(sum(x['out_tokens'] for x in final.values()) / max(1, len(final))),
        'gpu': torch.cuda.get_device_name(0), 'transformers': transformers.__version__, 'torch': torch.__version__,
        'python': platform.python_version()}
INFO.write_text(json.dumps(info, ensure_ascii=False, indent=1))
print(json.dumps(info, ensure_ascii=False, indent=1))
"""),
md("""## 5. 결과 내려받기
아래 셀이 다운로드 창을 띄운다. 안 뜨면 왼쪽 **파일(폴더 아이콘) → gemma_judge** 폴더에서 두 파일을 오른쪽 클릭 → 다운로드"""),
code("""files.download(str(RAW)); files.download(str(INFO))
"""),
]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
               "kernelspec": {"name": "python3", "display_name": "Python 3"}}
out = HERE / "09_검증엔진_gemma4_31b_dev100.ipynb"
nbf.write(nb, out)
print("→", out)
