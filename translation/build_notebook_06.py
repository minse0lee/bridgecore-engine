"""Gemma 4 31B 설정 실험 노트북(06_개츠비번역_gemma_설정실험.ipynb) 생성기.  python build_notebook_06.py

학습 없이 두 가지만 바꿔 개츠비 1·3장을 다시 번역한다 (기준: 05 노트북 = 앞 문단 2개, 생각 모드 끔).
  ctx5        앞 문단 5개, 생각 끔
  ctx5_think  앞 문단 5개, 생각 켬
  ctx2_think  앞 문단 2개, 생각 켬
+ 05 결과 파일을 함께 올리면 빠진 3문단(8문장)만 같은 기준 설정으로 채움(base_fill)
A100 80GB(bf16 원본) 전용 — 05와 같은 정밀도여야 비교가 된다.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
md("""# 개츠비 1·3장 — Gemma 4 31B 설정 실험 (학습 없음, A100 80GB)

기준(05 노트북): 앞 문단 2개 문맥, 생각 모드 끔 → major 이상 2.3%, 훼손 1개 이상 23.6% (6축 v1.1)
이번에 바꿔 보는 것 (다른 조건은 전부 같음: 같은 지침·greedy·bf16 원본)

| 이름 | 앞 문단 문맥 | 생각 모드 | 예상 시간(추정) |
|---|---|---|---|
| `ctx5` | 5개 | 끔 | 35~45분 |
| `ctx5_think` | 5개 | 켬 | 1~2시간 |
| `ctx2_think` | 2개 | 켬 | 1~2시간 |

`RUN` 목록 순서대로 돌고, **하나 끝날 때마다 결과 파일을 바로 내려받음**(세션이 끊겨도 끝난 것은 남음).
시간이 부족하면 `RUN`에서 빼면 됨. 우선순위: ctx5 → ctx5_think → ctx2_think

**올릴 파일** (1번 셀 실행 시 파일 선택 창)
- 필수: `gemma_translate.py`(**새 버전** — 문맥 문단 수 설정이 추가됨), `sentences.jsonl`
- 선택: 05 결과 `gemma_ch1_3_bf16.jsonl` → 올리면 그때 빠진 3문단(8문장)만 채운 `gemma_ch1_3_bf16_filled.jsonl`도 만듦(약 1분)

**실행**: ① 설치 셀 → '런타임 → 세션 다시 시작' → 1번 셀부터 끝까지. 끝날 때까지 창을 열어 둘 것
"""),
code("""# ① 맨 먼저 이 셀만 실행 → 끝나면 '런타임 → 세션 다시 시작' → 다음 셀부터
!pip list --format=freeze | grep -E '^(torch|torchvision|torchaudio)==' > /content/torch_pin.txt
!pip -q install -c /content/torch_pin.txt -U "transformers>=5.10.1" accelerate
!pip list --format=freeze | grep -E '^(transformers|torch|accelerate)=='
"""),
md("## 1. 설정·파일 올리기"),
code("""import json, sys, time, platform, inspect, re
from pathlib import Path
import torch
from google.colab import files

RUN = ['ctx5', 'ctx5_think', 'ctx2_think']     # 돌릴 설정 (순서대로)
CONFIGS = {'ctx5': dict(n_ctx=5, thinking=False), 'ctx5_think': dict(n_ctx=5, thinking=True),
           'ctx2_think': dict(n_ctx=2, thinking=True), 'base': dict(n_ctx=2, thinking=False)}
CHAPTERS = ['I', 'III']
MODEL_ID = 'google/gemma-4-31B-it'
BS = 4
THINK_BUDGET = 4096        # 생각 모드일 때 추가로 허용할 토큰 수

BASE = Path('/content/gemma'); OUT_DIR = BASE / 'out'; OUT_DIR.mkdir(parents=True, exist_ok=True)
print('파일 선택 창에서 고르기: gemma_translate.py, sentences.jsonl (+ 선택: gemma_ch1_3_bf16.jsonl)')
for name, data in files.upload().items():
    clean = re.sub(r' \(\d+\)(?=\.[^.]+$)', '', Path(name).name)   # 같은 이름을 다시 올리면 붙는 ' (1)' 제거
    (OUT_DIR if clean.startswith('gemma_ch1_3') else BASE).joinpath(clean).write_bytes(data)

GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(torch.cuda.get_device_name(0), f'{GPU_GB:.0f}GB')
assert GPU_GB > 70, '80GB GPU가 아닙니다 — 이 실험은 05와 같은 원본(bf16)이어야 비교가 됩니다'
assert (BASE / 'sentences.jsonl').exists() and (BASE / 'gemma_translate.py').exists(), '필수 파일 두 개를 올리세요'
sys.path.insert(0, str(BASE))
import gemma_translate as gt
assert 'n_ctx' in inspect.signature(gt.user_text).parameters, 'gemma_translate.py가 옛 버전입니다 — 새 파일을 올리세요'
PREV = OUT_DIR / 'gemma_ch1_3_bf16.jsonl'
print('프롬프트', gt.PROMPT_HASH, '· 05 결과 올림' if PREV.exists() else '· 05 결과 없음(빈칸 채우기 건너뜀)')
"""),
md("## 2. 모델 불러오기 (약 5분)"),
code("""from google.colab import userdata
from transformers import AutoProcessor, AutoModelForCausalLM
import transformers
from getpass import getpass

try:
    HF_TOKEN = userdata.get('HF_TOKEN')
except Exception:
    HF_TOKEN = None   # 공용 계정이면 열쇠 아이콘에 저장하지 말고, 필요할 때만 아래 줄 주석을 풀어 입력
# HF_TOKEN = getpass('HF 토큰: ')

t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
processor.tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map='cuda', token=HF_TOKEN).eval()
LOAD_SEC = time.time() - t0
print(f'불러오기 {LOAD_SEC/60:.1f}분 · GPU {torch.cuda.memory_allocated()/1e9:.1f}GB')

rows = [json.loads(l) for l in open(BASE / 'sentences.jsonl', encoding='utf-8') if l.strip()]
allb, idx = gt.blocks(rows, CHAPTERS)
WANT = {s['id'] for k in idx for s in allb[k]['sents'] if s['translate']}
print('문단', len(idx), '· 문장', len(WANT))
"""),
md("## 3. 번역 함수"),
code("""SAMPLE = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=64)   # 공식 권장값 — greedy가 실패한 문단 재시도에만 씀

def prompt_text(k, cfg):
    sents, user = gt.user_text(allb, k, n_ctx=cfg['n_ctx'])
    msgs = [{'role': 'system', 'content': gt.SYSTEM + gt.FORMAT}, {'role': 'user', 'content': user}]
    return sents, processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=cfg['thinking'])

def split_answer(ids):
    raw = processor.decode(ids, skip_special_tokens=False)
    try:
        r = processor.parse_response(raw)
        return raw, r.get('content') or '', r.get('thinking') or ''
    except Exception:
        return raw, processor.decode(ids, skip_special_tokens=True), ''

@torch.inference_mode()
def generate(ks, cfg, sample=False):
    packs = [prompt_text(k, cfg) for k in ks]
    enc = processor(text=[p[1] for p in packs], return_tensors='pt', padding=True).to(model.device)
    n_in = enc['input_ids'].shape[1]
    need = max(len(p[0]) for p in packs) * 200 + 300 + (THINK_BUDGET if cfg['thinking'] else 0)
    out = model.generate(**enc, max_new_tokens=need, **(SAMPLE if sample else dict(do_sample=False)))
    res = []
    for (sents, _), ids in zip(packs, out[:, n_in:]):
        raw, content, think = split_answer(ids)
        n_tok = int((ids != processor.tokenizer.pad_token_id).sum())
        try:
            got = gt.parse(content)
            miss = [s['id'] for s in sents if not got.get(s['id'])]
            err = f'id 누락 {miss}' if miss else None
        except Exception as e:
            got, err = {}, f'파싱 실패 {e.__class__.__name__}'
        res.append(dict(sents=sents, got=got, err=err, raw=raw, think_chars=len(think), out_tokens=n_tok))
    return res

def run_config(name, cfg, out_path, only=None):
    meta = {'model': MODEL_ID, 'runtime': 'colab_transformers_bf16', 'prompt_hash': gt.PROMPT_HASH, 'config': name,
            'n_ctx': cfg['n_ctx'], 'thinking': cfg['thinking'], 'decoding': 'greedy',
            'transformers': transformers.__version__, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
    done = {json.loads(l)['id'] for l in open(out_path, encoding='utf-8')} if out_path.exists() else set()
    todo = [k for k in (only if only is not None else idx)
            if any(s['translate'] and s['id'] not in done for s in allb[k]['sents'])]
    order = sorted(todo, key=lambda k: len(gt.user_text(allb, k, n_ctx=cfg['n_ctx'])[1]))
    stats = {'think_chars': [], 'out_tokens': [], 'retried_with_sampling': [], 'fails': {}}
    raw_log = open(OUT_DIR / f'raw_fail_{name}.jsonl', 'a', encoding='utf-8')

    def save(r, sampled=False):
        with open(out_path, 'a', encoding='utf-8') as f:
            for s in r['sents']:
                if s['id'] in done:      # 이미 있는 문장은 다시 쓰지 않음(중복 방지)
                    continue
                f.write(json.dumps({'id': s['id'], 'en': s['en'], 'mt': r['got'][s['id']],
                                    'mt_system': f'gemma-4-31b-it-bf16-{name}',
                                    'mt_meta': {**meta, 'decoding': 'sampling_retry' if sampled else 'greedy'}},
                                   ensure_ascii=False) + '\\n')

    t0, bs, i, fails, nb = time.time(), BS, 0, [], 0
    print(f'[{name}] 문단 {len(order)}개 시작')
    while i < len(order):
        ks = order[i:i + bs]
        try:
            res = generate(ks, cfg)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2); print('  메모리 부족 → 묶음', bs); continue
        for k, r in zip(ks, res):
            stats['think_chars'].append(r['think_chars']); stats['out_tokens'].append(r['out_tokens'])
            if r['err']:
                fails.append(k); raw_log.write(json.dumps({'k': k, 'err': r['err'], 'raw': r['raw']}, ensure_ascii=False) + '\\n')
            else:
                save(r)
        i += len(ks); nb += 1
        if nb % 5 == 0 or i >= len(order):
            el = time.time() - t0
            print(f'  {i}/{len(order)} 문단 · {el/60:.1f}분 · 남은 예상 {el/i*(len(order)-i)/60:.0f}분 · 실패 {len(fails)}')
    for k in fails:                      # greedy로 실패한 문단: 하나씩, 공식 권장 샘플링으로 한 번 더
        r = generate([k], cfg, sample=True)[0]
        if r['err']:
            stats['fails'][str(k)] = r['err']
            raw_log.write(json.dumps({'k': k, 'err': r['err'], 'raw': r['raw'], 'sampling': True}, ensure_ascii=False) + '\\n')
        else:
            save(r, sampled=True); stats['retried_with_sampling'].append(k)
    raw_log.close()
    got_ids = {json.loads(l)['id'] for l in open(out_path, encoding='utf-8')} if out_path.exists() else set()
    tc = [x for x in stats['think_chars'] if x]
    info = {**meta, 'gen_sec': round(time.time() - t0), 'n_sentences': len(got_ids & WANT), 'missing': sorted(WANT - got_ids),
            'retried_with_sampling': stats['retried_with_sampling'], 'fails': stats['fails'], 'batch_final': bs,
            'mean_out_tokens': round(sum(stats['out_tokens']) / max(1, len(stats['out_tokens']))),
            'mean_think_chars': round(sum(tc) / len(tc)) if tc else 0, 'load_sec': round(LOAD_SEC),
            'python': platform.python_version()}
    (OUT_DIR / f'run_info_{name}.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(f'[{name}] 완료 {info["gen_sec"]/60:.1f}분 · {info["n_sentences"]}/{len(WANT)}문장 · 빠짐 {len(info["missing"])}'
          f' · 샘플링 재시도 성공 {len(stats["retried_with_sampling"])} · 평균 생각 {info["mean_think_chars"]}자')
    return info

# 시험 1문단 (생각 켬) — 형식 확인
r = generate(idx[:1], CONFIGS['ctx5_think'])[0]
print('오류', r['err'], '· 생각', r['think_chars'], '자 · 출력 토큰', r['out_tokens'])
for s in r['sents']:
    print('EN', s['en']); print('KO', r['got'].get(s['id']))
print('--- 원출력 앞부분 ---'); print(r['raw'][:800])
"""),
md("## 4. (선택) 05에서 빠진 8문장 채우기 — 기준 설정 그대로"),
code("""if PREV.exists():
    import shutil
    FILLED = OUT_DIR / 'gemma_ch1_3_bf16_filled.jsonl'
    shutil.copy(PREV, FILLED)
    have = {json.loads(l)['id'] for l in open(FILLED, encoding='utf-8')}
    gaps = [k for k in idx if any(s['translate'] and s['id'] not in have for s in allb[k]['sents'])]
    print('빠진 문단', gaps)
    info = run_config('base', CONFIGS['base'], FILLED, only=gaps)
    files.download(str(FILLED)); files.download(str(OUT_DIR / 'run_info_base.json'))
else:
    print('05 결과를 올리지 않아 건너뜀')
"""),
md("## 5. 설정 실험 (하나 끝날 때마다 결과 자동 다운로드)"),
code("""ALL_INFO = {}
for name in RUN:
    out_path = OUT_DIR / f'gemma_ch1_3_bf16_{name}.jsonl'
    ALL_INFO[name] = run_config(name, CONFIGS[name], out_path)
    files.download(str(out_path)); files.download(str(OUT_DIR / f'run_info_{name}.json'))
    if (OUT_DIR / f'raw_fail_{name}.jsonl').exists() and (OUT_DIR / f'raw_fail_{name}.jsonl').stat().st_size:
        files.download(str(OUT_DIR / f'raw_fail_{name}.jsonl'))
print(json.dumps({k: {x: v[x] for x in ('gen_sec', 'n_sentences', 'mean_think_chars')} for k, v in ALL_INFO.items()},
                 ensure_ascii=False, indent=1))
"""),
]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
               "kernelspec": {"name": "python3", "display_name": "Python 3"}}
out = HERE / "06_개츠비번역_gemma_설정실험.ipynb"
nbf.write(nb, out)
print("→", out)
