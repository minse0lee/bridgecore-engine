"""책 번역 노트북(04_책번역_lora_v1.ipynb) 생성기.  python build_notebook_04.py

학습 없음. lora_v1 어댑터로 책 문장을 번역(학습 후)하고, 같은 실행에서 어댑터를 끈 번역(학습 전)도 만든다.
입력은 epub_tools.py extract가 만든 translate_input.jsonl.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
md("""# 책 번역 — MADLAD-400 10B + lora_v1

**준비 (드라이브)**
- `bridgecore_mt/runs/lora_v1/adapter_best/` (02 노트북 결과, 이미 있음)
- `bridgecore_mt/book/translate_input.jsonl` ← 민서님이 올림 (책 EPUB에서 뽑은 문장, `{id, en}`)

**결과** `bridgecore_mt/book/out/`
- `translate_input__lora.jsonl` (lora_v1 번역), `translate_input__base.jsonl` (학습 전 번역, `BASE_TOO=True`일 때)
- `run_info.json` (환경·시간·잘린 문장 수)

**실행**: ① 설치 셀만 실행 → '런타임 → 세션 다시 시작' → 1번 셀부터 끝까지.
끊겨도 다시 실행하면 이미 번역한 문장은 건너뛴다 (256문장마다 저장).
"""),
code("""# ① 맨 먼저 이 셀만 실행 → 끝나면 '런타임 → 세션 다시 시작' → 다음 셀부터
!pip list --format=freeze | grep -E '^(torch|torchvision|torchaudio)==' > /content/torch_pin.txt
!pip -q install -c /content/torch_pin.txt transformers peft accelerate sentencepiece protobuf
!pip -q uninstall -y torchao
"""),
md("## 1. 설정"),
code("""import json, time, platform
from pathlib import Path
import torch
from google.colab import drive
drive.mount('/content/drive')

BASE = Path('/content/drive/MyDrive/bridgecore_mt')
ADAPTER = BASE / 'runs' / 'lora_v1' / 'adapter_best'
INPUT = BASE / 'book' / 'translate_input.jsonl'
OUT_DIR = BASE / 'book' / 'out'
BASE_TOO = True        # 학습 전 번역도 만들기 (책에서 학습 전후 비교용). 시간은 약 2배

MODEL_ID = 'google/madlad400-10b-mt'
REVISION = '9f2797629c31e69617186dbe5f0ca43bf662f36d'   # 03 노트북에서 고정한 커밋 (v1·v2 학습 전 번역이 같았음)
PREFIX = '<2ko> '
GEN = dict(num_beams=4, do_sample=False, max_new_tokens=256)   # lora_v1 평가와 같은 설정
MAX_SRC = 256
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
BS = 32 if GPU_GB > 60 else 16
CHUNK = 256

assert ADAPTER.exists(), f'어댑터 없음: {ADAPTER}'
assert INPUT.exists(), f'입력 없음: {INPUT}'
OUT_DIR.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(INPUT, encoding='utf-8') if l.strip()]
assert len({r['id'] for r in rows}) == len(rows), 'id 중복'
print(f'GPU {torch.cuda.get_device_name(0)} {GPU_GB:.0f}GB | 문장 {len(rows)}개 | 단어 {sum(len(r["en"].split()) for r in rows)}')
"""),
md("## 2. 모델 불러오기"),
code("""from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel
import transformers, peft
tok = T5Tokenizer.from_pretrained(MODEL_ID, revision=REVISION)
try:
    m = T5ForConditionalGeneration.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.bfloat16, device_map={'': 0})
except TypeError:
    m = T5ForConditionalGeneration.from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16, device_map={'': 0})
pm = PeftModel.from_pretrained(m, str(ADAPTER)); pm.eval(); pm.config.use_cache = True
print('어댑터:', ADAPTER, '| transformers', transformers.__version__, '| peft', peft.__version__)
"""),
md("## 3. 번역 (256문장마다 저장, 끊기면 이어서)\n\n입력 토큰이 256을 넘는 긴 문장은 잘리지 않게 한 문장씩 길이 제한을 늘려 번역하고 `long: true`로 표시"),
code("""ntok = {r['id']: len(tok(PREFIX + r['en']).input_ids) for r in rows}
long_ids = {i for i, n in ntok.items() if n > MAX_SRC}
print('256토큰 넘는 문장:', len(long_ids))
meta = {"model": MODEL_ID, "revision": REVISION, "adapter": "lora_v1", "gen": GEN, "prefix": PREFIX,
        "transformers": transformers.__version__, "torch": torch.__version__}

def gen(texts, use_adapter, long=False):
    enc = tok([PREFIX + t for t in texts], return_tensors='pt', padding=True,
              truncation=not long, max_length=None if long else MAX_SRC).to(0)
    kw = dict(GEN)
    if long:
        kw['max_new_tokens'] = min(1024, 2 * enc.input_ids.shape[1])
    with torch.no_grad():
        if use_adapter:
            out = pm.generate(**enc, **kw)
        else:
            with pm.disable_adapter():
                out = pm.generate(**enc, **kw)
    return tok.batch_decode(out, skip_special_tokens=True)

def run(tag, use_adapter):
    path = OUT_DIR / f'translate_input__{tag}.jsonl'
    done = {json.loads(l)['id'] for l in open(path, encoding='utf-8')} if path.exists() else set()
    todo = [r for r in rows if r['id'] not in done]
    print(f'[{tag}] 이미 {len(done)} · 남은 {len(todo)}')
    t0 = time.time()
    for s in range(0, len(todo), CHUNK):
        chunk = todo[s:s + CHUNK]
        res = {}
        short = sorted([r for r in chunk if r['id'] not in long_ids], key=lambda r: len(r['en']))
        for b in range(0, len(short), BS):
            part = short[b:b + BS]
            for r, o in zip(part, gen([r['en'] for r in part], use_adapter)):
                res[r['id']] = o
        for r in chunk:
            if r['id'] in long_ids:
                res[r['id']] = gen([r['en']], use_adapter, long=True)[0]
        with open(path, 'a', encoding='utf-8') as f:
            for r in chunk:
                f.write(json.dumps({'id': r['id'], 'en': r['en'], 'mt': res[r['id']], 'long': r['id'] in long_ids,
                                    'mt_system': f'madlad10b_{tag}', 'mt_meta': meta}, ensure_ascii=False) + '\\n')
        n = s + len(chunk); el = time.time() - t0
        print(f'  {n}/{len(todo)} · {el/60:.1f}분 · 남은 예상 {el/n*(len(todo)-n)/60:.0f}분')
    return time.time() - t0

secs = {'lora': run('lora', True)}
if BASE_TOO:
    secs['base'] = run('base', False)
"""),
md("## 4. 확인"),
code("""info = {"n": len(rows), "long_sentences": len(long_ids), "seconds": {k: round(v) for k, v in secs.items()},
        "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(), **meta}
(OUT_DIR / 'run_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
print(json.dumps(info, ensure_ascii=False, indent=1))
outs = {t: {json.loads(l)['id']: json.loads(l)['mt'] for l in open(OUT_DIR / f'translate_input__{t}.jsonl', encoding='utf-8')}
        for t in (['lora', 'base'] if BASE_TOO else ['lora'])}
for t, o in outs.items():
    print(t, '번역 수', len(o), '/', len(rows), '| 빈 번역', sum(not v.strip() for v in o.values()))
import random
for r in random.Random(0).sample(rows, min(5, len(rows))):
    print('-', r['en'][:150])
    for t, o in outs.items():
        print(f'  {t}:', o[r['id']][:150])
"""),
md("""## 끝나면 보내 줄 것
`book/out/` 안의 `translate_input__lora.jsonl`, `translate_input__base.jsonl`, `run_info.json`
→ 한국어 EPUB 조립, 무작위 표본 6축 검증(학습 전후), 사람 확인용 표를 만든다"""),
]

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata.update({"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
                    "kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = HERE / "04_책번역_lora_v1.ipynb"
nbf.write(nb, out)
print(out)
