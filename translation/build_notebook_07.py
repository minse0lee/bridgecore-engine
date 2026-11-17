"""Gemma 4 31B QLoRA 학습 노트북(07_학습_gemma4_31b_QLoRA.ipynb) 생성기.  python build_notebook_07.py

- 데이터: MADLAD lora_v1과 같은 `bridgecore_mt/splits_v1/`(AI Hub 사회과학·교육, 영어 원문) → 두 모델을 같은 데이터로 비교
- 형식: 개츠비 번역(gemma_translate.py)과 같은 '문단 + id별 JSON' 형식으로 학습해, 학습 후에도 그 형식을 유지하게 한다
  AI Hub 문장은 서로 이어진 문단이 아니므로 앞 문단 문맥은 '(없음)', 1~4문장을 묶어 한 덩어리로 만든다
- 시스템 지침: 개츠비 지침(소설 전용) 대신 같은 원칙의 일반 지침(TRAIN_SYSTEM). 출력 형식 지시(FORMAT)는 같다
- A100 40GB(개인 계정, 드라이브)에서 4비트(QLoRA). 중간 저장·이어서 학습. 학습 후 번역 비교는 08 노트북(80GB, 원본+어댑터)
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
md("""# Gemma 4 31B 영→한 QLoRA 학습 (A100 40GB · 드라이브)

**하는 일**
1. MADLAD 때와 같은 AI Hub 분할(`splits_v1`)에서 학습 문장을 골라, 개츠비 번역과 **같은 입력·출력 형식**(문단 + id별 JSON)으로 학습 예제를 만든다
2. 4비트로 불러와 LoRA 학습 (QLoRA). 일정 스텝마다 검증 손실·중간 저장 → **끊기면 3번 셀부터 다시 실행하면 이어서 학습**
3. 학습 전(LoRA 끔)·후 번역을 같은 4비트 모델로 몇 덩어리 비교 (+ 선택: AI Hub 평가 500문장 chrF 전후 비교)
4. 어댑터를 `runs/gemma_lora_v1/adapter_best.zip`으로 저장 → 08 노트북(80GB)에서 원본 모델에 얹어 개츠비 번역

**준비 (드라이브 `MyDrive/bridgecore_mt/`)**
- `splits_v1/` (MADLAD 학습 때 만든 것 — 이미 있음)
- `gemma/gemma_translate.py` — **새 버전**(문맥 문단 수 설정이 있는 것)으로 바꿔 두기

**런타임**: A100(40GB) + 높은 RAM. ① 설치 셀 → '세션 다시 시작' → 1번 셀부터
**시간(추정, 실제로 재지 않음)**: 모델 불러오기 15~25분, 학습은 문장 2만 개·1에폭 기준 몇 시간. 첫 스텝들의 속도를 보고 2번 셀의 `N_TRAIN_SENT`를 조절
**한계**: AI Hub는 교육·사회과학 문장이라 소설 대화문 오역("Who with?" 등)이 고쳐진다는 보장은 없음. MADLAD 때처럼 말투 쏠림(Miss→선생님 등)도 학습 전후로 꼭 확인
"""),
code("""# ① 맨 먼저 이 셀만 실행 → 끝나면 '런타임 → 세션 다시 시작' → 다음 셀부터
!pip list --format=freeze | grep -E '^(torch|torchvision|torchaudio)==' > /content/torch_pin.txt
!pip -q install -c /content/torch_pin.txt -U "transformers>=5.10.1" accelerate peft sacrebleu
!pip -q install -U "bitsandbytes>=0.46.1"
!pip -q uninstall -y torchao
!pip list --format=freeze | grep -E '^(transformers|torch|accelerate|peft|bitsandbytes)=='
"""),
md("## 1. 설정"),
code("""import json, os, sys, time, math, random, glob, shutil, platform, inspect
from pathlib import Path
import torch
from google.colab import drive
drive.mount('/content/drive')

BASE = Path('/content/drive/MyDrive/bridgecore_mt')
SPLIT_DIR = BASE / 'splits_v1'
RUN_NAME = 'gemma_lora_v1'              # 설정을 바꿔 다시 학습할 땐 이름을 바꾸기
RUN_DIR = BASE / 'runs' / RUN_NAME; RUN_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ID = 'google/gemma-4-31B-it'
SEED = 20261117

N_TRAIN_SENT = 20000                    # 학습에 쓸 문장 수 (전체를 쓰려면 None). 처음엔 2만으로 속도·효과 확인
CHUNK = (1, 4)                          # 한 예제에 묶을 문장 수 범위
N_VALID_EX = 100                        # 검증 손실용 예제 수 (valid 분할에서)
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05)
ATTN = ('q_proj', 'k_proj', 'v_proj', 'o_proj')   # 언어 모델의 어텐션에만 LoRA
BS, ACCUM, LR, EPOCHS = 2, 8, 1e-4, 1
MAX_LEN = 1536
EVAL_STEPS = 50                         # 이 간격마다 검증 손실·중간 저장
RUN_TEST500 = True                      # 학습 후 AI Hub 평가 500문장 chrF 전후 비교 (4비트, 약 20~40분 추정)

GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(torch.cuda.get_device_name(0), f'{GPU_GB:.0f}GB')
assert GPU_GB > 35, 'A100(40GB 이상)이 아닙니다'
assert (SPLIT_DIR / 'train.jsonl').exists(), f'없음: {SPLIT_DIR}'
sys.path.insert(0, str(BASE / 'gemma'))
import gemma_translate as gt
assert 'n_ctx' in inspect.signature(gt.user_text).parameters, 'gemma/gemma_translate.py가 옛 버전입니다 — 새 파일로 바꾸세요'
random.seed(SEED); torch.manual_seed(SEED)
load_jsonl = lambda p: [json.loads(l) for l in open(p, encoding='utf-8')]
train_rows, valid_rows, test_rows = (load_jsonl(SPLIT_DIR / f'{n}.jsonl') for n in ('train', 'valid', 'test'))
print('분할', len(train_rows), len(valid_rows), len(test_rows))
"""),
md("## 2. 학습 예제 만들기 (개츠비 번역과 같은 형식)"),
code("""TRAIN_SYSTEM = '''당신은 전문 번역가입니다. 영어 원문을 한국어 독자를 위한 출판용 번역으로 옮깁니다.

원칙
1. 의미 보존: 원문의 내용·뉘앙스·의도를 빠짐없이 옮기고, 원문에 없는 내용을 덧붙이지 않습니다.
2. 문체: 자연스러운 한국어 문장으로 씁니다. 영어 어순·구문을 그대로 따르는 직역투를 피합니다.
3. 정렬: 입력의 각 문장(id)마다 번역을 정확히 하나씩 돌려줍니다. 문장을 합치거나 나누거나 순서를 바꾸지 않습니다.
'''

def make_examples(rows, n_sent, seed):
    rng = random.Random(seed)
    rows = rows[:]; rng.shuffle(rows)
    if n_sent:
        rows = rows[:n_sent]
    ex, i, sid = [], 0, 0
    while i < len(rows):
        k = rng.randint(*CHUNK); grp = rows[i:i + k]; i += k
        sents = []
        for r in grp:
            sid += 1; sents.append({'id': f'S{sid:06d}', 'en': r['en'], 'ko': r['ko'], 'translate': True})
        block = [{'para': ' '.join(s['en'] for s in sents), 'sents': sents}]
        _, user = gt.user_text(block, 0, n_ctx=0)      # 앞 문단 문맥 = (없음)
        answer = json.dumps({'translations': [{'id': s['id'], 'ko': s['ko']} for s in sents]}, ensure_ascii=False)
        ex.append({'user': user, 'answer': answer, 'n': len(sents)})
    return ex

train_ex = make_examples(train_rows, N_TRAIN_SENT, SEED)
valid_ex = make_examples(valid_rows, None, SEED + 1)[:N_VALID_EX]
print('학습 예제', len(train_ex), '(문장', sum(e['n'] for e in train_ex), ') · 검증 예제', len(valid_ex))
print('--- 예제 하나 ---'); print(train_ex[0]['user']); print('→', train_ex[0]['answer'])
"""),
md("## 3. 모델 불러오기 (4비트) + LoRA — 세션이 끊겼으면 여기부터 다시"),
code("""from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
import transformers, peft
from google.colab import userdata
try:
    HF_TOKEN = userdata.get('HF_TOKEN')
except Exception:
    HF_TOKEN = None

t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
tok = processor.tokenizer
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={'': 0}, token=HF_TOKEN,
                                             quantization_config=quant)
model.config.use_cache = False
# prepare_model_for_kbit_training은 4비트가 아닌 가중치(큰 임베딩 포함)를 fp32로 바꿔 메모리를 많이 써서 쓰지 않음
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
model.enable_input_require_grads()
targets = sorted({n for n, _ in model.named_modules()
                  if n.endswith(ATTN) and not any(x in n for x in ('vision', 'audio', 'multi_modal'))})
print('LoRA 대상 모듈', len(targets), '예:', targets[:2])
assert targets, 'LoRA 대상 모듈을 못 찾음 — 모듈 이름을 확인해야 함'
pm = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', target_modules=targets, **LORA))
pm.print_trainable_parameters()
print(f'불러오기 {(time.time()-t0)/60:.1f}분 · GPU {torch.cuda.memory_allocated()/1e9:.1f}GB')

# 대답 끝 표시: 생성 때 원출력이 '…<turn|>'로 끝났음 → 같은 토큰을 학습 목표 끝에 붙인다
END = '<turn|>'
END_ID = tok.convert_tokens_to_ids(END)
assert END_ID is not None and END_ID != tok.unk_token_id, f'{END} 토큰이 없음 — 끝 표시를 확인해야 함'
NEED_TT = 'token_type_ids' in inspect.signature(pm.base_model.model.forward).parameters

def prompt_of(user):
    msgs = [{'role': 'system', 'content': TRAIN_SYSTEM + gt.FORMAT}, {'role': 'user', 'content': user}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def encode(e):
    p = tok(prompt_of(e['user']))['input_ids']          # 번역(05·06·08) 때 processor(text=...)와 같은 방식으로 토큰화
    a = tok(e['answer'], add_special_tokens=False)['input_ids'] + [END_ID]
    ids = p + a
    out = {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': [-100] * len(p) + a}
    if NEED_TT:
        out['token_type_ids'] = [0] * len(ids)
    return out

enc_train = [x for x in map(encode, train_ex) if len(x['input_ids']) <= MAX_LEN]
enc_valid = [x for x in map(encode, valid_ex) if len(x['input_ids']) <= MAX_LEN]
lens = sorted(len(x['input_ids']) for x in enc_train)
print(f'학습 {len(enc_train)}/{len(train_ex)} · 검증 {len(enc_valid)} · 길이 중앙 {lens[len(lens)//2]} 최대 {lens[-1]} 토큰')
print('학습 목표 확인:', tok.decode([t for t in enc_train[0]['labels'] if t != -100]))
"""),
md("## 4. 학습 (끊기면 3번 셀 → 이 셀을 다시 실행하면 마지막 중간 저장에서 이어서)"),
code("""from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback

class DS(torch.utils.data.Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

_p = inspect.signature(TrainingArguments.__init__).parameters
opt = {}
opt.update({'train_sampling_strategy': 'group_by_length'} if 'train_sampling_strategy' in _p else {'group_by_length': True})
opt.update({'eval_strategy': 'steps'} if 'eval_strategy' in _p else {'evaluation_strategy': 'steps'})
total_steps = math.ceil(len(enc_train) / (BS * ACCUM)) * EPOCHS
args = TrainingArguments(
    output_dir=str(RUN_DIR / 'ckpt'), per_device_train_batch_size=BS, per_device_eval_batch_size=BS,
    gradient_accumulation_steps=ACCUM, learning_rate=LR, num_train_epochs=EPOCHS,
    warmup_steps=max(1, int(0.03 * total_steps)), lr_scheduler_type='cosine', optim='paged_adamw_8bit',
    logging_steps=10, eval_steps=EVAL_STEPS, save_steps=EVAL_STEPS, save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
    bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
    report_to=[], seed=SEED, dataloader_num_workers=2, remove_unused_columns=False, **opt)
trainer = Trainer(model=pm, args=args, train_dataset=DS(enc_train), eval_dataset=DS(enc_valid),
                  data_collator=DataCollatorForSeq2Seq(tok, model=pm, label_pad_token_id=-100, pad_to_multiple_of=8),
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
has_ckpt = bool(glob.glob(str(RUN_DIR / 'ckpt' / 'checkpoint-*')))
print('이어서 학습' if has_ckpt else '처음부터 학습', '| 전체 스텝', total_steps)
t0 = time.time()
trainer.train(resume_from_checkpoint=True if has_ckpt else None)
hours = (time.time() - t0) / 3600
pm.save_pretrained(str(RUN_DIR / 'adapter_best'))
shutil.make_archive(str(RUN_DIR / 'adapter_best'), 'zip', str(RUN_DIR / 'adapter_best'))
hist = trainer.state.log_history
summary = {'model': MODEL_ID, 'quant': 'nf4+double', 'lora': LORA, 'targets': len(targets), 'bs': BS, 'accum': ACCUM,
           'lr': LR, 'epochs': EPOCHS, 'n_train_examples': len(enc_train), 'n_train_sent': sum(e['n'] for e in train_ex),
           'best_checkpoint': trainer.state.best_model_checkpoint, 'best_eval_loss': trainer.state.best_metric,
           'evals': [(h['step'], round(h['eval_loss'], 4)) for h in hist if 'eval_loss' in h],
           'stopped_at_step': trainer.state.global_step, 'total_steps_planned': total_steps,
           'train_hours_this_session': round(hours, 2), 'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 1e9, 1),
           'transformers': transformers.__version__, 'peft': peft.__version__, 'torch': torch.__version__,
           'gpu': torch.cuda.get_device_name(0), 'train_system_sha': __import__('hashlib').sha256(TRAIN_SYSTEM.encode()).hexdigest()[:12],
           'format_prompt_hash': gt.PROMPT_HASH, 'seed': SEED, 'split_dir': str(SPLIT_DIR)}
(RUN_DIR / 'train_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
print(json.dumps(summary, ensure_ascii=False, indent=1))
"""),
md("## 5. 학습 전·후 비교 (같은 4비트 모델, LoRA 끔/켬)\n세션을 새로 시작했다면 1→2→3번 셀을 실행한 뒤 이 셀부터. 저장된 `adapter_best`를 불러온다."),
code("""if 'trainer' not in globals():          # 세션을 새로 시작해 학습 셀을 건너뛴 경우
    pm.load_adapter(str(RUN_DIR / 'adapter_best'), adapter_name='best'); pm.set_adapter('best')
    print('adapter_best 불러옴')
pm.eval(); pm.config.use_cache = True
tok.padding_side = 'left'

@torch.inference_mode()
def gen(users, use_adapter, system=TRAIN_SYSTEM, bs=8, max_new=1024):
    outs = []
    for s in range(0, len(users), bs):
        texts = [processor.apply_chat_template([{'role': 'system', 'content': system + gt.FORMAT},
                                                {'role': 'user', 'content': u}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False) for u in users[s:s+bs]]
        enc = processor(text=texts, return_tensors='pt', padding=True).to(0)
        if use_adapter:
            out = pm.generate(**enc, max_new_tokens=max_new, do_sample=False)
        else:
            with pm.disable_adapter():
                out = pm.generate(**enc, max_new_tokens=max_new, do_sample=False)
        outs += [processor.decode(o, skip_special_tokens=True) for o in out[:, enc['input_ids'].shape[1]:]]
    return outs

def parsed(text):
    try:
        return gt.parse(text)
    except Exception:
        return {}

# (a) 검증 예제 몇 개: 형식이 유지되는지, 번역이 어떻게 달라지는지
few = valid_ex[:4]
b, l = gen([e['user'] for e in few], False), gen([e['user'] for e in few], True)
for e, x, y in zip(few, b, l):
    gold = json.loads(e['answer'])['translations']
    px, py = parsed(x), parsed(y)
    for g in gold:
        print('정답', g['ko']); print('  전', px.get(g['id'], '(형식 오류)')); print('  후', py.get(g['id'], '(형식 오류)'))
    print()

# (b) 개츠비 지침으로도 형식이 유지되는지 (학습은 일반 지침으로 했으므로 확인)
gat = gen([few[0]['user']], True, system=gt.SYSTEM)[0]
print('개츠비 지침 + 학습 후 출력:', gat[:300])
"""),
md("## 6. (선택) AI Hub 평가 500문장 chrF — 학습 전후, MADLAD lora_v1과 같은 평가셋"),
code("""if RUN_TEST500:
    import sacrebleu
    t0 = time.time()
    test_ex = []
    for i in range(0, len(test_rows), 4):        # 4문장씩 묶어 같은 형식으로
        grp = test_rows[i:i+4]
        sents = [{'id': f'T{i+j:06d}', 'en': r['en'], 'translate': True} for j, r in enumerate(grp)]
        _, user = gt.user_text([{'para': ' '.join(s['en'] for s in sents), 'sents': sents}], 0, n_ctx=0)
        test_ex.append((user, [s['id'] for s in sents]))
    res = {}
    for tag, use in (('base', False), ('lora', True)):
        outs = gen([u for u, _ in test_ex], use)
        hyp = []
        for (u, ids), o in zip(test_ex, outs):
            p = parsed(o); hyp += [p.get(i, '') for i in ids]
        res[tag] = hyp
    refs = [r['ko'] for r in test_rows]
    sc = {t: {'chrF': round(sacrebleu.corpus_chrf(h, [refs]).score, 2), 'BLEU': round(sacrebleu.corpus_bleu(h, [refs]).score, 2),
              'empty': sum(not x for x in h)} for t, h in res.items()}
    rng = random.Random(SEED); n = len(refs); d = []
    for _ in range(1000):
        ix = [rng.randrange(n) for _ in range(n)]; rr = [[refs[i] for i in ix]]
        d.append(sacrebleu.corpus_chrf([res['lora'][i] for i in ix], rr).score - sacrebleu.corpus_chrf([res['base'][i] for i in ix], rr).score)
    d.sort()
    sc['chrF_diff'] = round(sc['lora']['chrF'] - sc['base']['chrF'], 2)
    sc['chrF_diff_95ci'] = [round(d[25], 2), round(d[974], 2)]
    sc['minutes'] = round((time.time() - t0) / 60, 1)
    sc['note'] = '4비트(nf4) 모델로 번역한 값 — 08 노트북의 원본(bf16)+어댑터 결과와 다를 수 있음'
    (RUN_DIR / 'aihub_test500_scores.json').write_text(json.dumps(sc, ensure_ascii=False, indent=1))
    with open(RUN_DIR / 'aihub_test500_outputs.jsonl', 'w', encoding='utf-8') as f:
        for r, x, y in zip(test_rows, res['base'], res['lora']):
            f.write(json.dumps({'id': r['id'], 'en': r['en'], 'ko_ref': r['ko'], 'base': x, 'lora': y}, ensure_ascii=False) + '\\n')
    print(json.dumps(sc, ensure_ascii=False, indent=1))
"""),
md("""## 끝나면
- 알려 줄 것: `runs/gemma_lora_v1/train_summary.json`, (했다면) `aihub_test500_scores.json`
- 다음: `adapter_best.zip`을 내려받아 08 노트북(공용 계정 80GB)에 올려 개츠비 1·3장 학습 후 번역 → 6축 비교
- 런타임은 바로 끊기. `runs/gemma_lora_v1/`은 지우지 말 것
"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
                                                 "kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = HERE / "07_학습_gemma4_31b_QLoRA.ipynb"
nbf.write(nb, out)
print(out)
