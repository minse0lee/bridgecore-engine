"""수정본 노트북(03_설정비교_본학습_v2.ipynb) 생성기.  python build_notebook_03.py

02(lora_v1)와 달라진 점
- 모델 커밋 해시 고정·기록, pip freeze 저장, 결정적 연산(같은 환경이면 같은 결과)
- 최적 시점을 검증 300문장 chrF로 고름 (빔 4, 최종 번역과 같은 설정)
- 학습데이터 검사 + 한/영 글자 비율이 극단적인 문장 제외
- BLEU는 한국어 형태소(ko-mecab) 기준, 실패하면 spBLEU(flores200)
- 학습 설정 A~D를 1에폭씩 비교(search) → 가장 좋은 설정으로 3에폭(final)
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
md("""# MADLAD-400 10B 영→한 LoRA — 설정 비교와 본 학습 (v2)

**02(lora_v1)에서 바뀐 점**
1. **재현**: 모델 커밋 해시 고정, 전체 라이브러리 목록 저장, 결정적 연산 켜기 (같은 GPU·같은 버전이면 같은 결과. 약간 느려짐)
2. **최적 시점 선택**: 검증 300문장을 실제로 번역해 chrF로 고름 (손실 대신 번역 품질로)
3. **데이터**: 학습데이터 검사 + 한/영 글자 비율이 0.2 미만·1.0 초과인 문장 제외
4. **BLEU**: 한국어 형태소(ko-mecab) 기준
5. **설정 비교**: A~D를 1에폭씩 학습해 검증 chrF로 비교 → 가장 좋은 설정으로 3에폭

**규칙 (결과 보기 전에 정해 둠)**
- 설정·모델 선택은 **검증셋 chrF로만**. 평가 500문장과 파일럿 60문항 결과는 고르는 데 쓰지 않는다
- 검증 chrF가 A보다 0.5점 넘게 높을 때만 A 대신 그 설정을 고른다 (300문장 chrF는 1점 안팎으로 흔들리므로)

**GPU**: 80GB A100이면 더 빠름 (시작 셀에 GPU 용량이 찍힘). 40GB가 배정되면 메모리 절약 설정을 자동으로 켜고 돌아감 (약 1.6배 느림)

**실행 방법**
- `MODE = 'search'`로 한 번 실행 (A~D, 약 4~5시간, 끊기면 다시 실행하면 끝난 설정은 건너뜀) → 결과 표 확인
- `MODE = 'final'`로 바꿔 실행 (고른 설정으로 3에폭 + 전후 번역 + 채점, 약 3시간)
- 매번: 설치 셀(①)만 실행 → '런타임 → 세션 다시 시작' → 설정 셀부터
"""),
code("""# ① 맨 먼저 이 셀만 실행 → 끝나면 '런타임 → 세션 다시 시작' → 다음 셀부터
!pip list --format=freeze | grep -E '^(torch|torchvision|torchaudio)==' > /content/torch_pin.txt
!cat /content/torch_pin.txt
!pip -q install -c /content/torch_pin.txt transformers peft accelerate sentencepiece "sacrebleu[ko]" protobuf huggingface_hub
!pip -q uninstall -y torchao
"""),
md("## 1. 설정 (결정적 연산은 torch를 쓰기 전에 켜야 해서 이 셀이 맨 앞)"),
code("""import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'   # 결정적 행렬곱 (CUDA를 쓰기 전에 설정)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # 메모리 조각화 줄이기
import json, time, math, random, platform, statistics, inspect, glob, gc, subprocess
from pathlib import Path
import torch

MODE = 'search'                 # 'search' = A~D 1에폭 비교 / 'final' = 고른 설정으로 3에폭 + 번역·채점
DETERMINISTIC = True
SEED = 20261104
if DETERMINISTIC:
    torch.use_deterministic_algorithms(True, warn_only=True)  # 결정적 구현이 없는 연산은 경고만
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from google.colab import drive
drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/bridgecore_mt')
SPLIT_DIR = BASE / 'splits_v1'
RUNS = BASE / 'runs'
IN_DIR = BASE / 'translate_inputs'

MODEL_ID = 'google/madlad400-10b-mt'
REVISION = None                 # None이면 지금 최신 커밋을 받아 그 해시로 고정·기록. 재현할 땐 기록된 해시를 여기에
PREFIX = '<2ko> '
GEN = dict(num_beams=4, do_sample=False, max_new_tokens=256)
MAX_SRC, MAX_TGT = 256, 256
BS, ACCUM = 16, 1
GRAD_CKPT = False               # 메모리 부족이면 True
EVAL_STEPS = 1000
VALID_GEN_N = 300               # 체크포인트마다 번역해서 chrF를 잴 검증 문장 수
PATIENCE = 3
RATIO_FILTER = (0.2, 1.0)       # 한/영 글자 비율(공백 제외)이 이 범위 밖이면 학습에서 제외
CU_PER_HOUR = 7.52

CONFIGS = {   # 학습률·LoRA 크기·붙이는 층
    'A': dict(r=16, lora_alpha=32, target_modules=['q', 'v'], lr=2e-4),   # = lora_v1 설정
    'B': dict(r=16, lora_alpha=32, target_modules=['q', 'v'], lr=1e-4),
    'C': dict(r=16, lora_alpha=32, target_modules=['q', 'v'], lr=5e-4),
    'D': dict(r=64, lora_alpha=128, target_modules='all-linear', lr=2e-4, grad_ckpt=True),
}
# lora_v1(A, 3에폭)에서 최대 메모리가 80.3GB(카드 85.1GB)라 여유가 작다. D는 LoRA가 커서 gradient checkpointing을 켬
SEARCH_EPOCHS, FINAL_EPOCHS = 1, 3
FINAL_CONFIG = None             # None이면 search 결과와 위 규칙으로 자동 선택

# GPU 메모리에 맞춰 자동 조정: Colab A100은 40GB가 배정될 때도 있다 (lora_v1은 80GB 카드에서 최대 80.3GB 사용)
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
EVAL_BS = 32
if GPU_GB < 60:
    GRAD_CKPT, EVAL_BS = True, 16
    print(f'⚠ GPU {GPU_GB:.0f}GB: 모든 설정에서 gradient checkpointing을 켜고 평가 배치를 16으로 줄입니다 (학습이 약 1.6배 느려짐)')
else:
    print(f'GPU {GPU_GB:.0f}GB')

random.seed(SEED); torch.manual_seed(SEED)
from huggingface_hub import HfApi
if REVISION is None:
    REVISION = HfApi().model_info(MODEL_ID).sha
print('모델 커밋:', REVISION)

import transformers, peft, accelerate, sentencepiece, sacrebleu
RUNS.mkdir(parents=True, exist_ok=True)
freeze = subprocess.run(['pip', 'freeze'], capture_output=True, text=True).stdout
env = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
       "transformers": transformers.__version__, "peft": peft.__version__, "accelerate": accelerate.__version__,
       "sentencepiece": sentencepiece.__version__, "sacrebleu": sacrebleu.__version__,
       "gpu": torch.cuda.get_device_name(0), "model_id": MODEL_ID, "revision": REVISION, "prefix": PREFIX,
       "gen": GEN, "seed": SEED, "deterministic": DETERMINISTIC, "bs": BS, "accum": ACCUM, "grad_ckpt": GRAD_CKPT,
       "gpu_gb": round(GPU_GB, 1), "eval_bs": EVAL_BS,
       "ratio_filter": RATIO_FILTER, "valid_gen_n": VALID_GEN_N, "configs": CONFIGS}
print(json.dumps({k: v for k, v in env.items() if k != 'configs'}, ensure_ascii=False, indent=1))
"""),
md("## 2. 데이터 불러오기와 학습데이터 검사"),
code("""load_jsonl = lambda p: [json.loads(l) for l in open(p, encoding='utf-8')]
train_all, valid, test = (load_jsonl(SPLIT_DIR / f'{n}.jsonl') for n in ('train', 'valid', 'test'))
man = json.loads((SPLIT_DIR / 'manifest.json').read_text())
env['split_sha256'] = man['split_sha256']

ratio = lambda r: len(r['ko'].replace(' ', '')) / max(1, len(r['en'].replace(' ', '')))
rs = sorted(ratio(r) for r in train_all)
qq = lambda p: round(rs[int(p * (len(rs) - 1))], 2)
lo, hi = RATIO_FILTER
dropped = [r for r in train_all if not (lo <= ratio(r) <= hi)]
train = [r for r in train_all if lo <= ratio(r) <= hi]
import re
audit = {"train_before": len(train_all), "train_after": len(train), "dropped_ratio": len(dropped),
         "ratio_pct_1_5_50_95_99": [qq(.01), qq(.05), qq(.5), qq(.95), qq(.99)],
         "ko_with_latin_word": sum(bool(re.search(r'[A-Za-z]{4,}', r['ko'])) for r in train_all),
         "dup_ko": len(train_all) - len({r['ko'] for r in train_all})}
env['data_audit'] = audit
print(json.dumps(audit, ensure_ascii=False, indent=1))
for r in dropped[:5]:
    print('  제외 예:', round(ratio(r), 2), '|', r['en'][:90], '|', r['ko'][:60])
valid_gen = valid[:VALID_GEN_N]   # valid는 이미 무작위 순서
"""),
md("## 3. 학습 함수 (검증 chrF로 최적 시점 선택)"),
code("""from transformers import (T5ForConditionalGeneration, T5Tokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

tok = T5Tokenizer.from_pretrained(MODEL_ID, revision=REVISION)

def load_base(grad_ckpt=None):
    gc_on = GRAD_CKPT if grad_ckpt is None else grad_ckpt
    try:
        m = T5ForConditionalGeneration.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.bfloat16, device_map={'': 0})
    except TypeError:
        m = T5ForConditionalGeneration.from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=torch.bfloat16, device_map={'': 0})
    m.config.use_cache = not gc_on
    m.generation_config.use_cache = True
    if gc_on:
        m.gradient_checkpointing_enable(); m.enable_input_require_grads()
    return m

class DS(torch.utils.data.Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x = tok(PREFIX + r['en'], max_length=MAX_SRC, truncation=True)
        x['labels'] = tok(text_target=r['ko'], max_length=MAX_TGT, truncation=True).input_ids
        return x

def compute_chrf(eval_pred):
    preds, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(preds, tuple): preds = preds[0]
    import numpy as np
    preds = np.where(preds < 0, tok.pad_token_id, preds)
    labels = np.where(labels < 0, tok.pad_token_id, labels)
    hyp = tok.batch_decode(preds, skip_special_tokens=True)
    ref = tok.batch_decode(labels, skip_special_tokens=True)
    return {"chrf": sacrebleu.corpus_chrf(hyp, [ref]).score}

def train_config(name, cfg, epochs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'env.json').write_text(json.dumps({**env, "config_name": name, "config": cfg, "epochs": epochs}, ensure_ascii=False, indent=1))
    (out_dir / 'pip_freeze.txt').write_text(freeze)
    gc_on = cfg.get('grad_ckpt', GRAD_CKPT)
    base = load_base(gc_on)
    lcfg = {k: v for k, v in cfg.items() if k not in ('lr', 'grad_ckpt')}
    pm = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, lora_dropout=0.05, **lcfg))
    pm.print_trainable_parameters()
    _p = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    opt = {'train_sampling_strategy': 'group_by_length'} if 'train_sampling_strategy' in _p else {'group_by_length': True}
    opt.update({'eval_strategy': 'steps'} if 'eval_strategy' in _p else {'evaluation_strategy': 'steps'})
    total = math.ceil(len(train) / (BS * ACCUM)) * epochs
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / 'ckpt'), per_device_train_batch_size=BS, per_device_eval_batch_size=EVAL_BS,
        gradient_accumulation_steps=ACCUM, learning_rate=cfg['lr'], num_train_epochs=epochs,
        warmup_steps=int(0.03 * total), lr_scheduler_type='linear',
        logging_steps=50, eval_steps=EVAL_STEPS, save_steps=EVAL_STEPS, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model='chrf', greater_is_better=True,
        predict_with_generate=True, generation_num_beams=GEN['num_beams'], generation_max_length=GEN['max_new_tokens'],
        bf16=True, gradient_checkpointing=gc_on, report_to=[], seed=SEED, data_seed=SEED,
        dataloader_num_workers=2, **opt)
    trainer = Seq2SeqTrainer(model=pm, args=args, train_dataset=DS(train), eval_dataset=DS(valid_gen),
                             data_collator=DataCollatorForSeq2Seq(tok, model=pm, label_pad_token_id=-100, pad_to_multiple_of=8),
                             compute_metrics=compute_chrf, callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)])
    has_ckpt = bool(glob.glob(str(out_dir / 'ckpt' / 'checkpoint-*')))
    print(name, cfg, '| 이어서' if has_ckpt else '| 처음부터', '| 전체 스텝', total)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)
    hours = (time.time() - t0) / 3600
    pm.save_pretrained(str(out_dir / 'adapter_best'))
    evals = [(h['step'], round(h.get('eval_loss', float('nan')), 4), round(h['eval_chrf'], 2))
             for h in trainer.state.log_history if 'eval_chrf' in h]
    summary = {"config_name": name, "config": cfg, "epochs": epochs, "best_valid_chrf": trainer.state.best_metric,
               "best_checkpoint": trainer.state.best_model_checkpoint, "evals_step_loss_chrf": evals,
               "stopped_at_step": trainer.state.global_step, "total_steps_planned": total,
               "train_hours_this_session": round(hours, 2), "cu_est": round(hours * CU_PER_HOUR, 1),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return pm, summary

"""),
md("## 4. MODE = 'search': A~D 1에폭씩 (끝난 설정은 건너뜀)"),
code("""if MODE == 'search':
    for name, cfg in CONFIGS.items():
        out = RUNS / f'search_{name}'
        if (out / 'summary.json').exists():
            print(name, '이미 끝남, 건너뜀'); continue
        pm, _ = train_config(name, cfg, SEARCH_EPOCHS, out)
        del pm; gc.collect(); torch.cuda.empty_cache()

res, pick = {}, 'A'
for name in CONFIGS:
    p = RUNS / f'search_{name}' / 'summary.json'
    if p.exists(): res[name] = json.loads(p.read_text())
print('설정 | 검증 chrF(최고) | 시간')
for n, s in res.items():
    print(f"{n} {CONFIGS[n]} | {s['best_valid_chrf']:.2f} | {s['train_hours_this_session']}h")
if 'A' in res:
    a = res['A']['best_valid_chrf']
    better = {n: s['best_valid_chrf'] for n, s in res.items() if s['best_valid_chrf'] > a + 0.5}
    pick = max(better, key=better.get) if better else 'A'
    print('규칙에 따른 선택:', pick, '(A보다 0.5점 넘게 높은 설정이 없으면 A)')
"""),
md("## 5. MODE = 'final': 고른 설정으로 3에폭 → 학습 전후 번역 → 채점"),
code("""if MODE == 'final':
    name = FINAL_CONFIG or pick
    RUN_NAME = f'lora_v2_{name}'
    RUN_DIR, OUT_DIR = RUNS / RUN_NAME, BASE / 'translations' / RUN_NAME
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if (RUN_DIR / 'summary.json').exists():   # 학습은 끝났고 번역만 남은 경우
        pm = PeftModel.from_pretrained(load_base(), str(RUN_DIR / 'adapter_best')); print('학습 결과 불러옴')
    else:
        pm, _ = train_config(name, CONFIGS[name], FINAL_EPOCHS, RUN_DIR)
    pm.eval(); pm.config.use_cache = True

    def translate(rows, use_adapter, bs=EVAL_BS):
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]['en']))
        outs = [None] * len(rows)
        for s in range(0, len(order), bs):
            idx = order[s:s+bs]
            enc = tok([PREFIX + rows[i]['en'] for i in idx], return_tensors='pt', padding=True,
                      truncation=True, max_length=MAX_SRC).to(0)
            with torch.no_grad():
                if use_adapter:
                    out = pm.generate(**enc, **GEN)
                else:
                    with pm.disable_adapter():
                        out = pm.generate(**enc, **GEN)
            for i, t in zip(idx, tok.batch_decode(out, skip_special_tokens=True)):
                outs[i] = t
        return outs

    def write_pair(stem, rows, b, l):
        meta = {"run": RUN_NAME, "config": CONFIGS[name], "revision": REVISION, "gen": GEN, "prefix": PREFIX,
                "transformers": transformers.__version__, "torch": torch.__version__}
        for tag, outs in (('base', b), ('lora', l)):
            with open(OUT_DIR / f'{stem}__{tag}.jsonl', 'w', encoding='utf-8') as f:
                for r, o in zip(rows, outs):
                    f.write(json.dumps({**r, 'mt': o, 'mt_system': f'madlad10b_{tag}', 'mt_meta': meta}, ensure_ascii=False) + '\\n')

    # lora_v1과 v2 중 무엇을 최종으로 보고할지는 검증 300문장 chrF로 정한다 (평가셋으로 고르지 않음)
    vref = [[r['ko'] for r in valid_gen]]
    v2_valid = sacrebleu.corpus_chrf(translate(valid_gen, True), vref).score
    v1_valid = None
    if (RUNS / 'lora_v1' / 'adapter_best').exists():
        pm.load_adapter(str(RUNS / 'lora_v1' / 'adapter_best'), adapter_name='v1')
        pm.set_adapter('v1'); v1_valid = sacrebleu.corpus_chrf(translate(valid_gen, True), vref).score
        pm.set_adapter('default')
    sel = {"v2_valid_chrf": round(v2_valid, 2), "v1_valid_chrf": None if v1_valid is None else round(v1_valid, 2),
           "report": RUN_NAME if (v1_valid is None or v2_valid > v1_valid + 0.5) else 'lora_v1',
           "rule": "v2가 검증 chrF에서 v1보다 0.5점 넘게 높을 때만 v2를 최종으로 보고"}
    (RUN_DIR / 'v1_vs_v2_valid.json').write_text(json.dumps(sel, ensure_ascii=False, indent=1))
    print(json.dumps(sel, ensure_ascii=False, indent=1))

    t0 = time.time()
    test_b, test_l = translate(test, False), translate(test, True)
    write_pair('aihub_test500', test, test_b, test_l)
    print(f'평가 500 전후 번역 {(time.time()-t0)/60:.1f}분')
    for p in sorted(IN_DIR.glob('*.jsonl')):
        rows = [r for r in load_jsonl(p) if (r.get('en') or '').strip()]
        b, l = translate(rows, False), translate(rows, True)
        write_pair(p.stem, rows, b, l)
        print(f'{p.name}: {len(rows)}문장 · 전후가 다른 문장 {sum(x != y for x, y in zip(b, l))}개')
"""),
md("### 채점: chrF(주), chrF++, BLEU(ko-mecab, 안 되면 flores200), chrF 차이 부트스트랩"),
code("""if MODE == 'final':
    from sacrebleu.metrics import BLEU, CHRF
    refs = [r['ko'] for r in test]
    bleu = None
    for t in ('ko-mecab', 'flores200', '13a'):
        try:
            bleu = BLEU(tokenize=t); bleu.corpus_score(test_b[:2], [refs[:2]]); break
        except Exception as e:
            print('BLEU 토크나이저', t, '사용 불가:', str(e)[:80]); bleu = None
    chrf, chrfpp = CHRF(), CHRF(word_order=2)
    def sc(h):
        return {"chrF": round(chrf.corpus_score(h, [refs]).score, 2), "chrF++": round(chrfpp.corpus_score(h, [refs]).score, 2),
                "BLEU": round(bleu.corpus_score(h, [refs]).score, 2)}
    sb, sl = sc(test_b), sc(test_l)
    rng = random.Random(SEED); n = len(refs); d = []
    for _ in range(1000):
        ix = [rng.randrange(n) for _ in range(n)]; rr = [[refs[i] for i in ix]]
        d.append(chrf.corpus_score([test_l[i] for i in ix], rr).score - chrf.corpus_score([test_b[i] for i in ix], rr).score)
    d.sort()
    out = {"run": RUN_NAME, "base": sb, "lora": sl, "chrF_diff": round(sl['chrF'] - sb['chrF'], 2),
           "chrF_diff_95ci": [round(d[int(.025 * len(d))], 2), round(d[int(.975 * len(d)) - 1], 2)],
           "signatures": {"chrF": str(chrf.get_signature()), "chrF++": str(chrfpp.get_signature()), "BLEU": str(bleu.get_signature())},
           "n": n}
    (RUN_DIR / 'aihub_test500_scores.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    for r, b, l in list(zip(test, test_b, test_l))[:5]:
        print('-', r['en'], '\\n  전:', b, '\\n  후:', l, '\\n  정답:', r['ko'])
"""),
md("""## 끝나면 알려 줄 것
- search: 위 표 (설정별 검증 chrF)와 '규칙에 따른 선택'
- final: `runs/lora_v2_*/summary.json`, `aihub_test500_scores.json`, 그리고 `translations/lora_v2_*/` 안의 `pilot60_src__base.jsonl`·`__lora.jsonl`
- 런타임은 끝나면 바로 끊기
"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
                                                 "kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = HERE / "03_설정비교_본학습_v2.ipynb"
nbf.write(nb, out)
print(out)
