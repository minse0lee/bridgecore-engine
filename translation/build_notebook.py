"""시험 학습 노트북(01_시험학습_MADLAD10B_LoRA.ipynb) 생성기.  python build_notebook.py"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
PREP = (HERE / "prep_data.py").read_text(encoding="utf-8")

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
md("""# MADLAD-400 10B 영→한 LoRA — 시험 학습

**목적**: 본 학습 전에 메모리·속도를 재서 전체 학습 시간과 Colab 컴퓨팅 단위 소모를 추정한다. 이 노트북의 짧은 학습 결과는 성능 비교에 쓰지 않는다.

**준비**
1. 런타임 → 런타임 유형 변경 → **A100**, **고용량 RAM 켜기**(80GB A100). 시작하면 오른쪽 위 리소스 패널에서 *시간당 컴퓨팅 단위*를 확인해 아래 `CU_PER_HOUR`에 적기
2. 구글 드라이브에 폴더 `MyDrive/bridgecore_mt/aihub/`를 만들고 `학습데이터.zip`, `검증데이터.zip`을 올리기 (AI Hub 원본 압축 그대로)
3. 위에서부터 차례로 실행

**순서**: 환경 기록 → 데이터 분할 저장(한 번만) → 모델 불러오기 → 기본 모델 번역 속도 → LoRA 짧은 학습(시간·메모리 측정) → 전체 학습 추정
"""),
code("""!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| bf16 지원:", torch.cuda.is_bf16_supported())
assert torch.cuda.is_bf16_supported(), "bf16을 지원하지 않는 GPU입니다. A100으로 바꿔 주세요"
"""),
code("""!pip -q install transformers peft accelerate sentencepiece sacrebleu protobuf
"""),
md("## 1. 설정과 환경 기록"),
code("""import json, os, time, platform, random
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')

BASE = Path('/content/drive/MyDrive/bridgecore_mt')
ZIP_DIR = BASE / 'aihub'
SPLIT_DIR = BASE / 'splits_v1'          # 분할 파일 (한 번 만들면 다시 만들지 않음)
RUN_DIR = BASE / 'runs' / 'pilot_test'  # 시험 학습 결과
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = 'google/madlad400-10b-mt'
PREFIX = '<2ko> '                       # 모델 카드 형식: "<2xx> " + 원문
SEED = 20261104
GEN = dict(num_beams=4, do_sample=False, max_new_tokens=256)  # 파일럿과 같은 탐색 설정
CU_PER_HOUR = 7.52                      # ← 리소스 패널에 나온 값으로 바꾸기 (A100 80GB 참고값)

import transformers, peft, accelerate, sentencepiece, sacrebleu
env = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
       "transformers": transformers.__version__, "peft": peft.__version__, "accelerate": accelerate.__version__,
       "sentencepiece": sentencepiece.__version__, "sacrebleu": sacrebleu.__version__,
       "gpu": torch.cuda.get_device_name(0),
       "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
       "model_id": MODEL_ID, "prefix": PREFIX, "gen": GEN, "seed": SEED}
(RUN_DIR / 'env.json').write_text(json.dumps(env, ensure_ascii=False, indent=1))
print(json.dumps(env, ensure_ascii=False, indent=1))
"""),
md("## 2. 데이터 분할 저장 (처음 한 번만)\n학습 = Training의 영어 원문 교육 문장, 검증 1,000 / 평가 500 = Validation에서 무작위. 평가·검증과 같은 영어 문장은 학습에서 뺀다. 이미 분할 파일이 있으면 다시 나누지 않는다 (나중에 같은 평가셋으로 다시 재기 위해)."),
code("%%writefile prep_data.py\n" + PREP),
code("""from prep_data import prepare
if (SPLIT_DIR / 'manifest.json').exists():
    man = json.loads((SPLIT_DIR / 'manifest.json').read_text())
    print('이미 있는 분할을 씁니다')
else:
    man = prepare(ZIP_DIR / '학습데이터.zip', ZIP_DIR / '검증데이터.zip', SPLIT_DIR, seed=SEED)
print(json.dumps(man, ensure_ascii=False, indent=1))

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding='utf-8')]
train, valid, test = (load_jsonl(SPLIT_DIR / f'{n}.jsonl') for n in ('train', 'valid', 'test'))
print(len(train), len(valid), len(test))
"""),
md("## 3. 모델 불러오기 (bf16)"),
code("""from transformers import T5ForConditionalGeneration, T5Tokenizer
t0 = time.time()
tok = T5Tokenizer.from_pretrained(MODEL_ID)
try:
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={'': 0})
except TypeError:  # 이전 버전 transformers
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map={'': 0})
load_min = (time.time() - t0) / 60
print(f'불러오기 {load_min:.1f}분 · 파라미터 {model.num_parameters()/1e9:.2f}B · GPU 메모리 {torch.cuda.memory_allocated()/1e9:.1f}GB')

# 토큰 길이 (최대 길이 정하기)
import statistics
samp = random.Random(SEED).sample(train, min(2000, len(train)))
ls = sorted(len(tok(PREFIX + r['en']).input_ids) for r in samp)
lt = sorted(len(tok(r['ko']).input_ids) for r in samp)
q = lambda xs, p: xs[int(p * (len(xs) - 1))]
print('원문 토큰 중앙값/95%/최대', statistics.median(ls), q(ls, .95), ls[-1], '| 번역 토큰', statistics.median(lt), q(lt, .95), lt[-1])
print('<2ko> 토큰:', tok.convert_ids_to_tokens(tok(PREFIX + 'Hello').input_ids)[:4])
"""),
md("## 4. 기본 모델 번역 속도 (평가 분할 64문장)\n여기 chrF는 속도 확인용 참고치다. 학습 전후 비교는 본 학습 뒤 평가 분할 전체로 따로 한다."),
code("""import sacrebleu
def translate(m, rows, bs=16):
    m.eval(); outs = []
    for i in range(0, len(rows), bs):
        enc = tok([PREFIX + r['en'] for r in rows[i:i+bs]], return_tensors='pt', padding=True, truncation=True, max_length=256).to(0)
        with torch.no_grad():
            out = m.generate(**enc, **GEN)
        outs += tok.batch_decode(out, skip_special_tokens=True)
    return outs

probe = test[:64]
torch.cuda.reset_peak_memory_stats(); t0 = time.time()
base_out = translate(model, probe)
gen_sec = (time.time() - t0) / len(probe)
chrf_base = sacrebleu.corpus_chrf(base_out, [[r['ko'] for r in probe]]).score
print(f'문장당 {gen_sec:.2f}초 · 최대 메모리 {torch.cuda.max_memory_allocated()/1e9:.1f}GB · chrF(64문장 참고) {chrf_base:.1f}')
for r, o in list(zip(probe, base_out))[:3]:
    print('-', r['en'], '\\n  기본:', o, '\\n  정답:', r['ko'])
"""),
md("## 5. LoRA 짧은 학습 (시간·메모리 측정)\n설정은 사전 실험과 같게 시작: r=16, alpha=32, dropout 0.05, 어텐션 q·v. 메모리 부족(OOM)이 나면 `BS`를 8로 줄이고 `ACCUM`을 2로."),
code("""from peft import LoraConfig, get_peft_model, TaskType
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback

MAX_SRC, MAX_TGT = 256, 256
BS, ACCUM, LR = 16, 1, 2e-4
TEST_STEPS = 150
TEST_N = 6000  # 시험용 학습 문장 수

model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
lcfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=['q', 'v'])
pm = get_peft_model(model, lcfg)
pm.print_trainable_parameters()

class DS(torch.utils.data.Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x = tok(PREFIX + r['en'], max_length=MAX_SRC, truncation=True)
        x['labels'] = tok(text_target=r['ko'], max_length=MAX_TGT, truncation=True).input_ids
        return x

class Timer(TrainerCallback):
    def __init__(self): self.t = []
    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize(); self.t.append(time.time())

sub = random.Random(SEED).sample(train, min(TEST_N, len(train)))
timer = Timer()
import inspect
# 길이가 비슷한 문장끼리 묶어 패딩 낭비 줄이기 (transformers 5부터 옵션 이름이 바뀜)
_p = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
group = {'train_sampling_strategy': 'group_by_length'} if 'train_sampling_strategy' in _p else {'group_by_length': True}
args = Seq2SeqTrainingArguments(output_dir=str(RUN_DIR / 'ckpt'), per_device_train_batch_size=BS,
                                gradient_accumulation_steps=ACCUM, learning_rate=LR, max_steps=TEST_STEPS,
                                warmup_steps=10, logging_steps=10, save_steps=TEST_STEPS, save_total_limit=1,
                                bf16=True, gradient_checkpointing=True, report_to=[], seed=SEED,
                                dataloader_num_workers=2, **group)
trainer = Seq2SeqTrainer(model=pm, args=args, train_dataset=DS(sub),
                         data_collator=DataCollatorForSeq2Seq(tok, model=pm, label_pad_token_id=-100, pad_to_multiple_of=8),
                         callbacks=[timer])
torch.cuda.reset_peak_memory_stats()
trainer.train()
steps = [b - a for a, b in zip(timer.t[10:], timer.t[11:])]  # 앞 10스텝(준비·워밍업) 제외
sec_step = statistics.median(steps)
peak = torch.cuda.max_memory_allocated() / 1e9
print(f'스텝당 {sec_step:.2f}초 (배치 {BS}×{ACCUM}) · 최대 메모리 {peak:.1f}GB')
print([round(l['loss'], 3) for l in trainer.state.log_history if 'loss' in l])
"""),
md("## 6. 저장·불러오기 확인과 전체 학습 추정"),
code("""ad = RUN_DIR / 'adapter_test'
pm.save_pretrained(str(ad))
print('어댑터 저장:', sorted(os.listdir(ad)))

import math
eff = BS * ACCUM
steps_per_epoch = math.ceil(len(train) / eff)
h_epoch = steps_per_epoch * sec_step / 3600
gen_hours = (len(test) + len(valid)) * gen_sec / 3600  # 평가 분할+검증 분할 번역 1회
est = {"train_sentences": len(train), "effective_batch": eff, "sec_per_step": round(sec_step, 3),
       "peak_mem_gb": round(peak, 1), "steps_per_epoch": steps_per_epoch,
       "hours_per_epoch": round(h_epoch, 2), "hours_3_epochs": round(3 * h_epoch, 2),
       "gen_sec_per_sentence": round(gen_sec, 3), "hours_translate_valid_test": round(gen_hours, 2),
       "cu_per_hour": CU_PER_HOUR, "cu_3_epochs": round(3 * h_epoch * CU_PER_HOUR, 1),
       "load_minutes": round(load_min, 1), "chrf_base_probe64": round(chrf_base, 1)}
(RUN_DIR / 'estimate.json').write_text(json.dumps(est, ensure_ascii=False, indent=1))
print(json.dumps(est, ensure_ascii=False, indent=1))
print(f"→ 1에폭 약 {h_epoch:.1f}시간, 3에폭 약 {3*h_epoch:.1f}시간, 컴퓨팅 단위 약 {3*h_epoch*CU_PER_HOUR:.0f}개 (번역·평가 시간 별도)")
"""),
code("""# 짧은 학습 뒤 번역이 깨지지 않았는지만 확인 (성능 비교 아님)
pm.config.use_cache = True
after = translate(pm, probe[:16])
for r, b, a in list(zip(probe, base_out, after))[:5]:
    print('-', r['en'], '\\n  기본:', b, '\\n  시험:', a)
"""),
md("""## 끝나면
`MyDrive/bridgecore_mt/runs/pilot_test/estimate.json`과 `env.json` 내용을 알려 주세요. 이걸로 본 학습 설정(에폭 수, 배치, 학습 문장 수)과 일정을 정한다.

- 분할 파일(`splits_v1/`)은 지우지 말 것. 본 학습과 학습 전후 평가에 같은 파일을 쓴다
- 런타임은 끝나면 바로 끊기 (켜 두는 동안 단위가 계속 줄어듦)
"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
                                                 "kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = HERE / "01_시험학습_MADLAD10B_LoRA.ipynb"
nbf.write(nb, out)
print(out)
