#!/usr/bin/env bash
# 번역 쪽 시험을 한 번에 돌린다. 먼저 모델을 받아 둔 뒤(python gatsby_full_gpu.py --download-only) 실행한다.
#   nohup bash run_policy.sh > run.log 2>&1 &      그다음   tail -f run.log
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
N=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "[$(date '+%H:%M')] GPU ${N}장"
$PY -c "import gatsby_full_gpu as g, sys; sys.exit(0 if g.local_snapshot() else 1)" \
  || { echo "모델이 아직 없습니다. 먼저  python gatsby_full_gpu.py --download-only  를 실행하세요."; exit 1; }

if [ "$N" -ge 4 ]; then
  echo "[$(date '+%H:%M')] 1단계: policy 와 policy_next5 를 동시에"
  CUDA_VISIBLE_DEVICES=0,1 $PY gatsby_policy_gpu.py --cond policy       > log_policy.txt 2>&1 &
  CUDA_VISIBLE_DEVICES=2,3 $PY gatsby_policy_gpu.py --cond policy_next5 > log_policy_next5.txt 2>&1 &
  wait
else
  echo "[$(date '+%H:%M')] 1단계: policy → policy_next5 (GPU가 4장 미만이라 차례로)"
  $PY gatsby_policy_gpu.py --cond policy       > log_policy.txt 2>&1
  $PY gatsby_policy_gpu.py --cond policy_next5 > log_policy_next5.txt 2>&1
fi
tail -n 2 log_policy.txt log_policy_next5.txt
echo "[$(date '+%H:%M')] 2단계: base (같은 기계에서의 비교 기준)"
$PY gatsby_policy_gpu.py --cond base > log_base.txt 2>&1
tail -n 2 log_base.txt
echo "[$(date '+%H:%M')] 모두 끝 — 내려받을 파일:"
ls -la out/policy_*.jsonl out/run_info_policy_*.jsonl
