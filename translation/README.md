# 번역엔진 (MADLAD-400 10B 영→한 LoRA)

- `prep_data.py`: AI Hub 한국어-영어 번역 말뭉치(사회과학)에서 영어 원문 교육 문장으로 학습·검증·평가 분할 저장
  - 학습 = Training 압축의 `source_language == en` · `domain == 교육`, 검증 1,000 / 평가 500 = Validation 압축에서 무작위 (시드 20261104)
  - 열: `en` 영어 원문, `ko` 사람 번역 (`mt`는 한→영 기계번역이라 안 씀)
  - Validation 압축 확인(2026-11-04): CSV 151,316행 중 영어 원문 9,713행(교육 9,653)
- `01_시험학습_MADLAD10B_LoRA.ipynb`: Colab A100에서 메모리·속도 측정용 짧은 학습 → 전체 학습 시간·컴퓨팅 단위 추정 (`build_notebook.py`로 생성)
- `02_본학습_MADLAD10B_LoRA.ipynb`: LoRA 본 학습(검증 손실로 최적 시점 저장·이어서 학습·조기 종료), 학습 전/후 번역을 같은 실행에서 같은 설정으로(학습 전 = `disable_adapter`), AI Hub 평가 500 chrF·BLEU + 부트스트랩, `translate_inputs/*.jsonl` 전후 번역 (`build_notebook_02.py`로 생성)
- `inputs/pilot60_src.jsonl`: 파일럿 60문항 원문 (6축 전후 비교용)
- `results/pilot_test_2026-11-04.json`: 시험 학습 결과 (A100 80GB, 스텝당 0.81초, 3에폭 약 3.3시간)
- AI Hub 데이터와 분할 파일은 저장소에 올리지 않는다 (재배포 금지)
- 점검: 작은 무작위 T5로 노트북 코드 전체를 CPU에서 끝까지 실행해 봄 (transformers 5.22, peft 0.21). 10B 실측은 Colab에서
