# 번역엔진 (MADLAD-400 10B 영→한 LoRA)

- `prep_data.py`: AI Hub 한국어-영어 번역 말뭉치(사회과학)에서 영어 원문 교육 문장으로 학습·검증·평가 분할 저장
  - 학습 = Training 압축의 `source_language == en` · `domain == 교육`, 검증 1,000 / 평가 500 = Validation 압축에서 무작위 (시드 20261104)
  - 열: `en` 영어 원문, `ko` 사람 번역 (`mt`는 한→영 기계번역이라 안 씀)
  - Validation 압축 확인(2026-11-04): CSV 151,316행 중 영어 원문 9,713행(교육 9,653)
- `01_시험학습_MADLAD10B_LoRA.ipynb`: Colab A100에서 메모리·속도 측정용 짧은 학습 → 전체 학습 시간·컴퓨팅 단위 추정 (`build_notebook.py`로 생성)
- AI Hub 데이터와 분할 파일은 저장소에 올리지 않는다 (재배포 금지)
- 점검: 작은 무작위 T5로 노트북 코드 전체를 CPU에서 끝까지 실행해 봄 (transformers 5.22, peft 0.21). 10B 실측은 Colab에서
