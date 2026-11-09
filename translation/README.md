# 번역엔진 (MADLAD-400 10B 영→한 LoRA)

- `prep_data.py`: AI Hub 한국어-영어 번역 말뭉치(사회과학)에서 영어 원문 교육 문장으로 학습·검증·평가 분할 저장
  - 학습 = Training 압축의 `source_language == en` · `domain == 교육`, 검증 1,000 / 평가 500 = Validation 압축에서 무작위 (시드 20261104)
  - 열: `en` 영어 원문, `ko` 사람 번역 (`mt`는 한→영 기계번역이라 안 씀)
  - Validation 압축 확인(2026-11-04): CSV 151,316행 중 영어 원문 9,713행(교육 9,653)
- `01_시험학습_MADLAD10B_LoRA.ipynb`: Colab A100에서 메모리·속도 측정용 짧은 학습 → 전체 학습 시간·컴퓨팅 단위 추정 (`build_notebook.py`로 생성)
- `02_본학습_MADLAD10B_LoRA.ipynb`: LoRA 본 학습(검증 손실로 최적 시점 저장·이어서 학습·조기 종료), 학습 전/후 번역을 같은 실행에서 같은 설정으로(학습 전 = `disable_adapter`), AI Hub 평가 500 chrF·BLEU + 부트스트랩, `translate_inputs/*.jsonl` 전후 번역 (`build_notebook_02.py`로 생성)
- `03_설정비교_본학습_v2.ipynb`: 02 보완판 — 모델 커밋 고정·pip freeze·결정적 연산, 검증 300문장 chrF로 최적 시점 선택, 한/영 글자 비율 0.2~1.0 밖 제외, BLEU ko-mecab, 설정 A~D 1에폭 비교(search) → 규칙(A보다 검증 chrF 0.5 초과일 때만 교체)으로 고른 설정 3에폭(final) (`build_notebook_03.py`)
- `inputs/pilot60_src.jsonl`: 파일럿 60문항 원문 (6축 전후 비교용)
- `results/pilot_test_2026-11-04.json`: 시험 학습 결과 (A100 80GB, 스텝당 0.81초, 3에폭 약 3.3시간)
- AI Hub 데이터와 분할 파일은 저장소에 올리지 않는다 (재배포 금지)
- 점검: 작은 무작위 T5로 노트북 코드 전체를 CPU에서 끝까지 실행해 봄 (transformers 5.22, peft 0.21). 10B 실측은 Colab에서
- `04_책번역_lora_v1.ipynb`: 학습 없이 lora_v1로 책 문장 번역(같은 실행에서 학습 전 번역도). 256문장마다 저장, 끊기면 이어서. 256토큰 넘는 문장은 자르지 않고 따로 번역 (`build_notebook_04.py`)
- `epub_tools.py`: EPUB → 문장(`extract`, DRM 검사·본문 추출 비율 기록) → 번역문으로 한국어 EPUB 재조립(`rebuild`, 문단 단위 교체, 각주 번호 링크만 유지) → 6축 검증용 무작위 표본(`sample`). 테스트 `tests/test_epub_tools.py`, Standard Ebooks 『The Time Machine』으로 추출·재조립 확인(1,781문장, 추출 비율 1.0, XML 엄격 파싱 통과)
