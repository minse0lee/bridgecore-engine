# 브릿지코어 6축 번역 검증엔진

영→한 번역문을 6축(Semantic·Textual·Context·Deontic·Register·Cultural)으로 판정하는 LLM 판정 엔진.
결정 사항·일정은 `PROJECT_NOTES.md`에 있다. 새 대화나 Claude Code에서는 "PROJECT_NOTES.md와 README.md 먼저 읽고 이어서 해줘".

## 설치

```bash
pip install -r requirements.txt
cp .env.example .env        # ANTHROPIC_API_KEY 입력 (깃에 올라가지 않음)
```

## 순서

```bash
python -m bridgecore.cli build            # 엑셀 3개 → data/processed/ (문항·평가·임시정답·데이터점검·그림)
python -m bridgecore.cli split            # 개발 40 / 확인 20 분할 (이미 있으면 거부)
python -m bridgecore.cli baseline         # 사람 간 기준선 → reports/baseline_pilot.md
python -m bridgecore.cli rules            # 숫자 검사·한글 비율 → data/processed/rules_pilot.csv
python -m bridgecore.cli prompt P036      # 문항 하나의 실제 프롬프트 확인 (API 없음)

python -m bridgecore.cli run --split dev                    # 드라이런: 요청만 저장
python -m bridgecore.cli run --split dev --live             # 실제 호출 (기본 모델 claude-haiku-5-5)
python -m bridgecore.cli run --split dev --live --model claude-sonnet-5-5
python -m bridgecore.cli run --items P007,P036 --live --name try_numbers   # 몇 문항만

python -m bridgecore.cli score runs/<실행폴더> --split dev   # → reports/<실행>_dev.md + _cases.csv
python -m pytest -q
```

확인용 20문항은 `--split holdout --i-know-holdout`일 때만 실행된다 (마지막 1회).

## 폴더

| 경로 | 내용 |
|---|---|
| `bridgecore/loader.py` | 엑셀 평가·문항·운영 시트 로더, 입력 규칙 점검, 그림 추출 |
| `bridgecore/schema.py` | 엔진 출력 JSON 형식·검증·평가 시트 행 변환 |
| `bridgecore/gold.py` | 다수결 임시 정답 (채점 규칙 v1) |
| `bridgecore/scoring.py` | 축별 P/R/F1, 보류 2가지 처리, 신뢰구간, 사람 간 기준선 |
| `bridgecore/split.py` | 장르 × 드문 축 훼손 층화 분할 |
| `bridgecore/rules.py` | 숫자 일치 힌트, 목표언어 미생성 필터, 규범 표현 힌트 |
| `bridgecore/prompt.py` | 시스템 프롬프트(캐시) + 문항 메시지 + 강제 도구 호출 |
| `bridgecore/runner.py` | 드라이런/실행, 재시작 시 이어서, 토큰 기록 |
| `prompts/v0/system.md` | 프롬프트 v0 (가이드라인 2026-10-07판 요약 + EX01·EX02) |
| `prices.json` | 모델 단가 (공식 가격표 확인 후 직접 입력 — 비어 있으면 비용 계산 안 함) |
| `data/raw/` | 원본 엑셀·가이드라인 (깃 제외) |
| `data/processed/` | 빌드 결과 (깃 제외, `split_pilot.json`만 포함) |
| `runs/`, `reports/` | 실행 결과, 채점 리포트 |

## 채점 규칙 v1

- 이진 정답: 3명 중 2명 이상 훼손 → 훼손 (평가자 3명이라 동률 없음). 사람 판단보류는 비훼손
- 심각도 정답: 다수 일치 값, 없으면 낮은 쪽 중앙값 (minor·major → minor)
- 4범주 정답(보조): 2명 이상 같은 값, 3명 모두 다르면 불일치 → 분모 제외
- 엔진 판단보류: 보류=비훼손(주 지표)과 보류=훼손 두 가지 F1 + 보류 건수
- 형식 오류 문항: 6축 비훼손으로 채점하고 건수 따로 보고
- 드문 축: 재현율 95% 윌슨 구간과 "n건 중 k건" 함께 표기

## 엔진 버전 기록

실행마다 `runs/<이름>/meta.json`에 날짜·모델·프롬프트 버전·프롬프트 해시가 남는다.
버전을 고정할 때 이 해시를 PROJECT_NOTES 개발일지에 적는다.
