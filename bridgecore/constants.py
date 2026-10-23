"""공통 상수: 축, 판정값, 심각도, 시트 열 위치.

가이드라인 2026-10-07판 기준. 축 순서는 가이드라인 1절 ④의 판정 순서와 같다.
"""

AXES = ["Semantic", "Textual", "Context", "Deontic", "Register", "Cultural"]

PRESERVED = "보존"
VIOLATED = "훼손"
NA = "N/A"
HOLD = "판단보류"
JUDGMENTS = [PRESERVED, VIOLATED, NA, HOLD]

SEVERITIES = ["minor", "major", "critical"]
SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}

RATER_IDS = ["R001", "R002", "R003"]

# 엑셀 시트 구조 (1-indexed 행 번호)
HEADER_ROW = 8          # 평가·문항 시트 공통 머리글 행
FIRST_DATA_ROW = 9

# 평가 시트 열 (0-indexed, 가이드라인 3절 A~L)
EVAL_COLS = {
    "item_id": 0, "axis": 1, "judgment": 2, "severity": 3, "error_ids": 4,
    "src_span": 5, "tgt_span": 6, "reason": 7, "final_judgment": 8,
    "final_severity": 9, "change_reason": 10, "opinion": 11,
}

# 문항 시트 열 (0-indexed)
ITEM_COLS = {
    "item_id": 0, "doc_info": 1, "brief": 2, "src_paragraph": 3,
    "tgt_paragraph": 4, "src": 5, "tgt": 6,
}

# 운영 시트 열 (0-indexed)
OPS_COLS = {"rater_id": 0, "batch": 1, "start": 2, "end": 3, "pause_min": 4}
