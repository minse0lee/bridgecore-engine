"""파이프라인 점검.  python -m pytest -q  (먼저 python -m bridgecore.cli build 실행)"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bridgecore import gold as G  # noqa: E402
from bridgecore.constants import AXES, HOLD, VIOLATED  # noqa: E402
from bridgecore.io_utils import read_jsonl  # noqa: E402
from bridgecore.rules import english_numbers, korean_numbers, number_check  # noqa: E402
from bridgecore.schema import to_sheet_rows, validate  # noqa: E402
from bridgecore.scoring import human_baseline, predictions_from_outputs, score  # noqa: E402

PROC = ROOT / "data" / "processed"


def _data():
    return read_jsonl(PROC / "items.jsonl"), read_jsonl(PROC / "ratings.jsonl")


def _output(item_id, judgments: dict, severity="major"):
    has_v = any(j == VIOLATED for j in judgments.values())
    return {
        "item_id": item_id, "evaluable": True, "evaluable_note": None,
        "errors": [{"id": "E001", "src_span": "x", "tgt_span": "y", "description": "가짜 오류 설명입니다"}] if has_v else [],
        "axes": {ax: {"judgment": judgments[ax],
                      "severity": severity if judgments[ax] == VIOLATED else None,
                      "error_ids": ["E001"] if judgments[ax] == VIOLATED else [],
                      "reason": "근거 문장입니다 (테스트)"} for ax in AXES},
    }


# ---------- 데이터 ----------
def test_loaded_shape():
    items, ratings = _data()
    assert len(items) == 60
    assert len(ratings) == 1080
    assert {r["rater_id"] for r in ratings} == {"R001", "R002", "R003"}
    assert sum(1 for i in items if i["figure"]) == 3


def test_binary_majority_always_decided():
    _, ratings = _data()
    gold = G.build_gold(ratings)
    assert len(gold) == 360
    assert all(v["n_raters"] == 3 for v in gold.values())  # 3명 → 이진 다수결에 동률 없음


# ---------- 정답 규칙 ----------
def test_severity_vote():
    assert G._severity_vote(["minor", "major"]) == "minor"
    assert G._severity_vote(["minor", "major", "critical"]) == "major"
    assert G._severity_vote(["major", "major", "minor"]) == "major"
    assert G._severity_vote(["major", None]) == "major"


def test_label4_undecided():
    rs = [{"item_id": "P1", "axis": "Semantic", "rater_id": r, "judgment": j, "severity": s}
          for r, j, s in [("A", "보존", None), ("B", "훼손", "minor"), ("C", "N/A", None)]]
    g = G.build_gold(rs)[("P1", "Semantic")]
    assert g["label4"] is None and g["violated"] is False


# ---------- 채점 ----------
def test_oracle_scores_perfect():
    """정답을 그대로 낸 가짜 엔진 → 모든 축 F1 1.0, 4범주 일치 1.0"""
    items, ratings = _data()
    gold = G.build_gold(ratings)
    outs = {}
    for it in items:
        js = {ax: gold[(it["item_id"], ax)]["label4"] or "보존" for ax in AXES}
        outs[it["item_id"]] = _output(it["item_id"], js)
    preds, checks = predictions_from_outputs(outs)
    assert not any(c["errors"] for c in checks.values())
    res = score(gold, preds)
    for ax in AXES:
        a = res["per_axis"][ax]
        if a["gold_violated"]:
            assert a["strict"]["f1"] == 1.0, ax
        assert a["label4_acc"] == 1.0, ax


def test_hold_counts_both_ways():
    """전부 판단보류 → strict 재현율 0, lenient 재현율 1"""
    items, ratings = _data()
    gold = G.build_gold(ratings)
    outs = {it["item_id"]: _output(it["item_id"], {ax: HOLD for ax in AXES}) for it in items}
    res = score(gold, predictions_from_outputs(outs)[0])["per_axis"]["ALL"]
    assert res["strict"]["recall"] == 0.0
    assert res["lenient"]["recall"] == 1.0
    assert res["hold"] == 360


def test_format_error_counts_as_negative():
    items, ratings = _data()
    gold = G.build_gold(ratings)
    bad = _output("P001", {ax: VIOLATED for ax in AXES})
    bad["axes"]["Semantic"]["reason"] = ""
    preds, checks = predictions_from_outputs({"P001": bad})
    assert checks["P001"]["errors"]
    assert all(preds[("P001", ax)]["judgment"] is None for ax in AXES)


def test_rater_as_engine_matches_baseline_direction():
    """평가자 한 명을 엔진처럼 채점 → 형식 통과, F1이 0~1 사이"""
    items, ratings = _data()
    gold = G.build_gold(ratings)
    by = {}
    for r in ratings:
        if r["rater_id"] == "R002":
            by.setdefault(r["item_id"], {})[r["axis"]] = r["judgment"]
    outs = {i: _output(i, js) for i, js in by.items()}
    res = score(gold, predictions_from_outputs(outs)[0])["per_axis"]["ALL"]
    assert 0 < res["strict"]["f1"] < 1
    bl = human_baseline(ratings)
    assert 0 < bl["ALL"]["mean_pairwise_f1"] < 1


# ---------- 출력 형식 ----------
def test_validate_catches_errors():
    out = _output("P001", {ax: "보존" for ax in AXES})
    assert validate(out, "P001") == ([], [])
    out["axes"]["Deontic"] = {"judgment": VIOLATED, "severity": None, "error_ids": ["E009"], "reason": "축 근거 문장 (테스트)"}
    errs, _ = validate(out, "P001")
    assert any("심각도" in e for e in errs) and any("E009" in e for e in errs)


def test_unevaluable_requires_all_hold():
    out = _output("P001", {ax: HOLD for ax in AXES})
    out["evaluable"] = False
    out["evaluable_note"] = "생성붕괴 검토 필요: 목표언어 미생성"
    assert validate(out, "P001")[0] == []
    out["axes"]["Semantic"]["judgment"] = "보존"
    assert validate(out, "P001")[0]


def test_sheet_rows_multi_error():
    out = _output("P001", {ax: "보존" for ax in AXES})
    out["errors"] = [{"id": "E001", "src_span": "a", "tgt_span": "가", "description": "가짜 오류 설명입니다"},
                     {"id": "E002", "src_span": "b", "tgt_span": "나", "description": "가짜 오류 설명입니다"}]
    out["axes"]["Semantic"] = {"judgment": VIOLATED, "severity": "major",
                               "error_ids": ["E001", "E002"], "reason": "축 근거 문장 (테스트)"}
    row = to_sheet_rows(out)[0]
    assert row["error_ids"] == "E001, E002"
    assert row["src_span"] == "E001: a / E002: b"  # 가이드라인 9절 형식


# ---------- 규칙 ----------
def test_numbers():
    assert english_numbers("fifty and eighty calories") == [50, 80]
    assert english_numbers("a hundred and twenty-five men") == [125]
    assert english_numbers("for the 2012–13 academic year") == [2012, 2013]
    assert korean_numbers("삼백만 달러") == [3_000_000]
    assert korean_numbers("300만 달러") == [3_000_000]
    assert korean_numbers("버그리를 세척했다") == []
    assert korean_numbers("세 명") == [3]
    assert number_check("the past fortnight", "지난 15일 동안")["mismatch"]
    assert not number_check("thirty days", "30일")["mismatch"]


def test_pilot_number_flags():
    """파일럿에서 숫자 불일치는 실제 오류 2건(P007 미터 첨가, P036 fortnight→15일)만 남아야 함"""
    from bridgecore.rules import run_rules
    items, _ = _data()
    flagged = sorted(i["item_id"] for i in items if run_rules(i)["number_check"]["mismatch"])
    assert flagged == ["P007", "P036"]


def test_deontic_longest_match():
    from bridgecore.rules import deontic_keywords
    assert deontic_keywords("You must not open the door.") == ["must not"]
    assert deontic_keywords("It may not exceed $30.") == ["may not"]
