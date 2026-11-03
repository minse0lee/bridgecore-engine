"""Deontic 범주 대응 규칙 점검 (API 호출 없음).  python -m pytest -q tests/test_deontic_rule.py

가이드라인 7절 조동사 표와 5절 예시를 그대로 옮긴 대응 쌍.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bridgecore.deontic_rule import check, english_markers, korean_markers  # noqa: E402


def flags(src, tgt):
    return [(f["type"], f["src_cat"], f["tgt_cat"], f["strength"]) for f in check(src, tgt)["flags"]]


# (원문, 번역, 기대 플래그 목록)
CASES = [
    # 맞는 대응 → 플래그 없음
    ("Students must wear eye protection.", "학생들은 보안경을 착용해야 합니다.", []),
    ("You must not enter the lab alone.", "혼자 실험실에 들어가서는 안 됩니다.", []),
    ("You must not enter the lab alone.", "혼자 실험실에 들어가면 안 돼요.", []),
    ("You need not bring a calculator.", "계산기를 가져올 필요는 없습니다.", []),
    ("You do not have to attend.", "참석하지 않아도 됩니다.", []),
    ("Students may use a dictionary.", "학생들은 사전을 사용해도 됩니다.", []),
    ("Students may use a dictionary.", "학생들은 사전을 사용할 수 있습니다.", []),
    ("Applicants should submit the form early.", "지원자는 서류를 일찍 제출해야 합니다.", []),
    ("Applicants should submit the form early.", "지원자는 서류를 일찍 제출하는 것이 좋습니다.", []),
    ("The total may not exceed 500 words.", "전체 분량은 500단어를 초과할 수 없습니다.", []),
    ("Do not touch the blade.", "칼날을 만지지 마십시오.", []),
    ("Smoking is prohibited on campus.", "교내 흡연은 금지되어 있습니다.", []),
    ("Schools are not required to notify parents.", "학교는 학부모에게 알릴 필요가 없습니다.", []),
    ("Schools are not required to notify parents.", "학교가 학부모에게 알리는 것은 요구되지 않습니다.", []),
    # 범주 뒤집힘 → 플래그
    ('"I must not tell anyone," she whispered.', "아무에게도 말하지 않아도 돼.", [("flip", "PROH", "NONOBL", "strong")]),
    ("You need not bring a calculator.", "계산기를 가져와서는 안 됩니다.", [("flip", "NONOBL", "PROH", "strong")]),
    ("Students must wear eye protection.", "학생들은 보안경을 착용하는 것이 좋습니다.", [("flip", "OBL", "REC", "weak")]),
    ("Students are required to wear goggles.", "학생들은 보안경을 착용해도 됩니다.", [("flip", "OBL", "PERM", "strong")]),
    ("Students are permitted to leave early.", "학생들은 일찍 떠나야 합니다.", [("flip", "PERM", "OBL", "strong")]),
    ("Smoking is prohibited on campus.", "교내에서 흡연해도 됩니다.", [("flip", "PROH", "PERM", "strong")]),
    # 규범 표지가 사라짐
    ("State funding mandates that colleges create threat assessment teams.",
     "주 정부가 대학에 위협 평가 팀을 구성하도록 지원한다.", [("vanished", "OBL", None, "strong")]),
    ("Food must be refrigerated.", "음식은 냉장 보관됩니다.", []),  # must는 weak라 '사라짐'을 내지 않음 (LLM 몫)
]


@pytest.mark.parametrize("src,tgt,expected", CASES)
def test_cases(src, tgt, expected):
    assert flags(src, tgt) == expected


def test_markers_basic():
    cats = [m["cat"] for m in english_markers("You must not and need not, but may.")]
    assert cats == ["NONOBL", "PROH", "PERM"]
    assert [m["cat"] for m in korean_markers("하지 않아도 됩니다")] == ["NONOBL"]  # '도 된다'(허가)로 다시 잡히지 않음
    assert [m["cat"] for m in korean_markers("해야합니다")] == ["OBL"]           # 띄어쓰기 없어도 잡힘


def test_midsentence_do_not_is_not_prohibition():
    # 문장 중간의 do not은 명령형 금지가 아님
    assert all(m["cat"] != "PROH" for m in english_markers("Many children do not outgrow it."))
