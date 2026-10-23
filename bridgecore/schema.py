"""엔진 출력 형식(PROJECT_NOTES 3-4)과 검증.

한 문항 = JSON 객체 하나:
{
  "item_id": "P053",
  "evaluable": true,              # false면 생성붕괴 의심 → 6축 모두 판단보류
  "evaluable_note": null,         # evaluable=false일 때 관찰한 현상
  "errors": [{"id": "E001", "src_span": "...", "tgt_span": "...", "description": "..."}],
  "axes": {"Semantic": {"judgment": "훼손", "severity": "major",
                         "error_ids": ["E001"], "reason": "..."}, ... 6축}
}

validate()는 (치명 오류 목록, 경고 목록)을 돌려준다. 치명 오류가 있으면 채점에서 해당 문항을
'형식 오류'로 집계하고, 경고는 기록만 한다.
"""
from __future__ import annotations

import json
import re

from .constants import AXES, HOLD, JUDGMENTS, NA, SEVERITIES, VIOLATED

ERROR_ID_RE = re.compile(r"^E\d{3}$")

_JUDGMENT_ALIASES = {
    "na": NA, "n/a": NA, "n.a.": NA, "해당없음": NA, "해당 없음": NA,
    "보류": HOLD, "판단 보류": HOLD,
    "preserved": "보존", "violated": VIOLATED, "hold": HOLD,
}

JSON_SCHEMA = {
    "type": "object",
    "required": ["item_id", "evaluable", "evaluable_note", "errors", "axes"],
    "properties": {
        "item_id": {"type": "string"},
        "evaluable": {"type": "boolean"},
        "evaluable_note": {"type": ["string", "null"]},
        "errors": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "src_span", "tgt_span", "description"],
                "properties": {
                    "id": {"type": "string"},
                    "src_span": {"type": "string"},
                    "tgt_span": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
        "axes": {
            "type": "object",
            "required": AXES,
            "properties": {
                ax: {
                    "type": "object",
                    "required": ["judgment", "severity", "error_ids", "reason"],
                    "properties": {
                        "judgment": {"enum": JUDGMENTS},
                        "severity": {"enum": SEVERITIES + [None]},
                        "error_ids": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string"},
                    },
                }
                for ax in AXES
            },
        },
    },
}


def parse_json_output(text: str) -> dict:
    """모델 응답 텍스트에서 JSON 객체를 꺼낸다 (```json 코드블록 허용)."""
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
    raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(raw)


def normalize(out: dict) -> dict:
    """의미를 바꾸지 않는 표기 차이만 정리한다 (대소문자, N/A 표기 등)."""
    for ax in AXES:
        a = (out.get("axes") or {}).get(ax)
        if not isinstance(a, dict):
            continue
        j = a.get("judgment")
        if isinstance(j, str):
            j = j.strip()
            a["judgment"] = _JUDGMENT_ALIASES.get(j.lower(), j)
        s = a.get("severity")
        if isinstance(s, str):
            s = s.strip().lower()
            a["severity"] = s or None
        if a.get("error_ids") is None:
            a["error_ids"] = []
    return out


def validate(out: dict, item_id: str | None = None) -> tuple[list[str], list[str]]:
    errs, warns = [], []
    if not isinstance(out, dict):
        return ["출력이 JSON 객체가 아님"], warns
    if item_id and out.get("item_id") != item_id:
        errs.append(f"item_id 불일치: {out.get('item_id')} != {item_id}")
    if not isinstance(out.get("evaluable"), bool):
        errs.append("evaluable이 true/false가 아님")

    error_ids = []
    for e in out.get("errors") or []:
        eid = e.get("id") if isinstance(e, dict) else None
        if not (isinstance(eid, str) and ERROR_ID_RE.match(eid)):
            errs.append(f"오류 ID 형식 오류: {eid}")
            continue
        if eid in error_ids:
            errs.append(f"오류 ID 중복: {eid}")
        error_ids.append(eid)
        if not (e.get("src_span") or e.get("tgt_span")):
            warns.append(f"{eid}: 근거 구간이 비어 있음")
    expected = [f"E{i:03d}" for i in range(1, len(error_ids) + 1)]
    if error_ids and error_ids != expected:
        warns.append(f"오류 ID가 E001부터 차례대로가 아님: {error_ids}")

    axes = out.get("axes")
    if not isinstance(axes, dict):
        return errs + ["axes 없음"], warns
    used = set()
    for ax in AXES:
        a = axes.get(ax)
        if not isinstance(a, dict):
            errs.append(f"{ax}: 축 판정 없음")
            continue
        j, sev, ids = a.get("judgment"), a.get("severity"), a.get("error_ids") or []
        if j not in JUDGMENTS:
            errs.append(f"{ax}: 판정값 오류 {j!r}")
        if not (isinstance(a.get("reason"), str) and a["reason"].strip()):
            errs.append(f"{ax}: 근거(reason) 없음")  # 라벨 복사 방지: 모든 축 근거 필수
        if j == VIOLATED:
            if sev not in SEVERITIES:
                errs.append(f"{ax}: 훼손인데 심각도 없음/오류 {sev!r}")
            if not ids:
                errs.append(f"{ax}: 훼손인데 오류 ID 없음")
        else:
            if sev is not None:
                warns.append(f"{ax}: {j}인데 심각도 {sev!r} (무시)")
            if ids:
                warns.append(f"{ax}: {j}인데 오류 ID {ids} (무시)")
        for i in ids:
            if i not in error_ids:
                errs.append(f"{ax}: errors에 없는 오류 ID {i}")
            used.add(i)
    for i in error_ids:
        if i not in used:
            warns.append(f"{i}: 어느 축에도 연결되지 않은 오류")
    if out.get("evaluable") is False:
        if any((axes.get(ax) or {}).get("judgment") != HOLD for ax in AXES):
            errs.append("evaluable=false인데 판단보류가 아닌 축이 있음")
        if not out.get("evaluable_note"):
            warns.append("evaluable=false인데 evaluable_note 없음")
    return errs, warns


def to_sheet_rows(out: dict) -> list[dict]:
    """엔진 출력을 평가 시트 행(문항ID·평가축·판정·심각도·오류ID·근거·이유)으로 변환."""
    errors = {e["id"]: e for e in out.get("errors") or [] if isinstance(e, dict) and "id" in e}
    rows = []
    for ax in AXES:
        a = (out.get("axes") or {}).get(ax) or {}
        ids = a.get("error_ids") or []
        is_v = a.get("judgment") == VIOLATED

        def join(field):
            parts = [errors[i].get(field) or "" for i in ids if i in errors]
            if len(parts) <= 1:
                return parts[0] if parts else None
            return " / ".join(f"{i}: {p}" for i, p in zip(ids, parts))

        rows.append({
            "item_id": out.get("item_id"), "axis": ax, "judgment": a.get("judgment"),
            "severity": a.get("severity") if is_v else None,
            "error_ids": ", ".join(ids) if is_v else None,
            "src_span": join("src_span") if is_v else None,
            "tgt_span": join("tgt_span") if is_v else None,
            "reason": a.get("reason"),
        })
    return rows
