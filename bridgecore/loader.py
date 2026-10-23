"""평가자 엑셀(평가·문항·운영 시트) 로더와 데이터 점검.

- load_items(xlsx): 문항 시트 60문항 + 문서정보 파싱 + 그림 자료 위치
- load_ratings(xlsx): 평가 시트 360행 (평가자 ID는 시트 상단 B3에서 읽음)
- check_ratings(rows): 가이드라인 3절 입력 규칙 위반 목록
- extract_figures(xlsx, out_dir): 문항 시트에 붙은 그림을 파일로 저장
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import openpyxl

from .constants import (AXES, EVAL_COLS, FIRST_DATA_ROW, ITEM_COLS, JUDGMENTS,
                        OPS_COLS, SEVERITIES, VIOLATED)

ITEM_ID_RE = re.compile(r"^P\d{3}$")
ERROR_ID_RE = re.compile(r"E\d{3}")
FIGURE_LABEL_RE = re.compile(r"^(P\d{3})\s*·\s*평가용 그림 자료")


def _clean(v):
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        return v or None
    return v


def parse_doc_info(text: str | None) -> dict:
    """'키: 값' 줄로 된 문서정보를 dict로. 키가 없는 줄은 앞 키에 이어 붙인다."""
    out: dict[str, str] = {}
    last = None
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([^:：]{1,12})[:：]\s*(.*)$", line)
        if m:
            last = m.group(1).strip()
            out[last] = m.group(2).strip()
        elif last:
            out[last] += " " + line
    return out


def load_items(xlsx: str | Path) -> list[dict]:
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    ws = wb["문항"]
    items, figure_labels = [], {}
    for row_idx, row in enumerate(ws.iter_rows(min_row=FIRST_DATA_ROW, values_only=True),
                                  FIRST_DATA_ROW):
        first = _clean(row[0])
        if isinstance(first, str) and ITEM_ID_RE.match(first):
            rec = {k: _clean(row[i]) for k, i in ITEM_COLS.items()}
            info = parse_doc_info(rec["doc_info"])
            rec["genre"] = info.get("장르")
            rec["title"] = info.get("자료명")
            rec["source_url"] = info.get("출처")
            rec["has_figure"] = "그림 위치" in info
            rec["figure"] = None
            items.append(rec)
        elif isinstance(first, str) and FIGURE_LABEL_RE.match(first):
            figure_labels[row_idx] = FIGURE_LABEL_RE.match(first).group(1)
    return items


def extract_figures(xlsx: str | Path, out_dir: str | Path) -> dict[str, str]:
    """문항 시트 그림을 '<문항ID>.<확장자>'로 저장. {문항ID: 경로} 반환.

    그림은 'Pxxx · 평가용 그림 자료' 라벨 바로 아래 행에 고정되어 있다.
    라벨 행 → 그 아래에서 가장 가까운 그림 앵커를 짝짓는다.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    ws = wb["문항"]
    labels = {}
    for row_idx, row in enumerate(ws.iter_rows(min_row=1, values_only=True), 1):
        v = _clean(row[0])
        if isinstance(v, str) and FIGURE_LABEL_RE.match(v):
            labels[row_idx] = FIGURE_LABEL_RE.match(v).group(1)

    with zipfile.ZipFile(xlsx) as z:
        names = z.namelist()
        drawing = next((n for n in names if re.match(r"xl/drawings/drawing\d+\.xml$", n)), None)
        if not drawing:
            return {}
        xml = z.read(drawing).decode("utf-8")
        rels_xml = z.read(drawing.replace("drawings/", "drawings/_rels/") + ".rels").decode("utf-8")
        rels = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="\.\./media/([^"]+)"', rels_xml))
        rels.update({k: v for v, k in re.findall(r'Target="\.\./media/([^"]+)"[^>]*Id="(rId\d+)"', rels_xml)})
        anchors = []  # (1-indexed 앵커 행, 미디어 파일명)
        for block in re.findall(r"<xdr:(?:oneCellAnchor|twoCellAnchor)[\s\S]*?</xdr:(?:oneCellAnchor|twoCellAnchor)>", xml):
            r = re.search(r"<xdr:from>[\s\S]*?<xdr:row>(\d+)</xdr:row>", block)
            e = re.search(r'r:embed="(rId\d+)"', block)
            if r and e and e.group(1) in rels:
                anchors.append((int(r.group(1)) + 1, rels[e.group(1)]))

        result = {}
        for label_row, item_id in sorted(labels.items()):
            below = [a for a in anchors if a[0] >= label_row]
            if not below:
                continue
            _, media = min(below, key=lambda a: a[0])
            ext = Path(media).suffix
            path = out_dir / f"{item_id}{ext}"
            path.write_bytes(z.read(f"xl/media/{media}"))
            result[item_id] = str(path)
        return result


def load_ratings(xlsx: str | Path) -> list[dict]:
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    ws = wb["평가"]
    rater_id = _clean(ws["B3"].value)
    rows = []
    for row_idx, row in enumerate(ws.iter_rows(min_row=FIRST_DATA_ROW, values_only=True),
                                  FIRST_DATA_ROW):
        item_id = _clean(row[0])
        if not (isinstance(item_id, str) and ITEM_ID_RE.match(item_id)):
            continue
        rec = {k: _clean(row[i]) for k, i in EVAL_COLS.items()}
        rec["rater_id"] = rater_id
        rec["sheet_row"] = row_idx
        rec["error_id_list"] = ERROR_ID_RE.findall(rec["error_ids"] or "")
        rows.append(rec)
    return rows


def load_ops(xlsx: str | Path) -> list[dict]:
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    ws = wb["운영"]
    out = []
    for row in ws.iter_rows(min_row=FIRST_DATA_ROW, values_only=True):
        batch = _clean(row[OPS_COLS["batch"]])
        if isinstance(batch, str) and re.match(r"^B\d{2}$", batch) and _clean(row[0]):
            out.append({k: _clean(row[i]) for k, i in OPS_COLS.items()})
    return out


def check_ratings(rows: list[dict], expected_items: list[str]) -> list[dict]:
    """가이드라인 3절·10쪽 '제출 전 확인' 규칙 위반을 찾는다. 수정은 하지 않는다."""
    issues = []

    def add(r, kind, detail=""):
        issues.append({"rater_id": r.get("rater_id"), "item_id": r.get("item_id"),
                       "axis": r.get("axis"), "sheet_row": r.get("sheet_row"),
                       "kind": kind, "detail": detail})

    seen = set()
    for r in rows:
        key = (r["rater_id"], r["item_id"], r["axis"])
        if key in seen:
            add(r, "중복 행")
        seen.add(key)
        if r["axis"] not in AXES:
            add(r, "알 수 없는 축", str(r["axis"]))
        if r["judgment"] not in JUDGMENTS:
            add(r, "판정값 누락/오류", str(r["judgment"]))
        if r["judgment"] == VIOLATED:
            if r["severity"] not in SEVERITIES:
                add(r, "훼손인데 심각도 없음", str(r["severity"]))
            if not r["error_id_list"]:
                add(r, "훼손인데 오류ID 없음", str(r["error_ids"]))
        else:
            if r["severity"]:
                add(r, "훼손이 아닌데 심각도 있음", str(r["severity"]))
            if r["error_id_list"]:
                add(r, "훼손이 아닌데 오류ID 있음", str(r["error_ids"]))
        if r["final_judgment"] or r["final_severity"]:
            add(r, "최종판정 입력됨(합의 후 상태)")

    by_rater: dict[str, set] = {}
    for r in rows:
        by_rater.setdefault(r["rater_id"], set()).add((r["item_id"], r["axis"]))
    for rater, keys in by_rater.items():
        for it in expected_items:
            for ax in AXES:
                if (it, ax) not in keys:
                    issues.append({"rater_id": rater, "item_id": it, "axis": ax,
                                   "sheet_row": None, "kind": "행 누락", "detail": ""})
    return issues
