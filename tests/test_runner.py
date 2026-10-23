"""실행기 점검 — 가짜 API 클라이언트로 동기·Batch·2단계·비교를 끝까지 돌린다 (네트워크 없음)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bridgecore import gold as G  # noqa: E402
from bridgecore import report as R  # noqa: E402
from bridgecore import runner as RN  # noqa: E402
from bridgecore.compare import compare_runs  # noqa: E402
from bridgecore.constants import AXES, VIOLATED  # noqa: E402
from bridgecore.io_utils import read_jsonl  # noqa: E402

PROC = ROOT / "data" / "processed"
ITEMS = {i["item_id"]: i for i in read_jsonl(PROC / "items.jsonl")}
RATINGS = read_jsonl(PROC / "ratings.jsonl")
GOLD = G.build_gold(RATINGS)
IDS = ["P002", "P007", "P010", "P036", "P053", "P060"]


def _judge(item_id: str, policy: str) -> dict:
    """가짜 판정: majority=다수결 정답, any=1명이라도 훼손이면 훼손(넉넉한 선별 흉내)"""
    axes, errors = {}, []
    for ax in AXES:
        g = GOLD[(item_id, ax)]
        v = g["violated"] if policy == "majority" else g["n_violated"] >= 1
        axes[ax] = {"judgment": VIOLATED if v else "보존", "severity": "major" if v else None,
                    "error_ids": ["E001"] if v else [], "reason": "가짜 근거 문장입니다"}
        if v and not errors:
            errors = [{"id": "E001", "src_span": "s", "tgt_span": "t", "description": "가짜 오류 설명입니다"}]
    return {"item_id": item_id, "evaluable": True, "evaluable_note": None, "errors": errors, "axes": axes}


def _message(item_id: str, model: str, policy: str) -> dict:
    return {"model": model, "stop_reason": "tool_use",
            "content": [{"type": "thinking", "thinking": "", "signature": "x"},  # 사고 블록은 건너뛰어야 함
                        {"type": "text", "text": json.dumps(_judge(item_id, policy), ensure_ascii=False)}],
            "usage": {"input_tokens": 1500, "output_tokens": 900,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 5000}}


def _item_id_of(req: dict) -> str:
    text = req["messages"][0]["content"][-1]["text"]
    return text.split("[문항 ID] ")[1].split("\n")[0].strip()


class FakeClient:
    def __init__(self, policy="majority", fail_once=()):
        self.policy, self.fail_once, self.calls = policy, set(fail_once), []
        self.messages = self
        self.batches = _FakeBatches(self)

    def create(self, **req):
        iid = _item_id_of(req)
        self.calls.append(iid)
        if iid in self.fail_once:
            self.fail_once.discard(iid)
            raise RuntimeError("일시 오류")
        return _message(iid, req["model"], self.policy)


class _FakeBatches:
    def __init__(self, parent):
        self.parent, self.store, self.ready = parent, {}, False

    def create(self, requests):
        bid = f"msgbatch_{len(self.store)}"
        self.store[bid] = requests
        return {"id": bid}

    def retrieve(self, bid):
        return {"processing_status": "ended" if self.ready else "in_progress",
                "request_counts": {"processing": 0 if self.ready else len(self.store[bid]),
                                   "succeeded": len(self.store[bid]) if self.ready else 0,
                                   "errored": 0, "canceled": 0, "expired": 0}}

    def results(self, bid):
        for r in self.store[bid]:
            iid = _item_id_of(r["params"])
            if iid == "P060":  # 한 문항은 실패 처리
                yield {"custom_id": r["custom_id"], "result": {"type": "errored", "error": {"type": "overloaded"}}}
            else:
                yield {"custom_id": r["custom_id"],
                       "result": {"type": "succeeded", "message": _message(iid, r["params"]["model"], self.parent.policy)}}


def _items():
    return [ITEMS[i] for i in IDS]


def test_sync_run_and_resume(tmp_path):
    c = FakeClient(fail_once={"P010"})
    out = RN.run(_items(), "claude-haiku-5-5", "v0", "t1", True, tmp_path, client=c, retries=0)
    recs = RN.latest_records(out)
    assert not RN._ok(recs["P010"]) and RN._ok(recs["P002"])
    RN.run(_items(), "claude-haiku-5-5", "v0", "t1", True, tmp_path, client=c)  # 재실행
    assert c.calls.count("P010") == 2 and c.calls.count("P002") == 1  # 실패분만 다시 부름
    assert all(RN._ok(r) for r in RN.latest_records(out).values())
    assert (out / "engine_sheet.csv").exists()


def test_auto_retry_and_concurrency(tmp_path):
    c = FakeClient(fail_once={"P010"})
    out = RN.run(_items(), "claude-haiku-5-5", "v0", "t3", True, tmp_path, client=c, workers=4)
    assert c.calls.count("P010") == 2  # 같은 실행 안에서 자동 재시도
    assert all(RN._ok(r) for r in RN.latest_records(out).values())


def test_placeholder_reason_is_format_error():
    from bridgecore.schema import validate
    out = _judge("P002", "majority")
    out["axes"]["Textual"]["reason"] = "placeholder"
    assert any("자리표시" in e for e in validate(out, "P002")[0])
    out["axes"]["Textual"]["reason"] = "x"
    assert any("자리표시" in e for e in validate(out, "P002")[0])


def test_rerun_with_other_prompt_refused(tmp_path):
    RN.run(_items()[:1], "claude-haiku-5-5", "v0", "t2", True, tmp_path, client=FakeClient())
    with pytest.raises(SystemExit):
        RN.run(_items()[:1], "claude-haiku-5-5", "v0", "t2", True, tmp_path, screen=True, client=FakeClient())


def test_batch_submit_collect_and_retry(tmp_path):
    c = FakeClient()
    out = RN.submit_batch(_items(), "claude-haiku-5-5", "v0", "b1", True, tmp_path, client=c)
    assert RN.collect_batch(out, client=c) is False  # 아직 처리 중
    c.batches.ready = True
    assert RN.collect_batch(out, client=c) is True
    recs = RN.latest_records(out)
    assert recs["P002"]["via_batch"] and RN._ok(recs["P002"])
    assert not RN._ok(recs["P060"])
    # 실패분만 다시 제출
    RN.submit_batch(_items(), "claude-haiku-5-5", "v0", "b1", True, tmp_path, client=c)
    info = json.loads((out / "batch.json").read_text(encoding="utf-8"))
    assert info["batches"][-1]["n"] == 1


def test_cascade_merge_and_compare(tmp_path):
    screen = RN.run(_items(), "claude-haiku-5-5", "v0", "s", True, tmp_path, screen=True,
                    client=FakeClient(policy="any"))
    esc = {r["item_id"]: r["escalate"] for r in RN.escalation_list(screen)}
    to2 = [ITEMS[i] for i, e in esc.items() if e]
    assert 0 < len(to2) < len(IDS)
    stage2 = RN.run(to2, "claude-sonnet-5-5", "v0", "s2", True, tmp_path, client=FakeClient("majority"))
    merged = RN.merge_cascade(screen, stage2, tmp_path / "s__cascade")
    recs = RN.latest_records(merged)
    for iid, e in esc.items():
        assert recs[iid]["stage"] == (2 if e else 1)
    # 기준 실행(정답과 같은 가짜 Sonnet 전체)과 비교: 'any' 선별은 다수결 훼손을 놓치지 않는다
    ref = RN.run(_items(), "claude-sonnet-5-5", "v0", "ref", True, tmp_path, client=FakeClient("majority"))
    md = compare_runs(screen, ref, GOLD, tmp_path / "reports").read_text(encoding="utf-8")
    assert "| ALL |" in md and "(1.00)" in md


def test_cascade_cost_counts_both_stages(tmp_path, monkeypatch):
    screen = RN.run(_items(), "claude-haiku-5-5", "v0", "c1", True, tmp_path, screen=True,
                    client=FakeClient(policy="any"))
    to2 = [ITEMS[r["item_id"]] for r in RN.escalation_list(screen) if r["escalate"]]
    stage2 = RN.run(to2, "claude-sonnet-5-5", "v0", "c2", True, tmp_path, client=FakeClient())
    merged = RN.merge_cascade(screen, stage2, tmp_path / "c__cascade")
    (tmp_path / "prices.json").write_text(json.dumps({
        "batch_multiplier": 0.5,
        "claude-haiku-5-5": {"input": 1, "output": 1, "cache_write": 1, "cache_read": 1},
        "claude-sonnet-5-5": {"input": 10, "output": 10, "cache_write": 10, "cache_read": 10}}))
    monkeypatch.setattr(R, "ROOT", tmp_path)
    meta = json.loads((merged / "meta.json").read_text(encoding="utf-8"))
    cost = R._cost(R._calls_for_run(merged, meta, RN.latest_records(merged), set(IDS)), len(IDS))
    per_call = (1500 + 900 + 5000) / 1e6
    expect = len(IDS) * per_call * 1 + len(to2) * per_call * 10  # 1차는 전 문항, 2차는 넘긴 문항만
    assert cost["by_model"]["claude-haiku-5-5"]["calls"] == len(IDS)
    assert abs(cost["usd"] - expect) < 1e-12


def test_score_run_on_cascade(tmp_path):
    screen = RN.run(_items(), "claude-haiku-5-5", "v0", "d1", True, tmp_path, screen=True,
                    client=FakeClient(policy="any"))
    to2 = [ITEMS[r["item_id"]] for r in RN.escalation_list(screen) if r["escalate"]]
    stage2 = RN.run(to2, "claude-sonnet-5-5", "v0", "d2", True, tmp_path, client=FakeClient())
    merged = RN.merge_cascade(screen, stage2, tmp_path / "d__cascade")
    path = R.score_run(merged, list(ITEMS.values()), RATINGS, tmp_path / "reports")
    md = path.read_text(encoding="utf-8")
    assert "2단계: 재판정" in md
    assert "| ALL |" in md


def test_request_shape_for_claude_5_5():
    """Sonnet 5.5: 도구 강제·temperature 금지 → 구조화 출력. 사고 끄기 값은 모델별로 다름"""
    from bridgecore.prompt import build_request
    from bridgecore.schema import JSON_SCHEMA
    r = build_request(ITEMS["P036"], "claude-sonnet-5-5", thinking=False, effort="low")
    assert "tool_choice" not in r and "tools" not in r and "temperature" not in r
    assert r["output_config"] == {"format": {"type": "json_schema", "schema": JSON_SCHEMA}, "effort": "low"}
    assert r["thinking"] == {"type": "between_tools"}
    assert build_request(ITEMS["P036"], "claude-haiku-5-5", thinking=False)["thinking"] == {"type": "disabled"}
    assert "thinking" not in build_request(ITEMS["P036"], "claude-haiku-5-5")


def test_schema_follows_structured_output_rules():
    from bridgecore.schema import JSON_SCHEMA

    def walk(node, unions=[0]):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node["required"]) == set(node["properties"])
            if isinstance(node.get("type"), list) or "anyOf" in node:
                unions[0] += 1
            for v in node.values():
                walk(v, unions)
        elif isinstance(node, list):
            for v in node:
                walk(v, unions)
        return unions[0]
    assert walk(JSON_SCHEMA) <= 16


def test_refusal_and_truncation_are_errors():
    msg = _message("P002", "claude-haiku-5-5", "majority")
    rec = RN.process_response("P002", {**msg, "stop_reason": "refusal"}, "claude-haiku-5-5")
    assert not RN._ok(rec)
    bad = {**msg, "content": [{"type": "text", "text": '{"item_id": "P002", "evaluable": tr'}],
           "stop_reason": "max_tokens"}
    rec = RN.process_response("P002", bad, "claude-haiku-5-5")
    assert any("JSON" in e for e in rec["errors"]) and any("max_tokens" in e for e in rec["errors"])


def test_paragraph_only_item():
    from bridgecore.prompt import PARAGRAPH_ONLY_NOTE, user_text
    it = {**ITEMS["P036"], "tgt": ""}
    text = user_text(it)
    assert PARAGRAPH_ONLY_NOTE in text and "(빈 출력)" not in text
