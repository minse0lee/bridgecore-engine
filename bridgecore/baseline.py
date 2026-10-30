"""가이드라인 효과 비교용 기준선: 6축·가이드라인 없이 "번역 오류를 찾아라"만 시키는 일반 LLM 판정.

엔진과 같은 모델·같은 문항 정보(문서정보·평가조건·문단 문맥)를 주고, 바꾸는 것은 지침뿐이다.
자동 점검 힌트(숫자 검사)도 엔진의 일부이므로 넣지 않는다.
출력: 오류 목록 [{src_span, tgt_span, description, severity}] → 문장 단위로 "오류 있음/없음"만 비교할 수 있다.

실행 폴더 runs/<이름>/ 에 meta.json(type=generic)과 outputs.jsonl을 남긴다 (엔진 실행과 같은 위치·형식).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from pathlib import Path

from .io_utils import write_json
from .runner import USAGE_KEYS, _dump, _get, get_client, latest_records
from .schema import parse_json_output

SEVERITIES = ["minor", "major", "critical"]

GENERIC_SYSTEM = """당신은 영→한 번역 검수자입니다. 사용자가 주는 평가대상 원문과 번역문을 비교해 번역 오류를 모두 찾고, 지정된 JSON 형식으로만 답합니다.
- 오류마다 원문 구간(src_span), 번역문 구간(tgt_span), 무엇이 잘못됐는지(description), 심각도(severity: minor, major, critical 중 하나)를 적습니다.
- 구간은 원문·번역문 표현을 그대로 옮깁니다.
- 오류가 없으면 errors를 빈 배열로 둡니다.
- 나머지 정보(문서정보, 평가조건, 문단)는 문맥으로 씁니다."""

GENERIC_USER = """[문서정보]
{doc_info}

[평가조건]
{brief}

[원문 문단] (문맥)
{src_paragraph}

[문단 전체 번역] (문맥)
{tgt_paragraph}

[평가대상 원문]
{src}

[평가대상 번역문]
{tgt}"""

GENERIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["errors"],
    "properties": {
        "errors": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["src_span", "tgt_span", "description", "severity"],
                "properties": {
                    "src_span": {"type": "string"},
                    "tgt_span": {"type": "string"},
                    "description": {"type": "string"},
                    "severity": {"type": "string", "enum": SEVERITIES},
                },
            },
        }
    },
}


def fingerprint() -> str:
    h = hashlib.sha256(GENERIC_SYSTEM.encode("utf-8"))
    h.update(GENERIC_USER.encode("utf-8"))
    h.update(json.dumps(GENERIC_SCHEMA, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:12]


def build_request(item: dict, model: str, max_tokens: int = 16000, effort: str | None = "medium") -> dict:
    text = GENERIC_USER.format(
        doc_info=item.get("doc_info") or "(없음)", brief=item.get("brief") or "(없음)",
        src_paragraph=item.get("src_paragraph") or "(없음)",
        tgt_paragraph=item.get("tgt_paragraph") or "(없음)",
        src=item.get("src") or "", tgt=item.get("tgt") or "(문단 전체 번역에서 대응 구간을 찾으세요)")
    content = []
    if item.get("figure"):  # 엔진과 같은 문맥: 원문 그림이 있으면 함께 보낸다
        import base64, mimetypes
        from .prompt import ROOT as _ROOT
        path = _ROOT / item["figure"]
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": mimetypes.guess_type(str(path))[0] or "image/png",
            "data": base64.b64encode(path.read_bytes()).decode("ascii")}})
        text += "\n\n[그림 자료] 이 문항의 원문 그림을 함께 첨부했습니다. 문맥으로만 사용하세요."
    content.append({"type": "text", "text": text})
    oc: dict = {"format": {"type": "json_schema", "schema": GENERIC_SCHEMA}}
    if effort:
        oc["effort"] = effort
    return {"model": model, "max_tokens": max_tokens,
            "system": [{"type": "text", "text": GENERIC_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": content}],
            "output_config": oc}


def validate(output) -> list[str]:
    if not isinstance(output, dict) or not isinstance(output.get("errors"), list):
        return ["errors 배열 없음"]
    bad = [i for i, e in enumerate(output["errors"])
           if not isinstance(e, dict) or e.get("severity") not in SEVERITIES
           or len((e.get("description") or "").strip()) < 4]
    return [f"형식이 맞지 않는 오류 항목 {bad}"] if bad else []


def _call(client, item: dict, req: dict, model: str, out: Path) -> dict:
    iid = item["item_id"]
    t0 = time.time()
    try:
        resp = client.messages.create(**req)
    except Exception as e:  # noqa: BLE001
        return {"item_id": iid, "model": model, "output": None, "usage": None,
                "errors": [f"API 오류 {e.__class__.__name__}: {e}"]}
    (out / "raw" / f"{iid}.json").write_text(_dump(resp), encoding="utf-8")
    texts = [_get(b, "text") or "" for b in (_get(resp, "content") or []) if _get(b, "type") == "text"]
    output, errs = None, []
    try:
        output = parse_json_output("".join(texts))
        errs = validate(output)
    except (ValueError, json.JSONDecodeError) as e:
        errs = [f"JSON 파싱 실패: {e}"]
    if _get(resp, "stop_reason") == "max_tokens":
        errs.append("max_tokens에서 잘림")
    u = _get(resp, "usage")
    usage = {k: (_get(u, k) or 0) for k in USAGE_KEYS} if u is not None else None
    return {"item_id": iid, "model": _get(resp, "model") or model, "output": output, "usage": usage,
            "stop_reason": _get(resp, "stop_reason"), "seconds": round(time.time() - t0, 1), "errors": errs}


def run_generic(items: list[dict], model: str, name: str, out_root: Path, live: bool,
                workers: int = 4, effort: str | None = "medium", extra_meta: dict | None = None) -> Path:
    out = out_root / name
    (out / "raw").mkdir(parents=True, exist_ok=True)
    reqs = {it["item_id"]: build_request(it, model, effort=effort) for it in items}
    write_json(out / "meta.json", {
        "created": dt.datetime.now().isoformat(timespec="seconds"), "type": "generic", "model": model,
        "prompt_version": "generic", "prompt_fingerprint": fingerprint(), "effort": effort,
        "thinking": "adaptive", "live": live, "items": [it["item_id"] for it in items], **(extra_meta or {})})
    if not live:
        print(f"드라이런: 요청 {len(reqs)}개를 만들었습니다. 실제 호출은 --live.")
        print(json.dumps(next(iter(reqs.values())), ensure_ascii=False, indent=1)[:1500])
        return out
    client = get_client()
    for attempt in range(2):
        prev = latest_records(out)
        todo = [it for it in items if not (prev.get(it["item_id"]) or {}).get("output")
                or (prev.get(it["item_id"]) or {}).get("errors")]
        if not todo:
            break
        if attempt:
            print(f"재시도: {[t['item_id'] for t in todo]}")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_call, client, it, reqs[it["item_id"]], model, out) for it in todo]
            for fu in as_completed(futs):
                rec = fu.result()
                with (out / "outputs.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n = len((rec.get("output") or {}).get("errors") or [])
                print(f"  {rec['item_id']}: {'ok' if not rec['errors'] else rec['errors']} · 오류 {n}개", flush=True)
    recs = latest_records(out)
    bad = [i for i, r in recs.items() if r.get("errors") or not r.get("output")]
    print(f"완료 {len(recs) - len(bad)}/{len(items)} · 실패 {bad} → {out}")
    return out
