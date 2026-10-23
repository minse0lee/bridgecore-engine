"""엔진 실행기.

기본은 드라이런: API를 부르지 않고 요청 본문만 runs/<이름>/requests.jsonl 에 저장한다
(프롬프트 확인, 토큰 수 사전 확인용). --live 일 때만 실제 호출한다.

실행 폴더 구성:
  meta.json        날짜·모델·프롬프트 버전·프롬프트 해시·문항 목록 (개발일지용 버전 기록)
  requests.jsonl   요청 본문 (그림 base64는 길이만 남김)
  outputs.jsonl    문항별 {item_id, output, usage, errors, warnings}  ← score 입력
  raw/<ID>.json    API 원응답
  engine_sheet.csv 사람 평가 시트와 같은 열로 펼친 결과
중간에 끊겨도 같은 --name으로 다시 실행하면 이미 끝난 문항은 건너뛴다.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

from .io_utils import read_jsonl, write_csv, write_json
from .prompt import TOOL_NAME, build_request, prompt_fingerprint
from .schema import normalize, to_sheet_rows, validate

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path = ROOT / ".env") -> None:
    """python-dotenv 없이 .env의 KEY=VALUE를 환경변수로 읽는다 (이미 있으면 덮지 않음)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _strip_images(req: dict) -> dict:
    req = json.loads(json.dumps(req))
    for m in req["messages"]:
        for c in m["content"]:
            if c.get("type") == "image":
                c["source"]["data"] = f"<base64 {len(c['source']['data'])} chars>"
    return req


def _usage(resp) -> dict:
    u = resp.usage
    return {k: getattr(u, k, 0) or 0 for k in
            ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}


def run(items: list[dict], model: str, version: str, run_name: str | None, live: bool,
        out_root: Path, max_tokens: int = 4000) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d")
    name = run_name or f"{stamp}_{version}_{model}{'' if live else '_dryrun'}"
    out = out_root / name
    (out / "raw").mkdir(parents=True, exist_ok=True)

    meta_path = out / "meta.json"
    meta = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "model": model, "prompt_version": version,
        "prompt_fingerprint": prompt_fingerprint(version),
        "max_tokens": max_tokens, "temperature": 0, "live": live,
        "items": [i["item_id"] for i in items],
    }
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if (old["model"], old["prompt_fingerprint"]) != (model, meta["prompt_fingerprint"]):
            sys.exit(f"같은 실행 이름에 다른 모델/프롬프트가 섞입니다: {name}. --name을 바꾸세요.")
        meta["created"] = old["created"]
    write_json(meta_path, meta)

    reqs = [build_request(it, model, version, max_tokens) for it in items]
    with (out / "requests.jsonl").open("w", encoding="utf-8") as f:
        for it, r in zip(items, reqs):
            f.write(json.dumps({"item_id": it["item_id"], "request": _strip_images(r)},
                               ensure_ascii=False) + "\n")
    print(f"실행 폴더: {out}")
    print(f"문항 {len(items)}개 · 모델 {model} · 프롬프트 {version} ({meta['prompt_fingerprint']})")

    if not live:
        chars = sum(len(r["system"][0]["text"]) for r in reqs[:1])
        user_chars = sum(len(r["messages"][0]["content"][-1]["text"]) for r in reqs)
        print(f"[드라이런] API를 부르지 않았습니다. 시스템 프롬프트 {chars:,}자(캐시 대상), "
              f"문항 본문 평균 {user_chars // max(len(reqs), 1):,}자.")
        print("실제 실행: 같은 명령에 --live 를 붙이세요 (.env 의 ANTHROPIC_API_KEY 필요).")
        return out

    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY가 없습니다. engine/.env 에 넣으세요 (.env.example 참고).")
    try:
        import anthropic
    except ImportError:
        sys.exit("anthropic 패키지가 없습니다: pip install -r requirements.txt")
    client = anthropic.Anthropic(max_retries=4)

    out_path = out / "outputs.jsonl"
    done = {r["item_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    totals = {"input_tokens": 0, "output_tokens": 0,
              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    with out_path.open("a", encoding="utf-8") as f:
        for it, req in zip(items, reqs):
            iid = it["item_id"]
            if iid in done:
                continue
            t0 = time.time()
            try:
                resp = client.messages.create(**req)
            except anthropic.APIError as e:
                print(f"  {iid}: API 오류 {e.__class__.__name__}: {e}")
                f.write(json.dumps({"item_id": iid, "output": None, "usage": None,
                                    "errors": [f"API 오류: {e}"], "warnings": []},
                                   ensure_ascii=False) + "\n")
                f.flush()
                continue
            (out / "raw" / f"{iid}.json").write_text(resp.model_dump_json(indent=2), encoding="utf-8")
            block = next((b for b in resp.content if b.type == "tool_use" and b.name == TOOL_NAME), None)
            output = normalize(dict(block.input)) if block else None
            errs, warns = validate(output, iid) if output else (["도구 호출 없음"], [])
            if resp.stop_reason == "max_tokens":
                errs.append("max_tokens에서 잘림")
            usage = _usage(resp)
            for k in totals:
                totals[k] += usage[k]
            rec = {"item_id": iid, "output": output, "usage": usage, "stop_reason": resp.stop_reason,
                   "seconds": round(time.time() - t0, 1), "errors": errs, "warnings": warns}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            flag = "형식오류 " + "; ".join(errs) if errs else "ok"
            print(f"  {iid}: {flag} · in {usage['input_tokens']} / cache_read "
                  f"{usage['cache_read_input_tokens']} / out {usage['output_tokens']}")

    recs = read_jsonl(out_path)
    rows = [r for rec in recs if rec.get("output") for r in to_sheet_rows(rec["output"])]
    write_csv(out / "engine_sheet.csv", rows)
    print(f"토큰 합계(이번 실행): {totals}")
    print(f"채점: python -m bridgecore.cli score {out}")
    return out
