"""엔진 실행기 — 동기 호출, Batch, 2단계(선별 → 재판정).

실행 폴더 runs/<이름>/:
  meta.json        날짜·모델·프롬프트 버전·해시·선별 모드·문항 목록 (버전 기록)
  requests.jsonl   요청 본문 (그림 base64는 길이만 남김)
  outputs.jsonl    문항별 {item_id, model, output, usage, errors, warnings}  ← score 입력
                   덧붙이기만 한다. 같은 문항이 여러 번 있으면 마지막 줄이 유효 (latest_records)
  raw/<ID>.json    API 원응답
  batch.json       Batch 제출 정보 (batch_id, custom_id ↔ 문항 ID)
  engine_sheet.csv 사람 평가 시트와 같은 열로 펼친 결과

드라이런(기본)은 요청만 저장한다. --live 일 때만 API를 부른다.
동기 실행은 중간에 끊겨도 같은 --name으로 다시 실행하면 성공한 문항은 건너뛰고 실패한 문항만 다시 부른다.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

from .constants import AXES, HOLD, VIOLATED
from .io_utils import read_json, read_jsonl, write_csv, write_json
from .prompt import build_request, prompt_fingerprint
from .schema import parse_json_output
from .schema import normalize, to_sheet_rows, validate

ROOT = Path(__file__).resolve().parent.parent
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


# ---------- 공통 ----------
def load_env(path: Path = ROOT / ".env") -> None:
    """python-dotenv 없이 .env의 KEY=VALUE를 환경변수로 읽는다 (이미 있으면 덮지 않음)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_client():
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY가 없습니다. engine/.env 에 넣으세요 (.env.example 참고).")
    try:
        import anthropic
    except ImportError:
        sys.exit("anthropic 패키지가 없습니다: pip install -r requirements.txt")
    return anthropic.Anthropic(max_retries=4)


def latest_records(out_dir: Path) -> dict[str, dict]:
    path = Path(out_dir) / "outputs.jsonl"
    recs = {}
    if path.exists():
        for r in read_jsonl(path):
            recs[r["item_id"]] = r
    return recs


def _strip_images(req: dict) -> dict:
    req = json.loads(json.dumps(req))
    for m in req["messages"]:
        for c in m["content"]:
            if c.get("type") == "image":
                c["source"]["data"] = f"<base64 {len(c['source']['data'])} chars>"
    return req


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _dump(resp) -> str:
    if hasattr(resp, "model_dump_json"):
        return resp.model_dump_json(indent=2)
    return json.dumps(resp, ensure_ascii=False, indent=2, default=str)


def process_response(item_id: str, resp, model: str, seconds: float | None = None) -> dict:
    """API 응답(Message) → outputs.jsonl 레코드. 동기·Batch 공통."""
    # 구조화 출력: JSON은 text 블록에 온다. 사고(thinking) 블록은 건너뛴다.
    content = _get(resp, "content") or []
    texts = [_get(b, "text") or "" for b in content if _get(b, "type") == "text"]
    stop = _get(resp, "stop_reason")
    output, errs = None, []
    if texts:
        try:
            output = normalize(parse_json_output("".join(texts)))
        except (ValueError, json.JSONDecodeError) as e:
            errs.append(f"JSON 파싱 실패: {e}")
    else:
        errs.append("text 블록 없음")
    warns: list[str] = []
    if output is not None:
        e2, warns = validate(output, item_id)
        errs += e2
    if stop == "max_tokens":
        errs.append("max_tokens에서 잘림 (사고 토큰 포함) — --max-tokens를 늘리거나 effort를 낮추세요")
    elif stop == "refusal":
        errs.append("모델이 응답을 거절함 (stop_reason=refusal)")
    u = _get(resp, "usage")
    usage = {k: (_get(u, k) or 0) for k in USAGE_KEYS} if u is not None else None
    return {"item_id": item_id, "model": _get(resp, "model") or model, "output": output,
            "usage": usage, "stop_reason": stop, "seconds": seconds,
            "errors": errs, "warnings": warns}


def error_record(item_id: str, model: str, message: str) -> dict:
    return {"item_id": item_id, "model": model, "output": None, "usage": None,
            "stop_reason": None, "seconds": None, "errors": [message], "warnings": []}


def _append(out_dir: Path, rec: dict) -> None:
    with (out_dir / "outputs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _ok(rec: dict | None) -> bool:
    return bool(rec and rec.get("output") and not rec.get("errors"))


def finalize(out_dir: Path) -> dict:
    """engine_sheet.csv 작성 + 토큰·상태 요약."""
    recs = latest_records(out_dir)
    rows = [r for rec in recs.values() if rec.get("output") for r in to_sheet_rows(rec["output"])]
    write_csv(out_dir / "engine_sheet.csv", rows)
    tot = {k: 0 for k in USAGE_KEYS}
    for rec in recs.values():
        for k in USAGE_KEYS:
            tot[k] += (rec.get("usage") or {}).get(k, 0)
    meta = read_json(out_dir / "meta.json")
    summary = {"items": len(meta["items"]), "done_ok": sum(_ok(recs.get(i)) for i in meta["items"]),
               "failed": [i for i in meta["items"] if i in recs and not _ok(recs[i])],
               "missing": [i for i in meta["items"] if i not in recs], "tokens": tot}
    return summary


def _print_summary(out_dir: Path, s: dict) -> None:
    print(f"완료 {s['done_ok']}/{s['items']} · 실패 {len(s['failed'])} {s['failed'][:10]} · "
          f"미실행 {len(s['missing'])}")
    print(f"토큰 합계: {s['tokens']}")
    if s["tokens"]["input_tokens"] and not s["tokens"]["cache_read_input_tokens"]:
        print("⚠ cache_read가 0입니다. 프롬프트 캐싱이 적용되지 않았을 수 있습니다 (최소 캐시 길이·모델 확인).")
    print(f"채점: python -m bridgecore.cli score {out_dir}")


# ---------- 준비 ----------
def prepare_run(items: list[dict], model: str, version: str, run_name: str | None, live: bool,
                out_root: Path, max_tokens: int = 16000, screen: bool = False,
                extra_meta: dict | None = None, effort: str | None = "medium",
                thinking: bool = True) -> tuple[Path, list[dict]]:
    stamp = dt.datetime.now().strftime("%Y%m%d")
    name = run_name or f"{stamp}_{version}{'_screen' if screen else ''}_{model}{'' if live else '_dryrun'}"
    out = Path(out_root) / name
    (out / "raw").mkdir(parents=True, exist_ok=True)
    meta = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "model": model, "prompt_version": version, "screen": screen,
        "prompt_fingerprint": prompt_fingerprint(version, screen),
        "max_tokens": max_tokens, "effort": effort, "thinking": "adaptive" if thinking else "off",
        "live": live,
        "items": [i["item_id"] for i in items], **(extra_meta or {}),
    }
    meta_path = out / "meta.json"
    if meta_path.exists():
        old = read_json(meta_path)
        keys = ("model", "prompt_fingerprint", "effort", "thinking")
        if tuple(old.get(k) for k in keys) != tuple(meta[k] for k in keys):
            sys.exit(f"같은 실행 이름에 다른 모델/프롬프트/effort/thinking 설정이 섞입니다: {name}. --name을 바꾸세요.")
        meta["created"] = old["created"]
        meta["items"] = list(dict.fromkeys(old["items"] + meta["items"]))
    write_json(meta_path, meta)
    reqs = [build_request(it, model, version, max_tokens, screen, effort, thinking) for it in items]
    with (out / "requests.jsonl").open("w", encoding="utf-8") as f:
        for it, r in zip(items, reqs):
            f.write(json.dumps({"item_id": it["item_id"], "request": _strip_images(r)},
                               ensure_ascii=False) + "\n")
    print(f"실행 폴더: {out}")
    print(f"문항 {len(items)}개 · 모델 {model} · 프롬프트 {version}"
          f"{' +선별' if screen else ''} ({meta['prompt_fingerprint']}) · effort {effort or '모델 기본'}"
          f" · 사고 {'켜짐' if thinking else '꺼짐'}")
    return out, reqs


def _dry_run_note(reqs: list[dict]) -> None:
    if not reqs:
        return
    sys_chars = len(reqs[0]["system"][0]["text"])
    user_chars = sum(len(r["messages"][0]["content"][-1]["text"]) for r in reqs) // len(reqs)
    print(f"[드라이런] API를 부르지 않았습니다. 시스템 프롬프트 {sys_chars:,}자(캐시 대상), "
          f"문항 본문 평균 {user_chars:,}자.")
    print("실제 실행: 같은 명령에 --live 를 붙이세요 (.env 의 ANTHROPIC_API_KEY 필요).")


# ---------- 동기 실행 ----------
def run(items: list[dict], model: str, version: str, run_name: str | None, live: bool,
        out_root: Path, max_tokens: int = 16000, screen: bool = False, client=None,
        extra_meta: dict | None = None, effort: str | None = "medium", thinking: bool = True) -> Path:
    out, reqs = prepare_run(items, model, version, run_name, live, out_root, max_tokens, screen,
                            extra_meta, effort, thinking)
    if not live:
        _dry_run_note(reqs)
        return out
    client = client or get_client()
    prev = latest_records(out)
    for it, req in zip(items, reqs):
        iid = it["item_id"]
        if _ok(prev.get(iid)):
            continue
        t0 = time.time()
        try:
            resp = client.messages.create(**req)
        except Exception as e:  # API 오류도 기록하고 다음 문항으로
            rec = error_record(iid, model, f"API 오류 {e.__class__.__name__}: {e}")
            _append(out, rec)
            print(f"  {iid}: {rec['errors'][0]}")
            continue
        (out / "raw" / f"{iid}.json").write_text(_dump(resp), encoding="utf-8")
        rec = process_response(iid, resp, model, round(time.time() - t0, 1))
        _append(out, rec)
        u = rec["usage"] or {}
        print(f"  {iid}: {'형식오류 ' + '; '.join(rec['errors']) if rec['errors'] else 'ok'} · "
              f"in {u.get('input_tokens')} / cache_read {u.get('cache_read_input_tokens')} / "
              f"out {u.get('output_tokens')}")
    _print_summary(out, finalize(out))
    return out


# ---------- Batch ----------
def submit_batch(items: list[dict], model: str, version: str, run_name: str | None, live: bool,
                 out_root: Path, max_tokens: int = 16000, screen: bool = False, client=None,
                 extra_meta: dict | None = None, effort: str | None = "medium",
                 thinking: bool = True) -> Path:
    """Message Batches로 제출만 하고 끝난다. 결과는 collect로 받는다.
    이미 성공한 문항은 제출하지 않는다 (실패분만 다시 제출할 때 같은 --name 사용)."""
    out, reqs = prepare_run(items, model, version, run_name, live, out_root, max_tokens, screen,
                            {**(extra_meta or {}), "mode": "batch"}, effort, thinking)
    if not live:
        _dry_run_note(reqs)
        print("Batch 제출: --live --batch")
        return out
    prev = latest_records(out)
    todo = [(it, r) for it, r in zip(items, reqs) if not _ok(prev.get(it["item_id"]))]
    if not todo:
        print("제출할 문항이 없습니다 (모두 완료).")
        return out
    batch_path = out / "batch.json"
    history = read_json(batch_path)["batches"] if batch_path.exists() else []
    if history and history[-1]["status"] != "collected":
        sys.exit(f"아직 수집하지 않은 Batch가 있습니다: {history[-1]['batch_id']}. 먼저 collect 하세요.")
    # custom_id는 영문·숫자·_- 64자 이하만 가능 → 순번으로 만들고 문항 ID와 대응표를 남긴다
    id_map = {f"r{len(history):02d}_{i:05d}": it["item_id"] for i, (it, _) in enumerate(todo)}
    requests = [{"custom_id": cid, "params": r} for cid, (_, r) in zip(id_map, todo)]
    client = client or get_client()
    b = client.messages.batches.create(requests=requests)
    history.append({"batch_id": _get(b, "id"), "submitted": dt.datetime.now().isoformat(timespec="seconds"),
                    "n": len(requests), "custom_ids": id_map, "status": "submitted"})
    write_json(batch_path, {"batches": history})
    print(f"Batch 제출: {_get(b, 'id')} · {len(requests)}문항. 결과 수집: python -m bridgecore.cli collect {out}")
    return out


def collect_batch(out_dir: Path, client=None) -> bool:
    """마지막 Batch 상태를 확인하고, 끝났으면 결과를 outputs.jsonl에 넣는다. 끝났으면 True."""
    out_dir = Path(out_dir)
    batch_path = out_dir / "batch.json"
    if not batch_path.exists():
        sys.exit(f"batch.json이 없습니다: {out_dir}")
    info = read_json(batch_path)
    last = info["batches"][-1]
    if last["status"] == "collected":
        print("이미 수집했습니다.")
        _print_summary(out_dir, finalize(out_dir))
        return True
    meta = read_json(out_dir / "meta.json")
    client = client or get_client()
    b = client.messages.batches.retrieve(last["batch_id"])
    status = _get(b, "processing_status")
    counts = _get(b, "request_counts")
    cdict = {k: _get(counts, k) for k in ("processing", "succeeded", "errored", "canceled", "expired")} \
        if counts is not None else {}
    print(f"Batch {last['batch_id']}: {status} {cdict}")
    if status != "ended":
        print("아직 처리 중입니다. 나중에 다시 collect 하세요.")
        return False
    seen = set()
    for entry in client.messages.batches.results(last["batch_id"]):
        cid = _get(entry, "custom_id")
        iid = last["custom_ids"].get(cid)
        if iid is None:
            continue
        seen.add(cid)
        result = _get(entry, "result")
        rtype = _get(result, "type")
        if rtype == "succeeded":
            msg = _get(result, "message")
            (out_dir / "raw" / f"{iid}.json").write_text(_dump(msg), encoding="utf-8")
            rec = process_response(iid, msg, meta["model"])
        else:
            err = _get(result, "error")
            rec = error_record(iid, meta["model"], f"Batch {rtype}: {err}")
        rec["via_batch"] = True  # 비용 계산 시 Batch 할인 적용
        _append(out_dir, rec)
    for cid, iid in last["custom_ids"].items():
        if cid not in seen:
            _append(out_dir, error_record(iid, meta["model"], "Batch 결과 없음"))
    last["status"] = "collected"
    last["collected"] = dt.datetime.now().isoformat(timespec="seconds")
    write_json(batch_path, info)
    s = finalize(out_dir)
    _print_summary(out_dir, s)
    if s["failed"]:
        print("실패한 문항만 다시 제출하려면 같은 --name 으로 run --batch --live 를 다시 실행하세요.")
    return True


# ---------- 2단계: 선별 → 재판정 ----------
ESCALATE_DEFAULT = ("violated", "hold")


def needs_escalation(rec: dict | None, on: tuple[str, ...] = ESCALATE_DEFAULT) -> tuple[bool, str]:
    """선별 결과로 2차 판정이 필요한지. 출력이 없거나 형식 오류면 항상 넘긴다."""
    if not _ok(rec):
        return True, "선별 실패·형식 오류"
    out = rec["output"]
    if out.get("evaluable") is False:
        return True, "평가 불가 표시"
    axes = out.get("axes") or {}
    hit = [ax for ax in AXES if
           ("violated" in on and axes.get(ax, {}).get("judgment") == VIOLATED) or
           ("hold" in on and axes.get(ax, {}).get("judgment") == HOLD)]
    return (bool(hit), ",".join(hit))


def escalation_list(screen_dir: Path, on: tuple[str, ...] = ESCALATE_DEFAULT) -> list[dict]:
    meta = read_json(Path(screen_dir) / "meta.json")
    recs = latest_records(screen_dir)
    rows = []
    for iid in meta["items"]:
        esc, why = needs_escalation(recs.get(iid), on)
        rows.append({"item_id": iid, "escalate": esc, "why": why})
    return rows


def merge_cascade(screen_dir: Path, stage2_dir: Path, out_dir: Path,
                  on: tuple[str, ...] = ESCALATE_DEFAULT) -> Path:
    """선별 실행 + 재판정 실행 → 최종 실행 폴더. 재판정 대상은 2차 결과, 나머지는 1차 결과.
    2차가 실패한 문항은 1차 결과를 쓰고 경고를 남긴다."""
    screen_dir, stage2_dir, out_dir = Path(screen_dir), Path(stage2_dir), Path(out_dir)
    m1, m2 = read_json(screen_dir / "meta.json"), read_json(stage2_dir / "meta.json")
    r1, r2 = latest_records(screen_dir), latest_records(stage2_dir)
    esc = {r["item_id"]: r for r in escalation_list(screen_dir, on)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "outputs.jsonl").write_text("", encoding="utf-8")
    n2 = fallback = 0
    for iid in m1["items"]:
        if esc[iid]["escalate"]:
            if _ok(r2.get(iid)):
                rec = {**r2[iid], "stage": 2, "escalate_reason": esc[iid]["why"]}
                n2 += 1
            else:
                fallback += 1
                rec = {**(r1.get(iid) or error_record(iid, m1["model"], "선별 결과 없음")), "stage": 1,
                       "escalate_reason": esc[iid]["why"]}
                rec["warnings"] = (rec.get("warnings") or []) + ["2차 판정 없음 → 1차 결과 사용"]
        else:
            rec = {**r1[iid], "stage": 1, "escalate_reason": None}
        _append(out_dir, rec)
    # 2단계 비용 계산을 위해 1차 호출 토큰도 함께 보관 (재판정 문항은 1차 비용도 들었으므로)
    write_json(out_dir / "stage1_usage.json",
               {iid: {"model": (r1.get(iid) or {}).get("model", m1["model"]),
                      "usage": (r1.get(iid) or {}).get("usage"),
                      "via_batch": (r1.get(iid) or {}).get("via_batch", False)} for iid in m1["items"]})
    write_json(out_dir / "meta.json", {
        "created": dt.datetime.now().isoformat(timespec="seconds"), "type": "cascade",
        "model": f"{m1['model']}→{m2['model']}", "prompt_version": f"{m1['prompt_version']}",
        "prompt_fingerprint": f"{m1['prompt_fingerprint']}+{m2['prompt_fingerprint']}",
        "screen_run": str(screen_dir), "stage2_run": str(stage2_dir), "escalate_on": list(on),
        "items": m1["items"], "escalated": n2 + fallback, "stage2_used": n2, "stage2_missing": fallback,
    })
    s = finalize(out_dir)
    print(f"2단계 병합: {out_dir}")
    print(f"문항 {len(m1['items'])} · 재판정 대상 {n2 + fallback} ({(n2 + fallback) / max(len(m1['items']), 1):.0%})"
          f" · 2차 결과 사용 {n2} · 2차 없음(1차 사용) {fallback}")
    print(f"채점: python -m bridgecore.cli score {out_dir}")
    return out_dir
