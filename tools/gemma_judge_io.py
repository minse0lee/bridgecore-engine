"""Gemma 4 31B를 검증엔진으로 쓸 때의 입출력 (Colab 노트북 09와 짝).

    python tools/gemma_judge_io.py export [--version v2] [--items data/golden/items_dev100.jsonl] --out gemma_judge_inputs.json
    python tools/gemma_judge_io.py import gemma_judge_raw_<tag>.jsonl --run runs/golden_dev100_gemma4_<tag>_r1
    python tools/golden_labels.py consensus <R001> <R002> <R003> --run runs/golden_dev100_gemma4_<tag>_r1

- 엔진(Claude)과 같은 시스템 프롬프트·같은 문항 틀·같은 자동 점검 힌트를 쓴다(prompt.py 그대로).
- 다른 점 하나: Claude는 API 구조화 출력이 JSON 스키마를 강제하지만, Gemma에는 그런 장치가 없어서
  같은 스키마를 시스템 프롬프트 끝에 글로 붙인다(GEMMA_FORMAT). 지문(fingerprint)에 이 차이가 기록된다.
- 결과는 엔진과 같은 outputs.jsonl 형식으로 바꿔서 golden_labels.py로 그대로 채점한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridgecore.prompt import prompt_fingerprint, system_text, user_text  # noqa: E402
from bridgecore.schema import JSON_SCHEMA, normalize, parse_json_output, validate  # noqa: E402

GEMMA_FORMAT = ("\n\n[출력 형식 — 반드시 지킬 것]\n코드블록이나 설명 없이 아래 JSON 스키마를 따르는 JSON 객체 하나만 출력합니다. "
                "키 이름·순서(item_id, evaluable, evaluable_note, errors, axes)와 여섯 축 이름(Semantic, Textual, Context, "
                "Deontic, Register, Cultural)을 그대로 씁니다. judgment는 보존·훼손·N/A·판단보류 중 하나, severity는 "
                "minor·major·critical 또는 null입니다.\n" + json.dumps(JSON_SCHEMA, ensure_ascii=False))


def cmd_export(a):
    items = [json.loads(l) for l in open(a.items, encoding="utf-8") if l.strip()]
    if any(it.get("figure") for it in items):
        sys.exit("그림 문항은 아직 지원하지 않음")
    system = system_text(a.version) + GEMMA_FORMAT
    fp = prompt_fingerprint(a.version)
    gfp = hashlib.sha256((fp + GEMMA_FORMAT).encode("utf-8")).hexdigest()[:12]
    data = {"prompt_version": a.version, "engine_fingerprint": fp, "gemma_fingerprint": gfp, "system": system,
            "items": [{"item_id": it["item_id"], "user": user_text(it, a.version)} for it in items]}
    Path(a.out).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"→ {a.out} · 문항 {len(items)} · 엔진 지문 {fp} · Gemma 지문 {gfp}")


def cmd_import(a):
    raws = [json.loads(l) for l in open(a.raw, encoding="utf-8") if l.strip()]
    last = {}
    for r in raws:                       # 같은 문항이 여러 번 있으면 마지막(재시도) 것을 쓴다
        last[r["item_id"]] = r
    out = Path(a.run); out.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(out / "outputs.jsonl", "w", encoding="utf-8") as f:
        for iid, r in sorted(last.items()):
            o, errs, warns = None, [], []
            try:
                o = normalize(parse_json_output(r["content"]))
                errs, warns = validate(o, iid)
            except (ValueError, json.JSONDecodeError) as e:
                errs = [f"JSON 파싱 실패: {e}"]
            if r.get("truncated"):
                errs.append("max_new_tokens에서 잘림")
            n_ok += not errs
            f.write(json.dumps({"item_id": iid, "model": r.get("model"), "output": o,
                                "usage": {"output_tokens": r.get("out_tokens")}, "stop_reason": None,
                                "seconds": r.get("sec"), "errors": errs, "warnings": warns,
                                "decoding": r.get("decoding")}, ensure_ascii=False) + "\n")
    info = json.loads(Path(a.info).read_text(encoding="utf-8")) if a.info else {}
    meta = {"model": raws[0].get("model") if raws else None, "runtime": "colab_transformers", **info,
            "items": sorted(last), "n_ok": n_ok, "source_raw": str(a.raw)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out} · 문항 {len(last)} · 정상 {n_ok} · 실패 {len(last) - n_ok}")


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("export"); s.add_argument("--version", default="v2")
    s.add_argument("--items", default=str(ROOT / "data/golden/items_dev100.jsonl")); s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_export)
    s = sub.add_parser("import"); s.add_argument("raw"); s.add_argument("--run", required=True); s.add_argument("--info")
    s.set_defaults(fn=cmd_import)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
