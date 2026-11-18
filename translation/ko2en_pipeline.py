"""한→영 도서 번역 파이프라인(수안님 프롬프트 v3.x의 A → B → E → B8)을 API로 자동 실행.

    python translation/ko2en_pipeline.py --prompts <prompts_v3.3.txt> --segments <segments.json> --out <폴더> [--model claude-opus-5-5] [--live]

- 프롬프트 원문(대외비)과 원고(미출간)는 저장소에 넣지 않는다. 경로로만 받는다.
- 대화창용 v3.x의 단계별 지침을 그대로 쓰고, 파일 첨부·묶음 MD 대신 API 입력/출력 형식(JSON)만 덧붙인다.
- E 단계에는 한국어(원문·기록)를 넣지 않는다(v3.x 사용법). 보존 목록도 영어와 문단 ID만.
- segments.json: {"book": {...도서 정보}, "settings": "G2 도서별 기준 텍스트", "segments": [
      {"id": "S1", "paras": [{"pid": "P3-4", "ko": "..."}], "before": ["..."], "after": ["..."]}]}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridgecore.runner import get_client  # noqa: E402

# ---------- 프롬프트 v3.x에서 단계별 지침 꺼내기 ----------

def _between(text, start, end):
    i = text.index(start)
    j = text.index(end, i + len(start))
    return text[i:j]


def load_prompts(path: Path) -> dict:
    t = path.read_text(encoding="utf-8")
    g = _between(t, "프롬프트 G 준비와 프롬프트 생성", "프롬프트 A 구간 번역")
    g0 = _between(g, "G0 공통 원칙", "G1 확인 범위").strip()

    def body(title, nxt):
        sec = _between(t, title, nxt) if nxt else t[t.index(title):]
        return _between(sec, "프롬프트 시작", "프롬프트 끝").replace("프롬프트 시작", "").strip()
    return {"G0": g0, "A": body("프롬프트 A 구간 번역", "프롬프트 B 원문 대조 검토"),
            "B": body("프롬프트 B 원문 대조 검토", "프롬프트 E 영어 편집"),
            "E": body("프롬프트 E 영어 편집", "프롬프트 C 독립 실행")}


API_NOTE = """

[API 실행 안내]
이 작업은 대화창이 아니라 자동 실행이다. 파일 첨부·묶음 MD·다음 창 안내 대신, 위 지침의 출력 항목을 지정된 JSON 형식에 담아 한 번에 답한다.
질문 때문에 멈추지 않는다. 판단이 필요한 항목은 질문 목록에 남기고 가장 보수적인 선택(원문 그대로의 의미)으로 계속한다.
문단 ID(pid)는 입력과 똑같이 유지한다. 모든 대상 문단을 빠짐없이 출력한다."""

S = {"type": "string"}
PARA = {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["pid", "en"],
                                    "properties": {"pid": S, "en": S}}}


def arr(props, req=None):
    return {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                        "required": req or list(props), "properties": props}}


Q = arr({"pid": S, "issue": S})
SCHEMA = {
    "A": {"type": "object", "additionalProperties": False,
          "required": ["translation", "preserve_candidates", "direct_quote_pids", "decisions", "questions", "author_queries"],
          "properties": {"translation": PARA, "preserve_candidates": arr({"pid": S, "en_phrase": S}),
                         "direct_quote_pids": {"type": "array", "items": S},
                         "decisions": arr({"pid": S, "ko": S, "en": S, "kind": S, "note": S}),
                         "questions": Q, "author_queries": Q}},
    "B": {"type": "object", "additionalProperties": False,
          "required": ["revised", "fixes", "questions", "author_queries", "preservation"],
          "properties": {"revised": PARA,
                         "fixes": arr({"pid": S, "before": S, "after": S, "meaning_before_ko": S, "meaning_after_ko": S, "reason": S}),
                         "questions": Q, "author_queries": Q,
                         "preservation": {"type": "object", "additionalProperties": False,
                                          "required": ["header", "direct_quote_pids", "author_phrases", "approved_terms", "numeric_pids"],
                                          "properties": {"header": {"type": "array", "items": S},
                                                         "direct_quote_pids": {"type": "array", "items": S},
                                                         "author_phrases": {"type": "array", "items": S},
                                                         "approved_terms": {"type": "array", "items": S},
                                                         "numeric_pids": {"type": "array", "items": S}}}}},
    "E": {"type": "object", "additionalProperties": False,
          "required": ["edited", "changes", "questions", "grammar_fix_count"],
          "properties": {"edited": PARA,
                         "changes": arr({"pid": S, "before": S, "after": S, "category": S, "reason": S, "check_needed": {"type": "boolean"}}),
                         "questions": Q, "grammar_fix_count": {"type": "integer"}}},
    "B8": {"type": "object", "additionalProperties": False, "required": ["final", "accepted", "rejected"],
           "properties": {"final": PARA, "accepted": arr({"pid": S, "change": S}),
                          "rejected": arr({"pid": S, "change": S, "reason_category": S, "note": S})}},
}


def call(client, model, system, user, schema, effort, max_tokens=32000):
    t0 = time.time()
    oc = {"format": {"type": "json_schema", "schema": schema}}
    if effort:
        oc["effort"] = effort
    with client.messages.stream(model=model, max_tokens=max_tokens,   # 긴 응답이라 스트리밍 필수
                                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                                messages=[{"role": "user", "content": user}], output_config=oc) as st:
        resp = st.get_final_message()
    text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
    u = resp.usage
    usage = {k: getattr(u, k, 0) or 0 for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("max_tokens에서 잘림")
    return json.loads(text), usage, round(time.time() - t0, 1)


def fmt_paras(paras):
    return "\n\n".join(f"[{p['pid']}] {p['ko']}" for p in paras)


def fmt_en(paras):
    return "\n\n".join(f"[{p['pid']}] {p['en']}" for p in paras)


def record_text(rec):
    """앞 구간에서 쌓인 기록(승인 표기·번역 결정)을 다음 구간에 넘긴다 — 책 전체 용어 일관성용"""
    if not rec["approved_terms"] and not rec["decisions"]:
        return ""
    dec = "\n".join(f"- {d['pid']} [{d['kind']}] {d['ko']} → {d['en']}" for d in rec["decisions"][-60:])
    return ("[최신 프로젝트 기록 — 앞 구간에서 정한 것. 같은 표현은 같은 영어로, 이미 한 첫 설명은 반복하지 않는다]\n승인 표기: "
            + " | ".join(rec["approved_terms"][-120:]) + "\n번역 결정:\n" + dec + "\n\n")


def run_segment(client, model, effort, P, book, settings, seg, log, rec=None):
    common = f"\n\n[공통 원칙 G0]\n{P['G0']}\n\n[도서별 기준(G2) — 사용자 승인 범위]\n{settings}"
    ctx = (record_text(rec or {"approved_terms": [], "decisions": []}) + f"[도서 정보]\n{json.dumps(book, ensure_ascii=False)}\n\n[앞 문맥 — 번역하지 않음]\n" + ("\n\n".join(seg['before']) or "(없음)") +
           f"\n\n[번역 대상 구간 {seg['id']}]\n{fmt_paras(seg['paras'])}\n\n[뒤 문맥 — 번역하지 않음]\n" + ("\n\n".join(seg['after']) or "(없음)"))
    out = {}
    a, u, s = call(client, model, P["A"] + common + API_NOTE, ctx + "\n\n위 구간을 번역해 주세요.", SCHEMA["A"], effort)
    out["A"] = a; log.append({"stage": "A", "seg": seg["id"], "usage": u, "sec": s})
    b_user = (ctx + "\n\n[A 영문 초고]\n" + fmt_en(a["translation"]) +
              "\n\n[A 기록]\n" + json.dumps({k: a[k] for k in ("preserve_candidates", "direct_quote_pids", "decisions", "questions", "author_queries")}, ensure_ascii=False) +
              "\n\n원문과 대조해 검토하고 수정본과 B7 보존 목록을 만들어 주세요.")
    b, u, s = call(client, model, P["B"] + common + API_NOTE, b_user, SCHEMA["B"], effort)
    out["B"] = b; log.append({"stage": "B", "seg": seg["id"], "usage": u, "sec": s})
    pres = b["preservation"]
    e_user = ("[보존 목록]\n" + "\n".join(pres["header"]) + "\n1. 직접 인용 범위: " + ", ".join(pres["direct_quote_pids"]) +
              "\n2. 저자 고유 표현: " + " | ".join(pres["author_phrases"]) + "\n3. 승인 표기: " + " | ".join(pres["approved_terms"]) +
              "\n4. 보존 수치·조건 문단: " + ", ".join(pres["numeric_pids"]) +
              "\n\n[영어 본문]\n" + fmt_en(b["revised"]) + "\n\n보존 목록을 지키며 편집해 주세요.")
    e, u, s = call(client, model, P["E"] + API_NOTE, e_user, SCHEMA["E"], effort)
    out["E"] = e; log.append({"stage": "E", "seg": seg["id"], "usage": u, "sec": s})
    b8_user = (ctx + "\n\n[B 검토 수정본]\n" + fmt_en(b["revised"]) + "\n\n[E 편집본]\n" + fmt_en(e["edited"]) +
               "\n\n[E 변경 목록]\n" + json.dumps(e["changes"], ensure_ascii=False) +
               "\n\nB8 절차대로 E 변경 목록을 원문과 대조해 확정본을 만들어 주세요.")
    f, u, s = call(client, model, P["B"] + common + API_NOTE, b8_user, SCHEMA["B8"], effort)
    out["B8"] = f; log.append({"stage": "B8", "seg": seg["id"], "usage": u, "sec": s})
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", type=Path, required=True); p.add_argument("--segments", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="medium"); p.add_argument("--live", action="store_true")
    a = p.parse_args(argv)
    P = load_prompts(a.prompts)
    data = json.loads(a.segments.read_text(encoding="utf-8"))
    print({k: len(v) for k, v in P.items()}, "· 구간", [s["id"] for s in data["segments"]])
    if not a.live:
        print("실제 실행: --live"); return
    client = get_client()
    a.out.mkdir(parents=True, exist_ok=True)
    log = []
    rec = {"approved_terms": [], "decisions": []}
    for seg in data["segments"]:
        res = run_segment(client, a.model, a.effort, P, data["book"], data["settings"], seg, log, rec)
        for t in res["B"]["preservation"]["approved_terms"]:
            if t not in rec["approved_terms"]:
                rec["approved_terms"].append(t)
        rec["decisions"] += res["A"]["decisions"]
        (a.out / "record.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        (a.out / f"{seg['id']}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(seg["id"], "완료", [(x["stage"], x["sec"]) for x in log if x["seg"] == seg["id"]], flush=True)
    tot = {k: sum(x["usage"][k] for x in log) for k in log[0]["usage"]}
    (a.out / "run_log.json").write_text(json.dumps({"model": a.model, "effort": a.effort, "log": log, "total": tot},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("토큰 합계", tot)


if __name__ == "__main__":
    main()
