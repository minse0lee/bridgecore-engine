"""Deontic 범주 대응 규칙 (가이드라인 7절 '자주 나오는 조동사' 표를 코드로 옮긴 것).

원문(영어)과 번역문(한국어)에서 규범 범주 표지를 찾고, 범주가 뒤집힌 경우를 표시한다.
규칙은 판정자가 아니다. 이 문장이 Deontic 적용 대상인지(추론의 must, 가능성의 may 등)는
LLM이 정한다. 규칙은 '적용 대상이라면 이 대응이 맞는지 확인하라'는 신호만 만든다.

범주
  OBL   의무        must, have to, need to, required to, shall(규정), mandate
  REC   권고        should, ought to, advised, recommended
  PERM  허가        may, can, allowed, permitted
  PROH  금지        must not, may not, shall not, cannot, not allowed, prohibited, 문두 Do not/Never
  NONOBL 의무 없음  need not, do not have to, not required

강도
  strong  뜻이 거의 하나인 표현 (must not, need not, required, prohibited, permitted ...)
  weak    규범이 아닌 뜻으로도 흔히 쓰이는 표현 (may, can, cannot, shall, should, must)
          should·must는 예상·추론으로도 쓰여 weak로 둔다
"""
from __future__ import annotations

import re

OBL, REC, PERM, PROH, NONOBL = "OBL", "REC", "PERM", "PROH", "NONOBL"
KO_NAME = {OBL: "의무", REC: "권고", PERM: "허가", PROH: "금지", NONOBL: "의무 없음"}

# ---------- 영어 원문 ----------
# (정규식, 범주, 강도). 위에서부터 맞추고, 맞은 구간은 지워서 아래 짧은 표현이 다시 잡히지 않게 한다.
_EN_PATTERNS: list[tuple[str, str, str]] = [
    # 의무 없음 (부정 + 의무) — must not 보다 먼저
    (r"\bneed(?:s)? not\b|\bneedn['’]t\b", NONOBL, "strong"),
    (r"\b(?:do|does|did)(?: not|n['’]t) (?:have|need) to\b", NONOBL, "strong"),
    (r"\b(?:is|are|was|were|be)(?: not|n['’]t) (?:required|obligated|obliged|mandated)\b", NONOBL, "strong"),
    (r"\bnot (?:required|obligated|obliged|mandatory)\b|\boptional\b", NONOBL, "strong"),
    (r"\bno (?:need|requirement|obligation) to\b", NONOBL, "strong"),
    # 금지
    (r"\bmust(?: not|n['’]t)\b", PROH, "strong"),
    (r"\bshall(?: not|n['’]t)\b", PROH, "strong"),
    (r"\bmay not\b", PROH, "weak"),          # 가능성 부정(may not be ...)일 수 있음
    (r"\bshould(?: not|n['’]t)\b|\bought not to\b", PROH, "weak"),
    (r"\b(?:is|are|was|were|be)(?: not|n['’]t) (?:allowed|permitted)\b|\bnot (?:allowed|permitted)\b", PROH, "strong"),
    (r"\bprohibit\w*|\bforbid\w*|\bbanned\b", PROH, "strong"),
    (r"(?:^|[.!?;:\"“”(]\s*)(?:do not|don['’]t|never)\b", PROH, "strong"),  # 문두 명령형
    (r"\bcannot\b|\bcan['’]t\b|\bcan not\b", PROH, "weak"),  # 능력 부정이 더 흔함
    # 의무
    (r"\bmust\b", OBL, "weak"),               # 추론의 must 가능
    (r"\b(?:has|have|had) to\b|\bhas got to\b|\bhave got to\b", OBL, "strong"),
    (r"\bneed(?:s)? to\b", OBL, "weak"),
    (r"\b(?:is|are|was|were|be) (?:required|obligated|obliged)\b|\brequire[sd]? (?:that|to|\w+ to)\b", OBL, "strong"),
    (r"\bmandat\w*|\bobligat\w*|\bcompulsory\b|\bmandatory\b", OBL, "strong"),
    (r"\bshall\b", OBL, "weak"),               # 제안 의문(Shall we ...?)일 수 있음
    # 권고
    (r"\bshould\b|\bought to\b", REC, "weak"),  # 예상의 should 가능
    (r"\b(?:is|are) (?:advised|recommended|encouraged)\b|\brecommend\w*|\badvis\w+ to\b", REC, "strong"),
    # 허가
    (r"\b(?:is|are|was|were|be) (?:allowed|permitted|entitled|eligible)\b", PERM, "strong"),
    (r"\bpermit\w*|\ballow\w*|\bentitled\b|\beligible\b", PERM, "strong"),
    (r"\bmay\b", PERM, "weak"),                # 가능성의 may가 더 흔함
    (r"\bcan\b", PERM, "weak"),                # 능력·가능성의 can이 더 흔함
]
_EN_RE = [(re.compile(p, re.I), c, s) for p, c, s in _EN_PATTERNS]


def english_markers(src: str) -> list[dict]:
    text = src or ""
    out = []
    for rx, cat, strength in _EN_RE:
        def take(m):
            out.append({"cat": cat, "strength": strength, "text": m.group(0).strip(" .!?;:\"“”(")})
            return " " * len(m.group(0))
        text = rx.sub(take, text)
    return out


# ---------- 한국어 번역문 ----------
# 띄어쓰기가 들쭉날쭉하므로 공백을 지운 문자열에서 찾는다.
_KO_PATTERNS: list[tuple[str, str, str]] = [
    # 의무 없음 — '해도 된다'(허가)보다 먼저
    (r"(?:않|안\S{0,2})아?도(?:된|됩|돼|되|무방|괜찮)", NONOBL, "strong"),
    (r"필요(?:가|는|도)?없", NONOBL, "strong"),
    (r"요구되지(?:는|도)?않|의무(?:가|는)?없|강제(?:가|는)?(?:아니|없)|선택사항", NONOBL, "strong"),
    # 금지
    (r"(?:면|서는|선|어선|아선|여선)안(?:된|됩|돼|되)", PROH, "strong"),
    (r"지(?:마|말|맙|맙시)", PROH, "strong"),
    (r"금지|금합|금한|불허|허용되지(?:는|도)?않|허락되지(?:는)?않", PROH, "strong"),
    (r"수(?:는|도)?없", PROH, "weak"),         # 능력 부정과 구분 안 됨
    # 의무
    (r"(?:야|어야|아야|여야)(?:하|한|합|해|할|했|만)", OBL, "strong"),
    (r"필수|의무(?:적|화|이|가|를|로)?|반드시|요구(?:된|됩|되며|되는|한다|합니다|하는)|하도록되어|하게되어있", OBL, "strong"),
    # 권고
    (r"(?:것이|게|편이)(?:좋|바람직|낫)|바랍니다|바란다|권장|권고|권합|권한다|추천", REC, "strong"),
    # 허가
    (r"도(?:된|됩|돼|되|좋|무방|괜찮)", PERM, "strong"),
    (r"허용|허가|허락|자격(?:이|을)?(?:있|갖|얻)", PERM, "strong"),
    (r"수(?:는|도)?있", PERM, "weak"),         # 가능성과 구분 안 됨
]
_KO_RE = [(re.compile(p), c, s) for p, c, s in _KO_PATTERNS]


def korean_markers(tgt: str) -> list[dict]:
    raw = tgt or ""
    pos = [i for i, ch in enumerate(raw) if not ch.isspace()]  # 공백 지운 문자열 위치 → 원래 위치
    text = "".join(raw[i] for i in pos)
    out = []
    for rx, cat, strength in _KO_RE:
        def take(m):
            a, b = pos[m.start()], pos[m.end() - 1] + 1
            out.append({"cat": cat, "strength": strength, "text": raw[a:b]})
            return "□" * len(m.group(0))
        text = rx.sub(take, text)
    return out


# ---------- 대응 비교 ----------
# 원문 범주별로 번역에 있으면 '맞는 대응'으로 보는 범주
_COMPATIBLE = {
    OBL: {OBL},               # must → -하는 것이 좋다는 의무가 권고로 약해진 것 (가이드라인 예시: Deontic 훼손)
    REC: {REC, OBL},          # should → -해야 한다는 통상 대응 (가이드라인 5절)
    PERM: {PERM},
    PROH: {PROH},
    NONOBL: {NONOBL},
}
# 원문 범주 → 번역에서 나오면 '범주가 뒤집힘'으로 보는 범주
_CONFLICT = {
    OBL: {NONOBL, PERM, REC},
    REC: {NONOBL, PROH},
    PERM: {OBL, PROH},
    PROH: {NONOBL, PERM},
    NONOBL: {PROH, OBL},
}


def check(src: str, tgt: str) -> dict:
    """반환: {src_markers, tgt_markers, flags: [{type, src_cat, tgt_cat, strength, src_text, tgt_text}]}

    flag type
      flip      원문 범주는 번역에 없고, 대신 충돌하는 범주가 있음 (예: 금지 → 의무 없음)
      vanished  원문에 뜻이 분명한(strong) 규범 표지가 있는데 번역에는 규범 표지가 하나도 없음
    """
    sm, tm = english_markers(src), korean_markers(tgt)
    tcats = {m["cat"] for m in tm}
    flags = []
    seen = set()
    for m in sm:
        c = m["cat"]
        if c in seen:
            continue
        seen.add(c)
        if tcats & _COMPATIBLE[c]:
            continue
        conflict = [t for t in tm if t["cat"] in _CONFLICT[c]]
        if conflict:
            # 강한 신호: 원문 표지가 strong이고 번역 충돌 표지도 strong
            strong = m["strength"] == "strong" and any(t["strength"] == "strong" for t in conflict)
            # must not(PROH strong) → 않아도 된다(NONOBL strong) 같은 경우
            flags.append({"type": "flip", "src_cat": c, "tgt_cat": conflict[0]["cat"],
                          "strength": "strong" if strong else "weak",
                          "src_text": m["text"], "tgt_text": conflict[0]["text"]})
        elif not tm and m["strength"] == "strong" and c != PERM:
            flags.append({"type": "vanished", "src_cat": c, "tgt_cat": None, "strength": "strong",
                          "src_text": m["text"], "tgt_text": None})
    return {"src_markers": sm, "tgt_markers": tm, "flags": flags}


def hint_lines(result: dict) -> list[str]:
    lines = []
    for f in result["flags"]:
        if f["type"] == "flip":
            lines.append(
                f"- 의무 범주 대응 확인: 원문 '{f['src_text']}'[{KO_NAME[f['src_cat']]}] → 번역 "
                f"'…{f['tgt_text']}…'[{KO_NAME[f['tgt_cat']]}]으로 범주가 바뀐 것으로 보입니다"
                f"{'' if f['strength'] == 'strong' else ' (자동 추출이라 확신도 낮음)'}. 이 문장이 Deontic 적용 대상이면 "
                f"범주가 바뀌었는지 직접 확인하세요. 원문 표현이 추론·가능성·능력·예상의 뜻이면 무시하세요.")
        else:
            lines.append(
                f"- 의무 표현 확인: 원문의 '{f['src_text']}'[{KO_NAME[f['src_cat']]}]에 대응하는 의무·금지 표현을 "
                f"번역에서 찾지 못했습니다. 이 문장이 Deontic 적용 대상이면 요구가 사실 서술이나 선택 사항으로 "
                f"읽히는지 확인하세요.")
    return lines
