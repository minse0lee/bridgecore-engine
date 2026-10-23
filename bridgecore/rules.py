"""보조 규칙 (PROJECT_NOTES 3-2). 판정자가 아니라 LLM에 주는 힌트 + 판정 전 필터.

- number_check: 원문·번역문의 수치 집합 비교 → 불일치면 "숫자 확인 필요" 힌트
  영어 숫자 단어(thirty, three million, first), 월 이름(June→6), dozen·fortnight,
  한국어 수사(세 명, 서른, 삼백만, 300만)를 정규화한다.
- hangul_ratio / target_language_missing: 가이드라인 6절 '목표언어 미생성' 필터
- deontic_keywords: 규범 표현 존재 힌트 (판정용 아님 — 파일럿에서 N/A 판정과 잘 맞지 않음)
"""
from __future__ import annotations

import re

# ---------- 영어 ----------
_EN_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_EN_TENS = {w: 10 * i for i, w in enumerate(
    "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_EN_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
# first·second는 "at first", 시간 단위 second 등 비수사 용법이 많아 제외
_EN_ORD = {"third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
           "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "twentieth": 20,
           "thirtieth": 30, "hundredth": 100}
_EN_MISC = {"dozen": 12, "fortnight": 14}
_MONTHS = {m: i for i, m in enumerate(
    "_ january february march april may june july august september october november december".split())}
# 'may' 'march'는 동사·이름일 때가 많아 날짜 문맥(뒤에 숫자)일 때만 월로 본다
_AMBIG_MONTHS = {"may", "march"}

_NUM_RE = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _to_num(s: str) -> float:
    v = float(s.replace(",", ""))
    return int(v) if v.is_integer() else v


def english_numbers(text: str) -> list:
    text = text or ""
    # 학년도 표기 "2012–13" → "2012–2013" (번역은 보통 네 자리로 씀)
    text = re.sub(r"\b((\d{2})\d{2})\s?[–—-]\s?(\d{2})\b", r"\1–\2\3", text)
    # "half an hour" → 30분
    text = re.sub(r"\bhalf(?: an)? hour\b", "30 minutes", text, flags=re.I)
    nums = [_to_num(m.group()) for m in _NUM_RE.finditer(text)]
    # "3 million" 같은 숫자+규모어 → 곱하기
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s+(hundred|thousand|million|billion)\b", text, re.I):
        nums.remove(_to_num(m.group(1)))
        nums.append(_to_num(str(float(m.group(1)) * _EN_SCALES[m.group(2).lower()])))
    tokens = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?|\d+", text)
    i = 0
    while i < len(tokens):
        t = tokens[i].lower()
        if t in _MONTHS and t != "_":
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            if t not in _AMBIG_MONTHS or nxt.isdigit():
                nums.append(_MONTHS[t])
            i += 1
            continue
        if t in _EN_ORD:
            nums.append(_EN_ORD[t]); i += 1; continue
        if t in _EN_MISC:
            nums.append(_EN_MISC[t]); i += 1; continue
        # 단어 숫자 구의 끝 위치 찾기 (e.g. "three hundred and twenty-five", "a hundred")
        used, j = False, i
        while j < len(tokens):
            w = tokens[j].lower()
            if all(p in _EN_UNITS or p in _EN_TENS for p in w.split("-")):
                used = True
            elif w in _EN_SCALES and (used or (j > 0 and tokens[j - 1].lower() == "a")):
                used = True
            # "and"는 규모어 뒤에서만 잇는다: "a hundred and twenty" (O), "fifty and eighty" (X)
            elif not (w == "and" and used and tokens[j - 1].lower() in _EN_SCALES
                      and j + 1 < len(tokens)
                      and all(p in _EN_UNITS or p in _EN_TENS for p in tokens[j + 1].lower().split("-"))):
                break
            j += 1
        if used:
            v = _parse_en_phrase(tokens[i:j], tokens[i - 1].lower() if i else "")
            if v != 1:  # 단독 "one"은 대명사(no one, one day)가 많아 제외
                nums.append(v)
            i = j
        else:
            i += 1
    return sorted(nums, key=float)


def _parse_en_phrase(words: list[str], prev: str):
    total, cur = 0, 0
    for w in (x.lower() for x in words):
        if w == "and":
            continue
        if w in _EN_SCALES:
            cur = max(cur, 1) if (cur or prev == "a") else 1
            if w == "hundred":
                cur *= 100
            else:
                total += cur * _EN_SCALES[w]
                cur = 0
        else:
            cur += sum(_EN_UNITS.get(p, 0) + _EN_TENS.get(p, 0) for p in w.split("-"))
    return total + cur


# ---------- 한국어 ----------
_KO_DIGIT = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
_KO_SMALL = {"십": 10, "백": 100, "천": 1000}
_KO_BIG = {"만": 10_000, "억": 100_000_000}
_NATIVE = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "석": 3, "네": 4, "넷": 4,
           "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10, "스무": 20, "스물": 20,
           "서른": 30, "마흔": 40, "쉰": 50, "예순": 60, "일흔": 70, "여든": 80, "아흔": 90}
_COUNTERS = ("명|개|살|번|마리|권|장|시간|시|분|초|달|개월|주|주일|해|년|일|가지|사람|채|대|병|잔|"
             "그릇|켤레|척|군데|곳|차례|배|층|편|통|쪽|페이지|학기|과목|달러|센트|파운드|피트|미터|"
             "마일|킬로미터|퍼센트|%|세기|세|학년|등|위|째|가구|명의|원")
_NATIVE_RE = re.compile(
    r"(?<![가-힣])((?:스물|스무|서른|마흔|쉰|예순|일흔|여든|아흔|열)?\s?"
    r"(?:하나|한|둘|두|셋|세|석|넷|네|다섯|여섯|일곱|여덟|아홉)?)\s?(?=" + _COUNTERS + r")")
_SINO_RE = re.compile(r"(?<![가-힣])([일이삼사오육칠팔구십백천만억]{2,})(?=\s?(?:" + _COUNTERS + r")|\s|$)")
_DIGIT_BIG_RE = re.compile(r"(\d+(?:\.\d+)?)\s?(천|만|억)(?:\s?(\d+(?:,\d{3})*))?")


def _sino(s: str):
    if not re.search(r"[십백천만억]", s):
        return None
    total, section, cur = 0, 0, 0
    for ch in s:
        if ch in _KO_DIGIT:
            cur = _KO_DIGIT[ch]
        elif ch in _KO_SMALL:
            section += (cur or 1) * _KO_SMALL[ch]; cur = 0
        elif ch in _KO_BIG:
            total += (section + cur or 1) * _KO_BIG[ch]; section = cur = 0
    return total + section + cur


def korean_numbers(text: str) -> list:
    text = text or ""
    nums = []
    consumed = text
    for m in _DIGIT_BIG_RE.finditer(text):
        v = float(m.group(1)) * {"천": 1000, "만": 10_000, "억": 100_000_000}[m.group(2)]
        if m.group(3):
            v += float(m.group(3).replace(",", ""))
        nums.append(_to_num(str(v)))
        consumed = consumed.replace(m.group(0), " " * len(m.group(0)), 1)
    nums += [_to_num(m.group()) for m in _NUM_RE.finditer(consumed)]
    for m in _SINO_RE.finditer(consumed):
        v = _sino(m.group(1))
        if v:
            nums.append(v)
    for m in _NATIVE_RE.finditer(consumed):
        phrase = m.group(1).replace(" ", "")
        if not phrase:
            continue
        # 한 음절 수사(한·두·세·네·석)는 뒤에 띄어쓰기가 있을 때만 수사로 본다: "세척"(X) "세 명"(O)
        if phrase in ("한", "두", "세", "네", "석") and not m.group(0).endswith(" "):
            continue
        v, rest = 0, phrase
        for w in sorted(_NATIVE, key=len, reverse=True):
            if rest.startswith(w):
                v += _NATIVE[w]; rest = rest[len(w):]
        for w in sorted(_NATIVE, key=len, reverse=True):
            if rest.startswith(w):
                v += _NATIVE[w]; rest = rest[len(w):]
        if v and not rest and v != 1:  # 단독 "한/하나"는 관형 용법이 많아 제외
            nums.append(v)
    return sorted(nums, key=float)


def number_check(src: str, tgt: str) -> dict:
    s, t = english_numbers(src), korean_numbers(tgt)
    ss, ts = {float(x) for x in s}, {float(x) for x in t}
    return {"src": s, "tgt": t,
            "only_src": sorted(ss - ts), "only_tgt": sorted(ts - ss),
            "mismatch": ss != ts}


# ---------- 목표언어 ----------
def hangul_ratio(text: str) -> float:
    text = text or ""
    hangul = len(re.findall(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return hangul / (hangul + latin) if hangul + latin else 0.0


def target_language_missing(tgt: str, threshold: float = 0.3) -> bool:
    return not (tgt or "").strip() or hangul_ratio(tgt) < threshold


# ---------- 규범 표현 ----------
_DEONTIC_TERMS = [
    "must not", "mustn't", "must", "shall not", "shall", "should not", "shouldn't", "should",
    "ought to", "may not", "may", "cannot", "can't", "can", "needs to", "need to", "have to", "has to",
    "required", "require", "requires", "permitted", "allowed", "prohibited", "forbidden",
    "eligible", "entitled", r"obligat\w+", r"mandat\w+", "do not", "don't", "never"]
# 긴 표현부터 맞춰야 "must not"이 "must"로 잘리지 않는다
_DEONTIC_RE = re.compile(r"\b(" + "|".join(sorted(_DEONTIC_TERMS, key=len, reverse=True)) + r")\b", re.I)


def deontic_keywords(src: str) -> list[str]:
    return sorted({m.group(0).lower() for m in _DEONTIC_RE.finditer(src or "")})


def run_rules(item: dict) -> dict:
    return {
        "number_check": number_check(item.get("src"), item.get("tgt")),
        "hangul_ratio": hangul_ratio(item.get("tgt")),
        "target_language_missing": target_language_missing(item.get("tgt")),
        "deontic_keywords": deontic_keywords(item.get("src")),
    }


def hints_text(rules: dict) -> str | None:
    """프롬프트에 넣을 힌트 문장. 판정을 지시하지 않고 확인할 곳만 알려준다."""
    lines = []
    nc = rules["number_check"]
    if nc["mismatch"]:
        fmt = lambda xs: ", ".join(f"{x:g}" for x in xs) or "없음"
        lines.append(f"- 숫자 확인 필요: 원문에만 있는 수치 [{fmt(nc['only_src'])}], "
                     f"번역문에만 있는 수치 [{fmt(nc['only_tgt'])}]. 자동 추출이라 단어·날짜 표기 차이로 "
                     f"생긴 오경보일 수 있으니 실제로 수량·시간·단위가 달라졌는지 직접 확인하세요.")
    if rules["deontic_keywords"]:
        lines.append(f"- 규범 표현 후보: {', '.join(rules['deontic_keywords'])}. 추론·가능성·능력의 뜻이면 "
                     f"규범 의미가 아닙니다(가이드라인 5절). 후보가 있다고 Deontic을 적용하거나, 없다고 "
                     f"N/A로 정하지 마세요.")
    return "\n".join(lines) or None
