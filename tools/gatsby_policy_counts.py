"""번역 쪽 방침 시험: 방침 항목을 지켰는지 코드로 세는 표 (2026-11-28).

    python tools/gatsby_policy_counts.py [--show]

대상은 방침 시험의 945문장(1·3장 + 7장 일부). 비교하는 번역: 기존(책 전체 번역), policy, policy_next5, (있으면) base.
말투 판정은 따옴표 안 문장의 끝말을 보는 어림셈이다. 사람이 읽고 판정한 것이 아니다.
"""
import collections
import json
import re
import sys
from pathlib import Path

D = Path("data/book/gatsby")
rows = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

# ── 방침 9: 고유명사 (원문 패턴, 방침 표기, 다른 표기들)
NAMES = [
    (r"Michaelis", "미카엘리스", ["마이클리스", "마이켈리스", "미켈리스", "마이캘리스"]),
    (r"Wolfshiem", "울프심", ["볼프심", "울프샤임", "볼프샤임", "울프스하임"]),
    (r"Klipspringer", "클립스프링거", ["클립스프링어", "클립스프링저"]),
    (r"[Oo]wl[- ][Ee]yes|owl-eyed", "부엉이", ["올빼미"]),
    (r"Catherine", "캐서린", ["캐더린", "캐서린느", "카트린"]),
    (r"McKee", "맥키", ["매키", "맥케이", "매케이"]),
    (r"Sloane", "슬론", ["슬로언", "슬로안", "슬로운"]),
    (r"Eckleburg", "에클버그", ["에클버그 박사 T", "에클벅", "에클부르크", "에클베르크"]),
    (r"valley of ashes", "재의 계곡", ["잿더미 계곡", "재의 골짜기", "잿더미의 계곡"]),
    (r"\bFinn\b", "핀란드인 가정부", []),
]
PEOPLE = "데이지|조던|베이커|톰|닉|개츠비|윌슨|머틀|캐서린|조지|뷰캐넌|캐러웨이|미카엘리스|마이클리스|맥키"
HON = re.compile(rf"({PEOPLE})\s?(양|씨|부인)(?![가-힣])|({PEOPLE})\s?(양|씨|부인)(?=[은는이가을를의에도과와께])")
POLITE = re.compile(r"(요|니다|니까|시오|시죠|죠|세요|군요|네요|데요)$")
HAGE = re.compile(r"자네|이보게|(하게|보게|주게|말게|하세|이네|하나|인가|겠나|았나|었나|던가|는가|로군|구먼|다네|라네|일세|ㄹ세)$")
QUOTE = re.compile(r"[“\"]([^”\"]*)[”\"]?")
SAY = "said|cried|asked|answered|demanded|remarked|inquired|insisted|whispered|repeated|shouted|exclaimed|suggested|explained|murmured|objected|admitted|broke out|went on|continued|snapped|muttered|called|agreed|added|began|urged|replied|ordered|interrupted|corrected|complained|confessed|announced|sobbed|laughed|nodded|protested"
SPEAKERS = {"개츠비": "Gatsby", "톰": "Tom", "데이지": "Daisy", "조던": "Jordan|Miss Baker", "윌슨": "Wilson", "미카엘리스": "Michaelis", "닉": "I"}


def speaker(en):
    """원문 문장의 따옴표 밖에 '말했다' 꼴로 화자가 적혀 있을 때만 화자를 돌려준다."""
    out = re.sub(r"“[^”]*”?", " ", en)
    for ko, pat in SPEAKERS.items():
        if re.search(rf"\b({SAY})\s+({pat})\b|\b({pat})\s+({SAY})\b", out):
            return ko
    return None


def speech(mt):
    """따옴표 안 문장들의 끝말로 존댓말/반말/하게체를 어림한다. 돌려주는 값: 'polite' | 'casual' | 'mixed' | None."""
    segs = []
    for q in QUOTE.findall(mt):
        for s in re.split(r"[.?!…]+", q):
            s = re.sub(r"[\s,—\-–’'”\")]+$", "", s.strip())
            s = re.sub(r"[,\s]*(친구|톰|닉|데이지|조던|개츠비|조지|윌슨|여보|자기)$", "", s)   # 끝에 붙은 부름말은 뗀다
            if len(s) >= 2 and re.search(r"[가-힣]$", s):
                segs.append(s)
    lab = set()
    for s in segs:
        if POLITE.search(s):
            lab.add("polite")
        elif re.search(r"(다|어|아|야|지|냐|니|게|나|군|네|자|래|걸|까|해|데|고|라|마|워|줘|봐|와|가|서|여|돼|세|구나|거든|잖아|는걸|텐데)$", s):
            lab.add("casual")
    if not lab:
        return None
    return "mixed" if len(lab) == 2 else lab.pop()


def hage(mt):
    for q in QUOTE.findall(mt):
        if re.search(r"자네|이보게", q):
            return True
        for s in re.split(r"[.?!…]+", q):
            s = re.sub(r"[\s,—\-–’'”\")]+$", "", s.strip())
            if re.search(r"(하게|보게|주게|말게|하세|겠나|았나|었나|는가|던가|로군|구먼|다네|라네|일세|겠네|았네|었네|하네|이네|있나|없나|하나|인가|보네|같네|같군|겠군|았군|었군|하군|이군)$", s):
                return True
    return False


def load():
    sents = {s["id"]: s for s in rows(D / "sentences.jsonl")}
    old = {r["id"]: r["mt"] for r in rows(D / "mt/translate_input__gemma_bf16_ctx5_full.jsonl")}
    conds = {"기존": old}
    for c in ("base", "policy", "policy_next5"):
        p = D / f"mt/policy_{c}.jsonl"
        if p.exists():
            conds[c] = {r["id"]: r["mt"] for r in rows(p)}
    ids = sorted(set(conds["policy"]))
    if "base" in conds and set(conds["base"]) != set(ids):
        print(f"(base는 {len(conds['base'])}/{len(ids)}문장만 있어 표에서 뺌)")
        del conds["base"]
    return sents, conds, ids


def main():
    show = "--show" in sys.argv
    sents, conds, ids = load()
    en = {i: sents[i]["en"] for i in ids}
    print(f"대상 {len(ids)}문장 · 조건 {list(conds)}\n")

    print("■ 고유명사 (원문에 그 이름이 있는 문장 수 → 방침 표기 / 다른 표기 / 둘 다 없음)")
    for pat, good, bads in NAMES:
        hit = [i for i in ids if re.search(pat, en[i])]
        if not hit:
            continue
        line = f"  {pat[:22]:22s} {len(hit):3d}문장 "
        for c, mt in conds.items():
            g = sum(1 for i in hit if good in mt[i]); b = sum(1 for i in hit if any(x in mt[i] for x in bads) and good not in mt[i])
            line += f"| {c}: {g}/{b}/{len(hit) - g - b} "
            if show and c != "기존":
                for i in hit:
                    if good not in mt[i]:
                        print("      ", c, i, en[i][:60], "→", mt[i][:60])
        print(line)

    print("\n■ 호칭: 원문에 Miss·Mr.·Mrs.가 없는데 번역에 '이름+양/씨/부인'이 있는 문장")
    for c, mt in conds.items():
        bad = [i for i in ids if HON.search(mt[i]) and not re.search(r"\b(Miss|Mr\.|Mrs\.|Mister)", en[i])]
        print(f"  {c}: {len(bad)}", [(i, HON.search(mt[i]).group(0)) for i in bad][: 12 if show else 6])

    print("\n■ 관계어: 원문 cousin → 번역에 '사촌'")
    hit = [i for i in ids if re.search(r"\bcousin", en[i])]
    for c, mt in conds.items():
        print(f"  {c}: 사촌 {sum(1 for i in hit if '사촌' in mt[i])} / 친척 {sum(1 for i in hit if '친척' in mt[i])} (원문 {len(hit)}문장)")

    print("\n■ old sport (원문에 있는 문장)")
    hit = [i for i in ids if re.search(r"old sport", en[i], re.I)]
    for c, mt in conds.items():
        k = collections.Counter("친구" if "친구" in mt[i] else ("이보게·자네" if re.search("이보게|자네", mt[i]) else "그 밖") for i in hit)
        print(f"  {c}: {dict(k)} (원문 {len(hit)}문장)")
        if show:
            for i in hit:
                if "친구" not in mt[i]:
                    print("      ", i, en[i][:70], "→", mt[i][:70])

    print("\n■ 말투(어림셈): 화자가 문장에 적힌 대사 — 존댓말 / 반말 / 섞임")
    sp = {i: speaker(en[i]) for i in ids}
    for who in SPEAKERS:
        hit = [i for i in ids if sp[i] == who and "“" in en[i]]
        line = f"  {who:6s} {len(hit):3d}문장 "
        for c, mt in conds.items():
            k = collections.Counter(speech(mt[i]) for i in hit)
            line += f"| {c}: {k['polite']}/{k['casual']}/{k['mixed']} "
        print(line)

    print("\n■ 개츠비의 old sport 대사 말투(톰이 흉내 내는 문장 제외) — 존댓말 / 반말 / 섞임 / 판정 못함")
    hit = [i for i in ids if re.search(r"old sport", en[i], re.I) and i not in ("S002076", "S002233")]
    for c, mt in conds.items():
        k = collections.Counter(speech(mt[i]) for i in hit)
        print(f"  {c}: {k['polite']}/{k['casual']}/{k['mixed']}/{k[None]} (원문 {len(hit)}문장)")
        if show and c != "기존":
            for i in hit:
                if speech(mt[i]) != "polite":
                    print("      ", i, mt[i][:80])

    print("\n■ 하게체(자네·이보게, -게/-나/-군/-네 끝말)가 있는 대사 문장")
    for c, mt in conds.items():
        bad = [i for i in ids if hage(mt[i])]
        print(f"  {c}: {len(bad)}", bad[:8] if show else "")


if __name__ == "__main__":
    main()
