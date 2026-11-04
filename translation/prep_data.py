"""AI Hub 한국어-영어 번역 말뭉치(사회과학)에서 영어 원문 교육 문장을 골라 학습·검증·평가 분할을 파일로 저장.

- 학습: Training 압축 파일의 영어 원문(source_language == en) · 교육 분야
- 검증·평가: Validation 압축 파일의 같은 조건 문장에서 무작위로 (서로 겹치지 않게)
- 평가·검증 문장과 영어 원문이 같은 문장은 학습에서 뺀다 (누출 방지)
- 열: en = 영어 원문, ko = 한국어 번역(사람). mt 열은 한→영 기계번역이라 쓰지 않는다
결과: out_dir/{train,valid,test}.jsonl + manifest.json (개수, 시드, 파일 해시)
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import zipfile
from pathlib import Path

HANGUL = re.compile(r"[가-힣]")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def hangul_ratio(s: str) -> float:
    chars = [c for c in s if not c.isspace()]
    return sum(bool(HANGUL.match(c)) for c in chars) / len(chars) if chars else 0.0


def read_zip(zip_path: Path, domain: str = "교육") -> list[dict]:
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with z.open(name) as f:
                for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig")):
                    if r.get("source_language") != "en" or (domain and r.get("domain") != domain):
                        continue
                    en, ko = norm(r.get("en")), norm(r.get("ko"))
                    rows.append({"id": r.get("sn"), "en": en, "ko": ko, "subdomain": r.get("subdomain"),
                                 "file_name": r.get("file_name"), "src_file": name})
    return rows


def keep(r: dict) -> bool:
    en, ko = r["en"], r["ko"]
    return (15 <= len(en) <= 500 and 5 <= len(ko) <= 500 and re.search(r"[A-Za-z]", en) is not None
            and hangul_ratio(ko) >= 0.3)


def dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        k = r["en"].lower()
        if k not in seen:
            seen.add(k); out.append(r)
    return out


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(p: Path, rows: list[dict]) -> None:
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare(train_zip: Path, valid_zip: Path, out_dir: Path, n_valid: int = 1000, n_test: int = 500,
            seed: int = 20261104, domain: str = "교육") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_tr, raw_va = read_zip(train_zip, domain), read_zip(valid_zip, domain)
    tr = dedupe([r for r in raw_tr if keep(r)])
    va = dedupe([r for r in raw_va if keep(r)])
    rng = random.Random(seed)
    va_sorted = sorted(va, key=lambda r: r["en"])  # 읽는 순서와 무관하게 같은 분할이 나오도록
    rng.shuffle(va_sorted)
    test, valid = va_sorted[:n_test], va_sorted[n_test:n_test + n_valid]
    held = {r["en"].lower() for r in test + valid}
    train = [r for r in tr if r["en"].lower() not in held]
    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        write_jsonl(out_dir / f"{name}.jsonl", rows)
    man = {"seed": seed, "domain": domain, "filters": "en 15~500자, ko 5~500자, ko 한글 비율 0.3 이상, 영어 원문 중복 제거",
           "raw": {"train_zip_en": len(raw_tr), "valid_zip_en": len(raw_va)},
           "after_filter": {"train_zip": len(tr), "valid_zip": len(va)},
           "removed_from_train_overlap": len(tr) - len(train),
           "counts": {"train": len(train), "valid": len(valid), "test": len(test)},
           "zip_sha256": {"train": sha256(train_zip), "valid": sha256(valid_zip)},
           "split_sha256": {n: sha256(out_dir / f"{n}.jsonl") for n in ("train", "valid", "test")}}
    (out_dir / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


if __name__ == "__main__":
    import sys
    print(json.dumps(prepare(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])), ensure_ascii=False, indent=1))
