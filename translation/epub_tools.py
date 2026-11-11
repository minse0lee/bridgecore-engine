"""EPUB 책 → 번역 입력 문장 → (Colab 04 노트북 번역) → 한국어 EPUB·대역본·대조표 · 6축 검증용 무작위 표본.

    python translation/epub_tools.py extract BOOK.epub OUT_DIR
        OUT_DIR/sentences.jsonl       모든 문장(원래 위치 포함). 번역 안 할 문장은 skip에 이유(pg_boilerplate·no_letters)
        OUT_DIR/translate_input.jsonl 번역할 문장만 {id, en} → 드라이브 bridgecore_mt/book/ 에 올림
        OUT_DIR/manifest.json         책 정보, 문장 수, 본문 대비 추출 비율, 너무 긴 문장 수, DRM 검사 결과
    python translation/epub_tools.py rebuild BOOK.epub OUT_DIR/sentences.jsonl MT.jsonl OUT.epub [--mode ko|bilingual]
        ko: 번역문으로 교체한 한국어 EPUB / bilingual: 문단마다 원문 아래 번역문을 붙인 대역 EPUB
    python translation/epub_tools.py export OUT_DIR/sentences.jsonl MT.jsonl REVIEW_DIR [--base BASE_MT.jsonl]
        paragraphs.csv·sentences.csv(엑셀용 원문·번역 대조), bilingual.txt, ko.txt
    python translation/epub_tools.py sample OUT_DIR/sentences.jsonl MT.jsonl ITEMS.jsonl --n 300 --seed 20261109
        검증엔진 입력 형식(파일럿 문항과 같은 필드)으로 본문 문장 무작위 표본

입력: 목차(EPUB3 nav, EPUB2 toc.ncx)·본문을 한 번에 처리. 구텐베르크 책은 START/END 표시 바깥과
pg-header·pg-footer(라이선스 문구)를 번역하지 않고 원문 그대로 둔다.

한계
- 문단(블록) 단위로 번역문을 갈아끼우므로 문단 안의 기울임·굵게 같은 서식은 사라진다.
  링크 하나로만 된 항목(목차 등)은 링크를 살리고, 각주 번호 링크(noteref)는 문단 끝에 다시 붙인다
- 문장 단위 번역(학습 데이터와 같은 단위)이라 앞뒤 문맥은 번역에 쓰이지 않는다
- 하위 목록을 가진 목록 항목처럼 블록 안에 블록이 섞인 경우 바깥 글자가 빠질 수 있음 → manifest의 coverage_chars로 확인
- DRM(암호화)이 걸린 EPUB은 처리하지 않는다 → 출판사에 DRM 없는 파일을 요청
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import posixpath
import random
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

from lxml import etree

BLOCK = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "dt", "dd", "td", "th",
         "figcaption", "caption", "div", "section", "aside", "header", "footer"}
HEADING = {"h1", "h2", "h3", "h4", "h5", "h6"}
SKIP = {"script", "style", "pre", "code", "math", "svg"}
FONT_OBFUSCATION = {"http://www.idpf.org/2008/embedding", "http://ns.adobe.com/pdf/enc#RC"}
LONG_WORDS = 150   # 이보다 긴 문장은 번역 길이 상한(256 토큰)에 걸려 잘릴 수 있음
NS_OPF = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}
PARSER = etree.XMLParser(recover=True, resolve_entities=False, no_network=True, huge_tree=True)

_seg = None


def split_sentences(text: str) -> list[str]:
    global _seg
    try:
        import pysbd
        _seg = _seg or pysbd.Segmenter(language="en", clean=False)
        out = [s.strip() for s in _seg.segment(text)]
    except ImportError:  # pysbd가 없으면 단순 규칙 (약어에서 잘못 끊길 수 있음)
        out = [s.strip() for s in re.split(r'(?<=[.!?]["”’)\]]?)\s+(?=["“‘(\[]?[A-Z0-9])', text)]
    return [s for s in out if s]


def _local(el) -> str:
    return etree.QName(el).localname.lower() if isinstance(el.tag, str) else ""


def _is_noteref(el) -> bool:
    attrs = {etree.QName(k).localname: v for k, v in el.attrib.items()}
    if "noteref" in attrs.get("type", "") or attrs.get("role") == "doc-noteref":
        return True
    if _local(el) == "sup":   # <sup><a href=...>12</a></sup> 형태의 각주 번호
        t = "".join(el.itertext()).strip()
        return len(t) <= 4 and any(_local(c) == "a" for c in el.iter())
    return False


def _noterefs(block) -> list:
    out = []
    for el in block.iter():
        if el is not block and isinstance(el.tag, str) and _is_noteref(el):
            if not any(el in set(o.iter()) for o in out):
                out.append(el)
    return out


def _text(block, skip_els=()) -> str:
    skip = set()
    for s in skip_els:
        skip.update(s.iter())
    parts = []

    def walk(el):
        if el in skip or _local(el) in SKIP:
            if el.tail:
                parts.append(el.tail)
            return
        if el.text:
            parts.append(el.text)
        for c in el:
            walk(c)
        if el is not block and el.tail:
            parts.append(el.tail)
    walk(block)
    s = re.sub("[\u00ad\u200b\u2060\ufeff]", "", "".join(parts))   # 보이지 않는 문자(소프트 하이픈·단어 결합자 등)
    return re.sub(r"\s+", " ", s).strip()


PG_MARK = re.compile(r"\*{3}\s*(START|END) OF (THE|THIS) PROJECT GUTENBERG", re.I)
PG_ATTR = re.compile(r"pg-?(header|footer|boilerplate)", re.I)


def _in_pg_boilerplate(el) -> bool:
    """구텐베르크 라이선스 머리말·꼬리말(id·class에 pg-header/pg-footer/pg-boilerplate)."""
    return any(PG_ATTR.search(f'{a.get("id", "")} {a.get("class", "")}') for a in [el, *el.iterancestors()])


def _blocks(root):
    """번역 단위 요소.
    - 본문: 블록 요소 중 안에 다른 블록이 없는 것 (div 안에 p가 있으면 p만)
    - 목차(<nav>): 중첩 목록이라 li 대신 각 항목의 링크(a)·span·제목
    - toc.ncx(EPUB2 목차): navLabel·docTitle 안의 text"""
    if _local(root) == "ncx":
        for el in root.iter():
            if _local(el) == "text" and any(_local(a) in ("navlabel", "doctitle") for a in el.iterancestors()):
                yield el
        return
    body = next((e for e in root.iter() if _local(e) == "body"), root)
    for el in body.iter():
        name = _local(el)
        anc = [_local(a) for a in el.iterancestors()]
        if any(a in SKIP for a in anc):
            continue
        if "nav" in anc:
            if (name in ("a", "span") and "a" not in anc and "span" not in anc) or name in HEADING:
                yield el
            continue
        if name not in BLOCK or any(_local(d) in BLOCK | {"nav"} for d in el.iterdescendants()):
            continue
        yield el


def _open_book(epub: Path):
    z = zipfile.ZipFile(epub)
    names = set(z.namelist())
    drm = {"encrypted": [], "rights_xml": "META-INF/rights.xml" in names}
    if "META-INF/encryption.xml" in names:
        enc = etree.fromstring(z.read("META-INF/encryption.xml"), PARSER)
        for em in enc.iter():
            if _local(em) == "encryptionmethod" and em.get("Algorithm") not in FONT_OBFUSCATION:
                drm["encrypted"].append(em.get("Algorithm"))
    if drm["encrypted"] or drm["rights_xml"]:
        raise SystemExit(f"DRM(암호화)이 걸린 EPUB입니다: {drm}. 출판사에 DRM 없는 파일을 요청하세요")
    cont = etree.fromstring(z.read("META-INF/container.xml"), PARSER)
    opf_path = next(e.get("full-path") for e in cont.iter() if _local(e) == "rootfile")
    opf = etree.fromstring(z.read(opf_path), PARSER)
    base = posixpath.dirname(opf_path)
    manifest = {}
    for it in opf.iter():
        if _local(it) == "item":
            manifest[it.get("id")] = (posixpath.normpath(posixpath.join(base, unquote(it.get("href")))),
                                      it.get("media-type", ""), it.get("properties", "") or "")
    docs, seen = [], set()
    for ir in opf.iter():
        if _local(ir) == "itemref" and ir.get("idref") in manifest:
            href, mt, props = manifest[ir.get("idref")]
            if "html" in mt and href not in seen:
                docs.append({"file": href, "kind": "nav" if "nav" in props.split() else "body", "in_spine": True}); seen.add(href)
    for href, mt, props in manifest.values():   # 본문 흐름(spine)에 없는 목차 문서도 번역
        if href in seen:
            continue
        if "nav" in props.split() and "html" in mt:
            docs.append({"file": href, "kind": "nav", "in_spine": False}); seen.add(href)
        elif mt == "application/x-dtbncx+xml":
            docs.append({"file": href, "kind": "ncx", "in_spine": False}); seen.add(href)
    meta = {k: (opf.findtext(f".//dc:{k}", namespaces=NS_OPF) or "").strip() for k in ("title", "creator", "language", "publisher")}
    return z, opf_path, docs, meta, drm


def extract(epub: Path, out_dir: Path) -> dict:
    z, _, docs, meta, drm = _open_book(epub)
    out_dir.mkdir(parents=True, exist_ok=True)
    has_start = any(re.search(rb"\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG", z.read(d["file"]), re.I)
                    for d in docs if d["kind"] == "body")
    state = "pre" if has_start else "body"   # 구텐베르크 START 표시 앞·END 표시 뒤는 라이선스 문구
    rows, out_docs = [], []
    nb = ns = 0
    body_chars = got_chars = 0
    for d in docs:
        root = etree.fromstring(z.read(d["file"]), PARSER)
        if root is None:
            out_docs.append({**d, "blocks": 0, "error": "파싱 실패"}); continue
        tree = root.getroottree()
        if d["kind"] != "ncx":
            body = next((e for e in root.iter() if _local(e) == "body"), root)
            body_chars += len(re.sub(r"\s+", "", _text(body)))
        chapter, n_here = "", 0
        for el in _blocks(root):
            refs = _noterefs(el)
            para = _text(el, refs)
            if not para:
                continue
            if d["kind"] != "ncx":
                got_chars += len(re.sub(r"\s+", "", para))
            skip = None
            if d["kind"] != "body" and re.search(r"PROJECT GUTENBERG", para, re.I):
                skip = "pg_boilerplate"   # 목차 속 라이선스 항목
            if d["kind"] == "body":
                m = PG_MARK.search(para)
                if m:
                    skip = "pg_boilerplate"
                    state = "body" if m.group(1).upper() == "START" else "post"
                elif state != "body" or _in_pg_boilerplate(el):
                    skip = "pg_boilerplate"
                if _local(el) in HEADING and not skip:
                    chapter = para[:120]
            # 제목·링크 하나로 된 항목(목차 줄)은 "Chapter I. The Start"처럼 마침표가 있어도 나누지 않는다
            whole = d["kind"] != "body" or _local(el) in HEADING or _target(el) is not el
            sents = [para] if whole else split_sentences(para)
            for k, s in enumerate(sents):
                ns += 1
                sk = skip or (None if re.search(r"[A-Za-z]{2,}", s) else "no_letters")
                if not sk and re.fullmatch(r"(CHAPTER |BOOK |PART )?[IVXLCDM]+\.?", s.strip()):
                    sk = "numeral"   # 로마 숫자 장 번호(II, XIV.)는 그대로 둔다
                rows.append({"id": f"S{ns:06d}", "block": f"B{nb:05d}", "file": d["file"], "kind": d["kind"],
                             "xpath": tree.getpath(el), "tag": _local(el), "sent_idx": k, "n_sents": len(sents),
                             "en": s, "paragraph": para, "chapter": chapter, "n_noterefs": len(refs),
                             "skip": sk, "translate": sk is None, "long": len(s.split()) > LONG_WORDS})
            nb += 1; n_here += 1
        out_docs.append({**d, "blocks": n_here})
    sha = hashlib.sha256(epub.read_bytes()).hexdigest()
    tr = [r for r in rows if r["translate"]]
    import collections
    man = {"source": epub.name, "sha256": sha, "meta": meta, "drm_check": drm, "docs": out_docs,
           "gutenberg_markers": has_start,
           "n_blocks": nb, "n_sentences": len(rows), "n_translate": len(tr),
           "skipped": dict(collections.Counter(r["skip"] for r in rows if r["skip"])),
           "translate_by_kind": dict(collections.Counter(r["kind"] for r in tr)),
           "words_translate": sum(len(r["en"].split()) for r in tr),
           "long_sentences": sum(r["long"] for r in tr),
           "coverage_chars": round(got_chars / body_chars, 3) if body_chars else None,
           "note": "coverage_chars가 0.95보다 낮으면 블록 밖 본문(예: 하위 목록이 있는 항목의 글자)이 빠졌을 수 있음 → 확인"}
    w = lambda p, rs: p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
    w(out_dir / "sentences.jsonl", rows)
    w(out_dir / "translate_input.jsonl", [{"id": r["id"], "en": r["en"]} for r in tr])
    (out_dir / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


_BYTES = re.compile(r"(?:<0x[0-9A-Fa-f]{2}>)+")


def fix_byte_tokens(s: str) -> str:
    """토크나이저가 드문 글자를 바이트 토큰(<0xEB><0x8A><0xAA>)으로 남긴 것을 원래 글자(늪)로 되돌린다.
    모델 출력이 아니라 디코딩 단계의 문제라 그대로 고친다."""
    return _BYTES.sub(lambda m: bytes(int(h, 16) for h in re.findall(r"<0x([0-9A-Fa-f]{2})>", m.group(0))).decode("utf-8", "replace"), s)


def _mt_map(p) -> dict:
    return {r["id"]: fix_byte_tokens((r.get("mt") or "").strip()) for r in _load(p)} if p else {}


def _blocks_from_rows(rows, mt) -> tuple[list[dict], int]:
    """문장 행을 블록(문단)으로 묶어 원문·번역 문단을 만든다. 번역 없는 문장은 원문 그대로."""
    blocks, missing = {}, 0
    for r in rows:
        t = r["en"]
        if r["translate"]:
            if mt.get(r["id"]):
                t = mt[r["id"]]
            else:
                missing += 1
        b = blocks.setdefault((r["file"], r["xpath"]), {"file": r["file"], "xpath": r["xpath"], "kind": r["kind"],
                                                        "tag": r["tag"], "chapter": r["chapter"], "en": r["paragraph"],
                                                        "skip": r["skip"], "parts": []})
        b["parts"].append(t)
    out = []
    for b in blocks.values():
        b["ko"] = " ".join(b.pop("parts"))
        out.append(b)
    return out, missing


def _target(el):
    """링크 하나로만 된 블록(본문 속 목차 항목 등)은 링크 안의 글자만 바꿔 링크를 살린다."""
    kids = [c for c in el if isinstance(c.tag, str)]
    if (len(kids) == 1 and _local(kids[0]) == "a" and not (el.text or "").strip()
            and not (kids[0].tail or "").strip()):
        return _target(kids[0])
    return el


def _sub(parent, name, **attrs):
    ns = etree.QName(parent).namespace
    return etree.SubElement(parent, f"{{{ns}}}{name}" if ns else name, **attrs)


def _replace(el, t):
    el = _target(el)
    refs = [copy.deepcopy(r) for r in _noterefs(el)]
    for r in refs:
        r.tail = None
    for c in list(el):
        el.remove(c)
    el.text = t
    for r in refs:
        el.append(r)


def _append_ko(el, t):
    if _local(el) == "text":   # toc.ncx는 글자만 가능
        el.text = f"{(el.text or '').strip()} / {t}"
        return
    el = _target(el)
    _sub(el, "br")
    s = _sub(el, "span", **{"class": "bc-ko", "lang": "ko"})
    s.text = t


def rebuild(epub: Path, sentences: Path, mt_path: Path, out: Path, mode: str = "ko") -> dict:
    """mode='ko': 번역문으로 교체 / 'bilingual': 원문 아래에 번역문을 덧붙인 대역본.
    구텐베르크 라이선스 문구·숫자만 있는 블록은 원문 그대로 둔다."""
    assert mode in ("ko", "bilingual")
    z, opf_path, _, _, _ = _open_book(epub)
    blocks, missing = _blocks_from_rows(_load(sentences), _mt_map(mt_path))
    by_file = {}
    for b in blocks:
        by_file.setdefault(b["file"], []).append(b)
    new, changed = {}, 0
    for f, bs in by_file.items():
        root = etree.fromstring(z.read(f), PARSER)
        tree = root.getroottree()
        targets = []
        for b in bs:   # 위치를 먼저 모두 찾고 나서 바꾼다
            hits = tree.xpath(b["xpath"])
            if len(hits) != 1:
                raise SystemExit(f"{f} {b['xpath']}: 위치를 찾지 못함 (추출한 EPUB과 다른 파일?)")
            targets.append((hits[0], b))
        for el, b in targets:
            if b["skip"] == "pg_boilerplate" or b["ko"] == b["en"]:
                continue
            (_replace if mode == "ko" else _append_ko)(el, b["ko"])
            changed += 1
        if mode == "ko" and _local(root) == "html":
            root.set("lang", "ko")
            root.set("{http://www.w3.org/XML/1998/namespace}lang", "ko")
        new[f] = etree.tostring(tree, xml_declaration=True, encoding="utf-8")
    if mode == "ko":
        opf = etree.fromstring(z.read(opf_path), PARSER)
        for el in opf.iter():
            if _local(el) == "language" and etree.QName(el).namespace == NS_OPF["dc"]:
                el.text = "ko"
        new[opf_path] = etree.tostring(opf.getroottree(), xml_declaration=True, encoding="utf-8")
    with zipfile.ZipFile(out, "w") as zo:
        zo.writestr(zipfile.ZipInfo("mimetype"), z.read("mimetype"), compress_type=zipfile.ZIP_STORED)
        for info in z.infolist():
            if info.filename == "mimetype":
                continue
            zo.writestr(info.filename, new.get(info.filename, z.read(info.filename)), compress_type=zipfile.ZIP_DEFLATED)
    rep = {"out": str(out), "mode": mode, "blocks_changed": changed, "sentences_missing_mt_kept_english": missing}
    print(json.dumps(rep, ensure_ascii=False))
    return rep


def export(sentences: Path, mt_path: Path, out_dir: Path, base_path: Path | None = None) -> dict:
    """사람이 보는 원문·번역 대조 파일. 라이선스 문구는 뺀다.
    paragraphs.csv(문단 대조, 엑셀용) · sentences.csv(문장 대조) · bilingual.txt(원문/번역 번갈아) · ko.txt(번역만)"""
    import csv
    rows = _load(sentences)
    mt, base = _mt_map(mt_path), _mt_map(base_path)
    blocks, missing = _blocks_from_rows(rows, mt)
    bblocks = {(b["file"], b["xpath"]): b["ko"] for b in _blocks_from_rows(rows, base)[0]} if base else {}
    keep = [b for b in blocks if b["skip"] != "pg_boilerplate"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "paragraphs.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["순번", "구분", "장", "원문", "번역"] + (["학습 전 번역"] if base else []))
        for k, b in enumerate(keep, 1):
            w.writerow([k, {"body": "본문", "nav": "목차", "ncx": "목차(ncx)"}[b["kind"]], b["chapter"], b["en"], b["ko"]]
                       + ([bblocks[(b["file"], b["xpath"])]] if base else []))
    with open(out_dir / "sentences.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "구분", "장", "원문 문장", "번역"] + (["학습 전 번역"] if base else []))
        for r in rows:
            if r["translate"]:
                w.writerow([r["id"], r["kind"], r["chapter"], r["en"], mt.get(r["id"], "")] + ([base.get(r["id"], "")] if base else []))
    body = [b for b in keep if b["kind"] == "body"]
    (out_dir / "bilingual.txt").write_text("\n\n".join(f"{b['en']}\n{b['ko']}" for b in body) + "\n", encoding="utf-8")
    (out_dir / "ko.txt").write_text("\n\n".join(b["ko"] for b in body) + "\n", encoding="utf-8")
    rep = {"out_dir": str(out_dir), "paragraphs": len(keep), "sentences_missing_mt": missing}
    print(json.dumps(rep, ensure_ascii=False))
    return rep


BRIEF = ("독자: 한국어 일반 독자\n목적: 책의 내용과 서술의 기능을 이해\n매체: 출판물 본문\n"
         "관계·상황: 문서의 실제 서술·대화 상황을 기준으로 읽음\n"
         "허용 범위: 자연스러운 한국어 어순·문장 분할 허용. 핵심 내용·의도·문화 함의·관계·구조·규범성 변경 불허")


def sample(sentences: Path, mt_path: Path, out: Path, n: int, seed: int, min_words: int = 3,
           title: str = "", author: str = "", mt_system: str = "madlad10b_lora_v1",
           chapters: list[str] | None = None, prefix: str = "BK", source: str = "출판사 PoC 원고") -> list[dict]:
    """번역 대상 문장 중 단어 min_words개 이상인 문장에서 무작위 n개(n<=0이면 전부). 목차(nav) 문서는 제외.
    chapters를 주면 그 장(가장 가까운 앞 제목)의 문장만."""
    rows = _load(sentences)
    mt = _mt_map(mt_path)
    blocks = {(b["file"], b["xpath"]): b["ko"] for b in _blocks_from_rows(rows, mt)[0]}
    pool = [r for r in rows if r["translate"] and r["kind"] == "body" and len(r["en"].split()) >= min_words and mt.get(r["id"])]
    if chapters:
        pool = [r for r in pool if r["chapter"] in chapters]
    pick = pool if n <= 0 else sorted(random.Random(seed).sample(pool, min(n, len(pool))), key=lambda r: r["id"])
    items = []
    for k, r in enumerate(pick, 1):
        items.append({"item_id": f"{prefix}{k:04d}", "doc_info": f"자료명: {title}\n저자·기관: {author}\n장르: 출판 콘텐츠\n"
                      f"출처: {source}\n장·절: {r['chapter'] or '-'}",
                      "brief": BRIEF, "src_paragraph": r["paragraph"], "tgt_paragraph": blocks[(r["file"], r["xpath"])],
                      "src": r["en"], "tgt": mt[r["id"]], "genre": "출판 콘텐츠", "title": title, "source_url": None,
                      "has_figure": False, "figure": None, "mt_system": mt_system, "book_sentence_id": r["id"]})
    out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    print(f"표본 {len(items)}개 (후보 {len(pool)}문장, seed {seed}) → {out}")
    return items


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("extract"); s.add_argument("epub", type=Path); s.add_argument("out_dir", type=Path)
    s = sub.add_parser("rebuild"); s.add_argument("epub", type=Path); s.add_argument("sentences", type=Path)
    s.add_argument("mt", type=Path); s.add_argument("out", type=Path)
    s.add_argument("--mode", choices=["ko", "bilingual"], default="ko")
    s = sub.add_parser("export"); s.add_argument("sentences", type=Path); s.add_argument("mt", type=Path)
    s.add_argument("out_dir", type=Path); s.add_argument("--base", type=Path, help="학습 전 번역(대조 열 추가)")
    s = sub.add_parser("sample"); s.add_argument("sentences", type=Path); s.add_argument("mt", type=Path)
    s.add_argument("out", type=Path); s.add_argument("--n", type=int, default=300); s.add_argument("--seed", type=int, default=20261109)
    s.add_argument("--title", default=""); s.add_argument("--author", default=""); s.add_argument("--mt-system", default="madlad10b_lora_v1")
    s.add_argument("--chapters", nargs="*"); s.add_argument("--prefix", default="BK"); s.add_argument("--source", default="출판사 PoC 원고")
    a = p.parse_args(argv)
    if a.cmd == "extract":
        m = extract(a.epub, a.out_dir)
        print(json.dumps({k: m[k] for k in ("meta", "gutenberg_markers", "n_blocks", "n_sentences", "n_translate", "skipped",
                                            "translate_by_kind", "words_translate", "long_sentences", "coverage_chars")},
                         ensure_ascii=False, indent=1))
    elif a.cmd == "rebuild":
        rebuild(a.epub, a.sentences, a.mt, a.out, a.mode)
    elif a.cmd == "export":
        export(a.sentences, a.mt, a.out_dir, a.base)
    else:
        sample(a.sentences, a.mt, a.out, a.n, a.seed, title=a.title, author=a.author, mt_system=a.mt_system,
               chapters=a.chapters, prefix=a.prefix, source=a.source)


if __name__ == "__main__":
    sys.exit(main())
