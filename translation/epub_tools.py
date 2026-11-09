"""EPUB 책 → 번역 입력 문장 → (Colab 번역) → 한국어 EPUB 재조립 · 6축 검증용 무작위 표본.

    python translation/epub_tools.py extract BOOK.epub OUT_DIR
        OUT_DIR/sentences.jsonl      모든 문장(원래 위치 포함). 번역 안 할 문장(숫자·기호만)은 translate=false
        OUT_DIR/translate_input.jsonl 번역할 문장만 {id, en} → 드라이브 translate_inputs/ 에 올림
        OUT_DIR/manifest.json        책 정보, 문장 수, 본문 대비 추출 비율, 너무 긴 문장 수, DRM 검사 결과
    python translation/epub_tools.py rebuild BOOK.epub OUT_DIR/sentences.jsonl MT.jsonl OUT.epub
        MT.jsonl: 노트북 출력({id, en, mt})
    python translation/epub_tools.py sample OUT_DIR/sentences.jsonl MT.jsonl ITEMS.jsonl --n 300 --seed 20261109
        검증엔진 입력 형식(파일럿 문항과 같은 필드)으로 무작위 표본

한계
- 문단(블록) 단위로 번역문을 갈아끼우므로 문단 안의 기울임·굵게·링크 같은 서식은 사라진다.
  각주 번호 링크(noteref)만 문단 끝에 다시 붙인다
- 문장 단위 번역(학습 데이터와 같은 단위)이라 앞뒤 문맥은 번역에 쓰이지 않는다
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


def _leaf_blocks(root):
    """블록 요소 중 안에 다른 블록이 없는 것 (div 안에 p가 있으면 p만)."""
    body = next((e for e in root.iter() if _local(e) == "body"), root)
    for el in body.iter():
        if _local(el) not in BLOCK or any(_local(a) in SKIP for a in el.iterancestors()):
            continue
        if any(_local(d) in BLOCK for d in el.iterdescendants()):
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
                                      it.get("media-type", ""), it.get("properties", ""))
    spine = []
    for ir in opf.iter():
        if _local(ir) == "itemref" and ir.get("idref") in manifest:
            href, mt, props = manifest[ir.get("idref")]
            if "html" in mt:
                spine.append({"file": href, "linear": ir.get("linear", "yes"), "nav": "nav" in props})
    meta = {k: (opf.findtext(f".//dc:{k}", namespaces=NS_OPF) or "").strip() for k in ("title", "creator", "language", "publisher")}
    return z, opf_path, spine, meta, drm


def extract(epub: Path, out_dir: Path) -> dict:
    z, _, spine, meta, drm = _open_book(epub)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, docs = [], []
    nb = ns = 0
    body_chars = got_chars = 0
    for d in spine:
        root = etree.fromstring(z.read(d["file"]), PARSER)
        if root is None:
            docs.append({**d, "blocks": 0, "error": "파싱 실패"}); continue
        tree = root.getroottree()
        body = next((e for e in root.iter() if _local(e) == "body"), root)
        body_chars += len(re.sub(r"\s+", "", _text(body)))
        chapter, n_here = "", 0
        for el in _leaf_blocks(root):
            refs = _noterefs(el)
            para = _text(el, refs)
            if not para:
                continue
            got_chars += len(re.sub(r"\s+", "", para))
            if _local(el) in HEADING:
                chapter = para[:120]
            sents = split_sentences(para)
            for k, s in enumerate(sents):
                ns += 1
                rows.append({"id": f"S{ns:06d}", "block": f"B{nb:05d}", "file": d["file"], "xpath": tree.getpath(el),
                             "tag": _local(el), "sent_idx": k, "n_sents": len(sents), "en": s, "paragraph": para,
                             "chapter": chapter, "nav": d["nav"], "n_noterefs": len(refs),
                             "translate": bool(re.search(r"[A-Za-z]{2,}", s)), "long": len(s.split()) > LONG_WORDS})
            nb += 1; n_here += 1
        docs.append({**d, "blocks": n_here})
    sha = hashlib.sha256(epub.read_bytes()).hexdigest()
    tr = [r for r in rows if r["translate"]]
    man = {"source": epub.name, "sha256": sha, "meta": meta, "drm_check": drm, "docs": docs,
           "n_blocks": nb, "n_sentences": len(rows), "n_translate": len(tr),
           "words_translate": sum(len(r["en"].split()) for r in tr),
           "long_sentences": sum(r["long"] for r in tr),
           "coverage_chars": round(got_chars / body_chars, 3) if body_chars else None,
           "note": "coverage_chars가 0.95보다 낮으면 블록 밖 본문(예: div 바로 아래 글자)이 빠졌을 수 있음 → 확인"}
    w = lambda p, rs: p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
    w(out_dir / "sentences.jsonl", rows)
    w(out_dir / "translate_input.jsonl", [{"id": r["id"], "en": r["en"]} for r in tr])
    (out_dir / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


def _load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _block_translations(rows, mt):
    blocks, missing = {}, 0
    for r in rows:
        t = r["en"]
        if r["translate"]:
            if r["id"] in mt and (mt[r["id"]] or "").strip():
                t = mt[r["id"]].strip()
            else:
                missing += 1
        blocks.setdefault((r["file"], r["xpath"]), []).append(t)
    return {k: " ".join(v) for k, v in blocks.items()}, missing


def rebuild(epub: Path, sentences: Path, mt_path: Path, out: Path, lang: str = "ko") -> dict:
    z, opf_path, spine, _, _ = _open_book(epub)
    rows = _load(sentences)
    mt = {r["id"]: r.get("mt") for r in _load(mt_path)}
    blocks, missing = _block_translations(rows, mt)
    by_file = {}
    for (f, xp), t in blocks.items():
        by_file.setdefault(f, {})[xp] = t
    new = {}
    replaced = 0
    for f, xmap in by_file.items():
        root = etree.fromstring(z.read(f), PARSER)
        tree = root.getroottree()
        targets = []
        for xp, t in xmap.items():   # 위치를 먼저 모두 찾고 나서 바꾼다
            hits = tree.xpath(xp)
            if len(hits) != 1:
                raise SystemExit(f"{f} {xp}: 위치를 찾지 못함 (추출한 EPUB과 다른 파일?)")
            targets.append((hits[0], t))
        for el, t in targets:
            refs = [copy.deepcopy(r) for r in _noterefs(el)]
            for r in refs:
                r.tail = None
            for c in list(el):
                el.remove(c)
            el.text = t
            for r in refs:
                el.append(r)
            replaced += 1
        root.set("lang", lang)
        root.set("{http://www.w3.org/XML/1998/namespace}lang", lang)
        new[f] = etree.tostring(tree, xml_declaration=True, encoding="utf-8")
    opf = etree.fromstring(z.read(opf_path), PARSER)
    for el in opf.iter():
        if _local(el) == "language" and etree.QName(el).namespace == NS_OPF["dc"]:
            el.text = lang
    new[opf_path] = etree.tostring(opf.getroottree(), xml_declaration=True, encoding="utf-8")
    with zipfile.ZipFile(out, "w") as zo:
        zo.writestr(zipfile.ZipInfo("mimetype"), z.read("mimetype"), compress_type=zipfile.ZIP_STORED)
        for info in z.infolist():
            if info.filename == "mimetype":
                continue
            zo.writestr(info.filename, new.get(info.filename, z.read(info.filename)), compress_type=zipfile.ZIP_DEFLATED)
    rep = {"out": str(out), "blocks_replaced": replaced, "sentences_missing_mt_kept_english": missing}
    print(json.dumps(rep, ensure_ascii=False))
    return rep


BRIEF = ("독자: 한국어 일반 독자\n목적: 책의 내용과 서술의 기능을 이해\n매체: 출판물 본문\n"
         "관계·상황: 문서의 실제 서술·대화 상황을 기준으로 읽음\n"
         "허용 범위: 자연스러운 한국어 어순·문장 분할 허용. 핵심 내용·의도·문화 함의·관계·구조·규범성 변경 불허")


def sample(sentences: Path, mt_path: Path, out: Path, n: int, seed: int, min_words: int = 3,
           title: str = "", author: str = "", mt_system: str = "madlad10b_lora_v1") -> list[dict]:
    """번역 대상 문장 중 단어 min_words개 이상인 문장에서 무작위 n개. 목차(nav) 문서는 제외."""
    rows = _load(sentences)
    mt = {r["id"]: r.get("mt") for r in _load(mt_path)}
    blocks, _ = _block_translations(rows, mt)
    pool = [r for r in rows if r["translate"] and not r["nav"] and len(r["en"].split()) >= min_words and mt.get(r["id"])]
    pick = sorted(random.Random(seed).sample(pool, min(n, len(pool))), key=lambda r: r["id"])
    items = []
    for k, r in enumerate(pick, 1):
        items.append({"item_id": f"BK{k:03d}", "doc_info": f"자료명: {title}\n저자·기관: {author}\n장르: 출판 콘텐츠\n"
                      f"출처: 출판사 PoC 원고\n장·절: {r['chapter'] or '-'}",
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
    s = sub.add_parser("sample"); s.add_argument("sentences", type=Path); s.add_argument("mt", type=Path)
    s.add_argument("out", type=Path); s.add_argument("--n", type=int, default=300); s.add_argument("--seed", type=int, default=20261109)
    s.add_argument("--title", default=""); s.add_argument("--author", default=""); s.add_argument("--mt-system", default="madlad10b_lora_v1")
    a = p.parse_args(argv)
    if a.cmd == "extract":
        m = extract(a.epub, a.out_dir)
        print(json.dumps({k: m[k] for k in ("meta", "n_blocks", "n_sentences", "n_translate", "words_translate",
                                            "long_sentences", "coverage_chars")}, ensure_ascii=False, indent=1))
    elif a.cmd == "rebuild":
        rebuild(a.epub, a.sentences, a.mt, a.out)
    else:
        sample(a.sentences, a.mt, a.out, a.n, a.seed, title=a.title, author=a.author, mt_system=a.mt_system)


if __name__ == "__main__":
    sys.exit(main())
