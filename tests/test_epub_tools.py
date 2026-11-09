import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "translation"))
import epub_tools as et  # noqa: E402

CH1 = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="en">
<head><title>Ch 1</title><style>p{}</style></head>
<body><section>
<h1>Chapter One</h1>
<p>Dr. Smith went to Washington, D.C. in 1941. He said it was <i>fine</i>.<a epub:type="noteref" href="#n1">1</a></p>
<div class="box"><p>Inside a box.</p><ul><li>First item here.</li><li>2012</li></ul></div>
<p>   </p>
<pre>code(); stays</pre>
</section></body></html>"""
CH2 = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Ch 2</title></head>
<body><h2>Two</h2><p>Last line&#160;here.<sup><a href="#n2">12</a></sup></p></body></html>"""
OPF = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Test Book</dc:title><dc:creator>A. Author</dc:creator>
<dc:language>en</dc:language><dc:identifier id="id">x</dc:identifier></metadata>
<manifest><item id="c1" href="text/ch%201.xhtml" media-type="application/xhtml+xml"/>
<item id="c2" href="text/ch2.xhtml" media-type="application/xhtml+xml"/></manifest>
<spine><itemref idref="c1"/><itemref idref="c2"/></spine></package>"""
CONT = """<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"""


def make_epub(p: Path, encrypted=False):
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", CONT)
        z.writestr("OEBPS/content.opf", OPF)
        z.writestr("OEBPS/text/ch 1.xhtml", CH1)
        z.writestr("OEBPS/text/ch2.xhtml", CH2)
        if encrypted:
            z.writestr("META-INF/encryption.xml", '<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                       '<EncryptedData xmlns="http://www.w3.org/2001/04/xmlenc#"><EncryptionMethod '
                       'Algorithm="http://www.w3.org/2001/04/xmlenc#aes128-cbc"/></EncryptedData></encryption>')


def test_roundtrip(tmp_path):
    book = tmp_path / "b.epub"; make_epub(book)
    man = et.extract(book, tmp_path / "x")
    rows = [json.loads(l) for l in open(tmp_path / "x" / "sentences.jsonl")]
    ens = [r["en"] for r in rows]
    assert ens[:3] == ["Chapter One", "Dr. Smith went to Washington, D.C. in 1941.", "He said it was fine."]
    assert "Inside a box." in ens and "First item here." in ens
    assert not any("code()" in e for e in ens)                      # pre 제외
    assert [r for r in rows if r["en"] == "2012"][0]["translate"] is False
    assert ens[-1] == "Last line here."                              # 각주 번호(12) 제외, nbsp 정리
    assert man["meta"]["title"] == "Test Book" and man["n_translate"] == len(rows) - 1
    assert rows[1]["chapter"] == "Chapter One" and rows[1]["n_noterefs"] == 1
    mt = tmp_path / "mt.jsonl"
    mt.write_text("".join(json.dumps({"id": r["id"], "en": r["en"], "mt": f"[KO {r['id']}]"}) + "\n" for r in rows if r["translate"]))
    out = tmp_path / "ko.epub"
    rep = et.rebuild(book, tmp_path / "x" / "sentences.jsonl", mt, out)
    assert rep["sentences_missing_mt_kept_english"] == 0
    z = zipfile.ZipFile(out)
    assert z.namelist()[0] == "mimetype" and z.getinfo("mimetype").compress_type == zipfile.ZIP_STORED
    c1 = z.read("OEBPS/text/ch 1.xhtml").decode()
    assert "[KO S000002] [KO S000003]" in c1 and 'noteref' in c1 and "code(); stays" in c1 and "Washington" not in c1
    assert ">2012<" in c1
    c2 = z.read("OEBPS/text/ch2.xhtml").decode()
    assert "<sup><a href=\"#n2\">12</a></sup>" in c2
    assert "<dc:language>ko</dc:language>" in z.read("OEBPS/content.opf").decode()
    items = et.sample(tmp_path / "x" / "sentences.jsonl", mt, tmp_path / "items.jsonl", n=3, seed=1, title="Test Book")
    assert len(items) == 3 and all(len(i["src"].split()) >= 3 for i in items)
    assert all(i["tgt_paragraph"] for i in items)


def test_drm_refused(tmp_path):
    book = tmp_path / "d.epub"; make_epub(book, encrypted=True)
    try:
        et.extract(book, tmp_path / "x")
    except SystemExit as e:
        assert "DRM" in str(e)
    else:
        raise AssertionError("DRM 파일을 통과시킴")


PG_CH = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>G</title></head><body>
<section class="pg-boilerplate pgheader" id="pg-header"><h2>The Project Gutenberg eBook of Test</h2>
<p>This ebook is for the use of anyone anywhere.</p>
<div id="pg-start-separator"><span>*** START OF THE PROJECT GUTENBERG EBOOK TEST ***</span></div></section>
<h1>A Test Novel</h1>
<p class="toc"><a href="#c1">Chapter I. The Start</a></p>
<h2 id="c1">Chapter I. The Start</h2>
<p>It was a dark night. The wind howled.</p>
<div id="pg-end-separator"><span>*** END OF THE PROJECT GUTENBERG EBOOK TEST ***</span></div>
<section class="pg-boilerplate pgfooter" id="pg-footer"><p>Updated editions will replace the previous one.</p></section>
<p>Trailing license line.</p>
</body></html>"""
PG_NAV = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><head><title>toc</title></head><body>
<nav epub:type="toc"><h2>Contents</h2><ol><li><a href="ch.xhtml#c1">Chapter I. The Start</a>
<ol><li><a href="ch.xhtml#c1">A Sub Part</a></li></ol></li></ol></nav></body></html>"""
PG_NCX = """<?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
<docTitle><text>A Test Novel</text></docTitle><navMap><navPoint id="n1" playOrder="1">
<navLabel><text>Chapter I. The Start</text></navLabel><content src="ch.xhtml#c1"/></navPoint></navMap></ncx>"""
PG_OPF = """<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>A Test Novel</dc:title><dc:language>en</dc:language>
<dc:identifier id="id">x</dc:identifier></metadata>
<manifest><item id="ch" href="ch.xhtml" media-type="application/xhtml+xml"/>
<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/></manifest>
<spine toc="ncx"><itemref idref="ch"/></spine></package>"""


def make_pg(p: Path):
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", CONT)
        for n, s in (("content.opf", PG_OPF), ("ch.xhtml", PG_CH), ("nav.xhtml", PG_NAV), ("toc.ncx", PG_NCX)):
            z.writestr(f"OEBPS/{n}", s)


def test_gutenberg_toc_bilingual_export(tmp_path):
    book = tmp_path / "g.epub"; make_pg(book)
    man = et.extract(book, tmp_path / "x")
    rows = [json.loads(l) for l in open(tmp_path / "x" / "sentences.jsonl")]
    tr = [r["en"] for r in rows if r["translate"]]
    assert man["gutenberg_markers"] is True
    assert not any("Gutenberg" in e or "license" in e.lower() or "Updated editions" in e for e in tr)
    assert "It was a dark night." in tr and "A Test Novel" in tr
    assert {r["kind"] for r in rows if r["translate"]} == {"body", "nav", "ncx"}
    assert "A Sub Part" in tr and "Contents" in tr                     # 중첩 목차 항목도
    assert man["translate_by_kind"]["ncx"] == 2                          # docTitle + navLabel
    mt = tmp_path / "mt.jsonl"
    mt.write_text("".join(json.dumps({"id": r["id"], "mt": f"[KO {r['en']}]"}) + "\n" for r in rows if r["translate"]))
    et.rebuild(book, tmp_path / "x" / "sentences.jsonl", mt, tmp_path / "ko.epub")
    z = zipfile.ZipFile(tmp_path / "ko.epub")
    ch = z.read("OEBPS/ch.xhtml").decode()
    assert '<a href="#c1">[KO Chapter I. The Start]</a>' in ch          # 본문 속 목차 링크 유지
    assert "This ebook is for the use of anyone anywhere." in ch and "Trailing license line." in ch
    assert "[KO It was a dark night.] [KO The wind howled.]" in ch
    nav = z.read("OEBPS/nav.xhtml").decode()
    assert 'href="ch.xhtml#c1">[KO A Sub Part]</a>' in nav
    assert "<text>[KO Chapter I. The Start]</text>" in z.read("OEBPS/toc.ncx").decode()
    et.rebuild(book, tmp_path / "x" / "sentences.jsonl", mt, tmp_path / "bi.epub", mode="bilingual")
    bi = zipfile.ZipFile(tmp_path / "bi.epub").read("OEBPS/ch.xhtml").decode()
    assert "It was a dark night. The wind howled.<br/><span class=\"bc-ko\" lang=\"ko\">[KO It was" in bi
    assert "<text>Chapter I. The Start / [KO Chapter I. The Start]</text>" in zipfile.ZipFile(tmp_path / "bi.epub").read("OEBPS/toc.ncx").decode()
    from lxml import etree
    for f in ("ko.epub", "bi.epub"):
        zz = zipfile.ZipFile(tmp_path / f)
        for n in zz.namelist():
            if n.endswith((".xhtml", ".ncx", ".opf")):
                etree.fromstring(zz.read(n))                              # 엄격 파싱 통과
    et.export(tmp_path / "x" / "sentences.jsonl", mt, tmp_path / "rv", base_path=mt)
    para = (tmp_path / "rv" / "paragraphs.csv").read_text(encoding="utf-8-sig")
    assert "Gutenberg" not in para and "학습 전 번역" in para and "It was a dark night." in para
    assert (tmp_path / "rv" / "bilingual.txt").read_text().startswith("A Test Novel\n[KO A Test Novel]")
