"""gatsby_full_gpu.py — 모델 없이(--fake) 이어하기·몫 나누기·합치기 확인."""
import json
import shutil
from pathlib import Path

import pytest

from translation import gatsby_full_gpu as g

ROOT = Path(__file__).resolve().parents[1]
SENT = ROOT / "data/book/gatsby/sentences.jsonl"


def _find_sentences():
    for p in [SENT, *ROOT.glob("data/**/sentences.jsonl")]:
        if p.exists() and sum(1 for _ in open(p, encoding="utf-8")) > 3000:
            return p
    return None


@pytest.fixture
def work(tmp_path):
    src = _find_sentences()
    if src is None:
        pytest.skip("개츠비 sentences.jsonl 없음")
    shutil.copy(src, tmp_path / "sentences.jsonl")
    shutil.copy(ROOT / "translation/gemma_translate.py", tmp_path / "gemma_translate.py")
    return tmp_path


def _rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _prev(work, n=40):
    """1·3장 기존 결과 흉내: I장 앞 문장 n개."""
    import importlib, sys
    sys.path.insert(0, str(work)); gt = importlib.import_module("gemma_translate")
    rows = [r for r in _rows(work / "sentences.jsonl") if r["kind"] == "body" and r["translate"] and r["chapter"] == "I"][:n]
    with open(work / "gemma_ch1_3_bf16_ctx5.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "en": r["en"], "mt": "기존:" + r["id"], "mt_system": "gemma-4-31b-it-bf16-ctx5",
                                "mt_meta": {"config": "ctx5", "prompt_hash": gt.PROMPT_HASH}}, ensure_ascii=False) + "\n")
    return {r["id"] for r in rows}


def test_shards_resume_merge(work, capsys):
    prev_ids = _prev(work)
    d = ["--dir", str(work), "--fake"]
    g.main(d + ["--shard", "0/2", "--limit", "10"])            # 속도 시험분
    n_test = len(_rows(work / "out/part_0of2.jsonl"))
    assert 0 < n_test < 200
    assert "속도 시험" in capsys.readouterr().out
    g.main(d + ["--shard", "0/2"]); g.main(d + ["--shard", "1/2"])
    a, b = _rows(work / "out/part_0of2.jsonl"), _rows(work / "out/part_1of2.jsonl")
    ids = [r["id"] for r in a + b]
    assert len(ids) == len(set(ids)) and not (set(ids) & prev_ids)      # 몫끼리·기존 결과와 겹치지 않음
    assert abs(len(a) - len(b)) < 0.2 * len(a)                             # 두 몫 크기가 비슷
    g.main(d + ["--shard", "0/2"])                                         # 다시 실행해도 더 쓰지 않음
    assert len(_rows(work / "out/part_0of2.jsonl")) == len(a)
    g.main(d + ["--merge"])
    final = _rows(work / "out" / g.FINAL)
    info = json.loads((work / "out/run_info_full_ctx5.json").read_text(encoding="utf-8"))
    assert len(final) == 2954 == info["n_sentences"] and not info["missing"] and not info["empty"]
    assert info["from_prev_ch1_3"] == len(prev_ids)
    assert all(r["mt"] == "기존:" + r["id"] for r in final if r["id"] in prev_ids)   # 기존 1·3장 번역은 그대로
    order = [r["id"] for r in final]
    assert order == sorted(order)                                          # 문장 순서대로


def test_unsharded_after_partial(work):
    d = ["--dir", str(work), "--fake"]
    g.main(d + ["--shard", "1/2"])                # 한 몫만 끝난 상태에서
    g.main(d)                                      # 나누지 않고 실행하면 남은 것만 번역
    g.main(d + ["--merge"])
    info = json.loads((work / "out/run_info_full_ctx5.json").read_text(encoding="utf-8"))
    assert info["n_sentences"] == 2954 and info["duplicates_dropped"] == 0


def test_rejects_wrong_prev(work):
    with open(work / "gemma_ch1_3_bf16_ctx5.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "S000001", "en": "x", "mt": "y", "mt_meta": {"config": "ctx5_think", "prompt_hash": "4f5cbcadc8d7"}}) + "\n")
    with pytest.raises(SystemExit):
        g.main(["--dir", str(work), "--fake"])
