"""gatsby_policy_gpu.py — 모델 없이(--fake) 대상 문장·이어하기·프롬프트 구성 확인."""
import importlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from translation import gatsby_policy_gpu as p

ROOT = Path(__file__).resolve().parents[1]
SENT = ROOT / "data/book/gatsby/sentences.jsonl"


@pytest.fixture
def work(tmp_path):
    if not SENT.exists():
        pytest.skip("개츠비 sentences.jsonl 없음")
    shutil.copy(SENT, tmp_path / "sentences.jsonl")
    shutil.copy(ROOT / "translation/gemma_translate.py", tmp_path / "gemma_translate.py")
    return tmp_path


def _rows(f):
    return [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]


def test_fake_run_and_resume(work):
    assert p.main(["--dir", str(work), "--cond", "policy", "--fake", "--limit", "10"]) == 0
    n1 = len(_rows(work / "out/policy_policy.jsonl"))
    assert n1 > 0
    assert p.main(["--dir", str(work), "--cond", "policy", "--fake"]) == 0
    rows = _rows(work / "out/policy_policy.jsonl")
    assert len(rows) == len({r["id"] for r in rows}) == 945          # 1·3장 638 + 7장 문단 307
    assert rows[0]["mt_meta"]["policy"] is True and rows[0]["mt_meta"]["n_next"] == 0


def test_prompt_parts(work):
    sys.path.insert(0, str(work)); gt = importlib.import_module("gemma_translate")
    allb, idx = gt.blocks(_rows(work / "sentences.jsonl"), None)
    k = p.targets(allb, idx)[40]
    assert p.user_text(gt, allb, k, 5, 0)[1] == gt.user_text(allb, k, n_ctx=5)[1]      # 뒤 문단 0개면 처음 번역과 같은 입력
    t = p.user_text(gt, allb, k, 5, 5)[1]
    assert "[뒤 문단 원문 — 문맥 참고용, 번역하지 마세요]" in t and t.rstrip().endswith(allb[k]["sents"][-1]["en"])
    assert p.prompt_hash(gt, "base") == gt.PROMPT_HASH != p.prompt_hash(gt, "policy")
    assert p.system_text(gt, "policy").endswith(gt.FORMAT)
