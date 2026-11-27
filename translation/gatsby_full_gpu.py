"""개츠비 책 전체 번역 — Gemma 4 31B(학습 없음, bf16 원본)를 Colab 밖의 GPU 머신에서 돌리는 스크립트.

노트북 10(Colab)과 같은 설정이다: 앞 문단 5개 문맥 · 생각 모드 끔 · greedy · 같은 지침(gemma_translate.py).
번역 함수는 06·10 노트북 셀을 그대로 옮겼다(설정이 같아야 11/17 1·3장 결과와 이어진다).

같은 폴더에 gemma_translate.py, sentences.jsonl, (있으면) gemma_ch1_3_bf16_ctx5.jsonl 을 둔다.

    python gatsby_full_gpu.py --download-only            # 모델만 내려받기(HF 토큰을 한 번 물어봄, 저장하지 않음)
    python gatsby_full_gpu.py --check-prev 20 --limit 24 # 시험: 1·3장 20문단을 다시 번역해 11/17 결과와 비교 + 24문단 속도 재기
    python gatsby_full_gpu.py                            # 책 전체(끊기면 같은 명령으로 다시 실행 → 이어서)
    python gatsby_full_gpu.py --merge                    # 결과 합치기 → out/gemma_full_bf16_ctx5.jsonl

GPU가 4장(48GB급)이면 2장씩 묶어 모델 두 개를 동시에 돌릴 수 있다(터미널 두 개):
    CUDA_VISIBLE_DEVICES=0,1 python gatsby_full_gpu.py --shard 0/2
    CUDA_VISIBLE_DEVICES=2,3 python gatsby_full_gpu.py --shard 1/2

- GPU 한 장이 70GB 이상이면 그 한 장에, 아니면 보이는 GPU 여러 장에 나눠 올린다(합계 80GB 이상 필요). 4비트 압축은 하지 않는다.
- 결과는 out/part_*.jsonl 에 문단 묶음마다 바로 저장한다. 이미 번역된 문장(1·3장 파일 포함)은 건너뛴다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import sys
import time
from getpass import getpass
from pathlib import Path

MODEL_ID = "google/gemma-4-31B-it"
NAME = "full_ctx5"
CFG = dict(n_ctx=5, thinking=False)
THINK_BUDGET = 4096
PROMPT_HASH = "4f5cbcadc8d7"
SAMPLE = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=64)   # 공식 권장값 — greedy가 실패한 문단 재시도에만 씀
FINAL = "gemma_full_bf16_ctx5.jsonl"


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_prev(path, gt):
    """11/17 1·3장 결과(ctx5, 같은 지침)만 받아들인다."""
    if not path or not Path(path).exists():
        return []
    rows = read_jsonl(path)
    bad = [r["id"] for r in rows if r["mt_meta"].get("config") not in ("ctx5", NAME) or r["mt_meta"].get("prompt_hash") != gt.PROMPT_HASH]
    if bad:
        sys.exit(f"{path}: ctx5 결과가 아닌 줄이 {len(bad)}개 있습니다(예: {bad[:3]}) — 11/17 1·3장 파일이 맞는지 확인하세요")
    return rows


def part_files(out_dir):
    return sorted(glob.glob(str(out_dir / "part_*.jsonl")))


def plan(gt, allb, idx, done, shard, limit):
    """번역할 문단 순서. 길이순으로 정렬해 비슷한 길이끼리 묶고, shard는 '전체' 순서에서 나눠 다시 실행해도 몫이 바뀌지 않게 한다."""
    static = sorted(idx, key=lambda k: len(gt.user_text(allb, k, n_ctx=CFG["n_ctx"])[1]))
    i, n = shard
    mine = static[i::n]
    todo = [k for k in mine if any(s["translate"] and s["id"] not in done for s in allb[k]["sents"])]
    if limit and len(todo) > limit:
        todo = todo[::len(todo) // limit][:limit]     # 짧은 문단부터 긴 문단까지 고르게 뽑음
    return todo


# ---------- 모델 ----------

def get_token():
    tok = os.environ.get("HF_TOKEN")
    return tok or getpass("Hugging Face 토큰(화면에 안 보임, 저장하지 않음): ").strip()


def local_snapshot():
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(MODEL_ID, local_files_only=True)
    except Exception:
        return None


def download():
    from huggingface_hub import snapshot_download
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache).free / 1e9
    print(f"모델 저장 위치 {cache} · 남은 공간 {free:.0f}GB")
    if free < 80:
        sys.exit("남은 공간이 80GB보다 작습니다. 큰 디스크 쪽으로 HF_HOME을 지정하세요(예: export HF_HOME=/workspace/hf)")
    t0 = time.time()
    path = snapshot_download(MODEL_ID, token=get_token())
    print(f"내려받기 완료 {(time.time() - t0) / 60:.1f}분 → {path}")
    return path


class Gemma:
    def __init__(self, gt, allb, bs):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.torch, self.gt, self.allb = torch, gt, allb
        n = torch.cuda.device_count()
        assert n >= 1, "GPU가 보이지 않습니다"
        gib = [torch.cuda.get_device_properties(i).total_memory / 2**30 for i in range(n)]
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        print("GPU:", ", ".join(f"{nm} {g:.0f}GiB" for nm, g in zip(names, gib)))
        if gib[0] * 2**30 / 1e9 > 70:
            dmap, mm, self.layout = "cuda", None, "single"
        else:
            # 여러 장에 나눠 올림. 문장 생성 때 쓸 여유를 장마다 남겨 둔다(입력이 들어가는 0번은 더 많이)
            mm = {i: f"{int(g * (0.70 if i == 0 else 0.85))}GiB" for i, g in enumerate(gib)}
            assert sum(int(v[:-3]) for v in mm.values()) > 62, \
                f"보이는 GPU 메모리 합계가 모자랍니다({sum(gib):.0f}GiB) — 원본 모델은 약 62GB가 필요합니다"
            dmap, self.layout = "auto", f"split_{n}gpu"
        path = local_snapshot() or download()
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(path)
        self.processor.tokenizer.padding_side = "left"
        kw = dict(dtype=torch.bfloat16, device_map=dmap)
        if mm:
            kw["max_memory"] = mm
        self.model = AutoModelForCausalLM.from_pretrained(path, **kw).eval()
        self.load_sec = time.time() - t0
        used = [torch.cuda.memory_allocated(i) / 1e9 for i in range(n)]
        print(f"불러오기 {self.load_sec / 60:.1f}분 · GPU 사용 " + " / ".join(f"{u:.1f}GB" for u in used))
        if dmap == "auto":
            where = sorted({str(v) for v in self.model.hf_device_map.values()})
            assert all(w.isdigit() for w in where), f"모델 일부가 GPU 밖({where})에 올라갔습니다 — 메모리가 모자랍니다"
        self.bs = bs
        self.meta = {"model": MODEL_ID, "runtime": f"gpu_script_transformers_bf16_{self.layout}", "prompt_hash": gt.PROMPT_HASH,
                     "config": NAME, "n_ctx": CFG["n_ctx"], "thinking": CFG["thinking"], "decoding": "greedy",
                     "transformers": transformers.__version__, "torch": torch.__version__, "gpu": names[0], "n_gpu": n}

    def prompt_text(self, k):
        sents, user = self.gt.user_text(self.allb, k, n_ctx=CFG["n_ctx"])
        msgs = [{"role": "system", "content": self.gt.SYSTEM + self.gt.FORMAT}, {"role": "user", "content": user}]
        return sents, self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                         enable_thinking=CFG["thinking"])

    def split_answer(self, ids):
        raw = self.processor.decode(ids, skip_special_tokens=False)
        try:
            r = self.processor.parse_response(raw)
            return raw, r.get("content") or "", r.get("thinking") or ""
        except Exception:
            return raw, self.processor.decode(ids, skip_special_tokens=True), ""

    def generate(self, ks, sample=False):
        torch = self.torch
        with torch.inference_mode():
            packs = [self.prompt_text(k) for k in ks]
            enc = self.processor(text=[p[1] for p in packs], return_tensors="pt", padding=True).to(self.model.device)
            n_in = enc["input_ids"].shape[1]
            need = max(len(p[0]) for p in packs) * 200 + 300 + (THINK_BUDGET if CFG["thinking"] else 0)
            out = self.model.generate(**enc, max_new_tokens=need, **(SAMPLE if sample else dict(do_sample=False)))
            res = []
            for (sents, _), ids in zip(packs, out[:, n_in:]):
                raw, content, think = self.split_answer(ids)
                n_tok = int((ids != self.processor.tokenizer.pad_token_id).sum())
                res.append(_check(self.gt, sents, content, raw, len(think), n_tok))
            return res

    def is_oom(self, e):
        return isinstance(e, self.torch.OutOfMemoryError)

    def free(self):
        self.torch.cuda.empty_cache()


def _check(gt, sents, content, raw, think_chars, n_tok):
    try:
        got = gt.parse(content)
        miss = [s["id"] for s in sents if not got.get(s["id"])]
        err = f"id 누락 {miss}" if miss else None
    except Exception as e:
        got, err = {}, f"파싱 실패 {e.__class__.__name__}"
    return dict(sents=sents, got=got, err=err, raw=raw, think_chars=think_chars, out_tokens=n_tok)


class Fake:
    """모델 없이 저장·이어하기·나누기·합치기만 확인하는 시험용(tests에서 씀)."""
    def __init__(self, gt, allb, bs):
        self.gt, self.allb, self.bs, self.load_sec = gt, allb, bs, 0.0
        self.meta = {"model": MODEL_ID, "runtime": "fake", "prompt_hash": gt.PROMPT_HASH, "config": NAME, "n_ctx": CFG["n_ctx"],
                     "thinking": False, "decoding": "greedy", "transformers": "-", "torch": "-", "gpu": "-", "n_gpu": 0}

    def generate(self, ks, sample=False):
        res = []
        for k in ks:
            sents, _ = self.gt.user_text(self.allb, k, n_ctx=CFG["n_ctx"])
            content = json.dumps({"translations": [{"id": s["id"], "ko": "번역:" + s["en"]} for s in sents]}, ensure_ascii=False)
            res.append(_check(self.gt, sents, content, content, 0, 10))
        return res

    def is_oom(self, e):
        return False

    def free(self):
        pass


# ---------- 실행 ----------

def run(m, gt, allb, order, done, out_path, out_dir, tag, want):
    meta = m.meta
    stats = {"out_tokens": [], "retried_with_sampling": [], "fails": {}}
    raw_log = open(out_dir / f"raw_fail_{tag}.jsonl", "a", encoding="utf-8")
    n_sent = 0

    def save(r, sampled=False):
        nonlocal n_sent
        with open(out_path, "a", encoding="utf-8") as f:
            for s in r["sents"]:
                if s["id"] in done:      # 이미 있는 문장은 다시 쓰지 않음(중복 방지)
                    continue
                f.write(json.dumps({"id": s["id"], "en": s["en"], "mt": r["got"][s["id"]],
                                    "mt_system": f"gemma-4-31b-it-bf16-{NAME}",
                                    "mt_meta": {**meta, "decoding": "sampling_retry" if sampled else "greedy"}},
                                   ensure_ascii=False) + "\n")
                done.add(s["id"]); n_sent += 1

    t0, bs, i, fails, nb = time.time(), m.bs, 0, [], 0
    print(f"[{tag}] 문단 {len(order)}개 시작 (한 번에 {bs}문단)")
    while i < len(order):
        ks = order[i:i + bs]
        try:
            res = m.generate(ks)
        except Exception as e:
            if not m.is_oom(e):
                raise
            m.free()
            if bs == 1:
                raise
            bs = max(1, bs // 2); print("  메모리 부족 → 묶음", bs); continue
        for k, r in zip(ks, res):
            stats["out_tokens"].append(r["out_tokens"])
            if r["err"]:
                fails.append(k); raw_log.write(json.dumps({"k": k, "err": r["err"], "raw": r["raw"]}, ensure_ascii=False) + "\n")
            else:
                save(r)
        i += len(ks); nb += 1
        if nb % 5 == 0 or i >= len(order):
            el = time.time() - t0
            print(f"  {i}/{len(order)} 문단 · {el / 60:.1f}분 · 남은 예상 {el / i * (len(order) - i) / 60:.0f}분 · 실패 {len(fails)}", flush=True)
    for k in fails:                      # greedy로 실패한 문단: 하나씩, 공식 권장 샘플링으로 한 번 더
        r = m.generate([k], sample=True)[0]
        if r["err"]:
            stats["fails"][str(k)] = r["err"]
            raw_log.write(json.dumps({"k": k, "err": r["err"], "raw": r["raw"], "sampling": True}, ensure_ascii=False) + "\n")
        else:
            save(r, sampled=True); stats["retried_with_sampling"].append(k)
    raw_log.close()
    gen_sec = time.time() - t0
    info = {**meta, "part": tag, "gen_sec": round(gen_sec), "n_sentences_this_run": n_sent,
            "retried_with_sampling": stats["retried_with_sampling"], "fails": stats["fails"], "batch_final": bs,
            "mean_out_tokens": round(sum(stats["out_tokens"]) / max(1, len(stats["out_tokens"]))),
            "load_sec": round(m.load_sec), "python": platform.python_version()}
    hist = out_dir / f"run_info_{tag}.jsonl"
    with open(hist, "a", encoding="utf-8") as f:
        f.write(json.dumps(info, ensure_ascii=False) + "\n")
    print(f"[{tag}] 끝 {gen_sec / 60:.1f}분 · 이번에 {n_sent}문장 · 샘플링 재시도 성공 {len(stats['retried_with_sampling'])} · 끝내 실패 {len(stats['fails'])}문단")
    return info, gen_sec, n_sent


def check_prev(m, gt, allb, idx, prev_rows, n, out_dir):
    """1·3장 문단 n개를 이 머신에서 다시 번역해 11/17(A100) 결과와 나란히 본다. 본 결과 파일에는 쓰지 않는다."""
    prev = {r["id"]: r["mt"] for r in prev_rows}
    cands = [k for k in idx if all(s["id"] in prev for s in allb[k]["sents"] if s["translate"])]
    if not cands:
        print("[비교] 1·3장 기존 결과 파일이 없어 건너뜁니다"); return None
    cands.sort(key=lambda k: len(gt.user_text(allb, k, n_ctx=CFG["n_ctx"])[1]))
    ks = cands[::max(1, len(cands) // n)][:n]
    t0, rows, bs, i = time.time(), [], m.bs, 0
    while i < len(ks):
        try:
            res = m.generate(ks[i:i + bs])
        except Exception as e:
            if not m.is_oom(e) or bs == 1:
                raise
            m.free(); bs = max(1, bs // 2); print("  메모리 부족 → 묶음", bs); continue
        for r in res:
            for s in r["sents"]:
                new = r["got"].get(s["id"])
                rows.append({"id": s["id"], "en": s["en"], "a100": prev[s["id"]], "here": new, "same": new == prev[s["id"]], "err": r["err"]})
        i += bs
    with open(out_dir / "check_prev.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    same, bad = sum(r["same"] for r in rows), sum(1 for r in rows if not r["here"])
    print(f"\n[비교] 1·3장 {len(ks)}문단 {len(rows)}문장 · {(time.time() - t0) / 60:.1f}분")
    print(f"  11/17 결과와 글자까지 같음 {same} · 다름 {len(rows) - same} · 번역 안 나옴 {bad}")
    print("  (GPU 종류와 묶음 구성이 달라 일부 문장은 표현이 달라질 수 있음 — 아래에서 뜻이 무너진 문장이 있는지 봄)")
    for r in [r for r in rows if not r["same"]][:12]:
        print(f"  {r['id']} EN   {r['en']}\n          A100 {r['a100']}\n          여기 {r['here']}")
    print(f"  전체 비교표: {out_dir / 'check_prev.jsonl'}\n")
    return rows


def merge(gt, rows, allb, idx, prev_rows, out_dir):
    want_order = [s["id"] for k in idx for s in allb[k]["sents"] if s["translate"]]
    got, dup = {}, 0
    for r in prev_rows + [r for p in part_files(out_dir) for r in read_jsonl(p)]:
        if r["id"] in got:
            dup += 1
            continue
        got[r["id"]] = r
    missing = [i for i in want_order if i not in got]
    empty = [i for i in want_order if i in got and not (got[i]["mt"] or "").strip()]
    final = out_dir / FINAL
    with open(final, "w", encoding="utf-8") as f:
        for i in want_order:
            if i in got:
                f.write(json.dumps(got[i], ensure_ascii=False) + "\n")
    runs = [json.loads(l) for p in sorted(glob.glob(str(out_dir / "run_info_part_*.jsonl"))) for l in open(p, encoding="utf-8") if l.strip()]
    info = {"model": MODEL_ID, "config": NAME, "prompt_hash": gt.PROMPT_HASH, "n_sentences": len(want_order) - len(missing),
            "n_want": len(want_order), "missing": missing, "empty": empty, "duplicates_dropped": dup,
            "from_prev_ch1_3": sum(1 for i in want_order if i in got and got[i]["mt_meta"].get("config") == "ctx5"),
            "gen_sec_total": sum(r["gen_sec"] for r in runs), "runs": runs}
    (out_dir / f"run_info_{NAME}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"합치기 → {final}")
    print(f"번역 {info['n_sentences']} / {len(want_order)} · 빠짐 {len(missing)} · 빈 번역 {len(empty)} · 1·3장 기존 결과 {info['from_prev_ch1_3']} · 중복 버림 {dup}")
    if missing:
        print("빠진 문장이 있습니다 — 같은 명령으로 번역을 다시 실행한 뒤 --merge 를 다시 하세요. 예:", missing[:5])
    else:
        print(f"내려받을 파일: {final} , {out_dir / f'run_info_{NAME}.json'}")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(Path(__file__).parent), help="gemma_translate.py·sentences.jsonl이 있는 폴더(기본: 이 파일 옆)")
    ap.add_argument("--prev", default=None, help="11/17 1·3장 결과(기본: 폴더에 gemma_ch1_3_bf16_ctx5.jsonl이 있으면 씀)")
    ap.add_argument("--shard", default="0/1", help="i/n — 전체를 n몫으로 나눈 것 중 i번째만(0부터)")
    ap.add_argument("--limit", type=int, default=0, help="이 수만큼의 문단만 번역(속도 시험)")
    ap.add_argument("--check-prev", type=int, default=0, help="1·3장 문단을 이 수만큼 다시 번역해 11/17 결과와 비교(결과 파일에는 쓰지 않음)")
    ap.add_argument("--bs", type=int, default=4, help="한 번에 번역할 문단 수(메모리 부족이면 자동으로 줄임)")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--fake", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.download_only:
        print("이미 받아 둠:", local_snapshot()) if local_snapshot() else download()
        return 0

    base = Path(a.dir)
    for n in ("gemma_translate.py", "sentences.jsonl"):
        assert (base / n).exists(), f"{base / n} 이 없습니다"
    sys.path.insert(0, str(base))
    import gemma_translate as gt
    assert gt.PROMPT_HASH == PROMPT_HASH, f"지침이 11/17과 다릅니다({gt.PROMPT_HASH}) — 같은 gemma_translate.py를 쓰세요"
    out_dir = base / "out"; out_dir.mkdir(exist_ok=True)
    rows = read_jsonl(base / "sentences.jsonl")
    allb, idx = gt.blocks(rows, None)      # 책 전체
    want = {s["id"] for k in idx for s in allb[k]["sents"] if s["translate"]}
    prev = a.prev or (str(base / "gemma_ch1_3_bf16_ctx5.jsonl") if (base / "gemma_ch1_3_bf16_ctx5.jsonl").exists() else None)
    prev_rows = load_prev(prev, gt)

    if a.merge:
        merge(gt, rows, allb, idx, prev_rows, out_dir)
        return 0

    si, sn = (int(x) for x in a.shard.split("/"))
    assert 0 <= si < sn, "--shard 는 0/2, 1/2 처럼 씁니다"
    tag = f"part_{si}of{sn}"
    done = {r["id"] for r in prev_rows} | {r["id"] for p in part_files(out_dir) for r in read_jsonl(p)}
    order = plan(gt, allb, idx, done, (si, sn), a.limit)
    print(f"문단 {len(idx)} · 문장 {len(want)} · 이미 번역된 문장 {len(done & want)}(1·3장 기존 결과 {len(prev_rows)}) · 이번에 할 문단 {len(order)}")
    if not order and not a.check_prev:
        print("이 몫에서 더 번역할 것이 없습니다 → 모든 몫이 끝났으면 --merge 를 실행하세요")
        return 0
    m = (Fake if a.fake else Gemma)(gt, allb, a.bs)
    if a.check_prev:
        check_prev(m, gt, allb, idx, prev_rows, a.check_prev, out_dir)
        if not a.limit:          # 비교만 할 때는 본 번역을 시작하지 않음
            return 0
    if not order:
        return 0
    before = len(done & want)
    info, sec, n_sent = run(m, gt, allb, order, done, out_dir / f"{tag}.jsonl", out_dir, tag, want)
    if a.limit and n_sent:
        left = len(want) - before - n_sent
        per = sec / n_sent
        print(f"\n[속도 시험] {n_sent}문장에 {sec / 60:.1f}분 → 문장당 {per:.1f}초")
        print(f"남은 {left}문장을 이 속도로 돌리면 약 {left * per / 3600:.1f}시간(추정)"
              + (f", 모델 두 개로 나눠 돌리면 약 {left * per / 7200:.1f}시간" if m.meta["n_gpu"] >= 4 or sn > 1 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
