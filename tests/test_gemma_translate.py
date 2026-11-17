import ast
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from translation import gemma_translate as g

ROOT = Path(__file__).resolve().parents[1]


def _claude_system():
    tree = ast.parse((ROOT / "translation/llm_translate.py").read_text(encoding="utf-8"))
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SYSTEM":
            return n.value.value


def test_system_same_as_claude():
    assert g.SYSTEM == _claude_system()


def test_parse_fenced():
    assert g.parse('```json\n{"translations":[{"id":"S1","ko":" 가 "}]}\n```') == {"S1": "가"}
    assert g.parse('설명\n{"translations":[{"id":"S1","ko":"가"}]}') == {"S1": "가"}
    broken = '{"translations": [{"id": "S1", "ko": "그가 "좋아"라고 했다."}, {"id": "S2", "ko": "나"}]}'
    assert g.parse(broken) == {"S1": '그가 "좋아"라고 했다.', "S2": "나"}


def test_end_to_end_mock(tmp_path, monkeypatch):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            ids = [l.split("\t")[0] for l in body["messages"][1]["content"].split("[번역할 문장 — id마다 하나씩]\n")[1].splitlines()]
            out = {"choices": [{"message": {"content": json.dumps({"translations": [{"id": i, "ko": "번역" + i} for i in ids]})}}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "provider": "mock"}
            data = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("K", "x")
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(k, raising=False)
    out = tmp_path / "o.jsonl"
    g.main([str(ROOT / "data/book/gatsby/sentences.jsonl"), str(out), "--chapters", "I", "III", "--live",
            "--base-url", f"http://127.0.0.1:{srv.server_port}", "--key-env", "K"])
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert len(rows) == 638 and len({r["id"] for r in rows}) == 638
    assert rows[0]["mt_system"] == "gemma-4-31b-it"
    srv.shutdown()
