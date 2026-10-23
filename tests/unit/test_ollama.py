"""Unit tests for app.llm.ollama.OllamaClient against a local fake Ollama HTTP server."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app.contracts import LlmError, LlmModelMissingError, LlmTimeoutError, LlmUnavailableError
from app.llm.ollama import NUM_CTX_BUCKETS, OllamaClient, pick_num_ctx

MODEL = "qwen2.5-coder:14b"


class FakeOllama:
    """Records requests; answers per the attributes below."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict | None]] = []
        self.models = [MODEL, "gemma4:12b"]
        self.answer = "## Goal\nDo X"
        self.chat_status = 200
        self.chat_body: bytes | None = None  # raw override
        self.delay_s = 0.0
        self.version = "0.40.4"
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # keep pytest output clean
                pass

            def _send(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                fake.requests.append(("GET", self.path, None))
                if self.path == "/api/version":
                    self._send(200, json.dumps({"version": fake.version}).encode())
                elif self.path == "/api/tags":
                    self._send(200, json.dumps({"models": [{"name": m} for m in fake.models]}).encode())
                else:
                    self._send(404, b'{"error":"404 page not found"}')

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.requests.append(("POST", self.path, body))
                if fake.delay_s:
                    threading.Event().wait(fake.delay_s)
                if fake.chat_body is not None:
                    self._send(fake.chat_status, fake.chat_body)
                    return
                if body.get("model") not in fake.models:
                    self._send(404, json.dumps({"error": f"model '{body.get('model')}' not found"}).encode())
                    return
                self._send(fake.chat_status, json.dumps({
                    "model": body["model"],
                    "message": {"role": "assistant", "content": fake.answer},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 42,
                    "eval_count": 7,
                    "total_duration": 1_500_000_000,
                }).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def ollama():
    fake = FakeOllama()
    yield fake
    fake.close()


def client(url: str, **kw) -> OllamaClient:
    return OllamaClient(url, kw.pop("model", MODEL), kw.pop("keep_alive", "10m"), kw.pop("timeout_s", 5), **kw)


def free_port_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"  # nothing listens there once the socket is closed


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


def test_chat_sends_a_deterministic_non_streaming_request(ollama):
    res = client(ollama.url + "/").chat("SYSTEM", "USER")
    assert res.text == "## Goal\nDo X"
    assert (res.prompt_tokens, res.output_tokens, res.total_ms) == (42, 7, 1500.0)
    method, path, body = ollama.requests[-1]
    assert (method, path) == ("POST", "/api/chat")
    assert body["model"] == MODEL
    assert body["stream"] is False
    assert body["keep_alive"] == "10m"
    assert body["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER"}]
    opts = body["options"]
    assert opts["temperature"] == 0 and opts["seed"] == 0
    assert opts["num_ctx"] in NUM_CTX_BUCKETS
    assert opts["num_predict"] == opts["num_ctx"] // 2


@pytest.mark.parametrize("keep_alive,expected", [("0", 0), ("-1", -1), ("30s", "30s")])
def test_keep_alive_numbers_are_sent_as_numbers(ollama, keep_alive, expected):
    client(ollama.url, keep_alive=keep_alive).chat("s", "u")
    assert ollama.requests[-1][2]["keep_alive"] == expected


def test_missing_model_is_actionable(ollama):
    with pytest.raises(LlmModelMissingError) as excinfo:
        client(ollama.url, model="qwen2.5-coder:32b").chat("s", "u")
    assert "ollama pull qwen2.5-coder:32b" in excinfo.value.user_message


def test_server_not_running():
    with pytest.raises(LlmUnavailableError) as excinfo:
        client(free_port_url()).chat("s", "u")
    assert "isn't running" in excinfo.value.user_message


def test_timeout(ollama):
    ollama.delay_s = 2.0
    with pytest.raises(LlmTimeoutError):
        client(ollama.url, timeout_s=0.5).chat("s", "u")


def test_server_error(ollama):
    ollama.chat_status = 500
    ollama.chat_body = b'{"error":"boom"}'
    with pytest.raises(LlmError) as excinfo:
        client(ollama.url).chat("s", "u")
    assert not isinstance(excinfo.value, (LlmModelMissingError, LlmUnavailableError, LlmTimeoutError))
    assert "boom" in excinfo.value.details


@pytest.mark.parametrize("raw", [b"not json", b'{"message": {}}', b"[]"])
def test_unreadable_answers(ollama, raw):
    ollama.chat_body = raw
    with pytest.raises(LlmError):
        client(ollama.url).chat("s", "u")


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


def test_health_ok(ollama):
    h = client(ollama.url).health()
    assert h.ok and h.reachable and h.model_present
    assert h.version == "0.40.4"
    assert MODEL in h.message


def test_health_model_missing(ollama):
    ollama.models = ["gemma4:12b"]
    h = client(ollama.url).health()
    assert not h.ok and h.reachable and not h.model_present
    assert f"ollama pull {MODEL}" in h.message


def test_health_latest_tag(ollama):
    ollama.models = ["qwen2.5-coder:latest"]
    assert client(ollama.url, model="qwen2.5-coder").health().ok


def test_health_unreachable():
    h = client(free_port_url()).health()
    assert not h.ok and not h.reachable


# ---------------------------------------------------------------------------
# num_ctx
# ---------------------------------------------------------------------------


def test_num_ctx_buckets():
    assert pick_num_ctx("a" * 100) == 4096
    assert pick_num_ctx("a" * 9_000) == 8192
    assert pick_num_ctx("a" * 20_000) == 16384
    assert pick_num_ctx("a" * 40_000) == 32768
    assert pick_num_ctx("a" * 500_000) == 32768
