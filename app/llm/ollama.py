"""Minimal client for a local Ollama server (https://github.com/ollama/ollama/blob/main/docs/api.md).

Stdlib only (urllib + json): two endpoints are needed, and the app already ships without a
pinned HTTP library on Windows. Calls block; run them off the Qt thread.

chat(system, user, num_ctx): POST /api/chat, non-streaming, deterministic options.
health(): GET /api/version + /api/tags -> HealthResult (never raises).
Errors: LlmUnavailableError (connection refused / DNS), LlmModelMissingError (404 "not found"),
LlmTimeoutError, LlmError (any other HTTP or protocol failure).
"""

from __future__ import annotations

import json
import logging
import math
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from app.contracts import LlmError, LlmModelMissingError, LlmTimeoutError, LlmUnavailableError

log = logging.getLogger("roenspeak.llm.ollama")

#: num_ctx values used. A fixed set keeps Ollama from reloading the model for every new size.
NUM_CTX_BUCKETS = (4096, 8192, 16384, 32768)
#: Health checks must answer fast even when the configured timeout is long.
HEALTH_TIMEOUT_S = 5.0


def pick_num_ctx(*texts: str) -> int:
    """Context window that fits the prompt plus an output of about the same size. Token count
    is estimated as chars / 3 (conservative for English, Romanian and code)."""
    prompt_tokens = math.ceil(sum(len(t) for t in texts) / 3)
    needed = prompt_tokens * 2 + 512
    for n in NUM_CTX_BUCKETS:
        if needed <= n:
            return n
    return NUM_CTX_BUCKETS[-1]


@dataclass(frozen=True)
class ChatResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    total_ms: float


@dataclass(frozen=True)
class HealthResult:
    ok: bool  # server reachable AND model present
    reachable: bool
    version: Optional[str]
    model_present: bool
    message: str  # user-facing one-liner


class OllamaClient:
    def __init__(self, base_url: str, model: str, keep_alive: str = "10m", timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.timeout_s = float(timeout_s)

    # --- public --------------------------------------------------------------------------

    def chat(self, system: str, user: str, num_ctx: Optional[int] = None) -> ChatResult:
        num_ctx = num_ctx or pick_num_ctx(system, user)
        body = {
            "model": self.model,
            "stream": False,
            "keep_alive": _keep_alive_value(self.keep_alive),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": 0,
                "top_p": 1,
                "seed": 0,
                "num_ctx": num_ctx,
                "num_predict": num_ctx // 2,
            },
        }
        data = self._request("POST", "/api/chat", body, self.timeout_s)
        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LlmError("Ollama returned an unexpected answer.", details=f"no message.content in {_short(data)}")
        if data.get("done_reason") == "length":
            log.warning("ollama: output hit num_predict=%d (truncated)", num_ctx // 2)
        return ChatResult(
            text=content,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            total_ms=float(data.get("total_duration") or 0) / 1e6,
        )

    def health(self) -> HealthResult:
        try:
            version_data = self._request("GET", "/api/version", None, HEALTH_TIMEOUT_S)
            version = version_data.get("version") if isinstance(version_data, dict) else None
            tags = self._request("GET", "/api/tags", None, HEALTH_TIMEOUT_S)
        except LlmError as e:
            reachable = not isinstance(e, LlmUnavailableError)
            return HealthResult(False, reachable, None, False, e.user_message)
        names = [m.get("name") for m in (tags.get("models") or []) if isinstance(m, dict)] if isinstance(tags, dict) else []
        present = _model_present(self.model, names)
        label = f"Ollama {version}" if version else "Ollama"
        if present:
            return HealthResult(True, True, version, True, f"{label} is running and {self.model} is installed.")
        return HealthResult(False, True, version, False,
                            f"{label} is running, but {self.model} is not installed. Run: ollama pull {self.model}")

    # --- transport -----------------------------------------------------------------------

    def _request(self, method: str, path: str, body: Optional[dict], timeout_s: float) -> Any:
        url = self.base_url + path
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=payload, method=method,
                                     headers={"Content-Type": "application/json"} if payload else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = _read_error_body(e)
            if e.code == 404 and "not found" in detail.lower():
                raise LlmModelMissingError(self.model, details=f"HTTP 404 {path}: {detail}") from None
            raise LlmError(f"Ollama answered with an error (HTTP {e.code}).",
                           details=f"{method} {url}: {detail}") from None
        except (socket.timeout, TimeoutError):
            raise LlmTimeoutError(
                f"Ollama did not answer within {timeout_s:.0f} s. The model may still be loading, or it "
                "runs partly on the CPU because the GPU is full. Try again, or raise the timeout in Settings.",
                details=f"{method} {url}",
            ) from None
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError)):
                raise LlmTimeoutError(
                    f"Ollama did not answer within {timeout_s:.0f} s.", details=f"{method} {url}"
                ) from None
            raise LlmUnavailableError(
                f"Ollama isn't running at {self.base_url}. Start Ollama and try again.",
                details=f"{method} {url}: {e.reason}",
            ) from None
        except (ConnectionError, OSError) as e:
            raise LlmUnavailableError(
                f"Ollama isn't running at {self.base_url}. Start Ollama and try again.",
                details=f"{method} {url}: {e}",
            ) from None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as e:
            raise LlmError("Ollama returned an unreadable answer.", details=f"{method} {url}: {e}") from None


def _keep_alive_value(keep_alive: str) -> Any:
    """Ollama wants a number for a bare count of seconds ("0", "-1") and a string otherwise."""
    try:
        return int(keep_alive)
    except ValueError:
        return keep_alive


def _model_present(model: str, names: list[Any]) -> bool:
    names = [n for n in names if isinstance(n, str)]
    if model in names:
        return True
    # "qwen2.5-coder" means "qwen2.5-coder:latest" to Ollama.
    return ":" not in model and f"{model}:latest" in names


def _read_error_body(e: urllib.error.HTTPError) -> str:
    try:
        raw = e.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - best effort
        return str(e)
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("error"), str):
            return data["error"]
    except ValueError:
        pass
    return raw[:300]


def _short(data: Any) -> str:
    text = repr(data)
    return text if len(text) <= 200 else text[:200] + "..."
