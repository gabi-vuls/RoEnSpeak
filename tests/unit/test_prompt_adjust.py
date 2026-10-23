"""Unit tests for app.core.prompt_adjust.PromptAdjuster and app.core.prompt_history."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Optional

import pytest

from app.contracts import (
    InjectionBlockedError,
    LlmConfig,
    LlmModelMissingError,
    LlmUnavailableError,
    ModifiersHeldError,
    NothingSelectedError,
    Notification,
    PromptStatus,
)
from app.core.events import SimpleEventBus
from app.core.prompt_adjust import PromptAdjuster
from app.core.prompt_history import PromptHistory
from app.llm.ollama import ChatResult, HealthResult

DRAFT = (
    "i want a new hotkey alt+p that copies the selected text and sends it to qwen2.5-coder:14b via "
    "ollama on http://localhost:11434, then pastes it back. keep the original in the logs, timeout 120 seconds"
)
REWRITE = """\
## Goal
A new hotkey alt+p that copies the selected text and sends it to qwen2.5-coder:14b via ollama on \
http://localhost:11434, then pastes it back.

## Requirements
- Keep the original in the logs.

## Constraints
- Timeout: 120 seconds."""


@dataclass
class Sel:
    text: str
    hwnd: int = 0x1000


class FakeGrabber:
    def __init__(self, text: str = DRAFT) -> None:
        self.text = text
        self.error: Optional[Exception] = None
        self.hwnd = 0x1000
        self.foreground = 0x1000
        self.grabs = 0

    def grab(self) -> Sel:
        self.grabs += 1
        if self.error is not None:
            raise self.error
        return Sel(self.text, self.hwnd)

    def foreground_window(self) -> int:
        return self.foreground


class FakeClient:
    def __init__(self, answer: str = REWRITE) -> None:
        self.answer = answer
        self.error: Optional[Exception] = None
        self.calls: list[tuple[str, str]] = []
        self.gate: Optional[threading.Event] = None

    def chat(self, system: str, user: str, num_ctx: Optional[int] = None) -> ChatResult:
        self.calls.append((system, user))
        if self.gate is not None:
            self.gate.wait(5)
        if self.error is not None:
            raise self.error
        return ChatResult(self.answer, 10, 20, 1234.0)

    def health(self) -> HealthResult:
        return HealthResult(True, True, "0.40.4", True, "ok")


class Harness:
    def __init__(self, tmp_path, cfg: LlmConfig = LlmConfig()) -> None:
        self.bus = SimpleEventBus()
        self.events: list = []
        self.bus.subscribe(self.events.append)
        self.grabber = FakeGrabber()
        self.client = FakeClient()
        self.pasted: list[str] = []
        self.clipboard: list[str] = []
        self.paste_error: Optional[Exception] = None
        self.dictating = False
        self.cfg = cfg
        self.history_path = tmp_path / "prompt_history.jsonl"
        self.done = threading.Event()
        self.bus.subscribe(lambda ev: self.done.set() if isinstance(ev, PromptStatus) and not ev.busy else None)
        self.adj = PromptAdjuster(
            grabber=self.grabber,
            paste=self._paste,
            set_clipboard=self.clipboard.append,
            get_config=lambda: self.cfg,
            is_dictating=lambda: self.dictating,
            bus=self.bus,
            history=PromptHistory(self.history_path),
            client_factory=lambda cfg: self.client,
        )
        self.adj.start()

    def _paste(self, text: str) -> None:
        if self.paste_error is not None:
            raise self.paste_error
        self.pasted.append(text)

    def run(self) -> None:
        self.done.clear()
        self.adj.trigger()
        assert self.done.wait(5), "job did not finish"

    def notifications(self) -> list[Notification]:
        return [e for e in self.events if isinstance(e, Notification)]

    def history(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        return [json.loads(line) for line in self.history_path.read_text(encoding="utf-8").splitlines()]

    def stop(self) -> None:
        self.adj.stop()


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    yield harness
    harness.stop()


def test_happy_path_pastes_the_rewrite_and_keeps_the_original_in_history(h):
    h.run()
    assert h.pasted == [REWRITE]
    assert h.clipboard == []
    assert h.notifications() == []
    (rec,) = h.history()
    assert rec["original"] == DRAFT and rec["result"] == REWRITE
    assert rec["pasted"] is True and rec["outcome"] == "pasted"
    assert rec["fidelity"]["ok"] is True
    system, user = h.client.calls[0]
    assert "prompt formatter" in system and DRAFT in user
    phases = [e.phase for e in h.events if isinstance(e, PromptStatus)]
    assert phases == ["copying", "rewriting", "pasting", "idle"]


def test_refused_while_dictating(h):
    h.dictating = True
    h.adj.trigger()
    assert h.grabber.grabs == 0
    assert "Stop dictation first" in h.notifications()[0].message


def test_second_press_while_busy_is_dropped(h):
    h.client.gate = threading.Event()
    h.done.clear()
    h.adj.trigger()
    for _ in range(200):
        if h.client.calls:
            break
        threading.Event().wait(0.01)
    h.adj.trigger()  # while the LLM call blocks
    assert any("Already adjusting" in n.message for n in h.notifications())
    h.client.gate.set()
    assert h.done.wait(5)
    assert h.grabber.grabs == 1 and h.pasted == [REWRITE]


@pytest.mark.parametrize("error,fragment", [
    (NothingSelectedError(), "No text is selected"),
    (ModifiersHeldError(""), "Release the hotkey keys"),
    (InjectionBlockedError("", reason="elevated"), "administrator"),
])
def test_copy_failures_notify_and_change_nothing(h, error, fragment):
    h.grabber.error = error
    h.run()
    assert h.pasted == [] and h.clipboard == [] and h.client.calls == []
    assert fragment in h.notifications()[0].message


@pytest.mark.parametrize("error", [
    LlmUnavailableError("Ollama isn't running at http://localhost:11434. Start Ollama and try again."),
    LlmModelMissingError("qwen2.5-coder:14b"),
])
def test_llm_failures_notify_keep_the_text_and_log_history(h, error):
    h.client.error = error
    h.run()
    assert h.pasted == [] and h.clipboard == []
    assert error.user_message in h.notifications()[0].message
    (rec,) = h.history()
    assert rec["original"] == DRAFT and type(error).__name__ in rec["error"]


def test_fidelity_failure_keeps_the_original_and_puts_the_rewrite_on_the_clipboard(h):
    h.client.answer = REWRITE.replace("120", "60")
    h.run()
    assert h.pasted == []
    assert h.clipboard == [REWRITE.replace("120", "60")]
    n = h.notifications()[0]
    assert n.warning and "120" in n.message
    (rec,) = h.history()
    assert rec["pasted"] is False and rec["fidelity"]["ok"] is False and "120" in rec["fidelity"]["missing"]


def test_fidelity_failure_pastes_with_a_warning_when_strict_check_is_off(tmp_path):
    h = Harness(tmp_path, LlmConfig(strict_check=False))
    try:
        h.client.answer = REWRITE.replace("120", "60")
        h.run()
        assert h.pasted == [REWRITE.replace("120", "60")]
        assert h.notifications()[0].warning
    finally:
        h.stop()


def test_focus_change_puts_the_rewrite_on_the_clipboard(h):
    h.grabber.foreground = 0x2000
    h.run()
    assert h.pasted == [] and h.clipboard == [REWRITE]
    assert "focus moved" in h.notifications()[0].message


def test_paste_failure_falls_back_to_the_clipboard(h):
    h.paste_error = InjectionBlockedError(REWRITE, reason="elevated")
    h.run()
    assert h.clipboard == [REWRITE]
    assert "clipboard" in h.notifications()[0].message
    assert h.history()[0]["outcome"] == "clipboard_paste_failed"


def test_too_long_selection_is_refused(tmp_path):
    h = Harness(tmp_path, LlmConfig(max_input_chars=100))
    try:
        h.run()
        assert h.client.calls == [] and h.pasted == []
        assert "limit" in h.notifications()[0].message
    finally:
        h.stop()


def test_unchanged_rewrite_is_not_pasted(h):
    h.client.answer = DRAFT
    h.run()
    assert h.pasted == []
    assert "already well structured" in h.notifications()[0].message


def test_chatty_wrapper_is_removed_before_pasting(h):
    h.client.answer = f"Here is the rewritten prompt:\n```markdown\n{REWRITE}\n```"
    h.run()
    assert h.pasted == [REWRITE]


def test_check_connection_uses_the_client(h):
    assert h.adj.check_connection().ok


# ---------------------------------------------------------------------------
# PromptHistory
# ---------------------------------------------------------------------------


def test_history_rotates_at_the_size_cap(tmp_path):
    path = tmp_path / "logs" / "prompt_history.jsonl"
    hist = PromptHistory(path, max_bytes=300, backups=2)
    for i in range(10):
        hist.append({"i": i, "original": "x" * 100})
    assert path.exists()
    assert path.with_name(path.name + ".1").exists()
    assert path.with_name(path.name + ".2").exists()
    assert not path.with_name(path.name + ".3").exists()
    last = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert last["i"] == 9 and "ts" in last


def test_history_keeps_unicode_readable(tmp_path):
    path = tmp_path / "h.jsonl"
    PromptHistory(path).append({"original": "Bună ziua, ș ț ă"})
    assert "Bună ziua, ș ț ă" in path.read_text(encoding="utf-8")


def test_history_default_path_is_in_the_log_folder(isolated_app_dirs):
    from app.contracts import logs_dir

    assert PromptHistory().path == logs_dir() / "prompt_history.jsonl"
