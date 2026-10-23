"""Prompt adjustment: select text anywhere, press the prompt hotkey, and the selection is
replaced by the same prompt restructured by a local LLM (Ollama).

One daemon thread, "prompt-adjust", runs one job at a time (an LLM call takes seconds to
minutes, so it must not block core-control). trigger() is called on the hotkey thread and
only enqueues; a press while a job runs is dropped with a notification.

A job:
  1. refuse while dictating (LISTENING / PROCESSING): both drive the clipboard and keystrokes;
  2. copy the selection (Ctrl+C) under io_lock; the user's clipboard is restored right away;
  3. ask the LLM (no lock held), clean the answer, run the fidelity check;
  4. paste over the selection under io_lock, but only if the check passed (or strict_check is
     off), dictation did not start meanwhile, and the same window still has the focus.
     Otherwise the result goes to the clipboard and a notification says why;
  5. append original + result to PromptHistory. roenspeak.log only gets counts and timings.

Not part of the AppState machine: it works whatever the Whisper model is doing (disarmed,
unloaded, error). Progress is published as PromptStatus events.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Optional, Protocol

from app.contracts import (
    AppError,
    EventBus,
    InjectionBlockedError,
    LlmConfig,
    LlmError,
    ModifiersHeldError,
    NothingSelectedError,
    Notification,
    PromptStatus,
)
from app.core.prompt_history import PromptHistory
from app.llm.ollama import ChatResult, HealthResult, OllamaClient
from app.llm.postprocess import check_fidelity, clean_output
from app.llm.prompt_text import SYSTEM_PROMPT, build_user_message

log = logging.getLogger("roenspeak.core.prompt_adjust")

TITLE = "Prompt adjustment"


class _Selection(Protocol):
    text: str
    hwnd: int


class SelectionGrabber(Protocol):
    def grab(self) -> _Selection: ...
    def foreground_window(self) -> int: ...


class LlmClient(Protocol):
    def chat(self, system: str, user: str, num_ctx: Optional[int] = None) -> ChatResult: ...
    def health(self) -> HealthResult: ...


def make_ollama_client(cfg: LlmConfig) -> LlmClient:
    return OllamaClient(cfg.ollama_url, cfg.model, cfg.keep_alive, cfg.timeout_s)


class PromptAdjuster:
    def __init__(
        self,
        *,
        grabber: SelectionGrabber,
        paste: Callable[[str], Any],
        set_clipboard: Callable[[str], None],
        get_config: Callable[[], LlmConfig],
        is_dictating: Callable[[], bool],
        bus: Optional[EventBus] = None,
        history: Optional[PromptHistory] = None,
        io_lock: Optional[threading.Lock] = None,
        client_factory: Callable[[LlmConfig], LlmClient] = make_ollama_client,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._grabber = grabber
        self._paste = paste
        self._set_clipboard = set_clipboard
        self._get_config = get_config
        self._is_dictating = is_dictating
        self._bus = bus
        self._history = history if history is not None else PromptHistory()
        self._io_lock = io_lock if io_lock is not None else threading.Lock()
        self._client_factory = client_factory
        self._clock = clock

        self._q: "queue.Queue[Optional[int]]" = queue.Queue()
        self._busy = False
        self._busy_lock = threading.Lock()
        self._job_ids = 0
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="prompt-adjust", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        """Lets a running job finish its current step; an LLM call in flight is abandoned
        (the thread is a daemon)."""
        if self._thread is None:
            return
        self._q.put(None)
        self._thread.join(timeout)
        self._thread = None

    @property
    def busy(self) -> bool:
        with self._busy_lock:
            return self._busy

    # -- entry points -------------------------------------------------------

    def trigger(self) -> None:
        """Hotkey thread. Never blocks."""
        if self._is_dictating():
            self._notify("Stop dictation first, then press the prompt hotkey again.")
            return
        with self._busy_lock:
            if self._busy:
                busy = True
            else:
                busy = False
                self._busy = True
                self._job_ids += 1
                job_id = self._job_ids
        if busy:
            self._notify("Already adjusting a prompt. Wait for it to finish.")
            return
        self._q.put(job_id)

    def check_connection(self) -> HealthResult:
        """Blocking; call off the Qt thread."""
        return self._client_factory(self._get_config()).health()

    # -- worker ---------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            job_id = self._q.get()
            if job_id is None:
                return
            try:
                self._run(job_id)
            except Exception:
                log.exception("prompt adjust job %d failed", job_id)
                self._notify("Prompt adjustment failed unexpectedly. See the log for details.", warning=True)
            finally:
                with self._busy_lock:
                    self._busy = False
                self._status(False, "idle")

    def _run(self, job_id: int) -> None:
        cfg = self._get_config()
        t0 = self._clock()
        record: dict[str, Any] = {"job": job_id, "model": cfg.model, "pasted": False}

        # 1-2. copy the selection
        self._status(True, "copying")
        try:
            with self._io_lock:
                selection = self._grabber.grab()
        except NothingSelectedError as e:
            log.info("prompt adjust %d: nothing selected (%s)", job_id, e.details)
            self._notify(e.user_message)
            return
        except ModifiersHeldError:
            self._notify("Release the hotkey keys so the selection can be copied.", warning=True)
            return
        except InjectionBlockedError as e:
            self._notify(f"Can't copy from this window. {e.user_message}", warning=True)
            return
        except AppError as e:
            log.warning("prompt adjust %d: copy failed: %s (%s)", job_id, e.user_message, e.details)
            self._notify(f"Could not copy the selection. {e.user_message}", warning=True)
            return

        original = selection.text
        record.update(chars_in=len(original), original=original)
        if len(original) > cfg.max_input_chars:
            record["error"] = "too_long"
            self._history.append(record)
            self._notify(f"The selection has {len(original):,} characters; the limit is "
                         f"{cfg.max_input_chars:,} (Settings). Nothing was changed.", warning=True)
            return

        # 3. ask the LLM
        self._status(True, "rewriting")
        try:
            answer = self._client_factory(cfg).chat(SYSTEM_PROMPT, build_user_message(original))
        except LlmError as e:
            log.warning("prompt adjust %d: LLM failed: %s (%s)", job_id, e.user_message, e.details)
            record["error"] = f"{type(e).__name__}: {e.details or e.user_message}"
            self._history.append(record)
            self._notify(f"{e.user_message} Nothing was changed.", warning=True)
            return

        result = clean_output(answer.text, original)
        report = check_fidelity(original, result)
        record.update(result=result, chars_out=len(result), llm_ms=round(answer.total_ms),
                      fidelity={"ok": report.ok, "coverage": round(report.coverage, 3),
                                "novelty": round(report.novelty, 3),
                                "length_ratio": round(report.length_ratio, 2),
                                "missing": list(report.missing[:20]), "reasons": list(report.reasons)})

        # 4. paste, or fall back to the clipboard
        if not result.strip():
            record["error"] = "empty_result"
            self._finish(job_id, record, t0)
            self._notify("The model returned an empty answer. Nothing was changed.", warning=True)
            return
        if result.strip() == original.strip():
            record["outcome"] = "unchanged"
            self._finish(job_id, record, t0)
            self._notify("The prompt is already well structured; nothing was changed.")
            return
        if not report.ok and cfg.strict_check:
            self._to_clipboard(result)
            record["outcome"] = "clipboard_fidelity"
            self._finish(job_id, record, t0)
            self._notify("The rewrite may have lost or added information "
                         f"({report.summary()}). Your text was left unchanged; the rewrite is on the "
                         "clipboard (Ctrl+V to use it anyway).", warning=True)
            return

        self._status(True, "pasting")
        with self._io_lock:
            focus_kept = self._grabber.foreground_window() == selection.hwnd
            if self._is_dictating() or not focus_kept:
                self._to_clipboard_locked(result)
                record["outcome"] = "clipboard_focus"
                pasted = False
                paste_error: Optional[AppError] = None
            else:
                try:
                    self._paste(result)
                    pasted = True
                    paste_error = None
                except AppError as e:
                    pasted = False
                    paste_error = e
                    self._to_clipboard_locked(result)
                    record["outcome"] = "clipboard_paste_failed"
        record["pasted"] = pasted
        if pasted:
            record["outcome"] = "pasted"
        self._finish(job_id, record, t0)

        if paste_error is not None:
            log.warning("prompt adjust %d: paste failed: %s (%s)", job_id, paste_error.user_message,
                        paste_error.details)
            self._notify(f"Could not paste the rewrite. {paste_error.user_message} "
                         "It is on the clipboard instead.", warning=True)
        elif not pasted:
            self._notify("The focus moved to another window, so nothing was replaced. "
                         "The rewrite is on the clipboard.", warning=True)
        elif not report.ok:
            self._notify(f"Pasted, but the rewrite may have lost or added information ({report.summary()}). "
                         "The original is in prompt_history.jsonl.", warning=True)

    # -- helpers --------------------------------------------------------------

    def _finish(self, job_id: int, record: dict[str, Any], t0: float) -> None:
        record["duration_ms"] = round((self._clock() - t0) * 1000.0)
        self._history.append(record)
        log.info("prompt adjust %d: outcome=%s chars_in=%d chars_out=%d llm_ms=%s total_ms=%d fidelity_ok=%s",
                 job_id, record.get("outcome", record.get("error", "none")), record.get("chars_in", 0),
                 record.get("chars_out", 0), record.get("llm_ms"), record["duration_ms"],
                 record.get("fidelity", {}).get("ok"))

    def _to_clipboard(self, text: str) -> None:
        with self._io_lock:
            self._to_clipboard_locked(text)

    def _to_clipboard_locked(self, text: str) -> None:
        try:
            self._set_clipboard(text)
        except AppError as e:
            log.warning("could not put the rewrite on the clipboard: %s (%s)", e.user_message, e.details)

    def _status(self, busy: bool, phase: str) -> None:
        if self._bus is not None:
            self._bus.publish(PromptStatus(busy=busy, phase=phase))

    def _notify(self, message: str, warning: bool = False) -> None:
        if self._bus is not None:
            self._bus.publish(Notification(title=TITLE, message=message, warning=warning))
