"""Hallucination filter for whisper segments (see contracts.FilterConfig for the rules)."""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Optional

from app.contracts import FilterConfig, SegmentStats

logger = logging.getLogger("roenspeak.engine.filter")

REASON_EMPTY = "empty"
REASON_NO_SPEECH = "no_speech"
REASON_COMPRESSION = "compression"
REASON_BLOCKLIST = "blocklist"

_WHITESPACE_RUN = re.compile(r"\s+")


def _keep_char(ch: str) -> bool:
    return ch.isalpha() or ch.isdigit() or ch.isspace() or ch == "." or ch == "*"


def normalize_text(text: str) -> str:
    """casefold, strip, replace every char that is not a letter, digit, space, '.' or '*'
    with a space, collapse whitespace runs, strip, then strip '.' from both ends.

    >>> normalize_text("  Thank you for watching!! ")
    'thank you for watching'
    """
    s = text.casefold().strip()
    s = "".join(ch if _keep_char(ch) else " " for ch in s)
    s = _WHITESPACE_RUN.sub(" ", s).strip()
    # Stripping dots can expose whitespace (e.g. "thank you. ."), so strip both until stable.
    while True:
        stripped = s.strip(".").strip()
        if stripped == s:
            return s
        s = stripped


def _has_letter_or_digit(s: str) -> bool:
    return any(ch.isalpha() or ch.isdigit() for ch in s)


@dataclass(frozen=True)
class _Compiled:
    cfg: FilterConfig
    exact: frozenset[str]
    prefixes: tuple[str, ...]


def _compile(cfg: FilterConfig) -> _Compiled:
    exact: set[str] = set()
    prefixes: list[str] = []
    for entry in cfg.blocklist:
        norm = normalize_text(entry)
        if norm.endswith("*"):
            prefix = norm[:-1]
            # An entry that is only "*" would drop every segment; it is ignored.
            if prefix.strip():
                prefixes.append(prefix)
        elif norm:
            exact.add(norm)
    return _Compiled(cfg=cfg, exact=frozenset(exact), prefixes=tuple(prefixes))


class ThresholdHallucinationFilter:
    """HallucinationFilter implementation. evaluate() is pure; set_config() swaps the
    compiled config atomically, so both are safe from any thread."""

    def __init__(self, cfg: FilterConfig) -> None:
        self._lock = threading.Lock()
        self._compiled = _compile(cfg)

    @property
    def config(self) -> FilterConfig:
        return self._compiled.cfg

    def set_config(self, cfg: FilterConfig) -> None:
        compiled = _compile(cfg)
        with self._lock:
            self._compiled = compiled

    def evaluate(self, stats: SegmentStats) -> Optional[str]:
        c = self._compiled  # single reference read: a consistent snapshot
        cfg = c.cfg
        norm = normalize_text(stats.text)
        if not _has_letter_or_digit(norm):
            return REASON_EMPTY
        if (
            stats.no_speech_prob > cfg.no_speech_prob_threshold
            and stats.avg_logprob < cfg.avg_logprob_threshold
        ):
            return REASON_NO_SPEECH
        if stats.compression_ratio > cfg.compression_ratio_threshold:
            return REASON_COMPRESSION
        if norm in c.exact or any(norm.startswith(p) for p in c.prefixes):
            return REASON_BLOCKLIST
        return None
