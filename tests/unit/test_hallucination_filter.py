"""Spec/contract tests for app.engine.filter (ThresholdHallucinationFilter, normalize_text).

Derived from contracts.FilterConfig / HallucinationFilter docstrings and the approved
decision "Blocklist excludes 'thank you'/'you'".
"""

from __future__ import annotations

import threading

import pytest

from app.contracts import DEFAULT_BLOCKLIST, FilterConfig, HallucinationFilter, SegmentStats
from app.engine.filter import ThresholdHallucinationFilter, normalize_text

# "Clean" stats: well below every threshold.
GOOD_LOGPROB = -0.3
GOOD_NOSPEECH = 0.05
GOOD_CR = 1.3


def stats(text="Hello, how are you today?", avg_logprob=GOOD_LOGPROB, no_speech_prob=GOOD_NOSPEECH,
          compression_ratio=GOOD_CR) -> SegmentStats:
    return SegmentStats(text=text, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob,
                        compression_ratio=compression_ratio)


@pytest.fixture
def filt():
    return ThresholdHallucinationFilter(FilterConfig())


def test_is_a_hallucination_filter(filt):
    assert isinstance(filt, HallucinationFilter)


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  Thank you for watching!! ", "thank you for watching"),  # documented example
        ("THANK YOU FOR WATCHING.", "thank you for watching"),
        ("Hello,   World", "hello world"),
        ("thank\tyou\nfor\r\nwatching", "thank you for watching"),
        ("...Hello...", "hello"),
        ("don't", "don t"),
        ("Straße", "strasse"),  # casefold, not lower
        ("Mulțumesc frumos", "mulțumesc frumos"),  # non-ASCII letters are kept
        ("Room 101", "room 101"),
        ("www.example.com", "www.example.com"),  # inner dots kept
        ("Subtitles by*", "subtitles by*"),  # '*' kept
        ("", ""),
        ("  !!? ", ""),
        ("♪ ♪", ""),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


# ---------------------------------------------------------------------------
# keep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Hello, how are you today?",
        "I would like to order a coffee, please.",
        "Thank you.",  # approved decision: not blocked
        "thank you",
        "Thank you!",
        "You",
        "you.",
        "Thank you so much for your help.",
        "Thank you for watching the kids yesterday.",  # exact entry, not a prefix entry
    ],
)
def test_normal_text_is_kept(filt, text):
    assert filt.evaluate(stats(text)) is None


def test_default_blocklist_contents():
    bl = FilterConfig().blocklist
    assert bl == DEFAULT_BLOCKLIST
    assert "thank you for watching" in bl
    assert "subtitles by*" in bl
    assert "thank you" not in bl
    assert "you" not in bl


# ---------------------------------------------------------------------------
# empty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "...", "!?", "♪", " - ", "*"])
def test_empty_rule(filt, text):
    assert filt.evaluate(stats(text)) == "empty"


def test_single_digit_is_not_empty(filt):
    assert filt.evaluate(stats("7")) is None


# ---------------------------------------------------------------------------
# no_speech (requires BOTH conditions, strict comparisons)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "no_speech_prob,avg_logprob,expected",
    [
        (0.61, -1.01, "no_speech"),
        (0.99, -5.0, "no_speech"),
        (0.6, -1.5, None),  # not strictly greater than 0.6
        (0.9, -1.0, None),  # not strictly less than -1.0
        (0.9, -0.5, None),  # only no_speech_prob high
        (0.3, -2.0, None),  # only avg_logprob low
    ],
)
def test_no_speech_rule(filt, no_speech_prob, avg_logprob, expected):
    assert filt.evaluate(stats(no_speech_prob=no_speech_prob, avg_logprob=avg_logprob)) == expected


# ---------------------------------------------------------------------------
# compression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cr,expected", [(2.41, "compression"), (5.0, "compression"), (2.4, None), (2.39, None)])
def test_compression_rule(filt, cr, expected):
    assert filt.evaluate(stats(compression_ratio=cr)) == expected


# ---------------------------------------------------------------------------
# blocklist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Thank you for watching!",
        "  thank you for watching. ",
        "THANKS FOR WATCHING",
        "Please subscribe.",
        "Please like and subscribe!",
        "Subtitles by the Amara.org community",
        "Subtitles by",
        "Subtitrare realizată de ...",
        "www.example.com",
        "Translated by John",
    ],
)
def test_blocklist_matches(filt, text):
    assert filt.evaluate(stats(text)) == "blocklist"


@pytest.mark.parametrize(
    "text",
    [
        "Thank you for watching the kids",  # exact entry: no prefix match
        "I said thank you for watching",  # not at the start
        "These subtitles by me",  # prefix must be at the start
    ],
)
def test_blocklist_exact_vs_prefix(filt, text):
    assert filt.evaluate(stats(text)) is None


def test_blocklist_entries_are_normalized():
    f = ThresholdHallucinationFilter(FilterConfig(blocklist=("  Hello, World!! ", "FOO BAR*")))
    assert f.evaluate(stats("hello world")) == "blocklist"
    assert f.evaluate(stats("Hello world.")) == "blocklist"
    assert f.evaluate(stats("Foo bar baz")) == "blocklist"
    assert f.evaluate(stats("Hello world again")) is None


def test_empty_blocklist_blocks_nothing():
    f = ThresholdHallucinationFilter(FilterConfig(blocklist=()))
    assert f.evaluate(stats("Thank you for watching!")) is None


# ---------------------------------------------------------------------------
# set_config
# ---------------------------------------------------------------------------


def test_set_config_replaces_blocklist(filt):
    assert filt.evaluate(stats("Thank you for watching")) == "blocklist"
    filt.set_config(FilterConfig(blocklist=("good morning", "hello there*")))
    assert filt.evaluate(stats("Thank you for watching")) is None
    assert filt.evaluate(stats("Good morning!")) == "blocklist"
    assert filt.evaluate(stats("Hello there, friend")) == "blocklist"


def test_set_config_replaces_thresholds(filt):
    assert filt.evaluate(stats(compression_ratio=2.5)) == "compression"
    assert filt.evaluate(stats(no_speech_prob=0.7, avg_logprob=-1.2)) == "no_speech"
    filt.set_config(FilterConfig(compression_ratio_threshold=3.0, no_speech_prob_threshold=0.8,
                                 avg_logprob_threshold=-2.0))
    assert filt.evaluate(stats(compression_ratio=2.5)) is None
    assert filt.evaluate(stats(compression_ratio=3.01)) == "compression"
    assert filt.evaluate(stats(no_speech_prob=0.7, avg_logprob=-1.2)) is None
    assert filt.evaluate(stats(no_speech_prob=0.81, avg_logprob=-2.01)) == "no_speech"


def test_constructor_config_is_used():
    f = ThresholdHallucinationFilter(FilterConfig(compression_ratio_threshold=1.5))
    assert f.evaluate(stats(compression_ratio=1.6)) == "compression"


def test_evaluate_is_pure(filt):
    s = stats("Thank you for watching!")
    assert filt.evaluate(s) == filt.evaluate(s) == "blocklist"
    assert filt.evaluate(stats()) is None


def test_concurrent_evaluate_and_set_config():
    f = ThresholdHallucinationFilter(FilterConfig())
    errors: list[BaseException] = []
    stop = threading.Event()

    def reader():
        try:
            while not stop.is_set():
                r = f.evaluate(stats("Hello, how are you today?"))
                assert r is None
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    for i in range(200):
        f.set_config(FilterConfig(blocklist=(f"entry {i}",), compression_ratio_threshold=2.0 + i / 100))
    stop.set()
    for t in threads:
        t.join(timeout=5)
    assert errors == []
