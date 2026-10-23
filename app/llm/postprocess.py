"""Guards around the LLM output for prompt adjustment.

clean_output(text, original): removes what a chat model tends to add around the answer
  (an outer ```markdown fence, a "Here is the rewritten prompt:" line, the draft markers).
check_fidelity(original, output): cheap, deterministic check that the rewrite kept the
  information and added nothing. It can't prove meaning is preserved; it catches the
  common failures: dropped numbers/URLs/paths/identifiers/code, answering the prompt
  instead of restructuring it, and summarising it away.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from app.llm.prompt_text import DRAFT_END, DRAFT_START

#: Thresholds for the soft (word-level) checks.
MIN_COVERAGE = 0.85  # share of the original's content words that must reappear
MAX_NOVELTY = 0.30  # share of the output's content words that may be new (typo fixes, headings)
MIN_LENGTH_RATIO = 0.6
MAX_LENGTH_RATIO = 2.5
#: Below this many content words the soft ratios are too noisy; only hard tokens are checked.
MIN_WORDS_FOR_SOFT_CHECK = 8

_OUTER_FENCE = re.compile(r"\A```[ \t]*(markdown|md|text|txt)?[ \t]*\n(?P<body>.*)\n```[ \t]*\Z",
                          re.DOTALL | re.IGNORECASE)
_PREAMBLE = re.compile(
    r"^\s*(here(?:'s| is| are)|sure|certainly|of course|below is|iat[ăa]|aici (?:este|ai)|sigur|desigur)"
    r"\b[^\n]{0,120}:\s*$",
    re.IGNORECASE,
)

_URL = re.compile(r"https?://[^\s<>()\"'`]+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PATH = re.compile(r"(?<![\w/\\])(?:[A-Za-z]:\\|\.{0,2}/|~/)?[\w.-]+(?:[/\\][\w.-]+)+[/\\]?")
_FLAG = re.compile(r"(?<!\w)--?[A-Za-z][\w-]*")
_IDENT = re.compile(
    r"\b(?:[A-Za-z_]\w*_\w+"  # snake_case
    r"|[a-z]+[A-Z]\w*"  # camelCase
    r"|[A-Z][a-z]+[A-Z]\w*"  # PascalCase with an inner capital
    r"|\w+(?:\.\w+)+)\b"  # dotted: file.py, a.b.c, 2.5
)
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,:]\d+)*")
_BACKTICK = re.compile(r"`([^`\n]+)`")
_FENCE_LINE = re.compile(r"^\s*(```|~~~)")
_WORD = re.compile(r"\w{3,}", re.UNICODE)

_STOPWORDS_RAW = (
    """
    the and for are but not you your with this that from have has was were will would should could
    can into about than then them they their there these those what when where which while who whom
    why how all any each few more most other some such only own same too very just also its our out
    over under again further once here both off very been being does did doing get got use using
    și sau dar nu cu de la pe în din pentru este sunt care ce cum când unde mai foarte doar
    dacă iar fie fiecare acest această aceste acești asta aia lui lor sau sub prin spre după
    """.split()
)
#: Words the structure itself introduces (section headings, EN + RO); never count as novel.
_STRUCTURE_WORDS_RAW = (
    """
    goal context requirements requirement constraints constraint input output format questions question
    scop context cerinte cerințe constrangeri constrângeri date intrare raspuns răspuns intrebari întrebări
    """.split()
)


@dataclass(frozen=True)
class FidelityReport:
    ok: bool
    missing: tuple[str, ...] = ()  # hard tokens of the original absent from the output
    coverage: float = 1.0
    novelty: float = 0.0
    length_ratio: float = 1.0
    reasons: tuple[str, ...] = field(default=())

    def summary(self) -> str:
        parts = list(self.reasons)
        if self.missing:
            shown = ", ".join(self.missing[:5]) + (" …" if len(self.missing) > 5 else "")
            parts.append(f"missing: {shown}")
        return "; ".join(parts) if parts else "ok"


def clean_output(text: str, original: str = "") -> str:
    out = text.strip().replace("\r\n", "\n")
    # Leftover draft markers.
    out = "\n".join(line for line in out.split("\n") if line.strip() not in (DRAFT_START, DRAFT_END)).strip()
    # A chatty first line ending in ":" ("Here is the rewritten prompt:", "Iată promptul:"),
    # unless the draft itself starts like that.
    lines = out.split("\n")
    if len(lines) > 1 and _PREAMBLE.match(lines[0]) and not _PREAMBLE.match(original.strip().split("\n")[0]):
        out = "\n".join(lines[1:]).strip()
    # One outer ```markdown fence around everything (unless the draft itself was one block).
    m = _OUTER_FENCE.match(out)
    if m and not original.lstrip().startswith("```"):
        out = m.group("body").strip()
    return out


def check_fidelity(original: str, output: str) -> FidelityReport:
    out_norm = _normalize_for_match(output)
    missing = tuple(t for t in hard_tokens(original) if _normalize_for_match(t) not in out_norm)

    orig_words = _content_words(original)
    out_words = _content_words(output)
    coverage = len(orig_words & out_words) / len(orig_words) if orig_words else 1.0
    novel = out_words - orig_words - _STRUCTURE_WORDS
    novelty = len(novel) / len(out_words) if out_words else 0.0
    length_ratio = len(output.strip()) / max(len(original.strip()), 1)

    reasons: list[str] = []
    if not output.strip():
        reasons.append("empty output")
    if len(orig_words) >= MIN_WORDS_FOR_SOFT_CHECK:
        if coverage < MIN_COVERAGE:
            reasons.append(f"only {coverage:.0%} of the original words kept")
        if novelty > MAX_NOVELTY:
            reasons.append(f"{novelty:.0%} new words added")
        if not MIN_LENGTH_RATIO <= length_ratio <= MAX_LENGTH_RATIO:
            reasons.append(f"length changed {length_ratio:.1f}x")
    ok = not reasons and not missing
    return FidelityReport(ok=ok, missing=missing, coverage=coverage, novelty=novelty,
                          length_ratio=length_ratio, reasons=tuple(reasons))


def hard_tokens(text: str) -> list[str]:
    """Tokens that must appear verbatim in the rewrite: URLs, emails, paths, flags,
    identifiers, numbers, `code spans` and every non-blank line inside fenced code."""
    found: list[str] = []
    seen: set[str] = set()

    def add(token: str, verbatim: bool = False) -> None:
        token = token.strip()
        if not verbatim:  # sentence punctuation glued to a prose token is not part of it
            token = token.rstrip(".,;:!?")
            while token.endswith(")") and token.count(")") > token.count("("):
                token = token[:-1].rstrip(".,;:!?")
        key = _normalize_for_match(token)
        if len(key) >= 1 and key not in seen:
            seen.add(key)
            found.append(token)

    prose_lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE_LINE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            if line.strip():
                add(line, verbatim=True)
        else:
            prose_lines.append(line)
    prose = "\n".join(prose_lines)

    for m in _BACKTICK.finditer(prose):
        add(m.group(1), verbatim=True)
    prose = _BACKTICK.sub(" ", prose)
    for rx in (_URL, _EMAIL):
        for m in rx.finditer(prose):
            add(m.group(0))
        prose = rx.sub(" ", prose)
    for rx in (_PATH, _FLAG, _IDENT, _NUMBER):
        for m in rx.finditer(prose):
            add(m.group(0))
    return found


# --- helpers ---------------------------------------------------------------------------


def _fold(text: str) -> str:
    """casefold + strip diacritics (a draft typed without ș/ț still matches a rewrite that has them)."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _normalize_for_match(text: str) -> str:
    # Markdown escapes (snake\_case) and soft formatting must not hide a token.
    text = re.sub(r"\\([_*`#\[\]()>\-])", r"\1", text)
    return " ".join(_fold(text).split())


_STOPWORDS = frozenset(_fold(w) for w in _STOPWORDS_RAW)
_STRUCTURE_WORDS = frozenset(_fold(w) for w in _STRUCTURE_WORDS_RAW)


def _content_words(text: str) -> set[str]:
    words = {_fold(w) for w in _WORD.findall(text)}
    return {w for w in words if w not in _STOPWORDS and not w.isdigit()}
