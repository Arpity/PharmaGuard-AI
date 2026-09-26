"""Numeric grounding check: every number in an LLM answer must appear in the facts it was given."""
from __future__ import annotations

import re

_CITATION = re.compile(r"\[(?:K|D)\d+\]")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[str]:
    text = _DATE.sub(" ", _CITATION.sub(" ", text))
    return [m.group(0).replace(",", "").lstrip("+") for m in _NUM.finditer(text)]


def _decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def ungrounded_numbers(answer: str, *fact_texts: str) -> list[str]:
    """Numbers in `answer` that are not present (up to rounding / sign) in the supplied facts.
    Small whole numbers (<= 10, e.g. list counts) are ignored."""
    allowed = []
    for t in fact_texts:
        allowed += [float(x) for x in _numbers(t)]
    bad = []
    for s in _numbers(answer):
        v = float(s)
        if v == int(v) and "." not in s and abs(v) <= 10:
            continue
        tol = 0.5 * 10 ** -_decimals(s) + 1e-9
        if not any(abs(abs(v) - abs(a)) <= tol for a in allowed):
            bad.append(s)
    return list(dict.fromkeys(bad))
