"""Output validation and unsupported-claim detection for LLM answers.

severity 'block' -> the answer is discarded and replaced by the deterministic summary.
severity 'warn'  -> the answer is shown, flagged 'needs verification'."""
from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field

from src.assistant.grounding import ungrounded_numbers

from .safe_errors import redact

CANARY = "PGCANARY-" + secrets.token_hex(4)          # planted in the system prompt; must never appear in output

_DISPOSITION = [re.compile(p, re.I) for p in [
    r"\b(i|we)\s+(hereby\s+|will\s+|would\s+|can\s+|do\s+)?(approve|reject|release|disposition|quarantine|certify)\b",
    r"\b(this\s+batch|the\s+batch|batch\s+b\d{5}|b\d{5})\s+(is|has\s+been|was|will\s+be|can\s+be|should\s+be|must\s+be|may\s+be)\s+(hereby\s+)?(approved|released|rejected|dispositioned|cleared|quarantined|passed)\b",
    r"\b(safe|fit|ok|okay|cleared?)\s+(to|for)\s+(release|distribution|market|shipment|use)\b",
    r"\b(recommend|advise|suggest|conclude)\w*\s+(that\s+)?(you\s+|qa\s+|we\s+)?(to\s+)?(release|reject|approve)\s+(this|the)\s+batch",
    r"\b(disposition|decision|verdict|status)\s*(is|:|-)\s*(release|reject|approve)d?\b",
    r"\bfinal\s+(decision|disposition)\s*(is|:)\s*",
    r"^\W*(approved|released|rejected|cleared|passed)\W*$",                       # the whole answer is just a verdict word
    r"\b(status|result)\s*(is|:|-)\s*(approved|released|cleared)\b",
]]
_CAUSAL = re.compile(r"\b(root\s+cause\s+(is|was)|definitely|certainly|undoubtedly|is\s+caused\s+by|was\s+caused\s+by|proves?\s+that)\b", re.I)
_CITATION = re.compile(r"\[K(\d+)\]")
_BATCH_ID = re.compile(r"\bB\d{5}\b", re.I)
_SOP_ID = re.compile(r"\bSOP-[A-Z]{2,4}-\d{2,4}\b", re.I)
_PROCEDURE_WORD = re.compile(r"\b(sop|procedure|guideline)\b", re.I)


@dataclass
class Violation:
    code: str
    severity: str          # "block" | "warn"
    detail: str


@dataclass
class OutputCheck:
    answer: str
    violations: list[Violation] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(v.severity == "block" for v in self.violations)

    @property
    def needs_verification(self) -> bool:
        return any(v.severity == "warn" for v in self.violations)


def validate_output(answer: str, facts: list[str], n_sources: int, max_chars: int = 3000,
                    block_numbers: bool = True) -> OutputCheck:
    v: list[Violation] = []
    text = answer or ""
    if not text.strip():
        return OutputCheck("", [Violation("EMPTY_ANSWER", "block", "The model returned no content.")])
    if len(text) > max_chars:
        v.append(Violation("ANSWER_TOO_LONG", "block", f"{len(text)} characters exceeds {max_chars}."))
    if CANARY in text:
        v.append(Violation("SYSTEM_PROMPT_LEAK", "block", "Answer contains internal prompt markers."))
    redacted = redact(text)
    if redacted != text:
        v.append(Violation("SECRET_REDACTED", "warn", "A credential-like string was removed from the answer."))
        text = redacted
    for p in _DISPOSITION:
        m = p.search(text)
        if m:
            v.append(Violation("DISPOSITION_CLAIM", "block", f"Answer appears to make a batch decision: '{m.group(0)}'."))
            break

    fact_blob = " ".join(facts)
    bad_nums = ungrounded_numbers(text, *facts)
    if bad_nums:
        v.append(Violation("UNGROUNDED_NUMBER", "block" if block_numbers else "warn",
                           "Numbers not found in the calculated analysis or sources: " + ", ".join(bad_nums)))
    cited = {int(n) for n in _CITATION.findall(text)}
    bad_cites = sorted(n for n in cited if not 1 <= n <= n_sources)
    if bad_cites:
        v.append(Violation("INVALID_CITATION", "block", "Cites sources that were not retrieved: " +
                           ", ".join(f"[K{n}]" for n in bad_cites)))
    known_ids = {x.upper() for x in _BATCH_ID.findall(fact_blob)}
    unknown_ids = sorted({x.upper() for x in _BATCH_ID.findall(text)} - known_ids)
    if unknown_ids:
        v.append(Violation("UNKNOWN_BATCH_REFERENCE", "block", "References batches not in the analysis: " + ", ".join(unknown_ids)))
    known_sops = {x.upper() for x in _SOP_ID.findall(fact_blob)}
    unknown_sops = sorted({x.upper() for x in _SOP_ID.findall(text)} - known_sops)
    if unknown_sops:
        v.append(Violation("UNKNOWN_PROCEDURE_REFERENCE", "block", "References procedures that were not retrieved: " + ", ".join(unknown_sops)))
    if _CAUSAL.search(text):
        v.append(Violation("UNSUPPORTED_CAUSAL_CLAIM", "warn", "States a cause or certainty the data does not establish."))
    if n_sources and _PROCEDURE_WORD.search(text) and not cited:
        v.append(Violation("MISSING_CITATION", "warn", "Mentions procedures without citing a retrieved source."))
    return OutputCheck(text, v)
