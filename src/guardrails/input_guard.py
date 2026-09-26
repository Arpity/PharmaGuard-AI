"""Input validation, prompt-injection detection and restricted-action detection.

Pattern matching is a first line of defence, not a guarantee (see README limitations). It is
layered with: delimiting user text as data in the prompt, output validation, and a permission
model in which the AI has no capability to change batch status at all."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .policy import DISPOSITION_REFUSAL, INJECTION_REFUSAL

BATCH_ID_RE = re.compile(r"^B\d{5}$")
SAFE_QUESTION = "Investigate this batch."
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b‌‍⁠﻿­"), None)

_INJECTION = [re.compile(p, re.I) for p in [
    r"\b(ignore|disregard|forget|override|bypass|circumvent|disable|turn\s+off)\b.{0,30}\b(previous|prior|above|earlier|preceding|all|any|your|these|the|safety)\b.{0,20}\b(instructions?|prompts?|rules?|guidelines?|guardrails?|safeguards?|restrictions?|filters?|policy|policies)\b",
    r"\b(reveal|show|print|repeat|output|leak|display|tell)\b.{0,25}\b(system|hidden|initial|developer|internal)\b.{0,10}\b(prompt|instructions?|message)\b",
    r"\b(system|developer)\s+prompt\b",
    r"\b(api[\s_-]?key|password|secret|credentials?|access\s+token)s?\b.{0,40}\b(show|reveal|print|give|tell|display|what)\b|\b(show|reveal|print|give|tell|display|what.{0,12}(is|are))\b.{0,40}\b(api[\s_-]?key|password|secret|credentials?)s?\b",
    r"\byou\s+are\s+now\b|\bfrom\s+now\s+on\s+you\b|\bpretend\s+(to\s+be|you\s+are)\b|\broleplay\s+as\b|\bact\s+as\s+(an?\s+)?(unrestricted|different|new|dan|admin|developer|root)\b",
    r"\b(jailbreak|dan\s+mode|developer\s+mode|do\s+anything\s+now)\b",
    r"</?\s*(system|assistant|instructions?)\s*>|\[/?(system|inst)\]|<\|im_(start|end)\|>|#{2,}\s*(system|instruction)",
    r"\bnew\s+(instructions?|rules?)\s*:",
    r"\b(disregard|ignore|forget)\b.{0,20}\b(everything|anything|all)\b.{0,20}\b(above|before|earlier|prior|said)\b",
    r"\brespond\s+(only\s+)?with\s+[\"']?(approved|released|pass|passed|ok)\b",
    r"\bwhat\s+(were|are)\s+you\s+(told|instructed|given)\b|\b(initial|original|hidden)\s+(prompt|instructions?)\b",
]]
_SQUASHED = ["ignorepreviousinstructions", "ignoreallinstructions", "ignoretheaboveinstructions",
             "disregardpreviousinstructions", "disregardallinstructions", "revealsystemprompt",
             "showsystemprompt", "developermode", "jailbreak"]

_VERBS = (r"(approv\w*|reject\w*|releas\w*|dispositio\w*|certif\w*|sign[\s-]?off|quarantin\w*|recall\w*|scrap\w*|"
          r"destroy\w*|discard\w*|clear(?:ed)?\s+for)")
_OBJ = r"(batch|lot|product|it|this|b\d{5})"
_RESTRICTED = re.compile(rf"\b{_VERBS}\b.{{0,40}}\b{_OBJ}\b|\b{_OBJ}\b.{{0,40}}\b{_VERBS}\b", re.I)
_RESTRICTED_EXTRA = [re.compile(p, re.I) for p in [
    r"\b(ship\w*|distribut\w*|accept\w*|green[\s-]?light|greenlight)\b.{0,30}\b(batch|lot|it|this|b\d{5})\b|\b(batch|lot|it|this|b\d{5})\b.{0,30}\b(ship\w*|distribut\w*|accept\w*|green[\s-]?light)\b",
    r"\b(pass|fail|clear|release|approve|reject)\s+(this|the|that)\s+(batch|lot)\b",
    r"\bgood\s+to\s+go\b|\bgo(?:es)?\s+to\s+market\b|\bsafe\s+to\s+(?:ship|sell|use|distribute)\b|\bfit\s+for\s+(?:sale|distribution|release)\b",
    r"\boverride\b.{0,30}\b(status|result|decision|flag|deviation)\b",
]]
_MUTATE = re.compile(r"\b(change|update|modify|edit|set|overwrite|delete|remove|alter|mark)\b.{0,30}"
                     r"\b(status|record|result|data|value|batch|quality)\b", re.I)
_EXEMPT = re.compile(r"\b(release|approval)\s+(criteria|checklist|requirements?|procedures?|sop|process|steps)\b", re.I)


@dataclass
class InputCheck:
    ok: bool
    sanitized: str
    question_for_llm: str
    message: str = ""
    injection: bool = False
    restricted: bool = False
    reasons: list[str] = field(default_factory=list)


def normalise(text: str, strip_tags: bool = True) -> str:
    t = unicodedata.normalize("NFKC", text or "").translate(_ZERO_WIDTH)
    t = "".join(ch for ch in t if ch in "\n\t " or unicodedata.category(ch)[0] != "C")
    if strip_tags:                                        # user text can never close our prompt delimiters
        t = t.replace("<", " ").replace(">", " ")
    return re.sub(r"\s+", " ", t).strip()


def validate_batch_id(batch_id: str) -> str | None:
    """Return the normalised id, or None if it is not B + 5 digits."""
    b = normalise(batch_id).upper()
    return b if BATCH_ID_RE.match(b) else None


# Look-alike folding used ONLY for detection (never applied to the text that is kept): common Cyrillic / Greek homoglyphs and leetspeak.
_FOLD = str.maketrans({"\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c", "\u0445": "x", "\u0456": "i", "\u0443": "y",
                       "\u03bf": "o", "\u03b1": "a", "\u03b5": "e", "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def detect_injection(text: str) -> bool:
    folded = text.translate(_FOLD)
    if any(p.search(text) or p.search(folded) for p in _INJECTION):
        return True
    squashed = re.sub(r"[^a-z]", "", folded.lower())
    return any(s in squashed for s in _SQUASHED)


def detect_restricted(text: str) -> bool:
    t = _EXEMPT.sub(" ", text)
    return bool(_RESTRICTED.search(t) or _MUTATE.search(t) or any(p.search(t) for p in _RESTRICTED_EXTRA))


def check_question(question: str, max_chars: int = 500) -> InputCheck:
    raw = question or ""
    if len(raw) > max_chars:
        return InputCheck(False, "", "", f"Question is too long ({len(raw)} characters; limit {max_chars}). Please shorten it.",
                          reasons=["too_long"])
    seen = normalise(raw, strip_tags=False)             # inspect BEFORE tags are stripped so </system> is caught
    q = normalise(raw)
    if not q:
        return InputCheck(False, "", "", "Please enter a question about the batch.", reasons=["empty"])
    if detect_injection(seen) or detect_injection(q):
        return InputCheck(False, q, "", INJECTION_REFUSAL, injection=True, reasons=["prompt_injection"])
    if detect_restricted(q):
        # Never forward a disposition request to the model: answer with the fixed refusal and a neutral investigation.
        return InputCheck(True, q, SAFE_QUESTION, DISPOSITION_REFUSAL, restricted=True, reasons=["restricted_action"])
    return InputCheck(True, q, q)
