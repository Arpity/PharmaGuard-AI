"""Deterministic question -> intent classification (no LLM needed)."""
from __future__ import annotations

INTENT_KEYWORDS = {
    "compare": ["compare", "comparison", "historical", "history", "similar", "peer", "versus", " vs ", "typical",
                "previous batches", "other batches"],
    "abnormal": ["abnormal", "outlier", "anomal", "out of spec", "out-of-spec", "oos", "which parameter", "parameters",
                 "unusual", "off-spec"],
    "procedure": ["procedure", "sop", "qa review", "what should", "next step", "action", "recommend", "capa",
                  "guideline", "how to", "follow", "checklist", "document"],
    "why_risk": ["why", "risk", "reason", "cause", "driver", "flag", "showing", "status"],
    "investigate": ["investigat", "overview", "summar", "tell me about", "analy", "review this batch", "full"],
}
ALL_INTENTS = ["investigate", "why_risk", "compare", "abnormal", "procedure"]


def classify(question: str) -> list[str]:
    q = f" {question.lower()} "
    found = [i for i, kws in INTENT_KEYWORDS.items() if any(k in q for k in kws)]
    return found or ["investigate"]


def sections_for(intents: list[str]) -> list[str]:
    """Which answer sections to produce. 'investigate' means everything."""
    if "investigate" in intents:
        return ["why_risk", "abnormal", "compare", "procedure"]
    return [i for i in ALL_INTENTS if i in intents and i != "investigate"]
