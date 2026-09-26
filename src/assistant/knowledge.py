"""Lightweight retrieval over Markdown knowledge documents (TF-IDF + cosine, no external services)."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOPWORDS = set("a an and are as at be by for from has have how in is it its of on or that the this to was were what "
                "when which who why with should would could can do does not no any all each per than then their there "
                "these those into over under about batch".split())
GENERIC_SECTIONS = {"Purpose and scope"}     # boilerplate: demoted so actionable sections rank first
_SUFFIXES = ("ification", "ify", "ations", "ation", "ings", "ing", "ate", "ed", "es", "s", "e")


def tokenize(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOPWORDS:
            continue
        for suf in _SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    doc_title: str
    section: str
    text: str
    path: str
    note: str = ""          # e.g. the SYNTHETIC banner line of the document

    @property
    def label(self) -> str:
        return f"{self.doc_title} > {self.section}"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def load_chunks(directory) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).glob("*.md")):
        title, note, section, buf = path.stem.replace("_", " ").title(), "", None, []

        def flush():
            body = "\n".join(buf).strip()
            if section and body:
                chunks.append(Chunk(path.stem, title, section, body, str(path), note))

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith(">"):
                note = line.lstrip("> ").strip()
            elif line.startswith("## "):
                flush()
                section, buf = line[3:].strip(), []
            elif section is not None:
                buf.append(line)
        flush()
    return chunks


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        docs = [tokenize(f"{c.doc_title} {c.section} {c.section} {c.text}") for c in chunks]
        df = Counter(t for d in docs for t in set(d))
        n = len(chunks)
        self.idf = {t: math.log((1 + n) / (1 + f)) + 1 for t, f in df.items()}
        self.vecs = [self._vector(d) for d in docs]

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        v = {t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in tf.items() if t in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(self, query: str, top_k: int = 5, per_doc: int = 2, min_score: float = 0.05) -> list[Hit]:
        q = self._vector(tokenize(query))
        scored = sorted(((sum(w * v.get(t, 0.0) for t, w in q.items()) * (0.5 if self.chunks[i].section in GENERIC_SECTIONS else 1.0), i)
                         for i, v in enumerate(self.vecs)), reverse=True)
        hits, per = [], Counter()
        for s, i in scored:                      # already sorted by score, best first
            c = self.chunks[i]
            if s < min_score or per[c.doc_id] >= per_doc:
                continue
            per[c.doc_id] += 1
            hits.append(Hit(c, round(float(s), 3)))
            if len(hits) == top_k:
                break
        return hits

    @classmethod
    def from_directory(cls, directory) -> "KnowledgeBase":
        return cls(load_chunks(directory))
