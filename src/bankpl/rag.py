"""Retrieval over the bank's terms (BM25, no external services).

The terms are a handful of short Markdown documents, split at ``##`` headings so that each
chunk answers one kind of question and carries a stable citation id such as
``tabela-oplat#przelewy``. BM25 with light Polish normalisation (lower case, diacritics
folded, crude suffix stripping) is enough at this size and keeps the eval deterministic.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from importlib import resources

SUFFIXES = sorted(
    "ami ach ach ego emu ymi ich ów om ie ia iu ów ej ą ę a e i o u y".split(),
    key=len,
    reverse=True,
)
STOP = set("i w na z do o a się nie jest to że po od za dla przy lub oraz czy jak co".split())


def fold(s: str) -> str:
    s = s.lower().replace("ł", "l")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"\w+", fold(text)):
        if w in STOP:
            continue
        for suf in SUFFIXES:
            suf = fold(suf)
            if len(w) > len(suf) + 3 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w[:7])  # prefix stemming: "przelewu", "przelewy" -> "przelew"
    return out


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", fold(s)).strip("-")


@dataclass
class Chunk:
    id: str
    doc: str
    heading: str
    text: str


def load_chunks() -> list[Chunk]:
    chunks = []
    for f in sorted(resources.files("bankpl").joinpath("terms").iterdir()):
        if not f.name.endswith(".md"):
            continue
        doc = f.name[:-3]
        for part in re.split(r"^## ", f.read_text("utf-8"), flags=re.M)[1:]:
            heading, _, body = part.partition("\n")
            chunks.append(Chunk(f"{doc}#{slug(heading)}", doc, heading.strip(), body.strip()))
    return chunks


class Index:
    def __init__(self, chunks: list[Chunk] | None = None, k1: float = 1.4, b: float = 0.7):
        self.chunks = chunks or load_chunks()
        self.k1, self.b = k1, b
        self.docs = [
            Counter(tokens(c.heading + " " + c.heading + " " + c.text)) for c in self.chunks
        ]
        self.avgdl = sum(sum(d.values()) for d in self.docs) / len(self.docs)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = tokens(query)
        scores = []
        for c, d in zip(self.chunks, self.docs, strict=True):
            dl = sum(d.values())
            s = sum(
                self.idf.get(t, 0)
                * d[t]
                * (self.k1 + 1)
                / (d[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                for t in q
                if t in d
            )
            scores.append((c, s))
        scores.sort(key=lambda cs: -cs[1])
        return [(c, s) for c, s in scores[:k] if s > 0]
