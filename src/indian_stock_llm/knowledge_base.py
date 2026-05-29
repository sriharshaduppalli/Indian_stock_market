from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class KnowledgeItem:
    id: str
    title: str
    content: str
    tags: list[str]
    source: str


_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


class KnowledgeBase:
    def __init__(self, items: Iterable[KnowledgeItem]):
        self.items = list(items)

    @classmethod
    def from_json(cls, path: Path) -> "KnowledgeBase":
        data = json.loads(path.read_text(encoding="utf-8"))
        items = [KnowledgeItem(**item) for item in data]
        return cls(items)

    def search(self, query: str, top_k: int = 3) -> List[KnowledgeItem]:
        query_tokens = _tokenize(query)
        scored = []
        for item in self.items:
            score = len(query_tokens & _tokenize(f"{item.title} {item.content} {' '.join(item.tags)}"))
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]
