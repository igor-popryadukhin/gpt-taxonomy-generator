from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from rapidfuzz.distance import Levenshtein
from razdel import tokenize

try:  # pragma: no cover - heavy dependency, exercised in integration tests
    from pymorphy2 import MorphAnalyzer
except Exception:  # pragma: no cover
    MorphAnalyzer = None  # type: ignore

_logger = logging.getLogger(__name__)

_STOP_WORDS = {
    "прочее",
    "прочие",
    "другое",
    "разное",
    "прочее и разное",
    "без категории",
    "общее",
    "другое направление",
}

_EMPTY_TOKENS = {"услуги", "работы", "товары"}


@dataclass(frozen=True)
class NormalizedName:
    display: str
    slug: str
    canonical: str


class Normalizer:
    def __init__(self) -> None:
        self._morph: Optional[MorphAnalyzer] = None
        if MorphAnalyzer is not None:
            try:
                self._morph = MorphAnalyzer()
            except Exception:  # pragma: no cover
                _logger.warning("Не удалось инициализировать pymorphy2, продолжаем без лемматизации")
                self._morph = None

    @staticmethod
    def normalize_name(value: str) -> str:
        text = value.strip()
        text = text.strip("/")
        text = re.sub(r"\s+/\s+", " / ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[.]+$", "", text)
        text = re.sub(r"\s*[–—-]\s*услуги$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*\(([^)]*)\)\s*$", "", text)
        text = text.replace("ё", "е").replace("Ё", "Е")
        return text.strip()

    def is_stop_word(self, value: str) -> bool:
        return value.lower() in _STOP_WORDS

    def slug(self, value: str) -> str:
        lowered = value.lower()
        cleaned = re.sub(r"[^0-9a-zа-я]+", "-", lowered)
        cleaned = re.sub(r"-+", "-", cleaned).strip("-")
        return cleaned

    def canonical_form(self, value: str) -> str:
        text = value.lower()
        text = text.replace("ё", "е")
        tokens = []
        for token in tokenize(text):
            if not token.text.strip():
                continue
            pure = re.sub(r"[^0-9a-zа-я]", "", token.text)
            if not pure:
                continue
            if pure in _EMPTY_TOKENS:
                continue
            if self._morph:
                try:
                    lemma = self._morph.parse(pure)[0].normal_form
                except Exception:  # pragma: no cover
                    lemma = pure
            else:
                lemma = pure
            tokens.append(lemma)
        return " ".join(tokens)

    def build(self, value: str, final_level: bool) -> Optional[NormalizedName]:
        display = self.normalize_name(value)
        if not display:
            return None
        if not final_level and self.is_stop_word(display):
            return None
        slug = self.slug(display)
        canonical = self.canonical_form(display)
        if not canonical:
            canonical = slug
        return NormalizedName(display=display, slug=slug, canonical=canonical)

    def are_similar(self, a: NormalizedName, b: NormalizedName) -> bool:
        if a.slug == b.slug:
            return True
        if a.canonical == b.canonical:
            return True
        distance = Levenshtein.normalized_distance(a.canonical, b.canonical)
        return distance < 0.2

    def path_key(self, parts: Sequence[str]) -> str:
        canonical_parts = [self.canonical_form(self.normalize_name(part)) for part in parts]
        canonical_parts = [part or self.slug(self.normalize_name(p)) for part, p in zip(canonical_parts, parts)]
        return " / ".join(canonical_parts)

    def deduplicate(self, items: Iterable[str], final_level: bool) -> list[str]:
        result: list[str] = []
        prepared: list[NormalizedName] = []
        for item in items:
            normalized = self.build(item, final_level)
            if not normalized:
                continue
            duplicate = any(self.are_similar(normalized, existing) for existing in prepared)
            if duplicate:
                continue
            prepared.append(normalized)
            result.append(normalized.display)
        return result


__all__ = ["Normalizer", "NormalizedName"]
