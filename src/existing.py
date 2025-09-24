from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, Sequence

from .normalize import NormalizedName, Normalizer

_logger = logging.getLogger(__name__)


class ExistingCategories:
    def __init__(self, normalizer: Normalizer) -> None:
        self.normalizer = normalizer
        self.children: DefaultDict[str, list[str]] = defaultdict(list)
        self._child_norm: DefaultDict[str, list[NormalizedName]] = defaultdict(list)
        self.paths: set[str] = set()
        self.total_loaded = 0

    @staticmethod
    def _iter_files(path: Path) -> Iterable[Path]:
        if path.is_file():
            yield path
            return
        for file in sorted(path.rglob("*.txt")):
            if file.is_file():
                yield file

    def load(self, paths: Iterable[Path]) -> None:
        for path in paths:
            if not path.exists():
                _logger.warning("Файл existing не найден: %s", path)
                continue
            for file in self._iter_files(path):
                self._load_file(file)

    def _load_file(self, file: Path) -> None:
        try:
            text = file.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - rare IO error
            _logger.error("Не удалось прочитать %s: %s", file, exc)
            return
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [self.normalizer.normalize_name(part) for part in stripped.split("/") if part.strip()]
            if not parts:
                continue
            self._add_path(parts)

    def _add_path(self, parts: Sequence[str]) -> None:
        canonical_parts = []
        for part in parts:
            canonical = self.normalizer.canonical_form(part)
            if not canonical:
                canonical = self.normalizer.slug(part)
            canonical_parts.append(canonical)
        for depth in range(1, len(parts)):
            parent_key = " / ".join(canonical_parts[:depth])
            child_display = parts[depth]
            child_norm = self.normalizer.build(child_display, final_level=True)
            if not child_norm:
                continue
            if any(self.normalizer.are_similar(child_norm, existing) for existing in self._child_norm[parent_key]):
                continue
            self._child_norm[parent_key].append(child_norm)
            self.children[parent_key].append(child_norm.display)
        path_key = " / ".join(canonical_parts)
        if path_key not in self.paths:
            self.paths.add(path_key)
            self.total_loaded += 1

    def children_for(self, parent_parts: Sequence[str]) -> list[str]:
        key = self.normalizer.path_key(parent_parts)
        return list(self.children.get(key, []))

    def has_data(self) -> bool:
        return bool(self.children)


__all__ = ["ExistingCategories"]
