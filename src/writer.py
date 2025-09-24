from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .utils import ensure_directory


def write_categories(paths: Sequence[Sequence[str]], output: Path) -> None:
    ensure_directory(output.parent)
    lines = [" / ".join(parts) for parts in sorted(paths)]
    unique: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line in seen:
            continue
        unique.append(line)
        seen.add(line)
    text = "\n".join(unique)
    if text:
        text += "\n"
    output.write_text(text, encoding="utf-8")


def write_report(report: dict, path: Path) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["write_categories", "write_report"]
