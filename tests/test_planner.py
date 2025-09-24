from pathlib import Path
from typing import Dict, List

import pytest

from src.config import AppConfig
from src.deepseek import ModelResponse
from src.existing import ExistingCategories
from src.normalize import Normalizer
from src.planner import CategoryPlanner


class FakeClient:
    def __init__(self, mapping: Dict[str, List[str]]) -> None:
        self.mapping = mapping
        self.calls = 0

    async def generate(self, parent_path: str, count: int, lang: str, existing, final_level: bool, attempt: int) -> ModelResponse:  # type: ignore[override]
        self.calls += 1
        variants = self.mapping.get(parent_path, [])[:count]
        return ModelResponse(candidates=variants, raw="\n".join(variants), usage={"prompt_tokens": 10, "completion_tokens": len(variants) * 5})

    def stats(self) -> Dict[str, object]:
        return {"requests": self.calls, "retries": 0, "failures": 0, "tokens": self.calls * 15, "usage": []}


@pytest.mark.asyncio
async def test_planner_builds_complete_tree(tmp_path: Path) -> None:
    mapping = {
        "Дизайнеры": ["Интерьеры", "Графика", "Конструкции"],
        "Дизайнеры / Интерьеры": [
            "Дизайн квартир",
            "Дизайн домов",
            "Дизайн офисов",
        ],
        "Дизайнеры / Графика": ["Логотипы", "Айдентика", "Иллюстрации"],
        "Дизайнеры / Конструкции": ["Стенды", "Оборудование", "Витрины"],
    }

    config = AppConfig(
        root="Дизайнеры",
        depth=3,
        breadth=3,
        out=tmp_path / "categories.txt",
        cache_dir=tmp_path / "cache",
        report=tmp_path / "report.json",
        log_file=tmp_path / "logs/app.log",
        dry_run=True,
    )

    normalizer = Normalizer()
    existing_store = ExistingCategories(normalizer)
    client = FakeClient(mapping)
    planner = CategoryPlanner(config, client, normalizer, existing_store)
    result = await planner.build()

    assert len(result.paths) == 9
    assert all(len(path) == config.depth for path in result.paths)
    assert result.stats["depth_reached"] == config.depth
    assert result.stats["existing_used"] == 0
