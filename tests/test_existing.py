from pathlib import Path

from src.existing import ExistingCategories
from src.normalize import Normalizer


def test_existing_loader(tmp_path: Path) -> None:
    data = """
    Ремонт / Отделка / Покраска стен
    Ремонт/Отделка/Штукатурка
    Ремонт / Отделка / Покраска стен
    
    # комментарий
    """.strip()
    source = tmp_path / "existing.txt"
    source.write_text(data, encoding="utf-8")

    normalizer = Normalizer()
    store = ExistingCategories(normalizer)
    store.load([source])

    level1 = store.children_for(["Ремонт"])
    assert level1 == ["Отделка"]

    level2 = store.children_for(["Ремонт", "Отделка"])
    assert sorted(level2) == ["Покраска стен", "Штукатурка"]
    assert store.total_loaded == 2
