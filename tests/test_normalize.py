from src.normalize import Normalizer


def test_normalize_name_and_slug():
    normalizer = Normalizer()
    assert normalizer.normalize_name("  Монтаж фасада (услуги)  ") == "Монтаж фасада"
    assert normalizer.normalize_name("Отделка фасада — услуги") == "Отделка фасада"
    assert normalizer.slug("Сантехника и отопление") == "сантехника-и-отопление"


def test_stop_words_filtering():
    normalizer = Normalizer()
    assert normalizer.deduplicate(["Разное", "Прочее"], final_level=False) == []
    assert normalizer.deduplicate(["Разное"], final_level=True) == ["Разное"]


def test_deduplicate_by_canonical_form():
    normalizer = Normalizer()
    variants = [
        "Монтаж дверей",
        "монтаж дверей",
        "Монтаж дверей.",
        "Установка дверей",
    ]
    result = normalizer.deduplicate(variants, final_level=True)
    assert result[0] == "Монтаж дверей"
    assert "Установка дверей" in result
    assert len(result) == 2
