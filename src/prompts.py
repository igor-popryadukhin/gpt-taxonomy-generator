from __future__ import annotations

from typing import Sequence

SYSTEM_PROMPT = (
    "Ты — помощник по таксономии. Говоришь кратко. Отвечаешь только списком вариантов "
    "для следующего уровня категорий на русском, без нумерации и лишних символов. Не "
    "повторяй уже существующие варианты."
)

_LEVEL_TEMPLATES = [
    (
        "Корневая цепочка: {parent}\n"
        "Нужны {n} новых подкатегорий для следующего уровня (только названия).\n"
        "Язык: {lang}\n"
        "Избегай дублей и слишком общих слов (\"Другое\", \"Разное\"), если не достигнут последний уровень.\n"
        "Существующие подкатегории (игнорируй их, не повторяй): {existing}"
    ),
    (
        "Текущий путь: {parent}.\n"
        "Добавь {n} свежих и уникальных идей для следующего уровня каталога. Только названия.\n"
        "Язык ответов — {lang}.\n"
        "Не используй слишком общие формулировки (\"Прочее\", \"Разное\"), пока не достигнут финальный уровень.\n"
        "Уже занятые подкатегории: {existing}"
    ),
    (
        "Путь категорий: {parent}.\n"
        "Предложи {n} новых подкатегорий. Нужны только короткие названия без нумерации.\n"
        "Ответь на {lang}.\n"
        "Пропускай любые повторы существующих вариантов: {existing}."
    ),
]

_FINAL_LEVEL_TEMPLATES = [
    (
        "Цепочка: {parent}\n"
        "Нужны {n} конкретных услуг/видов работ.\n"
        "Только реалистичные, востребованные формулировки для коммерческого каталога."
    ),
    (
        "Контекст: {parent}.\n"
        "Опиши {n} прикладных услуг или работ, которые реально можно заказать. Краткие формулировки."
    ),
    (
        "Категория: {parent}.\n"
        "Предложи {n} чётких названий услуг/работ без вводных слов и нумерации."
    ),
]


def _format_existing(existing: Sequence[str]) -> str:
    if not existing:
        return "(нет)"
    return ", ".join(existing)


def build_user_prompt(
    parent_path: str,
    count: int,
    lang: str,
    existing: Sequence[str],
    final_level: bool,
    attempt: int,
) -> str:
    templates = _FINAL_LEVEL_TEMPLATES if final_level else _LEVEL_TEMPLATES
    index = min(attempt, len(templates) - 1)
    template = templates[index]
    return template.format(parent=parent_path, n=count, lang=lang, existing=_format_existing(existing))


__all__ = ["SYSTEM_PROMPT", "build_user_prompt"]
