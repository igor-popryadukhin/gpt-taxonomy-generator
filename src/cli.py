from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional

import typer
from click.core import ParameterSource

from . import __version__
from .config import load_config
from .deepseek import DeepSeekClient
from .existing import ExistingCategories
from .normalize import Normalizer
from .planner import CategoryPlanner
from .utils import iso_now, setup_logging
from .writer import write_categories, write_report

app = typer.Typer(help="Генератор деревьев категорий с использованием DeepSeek")


def _collect_cli_options(ctx: typer.Context, values: Dict[str, object], option_names: List[str]) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    for name in option_names:
        if name not in values:
            continue
        source = ctx.get_parameter_source(name)
        if source in {ParameterSource.COMMANDLINE, ParameterSource.ENVIRONMENT}:
            value = values[name]
            if name == "existing" and not value:
                continue
            if isinstance(value, tuple):
                value = list(value)
            overrides[name] = value
    return overrides


@app.command()
def main(  # noqa: PLR0913 - many CLI options are expected
    ctx: typer.Context,
    root: str = typer.Option(..., "--root", help="Корневая категория"),
    depth: int = typer.Option(3, "--depth", min=2, max=8, help="Глубина дерева"),
    breadth: int = typer.Option(6, "--breadth", min=3, max=15, help="Целевое число подкатегорий"),
    existing: Optional[List[Path]] = typer.Option(
        None,
        "--existing",
        help="Путь к файлу или директории с известными категориями",
    ),
    lang: str = typer.Option("ru", "--lang", help="Язык генерации"),
    out: Path = typer.Option(Path("out/categories.txt"), "--out", help="Файл результата"),
    model: str = typer.Option("deepseek-chat", "--model", help="Название модели DeepSeek"),
    temperature: float = typer.Option(0.4, "--temperature", min=0.0, max=2.0, help="Температура выборки"),
    top_p: float = typer.Option(0.9, "--top-p", min=0.0, max=1.0, help="Порог nucleus sampling"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Фиксированный seed"),
    max_tokens: int = typer.Option(2048, "--max-tokens", help="Лимит токенов на ответ"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Прогон без записи", is_flag=True),
    resume: bool = typer.Option(False, "--resume", help="Продолжить из кэша", is_flag=True),
    dedupe: bool = typer.Option(True, "--dedupe/--no-dedupe", help="Включить дедупликацию"),
    concurrency: int = typer.Option(4, "--concurrency", help="Количество параллельных запросов"),
    rate_limit: float = typer.Option(1.0, "--rate-limit", help="Минимальный интервал между запросами"),
    token_budget: Optional[int] = typer.Option(None, "--token-budget", help="Лимит токенов на запуск"),
    model_config: Optional[Path] = typer.Option(None, "--config", help="Путь к конфигурационному файлу"),
    verbose: bool = typer.Option(False, "--verbose", help="Подробные логи", is_flag=True),
) -> None:
    start_time = time.perf_counter()
    cli_values = locals().copy()
    option_names = [
        "root",
        "depth",
        "breadth",
        "existing",
        "lang",
        "out",
        "model",
        "temperature",
        "top_p",
        "seed",
        "max_tokens",
        "dry_run",
        "resume",
        "dedupe",
        "concurrency",
        "rate_limit",
        "token_budget",
        "model_config",
    ]
    overrides = _collect_cli_options(ctx, cli_values, option_names)
    if model_config is not None:
        overrides["config"] = model_config

    config = load_config(overrides)
    setup_logging(config.log_file, verbose=verbose)
    typer.echo(f"Версия gpt-taxonomy: {__version__}")
    typer.echo(f"Корень: {config.root} | глубина: {config.depth} | ширина: {config.breadth}")

    normalizer = Normalizer()
    existing_store = ExistingCategories(normalizer)
    existing_store.load(config.existing)
    if existing_store.total_loaded:
        typer.echo(f"Загружено известных цепочек: {existing_store.total_loaded}")

    async def runner() -> tuple:
        async with DeepSeekClient(config) as client:
            planner = CategoryPlanner(config, client, normalizer, existing_store)
            plan_result = await planner.build()
            return plan_result, client.stats()

    started_at = iso_now()
    try:
        plan_result, api_stats = asyncio.run(runner())
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        typer.secho("Операция прервана пользователем", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    duration = time.perf_counter() - start_time
    finished_at = iso_now()
    typer.echo(f"Сгенерировано цепочек: {len(plan_result.paths)} за {duration:.2f} с")

    report = {
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(duration, 2),
        "config": {
            "root": config.root,
            "depth": config.depth,
            "breadth": config.breadth,
            "lang": config.lang,
            "model": config.model,
            "dedupe": config.dedupe,
            "resume": config.resume,
            "dry_run": config.dry_run,
        },
        "planner": plan_result.stats,
        "api": api_stats,
        "duplicates_before": plan_result.stats.get("candidates_seen", 0) - plan_result.stats.get("candidates_kept", 0),
        "duplicates_filtered": plan_result.stats.get("filtered_duplicates", 0),
        "existing_loaded": existing_store.total_loaded,
        "version": __version__,
    }

    if not config.dry_run:
        write_categories(plan_result.paths, config.out)
        write_report(report, config.report)
        typer.echo(f"Категории сохранены: {config.out}")
        typer.echo(f"Отчёт: {config.report}")
    else:
        typer.echo("Режим dry-run: файлы не записаны")


if __name__ == "__main__":  # pragma: no cover
    app()
