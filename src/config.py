from __future__ import annotations

import dataclasses
import json
import os
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union, get_args, get_origin

import yaml
from dotenv import load_dotenv

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore

from .utils import ensure_directory

ENV_PREFIX = "GPT_TAXONOMY_"


@dataclass
class AppConfig:
    root: str = ""
    depth: int = 2
    breadth: int = 5
    existing: list[Path] = field(default_factory=list)
    lang: str = "ru"
    out: Path = Path("out/categories.txt")
    model: str = "deepseek-chat"
    temperature: float = 0.4
    top_p: float = 0.9
    seed: Optional[int] = None
    max_tokens: int = 2048
    dry_run: bool = False
    resume: bool = False
    dedupe: bool = True
    concurrency: int = 4
    rate_limit: float = 1.0
    token_budget: Optional[int] = None
    cache_dir: Path = Path("out/cache")
    report: Path = Path("out/report.json")
    log_file: Path = Path("logs/app.log")
    config_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.out = Path(self.out)
        self.cache_dir = Path(self.cache_dir)
        self.report = Path(self.report)
        self.log_file = Path(self.log_file)
        if isinstance(self.existing, (str, os.PathLike)):
            self.existing = [Path(self.existing)]
        else:
            self.existing = [Path(p) for p in self.existing]
        self.validate()

    @property
    def out_dir(self) -> Path:
        return self.out.parent

    def validate(self) -> None:
        if not self.root:
            raise ValueError("Не указана корневая категория (root)")
        if not 2 <= self.depth <= 8:
            raise ValueError("Глубина (depth) должна быть в диапазоне 2..8")
        if not 3 <= self.breadth <= 15:
            raise ValueError("Средняя ширина (breadth) должна быть в диапазоне 3..15")
        if self.concurrency < 1:
            raise ValueError("Параллелизм (concurrency) должен быть >= 1")
        if self.rate_limit <= 0:
            raise ValueError("Интервал rate-limit должен быть положительным")
        if self.max_tokens <= 0:
            raise ValueError("max-tokens должен быть положительным")
        if self.token_budget is not None and self.token_budget <= 0:
            raise ValueError("token-budget должен быть положительным")

    def ensure_output_structure(self) -> None:
        ensure_directory(self.out_dir)
        ensure_directory(self.cache_dir)
        ensure_directory(self.log_file.parent)


def _defaults_dict() -> dict[str, Any]:
    defaults = AppConfig(root="__placeholder__")
    data: dict[str, Any] = {}
    for app_field in dataclasses.fields(AppConfig):
        if app_field.name == "root":
            data[app_field.name] = ""
            continue
        data[app_field.name] = json.loads(json.dumps(getattr(defaults, app_field.name), default=str))
    data["root"] = ""
    return data


def _load_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    with path.open("rb") as fh:
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(fh) or {}
        if suffix == ".toml":
            if tomllib is None:  # pragma: no cover
                raise RuntimeError("tomllib недоступен для чтения TOML")
            return tomllib.load(fh)
        if suffix == ".json":
            return json.load(fh)
        raise ValueError(f"Неизвестный формат файла конфигурации: {suffix}")


def _normalize_existing(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [Path(value)]
    if isinstance(value, Iterable):
        return [Path(v) for v in value]
    raise TypeError("Поле existing должно быть строкой, путём или списком путей")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "да"}


def _parse_value(value: Any, target_type: Any) -> Any:
    origin = get_origin(target_type)
    args = get_args(target_type)
    if origin is list:
        subtype = args[0]
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            return [Path(part) if subtype is Path else subtype(part) for part in parts]
        return value
    if target_type in {int, float, str}:
        return target_type(value)
    if target_type is bool:
        return _parse_bool(value)
    if target_type is Path:
        return Path(value)
    if origin in {Union, types.UnionType} and type(None) in args:
        subtypes = [arg for arg in args if arg is not type(None)]  # noqa: E721
        subtype = subtypes[0] if subtypes else str
        if value in {None, "", "null", "None"}:
            return None
        return _parse_value(value, subtype)
    return value


def _env_overrides() -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for app_field in dataclasses.fields(AppConfig):
        env_name = ENV_PREFIX + app_field.name.upper()
        raw = os.getenv(env_name)
        if raw is None:
            continue
        overrides[app_field.name] = _parse_value(raw, app_field.type)
    return overrides


def load_config(cli_options: Dict[str, Any]) -> AppConfig:
    load_dotenv()

    data = _defaults_dict()
    config_path: Optional[Path] = None
    cli_config_path = cli_options.get("config")
    if cli_config_path:
        config_path = Path(cli_config_path)
    else:
        for candidate in (Path("config.toml"), Path("config.yaml"), Path("config.yml")):
            if candidate.exists():
                config_path = candidate
                break
    if config_path:
        file_data = _load_file(config_path)
        if file_data:
            data.update(file_data)
        data["config_path"] = config_path

    data.update(_env_overrides())

    for key, value in cli_options.items():
        if value is None:
            continue
        data[key] = value

    if "existing" in data:
        data["existing"] = _normalize_existing(data["existing"])

    config = AppConfig(**data)
    config.ensure_output_structure()
    return config


__all__ = ["AppConfig", "load_config"]
