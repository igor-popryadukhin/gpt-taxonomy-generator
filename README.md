# GPT Taxonomy Generator

CLI-инструмент для генерации иерархии категорий на базе модели DeepSeek.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Запуск

```bash
export DEEPSEEK_API_KEY=...
python -m src.cli --root "Ремонт и строительство" --depth 5 --breadth 8
```
