# Contributing

Use Python 3.10 or later. Create an environment, install the development extra, then run all four
verification commands before opening a change:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src/voxreceipt
coverage run -m pytest && coverage report
```

Do not commit speech recordings, official competition audio, credentials, model weights, generated
reports, or local absolute paths. Add a focused test for each behavior change. Keep adapters offline
and require an explicit trust flag before importing custom adapter code.
