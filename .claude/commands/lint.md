Run linting with `ruff`.

Command:

```
uv run ruff format sleap tests && uv run ruff check --fix sleap tests
```

Then manually fix any remaining errors which cannot be automatically fixed by ruff.