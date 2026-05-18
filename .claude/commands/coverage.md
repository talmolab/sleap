Run tests with coverage.

Command to run:
```
uv run pytest -q --maxfail=1 --cov --cov-branch && rm -f .coverage.* && uv run coverage annotate
```

The result will be the terminal output and the line-by-line coverage will be in files sitting next to each module with the file naming `{module_name.py},cover`. 

If you are working on a PR, figure out which files were changed and look for coverage specifically in those. If you don't know which files to look for coverage in, use this:

```
git diff --name-only $(git merge-base origin/develop HEAD) | jq -R . | jq -s .
```