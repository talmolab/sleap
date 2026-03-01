---
name: pr
description: >
  Create a well-structured GitHub PR with proper branching, testing, formatting, and documentation.
  Use when the user says "create a PR", "make a PR", "open a pull request", or wants to submit
  changes for review. Handles the full workflow: branch creation, implementation, testing,
  formatting, committing, and PR creation with comprehensive descriptions.
---

# Create a GitHub Pull Request

## Overview

This skill guides the complete PR workflow from branch creation to PR submission. Follow all steps
in order to ensure high-quality, well-documented contributions.

## Step 1: Branch Setup

### Pull latest develop
```bash
git checkout develop
git pull origin develop
```

### Create feature branch
Use a descriptive branch name following the pattern: `{type}/{description}`

Types:
- `feature/` - New functionality
- `fix/` - Bug fixes
- `refactor/` - Code restructuring
- `docs/` - Documentation updates
- `test/` - Test additions/improvements

```bash
git checkout -b feature/descriptive-name
```

## Step 2: Understand the Problem

Before coding, clearly identify:
1. **Core problem**: What issue are we solving?
2. **Scope**: What files/modules will be affected?
3. **Approach**: What's the implementation strategy?
4. **Edge cases**: What scenarios need special handling?

If there's an associated GitHub issue, fetch it for context:
```bash
gh issue view <issue-number>
```

## Step 3: Implement Changes

- Make focused, incremental changes
- Follow existing code patterns and style
- Add docstrings and comments for complex logic
- Consider backwards compatibility

## Step 4: Write Tests

### Location
Tests go in the `tests/` directory, mirroring the source structure.

### Requirements
- Cover all new functionality
- Test edge cases and error conditions
- Test both success and failure paths
- Aim for high coverage of changed code

### Test file naming
- `test_{module_name}.py` for module tests
- Place in corresponding `tests/` subdirectory

## Step 5: Format Code

Run formatting and linting:
```bash
# Format code
uv run ruff format sleap tests

# Fix auto-fixable lint issues
uv run ruff check --fix sleap tests
```

Then manually fix any remaining errors which cannot be automatically fixed by ruff.

## Step 6: Run Tests with Coverage

Run the full test suite with coverage:
```bash
uv run pytest -q --maxfail=1 --cov --cov-branch && rm -f .coverage.* && uv run coverage annotate
```

### Check coverage for changed files
The coverage annotate command creates `{module_name.py},cover` files next to each module.

To find which files changed:
```bash
git diff --name-only $(git merge-base origin/develop HEAD)
```

Review the `,cover` files for your changed modules to ensure adequate coverage.

### Coverage markers in annotated files
- **>** means line was executed
- **!** means line was NOT executed (needs test coverage)
- **-** means line is not executable (comments, blank lines)

## Step 7: Commit Changes

### Commit structure
Make well-structured, atomic commits:
- Each commit should be a logical unit of work
- Write clear, descriptive commit messages
- Use conventional commit format when appropriate

### Commit message format
```
<type>: <short description>

<optional longer description>

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`

## Step 8: Push to GitHub

```bash
git push -u origin <branch-name>
```

## Step 9: Create Pull Request

### Create the PR
```bash
gh pr create --base develop --title "<descriptive title>" --body "$(cat <<'EOF'
## Summary
<1-3 bullet points describing the changes>

## Changes Made
- <detailed list of changes>

## Example Usage
```python
# For enhancements, show how to use the new functionality
```

## API Changes
- <list any API changes, new parameters, removed functionality>

## Testing
- <describe test coverage>
- <note any manual testing done>

## Design Decisions
- <explain key architectural choices>
- <note trade-offs considered>

## Future Considerations
- <potential improvements not in scope>
- <known limitations>

## Related Issues
Closes #<issue-number> (if applicable)

---
🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### If updating an existing PR

Fetch current PR description:
```bash
gh pr view <pr-number> --json body -q '.body'
```

Update PR description:
```bash
gh pr edit <pr-number> --body "<new body>"
```

### Fetch associated issue for context
If an issue is linked:
```bash
gh issue view <issue-number>
```

Use issue context to ensure PR description addresses all requirements.

## PR Description Checklist

- [ ] Summary clearly explains the "what" and "why"
- [ ] All significant changes are documented
- [ ] Example usage provided for new features
- [ ] API changes explicitly listed
- [ ] Breaking changes highlighted
- [ ] Test coverage described
- [ ] Design decisions explained with reasoning
- [ ] Related issues linked

## Docs Preview

PRs that modify `docs/`, `sleap/`, or `mkdocs.yml` will automatically get a docs preview deployed at:
```
https://docs.sleap.ai/pr/<pr-number>/
```

A comment will be posted on the PR with the preview link.

## Quick Reference Commands

```bash
# Branch setup
git checkout develop && git pull origin develop
git checkout -b feature/my-feature

# Format and lint
uv run ruff format sleap tests
uv run ruff check --fix sleap tests

# Test with coverage
uv run pytest -q --maxfail=1 --cov --cov-branch && rm -f .coverage.* && uv run coverage annotate

# Find changed files
git diff --name-only $(git merge-base origin/develop HEAD)

# Commit
git add <files>
git commit -m "feat: description"

# Push and create PR
git push -u origin <branch>
gh pr create --base develop --title "Title" --body "Description"

# View/edit existing PR
gh pr view <number>
gh pr edit <number> --body "New description"

# View linked issue
gh issue view <number>
```

## CI Checks

PRs trigger the following CI checks:
- **Lint**: `uv run ruff check sleap tests` and `uv run ruff format --check sleap tests`
- **Tests**: `uv run pytest --cov=sleap` on Ubuntu, Windows, and macOS
- **Docs Preview**: Auto-deployed for changes to `docs/`, `sleap/`, or `mkdocs.yml`

Ensure all checks pass before requesting review.
