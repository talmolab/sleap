---
name: investigation
description: Scaffolds a template for a small investigation for empirical experimentation. Use when the user asks for an investigation, experiment or research to support making design decisions, assess performance or try out a new idea that requires data or an MVP.
---

# Set up an investigation

## Instructions

1. Create a folder in `{REPO_ROOT}/scratch/` with the format `{YYYY-MM-DD}-{descriptive-name}`. You will mainly work inside this folder.
2. Create a `README.md` in this folder where you will:
   1. Describe the task.
   2. Take background notes for context.
   3. Create a task list for progress tracking.
   4. Summarize results and outcomes.
3. Create scripts and testing data files in this folder to conduct an empirical investigation.
4. For multi-step investigations with sub-experiments, consider generating individual markdown files to take interediate notes.
5. The `README.md` should contain actionable findings to support a subsequent implementation or design decision.

## Best practices
- Both for this library and other ones, it's worth writing a simple script or calling the library interactively to list its members or try out different constructions to explore the API and document it in a markdown file called `API.md`.
- Generate figures when applicable (e.g., plots, data visualizations) and reference them inline in the markdown files.
- Always use `uv` with self-contained dependencies when using Python.
- When you need a local web server, use `npx serve -p 8080 --cors --no-clipboard &` (run in background to avoid blocking).
- Prefer self-contained HTML+JS pages (without React if possible) and use Playwright MCP for rendering screenshots.
- Use subagents whenever possible to save context and delegate tasks in parallel.

## Important: Scratch work is gitignored
The `scratch/` directory is in `.gitignore` and will NOT be committed. When distilling findings into PRs:
- Do NOT assume local scratch notes will be checked in
- Include all relevant information inline in PR descriptions or code comments
- Copy key findings, code snippets, benchmark results, or data directly into the PR
- The PR should be self-contained and not rely on scratch files for context