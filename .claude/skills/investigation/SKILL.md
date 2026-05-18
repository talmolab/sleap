---
name: investigation
description: >
  Scaffolds a structured investigation in scratch/ for empirical research and documentation.
  Use when the user says "start an investigation" or wants to: trace code paths or data flow
  ("trace from X to Y", "what touches X", "follow the wiring"), document system architecture
  comprehensively ("document how the system works", "archeology"), investigate bugs
  ("figure out why X happens"), explore technical feasibility ("can we do X?"), or explore
  design options ("explore the API", "gather context", "design alternatives").
  Creates dated folder with README. NOT for simple code questions or single-file searches.
---

# Set up an investigation

## Instructions

1. Create a folder in `{REPO_ROOT}/scratch/` with the format `{YYYY-MM-DD}-{descriptive-name}`:
   ```bash
   mkdir scratch/$(uv run python -c "import datetime; print(datetime.date.today().isoformat())")-{descriptive-name}
   ```
2. Create a `README.md` in this folder with: task description, background context, task checklist. Update with findings as you progress.
3. Create scripts and data files as needed for empirical work.
4. For complex investigations, split into sub-documents as patterns emerge.

## Investigation Patterns

These are common patterns, not rigid categories. Most investigations blend multiple patterns.

**Tracing** - "trace from X to Y", "what touches X", "follow the wiring"
- Follow call stack or data flow from a focal component to its connections
- Can trace forward (X → where does it go?) or backward (what leads to X?)
- Useful for: assessing impact of changes, understanding coupling

**System Architecture Archeology** - "document how the system works", "archeology"
- Comprehensive documentation of an entire system or flow for reusable reference
- Start from entry points, trace through all layers, document relationships exhaustively
- For complex systems, consider numbered sub-documents (01-cli.md, 02-data.md, etc.)

**Bug Investigation** - "figure out why X happens", "this is broken"
- Reproduce → trace root cause → propose fix
- For cross-repo bugs, consider per-repo task breakdowns

**Technical Exploration** - "can we do X?", "is this possible?", "figure out how to"
- Feasibility testing with proof-of-concept scripts
- Document what works AND what doesn't

**Design Research** - "explore the API", "gather context", "design alternatives"
- Understand systems and constraints before building
- Compare alternatives, document trade-offs
- Include visual artifacts (mockups, screenshots) when relevant
- For iterative decisions, use numbered "Design Questions" (DQ1, DQ2...) to structure review

## Best Practices

- Use `uv` with inline dependencies for standalone scripts; for scripts importing local project code, use `python` directly (or `uv run python` if env not activated)
- Use subagents for parallel exploration to save context
- Write small scripts to explore APIs interactively
- Generate figures/diagrams and reference inline in markdown
- For web servers: `npx serve -p 8080 --cors --no-clipboard &`
- For screenshots: use Playwright MCP for web, Qt's grab() for GUI
- For external package API review: clone to `scratch/repos/` for direct source access

## Important: Scratch is Gitignored

The `scratch/` directory is in `.gitignore` and will NOT be committed.

- NEVER delete anything from scratch - it doesn't need cleanup
- When distilling findings into PRs, include all relevant info inline
- Copy key findings, code, and data directly into PR descriptions
- PRs must be self-contained; don't reference scratch files
