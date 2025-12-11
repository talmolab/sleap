# OpenSpec Instructions

Instructions for AI coding assistants using OpenSpec for spec-driven development.

## When to Use OpenSpec

Always open `@openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like "proposal", "spec", "change", "plan")
- Introduces new capabilities, breaking changes, architecture shifts, or significant performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

## Three-Stage Workflow

### 1. Create a Change (`/openspec:proposal`)

When you describe a change, OpenSpec generates:
- A proposal document
- Broken-down implementation tasks
- Technical design decisions
- Spec deltas showing how requirements will change

**Location**: `openspec/changes/<change-name>/`

### 2. Implement (`/openspec:apply`)

After user approval:
- Work through tasks systematically
- Update code according to spec deltas
- Follow TDD: write tests first
- Mark tasks complete as you go

### 3. Archive (`openspec archive <change>`)

When complete:
- Merge approved spec deltas back into `openspec/specs/`
- Archive the change folder
- Single source of truth updated

## Directory Structure

```
openspec/
├── project.md          # Project overview, tech stack, conventions
├── constitution.md     # Team principles and standards
├── AGENTS.md          # This file - instructions for AI
├── specs/             # Current truth - authoritative specifications
│   └── <feature>/
│       └── spec.md    # Feature requirements
└── changes/           # Proposed updates
    └── <change-name>/
        ├── proposal.md    # What and why
        ├── tasks.md       # Implementation checklist
        └── spec-deltas.md # Requirement changes
```

## Spec File Format

Each spec in `openspec/specs/<feature>/spec.md`:

```markdown
# Feature Name

## Purpose
Clear statement of what this feature does

## Requirements

### Requirement: <name>
The system SHALL/MUST <behavior>

#### Scenario: <name>
WHEN <condition>
THEN <expected behavior>
```

## Proposal Format

Each proposal in `openspec/changes/<change-name>/proposal.md`:

```markdown
# Proposal: <Change Name>

## Summary
Brief description of the change

## Motivation
Why this change is needed

## Design
How it will be implemented

## Spec Deltas

### ADDED Requirements
New capabilities being introduced

### MODIFIED Requirements
Changed behavior (complete updated text)

### REMOVED Requirements
Deprecated or eliminated features
```

## Commands

- `/openspec:proposal` - Create a new change proposal
- `/openspec:apply` - Implement an approved proposal
- `openspec list` - Show active changes
- `openspec show <change>` - Display change details
- `openspec validate <change>` - Check spec formatting
- `openspec archive <change>` - Complete and merge changes
- `openspec update` - Refresh agent instructions

## Best Practices

### For AI Assistants
1. **Read specs first**: Before coding, check `openspec/specs/` for existing requirements
2. **Propose before implementing**: Significant changes need user approval
3. **Update specs with code**: Keep spec deltas accurate as you implement
4. **Follow TDD**: Write tests that verify scenario acceptance criteria
5. **One task at a time**: Complete and mark tasks systematically

### For SLEAP Specifically
- Cross-repo awareness: Consider sleap, sleap-nn, and sleap-io impacts
- User-first: Our users are scientists, not developers - keep it accessible
- Test thoroughly: Use pytest fixtures, aim for high coverage
- Document: Update relevant docs when behavior changes
- Check labels: Use appropriate GitHub labels (see constitution.md)

## When NOT to Use OpenSpec

Skip OpenSpec for:
- Trivial fixes (typos, formatting)
- Documentation-only updates
- Test-only changes
- Obvious bugs with clear fixes

For these, just make the change directly.

## Troubleshooting

**"I can't find the spec for X"**: Check `openspec/specs/`, or ask the user if it exists

**"Should I create a proposal?"**: If the change affects behavior, architecture, or multiple files - yes

**"The spec conflicts with the code"**: The spec is the source of truth; propose an update if needed