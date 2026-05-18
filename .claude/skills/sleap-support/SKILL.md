---
name: sleap-support
description: >
  Handle SLEAP GitHub support workflow for issues and discussions. Use when
  the user says "support", provides a GitHub issue/discussion number like "#2512",
  or asks to investigate a user report from talmolab/sleap. Scaffolds investigation
  folders, downloads posts with images, analyzes problems, and drafts friendly responses.
---

# SLEAP Support Workflow

Handle GitHub issues and discussions from `talmolab/sleap` with a systematic investigation process.

## Quick Start

When given an issue/discussion number:

```bash
# Check both issues AND discussions (users often post in wrong category)
gh issue view 2512 --repo talmolab/sleap --json number,title,body,author,createdAt,comments 2>/dev/null || \
gh api repos/talmolab/sleap/discussions/2512 2>/dev/null
```

## Workflow Steps

### 1. Create Investigation Folder

First, check if an investigation already exists:

```bash
ls -d scratch/*-*2512* 2>/dev/null || echo "No existing investigation"
```

If none exists, create one:

```bash
mkdir -p scratch/$(date +%Y-%m-%d)-{issue|discussion}-2512-{short-description}
```

### 2. Fetch Post Content

**For Issues:**
```bash
gh issue view NUMBER --repo talmolab/sleap --json number,title,body,author,createdAt,comments,labels > scratch/.../issue.json
```

**For Discussions:**
```bash
gh api repos/talmolab/sleap/discussions/NUMBER > scratch/.../discussion.json
```

### 3. Download Images

Extract image URLs from the post body and download:

```bash
# Parse markdown image links: ![alt](url)
grep -oP '!\[.*?\]\(\K[^)]+' scratch/.../issue.json | while read url; do
  wget -P scratch/.../images/ "$url"
done
```

### 4. Create USER_POST.md

Convert the JSON to readable markdown with inline images:

```markdown
# Issue/Discussion #NUMBER: Title

**Author**: @username
**Created**: YYYY-MM-DD
**Platform**: (extract from post if mentioned)
**SLEAP Version**: (extract from post if mentioned)

## Original Post

[post body with images referenced inline]

## Comments

[any replies]
```

### 5. Write Investigation README

Create `scratch/.../README.md`:

```markdown
# Investigation: Issue/Discussion #NUMBER

**Date**: YYYY-MM-DD
**Post**: https://github.com/talmolab/sleap/{issues|discussions}/NUMBER
**Author**: @username
**Type**: Bug Report | Usage Question | Feature Request

## Summary

[1-2 sentence summary of the issue]

## Key Information

- **Platform**: Windows/macOS/Linux
- **SLEAP Version**: X.Y.Z
- **GPU**: (if relevant)
- **Dataset**: (if described)

## Preliminary Analysis

[Initial thoughts on what might be happening]

## Areas to Investigate

- [ ] Check area 1
- [ ] Check area 2

## Files

- `USER_POST.md` - Original post content
- `images/` - Downloaded screenshots
- `RESPONSE_DRAFT.md` - Draft response (when ready)
```

### 6. Check Release History First

**Before deep investigation, check if the issue was already fixed:**

```bash
# Get user's SLEAP version from their post (look for sleap doctor output or sleap.__version__)
USER_VERSION="1.4.0"  # example

# Check sleap releases for fixes
gh release list --repo talmolab/sleap --limit 20
gh release view v1.5.0 --repo talmolab/sleap --json body -q '.body' | grep -i "fix"

# For sleap-io issues
gh release list --repo talmolab/sleap-io --limit 10
gh release view v0.6.0 --repo talmolab/sleap-io --json body -q '.body'

# For sleap-nn/training issues
gh release list --repo talmolab/sleap-nn --limit 10
gh release view v0.2.0 --repo talmolab/sleap-nn --json body -q '.body'
```

**If a fix exists in a newer version:**
- Response should guide user to upgrade
- Include the specific version with the fix
- Mention what was fixed (link to PR/issue if available)

**If investigating an unfixed bug:**
- Checkout the user's version to see their actual code:

```bash
# Clone repos if not present
[ -d scratch/repos/sleap-io ] || gh repo clone talmolab/sleap-io scratch/repos/sleap-io
[ -d scratch/repos/sleap-nn ] || gh repo clone talmolab/sleap-nn scratch/repos/sleap-nn

# Checkout user's version
cd scratch/repos/sleap-io && git fetch --tags && git checkout v0.5.0
cd scratch/repos/sleap-nn && git fetch --tags && git checkout v0.1.5

# Now you're looking at the code they're actually running
```

**Use `git blame` to find potential culprits:**

```bash
# Find when a suspicious function was last changed
git blame -L 50,100 sleap/io/main.py

# Check if a line was changed recently
git log --oneline -5 -- path/to/file.py

# Find the commit that introduced a specific change
git log -S "function_name" --oneline
```

### 7. Analyze and Reproduce

Determine the issue type:

**Usage Question**: Check if documentation covers this. Common topics:
- Model configuration (skeleton, training params)
- Multi-animal vs single-animal tracking
- Inference and tracking settings
- Data format questions

**Bug Report**: Try to reproduce. Check:
- Version-specific issues
- Platform-specific behavior
- GPU/CUDA compatibility
- Data corruption signs

**Feature Request**: Note for tracking, no immediate action needed.

### 8. Determine Data Needs

**When to request SLP file:**
- Inference/tracking issues that can't be diagnosed from logs
- "Labels not showing" or display issues
- Merging or import problems
- Corruption or data loss reports

**Suggest upload to `https://slp.sh`** - our SLP file sharing service.

**If SLP provided:** Download and analyze:
```bash
sio show path/to/file.slp --summary
sio show path/to/file.slp --videos
sio show path/to/file.slp --skeleton
```

### 9. Draft Response

Create `RESPONSE_DRAFT.md` following this structure:

```markdown
Hi @{username},

Thanks for the post!

[Restate understanding: "If I understand correctly, you're seeing X when you try to Y..."]

[Provide solution OR request more info]

[If requesting info, give EXPLICIT instructions:]
- Use `sleap doctor` CLI for diagnostics
- Provide copy-paste terminal commands
- Assume non-technical user

Let us know if that works for you!

Cheers,

:heart: Talmo & Claude :robot:

<details>
<summary><b>Extended technical analysis</b></summary>

[Detailed investigation notes, code traces, version checks]

</details>
```

## Response Tone Guidelines

- **Opening**: Bright and positive ("Thanks for the post!", "Great question!")
- **Body**: Clear, concise, non-technical language
- **Instructions**: Step-by-step, assume terminal newbie
- **Closing**: Encouraging ("Let us know if that works for you!")
- **Signature**: `:heart: Talmo & Claude :robot:`

## When Requesting User Actions

**Terminal commands must be copy-paste ready:**

```bash
# Good - one-liner, no environment activation needed
sleap doctor

# Good - uses uvx for isolated execution
uvx sio show your_file.slp --summary

# Bad - assumes environment knowledge
source activate sleap && python -c "import sleap; print(sleap.__version__)"
```

## Related Repositories

**For data/I/O issues** - Check `talmolab/sleap-io`:
- Local: `../sleap-io` (preferred)
- Clone if needed: `gh repo clone talmolab/sleap-io scratch/repos/sleap-io`
- Key docs: `sleap-io/docs/examples.md`, `sleap-io/docs/formats/SLP.md`
- CLI: `sio show --help` for inspection commands

**For training/inference issues** - Check `talmolab/sleap-nn`:
- Local: `../sleap-nn` (preferred)
- Clone if needed: `gh repo clone talmolab/sleap-nn scratch/repos/sleap-nn`
- Topics: Model configs, training params, evaluation metrics, tracking

## Common Patterns

### Instance Duplication
- Check track assignment logic
- Look for ID switching during tracking
- May need `sleap-track` with different settings

### Training Issues
- GPU memory: suggest reducing batch size
- Loss not decreasing: check learning rate, augmentation
- NaN losses: data normalization issues

### Import/Export Issues
- Format compatibility (H5 vs SLP versions)
- Missing video paths
- Skeleton definition mismatches

### GUI Issues
- Qt/PySide6 version conflicts
- Display scaling on high-DPI
- Video codec issues

## Confirmation Before Posting

**ALWAYS** confirm with the developer before posting:
1. Show the full draft response
2. Ask: "Does this look good to post?"
3. Wait for explicit approval

**Never post automatically** - support responses represent the project.
