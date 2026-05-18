# Response Templates

## Opening Lines

Use variety - pick one that fits the context:

**For questions:**
- "Thanks for the post!"
- "Great question!"
- "Thanks for reaching out!"

**For bug reports:**
- "Thanks for the detailed report!"
- "Thanks for flagging this!"
- "Appreciate you reporting this!"

**For feature requests:**
- "Thanks for the suggestion!"
- "Interesting idea!"

## Requesting More Information

### Need SLP File

```markdown
To help diagnose this, it would be helpful to see your project file. Could you upload your `.slp` file to https://slp.sh and share the link here?

This is our secure file sharing service - files are automatically deleted after 7 days.
```

### Need System Info

```markdown
Could you share the output of `sleap doctor`? You can run this in a terminal:

1. Open a terminal (Command Prompt on Windows, Terminal on Mac/Linux)
2. Type `sleap doctor` and press Enter
3. Copy-paste the output here

This will help us understand your setup!
```

### Need Error Logs

```markdown
Could you share the full error message? If you're running from the GUI:

1. Open SLEAP from a terminal instead of clicking the icon
2. On Windows: Open Command Prompt, type `sleap-label`, press Enter
3. On Mac/Linux: Open Terminal, type `sleap-label`, press Enter
4. Reproduce the error
5. Copy-paste any red text from the terminal here
```

### Need Video Info

```markdown
Could you tell us more about your video?

1. What format is it? (mp4, avi, etc.)
2. How long is it? (minutes/hours)
3. What resolution? (you can check this by right-clicking the file → Properties → Details on Windows)
```

## Common Solutions

### GPU Memory Issues

```markdown
This looks like a GPU memory issue! Try reducing the batch size:

1. In the training dialog, look for "Batch Size"
2. Try halving it (e.g., if it's 4, try 2)
3. If that still fails, try 1

Smaller batch sizes use less GPU memory but may train a bit slower.
```

### Multi-Animal Tracking

```markdown
For multi-animal projects, make sure you're using the correct tracking method:

1. After inference, go to `Predict` → `Run Tracking...`
2. Try the "Simple" tracker first
3. If that doesn't work well, try "Flow" tracker for animals that move smoothly

The tracker connects detections across frames - without it, you'll see different track IDs each frame!
```

### Missing Labels After Import

```markdown
If labels aren't showing after import, the skeleton might not match:

1. Go to `Skeleton` menu → check that nodes and edges match your data
2. If they don't match, you may need to re-import with the correct skeleton

You can check your file's skeleton with:
```bash
uvx sio show your_file.slp --skeleton
```
```

## Closing Lines

**Standard:**
```markdown
Let us know if that works for you!

Cheers,

:heart: Talmo & Claude :robot:
```

**If uncertain about solution:**
```markdown
Let us know if this helps or if you're still running into issues!

Cheers,

:heart: Talmo & Claude :robot:
```

**If they need to try something:**
```markdown
Give that a try and let us know how it goes!

Cheers,

:heart: Talmo & Claude :robot:
```

## Technical Details Section

Always include this at the end:

```markdown
<details>
<summary><b>Extended technical analysis</b></summary>

## Investigation Notes

- [What you checked]
- [What you found]
- [Relevant code paths]

## Relevant Files

- `sleap/path/to/file.py:123` - [what it does]

## Version Info

- Reported version: X.Y.Z
- Current version: A.B.C
- Relevant changes: [if any]

</details>
```
