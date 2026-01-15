# Troubleshooting Patterns

Quick reference for common issues and diagnostic steps.

## Training Issues

### Loss Not Decreasing

**Symptoms**: Training runs but loss stays flat or increases

**Diagnostics**:
1. Check learning rate (too high?)
2. Check data augmentation (too aggressive?)
3. Check labels (are they correct?)

**Common fixes**:
- Reduce learning rate by 10x
- Disable rotation augmentation initially
- Verify skeleton is correct

### NaN Loss

**Symptoms**: Loss becomes NaN early in training

**Diagnostics**:
1. Check for empty frames (no labels)
2. Check image normalization
3. Check for corrupted images

**Common fixes**:
- Remove frames with no instances
- Check video codec compatibility

### Out of Memory (OOM)

**Symptoms**: CUDA out of memory error

**Diagnostics**:
```bash
# Check GPU memory
nvidia-smi
```

**Common fixes**:
- Reduce batch size
- Reduce input scale
- Use smaller backbone

## Inference Issues

### No Predictions

**Symptoms**: Model runs but no instances detected

**Diagnostics**:
1. Check confidence threshold
2. Check model was trained on similar data
3. Check input resolution matches training

**Common fixes**:
- Lower confidence threshold in inference settings
- Retrain with more diverse data

### Instance Duplication

**Symptoms**: Same animal gets multiple instances

**Diagnostics**:
1. Check NMS threshold
2. Check centroid model predictions
3. Review tracking settings

**Common fixes**:
- Increase NMS threshold
- Use tracking to merge duplicates
- Check multi-animal vs single-animal settings

### Track ID Switching

**Symptoms**: Animal identities swap between frames

**Diagnostics**:
1. Check tracker settings
2. Check for occlusions in video
3. Review prediction confidence

**Common fixes**:
- Try different tracker (Simple → Flow)
- Increase tracking window
- Manual correction in GUI

## GUI Issues

### SLEAP Won't Start

**Symptoms**: Nothing happens or immediate crash

**Diagnostics**:
```bash
# Run from terminal to see errors
sleap-label
```

**Common causes**:
- Qt/PySide6 conflicts
- Missing dependencies
- Corrupted installation

**Fixes**:
- Reinstall following the [installation guide](https://sleap.ai/installation.html)
- Check Python version compatibility
- Run `sleap doctor` to verify environment

### Video Won't Load

**Symptoms**: Error when opening video

**Diagnostics**:
```bash
# Check video info
ffprobe video.mp4
```

**Common causes**:
- Unsupported codec
- Corrupted file
- Path with special characters

**Fixes**:
- Re-encode video: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`
- Move to path without spaces/special chars

### Display Issues

**Symptoms**: UI elements too small/large, rendering problems

**Common causes**:
- High-DPI scaling issues
- OpenGL driver problems

**Fixes**:
- Set environment variable: `QT_SCALE_FACTOR=1.5`
- Update graphics drivers

## Data Issues

### Can't Open SLP File

**Symptoms**: Error loading project

**Diagnostics**:
```bash
uvx sio show file.slp --summary
```

**Common causes**:
- File corruption
- Version incompatibility
- Missing video files

**Fixes**:
- Try loading with `fix_videos=True`
- Check video paths match

### Missing Labels

**Symptoms**: Labels exist but don't display

**Diagnostics**:
1. Check skeleton matches
2. Check frame indices
3. Check instance visibility

**Fixes**:
- Verify skeleton: `uvx sio show file.slp --skeleton`
- Check track visibility in GUI

### Merge Conflicts

**Symptoms**: Issues when merging SLP files

**Diagnostics**:
```bash
# Compare skeletons
uvx sio show file1.slp --skeleton
uvx sio show file2.slp --skeleton
```

**Common causes**:
- Different skeleton definitions
- Overlapping frame ranges
- Different video sources

**Fixes**:
- Ensure identical skeletons before merge
- Use sleap-io merge with conflict resolution

## Version-Specific Issues

### Upgrading from v1.2 to v1.3+

- HDF5 format changed
- Need to re-export old projects

### Python 3.11+ Compatibility

- Some dependencies may have issues
- Check dependency versions

### GPU/CUDA Issues

**Diagnostics**:
```bash
# Check GPU availability (sleap doctor reports this)
sleap doctor

# Check NVIDIA driver
nvidia-smi
```

**Common fixes**:
- Ensure CUDA toolkit matches TensorFlow version
- Check cuDNN installation
- See [GPU setup guide](https://sleap.ai/installation.html#gpu-support)

## Version-Aware Investigation

**IMPORTANT**: Before diving deep into investigation, always check if the issue was already fixed.

### Step 1: Identify User's Versions

Look in the user's post for version information:
- `sleap doctor` output (preferred - contains all relevant info)
- `sleap.__version__` output
- Error tracebacks mentioning package paths

If version info is missing, request `sleap doctor` output from the user.

### Step 2: Check Release Notes for Fixes

```bash
# SLEAP main package releases
gh release list --repo talmolab/sleap --limit 20

# View specific release notes
gh release view v1.5.0 --repo talmolab/sleap

# Search release notes for keywords
gh release view v1.5.0 --repo talmolab/sleap --json body -q '.body' | grep -i "fix\|bug\|issue"

# sleap-io releases (data I/O issues)
gh release list --repo talmolab/sleap-io --limit 10
gh release view v0.6.0 --repo talmolab/sleap-io

# sleap-nn releases (training/inference issues)
gh release list --repo talmolab/sleap-nn --limit 10
gh release view v0.2.0 --repo talmolab/sleap-nn
```

### Step 3: Match User's Version to Code

If the user's version is older, checkout that version to see their actual code:

```bash
# Ensure repos are cloned
[ -d scratch/repos/sleap-io ] || gh repo clone talmolab/sleap-io scratch/repos/sleap-io
[ -d scratch/repos/sleap-nn ] || gh repo clone talmolab/sleap-nn scratch/repos/sleap-nn

# Checkout user's sleap-io version
cd scratch/repos/sleap-io
git fetch --tags
git checkout v0.5.0  # user's version

# Checkout user's sleap-nn version
cd scratch/repos/sleap-nn
git fetch --tags
git checkout v0.1.5  # user's version

# Return to sleap main repo
cd /home/talmo/code/sleap
```

### Step 4: Use Git Blame to Find Culprits

When you've identified a suspicious code region:

```bash
# Blame specific lines
git blame -L 50,100 path/to/file.py

# Show who changed a function recently
git log --oneline -10 -- path/to/file.py

# Find commits that added/removed a string
git log -S "suspicious_function" --oneline

# Find commits touching a pattern
git log -G "regex_pattern" --oneline -- path/to/file.py

# Show the diff for a specific commit
git show COMMIT_HASH --stat
git show COMMIT_HASH -- path/to/file.py
```

### Step 5: Compare Versions

```bash
# See what changed between user's version and current
git diff v1.4.0..HEAD -- path/to/file.py

# List commits between versions
git log --oneline v1.4.0..HEAD -- path/to/file.py

# Find when a bug was introduced (bisect conceptually)
git log --oneline --all -- path/to/file.py | head -20
```

### Response Scenarios

**Best case - Already fixed:**
```markdown
Good news! This issue was fixed in version X.Y.Z (PR #123).

To fix this, upgrade to the latest version of SLEAP. You can verify your current version with:
\`\`\`bash
sleap doctor
\`\`\`

See the [installation guide](https://sleap.ai/installation.html) for upgrade instructions for your setup.

Let us know if that resolves it!
```

**Issue exists in their version:**
```markdown
I checked the code for version X.Y.Z and I can see the issue.
This appears to be caused by [explanation].

[Solution or workaround]
```

**Need to backport a fix:**
- Note the specific commit that fixed it
- Check if it can be cleanly cherry-picked
- Consider releasing a patch version
