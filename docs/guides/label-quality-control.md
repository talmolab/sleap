# Label Quality Control

*Case: You want to check your labels for errors before training or after proofreading.*

SLEAP 1.6 introduces a Label Quality Control (QC) module that automatically detects common labeling errors using statistical analysis.

## Accessing Label QC

From the GUI: **Analyze** → **Label QC...**

## Types of Errors Detected

The QC module uses Gaussian Mixture Model (GMM) based anomaly detection to identify:

### Temporal Jitter
Detects frames where node positions jump unexpectedly compared to neighboring frames.

### Visibility Errors
Identifies inconsistencies in node visibility patterns.

### Scale Anomalies
Flags instances where the skeleton scale differs significantly from the norm.

### Potential Swaps
Detects possible identity swaps between tracked instances.

## Using QC Results

1. Open **Analyze** → **Label QC...**
2. Select which checks to run
3. Review flagged frames in the results panel
4. Double-click a result to navigate to that frame
5. Correct any actual errors found

## Programmatic Access

The QC module is available programmatically via `sleap.qc`:

```python
import sleap_io as sio
from sleap.qc import LabelQCDetector, QCConfig

# Load labels
labels = sio.load_file("labels.slp")

# Create detector with default config
detector = LabelQCDetector()

# Fit on labels (learns what "normal" looks like from your data)
detector.fit(labels)

# Score all instances
results = detector.score(labels)

# Get flagged instances above threshold (0.0-1.0, higher = more anomalous)
flagged = results.get_flagged(threshold=0.7)

# Inspect flagged instances
for flag in flagged:
    print(f"Video {flag.video_idx}, Frame {flag.frame_idx}, Instance {flag.instance_idx}")
    print(f"  Score: {flag.score:.2f}")
    print(f"  Top contributors: {flag.top_contributors}")
```

### Configuration Options

```python
from sleap.qc import QCConfig

config = QCConfig(
    instance_threshold=0.7,      # Score threshold for flagging
    gmm_n_components=3,          # Number of GMM components
    duplicate_iou_threshold=0.5, # IoU threshold for duplicate detection
)
detector = LabelQCDetector(config=config)
```

### Available Classes

- `LabelQCDetector`: Main detection interface
- `QCConfig`: Configuration settings
- `QCResults`: Container for scores, frame results, and feature contributions
- `QCFlag`: Individual flagged instance with score and contributing features

## Tips

- Run QC after initial labeling but before training
- Re-run after proofreading tracking results
- Not all flagged items are errors—use your judgment
- QC is most effective with consistent labeling practices
