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

The QC module is also available programmatically via `sleap.qc`:

```python
import sleap

labels = sleap.load_file("labels.slp")
# QC API documentation TBD based on sleap.qc module
```

## Tips

- Run QC after initial labeling but before training
- Re-run after proofreading tracking results
- Not all flagged items are errors—use your judgment
- QC is most effective with consistent labeling practices
