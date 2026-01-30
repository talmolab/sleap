# Label Quality Control

The Label QC feature helps you find and fix annotation errors by automatically detecting unusual or potentially incorrect labels.

Access via **Analyze → Label QC...**

---

## Overview

When you open Label QC, SLEAP analyzes all your labeled instances and assigns each an **anomaly score** (0-1). Higher scores indicate labels that look unusual compared to your other annotations—these may be errors worth reviewing.

---

## Using the QC Panel

### Score Distribution

The histogram shows the distribution of anomaly scores across all your labels:

- **X-axis**: Anomaly score (0 = normal, 1 = highly unusual)
- **Y-axis**: Number of instances
- **Red line**: Current threshold—instances above this are flagged

**Click anywhere on the histogram** to adjust the threshold.

### Flagged Instances Table

Below the histogram, a table lists all instances above the threshold:

| Column | Description |
|--------|-------------|
| **Video** | Video name |
| **Frame** | Frame number |
| **Instance** | Instance index |
| **Score** | Anomaly score |

**Click any row** to navigate directly to that instance in the main view.

### Actions

| Button | Description |
|--------|-------------|
| **Run Analysis** | Analyze labels (runs automatically on open) |
| **Add to Suggestions** | Add all flagged frames to labeling suggestions |
| **Export CSV** | Export flagged instances to a CSV file |

---

## What Gets Flagged?

The QC system looks for several types of issues:

- **Unusual poses** — Skeleton configurations that differ significantly from your other labels
- **Extreme positions** — Instances with unusual bounding box sizes or positions
- **Structural anomalies** — Anatomically implausible joint angles or limb lengths
- **Visibility patterns** — Unusual combinations of visible/non-visible nodes

---

## Workflow Tips

1. **Start with a high threshold** (0.8+) to catch obvious errors first
2. **Review flagged instances** by clicking rows in the table
3. **Fix or delete** problematic labels as needed
4. **Lower the threshold** gradually to find subtler issues
5. **Add remaining flags to suggestions** for systematic review

---

## Navigation

When the QC panel is open and has flagged instances:

- **Next Suggestion** (keyboard shortcut) navigates to the next flagged instance
- **Previous Suggestion** navigates to the previous flagged instance

This lets you quickly step through all flagged labels without using the mouse.

---

## Docking

The QC panel can be:

- **Docked** on the left or right side of the main window
- **Floated** as a separate window
- **Tabbed** with other panels (Videos, Skeleton, Instances)

Drag the title bar to rearrange, or use the dock/undock button.
