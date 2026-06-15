"""
Widget for visualizing label QC results.

Provides histogram and table views of instance anomaly scores,
with click-to-navigate support for reviewing flagged annotations.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np
from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtCore import QThread, Signal as QSignal

# Matplotlib setup with proper backend handling
import matplotlib
import os

if os.environ.get("MPLBACKEND") != "Agg":
    try:
        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
    except ImportError:
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas
else:
    from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas

from matplotlib.figure import Figure

if TYPE_CHECKING:
    import sleap_io as sio
    from sleap.qc.config import QCConfig
    from sleap.qc.results import QCResults, QCFlag


# Plain-language, biologist-friendly help for each detector. Each entry is
# (short title, body) and is shown when the user clicks the "?" help button next
# to a detector in the Detector Settings panel (issue #2769, item 3). The text
# deliberately avoids math/statistics jargon and describes what the detector
# catches and what the mistake looks like on a real animal.
DETECTOR_HELP = {
    "flip": (
        "Whole-instance L/R flip",
        "Flags an animal whose left and right body parts look swapped — for "
        "example, the left and right ears traced on the wrong sides, so the "
        "whole pose is mirror-flipped.\n\n"
        "Use this when you suspect a labeler clicked the left/right parts the "
        "wrong way around.",
    ),
    "chimera": (
        "Chimera (pose split)",
        "Flags one animal whose body parts actually belong to two different "
        "animals — like a single skeleton that jumps from one mouse's head to "
        "another mouse's tail.\n\n"
        "This often happens when two animals are close together and the labels "
        "get mixed up between them.",
    ),
    "duplicate": (
        "Duplicate / split",
        "Flags two labels that sit almost on top of each other — usually the "
        "same animal accidentally traced twice, or one animal split into two "
        "overlapping copies.\n\n"
        "Review these to delete the extra copy and keep a single clean label.",
    ),
    "chain": (
        "Wrong chain order",
        "Flags a body part traced out of order along a connected chain, such as "
        "a tail. Imagine the tail tip placed before the tail base, so the chain "
        "doubles back on itself instead of running smoothly from base to tip.\n\n"
        "Helpful for catching tail or limb points clicked in the wrong sequence.",
    ),
    "missing": (
        "Missing labelable node",
        "Flags an animal that is missing a body part its neighbors usually "
        "have. For example, most mice in your project have a visible nose, but "
        "this one is missing its nose label even though the nose looks "
        "visible.\n\n"
        "Use this to find parts that were skipped or forgotten during labeling.",
    ),
    "appearance": (
        "Appearance / wrong-object",
        "Flags a body part placed on the wrong-looking spot in the image — for "
        "example, a paw point that landed on the bedding or cage floor instead "
        "of on the animal's fur.\n\n"
        "It looks at the actual pixels around each point, so it catches points "
        "dragged off the animal.",
    ),
    "insample": (
        "In-sample model prediction",
        "Runs a trained pose model on your already-labeled frames and points "
        "out body parts the model is confident about but that you left "
        "unlabeled — likely parts that were visible but skipped.\n\n"
        "Note: this runs full model inference, so it can be slow on large "
        "projects.",
    ),
}


# Plain-language sentence templates for the primary issue on a flagged
# instance, keyed by the raw ``QCFlag.top_issue`` string (issue #2769, item 7).
# Used to rewrite the Selected Instance / Statistics panels in friendly terms
# instead of raw feature names. Keys cover the forced hard-rule labels, the
# per-channel labels, and the inferred issue labels from QCResults; anything not
# listed falls back to a cleaned-up version of the raw label.
ISSUE_FRIENDLY = {
    # Forced hard-rule issues.
    "Whole-instance L/R flip": "left/right sides look swapped",
    "Wrong keypoint order along chain": "body parts traced out of order",
    # Per-channel issues.
    "Missing labelable node": "a body part looks missing",
    "Appearance outlier": "a point sits on the wrong-looking spot",
    "Model expects a labeled part here": "a visible part looks unlabeled",
    # Inferred (structural / GMM) issues.
    "Unusual edge length": "two parts are an unusual distance apart",
    "Unusual proportions": "the body proportions look off",
    "Unusual joint angle": "a joint bends at an unusual angle",
    "Unusual pose structure": "the overall pose looks unusual",
    "Unusual node spacing": "the spacing between parts looks off",
    "Unusual scale": "the animal looks an unusual size",
    "Isolated node": "one part sits far from the rest",
    "Inconsistent spacing": "the spacing between parts is inconsistent",
    "Likely L/R swap": "left/right sides may be swapped",
    "Unusual visibility": "an unusual set of parts is visible",
    "Isolated invisible node": "a part is hidden where you'd expect it shown",
    "Unusual visibility pattern": "which parts are visible looks unusual",
    "Unusual pose shape": "the pose shape looks unusual",
    "Unusual curvature": "the body curves in an unusual way",
    "Unusual pose extent": "the pose spreads out unusually",
    "Unknown": "something looks unusual",
}


# Placeholder shown in the Selected Instance panel before a row is chosen.
SELECT_INSTANCE_PLACEHOLDER = "Click a row in the table to review why it was flagged."


def _friendly_issue(top_issue: str) -> str:
    """Map a raw ``QCFlag.top_issue`` to a short plain-language phrase.

    Args:
        top_issue: The raw issue label from a QCFlag (e.g.
            ``"Whole-instance L/R flip"`` or ``"Unusual joint angle"``).

    Returns:
        A lowercase clause suitable for dropping into a sentence such as
        "Flagged: <clause> (confidence ...)". Unknown labels fall back to a
        cleaned-up, lowercased version of the raw text.
    """
    if top_issue in ISSUE_FRIENDLY:
        return ISSUE_FRIENDLY[top_issue]
    # Fallback: strip any "High " prefix and the underscores from raw feature
    # names so even unmapped issues read reasonably.
    cleaned = top_issue
    if cleaned.startswith("High "):
        cleaned = cleaned[len("High ") :]
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned.lower() if cleaned else "something looks unusual"


class CheckableFilterMenu(QtWidgets.QMenu):
    """A QMenu that stays open while the user toggles its checkable items.

    Used for the multi-select issue-type filter on the flagged-instances list
    (issue #2769, item 5 / feature request #2758): a plain QMenu closes after
    every click, which is tedious when ticking several issue types in a row.
    Here a click on a *checkable* action toggles it in place and keeps the menu
    open; non-checkable actions (e.g. "Select all") behave normally.
    """

    def mouseReleaseEvent(self, event):
        action = self.activeAction()
        if action is not None and action.isEnabled() and action.isCheckable():
            # Toggle in place and keep the menu open for more selections.
            action.trigger()
            return
        super().mouseReleaseEvent(event)


class CollapsibleGroupBox(QtWidgets.QGroupBox):
    """A checkable group box that collapses its body like an HTML ``<details>``.

    Acts as a lightweight disclosure section: the check state of the group box
    header doubles as an expand/collapse toggle, hiding the body so the panel
    actually shrinks for a cleaner first-time view (issue #2769, item 4). The
    header reads like a GitHub ``<details>``/``<summary>`` disclosure -- a
    ``▶`` arrow when collapsed and a ``▼`` arrow when expanded prefix the title
    so the click-to-expand affordance is obvious (issue #2769 follow-up).

    All body widgets live inside a single inner :attr:`content` frame; callers
    add their layout to ``content`` (e.g. ``QVBoxLayout(group.content)``).
    Collapsing toggles only the *visibility* of ``content`` -- never its
    *enabled* state -- so per-widget enabled/disabled logic inside the body
    keeps working even while collapsed. (A plain checkable QGroupBox would
    instead disable every descendant when unchecked, which would clobber that
    logic.)

    Args:
        title: The header text (shown after the disclosure arrow).
        collapsed: If True, start collapsed (unchecked) with the body hidden.
        parent: Parent widget.
    """

    # Disclosure arrows shown before the title, mirroring an HTML ``<details>``.
    _ARROW_EXPANDED = "▼"  # ▼
    _ARROW_COLLAPSED = "▶"  # ▶

    def __init__(
        self,
        title: str = "",
        collapsed: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        # Keep the caller's title separate from the arrow prefix so toggling can
        # rebuild the displayed title without losing the original text.
        self._base_title = title
        super().__init__(title, parent)

        # Single body frame holding everything the caller adds.
        self.content = QtWidgets.QWidget(self)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.content)

        self.setCheckable(True)
        # Hide the native check-box indicator so the header shows ONLY the ▶/▼
        # disclosure arrow, not a check box. A checkable QGroupBox otherwise
        # draws a native checkbox next to the title (very visible on macOS)
        # (issue #2769 follow-up). The box stays checkable -- isChecked()/
        # setChecked()/toggled and click-to-expand keep working -- and styling
        # the indicator to zero size moves rendering onto the stylesheet engine
        # so the checkbox is gone on every platform.
        self.setStyleSheet("QGroupBox::indicator { width: 0px; height: 0px; }")
        # A pointing-hand cursor reinforces that the whole header is clickable,
        # like a disclosure summary.
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setChecked(not collapsed)
        # Re-apply collapse state whenever the header is toggled.
        self.toggled.connect(self._on_toggled)
        self._on_toggled(self.isChecked())

    def _on_toggled(self, checked: bool):
        """Show/hide the body and update the disclosure arrow to match state."""
        self.content.setVisible(checked)
        # A checkable QGroupBox disables its direct children when unchecked;
        # undo that on the body frame so the body's own enabled-state logic is
        # preserved (collapse is visibility-only here).
        self.content.setEnabled(True)
        self._update_title_arrow(checked)
        # Let the layout reclaim/release the space.
        self.updateGeometry()

    def _update_title_arrow(self, expanded: bool):
        """Prefix the title with a ▶/▼ disclosure arrow for the given state."""
        arrow = self._ARROW_EXPANDED if expanded else self._ARROW_COLLAPSED
        # ``setTitle`` triggers no signal, so this is safe inside ``_on_toggled``.
        super().setTitle(f"{arrow}  {self._base_title}" if self._base_title else arrow)

    def setTitle(self, title: str):  # noqa: N802 (Qt override)
        """Set the header text, keeping the leading disclosure arrow in sync."""
        self._base_title = title
        self._update_title_arrow(self.isChecked())

    def title(self) -> str:  # noqa: N802 (Qt override)
        """Return the caller-supplied title without the disclosure arrow."""
        return self._base_title

    def apply_collapsed_state(self):
        """Apply the current check state to body visibility.

        Kept for callers that add children after construction; toggling is
        already handled on construction and on every header click.
        """
        self._on_toggled(self.isChecked())


class QCScoreCanvas(Canvas):
    """Matplotlib canvas for displaying QC score distribution.

    Provides histogram visualization with threshold indicator and
    click-to-select functionality.

    Signals:
        threshold_changed: Emitted when user clicks to set threshold.
            Argument is the new threshold value (0-1).
    """

    threshold_changed = QtCore.Signal(float)

    def __init__(self, width: int = 6, height: int = 3, dpi: int = 100):
        """Initialize the canvas.

        Args:
            width: Figure width in inches.
            height: Figure height in inches.
            dpi: Dots per inch for the figure.
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.axes = self.fig.add_subplot(111)

        super().__init__(self.fig)

        # Use Preferred policy instead of Expanding to prevent unbounded growth
        # when docked. This respects size hints without fighting the splitter.
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        self.setMinimumSize(300, 150)

        self._scores: np.ndarray = np.array([])
        self._threshold: float = 0.7
        self._threshold_line = None

        # Connect click event for threshold adjustment
        self.mpl_connect("button_press_event", self._on_click)

        self._setup_axes()

    def _setup_axes(self):
        """Configure the axes appearance."""
        self.axes.set_xlabel("Anomaly Score", fontsize=10)
        self.axes.set_ylabel("Count", fontsize=10)
        self.axes.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        self.axes.tick_params(labelsize=9)

    def set_scores(self, scores: np.ndarray):
        """Set the anomaly scores to display.

        Args:
            scores: Array of anomaly scores (0-1).
        """
        self._scores = scores
        self.update_plot()

    def set_threshold(self, threshold: float):
        """Set the threshold line position.

        Args:
            threshold: Threshold value (0-1).
        """
        self._threshold = threshold
        self.update_plot()

    def update_plot(self):
        """Redraw the plot with current data and threshold."""
        self.axes.clear()
        self._setup_axes()

        if len(self._scores) == 0:
            self.axes.text(
                0.5,
                0.5,
                "No data\n\nClick 'Run Analysis' to start",
                ha="center",
                va="center",
                transform=self.axes.transAxes,
                fontsize=11,
                color="gray",
            )
            self.draw()
            return

        # Draw histogram with fixed bins from 0 to 1
        bins = np.linspace(0, 1, 51)  # 50 bins for finer detail
        n_flagged = np.sum(self._scores >= self._threshold)
        n_total = len(self._scores)

        # Color bars based on threshold
        counts, bin_edges, patches = self.axes.hist(
            self._scores,
            bins=bins,
            alpha=0.7,
            edgecolor="white",
        )

        # Color bars based on whether they're above/below threshold
        for patch, left_edge in zip(patches, bin_edges[:-1]):
            if left_edge >= self._threshold:
                patch.set_facecolor("#dc3545")  # Red for flagged
            else:
                patch.set_facecolor("#6c757d")  # Gray for normal

        # Draw threshold line
        self._threshold_line = self.axes.axvline(
            self._threshold,
            color="#007bff",
            linestyle="--",
            linewidth=2,
            label=f"Threshold: {self._threshold:.2f}",
        )

        # Add annotation for flagged count
        self.axes.annotate(
            f"{n_flagged} flagged\n({100 * n_flagged / n_total:.1f}%)",
            xy=(self._threshold + 0.02, self.axes.get_ylim()[1] * 0.9),
            fontsize=9,
            color="#dc3545",
            fontweight="bold",
        )

        self.axes.set_xlim(0, 1)
        self.axes.set_title(
            f"Score Distribution (n={n_total})",
            fontsize=11,
        )
        self.axes.legend(loc="upper left", fontsize=8)

        self.draw()

    def _on_click(self, event):
        """Handle click event to set threshold."""
        if event.inaxes != self.axes:
            return

        # Get x coordinate of click
        x = event.xdata
        if x is not None and 0 <= x <= 1:
            self.threshold_changed.emit(float(x))


class QCBreakdownCanvas(Canvas):
    """Matplotlib canvas for displaying error type breakdown.

    Shows a horizontal bar chart of top issues.
    """

    def __init__(self, width: int = 6, height: int = 2.5, dpi: int = 100):
        """Initialize the canvas."""
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.axes = self.fig.add_subplot(111)

        super().__init__(self.fig)

        # Use Preferred policy instead of Expanding to prevent unbounded growth
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        self.setMinimumSize(300, 120)

        self._issue_counts: dict = {}

    def set_issue_counts(self, issue_counts: dict):
        """Set the issue type counts to display.

        Args:
            issue_counts: Dict mapping issue name to count.
        """
        self._issue_counts = issue_counts
        self.update_plot()

    def update_plot(self):
        """Redraw the breakdown chart."""
        self.axes.clear()

        if not self._issue_counts:
            self.axes.text(
                0.5,
                0.5,
                "No flagged instances",
                ha="center",
                va="center",
                transform=self.axes.transAxes,
                fontsize=11,
                color="gray",
            )
            self.axes.set_title("Issue Breakdown", fontsize=11)
            self.draw()
            return

        # Sort by count descending, show ALL issue types
        sorted_issues = sorted(
            self._issue_counts.items(), key=lambda x: x[1], reverse=True
        )

        labels = [item[0] for item in sorted_issues]
        counts = [item[1] for item in sorted_issues]
        max_count = max(counts) if counts else 1

        # Horizontal bar chart
        y_pos = np.arange(len(labels))
        bars = self.axes.barh(y_pos, counts, color="#dc3545", alpha=0.7)

        self.axes.set_yticks(y_pos)
        self.axes.set_yticklabels(labels, fontsize=9)
        self.axes.invert_yaxis()  # Top to bottom
        self.axes.set_xlabel("Count", fontsize=10)
        self.axes.set_title("Issue Breakdown", fontsize=11)

        # Add count labels - inside bar (white) if bar is wide enough, else outside
        for bar, count in zip(bars, counts):
            bar_width = bar.get_width()
            y_center = bar.get_y() + bar.get_height() / 2

            # If bar is at least 20% of max width, put label inside
            if bar_width >= max_count * 0.2:
                self.axes.text(
                    bar_width - max_count * 0.02,  # Slightly inside right edge
                    y_center,
                    str(count),
                    va="center",
                    ha="right",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )
            else:
                # Put label outside the bar
                self.axes.text(
                    bar_width + max_count * 0.02,
                    y_center,
                    str(count),
                    va="center",
                    ha="left",
                    fontsize=9,
                    color="#333",
                )

        # Add some padding on the right for labels
        self.axes.set_xlim(0, max_count * 1.15)

        self.draw()


class QCFeatureCanvas(Canvas):
    """Matplotlib canvas for displaying feature distributions.

    Shows box plots comparing flagged vs non-flagged instances across
    top contributing features.
    """

    def __init__(self, width: int = 6, height: int = 2.5, dpi: int = 100):
        """Initialize the canvas."""
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.axes = self.fig.add_subplot(111)

        super().__init__(self.fig)

        # Use Preferred policy instead of Expanding to prevent unbounded growth
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        self.setMinimumSize(300, 120)

        self._feature_data: dict = {}  # {feature_name: (normal_values, flagged_values)}
        self._top_features: list = []

    def set_feature_data(
        self,
        feature_contributions: dict,
        instance_scores: dict,
        threshold: float,
        feature_names: list,
    ):
        """Set the feature data to display.

        Args:
            feature_contributions: Dict mapping InstanceKey to feature dict.
            instance_scores: Dict mapping InstanceKey to score.
            threshold: Threshold for flagging instances.
            feature_names: List of all feature names.
        """
        if not feature_contributions or not feature_names:
            self._feature_data = {}
            self._top_features = []
            self.update_plot()
            return

        # Separate flagged vs normal
        normal_features = {name: [] for name in feature_names}
        flagged_features = {name: [] for name in feature_names}

        for key, contributions in feature_contributions.items():
            score = instance_scores.get(key, 0)
            target = flagged_features if score >= threshold else normal_features

            for name in feature_names:
                val = contributions.get(name, 0)
                if np.isfinite(val):
                    target[name].append(val)

        # Find top discriminating features by difference in means
        feature_scores = []
        for name in feature_names:
            normal_vals = normal_features.get(name, [])
            flagged_vals = flagged_features.get(name, [])

            if normal_vals and flagged_vals:
                normal_mean = np.mean(normal_vals)
                normal_std = np.std(normal_vals) or 1.0
                flagged_mean = np.mean(flagged_vals)
                # Z-score of difference
                diff = abs(flagged_mean - normal_mean) / normal_std
                feature_scores.append((name, diff))

        # Sort by discriminating power and take top 6
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        self._top_features = [name for name, _ in feature_scores[:6]]

        # Store the data
        self._feature_data = {
            name: (normal_features[name], flagged_features[name])
            for name in self._top_features
        }

        self.update_plot()

    def update_plot(self):
        """Redraw the feature comparison chart."""
        self.axes.clear()

        if not self._feature_data or not self._top_features:
            self.axes.text(
                0.5,
                0.5,
                "No feature data\n\nRun analysis to see feature distributions",
                ha="center",
                va="center",
                transform=self.axes.transAxes,
                fontsize=11,
                color="gray",
            )
            self.axes.set_title("Feature Comparison", fontsize=11)
            self.draw()
            return

        # Prepare data for box plots
        positions = []
        box_data = []
        colors = []
        tick_labels = []

        for i, name in enumerate(self._top_features):
            normal_vals, flagged_vals = self._feature_data[name]
            base_pos = i * 2.5

            # Normal values
            if normal_vals:
                positions.append(base_pos)
                box_data.append(normal_vals)
                colors.append("#6c757d")  # Gray for normal
                tick_labels.append("")

            # Flagged values
            if flagged_vals:
                positions.append(base_pos + 0.8)
                box_data.append(flagged_vals)
                colors.append("#dc3545")  # Red for flagged
                tick_labels.append("")

        if not box_data:
            self.axes.text(
                0.5,
                0.5,
                "Insufficient data for comparison",
                ha="center",
                va="center",
                transform=self.axes.transAxes,
                fontsize=11,
                color="gray",
            )
            self.draw()
            return

        # Create box plots
        bp = self.axes.boxplot(
            box_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,  # Hide outliers for cleaner view
        )

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Set x-axis labels
        feature_positions = [i * 2.5 + 0.4 for i in range(len(self._top_features))]
        self.axes.set_xticks(feature_positions)
        # Shorten feature names
        short_names = [
            name.replace("_", " ").replace(" zscore", "")[:12]
            for name in self._top_features
        ]
        self.axes.set_xticklabels(short_names, fontsize=8, rotation=45, ha="right")

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#6c757d", alpha=0.7, label="Normal"),
            Patch(facecolor="#dc3545", alpha=0.7, label="Flagged"),
        ]
        self.axes.legend(handles=legend_elements, loc="upper right", fontsize=8)

        self.axes.set_ylabel("Feature Value", fontsize=9)
        self.axes.set_title("Top Discriminating Features", fontsize=11)
        self.axes.grid(True, alpha=0.3, axis="y")

        self.draw()


class QCSkeletonTraceCanvas(Canvas):
    """Matplotlib canvas that draws a skeleton and traces an ordered chain.

    This is the interactive heart of the chain-order "skeleton tracing" UX
    (issue #2769, item 2). Instead of typing node names, the user *clicks*
    nodes in order on a rendered skeleton; each click appends that node to the
    chain currently being traced, and the canvas highlights the trace in click
    order with numbered badges so the path is obvious at a glance.

    Where possible the skeleton is drawn over a *real* labeled frame: the owning
    widget picks one labeled instance with as many nodes present as possible,
    decodes that frame's image, and hands both the image and the instance's
    per-node *pixel* coordinates to :meth:`set_skeleton`. The canvas then shows
    the photo as the background (``imshow``, top-left origin like an image) and
    overlays the markers/edges/trace badges at their true pixel positions, so the
    user traces directly on the actual animal (issue #2769 follow-up). The view
    is zoomable (mouse wheel) and pannable (middle/right drag) so a small animal
    can be enlarged; double-click resets to the full image.

    When no frame image is available the canvas falls back to the abstract
    layout: a representative per-node coordinate (or a deterministic seeded
    spring layout, or a horizontal line if networkx is missing) normalized into a
    ``[-1, 1]`` frame on a plain white background. Either way the layout is
    stable across redraws.

    Clicking with the *left* button selects the nearest node within a pick radius
    and emits :attr:`node_clicked`; the owning widget owns the chain list and
    calls :meth:`set_trace` to update the highlighted order. The middle/right
    buttons pan instead of selecting.

    Signals:
        node_clicked: Emitted with the node *name* (str) when the user clicks a
            node. The owning widget decides whether to append/toggle it in the
            chain being traced.
    """

    node_clicked = QtCore.Signal(str)

    def __init__(self, width: int = 7, height: int = 5, dpi: int = 100):
        """Initialize the canvas.

        Args:
            width: Figure width in inches.
            height: Figure height in inches.
            dpi: Dots per inch for the figure.
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.axes = self.fig.add_subplot(111)

        super().__init__(self.fig)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        # Bigger minimum so the animal is clearly visible; the user explicitly
        # asked for a larger, zoomable canvas.
        self.setMinimumSize(480, 380)

        # Node display positions keyed by node name: {name: (x, y)}. In image
        # mode these are raw *pixel* coordinates; in the abstract fallback they
        # are *normalized* (roughly [-1, 1]); see :meth:`_compute_layout`.
        self._positions: dict = {}
        self._node_names: list = []
        self._edges: list = []  # list of (src_name, dst_name)
        # Optional representative per-node coordinates from real labeled
        # instances ({name: (x, y)} in raw label/image space). When present
        # these drive the abstract layout (a real animal shape); when empty the
        # canvas falls back to the spring/line layout.
        self._node_coords: dict = {}
        # The labeled frame image to show as the canvas background (np.ndarray,
        # 2D grayscale or HxWx3 RGB), or None to use the abstract white layout.
        self._background_image = None
        # The matplotlib AxesImage handle for the background (when shown).
        self._image_artist = None
        # The chain currently being traced, in click order (node names).
        self._trace: list = []
        # Click hit-test radius for the abstract layout (data coords). Both the
        # spring layout and the real-coordinate layout are normalized to
        # ~[-1, 1], so this fixed radius works for either. In image (pixel) mode
        # the pick radius is derived from the current view extent instead, so it
        # keeps working after zooming/panning -- see :meth:`node_at`.
        self._pick_radius = 0.18

        # Pan bookkeeping: the data-space anchor recorded on a middle/right
        # button press, used to translate the view on drag.
        self._pan_anchor = None

        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_pan_press)
        self.mpl_connect("motion_notify_event", self._on_pan_move)
        self.mpl_connect("button_release_event", self._on_pan_release)
        self.update_plot()

    def set_skeleton(
        self, node_names: list, edges: list, node_positions=None, image=None
    ):
        """Set the skeleton to display and recompute the layout.

        Args:
            node_names: Ordered list of node-name strings.
            edges: List of ``(source_name, destination_name)`` tuples.
            node_positions: Optional dict mapping node name to a representative
                ``(x, y)`` coordinate from real labeled instances (raw label
                space). When given (and non-empty), the skeleton is drawn using
                those coordinates. If an ``image`` is also given these are
                treated as raw *pixel* coordinates and drawn over the image at
                their true positions; otherwise they are centered/scaled into the
                abstract ``[-1, 1]`` frame (like the SLEAP skeleton builder).
                When omitted the canvas falls back to the spring/line layout.
                May also be set separately via :meth:`set_node_positions`.
            image: Optional labeled-frame image (``np.ndarray``; 2D grayscale,
                ``(H, W, 1)`` grayscale, or ``(H, W, 3)`` RGB) to show as the
                canvas background. When provided together with ``node_positions``
                the canvas enters *image mode*: the photo is shown with a
                top-left origin and the skeleton is overlaid at pixel coords.
                ``None`` keeps the plain abstract layout.
        """
        self._node_names = list(node_names)
        # Keep only edges whose endpoints are present, as name pairs.
        valid = set(self._node_names)
        self._edges = [(s, d) for (s, d) in edges if s in valid and d in valid]
        if node_positions is not None:
            # Keep only coords for nodes that exist in this skeleton.
            self._node_coords = {
                name: (float(x), float(y))
                for name, (x, y) in node_positions.items()
                if name in valid
            }
        else:
            # A new skeleton without coords drops any stale real coordinates.
            self._node_coords = {}
        # Image mode requires both an image and at least one real pixel coord;
        # otherwise we cannot place the skeleton on the photo, so fall back.
        self._background_image = self._coerce_image(image)
        if self._background_image is not None and self._node_coords:
            # Pixel mode: draw nodes at their true image coordinates (no
            # normalization), so they line up with the animal in the photo.
            self._positions = {
                name: (float(x), float(y))
                for name, (x, y) in self._node_coords.items()
                if np.isfinite(x) and np.isfinite(y)
            }
        else:
            self._background_image = None
            self._positions = self._compute_layout()
        # Drop any traced nodes that no longer exist in this skeleton.
        self._trace = [n for n in self._trace if n in valid]
        # A new skeleton/image is a structural change: start from the full
        # extent rather than preserving a stale zoom from a previous project.
        self.axes.set_xlim(0.0, 1.0)
        self.axes.set_ylim(0.0, 1.0)
        self._pan_anchor = None
        self.update_plot()

    @staticmethod
    def _coerce_image(image):
        """Normalize a frame image into a 2D/RGB array ``imshow`` can render.

        Squeezes a trailing singleton channel (``(H, W, 1)`` grayscale) down to
        ``(H, W)`` and passes ``(H, W, 3)`` RGB through unchanged. Anything that
        is not a usable 2D/RGB image (wrong rank, empty, or not array-like)
        returns ``None`` so the caller falls back to the abstract layout.

        Args:
            image: Candidate image (``np.ndarray`` or array-like), or ``None``.

        Returns:
            A 2D ``(H, W)`` or 3D ``(H, W, 3)`` ``np.ndarray``, or ``None``.
        """
        if image is None:
            return None
        try:
            arr = np.asarray(image)
        except Exception:
            return None
        if arr.size == 0:
            return None
        # Drop a leading singleton (e.g. a (1, H, W, C) single-frame stack).
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            # Keep RGB; drop an alpha channel for a stable imshow.
            return arr[..., :3]
        return None

    def set_node_positions(self, node_positions: dict):
        """Set representative per-node coordinates from real labeled instances.

        Drives the layout so the skeleton is drawn in the actual animal's shape
        (like the SLEAP skeleton builder) rather than an abstract spring layout.
        Coordinates are given in raw label/image space and are centered/scaled to
        fill the canvas; only nodes present in the current skeleton are kept. An
        empty dict reverts to the spring/line fallback.

        Args:
            node_positions: Dict mapping node name to an ``(x, y)`` coordinate.
        """
        valid = set(self._node_names)
        self._node_coords = {
            name: (float(x), float(y))
            for name, (x, y) in (node_positions or {}).items()
            if name in valid
        }
        # Setting positions without an image is an abstract-layout update; drop
        # any stale background so we do not draw normalized coords over a photo.
        self._background_image = None
        self._positions = self._compute_layout()
        # Structural change: reset any zoom/pan to the full extent.
        self.axes.set_xlim(0.0, 1.0)
        self.axes.set_ylim(0.0, 1.0)
        self._pan_anchor = None
        self.update_plot()

    def _compute_layout(self) -> dict:
        """Compute a node layout for the current skeleton.

        Prefers a *real* animal layout built from labeled coordinates
        (:attr:`_node_coords`) so the skeleton looks like the actual animal.
        When no labeled coordinates are available it falls back to a seeded
        spring layout (stable across redraws), or a horizontal line if networkx
        is unavailable. All layouts are normalized to roughly ``[-1, 1]`` so the
        fixed click-pick radius works regardless of which one is used.

        Returns:
            Dict mapping node name to an ``(x, y)`` position in the normalized
            display frame.
        """
        if not self._node_names:
            return {}
        # Prefer real labeled coordinates when we have at least two nodes with
        # finite positions (a single point gives no scale to normalize against).
        real = self._layout_from_coords()
        if real is not None:
            return real
        try:
            import networkx as nx

            graph = nx.Graph()
            graph.add_nodes_from(self._node_names)
            graph.add_edges_from(self._edges)
            # Seed keeps the picture stable so users build muscle memory.
            pos = nx.spring_layout(graph, seed=42)
            return {name: (float(p[0]), float(p[1])) for name, p in pos.items()}
        except Exception:
            # Fallback: lay nodes out on a horizontal line in node order.
            n = len(self._node_names)
            if n == 1:
                return {self._node_names[0]: (0.0, 0.0)}
            return {
                name: (-1.0 + 2.0 * i / (n - 1), 0.0)
                for i, name in enumerate(self._node_names)
            }

    def _layout_from_coords(self):
        """Build a normalized layout from real labeled node coordinates.

        Centers the labeled coordinates on their centroid and scales them
        uniformly (preserving the animal's aspect ratio) so the largest extent
        fills the ``[-1, 1]`` display frame. The image-space y-axis (which grows
        downward) is flipped so the animal is drawn the right way up. Nodes with
        no labeled coordinate are skipped entirely (no marker is drawn for them).

        Returns:
            A dict mapping node name to a normalized ``(x, y)`` position, or
            ``None`` when fewer than two nodes have finite labeled coordinates
            (in which case the caller falls back to the spring/line layout).
        """
        coords = self._node_coords
        if not coords:
            return None
        names = [
            name
            for name in self._node_names
            if name in coords
            and np.isfinite(coords[name][0])
            and np.isfinite(coords[name][1])
        ]
        if len(names) < 2:
            return None

        pts = np.array([coords[name] for name in names], dtype=float)
        center = pts.mean(axis=0)
        centered = pts - center
        # Uniform scale by the largest half-extent keeps the aspect ratio and
        # maps the widest axis into [-1, 1].
        half_extent = float(np.max(np.abs(centered))) if centered.size else 0.0
        scale = 1.0 / half_extent if half_extent > 0 else 1.0
        normalized = centered * scale
        # Flip y so image coordinates (y down) draw the animal upright.
        return {
            name: (float(normalized[i, 0]), float(-normalized[i, 1]))
            for i, name in enumerate(names)
        }

    def set_trace(self, trace: list):
        """Set the chain-in-progress (node names, in click order) and redraw.

        Args:
            trace: Ordered list of node names currently in the trace.
        """
        self._trace = [n for n in trace if n in self._positions]
        self.update_plot()

    @property
    def trace(self) -> list:
        """The current chain-in-progress as an ordered list of node names."""
        return list(self._trace)

    def update_plot(self):
        """Redraw the skeleton, edges, and the highlighted trace path.

        Draws over the labeled-frame photo at true pixel coordinates when an
        image is set (image mode), otherwise on a plain white background using
        the normalized abstract layout. Preserves the current zoom/pan view when
        only the trace changes, so wheel-zooming and then clicking nodes does not
        snap the view back to the full extent.
        """
        image_mode = self._background_image is not None
        # Remember the current view so a redraw triggered by a trace edit keeps
        # the user's zoom/pan. We only restore it when the axes already held a
        # real view (not the initial 0..1 default) and the data is unchanged.
        prev_xlim = self.axes.get_xlim()
        prev_ylim = self.axes.get_ylim()
        had_view = prev_xlim != (0.0, 1.0) or prev_ylim != (0.0, 1.0)

        self.axes.clear()
        self._image_artist = None
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        for spine in self.axes.spines.values():
            spine.set_visible(False)

        if not self._positions and not image_mode:
            self.axes.text(
                0.5,
                0.5,
                "No skeleton available.\nLoad a project with a skeleton to trace "
                "a chain,\nor type the chain below.",
                ha="center",
                va="center",
                transform=self.axes.transAxes,
                fontsize=9,
                color="gray",
            )
            self.draw()
            return

        # --- Background image (image mode) ---------------------------------
        if image_mode:
            img = self._background_image
            kwargs = {"origin": "upper", "interpolation": "nearest", "zorder": 0}
            if img.ndim == 2:
                kwargs["cmap"] = "gray"
            self._image_artist = self.axes.imshow(img, **kwargs)

        # In pixel (image) mode markers/labels are sized in pixels relative to
        # the image, so scale them off the image size; the abstract layout keeps
        # the original compact styling.
        if image_mode:
            h, w = self._background_image.shape[:2]
            span = float(max(h, w)) or 1.0
            marker_s = max(80.0, (span * 0.012) ** 2)
            edge_lw = max(1.2, span / 400.0)
            trace_lw = max(2.0, span / 250.0)
            badge_fs = 9
            label_fs = 8
            label_dy = 14
        else:
            marker_s = 260
            edge_lw = 1.2
            trace_lw = 2.5
            badge_fs = 9
            label_fs = 8
            label_dy = 12

        # Draw skeleton edges first so nodes sit on top. Skip any edge whose
        # endpoint lacks a position (e.g. a node with no labeled coordinate in
        # the real-animal layout).
        edge_color = "#e8f0ff" if image_mode else "#ced4da"
        for src, dst in self._edges:
            if src not in self._positions or dst not in self._positions:
                continue
            x0, y0 = self._positions[src]
            x1, y1 = self._positions[dst]
            self.axes.plot(
                [x0, x1], [y0, y1], color=edge_color, linewidth=edge_lw, zorder=1
            )

        # Draw the trace path (in click order) as a highlighted poly-line.
        drawable_trace = [n for n in self._trace if n in self._positions]
        if len(drawable_trace) >= 2:
            tx = [self._positions[n][0] for n in drawable_trace]
            ty = [self._positions[n][1] for n in drawable_trace]
            self.axes.plot(
                tx, ty, color="#007bff", linewidth=trace_lw, zorder=2, alpha=0.9
            )

        # Map node -> 1-based position in the trace (for numbered badges).
        trace_order = {name: i + 1 for i, name in enumerate(self._trace)}

        # Draw nodes.
        for name, (x, y) in self._positions.items():
            in_trace = name in trace_order
            self.axes.scatter(
                [x],
                [y],
                s=marker_s,
                facecolor="#007bff" if in_trace else "#e9ecef",
                edgecolor="#0056b3" if in_trace else "#adb5bd",
                linewidths=1.5,
                zorder=3,
            )
            # Numbered badge for traced nodes; small name label for all.
            if in_trace:
                self.axes.text(
                    x,
                    y,
                    str(trace_order[name]),
                    ha="center",
                    va="center",
                    fontsize=badge_fs,
                    fontweight="bold",
                    color="white",
                    zorder=4,
                )
            self.axes.annotate(
                name,
                xy=(x, y),
                xytext=(0, label_dy),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=label_fs,
                color=(
                    "#f1f3f5" if image_mode else ("#212529" if in_trace else "#495057")
                ),
                zorder=4,
                bbox=(
                    dict(boxstyle="round,pad=0.15", fc="#212529aa", ec="none")
                    if image_mode
                    else None
                ),
            )

        if image_mode:
            self.axes.set_aspect("equal", adjustable="box")
            if had_view:
                # Preserve the user's zoom/pan across a trace-only redraw.
                self.axes.set_xlim(prev_xlim)
                self.axes.set_ylim(prev_ylim)
            else:
                # Default view: fit the instance (node bounding box + a margin)
                # rather than the whole frame, so the animal is large and easy to
                # trace (issue #2769 follow-up). y inverted -> top-left origin.
                xlim, ylim = self._instance_fit_limits()
                self.axes.set_xlim(xlim)
                self.axes.set_ylim(ylim)
        else:
            # Pad the limits so labels are not clipped. Use ``adjustable="box"``
            # so the equal-aspect constraint resizes the axes box rather than
            # silently overriding the explicit limits (which would warn).
            xs = [p[0] for p in self._positions.values()]
            ys = [p[1] for p in self._positions.values()]
            pad = 0.35
            self.axes.set_aspect("equal", adjustable="box")
            self.axes.set_xlim(min(xs) - pad, max(xs) + pad)
            self.axes.set_ylim(min(ys) - pad, max(ys) + pad)

        self.draw()

    def _instance_fit_limits(self):
        """Default image-mode view limits: fit the instance plus a margin.

        Returns ``(xlim, ylim)`` in pixel coordinates (``ylim`` inverted for a
        top-left origin) framing the instance's node bounding box plus a margin
        and clamped to the image, so the animal fills the canvas instead of
        sitting tiny inside the full frame (issue #2769 follow-up). Falls back to
        the full image extent when there are no finite node positions.
        """
        h, w = self._background_image.shape[:2]
        full = ((-0.5, w - 0.5), (h - 0.5, -0.5))
        pts = [
            p
            for p in self._positions.values()
            if np.isfinite(p[0]) and np.isfinite(p[1])
        ]
        if not pts:
            return full
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        # Margin: a quarter of the larger bbox side, with a floor so a tiny or
        # single-node box still gets a sensible window.
        span = max(xmax - xmin, ymax - ymin)
        margin = max(span * 0.25, 0.05 * max(w, h), 10.0)
        x0 = max(xmin - margin, -0.5)
        x1 = min(xmax + margin, w - 0.5)
        y0 = max(ymin - margin, -0.5)
        y1 = min(ymax + margin, h - 0.5)
        if x1 <= x0 or y1 <= y0:
            return full
        return (x0, x1), (y1, y0)  # ylim inverted -> top-left origin

    def reset_view(self):
        """Reset to the default view (fit the instance, or the layout extent)."""
        # Forget any zoom/pan, then let update_plot recompute the default view:
        # instance-fit in image mode, full layout extent otherwise.
        self.axes.set_xlim(0.0, 1.0)
        self.axes.set_ylim(0.0, 1.0)
        self.update_plot()

    def _current_pick_radius(self) -> float:
        """Pick radius in *data* units for the currently active layout.

        The abstract layout is normalized to ``~[-1, 1]`` so a fixed radius
        works. In image (pixel) mode the data units are image pixels and the
        view changes with zoom, so the radius is taken as a fraction of the
        current visible width -- this keeps the clickable target a roughly
        constant on-screen size after zooming and panning.

        Returns:
            The hit-test radius in the active data coordinate space.
        """
        if self._background_image is None:
            return self._pick_radius
        x0, x1 = self.axes.get_xlim()
        view_w = abs(x1 - x0)
        if not np.isfinite(view_w) or view_w <= 0:
            # No real view yet: fall back to a fraction of the image width.
            view_w = float(self._background_image.shape[1]) or 1.0
        # ~6% of the visible width; clamp so it never collapses to nothing.
        return max(view_w * 0.06, 1.0)

    def node_at(self, x: float, y: float) -> Optional[str]:
        """Return the node name nearest to ``(x, y)`` within the pick radius.

        Works in whichever coordinate space is currently active: normalized
        units for the abstract layout, image pixels for the photo overlay. In
        pixel mode the radius scales with the current view (see
        :meth:`_current_pick_radius`) so picking keeps working after zoom/pan.

        Args:
            x: X position in data coordinates.
            y: Y position in data coordinates.

        Returns:
            The nearest node name within the active pick radius, or None.
        """
        best_name = None
        best_dist = self._current_pick_radius()
        for name, (nx_, ny_) in self._positions.items():
            dist = ((nx_ - x) ** 2 + (ny_ - y) ** 2) ** 0.5
            if dist <= best_dist:
                best_dist = dist
                best_name = name
        return best_name

    def _on_click(self, event):
        """Handle a left click: select the nearest node, or reset on dbl-click.

        Left-button single clicks emit :attr:`node_clicked` for the nearest node
        (the tracing interaction). A left-button double-click resets the view to
        the full extent. Middle/right buttons are reserved for panning and are
        ignored here.
        """
        if event.inaxes != self.axes:
            return
        # Only the left button drives selection; middle/right pan instead.
        if event.button not in (1, None):
            return
        if getattr(event, "dblclick", False):
            self.reset_view()
            return
        if event.xdata is None or event.ydata is None:
            return
        name = self.node_at(event.xdata, event.ydata)
        if name is not None:
            self.node_clicked.emit(name)

    def _on_scroll(self, event):
        """Zoom the view in/out around the cursor on a mouse-wheel scroll.

        Scales the current x/y limits about ``(event.xdata, event.ydata)`` so the
        point under the cursor stays put: scrolling up zooms in, down zooms out.
        """
        if event.inaxes != self.axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        # Up zooms in (shrink the view), down zooms out (grow it).
        scale = 0.8 if event.button == "up" else 1.25
        x0, x1 = self.axes.get_xlim()
        y0, y1 = self.axes.get_ylim()
        cx, cy = event.xdata, event.ydata
        # Keep the cursor anchored: scale each side's distance from the cursor.
        self.axes.set_xlim(cx + (x0 - cx) * scale, cx + (x1 - cx) * scale)
        self.axes.set_ylim(cy + (y0 - cy) * scale, cy + (y1 - cy) * scale)
        self.draw()

    def _on_pan_press(self, event):
        """Record the pan anchor when a non-left button is pressed in the axes.

        The anchor is the press location in *display pixels* together with the
        axes limits at press time; tracking pixels (rather than data coords)
        avoids feedback when the limits move out from under the cursor mid-drag.
        """
        if event.inaxes != self.axes:
            return
        # Middle (2) or right (3) button starts a pan; left is for selection.
        if event.button in (2, 3):
            x0, x1 = self.axes.get_xlim()
            y0, y1 = self.axes.get_ylim()
            bbox = self.axes.get_window_extent()
            # Data units per display pixel, frozen at press time so the mapping
            # cannot drift as the limits move during the drag.
            self._pan_anchor = (
                event.x,
                event.y,
                x0,
                x1,
                y0,
                y1,
                (x1 - x0) / bbox.width if bbox.width else 0.0,
                (y1 - y0) / bbox.height if bbox.height else 0.0,
            )

    def _on_pan_move(self, event):
        """Translate the view to follow the cursor while panning.

        Converts the pixel movement since the press into a data-space shift
        using the frozen press-time pixel scale, then offsets the press-time
        limits by that shift so the grabbed point stays under the cursor.
        """
        if self._pan_anchor is None:
            return
        if event.x is None or event.y is None:
            return
        px, py, x0, x1, y0, y1, sx, sy = self._pan_anchor
        shift_x = (event.x - px) * sx
        shift_y = (event.y - py) * sy
        self.axes.set_xlim(x0 - shift_x, x1 - shift_x)
        self.axes.set_ylim(y0 - shift_y, y1 - shift_y)
        self.draw()

    def _on_pan_release(self, event):
        """End the current pan gesture."""
        if event.button in (2, 3):
            self._pan_anchor = None


class QCFlagTableModel(QtCore.QAbstractTableModel):
    """Table model for QC flagged instances.

    The trailing "Reviewed" column (issue #2769, item 6) is a user-toggled
    checkbox so reviewers can mark and see which flagged instances they have
    already looked at. The reviewed state is *not* stored on the row: it lives
    in a shared set of instance keys owned by the widget, so it survives
    threshold re-filters and issue-type filtering (which rebuild the row list).
    """

    # "Reviewed" is appended last so the existing data/sort column indices
    # (Frame=0 ... Issue=4) are unchanged.
    COLUMNS = ["Frame", "Instance", "Score", "Confidence", "Issue", "Reviewed"]
    REVIEWED_COL = 5

    def __init__(self, parent=None, reviewed_keys: Optional[set] = None):
        """Initialize the model.

        Args:
            parent: Parent QObject.
            reviewed_keys: Optional shared set of ``instance_key`` tuples that
                have been marked reviewed. The model reads/writes this set for
                the Reviewed column so the state is keyed by instance identity
                (video, frame, instance) rather than row index. If None, a new
                empty set is created.
        """
        super().__init__(parent)
        self._items: List["QCFlag"] = []
        self._reviewed_keys: set = reviewed_keys if reviewed_keys is not None else set()

    @property
    def items(self) -> List["QCFlag"]:
        """Get the current items."""
        return self._items

    @items.setter
    def items(self, value: List["QCFlag"]):
        """Set items and refresh the model."""
        self.beginResetModel()
        self._items = value
        self.endResetModel()

    @property
    def reviewed_keys(self) -> set:
        """Set of instance keys marked reviewed (shared with the widget)."""
        return self._reviewed_keys

    def is_reviewed(self, item: "QCFlag") -> bool:
        """Return True if the given flag's instance is marked reviewed."""
        return item.instance_key in self._reviewed_keys

    def set_reviewed(self, item: "QCFlag", reviewed: bool):
        """Mark a flag's instance reviewed/unreviewed and repaint its row.

        Args:
            item: The flag whose instance reviewed-state to set.
            reviewed: Target reviewed state.
        """
        key = item.instance_key
        changed = (key in self._reviewed_keys) != bool(reviewed)
        if reviewed:
            self._reviewed_keys.add(key)
        else:
            self._reviewed_keys.discard(key)
        if changed and item in self._items:
            row = self._items.index(item)
            top = self.index(row, 0)
            bottom = self.index(row, self.columnCount() - 1)
            self.dataChanged.emit(top, bottom)

    def reviewed_count(self) -> int:
        """Number of *currently shown* rows whose instance is reviewed."""
        return sum(
            1 for item in self._items if item.instance_key in self._reviewed_keys
        )

    def rowCount(self, parent=None) -> int:
        return len(self._items)

    def columnCount(self, parent=None) -> int:
        return len(self.COLUMNS)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return self.COLUMNS[section]
        return None

    def flags(self, index):
        base = super().flags(index)
        if index.isValid() and index.column() == self.REVIEWED_COL:
            # Reviewed is a user-checkable column.
            return base | QtCore.Qt.ItemIsUserCheckable
        return base

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._items):
            return None

        item = self._items[index.row()]
        col = index.column()

        if role == QtCore.Qt.DisplayRole:
            if col == 0:  # Frame
                return str(item.frame_idx)
            elif col == 1:  # Instance
                return str(item.instance_idx)
            elif col == 2:  # Score
                return f"{item.score:.3f}"
            elif col == 3:  # Confidence
                return item.confidence.title()
            elif col == 4:  # Issue
                return item.top_issue.replace("_", " ").title()
            # Reviewed column shows a checkbox (via CheckStateRole), no text.

        elif role == QtCore.Qt.CheckStateRole:
            if col == self.REVIEWED_COL:
                return (
                    QtCore.Qt.Checked
                    if item.instance_key in self._reviewed_keys
                    else QtCore.Qt.Unchecked
                )

        elif role == QtCore.Qt.TextAlignmentRole:
            if col == self.REVIEWED_COL:
                return int(QtCore.Qt.AlignCenter)

        elif role == QtCore.Qt.ForegroundRole:
            if col == 2:  # Score column
                if item.score >= 0.8:
                    return QtGui.QBrush(QtGui.QColor(220, 53, 69))  # Red
                elif item.score >= 0.6:
                    return QtGui.QBrush(QtGui.QColor(255, 193, 7))  # Yellow
            elif col == 3:  # Confidence column
                if item.confidence == "high":
                    return QtGui.QBrush(QtGui.QColor(220, 53, 69))
                elif item.confidence == "medium":
                    return QtGui.QBrush(QtGui.QColor(255, 193, 7))
                else:
                    return QtGui.QBrush(QtGui.QColor(108, 117, 125))

        return None

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        """Toggle reviewed-state when the user clicks the Reviewed checkbox."""
        if (
            index.isValid()
            and index.column() == self.REVIEWED_COL
            and role == QtCore.Qt.CheckStateRole
            and index.row() < len(self._items)
        ):
            item = self._items[index.row()]
            # value may arrive as a Qt.CheckState enum or its int form.
            reviewed = QtCore.Qt.CheckState(value) == QtCore.Qt.Checked
            self.set_reviewed(item, reviewed)
            return True
        return False

    def sort(self, column: int, order: QtCore.Qt.SortOrder = QtCore.Qt.AscendingOrder):
        """Sort the model by the given column.

        Args:
            column: Column index to sort by.
            order: Sort order (AscendingOrder or DescendingOrder).
        """
        self.beginResetModel()

        reverse = order == QtCore.Qt.DescendingOrder

        # Define sort key for each column
        if column == 0:  # Frame
            key = lambda x: x.frame_idx
        elif column == 1:  # Instance
            key = lambda x: x.instance_idx
        elif column == 2:  # Score
            key = lambda x: x.score
        elif column == 3:  # Confidence (high > medium > low)
            conf_order = {"high": 2, "medium": 1, "low": 0}
            key = lambda x: conf_order.get(x.confidence, -1)
        elif column == 4:  # Issue (alphabetical)
            key = lambda x: x.top_issue
        elif column == self.REVIEWED_COL:  # Reviewed (unreviewed first ascending)
            key = lambda x: x.instance_key in self._reviewed_keys
        else:
            key = lambda x: 0

        self._items.sort(key=key, reverse=reverse)
        self.endResetModel()


class QCAnalysisWorker(QThread):
    """Worker thread for running QC analysis in background.

    Args:
        labels: A sleap_io.Labels object to analyze.
        config: Optional QCConfig controlling which detectors run and their
            thresholds. If None, the detector falls back to QCConfig() defaults.

    Signals:
        progress: Emitted with (step_name, progress_pct, detail) during analysis.
        finished: Emitted with QCResults when analysis completes.
        error: Emitted with error message if analysis fails.
    """

    progress = QSignal(str, int, str)  # (step_name, progress_percent, detail)
    finished = QSignal(object)  # QCResults
    error = QSignal(str)

    def __init__(self, labels, config=None, parent=None):
        super().__init__(parent)
        self._labels = labels
        self._config = config
        self._results = None
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the analysis."""
        self._cancelled = True

    def run(self):
        """Run the QC analysis."""
        try:
            from sleap.qc import LabelQCDetector

            def progress_callback(step_name, progress_fraction, detail=None):
                """Handle progress updates from detector."""
                if self._cancelled:
                    raise InterruptedError("Analysis cancelled")
                progress_pct = int(progress_fraction * 100)
                self.progress.emit(step_name, progress_pct, detail or "")

            # Create detector (config may be None; LabelQCDetector falls back
            # to QCConfig() defaults in that case).
            self.progress.emit("Initializing...", 0, "")
            detector = LabelQCDetector(self._config)

            # Fit model with progress callback
            detector.fit(self._labels, progress_callback=progress_callback)

            if self._cancelled:
                return

            # Score instances with progress callback
            results = detector.score(self._labels, progress_callback=progress_callback)

            if self._cancelled:
                return

            # Complete
            self.progress.emit("Complete", 100, "")
            self.finished.emit(results)

        except InterruptedError:
            # Analysis was cancelled, just return silently
            pass
        except Exception as e:
            self.error.emit(str(e))


class QCChainTraceDialog(QtWidgets.QDialog):
    """Pop-up dialog that hosts the ordered-chain skeleton-tracing UI.

    The full chain editor (a :class:`QCSkeletonTraceCanvas`, the "Tracing:" chip
    readout with Undo/Clear/Add-chain, the saved-chains list, and the advanced
    "type chains as text" box) used to be embedded directly in the Detector
    Settings panel, which felt cramped. It now lives in this dialog instead
    (issue #2769 follow-up); the Detector Settings row just has a
    "Configure chains..." button that opens it.

    The dialog does *not* own any chain state: it simply re-parents and shows the
    owning :class:`QCWidget`'s live ``_chain_trace_panel``. Tracing therefore
    edits the widget's ``_traced_chains`` / free-text box directly, so the chains
    are already written back through :meth:`QCWidget._collect_ordered_chains`
    when the dialog closes -- no explicit accept/apply step is needed. Closing
    via the Close button (or the window chrome) simply hides the dialog; the
    panel is kept alive and reused the next time it is opened.

    Args:
        parent: The owning :class:`QCWidget`.
        panel: The chain-trace panel widget to host (built by
            :meth:`QCWidget._build_chain_trace_panel`).
    """

    def __init__(self, parent: QtWidgets.QWidget, panel: QtWidgets.QWidget):
        super().__init__(parent)
        self.setWindowTitle("Configure ordered chains")
        # Non-modal would let the underlying project change under the editor;
        # a modal dialog keeps the chain state edit self-contained.
        self.setModal(True)
        # Larger so the real labeled frame is clearly visible for tracing
        # (issue #2769 follow-up).
        self.setMinimumSize(640, 680)
        self.resize(820, 820)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Host the live panel; it stays visible while owned by the dialog.
        self._panel = panel
        panel.setParent(self)
        panel.setVisible(True)
        layout.addWidget(panel, stretch=1)

        # A single Close button: chains are written back live, so this just
        # dismisses the editor.
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class QCWidget(QtWidgets.QWidget):
    """Widget for label quality control analysis with visualizations.

    Provides controls for running QC analysis, viewing score distributions,
    and navigating to flagged instances.

    Signals:
        navigate_to_instance: Emitted when user wants to navigate to an instance.
            Arguments are (video_idx, frame_idx, instance_idx).
    """

    navigate_to_instance = QtCore.Signal(int, int, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        """Initialize the widget.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)

        self._labels: Optional["sio.Labels"] = None
        self._detector = None
        self._results: Optional["QCResults"] = None
        self._selected_flag: Optional["QCFlag"] = None
        self._worker: Optional[QCAnalysisWorker] = None
        self._last_export_dir: Optional[str] = None  # Persist export directory

        # All flagged instances at the current threshold, before the issue-type
        # filter is applied (item 5). The table shows a filtered subset of this.
        self._all_flagged: List["QCFlag"] = []
        # Raw ``top_issue`` strings the user has chosen to SHOW in the table.
        # None means "no filter yet" -> show everything (item 5).
        self._visible_issue_types: Optional[set] = None
        # Checkable filter actions by raw issue, and every issue type seen so
        # far (so a re-filter can tell genuinely new types from de-selected
        # ones and default the new ones to shown).
        self._issue_filter_actions: dict = {}
        self._issue_filter_seen: set = set()
        # Session-scoped reviewed-state, keyed by the instance's identity tuple
        # (video, frame, instance) so it survives threshold/issue re-filters
        # (item 6). Shared by reference with the table model.
        self._reviewed_keys: set = set()
        # Guards against stacking multiple deferred "Hide reviewed" re-filters
        # when several rows are marked reviewed in quick succession (Group C).
        self._hide_reviewed_refilter_pending: bool = False

        # Ordered chains the user has traced on the skeleton (item 2): each
        # entry is a list of node names in order. These feed QCConfig together
        # with any chains typed into the advanced free-text fallback.
        self._traced_chains: List[List[str]] = []
        # The chain currently being traced by clicking nodes (node names, in
        # click order); committed to ``_traced_chains`` via "Add chain".
        self._trace_in_progress: List[str] = []

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the widget UI."""
        # The whole panel lives in a scroll area so a tall panel (charts +
        # table + details) scrolls within the dock instead of forcing the dock
        # and window to grow without bound (issue #2769, item 4 follow-up).
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        outer.addWidget(self._scroll)

        container = QtWidgets.QWidget()
        self._scroll.setWidget(container)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # === Top row: title and run button ===
        title_layout = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("<b>Label Quality Control</b>")
        title_layout.addWidget(title)
        title_layout.addStretch()

        self._run_button = QtWidgets.QPushButton("Run Analysis")
        self._run_button.setToolTip(
            "Analyze all labeled instances for potential annotation errors"
        )
        self._run_button.setFixedWidth(100)
        title_layout.addWidget(self._run_button)
        layout.addLayout(title_layout)

        # Progress area (hidden by default). The status TEXT gets its own line
        # ABOVE the progress bar so neither is squeezed (issue #2769, item 1).
        progress_box = QtWidgets.QVBoxLayout()
        progress_box.setSpacing(2)

        # Status text on its own line.
        self._progress_label = QtWidgets.QLabel("")
        self._progress_label.setVisible(False)
        self._progress_label.setWordWrap(True)
        progress_box.addWidget(self._progress_label)

        # Progress bar + Cancel button share the second line.
        progress_layout = QtWidgets.QHBoxLayout()
        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setVisible(False)
        self._progress_bar.setTextVisible(True)
        progress_layout.addWidget(self._progress_bar, stretch=1)

        # Cancel button
        self._cancel_button = QtWidgets.QPushButton("Cancel")
        self._cancel_button.setVisible(False)
        self._cancel_button.setFixedWidth(60)
        self._cancel_button.setToolTip("Cancel the running analysis")
        progress_layout.addWidget(self._cancel_button)

        progress_box.addLayout(progress_layout)
        layout.addLayout(progress_box)

        # Timer for spinner animation during analysis
        self._spinner_timer = QtCore.QTimer(self)
        self._spinner_timer.setInterval(100)  # 100ms
        self._spinner_timer.timeout.connect(self._update_spinner)
        self._spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx = 0

        # === Threshold control ===
        threshold_layout = QtWidgets.QHBoxLayout()
        threshold_layout.addWidget(QtWidgets.QLabel("Sensitivity:"))
        threshold_layout.addWidget(QtWidgets.QLabel("More"))

        self._threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._threshold_slider.setMinimum(30)
        self._threshold_slider.setMaximum(90)
        self._threshold_slider.setValue(70)
        self._threshold_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self._threshold_slider.setTickInterval(10)
        self._threshold_slider.setToolTip(
            "Lower threshold = more instances flagged (higher sensitivity)\n"
            "Click on the histogram to set threshold visually"
        )
        threshold_layout.addWidget(self._threshold_slider, stretch=1)

        threshold_layout.addWidget(QtWidgets.QLabel("Fewer"))

        self._threshold_label = QtWidgets.QLabel("0.70")
        self._threshold_label.setMinimumWidth(40)
        self._threshold_label.setAlignment(QtCore.Qt.AlignCenter)
        self._threshold_label.setStyleSheet(
            # Explicit dark text: without it the label inherits the palette text
            # color, which is white on a dark Linux theme -> white-on-light and
            # unreadable (issue #2769 follow-up).
            "color: #212529; font-weight: bold; background: #f8f9fa; "
            "padding: 2px 6px; border-radius: 3px;"
        )
        threshold_layout.addWidget(self._threshold_label)

        layout.addLayout(threshold_layout)

        # === Detector settings ===
        self._setup_detector_settings(layout)

        # === Tabbed visualization area (collapsible) ===
        # Wrap the charts in a collapsible section so users can hide the heavy
        # plots for a cleaner panel (issue #2769, item 4). Expanded by default
        # since the charts are the main view.
        self._charts_group = CollapsibleGroupBox("Charts", collapsed=False)
        self._charts_group.setToolTip(
            "Score, issue-breakdown and feature charts.\n"
            "Uncheck the header to collapse this section."
        )
        charts_layout = QtWidgets.QVBoxLayout(self._charts_group.content)
        charts_layout.setContentsMargins(6, 4, 6, 6)

        self._viz_tabs = QtWidgets.QTabWidget()
        self._viz_tabs.setMinimumHeight(180)
        # Lock the charts height so the matplotlib canvases can't keep expanding
        # the panel without bound (issue #2769, item 4 follow-up); the
        # panel-level scroll area handles any overflow below.
        self._viz_tabs.setMaximumHeight(300)
        self._viz_tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )

        # Score distribution tab
        self._score_canvas = QCScoreCanvas(width=6, height=2.2)
        self._viz_tabs.addTab(self._score_canvas, "Score Distribution")

        # Issue breakdown tab
        self._breakdown_canvas = QCBreakdownCanvas(width=6, height=2.2)
        self._viz_tabs.addTab(self._breakdown_canvas, "Issue Breakdown")

        # Features tab
        self._feature_canvas = QCFeatureCanvas(width=6, height=2.2)
        self._viz_tabs.addTab(self._feature_canvas, "Features")

        charts_layout.addWidget(self._viz_tabs)
        layout.addWidget(self._charts_group)

        # === Flagged instances table ===
        table_group = QtWidgets.QGroupBox("Flagged Instances")
        table_layout = QtWidgets.QVBoxLayout(table_group)
        table_layout.setContentsMargins(4, 4, 4, 4)

        # --- Toolbar row above the table: issue-type filter + reviewed count ---
        # Item 5: a checkable dropdown to include/exclude flagged instances by
        # issue type. Item 6: a running "X / Y reviewed" counter.
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(6)

        toolbar.addWidget(QtWidgets.QLabel("Show:"))

        # Multi-select issue-type filter as a QMenu of checkboxes on a button.
        self._issue_filter_button = QtWidgets.QToolButton()
        self._issue_filter_button.setText("Issue types: all")
        self._issue_filter_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self._issue_filter_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self._issue_filter_button.setToolTip(
            "Filter the flagged list by issue type.\n"
            "Tick the issue types you want to see; the table updates live."
        )
        # Stays open while the user ticks multiple issue types (item 5).
        self._issue_filter_menu = CheckableFilterMenu(self._issue_filter_button)
        self._issue_filter_button.setMenu(self._issue_filter_menu)
        # Disabled until results exist (nothing to filter yet).
        self._issue_filter_button.setEnabled(False)
        toolbar.addWidget(self._issue_filter_button)

        # "Hide reviewed" filter (Group C / feedback on #2769): show only the
        # not-yet-reviewed flagged instances, AND-combined with the issue-type
        # filter above. Sits next to the issue-type button so both live filters
        # read together. Disabled until results exist (nothing to hide yet).
        self._hide_reviewed_check = QtWidgets.QCheckBox("Hide reviewed")
        self._hide_reviewed_check.setChecked(False)
        self._hide_reviewed_check.setEnabled(False)
        self._hide_reviewed_check.setToolTip(
            "Show only instances you have NOT marked reviewed.\n"
            "Combined (AND) with the issue-type filter; the table updates live "
            "as you tick rows reviewed."
        )
        toolbar.addWidget(self._hide_reviewed_check)

        toolbar.addStretch()

        # Running reviewed counter ("12 / 45 reviewed"), item 6.
        self._reviewed_count_label = QtWidgets.QLabel("0 / 0 reviewed")
        self._reviewed_count_label.setToolTip(
            "How many of the flagged instances shown you have marked reviewed.\n"
            "Tick the Reviewed box (or just navigate to a row) to mark one."
        )
        toolbar.addWidget(self._reviewed_count_label)

        table_layout.addLayout(toolbar)

        # Share the reviewed-keys set by reference so the model's Reviewed
        # column reflects session-scoped, identity-keyed state (item 6).
        self._table_model = QCFlagTableModel(reviewed_keys=self._reviewed_keys)
        self._table_view = QtWidgets.QTableView()
        self._table_view.setModel(self._table_model)
        self._table_view.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
        self._table_view.setSelectionMode(QtWidgets.QTableView.SingleSelection)
        self._table_view.setAlternatingRowColors(True)
        self._table_view.setSortingEnabled(True)
        self._table_view.setMinimumHeight(120)

        # Column widths: stretch the Issue column, keep the trailing Reviewed
        # checkbox column compact (so it does not grab the leftover width).
        header = self._table_view.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)  # Issue
        header.setSectionResizeMode(
            QCFlagTableModel.REVIEWED_COL, QtWidgets.QHeaderView.ResizeToContents
        )

        table_layout.addWidget(self._table_view)
        layout.addWidget(table_group, stretch=1)

        # === Bottom panel: selected instance info and statistics ===
        # Both are collapsible (issue #2769, item 4) but expanded by default,
        # since their plain-language summaries (item 7) are useful to everyone.
        bottom_layout = QtWidgets.QHBoxLayout()

        # Selected instance details
        self._details_group = CollapsibleGroupBox("Selected Instance", collapsed=False)
        self._details_group.setToolTip(
            "Why the selected instance was flagged.\n"
            "Uncheck the header to collapse this section."
        )
        details_layout = QtWidgets.QVBoxLayout(self._details_group.content)
        details_layout.setContentsMargins(6, 6, 6, 6)

        self._details_label = QtWidgets.QLabel(SELECT_INSTANCE_PLACEHOLDER)
        self._details_label.setWordWrap(True)
        self._details_label.setMinimumHeight(70)
        details_layout.addWidget(self._details_label)

        bottom_layout.addWidget(self._details_group)

        # Statistics panel
        self._stats_group = CollapsibleGroupBox("Statistics", collapsed=False)
        self._stats_group.setToolTip(
            "Summary of how many instances were flagged.\n"
            "Uncheck the header to collapse this section."
        )
        stats_layout = QtWidgets.QVBoxLayout(self._stats_group.content)
        stats_layout.setContentsMargins(6, 6, 6, 6)

        self._stats_label = QtWidgets.QLabel("No analysis run yet")
        self._stats_label.setWordWrap(True)
        self._stats_label.setMinimumHeight(70)
        stats_layout.addWidget(self._stats_label)

        bottom_layout.addWidget(self._stats_group)

        layout.addLayout(bottom_layout)

    def _setup_detector_settings(self, layout: QtWidgets.QVBoxLayout):
        """Build the collapsible "Detector Settings" group.

        Exposes the per-detector QCConfig toggles and thresholds so users can
        enable/disable and tune each detector for the current project. The
        resulting controls are read by :meth:`_build_qc_config`.

        The group is an advanced panel, so it starts COLLAPSED to keep the
        first-time view clean (issue #2769, item 4); each detector row carries a
        plain-language "?" help button (item 3).

        Args:
            layout: The parent layout to append the group box to.
        """
        # Collapsible header; advanced panel starts collapsed for a clean view.
        group = CollapsibleGroupBox("Detector Settings", collapsed=True)
        group.setToolTip(
            "Enable/disable and tune individual QC detectors for this project.\n"
            "Click the header to expand; uncheck it to collapse again."
        )
        self._detector_settings_group = group

        grid = QtWidgets.QGridLayout(group.content)
        grid.setContentsMargins(8, 4, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        # Column 2 (the threshold area) takes any extra width. Column 3 holds
        # the per-detector "?" help buttons at the far right.
        grid.setColumnStretch(2, 1)
        self._detector_settings_grid = grid

        # --- Whole-instance L/R flip (chirality), reliable, default-ON ---
        self._cb_flip = QtWidgets.QCheckBox("Whole-instance L/R flip")
        self._cb_flip.setChecked(True)
        self._cb_flip.setToolTip(
            "Flag instances whose left/right keypoints look mirror-flipped "
            "(chirality). Reliable detector; on by default."
        )
        self._sb_flip_thr = QtWidgets.QDoubleSpinBox()
        self._sb_flip_thr.setRange(0.0, 1.0)
        self._sb_flip_thr.setSingleStep(0.05)
        self._sb_flip_thr.setValue(0.5)
        self._sb_flip_thr.setToolTip(
            "Fraction of symmetric pairs that must look flipped before the "
            "whole instance is force-flagged (chirality_flip_threshold)."
        )
        flip_thr_row = self._make_threshold_row(
            [(QtWidgets.QLabel("flip frac ≥"), self._sb_flip_thr)]
        )
        self._add_detector_row(0, self._cb_flip, flip_thr_row, help_key="flip")

        # --- Chimera / pose split, reliable, default-ON (no user threshold) ---
        self._cb_chimera = QtWidgets.QCheckBox("Chimera (pose split)")
        self._cb_chimera.setChecked(True)
        self._cb_chimera.setToolTip(
            "Flag a single instance whose pose spans two animals (a chimera).\n"
            "Relies on the learned GMM, so it has no hard threshold to tune."
        )
        # No tunable threshold: chimera has no hard rule, the GMM decides.
        chimera_note = QtWidgets.QLabel("(no threshold)")
        chimera_note.setEnabled(False)
        chimera_note.setToolTip("Chimera detection has no hard threshold to set.")
        self._add_detector_row(1, self._cb_chimera, chimera_note, help_key="chimera")

        # --- Duplicate / split, reliable, default-ON ---
        self._cb_duplicate = QtWidgets.QCheckBox("Duplicate / split")
        self._cb_duplicate.setChecked(True)
        self._cb_duplicate.setToolTip(
            "Fold the split-duplicate signal into frame-level duplicate "
            "detection. Reliable detector; on by default."
        )
        self._sb_dup_thr = QtWidgets.QDoubleSpinBox()
        self._sb_dup_thr.setRange(0.0, 1.0)
        self._sb_dup_thr.setSingleStep(0.05)
        self._sb_dup_thr.setValue(0.5)
        self._sb_dup_thr.setToolTip(
            "Combined duplicate score at/above which a pair of instances is "
            "flagged as a duplicate (duplicate_score_threshold)."
        )
        dup_thr_row = self._make_threshold_row(
            [(QtWidgets.QLabel("score ≥"), self._sb_dup_thr)]
        )
        self._add_detector_row(2, self._cb_duplicate, dup_thr_row, help_key="duplicate")

        # --- Wrong chain order, experimental, default-OFF ---
        self._cb_chain = QtWidgets.QCheckBox("Wrong chain order")
        self._cb_chain.setChecked(False)
        self._cb_chain.setToolTip(
            "Flag instances whose keypoints run out of order along an ordered "
            "chain (e.g. a tail). Experimental; off by default."
        )
        self._sb_chain_angle = QtWidgets.QSpinBox()
        self._sb_chain_angle.setRange(10, 150)
        self._sb_chain_angle.setSuffix("°")
        self._sb_chain_angle.setValue(60)
        self._sb_chain_angle.setToolTip(
            "Per-node turning angle (degrees) above which a chain node counts "
            "as an ordering inversion (chain_turn_angle_deg)."
        )
        self._sb_order_thr = QtWidgets.QDoubleSpinBox()
        self._sb_order_thr.setRange(0.0, 1.0)
        self._sb_order_thr.setSingleStep(0.05)
        self._sb_order_thr.setValue(0.3)
        self._sb_order_thr.setToolTip(
            "Fraction of chain nodes that must be inverted before the instance "
            "is force-flagged (order_inversion_threshold)."
        )
        chain_thr_row = self._make_threshold_row(
            [
                (QtWidgets.QLabel("turn ≥"), self._sb_chain_angle),
                (QtWidgets.QLabel("inv frac ≥"), self._sb_order_thr),
            ]
        )
        self._add_detector_row(3, self._cb_chain, chain_thr_row, help_key="chain")

        # Skeleton-tracing UX for defining ordered chains (issue #2769, item 2):
        # a click-to-trace skeleton plus an ordered chip list, instead of forcing
        # users to type node names. The full tracing UI lives in a *pop-up*
        # dialog (issue #2769 follow-up: the inline panel felt too cramped), so
        # the grid row only carries a "Configure chains..." button and a short
        # summary of the current chains. The panel is built eagerly (so its
        # controls exist for callers/tests) but is hosted in the dialog on
        # demand rather than embedded here.
        self._chain_trace_panel = self._build_chain_trace_panel()
        # Hold a Qt parent so the (currently hidden) panel stays alive between
        # dialog openings; it is reparented into the dialog when opened.
        self._chain_trace_panel.setParent(self)
        self._chain_trace_panel.setVisible(False)
        self._chain_trace_dialog: Optional["QCChainTraceDialog"] = None

        chain_config_row = QtWidgets.QHBoxLayout()
        chain_config_row.setContentsMargins(0, 0, 0, 0)
        chain_config_row.setSpacing(6)
        self._chain_config_btn = QtWidgets.QPushButton("Configure chains...")
        self._chain_config_btn.setToolTip(
            "Open the chain editor to trace ordered chains on the skeleton "
            "(e.g. tail base → tail tip) used by the chain-order detector."
        )
        self._chain_config_btn.clicked.connect(self._open_chain_trace_dialog)
        chain_config_row.addWidget(self._chain_config_btn)

        self._chain_summary_label = QtWidgets.QLabel("No chains")
        self._chain_summary_label.setWordWrap(True)
        self._chain_summary_label.setStyleSheet("color:#6c757d;")
        self._chain_summary_label.setToolTip(
            "The ordered chains currently configured for the chain-order "
            "detector. Click 'Configure chains...' to edit them."
        )
        chain_config_row.addWidget(self._chain_summary_label, stretch=1)

        chain_config_container = QtWidgets.QWidget()
        chain_config_container.setLayout(chain_config_row)
        self._chain_config_container = chain_config_container
        grid.addWidget(chain_config_container, 4, 0, 1, 4)
        self._refresh_chain_summary()

        # --- Missing labelable node, experimental, default-OFF ---
        self._cb_missing = QtWidgets.QCheckBox("Missing labelable node")
        self._cb_missing.setChecked(False)
        self._cb_missing.setToolTip(
            "Flag an instance that is missing a node its peers usually keep. "
            "Experimental; off by default."
        )
        self._sb_missing_thr = QtWidgets.QDoubleSpinBox()
        self._sb_missing_thr.setRange(0.0, 1.0)
        self._sb_missing_thr.setSingleStep(0.05)
        self._sb_missing_thr.setValue(0.9)
        self._sb_missing_thr.setToolTip(
            "Minimum expected-visibility probability for an absent node to be "
            "flagged as suspicious (missing_node_prob_threshold)."
        )
        missing_thr_row = self._make_threshold_row(
            [(QtWidgets.QLabel("prob ≥"), self._sb_missing_thr)]
        )
        self._add_detector_row(5, self._cb_missing, missing_thr_row, help_key="missing")

        # --- Appearance / wrong-object (B2 channel), experimental, default-OFF ---
        self._cb_appearance = QtWidgets.QCheckBox("Appearance / wrong-object")
        self._cb_appearance.setChecked(False)
        self._cb_appearance.setToolTip(
            "Flag a keypoint placed on visually-wrong pixels (e.g. on bedding "
            "instead of fur), using a per-node image-appearance model. "
            "Experimental; off by default."
        )
        # No tunable hard threshold in the GUI: appearance is a channel score.
        appearance_note = QtWidgets.QLabel("(image-based)")
        appearance_note.setEnabled(False)
        appearance_note.setToolTip(
            "Appearance scoring reads image patches around each node; it has no "
            "hard threshold to tune here."
        )
        self._add_detector_row(
            6, self._cb_appearance, appearance_note, help_key="appearance"
        )

        # --- In-sample model prediction (B2 channel), experimental, default-OFF ---
        self._cb_insample = QtWidgets.QCheckBox("In-sample model prediction")
        self._cb_insample.setChecked(False)
        self._cb_insample.setToolTip(
            "Run a trained sleap-nn model on the labeled frames and flag "
            "unlabeled nodes the model confidently localizes (labelable-but-"
            "skipped parts).\n"
            "WARNING: runs full model inference and can be slow on large "
            "projects. Experimental; off by default."
        )
        # Model-path picker: a (display-only) line edit plus a Browse button.
        self._insample_model_edit = QtWidgets.QLineEdit()
        self._insample_model_edit.setReadOnly(True)
        self._insample_model_edit.setPlaceholderText("trained sleap-nn model folder")
        self._insample_model_edit.setToolTip(
            "Folder of a trained sleap-nn model (with best.ckpt + "
            "training_config.yaml) to run in-sample. Empty disables the channel."
        )
        self._insample_browse_btn = QtWidgets.QPushButton("Browse...")
        self._insample_browse_btn.setToolTip(
            "Choose the trained sleap-nn model folder for in-sample prediction."
        )
        insample_picker = self._make_threshold_row(
            [(self._insample_model_edit, self._insample_browse_btn)]
        )
        self._add_detector_row(
            7, self._cb_insample, insample_picker, help_key="insample"
        )

        # Disable each detector's tunable widgets when its checkbox is off.
        self._cb_flip.toggled.connect(self._sb_flip_thr.setEnabled)
        self._sb_flip_thr.setEnabled(self._cb_flip.isChecked())
        self._cb_duplicate.toggled.connect(self._sb_dup_thr.setEnabled)
        self._sb_dup_thr.setEnabled(self._cb_duplicate.isChecked())
        self._cb_missing.toggled.connect(self._sb_missing_thr.setEnabled)
        self._sb_missing_thr.setEnabled(self._cb_missing.isChecked())

        def _set_chain_enabled(on: bool):
            self._sb_chain_angle.setEnabled(on)
            self._sb_order_thr.setEnabled(on)
            # The "Configure chains..." button + summary (the inline entry point)
            # and the trace panel hosted in the pop-up dialog all follow the
            # chain checkbox.
            self._chain_config_btn.setEnabled(on)
            self._chain_summary_label.setEnabled(on)
            self._chain_trace_panel.setEnabled(on)

        self._cb_chain.toggled.connect(_set_chain_enabled)
        _set_chain_enabled(self._cb_chain.isChecked())
        # Warn (only on a real user click, not programmatic restore/init) if the
        # detector is turned on with no chains defined -- it would do nothing.
        self._cb_chain.clicked.connect(self._on_chain_checked)

        # Disable the in-sample model picker + Browse button when its checkbox
        # is off (mirrors the other detectors' disable-on-uncheck behavior).
        def _set_insample_enabled(on: bool):
            self._insample_model_edit.setEnabled(on)
            self._insample_browse_btn.setEnabled(on)

        self._cb_insample.toggled.connect(_set_insample_enabled)
        _set_insample_enabled(self._cb_insample.isChecked())
        self._insample_browse_btn.clicked.connect(self._on_browse_insample_model)

        # --- Restore-defaults footer row, spanning the full grid width. ---
        # A thin separator above a right-aligned "Restore defaults" button that
        # resets every control back to QCConfig() defaults (issue #2769
        # follow-up, "restore to default setting").
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        grid.addWidget(separator, 8, 0, 1, 4)

        restore_row = QtWidgets.QHBoxLayout()
        restore_row.setContentsMargins(0, 0, 0, 0)
        restore_row.addStretch()
        self._restore_defaults_btn = QtWidgets.QPushButton("Restore defaults")
        self._restore_defaults_btn.setToolTip(
            "Reset every detector setting on this panel (enable toggles, "
            "thresholds, ordered chains and the in-sample model path) back to "
            "the shipped defaults."
        )
        self._restore_defaults_btn.clicked.connect(self._on_restore_detector_defaults)
        restore_row.addWidget(self._restore_defaults_btn)
        restore_container = QtWidgets.QWidget()
        restore_container.setLayout(restore_row)
        grid.addWidget(restore_container, 9, 0, 1, 4)

        # Apply the (collapsed) start state now that all children exist so the
        # panel actually starts hidden for a clean first-time view.
        group.apply_collapsed_state()

        layout.addWidget(group)

    def _build_chain_trace_panel(self) -> QtWidgets.QWidget:
        """Build the skeleton-tracing UI for defining ordered chains.

        Lets the user define an ordered chain by *clicking nodes in order* on a
        rendered skeleton (issue #2769, item 2) instead of typing node names.
        The panel contains:

        * a :class:`QCSkeletonTraceCanvas` (click nodes to build a chain),
        * a live ordered-chip readout of the chain being traced plus
          Add / Undo / Clear controls,
        * a list of saved chains with up/down reorder and remove, and
        * an advanced, collapsible free-text fallback (the original typed-chain
          box) for power users / projects with no skeleton.

        The panel is hosted inside the :class:`QCChainTraceDialog` pop-up rather
        than embedded in the Detector Settings grid (issue #2769 follow-up), but
        it still reads/writes the same widget state (``_traced_chains`` and the
        free-text box), so both the traced chains and the free-text box feed
        :meth:`_build_qc_config` via :meth:`_collect_ordered_chains`.

        Returns:
            The assembled panel widget.
        """
        panel = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(panel)
        outer.setContentsMargins(0, 4, 0, 0)
        outer.setSpacing(4)

        heading = QtWidgets.QLabel(
            "Trace a chain by clicking nodes on the skeleton in order "
            "(e.g. tail base → tail tip), then click <b>Add chain</b>. "
            "Repeat to add more chains."
        )
        heading.setWordWrap(True)
        heading.setToolTip(
            "Define the ground-truth order of a connected chain of nodes (like a "
            "tail or a limb) so the detector can spot points clicked out of "
            "sequence. Click nodes on the skeleton in the order they should run."
        )
        outer.addWidget(heading)

        # --- Interactive skeleton canvas: click nodes in order to trace. ---
        # Bigger + zoomable so the real labeled animal is clearly visible
        # (issue #2769 follow-up: trace on the actual frame photo).
        self._skeleton_canvas = QCSkeletonTraceCanvas(width=7, height=5)
        self._skeleton_canvas.setToolTip(
            "Click nodes in order to trace a chain. The numbered blue nodes show "
            "the current order; click 'Add chain' to save it.\n"
            "Scroll to zoom, drag with the middle/right mouse button to pan, "
            "and double-click to reset the view."
        )
        outer.addWidget(self._skeleton_canvas, stretch=1)

        # A small view-control row: a hint plus an explicit Reset view button
        # (double-click also resets).
        view_row = QtWidgets.QHBoxLayout()
        view_row.setSpacing(4)
        view_hint = QtWidgets.QLabel(
            "Scroll to zoom · middle/right-drag to pan · double-click to reset"
        )
        view_hint.setStyleSheet("color:#6c757d;")
        view_row.addWidget(view_hint)
        view_row.addStretch()
        self._reset_view_btn = QtWidgets.QToolButton()
        self._reset_view_btn.setText("Reset view")
        self._reset_view_btn.setToolTip("Reset zoom/pan to show the whole frame.")
        self._reset_view_btn.clicked.connect(self._skeleton_canvas.reset_view)
        view_row.addWidget(self._reset_view_btn)
        outer.addLayout(view_row)

        # --- Current trace readout (ordered chips) + actions. ---
        trace_row = QtWidgets.QHBoxLayout()
        trace_row.setSpacing(4)
        trace_row.addWidget(QtWidgets.QLabel("Tracing:"))

        self._trace_chips_label = QtWidgets.QLabel("(click nodes to start)")
        self._trace_chips_label.setWordWrap(True)
        self._trace_chips_label.setTextFormat(QtCore.Qt.RichText)
        self._trace_chips_label.setToolTip(
            "The chain you are currently tracing, in click order."
        )
        trace_row.addWidget(self._trace_chips_label, stretch=1)

        self._trace_undo_btn = QtWidgets.QToolButton()
        self._trace_undo_btn.setText("Undo")
        self._trace_undo_btn.setToolTip("Remove the last node from the trace.")
        self._trace_undo_btn.clicked.connect(self._on_trace_undo)
        trace_row.addWidget(self._trace_undo_btn)

        self._trace_clear_btn = QtWidgets.QToolButton()
        self._trace_clear_btn.setText("Clear")
        self._trace_clear_btn.setToolTip("Clear the current trace and start over.")
        self._trace_clear_btn.clicked.connect(self._on_trace_clear)
        trace_row.addWidget(self._trace_clear_btn)

        self._trace_add_btn = QtWidgets.QToolButton()
        self._trace_add_btn.setText("Add chain")
        self._trace_add_btn.setToolTip(
            "Save the traced chain to the list of ordered chains below."
        )
        self._trace_add_btn.clicked.connect(self._on_trace_add_chain)
        trace_row.addWidget(self._trace_add_btn)

        outer.addLayout(trace_row)

        # --- Saved chains list with up/down reorder + remove. ---
        saved_row = QtWidgets.QHBoxLayout()
        saved_row.setSpacing(4)

        self._chains_list = QtWidgets.QListWidget()
        self._chains_list.setToolTip(
            "Ordered chains used by the chain-order detector. Select one to "
            "reorder or remove it."
        )
        self._chains_list.setMaximumHeight(78)
        self._chains_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        saved_row.addWidget(self._chains_list, stretch=1)

        # Vertical button strip for the saved-chains list.
        chain_btns = QtWidgets.QVBoxLayout()
        chain_btns.setSpacing(2)
        self._chain_up_btn = QtWidgets.QToolButton()
        self._chain_up_btn.setArrowType(QtCore.Qt.UpArrow)
        self._chain_up_btn.setToolTip("Move the selected chain up.")
        self._chain_up_btn.clicked.connect(lambda: self._move_selected_chain(-1))
        chain_btns.addWidget(self._chain_up_btn)

        self._chain_down_btn = QtWidgets.QToolButton()
        self._chain_down_btn.setArrowType(QtCore.Qt.DownArrow)
        self._chain_down_btn.setToolTip("Move the selected chain down.")
        self._chain_down_btn.clicked.connect(lambda: self._move_selected_chain(1))
        chain_btns.addWidget(self._chain_down_btn)

        self._chain_remove_btn = QtWidgets.QToolButton()
        self._chain_remove_btn.setText("✕")
        self._chain_remove_btn.setToolTip("Remove the selected chain.")
        self._chain_remove_btn.clicked.connect(self._on_remove_selected_chain)
        chain_btns.addWidget(self._chain_remove_btn)
        chain_btns.addStretch()
        saved_row.addLayout(chain_btns)

        outer.addLayout(saved_row)

        # --- Advanced free-text fallback (the original typed-chain box). ---
        # Kept for power users and projects without a skeleton; collapsed by
        # default so the trace UI is the primary path.
        advanced = CollapsibleGroupBox("Advanced: type chains as text", collapsed=True)
        advanced.setToolTip(
            "Fallback for typing ordered chains by hand (one chain per line, "
            "node names comma-separated). Useful when no skeleton is loaded or "
            "for pasting chains. Combined with any chains traced above."
        )
        adv_layout = QtWidgets.QVBoxLayout(advanced.content)
        adv_layout.setContentsMargins(6, 4, 6, 6)

        self._ordered_chains_edit = QtWidgets.QPlainTextEdit()
        self._ordered_chains_edit.setPlaceholderText(
            "One chain per line, node names comma-separated, "
            "e.g. TTI, Tail_0, Tail_1, Tail_2, TailTip"
        )
        self._ordered_chains_edit.setToolTip(
            "Optional ground-truth ordered node chains for the chain-order "
            "detector (one chain per line, node names comma-separated).\n"
            "Combined with any chains traced on the skeleton above. Leave both "
            "empty to fall back to auto-detected skeleton chains."
        )
        # Keep it compact (about three lines tall).
        fm = self._ordered_chains_edit.fontMetrics()
        self._ordered_chains_edit.setMaximumHeight(fm.height() * 3 + 8)
        adv_layout.addWidget(self._ordered_chains_edit)
        advanced.apply_collapsed_state()
        outer.addWidget(advanced)

        # Wire the canvas click -> append to the trace.
        self._skeleton_canvas.node_clicked.connect(self._on_trace_node_clicked)

        # Initialize the readout/buttons for an empty trace.
        self._refresh_trace_readout()
        self._refresh_chains_list()

        return panel

    def _on_chain_checked(self, checked: bool):
        """Prompt to define a chain when chain-order is enabled with none set.

        The "Wrong chain order" detector does nothing without at least one
        ordered chain, so when the user ticks it on with none configured, offer
        to open the chain editor right away (issue #2769 follow-up). Connected to
        ``clicked`` (not ``toggled``) so it never fires on programmatic changes
        like Restore defaults or initial setup.

        Args:
            checked: The checkbox's new state from the ``clicked`` signal.
        """
        if not checked or self._collect_ordered_chains():
            return
        resp = QtWidgets.QMessageBox.question(
            self,
            "No chains defined",
            "The 'Wrong chain order' detector needs at least one ordered chain "
            "to check, but none are defined yet.\n\n"
            "Open the chain editor to trace one now?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if resp == QtWidgets.QMessageBox.Yes:
            self._open_chain_trace_dialog()

    def _open_chain_trace_dialog(self):
        """Open the pop-up chain editor hosting the full tracing UI.

        Lazily creates a :class:`QCChainTraceDialog`, hands it the live
        :attr:`_chain_trace_panel` (so editing operates directly on the widget's
        chain state), shows it modally, and refreshes the inline summary on
        close (issue #2769 follow-up: the embedded panel was too cramped).
        """
        if self._chain_trace_dialog is None:
            self._chain_trace_dialog = QCChainTraceDialog(self, self._chain_trace_panel)
        dialog = self._chain_trace_dialog
        # Make sure the trace readout/canvas reflect the current state each time.
        self._refresh_trace_readout()
        self._refresh_chains_list()
        dialog.exec_()
        # Editing happened in-place on the shared panel; refresh the summary.
        self._refresh_chain_summary()

    def _chain_summary_text(self) -> str:
        """Plain-language summary of the configured ordered chains.

        Combines the traced chains with any advanced free-text chains (the same
        set :meth:`_collect_ordered_chains` feeds into the config) into a short
        caption like ``"2 chains: TTI→…→TailTip, Head→Neck"``. Returns
        ``"No chains"`` when nothing is configured.

        Returns:
            A short caption describing the current chains.
        """
        chains = self._collect_ordered_chains()
        if not chains:
            return "No chains"

        def _abbrev(chain: list) -> str:
            # Show endpoints (with an ellipsis between) so long tail chains stay
            # compact: e.g. ["TTI","Tail_0","TailTip"] -> "TTI→…→TailTip".
            if len(chain) <= 2:
                return "→".join(chain)
            return f"{chain[0]}→…→{chain[-1]}"

        # Cap how many chains we spell out so the label can't run away.
        shown = [_abbrev(chain) for chain in chains[:3]]
        summary = ", ".join(shown)
        if len(chains) > 3:
            summary += f", +{len(chains) - 3} more"
        n = len(chains)
        return f"{n} chain{'s' if n != 1 else ''}: {summary}"

    def _refresh_chain_summary(self):
        """Update the inline chain summary label from the current chains."""
        if hasattr(self, "_chain_summary_label"):
            self._chain_summary_label.setText(self._chain_summary_text())

    def _on_trace_node_clicked(self, name: str):
        """Append a clicked node to the chain currently being traced.

        Consecutive duplicate clicks on the same node are ignored so a double
        click does not insert a degenerate zero-length segment.

        Args:
            name: The node name that was clicked on the skeleton canvas.
        """
        if self._trace_in_progress and self._trace_in_progress[-1] == name:
            return
        self._trace_in_progress.append(name)
        self._refresh_trace_readout()

    def _on_trace_undo(self):
        """Remove the last node from the chain being traced."""
        if self._trace_in_progress:
            self._trace_in_progress.pop()
            self._refresh_trace_readout()

    def _on_trace_clear(self):
        """Clear the chain currently being traced."""
        if self._trace_in_progress:
            self._trace_in_progress = []
            self._refresh_trace_readout()

    def _on_trace_add_chain(self):
        """Commit the traced chain (>= 2 nodes) to the saved-chains list."""
        chain = list(self._trace_in_progress)
        if len(chain) < 2:
            QtWidgets.QMessageBox.information(
                self,
                "Trace a chain first",
                "Click at least two nodes on the skeleton (in order) before "
                "adding a chain.",
            )
            return
        self._traced_chains.append(chain)
        self._trace_in_progress = []
        self._refresh_trace_readout()
        self._refresh_chains_list()

    def _refresh_trace_readout(self):
        """Sync the trace chips, canvas highlight, and trace-action buttons."""
        chain = self._trace_in_progress
        # Mirror the trace onto the canvas highlight.
        if hasattr(self, "_skeleton_canvas"):
            self._skeleton_canvas.set_trace(chain)

        if chain:
            chips = "  →  ".join(
                f"<span style='background:#e7f1ff; color:#0056b3; "
                f"padding:1px 5px; border-radius:6px;'>{i + 1}. {name}</span>"
                for i, name in enumerate(chain)
            )
            self._trace_chips_label.setText(chips)
        else:
            self._trace_chips_label.setText(
                "<span style='color:#6c757d;'>(click nodes to start)</span>"
            )

        has_any = bool(chain)
        self._trace_undo_btn.setEnabled(has_any)
        self._trace_clear_btn.setEnabled(has_any)
        # Need at least two nodes to make a meaningful chain.
        self._trace_add_btn.setEnabled(len(chain) >= 2)

    def _refresh_chains_list(self):
        """Repopulate the saved-chains list from ``_traced_chains``.

        Also refreshes the inline summary label, since every saved-chains change
        (add / remove / reorder / restore) flows through here.
        """
        self._chains_list.clear()
        for chain in self._traced_chains:
            self._chains_list.addItem(" → ".join(chain))
        has_rows = self._chains_list.count() > 0
        self._chain_up_btn.setEnabled(has_rows)
        self._chain_down_btn.setEnabled(has_rows)
        self._chain_remove_btn.setEnabled(has_rows)
        self._refresh_chain_summary()

    def _move_selected_chain(self, delta: int):
        """Move the selected saved chain up or down by one.

        Args:
            delta: -1 to move up, +1 to move down.
        """
        row = self._chains_list.currentRow()
        if row < 0:
            return
        new_row = row + delta
        if not (0 <= new_row < len(self._traced_chains)):
            return
        self._traced_chains[row], self._traced_chains[new_row] = (
            self._traced_chains[new_row],
            self._traced_chains[row],
        )
        self._refresh_chains_list()
        self._chains_list.setCurrentRow(new_row)

    def _on_remove_selected_chain(self):
        """Remove the currently selected saved chain."""
        row = self._chains_list.currentRow()
        if 0 <= row < len(self._traced_chains):
            del self._traced_chains[row]
            self._refresh_chains_list()

    def _update_skeleton_trace(self):
        """Push the current project's skeleton into the trace canvas.

        Reads the first skeleton from the loaded labels (if any) and hands its
        nodes/edges to the :class:`QCSkeletonTraceCanvas` so the user can trace
        on the real skeleton. It then picks one real labeled instance with as
        many nodes present as possible and decodes that frame's image
        (:meth:`_best_labeled_instance_image`), so the canvas can show the actual
        photo with the skeleton overlaid at true pixel coordinates -- the user
        traces on the real animal (issue #2769 follow-up). When no image can be
        decoded (e.g. the video file is missing) it falls back to a centered,
        abstract layout from a representative instance
        (:meth:`_representative_node_positions`). Clears the canvas when no
        skeleton is available.

        Uses ``labels.skeletons`` (a list) rather than the ``labels.skeleton``
        convenience property, since the latter *raises* when there are zero or
        multiple skeletons. Nothing here may throw: any backend hiccup degrades
        to a less rich view (image -> abstract -> spring -> empty) so the dialog
        always opens.
        """
        node_names: list = []
        edges: list = []
        labels = self._labels
        skeleton = None
        try:
            skeletons = (
                getattr(labels, "skeletons", None) if labels is not None else None
            )
            skeleton = skeletons[0] if skeletons else None
            if skeleton is not None:
                node_names = list(skeleton.node_names)
                edges = [
                    (edge.source.name, edge.destination.name) for edge in skeleton.edges
                ]
        except Exception:
            node_names, edges, skeleton = [], [], None

        # Prefer the real labeled frame: one instance's pixel coords + the photo.
        pixel_positions, image = self._best_labeled_instance_image(skeleton, node_names)
        if image is not None and pixel_positions:
            self._skeleton_canvas.set_skeleton(
                node_names, edges, node_positions=pixel_positions, image=image
            )
            return

        # No usable image: draw an abstract animal shape from a representative
        # instance (centered/scaled), falling back to the spring layout inside
        # the canvas when even that is unavailable.
        node_positions = self._representative_node_positions(skeleton, node_names)
        self._skeleton_canvas.set_skeleton(
            node_names, edges, node_positions=node_positions, image=None
        )

    def _best_labeled_instance_image(self, skeleton, node_names: list):
        """Pick the best-labeled instance and decode its frame image.

        Scans every labeled frame for *user* instances that share ``skeleton``
        and counts how many nodes are present (finite/visible). The instance with
        the most present nodes wins (preferring one with *all* nodes present;
        ties break to the first found), since a fully-labeled animal makes the
        clearest tracing target. That instance's frame image is decoded via
        ``labeled_frame.image`` (which works for embedded ``pkg.slp`` frames).

        The whole scan is wrapped defensively: a missing video, an undecodable
        frame, or any backend error returns ``(positions, None)`` (or ``({},
        None)``) so the caller falls back to the abstract layout and the dialog
        still opens.

        Args:
            skeleton: The skeleton whose ``node_names`` order aligns with each
                instance's ``numpy()`` rows, or ``None``.
            node_names: The skeleton node names (used as result keys).

        Returns:
            A ``(positions, image)`` tuple. ``positions`` maps node name to the
            chosen instance's ``(x, y)`` *pixel* coordinate (present nodes only);
            ``image`` is the decoded frame as an ``np.ndarray`` or ``None``.
        """
        labels = self._labels
        if labels is None or skeleton is None or not node_names:
            return {}, None

        n_nodes = len(node_names)
        best_lf = None
        best_pts = None
        best_count = -1
        try:
            frames = getattr(labels, "labeled_frames", None)
            if frames is None:
                frames = labels  # iterate the Labels object directly
            for lf in frames:
                for inst in getattr(lf, "user_instances", []):
                    # Only instances sharing this skeleton align row-for-row.
                    if getattr(inst, "skeleton", None) is not skeleton:
                        continue
                    pts = np.asarray(inst.numpy(invisible_as_nan=True), dtype=float)
                    if pts.shape != (n_nodes, 2):
                        continue
                    present = int(np.isfinite(pts).all(axis=1).sum())
                    if present > best_count:
                        best_count = present
                        best_pts = pts
                        best_lf = lf
                        # A fully-present instance is ideal; stop early.
                        if present == n_nodes:
                            raise StopIteration
        except StopIteration:
            pass
        except Exception:
            return {}, None

        if best_lf is None or best_pts is None or best_count <= 0:
            return {}, None

        positions = {
            name: (float(best_pts[i, 0]), float(best_pts[i, 1]))
            for i, name in enumerate(node_names)
            if np.isfinite(best_pts[i, 0]) and np.isfinite(best_pts[i, 1])
        }

        # Decode the frame image; any failure (missing video, bad backend) falls
        # back to an image-less (abstract) overlay.
        image = None
        try:
            image = np.asarray(best_lf.image)
        except Exception:
            image = None

        return positions, image

    def _representative_node_positions(
        self, skeleton, node_names: list, max_instances: int = 200
    ) -> dict:
        """Median per-node coordinate over a sample of labeled instances.

        For each skeleton node, takes the median ``(x, y)`` over up to
        ``max_instances`` labeled (user) instances in which that node is
        visible, in the raw label coordinate frame. The medians give a stable,
        outlier-resistant "representative animal" the trace canvas can draw in
        the real animal's shape. Each instance is recentered on its own centroid
        first so animals labeled in very different parts of the frame still
        average into a coherent shape rather than smearing across the image.

        Args:
            skeleton: The skeleton whose ``node_names`` order aligns with each
                instance's ``numpy()`` rows, or ``None``.
            node_names: The skeleton node names (used as the result keys).
            max_instances: Cap on how many labeled instances to sample.

        Returns:
            Dict mapping node name to ``(x, y)``, restricted to nodes seen
            visible in at least one sampled instance. Empty when there is no
            usable labeled data (the canvas then falls back to a spring layout).
        """
        labels = self._labels
        if labels is None or skeleton is None or not node_names:
            return {}

        n_nodes = len(node_names)
        # Collect centered point arrays, one (n_nodes, 2) row-block per instance.
        stacks: list = []
        try:
            for lf in labels:
                for inst in lf.user_instances:
                    # Only instances sharing this skeleton align row-for-row.
                    if getattr(inst, "skeleton", None) is not skeleton:
                        continue
                    pts = np.asarray(inst.numpy(invisible_as_nan=True), dtype=float)
                    if pts.shape != (n_nodes, 2):
                        continue
                    visible = np.isfinite(pts).all(axis=1)
                    if not visible.any():
                        continue
                    # Recenter on this instance's visible centroid so instances
                    # in different frame regions overlay coherently.
                    centroid = np.nanmean(pts[visible], axis=0)
                    stacks.append(pts - centroid)
                    if len(stacks) >= max_instances:
                        raise StopIteration
        except StopIteration:
            pass
        except Exception:
            return {}

        if not stacks:
            return {}

        # Shape: (n_instances, n_nodes, 2); median over instances, ignoring NaNs.
        arr = np.stack(stacks, axis=0)
        with np.errstate(all="ignore"):
            # Suppress the all-NaN-slice warning for never-visible nodes.
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                medians = np.nanmedian(arr, axis=0)  # (n_nodes, 2)

        positions: dict = {}
        for i, name in enumerate(node_names):
            x, y = medians[i]
            if np.isfinite(x) and np.isfinite(y):
                positions[name] = (float(x), float(y))
        return positions

    def _on_browse_insample_model(self):
        """Open a folder picker and set the in-sample model path line edit."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select trained sleap-nn model folder",
            self._insample_model_edit.text() or "",
        )
        if directory:
            self._insample_model_edit.setText(directory)

    def _make_threshold_row(self, items: list) -> QtWidgets.QWidget:
        """Pack labeled threshold widgets into a single compact row widget.

        Args:
            items: List of (label_widget, control_widget) pairs.

        Returns:
            A container widget laying the pairs out left-to-right.
        """
        container = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        for label, control in items:
            row.addWidget(label)
            row.addWidget(control)
        row.addStretch()
        return container

    def _add_detector_row(
        self,
        grid_row: int,
        checkbox: QtWidgets.QCheckBox,
        threshold_widget: QtWidgets.QWidget,
        help_key: Optional[str] = None,
    ):
        """Add a [enable checkbox | name | threshold | "?"] row to the grid.

        The checkbox carries its own label text, so it spans the enable and
        name columns; the threshold widget sits in the trailing column, and an
        optional plain-language "?" help button sits at the far right.

        Args:
            grid_row: Row index in the settings grid.
            checkbox: The enable checkbox (its text is the detector name).
            threshold_widget: Widget holding the detector's threshold control(s).
            help_key: Key into :data:`DETECTOR_HELP` for the "?" help button. If
                None, no help button is added for this row.
        """
        grid = self._detector_settings_grid
        grid.addWidget(checkbox, grid_row, 0, 1, 2)
        grid.addWidget(threshold_widget, grid_row, 2)
        if help_key is not None:
            grid.addWidget(self._make_help_button(help_key), grid_row, 3)

    def _make_help_button(self, help_key: str) -> QtWidgets.QToolButton:
        """Create a small "?" button that explains a detector in plain language.

        Args:
            help_key: Key into :data:`DETECTOR_HELP`.

        Returns:
            A compact tool button that pops up the detector's explanation
            (issue #2769, item 3).
        """
        button = QtWidgets.QToolButton()
        button.setText("?")
        button.setAutoRaise(True)
        button.setFocusPolicy(QtCore.Qt.NoFocus)
        button.setCursor(QtCore.Qt.PointingHandCursor)
        # Keep it small and square so it does not bloat the row.
        button.setFixedSize(20, 20)
        title = DETECTOR_HELP.get(help_key, ("", ""))[0]
        button.setToolTip(f"What does '{title}' catch?")
        button.setAccessibleName(f"Help for {title}")
        button.clicked.connect(lambda: self._show_detector_help(help_key))
        return button

    def _show_detector_help(self, help_key: str):
        """Show a plain-language explanation of a detector in a popup.

        Args:
            help_key: Key into :data:`DETECTOR_HELP`.
        """
        title, body = DETECTOR_HELP.get(
            help_key, ("Detector", "No description available.")
        )
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Information)
        box.setWindowTitle(f"{title} — what it catches")
        box.setText(f"<b>{title}</b>")
        box.setInformativeText(body)
        box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        box.exec_()

    def _build_qc_config(self) -> "QCConfig":
        """Build a QCConfig from the current detector-settings controls.

        Reads every per-detector control and maps it onto the corresponding
        QCConfig field. All fields not exposed in the GUI are left at their
        QCConfig defaults.

        Returns:
            A QCConfig reflecting the current control states.
        """
        from sleap.qc.config import QCConfig

        return QCConfig(
            use_chirality=self._cb_flip.isChecked(),
            chirality_flip_threshold=self._sb_flip_thr.value(),
            use_split_detection=self._cb_chimera.isChecked(),
            use_duplicate_score=self._cb_duplicate.isChecked(),
            duplicate_score_threshold=self._sb_dup_thr.value(),
            use_chain_ordering=self._cb_chain.isChecked(),
            chain_turn_angle_deg=float(self._sb_chain_angle.value()),
            order_inversion_threshold=self._sb_order_thr.value(),
            ordered_chains=self._collect_ordered_chains(),
            use_missing_node_check=self._cb_missing.isChecked(),
            missing_node_prob_threshold=self._sb_missing_thr.value(),
            use_appearance=self._cb_appearance.isChecked(),
            use_insample_prediction=self._cb_insample.isChecked(),
            insample_model_path=self._insample_model_edit.text().strip(),
        )

    @staticmethod
    def _config_flag(value, default: bool) -> bool:
        """Resolve a QCConfig toggle (which may be ``"auto"``) to a checkbox bool.

        Several QCConfig toggles accept a literal ``"auto"`` sentinel as well as
        a plain ``bool`` (e.g. ``use_chirality``/``use_chain_ordering``). The
        Detector Settings checkboxes are plain on/off, so map a real bool
        straight through and fall back to the detector's documented GUI default
        for ``"auto"`` (or any other non-bool) value.

        Args:
            value: The QCConfig field value (``bool`` or ``"auto"``).
            default: The checkbox state to use when ``value`` is not a bool.

        Returns:
            The boolean checkbox state to apply.
        """
        return value if isinstance(value, bool) else default

    def _apply_config_to_widgets(self, config: "QCConfig"):
        """Push a QCConfig's values back into the Detector Settings controls.

        Inverse of :meth:`_build_qc_config`: every per-detector control that
        feeds the config is set from the matching field, so the panel can be
        reset to a known configuration (e.g. the "Restore defaults" button feeds
        a fresh ``QCConfig()``). Toggles that accept ``"auto"`` fall back to the
        detector's documented GUI default via :meth:`_config_flag`.

        Args:
            config: The configuration to display in the controls.
        """
        # Enable checkboxes (map "auto" sentinels to the documented GUI default:
        # reliable detectors ON, experimental detectors OFF).
        self._cb_flip.setChecked(self._config_flag(config.use_chirality, True))
        self._cb_chimera.setChecked(self._config_flag(config.use_split_detection, True))
        self._cb_duplicate.setChecked(
            self._config_flag(config.use_duplicate_score, True)
        )
        self._cb_chain.setChecked(self._config_flag(config.use_chain_ordering, False))
        self._cb_missing.setChecked(
            self._config_flag(config.use_missing_node_check, False)
        )
        self._cb_appearance.setChecked(self._config_flag(config.use_appearance, False))
        self._cb_insample.setChecked(
            self._config_flag(config.use_insample_prediction, False)
        )

        # Per-detector thresholds.
        self._sb_flip_thr.setValue(config.chirality_flip_threshold)
        self._sb_dup_thr.setValue(config.duplicate_score_threshold)
        self._sb_chain_angle.setValue(int(config.chain_turn_angle_deg))
        self._sb_order_thr.setValue(config.order_inversion_threshold)
        self._sb_missing_thr.setValue(config.missing_node_prob_threshold)

        # In-sample model path.
        self._insample_model_edit.setText(config.insample_model_path or "")

        # Ordered chains: drop the free-text box and any half-finished trace,
        # then load the config's chains as the saved-chains list.
        self._ordered_chains_edit.setPlainText("")
        self._trace_in_progress = []
        self._traced_chains = [list(chain) for chain in (config.ordered_chains or [])]
        self._refresh_trace_readout()
        self._refresh_chains_list()

    def _on_restore_detector_defaults(self):
        """Reset every Detector Settings control back to ``QCConfig()`` defaults.

        Builds a fresh :class:`~sleap.qc.config.QCConfig` and pushes its values
        into the controls via :meth:`_apply_config_to_widgets`, so enable
        checkboxes, thresholds, ordered chains, the in-sample model path and the
        appearance/missing/prediction toggles all return to the shipped defaults
        (issue #2769 follow-up, "restore to default setting").
        """
        from sleap.qc.config import QCConfig

        self._apply_config_to_widgets(QCConfig())

    def _parse_ordered_chains(self) -> list:
        """Parse the ordered-chains text box into a list of node-name lists.

        Each non-empty line becomes one chain; node names are split on commas
        and stripped, with empty names dropped. Empty text yields ``[]``.

        Returns:
            A list of lists of node-name strings.
        """
        text = self._ordered_chains_edit.toPlainText()
        chains = []
        for line in text.splitlines():
            names = [name.strip() for name in line.split(",")]
            names = [name for name in names if name]
            if names:
                chains.append(names)
        return chains

    def _collect_ordered_chains(self) -> list:
        """Combine skeleton-traced chains with the advanced free-text chains.

        The chains the user traced by clicking on the skeleton (item 2) come
        first, followed by any chains typed into the advanced free-text box.
        Exact duplicate chains are dropped so tracing and then typing the same
        chain does not double it up.

        Returns:
            A list of node-name lists for ``QCConfig.ordered_chains``.
        """
        chains: list = []
        seen: set = set()
        for chain in list(self._traced_chains) + self._parse_ordered_chains():
            key = tuple(chain)
            if key in seen:
                continue
            seen.add(key)
            chains.append(list(chain))
        return chains

    def _connect_signals(self):
        """Connect UI signals."""
        self._run_button.clicked.connect(self._on_run_analysis)
        self._cancel_button.clicked.connect(self._on_cancel_analysis)
        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self._score_canvas.threshold_changed.connect(self._on_canvas_threshold_changed)
        self._table_view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )
        self._table_view.doubleClicked.connect(self._on_row_double_clicked)
        # Keep the "X / Y reviewed" counter in sync whenever a Reviewed
        # checkbox is toggled in the model (item 6), and -- when "Hide reviewed"
        # is active -- drop the now-reviewed row from the view live (Group C).
        self._table_model.dataChanged.connect(self._on_reviewed_changed)
        # Re-apply the filters whenever the "Hide reviewed" toggle changes.
        self._hide_reviewed_check.toggled.connect(self._on_hide_reviewed_toggled)

    def _update_spinner(self):
        """Update the spinner animation character."""
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_chars)
        # Update the progress label with spinner
        current_text = self._progress_label.text()
        # Remove old spinner if present
        for char in self._spinner_chars:
            if current_text.startswith(char + " "):
                current_text = current_text[2:]
                break
        self._progress_label.setText(
            f"{self._spinner_chars[self._spinner_idx]} {current_text}"
        )

    def _on_cancel_analysis(self):
        """Handle cancel button click."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._progress_label.setText("Cancelling...")
            self._cancel_button.setEnabled(False)

    def set_labels(self, labels: "sio.Labels"):
        """Set the labels to analyze.

        Args:
            labels: A sleap_io.Labels object.
        """
        self._labels = labels
        self._detector = None
        self._results = None
        self._selected_flag = None

        # Reset the per-results filter/reviewed bookkeeping for a fresh project.
        self._all_flagged = []
        self._visible_issue_types = None
        self._reviewed_keys.clear()
        # Reset the "Hide reviewed" filter; block its toggled signal so this
        # reset does not trigger a stray re-filter mid-reset (Group C).
        self._hide_reviewed_refilter_pending = False
        self._hide_reviewed_check.blockSignals(True)
        self._hide_reviewed_check.setChecked(False)
        self._hide_reviewed_check.setEnabled(False)
        self._hide_reviewed_check.blockSignals(False)

        # Reset the chain being traced (a new project may have a new skeleton);
        # saved chains are kept so the user does not lose deliberate work.
        self._trace_in_progress = []

        # Update UI
        self._score_canvas.set_scores(np.array([]))
        self._breakdown_canvas.set_issue_counts({})
        self._table_model.items = []
        self._rebuild_issue_filter_menu([])
        self._update_reviewed_count()
        self._update_statistics()
        self._details_label.setText(SELECT_INSTANCE_PLACEHOLDER)
        # Refresh the skeleton-trace canvas with this project's skeleton.
        self._update_skeleton_trace()
        self._refresh_trace_readout()

    def _on_run_analysis(self):
        """Run QC analysis on current labels."""
        if self._labels is None:
            QtWidgets.QMessageBox.warning(
                self, "No Labels", "Please load a labels file first."
            )
            return

        n_instances = sum(len(lf.user_instances) for lf in self._labels)
        if n_instances < 2:
            QtWidgets.QMessageBox.warning(
                self,
                "Insufficient Data",
                "Need at least 2 instances to run QC analysis.",
            )
            return

        # If already running, don't start another
        if self._worker is not None and self._worker.isRunning():
            return

        # Show progress UI
        self._run_button.setEnabled(False)
        self._progress_label.setVisible(True)
        self._progress_label.setText("Starting...")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._cancel_button.setVisible(True)
        self._cancel_button.setEnabled(True)

        # Start spinner animation
        self._spinner_idx = 0
        self._spinner_timer.start()

        # Create and start worker thread with the per-detector config from the
        # Detector Settings controls.
        config = self._build_qc_config()
        self._worker = QCAnalysisWorker(self._labels, config=config)
        self._worker.progress.connect(self._on_analysis_progress)
        self._worker.finished.connect(self._on_analysis_finished)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.start()

    def _on_analysis_progress(self, step_name: str, progress: int, detail: str):
        """Handle progress update from worker."""
        # Format the label: step name with optional detail
        if detail:
            text = f"{step_name} ({detail})"
        else:
            text = step_name
        self._progress_label.setText(text)
        self._progress_bar.setValue(progress)

    def _on_analysis_finished(self, results):
        """Handle successful analysis completion."""
        self._results = results

        # Stop spinner and hide progress UI
        self._spinner_timer.stop()
        self._progress_label.setVisible(False)
        self._progress_bar.setVisible(False)
        self._cancel_button.setVisible(False)
        self._run_button.setEnabled(True)

        # Update all displays
        self._update_all_displays()

    def _on_analysis_error(self, error_msg: str):
        """Handle analysis error."""
        # Stop spinner and hide progress UI
        self._spinner_timer.stop()
        self._progress_label.setVisible(False)
        self._progress_bar.setVisible(False)
        self._cancel_button.setVisible(False)
        self._run_button.setEnabled(True)

        QtWidgets.QMessageBox.critical(
            self, "Analysis Error", f"Error during QC analysis:\n{error_msg}"
        )

    def _on_threshold_changed(self, value: int):
        """Handle threshold slider change."""
        threshold = value / 100.0
        self._threshold_label.setText(f"{threshold:.2f}")
        self._score_canvas.set_threshold(threshold)

        if self._results is not None:
            self._update_flagged_display()

    def _on_canvas_threshold_changed(self, threshold: float):
        """Handle threshold change from clicking on histogram."""
        # Clamp to slider range
        slider_value = int(threshold * 100)
        slider_value = max(30, min(90, slider_value))
        self._threshold_slider.setValue(slider_value)

    def _update_all_displays(self):
        """Update all display components after analysis."""
        if self._results is None:
            return

        # Get all scores for histogram
        scores = np.array(list(self._results.instance_scores.values()))
        self._score_canvas.set_scores(scores)

        threshold = self._threshold_slider.value() / 100.0
        self._score_canvas.set_threshold(threshold)

        self._update_flagged_display()
        self._update_statistics()

    def _update_flagged_display(self):
        """Update the flagged instances table and breakdown chart.

        Re-filters the full flagged list at the current threshold, rebuilds the
        issue-type filter menu to match the categories now present (item 5),
        then pushes only the user-selected issue types into the table. The
        breakdown chart keeps showing the FULL set so the distribution stays
        complete regardless of which types are toggled in the table.
        """
        if self._results is None:
            return

        threshold = self._threshold_slider.value() / 100.0
        flagged = self._results.get_flagged(threshold=threshold)
        self._all_flagged = flagged

        # The "Hide reviewed" filter is usable as soon as anything is flagged
        # (Group C); gate it on the full flagged set rather than the issue-type
        # categories so it works even with a single issue type.
        self._hide_reviewed_check.setEnabled(bool(flagged))

        # Rebuild the issue-type filter to match the categories now present,
        # preserving the user's current show/hide choices where possible.
        # de-dup while keeping first-seen (score-descending) order.
        present_types = list(dict.fromkeys(f.top_issue for f in flagged))
        self._rebuild_issue_filter_menu(present_types)

        # Push only the selected issue types into the table (item 5).
        self._apply_issue_filter()

        # Update breakdown chart from the FULL flagged set (not the filtered
        # view) so the distribution is always complete.
        issue_counts = {}
        for flag in flagged:
            issue = flag.top_issue.replace("_", " ").title()
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        self._breakdown_canvas.set_issue_counts(issue_counts)

        # Update feature comparison chart
        self._feature_canvas.set_feature_data(
            self._results.feature_contributions,
            self._results.instance_scores,
            threshold,
            self._results.feature_names,
        )

    def _rebuild_issue_filter_menu(self, present_types: List[str]):
        """Rebuild the issue-type filter menu for the categories present.

        Builds one checkable action per raw ``top_issue`` value currently in the
        flagged set, labeled with its plain-language category name. The user's
        previous show/hide choices are preserved for types that still exist;
        newly appeared types default to shown (item 5).

        Args:
            present_types: Raw ``top_issue`` strings present in the flagged set,
                in display order.
        """
        menu = self._issue_filter_menu
        menu.clear()
        self._issue_filter_actions = {}

        if not present_types:
            self._visible_issue_types = None
            self._issue_filter_button.setEnabled(False)
            self._update_issue_filter_button_text()
            return

        # Carry over previous selections; default new types to shown.
        prev = self._visible_issue_types
        if prev is None:
            selected = set(present_types)
        else:
            selected = {t for t in present_types if t in prev}
            # Any brand-new type (not seen when prev was captured) starts shown.
            known = set(prev) | self._issue_filter_seen
            selected |= {t for t in present_types if t not in known}
        self._visible_issue_types = selected
        self._issue_filter_seen = set(present_types)

        # Convenience "All" / "None" actions at the top.
        all_action = menu.addAction("Select all")
        all_action.triggered.connect(lambda: self._set_all_issue_types(True))
        none_action = menu.addAction("Select none")
        none_action.triggered.connect(lambda: self._set_all_issue_types(False))
        menu.addSeparator()

        for raw in present_types:
            label = self._issue_category_label(raw)
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(raw in selected)
            # Bind the raw key for this action.
            action.toggled.connect(
                lambda checked, key=raw: self._on_issue_type_toggled(key, checked)
            )
            self._issue_filter_actions[raw] = action

        self._issue_filter_button.setEnabled(True)
        self._update_issue_filter_button_text()

    def _issue_category_label(self, raw_issue: str) -> str:
        """Plain-language menu label for a raw ``top_issue`` category.

        Title-cases the raw label the same way the table's Issue column does so
        the filter entries read identically to the rows they hide/show.

        Args:
            raw_issue: The raw ``top_issue`` string.

        Returns:
            A human-friendly category label.
        """
        return raw_issue.replace("_", " ").title()

    def _on_issue_type_toggled(self, raw_issue: str, checked: bool):
        """Handle a single issue-type checkbox toggle in the filter menu."""
        if self._visible_issue_types is None:
            self._visible_issue_types = set()
        if checked:
            self._visible_issue_types.add(raw_issue)
        else:
            self._visible_issue_types.discard(raw_issue)
        self._update_issue_filter_button_text()
        self._apply_issue_filter()

    def _set_all_issue_types(self, checked: bool):
        """Check or uncheck every issue-type entry at once."""
        actions = getattr(self, "_issue_filter_actions", {})
        if checked:
            self._visible_issue_types = set(actions.keys())
        else:
            self._visible_issue_types = set()
        # Reflect in the checkboxes without re-triggering per-item handlers.
        for raw, action in actions.items():
            action.blockSignals(True)
            action.setChecked(raw in self._visible_issue_types)
            action.blockSignals(False)
        self._update_issue_filter_button_text()
        self._apply_issue_filter()

    def _update_issue_filter_button_text(self):
        """Update the filter button caption to reflect the active selection."""
        actions = getattr(self, "_issue_filter_actions", {})
        total = len(actions)
        if total == 0:
            self._issue_filter_button.setText("Issue types: all")
            return
        selected = self._visible_issue_types or set()
        n_sel = sum(1 for raw in actions if raw in selected)
        if n_sel == total:
            self._issue_filter_button.setText("Issue types: all")
        elif n_sel == 0:
            self._issue_filter_button.setText("Issue types: none")
        else:
            self._issue_filter_button.setText(f"Issue types: {n_sel} of {total}")

    def _filtered_flagged(self) -> List["QCFlag"]:
        """Return the flagged list under the active filters.

        Applies the issue-type filter and the "Hide reviewed" filter together
        (logical AND): when both are active a row must match a selected issue
        type *and* not be in :attr:`_reviewed_keys` to be shown (Group C).
        """
        if self._visible_issue_types is None:
            flagged = list(self._all_flagged)
        else:
            flagged = [
                f for f in self._all_flagged if f.top_issue in self._visible_issue_types
            ]
        if self._hide_reviewed_check.isChecked():
            flagged = [f for f in flagged if f.instance_key not in self._reviewed_keys]
        return flagged

    def _apply_issue_filter(self):
        """Push the issue-type-filtered flagged subset into the table model."""
        self._table_model.items = self._filtered_flagged()
        self._update_reviewed_count()

    def _update_reviewed_count(self, *args):
        """Refresh the "X / Y reviewed" counter for the rows currently shown.

        Args:
            *args: Ignored; lets this slot accept ``dataChanged`` signal args.
        """
        total = self._table_model.rowCount()
        reviewed = self._table_model.reviewed_count()
        self._reviewed_count_label.setText(f"{reviewed} / {total} reviewed")

    def _on_reviewed_changed(self, *args):
        """React to a Reviewed-state change in the model (Group C, item 6).

        Always refreshes the running "X / Y reviewed" counter. When the "Hide
        reviewed" filter is active, also re-applies the filters so a row just
        ticked reviewed drops out of the not-reviewed view live. The re-filter is
        *deferred* to the next event-loop turn because the ``dataChanged`` that
        triggers this slot is emitted from inside the model's own ``setData`` (or
        the selection-driven auto-mark); resetting the model's rows synchronously
        at that point would yank the data out from under the in-flight edit.

        Args:
            *args: Ignored; lets this slot accept ``dataChanged`` signal args.
        """
        self._update_reviewed_count()
        if (
            self._hide_reviewed_check.isChecked()
            and not self._hide_reviewed_refilter_pending
        ):
            self._hide_reviewed_refilter_pending = True
            QtCore.QTimer.singleShot(0, self._apply_hide_reviewed_refilter)

    def _apply_hide_reviewed_refilter(self):
        """Deferred re-filter after a reviewed change while hiding reviewed rows.

        Runs on the next event-loop turn (scheduled by :meth:`_on_reviewed_changed`)
        so the model reset happens cleanly after the originating edit returns.
        """
        self._hide_reviewed_refilter_pending = False
        if self._hide_reviewed_check.isChecked():
            self._apply_issue_filter()

    def _on_hide_reviewed_toggled(self, checked: bool):
        """Re-apply the filters when the "Hide reviewed" toggle changes (Group C).

        Args:
            checked: New checkbox state (True hides reviewed rows).
        """
        self._apply_issue_filter()

    def _update_statistics(self):
        """Update the statistics panel in plain language.

        Summarizes the analysis as a short sentence ("45 of 1,200 instances
        flagged (3.8%). Most common issue: ...") instead of bare numbers
        (issue #2769, item 7).
        """
        if self._labels is None:
            self._stats_label.setText("No labels loaded yet.")
            return

        n_instances = sum(len(lf.user_instances) for lf in self._labels)
        n_frames = len(self._labels)

        if self._results is None:
            self._stats_label.setText(
                f"Ready to analyze {n_instances:,} instances "
                f"across {n_frames:,} frames.<br/>"
                f"Click <b>Run Analysis</b> to find labeling issues."
            )
            return

        threshold = self._threshold_slider.value() / 100.0
        flagged = self._results.get_flagged(threshold=threshold)
        n_flagged = len(flagged)
        pct_flagged = (n_flagged / n_instances * 100) if n_instances > 0 else 0

        # Nothing flagged at the current sensitivity: reassure the user.
        if n_flagged == 0:
            self._stats_label.setText(
                f"No issues flagged out of {n_instances:,} instances "
                f"at this sensitivity.<br/>"
                f"Drag the Sensitivity slider toward <b>More</b> to flag "
                f"borderline cases."
            )
            return

        # Most common issue, in the same friendly wording as the table column.
        issue_counts = {}
        for flag in flagged:
            label = flag.top_issue.replace("_", " ").title()
            issue_counts[label] = issue_counts.get(label, 0) + 1
        top_label, top_count = max(issue_counts.items(), key=lambda kv: kv[1])

        # High-confidence count gives a sense of how many are clear-cut.
        high_conf = sum(1 for f in flagged if f.confidence == "high")

        lines = [
            f"<b>{n_flagged:,} of {n_instances:,}</b> instances flagged "
            f"({pct_flagged:.1f}%).",
            f"Most common issue: <b>{top_label}</b> ({top_count}).",
        ]
        if high_conf:
            lines.append(f"{high_conf:,} are high-confidence.")
        lines.append(
            "<span style='color:#6c757d;'>Click a row to review each one.</span>"
        )
        self._stats_label.setText("<br/>".join(lines))

    def _on_selection_changed(self, selected, deselected):
        """Handle selection change in table."""
        indexes = self._table_view.selectionModel().selectedRows()
        if indexes:
            row = indexes[0].row()
            if row < len(self._table_model.items):
                self._selected_flag = self._table_model.items[row]
                self._update_selected_details()

                # Auto-mark reviewed on navigate: selecting a row jumps the user
                # to that instance, so count it as "looked at" (item 6). The
                # model emits dataChanged, refreshing the running counter.
                self._table_model.set_reviewed(self._selected_flag, True)

                # Navigate to the instance
                self.navigate_to_instance.emit(
                    self._selected_flag.video_idx,
                    self._selected_flag.frame_idx,
                    self._selected_flag.instance_idx,
                )
        else:
            self._selected_flag = None
            self._details_label.setText(SELECT_INSTANCE_PLACEHOLDER)

    def _on_row_double_clicked(self, index):
        """Handle double-click on table row."""
        row = index.row()
        if row < len(self._table_model.items):
            flag = self._table_model.items[row]
            self.navigate_to_instance.emit(
                flag.video_idx,
                flag.frame_idx,
                flag.instance_idx,
            )

    def _update_selected_details(self):
        """Update the selected instance details panel in plain language.

        Maps the flag's primary issue + score onto a friendly sentence telling
        the user *why* the instance was flagged and where to find it, instead of
        showing raw feature names (issue #2769, item 7).
        """
        if self._selected_flag is None:
            self._details_label.setText(SELECT_INSTANCE_PLACEHOLDER)
            return

        flag = self._selected_flag

        reason = _friendly_issue(flag.top_issue)
        # Capitalize the first letter of the reason for a readable sentence.
        reason_sentence = reason[:1].upper() + reason[1:] if reason else reason

        self._details_label.setText(
            f"<b>Flagged:</b> {reason_sentence} "
            f"(confidence {flag.score:.2f}).<br/>"
            f"Frame {flag.frame_idx}, instance {flag.instance_idx}.<br/>"
            f"<span style='color:#6c757d;'>Double-click the row to jump there.</span>"
        )

    @property
    def has_results(self) -> bool:
        """Return True if analysis results are available."""
        return self._results is not None

    @property
    def has_flags(self) -> bool:
        """Return True if there are flagged items to navigate."""
        return len(self._table_model.items) > 0

    def goto_next_flag(self) -> bool:
        """Navigate to the next flagged instance in the table.

        Returns:
            True if navigation occurred, False if no items or at end.
        """
        if not self.has_flags:
            return False

        # Get current selection
        indexes = self._table_view.selectionModel().selectedRows()
        current_row = indexes[0].row() if indexes else -1

        # Move to next row (wrap around)
        next_row = (current_row + 1) % len(self._table_model.items)

        # Select the row (this triggers navigation via _on_selection_changed)
        self._table_view.selectRow(next_row)
        return True

    def goto_prev_flag(self) -> bool:
        """Navigate to the previous flagged instance in the table.

        Returns:
            True if navigation occurred, False if no items.
        """
        if not self.has_flags:
            return False

        # Get current selection
        indexes = self._table_view.selectionModel().selectedRows()
        n_items = len(self._table_model.items)
        current_row = indexes[0].row() if indexes else 0

        # Move to previous row (wrap around)
        prev_row = (current_row - 1) % n_items

        # Select the row (this triggers navigation via _on_selection_changed)
        self._table_view.selectRow(prev_row)
        return True

    def export_results(self):
        """Export QC results to CSV (public method for dialog)."""
        import os

        if self._results is None:
            QtWidgets.QMessageBox.warning(
                self, "No Results", "Please run analysis first."
            )
            return

        # Determine default directory: use last export dir, or labels folder, or CWD
        default_dir = self._last_export_dir
        if default_dir is None and self._labels is not None:
            # Try to get directory from labels provenance
            provenance = getattr(self._labels, "provenance", None)
            if provenance is not None:
                labels_path = getattr(provenance, "filename", None)
                if labels_path:
                    default_dir = os.path.dirname(labels_path)

        default_filename = "qc_results.csv"
        if default_dir:
            default_path = os.path.join(default_dir, default_filename)
        else:
            default_path = default_filename

        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export QC Results",
            default_path,
            "CSV Files (*.csv);;All Files (*)",
        )

        if filepath:
            try:
                df = self._results.to_dataframe()
                df.to_csv(filepath, index=False)
                # Persist the directory for next export
                self._last_export_dir = os.path.dirname(filepath)
                QtWidgets.QMessageBox.information(
                    self, "Export Complete", f"Results exported to:\n{filepath}"
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Export Error", f"Error exporting results:\n{str(e)}"
                )

    def export_to_suggestions(self) -> int:
        """Export flagged frames to the suggestions list.

        Creates SuggestionFrame objects for each unique frame that contains
        flagged instances and adds them to labels.suggestions.

        Returns:
            Number of suggestions added, or -1 if export failed.
        """
        from sleap_io import SuggestionFrame

        if self._results is None:
            QtWidgets.QMessageBox.warning(
                self, "No Results", "Please run analysis first."
            )
            return -1

        if self._labels is None:
            QtWidgets.QMessageBox.warning(self, "No Labels", "No labels file loaded.")
            return -1

        threshold = self._threshold_slider.value() / 100.0
        flagged = self._results.get_flagged(threshold=threshold)

        if not flagged:
            QtWidgets.QMessageBox.information(
                self,
                "No Flagged Instances",
                "No instances are flagged at the current threshold.",
            )
            return 0

        # Get unique frames (video_idx, frame_idx pairs)
        # Track the highest score for each frame for metadata
        unique_frames = {}
        for flag in flagged:
            key = (flag.video_idx, flag.frame_idx)
            if key not in unique_frames or flag.score > unique_frames[key].score:
                unique_frames[key] = flag

        # Filter out frames that are already in suggestions
        existing_suggestions = set()
        for sugg in self._labels.suggestions:
            video_idx = self._labels.videos.index(sugg.video)
            existing_suggestions.add((video_idx, sugg.frame_idx))

        new_frames = {
            key: flag
            for key, flag in unique_frames.items()
            if key not in existing_suggestions
        }

        if not new_frames:
            QtWidgets.QMessageBox.information(
                self,
                "Already Added",
                f"All {len(unique_frames)} flagged frames are already in suggestions.",
            )
            return 0

        # Create SuggestionFrame objects
        suggestions = []
        for (video_idx, frame_idx), flag in new_frames.items():
            video = self._labels.videos[video_idx]
            suggestion = SuggestionFrame(video=video, frame_idx=frame_idx)
            suggestions.append(suggestion)

        # Add to labels
        self._labels.suggestions.extend(suggestions)

        n_added = len(suggestions)
        n_skipped = len(unique_frames) - n_added

        msg = f"Added {n_added} frame(s) to suggestions."
        if n_skipped > 0:
            msg += f"\n({n_skipped} already in suggestions)"

        QtWidgets.QMessageBox.information(self, "Export Complete", msg)

        return n_added

    def cleanup(self):
        """Clean up resources, stopping any running analysis.

        Should be called before the widget is destroyed.
        """
        # Stop spinner timer
        self._spinner_timer.stop()

        # Cancel and wait for worker thread
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            # Wait up to 2 seconds for thread to finish
            if not self._worker.wait(2000):
                # Thread didn't finish, terminate it
                self._worker.terminate()
                self._worker.wait()
            self._worker = None

    def closeEvent(self, event):
        """Handle widget close event."""
        self.cleanup()
        super().closeEvent(event)
