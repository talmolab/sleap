"""
Dialog/widgets for showing metrics on trained models.
"""

from PySide2 import QtWidgets, QtCore

import numpy as np
import pandas as pd
import seaborn as sns

from sleap.gui.dataviews import GenericTableModel, GenericTableView
from sleap.gui.dialogs.filedialog import FileDialog
from sleap.gui.learning.configs import TrainingConfigsGetter, ConfigFileInfo
from sleap.gui.learning.dialog import TrainingEditorWidget

from sleap.gui.widgets.mpl import MplCanvas

from typing import Optional, Text


class MetricsTableDialog(QtWidgets.QWidget):
    """Dialog for showing table with multiple models.

    The dialog is can show multiple models, including those which don't
    already have metrics (ideally you'd be able to generate evals but this isn't
    currently supported).

    You can then view details on the models (hyperparameters or more detailed
    metrics).

    The typical use-case is to init dialog with path to labels file, and it
    will then show all trained models found within subdirectories.
    """

    def __init__(self, labels_filename: Text = ""):
        super(MetricsTableDialog, self).__init__()

        labels_filename = labels_filename or ""

        self._cfg_getter = TrainingConfigsGetter.make_from_labels_filename(
            labels_filename,
        )
        self._cfg_getter.search_depth = 4

        self.table_model = MetricsTableModel(items=[])
        self.table_view = GenericTableView(
            model=self.table_model, is_activatable=True, row_name="trained_model"
        )
        self.table_view.state.connect("trained_model", self._show_metric_details)
        self.table_view.state.connect("selected_trained_model", self._update_gui)

        button_layout = QtWidgets.QHBoxLayout()
        buttons = QtWidgets.QWidget()
        buttons.setLayout(button_layout)

        btn = QtWidgets.QPushButton("Add Trained Model(s)")
        btn.clicked.connect(self._add_model_action)
        button_layout.addWidget(btn)

        btn = QtWidgets.QPushButton("View Hyperparameters")
        btn.clicked.connect(self._show_model_params)
        button_layout.addWidget(btn)
        self._view_model_btn = btn

        btn = QtWidgets.QPushButton("View Metrics")
        btn.clicked.connect(self._show_metric_details)
        button_layout.addWidget(btn)
        self._view_metrics_btn = btn

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.table_view)
        layout.addWidget(buttons)
        self.setLayout(layout)

        self.setWindowTitle("Metrics for Trained Models")

        self._update_cfgs()
        self._update_gui()

        self.setMinimumWidth(1200)

    def _update_gui(self, *args):
        """Enables/disables buttons as appropriate for table row selection."""
        is_selected = self.table_view.state["selected_trained_model"] is not None
        self._view_model_btn.setEnabled(is_selected)
        self._view_metrics_btn.setEnabled(is_selected)

    def _update_cfgs(self):
        """Searches for models and updates table."""
        self._cfg_getter.update()
        cfgs = self._cfg_getter.get_filtered_configs(only_trained=True)
        self.table_model.items = cfgs
        self.table_view.resizeColumnsToContents()

    def _add_model_action(self):
        """Method called when user clicks 'add models' button."""
        dir = FileDialog.openDir(None, dir=None, caption="")

        if dir:
            self._cfg_getter.dir_paths.append(dir)
            self._update_cfgs()

    def _show_model(self, cfg_info: Optional[ConfigFileInfo] = None):
        """Method to show both hyperparam and metrics windows."""
        self._show_model_params(cfg_info)
        self._show_metric_details(cfg_info)

    def _show_model_params(
        self, cfg_info: Optional[ConfigFileInfo] = None, model_detail_widgets=dict()
    ):
        """
        Method to show dialog with hyperparameters for model.

        Args:
            cfg_info: The `ConfigFileInfo` for the model to show; if None,
                then show for model currently selected in table.
            model_detail_widgets: Not user param; cache for widgets so that we
                don't create new window if user views same model twice.
        """
        if cfg_info is None:
            cfg_info = self.table_view.getSelectedRowItem()

        key = cfg_info.path
        if key not in model_detail_widgets:
            model_detail_widgets[key] = TrainingEditorWidget.from_trained_config(
                cfg_info
            )

        model_detail_widgets[key].show()
        model_detail_widgets[key].raise_()
        model_detail_widgets[key].activateWindow()

    def _show_metric_details(
        self, cfg_info: Optional[ConfigFileInfo] = None, metric_detail_widgets=dict()
    ):
        """
        Method to show dialog with metrics for model.

        Args:
            cfg_info: The `ConfigFileInfo` for the model to show; if None,
                then show for model currently selected in table.
            metric_detail_widgets: Not user param; cache for widgets so that we
                don't create new window if user views same model twice.
        """
        if cfg_info is None:
            cfg_info = self.table_view.getSelectedRowItem()

        key = cfg_info.path
        if key not in metric_detail_widgets:
            metric_detail_widgets[key] = DetailedMetricsDialog(cfg_info)

        metric_detail_widgets[key].show()
        metric_detail_widgets[key].raise_()
        metric_detail_widgets[key].activateWindow()


class MetricsTableModel(GenericTableModel):
    """
    Model (i.e. Qt model/view) for table in MetricsTableDialog.
    """

    properties = (
        "Path",
        "Timestamp",
        # "Run Name",
        "Model Type",
        "Architecture",
        "Training Instances",
        "Validation Instances",
        "OKS mAP",
        "Vis Precision",
        "Vis Recall",
        "Dist: 95%",
        "Dist: 75%",
        "Dist: Avg",
    )
    show_row_numbers = False

    def item_to_data(self, obj, cfg: ConfigFileInfo):

        if cfg.training_frame_count:
            n_train_str = (
                f"{cfg.training_instance_count} ({cfg.training_frame_count} frames)"
            )
        else:
            n_train_str = ""

        if cfg.validation_frame_count:
            n_val_str = (
                f"{cfg.validation_instance_count} ({cfg.validation_frame_count} frames)"
            )
        else:
            n_val_str = ""

        arch_str = cfg.config.model.backbone.which_oneof_attrib_name()

        backbone = cfg.config.model.backbone.which_oneof()
        if hasattr(backbone, "max_stride"):
            arch_str = f"{arch_str}, max stride: {backbone.max_stride}"
        if hasattr(backbone, "filters"):
            arch_str = f"{arch_str}, filters: {backbone.filters}"

        # scale = cfg.config.data.preprocessing.input_scaling
        # if scale != 1.0:
        #     arch_str = f"{arch_str}, scale: {scale}"

        item_data = {
            "Timestamp": str(cfg.timestamp),
            # "Run Name": cfg.config.outputs.run_name,
            "Path": cfg.path_dir,
            "Model Type": cfg.head_name,
            "Architecture": arch_str,
            "Training Instances": n_train_str,
            "Validation Instances": n_val_str,
        }

        metrics = cfg.metrics

        # import pprint
        # pp = pprint.PrettyPrinter()
        # pp.pprint(metrics)

        if metrics:
            item_data = {
                **item_data,
                "OKS mAP": f"{metrics['oks_voc.mAP']:.5f}",
                "Vis Precision": f"{metrics['vis.precision']:.5f}",
                "Vis Recall": f"{metrics['vis.recall']:.5f}",
                "Dist: 95%": f"{metrics['dist.p95']:.5f}",
                "Dist: 75%": f"{metrics['dist.p75']:.5f}",
                "Dist: Avg": f"{metrics['dist.avg']:.5f}",
            }

        return item_data


METRICS_KEY_LABELS = {
    "vis.tp": "Visibility - True Positives",
    "vis.fp": "Visibility - False Positives",
    "vis.tn": "Visibility - True Negatives",
    "vis.fn": "Visibility - False Negatives",
    "vis.precision": "Visibility - Precision",
    "vis.recall": "Visibility - Recall",
    "dist.avg": "Average Distance (ground truth vs prediction)",
    "dist.p50": "Distance for 50th percentile",
    "dist.p75": "Distance for 75th percentile",
    "dist.p90": "Distance for 90th percentile",
    "dist.p95": "Distance for 95th percentile",
    "dist.p99": "Distance for 99th percentile",
    "pck.mPCK": "Mean Percentage of Correct Keypoints (PCK)",
    "oks.mOKS": "Mean Object Keypoint Similarity (OKS)",
    "oks_voc.mAP": "VOC with OKS scores - mean Average Precision (mAP)",
    "oks_voc.mAR": "VOC with OKS scores - mean Average Recall (mAR)",
    "pck_voc.mAP": "VOC with PCK scores - mean Average Precision (mAP)",
    "pck_voc.mAR": "VOC with PCK scores - mean Average Recall (mAR)",
}


class DetailedMetricsDialog(QtWidgets.QWidget):
    """
    Dialog to show detailed metrics for a trained model.

    Args:
        cfg_info: The `ConfigFileInfo` object (from `TrainingConfigsGetter`)
            for the model we want to show.
    """

    def __init__(self, cfg_info: ConfigFileInfo):
        super(DetailedMetricsDialog, self).__init__()

        self.setWindowTitle(cfg_info.path_dir)
        self.setMinimumWidth(800)

        self.cfg_info = cfg_info
        self.skeleton = cfg_info.skeleton

        self.metrics = self.cfg_info.metrics

        layout = QtWidgets.QHBoxLayout()
        metrics_layout = QtWidgets.QFormLayout()

        if self.metrics:
            for key, val in self.metrics.items():
                if (
                    isinstance(val, np.float)
                    or isinstance(val, np.ndarray)
                    and not len(val.shape)
                ):
                    val_str = str(val)

                    key_str = (
                        METRICS_KEY_LABELS[key] if key in METRICS_KEY_LABELS else key
                    )

                    text_widget = QtWidgets.QLabel(val_str)
                    text_widget.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                    metrics_layout.addRow(f"<b>{key_str}</b>:", text_widget)

            metrics_widget = QtWidgets.QWidget()
            metrics_widget.setLayout(metrics_layout)

            self.canvas = MplCanvas(dpi=50)

            layout.addWidget(metrics_widget)
            layout.addWidget(self.canvas)

            self._plot_distances()
        else:
            text_widget = QtWidgets.QLabel(
                "Metrics have not been generated for this model."
            )
            layout.addWidget(text_widget)

        self.setLayout(layout)

    def _plot_distances(self):
        """Plots node distances (using matplotlib widget)."""
        ax = self.canvas.axes

        node_names = self.skeleton.node_names if self.skeleton else None

        dists = pd.DataFrame(self.metrics["dist.dists"], columns=node_names).melt(
            var_name="Part", value_name="Error"
        )

        sns.boxplot(data=dists, x="Error", y="Part", fliersize=0, ax=ax)

        sns.stripplot(
            data=dists, x="Error", y="Part", alpha=0.25, linewidth=1, jitter=0.2, ax=ax
        )

        ax.set_title("Node distances (ground truth vs prediction)")
        dist_1d = self.metrics["dist.dists"].flatten()

        xmax = np.ceil(np.ceil(np.nanpercentile(dist_1d, 95) / 5) + 1) * 5
        ax.set_xlim([0, xmax])
        ax.set_xlabel("Error (px)")

    def _plot_oks(self):
        """Plots OKS -- not currently used."""
        ax = self.canvas.axes
        metrics = self.metrics

        for match_threshold, precision in zip(
            metrics["oks_voc.match_score_thresholds"], metrics["oks_voc.precisions"]
        ):
            ax.plot(
                metrics["oks_voc.recall_thresholds"],
                precision,
                "-",
                label=f"OKS @ {match_threshold:.2f}",
            )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")


# if __name__ == "__main__":
#     from PySide2.QtWidgets import QApplication
#
#     app = QApplication([])
#     window = MetricsTableDialog("")
#     window.show()
#
#     app.exec_()
