"""Module for creating dock widgets for the `MainWindow`."""

from typing import Callable, Iterable, List, Optional, Type, Union

from qtpy import QtGui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDockWidget,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sleap.gui.dataviews import (
    GenericTableModel,
    GenericTableView,
    LabeledFrameTableModel,
    SkeletonEdgesTableModel,
    SkeletonNodeModel,
    SkeletonNodesTableModel,
    SuggestionsTableModel,
    VideosTableModel,
)
from sleap.gui.dialogs.formbuilder import YamlFormWidget
from sleap.gui.widgets.views import CollapsibleWidget
from sleap.skeleton import Skeleton, SkeletonDecoder
from sleap.util import find_files_by_suffix, get_package_file


class DockWidget(QDockWidget):
    """'Abstract' class for a dockable widget attached to the `MainWindow`."""

    def __init__(
        self,
        name: str,
        main_window: Optional[QMainWindow] = None,
        model_type: Optional[
            Union[Type[GenericTableModel], List[Type[GenericTableModel]]]
        ] = None,
        widgets: Optional[Iterable[QWidget]] = None,
        tab_with: Optional[QLayout] = None,
    ):
        # Create the dock and add it to the main window.
        super().__init__(name)
        self.name = name
        self.main_window = main_window
        self.setup_dock(widgets, tab_with)

        # Create the model and table for the dock.
        self.model_type = model_type
        if self.model_type is None:
            self.model = None
            self.table = None
        else:
            self.model = self.create_models()
            self.table = self.create_tables()

        # Lay out the dock widget, adding/creating other widgets if needed.
        self.lay_everything_out()

    @property
    def wgt_layout(self) -> QVBoxLayout:
        return self.widget().layout()

    def setup_dock(self, widgets, tab_with):
        """Create a dock widget.

        Args:
            widgets: The widgets to add to the dock.
            tab_with: The `QLayout` to tabify the `DockWidget` with.
        """

        self.setObjectName(self.name + "Dock")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        dock_widget = QWidget()
        dock_widget.setObjectName(self.name + "Widget")
        layout = QVBoxLayout()

        widgets = widgets or []
        for widget in widgets:
            layout.addWidget(widget)

        dock_widget.setLayout(layout)
        self.setWidget(dock_widget)

        self.add_to_window(self.main_window, tab_with)

    def add_to_window(self, main_window: QMainWindow, tab_with: QVBoxLayout):
        """Add the dock to the `MainWindow`.

        Args:
            tab_with: The `QLayout` to tabify the `DockWidget` with.
        """
        self.main_window = main_window
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self)
        self.main_window.viewMenu.addAction(self.toggleViewAction())

        if tab_with is not None:
            self.main_window.tabifyDockWidget(tab_with, self)

    def add_button(self, to: QLayout, label: str, action: Callable, key=None):
        key = key or label.lower()
        btn = QPushButton(label)
        btn.clicked.connect(action)
        to.addWidget(btn)
        self.main_window._buttons[key] = btn
        return btn

    def create_models(self) -> GenericTableModel:
        """Create the model for the table in the dock (if any).

        Implement this in the subclass.
        Ex:
            self.model = self.model_type(items=[], context=self.main_window.commands)

        Returns:
            The model.
        """
        raise NotImplementedError

    def create_tables(self) -> GenericTableView:
        """Add a table to the dock.

        Implement this in the subclass.
        Ex:
            self.table = GenericTableView(
                state=self.main_window.state,
                model=self.model or self.create_models(),
            )
            self.wgt_layout.addWidget(self.table)

        Returns:
            The table widget.
        """
        raise NotImplementedError

    def lay_everything_out(self) -> None:
        """Lay out the dock widget, adding/creating other widgets if needed.

        Implement this in the subclass. No example as this is extremely custom.
        """
        raise NotImplementedError


class VideosDock(DockWidget):
    """Dock widget for displaying video information."""

    def __init__(
        self,
        main_window: Optional[QMainWindow] = None,
    ):
        super().__init__(
            name="Videos", main_window=main_window, model_type=VideosTableModel
        )

    def create_models(self) -> VideosTableModel:
        self.model = self.model_type(
            items=self.main_window.labels.videos, context=self.main_window.commands
        )
        return self.model

    def create_tables(self) -> GenericTableView:
        if self.model is None:
            self.create_models()

        main_window = self.main_window
        self.table = GenericTableView(
            state=main_window.state,
            row_name="video",
            is_activatable=True,
            model=self.model,
            ellipsis_left=True,
            multiple_selection=True,
        )

        return self.table

    def create_video_edit_and_nav_buttons(self) -> QWidget:
        """Create the buttons for editing and navigating videos in table."""
        main_window = self.main_window

        hb = QHBoxLayout()
        self.add_button(hb, "Toggle Grayscale", main_window.commands.toggleGrayscale)
        self.add_button(hb, "Show Video", self.table.activateSelected)
        self.add_button(hb, "Add Videos", main_window.commands.addVideo)
        self.add_button(hb, "Remove Video", main_window.commands.removeVideo)
        hbw = QWidget()
        hbw.setLayout(hb)
        return hbw

    def lay_everything_out(self):
        """Lay out the dock widget, adding/creating other widgets if needed."""
        self.wgt_layout.addWidget(self.table)

        video_edit_and_nav_buttons = self.create_video_edit_and_nav_buttons()
        self.wgt_layout.addWidget(video_edit_and_nav_buttons)


class SkeletonDock(DockWidget):
    """Dock widget for displaying skeleton information."""

    def __init__(
        self,
        main_window: Optional[QMainWindow] = None,
        tab_with: Optional[QLayout] = None,
    ):
        self.nodes_model_type = SkeletonNodesTableModel
        self.edges_model_type = SkeletonEdgesTableModel
        super().__init__(
            name="Skeleton",
            main_window=main_window,
            model_type=[self.nodes_model_type, self.edges_model_type],
            tab_with=tab_with,
        )

    def create_models(self) -> GenericTableModel:
        main_window = self.main_window
        self.nodes_model = self.nodes_model_type(
            items=main_window.state["skeleton"], context=main_window.commands
        )
        self.edges_model = self.edges_model_type(
            items=main_window.state["skeleton"], context=main_window.commands
        )
        return [self.nodes_model, self.edges_model]

    def create_tables(self) -> GenericTableView:
        if self.model is None:
            self.create_models()

        main_window = self.main_window
        self.nodes_table = GenericTableView(
            state=main_window.state,
            row_name="node",
            model=self.nodes_model,
        )

        self.edges_table = GenericTableView(
            state=main_window.state,
            row_name="edge",
            model=self.edges_model,
        )

        return [self.nodes_table, self.edges_table]

    def create_project_skeleton_groupbox(self) -> QGroupBox:
        """Create the groupbox for the project skeleton."""
        main_window = self.main_window
        gb = QGroupBox("Project Skeleton")
        vgb = QVBoxLayout()

        nodes_widget = QWidget()
        vb = QVBoxLayout()
        graph_tabs = QTabWidget()

        vb.addWidget(self.nodes_table)
        hb = QHBoxLayout()
        self.add_button(hb, "New Node", main_window.commands.newNode)
        self.add_button(hb, "Delete Node", main_window.commands.deleteNode)

        hbw = QWidget()
        hbw.setLayout(hb)
        vb.addWidget(hbw)
        nodes_widget.setLayout(vb)
        graph_tabs.addTab(nodes_widget, "Nodes")

        def _update_edge_src():
            self.skeletonEdgesDst.model().skeleton = main_window.state["skeleton"]

        edges_widget = QWidget()

        vb = QVBoxLayout()
        vb.addWidget(self.edges_table)

        hb = QHBoxLayout()
        self.skeletonEdgesSrc = QComboBox()
        self.skeletonEdgesSrc.setEditable(False)
        self.skeletonEdgesSrc.currentIndexChanged.connect(_update_edge_src)
        self.skeletonEdgesSrc.setModel(SkeletonNodeModel(main_window.state["skeleton"]))
        hb.addWidget(self.skeletonEdgesSrc)
        hb.addWidget(QLabel("to"))
        self.skeletonEdgesDst = QComboBox()
        self.skeletonEdgesDst.setEditable(False)
        hb.addWidget(self.skeletonEdgesDst)
        self.skeletonEdgesDst.setModel(
            SkeletonNodeModel(
                main_window.state["skeleton"],
                lambda: self.skeletonEdgesSrc.currentText(),
            )
        )

        def new_edge():
            src_node = self.skeletonEdgesSrc.currentText()
            dst_node = self.skeletonEdgesDst.currentText()
            main_window.commands.newEdge(src_node, dst_node)

        self.add_button(hb, "Add Edge", new_edge)
        self.add_button(hb, "Delete Edge", main_window.commands.deleteEdge)
        hbw = QWidget()
        hbw.setLayout(hb)
        vb.addWidget(hbw)
        edges_widget.setLayout(vb)
        graph_tabs.addTab(edges_widget, "Edges")
        vgb.addWidget(graph_tabs)

        hb = QHBoxLayout()
        self.add_button(hb, "Load From File...", main_window.commands.openSkeleton)
        self.add_button(hb, "Save As...", main_window.commands.saveSkeleton)

        hbw = QWidget()
        hbw.setLayout(hb)
        vgb.addWidget(hbw)

        # Add graph tabs to "Project Skeleton" group box
        gb.setLayout(vgb)
        return gb

    def create_templates_groupbox(self) -> QGroupBox:
        """Create the groupbox for the skeleton templates."""
        main_window = self.main_window

        gb = CollapsibleWidget("Templates")
        vb = QVBoxLayout()
        hb = QHBoxLayout()

        skeletons_folder = get_package_file("skeletons")
        skeletons_json_files = find_files_by_suffix(
            skeletons_folder, suffix=".json", depth=1
        )
        skeletons_names = [json.name.split(".")[0] for json in skeletons_json_files]
        self.skeleton_templates = QComboBox()
        self.skeleton_templates.addItems(skeletons_names)
        self.skeleton_templates.setEditable(False)
        hb.addWidget(self.skeleton_templates)
        self.add_button(hb, "Load", main_window.commands.openSkeletonTemplate)
        hbw = QWidget()
        hbw.setLayout(hb)
        vb.addWidget(hbw)

        hb = QHBoxLayout()
        self.skeleton_preview_image = QLabel("Preview Skeleton")
        hb.addWidget(self.skeleton_preview_image)
        hb.setAlignment(self.skeleton_preview_image, Qt.AlignLeft)

        self.skeleton_description = QLabel(
            f'<strong>Description:</strong> {main_window.state["skeleton_description"]}'
        )
        self.skeleton_description.setWordWrap(True)
        hb.addWidget(self.skeleton_description)
        hb.setAlignment(self.skeleton_description, Qt.AlignLeft)

        hbw = QWidget()
        hbw.setLayout(hb)
        vb.addWidget(hbw)

        def updatePreviewImage(preview_image_bytes: bytes):

            # Decode the preview image
            preview_image = SkeletonDecoder.decode_preview_image(preview_image_bytes)

            # Create a QImage from the Image
            preview_image = QtGui.QImage(
                preview_image.tobytes(),
                preview_image.size[0],
                preview_image.size[1],
                QtGui.QImage.Format_RGBA8888,  # Format for RGBA images (see Image.mode)
            )

            preview_image = QtGui.QPixmap.fromImage(preview_image)

            self.skeleton_preview_image.setPixmap(preview_image)

        def update_skeleton_preview(idx: int):
            skel = Skeleton.load_json(skeletons_json_files[idx])
            main_window.state["skeleton_description"] = (
                f"<strong>Description:</strong> {skel.description}<br><br>"
                f"<strong>Nodes ({len(skel)}):</strong> {', '.join(skel.node_names)}"
            )
            self.skeleton_description.setText(main_window.state["skeleton_description"])
            updatePreviewImage(skel.preview_image)

        self.skeleton_templates.currentIndexChanged.connect(update_skeleton_preview)
        update_skeleton_preview(idx=0)

        gb.set_content_layout(vb)
        return gb

    def lay_everything_out(self):
        """Lay out the dock widget, adding/creating other widgets if needed."""
        templates_gb = self.create_templates_groupbox()
        self.wgt_layout.addWidget(templates_gb)

        project_skeleton_groupbox = self.create_project_skeleton_groupbox()
        self.wgt_layout.addWidget(project_skeleton_groupbox)


class SuggestionsDock(DockWidget):
    """Dock widget for displaying suggestions."""

    def __init__(self, main_window: QMainWindow, tab_with: Optional[QLayout] = None):
        super().__init__(
            name="Labeling Suggestions",
            main_window=main_window,
            model_type=SuggestionsTableModel,
            tab_with=tab_with,
        )

    def create_models(self) -> SuggestionsTableModel:
        self.model = self.model_type(
            items=self.main_window.labels.suggestions, context=self.main_window.commands
        )
        return self.model

    def create_tables(self) -> GenericTableView:
        self.table = GenericTableView(
            state=self.main_window.state,
            is_sortable=True,
            model=self.model,
        )

        # Connect some actions to the table
        def goto_suggestion(*args):
            selected_frame = self.table.getSelectedRowItem()
            self.main_window.commands.gotoVideoAndFrame(
                selected_frame.video, selected_frame.frame_idx
            )

        self.table.doubleClicked.connect(goto_suggestion)
        self.main_window.state.connect("suggestion_idx", self.table.selectRow)

        return self.table

    def lay_everything_out(self) -> None:
        self.wgt_layout.addWidget(self.table)

        table_edit_buttons = self.create_table_edit_buttons()
        self.wgt_layout.addWidget(table_edit_buttons)

        table_nav_buttons = self.create_table_nav_buttons()
        self.wgt_layout.addWidget(table_nav_buttons)

        self.suggestions_form_widget = self.create_suggestions_form()
        self.wgt_layout.addWidget(self.suggestions_form_widget)

    def create_table_nav_buttons(self) -> QWidget:
        main_window = self.main_window
        hb = QHBoxLayout()

        self.add_button(
            hb,
            "Previous",
            main_window.process_events_then(main_window.commands.prevSuggestedFrame),
            "goto previous suggestion",
        )

        self.suggested_count_label = QLabel()
        hb.addWidget(self.suggested_count_label)

        self.add_button(
            hb,
            "Next",
            main_window.process_events_then(main_window.commands.nextSuggestedFrame),
            "goto next suggestion",
        )

        hbw = QWidget()
        hbw.setLayout(hb)
        return hbw

    def create_suggestions_form(self) -> QWidget:
        main_window = self.main_window
        suggestions_form_widget = YamlFormWidget.from_name(
            "suggestions",
            title="Generate Suggestions",
        )
        suggestions_form_widget.mainAction.connect(
            main_window.process_events_then(main_window.commands.generateSuggestions)
        )
        return suggestions_form_widget

    def create_table_edit_buttons(self) -> QWidget:
        main_window = self.main_window
        hb = QHBoxLayout()

        self.add_button(
            hb,
            "Add current frame",
            main_window.process_events_then(
                main_window.commands.addCurrentFrameAsSuggestion
            ),
            "add current frame as suggestion",
        )

        self.add_button(
            hb,
            "Remove",
            main_window.process_events_then(main_window.commands.removeSuggestion),
            "remove suggestion",
        )

        self.add_button(
            hb,
            "Clear all",
            main_window.process_events_then(main_window.commands.clearSuggestions),
            "clear suggestions",
        )

        hbw = QWidget()
        hbw.setLayout(hb)
        return hbw


class InstancesDock(DockWidget):
    """Dock widget for displaying instances."""

    def __init__(self, main_window: QMainWindow, tab_with: Optional[QLayout] = None):
        super().__init__(
            name="Instances",
            main_window=main_window,
            model_type=LabeledFrameTableModel,
            tab_with=tab_with,
        )

    def create_models(self) -> LabeledFrameTableModel:
        self.model = self.model_type(
            items=self.main_window.state["labeled_frame"],
            context=self.main_window.commands,
        )
        return self.model

    def create_tables(self) -> GenericTableView:
        self.table = GenericTableView(
            state=self.main_window.state,
            row_name="instance",
            name_prefix="",
            model=self.model,
        )
        return self.table

    def lay_everything_out(self) -> None:
        self.wgt_layout.addWidget(self.table)

        table_edit_buttons = self.create_table_edit_buttons()
        self.wgt_layout.addWidget(table_edit_buttons)

    def create_table_edit_buttons(self) -> QWidget:
        main_window = self.main_window

        hb = QHBoxLayout()
        self.add_button(
            hb, "New Instance", lambda x: main_window.commands.newInstance(offset=10)
        )
        self.add_button(
            hb, "Delete Instance", main_window.commands.deleteSelectedInstance
        )

        hbw = QWidget()
        hbw.setLayout(hb)
        return hbw
