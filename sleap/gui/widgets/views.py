"""GUI code for the views (e.g. Videos, Skeleton, Labeling Suggestions, etc.)."""

from qtpy.QtWidgets import (
    QWidget,
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
    QCheckBox,
    QGridLayout,
    QToolButton,
    QFrame,
    QSizePolicy,
    QScrollArea,
)
from qtpy.QtCore import (
    Qt,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QAbstractAnimation,
)

from sleap.util import get_package_file


class CollapsibleGroupBox(QGroupBox):
    """A collapsible group box."""

    def __init__(self, title, parent=None):
        super().__init__(title, parent=parent)

        self.setCheckable(True)
        self.setChecked(True)

        self.toggled.connect(self.on_toggled)

        # Set checkbox to be a down arrow when collapsed
        get_package_file("sleap/gui/down-arrow.png")
        self.setStyleSheet(
            """
            QGroupBox::title {
            left: 10px;
            margin-left: 4px;
            }
            QGroupBox::indicator {
                width: 13px;
                height: 13px;
            }
            QGroupBox::indicator:unchecked {
            image: url(../down-arrow.png);
            }
            
            QGroupBox::indicator:checked {
            image: url(down-arrow.png);
            }
            """
        )

    def on_toggled(self, checked):
        """Called when the group box is toggled."""

        # Hide children if we're collapsing
        for child in self.findChildren(QWidget):
            child.setVisible(checked)


class AnimatedCollapsibleWidget(QWidget):
    """An animated collapsible QWidget.

    Derived from: https://stackoverflow.com/a/37119983/13281260
    """

    def __init__(self, title: str, animation_duration: int = 3, parent: QWidget = None):
        super().__init__(parent=parent)

        self.animation_is_working = False
        self.animation_duration = animation_duration

        # Create our custom toggle button
        self.toggle_button = QToolButton()
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)

        # Create the header line
        header_line = QFrame()
        header_line.setFrameShape(QFrame.HLine)
        header_line.setFrameShadow(QFrame.Sunken)
        header_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        # Created the layout for the header
        header_layout = QHBoxLayout()
        header_layout.addWidget(self.toggle_button)
        header_layout.addWidget(header_line)

        # Create a widget to apply the header layout to
        header_widget = QWidget()
        header_widget.setLayout(header_layout)

        self.content_area = QWidget()

        if self.animation_is_working:
            self.content_area = QScrollArea()
            self.content_area.setStyleSheet(
                "QScrollArea { background-color: white; border: none; }"
            )
            self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            # Start out collapsed
            self.content_area.setMaximumHeight(0)
            self.content_area.setMinimumHeight(0)

        # Let the entire widget grow and shrink with its content
        self.toggle_animation = QParallelAnimationGroup()
        if self.animation_is_working:
            self.toggle_animation.addAnimation(
                QPropertyAnimation(self, "minimumHeight")
            )
            self.toggle_animation.addAnimation(
                QPropertyAnimation(self, "maximumHeight")
            )
            self.toggle_animation.addAnimation(
                QPropertyAnimation(self.content_area, "maximumHeight")
            )

        # Do not  waste space
        main_layout = QVBoxLayout()
        # main_layout.setVerticalSpacing(0)
        # main_layout.setContentsMargins(0, 0, 0, 0)
        # row = 0
        # main_layout.addWidget(self.toggle_button, row, 0, 1, 1, Qt.AlignLeft)
        # main_layout.addWidget(header_line, row + 1, 2, 1, 1)
        # main_layout.addWidget(self.content_area, row, 0, 1, 3)
        main_layout.addWidget(header_widget)
        main_layout.addWidget(self.content_area)
        self.setLayout(main_layout)

        self.toggle_button.clicked.connect(self.toggle_button_callback)

    def toggle_button_callback(self, checked: bool):
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

        if self.animation_is_working:
            self.toggle_animation.setDirection(
                QAbstractAnimation.Forward if checked else QAbstractAnimation.Backward
            )
            self.toggle_animation.start()
        else:
            # Hide children if we're collapsing
            for child in self.content_area.findChildren(QWidget):
                child.setVisible(checked)

    def setContentLayout(self, content_layout):
        self.content_area.setLayout(content_layout)

        if self.animation_is_working:
            collapsed_height = (
                self.sizeHint().height() - self.content_area.maximumHeight()
            )
            content_height = content_layout.sizeHint().height()

            for i in range(self.toggle_animation.animationCount()):
                spoiler_animation: QPropertyAnimation = (
                    self.toggle_animation.animationAt(i)
                )
                spoiler_animation.setDuration(self.animation_duration)
                spoiler_animation.setStartValue(collapsed_height)
                spoiler_animation.setEndValue(collapsed_height + content_height)

            content_animation: QPropertyAnimation = self.toggle_animation.animationAt(
                self.toggle_animation.animationCount() - 1
            )
            content_animation.setDuration(self.animation_duration)
            content_animation.setStartValue(0)
            content_animation.setEndValue(content_height)
