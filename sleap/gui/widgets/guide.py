"""This module defines the guide widget for navigating the core workflow."""

from PySide2.QtCore import Qt, Signal
from PySide2.QtWidgets import QWidget, QLabel
from PySide2.QtWidgets import QVBoxLayout


CHECKMARK = "&#x2714;"


class StepLabel(QLabel):
    completed = Signal(QWidget, bool)

    def __init__(self, step: int, name: str, label: str, *args, **kwargs):
        super(StepLabel, self).__init__(*args, **kwargs)
        self.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.setWordWrap(True)
        self.setTextFormat(Qt.RichText)
        self._complete = False
        self._step = step
        self._name = name
        self._label = label
        self.update()

    def update(self):
        if self.complete():
            self.setText(
                f'<span style="color:green">{CHECKMARK} {self.step()}. '
                f'<a href="{self.name()}"><span style="color:green">{self.label()}'
                "</span></a></span>"
            )
        else:
            self.setText(f'  {self.step()}. <a href="{self.name()}">{self.label()}</a>')

    def step(self) -> int:
        return self._step

    def setStep(self, step: int):
        self._step = step
        self.update()

    def name(self) -> str:
        return self._name

    def setName(self, name: str):
        self._name = name
        self.update()

    def label(self) -> str:
        return self._label

    def setLabel(self, label: str):
        self._label = label
        self.update()

    def complete(self) -> bool:
        return self._complete

    def setComplete(self, complete: bool):
        self._complete = complete
        self.update()
        self.completed.emit(self, complete)


class GuideWidget(QWidget):
    stepClicked = Signal(QWidget, str)

    def __init__(self, *args, **kwargs):
        super(GuideWidget, self).__init__(*args, **kwargs)
        self.steps = []
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

    def addStep(self, name: str, label: str):
        label = StepLabel(step=len(self.steps) + 1, name=name, label=label)
        label.linkActivated.connect(lambda x: self.stepClicked.emit(self, name))
        self.steps.append(label)
        self._layout.addWidget(label)

    def setComplete(self, step_name: str, complete: bool):
        for i, step in enumerate(self.steps):
            if step.name() == step_name:
                step.setComplete(complete)
