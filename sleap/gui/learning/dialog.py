"""
Stub dialog for training/inference - neural network functionality has been removed.
"""
from qtpy import QtWidgets, QtCore

class LearningDialog(QtWidgets.QDialog):
    """Stub dialog - neural network functionality has been removed."""
    
    _handle_learning_finished = QtCore.Signal(int)
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.setWindowTitle("Training/Inference Unavailable")
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel(
            "Neural network functionality has been removed from this build.\n\n"
            "Training and inference features are not available."
        )
        layout.addWidget(label)
        button = QtWidgets.QPushButton("Close")
        button.clicked.connect(self.close)
        layout.addWidget(button)
        self.setLayout(layout)
    
    def run(self):
        """Stub method for compatibility."""
        pass

class TrainingEditorWidget(QtWidgets.QWidget):
    """Stub widget for compatibility."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel("Neural network functionality has been removed.")
        layout.addWidget(label)
        self.setLayout(layout)