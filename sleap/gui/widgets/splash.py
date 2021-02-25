from PySide2.QtCore import Qt
from PySide2.QtWidgets import QWidget, QLabel, QDialog
from PySide2.QtWidgets import QHBoxLayout


class SplashWidget(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QHBoxLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel("New"))
        layout.addWidget(QLabel("Open"))

        self.setModal(True)
        self.setSizeGripEnabled(False)


if __name__ == "__main__":
    from PySide2.QtWidgets import QApplication

    app = QApplication([])
    window = SplashWidget()
    window.show()
    app.exec_()
