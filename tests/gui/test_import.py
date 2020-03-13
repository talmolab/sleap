from sleap.gui.dialogs.importvideos import ImportParamDialog

import PySide2.QtCore as QtCore


def test_gui_import(qtbot):
    file_names = [
        "tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5",
        "tests/data/videos/small_robot.mp4",
    ]

    importer = ImportParamDialog(file_names)
    importer.show()

    qtbot.addWidget(importer)

    data = importer.get_data()
    assert len(data) == 2
    assert len(data[0]["params"]) > 1

    for import_item in importer.import_widgets:
        btn = import_item.enabled_checkbox_widget
        with qtbot.waitSignal(btn.stateChanged, timeout=10):
            qtbot.mouseClick(btn, QtCore.Qt.LeftButton)
            assert not import_item.is_enabled()

    assert len(importer.get_data()) == 0

    for import_item in importer.import_widgets:
        btn = import_item.enabled_checkbox_widget
        with qtbot.waitSignal(btn.stateChanged, timeout=10):
            qtbot.mouseClick(btn, QtCore.Qt.LeftButton)
            assert import_item.is_enabled()

    assert len(importer.get_data()) == 2


def test_video_import_detect_grayscale():
    importer = ImportParamDialog(
        [
            "tests/data/videos/centered_pair_small.mp4",
            "tests/data/videos/small_robot.mp4",
        ]
    )
    data = importer.get_data()

    assert data[0]["params"]["grayscale"] == True
    assert data[1]["params"]["grayscale"] == False


def test_video_import_detect_h5_shape():
    importer = ImportParamDialog(
        ["tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5"]
    )
    data = importer.get_data()

    assert data[0]["params"]["input_format"] == "channels_first"

    assert importer.import_widgets[0].video is not None
    assert importer.import_widgets[0].video.num_frames == 42
    assert importer.import_widgets[0].video.height == 512
    assert importer.import_widgets[0].video.width == 512
    assert importer.import_widgets[0].video.channels == 1
