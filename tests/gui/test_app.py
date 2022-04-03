import pytest
from PySide2.QtWidgets import QApplication

from sleap.gui.app import MainWindow
from sleap.gui.commands import *


def test_app_workflow(qtbot, centered_pair_vid, small_robot_mp4_vid):
    app = MainWindow()

    # Add nodes
    app.commands.newNode()
    app.commands.newNode()
    app.commands.newNode()

    assert len(app.state["skeleton"].nodes) == 3

    # Name nodes
    app.commands.setNodeName(
        skeleton=app.state["skeleton"], node=app.state["skeleton"].nodes[0], name="a"
    )
    app.commands.setNodeName(
        skeleton=app.state["skeleton"], node=app.state["skeleton"].nodes[1], name="b"
    )
    app.commands.setNodeName(
        skeleton=app.state["skeleton"], node=app.state["skeleton"].nodes[2], name="c"
    )

    assert app.state["skeleton"].nodes[0].name == "a"
    assert app.state["skeleton"].nodes[1].name == "b"
    assert app.state["skeleton"].nodes[2].name == "c"

    # Select and delete the third node
    app.skeletonNodesTable.selectRowItem(app.state["skeleton"].nodes[2])
    app.commands.deleteNode()

    assert len(app.state["skeleton"].nodes) == 2

    # Add back the third node
    app.commands.newNode()
    app.commands.setNodeName(
        skeleton=app.state["skeleton"], node=app.state["skeleton"].nodes[2], name="c"
    )

    assert len(app.state["skeleton"].nodes) == 3
    assert app.state["skeleton"].nodes[2].name == "c"

    # Add edges
    app.commands.newEdge("a", "b")
    app.commands.newEdge("b", "c")

    assert len(app.state["skeleton"].edges) == 2

    # Add and remove symmetry
    app.commands.setNodeSymmetry(app.state["skeleton"], "b", "c")
    assert app.state["skeleton"].get_symmetry_name("c") == "b"
    app.commands.setNodeSymmetry(app.state["skeleton"], "b", "")
    assert app.state["skeleton"].get_symmetry("c") is None

    # Remove an edge
    app.skeletonEdgesTable.selectRowItem(dict(source="b", destination="c"))
    app.commands.deleteEdge()

    assert len(app.state["skeleton"].edges) == 1

    # FIXME: for now we'll bypass the video adding gui
    app.labels.add_video(centered_pair_vid)
    app.labels.add_video(small_robot_mp4_vid)
    app.on_data_update([UpdateTopic.video])

    assert len(app.labels.videos) == 2

    app.state["video"] = centered_pair_vid

    # Activate video using table
    app.videosTable.selectRowItem(small_robot_mp4_vid)
    app.videosTable.activateSelected()

    assert app.state["video"] == small_robot_mp4_vid

    # Select remaining video
    app.videosTable.selectRowItem(small_robot_mp4_vid)
    assert app.state["selected_video"] == small_robot_mp4_vid

    # Delete selected video
    app.commands.removeVideo()

    assert len(app.labels.videos) == 1
    assert app.state["video"] == centered_pair_vid

    # Add instances
    app.state["frame_idx"] = 27
    app.commands.newInstance()
    app.commands.newInstance()

    assert len(app.state["labeled_frame"].instances) == 2

    inst_27_0 = app.state["labeled_frame"].instances[0]
    inst_27_1 = app.state["labeled_frame"].instances[1]

    # Move instance nodes
    app.commands.setPointLocations(inst_27_0, {"a": (15, 20)})

    assert inst_27_0["a"].x == 15
    assert inst_27_0["a"].y == 20

    # Toggle node visibility
    assert inst_27_0["b"].visible
    app.commands.setInstancePointVisibility(inst_27_0, "b", False)
    assert not inst_27_0["b"].visible

    # Select and delete instance
    app.state["instance"] = inst_27_1
    app.commands.deleteSelectedInstance()

    assert len(app.state["labeled_frame"].instances) == 1
    assert app.state["labeled_frame"].instances == [inst_27_0]

    # Add instances on another frame
    app.state["frame_idx"] = 29
    app.commands.newInstance()
    app.commands.newInstance()

    assert len(app.state["labeled_frame"].instances) == 2

    inst_29_0 = app.state["labeled_frame"].instances[0]
    inst_29_1 = app.state["labeled_frame"].instances[1]

    app.state["instance"] = inst_29_0

    # Make and set track
    app.commands.addTrack()

    assert len(app.labels.tracks) == 1
    track_a = app.labels.tracks[0]

    assert inst_29_0.track == track_a

    # Set track name
    app.commands.setTrackName(track_a, "Track A")
    assert track_a.name == "Track A"

    # Set track on existing instance in another frame
    app.state["frame_idx"] = 27
    app.state["instance"] = inst_27_0

    app.commands.setInstanceTrack(new_track=track_a)
    assert inst_27_0.track == track_a

    # Delete all instances in track
    app.commands.deleteSelectedInstanceTrack()

    assert len(app.state["labeled_frame"].instances) == 0
    app.state["frame_idx"] = 29
    assert len(app.state["labeled_frame"].instances) == 1

    # Set up new frame/tracks for transposing instances
    app.state["frame_idx"] = 31

    app.commands.newInstance()
    app.commands.newInstance()

    inst_31_0 = app.state["labeled_frame"].instances[0]
    inst_31_1 = app.state["labeled_frame"].instances[1]

    app.commands.addTrack()
    assert len(app.labels.tracks) == 2
    track_b = app.labels.tracks[1]
    app.commands.setTrackName(track_b, "Track B")
    assert track_b.name == "Track B"

    app.state["instance"] = inst_31_0
    app.commands.setInstanceTrack(track_a)
    assert inst_31_0.track == track_a

    app.state["instance"] = inst_31_1
    app.commands.setInstanceTrack(track_b)
    assert inst_31_1.track == track_b

    # Here we do the actual transpose for the pair of instances we just made
    app.commands.transposeInstance()
    assert inst_31_0.track == track_b
    assert inst_31_1.track == track_a

    # Try transposing back
    app.commands.transposeInstance()
    assert inst_31_0.track == track_a
    assert inst_31_1.track == track_b

    # Add a third instance on a new track

    app.commands.newInstance()
    inst_31_2 = app.state["labeled_frame"].instances[2]

    app.state["instance"] = None  # so we don't set track when creating
    app.commands.addTrack()
    assert len(app.labels.tracks) == 3
    track_c = app.labels.tracks[2]

    app.state["instance"] = inst_31_2
    app.commands.setInstanceTrack(track_c)
    assert inst_31_2.track == track_c

    # Try transposing instance 2 and instance 0
    app.commands.transposeInstance()
    app.state["instance"] = inst_31_0
    assert inst_31_0.track == track_c
    assert inst_31_2.track == track_a
    assert inst_31_1.track == track_b


def test_app_new_window(qtbot):
    app = QApplication.instance()
    app.closeAllWindows()
    win = MainWindow()

    assert win.commands.has_any_changes == False
    assert win.state["project_loaded"] == False

    start_wins = sum(
        (1 for widget in app.topLevelWidgets() if isinstance(widget, MainWindow))
    )

    # there's no loaded project, so this should load into same window
    OpenProject.do_action(
        win.commands, dict(filename="tests/data/json_format_v1/centered_pair.json")
    )

    assert win.state["project_loaded"] == True
    wins = sum(
        (1 for widget in app.topLevelWidgets() if isinstance(widget, MainWindow))
    )
    assert wins == start_wins

    # this time it will open in new window, so current window shouldn't change
    OpenProject.do_action(
        win.commands, dict(filename="tests/data/slp_hdf5/minimal_instance.slp")
    )

    assert win.state["filename"] == "tests/data/json_format_v1/centered_pair.json"

    wins = sum(
        (1 for widget in app.topLevelWidgets() if isinstance(widget, MainWindow))
    )
    assert wins == (start_wins + 1)

    new_win = MainWindow()

    wins = sum(
        (1 for widget in app.topLevelWidgets() if isinstance(widget, MainWindow))
    )
    assert wins == (start_wins + 2)

    # add something so this is no longer empty project
    new_win.commands.newNode()

    # should open in new window
    OpenProject.do_action(
        win.commands, dict(filename="tests/data/json_format_v1/centered_pair.json")
    )

    wins = sum(
        (1 for widget in app.topLevelWidgets() if isinstance(widget, MainWindow))
    )
    assert wins == (start_wins + 3)

    app.closeAllWindows()


@pytest.mark.skipif(
    sys.platform.startswith("li"), reason="qtbot.waitActive times out on ubuntu"
)
def test_menu_actions(qtbot, centered_pair_predictions: Labels):
    def verify_visibility(expected_visibility: bool = True):
        """Verify the visibility status of all instances in video player.

        Args:
            expected_visibility: Expected visibility of instance. Defaults to True.
        """
        for inst in vp.instances:
            assert inst.isVisible() == expected_visibility
        for inst in vp.predicted_instances:
            assert inst.isVisible() == expected_visibility

    def toggle_and_verify_visibility(expected_visibility: bool = True):
        """Toggle the visibility of all instances within video player, then verify
        expected visibility of all instances.

        Args:
            expected_visibility: Expected visibility of instances. Defaults to True.
        """
        qtbot.keyClick(window.menuBar(), window.shortcuts["show instances"].toString())
        verify_visibility(expected_visibility)

    # Test hide instances menu action (and its effect on instance color)

    # Instantiate the window and load labels
    window: MainWindow = MainWindow()
    window.loadLabelsObject(centered_pair_predictions)
    # TODO: window does not seem to show as expected on ubuntu
    with qtbot.waitActive(window, timeout=2000):
        window.showNormal()
    vp = window.player

    # Enable distinct colors
    window.state["color predicted"] = True

    # Read colors for each instance in view
    # TODO: revisit with LabeledFrame.unused_predictions() & instances_to_show()
    visible_instances = window.state["labeled_frame"].instances_to_show
    color_of_instances = {}
    for inst in visible_instances:
        item_color = window.color_manager.get_item_color(inst)
        color_of_instances[inst] = item_color

    # Turn predicted instance into user labeled instance
    predicted_instance = vp.view.predicted_instances[0].instance
    window.commands.newInstance(copy_instance=predicted_instance, mark_complete=False)

    # Ensure colors of instances do not change
    for inst in visible_instances:
        item_color = window.color_manager.get_item_color(inst)
        assert item_color == color_of_instances[inst]

    # Ensure instances are visible - should be the visible by default
    verify_visibility(True)

    # Toggle instance visibility with shortcut, hiding instances
    toggle_and_verify_visibility(False)

    # Toggle instance visibility with shortcut, showing instances
    toggle_and_verify_visibility(True)
