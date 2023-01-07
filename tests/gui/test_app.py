import pytest
from qtpy.QtWidgets import QApplication

from sleap.gui.app import MainWindow
from sleap.gui.commands import *


def test_app_workflow(
    qtbot, centered_pair_vid, small_robot_mp4_vid, min_tracks_2node_labels: Labels
):
    app = MainWindow(no_usage_data=True)

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

    # Prepare to check suggestion ui update upon video state change
    def assert_frame_chunk_suggestion_ui_updated(
        app, frame_to_spinbox, frame_from_spinbox
    ):
        assert frame_to_spinbox.maximum() == app.state["video"].num_frames
        assert frame_from_spinbox.maximum() == app.state["video"].num_frames

    method_layout = app.suggestions_form_widget.form_layout.fields["method"]
    frame_chunk_layout = method_layout.page_layouts["frame chunk"]
    frame_to_spinbox = frame_chunk_layout.fields["frame_to"]
    frame_from_spinbox = frame_chunk_layout.fields["frame_from"]

    # Verify the max of frame_chunk spinboxes is updated
    assert_frame_chunk_suggestion_ui_updated(app, frame_to_spinbox, frame_from_spinbox)

    # Activate video using table
    app.videosTable.selectRowItem(small_robot_mp4_vid)
    app.videosTable.activateSelected()

    assert app.state["video"] == small_robot_mp4_vid

    # Verify the max of frame_to in frame_chunk is updated
    assert_frame_chunk_suggestion_ui_updated(app, frame_to_spinbox, frame_from_spinbox)

    # Select remaining video
    app.videosTable.selectRowItem(small_robot_mp4_vid)
    assert app.state["selected_video"] == small_robot_mp4_vid

    # Verify the max of frame_to in frame_chunk is updated
    assert_frame_chunk_suggestion_ui_updated(app, frame_to_spinbox, frame_from_spinbox)

    # Delete selected video
    app.commands.removeVideo()

    assert len(app.labels.videos) == 1
    assert app.state["video"] == centered_pair_vid

    # Verify the max of frame_to in frame_chunk is updated
    assert_frame_chunk_suggestion_ui_updated(app, frame_to_spinbox, frame_from_spinbox)

    # Add instances
    app.state["frame_idx"] = 27
    app.commands.newInstance()
    app.commands.newInstance()

    assert len(app.state["labeled_frame"].instances) == 2

    inst_27_0 = app.state["labeled_frame"].instances[0]
    inst_27_1 = app.state["labeled_frame"].instances[1]

    # Move instance nodes
    app.commands.setPointLocations(
        inst_27_0, {"a": (15, 20), "b": (15, 40), "c": (40, 40)}
    )

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

    # Set up to test labeled frames data cache
    app.labels = min_tracks_2node_labels
    video = app.labels.video
    num_samples = 5
    frame_delta = video.num_frames // num_samples

    # Add suggestions
    app.labels.suggestions = VideoFrameSuggestions.suggest(
        labels=app.labels,
        params=dict(
            videos=app.labels.videos,
            method="sample",
            per_video=num_samples,
            sampling_method="stride",
        ),
    )
    assert len(app.labels.suggestions) == num_samples

    # The on_data_update function uses labeled frames cache
    app.on_data_update([UpdateTopic.suggestions])
    assert len(app.suggestionsTable.model().items) == num_samples
    assert f"{num_samples}/{num_samples}" in app.suggested_count_label.text()

    # Check that frames returned by labeled frames cache are correct
    prev_idx = -frame_delta
    for l_suggestion, st_suggestion in list(
        zip(app.labels.get_suggestions(), app.suggestionsTable.model().items)
    ):
        assert l_suggestion == st_suggestion["SuggestionFrame"]
        lf = app.labels.get(
            (l_suggestion.video, l_suggestion.frame_idx), use_cache=True
        )
        assert type(lf) == LabeledFrame
        assert lf.video == video
        assert lf.frame_idx == prev_idx + frame_delta
        prev_idx = l_suggestion.frame_idx

    # Add video, add frame suggestions, remove the video, verify the frame suggestions are also removed
    app.labels.add_video(small_robot_mp4_vid)
    app.on_data_update([UpdateTopic.video])

    assert len(app.labels.videos) == 2

    app.state["video"] = centered_pair_vid

    # Generate suggested frames in both videos
    app.labels.clear_suggestions()
    num_samples = 3
    app.labels.suggestions = VideoFrameSuggestions.suggest(
        labels=app.labels,
        params=dict(
            videos=app.labels.videos,
            method="sample",
            per_video=num_samples,
            sampling_method="random",
        ),
    )

    # Verify that suggestions contain frames from both videos
    video_source = []
    for sugg in app.labels.suggestions:
        if not (sugg.video in video_source):
            video_source.append(sugg.video)
    assert len(video_source) == 2

    # Remove video 1, keep video 0
    app.videosTable.selectRowItem(small_robot_mp4_vid)
    assert app.state["selected_video"] == small_robot_mp4_vid
    app.commands.removeVideo()
    assert len(app.labels.videos) == 1
    assert app.state["video"] == centered_pair_vid

    # Verify frame suggestions from video 1 are removed
    for sugg in app.labels.suggestions:
        assert sugg.video == app.labels.videos[0]


def test_app_new_window(qtbot):
    app = QApplication.instance()
    app.closeAllWindows()
    win = MainWindow(no_usage_data=True)

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

    new_win = MainWindow(no_usage_data=True)

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
    window: MainWindow = MainWindow(no_usage_data=True)
    window.commands.loadLabelsObject(centered_pair_predictions)
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
