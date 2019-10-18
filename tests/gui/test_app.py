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

    # Remove an edge
    app.skeletonEdgesTable.selectRowItem(dict(source="b", destination="c"))
    app.commands.deleteEdge()

    assert len(app.state["skeleton"].edges) == 1

    # FIXME: for now we'll bypass the video adding gui
    app.labels.add_video(centered_pair_vid)
    app.labels.add_video(small_robot_mp4_vid)

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
    track = app.labels.tracks[0]

    assert inst_29_0.track == track

    # Set track on existing instance in another frame
    app.state["frame_idx"] = 27
    app.state["instance"] = inst_27_0

    app.commands.setInstanceTrack(new_track=track)
    assert inst_27_0.track == track

    # Delete all instances in track
    app.commands.deleteSelectedInstanceTrack()

    assert len(app.state["labeled_frame"].instances) == 0
    app.state["frame_idx"] = 29
    assert len(app.state["labeled_frame"].instances) == 1
