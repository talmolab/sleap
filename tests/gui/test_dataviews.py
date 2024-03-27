from sleap.gui.dataviews import *


def test_skeleton_nodes(qtbot, centered_pair_predictions):
    table = GenericTableView(
        model=SkeletonNodesTableModel(items=centered_pair_predictions.skeletons[0])
    )

    table.selectRow(1)
    assert table.model().data(table.currentIndex()) == "neck"

    table = GenericTableView(
        model=SkeletonEdgesTableModel(items=centered_pair_predictions.skeletons[0])
    )
    table.selectRow(2)
    assert table.model().data(table.currentIndex()) == "thorax"

    table = GenericTableView(
        row_name="video",
        model=VideosTableModel(items=centered_pair_predictions.videos),
        multiple_selection=True,
    )
    table.selectRow(0)
    assert (
        table.model().data(table.currentIndex()).find("centered_pair_low_quality.mp4")
        > -1
    )
    assert table.state["selected_video"] == centered_pair_predictions.videos[0]

    table = GenericTableView(
        row_name="instance",
        name_prefix="",
        model=LabeledFrameTableModel(items=centered_pair_predictions.labels[13]),
    )
    table.selectRow(1)
    assert table.model().data(table.currentIndex()) == "21/24"


def test_table_sort(qtbot, centered_pair_predictions):
    table = GenericTableView(
        row_name="instance",
        is_sortable=True,
        name_prefix="",
        model=LabeledFrameTableModel(items=centered_pair_predictions.labels[13]),
    )
    table.selectRow(1)
    assert table.model().data(table.currentIndex()) == "21/24"

    inst = centered_pair_predictions.labels[13].instances[0]
    table.selectRow(0)
    assert table.getSelectedRowItem().score == inst.score

    inst = centered_pair_predictions.labels[13].instances[1]
    table.selectRow(1)
    assert table.getSelectedRowItem().score == inst.score

    # Now sort the instances and make sure things are different
    table.model().sort(2)  # "score" column, should reverse initial order
    table.selectRow(1)
    assert table.model().data(table.currentIndex()) == "24/24"

    # Instance 0 should be in row 1
    inst = centered_pair_predictions.labels[13].instances[0]
    table.selectRow(1)
    assert table.getSelectedRowItem().score == inst.score

    # Instance 1 should be in row 0
    inst = centered_pair_predictions.labels[13].instances[1]
    table.selectRow(0)
    assert table.getSelectedRowItem().score == inst.score


def test_sessions_table(qtbot, min_session_session, hdf5_vid):
    sessions = []
    sessions.append(min_session_session)
    table = GenericTableView(
        row_name="session",
        is_sortable=True,
        name_prefix="",
        model=SessionsTableModel(items=sessions),
    )
    table.selectRow(0)
    assert len(table.getSelectedRowItem().videos) == 0
    assert len(table.getSelectedRowItem().camera_cluster.cameras) == 8
    assert len(table.getSelectedRowItem().camera_cluster.sessions) == 1

    video = hdf5_vid
    min_session_session.add_video(
        video,
        table.getSelectedRowItem().camera_cluster.cameras[0],
    )

    # Verify that modification of the recording session is reflected in the recording session stored in the table
    assert len(table.getSelectedRowItem().videos) == 1

    min_session_session.remove_video(video)
    assert len(table.getSelectedRowItem().videos) == 0


def test_table_sort_string(qtbot):
    table_model = GenericTableModel(
        items=[dict(a=1, b=2), dict(a=2, b="")], properties=["a", "b"]
    )

    table = GenericTableView(is_sortable=True, model=table_model)

    # Make sure we can sort with both numbers and strings (i.e., "")
    table.model().sort(0)
    table.model().sort(1)


def test_camera_table(qtbot, multiview_min_session_labels):

    session = multiview_min_session_labels.sessions[0]
    camcorders = session.camera_cluster.cameras

    table_model = CamerasTableModel(items=session)
    num_rows = table_model.rowCount()

    assert table_model.columnCount() == 2
    assert num_rows == len(camcorders)

    table = GenericTableView(
        row_name="camera",
        model=table_model,
    )

    # Test if all comcorders are presented in the correct row
    for i in range(num_rows):
        table.selectRow(i)

        # Check first column
        assert table.getSelectedRowItem() == camcorders[i]
        assert table.model().data(table.currentIndex()) == camcorders[i].name

        # Check second column
        index = table.model().index(i, 1)
        linked_video_filename = camcorders[i].get_video(session).filename
        assert table.model().data(index) == linked_video_filename

    # Test if a comcorder change is reflected
    idxs_to_remove = [1, 2, 7]
    for idx in idxs_to_remove:
        multiview_min_session_labels.sessions[0].remove_video(
            camcorders[idx].get_video(multiview_min_session_labels.sessions[0])
        )
    table.model().items = session

    for i in range(num_rows):
        table.selectRow(i)

        # Check first column
        assert table.getSelectedRowItem() == camcorders[i]
        assert table.model().data(table.currentIndex()) == camcorders[i].name

        # Check second column
        index = table.model().index(i, 1)
        linked_video = camcorders[i].get_video(session)
        if i in idxs_to_remove:
            assert table.model().data(index) == ""
        else:
            linked_video_filename = linked_video.filename
            assert table.model().data(index) == linked_video_filename

def test_unlinked_videos_table(qtbot, multiview_min_session_labels, hdf5_vid):
    # Test if the unlinked videos table is loaded correctly
    
    # Test if the linked videos are unlinked correctly
    
    # Test if the linked videos are linked correctly