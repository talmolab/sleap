from sleap.io.dataset import Labels
from sleap.io.format import read, dispatch, adaptor, text, genericjson, hdf5, filehandle
from sleap.io.format.adaptor import SleapObjectType
from sleap.io.format.alphatracker import AlphaTrackerAdaptor
from sleap.gui.commands import ImportAlphaTracker
from sleap.gui.app import MainWindow
from sleap.gui.state import GuiState

import pytest
import os
from pathlib import Path
import numpy as np
from numpy.testing import assert_array_equal


def test_text_adaptor(tmpdir):
    disp = dispatch.Dispatch()
    disp.register(text.TextAdaptor())

    filename = os.path.join(tmpdir, "textfile.txt")
    some_text = "some text to save in a file"

    disp.write(filename, some_text)

    read_text = disp.read(filename)

    assert some_text == read_text


def test_json_adaptor(tmpdir):
    disp = dispatch.Dispatch()
    disp.register(genericjson.GenericJsonAdaptor())

    filename = os.path.join(tmpdir, "jsonfile.json")
    d = dict(foo=123, bar="zip")

    disp.write(filename, d)

    read_dict = disp.read(filename)

    assert d == read_dict

    assert disp.open(filename).is_json


def test_invalid_json(tmpdir):
    # Write an "invalid" json file
    filename = os.path.join(tmpdir, "textfile.json")
    some_text = "some text to save in a file"
    with open(filename, "w") as f:
        f.write(some_text)

    disp = dispatch.Dispatch()
    disp.register(genericjson.GenericJsonAdaptor())

    assert not disp.open(filename).is_json

    with pytest.raises(TypeError):
        disp.read(filename)


def test_no_matching_adaptor():
    disp = dispatch.Dispatch()

    with pytest.raises(TypeError):
        disp.write("foo.txt", "foo")

    err = disp.write_safely("foo.txt", "foo")

    assert err is not None


def test_failed_read():
    disp = dispatch.Dispatch()
    disp.register(text.TextAdaptor())

    # Attempt to read hdf5 using text adaptor
    hdf5_filename = "tests/data/hdf5_format_v1/training.scale=0.50,sigma=10.h5"
    x, err = disp.read_safely(hdf5_filename)

    # There should be an error
    assert err is not None


def test_missing_file():
    disp = dispatch.Dispatch()
    disp.register(text.TextAdaptor())

    with pytest.raises(FileNotFoundError):
        disp.read("missing_file.txt")


def test_hdf5_v1(tmpdir, centered_pair_predictions_hdf5_path):
    filename = centered_pair_predictions_hdf5_path
    disp = dispatch.Dispatch.make_dispatcher(adaptor.SleapObjectType.labels)

    # Make sure reading works
    x = disp.read(filename)
    assert len(x.labeled_frames) == 1100

    # Make sure writing works
    filename = os.path.join(tmpdir, "test.h5")
    disp.write(filename, x)

    # Make sure we can read the file we just wrote
    y = disp.read(filename)
    assert len(y.labeled_frames) == 1100


def test_hdf5_v1_filehandle(centered_pair_predictions_hdf5_path):

    filename = centered_pair_predictions_hdf5_path

    labels = hdf5.LabelsV1Adaptor.read_headers(filehandle.FileHandle(filename))

    assert len(labels.videos) == 1
    assert (
        labels.videos[0].backend.filename
        == "tests/data/json_format_v1/centered_pair_low_quality.mp4"
    )


def test_analysis_hdf5(tmpdir, centered_pair_predictions):
    from sleap.info.write_tracking_h5 import main as write_analysis

    filename = os.path.join(tmpdir, "analysis.h5")
    video = centered_pair_predictions.videos[0]

    write_analysis(centered_pair_predictions, output_path=filename, all_frames=True)

    labels = read(
        filename,
        for_object="labels",
        as_format="analysis",
        video=video,
    )

    assert len(labels) == len(centered_pair_predictions)
    assert len(labels.tracks) == len(centered_pair_predictions.tracks)
    assert len(labels.all_instances) == len(centered_pair_predictions.all_instances)


def test_json_v1(tmpdir, centered_pair_labels):
    filename = os.path.join(tmpdir, "test.json")
    disp = dispatch.Dispatch.make_dispatcher(adaptor.SleapObjectType.labels)

    disp.write(filename, centered_pair_labels)

    # Make sure we can read the file we just wrote
    y = disp.read(filename)
    assert len(y.labeled_frames) == len(centered_pair_labels.labeled_frames)


def test_matching_adaptor(centered_pair_predictions_hdf5_path):
    from sleap.io.format import read

    read(
        centered_pair_predictions_hdf5_path,
        for_object="labels",
        as_format="*",
    )

    read(
        "tests/data/json_format_v1/centered_pair.json",
        for_object="labels",
        as_format="*",
    )


@pytest.mark.parametrize(
    "test_data",
    [
        "tests/data/dlc/madlc_testdata.csv",
        "tests/data/dlc/madlc_testdata_v2.csv",
    ],
)
def test_madlc(test_data):
    labels = read(
        test_data,
        for_object="labels",
        as_format="deeplabcut",
    )

    assert labels.skeleton.node_names == ["A", "B", "C"]
    assert len(labels.videos) == 1
    assert len(labels.video.filenames) == 4
    assert labels.videos[0].filenames[0].endswith("img000.png")
    assert labels.videos[0].filenames[1].endswith("img001.png")
    assert labels.videos[0].filenames[2].endswith("img002.png")
    assert labels.videos[0].filenames[3].endswith("img003.png")

    # Assert frames without any coor are not labeled
    assert len(labels) == 3

    # Assert number of instances per frame is correct
    assert len(labels[0]) == 2
    assert len(labels[1]) == 2
    assert len(labels[2]) == 1

    assert_array_equal(labels[0][0].numpy(), [[0, 1], [2, 3], [4, 5]])
    assert_array_equal(labels[0][1].numpy(), [[6, 7], [8, 9], [10, 11]])
    assert_array_equal(labels[1][0].numpy(), [[12, 13], [np.nan, np.nan], [15, 16]])
    assert_array_equal(labels[1][1].numpy(), [[17, 18], [np.nan, np.nan], [20, 21]])
    assert_array_equal(labels[2][0].numpy(), [[22, 23], [24, 25], [26, 27]])
    assert labels[2].frame_idx == 3


@pytest.mark.parametrize(
    "test_data",
    ["tests/data/dlc/dlc_testdata.csv", "tests/data/dlc/dlc_testdata_v2.csv"],
)
def test_sadlc(test_data):
    labels = read(
        test_data,
        for_object="labels",
        as_format="deeplabcut",
    )

    assert labels.skeleton.node_names == ["A", "B", "C"]
    assert len(labels.videos) == 1
    assert len(labels.video.filenames) == 4
    assert labels.videos[0].filenames[0].endswith("img000.png")
    assert labels.videos[0].filenames[1].endswith("img001.png")
    assert labels.videos[0].filenames[2].endswith("img002.png")
    assert labels.videos[0].filenames[3].endswith("img003.png")

    # Assert frames without any coor are not labeled
    assert len(labels) == 3

    # Assert number of instances per frame is correct
    assert len(labels[0]) == 1
    assert len(labels[1]) == 1
    assert len(labels[2]) == 1

    assert_array_equal(labels[0][0].numpy(), [[0, 1], [2, 3], [4, 5]])
    assert_array_equal(labels[1][0].numpy(), [[12, 13], [np.nan, np.nan], [15, 16]])
    assert_array_equal(labels[2][0].numpy(), [[22, 23], [24, 25], [26, 27]])
    assert labels[2].frame_idx == 3


def test_alphatracker(qtbot):

    # Checks on properties
    at_adaptor = AlphaTrackerAdaptor()
    assert at_adaptor.handles == SleapObjectType.labels
    assert at_adaptor.default_ext == "json"
    assert at_adaptor.name == "AlphaTracker Dataset JSON"
    assert at_adaptor.can_write_filename("cannot_write_this.txt") == False
    assert at_adaptor.does_read() == True
    assert at_adaptor.does_write() == False
    with pytest.raises(NotImplementedError):
        at_adaptor.write("file_that_will_not_be_written", Labels())
    assert at_adaptor.formatted_ext_options == f"AlphaTracker Dataset JSON (json)"

    # Begin checks on functionality

    filename = "tests/data/alphatracker/at_testdata.json"
    disp = dispatch.Dispatch()
    disp.register(AlphaTrackerAdaptor)

    # Ensure reading works
    labels: Labels = disp.read(filename)
    lfs = labels.labeled_frames

    # Ensure video and frames are read correctly
    assert len(lfs) == 4
    for file_idx, file in enumerate(labels.video.backend.filenames):
        f = Path(file)
        assert f.stem == f"img00{file_idx}"

    # Ensure nodes are read correctly
    nodes = labels.skeleton.node_names
    assert nodes[0] == "1"
    assert nodes[1] == "2"
    assert nodes[2] == "3"

    # Ensure points are read correctly
    for lf_idx, lf in enumerate(lfs):
        assert len(lf.instances) == 2
        for inst_idx, inst in enumerate(lf.instances):
            for point_idx, point in enumerate(inst.points):
                assert point[0] == ((lf_idx + 1) * (inst_idx + 1))
                assert point[1] == (point_idx + 2)

    # Run through GUI display

    app = MainWindow()
    app.state = GuiState()
    app.state["filename"] = filename

    # Only test do_action because ask method opens FileDialog
    ImportAlphaTracker().do_action(context=app.commands, params=app.state)


def test_tracking_scores(tmpdir, centered_pair_predictions_slp_path):

    # test reading
    filename = centered_pair_predictions_slp_path

    fh = filehandle.FileHandle(filename)

    assert fh.format_id is not None

    labels = hdf5.LabelsV1Adaptor.read(fh)

    for instance in labels.instances():
        assert hasattr(instance, "tracking_score")

    # test writing
    filename = os.path.join(tmpdir, "test.slp")
    labels.save(filename)

    labels = hdf5.LabelsV1Adaptor.read(filehandle.FileHandle(filename))

    for instance in labels.instances():
        assert hasattr(instance, "tracking_score")
