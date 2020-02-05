from sleap.io.format import dispatch, adaptor, text, genericjson, hdf5, filehandle
import pytest
import os


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


def test_hdf5_v1(tmpdir):
    filename = "tests/data/hdf5_format_v1/centered_pair_predictions.h5"
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


def test_hdf5_v1():
    filename = "tests/data/hdf5_format_v1/centered_pair_predictions.h5"

    labels = hdf5.LabelsV1Adaptor.read_headers(filehandle.FileHandle(filename))

    assert len(labels.videos) == 1
    assert (
        labels.videos[0].backend.filename
        == "tests/data/json_format_v1/centered_pair_low_quality.mp4"
    )


def test_json_v1(tmpdir, centered_pair_labels):
    filename = os.path.join(tmpdir, "test.json")
    disp = dispatch.Dispatch.make_dispatcher(adaptor.SleapObjectType.labels)

    disp.write(filename, centered_pair_labels)

    # Make sure we can read the file we just wrote
    y = disp.read(filename)
    assert len(y.labeled_frames) == len(centered_pair_labels.labeled_frames)
