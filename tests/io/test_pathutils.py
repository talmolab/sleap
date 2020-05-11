from sleap.io.pathutils import *


def test_find_missing():
    missing = list_file_missing(
        ["tests/data/videos/small_robot.mp4", "tests/data/videos/does_not_exist.mp4"]
    )
    assert missing == [False, True]


def test_prefix_change_no_filenames():
    filenames_prefix_change([], "foo", "bar")


def test_filenames_prefix_change():
    # First path is fixable, second is not (since file doesn't exist anywhere)
    filenames = ["foo/small_robot.mp4", "foo/does_not_exist.mp4"]
    filenames_prefix_change(filenames, "foo", "tests/data/videos")

    # Check that first path was correctly changed and second was left alone
    assert filenames[0] == "tests/data/videos/small_robot.mp4"
    assert filenames[1] == "foo/does_not_exist.mp4"


def test_partial_missing_list():
    # First path is fixable, second is not (since file doesn't exist anywhere)
    filenames = ["foo/small_robot.mp4", "foo/does_not_exist.mp4"]
    filenames_prefix_change(filenames, "foo", "tests/data/videos", missing=[True])

    # Check that first path was correctly changed and second was left alone
    assert filenames[0] == "tests/data/videos/small_robot.mp4"
    assert filenames[1] == "foo/does_not_exist.mp4"


def test_filename_fix_from_saved_prefix():
    path_prefix_conversions = dict(foo="tests/data/videos")

    # First path is fixable, second is not (since file doesn't exist anywhere)
    filenames = ["foo/small_robot.mp4", "foo/does_not_exist.mp4"]
    fix_paths_with_saved_prefix(
        filenames, path_prefix_conversions=path_prefix_conversions
    )

    # Check that first path was correctly changed and second was left alone
    assert filenames[0] == "tests/data/videos/small_robot.mp4"
    assert filenames[1] == "foo/does_not_exist.mp4"


def test_changed_subpath():
    # Check partial overlap, with different types of paths
    a, b = find_changed_subpath(
        "m:\\one\\two\\three.mp4", "/Volumes/fileset/two/three.mp4"
    )
    assert a == "m:\\one"
    assert b == "/Volumes/fileset"

    # Check no overlap
    a, b = find_changed_subpath(
        "m:\\one\\two\\three.mp4", "/Volumes/fileset/two/three.mp4z"
    )
    assert a == "m:\\one\\two\\three.mp4"
    assert b == "/Volumes/fileset/two/three.mp4z"
