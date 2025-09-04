from sleap.io.convert import default_analysis_filename, main as sleap_convert
from sleap_io import Video, Labels
from sleap_io.model.instance import Instance
from sleap.sleap_io_adaptors.lf_labels_utils import (
    labels_add_video,
    labels_get,
    labels_add_instance,
)
from pathlib import PurePath, Path
import re
import pytest
import numpy as np


@pytest.mark.parametrize("format", ["analysis", "analysis.csv"])
def test_analysis_format(
    min_labels_slp: Labels,
    min_labels_slp_path: Labels,
    small_robot_mp4_vid: Video,
    format: str,
    tmpdir,
):
    labels = min_labels_slp
    slp_path = PurePath(min_labels_slp_path)
    tmpdir = PurePath(tmpdir)

    def generate_filenames(paths, format="analysis"):
        output_paths = [path for path in paths]

        # Generate filenames if user has not specified (enough) output filenames
        labels_path = str(slp_path)
        fn = re.sub("(\\.json(\\.zip)?|\\.h5|\\.slp)$", "", labels_path)
        fn = PurePath(fn)
        out_suffix = "csv" if "csv" in format else "h5"
        default_names = [
            default_analysis_filename(
                labels=labels,
                video=video,
                output_path=str(fn.parent),
                output_prefix=str(fn.stem),
                format_suffix=out_suffix,
            )
            for video in labels.videos[len(paths) :]
        ]

        output_paths.extend(default_names)
        return output_paths

    def assert_analysis_existance(output_paths: list, format="analysis"):
        output_paths = generate_filenames(output_paths, format)
        for video, path in zip(labels.videos, output_paths):
            video_exists = Path(path).exists()
            if len(labels_get(labels, video)) == 0:
                assert not video_exists
            else:
                assert video_exists

    def sleap_convert_assert(output_paths, slp_path, format="analysis"):
        output_args = ""
        for path in output_paths:
            output_args += f"-o {path} "
        args = f"--format {format} {output_args}{slp_path}".split()
        sleap_convert(args)
        assert_analysis_existance(output_paths, format)

    # No output specified
    output_paths = []
    sleap_convert_assert(output_paths, slp_path, format)

    # Specify output and retest
    output_paths = [str(tmpdir.with_name("prefix")), str(tmpdir.with_name("prefix2"))]
    sleap_convert_assert(output_paths, slp_path, format)

    # Add video and retest
    labels_add_video(labels, small_robot_mp4_vid)
    slp_path = tmpdir.with_name("new_slp.slp")
    labels.save(filename=str(slp_path))

    output_paths = [str(tmpdir.with_name("prefix"))]
    sleap_convert_assert(output_paths, slp_path, format)

    # Add labeled frame to video and retest
    labeled_frame = labels.find(video=labels.videos[1], frame_idx=0, return_new=True)[0]
    instance = Instance.from_numpy(
        np.zeros((len(labels.skeleton), 2)), skeleton=labels.skeleton
    )
    labels_add_instance(labels, frame=labeled_frame, instance=instance)
    labels.append(labeled_frame)
    slp_path = tmpdir.with_name("new_slp.slp")
    labels.save(filename=str(slp_path))

    output_paths = [str(tmpdir.with_name("prefix"))]
    sleap_convert_assert(output_paths, slp_path, format)


def test_sleap_format(
    min_labels_slp: Labels,
    min_labels_slp_path: Labels,
    tmpdir,
):
    def sleap_convert_assert(output_path, slp_path):
        args = f"-o {output_path} {slp_path}".split()
        sleap_convert(args)
        assert Path(output_path).exists()

    slp_path = PurePath(min_labels_slp_path)
    tmpdir = PurePath(tmpdir)

    output_path = Path(tmpdir, slp_path)
    # Create intermediate directories
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sleap_convert_assert(output_path, str(slp_path))


@pytest.mark.parametrize("suffix", [".slp", ".json"])
def test_auto_slp_h5_json_format(
    min_labels_slp: Labels,
    min_labels_slp_path: Labels,
    tmpdir,
    suffix,
):
    def sleap_convert_assert(output_path: Path, slp_path):
        args = f"--format {output_path.suffix[1:]} -o {output_path} {slp_path}".split()
        print(f"args = {args}")
        sleap_convert(args)
        assert Path(output_path).exists()

    labels = min_labels_slp
    slp_path = PurePath(min_labels_slp_path)
    new_slp_path = PurePath(tmpdir, slp_path.name)
    labels.save(str(new_slp_path))  # Convert to string for sleap-io

    output_path = Path(f"{new_slp_path}{suffix}")
    print(f"output_path = {output_path}")
    sleap_convert_assert(output_path, str(new_slp_path))  # Convert to string
