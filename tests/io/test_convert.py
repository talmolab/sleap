from sleap.io.convert import default_analysis_filename, main as sleap_convert
from sleap.io.dataset import Labels
from sleap.io.video import Video
from sleap.instance import Instance

from pathlib import PurePath, Path
import re
import pytest


def test_analysis_format(
    min_labels_slp: Labels,
    min_labels_slp_path: Labels,
    small_robot_mp4_vid: Video,
    tmpdir,
):
    labels = min_labels_slp
    slp_path = PurePath(min_labels_slp_path)
    tmpdir = PurePath(tmpdir)

    def generate_filenames(paths):
        output_paths = [path for path in paths]

        # Generate filenames if user has not specified (enough) output filenames
        labels_path = str(slp_path)
        fn = re.sub("(\\.json(\\.zip)?|\\.h5|\\.slp)$", "", labels_path)
        fn = PurePath(fn)
        default_names = [
            default_analysis_filename(
                labels=labels,
                video=video,
                output_path=str(fn.parent),
                output_prefix=str(fn.stem),
            )
            for video in labels.videos[len(paths) :]
        ]

        output_paths.extend(default_names)
        return output_paths

    def assert_analysis_existance(output_paths: list):
        output_paths = generate_filenames(output_paths)
        for video, path in zip(labels.videos, output_paths):
            video_exists = Path(path).exists()
            if len(labels.get(video)) == 0:
                assert not video_exists
            else:
                assert video_exists

    def sleap_convert_assert(output_paths, slp_path):
        output_args = ""
        for path in output_paths:
            output_args += f"-o {path} "
        args = f"--format analysis {output_args}{slp_path}".split()
        sleap_convert(args)
        assert_analysis_existance(output_paths)

    # No output specified
    output_paths = []
    sleap_convert_assert(output_paths, slp_path)

    # Specify output and retest
    output_paths = [str(tmpdir.with_name("prefix")), str(tmpdir.with_name("prefix2"))]
    sleap_convert_assert(output_paths, slp_path)

    # Add video and retest
    labels.add_video(small_robot_mp4_vid)
    slp_path = tmpdir.with_name("new_slp.slp")
    labels.save(filename=slp_path)

    output_paths = [str(tmpdir.with_name("prefix"))]
    sleap_convert_assert(output_paths, slp_path)

    # Add labeled frame to video and retest
    labeled_frame = labels.find(video=labels.videos[1], frame_idx=0, return_new=True)[0]
    instance = Instance(skeleton=labels.skeleton, frame=labeled_frame)
    labels.add_instance(frame=labeled_frame, instance=instance)
    labels.append(labeled_frame)
    slp_path = tmpdir.with_name("new_slp.slp")
    labels.save(filename=slp_path)

    output_paths = [str(tmpdir.with_name("prefix"))]
    sleap_convert_assert(output_paths, slp_path)


def test_sleap_format(
    min_labels_slp: Labels,
    min_labels_slp_path: Labels,
    tmpdir,
):
    def sleap_convert_assert(output_path, slp_path):
        args = f"-o {output_path} {slp_path}".split()
        sleap_convert(args)
        assert Path(output_path).exists()

    labels = min_labels_slp
    slp_path = PurePath(min_labels_slp_path)
    tmpdir = PurePath(tmpdir)

    output_path = Path(tmpdir, slp_path)
    sleap_convert_assert(output_path, slp_path)


@pytest.mark.parametrize("suffix", [".slp", ".json", ".h5"])
def test_auto_slp_h5_json_format(
    min_labels_slp: Labels,
    min_labels_slp_path: Labels,
    tmpdir,
    suffix,
):
    def sleap_convert_assert(output_path: Path, slp_path):
        args = f"--format {output_path.suffix[1:]} {slp_path}".split()
        print(f"args = {args}")
        sleap_convert(args)
        assert Path(output_path).exists()

    labels = min_labels_slp
    slp_path = PurePath(min_labels_slp_path)
    new_slp_path = PurePath(tmpdir, slp_path.name)
    labels.save(new_slp_path)

    output_path = Path(f"{new_slp_path}{suffix}")
    print(f"output_path = {output_path}")
    sleap_convert_assert(output_path, new_slp_path)
