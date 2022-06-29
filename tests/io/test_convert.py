from sleap.io.convert import default_analysis_filename, main as sleap_convert
from sleap.io.dataset import Labels
from sleap.io.video import Video
from sleap.instance import Instance

from pathlib import PurePath, Path


def test_analysis_format(
    min_labels_slp: Labels,
    min_labels_slp_path: Labels,
    small_robot_mp4_vid: Video,
    tmpdir,
):
    labels = min_labels_slp
    slp_path = PurePath(min_labels_slp_path)
    tmpdir = PurePath(tmpdir)

    def assert_analysis_existance(output_prefix: str):
        output_prefix = PurePath(output_prefix)
        for video in labels.videos:
            output_name = default_analysis_filename(
                labels=labels,
                video=video,
                output_path=str(output_prefix.parent),
                output_prefix=str(output_prefix.stem),
            )
            video_exists = Path(output_name).exists()
            if len(labels.get(video)) == 0:
                assert not video_exists
            else:
                assert video_exists

    def sleap_convert_assert(output_prefix, slp_path, with_output: bool = True):
        args = (
            f"--format analysis -o {output_prefix} {slp_path}".split()
            if with_output
            else f"--format analysis {slp_path}".split()
        )
        sleap_convert(args)
        assert_analysis_existance(output_prefix)

    # No output specified
    output_prefix = slp_path.with_suffix("")
    sleap_convert_assert(output_prefix, slp_path, with_output=False)

    # Specify output and retest
    output_prefix = tmpdir.with_name("prefix")
    sleap_convert_assert(output_prefix, slp_path, with_output=True)

    # Add video and retest
    labels.add_video(small_robot_mp4_vid)
    slp_path = tmpdir.with_name("new_slp.slp")
    labels.save(filename=slp_path)

    output_prefix = tmpdir.with_name("prefix")
    sleap_convert_assert(output_prefix, slp_path, with_output=True)

    # Add labeled frame to video and retest
    labeled_frame = labels.find(video=labels.videos[1], frame_idx=0, return_new=True)[0]
    instance = Instance(skeleton=labels.skeleton, frame=labeled_frame)
    labels.add_instance(frame=labeled_frame, instance=instance)
    labels.append(labeled_frame)
    slp_path = tmpdir.with_name("new_slp.slp")
    labels.save(filename=slp_path)

    output_prefix = tmpdir.with_name("prefix")
    sleap_convert_assert(output_prefix, slp_path, with_output=True)
