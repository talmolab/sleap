"""Module to test all functions in sleap.nn.viz module."""

import sleap
from sleap.instance import LabeledFrame, Track
from sleap.io.dataset import Labels
from sleap.nn.viz import generate_skeleton_preview_image


def test_generate_skeleton_preview_image(
    centered_pair_predictions_slp_path: str,
    centered_pair_vid_path: str,
):
    """Encode/decode preview images for all skeletons in sleap.skeletons directory."""

    video_file = centered_pair_vid_path
    labels: Labels = sleap.load_file(
        centered_pair_predictions_slp_path, search_paths=[video_file]
    )
    lf: LabeledFrame = labels.labeled_frames[0]
    track: Track = labels.tracks[0]

    if track is None:
        inst = lf.instances[0]
    else:
        inst = next(
            instance for instance in lf.instances if instance.track.matches(track)
        )

    img_b64: bytes = generate_skeleton_preview_image(inst)
    assert isinstance(img_b64, bytes)
