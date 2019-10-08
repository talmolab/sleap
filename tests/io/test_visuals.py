import os
from sleap.io.visuals import save_labeled_video


def test_write_visuals(tmpdir, centered_pair_predictions):
    path = os.path.join(tmpdir, "clip.avi")
    save_labeled_video(
        filename=path,
        labels=centered_pair_predictions,
        video=centered_pair_predictions.videos[0],
        frames=(0, 1, 2),
        fps=15,
    )
    assert os.path.exists(path)
