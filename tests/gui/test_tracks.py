import pytest
import sleap_io as sio
from qtpy import QtGui

from sleap.gui.overlays.tracks import TrackTrailOverlay


class _FakeColorManager:
    def get_track_color(self, key):
        return (10, 20, 30)


class _FakeScene:
    def __init__(self):
        self.paths = []

    def addPath(self, path, pen):
        self.paths.append((path, QtGui.QColor(pen.color())))
        return object()


class _FakePlayer:
    def __init__(self):
        self.scene = _FakeScene()
        self.color_manager = _FakeColorManager()


def test_track_trails(centered_pair_predictions):
    labels = centered_pair_predictions
    trail_manager = TrackTrailOverlay(labels, player=None, trail_length=6)

    frames = trail_manager.get_frame_selection(labels.videos[0], 27)
    assert len(frames) == 6
    assert frames[0].frame_idx == 22

    tracks = trail_manager.get_tracks_in_frame(labels.videos[0], 27)
    assert len(tracks) == 2
    assert tracks[0].name == "1"
    assert tracks[1].name == "2"

    tracks_with_trails = trail_manager.get_tracks_in_frame(
        labels.videos[0], 27, include_trails=True
    )
    assert len(tracks_with_trails) == 13

    all_trails = trail_manager.get_track_trails(frames)
    # The 6-frame trail window can include tracks not present in frame 27
    # itself (e.g. a track that appeared earlier and dropped out).
    assert set(tracks) <= set(all_trails.keys())

    # Default `trail_node` is "centroid".
    trail = all_trails[tracks[0]]
    assert len(trail) == 6
    assert trail[-1] == labels.find(labels.videos[0], 27)[0].instances[0].centroid_xy


def test_track_trails_named_node(centered_pair_predictions):
    """A named `trail_node` follows that node's position, not the centroid."""
    labels = centered_pair_predictions
    trail_manager = TrackTrailOverlay(
        labels, player=None, trail_length=6, trail_node="head"
    )

    frames = trail_manager.get_frame_selection(labels.videos[0], 27)
    tracks = trail_manager.get_tracks_in_frame(labels.videos[0], 27)
    all_trails = trail_manager.get_track_trails(frames)

    trail = all_trails[tracks[0]]
    expected = [
        (192.0, 189.0),
        (192.0, 188.0),
        (193.0, 187.0),
        (194.0, 186.0),
        (195.0, 185.0),
        (196.0, 185.0),
    ]
    assert trail == expected


def test_track_trails_unknown_node_falls_back_to_centroid(centered_pair_predictions):
    """A stale/unknown `trail_node` (e.g. after switching skeletons) falls back to
    centroid instead of raising.
    """
    labels = centered_pair_predictions
    trail_manager = TrackTrailOverlay(
        labels, player=None, trail_length=6, trail_node="not_a_real_node"
    )

    frames = trail_manager.get_frame_selection(labels.videos[0], 27)
    tracks = trail_manager.get_tracks_in_frame(labels.videos[0], 27)
    all_trails = trail_manager.get_track_trails(frames)

    trail = all_trails[tracks[0]]
    assert trail[-1] == labels.find(labels.videos[0], 27)[0].instances[0].centroid_xy


def test_get_node_options(centered_pair_predictions):
    labels = centered_pair_predictions
    options = TrackTrailOverlay.get_node_options(labels)
    assert options[0] == "centroid"
    assert options[1:] == list(labels.skeletons[0].node_names)


def test_get_node_options_no_skeleton():
    labels = sio.Labels()
    assert TrackTrailOverlay.get_node_options(labels) == ["centroid"]


def _make_untracked_labels(n_frames: int = 4) -> sio.Labels:
    """Single untracked instance moving diagonally, no tracks assigned."""
    skel = sio.Skeleton(["head", "tail"])
    video = sio.Video(filename="a.mp4")
    lfs = []
    for i in range(n_frames):
        inst = sio.PredictedInstance.from_numpy(
            points_data=[[10.0 + i, 20.0 + i], [15.0 + i, 25.0 + i]],
            skeleton=skel,
            point_scores=[1.0, 1.0],
            score=1.0,
        )
        lfs.append(sio.LabeledFrame(video=video, frame_idx=i, instances=[inst]))
    return sio.Labels(labeled_frames=lfs, videos=[video], skeletons=[skel])


def test_track_trails_without_tracks():
    """Untracked / single-instance data still gets a trail, keyed by instance
    position within the frame (matching sleap-io's index-keyed fallback) rather
    than by `Track` identity.
    """
    labels = _make_untracked_labels(n_frames=4)
    assert len(labels.tracks) == 0

    trail_manager = TrackTrailOverlay(labels, player=None, trail_length=4)
    frames = trail_manager.get_frame_selection(labels.videos[0], 3)
    all_trails = trail_manager.get_track_trails(frames)

    assert list(all_trails.keys()) == [0]
    assert all_trails[0] == [
        (12.5, 22.5),
        (13.5, 23.5),
        (14.5, 24.5),
        (15.5, 25.5),
    ]


def test_track_trails_carries_forward_missing_detections():
    """A momentarily invisible node carries its last known position forward
    rather than breaking the trail's length/fade bookkeeping.
    """
    skel = sio.Skeleton(["head"])
    video = sio.Video(filename="a.mp4")
    track = sio.Track(name="1")
    points_by_frame = [[10.0, 20.0], [float("nan"), float("nan")], [12.0, 22.0]]
    lfs = []
    for i, xy in enumerate(points_by_frame):
        inst = sio.PredictedInstance.from_numpy(
            points_data=[xy],
            skeleton=skel,
            point_scores=[1.0],
            score=1.0,
            track=track,
        )
        lfs.append(sio.LabeledFrame(video=video, frame_idx=i, instances=[inst]))
    labels = sio.Labels(
        labeled_frames=lfs, videos=[video], skeletons=[skel], tracks=[track]
    )

    trail_manager = TrackTrailOverlay(
        labels, player=None, trail_length=3, trail_node="head"
    )
    frames = trail_manager.get_frame_selection(video, 2)
    all_trails = trail_manager.get_track_trails(frames)

    assert len(all_trails[track]) == 3
    # Frame 1's invisible "head" point carries frame 0's position forward.
    assert all_trails[track][1] == all_trails[track][0] == (10.0, 20.0)
    assert all_trails[track][2] == (12.0, 22.0)


def test_add_to_scene_fades_oldest_to_newest():
    """With `trail_alpha_fade` on (the default), drawn segments' alpha rises
    monotonically from the oldest to the newest, and the newest is fully opaque.

    Uses a single untracked instance so all drawn segments belong to one trail
    (multiple tracks would each restart their own oldest->newest ramp,
    interleaved in draw order).
    """
    labels = _make_untracked_labels(n_frames=6)
    player = _FakePlayer()
    overlay = TrackTrailOverlay(labels, player=player, trail_length=6)

    overlay.add_to_scene(labels.videos[0], 5)

    assert len(overlay.items) == len(player.scene.paths)
    assert len(player.scene.paths) > 1

    alphas = [color.alphaF() for _, color in player.scene.paths]
    assert alphas == sorted(alphas)
    assert alphas[-1] == pytest.approx(1.0)
    assert alphas[0] < alphas[-1]


def test_add_to_scene_without_fade_uses_uniform_alpha():
    """With `trail_alpha_fade` off, every drawn segment uses the same
    `trail_alpha` -- no oldest/newest gradient.
    """
    labels = _make_untracked_labels(n_frames=6)
    player = _FakePlayer()
    overlay = TrackTrailOverlay(
        labels,
        player=player,
        trail_length=6,
        trail_alpha_fade=False,
        trail_alpha=0.5,
    )

    overlay.add_to_scene(labels.videos[0], 5)

    assert player.scene.paths
    # `abs` tolerance accounts for QColor's 8-bit alpha-channel quantization.
    alphas = [color.alphaF() for _, color in player.scene.paths]
    assert all(a == pytest.approx(0.5, abs=1e-2) for a in alphas)


def test_add_to_scene_hidden_or_zero_length_draws_nothing(centered_pair_predictions):
    labels = centered_pair_predictions
    player = _FakePlayer()

    overlay = TrackTrailOverlay(labels, player=player, trail_length=0)
    overlay.add_to_scene(labels.videos[0], 27)
    assert overlay.items == []

    overlay = TrackTrailOverlay(labels, player=player, trail_length=6, show=False)
    overlay.add_to_scene(labels.videos[0], 27)
    assert overlay.items == []
