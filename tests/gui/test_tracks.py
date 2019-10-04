from sleap.gui.overlays.tracks import TrackColorManager, TrackTrailOverlay
from sleap.io.video import Video


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

    trails = trail_manager.get_track_trails(frames, tracks[0])

    assert len(trails) == 24

    test_trail = [
        (245.0, 208.0),
        (245.0, 207.0),
        (245.0, 206.0),
        (246.0, 205.0),
        (247.0, 203.0),
        (248.0, 202.0),
    ]
    assert test_trail in trails

    # Test track colors
    color_manager = TrackColorManager(labels=labels)

    tracks = trail_manager.get_tracks_in_frame(labels.videos[0], 1099)
    assert len(tracks) == 5

    assert color_manager.get_color(tracks[3]) == [119, 172, 48]
