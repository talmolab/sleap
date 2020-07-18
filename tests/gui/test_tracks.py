from sleap.gui.overlays.tracks import TrackTrailOverlay


def test_track_trails(centered_pair_predictions):

    labels = centered_pair_predictions
    trail_manager = TrackTrailOverlay(
        labels, player=None, trail_length=6, max_node_count=24
    )

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
    trails = all_trails[tracks[0]]

    assert len(trails) == 24

    # points for first (HEAD) node
    test_trail = [
        (192.0, 189.0),
        (192.0, 188.0),
        (193.0, 187.0),
        (194.0, 186.0),
        (195.0, 185.0),
        (196.0, 185.0),
    ]

    assert test_trail in trails
