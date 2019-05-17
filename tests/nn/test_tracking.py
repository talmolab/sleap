from sleap.nn.tracking import FlowShiftTracker

def test_flow_tracker(centered_pair_vid, centered_pair_predictions):

    # We are going to test tracking. The dataset we have loaded
    # has already had tracking done so lets forget it.
    expected_num_tracks = len(centered_pair_predictions.tracks)
    for frame in centered_pair_predictions:
        for instance in frame:
            instance.track = None

    # Create a tracker to run
    tracker = FlowShiftTracker(window=10, verbosity=0)

    # Run tracking
    tracker.process(centered_pair_vid, centered_pair_predictions.labels)

    # Get new tracks list
    tracks = list({i.track for f in centered_pair_predictions for i in f if i.track})
    tracks.sort(key=lambda x: (x.spawned_on, x.name))

    # FIXME: Old tracking results seems to produce a few more tracks. This number probably
    # shouldn't be hard coded.
    assert len(tracks) == 24

