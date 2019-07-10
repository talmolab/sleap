from sleap.nn.tracking import FlowShiftTracker
from sleap.io.dataset import Labels

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

# def test_tracking_optflow_fail(centered_pair_vid, centered_pair_predictions):
#     frame_nums = range(0, len(centered_pair_predictions), 2)
#     labels = Labels([centered_pair_predictions[i] for i in frame_nums])
#     imgs = centered_pair_vid.get_frames(frame_nums)
#     #labels = Labels.load_json('tests/data/tracking/tracking_bug2.json.zip')
#     #imgs = labels.videos[0].get_frames([f.frame_idx for f in labels])
#
#     tracker = FlowShiftTracker(window=15, verbosity=1)
#
#     tracker.process(imgs, labels.labels)
#
#     # Get new tracks list
#     tracks = list({i.track for f in labels for i in f if i.track})
#     tracks.sort(key=lambda x: (x.spawned_on, x.name))
#
#     # FIXME: Old tracking results seems to produce a few more tracks. This number probably
#     # shouldn't be hard coded.
#     assert len(tracks) == 3
#
