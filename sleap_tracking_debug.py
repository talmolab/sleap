import sleap

PREDICTIONS_FILE = (
    "/Users/elizabethberrigan/repos/sleap/tests/data/tracks/clip.2node.slp"
)

# Load predictions
labels = sleap.load_file(PREDICTIONS_FILE)

# Here I'm removing the tracks so we just have instances without any tracking applied.
for instance in labels.instances():
    instance.track = None
labels.tracks = []

tracker = sleap.nn.tracking.Tracker.make_tracker_by_name(
    tracker="flow",
    track_window=5,
    # Matching options
    similarity="instance",
    match="hungarian",
    max_tracking=True,
    max_tracks=2,
    kf_node_indices=[0, 1],
    kf_init_frame_count=10,
)

tracked_lfs = []
for lf in labels:
    lf.instances = tracker.track(lf.instances, img=lf.image)
    tracked_lfs.append(lf)
tracked_labels = sleap.Labels(tracked_lfs)
tracked_labels.save(
    "/Users/elizabethberrigan/repos/sleap/tests/data/tracks/clip.2node.tracked.slp"
)
