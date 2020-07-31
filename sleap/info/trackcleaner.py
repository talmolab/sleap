"""
CLI for TrackCleaner (mostly deprecated).

This runs the track cleaner to (i) cull instances and (ii) connect track breaks
in predictions which have already been tracked.

The `sleap-track` CLI lets you run (or re-run) tracking on predictions, and
includes options to cull instances *before* tracking (with more control over
how instances are selected for culling) and connect track breaks. In most cases
it will be better to use the `sleap-track` CLI.
"""

import operator
from typing import Text

from sleap import Labels
from sleap.nn.tracking import TrackCleaner


def fit_tracks(filename: Text, instance_count: int):
    """Wraps `TrackCleaner` for easier cli api."""

    labels = Labels.load_file(filename)
    video = labels.videos[0]
    frames = labels.find(video)

    TrackCleaner(instance_count=instance_count).run(frames=frames)

    # Rebuild list of tracks
    labels.tracks = list(
        {
            instance.track
            for frame in labels
            for instance in frame.instances
            if instance.track
        }
    )

    labels.tracks.sort(key=operator.attrgetter("spawned_on", "name"))

    # Save new file
    save_filename = filename
    save_filename = save_filename.replace(".slp", ".cleaned.slp")
    save_filename = save_filename.replace(".h5", ".cleaned.h5")
    save_filename = save_filename.replace(".json", ".cleaned.json")
    Labels.save_file(labels, save_filename)

    print(f"Saved: {save_filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to labels json file")
    parser.add_argument(
        "-c",
        "--instance_count",
        type=int,
        default=2,
        help="Count of instances to keep in each frame",
    )

    args = parser.parse_args()

    fit_tracks(filename=args.data_path, instance_count=args.instance_count)

    # print(args)
