"""Adaptor to read and write ndx-pose files."""

import attr
import datetime
import numpy as np

from pathlib import PurePath
from typing import List
from pynwb import NWBFile, NWBHDF5IO, ProcessingModule
from ndx_pose import PoseEstimationSeries, PoseEstimation

import sleap
from sleap.instance import LabeledFrame, PredictedInstance, Track
from sleap.io.format.filehandle import FileHandle
from sleap.io.format.adaptor import Adaptor, SleapObjectType
from sleap.io.dataset import Labels
from sleap.io.video import Video
from sleap.skeleton import Skeleton


class NDXPoseAdaptor(Adaptor):
    """Adaptor to read and write ndx-pose files."""

    @property
    def handles(self) -> SleapObjectType:
        return SleapObjectType.labels

    @property
    def default_ext(self) -> str:
        return "nwb"

    @property
    def all_exts(self) -> List[str]:
        return ["nwb"]

    @property
    def name(self) -> str:
        return "NWB Format"

    def can_read_file(self, file: FileHandle) -> bool:
        return self.does_match_ext(file.filename)

    def can_write_filename(self, filename: str) -> bool:
        return self.does_match_ext(filename)

    def does_read(self) -> bool:
        raise True

    def does_write(self) -> bool:
        return True

    def read(self, file: FileHandle) -> Labels:
        """Read the NWB file and returns the appropriate deserialized `Labels` object.

        Args:
            file: `FileHandle` object for the NWB file to read.

        Returns:
            A `Labels` object.
        """
        labels = Labels()

        path = file.filename
        io = NWBHDF5IO(path, mode="r", load_namespaces=True)
        try:
            read_nwbfile = io.read()
            nwb_file = read_nwbfile.processing

            # Get list of videos
            video_keys: List[str] = list(nwb_file.keys())
            video_tracks = dict()

            # Get track keys
            test_processing_module: ProcessingModule = nwb_file[video_keys[0]]
            track_keys: List[str] = list(
                test_processing_module.fields["data_interfaces"]
            )

            # Get track
            test_pose_estimation: PoseEstimation = test_processing_module[track_keys[0]]
            node_names = test_pose_estimation.nodes[:]
            edge_inds = test_pose_estimation.edges[:]

            for processing_module in nwb_file.values():

                # Get track keys
                track_keys: List[str] = list(
                    processing_module.fields["data_interfaces"]
                )

                # Extract info needed to create video and tracks_numpy
                test_pose_estimation = processing_module[track_keys[0]]
                test_pose_estimation_series = test_pose_estimation[node_names[0]]

                # Recreate Labels numpy (same as output of Labels.numpy())
                n_tracks = len(track_keys)
                n_frames = test_pose_estimation_series.data[:].shape[0]
                n_nodes = len(node_names)
                tracks_numpy = np.full(
                    (n_frames, n_tracks, n_nodes, 2), np.nan, np.float32
                )
                confidence = np.full(
                    (n_frames, n_tracks, n_nodes, 1), np.nan, np.float32
                )
                for track_idx, track_key in enumerate(track_keys):
                    pose_estimation = processing_module[track_key]

                    for node_idx, node_name in enumerate(node_names):
                        pose_estimation_series = pose_estimation[node_name]

                        tracks_numpy[
                            :, track_idx, node_idx, :
                        ] = pose_estimation_series.data[:]
                        confidence[:, track_idx, node_idx, :] = np.expand_dims(
                            pose_estimation_series.confidence[:], axis=1
                        )

                video_tracks[str(PurePath(test_pose_estimation.original_videos[0]))] = (
                    tracks_numpy,
                    confidence,
                )

        except Exception as e:
            raise (e)
        finally:
            io.close()

        # Create skeleton
        skeleton = Skeleton.from_names_and_edge_inds(
            node_names=node_names,
            edge_inds=edge_inds,
        )
        labels.skeletons = [skeleton]

        # Now add instances to Labels object
        for video_fn, (tracks_numpy, confidence) in zip(
            video_tracks.keys(), video_tracks.values()
        ):
            video = Video.from_filename(video_fn)
            labels.add_video(video=video)
            n_frames, n_tracks, n_nodes, _ = tracks_numpy.shape
            for track_idx in list(range(n_tracks)):
                # Decide whether to create new track
                add_track = True
                new_track = Track(name=str(track_idx))
                track = new_track
                for l_track in labels.tracks:
                    if l_track.matches(new_track):
                        add_track = False
                        track = l_track
                        continue

                for frame_idx in list(range(n_frames)):
                    points = tracks_numpy[frame_idx, track_idx, :, :]
                    if np.isnan(points).all():
                        continue

                    frame = LabeledFrame(video=video, frame_idx=frame_idx)
                    labels.append(frame)

                    if not np.isnan(points).all():
                        inst_confidence = confidence[frame_idx, track_idx, :, :]
                        instance = PredictedInstance.from_numpy(
                            track=track,
                            points=tracks_numpy[frame_idx, track_idx, :, :],
                            point_confidences=inst_confidence,
                            instance_score=np.mean(inst_confidence),
                            skeleton=skeleton,
                        )

                        labels.add_instance(frame=frame, instance=instance)

                # Need to add track AFTER adding instances (see LabelsDataCache)
                if add_track:
                    labels.add_track(video=video, track=track)

        return labels

    def write(self, filename: str, labels: Labels):
        """Write all `PredictedInstance` objects in a `Labels` object to an NWB file.

            Use `Labels.numpy` to create a `pynwb.NWBFile` with a separate
            `pynwb.ProcessingModule` for each `Video` in the `Labels` object.

            To access the `pynwb.ProcessingModule` for a specific `Video`, use the key
            '{video_idx:03}_{video_fn.stem}' where
            `isinstance(video_fn, pathlib.PurePath)`. Ex:
                video: 'path_to_video/my_video.mp4'
                video index: 3/5
                key: '003_my_video'

            Within each `pynwb.ProcessingModule` is a `ndx_pose.PoseEstimation` for
            each unique track in the `Video`.

            The `ndx_pose.PoseEstimation` for each unique `Track` is stored under the
            key 'Track{track_idx:03}' where `track_idx` ranges from
            0 to (number of tracks) - 1. Ex:
                track_idx: 1
                key: 'Track001'

            Each `ndx_pose.PoseEstimation` has a `ndx_pose.PoseEstimationSeries` for
            every `Node` in the `Skeleton`.

            The `ndx_pose.PoseEstimationSeries` for a specific `Node` is stored under
            the key '`Node.name`'. Ex:
                node name: 'head'
                key: 'head'

        Args:
            filename: Output path for the NWB format file.
            labels: The `Labels` object to covert to a NWB format file.

        Returns:
            A `pynwb.NWBFile` with a separate `pynwb.ProcessingModule` for each
            `Video` in the `Labels` object.

        """

        skeleton = labels.skeleton

        print(f"\nCreating NWB file...")
        nwb_file = NWBFile(
            session_description="session_description",
            identifier="identifier",
            session_start_time=datetime.datetime.now(datetime.timezone.utc),
        )

        for video_idx, video in enumerate(labels.videos):
            # Create new processing module for each video
            video_fn = PurePath(video.backend.filename)
            nwb_processing_module = nwb_file.create_processing_module(
                name=f"{video_idx:03}_{video_fn.stem}",
                description=f"Processed SLEAP pose data for {video_fn.name} with "
                f"{skeleton.name} skeleton.",
            )

            # Get tracks for each video
            untracked = True if len(labels.tracks) == 0 else False
            tracks_numpy = labels.numpy(
                video=video, all_frames=True, untracked=untracked, get_confidence=True
            )
            n_frames, n_tracks, n_nodes, _ = tracks_numpy.shape
            timestamps = np.arange(n_frames)
            for track_idx in list(range(n_tracks)):
                pose_estimation_series: List[PoseEstimationSeries] = []

                for node_idx, node in enumerate(skeleton.nodes):

                    # Create instance of PoseEstimationSeries for each node
                    # Get confidence value of every frame (-1 if user labeled)
                    data = tracks_numpy[:, track_idx, node_idx, :2]
                    confidence = tracks_numpy[:, track_idx, node_idx, 2]

                    pose_estimation_series.append(
                        PoseEstimationSeries(
                            name=f"{node.name}",
                            description=f"Sequential trajectory of {node.name}.",
                            data=data,
                            unit="pixels",
                            reference_frame="No reference.",
                            timestamps=timestamps,
                            confidence=confidence,
                            confidence_definition="Point-wise confidence scores.",
                        )
                    )

                # Combine each nodes PoseEstimationSeries to create a PoseEstimation
                pose_estimation = PoseEstimation(
                    name=f"Track{track_idx:03}",
                    pose_estimation_series=pose_estimation_series,
                    description=(
                        f"Estimated positions of {skeleton.name} in video {video_fn} "
                        f"using SLEAP."
                    ),
                    original_videos=[f"{video_fn}"],
                    labeled_videos=[f"{video_fn}"],
                    dimensions=np.array([[video.backend.height, video.backend.width]]),
                    # scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
                    source_software="SLEAP",
                    source_software_version=f"{sleap.__version__}",
                    nodes=skeleton.node_names,
                    edges=skeleton.edge_inds,
                )

                # Create a processing module for each
                nwb_processing_module.add(pose_estimation)

        path = filename
        with NWBHDF5IO(path, mode="w") as io:
            io.write(nwb_file)

        print(f"Finished writing NWB file to {filename}\n")
