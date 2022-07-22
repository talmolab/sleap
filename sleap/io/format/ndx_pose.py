"""Adaptor to read and write ndx-pose files."""

import attr
import datetime
import re
import numpy as np
import uuid

from pathlib import Path, PurePath
from typing import List, Optional
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
            video_keys: List[str] = [
                key for key in nwb_file.keys() if "SLEAP_VIDEO" in key
            ]
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
                is_tracked: bool = re.sub("[0-9]+", "", track_keys[0]) == "track"

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
                confidence = np.full((n_frames, n_tracks, n_nodes), np.nan, np.float32)
                for track_idx, track_key in enumerate(track_keys):
                    pose_estimation = processing_module[track_key]

                    for node_idx, node_name in enumerate(node_names):
                        pose_estimation_series = pose_estimation[node_name]

                        tracks_numpy[
                            :, track_idx, node_idx, :
                        ] = pose_estimation_series.data[:]
                        confidence[
                            :, track_idx, node_idx
                        ] = pose_estimation_series.confidence[:]

                video_tracks[str(PurePath(test_pose_estimation.original_videos[0]))] = (
                    tracks_numpy,
                    confidence,
                    is_tracked,
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

        # Add instances to labeled frames
        lfs = []
        for video_fn, (tracks_numpy, confidence, is_tracked) in video_tracks.items():
            video = Video.from_filename(video_fn)
            n_frames, n_tracks, n_nodes, _ = tracks_numpy.shape
            tracks = [Track(name=f"track{track_idx}") for track_idx in range(n_tracks)]
            for frame_idx, (frame_pts, frame_confs) in enumerate(
                zip(tracks_numpy, confidence)
            ):
                insts = []
                for track, (inst_pts, inst_confs) in zip(
                    tracks, zip(frame_pts, frame_confs)
                ):
                    if np.isnan(inst_pts).all():
                        continue
                    insts.append(
                        PredictedInstance.from_numpy(
                            points=inst_pts,  # (n_nodes, 2)
                            point_confidences=inst_confs,  # (n_nodes,)
                            instance_score=inst_confs.mean(),  # ()
                            skeleton=skeleton,
                            track=track if is_tracked else None,
                        )
                    )
                if len(insts) > 0:
                    lfs.append(
                        LabeledFrame(video=video, frame_idx=frame_idx, instances=insts)
                    )
        labels = Labels(lfs)
        return labels

    def write(
        self,
        filename: str,
        labels: Labels,
        overwrite: bool = False,
        session_description: str = "Processed SLEAP pose data",
        identifier: Optional[str] = None,
        session_start_time: Optional[datetime.datetime] = None,
    ):
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
            key 'track{track_idx:03}' if tracks are set or 'untrack{track_idx:03}' if
            untracked where `track_idx` ranges from
            0 to (number of tracks) - 1. Ex:
                track_idx: 1
                key: 'track001'

            Each `ndx_pose.PoseEstimation` has a `ndx_pose.PoseEstimationSeries` for
            every `Node` in the `Skeleton`.

            The `ndx_pose.PoseEstimationSeries` for a specific `Node` is stored under
            the key '`Node.name`'. Ex:
                node name: 'head'
                key: 'head'

        Args:
            filename: Output path for the NWB format file.
            labels: The `Labels` object to covert to a NWB format file.
            overwrite: Boolean that overwrites existing NWB file if True. If False, data
                will be appended to existing NWB file.
            session_description: Description for entire project. Stored under
                NWBFile "session_description" key. If appending data to a preexisting
                file, then the session_description will not be used.
            identifier: Unique identifier for project. If no identifier is
                specified, then will generate a GUID. If appending data to a
                preexisting file, then the identifier will not be used.
            session_start_time: THe datetime associated with the project. If no
                session_start_time is given, then the current datetime will be used. If
                appending data to a preexisting file, then the session_start_time will
                not be used.

        Returns:
            A `pynwb.NWBFile` with a separate `pynwb.ProcessingModule` for each
            `Video` in the `Labels` object.

        """

        # Check that this project contains predicted instances
        if len(labels.predicted_instances) == 0:
            raise TypeError(
                "Only predicted instances are written to the NWB format. "
                "This project has no predicted instances."
            )

        # Set optional kwargs if not specified by user
        if session_start_time is None:
            session_start_time = datetime.datetime.now(datetime.timezone.utc)
        identifier = str(uuid.uuid4()) if identifier is None else identifier

        try:
            io = None
            if Path(filename).exists() and not overwrite:
                # Append to file if it exists and we do not want to overwrite
                print(f"\nOpening existing NWB file...")
                io = NWBHDF5IO(filename, mode="a", load_namespaces=True)
                nwb_file = io.read()
            else:
                # If file does not exist or we want to overwrite, create new file
                if not overwrite:
                    print(f"\nCould not find the file specified: {filename}")
                print(f"\nCreating NWB file...")
                nwb_file = NWBFile(
                    session_description=session_description,
                    identifier=identifier,
                    session_start_time=session_start_time,
                )
                io = NWBHDF5IO(filename, mode="w")

            skeleton = labels.skeleton

            for video_idx, video in enumerate(labels.videos):
                # Create new processing module for each video
                video_fn = PurePath(video.backend.filename)
                try:
                    name = f"SLEAP_VIDEO_{video_idx:03}_{video_fn.stem}"
                    nwb_processing_module = nwb_file.create_processing_module(
                        name=name,
                        description=f"{session_description} for {video_fn.name} with "
                        f"{skeleton.name} skeleton.",
                    )
                except ValueError:
                    # Cannot overwrite or delete processing modules
                    print(
                        f"Processing module for {video_fn.name} already exists... "
                        f"Skipping: {name}"
                    )
                    continue

                # Get tracks for each video
                video_lfs = labels.get(video)
                untracked = all(
                    [inst.track is None for lf in video_lfs for inst in lf.instances]
                )
                tracks_numpy = labels.numpy(
                    video=video,
                    all_frames=True,
                    untracked=untracked,
                    return_confidence=True,
                )
                n_frames, n_tracks, n_nodes, _ = tracks_numpy.shape
                timestamps = np.arange(n_frames)
                for track_idx in list(range(n_tracks)):
                    pose_estimation_series: List[PoseEstimationSeries] = []

                    for node_idx, node in enumerate(skeleton.nodes):

                        # Create instance of PoseEstimationSeries for each node
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

                    # Combine each node's PoseEstimationSeries to create a PoseEstimation
                    name_prefix = "untracked" if untracked else "track"
                    pose_estimation = PoseEstimation(
                        name=f"{name_prefix}{track_idx:03}",
                        pose_estimation_series=pose_estimation_series,
                        description=(
                            f"Estimated positions of {skeleton.name} in video {video_fn} "
                            f"using SLEAP."
                        ),
                        original_videos=[f"{video_fn}"],
                        labeled_videos=[f"{video_fn}"],
                        dimensions=np.array(
                            [[video.backend.height, video.backend.width]]
                        ),
                        scorer=str(labels.provenance),
                        source_software="SLEAP",
                        source_software_version=f"{sleap.__version__}",
                        nodes=skeleton.node_names,
                        edges=skeleton.edge_inds,
                    )

                    # Create a processing module for each
                    nwb_processing_module.add(pose_estimation)

            io.write(nwb_file)

        except Exception as e:
            raise e

        finally:
            if io is not None:
                io.close()

        print(f"Finished writing NWB file to {filename}\n")
