import numpy as np
import nixio as nix

from pathlib import Path
from typing import Dict, List, Optional, cast
from sleap.instance import Track

from sleap.io.format.adaptor import Adaptor, SleapObjectType
from sleap.io.format.filehandle import FileHandle
from sleap.io.dataset import Labels
from sleap.io.video import Video
from sleap.skeleton import Node, Skeleton


class NixAdaptor(Adaptor):
    """Adaptor class for export of tracking analysis results to the generic
    [NIX](https://github.com/g-node/nix) format.
    NIX defines a generic data model for scientific data that combines data and data
    annotations within the same container. The written files are hdf5 files that can
    be read with any hdf5 library but follow the entity definitions of the NIX data
    model. For reading nix-files with python install the nixio low-level library
    ```pip install nixio``` or use the high-level api
    [nixtrack](https://github.com/bendalab/nixtrack).

    So far the adaptor exports the tracked positions for each node of each instance,
    the track and skeleton information along with the respective scores and the
    centroid. Additionally, the video information is exported as metadata.
    For more information on the mapping from sleap to nix see the docs on
    [nixtrack](https://github.com/bendalab/nixtrack) (work in progress).
    The adaptor uses a chunked writing approach which avoids numpy out of memory
    exceptions when exporting large datasets.

    author: Jan Grewe (jan.grewe@g-node.org)
    """

    @property
    def default_ext(self):
        return "nix"

    @property
    def all_exts(self) -> List[str]:
        return [self.default_ext]

    @property
    def handles(self):
        return SleapObjectType.misc

    @property
    def name(self) -> str:
        """Human-reading name of the file format"""
        return (
            "NIX file flavoured for animal tracking data https://github.com/g-node/nix"
        )

    @classmethod
    def can_read_file(cls, file: FileHandle) -> bool:
        """Returns whether this adaptor can read this file."""
        return False

    def can_write_filename(self, filename: str) -> bool:
        """Returns whether this adaptor can write format of this filename."""
        return filename.endswith(tuple(self.all_exts))

    @classmethod
    def does_read(cls) -> bool:
        """Returns whether this adaptor supports reading."""
        return False

    @classmethod
    def does_write(cls) -> bool:
        """Returns whether this adaptor supports writing."""
        return True

    @classmethod
    def read(cls, file: FileHandle) -> object:
        """Reads the file and returns the appropriate deserialized object."""
        raise NotImplementedError("NixAdaptor does not support reading.")

    @classmethod
    def __check_video(cls, labels: Labels, video: Optional[Video]):
        if (video is None) and (len(labels.videos) == 0):
            raise ValueError(
                f"There are no videos in this project. "
                "No analysis file will be be written."
            )
        if video is not None:
            if video not in labels.videos:
                raise ValueError(
                    f"Specified video {video} is not part of this project. "
                    "Skipping the analysis file for this video."
                )
            if len(labels.get(video)) == 0:
                raise ValueError(
                    f"No labeled frames in {video.backend.filename}. "
                    "Skipping the analysis file for this video."
                )

    @classmethod
    def write(
        cls,
        filename: str,
        source_object: object,
        source_path: Optional[str] = None,
        video: Optional[Video] = None,
    ):
        """Writes the object to a file."""
        source_object = cast(Labels, source_object)

        cls.__check_video(source_object, video)

        def create_file(filename: str, project: Optional[str], video: Video):
            print(f"Creating nix file...", end="\t")
            nf = nix.File.open(filename, nix.FileMode.Overwrite)
            try:
                s = nf.create_section("TrackingAnalysis", "nix.tracking.metadata")
                s["version"] = "0.1.0"
                s["format"] = "nix.tracking"
                s["definitions"] = "https://github.com/bendalab/nixtrack"
                s["writer"] = str(cls)[8:-2]
                if project is not None:
                    s["project"] = project

                name = Path(video.backend.filename).name
                b = nf.create_block(name, "nix.tracking_results")

                # add video metadata, if exists
                src = b.create_source(name, "nix.tracking.source.video")
                sec = src.file.create_section(
                    name, "nix.tracking.source.video.metadata"
                )
                sec["filename"] = video.backend.filename
                sec["fps"] = getattr(video.backend, "fps", 0.0)
                sec.props["fps"].unit = "Hz"
                sec["frames"] = video.num_frames
                sec["grayscale"] = getattr(video.backend, "grayscale", None)
                sec["height"] = video.backend.height
                sec["width"] = video.backend.width
                src.metadata = sec
            except Exception as e:
                nf.close()
                raise e

            print("done")
            return nf

        def track_map(source: Labels) -> Dict[Track, int]:
            track_map: Dict[Track, int] = {}
            for track in source.tracks:
                if track in track_map:
                    continue
                track_map[track] = len(track_map)
            return track_map

        def skeleton_map(source: Labels) -> Dict[Skeleton, int]:
            skel_map: Dict[Skeleton, int] = {}
            for skeleton in source.skeletons:
                if skeleton in skel_map:
                    continue
                skel_map[skeleton] = len(skel_map)
            return skel_map

        def node_map(source: Labels) -> Dict[Node, int]:
            n_map: Dict[Node, int] = {}
            for node in source.nodes:
                if node in n_map:
                    continue
                n_map[node] = len(n_map)
            return n_map

        def create_feature_array(name, type, block, frame_index_array, shape, dtype):
            array = block.create_data_array(name, type, dtype=dtype, shape=shape)
            rd = array.append_range_dimension()
            rd.link_data_array(frame_index_array, [-1])
            return array

        def create_positions_array(
            name, type, block, frame_index_array, node_names, shape, dtype
        ):
            array = block.create_data_array(
                name, type, dtype=dtype, shape=shape, label="pixel"
            )
            rd = array.append_range_dimension()
            rd.link_data_array(frame_index_array, [-1])
            array.append_set_dimension(["x", "y"])
            array.append_set_dimension(node_names)
            return array

        def chunked_write(
            instances,
            frameid_array,
            positions_array,
            track_array,
            skeleton_array,
            pointscore_array,
            instancescore_array,
            trackingscore_array,
            centroid_array,
            track_map,
            node_map,
            skeleton_map,
            chunksize=10000,
        ):
            data_written = 0
            indices = np.zeros(chunksize, dtype=int)
            track = np.zeros_like(indices)
            skeleton = np.zeros_like(indices)
            instscore = np.zeros_like(indices, dtype=float)
            positions = np.zeros((chunksize, 2, len(node_map.keys())), dtype=float)
            centroids = np.zeros((chunksize, 2), dtype=float)
            trackscore = np.zeros_like(instscore)
            pointscore = np.zeros((chunksize, len(node_map.keys())), dtype=float)
            dflt_pointscore = [0.0 for n in range(len(node_map.keys()))]

            while data_written < len(instances):
                print(".", end="")
                start = data_written
                end = (
                    len(instances)
                    if start + chunksize >= len(instances)
                    else start + chunksize
                )
                for i in range(start, end):
                    inst = instances[i]
                    index = i - start
                    indices[index] = inst.frame_idx
                    if inst.track is not None:
                        track[index] = track_map[inst.track]
                    else:
                        track[index] = -1

                    skeleton[index] = skeleton_map[inst.skeleton]

                    all_nodes = set([n.name for n in inst.nodes])
                    used_nodes = set([n.name for n in node_map.keys()])
                    missing_nodes = all_nodes.difference(used_nodes)
                    for n, p in zip(inst.nodes, inst.points):
                        positions[index, :, node_map[n]] = np.array([p.x, p.y])
                    for m in missing_nodes:
                        positions[index, :, node_map[m]] = np.array([np.nan, np.nan])

                    centroids[index, :] = inst.centroid
                    if hasattr(inst, "score"):
                        instscore[index] = inst.score
                        trackscore[index] = inst.tracking_score
                        pointscore[index, :] = inst.scores
                    else:
                        instscore[index] = 0.0
                        trackscore[index] = 0.0
                        pointscore[index, :] = dflt_pointscore

                frameid_array[start:end] = indices[: end - start]
                track_array[start:end] = track[: end - start]
                positions_array[start:end, :, :] = positions[: end - start, :, :]
                centroid_array[start:end, :] = centroids[: end - start, :]
                skeleton_array[start:end] = skeleton[: end - start]
                pointscore_array[start:end] = pointscore[: end - start]
                instancescore_array[start:end] = instscore[: end - start]
                trackingscore_array[start:end] = trackscore[: end - start]
                data_written += end - start

        def write_data(block, source: Labels, video: Video):
            instances = [
                instance
                for instance in source.instances(video=video)
                if instance.frame_idx is not None
            ]
            instances = sorted(instances, key=lambda i: i.frame_idx)
            nodes = node_map(source)
            tracks = track_map(source)
            skeletons = skeleton_map(source)
            positions_shape = (len(instances), 2, len(nodes))

            frameid_array = block.create_data_array(
                "frame",
                "nix.tracking.instance_frameidx",
                label="frame index",
                shape=(len(instances),),
                dtype=nix.DataType.Int64,
            )
            frameid_array.append_range_dimension_using_self()

            positions_array = create_positions_array(
                "position",
                "nix.tracking.instance_position",
                block,
                frameid_array,
                [node.name for node in nodes.keys()],
                positions_shape,
                nix.DataType.Float,
            )

            track_array = create_feature_array(
                "track",
                "nix.tracking.instance_track",
                block,
                frameid_array,
                shape=(len(instances),),
                dtype=nix.DataType.Int64,
            )

            skeleton_array = create_feature_array(
                "skeleton",
                "nix.tracking.instance_skeleton",
                block,
                frameid_array,
                (len(instances),),
                nix.DataType.Int64,
            )

            point_score = create_feature_array(
                "node score",
                "nix.tracking.nodes_score",
                block,
                frameid_array,
                (len(instances), len(nodes)),
                nix.DataType.Float,
            )
            point_score.append_set_dimension([node.name for node in nodes.keys()])

            centroid_array = create_feature_array(
                "centroid",
                "nix.tracking.centroid_position",
                block,
                frameid_array,
                (len(instances), 2),
                nix.DataType.Float,
            )

            centroid_array.append_set_dimension(["x", "y"])
            instance_score = create_feature_array(
                "instance score",
                "nix.tracking.instance_score",
                block,
                frameid_array,
                (len(instances),),
                nix.DataType.Float,
            )

            tracking_score = create_feature_array(
                "tracking score",
                "nix.tracking.tack_score",
                block,
                frameid_array,
                (len(instances),),
                nix.DataType.Float,
            )

            # bind all together using a nix.MultiTag
            mtag = block.create_multi_tag(
                "tracking results", "nix.tracking.results", positions=frameid_array
            )
            mtag.references.append(positions_array)
            mtag.create_feature(track_array, nix.LinkType.Indexed)
            mtag.create_feature(skeleton_array, nix.LinkType.Indexed)
            mtag.create_feature(point_score, nix.LinkType.Indexed)
            mtag.create_feature(instance_score, nix.LinkType.Indexed)
            mtag.create_feature(tracking_score, nix.LinkType.Indexed)
            mtag.create_feature(centroid_array, nix.LinkType.Indexed)

            sm = block.create_data_frame(
                "skeleton map",
                "nix.tracking.skeleton_map",
                col_names=["name", "index"],
                col_dtypes=[nix.DataType.String, nix.DataType.Int8],
            )
            table_data = []
            for track in skeletons.keys():
                table_data.append((track.name, skeletons[track]))
            sm.append_rows(table_data)

            nm = block.create_data_frame(
                "node map",
                "nix.tracking.node_map",
                col_names=["name", "weight", "index", "skeleton"],
                col_dtypes=[
                    nix.DataType.String,
                    nix.DataType.Float,
                    nix.DataType.Int8,
                    nix.DataType.Int8,
                ],
            )
            table_data = []
            for node in nodes.keys():
                skel_index = -1  # if node is not assigned to a skeleton
                for track in skeletons:
                    if node in track.nodes:
                        skel_index = skeletons[track]
                        break
                table_data.append((node.name, node.weight, nodes[node], skel_index))
            nm.append_rows(table_data)

            tm = block.create_data_frame(
                "track map",
                "nix.tracking.track_map",
                col_names=["name", "spawned_on", "index"],
                col_dtypes=[nix.DataType.String, nix.DataType.Int64, nix.DataType.Int8],
            )
            table_data = [("none", -1, -1)]  # default for user-labeled instances
            for track in tracks.keys():
                table_data.append((track.name, track.spawned_on, tracks[track]))
            tm.append_rows(table_data)

            # Print shape info
            data_dict = {
                "instances": instances,
                "frameid_array": frameid_array,
                "positions_array": positions_array,
                "track_array": track_array,
                "skeleton_array": skeleton_array,
                "point_score": point_score,
                "instance_score": instance_score,
                "tracking_score": tracking_score,
                "centroid_array": centroid_array,
                "tracks": tracks,
                "nodes": nodes,
                "skeletons": skeletons,
            }
            for key, val in data_dict.items():
                print(f"\t{key}:", end=" ")
                if hasattr(val, "shape"):
                    print(f"{val.shape}")
                else:
                    print(f"{len(val)}")

            # Print labels/video info
            print(
                f"\tlabels path: {source_path}\n"
                f"\tvideo path: {video.backend.filename}\n"
                f"\tvideo index = {source_object.videos.index(video)}"
            )

            print(f"Writing to NIX file...")
            chunked_write(
                instances,
                frameid_array,
                positions_array,
                track_array,
                skeleton_array,
                point_score,
                instance_score,
                tracking_score,
                centroid_array,
                tracks,
                nodes,
                skeletons,
            )
            print(f"done")

        print(f"\nExporting to NIX analysis file...")
        if video is None:
            video = source_object.videos[0]
            print(f"No video specified, exporting the first one...")

        nix_file = None
        try:
            nix_file = create_file(filename, source_path, video)
            write_data(nix_file.blocks[0], source_object, video)
            print(f"Saved as {filename}")
        except Exception as e:
            print(f"\n\tWriting failed with following error:\n{e}!")
        finally:
            if nix_file is not None:
                nix_file.close()
