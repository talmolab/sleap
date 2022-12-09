import numpy as np

from pathlib import Path
from typing import List

from sleap.io.format.adaptor import Adaptor, SleapObjectType
from sleap.io.format.filehandle import FileHandle
from sleap.io.dataset import Labels
from sleap.io.video import Video

try:
    import nixio as nix

    nix_available = True
except ImportError:
    nix_available = False


class NixAdaptor(Adaptor):
    """Adaptor class for export of tracking analysis results to the generic [NIX](https://github.com/g-node/nix) format.
    NIX defines a generic data model for scientific data that combines data and data annotations within the same container. The written files are hdf5 files that can be read with any hdf5 library but follow the entity definitions of the NIX data model. For reading nix-files with python install the nixio low-level library ```pip install nixio``` or use the high-level api [nixtrack](https://github.com/bendalab/nixtrack).

    So far the adaptor exports the tracked positions for each node of each instance, the track and skeleton information along with the respective scores and the centroid. Additionally, the video information is exported as metadata.
    For more information on the mapping from sleap to nix see the docs on [nixtrack](https://github.com/bendalab/nixtrack) (work in progress).
    The adaptor uses a chunked writing approach which avoids numpy out of memory exceptions when exporting large datasets.

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

    @classmethod
    def can_write_filename(cls, filename: str) -> bool:
        """Returns whether this adaptor can write format of this filename."""
        raise NotImplementedError

    @classmethod
    def does_read(cls) -> bool:
        """Returns whether this adaptor supports reading."""
        return False

    @classmethod
    def does_write(cls) -> bool:
        """Returns whether this adaptor supports writing."""
        return nix_available

    @classmethod
    def read(cls, file: FileHandle) -> object:
        """Reads the file and returns the appropriate deserialized object."""
        raise NotImplementedError("NixAdaptor does not support reading.")

    @classmethod
    def __check_video(cls, labels, video):
        if video is None and len(labels.videos) == 0:
            raise ValueError(
                f"There are no videos in this project. NO output file will be be written."
            )
        if video is not None and video not in labels.videos:
            raise ValueError(
                f"Specified video {video} is not part of this project. NO output file will be be written."
            )

    @classmethod
    def write(
        cls,
        filename: str,
        source_object: object,
        source_path: str = "None",
        video: Video = None,
    ):
        """Writes the object to a file."""
        if not nix_available:
            raise ImportError(
                "NIX library not installed, export to NIX not possible (run pip install nixio)."
            )
        cls.__check_video(source_object, video)

        def create_file(filename, project, video):
            print(f"...creating nix file {filename} for {project}", end="\t")
            nf = nix.File.open(filename, nix.FileMode.Overwrite)
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
            sec = src.file.create_section(name, "nix.tracking.source.video.metadata")
            sec["filename"] = video.backend.filename
            sec["fps"] = video.backend.fps
            sec.props["fps"].unit = "Hz"
            sec["frames"] = video.num_frames
            sec["grayscale"] = video.backend.grayscale
            sec["height"] = video.backend.height
            sec["width"] = video.backend.width
            src.metadata = sec

            print("done")
            return nf

        def track_map(source: Labels):
            track_map = {}
            for t in source.tracks:
                if t.name in track_map:
                    continue
                track_map[t.name] = len(track_map)
            return track_map

        def skeleton_map(source: Labels):
            skel_map = {}
            for s in source.skeletons:
                if s.name in skel_map:
                    continue
                skel_map[s.name] = len(skel_map)
            return skel_map

        def node_map(source: Labels):
            n_map = {}
            for skeleton in source.skeletons:
                for n in skeleton.nodes:
                    if n.name in n_map:
                        continue
                    n_map[n.name] = len(n_map)
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
            positions = np.zeros((chunksize, 2, len(node_map.keys())), dtype=float)
            centroids = np.zeros((chunksize, 2), dtype=float)
            track = np.zeros_like(indices)
            skeleton = np.zeros_like(indices)
            pointscore = np.zeros((chunksize, len(node_map.keys())), dtype=float)
            instscore = np.zeros_like(track, dtype=float)
            trackscore = np.zeros_like(instscore)
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
                        track[index] = track_map[inst.track.name]
                    else:
                        track[index] = -1
                    skeleton[index] = skeleton_map[inst.skeleton.name]

                    fnames = set([n.name for n in inst.nodes])
                    nnames = set(list(node_map.keys()))
                    missing = nnames.difference(fnames)

                    for n, p in zip(inst.nodes, inst.points):
                        positions[index, :, node_map[n.name]] = np.array([p.x, p.y])
                    for m in missing:
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
                list(nodes.keys()),
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
            point_score.append_set_dimension(nodes.keys())
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
            for k in skeletons.keys():
                table_data.append((k, skeletons[k]))
            sm.append_rows(table_data)

            tm = block.create_data_frame(
                "track map",
                "nix.tracking.track_map",
                col_names=["name", "index"],
                col_dtypes=[nix.DataType.String, nix.DataType.Int8],
            )
            table_data = [("none", -1)]  # default for user-labeled instances
            for k in tracks.keys():
                table_data.append((k, tracks[k]))
            tm.append_rows(table_data)
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

        print(f"Exporting analysis to NIX file {filename} ...", end="\n")
        if video is None:
            print(f"No video specified, exporting the first one {video}...")
            video = source_object.videos[0]

        nix_file = create_file(filename, source_path, video)
        try:
            write_data(nix_file.blocks[0], source_object, video)
            print(" done")
        except Exception as e:
            print(f"\n\t Writing failed withe error message {e}!")
        finally:
            nix_file.close()
