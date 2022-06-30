import os
import numpy as np

from typing import List
from ..dataset import Labels
from ..video import Video
from ...skeleton import Skeleton
from .adaptor import Adaptor, SleapObjectType
from .filehandle import FileHandle

try:
    import nixio as nix
    nix_available = True
except ImportError:
    nix_available = False


class NixAdaptor(Adaptor):

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
        return "NIX file flavoured for animal tracking data https://github.com/g-node/nix"

    @classmethod
    def can_read_file(file: FileHandle) -> bool:
        """Returns whether this adaptor can read this file."""
        return False

    @classmethod
    def can_write_filename(filename: str) -> bool:
        """Returns whether this adaptor can write format of this filename."""
        raise NotImplementedError

    @classmethod
    def does_read() -> bool:
        """Returns whether this adaptor supports reading."""
        return False

    @classmethod
    def does_write() -> bool:
        """Returns whether this adaptor supports writing."""
        return nix_available
     
    @classmethod
    def read(file: FileHandle) -> object:
        """Reads the file and returns the appropriate deserialized object."""
        raise NotImplementedError("NixAdaptor does not support reading.")

    @classmethod
    def write(cls, filename: str, source_object: object, source_path: str = None, video: Video = None):
        """Writes the object to a file."""
        def check(labels, video):
            if video is None and len(labels.videos) == 0:
                return False
            return True

        def create_file(filename, project, video):
            nf = nix.File.open(filename, nix.FileMode.Overwrite)
            s = nf.create_section("TrackingAnalysis", "nix.tracking.metadata")
            s["version"] = "0.1.0"
            s["format"] = "nix.tracking"
            s["definitions"] = "github.com/bendalab/nix_tracking"
            s["writer"] = str(cls)[8:-2]
            if project is not None:
                s["project"] = project
            if video is not None:
                name = os.path.split(video.backend.filename)[-1]
                b = nf.create_block(name, "nix.tracking_results")
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

        def write(block, source: Labels):
            instances = list(source.instances(video=video))
            instances = sorted(instances, key=lambda i: i.frame_idx)
            nodes = node_map(source)
            tracks = track_map(source)
            skeletons = skeleton_map(source)
            positions_shape = (len(instances), 2, len(nodes))

            frameid_array = block.create_data_array("frame", "nix.tracking.instance_frameidx",
                                                    shape=(len(instances),), dtype=nix.DataType.Int64)
            frameid_array.label = "frame index"
            frameid_array.append_range_dimension_using_self()

            positions_array = block.create_data_array("position", "nix.tracking.instance_positions",
                                                      dtype=nix.DataType.Float, shape=positions_shape)
            positions_array.label = "pixel"
            rd = positions_array.append_range_dimension()

            rd.link_data_array(frameid_array, [-1])
            positions_array.append_set_dimension(["x", "y"])
            positions_array.append_set_dimension(list(nodes.keys()))

            track_array = block.create_data_array("track", "nix.tracking.instance_track", 
                                                  dtype=nix.DataType.Int64, shape=(len(instances),))
            rd = track_array.append_range_dimension()
            rd.link_data_array(frameid_array, [-1])

            skeleton_array = block.create_data_array("skeleton", "nix.tracking.instance_skeleton",
                                                      dtype=nix.DataType.Int64, shape=(len(instances),))
            rd = skeleton_array.append_range_dimension()
            rd.link_data_array(frameid_array, [-1])

            for i, inst in enumerate(instances):
                frameid_array[i] = inst.frame_idx
                if inst.track is not None:
                    track_array[i] = tracks[inst.track.name]
                else:
                    track_array[i] = -1
                skeleton_array[i] = skeletons[inst.skeleton.name]
                for node, point in zip(inst.nodes, inst.points_array):
                     positions_array[i, :, nodes[node.name]] = point

            sm = block.create_data_frame("skeleton map", "nix.tracking.skeleton_map", 
                                         col_names=["name", "index"],
                                         col_dtypes=[nix.DataType.String, nix.DataType.Int8])
            table_data = []
            for k in skeletons.keys():
                table_data.append((k, skeletons[k]))
            sm.append_rows(table_data)

            tm = block.create_data_frame("track map", "nix.tracking.track_map", 
                                         col_names=["name", "index"],
                                         col_dtypes=[nix.DataType.String, nix.DataType.Int8])
            table_data = [("none", -1)]
            for k in tracks.keys():
                table_data.append((k, tracks[k]))
            tm.append_rows(table_data)

        if not nix_available:
            raise ImportError("NIX library not installed, export to nix not possible. (run pip install nixio)")
        if not check(source_object, video):
            raise ValueError(f"There are no videos in this project. Output file will not be written.")

        nix_file = create_file(filename, source_path, video)
        write(nix_file.blocks[0], source_object)

        nix_file.close()
