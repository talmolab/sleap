"""
Adaptor for reading/writing SLEAP datasets using sleap-io backend.

This adaptor provides a bridge between sleap's format system and sleap-io's
SLP reading/writing capabilities.
"""

from sleap.io import format
from sleap.io.format.adaptor import SleapObjectType
from sleap.io.format.filehandle import FileHandle
from sleap.io.dataset import Labels
from sleap_io.io.slp import read_labels, write_labels
from sleap_io.model.labels import Labels as SleapIOLabels

from typing import Optional, Union, Callable, List, Text


class SleapIOSLPAdaptor(format.adaptor.Adaptor):
    """Adaptor for reading/writing SLEAP datasets using sleap-io backend."""
    
    FORMAT_ID = 2.0  # Higher than the old HDF5 adaptor to take precedence

    @property
    def handles(self):
        return SleapObjectType.labels

    @property
    def default_ext(self):
        return "slp"

    @property
    def all_exts(self):
        return ["slp", "pkg.slp"]

    @property
    def name(self):
        return "SLEAP-IO SLP"

    def can_read_file(self, file: FileHandle):
        """Check if this adaptor can read the given file."""
        if not self.does_match_ext(file.filename):
            return False
        if not file.is_hdf5:
            return False
        # Check if it has the required HDF5 structure for SLP files
        try:
            with file.file as f:
                if "metadata" not in f:
                    return False
                if "frames" not in f:
                    return False
                if "instances" not in f:
                    return False
                if "points" not in f:
                    return False
                return True
        except Exception:
            return False

    def can_write_filename(self, filename: str):
        """Check if this adaptor can write to the given filename."""
        return self.does_match_ext(filename)

    def does_read(self) -> bool:
        return True

    def does_write(self) -> bool:
        return True

    def read(
        self,
        file: FileHandle,
        video_search: Union[Callable, List[Text], None] = None,
        match_to: Optional[Labels] = None,
        *args,
        **kwargs,
    ) -> Labels:
        """Read SLP file using sleap-io backend."""
        # Convert sleap-io Labels to sleap Labels
        sleap_io_labels = read_labels(file.filename, open_videos=True)
        
        # Convert to sleap Labels format
        return self._convert_sleap_io_to_sleap(sleap_io_labels, match_to)

    def write(self, filename: str, source_object: Labels, *args, **kwargs):
        """Write Labels to SLP file using sleap-io backend."""
        # Convert sleap Labels to sleap-io Labels
        sleap_io_labels = self._convert_sleap_to_sleap_io(source_object)
        
        # Write using sleap-io
        write_labels(filename, sleap_io_labels, embed=False, verbose=True)

    def _convert_sleap_io_to_sleap(self, sleap_io_labels: SleapIOLabels, match_to: Optional[Labels] = None) -> Labels:
        """Convert sleap-io Labels to sleap Labels."""
        from sleap.instance import LabeledFrame, Instance, PredictedInstance, Point, PredictedPoint
        from sleap.io.video import Video as SleapVideo
        from sleap.skeleton import Skeleton as SleapSkeleton, Node as SleapNode
        
        # Convert videos
        videos = []
        for video in sleap_io_labels.videos:
            sleap_video = SleapVideo.from_filename(video.filename)
            videos.append(sleap_video)
        
        # Convert skeletons
        skeletons = []
        for skeleton in sleap_io_labels.skeletons:
            sleap_skeleton = SleapSkeleton(name=skeleton.name)
            # Add nodes to the skeleton
            for node in skeleton.nodes:
                sleap_skeleton.add_node(node.name)
            # Add edges to the skeleton
            for edge in skeleton.edges:
                sleap_skeleton.add_edge(edge.source.name, edge.destination.name)
            skeletons.append(sleap_skeleton)
        
        # Convert tracks
        tracks = []
        for track in sleap_io_labels.tracks:
            from sleap.instance import Track as SleapTrack
            sleap_track = SleapTrack(track.id, track.name)
            tracks.append(sleap_track)
        
        # Convert labeled frames
        labeled_frames = []
        for lf in sleap_io_labels.labeled_frames:
            # Convert instances
            instances = []
            for instance in lf.instances:
                # Convert points
                points = []
                for i, point_data in enumerate(instance.points):
                    if hasattr(instance, 'score'):  # PredictedInstance
                        sleap_point = PredictedPoint(
                            x=point_data["xy"][0],
                            y=point_data["xy"][1],
                            visible=point_data["visible"],
                            complete=point_data["complete"],
                            score=point_data["score"] if "score" in instance.points.dtype.names else 1.0
                        )
                    else:  # Instance
                        sleap_point = Point(
                            x=point_data["xy"][0],
                            y=point_data["xy"][1],
                            visible=point_data["visible"],
                            complete=point_data["complete"]
                        )
                    points.append(sleap_point)
                
                if hasattr(instance, 'score'):  # PredictedInstance
                    sleap_instance = PredictedInstance(
                        skeleton=skeletons[0] if skeletons else None,  # Use first skeleton for now
                        points=points,
                        score=instance.score,
                        tracking_score=getattr(instance, 'tracking_score', 0.0)
                    )
                else:  # Instance
                    sleap_instance = Instance(
                        skeleton=skeletons[0] if skeletons else None,  # Use first skeleton for now
                        points=points,
                        track=tracks[instance.track.id] if instance.track and instance.track.id < len(tracks) else None
                    )
                instances.append(sleap_instance)
            
            sleap_lf = LabeledFrame(
                video=videos[lf.video.id] if lf.video and lf.video.id < len(videos) else videos[0] if videos else None,
                frame_idx=lf.frame_idx,
                instances=instances
            )
            labeled_frames.append(sleap_lf)
        
        # Create sleap Labels object
        sleap_labels = Labels(
            labeled_frames=labeled_frames,
            videos=videos,
            skeletons=skeletons,
            tracks=tracks
        )
        
        return sleap_labels

    def _convert_sleap_to_sleap_io(self, sleap_labels: Labels) -> SleapIOLabels:
        """Convert sleap Labels to sleap-io Labels."""
        from sleap_io.model.labeled_frame import LabeledFrame as SleapIOLabeledFrame
        from sleap_io.model.instance import Instance as SleapIOInstance, PredictedInstance as SleapIOPredictedInstance
        from sleap_io.model.skeleton import Skeleton as SleapIOSkeleton, Node as SleapIONode
        from sleap_io.model.video import Video as SleapIOVideo
        from sleap_io.model.instance import Track as SleapIOTrack
        import numpy as np
        
        # Convert videos
        videos = []
        for video in sleap_labels.videos:
            sleap_io_video = SleapIOVideo.from_filename(video.filename)
            videos.append(sleap_io_video)
        
        # Convert skeletons
        skeletons = []
        for skeleton in sleap_labels.skeletons:
            nodes = [SleapIONode(name=node.name) for node in skeleton.nodes]
            edges = []
            for edge in skeleton.edges:
                source_node = next(n for n in nodes if n.name == edge[0].name)
                dest_node = next(n for n in nodes if n.name == edge[1].name)
                edges.append((source_node, dest_node))
            sleap_io_skeleton = SleapIOSkeleton(nodes=nodes, edges=edges, name=skeleton.name)
            skeletons.append(sleap_io_skeleton)
        
        # Convert tracks
        tracks = []
        for track in sleap_labels.tracks:
            if track:
                sleap_io_track = SleapIOTrack(track.id, track.name)
                tracks.append(sleap_io_track)
        
        # Convert labeled frames
        labeled_frames = []
        for lf in sleap_labels.labeled_frames:
            # Convert instances
            instances = []
            for instance in lf.instances:
                # Convert points - sleap uses numpy arrays, not individual Point objects
                # Create numpy array from sleap points
                points_array = np.array([[p.x, p.y] for p in instance.points])
                visible_array = np.array([p.visible for p in instance.points])
                complete_array = np.array([p.complete for p in instance.points])
                
                # Create structured array for sleap-io
                if hasattr(instance, 'score'):  # PredictedInstance
                    score_array = np.array([p.score if hasattr(p, 'score') else 1.0 for p in instance.points])
                    from sleap_io.model.instance import PredictedPointsArray
                    points_data = np.column_stack([points_array, score_array, visible_array, complete_array])
                    points = PredictedPointsArray.from_array(points_data)
                else:  # Instance  
                    from sleap_io.model.instance import PointsArray
                    points_data = np.column_stack([points_array, visible_array, complete_array])
                    points = PointsArray.from_array(points_data)
                
                if hasattr(instance, 'score'):  # PredictedInstance
                    sleap_io_instance = SleapIOPredictedInstance(
                        skeleton=skeletons[0] if skeletons else None,  # Use first skeleton for now
                        points=points,
                        score=instance.score,
                        tracking_score=getattr(instance, 'tracking_score', 0.0)
                    )
                else:  # Instance
                    sleap_io_instance = SleapIOInstance(
                        skeleton=skeletons[0] if skeletons else None,  # Use first skeleton for now
                        points=points,
                        track=tracks[instance.track.id] if instance.track and instance.track.id < len(tracks) else None
                    )
                instances.append(sleap_io_instance)
            
            sleap_io_lf = SleapIOLabeledFrame(
                video=videos[lf.video.id] if lf.video and lf.video.id < len(videos) else videos[0] if videos else None,
                frame_idx=lf.frame_idx,
                instances=instances
            )
            labeled_frames.append(sleap_io_lf)
        
        # Create sleap-io Labels object
        sleap_io_labels = SleapIOLabels(
            labeled_frames=labeled_frames,
            videos=videos,
            skeletons=skeletons,
            tracks=tracks
        )
        
        return sleap_io_labels
