"""
Adaptor for reading/writing old, JSON dataset format (kind of deprecated).

This supports reading and writing both `.json` and `.json.zip`. The zip allows
you to include image files, in imgstore videos. A better option now is to save
a single HDF5 file which include both the SLEAP dataset (i.e., `Labels`) and
also the videos/frames as HDF5 datasets.
"""
import atexit
import os
import re
import shutil
import tempfile
import zipfile
from typing import Optional, Union, Dict, List, Callable, Text

import cattr

from .adaptor import Adaptor, SleapObjectType
from .filehandle import FileHandle

from sleap import Labels, Video
from sleap.gui.suggestions import SuggestionFrame
from sleap.instance import (
    LabeledFrame,
    Track,
    make_instance_cattr,
)
from sleap.io.legacy import load_labels_json_old
from sleap.skeleton import Node, Skeleton
from sleap.util import json_loads, json_dumps, weak_filename_match


class LabelsJsonAdaptor(Adaptor):
    FORMAT_ID = 1

    @property
    def handles(self):
        return SleapObjectType.labels

    @property
    def default_ext(self):
        return "json"

    @property
    def all_exts(self):
        return ["json", "json.zip"]

    @property
    def name(self):
        return "Labels JSON"

    def can_read_file(self, file: FileHandle):
        if not self.does_match_ext(file.filename):
            print(f"{file.filename} doesn't match ext for json or json.zip")
            return False

        if file.filename.endswith(".zip"):
            # We can't check inside zip so assume it's correct
            return True

        if not file.is_json:
            return False
        if file.format_id not in (None, self.FORMAT_ID):
            return False
        return True

    def can_write_filename(self, filename: str):
        return self.does_match_ext(filename)

    def does_read(self) -> bool:
        return True

    def does_write(self) -> bool:
        return True

    @classmethod
    def read(
        cls,
        file: FileHandle,
        video_search: Union[Callable, List[Text], None] = None,
        match_to: Optional[Labels] = None,
        *args,
        **kwargs,
    ) -> Labels:
        pass

        """
        Deserialize JSON file as new :class:`Labels` instance.

        Args:
            filename: Path to JSON file.
            video_callback: A callback function that which can modify
                video paths before we try to create the corresponding
                :class:`Video` objects. Usually you'll want to pass
                a callback created by :meth:`make_video_callback`
                or :meth:`make_gui_video_callback`.
                Alternately, if you pass a list of strings we'll construct a
                non-gui callback with those strings as the search paths.
            match_to: If given, we'll replace particular objects in the
                data dictionary with *matching* objects in the match_to
                :class:`Labels` object. This ensures that the newly
                instantiated :class:`Labels` can be merged without
                duplicate matching objects (e.g., :class:`Video` objects ).
        Returns:
            A new :class:`Labels` object.
        """

        tmp_dir = None
        filename = file.filename

        # Check if the file is a zipfile for not.
        if zipfile.is_zipfile(filename):

            # Make a tmpdir, located in the directory that the file exists, to unzip
            # its contents.
            tmp_dir = os.path.join(
                os.path.dirname(filename),
                f"tmp_{os.getpid()}_{os.path.basename(filename)}",
            )
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            try:
                os.mkdir(tmp_dir)
            except FileExistsError:
                pass

            # tmp_dir = tempfile.mkdtemp(dir=os.path.dirname(filename))

            try:

                # Register a cleanup routine that deletes the tmpdir on program exit
                # if something goes wrong. The True is for ignore_errors
                atexit.register(shutil.rmtree, tmp_dir, True)

                # Uncompress the data into the directory
                shutil.unpack_archive(filename, extract_dir=tmp_dir)

                # We can now open the JSON file, save the zip file and
                # replace file with the first JSON file we find in the archive.
                json_files = [
                    os.path.join(tmp_dir, file)
                    for file in os.listdir(tmp_dir)
                    if file.endswith(".json")
                ]

                if len(json_files) == 0:
                    raise ValueError(
                        f"No JSON file found inside {filename}. Are you sure this is a valid sLEAP dataset."
                    )

                filename = json_files[0]

            except Exception as ex:
                # If we had problems, delete the temp directory and reraise the exception.
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

        # Open and parse the JSON in filename
        with open(filename, "r") as file:

            # FIXME: Peek into the json to see if there is version string.
            # We do this to tell apart old JSON data from leap_dev vs the
            # newer format for sLEAP.
            json_str = file.read()
            dicts = json_loads(json_str)

            # If we have a version number, then it is new sLEAP format
            if "version" in dicts:

                # Cache the working directory.
                cwd = os.getcwd()
                # Replace local video paths (for imagestore)
                if tmp_dir:
                    for vid in dicts["videos"]:
                        vid["backend"]["filename"] = os.path.join(
                            tmp_dir, vid["backend"]["filename"]
                        )

                # Use the video_callback for finding videos with broken paths:

                # 1. Accept single string as video search path
                if isinstance(video_search, str):
                    video_search = [video_search]

                # 2. Accept list of strings as video search paths
                if hasattr(video_search, "__iter__"):
                    # If the callback is an iterable, then we'll expect it to be a
                    # list of strings and build a non-gui callback with those as
                    # the search paths.
                    # When path is to a file, use the path of parent directory.
                    search_paths = [
                        os.path.dirname(path) if os.path.isfile(path) else path
                        for path in video_search
                    ]

                    # Make the search function from list of paths
                    video_search = Labels.make_video_callback(search_paths)

                # 3. Use the callback function (either given as arg or build from paths)
                if callable(video_search):
                    abort = video_search(dicts["videos"])
                    if abort:
                        raise FileNotFoundError

                # Try to load the labels filename.
                try:
                    labels = cls.from_json_data(dicts, match_to=match_to)

                except FileNotFoundError:

                    # FIXME: We are going to the labels JSON that has references to
                    # video files. Lets change directory to the dirname of the json file
                    # so that relative paths will be from this directory. Maybe
                    # it is better to feed the dataset dirname all the way down to
                    # the Video object. This seems like less coupling between classes
                    # though.
                    if os.path.dirname(filename) != "":
                        os.chdir(os.path.dirname(filename))

                    # Try again
                    labels = cls.from_json_data(dicts, match_to=match_to)

                except Exception as ex:
                    # Ok, we give up, where the hell are these videos!
                    raise  # Re-raise.
                finally:
                    os.chdir(cwd)  # Make sure to change back if we have problems.

                return labels

            else:
                frames = load_labels_json_old(data_path=filename, parsed_json=dicts)
                return Labels(frames)

    @classmethod
    def write(
        cls,
        filename: str,
        source_object: str,
        compress: Optional[bool] = None,
        save_frame_data: bool = False,
        frame_data_format: str = "png",
    ):
        """
        Save a Labels instance to a JSON format.

        Args:
            filename: The filename to save the data to.
            source_object: The labels dataset to save.
            compress: Whether the data be zip compressed or not? If True,
                the JSON will be compressed using Python's shutil.make_archive
                command into a PKZIP zip file. If compress is True then
                filename will have a .zip appended to it.
            save_frame_data: Whether to save the image data for each frame.
                For each video in the dataset, all frames that have labels
                will be stored as an imgstore dataset.
                If save_frame_data is True then compress will be forced to True
                since the archive must contain both the JSON data and image
                data stored in ImgStores.
            frame_data_format: If save_frame_data is True, then this argument
                is used to set the data format to use when writing frame
                data to ImgStore objects. Supported formats should be:

                 * 'pgm',
                 * 'bmp',
                 * 'ppm',
                 * 'tif',
                 * 'png',
                 * 'jpg',
                 * 'npy',
                 * 'mjpeg/avi',
                 * 'h264/mkv',
                 * 'avc1/mp4'

                 Note: 'h264/mkv' and 'avc1/mp4' require separate installation
                 of these codecs on your system. They are excluded from SLEAP
                 because of their GPL license.

        Returns:
            None
        """

        labels = source_object

        if compress is None:
            compress = filename.endswith(".zip")

        # Lets make a temporary directory to store the image frame data or pre-
        # compressed json in case we need it.
        with tempfile.TemporaryDirectory() as tmp_dir:

            # If we are saving frame data along with the datasets. We will replace
            # videos with new video object that represent video data from just the
            # labeled frames.
            if save_frame_data:

                # Create a set of new Video objects with imgstore backends. One for each
                # of the videos. We will only include the labeled frames though. We will
                # then replace each video with this new video
                new_videos = labels.save_frame_data_imgstore(
                    output_dir=tmp_dir, format=frame_data_format
                )

                # Make video paths relative
                for vid in new_videos:
                    tmp_path = vid.filename
                    # Get the parent dir of the YAML file.
                    # Use "/" since this works on Windows and posix
                    img_store_dir = (
                        os.path.basename(os.path.split(tmp_path)[0])
                        + "/"
                        + os.path.basename(tmp_path)
                    )
                    # Change to relative path
                    vid.backend.filename = img_store_dir

                # Convert to a dict, not JSON yet, because we need to patch up the
                # videos
                d = labels.to_dict()
                d["videos"] = Video.cattr().unstructure(new_videos)

            else:
                d = labels.to_dict()

            # Set file format version
            d["format_id"] = cls.FORMAT_ID

            if compress or save_frame_data:

                # Ensure that filename ends with .json
                # shutil will append .zip
                filename = re.sub(r"(\.json\.zip)$", ".json", filename)

                # Write the json to the tmp directory, we will zip it up with the frame
                # data.
                full_out_filename = os.path.join(tmp_dir, os.path.basename(filename))
                json_dumps(d, full_out_filename)

                # Create the archive
                shutil.make_archive(base_name=filename, root_dir=tmp_dir, format="zip")

            # If the user doesn't want to compress, then just write the json to the
            # filename
            else:
                json_dumps(d, filename)

    @classmethod
    def from_json_data(
        cls, data: Union[str, dict], match_to: Optional["Labels"] = None
    ) -> "Labels":
        """
        Create instance of class from data in dictionary.

        Method is used by other methods that load from JSON.

        Args:
            data: Dictionary, deserialized from JSON.
            match_to: If given, we'll replace particular objects in the
                data dictionary with *matching* objects in the match_to
                :class:`Labels` object. This ensures that the newly
                instantiated :class:`Labels` can be merged without
                duplicate matching objects (e.g., :class:`Video` objects ).
        Returns:
            A new :class:`Labels` object.
        """

        # Parse the json string if needed.
        if type(data) is str:
            dicts = json_loads(data)
        else:
            dicts = data

        dicts["tracks"] = dicts.get(
            "tracks", []
        )  # don't break if json doesn't include tracks

        # First, deserialize the skeletons, videos, and nodes lists.
        # The labels reference these so we will need them while deserializing.
        nodes = cattr.structure(dicts["nodes"], List[Node])

        idx_to_node = {i: nodes[i] for i in range(len(nodes))}
        skeletons = Skeleton.make_cattr(idx_to_node).structure(
            dicts["skeletons"], List[Skeleton]
        )
        videos = Video.cattr().structure(dicts["videos"], List[Video])

        try:
            # First try unstructuring tuple (newer format)
            track_cattr = cattr.Converter(
                unstruct_strat=cattr.UnstructureStrategy.AS_TUPLE
            )
            tracks = track_cattr.structure(dicts["tracks"], List[Track])
        except:
            # Then try unstructuring dict (older format)
            try:
                tracks = cattr.structure(dicts["tracks"], List[Track])
            except:
                raise ValueError("Unable to load tracks as tuple or dict!")

        # if we're given a Labels object to match, use its objects when they match
        if match_to is not None:
            if len(skeletons) > 1 or len(match_to.skeletons) > 1:
                # Match full skeletons
                for idx, sk in enumerate(skeletons):
                    for old_sk in match_to.skeletons:
                        if sk.matches(old_sk):
                            # use nodes from matched skeleton
                            for (node, match_node) in zip(sk.nodes, old_sk.nodes):
                                node_idx = nodes.index(node)
                                nodes[node_idx] = match_node
                            # use skeleton from match
                            skeletons[idx] = old_sk
                            break
            elif len(skeletons) == 1 and len(match_to.skeletons) == 1:
                # Match by node names
                old_skel = match_to.skeleton
                new_skel = skeletons[0]

                old_node_names = old_skel.node_names
                for i, node in enumerate(nodes):
                    if node.name in old_node_names:
                        nodes[i] = old_skel.nodes[old_node_names.index(node.name)]
                    else:
                        old_skel._graph.add_node(node)

                skeletons[0] = old_skel

            # Match videos
            for idx, vid in enumerate(videos):
                for old_vid in match_to.videos:

                    # Try to match videos using either their current or source filename
                    # if available.
                    old_vid_paths = [old_vid.filename]
                    if getattr(old_vid.backend, "has_embedded_images", False):
                        old_vid_paths.append(old_vid.backend._source_video.filename)

                    new_vid_paths = [vid.filename]
                    if getattr(vid.backend, "has_embedded_images", False):
                        new_vid_paths.append(vid.backend._source_video.filename)

                    is_match = False
                    for old_vid_path in old_vid_paths:
                        for new_vid_path in new_vid_paths:
                            if old_vid_path == new_vid_path or weak_filename_match(
                                old_vid_path, new_vid_path
                            ):
                                is_match = True
                                videos[idx] = old_vid
                                break
                        if is_match:
                            break
                    if is_match:
                        break

        suggestions = []
        if "suggestions" in dicts:
            suggestions_cattr = cattr.Converter()
            suggestions_cattr.register_structure_hook(
                Video, lambda x, type: videos[int(x)]
            )
            try:
                suggestions = suggestions_cattr.structure(
                    dicts["suggestions"], List[SuggestionFrame]
                )
            except Exception as e:
                print("Error while loading suggestions (1)")
                print(e)

                try:
                    # Convert old suggestion format to new format.
                    # Old format: {video: list of frame indices}
                    # New format: [SuggestionFrames]
                    old_suggestions = suggestions_cattr.structure(
                        dicts["suggestions"], Dict[Video, List]
                    )
                    for video in old_suggestions.keys():
                        suggestions.extend(
                            [
                                SuggestionFrame(video, idx)
                                for idx in old_suggestions[video]
                            ]
                        )
                except Exception as e:
                    print("Error while loading suggestions (2)")
                    print(e)
                    pass

        if "negative_anchors" in dicts:
            negative_anchors_cattr = cattr.Converter()
            negative_anchors_cattr.register_structure_hook(
                Video, lambda x, type: videos[int(x)]
            )
            negative_anchors = negative_anchors_cattr.structure(
                dicts["negative_anchors"], Dict[Video, List]
            )
        else:
            negative_anchors = dict()

        if "provenance" in dicts:
            provenance = dicts["provenance"]
        else:
            provenance = dict()

        # If there is actual labels data, get it.
        if "labels" in dicts:
            label_cattr = make_instance_cattr()
            label_cattr.register_structure_hook(
                Skeleton, lambda x, type: skeletons[int(x)]
            )
            label_cattr.register_structure_hook(Video, lambda x, type: videos[int(x)])
            label_cattr.register_structure_hook(
                Node, lambda x, type: x if isinstance(x, Node) else nodes[int(x)]
            )
            label_cattr.register_structure_hook(
                Track, lambda x, type: None if x is None else tracks[int(x)]
            )

            labels = label_cattr.structure(dicts["labels"], List[LabeledFrame])
        else:
            labels = []

        return Labels(
            labeled_frames=labels,
            videos=videos,
            skeletons=skeletons,
            nodes=nodes,
            suggestions=suggestions,
            negative_anchors=negative_anchors,
            tracks=tracks,
            provenance=provenance,
        )
