"""Adaptor to read and write analysis HDF5 files.

These contain location and track data, but lack other metadata included in a
full SLEAP dataset file.

This adaptor uses sleap-io for both reading and writing analysis HDF5 files,
providing a consistent format with additional metadata like dimension labels
and skeleton symmetries.

To determine whether this adaptor can read a file, we check it's an HDF5 file
with a `track_occupancy` dataset.
"""

from typing import Union

from sleap_io import Labels, Video
import sleap_io as sio

from .adaptor import Adaptor, SleapObjectType
from .filehandle import FileHandle


class SleapAnalysisAdaptor(Adaptor):
    @property
    def handles(self):
        return SleapObjectType.labels

    @property
    def default_ext(self):
        return "h5"

    @property
    def all_exts(self):
        return ["h5", "hdf5"]

    @property
    def name(self):
        return "SLEAP Analysis HDF5"

    def can_read_file(self, file: FileHandle):
        if not self.does_match_ext(file.filename):
            return False
        if not file.is_hdf5:
            return False
        if "track_occupancy" not in file.file:
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
        video: Union[Video, str],
        *args,
        **kwargs,
    ) -> Labels:
        """Reads analysis HDF5 file using sleap-io.

        Args:
            file: The file handle for the HDF5 file.
            video: The video to associate with the data. Can be a Video object
                or a path string.

        Returns:
            Labels object with loaded pose data.
        """
        # Use sleap-io's load_analysis_h5 which handles all format variants
        return sio.load_analysis_h5(file.filename, video=video)

    @classmethod
    def write(
        cls,
        filename: str,
        source_object: Labels,
        source_path: str = None,
        video: Video = None,
    ):
        """Writes analysis file for :py:class:`Labels` `source_object`.

        Args:
            filename: The filename for the output file.
            source_object: The :py:class:`Labels` from which to get data from.
            source_path: Path to the source labels file (stored as metadata).
            video: The :py:class:`Video` from which to get data from. If no `video` is
                specified, then the first video in `source_object` videos list will be
                used. If there are no :py:class:`LabeledFrame`s in the `video`,
                then no analysis file will be written.
        """
        try:
            sio.save_analysis_h5(
                source_object,
                filename,
                video=video,
                labels_path=source_path,
                all_frames=True,
                preset="matlab",  # SLEAP-compatible format
            )
        except ValueError as e:
            # Handle case where video has no labeled frames
            # sleap-io raises ValueError, but we silently skip like old behavior
            if "No labeled frames" in str(e):
                print("No labeled frames in video. Skipping analysis export.")
            else:
                raise
