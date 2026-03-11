"""Adaptor for writing SLEAP analysis as CSV.

This adaptor uses sleap-io for CSV export, providing a consistent format
with the analysis HDF5 export.
"""

from sleap.io import format

from sleap_io import Labels, Video
import sleap_io as sio


class CSVAdaptor(format.adaptor.Adaptor):
    FORMAT_ID = 1.0

    # 1.0 initial implementation

    @property
    def handles(self):
        return format.adaptor.SleapObjectType.labels

    @property
    def default_ext(self):
        return "csv"

    @property
    def all_exts(self):
        return ["csv", "xlsx"]

    @property
    def name(self):
        return "CSV"

    def can_read_file(self, file: format.filehandle.FileHandle):
        return False

    def can_write_filename(self, filename: str):
        return self.does_match_ext(filename)

    def does_read(self) -> bool:
        return False

    def does_write(self) -> bool:
        return True

    @classmethod
    def write(
        cls,
        filename: str,
        source_object: Labels,
        source_path: str = None,
        video: Video = None,
    ):
        """Writes CSV file for :py:class:`Labels` `source_object`.

        Args:
            filename: The filename for the output file.
            source_object: The :py:class:`Labels` from which to get data from.
            source_path: Path for the labels object (stored as metadata).
            video: The :py:class:`Video` from which to get data from. If no `video` is
                specified, then the first video in `source_object` videos list will be
                used. If there are no :py:class:`LabeledFrame`s in the `video`,
                then no analysis file will be written.
        """
        # Resolve video
        if video is None:
            video = source_object.videos[0] if source_object.videos else None

        # Check for labeled frames before exporting (sleap-io may not raise error)
        if video is not None:
            labeled_frames = source_object.find(video)
            if not labeled_frames:
                print("No labeled frames in video. Skipping CSV export.")
                return

        sio.save_csv(
            source_object,
            filename,
            format="sleap",
            video=video,
            include_score=True,
            include_empty=True,  # Include all frames from 0 to last labeled
            save_metadata=True,
        )
