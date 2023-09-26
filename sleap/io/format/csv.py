"""Adaptor for writing SLEAP analysis as csv."""

from sleap.io import format

from sleap import Labels, Video
from typing import Optional, Callable, List, Text, Union


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
        """Writes csv file for :py:class:`Labels` `source_object`.

        Args:
            filename: The filename for the output file.
            source_object: The :py:class:`Labels` from which to get data from.
            source_path: Path for the labels object
            video: The :py:class:`Video` from which toget data from. If no `video` is
                specified, then the first video in `source_object` videos list will be
                used. If there are no :py:class:`Labeled Frame`s in the `video`, then no
                analysis file will be written.
        """
        from sleap.info.write_tracking_h5 import main as write_analysis

        write_analysis(
            labels=source_object,
            output_path=filename,
            labels_path=source_path,
            all_frames=True,
            video=video,
            csv=True,
        )
