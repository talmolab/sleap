"""
Adaptor for reading and writing any generic text file.

This is a good example of a very simple adaptor class.
"""

from .adaptor import Adaptor, SleapObjectType
from .filehandle import FileHandle


class TextAdaptor(Adaptor):
    @property
    def handles(self):
        return SleapObjectType.misc

    @property
    def default_ext(self):
        return "txt"

    @property
    def all_exts(self):
        return ["txt", "log"]

    @property
    def name(self):
        return "Text file"

    def can_read_file(self, file: FileHandle):
        return True

    def can_write_filename(self, filename: str) -> bool:
        return True

    def does_read(self) -> bool:
        return True

    def does_write(self) -> bool:
        return True

    def read(self, file: FileHandle, *args, **kwargs):
        return file.text

    def write(self, filename: str, source_object: str):
        with open(filename, "w") as f:
            f.write(source_object)
