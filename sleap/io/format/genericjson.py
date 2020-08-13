"""
Adaptor for reading and writing any generic JSON file.

This is a good example of a very simple adaptor class.
"""

from .adaptor import Adaptor, SleapObjectType
from .filehandle import FileHandle

from sleap.util import json_dumps


class GenericJsonAdaptor(Adaptor):
    @property
    def handles(self):
        return SleapObjectType.misc

    @property
    def default_ext(self):
        return "json"

    @property
    def all_exts(self):
        return ["json", "txt"]

    @property
    def name(self):
        return "JSON file"

    def can_read_file(self, file: FileHandle):
        if not self.does_match_ext(file.filename):
            return False
        return file.is_json

    def can_write_filename(self, filename: str) -> bool:
        return True

    def does_read(self) -> bool:
        return True

    def does_write(self) -> bool:
        return True

    def read(self, file: FileHandle, *args, **kwargs):
        return file.json

    def write(self, filename: str, source_object: dict):
        json_dumps(source_object, filename)
