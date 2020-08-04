"""
Dispatcher for dynamically supporting multiple dataset file formats.

See `read()` and `write()` in `sleap.io.format` for usage.
"""

import attr
from typing import List, Optional, Tuple, Union

from sleap.io.format.adaptor import Adaptor, SleapObjectType
from sleap.io.format.filehandle import FileHandle


@attr.s(auto_attribs=True)
class Dispatch(object):

    _adaptors: List[Adaptor] = attr.ib(default=attr.Factory(list))

    def register(self, adaptor: Union[Adaptor, type, List[Adaptor]]):
        """
        Registers the class which reads/writes specific file format.
        """
        # If given a class, then instantiate it since we want the object
        if hasattr(adaptor, "__iter__"):
            self.register_list(adaptor)

        else:
            if type(adaptor) == type:
                adaptor = adaptor()

            self._adaptors.append(adaptor)

    def register_list(self, adaptor_list: List[Union[Adaptor, type]]):
        """Convenience function for registering multiple adaptors."""
        for adaptor in adaptor_list:
            self.register(adaptor)

    def get_formatted_ext_options(self) -> List[str]:
        """
        Returns the file extensions that can be used for specified type.

        This is used for determining which extensions to list in save dialog.
        """
        return [adaptor.formatted_ext_options for adaptor in self._adaptors]

    def open(self, filename: str) -> FileHandle:
        """Returns FileHandle for file."""
        return FileHandle(filename)

    def read(self, filename: str, *args, **kwargs) -> object:
        """Reads file and returns the deserialized object."""

        with self.open(filename) as file:
            for adaptor in self._adaptors:
                if adaptor.can_read_file(file):
                    return adaptor.read(file, *args, **kwargs)

        raise TypeError("No file format adaptor could read this file.")

    def read_safely(self, *args, **kwargs) -> Tuple[object, Optional[BaseException]]:
        """Wrapper for reading file without throwing exception."""
        try:
            return self.read(*args, **kwargs), None
        except Exception as e:
            return None, e

    def write(self, filename: str, source_object: object, *args, **kwargs):
        """
        Writes an object to a file.

        Args:
            filename: The full name (including path) of the file to write.
            source_object: The object to write.
        """

        for adaptor in self._adaptors:
            if adaptor.can_write_filename(filename):
                return adaptor.write(filename, source_object, *args, **kwargs)

        raise TypeError("No file format adaptor could write this file.")

    def write_safely(self, *args, **kwargs) -> Optional[BaseException]:
        """Wrapper for writing file without throwing exception."""
        try:
            self.write(*args, **kwargs)
            return None
        except Exception as e:
            return e

    @classmethod
    def make_dispatcher(cls, object_type: SleapObjectType) -> "Dispatch":
        """Factory method which automatically registers some adaptors."""
        dispatcher = cls()
        if object_type == SleapObjectType.labels:
            from .hdf5 import LabelsV1Adaptor
            from .labels_json import LabelsJsonAdaptor
            from .deeplabcut import LabelsDeepLabCutCsvAdaptor

            dispatcher.register(LabelsV1Adaptor())
            dispatcher.register(LabelsJsonAdaptor())
            dispatcher.register(LabelsDeepLabCutCsvAdaptor())

        elif object_type == SleapObjectType.misc:
            from .text import TextAdaptor

            dispatcher.register(TextAdaptor())

        return dispatcher
