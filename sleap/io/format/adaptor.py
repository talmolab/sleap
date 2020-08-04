"""
File format adaptor base class.
"""

from enum import Enum
from typing import List

import attr

from sleap.io.format.filehandle import FileHandle


class SleapObjectType(Enum):
    """Types of files that an adaptor could read/write."""

    misc = 0
    labels = 1


@attr.s(auto_attribs=True)
class Adaptor:
    """
    File format adaptor base class.

    An adaptor handles reading and/or writing a specific file format. To add
    support for a new file format, you'll create a new class which inherits from
    the Adaptor base class and implements the relevant functions.
    """

    @property
    def handles(self) -> SleapObjectType:
        """
        Returns the type of object that can be read/written.

        The Dispatch class calls this method on all registered adaptors to
        determine which to use for reading/writing.
        """
        raise NotImplementedError

    @property
    def default_ext(self) -> str:
        """The default file extension, e.g., 'json' (without '.')."""
        raise NotImplementedError

    @property
    def all_exts(self) -> List[str]:
        """List of all file extensions supported by adaptor."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Human-reading name of the file format"""
        raise NotImplementedError

    def can_read_file(self, file: FileHandle) -> bool:
        """Returns whether this adaptor can read this file."""
        raise NotImplementedError

    def can_write_filename(self, filename: str) -> bool:
        """Returns whether this adaptor can write format of this filename."""
        raise NotImplementedError

    def does_read(self) -> bool:
        """Returns whether this adaptor supports reading."""
        raise NotImplementedError

    def does_write(self) -> bool:
        """Returns whether this adaptor supports writing."""
        raise NotImplementedError

    def read(self, file: FileHandle) -> object:
        """Reads the file and returns the appropriate deserialized object."""
        raise NotImplementedError

    def write(self, filename: str, source_object: object):
        """Writes the object to a file."""
        raise NotImplementedError

    # Methods with default implementation

    def does_match_ext(self, filename: str) -> bool:
        """Returns whether this adaptor can write format of this filename."""

        # We don't match the ext against the result of os.path.splitext because
        # we want to match extensions like ".json.zip".

        return filename.endswith(tuple(self.all_exts))

    @property
    def formatted_ext_options(self):
        """String for Qt file dialog extension options."""
        return f"{self.name} ({' '.join(self.all_exts)})"
