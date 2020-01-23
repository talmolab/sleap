import os
from typing import Optional

import attr
import h5py

from sleap.util import json_loads


@attr.s(auto_attribs=True)
class FileHandle(object):
    """Reference to a file; can hold loaded data so it needn't be read twice."""

    filename: str
    _is_hdf5: bool = False
    _is_json: Optional[bool] = None
    _is_open: bool = False
    _file: object = None
    _text: str = None
    _json: object = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def open(self):
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Could not find {self.filename}")

        if self._file is None:
            try:
                self._file = h5py.File(self.filename, "r")
                self._is_hdf5 = True
            except OSError as e:
                # We get OSError when trying to read non-HDF5 file with h5py
                pass

        if self._file is None:
            self._file = open(self.filename, "r")
            self._is_hdf5 = False

    def close(self):
        if self._file is not None:
            self._file.close()

    @property
    def file(self):
        self.open()
        return self._file

    @property
    def text(self):
        if self._text is None:
            self._text = self.file.read()
        return self._text

    @property
    def json(self):
        if self._json is None:
            self._json = json_loads(self.text)
        return self._json

    @property
    def is_json(self):
        if self._is_json is None:
            try:
                self.json
                self._is_json = True
            except Exception as e:
                self._is_json = False
        return self._is_json

    @property
    def is_hdf5(self):
        self.open()
        return self._is_hdf5

    @property
    def format_id(self):
        if self.is_hdf5:
            if "metadata" in self.file:
                meta_group = self.file.require_group("metadata")
                if "format_id" in meta_group.attrs:
                    return meta_group.attrs["format_id"]

        elif self.is_json:
            if "format_id" in self.json:
                return self.json["format_id"]

        return None
