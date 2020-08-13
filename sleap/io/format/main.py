"""
Read/write for multiple dataset formats.

File adaptors provide a common API for code that reads and/or writes data files.
Usually these are for reading or writing SLEAP datasets or something with
roughly equivalent data (e.g., a COCO keypoint dataset), although the code
can in principle be used for reading/writing different types of data.

For reading or writing SLEAP datasets, use `main.read()` or `main.write()`,
optionally specifying the `as_format` parameter (e.g., if you want to write a
specific, non-default format). `sleap.io.convert` is a nice usage example.

To add support for a new file format:

1. Create an adaptor class which implements all virtual functions in the
   `Adaptor` class. Take a look at `GenericJsonAdaptor` for a simple example
   or `SleapAnalysisAdaptor` for an adaptor which supports reading and writing
   datasets (this would be a good adaptor to use as a template for your own).

2. If it's for reading and/or writing `Labels` datasets (the typical case),
   add it to `all_labels_adaptors` dictionary in `main.py`.

If your file format has a file extension that's distinct from other
supported file formats, then read/write code will automatically detect the
correct format. For example, if your adaptor supports save and its default file
ext is `.foo`, then calling `Labels.save_file(labels, "filename.foo")` will use
your file adaptor.

If your file format does not have a distinct file extension, then additional
work is required. For an example, take a look at the `ExportAnalysisFile` and
`ImportAnalysisFile` command classes (in `sleap.gui.commands`). For the analysis
HDF5 we need custom code since these files have a `.h5` extension, and this is
also a non-default file extension for the `LabelsV1Adaptor` adaptor.
"""

from .coco import LabelsCocoAdaptor
from .deeplabcut import LabelsDeepLabCutCsvAdaptor, LabelsDeepLabCutYamlAdaptor
from .deepposekit import LabelsDeepPoseKitAdaptor
from .hdf5 import LabelsV1Adaptor
from .labels_json import LabelsJsonAdaptor
from .leap_matlab import LabelsLeapMatlabAdaptor
from .sleap_analysis import SleapAnalysisAdaptor

from . import adaptor, dispatch, filehandle

from typing import Text, Optional, Union


# Default adaptors to use when input/output format isn't specified.
default_labels_adaptors = [LabelsV1Adaptor, LabelsJsonAdaptor]

# All supported adaptors for reading and/or writing SLEAP datasets.
# Key is string used to specify format (`as_format` param), value is either an
# adaptor class (i.e., class which inherits from `adaptor.Adaptor`) or a list
# of adaptor classes.
all_labels_adaptors = {
    "hdf5_v1": LabelsV1Adaptor,
    "json": LabelsJsonAdaptor,
    "leap": LabelsLeapMatlabAdaptor,
    "deeplabcut": (LabelsDeepLabCutYamlAdaptor, LabelsDeepLabCutCsvAdaptor),
    "deepposekit": LabelsDeepPoseKitAdaptor,
    "coco": LabelsCocoAdaptor,
    "analysis": SleapAnalysisAdaptor,
}


def read(
    filename: Text,
    for_object: Union[Text, object],
    as_format: Optional[Text] = None,
    *args,
    **kwargs,
) -> object:
    """
    Reads file using the appropriate file format adaptor.

    Args:
        filename: Full filename of the file to read.
        for_object: The type of object we're trying to read; can be given as
            string (e.g., "labels") or instance of the object.
        as_format: Allows you to specify the format adaptor to use;
            if not specified, then we'll try the default adaptors for this
            object type.

    Exceptions:
        NotImplementedError if appropriate adaptor cannot be found.

        TypeError if adaptor does not support reading
        (shouldn't happen unless you specify `as_format` adaptor).

        Any file-related exception thrown while trying to read.
    """

    disp = dispatch.Dispatch()

    if as_format in all_labels_adaptors:

        disp.register(all_labels_adaptors[as_format])
        return disp.read(filename, *args, **kwargs)

    if for_object == "labels" or hasattr(for_object, "labeled_frames"):
        if as_format == "*":
            for format_name, adaptor in all_labels_adaptors.items():
                disp.register(adaptor)
                # print(f"[registering format adaptor for {format_name}]")
        else:
            disp.register_list(default_labels_adaptors)

        return disp.read(filename, *args, **kwargs)

    raise NotImplementedError("No adaptors for this object type.")


def write(
    filename: str,
    source_object: object,
    as_format: Optional[Text] = None,
    *args,
    **kwargs,
):
    """
    Writes SLEAP dataset file using the appropriate file format adaptor.

    Args:
        filename: Full filename of the file to write.
            All directories should exist.
        source_object: The object we want to write to a file.
        as_format: Allows you to specify the format adaptor to use;
            if not specified, then this will use the privileged adaptor for
            the type of object.

    Exceptions:
        NotImplementedError if appropriate adaptor cannot be found.

        TypeError if adaptor does not support writing
        (shouldn't happen unless you specify `as_format` adaptor).

        Any file-related exception thrown while trying to write.
    """
    disp = dispatch.Dispatch()

    # User specified known output format
    if as_format in all_labels_adaptors:
        # Register adaptors which support this format
        disp.register(all_labels_adaptors[as_format])
        # Write file using dispatch (which finds best adaptor)
        return disp.write(filename, source_object, *args, **kwargs)

    elif as_format is not None:
        raise KeyError(f"No adaptor for {as_format}.")

    # If we're still here, then user didn't specify output format.

    # Check if we're trying to save a SLEAP dataset (i.e., `Labels` object)
    # and if so, register default adaptors and try writing.
    if hasattr(source_object, "labeled_frames"):
        disp.register_list(default_labels_adaptors)
        return disp.write(filename, source_object, *args, **kwargs)

    raise NotImplementedError(
        f"No adaptors for object type {type(source_object)} ({as_format})."
    )
