from .coco import LabelsCocoAdaptor
from .deeplabcut import LabelsDeepLabCutCsvAdaptor, LabelsDeepLabCutYamlAdaptor
from .deepposekit import LabelsDeepPoseKitAdaptor
from .hdf5 import LabelsV1Adaptor
from .labels_json import LabelsJsonAdaptor
from .leap_matlab import LabelsLeapMatlabAdaptor
from .sleap_analysis import SleapAnalysisAdaptor

from . import adaptor, dispatch, filehandle

from typing import Text, Optional, Union

default_labels_adaptors = [LabelsV1Adaptor, LabelsJsonAdaptor]

all_labels_adaptors = {
    "hdf5_v1": LabelsV1Adaptor,
    "json": LabelsJsonAdaptor,
    "leap": LabelsLeapMatlabAdaptor,
    # "deeplabcut_csv": LabelsDeepLabCutCsvAdaptor,
    # "deeplabcut_yaml": LabelsDeepLabCutYamlAdaptor,
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
):
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
    Writes file using the appropriate file format adaptor.

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

    if as_format in all_labels_adaptors:
        disp.register(all_labels_adaptors[as_format])
        return disp.write(filename, source_object, *args, **kwargs)

    elif as_format is not None:
        raise KeyError(f"No adaptor for {as_format}.")

    if hasattr(source_object, "labeled_frames"):
        disp.register_list(default_labels_adaptors)
        return disp.write(filename, source_object, *args, **kwargs)

    raise NotImplementedError(
        f"No adaptors for object type {type(source_object)} ({as_format})."
    )
