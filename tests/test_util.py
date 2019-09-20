import numpy as np
import attr
import pytest

from typing import List, Dict

from sleap.util import attr_to_dtype, frame_list, weak_filename_match

def test_attr_to_dtype():
    """
    Test that we can convert classes with basic types to numpy composite
    dtypes.
    """
    @attr.s
    class TestAttr:
        a: int = attr.ib()
        b: float = attr.ib()
        c: bool = attr.ib()
        d: str = attr.ib()

    @attr.s
    class TestAttr2:
        a: int = attr.ib()
        b = attr.ib()  # No type annotation!

    @attr.s
    class TestAttr3:
        a: int = attr.ib()
        b: List = attr.ib()  # List should throw exception!
        c: Dict = attr.ib()  # Dict should throw exception!

    dtype = attr_to_dtype(TestAttr)
    dtype.fields['a'][0] == np.dtype(int)
    dtype.fields['b'][0] == np.dtype(float)
    dtype.fields['c'][0] == np.dtype(bool)
    dtype.fields['d'][0] == np.dtype(object)

    with pytest.raises(TypeError):
        attr_to_dtype(TestAttr2)

    with pytest.raises(TypeError):
        attr_to_dtype(TestAttr3)

def test_frame_list():
    assert frame_list("3-5") == [3,4,5]
    assert frame_list("7,10") == [7,10]

def test_weak_match():
    assert weak_filename_match("one/two", "one/two")
    assert weak_filename_match(
        "M:\\code\\sandbox\\sleap_nas\\pilot_6pts\\tmp_11576_FoxP1_6pts.training.n=468.json.zip\\frame_data_vid0\\metadata.yaml",
        "D:\\projects\\code\\sandbox\\sleap_nas\\pilot_6pts\\tmp_99713_FoxP1_6pts.training.n=468.json.zip\\frame_data_vid0\\metadata.yaml")
    assert weak_filename_match("zero/one/two/three.mp4","other\\one\\two\\three.mp4")

    assert not weak_filename_match("one/two/three", "two/three")
    assert not weak_filename_match("one/two/three.mp4","one/two/three.avi")
    assert not weak_filename_match("foo.mp4","bar.mp4")
