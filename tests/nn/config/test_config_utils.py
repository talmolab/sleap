import attr
import pytest
from typing import Optional, Text

from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test

from sleap.nn.config import utils


def test_one_of():
    @utils.oneof
    @attr.s(auto_attribs=True)
    class ExclusiveClass:
        a: Optional[Text] = None
        b: Optional[Text] = None

    c = ExclusiveClass(a="hello")

    assert c.which_oneof_attrib_name() == "a"
    assert c.which_oneof() == "hello"

    with pytest.raises(ValueError):
        c = ExclusiveClass(a="hello", b="too many values!")
