from sleap.gui.learning.configs import TrainingConfigsGetter
from sleap.gui.learning.utils import ScopedKeyDict


def test_config_list_load():
    configs = TrainingConfigsGetter.make_from_labels_filename("").get_filtered_configs(
        "centroid"
    )

    assert 1 == len(configs)


def test_scoped_key_dict():
    d = {"foo": 1, "bar": {"cat": {"dog": 2}, "elephant": 3}}

    x = ScopedKeyDict.from_hierarchical_dict(d).key_val_dict

    assert x["foo"] == 1
    assert x["bar.cat.dog"] == 2
    assert x["bar.elephant"] == 3
