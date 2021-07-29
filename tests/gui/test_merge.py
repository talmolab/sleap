from sleap.gui.dialogs.merge import show_instance_type_counts


def test_count_string(simple_predictions):
    assert show_instance_type_counts(simple_predictions[0]) == "0 (user) / 2 (pred)"
