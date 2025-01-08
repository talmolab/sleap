"""Module to test all overlays in the sleap/gui/overlays/base.py."""

from sleap.gui.overlays.base import DataOverlay, ModelData


def test_data_overlay(qtbot, min_bottomup_model_path, centered_pair_vid):
    """Test the data overlay."""

    model_path = min_bottomup_model_path
    video = centered_pair_vid

    predictor = DataOverlay.make_predictor(filename=model_path)

    overlay = DataOverlay.from_predictor(
        predictor=predictor,
        video=video,
        show_pafs=True,
    )


if __name__ == "__main__":
    import pytest

    pytest.main([f"{__file__}::test_data_overlay"])
