"""Tests for sleap.gui.learning.receptivefield crop size functions."""

import numpy as np
import sleap_io as sio

import pytest

from sleap.gui.learning.receptivefield import (
    DEFAULT_MIN_CROP_SIZE,
    compute_augmentation_padding,
    compute_eff_scale,
    find_instance_crop_size,
    find_max_instance_bbox_size,
    iter_required_crop_sizes,
    resolve_anchor_ind,
    resolve_centroid_method,
    resolve_crop_centroid,
    resolve_crop_inputs,
    resolve_max_hw,
)


def _video(filename="test.mp4", height=None, width=None):
    """A video whose shape is readable without a backend."""
    if height is None or width is None:
        return sio.Video(filename=filename)
    return sio.Video(
        filename=filename, backend_metadata={"shape": (10, height, width, 1)}
    )


def _make_labels(user_points, predicted_points=None, nodes=("a", "b"), video=None):
    """Build a Labels object with user and optionally predicted instances."""
    skeleton = sio.Skeleton(nodes=[sio.Node(n) for n in nodes])
    video = video if video is not None else _video()
    frames = []
    for frame_idx, pts in enumerate(user_points):
        instances = [sio.Instance.from_numpy(np.array(pts), skeleton=skeleton)]
        if predicted_points and frame_idx < len(predicted_points):
            instances.append(
                sio.PredictedInstance.from_numpy(
                    np.array(predicted_points[frame_idx]),
                    skeleton=skeleton,
                    point_scores=np.ones(len(predicted_points[frame_idx])),
                )
            )
        frames.append(
            sio.LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
        )
    return sio.Labels(labeled_frames=frames)


class TestFindMaxInstanceBboxSize:
    def test_basic(self):
        labels = _make_labels([[[0, 0], [100, 50]]])
        assert find_max_instance_bbox_size(labels) == 100.0

    def test_skips_predicted_instances(self):
        """Predicted instances with large bboxes should not inflate the result."""
        labels = _make_labels(
            user_points=[[[0, 0], [100, 50]]],
            predicted_points=[[[0, 0], [800, 800]]],
        )
        assert find_max_instance_bbox_size(labels) == 100.0

    def test_multiple_frames(self):
        labels = _make_labels([[[0, 0], [50, 50]], [[0, 0], [120, 30]]])
        assert find_max_instance_bbox_size(labels) == 120.0


class TestFindInstanceCropSize:
    def test_basic_stride_rounding(self):
        labels = _make_labels([[[0, 0], [100, 50]]])
        assert find_instance_crop_size(labels, maximum_stride=16) == 112

    def test_skips_predicted_instances(self):
        """Predicted instances should not affect crop size computation."""
        labels = _make_labels(
            user_points=[[[0, 0], [100, 50]]],
            predicted_points=[[[0, 0], [800, 800]]],
        )
        assert find_instance_crop_size(labels, maximum_stride=16) == 112

    def test_with_padding(self):
        labels = _make_labels([[[0, 0], [100, 50]]])
        assert find_instance_crop_size(labels, padding=20, maximum_stride=16) == 128

    def test_min_crop_size_is_a_floor_not_an_early_return(self):
        """min_crop_size raises a small crop, but never caps a large one.

        A value divisible by the stride used to return immediately without
        reading the labels, pinning the crop however large the animals were
        (sleap-nn #748).
        """
        small = _make_labels([[[0, 0], [100, 50]]])
        assert find_instance_crop_size(small, maximum_stride=16, min_crop_size=256) == (
            256
        )

        large = _make_labels([[[0, 0], [300, 300]]])
        assert find_instance_crop_size(large, maximum_stride=16, min_crop_size=96) == (
            304
        )

    def test_min_crop_size_applies_with_nothing_to_measure(self):
        """The floor is applied outside the per-instance loop.

        Applying it inside meant an empty project, or one whose only instance is
        all-NaN, produced a 0px crop size.
        """
        empty = sio.Labels(labeled_frames=[])
        assert find_instance_crop_size(empty, maximum_stride=16, min_crop_size=100) == (
            112
        )
        assert (
            find_instance_crop_size(
                empty, maximum_stride=16, min_crop_size=100, padding=30
            )
            == 112
        )

        all_nan = _make_labels([[[np.nan, np.nan], [np.nan, np.nan]]])
        assert (
            find_instance_crop_size(all_nan, maximum_stride=16, min_crop_size=100)
            == 112
        )

    def test_with_augmentation_padding_matches_sleap_nn(self):
        """Crop size with aug padding should match sleap-nn's computation."""
        from sleap_nn.data.instance_cropping import compute_augmentation_padding

        labels = _make_labels([[[0, 0], [100, 50]]])
        bbox_size = find_max_instance_bbox_size(labels)
        padding = compute_augmentation_padding(
            bbox_size, rotation_max=180.0, scale_max=1.1
        )
        crop_size = find_instance_crop_size(labels, padding=padding, maximum_stride=16)
        assert crop_size == 160


class TestAnchorAwareSizing:
    """Crops are centered on the centroid, not the bbox midpoint.

    So an instance needs twice its greatest node offset from that center, which
    only equals its bbox extent when the center sits mid-bbox (sleap-nn #748).
    """

    def test_centered_anchor_matches_bbox_extent(self):
        """When the anchor sits mid-bbox, the old bbox criterion was correct."""
        labels = _make_labels([[[0, 0], [50, 0], [100, 0]]], nodes=("a", "b", "c"))
        assert find_max_instance_bbox_size(labels) == 100.0
        assert list(iter_required_crop_sizes(labels, anchor_ind=1)) == [100.0]
        assert find_instance_crop_size(labels, maximum_stride=16, anchor_ind=1) == 112

    def test_anchor_at_one_end_doubles_the_crop(self):
        """An anchor on an end node needs 2x the extent to reach the far one."""
        labels = _make_labels([[[0, 0], [50, 0], [100, 0]]], nodes=("a", "b", "c"))
        assert list(iter_required_crop_sizes(labels, anchor_ind=0)) == [200.0]
        assert find_instance_crop_size(labels, maximum_stride=16, anchor_ind=0) == 208

    def test_unresolvable_anchor_degrades_to_the_fallback(self):
        """An anchor index off the end of the skeleton must not crash."""
        labels = _make_labels([[[0, 0], [100, 50]]])
        required = list(iter_required_crop_sizes(labels, anchor_ind=99))
        assert required == list(iter_required_crop_sizes(labels))

    def test_occluded_anchor_uses_the_fallback(self):
        """A NaN anchor node falls back rather than yielding a NaN center."""
        labels = _make_labels(
            [[[np.nan, np.nan], [50, 0], [100, 0]]], nodes=("a", "b", "c")
        )
        # bbox_center of the visible nodes is (75, 0), so the reach is 25.
        required = list(
            iter_required_crop_sizes(
                labels,
                anchor_ind=0,
                centroid_method="anchor",
                centroid_fallback="bbox_center",
            )
        )
        assert required == [50.0]

    def test_empty_instance_is_skipped(self):
        """An all-NaN instance is filtered out before it is measured."""
        labels = _make_labels([[[np.nan, np.nan], [np.nan, np.nan]]])
        assert list(iter_required_crop_sizes(labels)) == []

    def test_instance_with_no_definable_centroid_is_skipped(self):
        """A non-empty instance can still have no centroid.

        `geometric_median` needs both coordinates of a point, so an instance
        whose every point is half-NaN is not empty but has no center to
        measure a reach from.
        """
        labels = _make_labels([[[5, 10], [5, 40]]])
        instance = labels[0].instances[0]
        # sleap-io derives visibility from x, so a half-NaN point has to be
        # written directly; `from_numpy` would mark it invisible instead.
        instance.points["xy"][:, 0] = np.nan
        instance.points["visible"][:] = True

        assert not instance.is_empty
        assert (
            list(iter_required_crop_sizes(labels, centroid_method="geometric_median"))
            == []
        )


class TestReduceMethodErrors:
    def test_unknown_reduce_method_raises(self):
        """A bad method name must not silently measure the wrong thing."""
        labels = _make_labels([[[0, 0], [100, 50]]])
        with pytest.raises(ValueError, match="Unknown centroid reduce method"):
            list(iter_required_crop_sizes(labels, centroid_method="nonsense"))

    def test_labels_without_a_skeleton_have_no_anchor(self):
        empty = sio.Labels(labeled_frames=[])
        assert resolve_anchor_ind(empty, "a") is None

    @pytest.mark.parametrize(
        "method", ["center_of_mass", "bbox_center", "geometric_median"]
    )
    def test_each_centroid_method_is_supported(self, method):
        """Every method in the vocabulary yields a finite required size."""
        pts = [[10, 10], [200, 12], [33, 90], [140, 250], [7, 300], [5, 5], [180, 30]]
        labels = _make_labels([pts], nodes=tuple("abcdefg"))
        (required,) = iter_required_crop_sizes(labels, centroid_method=method)
        assert np.isfinite(required) and required > 0


class TestSizeMatchedMeasurement:
    """Cropping happens after the size matcher rescales every frame."""

    def test_upscaled_video_measures_larger(self):
        """A lower-resolution video is upscaled, so its animal needs more crop.

        A 2048px video with a 200px animal plus a 1024px video with a 150px
        animal gave a crop of 200, but the second is upscaled 2x, so its animal
        spans 300px and ran off the edge.
        """
        skeleton = sio.Skeleton(nodes=[sio.Node("a"), sio.Node("b")])
        big = _video("big.mp4", 2048, 2048)
        small = _video("small.mp4", 1024, 1024)
        labels = sio.Labels(
            labeled_frames=[
                sio.LabeledFrame(
                    video=big,
                    frame_idx=0,
                    instances=[
                        sio.Instance.from_numpy(
                            np.array([[0, 0], [200, 100]]), skeleton=skeleton
                        )
                    ],
                ),
                sio.LabeledFrame(
                    video=small,
                    frame_idx=0,
                    instances=[
                        sio.Instance.from_numpy(
                            np.array([[0, 0], [150, 75]]), skeleton=skeleton
                        )
                    ],
                ),
            ]
        )

        # Native pixels: the 2048px video's 200px animal is the largest.
        assert find_max_instance_bbox_size(labels) == 200.0

        # Size matched to 2048: the 1024px video's animal spans 300px.
        assert find_max_instance_bbox_size(labels, max_hw=(2048, 2048)) == 300.0
        assert (
            find_instance_crop_size(
                labels, maximum_stride=16, max_hw=(2048, 2048), min_crop_size=0
            )
            == 304
        )

    def test_unreadable_video_degrades_to_native_pixels(self):
        """A video with no readable shape must not crash the preview."""
        labels = _make_labels([[[0, 0], [100, 50]]], video=_video("missing.mp4"))
        assert (
            find_instance_crop_size(
                labels, maximum_stride=16, max_hw=(2048, 2048), min_crop_size=0
            )
            == 112
        )


class TestComputeEffScale:
    def test_no_target_is_identity(self):
        assert compute_eff_scale((600, 800), None) == 1.0
        assert compute_eff_scale((600, 800), (600, 800)) == 1.0

    def test_uses_the_smaller_ratio(self):
        """Aspect ratio is preserved, so the limiting axis wins."""
        assert compute_eff_scale((600, 1200), (1024, 1280)) == pytest.approx(
            1280 / 1200
        )

    def test_none_axis_keeps_native_size(self):
        """A `None` axis is read as the frame's own size, so it never limits."""
        assert compute_eff_scale((600, 800), (None, 800)) == 1.0
        # Height could double, but the unset width axis holds the scale at 1.0
        # because aspect ratio is preserved.
        assert compute_eff_scale((600, 800), (1200, None)) == 1.0
        assert compute_eff_scale((600, 800), (1200, 1600)) == pytest.approx(2.0)


class TestResolveMaxHw:
    def test_configured_values_win(self):
        labels = _make_labels([[[0, 0], [1, 1]]], video=_video("a.mp4", 100, 100))
        assert resolve_max_hw(labels, 512, 640) == (512, 640)

    def test_derived_from_videos_like_the_trainer(self):
        """An unset target is filled from the largest video, as training does."""
        skeleton = sio.Skeleton(nodes=[sio.Node("a")])
        labels = sio.Labels(
            labeled_frames=[
                sio.LabeledFrame(
                    video=_video("a.mp4", 600, 800),
                    frame_idx=0,
                    instances=[
                        sio.Instance.from_numpy(np.array([[1, 1]]), skeleton=skeleton)
                    ],
                ),
                sio.LabeledFrame(
                    video=_video("b.mp4", 1024, 500),
                    frame_idx=0,
                    instances=[
                        sio.Instance.from_numpy(np.array([[1, 1]]), skeleton=skeleton)
                    ],
                ),
            ]
        )
        assert resolve_max_hw(labels) == (1024, 800)

    def test_no_readable_shape_gives_none(self):
        labels = _make_labels([[[0, 0], [1, 1]]], video=_video("missing.mp4"))
        assert resolve_max_hw(labels) is None
        assert resolve_max_hw(None) is None


class TestResolveCentroid:
    def test_anchor_part_implies_the_anchor_method(self):
        assert resolve_centroid_method(anchor_part="a") == ("anchor", "center_of_mass")

    def test_explicit_fallback_is_kept(self):
        resolved = resolve_centroid_method(
            anchor_part="a", centroid_fallback="bbox_center"
        )
        assert resolved == ("anchor", "bbox_center")

    def test_no_anchor_defaults_to_center_of_mass(self):
        assert resolve_centroid_method() == ("center_of_mass", None)

    def test_unknown_values_degrade_rather_than_raise(self):
        """This is a preview; training reports a bad config properly."""
        assert resolve_centroid_method(centroid_method="nonsense") == (
            "center_of_mass",
            None,
        )
        assert resolve_centroid_method(centroid_method="anchor") == (
            "center_of_mass",
            None,
        )

    def test_resolve_anchor_ind(self):
        labels = _make_labels([[[0, 0], [1, 1]]])
        assert resolve_anchor_ind(labels, "b") == 1
        assert resolve_anchor_ind(labels, "not_a_node") is None
        assert resolve_anchor_ind(labels, None) is None
        assert resolve_anchor_ind(None, "a") is None

    def test_resolve_crop_centroid_reads_the_head_leaf(self):
        from omegaconf import OmegaConf

        labels = _make_labels([[[0, 0], [1, 1]]])
        model_cfg = OmegaConf.create(
            {
                "model_config": {
                    "head_configs": {
                        "centered_instance": {"confmaps": {"anchor_part": "b"}}
                    }
                }
            }
        )
        anchor_ind, anchor_part, method, fallback = resolve_crop_centroid(
            model_cfg, labels
        )
        assert (anchor_ind, anchor_part, method, fallback) == (
            1,
            "b",
            "anchor",
            "center_of_mass",
        )

    def test_resolve_crop_centroid_degrades_an_unresolvable_anchor(self):
        """An anchor naming no node in the skeleton is not the center in force."""
        from omegaconf import OmegaConf

        labels = _make_labels([[[0, 0], [1, 1]]])
        model_cfg = OmegaConf.create(
            {
                "model_config": {
                    "head_configs": {
                        "multi_class_topdown": {"confmaps": {"anchor_part": "gone"}}
                    }
                }
            }
        )
        assert resolve_crop_centroid(model_cfg, labels) == (
            None,
            None,
            "center_of_mass",
            None,
        )

    def test_resolve_crop_centroid_tolerates_a_partial_config(self):
        """The form hands over partial configs while the user is still typing."""
        from omegaconf import OmegaConf

        assert resolve_crop_centroid(OmegaConf.create({"model_config": {}})) == (
            None,
            None,
            "center_of_mass",
            None,
        )


class TestResolveCropInputs:
    def test_defaults_match_the_training_config(self):
        """Fields absent from the GUI form still apply, as training applies them."""
        from omegaconf import OmegaConf

        labels = _make_labels([[[0, 0], [1, 1]]], video=_video("a.mp4", 512, 640))
        inputs = resolve_crop_inputs(
            OmegaConf.create({"data_config": {"preprocessing": {}}}),
            OmegaConf.create({"model_config": {}}),
            labels,
        )
        assert inputs["min_crop_size"] == DEFAULT_MIN_CROP_SIZE
        assert inputs["user_instances_only"] is True
        assert inputs["max_hw"] == (512, 640)
        assert inputs["centroid_method"] == "center_of_mass"

    def test_configured_values_are_read(self):
        from omegaconf import OmegaConf

        labels = _make_labels([[[0, 0], [1, 1]]], video=_video("a.mp4", 512, 640))
        data_cfg = OmegaConf.create(
            {
                "data_config": {
                    "preprocessing": {
                        "max_height": 256,
                        "max_width": 256,
                        "min_crop_size": 64,
                    },
                    "user_instances_only": False,
                }
            }
        )
        inputs = resolve_crop_inputs(
            data_cfg, OmegaConf.create({"model_config": {}}), labels
        )
        assert inputs["max_hw"] == (256, 256)
        assert inputs["min_crop_size"] == 64
        assert inputs["user_instances_only"] is False


class TestUserInstancesOnly:
    def test_predicted_instances_included_when_requested(self):
        """`user_instances_only=False` must count predictions, as training does."""
        labels = _make_labels(
            user_points=[[[0, 0], [100, 50]]],
            predicted_points=[[[0, 0], [800, 800]]],
        )
        assert (
            find_instance_crop_size(
                labels, maximum_stride=16, user_instances_only=False, min_crop_size=0
            )
            == 800
        )
        assert find_max_instance_bbox_size(labels, user_instances_only=False) == 800.0


class TestAugmentationPadding:
    def test_no_geometric_augmentation_is_no_padding(self):
        assert compute_augmentation_padding(100.0, 0.0, 1.0) == 0
        assert compute_augmentation_padding(100.0, 0.0, 0.9) == 0

    def test_worst_case_rotation_is_45_degrees(self):
        """Any range including 45 degrees hits the sqrt(2) expansion."""
        at_45 = compute_augmentation_padding(100.0, 45.0, 1.0)
        assert at_45 == compute_augmentation_padding(100.0, 180.0, 1.0)
        assert at_45 == int(np.ceil(100.0 * (np.sqrt(2) - 1)))

    def test_matches_sleap_nn(self):
        """The local copy must agree with sleap-nn's, which it forks to skip torch."""
        from sleap_nn.data.instance_cropping import (
            compute_augmentation_padding as nn_padding,
        )

        for bbox in (50.0, 100.0, 137.5, 1000.0):
            for rot in (0.0, 15.0, 45.0, 90.0, 180.0):
                for scale in (1.0, 1.1, 1.5):
                    assert compute_augmentation_padding(bbox, rot, scale) == nn_padding(
                        bbox, rot, scale
                    )


nn_crop = pytest.importorskip("sleap_nn.data.instance_cropping")


@pytest.mark.skipif(
    not hasattr(nn_crop, "iter_required_crop_sizes"),
    reason="installed sleap-nn predates the anchor-aware crop sizing (#748)",
)
class TestParityWithSleapNn:
    """Guard against this fork drifting from sleap-nn again.

    Skipped until the pinned sleap-nn is new enough; it then compares the two
    implementations directly rather than restating their expected numbers.
    """

    def _cases(self):
        big = _video("big.mp4", 2048, 2048)
        small = _video("small.mp4", 1024, 1024)
        wide = _video("wide.mp4", 600, 1200)
        pts7 = [[10, 10], [200, 12], [33, 90], [140, 250], [7, 300], [5, 5], [180, 30]]
        skeleton7 = tuple("abcdefg")
        yield _make_labels([[[0, 0], [100, 50]]], video=big), dict(maximum_stride=16)
        yield (
            _make_labels([[[0, 0], [50, 0], [100, 0]]], nodes=("a", "b", "c")),
            dict(maximum_stride=16, anchor_ind=0),
        )
        yield (
            _make_labels([[[10, 10], [210, 110]]], video=wide),
            dict(maximum_stride=16, max_hw=(1024, 1280)),
        )
        yield (
            _make_labels([[[0, 0], [150, 75]]], video=small),
            dict(maximum_stride=16, max_hw=(2048, 2048)),
        )
        yield (
            _make_labels([[[0, 0], [300, 300]]]),
            dict(maximum_stride=16, min_crop_size=96),
        )
        yield sio.Labels(labeled_frames=[]), dict(maximum_stride=16, min_crop_size=100)
        yield (
            _make_labels([[[0, 0], [40, 40]]]),
            dict(maximum_stride=16, min_crop_size=100, padding=30),
        )
        for method in ("center_of_mass", "bbox_center", "geometric_median"):
            yield (
                _make_labels([pts7], nodes=skeleton7),
                dict(maximum_stride=1, centroid_method=method),
            )

    def test_find_instance_crop_size_agrees(self):
        for labels, kwargs in self._cases():
            mine = find_instance_crop_size(labels, **kwargs)
            theirs = nn_crop.find_instance_crop_size(
                labels, user_instances_only=True, **kwargs
            )
            assert mine == theirs, kwargs

    def test_required_crop_sizes_agree(self):
        for labels, kwargs in self._cases():
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ("maximum_stride", "min_crop_size", "padding")
            }
            mine = list(iter_required_crop_sizes(labels, **kwargs))
            theirs = list(
                nn_crop.iter_required_crop_sizes(
                    labels, user_instances_only=True, **kwargs
                )
            )
            assert len(mine) == len(theirs)
            assert np.allclose(mine, theirs, atol=1e-4), kwargs

    def test_max_instance_bbox_size_agrees(self):
        for labels, kwargs in self._cases():
            max_hw = kwargs.get("max_hw")
            assert np.isclose(
                find_max_instance_bbox_size(labels, max_hw=max_hw),
                nn_crop.find_max_instance_bbox_size(
                    labels, max_hw=max_hw, user_instances_only=True
                ),
            )


class TestCropBoxPreview:
    """The crop box is drawn over a frame at its video's native resolution.

    The crop size is measured in size-matched pixels, so the box has to divide
    that scale back out or it covers the wrong region (sleap-nn #748).
    """

    def _widget_with_labels(self, qtbot, height, width):
        from sleap.gui.learning.receptivefield import ReceptiveFieldWidget

        video = _video("preview.mp4", height, width)
        labels = _make_labels([[[10, 10], [60, 40]]], video=video)

        widget = ReceptiveFieldWidget("centered_instance", show_crop_box=True)
        qtbot.addWidget(widget)
        widget.setLabels(labels)
        return widget, labels

    def test_preview_video_is_recorded(self, qtbot):
        """`setLabels` must remember which video the shown frame came from."""
        widget, labels = self._widget_with_labels(qtbot, 512, 512)
        assert widget._preview_video is labels.videos[0]

    def test_box_is_scaled_out_of_the_size_matched_space(self, qtbot):
        """A 512px video matched up to 1024 is upscaled 2x, so the box halves."""
        widget, _ = self._widget_with_labels(qtbot, 512, 512)

        widget.setCropConfig(crop_size=200, scale=1.0, max_hw=(1024, 1024))
        image_widget = widget._field_image_widget
        assert image_widget._crop_scale == pytest.approx(2.0)
        # 200 size-matched px is 100 px of this video's own frame.
        assert image_widget.crop_box.rect().width() == pytest.approx(100.0)

    def test_no_size_matching_leaves_the_box_alone(self, qtbot):
        widget, _ = self._widget_with_labels(qtbot, 1024, 1024)

        widget.setCropConfig(crop_size=200, scale=1.0, max_hw=(1024, 1024))
        image_widget = widget._field_image_widget
        assert image_widget._crop_scale == pytest.approx(1.0)
        assert image_widget.crop_box.rect().width() == pytest.approx(200.0)

        # And with no target at all.
        widget.setCropConfig(crop_size=200, scale=1.0, max_hw=None)
        assert image_widget._crop_scale == pytest.approx(1.0)

    def test_input_scaling_still_applies(self, qtbot):
        """`crop_size` arrives already multiplied by the input scale."""
        widget, _ = self._widget_with_labels(qtbot, 1024, 1024)

        # compute_crop_size_from_cfg would hand over 200 * 0.5 = 100 here.
        widget.setCropConfig(crop_size=100, scale=0.5, max_hw=(1024, 1024))
        image_widget = widget._field_image_widget
        # 100 / 0.5 recovers the 200px region the crop covers.
        assert image_widget.crop_box.rect().width() == pytest.approx(200.0)

    def test_crop_box_scale_does_not_disturb_the_receptive_field_box(self, qtbot):
        """The two boxes live in different pixel spaces and must not share one.

        They both used `self._scale`, so a size-matched crop scale would have
        resized the receptive field box on the next repaint.
        """
        widget, _ = self._widget_with_labels(qtbot, 512, 512)
        image_widget = widget._field_image_widget

        image_widget._set_field_size(64, 1.0)
        rf_width = image_widget.box.rect().width()

        widget.setCropConfig(crop_size=200, scale=1.0, max_hw=(1024, 1024))
        # Re-drawing with no arguments is what viewportEvent does on repaint.
        image_widget._set_field_size()

        assert image_widget._scale == pytest.approx(1.0)
        assert image_widget.box.rect().width() == pytest.approx(rf_width)

    def test_unreadable_preview_video_falls_back(self, qtbot):
        """An unreadable video must not stop the box from being drawn."""
        from sleap.gui.learning.receptivefield import ReceptiveFieldWidget

        labels = _make_labels([[[10, 10], [60, 40]]], video=_video("missing.mp4"))
        widget = ReceptiveFieldWidget("centered_instance", show_crop_box=True)
        qtbot.addWidget(widget)
        widget.setLabels(labels)

        widget.setCropConfig(crop_size=200, scale=1.0, max_hw=(1024, 1024))
        assert widget._field_image_widget._crop_scale == pytest.approx(1.0)


class TestGetFirstLabeledFrameAndInstance:
    def test_reports_the_video_it_read_from(self):
        """The video is needed to know what pixel space the image is in."""
        from sleap.gui.learning.receptivefield import (
            get_first_labeled_frame_and_instance,
        )

        video = _video("preview.mp4", 512, 512)
        labels = _make_labels([[[10, 10], [60, 40]]], video=video)

        frame_image, instance, read_video = get_first_labeled_frame_and_instance(labels)
        # The file does not exist, so the frame cannot be read -- but the
        # instance and the video it belongs to still resolve.
        assert frame_image is None
        assert instance is labels[0].instances[0]
        assert read_video is video

    def test_no_labels(self):
        from sleap.gui.learning.receptivefield import (
            get_first_labeled_frame_and_instance,
        )

        assert get_first_labeled_frame_and_instance(None) == (None, None, None)
        assert get_first_labeled_frame_and_instance(sio.Labels(labeled_frames=[])) == (
            None,
            None,
            None,
        )
