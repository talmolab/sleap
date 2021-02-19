import tensorflow as tf

import sleap
from sleap.nn.heads import (
    Head,
    SingleInstanceConfmapsHead,
    CentroidConfmapsHead,
    CenteredInstanceConfmapsHead,
    MultiInstanceConfmapsHead,
    PartAffinityFieldsHead,
    ClassMapsHead,
    OffsetRefinementHead,
)
from sleap.nn.config import (
    SingleInstanceConfmapsHeadConfig,
    CentroidsHeadConfig,
    CenteredInstanceConfmapsHeadConfig,
    MultiInstanceConfmapsHeadConfig,
    PartAffinityFieldsHeadConfig,
    ClassMapsHeadConfig,
)


sleap.use_cpu_only()


def test_single_instance_confmaps_head():
    head = SingleInstanceConfmapsHead(
        part_names=["a", "b", "c"],
        sigma=1.0,
        output_stride=1,
        loss_weight=1.0,
    )

    x_in = tf.keras.layers.Input([4, 4, 4])
    x = head.make_head(x_in)

    assert head.channels == 3
    assert tuple(x.shape) == (None, 4, 4, 3)
    assert tf.keras.Model(x_in, x).output_names[0] == "SingleInstanceConfmapsHead"
    assert tf.keras.Model(x_in, x).layers[-1].activation.__name__ == "linear"

    head = SingleInstanceConfmapsHead.from_config(
        SingleInstanceConfmapsHeadConfig(
            part_names=None,
            sigma=1.5,
            output_stride=2,
            loss_weight=2.0,
            offset_refinement=False,
        ),
        part_names=["c", "b", "a"],
    )
    assert head.part_names == ["c", "b", "a"]
    assert head.sigma == 1.5
    assert head.output_stride == 2
    assert head.loss_weight == 2.0
    x = head.make_head(x_in)


def test_centroid_confmaps_head():
    head = CentroidConfmapsHead(
        anchor_part="a",
        sigma=1.0,
        output_stride=1,
        loss_weight=1.0,
    )

    x_in = tf.keras.layers.Input([4, 4, 4])
    x = head.make_head(x_in)

    assert head.channels == 1
    assert tuple(x.shape) == (None, 4, 4, 1)
    assert tf.keras.Model(x_in, x).output_names[0] == "CentroidConfmapsHead"
    assert tf.keras.Model(x_in, x).layers[-1].activation.__name__ == "linear"

    head = CentroidConfmapsHead.from_config(
        CentroidsHeadConfig(
            anchor_part="a",
            sigma=1.5,
            output_stride=2,
            loss_weight=2.0,
            offset_refinement=False,
        )
    )
    assert head.anchor_part == "a"
    assert head.sigma == 1.5
    assert head.output_stride == 2
    assert head.loss_weight == 2.0
    x = head.make_head(x_in)


def test_centroid_confmaps_head():
    head = CentroidConfmapsHead(
        anchor_part="a",
        sigma=1.0,
        output_stride=1,
        loss_weight=1.0,
    )

    x_in = tf.keras.layers.Input([4, 4, 4])
    x = head.make_head(x_in)

    assert head.channels == 1
    assert tuple(x.shape) == (None, 4, 4, 1)
    assert tf.keras.Model(x_in, x).output_names[0] == "CentroidConfmapsHead"
    assert tf.keras.Model(x_in, x).layers[-1].activation.__name__ == "linear"

    head = CentroidConfmapsHead.from_config(
        CentroidsHeadConfig(
            anchor_part="a",
            sigma=1.5,
            output_stride=2,
            loss_weight=2.0,
            offset_refinement=False,
        )
    )
    assert head.anchor_part == "a"
    assert head.sigma == 1.5
    assert head.output_stride == 2
    assert head.loss_weight == 2.0
    x = head.make_head(x_in)


def test_centered_instance_confmaps_head():
    head = CenteredInstanceConfmapsHead(
        part_names=["a", "b", "c"],
        anchor_part="a",
        sigma=1.0,
        output_stride=1,
        loss_weight=1.0,
    )

    x_in = tf.keras.layers.Input([4, 4, 4])
    x = head.make_head(x_in)

    assert head.channels == 3
    assert tuple(x.shape) == (None, 4, 4, 3)
    assert tf.keras.Model(x_in, x).output_names[0] == "CenteredInstanceConfmapsHead"
    assert tf.keras.Model(x_in, x).layers[-1].activation.__name__ == "linear"

    head = CenteredInstanceConfmapsHead.from_config(
        CenteredInstanceConfmapsHeadConfig(
            part_names=None,
            anchor_part="a",
            sigma=1.5,
            output_stride=2,
            loss_weight=2.0,
            offset_refinement=False,
        ),
        part_names=["c", "b", "a"],
    )
    assert head.part_names == ["c", "b", "a"]
    assert head.sigma == 1.5
    assert head.output_stride == 2
    assert head.loss_weight == 2.0


def test_multi_instance_confmaps_head():
    head = MultiInstanceConfmapsHead(
        part_names=["a", "b", "c"],
        sigma=1.0,
        output_stride=1,
        loss_weight=1.0,
    )

    x_in = tf.keras.layers.Input([4, 4, 4])
    x = head.make_head(x_in)

    assert head.channels == 3
    assert tuple(x.shape) == (None, 4, 4, 3)
    assert tf.keras.Model(x_in, x).output_names[0] == "MultiInstanceConfmapsHead"
    assert tf.keras.Model(x_in, x).layers[-1].activation.__name__ == "linear"

    head = MultiInstanceConfmapsHead.from_config(
        MultiInstanceConfmapsHeadConfig(
            part_names=None,
            sigma=1.5,
            output_stride=2,
            loss_weight=2.0,
            offset_refinement=False,
        ),
        part_names=["c", "b", "a"],
    )
    assert head.part_names == ["c", "b", "a"]
    assert head.sigma == 1.5
    assert head.output_stride == 2
    assert head.loss_weight == 2.0


def test_part_affinity_fields_head():
    head = PartAffinityFieldsHead(
        edges=[("a", "b"), ("b", "c")],
        sigma=1.0,
        output_stride=1,
        loss_weight=1.0,
    )

    x_in = tf.keras.layers.Input([4, 4, 5])
    x = head.make_head(x_in)

    assert head.channels == 4
    assert tuple(x.shape) == (None, 4, 4, 4)
    assert tf.keras.Model(x_in, x).output_names[0] == "PartAffinityFieldsHead"
    assert tf.keras.Model(x_in, x).layers[-1].activation.__name__ == "linear"

    head = PartAffinityFieldsHead.from_config(
        PartAffinityFieldsHeadConfig(
            edges=None,
            sigma=1.5,
            output_stride=2,
            loss_weight=2.0,
        ),
        edges=[("a", "b"), ("b", "c")],
    )
    assert head.edges == [("a", "b"), ("b", "c")]
    assert head.sigma == 1.5
    assert head.output_stride == 2
    assert head.loss_weight == 2.0


def test_class_maps_head():
    head = ClassMapsHead(
        classes=["1", "2"],
        sigma=1.0,
        output_stride=1,
        loss_weight=1.0,
    )

    x_in = tf.keras.layers.Input([4, 4, 4])
    x = head.make_head(x_in)

    assert head.channels == 2
    assert tuple(x.shape) == (None, 4, 4, 2)
    assert tf.keras.Model(x_in, x).output_names[0] == "ClassMapsHead"
    assert tf.keras.Model(x_in, x).layers[-1].activation.__name__ == "sigmoid"

    head = ClassMapsHead.from_config(
        ClassMapsHeadConfig(
            classes=None,
            sigma=1.5,
            output_stride=2,
            loss_weight=2.0,
        ),
        classes=["1", "2"],
    )
    assert head.classes == ["1", "2"]
    assert head.sigma == 1.5
    assert head.output_stride == 2
    assert head.loss_weight == 2.0


def test_offset_refinement_head():
    head = OffsetRefinementHead(
        part_names=["a", "b", "c"],
        sigma_threshold=0.3,
        output_stride=1,
        loss_weight=1.0,
    )

    x_in = tf.keras.layers.Input([4, 4, 8])
    x = head.make_head(x_in)

    assert head.channels == 6
    assert tuple(x.shape) == (None, 4, 4, 6)
    assert tf.keras.Model(x_in, x).output_names[0] == "OffsetRefinementHead"
    assert tf.keras.Model(x_in, x).layers[-1].activation.__name__ == "linear"

    head = OffsetRefinementHead.from_config(
        MultiInstanceConfmapsHeadConfig(
            part_names=["a", "b"],
            sigma=1.5,
            output_stride=2,
            loss_weight=2.0,
            offset_refinement=False,
        ),
        sigma_threshold=0.4,
    )
    assert head.part_names == ["a", "b"]
    assert head.output_stride == 2
    assert head.sigma_threshold == 0.4

    head = OffsetRefinementHead.from_config(
        MultiInstanceConfmapsHeadConfig(), part_names=["a", "b"]
    )
    assert head.part_names == ["a", "b"]

    head = OffsetRefinementHead.from_config(CentroidsHeadConfig(anchor_part="a"))
    assert head.part_names == ["a"]

    head = OffsetRefinementHead.from_config(CentroidsHeadConfig(anchor_part=None))
    assert head.part_names == [None]
    assert head.channels == 2
