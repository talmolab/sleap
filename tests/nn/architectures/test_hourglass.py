import numpy as np
import tensorflow as tf

tf.config.experimental.set_visible_devices([], device_type="GPU")  # hide GPUs for test

from sleap.nn.architectures import hourglass


class UnetTests(tf.test.TestCase):
    def test_hourglass_reference(self):
        # Reference implementation from the original paper.
        arch = hourglass.Hourglass(
            down_blocks=4,
            up_blocks=4,
            stem_filters=128,
            stem_stride=4,
            filters=256,
            filter_increase=128,
            interp_method="nearest",
            stacks=3
        )
        x_in = tf.keras.layers.Input((256, 256, 1))
        x, x_mid = arch.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        self.assertAllEqual(
            [out.shape for out in model.output],
            [(None, 64, 64, 256)] * 3)
        self.assertEqual(arch.encoder_features_stride, 64)
        self.assertEqual(arch.decoder_features_stride, 4)
        self.assertEqual(len(model.layers), 116)
        self.assertEqual(len(model.trainable_weights), 156)
        self.assertEqual(np.sum(param_counts), 65969408)
        self.assertEqual(model.count_params(), 66002944)
        self.assertEqual(len(x_mid), 3)