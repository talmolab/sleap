import numpy as np
import tensorflow as tf

tf.config.experimental.set_visible_devices([], device_type="GPU")  # hide GPUs for test

from sleap.nn.architectures import unet


class UnetTests(tf.test.TestCase):
    def test_unet_reference(self):
        # Reference implementation from the original paper.
        arch = unet.Unet(
            filters=64,
            filters_rate=2,
            kernel_size=3,
            convs_per_block=2,
            down_blocks=4,
            middle_block=True,
            up_blocks=4,
            up_interpolate=False,
        )
        x_in = tf.keras.layers.Input((192, 192, 1))
        x, x_mid = arch.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        self.assertEqual(len(model.layers), 61)
        self.assertEqual(len(model.trainable_weights), 60)
        self.assertEqual(np.sum(param_counts), 34515968)
        self.assertEqual(model.count_params(), 34519808)
        self.assertAllEqual(model.output.shape, (None, 192, 192, 64))
        self.assertEqual(len(x_mid), 4)
        self.assertEqual(arch.encoder_features_stride, 16)
        self.assertEqual(arch.decoder_features_stride, 1)
