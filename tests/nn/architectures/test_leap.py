import numpy as np
import tensorflow as tf

tf.config.experimental.set_visible_devices([], device_type="GPU")  # hide GPUs for test

from sleap.nn.architectures import leap


class LeapTests(tf.test.TestCase):
    def test_leap_cnn_reference(self):
        # Reference implementation from the original paper.
        arch = leap.LeapCNN(
            filters=64,
            filters_rate=2,
            down_blocks=3,
            down_convs_per_block=3,
            up_blocks=3,
            up_interpolate=False,
            up_convs_per_block=2,
        )
        x_in = tf.keras.layers.Input((192, 192, 1))
        x, x_mid = arch.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        self.assertEqual(len(model.layers), 40)
        self.assertEqual(len(model.trainable_weights), 36)
        self.assertEqual(np.sum(param_counts), 10768896)
        self.assertEqual(model.count_params(), 10768896)
        self.assertAllEqual(model.output.shape, (None, 192, 192, 128))
        self.assertEqual(len(x_mid), 3)
        self.assertEqual(arch.encoder_features_stride, 8)
        self.assertEqual(arch.decoder_features_stride, 1)
        self.assertIsInstance(
            model.get_layer("dec0_s8_to_s4_trans_conv"),
            tf.keras.layers.Conv2DTranspose,
        )

    def test_leap_cnn_interp(self):
        arch = leap.LeapCNN(
            filters=8,
            filters_rate=2,
            down_blocks=3,
            down_convs_per_block=3,
            up_blocks=3,
            up_interpolate=True,
            up_convs_per_block=2,
        )
        x_in = tf.keras.layers.Input((64, 64, 1))
        x, x_mid = arch.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        self.assertEqual(len(model.layers), 37)
        self.assertEqual(len(model.trainable_weights), 30)
        self.assertEqual(np.sum(param_counts), 120272)
        self.assertEqual(model.count_params(), 120272)
        self.assertAllEqual(model.output.shape, (None, 64, 64, 16))
        self.assertIsInstance(
            model.get_layer("dec0_s8_to_s4_interp_bilinear"),
            tf.keras.layers.UpSampling2D,
        )
